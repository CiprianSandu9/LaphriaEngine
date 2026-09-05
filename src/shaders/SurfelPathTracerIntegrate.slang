#include "SurfelPathTracerCommon.slang"

[[vk::binding(4, 0)]] RWByteAddressBuffer counters;
[[vk::binding(5, 0)]] RWStructuredBuffer<SurfelPathTracerSurfel> surfelBuffer;
[[vk::binding(10, 0)]] RWStructuredBuffer<uint> rayBuffer;
[[vk::binding(11, 0)]] RWStructuredBuffer<SurfelPathTracerCellInfo> cellInfoBuffer;
[[vk::binding(13, 0)]] RWStructuredBuffer<uint> cellToSurfelBuffer;
[[vk::binding(14, 0)]] RWTexture2D<float4> irradianceAtlas;

[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

struct IntegratePushConstants {
    uint maxSurfels;
    uint enableRadianceSharing;
    uint maxRadianceSharingSamples;
    uint cellDimension;
    float cellSize;
    uint enableGuidedSampling;
    float surfelSupportRadius;
    uint pad1;
};

[[vk::push_constant]] IntegratePushConstants push;

float3 loadRayRadiance(uint rayIndex) {
    uint base = rayIndex * SURFEL_PT_RAY_RECORD_UINTS;
    return float3(asfloat(rayBuffer[base + SURFEL_PT_RAY_RADIANCE_X]),
                  asfloat(rayBuffer[base + SURFEL_PT_RAY_RADIANCE_Y]),
                  asfloat(rayBuffer[base + SURFEL_PT_RAY_RADIANCE_Z]));
}

float loadRayPdf(uint rayIndex) {
    return asfloat(rayBuffer[rayIndex * SURFEL_PT_RAY_RECORD_UINTS + SURFEL_PT_RAY_PDF]);
}

float3 loadRayLocalDirection(uint rayIndex) {
    return unpackRayLocalDirection(rayBuffer[rayIndex * SURFEL_PT_RAY_RECORD_UINTS + SURFEL_PT_RAY_LOCAL_DIRECTION]);
}

void writeAtlas(uint surfelIndex,
                uint guideToken,
                float3 localDirection,
                float3 incidentRadiance,
                uint width,
                uint height) {
    if (width == 0u || height == 0u) {
        return;
    }

    uint2 coord = atlasTileBase(surfelIndex, width) + atlasDirectionCoord(localDirection);
    if (coord.x >= width || coord.y >= height) {
        return;
    }

    float3 sampleRadiance = clampLuminance(max(incidentRadiance, float3(0.0)),
                                           SURFEL_PT_MAX_RADIANCE_LUMINANCE);
    float4 previous = irradianceAtlas[coord];
    float decodedPreviousToken = -previous.a;
    uint previousToken = isfinite(decodedPreviousToken) && decodedPreviousToken >= 1.0
                         ? uint(round(decodedPreviousToken))
                         : 0u;
    float3 filteredRadiance = previousToken == guideToken
                              ? lerp(previous.rgb, sampleRadiance, 0.125)
                              : sampleRadiance;
    irradianceAtlas[coord] = float4(filteredRadiance, -float(guideToken));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint surfelIndex = tid.x;
    uint requestedRays = counters.Load(SURFEL_PT_COUNTER_REQUESTED_RAYS_OFFSET);
    if (surfelIndex >= push.maxSurfels) {
        return;
    }

    SurfelPathTracerSurfel surfel = surfelBuffer[surfelIndex];
    if (!isActiveSurfel(surfel)) {
        return;
    }

    if (surfel.rayCount == 0u) {
        surfel.rayCount = 1u;
        surfelBuffer[surfelIndex] = surfel;
        return;
    }

    float3 accumulatedRadiance = float3(0.0, 0.0, 0.0);
    uint sampleCount = 0u;
    uint atlasWidth = 0u;
    uint atlasHeight = 0u;
    irradianceAtlas.GetDimensions(atlasWidth, atlasHeight);
    for (uint ray = 0u; ray < surfel.rayCount; ++ray) {
        uint rayIndex = surfel.rayOffset + ray;
        if (rayIndex >= requestedRays) {
            continue;
        }

        uint rayBase = rayIndex * SURFEL_PT_RAY_RECORD_UINTS;
        if (rayBuffer[rayBase + SURFEL_PT_RAY_FLAGS] == 0u ||
            rayBuffer[rayBase + SURFEL_PT_RAY_SURFEL_INDEX] != surfelIndex) {
            continue;
        }

        float3 rayRadiance = loadRayRadiance(rayIndex);
        float3 localDirection = loadRayLocalDirection(rayIndex);
        float pdf = max(loadRayPdf(rayIndex), 1e-5);
        float cosine = max(localDirection.z, 0.0);
        float integrationWeight = cosine / pdf;
        float3 weightedRadiance = rayRadiance * integrationWeight;
        accumulatedRadiance += weightedRadiance;
        sampleCount += 1u;
        // The guide estimates the directional incident-radiance distribution.
        // Do not feed the guide its own cosine/PDF importance weight: that
        // creates a positive feedback loop and amplifies noisy low-PDF rays.
        if (push.enableGuidedSampling != 0u) {
            writeAtlas(surfelIndex,
                       surfel.irradianceAtlasOffset,
                       localDirection,
                       rayRadiance,
                       atlasWidth,
                       atlasHeight);
        }
    }

    if (sampleCount == 0u) {
        surfel.rayCount = 1u;
        surfelBuffer[surfelIndex] = surfel;
        return;
    }

    float invSampleCount = 1.0 / float(sampleCount);
    float3 sampleRadiance = clampLuminance(accumulatedRadiance * invSampleCount, SURFEL_PT_MAX_RADIANCE_LUMINANCE);
    if (push.enableRadianceSharing != 0u) {
        float3 shared;
        bool cacheHit = sampleSurfelIrradianceCell(surfel.position,
                                                   unpackNormalOctahedral(surfel.packedNormal),
                                                   push.maxRadianceSharingSamples,
                                                   surfelIndex ^ counters.Load(SURFEL_PT_COUNTER_FRAME_INDEX_OFFSET),
                                                   surfelIndex /* never blend a surfel with itself */,
                                                   1u /* full 3x3x3 neighbourhood */,
                                                   push.maxSurfels,
                                                   push.cellSize,
                                                   push.surfelSupportRadius,
                                                   push.cellDimension,
                                                   ubo.cameraPos.xyz,
                                                   surfelBuffer,
                                                   cellInfoBuffer,
                                                   cellToSurfelBuffer,
                                                   shared);
        if (cacheHit && luminance(shared) > 1e-5) {
            sampleRadiance = lerp(sampleRadiance, shared, 0.25);
        }
    }

    // The first traced estimate replaces the placement seed outright (4.0 makes
    // both the short and the long mean blend at 1.0 inside updateMsme). Life is
    // incremented by Update before the first samples ever arrive here, so a
    // life-based test never fired and new surfels converged at the steady-state
    // rate from a seed that is frequently zero.
    // Warm-up is a cumulative average (1, 1/2, 1/3, ...) that settles on the
    // steady-state exponential blends. Both means start from the first traced
    // batch instead of the placement seed, but no single noisy batch dominates.
    uint batchesIntegrated = surfelSampleCount(surfel);
    float cumulative = 1.0 / float(batchesIntegrated + 1u);
    float shortAlpha = max(0.25, cumulative);
    float longAlpha = max(0.0625, cumulative);
    float3 longMean = updateMsme(sampleRadiance, surfel, shortAlpha, longAlpha);

    surfel.radiance = longMean;
    surfel.rayCount = 1u;
    surfel.flags = withSurfelSampleCount(surfel.flags | SURFEL_PT_SURFEL_FLAG_HAS_SAMPLES,
                                         batchesIntegrated + 1u);
    surfelBuffer[surfelIndex] = surfel;
}
