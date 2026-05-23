#include "SurfelPathTracerCommon.slang"

[[vk::binding(4, 0)]] RWByteAddressBuffer counters;
[[vk::binding(5, 0)]] RWStructuredBuffer<SurfelPathTracerSurfel> surfelBuffer;
[[vk::binding(10, 0)]] RWStructuredBuffer<uint> rayBuffer;
[[vk::binding(11, 0)]] RWStructuredBuffer<SurfelPathTracerCellInfo> cellInfoBuffer;
[[vk::binding(13, 0)]] RWStructuredBuffer<uint> cellToSurfelBuffer;
[[vk::binding(14, 0)]] RWTexture2D<float4> irradianceAtlas;
[[vk::binding(15, 0)]] RWTexture2D<float> surfelDepthAtlas;

[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

struct IntegratePushConstants {
    uint maxSurfels;
    uint enableRadianceSharing;
    uint maxRadianceSharingSamples;
    uint cellDimension;
    float cellSize;
};

[[vk::push_constant]] IntegratePushConstants push;

float3 loadRayRadiance(uint rayIndex) {
    uint base = rayIndex * SURFEL_PT_RAY_RECORD_UINTS;
    return float3(asfloat(rayBuffer[base + SURFEL_PT_RAY_RADIANCE_X]),
                  asfloat(rayBuffer[base + SURFEL_PT_RAY_RADIANCE_Y]),
                  asfloat(rayBuffer[base + SURFEL_PT_RAY_RADIANCE_Z]));
}

float loadRayHitT(uint rayIndex) {
    return asfloat(rayBuffer[rayIndex * SURFEL_PT_RAY_RECORD_UINTS + SURFEL_PT_RAY_HIT_T]);
}

float loadRayPdf(uint rayIndex) {
    return asfloat(rayBuffer[rayIndex * SURFEL_PT_RAY_RECORD_UINTS + SURFEL_PT_RAY_PDF]);
}

float3 loadRayLocalDirection(uint rayIndex) {
    return unpackRayLocalDirection(rayBuffer[rayIndex * SURFEL_PT_RAY_RECORD_UINTS + SURFEL_PT_RAY_LOCAL_DIRECTION]);
}

void writeAtlas(uint surfelIndex, float3 localDirection, float3 radiance, float depth, uint width, uint height) {
    if (width == 0u || height == 0u) {
        return;
    }

    uint2 coord = atlasTileBase(surfelIndex, width) + atlasDirectionCoord(localDirection);
    if (coord.x >= width || coord.y >= height) {
        return;
    }

    float newLuma = luminance(radiance);
    irradianceAtlas[coord] = float4(radiance, max(newLuma, 1e-5));
    surfelDepthAtlas[coord] = depth;
}

float3 terminatePathWithSurfels(float3 position,
                                float3 normal,
                                uint maxSamples,
                                uint maxSurfels,
                                float cellSize,
                                uint cellDimension,
                                float3 cameraPosition)
{
    uint cellIndex = cameraRelativeCellIndexForPosition(position, cameraPosition, cellSize, cellDimension);
    uint cellCount = cellDimension * cellDimension * cellDimension;
    if (cellIndex == SURFEL_PT_INVALID_INDEX || cellIndex >= cellCount) {
        return float3(0.0);
    }

    SurfelPathTracerCellInfo cellInfo = cellInfoBuffer[cellIndex];
    uint count = min(cellInfo.surfelCount, maxSamples);
    float3 sum = float3(0.0);
    float weightSum = 0.0;
    for (uint i = 0u; i < count; ++i) {
        uint surfelIndex = cellToSurfelBuffer[cellInfo.surfelOffset + i];
        if (surfelIndex == SURFEL_PT_INVALID_INDEX || surfelIndex >= maxSurfels) {
            continue;
        }

        SurfelPathTracerSurfel surfel = surfelBuffer[surfelIndex];
        if (!isActiveSurfel(surfel)) {
            continue;
        }

        float3 surfelNormal = unpackNormalOctahedral(surfel.packedNormal);
        float distanceWeight = saturate(1.0 - length(position - surfel.position) / max(surfel.radius, 1e-4));
        float normalWeight = saturate(dot(normal, surfelNormal));
        float weight = distanceWeight * normalWeight;
        sum += surfel.radiance * weight;
        weightSum += weight;
    }
    return weightSum > 1e-5 ? sum / weightSum : float3(0.0);
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
    float accumulatedDepth = 0.0;
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
        float rayDepth = max(loadRayHitT(rayIndex), 0.0);
        accumulatedRadiance += weightedRadiance;
        accumulatedDepth += rayDepth;
        sampleCount += 1u;
        writeAtlas(surfelIndex, localDirection, rayRadiance * integrationWeight, rayDepth, atlasWidth, atlasHeight);
    }

    if (sampleCount == 0u) {
        surfel.rayCount = 1u;
        surfelBuffer[surfelIndex] = surfel;
        return;
    }

    float invSampleCount = 1.0 / float(sampleCount);
    float3 sampleRadiance = clampLuminance(accumulatedRadiance * invSampleCount, SURFEL_PT_MAX_RADIANCE_LUMINANCE);
    if (push.enableRadianceSharing != 0u) {
        float3 shared = terminatePathWithSurfels(surfel.position,
                                                 unpackNormalOctahedral(surfel.packedNormal),
                                                 push.maxRadianceSharingSamples,
                                                 push.maxSurfels,
                                                 push.cellSize,
                                                 push.cellDimension,
                                                 ubo.cameraPos.xyz);
        if (luminance(shared) > 1e-5) {
            sampleRadiance = lerp(sampleRadiance, shared, 0.25);
        }
    }

    float shortAlpha = surfel.shortMeanAndLife.w <= 1.0 ? 4.0 : 0.25;
    float3 longMean = updateMsme(sampleRadiance, surfel, shortAlpha);

    surfel.radiance = longMean;
    surfel.rayCount = 1u;
    surfelBuffer[surfelIndex] = surfel;
}
