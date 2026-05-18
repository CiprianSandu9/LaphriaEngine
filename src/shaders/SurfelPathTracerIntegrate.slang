#include "SurfelPathTracerCommon.slang"

[[vk::binding(4, 0)]] RWByteAddressBuffer counters;
[[vk::binding(5, 0)]] RWStructuredBuffer<SurfelPathTracerSurfel> surfelBuffer;
[[vk::binding(10, 0)]] RWStructuredBuffer<uint> rayBuffer;
[[vk::binding(14, 0)]] RWTexture2D<float4> irradianceAtlas;
[[vk::binding(15, 0)]] RWTexture2D<float> surfelDepthAtlas;

struct IntegratePushConstants {
    uint maxSurfels;
    uint pad0;
    uint pad1;
    uint pad2;
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

void writeAtlas(uint atlasOffset, float3 radiance, float depth) {
    uint width = 0u;
    uint height = 0u;
    irradianceAtlas.GetDimensions(width, height);
    if (width == 0u || height == 0u) {
        return;
    }

    uint texelCount = width * height;
    uint wrappedOffset = atlasOffset % max(texelCount, 1u);
    uint2 coord = uint2(wrappedOffset % width, wrappedOffset / width);
    irradianceAtlas[coord] = float4(radiance, 1.0);
    surfelDepthAtlas[coord] = depth;
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
    if (surfel.flags == 0u || surfel.radius <= 0.0 || !isFinite3(surfel.position)) {
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

        accumulatedRadiance += loadRayRadiance(rayIndex);
        accumulatedDepth += loadRayHitT(rayIndex);
        sampleCount += 1u;
    }

    if (sampleCount == 0u) {
        surfel.rayCount = 1u;
        surfelBuffer[surfelIndex] = surfel;
        return;
    }

    float invSampleCount = 1.0 / float(sampleCount);
    float3 sampleRadiance = clampLuminance(accumulatedRadiance * invSampleCount, SURFEL_PT_MAX_RADIANCE_LUMINANCE);
    float3 previousLongMean = surfel.meanAndVariance.xyz;
    float3 previousShortMean = surfel.shortMeanAndLife.xyz;
    float longAlpha = surfel.shortMeanAndLife.w <= 1.0 ? 1.0 : 0.05;
    float shortAlpha = 0.25;

    float3 longMean = msmeBlend(previousLongMean, sampleRadiance, longAlpha);
    float3 shortMean = msmeBlend(previousShortMean, sampleRadiance, shortAlpha);
    float error = luminance(sampleRadiance - longMean);
    float variance = lerp(surfel.meanAndVariance.w, error * error, 0.1);

    surfel.meanAndVariance = float4(longMean, variance);
    surfel.shortMeanAndLife = float4(shortMean, surfel.shortMeanAndLife.w);
    surfel.radiance = longMean;
    surfel.rayCount = 1u;
    surfelBuffer[surfelIndex] = surfel;

    writeAtlas(surfel.irradianceAtlasOffset, longMean, accumulatedDepth * invSampleCount);
}
