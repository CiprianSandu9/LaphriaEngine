#include "SurfelPathTracerCommon.slang"

[[vk::binding(1, 0)]] RWTexture2D<float4> gBufferNormal;
[[vk::binding(2, 0)]] RWTexture2D<float> gBufferDepth;
[[vk::binding(3, 0)]] RWTexture2D<float4> gBufferMotionMaterial;
[[vk::binding(16, 0)]] RWTexture2D<float4> reflectionImages;
[[vk::binding(17, 0)]] RWTexture2D<float4> filteredReflectionImages;

struct ReflectionFilterPushConstants {
    uint width;
    uint height;
    uint fullWidth;
    uint fullHeight;
    uint enabled;
    uint resetHistory;
    uint frameIndex;
    uint pad0;
};

[[vk::push_constant]] ReflectionFilterPushConstants push;

uint2 fullPixelForHalf(uint2 halfPixel) {
    return min(halfPixel * 2u + uint2(1u, 1u), uint2(push.fullWidth - 1u, push.fullHeight - 1u));
}

bool consistentWithCenter(uint2 sampleHalf, float centerDepth, float3 centerNormal, float2 centerMaterial) {
    uint2 fullPixel = fullPixelForHalf(sampleHalf);
    float sampleDepth = gBufferDepth[fullPixel];
    float3 sampleNormalRaw = gBufferNormal[fullPixel].xyz;
    if (!isFinite3(sampleNormalRaw) || length(sampleNormalRaw) <= 0.001) {
        return false;
    }
    float3 sampleNormal = normalize(sampleNormalRaw);
    float4 sampleMotionMaterial = gBufferMotionMaterial[fullPixel];
    float2 sampleMaterial = sampleMotionMaterial.zw;
    float depthTolerance = max(0.05, centerDepth * 0.03);
    return sampleDepth > 0.0 &&
           abs(sampleDepth - centerDepth) <= depthTolerance &&
           dot(centerNormal, sampleNormal) >= 0.80 &&
           all(abs(sampleMaterial - centerMaterial) < float2(0.5, 0.5));
}

float makeReflectionHistoryKey(float depth, float3 normal, float2 material) {
    uint depthKey = uint(clamp(depth * 0.5, 0.0, 31.0));
    uint materialKey = (uint(abs(material.x) + 0.5) ^ (uint(abs(material.y) + 0.5) * 17u)) & 7u;
    uint normalKey = (normal.z >= 0.0 ? 1u : 0u) | ((normal.y >= 0.0 ? 1u : 0u) << 1u);
    uint packedKey = (depthKey << 5u) | (materialKey << 2u) | normalKey;
    return float(packedKey) / 1023.0;
}

bool reflectionHistoryKeyMatches(float storedKey, float currentKey) {
    return abs(storedKey - currentKey) <= (1.5 / 1023.0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint2 pixel = tid.xy;
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    float4 raw = reflectionImages[pixel];
    if (push.enabled == 0u) {
        filteredReflectionImages[pixel] = raw;
        return;
    }

    uint2 fullPixel = fullPixelForHalf(pixel);
    float centerDepth = gBufferDepth[fullPixel];
    float3 centerNormalRaw = gBufferNormal[fullPixel].xyz;
    if (centerDepth <= 0.0 || !isFinite3(centerNormalRaw) || length(centerNormalRaw) <= 0.001) {
        filteredReflectionImages[pixel] = float4(0.0);
        return;
    }

    float3 centerNormal = normalize(centerNormalRaw);
    float2 centerMaterial = gBufferMotionMaterial[fullPixel].zw;
    float currentKey = makeReflectionHistoryKey(centerDepth, centerNormal, centerMaterial);

    float3 sum = float3(0.0);
    float weightSum = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            int2 coord = int2(pixel) + int2(x, y);
            if (any(coord < int2(0)) || coord.x >= int(push.width) || coord.y >= int(push.height)) {
                continue;
            }
            uint2 samplePixel = uint2(coord);
            if (!consistentWithCenter(samplePixel, centerDepth, centerNormal, centerMaterial)) {
                continue;
            }

            float spatial = exp(-dot(float2(x, y), float2(x, y)) * 0.35);
            float4 sampleValue = reflectionImages[samplePixel];
            float varianceWeight = 1.0 / (1.0 + max(sampleValue.a, 0.0) * 4.0);
            float weight = spatial * varianceWeight;
            sum += sampleValue.rgb * weight;
            weightSum += weight;
        }
    }

    float3 spatialReconstruction = weightSum > 0.0001 ? sum / weightSum : raw.rgb;

    float4 previousFiltered = filteredReflectionImages[pixel];
    bool historyUsable = push.resetHistory == 0u &&
                         isFinite3(previousFiltered.rgb) &&
                         reflectionHistoryKeyMatches(previousFiltered.a, currentKey);
    float historyAlpha = 0.16;
    float3 filtered = historyUsable ? lerp(previousFiltered.rgb, spatialReconstruction, historyAlpha) : spatialReconstruction;
    filteredReflectionImages[pixel] = float4(clampLuminance(filtered, SURFEL_PT_MAX_RADIANCE_LUMINANCE), currentKey);
}
