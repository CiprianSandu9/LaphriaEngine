#include "SurfelPathTracerCommon.slang"

[[vk::binding(1, 0)]] RWTexture2D<float4> gBufferNormal;
[[vk::binding(2, 0)]] RWTexture2D<float> gBufferDepth;
[[vk::binding(16, 0)]] RWTexture2D<float4> reflectionImages;
[[vk::binding(17, 0)]] RWTexture2D<float4> filteredReflectionImages;

struct BilateralPushConstants {
    uint width;
    uint height;
    uint fullWidth;
    uint fullHeight;
    uint enabled;
    uint pad0;
    uint pad1;
    uint pad2;
};

[[vk::push_constant]] BilateralPushConstants push;

uint2 fullPixelForHalf(uint2 halfPixel) {
    return min(halfPixel * 2u + uint2(1u, 1u), uint2(push.fullWidth - 1u, push.fullHeight - 1u));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint2 pixel = tid.xy;
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    float4 centerValue = filteredReflectionImages[pixel];
    if (push.enabled == 0u) {
        return;
    }

    uint2 fullPixel = fullPixelForHalf(pixel);
    float centerDepth = gBufferDepth[fullPixel];
    float3 centerNormalRaw = gBufferNormal[fullPixel].xyz;
    if (centerDepth <= 0.0 || !isFinite3(centerNormalRaw) || length(centerNormalRaw) <= 0.001) {
        reflectionImages[pixel] = float4(0.0);
        return;
    }

    float3 centerNormal = normalize(centerNormalRaw);
    float3 sum = float3(0.0);
    float weightSum = 0.0;
    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            int2 coord = int2(pixel) + int2(x, y);
            if (any(coord < int2(0)) || coord.x >= int(push.width) || coord.y >= int(push.height)) {
                continue;
            }

            uint2 samplePixel = uint2(coord);
            uint2 sampleFullPixel = fullPixelForHalf(samplePixel);
            float sampleDepth = gBufferDepth[sampleFullPixel];
            float3 sampleNormalRaw = gBufferNormal[sampleFullPixel].xyz;
            if (sampleDepth <= 0.0 || !isFinite3(sampleNormalRaw) || length(sampleNormalRaw) <= 0.001) {
                continue;
            }

            float3 sampleNormal = normalize(sampleNormalRaw);
            float spatialWeight = exp(-dot(float2(x, y), float2(x, y)) * 0.22);
            float depthWeight = exp(-abs(sampleDepth - centerDepth) / max(centerDepth * 0.02, 0.03));
            float normalWeight = pow(saturate(dot(centerNormal, sampleNormal)), 16.0);
            float4 sampleValue = filteredReflectionImages[samplePixel];
            float weight = spatialWeight * depthWeight * normalWeight;
            sum += sampleValue.rgb * weight;
            weightSum += weight;
        }
    }

    if (weightSum > 0.0001) {
        float3 filtered = sum / weightSum;
        reflectionImages[pixel] = float4(clampLuminance(filtered, SURFEL_PT_MAX_RADIANCE_LUMINANCE), centerValue.a);
    } else {
        reflectionImages[pixel] = centerValue;
    }
}
