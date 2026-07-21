#include "SurfelPathTracerCommon.slang"

[[vk::binding(0, 0)]] RWTexture2D<float4> outputImage;
[[vk::binding(1, 0)]] RWTexture2D<float4> gBufferNormal;
[[vk::binding(2, 0)]] RWTexture2D<float> gBufferDepth;
[[vk::binding(3, 0)]] RWTexture2D<float4> gBufferMotionMaterial;
[[vk::binding(18, 0)]] RWTexture2D<float4> lightingImages;
[[vk::binding(25, 0)]] RWTexture2D<float4> previousTaaHistory;
[[vk::binding(26, 0)]] RWTexture2D<float4> currentTaaHistory;

[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

struct TaaPushConstants {
    uint width;
    uint height;
    uint enabled;
    uint resetHistory;
    uint frameIndex;
    uint pad0;
    uint pad1;
    uint pad2;
};

[[vk::push_constant]] TaaPushConstants push;

void neighborhoodBounds(uint2 pixel, out float3 lo, out float3 hi) {
    lo = float3(1e20);
    hi = float3(-1e20);
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            int2 coord = int2(pixel) + int2(x, y);
            if (any(coord < int2(0)) || coord.x >= int(push.width) || coord.y >= int(push.height)) {
                continue;
            }
            float3 sampleValue = lightingImages[uint2(coord)].rgb;
            lo = min(lo, sampleValue);
            hi = max(hi, sampleValue);
        }
    }
}

float3 primaryRayDirection(uint2 pixel) {
    float2 uv = (float2(pixel) + float2(0.5, 0.5)) / float2(max(push.width, 1u), max(push.height, 1u));
    float2 d = uv * 2.0 - 1.0;
    float4 target = mul(ubo.projInverse, float4(d.x, -d.y, 1.0, 1.0));
    return normalize(mul(ubo.viewInverse, float4(normalize(target.xyz / target.w), 0.0)).xyz);
}

float3 reconstructWorldPosition(uint2 pixel, float depth) {
    return ubo.cameraPos.xyz + primaryRayDirection(pixel) * depth;
}

bool reprojectWithPrevViewProj(uint2 pixel, float depth, float2 motion, out float2 previousUv) {
    float3 worldPosition = reconstructWorldPosition(pixel, depth);
    float4 previousClip = mul(ubo.prevViewProj, float4(worldPosition, 1.0));
    if (abs(previousClip.w) <= 0.0001) {
        previousUv = float2(-1.0);
        return false;
    }

    previousUv = (previousClip.xy / previousClip.w) * float2(0.5, -0.5) + 0.5;
    float2 uv = (float2(pixel) + 0.5) / float2(push.width, push.height);
    float2 motionUv = uv - motion;
    float agreementPixels = length((previousUv - motionUv) * float2(push.width, push.height));
    return all(previousUv >= float2(0.0)) &&
           all(previousUv <= float2(1.0)) &&
           agreementPixels <= 2.0;
}

float makeTaaHistoryKey(float depth, float4 motionMaterial, float3 normal) {
    uint depthKey = uint(clamp(depth * 0.5, 0.0, 31.0));
    uint materialKey = (uint(abs(motionMaterial.z) + 0.5) ^ (uint(abs(motionMaterial.w) + 0.5) * 17u)) & 7u;
    uint normalKey = (normal.z >= 0.0 ? 1u : 0u) | ((normal.y >= 0.0 ? 1u : 0u) << 1u);
    uint motionKey = uint(clamp(length(motionMaterial.xy) * 128.0, 0.0, 1.0));
    uint packedKey = (depthKey << 6u) | (materialKey << 3u) | (normalKey << 1u) | motionKey;
    return float(packedKey) / 1023.0;
}

bool taaHistoryKeyMatches(float storedKey, float currentKey) {
    return abs(storedKey - currentKey) <= (1.5 / 1023.0);
}

void accumulateTaaHistoryTap(int2 pixel,
                             float weight,
                             float currentKey,
                             inout float3 historySum,
                             inout float validWeight) {
    if (weight <= 0.0 ||
        any(pixel < int2(0)) ||
        pixel.x >= int(push.width) ||
        pixel.y >= int(push.height)) {
        return;
    }

    float4 sampleValue = previousTaaHistory[uint2(pixel)];
    if (!taaHistoryKeyMatches(sampleValue.a, currentKey)) {
        return;
    }

    historySum += sampleValue.rgb * weight;
    validWeight += weight;
}

bool sampleReprojectedTaaHistory(float2 previousUv,
                                 float currentKey,
                                 out float3 history) {
    // Convert normalized UVs to texel-center space before selecting the four
    // bilinear taps. Truncating previousUv * dimensions to one texel makes the
    // history jump at pixel boundaries during sub-pixel camera motion, which
    // presents as a subtle wave or shimmer over otherwise stable geometry.
    float2 texelPosition = previousUv * float2(push.width, push.height) - 0.5;
    int2 basePixel = int2(floor(texelPosition));
    float2 fraction = frac(texelPosition);

    float3 historySum = float3(0.0);
    float validWeight = 0.0;
    accumulateTaaHistoryTap(basePixel,
                            (1.0 - fraction.x) * (1.0 - fraction.y),
                            currentKey,
                            historySum,
                            validWeight);
    accumulateTaaHistoryTap(basePixel + int2(1, 0),
                            fraction.x * (1.0 - fraction.y),
                            currentKey,
                            historySum,
                            validWeight);
    accumulateTaaHistoryTap(basePixel + int2(0, 1),
                            (1.0 - fraction.x) * fraction.y,
                            currentKey,
                            historySum,
                            validWeight);
    accumulateTaaHistoryTap(basePixel + int2(1, 1),
                            fraction.x * fraction.y,
                            currentKey,
                            historySum,
                            validWeight);

    if (validWeight <= 1.0e-4) {
        history = float3(0.0);
        return false;
    }

    // Renormalizing valid taps prevents a rejected foreground/background tap
    // from darkening the reconstructed history near disocclusions.
    history = historySum / validWeight;
    return true;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint2 pixel = tid.xy;
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    float3 currentLighting = max(lightingImages[pixel].rgb, float3(0.0));
    if (push.enabled == 0u) {
        outputImage[pixel] = float4(applyAcesTonemap(currentLighting, ubo.exposure), 1.0);
        return;
    }

    float3 history = currentLighting;
    float depth = gBufferDepth[pixel];
    float4 motionMaterial = gBufferMotionMaterial[pixel];
    float3 normalRaw = gBufferNormal[pixel].xyz;
    bool validNormal = isFinite3(normalRaw) && length(normalRaw) > 0.001;
    float3 normal = validNormal ? normalize(normalRaw) : float3(0.0, 1.0, 0.0);
    float currentKey = makeTaaHistoryKey(depth, motionMaterial, normal);
    bool useHistory = push.resetHistory == 0u &&
                      depth > 0.0 &&
                      validNormal &&
                      motionMaterial.z >= 0.0 &&
                      motionMaterial.w >= 0.0;

    if (useHistory) {
        float2 previousUv;
        if (reprojectWithPrevViewProj(pixel, depth, motionMaterial.xy, previousUv)) {
            float3 lo;
            float3 hi;
            neighborhoodBounds(pixel, lo, hi);
            float3 reconstructedHistory;
            if (sampleReprojectedTaaHistory(previousUv, currentKey, reconstructedHistory)) {
                history = clamp(reconstructedHistory, lo, hi);
            } else {
                useHistory = false;
            }
        } else {
            useHistory = false;
        }
    }

    float cameraMotion = saturate(length(motionMaterial.xy) * 64.0);
    float alpha = lerp(0.08, 0.45, cameraMotion);
    float3 blended = useHistory ? lerp(history, currentLighting, alpha) : currentLighting;

    currentTaaHistory[pixel] = float4(blended, currentKey);
    outputImage[pixel] = float4(applyAcesTonemap(blended, ubo.exposure), 1.0);
}
