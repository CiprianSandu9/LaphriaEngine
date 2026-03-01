#include "ShaderCommon.slang"

// Set 0 = computeDescriptorSetLayout: one storage image, written by the starfield compute shader.
[[vk::binding(0, 0)]]
RWTexture2D<float4> resultImage;

[[vk::push_constant]]
ScenePushConstants push;

// ============================================================================
// Starfield helpers
// ============================================================================

float Noise2d(float2 x) {
    float xhash = cos(x.x * 37.0);
    float yhash = cos(x.y * 57.0);
    return fract(415.92653 * (xhash + yhash));
}

float NoisyStarField(float2 vSamplePos, float fThreshhold) {
    float StarVal = Noise2d(vSamplePos);
    if (StarVal >= fThreshhold)
        StarVal = pow((StarVal - fThreshhold) / (1.0 - fThreshhold), 6.0);
    else
        StarVal = 0.0;
    return StarVal;
}

// Bilinearly interpolate four neighbouring noise samples so stars don't shimmer
// when the window is resized or the sample position drifts by a sub-pixel amount.
float StableStarField(float2 vSamplePos, float fThreshhold) {
    float fractX = fract(vSamplePos.x);
    float fractY = fract(vSamplePos.y);
    float2 floorSample = floor(vSamplePos);
    float v1 = NoisyStarField(floorSample, fThreshhold);
    float v2 = NoisyStarField(floorSample + float2(0.0, 1.0), fThreshhold);
    float v3 = NoisyStarField(floorSample + float2(1.0, 0.0), fThreshhold);
    float v4 = NoisyStarField(floorSample + float2(1.0, 1.0), fThreshhold);

    float StarVal = v1 * (1.0 - fractX) * (1.0 - fractY)
                  + v2 * (1.0 - fractX) * fractY
                  + v3 * fractX * (1.0 - fractY)
                  + v4 * fractX * fractY;
    return StarVal;
}

// ============================================================================
// Compute entry point — writes the starfield background to resultImage.
// Workgroup size 16×16 matches the C++ dispatch calculation (groupCount = ceil(dim/16)).
// ============================================================================

[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint2 texelCoord = dispatchThreadID.xy;
    uint2 size;
    resultImage.GetDimensions(size.x, size.y);

    if (texelCoord.x < size.x && texelCoord.y < size.y) {
        float2 iResolution = float2(size);
        float2 fragCoord = float2(texelCoord);

        // Sky gradient: push.skyData.xyz = sky color, push.skyData.w = star density threshold.
        float3 vColor = push.skyData.xyz * fragCoord.y / iResolution.y;
        float StarFieldThreshhold = push.skyData.w;

        float xRate = 0.2;
        float yRate = -0.06;
        float2 vSamplePos = fragCoord + float2(xRate * 1.0, yRate * 1.0); // time = 1.0 for now
        float StarVal = StableStarField(vSamplePos, StarFieldThreshhold);
        vColor += float3(StarVal);

        resultImage[texelCoord] = float4(vColor, 1.0);
    }
}
