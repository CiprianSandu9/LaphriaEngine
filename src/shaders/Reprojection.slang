#include "ShaderCommon.slang"

// Denoiser descriptor set (Set 0) — all bindings are storage images.
[[vk::binding(0,  0)]] RWTexture2D<float4> noisyColor;           // current noisy 1-SPP input
[[vk::binding(1,  0)]] RWTexture2D<float4> gBufferNormals;       // current G-Buffer normals
[[vk::binding(2,  0)]] RWTexture2D<float>  gBufferDepth;         // current G-Buffer depth
[[vk::binding(3,  0)]] RWTexture2D<float2> motionVectors;        // current motion vectors
[[vk::binding(4,  0)]] RWTexture2D<float4> historyColorIn;       // previous frame colour history (read)
[[vk::binding(5,  0)]] RWTexture2D<float4> historyColorOut;      // this frame colour history (write)
[[vk::binding(6,  0)]] RWTexture2D<float2> historyMomentsIn;     // previous frame moments (read)
[[vk::binding(7,  0)]] RWTexture2D<float2> historyMomentsOut;    // this frame moments (write)
[[vk::binding(8,  0)]] RWTexture2D<float4> atrousTempA;          // A-Trous ping-pong A (reprojection writes here)
// bindings 9, 10 unused in this shader
[[vk::binding(11, 0)]] RWTexture2D<float4> prevGBufferNormals;   // previous-frame G-Buffer normals
[[vk::binding(12, 0)]] RWTexture2D<float>  prevGBufferDepth;     // previous-frame G-Buffer depth

// Push constants (shared layout with A-Trous; stepSize == 0 identifies this pass)
struct DenoisePushConstants {
    int   stepSize;
    int   isLastPass;
    float phiColor;
    float phiNormal;
};
[[vk::push_constant]] DenoisePushConstants push;

float luminance(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

[shader("compute")]
[numthreads(16, 16, 1)]
void reprojectionMain(uint3 dispatchID : SV_DispatchThreadID)
{
    uint2 dims;
    noisyColor.GetDimensions(dims.x, dims.y);
    uint2 pixel = dispatchID.xy;
    if (pixel.x >= dims.x || pixel.y >= dims.y) return;

    float3 currentColor  = noisyColor[pixel].rgb;
    float3 currentNormal = gBufferNormals[pixel].xyz;
    float  currentDepth  = gBufferDepth[pixel];

    // ── Firefly suppression: clamp outlier luminance before temporal blend ───
    float currentLum = luminance(currentColor);
    float2 prevMomentsVal = historyMomentsIn[pixel];
    float  prevMean       = prevMomentsVal.x;
    float  prevVar        = max(prevMomentsVal.y - prevMean * prevMean, 0.0);
    float  sigma          = sqrt(prevVar);
    float  clampMax       = prevMean + 4.0 * sigma;
    if (prevVar > 0.001 && currentLum > clampMax)
        currentColor *= clampMax / max(currentLum, 0.0001);

    // ── Temporal reprojection ────────────────────────────────────────────────
    // mv = currentCenterUV - prevCenterUV, where centerUV = (pixel + 0.5) / dims.
    // Rearranged: prevPixelIndex = floor(pixel + 0.5 - mv*dims - 0.5) = floor(pixel - mv*dims).
    // The old code subtracted an extra 0.5 on top of the correct derivation, shifting
    // every lookup one pixel up-left and breaking accumulation even for a static camera.
    float2 mv        = motionVectors[pixel];
    int2   prevPixel = int2(float2(pixel) - mv * float2(dims));

    // Rejection heuristics: normals must align and depths must be close.
    bool   valid = all(prevPixel >= int2(0, 0)) && all(prevPixel < int2(dims));
    float3 histColor = float3(0.0, 0.0, 0.0);
    float  alpha     = 1.0;   // 1.0 = trust only current (no history)

    if (valid) {
        float3 prevNormal = prevGBufferNormals[prevPixel].xyz;
        float  prevDepth  = prevGBufferDepth[prevPixel];

        float normalDot   = dot(normalize(currentNormal), normalize(prevNormal));
        float depthRelErr = abs(currentDepth - prevDepth) / max(abs(currentDepth), 0.001);

        if (normalDot > 0.9 && depthRelErr < 0.1 && currentDepth > 0.0 && prevDepth > 0.0) {
            histColor = historyColorIn[prevPixel].rgb;
            // push.phiColor carries historyAlpha from C++:
            //   0.1 when camera is static  → blend 10% current / 90% history for fast convergence
            //   1.0 when camera moved      → discard history entirely to prevent ghosting bands
            alpha = push.phiColor;
        }
    }

    float3 blended = lerp(histColor, currentColor, alpha);

    // ── Temporal moments (mean and mean²) for variance estimation ────────────
    float blendedLum = luminance(blended);
    float2 prevMom   = valid ? historyMomentsIn[prevPixel] : float2(0.0, 0.0);
    float  newMean   = lerp(prevMom.x, blendedLum, alpha);
    float  newMean2  = lerp(prevMom.y, blendedLum * blendedLum, alpha);

    // ── Write outputs ────────────────────────────────────────────────────────
    atrousTempA[pixel]        = float4(blended, 1.0);   // A-Trous input for first iteration
    historyColorOut[pixel]    = float4(blended, 1.0);   // history for next frame
    historyMomentsOut[pixel]  = float2(newMean, newMean2);
}
