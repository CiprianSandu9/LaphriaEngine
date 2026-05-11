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
[[vk::binding(13, 0)]] RWTexture2D<float4> analysisReprojectionDebug; // R=valid, G=alpha, B=motion magnitude
[[vk::binding(14, 0)]] RWByteAddressBuffer ptAnalysisCounters;

// Push constants (shared layout with A-Trous; stepSize == 0 identifies this pass)
struct DenoisePushConstants {
    int   stepSize;
    int   isLastPass;
    float phiColor;
    float phiNormal;
    float exposureScale;
    int   useRawInput;
    int   debugAov;
    int   debugAtrousIteration;
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
    ptAnalysisCounters.InterlockedAdd(0, 1u); // pixel count

    // ── Firefly suppression: clamp outlier luminance before temporal blend ───
    float currentLum = luminance(currentColor);
    float2 prevMomentsVal = historyMomentsIn[pixel];
    float  prevMean       = prevMomentsVal.x;
    float  prevVar        = max(prevMomentsVal.y - prevMean * prevMean, 0.0);
    float  sigma          = sqrt(prevVar);
    float  clampMax       = prevMean + 4.0 * sigma;
    if (prevVar > 0.001 && currentLum > clampMax)
    {
        currentColor *= clampMax / max(currentLum, 0.0001);
        ptAnalysisCounters.InterlockedAdd(12, 1u); // firefly clamp hits
    }

    // ── Temporal reprojection ────────────────────────────────────────────────
    // mv = currentCenterUV - prevCenterUV, where centerUV = (pixel + 0.5) / dims.
    // Rearranged: prevCenterInPixels = pixel + 0.5 - mv*dims.
    // Convert center-space back to integer texel index with floor(centerInPixels).
    // Using trunc(pixel - mv*dims) biases lookups by up to 1 texel for subpixel motion,
    // which appears as watery blur / crawling in temporally accumulated details.
    float2 mv        = motionVectors[pixel];
    int2   prevPixel = int2(floor(float2(pixel) + 0.5 - mv * float2(dims)));

    // Rejection heuristics: normals must align and depths must be close.
    bool   valid = all(prevPixel >= int2(0, 0)) && all(prevPixel < int2(dims));
    float3 histColor = float3(0.0, 0.0, 0.0);
    float  alpha     = 1.0;   // 1.0 = trust only current (no history)

    if (valid) {
        float3 prevNormal = prevGBufferNormals[prevPixel].xyz;
        float  prevDepth  = prevGBufferDepth[prevPixel];

        float3 currN      = (dot(currentNormal, currentNormal) > 0.0001) ? normalize(currentNormal) : float3(0.0, 0.0, 1.0);
        float3 prevN      = (dot(prevNormal, prevNormal) > 0.0001) ? normalize(prevNormal) : float3(0.0, 0.0, 1.0);
        float normalDot   = dot(currN, prevN);
        float depthRelErr = abs(currentDepth - prevDepth) / max(abs(currentDepth), 0.001);

        if (normalDot > 0.9 && depthRelErr < 0.1 && currentDepth > 0.0 && prevDepth > 0.0) {
            histColor = historyColorIn[prevPixel].rgb;
            // Conservative neighborhood clamp on reprojected history to limit ghost trails.
            float minLum = 1e20;
            float maxLum = -1e20;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int2 samplePixel = clamp(int2(pixel) + int2(dx, dy), int2(0, 0), int2(dims) - int2(1, 1));
                    float sampleLum = luminance(noisyColor[samplePixel].rgb);
                    minLum = min(minLum, sampleLum);
                    maxLum = max(maxLum, sampleLum);
                }
            }

            float histLum = luminance(histColor);
            float centerLum = 0.5 * (minLum + maxLum);
            float baseHalfRange = max(0.5 * (maxLum - minLum) * 1.1, 0.0001);
            float temporalHalfRange = min(max(prevMean, 0.0) + 2.0 * sigma, 0.08);
            float halfRange = max(baseHalfRange, temporalHalfRange);
            float clampedHistLum = clamp(histLum, centerLum - halfRange, centerLum + halfRange);
            if (histLum > 0.0001) {
                histColor *= clampedHistLum / histLum;
            }

            // push.phiColor carries historyAlpha from C++:
            // lower alpha = stronger history reuse, higher alpha = more current-frame trust.
            alpha = push.phiColor;
            ptAnalysisCounters.InterlockedAdd(4, 1u); // history accepted
        } else {
            ptAnalysisCounters.InterlockedAdd(8, 1u); // history rejected
        }
    } else {
        ptAnalysisCounters.InterlockedAdd(8, 1u); // history rejected
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
    analysisReprojectionDebug[pixel] = float4(valid ? 1.0 : 0.0, alpha, length(mv), 1.0);
}
