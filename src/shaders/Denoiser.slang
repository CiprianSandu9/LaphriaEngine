#include "ShaderCommon.slang"

// Denoiser descriptor set (Set 0) — all storage images.
[[vk::binding(0,  0)]] RWTexture2D<float4> noisyColor;       // raw RT output
[[vk::binding(1,  0)]] RWTexture2D<float4> gBufferNormals;     // G-Buffer normals (edge-stopping)
[[vk::binding(2,  0)]] RWTexture2D<float>  gBufferDepth;       // G-Buffer depth (edge-stopping)
[[vk::binding(7,  0)]] RWTexture2D<float2> historyMomentsOut;  // variance estimate (R=mean, G=mean²)
[[vk::binding(8,  0)]] RWTexture2D<float4> atrousTempA;        // A-Trous ping-pong buffer A
[[vk::binding(9,  0)]] RWTexture2D<float4> atrousTempB;        // A-Trous ping-pong buffer B
[[vk::binding(10, 0)]] RWTexture2D<float4> finalOutput;        // final denoised output (last pass only)
[[vk::binding(13, 0)]] RWTexture2D<float4> analysisReprojectionDebug;
[[vk::binding(14, 0)]] RWByteAddressBuffer ptAnalysisCounters;

struct DenoisePushConstants {
    int   stepSize;    // 1, 2, 4, 8, 16 for iterations 0-4
    int   isLastPass;  // 1 on iteration 4: write to finalOutput with tone mapping
    float phiColor;    // luminance edge-stopping weight (typical: 10.0)
    float phiNormal;   // normal edge-stopping exponent  (typical: 128.0)
    float exposureScale;
    int   useRawInput; // 1 if Reprojection was skipped
    int   debugAov;
    int   debugAtrousIteration;
};
[[vk::push_constant]] DenoisePushConstants push;

static const int DEBUG_AOV_PATH_RAW_FINAL_COLOR   = 6;
static const int DEBUG_AOV_PATH_DIRECT_LIGHTING   = 7;
static const int DEBUG_AOV_PATH_INDIRECT_LIGHTING = 8;
static const int DEBUG_AOV_PATH_SKY_CONTRIBUTION  = 9;
static const int DEBUG_AOV_PATH_THROUGHPUT        = 10;
static const int DEBUG_AOV_PATH_BOUNCE_COUNT      = 11;
static const int DEBUG_AOV_PATH_SHADOW_VISIBILITY = 12;
static const int DEBUG_AOV_PATH_ENVIRONMENT_NEE   = 13;
static const int DEBUG_AOV_PATH_FIRST_HIT_BOUNCE  = 14;
static const int DEBUG_AOV_PATH_SECONDARY_SUN     = 15;
static const int DEBUG_AOV_PATH_BASELINE_CONTINUATION = 16;
static const int DEBUG_AOV_PATH_RESERVOIR_GI_CONTRIBUTION = 17;
static const int DEBUG_AOV_PATH_RESERVOIR_GI_ACCEPTED_LUMA = 18;

float luminance(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

// A-Trous 5×5 kernel weights (separable gaussian approximation used in SVGF).
static const float kWeights[3] = { 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0 };

[shader("compute")]
[numthreads(16, 16, 1)]
void atrousMain(uint3 dispatchID : SV_DispatchThreadID)
{
    uint2 dims;
    atrousTempA.GetDimensions(dims.x, dims.y);
    uint2 pixel = dispatchID.xy;
    if (pixel.x >= dims.x || pixel.y >= dims.y) return;

    // Determine which ping-pong buffer is the input this iteration.
    // Iteration i: input is A if (i % 2 == 0), B if (i % 2 == 1).
    // stepSize tells us the iteration index: iter = log2(stepSize).
    int  iter  = (push.stepSize == 1) ? 0 :
                 (push.stepSize == 2) ? 1 :
                 (push.stepSize == 4) ? 2 :
                 (push.stepSize == 8) ? 3 : 4;
    bool readA = (iter % 2 == 0);

    if (push.stepSize == 0) {
        // Pass-through mode: denoiser disabled, just apply tonemap.
        float3 color = (push.useRawInput != 0) ? noisyColor[pixel].rgb : atrousTempA[pixel].rgb;
        if (push.isLastPass != 0) {
            float3 tonemapped = applyAcesTonemap(color, push.exposureScale);
            finalOutput[pixel] = float4(tonemapped, 1.0);
        }
        return;
    }

    float3 centerColor;
    if (push.useRawInput != 0 && iter == 0) {
        centerColor = noisyColor[pixel].rgb;
    } else {
        centerColor = readA ? atrousTempA[pixel].rgb : atrousTempB[pixel].rgb;
    }
    // Sky pixels write a zero normal (length = 0); guard against the NaN that normalize() would produce.
    float3 rawNorm    = gBufferNormals[pixel].xyz;
    float3 centerNorm  = (dot(rawNorm, rawNorm) > 0.0001) ? normalize(rawNorm) : float3(0.0, 0.0, 1.0);
    float  centerDepth = gBufferDepth[pixel];

    // Variance from temporal moments: var = E[x²] - E[x]²
    // Clamp to a minimum so the luminance edge-stopping (phiColor * sqrt(variance)) never
    // collapses to zero on the first frame when there is no temporal history yet.
    // Without this floor, all neighbor weights become ~0 and the A-Trous filter degenerates
    // into a pass-through, preserving raw 1-SPP salt-and-pepper noise.
    float variance;
    if (push.useRawInput != 0) {
        // Reprojection was skipped, so temporal moments are stale/undefined.
        // Use a deterministic floor variance to keep edge-stopping stable.
        variance = 0.01;
    } else {
        float2 moments = historyMomentsOut[pixel];
        variance = max(moments.y - moments.x * moments.x, 0.01);
    }
    float  centerLum = luminance(centerColor);

    float3 colorSum  = float3(0.0, 0.0, 0.0);
    float  weightSum = 0.0;

    // 5×5 A-Trous kernel (step size doubles each iteration for O(N log N) coverage).
    for (int dy = -2; dy <= 2; ++dy)
    {
        for (int dx = -2; dx <= 2; ++dx)
        {
            int2  samplePixel = int2(pixel) + int2(dx, dy) * push.stepSize;
            float kernelW     = kWeights[abs(dx)] * kWeights[abs(dy)];

            // Clamp to image boundary.
            samplePixel = clamp(samplePixel, int2(0, 0), int2(dims) - int2(1, 1));

            float3 sampleColor;
            if (push.useRawInput != 0 && iter == 0) {
                sampleColor = noisyColor[samplePixel].rgb;
            } else {
                sampleColor = readA ? atrousTempA[samplePixel].rgb : atrousTempB[samplePixel].rgb;
            }
            float3 rawSampleNorm = gBufferNormals[samplePixel].xyz;
            float3 sampleNorm    = (dot(rawSampleNorm, rawSampleNorm) > 0.0001) ? normalize(rawSampleNorm) : float3(0.0, 0.0, 1.0);
            float  sampleDepth   = gBufferDepth[samplePixel];

            // Normal edge-stopping: high power to sharply preserve geometry boundaries.
            float normalW = pow(max(dot(centerNorm, sampleNorm), 0.0), push.phiNormal);

            // Depth edge-stopping: exponential falloff on relative depth difference.
            float depthDiff = abs(centerDepth - sampleDepth) / max(abs(centerDepth), 0.001);
            float depthW    = exp(-depthDiff * 10.0);

            // Luminance edge-stopping: variance-guided (tighter in low-variance regions).
            float lumDiff  = abs(centerLum - luminance(sampleColor));
            float lumW     = exp(-lumDiff / max(push.phiColor * sqrt(variance) + 0.0001, 0.0001));

            float w = kernelW * normalW * depthW * lumW;
            colorSum  += sampleColor * w;
            weightSum += w;
        }
    }

    float3 filtered = colorSum / max(weightSum, 0.0001);

    if (push.isLastPass != 0) {
        float3 outColor = applyAcesTonemap(filtered, push.exposureScale);
        if (push.debugAov == 1) {
            float validity = analysisReprojectionDebug[pixel].r;
            outColor = float3(validity, validity, validity);
        } else if (push.debugAov == 2) {
            float alpha = analysisReprojectionDebug[pixel].g;
            outColor = float3(alpha, alpha, alpha);
        } else if (push.debugAov == 3) {
            float motionMag = saturate(analysisReprojectionDebug[pixel].b * 20.0);
            outColor = float3(motionMag, motionMag, motionMag);
        } else if (push.debugAov == 4) {
            float2 moments = historyMomentsOut[pixel];
            float variance = max(moments.y - moments.x * moments.x, 0.0);
            float v = saturate(variance * 8.0);
            outColor = float3(v, v, v);
        } else if (push.debugAov == 5) {
            outColor = applyAcesTonemap(filtered, push.exposureScale);
        } else if (push.debugAov == DEBUG_AOV_PATH_RAW_FINAL_COLOR ||
                   push.debugAov == DEBUG_AOV_PATH_DIRECT_LIGHTING ||
                   push.debugAov == DEBUG_AOV_PATH_INDIRECT_LIGHTING ||
                   push.debugAov == DEBUG_AOV_PATH_SKY_CONTRIBUTION ||
                   push.debugAov == DEBUG_AOV_PATH_ENVIRONMENT_NEE ||
                   push.debugAov == DEBUG_AOV_PATH_FIRST_HIT_BOUNCE ||
                   push.debugAov == DEBUG_AOV_PATH_SECONDARY_SUN ||
                   push.debugAov == DEBUG_AOV_PATH_BASELINE_CONTINUATION ||
                   push.debugAov == DEBUG_AOV_PATH_RESERVOIR_GI_CONTRIBUTION ||
                   push.debugAov == DEBUG_AOV_PATH_RESERVOIR_GI_ACCEPTED_LUMA) {
            float3 debugRaw = noisyColor[pixel].rgb;
            outColor = applyAcesTonemap(debugRaw, push.exposureScale);
        } else if (push.debugAov == DEBUG_AOV_PATH_THROUGHPUT ||
                   push.debugAov == DEBUG_AOV_PATH_BOUNCE_COUNT ||
                   push.debugAov == DEBUG_AOV_PATH_SHADOW_VISIBILITY) {
            outColor = saturate(noisyColor[pixel].rgb);
        }
        finalOutput[pixel] = float4(outColor, 1.0);
    } else {
        // Intermediate iteration: write to the OTHER ping-pong buffer.
        if (readA)
            atrousTempB[pixel] = float4(filtered, 1.0);
        else
            atrousTempA[pixel] = float4(filtered, 1.0);
    }
}
