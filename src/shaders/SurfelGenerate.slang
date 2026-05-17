#include "SurfelCommon.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<SurfelGiRecord> surfelGiRecords;
[[vk::binding(1, 0)]] RWStructuredBuffer<SurfelGiCell> surfelGiCells;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> surfelGiCellSlots;
[[vk::binding(3, 0)]] RWByteAddressBuffer surfelGiCounters;
[[vk::binding(4, 0)]] RWByteAddressBuffer ptAnalysisCounters;
[[vk::binding(5, 0)]] RWTexture2D<float4> surfelGiDebug;
[[vk::binding(6, 0)]] RWTexture2D<float4> gBufferNormalsRead;
[[vk::binding(7, 0)]] RWTexture2D<float> gBufferDepthRead;
[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

struct SurfelGiPushConstants
{
    uint renderWidth;
    uint renderHeight;
    uint maxEvalCandidates;
    uint pad1;
};
[[vk::push_constant]] SurfelGiPushConstants push;

[shader("compute")]
[numthreads(8, 8, 1)]
void surfelGenerateMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;
    const uint width = push.renderWidth;
    const uint height = push.renderHeight;

    if (pixel.x >= width || pixel.y >= height)
        return;

    float depth = gBufferDepthRead[pixel];
    float3 normal = gBufferNormalsRead[pixel].xyz;
    ptAnalysisCounters.InterlockedAdd(surfelGiGenerateAttemptsOffset, 1u);

    if (depth <= 0.0f || length(normal) < 0.5f)
    {
        ptAnalysisCounters.InterlockedAdd(surfelGiGenerateRejectInvalidOffset, 1u);
        return;
    }

    if (((pixel.x + pixel.y) & 15u) != 0u)
    {
        ptAnalysisCounters.InterlockedAdd(surfelGiGenerateRejectCoverageOffset, 1u);
        return;
    }

    uint surfelIndex;
    surfelGiCounters.InterlockedAdd(0u, 1u, surfelIndex);
    if (surfelIndex >= SURFEL_GI_MAX_SURFELS)
        return;

    float3 worldPos = reconstructSurfelGiFirstHitWorldPosition(
        pixel, uint2(width, height), depth, ubo.cameraPos.xyz, ubo.projInverse, ubo.viewInverse);

    SurfelGiRecord record;
    record.positionRadius = float4(worldPos, SURFEL_GI_MIN_RADIUS);
    record.normalConfidence = float4(normalize(normal), 0.1f);
    record.radianceAge = float4(0.0f, 0.0f, 0.0f, 0.0f);
    record.flags = 1u;
    record.lastSeenFrame = ubo.frameCount;
    record.cellIndex = 0xffffffffu;
    record.pad0 = 0u;
    surfelGiRecords[surfelIndex] = record;

    ptAnalysisCounters.InterlockedAdd(surfelGiGeneratedOffset, 1u);
}
