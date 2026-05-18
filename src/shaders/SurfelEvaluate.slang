#include "SurfelCommon.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<SurfelGiRecord> surfelGiRecords;
[[vk::binding(1, 0)]] RWStructuredBuffer<SurfelGiCell> surfelGiCells;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> surfelGiCellToSurfel;
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

bool acceptsSurfelCandidate(SurfelGiRecord record, float3 receiverPos, float3 receiverNormal)
{
    if ((record.flags & 1u) == 0u)
        return false;

    float3 surfelNormal = normalize(record.normalConfidence.xyz);
    if (dot(receiverNormal, surfelNormal) < 0.25f)
        return false;

    float radius = clamp(record.positionRadius.w, SURFEL_GI_MIN_RADIUS, SURFEL_GI_MAX_RADIUS) * 4.0f;
    float3 toReceiver = receiverPos - record.positionRadius.xyz;
    return dot(toReceiver, toReceiver) <= radius * radius;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void surfelEvaluateMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;
    const uint width = push.renderWidth;
    const uint height = push.renderHeight;

    if (pixel.x >= width || pixel.y >= height)
        return;

    ptAnalysisCounters.InterlockedAdd(surfelGiEvalAttemptsOffset, 1u);

    float depth = gBufferDepthRead[pixel];
    float3 normal = gBufferNormalsRead[pixel].xyz;
    if (depth <= 0.0f || length(normal) < 0.5f)
    {
        surfelGiDebug[pixel] = float4(0.0f, 0.0f, 0.0f, 1.0f);
        return;
    }

    float3 receiverNormal = normalize(normal);
    float3 receiverPos = reconstructSurfelGiFirstHitWorldPosition(
        pixel, uint2(width, height), depth, ubo.cameraPos.xyz, ubo.projInverse, ubo.viewInverse);

    int3 gridCell = calcSurfelGiCell(receiverPos, ubo.cameraPos.xyz);
    if (!isSurfelGiCellValid(gridCell))
    {
        ptAnalysisCounters.InterlockedAdd(surfelGiEvalCellEmptyOffset, 1u);
        surfelGiDebug[pixel] = float4(0.0f, 0.0f, 0.0f, 1.0f);
        return;
    }

    uint cellIndex = flattenSurfelGiCell(gridCell);
    SurfelGiCell cell = surfelGiCells[cellIndex];
    if (cell.count == 0u || cell.offset == 0xffffffffu)
    {
        ptAnalysisCounters.InterlockedAdd(surfelGiEvalCellEmptyOffset, 1u);
        surfelGiDebug[pixel] = float4(0.0f, 0.0f, 0.0f, 1.0f);
        return;
    }

    if (cell.count > SURFEL_GI_DENSE_CELL_LIMIT)
    {
        ptAnalysisCounters.InterlockedAdd(surfelGiEvalDenseCellOffset, 1u);
        ptAnalysisCounters.InterlockedAdd(surfelGiEvalDenseCellSkippedOffset, 1u);
        surfelGiDebug[pixel] = float4(0.0f, 0.5f, 1.0f, 1.0f);
        return;
    }

    uint configuredCandidateCount = clamp(push.maxEvalCandidates, 1u, SURFEL_GI_MAX_EVAL_CANDIDATES);
    uint boundedCandidateCount = min(cell.count, configuredCandidateCount);
    uint accepted = 0u;
    for (uint slot = 0u; slot < boundedCandidateCount; ++slot)
    {
        uint memberIndex = slot;
        if (cell.count > boundedCandidateCount)
        {
            memberIndex = pcgHash(pixel.x * 73856093u ^
                                  pixel.y * 19349663u ^
                                  ubo.frameCount * 83492791u ^
                                  slot * 2654435761u);
            memberIndex = memberIndex % cell.count;
        }

        uint membershipIndex = cell.offset + memberIndex;
        if (membershipIndex < cell.offset || membershipIndex >= SURFEL_GI_CELL_TO_SURFEL_CAPACITY)
            continue;

        uint surfelIndex = surfelGiCellToSurfel[cell.offset + memberIndex];
        if (surfelIndex == 0xffffffffu || surfelIndex >= SURFEL_GI_MAX_SURFELS)
            continue;

        ptAnalysisCounters.InterlockedAdd(surfelGiEvalCandidatesOffset, 1u);
        SurfelGiRecord record = surfelGiRecords[surfelIndex];
        if (acceptsSurfelCandidate(record, receiverPos, receiverNormal))
            ++accepted;
    }

    if (accepted > 0u)
        ptAnalysisCounters.InterlockedAdd(surfelGiEvalAcceptedOffset, accepted);

    float acceptedRatio = float(accepted) / max(float(boundedCandidateCount), 1.0f);
    float occupancyRatio = float(boundedCandidateCount) / max(float(configuredCandidateCount), 1.0f);
    surfelGiDebug[pixel] = float4(saturate(acceptedRatio), 0.0f, saturate(occupancyRatio), 1.0f);
}
