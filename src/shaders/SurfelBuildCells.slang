#include "SurfelCommon.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<SurfelGiRecord> surfelGiRecords;
[[vk::binding(1, 0)]] RWStructuredBuffer<SurfelGiCell> surfelGiCells;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> surfelGiCellSlots;
[[vk::binding(3, 0)]] RWByteAddressBuffer surfelGiCounters;
[[vk::binding(4, 0)]] RWByteAddressBuffer ptAnalysisCounters;
[[vk::binding(5, 0)]] RWTexture2D<float4> surfelGiDebug;
[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

[shader("compute")]
[numthreads(128, 1, 1)]
void surfelBuildCellsMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint surfelIndex = dispatchThreadID.x;
    if (surfelIndex >= SURFEL_GI_MAX_SURFELS)
        return;

    SurfelGiRecord record = surfelGiRecords[surfelIndex];
    if ((record.flags & 1u) == 0u)
        return;

    ptAnalysisCounters.InterlockedAdd(surfelGiCellInsertAttemptsOffset, 1u);

    const int3 gridCell = calcSurfelGiCell(record.positionRadius.xyz, ubo.cameraPos.xyz);
    if (!isSurfelGiCellValid(gridCell))
    {
        ptAnalysisCounters.InterlockedAdd(surfelGiCellOverflowOffset, 1u);
        return;
    }

    const uint cellIndex = flattenSurfelGiCell(gridCell);
    uint slotIndex;
#define cell surfelGiCells[cellIndex]
    InterlockedAdd(cell.count, 1u, slotIndex);

    if (slotIndex >= SURFEL_GI_CELL_SLOT_COUNT)
    {
        InterlockedAdd(cell.overflow, 1u);
        ptAnalysisCounters.InterlockedAdd(surfelGiCellOverflowOffset, 1u);
        return;
    }
#undef cell

    surfelGiCellSlots[cellIndex * SURFEL_GI_CELL_SLOT_COUNT + slotIndex] = surfelIndex;
    record.cellIndex = cellIndex;
    surfelGiRecords[surfelIndex] = record;

    ptAnalysisCounters.InterlockedAdd(surfelGiCellInsertedOffset, 1u);
}
