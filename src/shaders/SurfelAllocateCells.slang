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
void surfelAllocateCellsMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint cellIndex = dispatchThreadID.x;
    if (cellIndex >= SURFEL_GI_CELL_COUNT)
        return;

    SurfelGiCell cell = surfelGiCells[cellIndex];
    cell.offset = 0xffffffffu;
    cell.writeCursor = 0u;
    cell.overflow = 0u;

    if (cell.count > 0u)
    {
        ptAnalysisCounters.InterlockedAdd(surfelGiCellNonEmptyOffset, 1u);
        ptAnalysisCounters.InterlockedMax(surfelGiCellMaxPopulationOffset, cell.count);

        uint allocatedOffset;
        surfelGiCounters.InterlockedAdd(4u, cell.count, allocatedOffset);
        if (allocatedOffset + cell.count <= SURFEL_GI_CELL_TO_SURFEL_CAPACITY)
        {
            cell.offset = allocatedOffset;
            ptAnalysisCounters.InterlockedAdd(surfelGiCellAllocatedMembershipsOffset, cell.count);
        }
        else
        {
            cell.overflow = cell.count;
            ptAnalysisCounters.InterlockedAdd(surfelGiCellAllocationOverflowOffset, cell.count);
        }
    }

    surfelGiCells[cellIndex] = cell;
}
