#include "SurfelCommon.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<SurfelGiRecord> surfelGiRecords;
[[vk::binding(1, 0)]] RWStructuredBuffer<SurfelGiCell> surfelGiCells;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> surfelGiCellToSurfel;
[[vk::binding(3, 0)]] RWByteAddressBuffer surfelGiCounters;
[[vk::binding(4, 0)]] RWByteAddressBuffer ptAnalysisCounters;
[[vk::binding(5, 0)]] RWTexture2D<float4> surfelGiDebug;

[shader("compute")]
[numthreads(128, 1, 1)]
void surfelClearMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;

    if (index == 0u)
    {
        surfelGiCounters.Store(0u, 0u);
        surfelGiCounters.Store(4u, 0u);
        surfelGiCounters.Store(8u, 0u);
        surfelGiCounters.Store(12u, 0u);
        ptAnalysisCounters.InterlockedAdd(surfelGiClearDispatchesOffset, 1u);
    }

    if (index < SURFEL_GI_CELL_COUNT)
    {
        SurfelGiCell cell;
        cell.count = 0u;
        cell.offset = 0xffffffffu;
        cell.writeCursor = 0u;
        cell.overflow = 0u;
        surfelGiCells[index] = cell;
    }

    if (index < SURFEL_GI_CELL_TO_SURFEL_CAPACITY)
    {
        surfelGiCellToSurfel[index] = 0xffffffffu;
    }
}
