#include "SurfelCommon.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<SurfelGiRecord> surfelGiRecords;
[[vk::binding(1, 0)]] RWStructuredBuffer<SurfelGiCell> surfelGiCells;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> surfelGiCellSlots;
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
        cell.overflow = 0u;
        cell.pad0 = 0u;
        cell.pad1 = 0u;
        surfelGiCells[index] = cell;

        const uint slotBase = index * SURFEL_GI_CELL_SLOT_COUNT;
        for (uint slot = 0u; slot < SURFEL_GI_CELL_SLOT_COUNT; ++slot)
        {
            surfelGiCellSlots[slotBase + slot] = 0xffffffffu;
        }
    }
}
