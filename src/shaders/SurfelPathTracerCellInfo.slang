#include "SurfelPathTracerCommon.slang"

[[vk::binding(4, 0)]] RWByteAddressBuffer counters;
[[vk::binding(11, 0)]] RWStructuredBuffer<SurfelPathTracerCellInfo> cellInfoBuffer;
[[vk::binding(12, 0)]] RWByteAddressBuffer cellCounterBuffer;

struct CellInfoPushConstants {
    uint cellCount;
    uint perCellSurfelLimit;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] CellInfoPushConstants push;

static const uint CELL_RUNNING_TOTAL_OFFSET = 0u;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint cellIndex = tid.x;
    if (cellIndex == 0u) {
        counters.Store(SURFEL_PT_INDIRECT_RAY_WIDTH_OFFSET,
                       counters.Load(SURFEL_PT_COUNTER_REQUESTED_RAYS_OFFSET));
        counters.Store(SURFEL_PT_INDIRECT_RAY_HEIGHT_OFFSET, 1u);
        counters.Store(SURFEL_PT_INDIRECT_RAY_DEPTH_OFFSET, 1u);
    }
    if (cellIndex >= push.cellCount) {
        return;
    }

    uint countOffset = (1u + cellIndex) * 4u;
    uint cursorOffset = (1u + push.cellCount + cellIndex) * 4u;
    uint rawCount = cellCounterBuffer.Load(countOffset);
    uint clampedCount = min(rawCount, max(push.perCellSurfelLimit, 1u));

    uint surfelOffset = 0u;
    cellCounterBuffer.InterlockedAdd(CELL_RUNNING_TOTAL_OFFSET, clampedCount, surfelOffset);
    cellCounterBuffer.Store(cursorOffset, 0u);

    SurfelPathTracerCellInfo cellInfo;
    cellInfo.surfelOffset = surfelOffset;
    cellInfo.surfelCount = clampedCount;
    cellInfoBuffer[cellIndex] = cellInfo;

    if (clampedCount > 0u) {
        counters.InterlockedAdd(SURFEL_PT_COUNTER_FILLED_CELLS_OFFSET, 1u);
    }
}
