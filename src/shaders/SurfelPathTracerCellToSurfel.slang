#include "SurfelPathTracerCommon.slang"

[[vk::binding(4, 0)]] RWByteAddressBuffer counters;
[[vk::binding(5, 0)]] RWStructuredBuffer<SurfelPathTracerSurfel> surfelBuffer;
[[vk::binding(11, 0)]] RWStructuredBuffer<SurfelPathTracerCellInfo> cellInfoBuffer;
[[vk::binding(12, 0)]] RWByteAddressBuffer cellCounterBuffer;
[[vk::binding(13, 0)]] RWStructuredBuffer<uint> cellToSurfelBuffer;

struct CellToSurfelPushConstants {
    uint maxSurfels;
    float cellSize;
    uint cellDimension;
    uint perCellSurfelLimit;
    float4 cameraPosition;
};

[[vk::push_constant]] CellToSurfelPushConstants push;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint surfelIndex = tid.x;
    if (surfelIndex >= push.maxSurfels) {
        return;
    }

    SurfelPathTracerSurfel surfel = surfelBuffer[surfelIndex];
    if (surfel.flags == 0u || surfel.radius <= 0.0f || !isFinite3(surfel.position)) {
        return;
    }

    uint cellIndex = cameraRelativeCellIndexForPosition(surfel.position,
                                                        push.cameraPosition.xyz,
                                                        push.cellSize,
                                                        push.cellDimension);
    if (cellIndex == SURFEL_PT_INVALID_INDEX) {
        return;
    }
    SurfelPathTracerCellInfo cellInfo = cellInfoBuffer[cellIndex];
    if (cellInfo.surfelCount == 0u) {
        counters.InterlockedAdd(SURFEL_PT_COUNTER_REJECTED_STORES_OFFSET, 1u);
        return;
    }

    uint cursorOffset = (1u + push.cellDimension * push.cellDimension * push.cellDimension + cellIndex) * 4u;
    uint cellSlot = 0u;
    cellCounterBuffer.InterlockedAdd(cursorOffset, 1u, cellSlot);

    if (cellSlot < cellInfo.surfelCount) {
        cellToSurfelBuffer[cellInfo.surfelOffset + cellSlot] = surfelIndex;
    } else {
        counters.InterlockedAdd(SURFEL_PT_COUNTER_REJECTED_STORES_OFFSET, 1u);
    }
}
