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
};

[[vk::push_constant]] CellToSurfelPushConstants push;

uint cellIndexForPosition(float3 position, float cellSize, uint cellDimension) {
    float safeCellSize = max(cellSize, 0.0001f);
    uint dim = max(cellDimension, 1u);
    float halfDim = float(dim) * 0.5f;
    float maxCoord = float(dim - 1u);
    float3 centeredCell = floor(position / safeCellSize) + float3(halfDim, halfDim, halfDim);
    float3 clampedCell = clamp(centeredCell, float3(0.0f, 0.0f, 0.0f), float3(maxCoord, maxCoord, maxCoord));
    uint3 coord = uint3(clampedCell);
    return coord.x + coord.y * dim + coord.z * dim * dim;
}

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

    uint cellIndex = cellIndexForPosition(surfel.position, push.cellSize, push.cellDimension);
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
