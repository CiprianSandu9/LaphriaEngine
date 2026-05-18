#include "SurfelPathTracerCommon.slang"

[[vk::binding(4, 0)]] RWByteAddressBuffer counters;
[[vk::binding(5, 0)]] RWStructuredBuffer<SurfelPathTracerSurfel> surfelBuffer;
[[vk::binding(11, 0)]] RWStructuredBuffer<SurfelPathTracerCellInfo> cellInfoBuffer;
[[vk::binding(12, 0)]] RWByteAddressBuffer cellCounterBuffer;
[[vk::binding(13, 0)]] RWStructuredBuffer<uint> cellToSurfelBuffer;
[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

struct CellToSurfelPushConstants {
    uint maxSurfels;
    float cellSize;
    uint cellDimension;
    uint perCellSurfelLimit;
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

    int3 centerCoord = cameraRelativeCellCoord(surfel.position, ubo.cameraPos.xyz, push.cellSize);
    for (int z = -1; z <= 1; ++z)
    for (int y = -1; y <= 1; ++y)
    for (int x = -1; x <= 1; ++x)
    {
        int3 cellCoord = centerCoord + int3(x, y, z);
        if (!isCellCoordValid(cellCoord, push.cellDimension) ||
            !isSurfelIntersectCell(surfel, cellCoord, ubo.cameraPos.xyz, push.cellSize))
        {
            continue;
        }

        uint cellIndex = flattenCameraRelativeCellCoord(cellCoord, push.cellDimension);
        SurfelPathTracerCellInfo cellInfo = cellInfoBuffer[cellIndex];
        if (cellInfo.surfelCount == 0u) {
            counters.InterlockedAdd(SURFEL_PT_COUNTER_REJECTED_STORES_OFFSET, 1u);
            continue;
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
}
