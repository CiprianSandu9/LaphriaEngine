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

void insertDeterministicOverflowSurfel(uint cellEntry,
                                       SurfelPathTracerCellInfo cellInfo)
{
    uint candidate = cellEntry;
    for (uint slot = 0u; slot < cellInfo.surfelCount; ++slot)
    {
        uint previous = SURFEL_PT_INVALID_INDEX;
        InterlockedMin(cellToSurfelBuffer[cellInfo.surfelOffset + slot],
                       candidate,
                       previous);

        if (previous == SURFEL_PT_INVALID_INDEX)
        {
            return;
        }
        if (previous > candidate)
        {
            // Continue inserting the entry displaced by the smaller stable key.
            candidate = previous;
        }
    }

    counters.InterlockedAdd(SURFEL_PT_COUNTER_REJECTED_STORES_OFFSET, 1u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint surfelIndex = tid.x;
    if (surfelIndex >= push.maxSurfels) {
        return;
    }

    SurfelPathTracerSurfel surfel = surfelBuffer[surfelIndex];
    if (!isActiveSurfel(surfel)) {
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

        int3 absoluteCellCoord = cellCoord + worldCellCoord(ubo.cameraPos.xyz, push.cellSize);
        uint cellEntry = packCellSurfelEntry(surfelIndex, absoluteCellCoord);
        uint rawCount = cellCounterBuffer.Load((1u + cellIndex) * 4u);
        if (rawCount <= cellInfo.surfelCount)
        {
            uint cursorOffset = (1u + push.cellDimension * push.cellDimension * push.cellDimension + cellIndex) * 4u;
            uint cellSlot = 0u;
            cellCounterBuffer.InterlockedAdd(cursorOffset, 1u, cellSlot);
            uint previousSurfel = SURFEL_PT_INVALID_INDEX;
            InterlockedExchange(cellToSurfelBuffer[cellInfo.surfelOffset + cellSlot],
                                cellEntry,
                                previousSurfel);
        }
        else
        {
            // Overflowed cells retain the same stable hashed sample for any GPU
            // invocation order without biasing retention toward old/low IDs.
            insertDeterministicOverflowSurfel(cellEntry, cellInfo);
        }
    }
}
