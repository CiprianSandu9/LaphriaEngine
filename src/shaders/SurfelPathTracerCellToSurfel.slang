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

uint reservoirSlotForCellSurfel(uint surfelIndex, uint cellIndex, uint cellSlot, uint cellCapacity)
{
    if (cellCapacity == 0u)
    {
        return SURFEL_PT_INVALID_INDEX;
    }

    uint streamLength = cellSlot + 1u;
    uint candidate = pcgHash(surfelIndex ^ (cellIndex * 1664525u) ^ cellSlot) % streamLength;
    return candidate < cellCapacity ? candidate : SURFEL_PT_INVALID_INDEX;
}

uint dominantNormalBucket(float3 normal)
{
    float3 n = normalize(normal);
    float3 a = abs(n);
    if (a.x >= a.y && a.x >= a.z)
    {
        return n.x >= 0.0 ? 0u : 1u;
    }
    if (a.y >= a.z)
    {
        return n.y >= 0.0 ? 2u : 3u;
    }
    return n.z >= 0.0 ? 4u : 5u;
}

uint representativeSlotForCellSurfel(SurfelPathTracerSurfel surfel,
                                     int3 cellCoord,
                                     float3 cameraPosition,
                                     float cellSize,
                                     uint cellCapacity)
{
    if (cellCapacity == 0u)
    {
        return SURFEL_PT_INVALID_INDEX;
    }

    float safeCellSize = max(cellSize, 0.0001);
    float3 cellCenter = float3(cellCoord) * safeCellSize + cameraPosition;
    float3 cellMin = cellCenter - float3(safeCellSize * 0.5);
    float3 localPosition = saturate((surfel.position - cellMin) / safeCellSize);
    uint3 localBucket = uint3(localPosition.x >= 0.5 ? 1u : 0u,
                              localPosition.y >= 0.5 ? 1u : 0u,
                              localPosition.z >= 0.5 ? 1u : 0u);
    uint spatialBucket = localBucket.x + localBucket.y * 2u + localBucket.z * 4u;
    uint normalBucket = dominantNormalBucket(unpackNormalOctahedral(surfel.packedNormal));
    uint bucket = spatialBucket + normalBucket * 8u;
    return bucket % cellCapacity;
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

        uint cursorOffset = (1u + push.cellDimension * push.cellDimension * push.cellDimension + cellIndex) * 4u;
        uint cellSlot = 0u;
        cellCounterBuffer.InterlockedAdd(cursorOffset, 1u, cellSlot);

        if (cellSlot < cellInfo.surfelCount) {
            cellToSurfelBuffer[cellInfo.surfelOffset + cellSlot] = surfelIndex;
        } else {
            uint replaceSlot = reservoirSlotForCellSurfel(surfelIndex, cellIndex, cellSlot, cellInfo.surfelCount);
            if (replaceSlot == SURFEL_PT_INVALID_INDEX) {
                counters.InterlockedAdd(SURFEL_PT_COUNTER_REJECTED_STORES_OFFSET, 1u);
            } else {
                uint representativeSlot = representativeSlotForCellSurfel(surfel,
                                                                          cellCoord,
                                                                          ubo.cameraPos.xyz,
                                                                          push.cellSize,
                                                                          cellInfo.surfelCount);
                replaceSlot = representativeSlot;
                cellToSurfelBuffer[cellInfo.surfelOffset + replaceSlot] = surfelIndex;
            }
        }
    }
}
