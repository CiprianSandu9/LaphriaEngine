#include "SurfelCommon.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<SurfelGiRecord> surfelGiRecords;
[[vk::binding(1, 0)]] RWStructuredBuffer<SurfelGiCell> surfelGiCells;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> surfelGiCellSlots;
[[vk::binding(3, 0)]] RWByteAddressBuffer surfelGiCounters;
[[vk::binding(4, 0)]] RWByteAddressBuffer ptAnalysisCounters;
[[vk::binding(5, 0)]] RWTexture2D<float4> surfelGiDebug;
[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

bool surfelIntersectsCell(SurfelGiRecord record, int3 cell, float3 cameraPos)
{
    const float3 gridOrigin = cameraPos -
        float3(SURFEL_GI_GRID_DIM, SURFEL_GI_GRID_DIM, SURFEL_GI_GRID_DIM) *
        (0.5f * SURFEL_GI_CELL_SIZE);
    const float3 cellMin = gridOrigin + float3(cell) * SURFEL_GI_CELL_SIZE;
    const float3 cellMax = cellMin + float3(SURFEL_GI_CELL_SIZE, SURFEL_GI_CELL_SIZE, SURFEL_GI_CELL_SIZE);
    const float radius = clamp(record.positionRadius.w, SURFEL_GI_MIN_RADIUS, SURFEL_GI_MAX_RADIUS);
    const float3 closestPoint = clamp(record.positionRadius.xyz, cellMin, cellMax);
    const float3 delta = record.positionRadius.xyz - closestPoint;
    return dot(delta, delta) <= radius * radius;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void surfelCountCellsMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint surfelIndex = dispatchThreadID.x;
    if (surfelIndex >= SURFEL_GI_MAX_SURFELS)
        return;

    SurfelGiRecord record = surfelGiRecords[surfelIndex];
    if ((record.flags & 1u) == 0u)
        return;

    const int3 centerCell = calcSurfelGiCell(record.positionRadius.xyz, ubo.cameraPos.xyz);
    for (int z = -1; z <= 1; ++z)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int x = -1; x <= 1; ++x)
            {
                const int3 cell = centerCell + int3(x, y, z);
                if (!isSurfelGiCellValid(cell) ||
                    !surfelIntersectsCell(record, cell, ubo.cameraPos.xyz))
                {
                    continue;
                }

                const uint cellIndex = flattenSurfelGiCell(cell);
                uint previousCount;
                InterlockedAdd(surfelGiCells[cellIndex].count, 1u, previousCount);
                ptAnalysisCounters.InterlockedAdd(surfelGiCellTotalMembershipsOffset, 1u);
            }
        }
    }
}
