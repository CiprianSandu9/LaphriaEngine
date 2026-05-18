#include "SurfelPathTracerCommon.slang"

[[vk::binding(4, 0)]] RWByteAddressBuffer counters;
[[vk::binding(5, 0)]] RWStructuredBuffer<SurfelPathTracerSurfel> surfelBuffer;
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> aliveBuffer;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> deadBuffer;
[[vk::binding(8, 0)]] RWStructuredBuffer<uint> dirtyBuffer;
[[vk::binding(9, 0)]] RWStructuredBuffer<uint> recycleBuffer;
[[vk::binding(10, 0)]] RWStructuredBuffer<uint> rayBuffer;
[[vk::binding(11, 0)]] RWStructuredBuffer<SurfelPathTracerCellInfo> cellInfoBuffer;
[[vk::binding(12, 0)]] RWByteAddressBuffer cellCounterBuffer;
[[vk::binding(13, 0)]] RWStructuredBuffer<uint> cellToSurfelBuffer;

struct PreparePushConstants {
    uint maxSurfels;
    uint resetPersistent;
    uint cellCount;
    uint perCellSurfelLimit;
};

[[vk::push_constant]] PreparePushConstants push;

static const uint COUNTER_ALIVE_SURFELS_OFFSET = 0u;
static const uint COUNTER_DEAD_SURFELS_OFFSET = 4u;
static const uint COUNTER_DIRTY_SURFELS_OFFSET = 8u;
static const uint COUNTER_REQUESTED_RAYS_OFFSET = 12u;
static const uint COUNTER_FILLED_CELLS_OFFSET = 16u;
static const uint COUNTER_REJECTED_STORES_OFFSET = 20u;
static const uint COUNTER_FRAME_INDEX_OFFSET = 24u;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint index = tid.x;
    uint maxSurfels = max(push.maxSurfels, 1u);
    uint cellCount = push.cellCount;
    uint perCellSurfelLimit = max(push.perCellSurfelLimit, 1u);
    uint cellToSurfelCount = cellCount * perCellSurfelLimit;
    uint cellCounterCount = 1u + cellCount * 2u;

    if (index == 0u) {
        counters.Store(COUNTER_DIRTY_SURFELS_OFFSET, 0u);
        counters.Store(COUNTER_REQUESTED_RAYS_OFFSET, 0u);
        counters.Store(COUNTER_FILLED_CELLS_OFFSET, 0u);
        counters.Store(COUNTER_REJECTED_STORES_OFFSET, 0u);
        counters.InterlockedAdd(COUNTER_FRAME_INDEX_OFFSET, 1u);

        if (push.resetPersistent != 0u) {
            counters.Store(COUNTER_ALIVE_SURFELS_OFFSET, 0u);
            counters.Store(COUNTER_DEAD_SURFELS_OFFSET, maxSurfels);
        }
    }

    if (index < cellCounterCount) {
        cellCounterBuffer.Store(index * 4u, 0u);
    }

    if (index < cellCount) {
        SurfelPathTracerCellInfo cellInfo;
        cellInfo.surfelOffset = 0u;
        cellInfo.surfelCount = 0u;
        cellInfoBuffer[index] = cellInfo;
    }

    if (index < cellToSurfelCount) {
        cellToSurfelBuffer[index] = SURFEL_PT_INVALID_INDEX;
    }

    if (push.resetPersistent == 0u) {
        return;
    }

    if (index < maxSurfels) {
        aliveBuffer[index] = SURFEL_PT_INVALID_INDEX;
        deadBuffer[index] = index;
        dirtyBuffer[index] = SURFEL_PT_INVALID_INDEX;

        SurfelPathTracerSurfel surfel;
        surfel.position = float3(0.0f, 0.0f, 0.0f);
        surfel.radius = 0.0f;
        surfel.radiance = float3(0.0f, 0.0f, 0.0f);
        surfel.packedNormal = 0u;
        surfel.rayOffset = 0u;
        surfel.rayCount = 0u;
        surfel.irradianceAtlasOffset = 0u;
        surfel.flags = 0;
        surfel.meanAndVariance = float4(0.0f, 0.0f, 0.0f, 0.0f);
        surfel.shortMeanAndLife = float4(0.0f, 0.0f, 0.0f, 0.0f);
        surfelBuffer[index] = surfel;
    }

    uint recycleCount = maxSurfels * 4u;
    if (index < recycleCount) {
        recycleBuffer[index] = SURFEL_PT_INVALID_INDEX;
    }

}
