#include "SurfelPathTracerCommon.slang"

[[vk::binding(4, 0)]] RWByteAddressBuffer counters;
[[vk::binding(5, 0)]] RWStructuredBuffer<SurfelPathTracerSurfel> surfelBuffer;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> deadBuffer;
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

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint index = tid.x;
    uint maxSurfels = max(push.maxSurfels, 1u);
    uint cellCount = push.cellCount;
    uint perCellSurfelLimit = max(push.perCellSurfelLimit, 1u);
    uint cellCounterCount = 1u + cellCount * 2u;

    if (index == 0u) {
        counters.Store(SURFEL_PT_COUNTER_DIRTY_SURFELS_OFFSET, 0u);
        counters.Store(SURFEL_PT_COUNTER_REQUESTED_RAYS_OFFSET, 0u);
        counters.Store(SURFEL_PT_COUNTER_FILLED_CELLS_OFFSET, 0u);
        counters.Store(SURFEL_PT_COUNTER_REJECTED_STORES_OFFSET, 0u);
        counters.Store(SURFEL_PT_COUNTER_RECYCLED_SURFELS_OFFSET, 0u);
        counters.Store(SURFEL_PT_COUNTER_SPAWNED_SURFELS_OFFSET, 0u);
        counters.Store(SURFEL_PT_COUNTER_REMOVED_SURFELS_OFFSET, 0u);
        counters.Store(SURFEL_PT_COUNTER_GUIDED_RAYS_OFFSET, 0u);
        counters.Store(SURFEL_PT_COUNTER_COSINE_RAYS_OFFSET, 0u);
        counters.Store(SURFEL_PT_COUNTER_SURFEL_TERMINATION_ATTEMPTS_OFFSET, 0u);
        counters.Store(SURFEL_PT_COUNTER_PATH_MISSES_OFFSET, 0u);
		counters.Store(SURFEL_PT_COUNTER_SURFEL_TERMINATION_HITS_OFFSET, 0u);
		counters.Store(SURFEL_PT_COUNTER_DEMANDED_RAYS_OFFSET, 0u);
		counters.Store(SURFEL_PT_INDIRECT_RAY_WIDTH_OFFSET, 0u);
		counters.Store(SURFEL_PT_INDIRECT_RAY_HEIGHT_OFFSET, 1u);
		counters.Store(SURFEL_PT_INDIRECT_RAY_DEPTH_OFFSET, 1u);
        if (push.resetPersistent != 0u) {
            // Start every cache generation from the same sampling sequence.
            // Lifecycle ages are then local to the current cache as well.
            counters.Store(SURFEL_PT_COUNTER_FRAME_INDEX_OFFSET, 0u);
            counters.Store(SURFEL_PT_COUNTER_ALIVE_SURFELS_OFFSET, 0u);
            counters.Store(SURFEL_PT_COUNTER_DEAD_SURFELS_OFFSET, maxSurfels);
        } else {
            counters.InterlockedAdd(SURFEL_PT_COUNTER_FRAME_INDEX_OFFSET, 1u);
        }
    }

    if (index < cellCounterCount) {
        cellCounterBuffer.Store(index * 4u, 0u);
    }

    if (push.resetPersistent != 0u && index < cellCount) {
        SurfelPathTracerCellInfo cellInfo;
        cellInfo.surfelOffset = 0u;
        cellInfo.surfelCount = 0u;
        cellInfoBuffer[index] = cellInfo;
    }

    if (push.resetPersistent == 0u) {
        return;
    }

    if (index < maxSurfels) {
        deadBuffer[index] = index;

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
        surfel.lastSeenFrame = 0u;
        surfel.lastReferencedFrame = 0u;
        surfel.sleepState = SURFEL_PT_SLEEP_AWAKE;
        surfel.materialKey = 0u;
        surfel.varianceAndInconsistency = float4(1.0, 1.0, 1.0, 1.0);
        surfelBuffer[index] = surfel;
    }
}
