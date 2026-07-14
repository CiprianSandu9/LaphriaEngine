#include "SurfelPathTracerCommon.slang"

[[vk::binding(4, 0)]] RWByteAddressBuffer counters;
[[vk::binding(5, 0)]] RWStructuredBuffer<SurfelPathTracerSurfel> surfelBuffer;
[[vk::binding(10, 0)]] RWStructuredBuffer<uint> rayBuffer;
[[vk::binding(11, 0)]] StructuredBuffer<SurfelPathTracerCellInfo> cellInfoBuffer;
[[vk::binding(13, 0)]] StructuredBuffer<uint> cellToSurfelBuffer;

struct RaySchedulePushConstants {
    uint cellCount;
    uint maxSurfels;
    uint maxRays;
    uint minRaysPerSurfel;
    uint maxRaysPerSurfel;
    float varianceSensitivity;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] RaySchedulePushConstants push;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint cellIndex = tid.x;
    if (cellIndex >= push.cellCount) {
        return;
    }

    SurfelPathTracerCellInfo cellInfo = cellInfoBuffer[cellIndex];
    for (uint slot = 0u; slot < cellInfo.surfelCount; ++slot) {
        uint surfelIndex = cellToSurfelBuffer[cellInfo.surfelOffset + slot];
        if (surfelIndex == SURFEL_PT_INVALID_INDEX || surfelIndex >= push.maxSurfels) {
            continue;
        }

        // A surfel can overlap multiple cells. Atomically claim it so the final
        // compact cell map schedules each represented surfel exactly once.
        uint previousFlags = 0u;
        InterlockedOr(surfelBuffer[surfelIndex].flags,
                      SURFEL_PT_SURFEL_FLAG_RAY_SCHEDULED,
                      previousFlags);
        if ((previousFlags & SURFEL_PT_SURFEL_FLAG_RAY_SCHEDULED) != 0u) {
            continue;
        }

        SurfelPathTracerSurfel surfel = surfelBuffer[surfelIndex];
        if (!isActiveSurfel(surfel)) {
            continue;
        }

        uint frameIndex = counters.Load(SURFEL_PT_COUNTER_FRAME_INDEX_OFFSET);
        // This pass has uniquely claimed a surfel that survived compact-map
        // selection. Track representation here instead of relying on one
        // arbitrary resolve pixel to mark the surfel as seen.
        surfel.lastSeenFrame = frameIndex;
        uint rayRequestCount = adaptiveRayCountForSurfel(surfel,
                                                         push.minRaysPerSurfel,
                                                         push.maxRaysPerSurfel,
                                                         push.varianceSensitivity,
                                                         frameIndex);
        if (rayRequestCount == 0u || push.maxRays == 0u) {
            continue;
        }

        uint rayOffset = 0u;
        counters.InterlockedAdd(SURFEL_PT_COUNTER_REQUESTED_RAYS_OFFSET,
                                rayRequestCount,
                                rayOffset);
        uint writableRayCount = rayOffset < push.maxRays
                                    ? min(rayRequestCount, push.maxRays - rayOffset)
                                    : 0u;
        if (writableRayCount < rayRequestCount) {
            counters.InterlockedMin(SURFEL_PT_COUNTER_REQUESTED_RAYS_OFFSET, push.maxRays);
        }
        if (writableRayCount == 0u) {
            continue;
        }

        surfel.rayOffset = rayOffset;
        surfel.rayCount = writableRayCount;
        surfel.flags |= SURFEL_PT_SURFEL_FLAG_RAY_SCHEDULED;
        surfelBuffer[surfelIndex] = surfel;
        for (uint ray = 0u; ray < writableRayCount; ++ray) {
            uint rayIndex = rayOffset + ray;
            uint base = rayIndex * SURFEL_PT_RAY_RECORD_UINTS;
            rayBuffer[base + SURFEL_PT_RAY_SURFEL_INDEX] = surfelIndex;
            rayBuffer[base + SURFEL_PT_RAY_FLAGS] = 0u;
        }

        counters.InterlockedMax(SURFEL_PT_INDIRECT_RAY_WIDTH_OFFSET,
                                rayOffset + writableRayCount);
    }
}
