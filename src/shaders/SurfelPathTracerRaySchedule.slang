#include "SurfelPathTracerCommon.slang"

[[vk::binding(4, 0)]] RWByteAddressBuffer counters;
[[vk::binding(5, 0)]] RWStructuredBuffer<SurfelPathTracerSurfel> surfelBuffer;
[[vk::binding(10, 0)]] RWStructuredBuffer<uint> rayBuffer;
[[vk::binding(11, 0)]] StructuredBuffer<SurfelPathTracerCellInfo> cellInfoBuffer;
[[vk::binding(13, 0)]] StructuredBuffer<uint> cellToSurfelBuffer;

[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

struct RaySchedulePushConstants {
    uint cellCount;
    uint maxSurfels;
    uint maxRays;
    uint minRaysPerSurfel;
    uint maxRaysPerSurfel;
    float varianceSensitivity;
    uint offscreenRayInterval;   // 1 = trace off-screen surfels every frame
    float supportRadius;         // projected margin for the frustum test
};

[[vk::push_constant]] RaySchedulePushConstants push;

// Conservative view-frustum test for a surfel's support sphere. Returns true when
// no part of the sphere can be visible: fully behind the camera, or outside the
// NDC square by more than the sphere's projected radius.
bool isSurfelOutsideFrustum(float3 position, float radius)
{
    float4 viewPos = mul(ubo.view, float4(position, 1.0));
    float4 clip = mul(ubo.proj, viewPos);
    if (clip.w <= radius)
    {
        // At or behind the near plane (clip.w is the view-space depth).
        return clip.w < -radius;
    }
    float2 ndc = clip.xy / clip.w;
    // Projected radius in NDC: focal length (proj[0][0], proj[1][1]) times r / depth.
    float2 margin = abs(float2(ubo.proj[0][0], ubo.proj[1][1])) * radius / clip.w;
    return any(abs(ndc) > 1.0 + margin);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= push.cellCount) {
        return;
    }

    // Ray records are claimed first-come from one counter, so when demand
    // exceeds the budget the cells dispatched last always lose. Rotate which
    // cell each lane handles by a per-frame offset so budget starvation is
    // spread across the whole window instead of pinned to the highest z-slabs.
    uint frameIndex = counters.Load(SURFEL_PT_COUNTER_FRAME_INDEX_OFFSET);
    uint rotation = pcgHash(frameIndex ^ 0x2C1B3C6Du) % max(push.cellCount, 1u);
    // Budget scaling from last frame's demand (see PREVIOUS_DEMANDED_RAYS).
    // Requests never drop below minRaysPerSurfel; if even that exceeds the
    // budget, the rotated first-come clamp below still applies.
    uint previousDemand = counters.Load(SURFEL_PT_COUNTER_PREVIOUS_DEMANDED_RAYS_OFFSET);
    float budgetScale = previousDemand > push.maxRays
                        ? float(push.maxRays) / float(previousDemand)
                        : 1.0;
    uint cellIndex = tid.x + rotation;
    if (cellIndex >= push.cellCount) {
        cellIndex -= push.cellCount;
    }

    SurfelPathTracerCellInfo cellInfo = cellInfoBuffer[cellIndex];
    for (uint slot = 0u; slot < cellInfo.surfelCount; ++slot) {
        uint surfelIndex = unpackCellSurfelIndex(cellToSurfelBuffer[cellInfo.surfelOffset + slot]);
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

        // This pass has uniquely claimed a surfel that survived compact-map
        // selection. Track representation here instead of relying on one
        // arbitrary resolve pixel to mark the surfel as seen.
        uint previousSeenFrame = 0u;
        InterlockedMax(surfelBuffer[surfelIndex].lastSeenFrame,
                       frameIndex,
                       previousSeenFrame);
        surfel.lastSeenFrame = frameIndex;
        uint rayRequestCount = adaptiveRayCountForSurfel(surfel,
                                                         push.minRaysPerSurfel,
                                                         push.maxRaysPerSurfel,
                                                         push.varianceSensitivity);
        rayRequestCount = max(push.minRaysPerSurfel,
                              uint(round(float(rayRequestCount) * budgetScale)));
        if (rayRequestCount == 0u || push.maxRays == 0u) {
            continue;
        }

        // Off-screen surfels stay represented (lastSeenFrame above) but only trace
        // every offscreenRayInterval-th frame, staggered by a stable per-surfel hash
        // so the skipped set is spread evenly across frames. Surfels still in warm-up
        // always trace: they have no usable estimate yet.
        bool inWarmup = surfel.shortMeanAndLife.w <= 16.0;
        if (push.offscreenRayInterval > 1u && !inWarmup &&
            isSurfelOutsideFrustum(surfel.position, max(surfel.radius, push.supportRadius)))
        {
            uint phase = (pcgHash(surfelIndex ^ 0x68E31DA4u) + frameIndex) % push.offscreenRayInterval;
            if (phase != 0u) {
                continue;
            }
        }

        // Record the unclamped demand so the budget saturation ratio is observable.
        counters.InterlockedAdd(SURFEL_PT_COUNTER_DEMANDED_RAYS_OFFSET, rayRequestCount);

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
