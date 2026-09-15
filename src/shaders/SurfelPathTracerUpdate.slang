#include "SurfelPathTracerCommon.slang"

[[vk::binding(4, 0)]] RWByteAddressBuffer counters;
[[vk::binding(5, 0)]] RWStructuredBuffer<SurfelPathTracerSurfel> surfelBuffer;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> deadBuffer;
[[vk::binding(10, 0)]] RWStructuredBuffer<uint> rayBuffer;
[[vk::binding(11, 0)]] StructuredBuffer<SurfelPathTracerCellInfo> cellInfoBuffer;
[[vk::binding(12, 0)]] RWByteAddressBuffer cellCounterBuffer;
[[vk::binding(29, 0)]] RWStructuredBuffer<SurfelPathTracerSource> surfelSourceBuffer;
[[vk::binding(31, 0)]] StructuredBuffer<SurfelPathTracerSourceTransform> sourceTransformBuffer;
[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

struct UpdatePushConstants {
    uint maxSurfels;
    float cellSize;
    uint cellDimension;
    uint maxRays;
    uint lockSurfels;
    uint minRaysPerSurfel;
    uint maxRaysPerSurfel;
    float varianceSensitivity;
    uint sourceTransformCount;
    uint perCellSurfelLimit;
    uint pad1;
    uint pad2;
};

[[vk::push_constant]] UpdatePushConstants push;

// Surfels that are unreferenced for this long while their cell is saturated are
// dead weight: they cannot enter the compact map ahead of referenced entries and
// nothing else reclaims them below the global pressure threshold. Foliage cells
// otherwise only ever grow, and the frame time never recovers without a reset.
static const uint SURFEL_PT_OVERFLOW_EVICTION_FRAMES = 240u;

bool isStrandedInSaturatedCell(SurfelPathTracerSurfel surfel, int3 centerCoord, uint frameIndex)
{
    if (!isCellCoordValid(centerCoord, push.cellDimension))
    {
        return false;
    }
    // cellInfoBuffer still holds last frame's map at this point (CellInfo runs
    // after Update). A saturated cell reports surfelCount == limit.
    uint cellIndex = flattenCameraRelativeCellCoord(centerCoord, push.cellDimension);
    if (cellInfoBuffer[cellIndex].surfelCount < max(push.perCellSurfelLimit, 1u))
    {
        return false;
    }
    uint ageSinceRef = frameIndex >= surfel.lastReferencedFrame ? frameIndex - surfel.lastReferencedFrame : 0u;
    return ageSinceRef > SURFEL_PT_OVERFLOW_EVICTION_FRAMES;
}

bool refreshAnchoredSurfel(uint surfelIndex, inout SurfelPathTracerSurfel surfel)
{
    SurfelPathTracerSource source = surfelSourceBuffer[surfelIndex];
    if ((source.sourceFlags & SURFEL_PT_SOURCE_FLAG_VALID) == 0u)
    {
        return false;
    }

    // Only store the failure flag on the transition; rewriting the whole source
    // record every frame for every unanchored surfel is pure bandwidth.
    bool alreadyFailed = (source.sourceFlags & SURFEL_PT_SOURCE_FLAG_REFRESH_FAILED) != 0u;
    if (source.sourceNodeId >= push.sourceTransformCount)
    {
        if (!alreadyFailed)
        {
            source.sourceFlags |= SURFEL_PT_SOURCE_FLAG_REFRESH_FAILED;
            surfelSourceBuffer[surfelIndex] = source;
        }
        return false;
    }

    SurfelPathTracerSourceTransform transform = sourceTransformBuffer[source.sourceNodeId];
    if ((transform.flags & SURFEL_PT_SOURCE_FLAG_VALID) == 0u)
    {
        if (!alreadyFailed)
        {
            source.sourceFlags |= SURFEL_PT_SOURCE_FLAG_REFRESH_FAILED;
            surfelSourceBuffer[surfelIndex] = source;
        }
        return false;
    }

    float4 worldPosition = mul(transform.objectToWorld, float4(source.sourceObjectPosition, 1.0));
    float3 objectNormal = unpackNormalOctahedral(source.sourceObjectNormal);
    float3 worldNormal = normalize(mul(transpose((float3x3)transform.worldToObject), objectNormal));

    surfel.position = worldPosition.xyz;
    surfel.packedNormal = packNormalOctahedral(worldNormal);
    surfel.materialKey = source.sourceMaterialKey;

    if ((source.sourceFlags & SURFEL_PT_SOURCE_FLAG_REFRESH_FAILED) != 0u)
    {
        source.sourceFlags &= ~SURFEL_PT_SOURCE_FLAG_REFRESH_FAILED;
        surfelSourceBuffer[surfelIndex] = source;
    }

    return true;
}

bool pushDeadSurfel(uint surfelIndex, uint maxSurfels)
{
    // Each invocation owns a distinct active surfel, so every successful call
    // contributes exactly one new dead slot. The old CAS retry loop serialized
    // recycling waves for milliseconds. A single atomic reservation preserves
    // the dead-list invariant: deadCount + activeCount cannot exceed capacity.
    uint deadSlot = 0u;
    counters.InterlockedAdd(SURFEL_PT_COUNTER_DEAD_SURFELS_OFFSET, 1u, deadSlot);
    if (deadSlot >= maxSurfels)
    {
        // Defensive recovery for a corrupted counter; this path is not expected
        // while each surfel is freed at most once.
        counters.InterlockedMin(SURFEL_PT_COUNTER_DEAD_SURFELS_OFFSET, maxSurfels);
        return false;
    }

    deadBuffer[deadSlot] = surfelIndex;
    return true;
}

bool isRepresentedByCurrentWindow(SurfelPathTracerSurfel surfel, int3 centerCoord)
{
    // The center cell covers the common case. Only surfels whose centers sit
    // just outside the bounded window need the exact overlap test.
    if (isCellCoordValid(centerCoord, push.cellDimension))
    {
        return true;
    }

    for (int z = -1; z <= 1; ++z)
    for (int y = -1; y <= 1; ++y)
    for (int x = -1; x <= 1; ++x)
    {
        int3 cellCoord = centerCoord + int3(x, y, z);
        if (isCellCoordValid(cellCoord, push.cellDimension) &&
            isSurfelIntersectCell(surfel, cellCoord, ubo.cameraPos.xyz, push.cellSize))
        {
            return true;
        }
    }
    return false;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint surfelIndex = tid.x;
    if (surfelIndex >= push.maxSurfels) {
        return;
    }

    SurfelPathTracerSurfel surfel = surfelBuffer[surfelIndex];
    surfel.flags &= ~SURFEL_PT_SURFEL_FLAG_RAY_SCHEDULED;
    surfel.rayOffset = 0u;
    surfel.rayCount = 0u;
    if ((surfel.flags & SURFEL_PT_SURFEL_FLAG_PENDING_FREE) != 0u)
    {
        if (pushDeadSurfel(surfelIndex, push.maxSurfels))
        {
            surfel.flags = 0u;
            surfel.radius = 0.0;
            surfel.rayOffset = 0u;
            surfel.rayCount = 0u;
            surfelBuffer[surfelIndex] = surfel;
            counters.InterlockedAdd(SURFEL_PT_COUNTER_REMOVED_SURFELS_OFFSET, 1u);
        }
        return;
    }

    if (!isActiveSurfel(surfel)) {
        if ((surfel.flags & SURFEL_PT_SURFEL_FLAG_ACTIVE) != 0u &&
            push.lockSurfels == 0u &&
            pushDeadSurfel(surfelIndex, push.maxSurfels))
        {
            surfel.flags = 0u;
            surfel.radius = 0.0;
            surfel.rayOffset = 0u;
            surfel.rayCount = 0u;
            surfelBuffer[surfelIndex] = surfel;
            counters.InterlockedAdd(SURFEL_PT_COUNTER_RECYCLED_SURFELS_OFFSET, 1u);
        }
        return;
    }

    if (push.lockSurfels == 0u)
    {
        refreshAnchoredSurfel(surfelIndex, surfel);

        uint frameIndex = counters.Load(SURFEL_PT_COUNTER_FRAME_INDEX_OFFSET);
        int3 centerCoord = cameraRelativeCellCoord(surfel.position, ubo.cameraPos.xyz, push.cellSize);
        // Detect a real outside->inside transition of the lookup window with a
        // sticky flag. The previous age-based test (ageSinceSeen > 1) re-fired
        // every other frame for any surfel inside the window that had merely
        // dropped out of the compact map. That refreshed lastSeenFrame forever
        // and made the "unrepresented" eviction rule unreachable inside the
        // window: surfels evicted from the map became immortal and untraced.
        bool inWindow = isRepresentedByCurrentWindow(surfel, centerCoord);
        bool wasInWindow = (surfel.flags & SURFEL_PT_SURFEL_FLAG_IN_WINDOW) != 0u;
        bool reenteredCurrentWindow = inWindow && !wasInWindow;
        surfel.flags = inWindow ? (surfel.flags | SURFEL_PT_SURFEL_FLAG_IN_WINDOW)
                                : (surfel.flags & ~SURFEL_PT_SURFEL_FLAG_IN_WINDOW);
        if (reenteredCurrentWindow)
        {
            // Update runs before this frame's resolve. Give a surfel that has
            // just returned to the camera window one frame to be referenced by
            // visible pixels before applying stale-reference eviction.
            surfel.lastSeenFrame = frameIndex;
        }

        uint deadCount = counters.Load(SURFEL_PT_COUNTER_DEAD_SURFELS_OFFSET);
        float aliveRatio = 1.0 - saturate(float(deadCount) / float(max(push.maxSurfels, 1u)));
        float distanceToCamera = length(surfel.position - ubo.cameraPos.xyz);
        if (!reenteredCurrentWindow &&
            (shouldRecycleSurfel(surfel, frameIndex, distanceToCamera, aliveRatio) ||
             isStrandedInSaturatedCell(surfel, centerCoord, frameIndex)))
        {
            if (pushDeadSurfel(surfelIndex, push.maxSurfels))
            {
                surfel.flags = 0u;
                surfel.radius = 0.0;
                surfel.rayOffset = 0u;
                surfel.rayCount = 0u;
                surfelBuffer[surfelIndex] = surfel;
                counters.InterlockedAdd(SURFEL_PT_COUNTER_RECYCLED_SURFELS_OFFSET, 1u);
            }
            return;
        }

        surfel.shortMeanAndLife.w = min(surfel.shortMeanAndLife.w + 1.0f, float(SURFEL_PT_MAX_LIFE));
        // Relative to the surfel's brightness, so dark surfels are judged by the
        // noise the viewer sees rather than by tiny absolute deltas.
        float varianceMagnitude = surfelRelativeInconsistency(surfel);
        // Sleeping is a convergence state, not a visibility state. RaySchedule
        // marks represented surfels as seen every update, so lastSeenFrame could
        // never sustain sleeping. Stable mature surfels now remain asleep and
        // continue receiving a quarter-rate update budget until inconsistency
        // rises enough to wake them.
        if (surfel.shortMeanAndLife.w > 64.0 && varianceMagnitude < 0.05)
        {
            surfel.sleepState = SURFEL_PT_SLEEP_SLEEPING;
        }
        else if (surfel.shortMeanAndLife.w <= SURFEL_PT_WARMUP_LIFE || varianceMagnitude > 0.2)
        {
            surfel.sleepState = SURFEL_PT_SLEEP_AWAKE;
        }
    }
    else
    {
        // Freezing may last longer than the recycling grace window. Keep the
        // lifecycle clock aligned so unfreezing does not immediately recycle
        // the entire otherwise unchanged cache.
        surfel.lastSeenFrame = counters.Load(SURFEL_PT_COUNTER_FRAME_INDEX_OFFSET);
    }

    int3 centerCoord = cameraRelativeCellCoord(surfel.position, ubo.cameraPos.xyz, push.cellSize);
    bool countedCell = false;
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
        uint cellCountOffset = (1u + cellIndex) * 4u;
        cellCounterBuffer.InterlockedAdd(cellCountOffset, 1u);
        countedCell = true;
    }

    if (!countedCell) {
        surfel.rayOffset = 0u;
        surfel.rayCount = 0u;
        surfelBuffer[surfelIndex] = surfel;
        return;
    }

    surfelBuffer[surfelIndex] = surfel;
}
