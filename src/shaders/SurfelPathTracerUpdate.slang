#include "SurfelPathTracerCommon.slang"

[[vk::binding(4, 0)]] RWByteAddressBuffer counters;
[[vk::binding(5, 0)]] RWStructuredBuffer<SurfelPathTracerSurfel> surfelBuffer;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> deadBuffer;
[[vk::binding(10, 0)]] RWStructuredBuffer<uint> rayBuffer;
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
    uint pad0;
    uint pad1;
    uint pad2;
};

[[vk::push_constant]] UpdatePushConstants push;

bool refreshAnchoredSurfel(uint surfelIndex, inout SurfelPathTracerSurfel surfel)
{
    SurfelPathTracerSource source = surfelSourceBuffer[surfelIndex];
    if ((source.sourceFlags & SURFEL_PT_SOURCE_FLAG_VALID) == 0u)
    {
        return false;
    }

    if (source.sourceNodeId >= push.sourceTransformCount)
    {
        source.sourceFlags |= SURFEL_PT_SOURCE_FLAG_REFRESH_FAILED;
        surfelSourceBuffer[surfelIndex] = source;
        return false;
    }

    SurfelPathTracerSourceTransform transform = sourceTransformBuffer[source.sourceNodeId];
    if ((transform.flags & SURFEL_PT_SOURCE_FLAG_VALID) == 0u)
    {
        source.sourceFlags |= SURFEL_PT_SOURCE_FLAG_REFRESH_FAILED;
        surfelSourceBuffer[surfelIndex] = source;
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
    uint deadCount = counters.Load(SURFEL_PT_COUNTER_DEAD_SURFELS_OFFSET);
    uint previousDeadCount = deadCount;
    for (uint attempt = 0u; attempt < 16u; ++attempt)
    {
        if (deadCount >= maxSurfels)
        {
            return false;
        }

        uint desiredDeadCount = deadCount + 1u;
        counters.InterlockedCompareExchange(SURFEL_PT_COUNTER_DEAD_SURFELS_OFFSET,
                                            deadCount,
                                            desiredDeadCount,
                                            previousDeadCount);
        if (previousDeadCount == deadCount)
        {
            deadBuffer[deadCount] = surfelIndex;
            return true;
        }
        deadCount = previousDeadCount;
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

    refreshAnchoredSurfel(surfelIndex, surfel);

    uint frameIndex = counters.Load(SURFEL_PT_COUNTER_FRAME_INDEX_OFFSET);
    uint deadCount = counters.Load(SURFEL_PT_COUNTER_DEAD_SURFELS_OFFSET);
    float aliveRatio = 1.0 - saturate(float(deadCount) / float(max(push.maxSurfels, 1u)));
    float distanceToCamera = length(surfel.position - ubo.cameraPos.xyz);
    if (push.lockSurfels == 0u && shouldRecycleSurfel(surfel, frameIndex, distanceToCamera, aliveRatio))
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
    uint ageSinceSeen = frameIndex >= surfel.lastSeenFrame ? frameIndex - surfel.lastSeenFrame : 0u;
    float varianceMagnitude = length(max(surfel.varianceAndInconsistency.xyz, float3(0.0)));
    if (ageSinceSeen > 60u && varianceMagnitude < 0.05)
    {
        surfel.sleepState = SURFEL_PT_SLEEP_SLEEPING;
    }
    else if (ageSinceSeen < 20u || varianceMagnitude > 0.2)
    {
        surfel.sleepState = SURFEL_PT_SLEEP_AWAKE;
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

    uint rayRequestCount = adaptiveRayCountForSurfel(surfel,
                                                     push.minRaysPerSurfel,
                                                     push.maxRaysPerSurfel,
                                                     push.varianceSensitivity,
                                                     frameIndex);
    if (rayRequestCount > 0u && push.maxRays > 0u) {
        uint rayOffset = 0u;
        counters.InterlockedAdd(SURFEL_PT_COUNTER_REQUESTED_RAYS_OFFSET, rayRequestCount, rayOffset);
        uint writableRayCount = rayOffset < push.maxRays ? min(rayRequestCount, push.maxRays - rayOffset) : 0u;
        if (writableRayCount < rayRequestCount) {
            counters.InterlockedMin(SURFEL_PT_COUNTER_REQUESTED_RAYS_OFFSET, push.maxRays);
        }

        surfel.rayOffset = rayOffset;
        surfel.rayCount = writableRayCount;
        for (uint ray = 0u; ray < writableRayCount; ++ray) {
            uint rayIndex = rayOffset + ray;
            uint base = rayIndex * SURFEL_PT_RAY_RECORD_UINTS;
            rayBuffer[base + SURFEL_PT_RAY_SURFEL_INDEX] = surfelIndex;
            rayBuffer[base + SURFEL_PT_RAY_FLAGS] = 0u;
        }
    } else {
        surfel.rayOffset = 0u;
        surfel.rayCount = 0u;
    }

    surfelBuffer[surfelIndex] = surfel;
}
