#include "SurfelPathTracerCommon.slang"

[[vk::binding(0, 0)]] RWTexture2D<float4> outputImage;
[[vk::binding(1, 0)]] RWTexture2D<float4> gBufferNormal;
[[vk::binding(2, 0)]] RWTexture2D<float> gBufferDepth;
[[vk::binding(4, 0)]] RWByteAddressBuffer counters;
[[vk::binding(5, 0)]] RWStructuredBuffer<SurfelPathTracerSurfel> surfelBuffer;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> deadBuffer;
[[vk::binding(8, 0)]] RWStructuredBuffer<uint> dirtyBuffer;
[[vk::binding(11, 0)]] RWStructuredBuffer<SurfelPathTracerCellInfo> cellInfoBuffer;
[[vk::binding(13, 0)]] RWStructuredBuffer<uint> cellToSurfelBuffer;
[[vk::binding(28, 0)]] StructuredBuffer<SurfelPathTracerPixelSource> gBufferSourceBuffer;
[[vk::binding(29, 0)]] RWStructuredBuffer<SurfelPathTracerSource> surfelSourceBuffer;

[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

struct EvaluatePushConstants {
    uint mode;
    uint width;
    uint height;
    float cellSize;
    uint cellDimension;
    uint maxSurfels;
    float placementThreshold;
    float removalThreshold;
    float surfelTargetArea;
    float surfelMinRadius;
    float surfelMaxRadiusScale;
    uint frameIndex;
    uint lockSurfels;
    uint enablePlacement;
    uint enableRemoval;
    uint maxSurfelSamplesPerQuery;
    uint perCellSurfelLimit;
    uint pad1;
};

[[vk::push_constant]] EvaluatePushConstants push;

static const uint SURFEL_PT_EVALUATE_GENERATE = 0u;
static const uint SURFEL_PT_EVALUATE_RESOLVE = 1u;

float3 primaryRayDirection(uint2 pixel) {
    float2 uv = (float2(pixel) + float2(0.5, 0.5)) / float2(max(push.width, 1u), max(push.height, 1u));
    float2 d = uv * 2.0 - 1.0;
    float4 target = mul(ubo.projInverse, float4(d.x, -d.y, 1.0, 1.0));
    float3 rayDir = mul(ubo.viewInverse, float4(normalize(target.xyz / target.w), 0.0)).xyz;
    return normalize(rayDir);
}

float3 reconstructWorldPosition(uint2 pixel, float depth) {
    return ubo.cameraPos.xyz + primaryRayDirection(pixel) * depth;
}

float estimateCoverage(float3 position, float3 normal, out uint closestSurfelIndex, bool stampClosest) {
    closestSurfelIndex = SURFEL_PT_INVALID_INDEX;
    uint dim = max(push.cellDimension, 1u);
    int3 centerCoord = cameraRelativeCellCoord(position, ubo.cameraPos.xyz, push.cellSize);
    if (!isCellCoordValid(centerCoord, dim)) {
        return 0.0;
    }

    float coverage = 0.0;
    float bestWeight = 0.0;
    uint maxSamples = max(push.maxSurfelSamplesPerQuery, 1u);
    uint samplesVisited = 0u;
    uint entriesInspected = 0u;
    uint maxEntriesInspected = maxSamples * 8u;
    for (int shell = 0; shell <= 1; ++shell) {
        for (int z = -shell; z <= shell; ++z) {
            for (int y = -shell; y <= shell; ++y) {
                for (int x = -shell; x <= shell; ++x) {
                    if (samplesVisited >= maxSamples || entriesInspected >= maxEntriesInspected) {
                        break;
                    }
                    if (max(abs(x), max(abs(y), abs(z))) != shell) {
                        continue;
                    }

                    int3 cellCoord = centerCoord + int3(x, y, z);
                    if (!isCellCoordValid(cellCoord, dim)) {
                        continue;
                    }

                    uint cellIndex = flattenCameraRelativeCellCoord(cellCoord, dim);
                    SurfelPathTracerCellInfo cellInfo = cellInfoBuffer[cellIndex];
                    // Never stop part-way through a cell: slots in a complete
                    // under-capacity cell are assigned by atomics and therefore
                    // have no stable order. Treat maxSamples as a between-cell
                    // soft budget so the retained set, not insertion order,
                    // determines coverage.
                    for (uint slot = 0u; slot < cellInfo.surfelCount; ++slot) {
                        uint surfelIndex = unpackCellSurfelIndex(cellToSurfelBuffer[cellInfo.surfelOffset + slot]);
                        if (surfelIndex == SURFEL_PT_INVALID_INDEX || surfelIndex >= push.maxSurfels) {
                            continue;
                        }

                        SurfelPathTracerSurfel surfel = surfelBuffer[surfelIndex];
                        if (!isActiveSurfel(surfel)) {
                            continue;
                        }

                        entriesInspected += 1u;
                        float3 surfelNormal = unpackNormalOctahedral(surfel.packedNormal);
                        float normalWeight = saturate(dot(normal, surfelNormal));
                        float distanceWeight = saturate(1.0 - length(position - surfel.position) / max(surfel.radius, 0.0001));
                        float weight = normalWeight * distanceWeight;
                        // A spatial cell can contain several disconnected
                        // walls or arches. Candidates with no geometric support
                        // must not consume the query budget and prevent the
                        // neighboring cells from being searched.
                        if (weight <= 0.0001) {
                            continue;
                        }
                        samplesVisited += 1u;
                        coverage += weight;
                        if (weight > bestWeight) {
                            bestWeight = weight;
                            closestSurfelIndex = surfelIndex;
                        }
                    }
                }
            }
        }
    }
    if (stampClosest && closestSurfelIndex != SURFEL_PT_INVALID_INDEX) {
        uint previousReferenceFrame = 0u;
        InterlockedMax(surfelBuffer[closestSurfelIndex].lastReferencedFrame,
                       counters.Load(SURFEL_PT_COUNTER_FRAME_INDEX_OFFSET),
                       previousReferenceFrame);
    }
    return coverage;
}

uint sourcePixelIndex(uint2 pixel)
{
    return pixel.y * max(push.width, 1u) + pixel.x;
}

uint loadPixelMaterialKey(uint2 pixel)
{
    return gBufferSourceBuffer[sourcePixelIndex(pixel)].sourceMaterialKey;
}

void writeSurfelSource(uint surfelIndex, uint2 pixel, uint materialKey)
{
    uint pixelIndex = sourcePixelIndex(pixel);
    SurfelPathTracerPixelSource pixelSource = gBufferSourceBuffer[pixelIndex];

    SurfelPathTracerSource source;
    source.sourceNodeId = pixelSource.sourceNodeId;
    source.sourceInstanceId = pixelSource.sourceInstanceId;
    source.sourceInstanceCustomIndex = pixelSource.sourceInstanceCustomIndex;
    source.sourcePrimitiveIndex = pixelSource.sourcePrimitiveIndex;
    source.sourceBarycentrics = pixelSource.sourceBarycentrics;
    source.sourceMaterialKey = materialKey;
    bool validSource = (pixelSource.sourceFlags & SURFEL_PT_SOURCE_FLAG_VALID) != 0u;
    source.sourceFlags = validSource ? pixelSource.sourceFlags : 0u;
    source.sourceObjectPosition = pixelSource.sourceObjectPosition;
    source.sourceObjectNormal = pixelSource.sourceObjectNormal;

    surfelSourceBuffer[surfelIndex] = source;
}

void allocateSurfel(uint2 pixel, float3 position, float3 normal, float radius, float3 seedRadiance, uint materialKey) {
    uint maxSurfels = max(push.maxSurfels, 1u);
    uint frameIndex = counters.Load(SURFEL_PT_COUNTER_FRAME_INDEX_OFFSET);
    uint deadCount = counters.Load(SURFEL_PT_COUNTER_DEAD_SURFELS_OFFSET);
    uint previousDeadCount = deadCount;
    uint deadSlot = SURFEL_PT_INVALID_INDEX;
    for (uint attempt = 0u; attempt < 16u; ++attempt) {
        if (deadCount == 0u || deadCount > maxSurfels) {
            return;
        }

        uint desiredDeadCount = deadCount - 1u;
        counters.InterlockedCompareExchange(SURFEL_PT_COUNTER_DEAD_SURFELS_OFFSET,
                                            deadCount,
                                            desiredDeadCount,
                                            previousDeadCount);
        if (previousDeadCount == deadCount) {
            deadSlot = desiredDeadCount;
            break;
        }

        deadCount = previousDeadCount;
    }

    if (deadSlot == SURFEL_PT_INVALID_INDEX || deadSlot >= maxSurfels) {
        return;
    }

    uint surfelIndex = deadBuffer[deadSlot];
    if (surfelIndex == SURFEL_PT_INVALID_INDEX || surfelIndex >= maxSurfels) {
        return;
    }

    SurfelPathTracerSurfel surfel;
    surfel.position = position;
    surfel.radius = max(radius, push.surfelMinRadius);
    surfel.radiance = clampLuminance(seedRadiance, SURFEL_PT_MAX_RADIANCE_LUMINANCE);
    surfel.packedNormal = packNormalOctahedral(normal);
    surfel.rayOffset = 0u;
    surfel.rayCount = 1u;
    // The atlas tile is still addressed by surfelIndex. This value identifies
    // the allocation currently owning the tile so stale per-frame atlas data
    // from a recycled surfel is ignored until fresh samples arrive.
    surfel.irradianceAtlasOffset = (frameIndex % SURFEL_PT_GUIDE_TOKEN_COUNT) + 1u;
    surfel.flags = SURFEL_PT_SURFEL_FLAG_ACTIVE;
    surfel.meanAndVariance = float4(surfel.radiance, 1.0);
    surfel.shortMeanAndLife = float4(surfel.radiance, 1.0);
    surfel.lastSeenFrame = frameIndex;
    surfel.lastReferencedFrame = frameIndex;
    surfel.sleepState = SURFEL_PT_SLEEP_AWAKE;
    surfel.materialKey = materialKey;
    surfel.varianceAndInconsistency = float4(1.0, 1.0, 1.0, 1.0);
    surfelBuffer[surfelIndex] = surfel;
    writeSurfelSource(surfelIndex, pixel, surfel.materialKey);

    counters.InterlockedAdd(SURFEL_PT_COUNTER_SPAWNED_SURFELS_OFFSET, 1u);

    uint dirtySlot = 0u;
    counters.InterlockedAdd(SURFEL_PT_COUNTER_DIRTY_SURFELS_OFFSET, 1u, dirtySlot);
    if (dirtySlot < maxSurfels) {
        dirtyBuffer[dirtySlot] = surfelIndex;
    }
}

float3 sampleWeightedSurfelRadiance(float3 position, float3 normal) {
    uint dim = max(push.cellDimension, 1u);
    int3 centerCoord = cameraRelativeCellCoord(position, ubo.cameraPos.xyz, push.cellSize);
    if (!isCellCoordValid(centerCoord, dim)) {
        return float3(0.0);
    }

    float3 weightedRadiance = float3(0.0);
    float weightSum = 0.0;
    uint maxSamples = max(push.maxSurfelSamplesPerQuery, 1u);
    uint samplesVisited = 0u;
    uint entriesInspected = 0u;
    uint maxEntriesInspected = maxSamples * 8u;
    for (int shell = 0; shell <= 1; ++shell) {
        for (int z = -shell; z <= shell; ++z) {
            for (int y = -shell; y <= shell; ++y) {
                for (int x = -shell; x <= shell; ++x) {
                    if (samplesVisited >= maxSamples || entriesInspected >= maxEntriesInspected) {
                        break;
                    }
                    if (max(abs(x), max(abs(y), abs(z))) != shell) {
                        continue;
                    }

                    int3 cellCoord = centerCoord + int3(x, y, z);
                    if (!isCellCoordValid(cellCoord, dim)) {
                        continue;
                    }

                    uint cellIndex = flattenCameraRelativeCellCoord(cellCoord, dim);
                    SurfelPathTracerCellInfo cellInfo = cellInfoBuffer[cellIndex];
                    // As in coverage, finish the current cell before applying
                    // the query budget. This removes partial-cell dependence on
                    // the parallel insertion order.
                    for (uint slot = 0u; slot < cellInfo.surfelCount; ++slot) {
                        uint surfelIndex = unpackCellSurfelIndex(cellToSurfelBuffer[cellInfo.surfelOffset + slot]);
                        if (surfelIndex == SURFEL_PT_INVALID_INDEX || surfelIndex >= push.maxSurfels) {
                            continue;
                        }

                        SurfelPathTracerSurfel surfel = surfelBuffer[surfelIndex];
                        if (!isActiveSurfel(surfel)) {
                            continue;
                        }

                        entriesInspected += 1u;
                        float3 surfelNormal = unpackNormalOctahedral(surfel.packedNormal);
                        float normalWeight = saturate(dot(normal, surfelNormal));
                        // Resolve must use the same physical support radius as
                        // placement and cell intersection. Doubling it here made
                        // the intended influence extend beyond the fixed +/-1
                        // query neighborhood and increased cross-surface bleed.
                        float influenceRadius = max(surfel.radius, push.cellSize * 0.75);
                        float distanceWeight = saturate(1.0 - length(position - surfel.position) / max(influenceRadius, 0.0001));
                        float weight = normalWeight * distanceWeight * distanceWeight;
                        if (weight <= 0.0001) {
                            continue;
                        }
                        samplesVisited += 1u;
                        weightedRadiance += clampLuminance(surfel.radiance, SURFEL_PT_MAX_RADIANCE_LUMINANCE) * weight;
                        weightSum += weight;
                    }
                }
            }
        }
    }

    return weightSum > 0.0001 ? weightedRadiance / weightSum : float3(0.0);
}

float3 resolveSurfelRadiance(float3 position, float3 normal) {
    return sampleWeightedSurfelRadiance(position, normal);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint2 pixel = tid.xy;
    if (push.mode == SURFEL_PT_EVALUATE_GENERATE) {
        uint frameIndex = counters.Load(SURFEL_PT_COUNTER_FRAME_INDEX_OFFSET);
        uint generationPhaseX = frameIndex & 7u;
        uint generationPhaseY = (frameIndex >> 3u) & 7u;
        // The generation dispatch is 1/8 resolution in each dimension. Map its
        // densely packed lanes to the same one-pixel-per-8x8-tile temporal phase
        // that placement has always used. This avoids mostly-empty waves while
        // still considering every screen pixel once over a 64-frame cycle.
        pixel = tid.xy * 8u + uint2(generationPhaseX, generationPhaseY);
    }

    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    float depth = gBufferDepth[pixel];
    float3 rawNormal = gBufferNormal[pixel].xyz;
    if (depth <= 0.0 || length(rawNormal) <= 0.001) {
        if (push.mode == SURFEL_PT_EVALUATE_RESOLVE) {
            float3 sunDir = normalize(-ubo.lightDir.xyz);
            outputImage[pixel] = float4(evalSkyColor(primaryRayDirection(pixel), sunDir), 1.0);
        }
        return;
    }

    float3 normal = normalize(rawNormal);
    float3 position = reconstructWorldPosition(pixel, depth);
    uint closestSurfelIndex = SURFEL_PT_INVALID_INDEX;

    if (push.mode == SURFEL_PT_EVALUATE_GENERATE) {
        float coverage = estimateCoverage(position, normal, closestSurfelIndex, false);
        bool wantsPlacement = push.lockSurfels == 0u &&
                              push.enablePlacement != 0u &&
                              coverage < push.placementThreshold;
        // perCellSurfelLimit bounds the compact lookup representation, not the
        // persistent surfel population. A full cell can contain unrelated
        // surfaces; blocking placement here permanently leaves a newly visible
        // surface without GI. Deterministic overflow selection and global
        // pressure recycling provide the actual capacity controls.
        // Thousands of uncovered pixels used to compute seed radiance and then
        // spin on the same dead-list CAS, although only a small fraction could
        // win. Keep one deterministic candidate per subgroup. This preserves
        // broad screen coverage while reducing both seed searches and global
        // atomic contention by approximately the subgroup width.
        bool placeFromThisLane = wantsPlacement && WavePrefixCountBits(wantsPlacement) == 0u;
        if (placeFromThisLane) {
            float radius = surfelRadius(depth, push.surfelTargetArea, push.surfelMinRadius, push.cellSize * push.surfelMaxRadiusScale);
            float3 seedRadiance = resolveSurfelRadiance(position, normal);
            allocateSurfel(pixel, position, normal, radius, seedRadiance, loadPixelMaterialKey(pixel));
        }
        if (push.lockSurfels == 0u &&
            push.enableRemoval != 0u &&
            coverage > push.removalThreshold &&
            closestSurfelIndex != SURFEL_PT_INVALID_INDEX) {
            SurfelPathTracerSurfel surfel = surfelBuffer[closestSurfelIndex];
            if (isActiveSurfel(surfel)) {
                surfel.flags |= SURFEL_PT_SURFEL_FLAG_PENDING_FREE;
                surfelBuffer[closestSurfelIndex] = surfel;
            }
        }
        return;
    }

    float coverage = estimateCoverage(position, normal, closestSurfelIndex, true);
    outputImage[pixel] = float4(resolveSurfelRadiance(position, normal), coverage);
}
