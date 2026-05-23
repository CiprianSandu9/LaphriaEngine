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
    uint pad0;
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
    for (int shell = 0; shell <= 1; ++shell) {
        for (int z = -shell; z <= shell; ++z) {
            for (int y = -shell; y <= shell; ++y) {
                for (int x = -shell; x <= shell; ++x) {
                    if (samplesVisited >= maxSamples) {
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
                    for (uint slot = 0u; slot < cellInfo.surfelCount; ++slot) {
                        if (samplesVisited >= maxSamples) {
                            break;
                        }

                        uint surfelIndex = cellToSurfelBuffer[cellInfo.surfelOffset + slot];
                        if (surfelIndex == SURFEL_PT_INVALID_INDEX || surfelIndex >= push.maxSurfels) {
                            continue;
                        }

                        SurfelPathTracerSurfel surfel = surfelBuffer[surfelIndex];
                        if (!isActiveSurfel(surfel)) {
                            continue;
                        }

                        samplesVisited += 1u;
                        float3 surfelNormal = unpackNormalOctahedral(surfel.packedNormal);
                        float normalWeight = saturate(dot(normal, surfelNormal));
                        float distanceWeight = saturate(1.0 - length(position - surfel.position) / max(surfel.radius, 0.0001));
                        float weight = normalWeight * distanceWeight;
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
        SurfelPathTracerSurfel surfel = surfelBuffer[closestSurfelIndex];
        surfel.lastReferencedFrame = counters.Load(SURFEL_PT_COUNTER_FRAME_INDEX_OFFSET);
        surfel.lastSeenFrame = counters.Load(SURFEL_PT_COUNTER_FRAME_INDEX_OFFSET);
        surfelBuffer[closestSurfelIndex] = surfel;
    }
    return coverage;
}

void allocateSurfel(uint2 pixel, float3 position, float3 normal, float radius, float3 seedRadiance, uint materialKey) {
    uint maxSurfels = max(push.maxSurfels, 1u);
    uint frameIndex = counters.Load(SURFEL_PT_COUNTER_FRAME_INDEX_OFFSET);
    uint generationPhaseX = frameIndex & 7u;
    uint generationPhaseY = (frameIndex >> 3u) & 7u;
    if ((pixel.x & 7u) != generationPhaseX || (pixel.y & 7u) != generationPhaseY) {
        return;
    }

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
    surfel.irradianceAtlasOffset = surfelIndex;
    surfel.flags = SURFEL_PT_SURFEL_FLAG_ACTIVE;
    surfel.meanAndVariance = float4(surfel.radiance, 1.0);
    surfel.shortMeanAndLife = float4(surfel.radiance, 1.0);
    surfel.lastSeenFrame = frameIndex;
    surfel.lastReferencedFrame = frameIndex;
    surfel.sleepState = SURFEL_PT_SLEEP_AWAKE;
    surfel.materialKey = materialKey;
    surfel.varianceAndInconsistency = float4(1.0, 1.0, 1.0, 1.0);
    surfelBuffer[surfelIndex] = surfel;

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
    for (int shell = 0; shell <= 1; ++shell) {
        for (int z = -shell; z <= shell; ++z) {
            for (int y = -shell; y <= shell; ++y) {
                for (int x = -shell; x <= shell; ++x) {
                    if (samplesVisited >= maxSamples) {
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
                    for (uint slot = 0u; slot < cellInfo.surfelCount; ++slot) {
                        if (samplesVisited >= maxSamples) {
                            break;
                        }

                        uint surfelIndex = cellToSurfelBuffer[cellInfo.surfelOffset + slot];
                        if (surfelIndex == SURFEL_PT_INVALID_INDEX || surfelIndex >= push.maxSurfels) {
                            continue;
                        }

                        SurfelPathTracerSurfel surfel = surfelBuffer[surfelIndex];
                        if (!isActiveSurfel(surfel)) {
                            continue;
                        }

                        samplesVisited += 1u;
                        float3 surfelNormal = unpackNormalOctahedral(surfel.packedNormal);
                        float normalWeight = saturate(dot(normal, surfelNormal));
                        float influenceRadius = max(surfel.radius * 2.0, push.cellSize * 0.75);
                        float distanceWeight = saturate(1.0 - length(position - surfel.position) / max(influenceRadius, 0.0001));
                        float weight = normalWeight * distanceWeight * distanceWeight;
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
        if (push.lockSurfels == 0u && push.enablePlacement != 0u && coverage < push.placementThreshold) {
            float radius = surfelRadius(depth, push.surfelTargetArea, push.surfelMinRadius, push.cellSize * push.surfelMaxRadiusScale);
            allocateSurfel(pixel, position, normal, radius, resolveSurfelRadiance(position, normal), 0u);
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
