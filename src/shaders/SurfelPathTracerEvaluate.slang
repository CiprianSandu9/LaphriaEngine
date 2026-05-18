#include "SurfelPathTracerCommon.slang"

[[vk::binding(0, 0)]] RWTexture2D<float4> outputImage;
[[vk::binding(1, 0)]] RWTexture2D<float4> gBufferNormal;
[[vk::binding(2, 0)]] RWTexture2D<float> gBufferDepth;
[[vk::binding(4, 0)]] RWByteAddressBuffer counters;
[[vk::binding(5, 0)]] RWStructuredBuffer<SurfelPathTracerSurfel> surfelBuffer;
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> aliveBuffer;
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
    uint pad1;
    uint pad2;
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

uint cellIndexForPosition(float3 position) {
    uint cellDimension = max(push.cellDimension, 1u);
    float cellSize = max(push.cellSize, 0.0001);
    float dim = float(cellDimension);
    float halfDim = dim * 0.5;
    float maxCoord = dim - 1.0;
    float3 centeredCell = floor(position / cellSize) + float3(halfDim, halfDim, halfDim);
    float3 clampedCell = clamp(centeredCell, float3(0.0, 0.0, 0.0), float3(maxCoord, maxCoord, maxCoord));
    uint3 coord = uint3(clampedCell);
    return coord.x + coord.y * cellDimension + coord.z * cellDimension * cellDimension;
}

float estimateCoverage(float3 position, float3 normal, out uint closestSurfelIndex) {
    closestSurfelIndex = SURFEL_PT_INVALID_INDEX;
    uint centerCell = cellIndexForPosition(position);
    uint dim = max(push.cellDimension, 1u);
    uint3 centerCoord = uint3(centerCell % dim, (centerCell / dim) % dim, centerCell / (dim * dim));

    float coverage = 0.0;
    float bestWeight = 0.0;
    for (int z = -1; z <= 1; ++z) {
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                int3 coord = int3(centerCoord) + int3(x, y, z);
                if (any(coord < int3(0, 0, 0)) || any(coord >= int3(int(dim), int(dim), int(dim)))) {
                    continue;
                }

                uint cellIndex = uint(coord.x) + uint(coord.y) * dim + uint(coord.z) * dim * dim;
                SurfelPathTracerCellInfo cellInfo = cellInfoBuffer[cellIndex];
                for (uint slot = 0u; slot < cellInfo.surfelCount; ++slot) {
                    uint surfelIndex = cellToSurfelBuffer[cellInfo.surfelOffset + slot];
                    if (surfelIndex == SURFEL_PT_INVALID_INDEX) {
                        continue;
                    }

                    SurfelPathTracerSurfel surfel = surfelBuffer[surfelIndex];
                    if (surfel.flags == 0u || surfel.radius <= 0.0 || !isFinite3(surfel.position)) {
                        continue;
                    }

                    float3 surfelNormal = unpackNormalOctahedral(surfel.packedNormal);
                    float normalWeight = saturate(dot(normal, surfelNormal));
                    float distanceWeight = saturate(1.0 - length(position - surfel.position) / max(surfel.radius, 0.0001));
                    float weight = normalWeight * distanceWeight;
                    coverage = max(coverage, weight);
                    if (weight > bestWeight) {
                        bestWeight = weight;
                        closestSurfelIndex = surfelIndex;
                    }
                }
            }
        }
    }
    return coverage;
}

void allocateSurfel(uint2 pixel, float3 position, float3 normal, float radius) {
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
    surfel.radius = max(radius, 0.05);
    surfel.radiance = float3(0.0, 0.0, 0.0);
    surfel.packedNormal = packNormalOctahedral(normal);
    surfel.rayOffset = 0u;
    surfel.rayCount = 1u;
    surfel.irradianceAtlasOffset = surfelIndex;
    surfel.flags = 1u;
    surfel.meanAndVariance = float4(0.0, 0.0, 0.0, 0.0);
    surfel.shortMeanAndLife = float4(0.0, 0.0, 0.0, 1.0);
    surfelBuffer[surfelIndex] = surfel;

    uint aliveSlot = 0u;
    counters.InterlockedAdd(SURFEL_PT_COUNTER_ALIVE_SURFELS_OFFSET, 1u, aliveSlot);
    if (aliveSlot < maxSurfels) {
        aliveBuffer[aliveSlot] = surfelIndex;
    }

    uint dirtySlot = 0u;
    counters.InterlockedAdd(SURFEL_PT_COUNTER_DIRTY_SURFELS_OFFSET, 1u, dirtySlot);
    if (dirtySlot < maxSurfels) {
        dirtyBuffer[dirtySlot] = surfelIndex;
    }
}

float3 resolveSurfelRadiance(uint2 pixel, float depth, float3 normal) {
    float3 position = reconstructWorldPosition(pixel, depth);
    uint closestSurfelIndex = SURFEL_PT_INVALID_INDEX;
    float coverage = estimateCoverage(position, normal, closestSurfelIndex);
    if (closestSurfelIndex == SURFEL_PT_INVALID_INDEX) {
        return normal * 0.5 + 0.5;
    }

    SurfelPathTracerSurfel surfel = surfelBuffer[closestSurfelIndex];
    float3 gi = clampLuminance(surfel.radiance, SURFEL_PT_MAX_RADIANCE_LUMINANCE);
    return lerp(normal * 0.5 + 0.5, gi, saturate(coverage));
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
    float coverage = estimateCoverage(position, normal, closestSurfelIndex);

    if (push.mode == SURFEL_PT_EVALUATE_GENERATE) {
        if (coverage < 0.35) {
            float radius = max(depth * 0.005, 0.08);
            allocateSurfel(pixel, position, normal, radius);
        }
        return;
    }

    outputImage[pixel] = float4(resolveSurfelRadiance(pixel, depth, normal), 1.0);
}
