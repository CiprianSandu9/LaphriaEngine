#ifndef SURFEL_PATH_TRACER_COMMON_H
#define SURFEL_PATH_TRACER_COMMON_H

#include "ShaderCommon.slang"

static const uint SURFEL_PT_MAX_LIFE = 255u;
static const uint SURFEL_PT_INVALID_INDEX = 0xFFFFFFFFu;
static const uint SURFEL_PT_ATLAS_TILE_SIZE = 6u;
static const uint SURFEL_PT_SLEEP_AWAKE = 0u;
static const uint SURFEL_PT_SLEEP_SLEEPING = 1u;
static const uint SURFEL_PT_STATUS_LAST_SEEN = 0x2u;
static const uint SURFEL_PT_STATUS_LAST_REFERENCED = 0x4u;
static const float SURFEL_PT_RAY_BIAS = 0.001;
static const float SURFEL_PT_MAX_RADIANCE_LUMINANCE = 32.0;

static const uint SURFEL_PT_COUNTER_ALIVE_SURFELS_OFFSET = 0u;
static const uint SURFEL_PT_COUNTER_DEAD_SURFELS_OFFSET = 4u;
static const uint SURFEL_PT_COUNTER_DIRTY_SURFELS_OFFSET = 8u;
static const uint SURFEL_PT_COUNTER_REQUESTED_RAYS_OFFSET = 12u;
static const uint SURFEL_PT_COUNTER_FILLED_CELLS_OFFSET = 16u;
static const uint SURFEL_PT_COUNTER_REJECTED_STORES_OFFSET = 20u;
static const uint SURFEL_PT_COUNTER_FRAME_INDEX_OFFSET = 24u;
static const uint SURFEL_PT_COUNTER_RECYCLED_SURFELS_OFFSET = 32u;
static const uint SURFEL_PT_COUNTER_SPAWNED_SURFELS_OFFSET = 36u;
static const uint SURFEL_PT_COUNTER_REMOVED_SURFELS_OFFSET = 40u;
static const uint SURFEL_PT_COUNTER_GUIDED_RAYS_OFFSET = 44u;
static const uint SURFEL_PT_COUNTER_COSINE_RAYS_OFFSET = 48u;
static const uint SURFEL_PT_COUNTER_SURFEL_TERMINATED_PATHS_OFFSET = 52u;
static const uint SURFEL_PT_COUNTER_PATH_MISSES_OFFSET = 56u;

static const uint SURFEL_PT_RAY_RECORD_UINTS = 8u;
static const uint SURFEL_PT_RAY_SURFEL_INDEX = 0u;
static const uint SURFEL_PT_RAY_RADIANCE_X = 1u;
static const uint SURFEL_PT_RAY_RADIANCE_Y = 2u;
static const uint SURFEL_PT_RAY_RADIANCE_Z = 3u;
static const uint SURFEL_PT_RAY_PDF = 4u;
static const uint SURFEL_PT_RAY_HIT_T = 5u;
static const uint SURFEL_PT_RAY_FLAGS = 6u;

struct SurfelPathTracerSurfel {
    float3 position;
    float radius;
    float3 radiance;
    uint packedNormal;
    uint rayOffset;
    uint rayCount;
    uint irradianceAtlasOffset;
    uint flags;
    float4 meanAndVariance;
    float4 shortMeanAndLife;
    uint lastSeenFrame;
    uint lastReferencedFrame;
    uint sleepState;
    uint materialKey;
    float4 varianceAndInconsistency;
};

struct SurfelPathTracerCellInfo {
    uint surfelOffset;
    uint surfelCount;
};

struct SurfelPathTracerCounters {
    uint aliveSurfels;
    uint deadSurfels;
    uint dirtySurfels;
    uint requestedRays;
    uint filledCells;
    uint rejectedStores;
    uint frameIndex;
    uint pad0;
    uint recycledSurfels;
    uint spawnedSurfels;
    uint removedSurfels;
    uint guidedRays;
    uint cosineRays;
    uint surfelTerminatedPaths;
    uint pathMisses;
    uint pad1;
};

uint packNormalOctahedral(float3 normal) {
    float3 n = normalize(normal);
    float invL1 = 1.0 / max(abs(n.x) + abs(n.y) + abs(n.z), 0.000001);
    float2 oct = n.xy * invL1;
    if (n.z < 0.0) {
        float2 signNotZero = float2(oct.x >= 0.0 ? 1.0 : -1.0, oct.y >= 0.0 ? 1.0 : -1.0);
        oct = (1.0 - abs(oct.yx)) * signNotZero;
    }
    uint2 encoded = uint2(round(saturate(oct * 0.5 + 0.5) * 65535.0));
    return encoded.x | (encoded.y << 16u);
}

float3 unpackNormalOctahedral(uint packedNormal) {
    uint2 encoded = uint2(packedNormal & 0xFFFFu, packedNormal >> 16u);
    float2 oct = float2(encoded) / 65535.0 * 2.0 - 1.0;
    float3 n = float3(oct.x, oct.y, 1.0 - abs(oct.x) - abs(oct.y));
    if (n.z < 0.0) {
        float2 signNotZero = float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
        n.xy = (1.0 - abs(n.yx)) * signNotZero;
    }
    return normalize(n);
}

bool isFinite3(float3 value) {
    return abs(value.x) < 3.402823e38 &&
           abs(value.y) < 3.402823e38 &&
           abs(value.z) < 3.402823e38;
}

int3 cameraRelativeCellCoord(float3 position, float3 cameraPosition, float cellSize)
{
    return int3(round((position - cameraPosition) / max(cellSize, 0.0001)));
}

bool isCellCoordValid(int3 coord, uint cellDimension)
{
    uint dim = max(cellDimension, 1u);
    int halfDim = int(dim) / 2;
    int upperDim = int(dim) - halfDim;
    return coord.x >= -halfDim && coord.x < upperDim &&
           coord.y >= -halfDim && coord.y < upperDim &&
           coord.z >= -halfDim && coord.z < upperDim;
}

uint flattenCameraRelativeCellCoord(int3 coord, uint cellDimension)
{
    uint dim = max(cellDimension, 1u);
    int halfDim = int(dim) / 2;
    uint3 unsignedCoord = uint3(coord + int3(halfDim, halfDim, halfDim));
    return unsignedCoord.x + unsignedCoord.y * dim + unsignedCoord.z * dim * dim;
}

uint cameraRelativeCellIndexForPosition(float3 position, float3 cameraPosition, float cellSize, uint cellDimension)
{
    int3 coord = cameraRelativeCellCoord(position, cameraPosition, cellSize);
    if (!isCellCoordValid(coord, cellDimension))
    {
        return SURFEL_PT_INVALID_INDEX;
    }
    return flattenCameraRelativeCellCoord(coord, cellDimension);
}

bool isSurfelIntersectCell(SurfelPathTracerSurfel surfel, int3 cellCoord, float3 cameraPosition, float cellSize)
{
    if (surfel.flags == 0u || surfel.radius <= 0.0 || !isFinite3(surfel.position))
    {
        return false;
    }

    float safeCellSize = max(cellSize, 0.0001);
    float3 cellCenter = float3(cellCoord) * safeCellSize + cameraPosition;
    float3 halfExtent = float3(safeCellSize * 0.5);
    float3 closest = clamp(surfel.position, cellCenter - halfExtent, cellCenter + halfExtent);
    return length(closest - surfel.position) <= surfel.radius;
}

bool shouldRecycleSurfel(SurfelPathTracerSurfel surfel, uint frameIndex, float distanceToCamera, float aliveRatio)
{
    if (surfel.flags == 0u)
    {
        return false;
    }
    if (surfel.radius <= 0.0 || !isFinite3(surfel.position))
    {
        return true;
    }

    uint ageSinceSeen = frameIndex >= surfel.lastSeenFrame ? frameIndex - surfel.lastSeenFrame : 0u;
    uint ageSinceRef = frameIndex >= surfel.lastReferencedFrame ? frameIndex - surfel.lastReferencedFrame : 0u;
    bool stale = ageSinceSeen > 240u && ageSinceRef > 120u;
    bool distant = distanceToCamera > 160.0;
    bool pressure = aliveRatio > 0.85;
    return stale && (distant || pressure);
}

float surfelRadius(float distanceToCamera, float targetArea, float minRadius, float maxRadius)
{
    float radius = max(distanceToCamera * sqrt(max(targetArea, 1.0)) * 0.0007, minRadius);
    return min(radius, maxRadius);
}

float luminance(float3 value) {
    return dot(value, float3(0.2126, 0.7152, 0.0722));
}

float3 clampLuminance(float3 radiance, float maxLuminance) {
    float safeMax = max(maxLuminance, 0.0);
    float luma = max(luminance(max(radiance, float3(0.0, 0.0, 0.0))), 0.0);
    if (luma <= safeMax || luma <= 0.0001) {
        return max(radiance, float3(0.0, 0.0, 0.0));
    }
    return max(radiance, float3(0.0, 0.0, 0.0)) * (safeMax / luma);
}

float3 msmeBlend(float3 previousMean, float3 sampleRadiance, float alpha)
{
    return lerp(previousMean, sampleRadiance, clamp(alpha, 0.0, 1.0));
}

#endif // SURFEL_PATH_TRACER_COMMON_H
