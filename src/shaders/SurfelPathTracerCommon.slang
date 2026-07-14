#ifndef SURFEL_PATH_TRACER_COMMON_H
#define SURFEL_PATH_TRACER_COMMON_H

#include "ShaderCommon.slang"

static const uint SURFEL_PT_MAX_LIFE = 255u;
static const uint SURFEL_PT_INVALID_INDEX = 0xFFFFFFFFu;
static const uint SURFEL_PT_ATLAS_TILE_SIZE = 6u;
static const uint SURFEL_PT_SLEEP_AWAKE = 0u;
static const uint SURFEL_PT_SLEEP_SLEEPING = 1u;
static const uint SURFEL_PT_SURFEL_FLAG_ACTIVE = 1u;
static const uint SURFEL_PT_SURFEL_FLAG_PENDING_FREE = 2u;
static const uint SURFEL_PT_SURFEL_FLAG_RAY_SCHEDULED = 4u;
static const uint SURFEL_PT_SOURCE_FLAG_VALID = 1u << 0u;
static const uint SURFEL_PT_SOURCE_FLAG_REFRESH_FAILED = 1u << 1u;
static const uint SURFEL_PT_STATUS_LAST_SEEN = 0x2u;
static const uint SURFEL_PT_STATUS_LAST_REFERENCED = 0x4u;
static const float SURFEL_PT_RAY_BIAS = 0.001;
static const float SURFEL_PT_MAX_RADIANCE_LUMINANCE = 32.0;
// Stored as a negative value in the R16F alpha channel of each guide texel.
// Integers in this range are represented exactly, allowing recycled surfels
// to reject atlas data that belongs to an older allocation without another
// GPU resource. The sign also distinguishes it from the former luma layout.
static const uint SURFEL_PT_GUIDE_TOKEN_COUNT = 1023u;

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
static const uint SURFEL_PT_COUNTER_SURFEL_TERMINATION_ATTEMPTS_OFFSET = 52u;
static const uint SURFEL_PT_COUNTER_PATH_MISSES_OFFSET = 56u;
static const uint SURFEL_PT_COUNTER_SURFEL_TERMINATION_HITS_OFFSET = 60u;
static const uint SURFEL_PT_INDIRECT_RAY_WIDTH_OFFSET = 64u;
static const uint SURFEL_PT_INDIRECT_RAY_HEIGHT_OFFSET = 68u;
static const uint SURFEL_PT_INDIRECT_RAY_DEPTH_OFFSET = 72u;

static const uint SURFEL_PT_RAY_RECORD_UINTS = 8u;
static const uint SURFEL_PT_RAY_SURFEL_INDEX = 0u;
static const uint SURFEL_PT_RAY_RADIANCE_X = 1u;
static const uint SURFEL_PT_RAY_RADIANCE_Y = 2u;
static const uint SURFEL_PT_RAY_RADIANCE_Z = 3u;
static const uint SURFEL_PT_RAY_PDF = 4u;
static const uint SURFEL_PT_RAY_HIT_T = 5u;
static const uint SURFEL_PT_RAY_FLAGS = 6u;
static const uint SURFEL_PT_RAY_LOCAL_DIRECTION = 7u;

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

struct SurfelPathTracerSource {
    uint sourceNodeId;
    uint sourceInstanceId;
    uint sourceInstanceCustomIndex;
    uint sourcePrimitiveIndex;
    float2 sourceBarycentrics;
    uint sourceMaterialKey;
    uint sourceFlags;
    float3 sourceObjectPosition;
    uint sourceObjectNormal;
};

struct SurfelPathTracerPixelSource {
    float3 sourceObjectPosition;
    uint sourceObjectNormal;
    float2 sourceBarycentrics;
    uint sourcePrimitiveIndex;
    uint sourceInstanceId;
    uint sourceNodeId;
    uint sourceInstanceCustomIndex;
    uint sourceMaterialKey;
    uint sourceFlags;
};

struct SurfelPathTracerSourceInstance {
    uint sourceNodeId;
    uint modelId;
    uint primitiveOffset;
    uint flags;
};

struct SurfelPathTracerSourceTransform {
    float4x4 objectToWorld;
    float4x4 worldToObject;
    uint flags;
    uint padding0;
    uint padding1;
    uint padding2;
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
    uint surfelTerminationAttempts;
    uint pathMisses;
    uint surfelTerminationHits;
    uint indirectRayWidth;
    uint indirectRayHeight;
    uint indirectRayDepth;
    uint pad2;
};

struct SurfelPathTracerPayload {
    float3 radiance;
    float hitT;
    float3 hitPosition;
    float3 hitNormal;
    float3 baseColor;
    float3 emission;
    float3 f0;
    float ao;
    float metallic;
    float roughness;
    uint hitKind;
    uint instanceID;
};

SurfelPathTracerPayload makeEmptySurfelPayload()
{
    SurfelPathTracerPayload payload;
    payload.radiance = float3(0.0);
    payload.hitT = -1.0;
    payload.hitPosition = float3(0.0);
    payload.hitNormal = float3(0.0, 1.0, 0.0);
    payload.baseColor = float3(0.0);
    payload.emission = float3(0.0);
    payload.f0 = float3(0.0);
    payload.ao = 1.0;
    payload.metallic = 0.0;
    payload.roughness = 1.0;
    payload.hitKind = 0u;
    payload.instanceID = 0u;
    return payload;
}

SurfelPathTracerPayload makeVisibilitySurfelPayload()
{
    SurfelPathTracerPayload payload = makeEmptySurfelPayload();
    payload.hitT = 1.0;
    payload.hitKind = 1u;
    return payload;
}

uint makeSurfelMaterialKey(uint modelId, uint materialIndex)
{
    // Packs 16-bit model and material values into a single stable surfel key.
    return ((modelId & 0xFFFFu) << 16u) | (materialIndex & 0xFFFFu);
}

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

float2 dirToOctUv(float3 direction) {
    float3 n = normalize(direction);
    float invL1 = 1.0 / max(abs(n.x) + abs(n.y) + abs(n.z), 0.000001);
    float2 uv = n.xy * invL1;
    if (n.z < 0.0) {
        float2 signNotZero = float2(uv.x >= 0.0 ? 1.0 : -1.0, uv.y >= 0.0 ? 1.0 : -1.0);
        uv = (1.0 - abs(uv.yx)) * signNotZero;
    }
    return clamp(uv, float2(-1.0), float2(1.0));
}

float3 octUvToDir(float2 uv) {
    float3 n = float3(uv.x, uv.y, 1.0 - abs(uv.x) - abs(uv.y));
    if (n.z < 0.0) {
        float2 signNotZero = float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
        n.xy = (1.0 - abs(n.yx)) * signNotZero;
    }
    return normalize(n);
}

float3 unpackNormalOctahedral(uint packedNormal) {
    uint2 encoded = uint2(packedNormal & 0xFFFFu, packedNormal >> 16u);
    float2 oct = float2(encoded) / 65535.0 * 2.0 - 1.0;
    return octUvToDir(oct);
}

uint packRayLocalDirection(float3 localDirection) {
    float2 uv = dirToOctUv(localDirection);
    uint2 encoded = uint2(round(saturate(uv * 0.5 + 0.5) * 65535.0));
    return encoded.x | (encoded.y << 16u);
}

float3 unpackRayLocalDirection(uint packedDirection) {
    uint2 encoded = uint2(packedDirection & 0xFFFFu, packedDirection >> 16u);
    return octUvToDir(float2(encoded) / 65535.0 * 2.0 - 1.0);
}

void surfelTangentFrame(float3 normal, out float3 tangent, out float3 bitangent) {
    float3 n = normalize(normal);
    float3 up = abs(n.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(0.0, 1.0, 0.0);
    tangent = normalize(cross(up, n));
    bitangent = cross(n, tangent);
}

float3 worldToLocalDirection(float3 worldDirection, float3 normal) {
    float3 tangent;
    float3 bitangent;
    surfelTangentFrame(normal, tangent, bitangent);
    return normalize(float3(dot(worldDirection, tangent),
                            dot(worldDirection, bitangent),
                            dot(worldDirection, normalize(normal))));
}

float3 localToWorldDirection(float3 localDirection, float3 normal) {
    float3 tangent;
    float3 bitangent;
    surfelTangentFrame(normal, tangent, bitangent);
    float3 n = normalize(normal);
    return normalize(tangent * localDirection.x + bitangent * localDirection.y + n * localDirection.z);
}

uint2 atlasTileBase(uint surfelIndex, uint atlasWidth)
{
    uint tilesPerRow = max(atlasWidth / SURFEL_PT_ATLAS_TILE_SIZE, 1u);
    uint tileX = surfelIndex % tilesPerRow;
    uint tileY = surfelIndex / tilesPerRow;
    return uint2(tileX, tileY) * SURFEL_PT_ATLAS_TILE_SIZE;
}

uint2 atlasDirectionCoord(float3 localDirection)
{
    float2 uv = dirToOctUv(localDirection);
    float2 unitUv = saturate(uv * 0.5 + 0.5);
    uint2 offset = min(uint2(unitUv * float(SURFEL_PT_ATLAS_TILE_SIZE)),
                       uint2(SURFEL_PT_ATLAS_TILE_SIZE - 1u));
    return offset;
}

bool isFiniteScalar(float value) {
    return abs(value) < 3.402823e38;
}

bool isFinite3(float3 value) {
    return isFiniteScalar(value.x) &&
           isFiniteScalar(value.y) &&
           isFiniteScalar(value.z);
}

bool isActiveSurfel(SurfelPathTracerSurfel surfel)
{
    return (surfel.flags & SURFEL_PT_SURFEL_FLAG_ACTIVE) != 0u &&
           (surfel.flags & SURFEL_PT_SURFEL_FLAG_PENDING_FREE) == 0u &&
           surfel.radius > 0.0 &&
           isFinite3(surfel.position);
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
    if (!isActiveSurfel(surfel))
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
    if (!isActiveSurfel(surfel))
    {
        return false;
    }

    uint ageSinceSeen = frameIndex >= surfel.lastSeenFrame ? frameIndex - surfel.lastSeenFrame : 0u;
    uint ageSinceRef = frameIndex >= surfel.lastReferencedFrame ? frameIndex - surfel.lastReferencedFrame : 0u;
    bool pressure = aliveRatio > 0.85;
    uint representationGraceFrames = pressure ? 120u : 240u;
    bool unrepresented = ageSinceSeen > representationGraceFrames;
    bool distant = distanceToCamera > 160.0;
    bool distantAndUnused = distant && ageSinceSeen > 30u && ageSinceRef > 30u;
    return unrepresented || distantAndUnused;
}

uint adaptiveRayCountForSurfel(SurfelPathTracerSurfel surfel,
                               uint minRays,
                               uint maxRays,
                               float varianceSensitivity)
{
    uint safeMinRays = max(minRays, 1u);
    uint safeMaxRays = max(maxRays, safeMinRays);
    float variance = length(max(surfel.varianceAndInconsistency.xyz, float3(0.0)));
    // Give newly allocated surfels a short full-budget warm-up. Using
    // ageSinceSeen here kept every visible surfel permanently at max rays,
    // because RaySchedule marks it seen immediately before this call.
    float warmupBoost = surfel.shortMeanAndLife.w <= 16.0 ? 1.0 : 0.0;
    float varianceT = saturate(variance * varianceSensitivity + warmupBoost);
    uint count = uint(round(lerp(float(safeMinRays), float(safeMaxRays), varianceT)));
    if (surfel.sleepState == SURFEL_PT_SLEEP_SLEEPING)
    {
        count = max(safeMinRays, count / 4u);
    }
    return clamp(count, safeMinRays, safeMaxRays);
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

bool sampleSurfelIrradianceCell(float3 position,
                               float3 normal,
                               uint maxSamples,
                               uint sampleSeed,
                               uint maxSurfels,
                               float cellSize,
                               uint cellDimension,
                               float3 cameraPosition,
                               RWStructuredBuffer<SurfelPathTracerSurfel> surfels,
                               RWStructuredBuffer<SurfelPathTracerCellInfo> cells,
                               RWStructuredBuffer<uint> cellToSurfels,
                               out float3 irradiance)
{
    irradiance = float3(0.0);
    uint dim = max(cellDimension, 1u);
    int3 cellCoord = cameraRelativeCellCoord(position, cameraPosition, cellSize);
    if (!isCellCoordValid(cellCoord, dim))
    {
        return false;
    }

    uint cellIndex = flattenCameraRelativeCellCoord(cellCoord, dim);
    SurfelPathTracerCellInfo cellInfo = cells[cellIndex];
    uint slotCount = cellInfo.surfelCount;
    if (slotCount == 0u)
    {
        return false;
    }

    float3 weightedRadiance = float3(0.0);
    float weightSum = 0.0;
    uint sampleCount = min(slotCount, max(maxSamples, 1u));
    uint startSlot = pcgHash(sampleSeed) % slotCount;
    for (uint sampleIndex = 0u; sampleIndex < sampleCount; ++sampleIndex)
    {
        uint slot = (startSlot + sampleIndex) % slotCount;
        uint surfelIndex = cellToSurfels[cellInfo.surfelOffset + slot];
        if (surfelIndex == SURFEL_PT_INVALID_INDEX || surfelIndex >= maxSurfels)
        {
            continue;
        }

        SurfelPathTracerSurfel surfel = surfels[surfelIndex];
        if (!isActiveSurfel(surfel))
        {
            continue;
        }

        float3 surfelNormal = unpackNormalOctahedral(surfel.packedNormal);
        float normalWeight = saturate(dot(normal, surfelNormal));
        float distanceWeight = saturate(1.0 - length(position - surfel.position) /
                                             max(surfel.radius, 0.0001));
        float weight = normalWeight * distanceWeight;
        weightedRadiance += clampLuminance(surfel.radiance,
                                           SURFEL_PT_MAX_RADIANCE_LUMINANCE) * weight;
        weightSum += weight;
    }

    if (weightSum <= 0.0001)
    {
        return false;
    }

    irradiance = weightedRadiance / weightSum;
    return true;
}

float ggxPdf(float3 L, float3 N, float3 V, float roughness)
{
    float3 H = normalize(L + V);
    float nDotH = max(dot(N, H), 0.0);
    float vDotH = max(dot(V, H), 1e-5);
    if (nDotH <= 0.0) { return 0.0; }
    float D = distributionGGX(N, H, max(roughness, MIN_ROUGHNESS));
    return max(D * nDotH / (4.0 * vDotH), 1e-5);
}

float3 ggxSpecularBrdf(float3 L, float3 N, float3 V, float3 f0, float roughness)
{
    float nDotL = max(dot(N, L), 0.0);
    float nDotV = max(dot(N, V), 0.0);
    if (nDotL <= 0.0 || nDotV <= 0.0) { return float3(0.0); }
    float3 H = normalize(L + V);
    float D = distributionGGX(N, H, max(roughness, MIN_ROUGHNESS));
    float G = geometrySmith(N, V, L, max(roughness, MIN_ROUGHNESS));
    float3 F = fresnelSchlickRoughness(saturate(dot(H, V)), f0, roughness);
    return (D * G * F) / max(4.0 * nDotV * nDotL, 1e-5);
}

struct SurfelPtReservoir {
    float3 direction;
    float3 brdfWeight;
    float pdf;
    float weightSum;
    float selectedWeight;
};

void addReflectionCandidate(inout SurfelPtReservoir reservoir,
                            float3 direction,
                            float3 brdfWeight,
                            float pdf,
                            float candidateWeight,
                            float randomValue)
{
    reservoir.weightSum += candidateWeight;
    if (reservoir.weightSum > 0.0 && randomValue < candidateWeight / reservoir.weightSum)
    {
        reservoir.direction = direction;
        reservoir.brdfWeight = brdfWeight;
        reservoir.pdf = max(pdf, 1e-5);
        reservoir.selectedWeight = candidateWeight;
    }
}

float3 updateMsme(float3 sampleRadiance, inout SurfelPathTracerSurfel surfel, float shortBlend)
{
    float3 previousMean = surfel.meanAndVariance.xyz;
    float3 previousShort = surfel.shortMeanAndLife.xyz;
    float3 shortMean = lerp(previousShort, sampleRadiance, saturate(shortBlend));
    float3 longMean = lerp(previousMean, sampleRadiance, saturate(shortBlend * 0.25));
    float3 delta = sampleRadiance - longMean;
    float variance = lerp(surfel.meanAndVariance.w, dot(delta, delta), saturate(shortBlend * 0.5));
    surfel.meanAndVariance = float4(longMean, variance);
    surfel.shortMeanAndLife = float4(shortMean, surfel.shortMeanAndLife.w);
    surfel.varianceAndInconsistency = float4(abs(shortMean - longMean), variance);
    surfel.radiance = longMean;
    return longMean;
}

#endif // SURFEL_PATH_TRACER_COMMON_H
