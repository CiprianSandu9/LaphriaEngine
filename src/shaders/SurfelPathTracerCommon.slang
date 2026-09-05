#ifndef SURFEL_PATH_TRACER_COMMON_H
#define SURFEL_PATH_TRACER_COMMON_H

#include "ShaderCommon.slang"

static const uint SURFEL_PT_MAX_LIFE = 255u;
static const uint SURFEL_PT_INVALID_INDEX = 0xFFFFFFFFu;
// The atlas-backed capacity is below 2^19 (524,288). Compact cell entries use
// the remaining 13 bits for a deterministic per-world-cell retention priority.
static const uint SURFEL_PT_CELL_ENTRY_INDEX_BITS = 19u;
static const uint SURFEL_PT_CELL_ENTRY_INDEX_MASK = (1u << SURFEL_PT_CELL_ENTRY_INDEX_BITS) - 1u;
static const uint SURFEL_PT_CELL_ENTRY_PRIORITY_MASK = (1u << (32u - SURFEL_PT_CELL_ENTRY_INDEX_BITS)) - 1u;
static const uint SURFEL_PT_ATLAS_TILE_SIZE = 6u;
static const uint SURFEL_PT_SLEEP_AWAKE = 0u;
static const uint SURFEL_PT_SLEEP_SLEEPING = 1u;
static const uint SURFEL_PT_SURFEL_FLAG_ACTIVE = 1u;
static const uint SURFEL_PT_SURFEL_FLAG_PENDING_FREE = 2u;
static const uint SURFEL_PT_SURFEL_FLAG_RAY_SCHEDULED = 4u;
// Set by Update while the surfel overlaps the camera-relative lookup window.
// Lets Update detect a real outside->inside transition instead of re-firing the
// re-entry grace every other frame for surfels that merely fell out of the
// compact map (which would make them immortal).
static const uint SURFEL_PT_SURFEL_FLAG_IN_WINDOW = 8u;
// Set by Integrate once the surfel has accumulated at least one traced sample.
// Until then the first sample replaces the placement seed outright.
static const uint SURFEL_PT_SURFEL_FLAG_HAS_SAMPLES = 16u;
// Bits 24..31 of flags: saturating count of integrated sample batches. Only
// Integrate changes it (full struct write); the atomic OR users of the low bits
// never touch it. Drives the cumulative-average warm-up blend.
static const uint SURFEL_PT_SURFEL_SAMPLE_COUNT_SHIFT = 24u;
static const uint SURFEL_PT_SURFEL_SAMPLE_COUNT_MAX = 255u;
// Luminance floor for the relative-inconsistency metric (see
// surfelRelativeInconsistency). Roughly the darkest irradiance worth chasing.
static const float SURFEL_PT_RELATIVE_NOISE_FLOOR = 0.05;
// Warm-up length in frames: full ray budget, no sleeping, no removal.
static const float SURFEL_PT_WARMUP_LIFE = 16.0;
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
// Unclamped ray demand from RaySchedule. REQUESTED_RAYS is clamped to the
// frame budget, so this is the only way to measure budget saturation.
static const uint SURFEL_PT_COUNTER_DEMANDED_RAYS_OFFSET = 76u;
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
    uint demandedRays;
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

int3 worldCellCoord(float3 position, float cellSize)
{
    return int3(round(position / max(cellSize, 0.0001)));
}

uint hashWorldCellCoord(int3 cellCoord)
{
    uint hash = pcgHash(asuint(cellCoord.x) ^ 0x9E3779B9u);
    hash = pcgHash(hash ^ asuint(cellCoord.y) ^ 0x85EBCA6Bu);
    return pcgHash(hash ^ asuint(cellCoord.z) ^ 0xC2B2AE35u);
}

static const uint SURFEL_PT_CELL_RETENTION_KEY_BITS = 6u;
static const uint SURFEL_PT_CELL_RETENTION_KEY_MAX = (1u << SURFEL_PT_CELL_RETENTION_KEY_BITS) - 1u;
static const uint SURFEL_PT_CELL_RETENTION_HASH_BITS = 32u - SURFEL_PT_CELL_ENTRY_INDEX_BITS -
                                                       SURFEL_PT_CELL_RETENTION_KEY_BITS;
static const uint SURFEL_PT_CELL_RETENTION_HASH_MASK = (1u << SURFEL_PT_CELL_RETENTION_HASH_BITS) - 1u;
// Stale entries re-shuffle their tie-break every 2^shift frames. A surfel that
// rotates back into the map needs a few frames to be traced and referenced.
static const uint SURFEL_PT_CELL_RETENTION_ROTATION_SHIFT = 2u;

// Retention key for the compact cell map: smaller keys survive when a cell holds
// more surfels than its limit. The key is the age since a visible pixel last
// referenced the surfel, so the retained set of a full cell is the set the
// screen actually uses instead of a hash lottery that starved e.g. wall surfels
// next to dense foliage. Surfels still in warm-up are capped at key 9: they beat
// anything unreferenced for more than eight frames (so a new surfel can enter a
// cell full of stale entries and be evaluated), but they cannot displace surfels
// the screen used in the last eight frames. Without that cap a burst of
// placements would evict every referenced surfel in the cell.
// The age saturates after ~60 frames. Only map members can be referenced, so
// without a cap a surfel once evicted could never out-rank a stale retained one
// and would sit untraced and unreferenced forever. Saturated ("stale") entries
// instead compete through a slowly rotating hash (see packCellSurfelEntry), so
// every stale surfel periodically re-enters the map, gets rays, and has a chance
// to be referenced again.
static const uint SURFEL_PT_CELL_RETENTION_WARMUP_KEY = 9u;
uint cellRetentionKey(SurfelPathTracerSurfel surfel, uint frameIndex)
{
    uint age = frameIndex >= surfel.lastReferencedFrame ? frameIndex - surfel.lastReferencedFrame : 0u;
    uint key = min(age + 1u, SURFEL_PT_CELL_RETENTION_KEY_MAX);
    if (surfel.shortMeanAndLife.w <= 16.0)
    {
        key = min(key, SURFEL_PT_CELL_RETENTION_WARMUP_KEY);
    }
    return key;
}

uint packCellSurfelEntry(uint surfelIndex, int3 absoluteCellCoord, uint retentionKey, uint frameIndex)
{
    // 13 priority bits: 6 bits of retention key, then 7 hash bits as an
    // order-independent tie-break among equally useful surfels. The tie-break is
    // stable for recently used surfels (no flicker among them) and rotates every
    // few frames for stale ones so eviction from the map is never permanent.
    uint clampedKey = min(retentionKey, SURFEL_PT_CELL_RETENTION_KEY_MAX);
    uint rotation = clampedKey >= SURFEL_PT_CELL_RETENTION_KEY_MAX
                    ? (frameIndex >> SURFEL_PT_CELL_RETENTION_ROTATION_SHIFT) * 0x9E3779B9u
                    : 0u;
    uint hashBits = pcgHash(surfelIndex ^ hashWorldCellCoord(absoluteCellCoord) ^ rotation) &
                    SURFEL_PT_CELL_RETENTION_HASH_MASK;
    uint priority = ((clampedKey << SURFEL_PT_CELL_RETENTION_HASH_BITS) | hashBits) &
                    SURFEL_PT_CELL_ENTRY_PRIORITY_MASK;
    return (priority << SURFEL_PT_CELL_ENTRY_INDEX_BITS) |
           (surfelIndex & SURFEL_PT_CELL_ENTRY_INDEX_MASK);
}

uint unpackCellSurfelIndex(uint entry)
{
    return entry == SURFEL_PT_INVALID_INDEX
           ? SURFEL_PT_INVALID_INDEX
           : (entry & SURFEL_PT_CELL_ENTRY_INDEX_MASK);
}

int3 cameraRelativeCellCoord(float3 position, float3 cameraPosition, float cellSize)
{
    // Keep the bounded lookup window camera-relative, but anchor its cells in
    // world space. Sub-cell camera motion must not translate every lookup cell
    // and reshuffle otherwise stationary surfels.
    return worldCellCoord(position, cellSize) - worldCellCoord(cameraPosition, cellSize);
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
    int3 cameraCell = worldCellCoord(cameraPosition, safeCellSize);
    float3 cellCenter = float3(cellCoord + cameraCell) * safeCellSize;
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
    // Ray scheduling marks every surfel represented by the camera-centered
    // spatial window as seen. In compact scenes that includes surfels behind
    // the camera and behind walls, so representation alone cannot be allowed
    // to keep the cache alive indefinitely. Once capacity is under pressure,
    // reclaim entries that have not actually been selected by a visible
    // resolve pixel for several seconds. Newly placed and recently revisited
    // surfels retain a generous grace period to avoid camera-motion churn.
    // Retire only a small deterministic fraction per frame. Thousands of GPU
    // invocations can observe the same near-full counter concurrently; without
    // throttling they could all evict at once and turn capacity recovery into a
    // visible cache flush.
    bool evictionTurn = (pcgHash(surfel.materialKey ^ surfel.packedNormal ^
                                 asuint(surfel.position.x) ^ frameIndex) & 63u) == 0u;
    bool unreferencedUnderPressure = pressure && ageSinceRef > 480u && evictionTurn;
    bool distant = distanceToCamera > 160.0;
    bool distantAndUnused = distant && ageSinceSeen > 30u && ageSinceRef > 30u;
    return unrepresented || unreferencedUnderPressure || distantAndUnused;
}

uint surfelSampleCount(SurfelPathTracerSurfel surfel)
{
    return surfel.flags >> SURFEL_PT_SURFEL_SAMPLE_COUNT_SHIFT;
}

uint withSurfelSampleCount(uint flags, uint sampleCount)
{
    return (flags & ((1u << SURFEL_PT_SURFEL_SAMPLE_COUNT_SHIFT) - 1u)) |
           (min(sampleCount, SURFEL_PT_SURFEL_SAMPLE_COUNT_MAX) << SURFEL_PT_SURFEL_SAMPLE_COUNT_SHIFT);
}

// Inconsistency (|short mean - long mean|) relative to the surfel's own
// brightness. The absolute metric let every dark surfel fall to the minimum ray
// count and go to sleep although its relative noise was the highest in the
// scene, which showed as shimmer in shadowed areas.
float surfelRelativeInconsistency(SurfelPathTracerSurfel surfel)
{
    const float3 lumaWeights = float3(0.2126, 0.7152, 0.0722);
    float inconsistency = dot(max(surfel.varianceAndInconsistency.xyz, float3(0.0)), lumaWeights);
    float reference = dot(max(surfel.meanAndVariance.xyz, float3(0.0)), lumaWeights);
    return inconsistency / max(reference, SURFEL_PT_RELATIVE_NOISE_FLOOR);
}

uint adaptiveRayCountForSurfel(SurfelPathTracerSurfel surfel,
                               uint minRays,
                               uint maxRays,
                               float varianceSensitivity)
{
    uint safeMinRays = max(minRays, 1u);
    uint safeMaxRays = max(maxRays, safeMinRays);
    float variance = surfelRelativeInconsistency(surfel);
    // Give newly allocated surfels a short full-budget warm-up. Using
    // ageSinceSeen here kept every visible surfel permanently at max rays,
    // because RaySchedule marks it seen immediately before this call.
    float warmupBoost = surfel.shortMeanAndLife.w <= SURFEL_PT_WARMUP_LIFE ? 1.0 : 0.0;
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

// excludeSurfelIndex: a surfel querying its own neighbourhood (radiance sharing)
// must not read itself back, or the "shared" value is dominated by its own
// previous estimate and the blend degenerates into extra temporal lag.
bool sampleSurfelIrradianceCell(float3 position,
                               float3 normal,
                               uint maxSamples,
                               uint sampleSeed,
                               uint excludeSurfelIndex,
                               uint searchShells,
                               uint maxSurfels,
                               float cellSize,
                               float supportRadius,
                               uint cellDimension,
                               float3 cameraPosition,
                               RWStructuredBuffer<SurfelPathTracerSurfel> surfels,
                               RWStructuredBuffer<SurfelPathTracerCellInfo> cells,
                               RWStructuredBuffer<uint> cellToSurfels,
                               out float3 irradiance)
{
    irradiance = float3(0.0);
    uint dim = max(cellDimension, 1u);
    int3 centerCoord = cameraRelativeCellCoord(position, cameraPosition, cellSize);
    if (!isCellCoordValid(centerCoord, dim))
    {
        return false;
    }

    float3 weightedRadiance = float3(0.0);
    float weightSum = 0.0;
    uint samplesVisited = 0u;
    uint softBudget = max(maxSamples, 1u);
    // Search the 3x3x3 neighbourhood (searchShells = 1), centre cell first,
    // exactly like the screen-space resolve. The support radius is clamped to the
    // cell size on the host, so every surfel that can reach `position` lives in
    // one of these cells; a single-cell lookup missed hit points near cell
    // boundaries and made termination and sharing success depend on the cell
    // size. Callers with ~1M lookups per frame (glossy reflections) pass
    // searchShells = 0 to keep the centre-cell-only cost.
    // Each cell's retained list is consumed completely (its slot order comes from
    // atomics and is not stable); maxSamples is a soft budget between cells.
    int maxShell = int(min(searchShells, 1u));
    for (int shell = 0; shell <= maxShell; ++shell)
    {
        for (int z = -shell; z <= shell; ++z)
        for (int y = -shell; y <= shell; ++y)
        for (int x = -shell; x <= shell; ++x)
        {
            if (samplesVisited >= softBudget)
            {
                break;
            }
            if (max(abs(x), max(abs(y), abs(z))) != shell)
            {
                continue;
            }
            int3 cellCoord = centerCoord + int3(x, y, z);
            if (!isCellCoordValid(cellCoord, dim))
            {
                continue;
            }
            uint cellIndex = flattenCameraRelativeCellCoord(cellCoord, dim);
            SurfelPathTracerCellInfo cellInfo = cells[cellIndex];
            for (uint slot = 0u; slot < cellInfo.surfelCount; ++slot)
            {
                uint surfelIndex = unpackCellSurfelIndex(cellToSurfels[cellInfo.surfelOffset + slot]);
                if (surfelIndex == SURFEL_PT_INVALID_INDEX || surfelIndex >= maxSurfels ||
                    surfelIndex == excludeSurfelIndex)
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
                // Same support radius and squared falloff as the screen-space
                // resolve (Evaluate). Weighting by the raw surfel radius (often
                // the 5 cm minimum) made this lookup miss almost every hit point.
                float influenceRadius = max(surfel.radius, supportRadius);
                float distanceWeight = saturate(1.0 - length(position - surfel.position) /
                                                     max(influenceRadius, 0.0001));
                float weight = normalWeight * distanceWeight * distanceWeight;
                if (weight <= 0.0001)
                {
                    continue;
                }
                samplesVisited += 1u;
                weightedRadiance += clampLuminance(surfel.radiance,
                                                   SURFEL_PT_MAX_RADIANCE_LUMINANCE) * weight;
                weightSum += weight;
            }
        }
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

float3 updateMsme(float3 sampleRadiance, inout SurfelPathTracerSurfel surfel, float shortBlend, float longBlend)
{
    float3 previousMean = surfel.meanAndVariance.xyz;
    float3 previousShort = surfel.shortMeanAndLife.xyz;
    float3 shortMean = lerp(previousShort, sampleRadiance, saturate(shortBlend));
    float3 longMean = lerp(previousMean, sampleRadiance, saturate(longBlend));
    float3 delta = sampleRadiance - longMean;
    float variance = lerp(surfel.meanAndVariance.w, dot(delta, delta), saturate(shortBlend * 0.5));
    surfel.meanAndVariance = float4(longMean, variance);
    surfel.shortMeanAndLife = float4(shortMean, surfel.shortMeanAndLife.w);
    surfel.varianceAndInconsistency = float4(abs(shortMean - longMean), variance);
    surfel.radiance = longMean;
    return longMean;
}

#endif // SURFEL_PATH_TRACER_COMMON_H
