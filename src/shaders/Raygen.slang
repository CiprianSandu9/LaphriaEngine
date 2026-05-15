#include "ShaderCommon.slang"

// Path tracer RayPayload — surface properties returned by ClosestHit to Raygen.
// hitT < 0 is the escape sentinel set by Miss to indicate a sky hit.
// Total size: 3+3+3+3+1+1+3+1+1+1 floats × 4 = 20 × 4 = 80 bytes (within 128-byte limit).
struct RayPayload {
    float3 albedo;       // Linear base color × (1 - metallic) diffuse weight
    float3 emission;     // Emissive radiance; Miss writes sky radiance here
    float3 worldNormal;  // Shading normal in world space (after normal mapping)
    float3 hitPos;       // World-space position of the surface hit
    float  roughness;    // PBR roughness [MIN_ROUGHNESS, 1]
    float  metallic;     // PBR metallic [0, 1]
    float3 F0;           // Fresnel base reflectance (precomputed from albedo + metallic)
    float  ao;           // Baked ambient occlusion factor [0, 1]
    float  hitT;         // Ray t-value; negative means ray escaped to sky (Miss fired)
    uint   instanceID;   // InstanceID() from ClosestHit; 0 on sky hit
};

// Set 0 — RT descriptor set
[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(1, 0)]] RWTexture2D<float4> noisyColorOutput;   // 1 SPP raw radiance
[[vk::binding(2, 0)]] RWTexture2D<float4> gBufferNormals;     // world-space normal (xyz), w unused
[[vk::binding(3, 0)]] RWTexture2D<float>  gBufferDepth;       // linear ray hit distance (negative = sky)
[[vk::binding(4, 0)]] RWTexture2D<float2> motionVectors;      // screen-space UV offset current→previous
[[vk::binding(9, 0)]] RWByteAddressBuffer ptAnalysisCounters;
[[vk::binding(10, 0)]] RWByteAddressBuffer ptSunVisibleCandidateCache;
[[vk::binding(11, 0)]] RWByteAddressBuffer ptSunVisibleConnectionCache;
[[vk::binding(12, 0)]] RWByteAddressBuffer ptReservoirGiCurrent;
[[vk::binding(13, 0)]] RWByteAddressBuffer ptReservoirGiHistory;

// Set 1 — global UBO
[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

[[vk::push_constant]] ScenePushConstants push;

static const int SUN_VISIBLE_CANDIDATE_CACHE_BINDING = 10;
static const int SUN_VISIBLE_CONNECTION_CACHE_BINDING = 11;
static const int RESERVOIR_GI_CURRENT_BINDING = 12;
static const int RESERVOIR_GI_HISTORY_BINDING = 13;
static const uint RESERVOIR_GI_CURRENT_CAPACITY = 262144u;
static const uint SUN_VISIBLE_CANDIDATE_CACHE_CAPACITY = 4096;
static const uint SUN_VISIBLE_CANDIDATE_CACHE_BANK_COUNT = 2;
static const uint SUN_VISIBLE_CANDIDATE_CACHE_BANK_CAPACITY =
    SUN_VISIBLE_CANDIDATE_CACHE_CAPACITY / SUN_VISIBLE_CANDIDATE_CACHE_BANK_COUNT;
static const uint SUN_VISIBLE_CANDIDATE_CACHE_FRAME_SLOT_COUNT = 2;
static const uint SUN_VISIBLE_CANDIDATE_CACHE_HEADER_SIZE = 16;
static const uint SUN_VISIBLE_CANDIDATE_RECORD_SIZE = 96;
static const uint SUN_VISIBLE_CONNECTION_CACHE_CAPACITY = 4096;
static const uint SUN_VISIBLE_CONNECTION_CACHE_BANK_COUNT = 2;
static const uint SUN_VISIBLE_CONNECTION_CACHE_BANK_CAPACITY =
    SUN_VISIBLE_CONNECTION_CACHE_CAPACITY / SUN_VISIBLE_CONNECTION_CACHE_BANK_COUNT;
static const uint SUN_VISIBLE_CONNECTION_CACHE_FRAME_SLOT_COUNT = 2;
static const uint SUN_VISIBLE_CONNECTION_CACHE_HEADER_SIZE = 16;
static const uint SUN_VISIBLE_CONNECTION_RECORD_SIZE = 128;
static const uint RESERVOIR_GI_CURRENT_HEADER_SIZE = 16;
static const uint RESERVOIR_GI_RECORD_SIZE = 96;

static const int DEBUG_AOV_PATH_RAW_FINAL_COLOR   = 6;
static const int DEBUG_AOV_PATH_DIRECT_LIGHTING   = 7;
static const int DEBUG_AOV_PATH_INDIRECT_LIGHTING = 8;
static const int DEBUG_AOV_PATH_SKY_CONTRIBUTION  = 9;
static const int DEBUG_AOV_PATH_THROUGHPUT        = 10;
static const int DEBUG_AOV_PATH_BOUNCE_COUNT      = 11;
static const int DEBUG_AOV_PATH_SHADOW_VISIBILITY = 12;
static const int DEBUG_AOV_PATH_ENVIRONMENT_NEE   = 13;
static const int DEBUG_AOV_PATH_FIRST_HIT_BOUNCE  = 14;
static const int DEBUG_AOV_PATH_SECONDARY_SUN     = 15;
static const int DEBUG_AOV_PATH_BASELINE_CONTINUATION = 16;
static const int ENV_NEE_SAMPLING_COSINE_HEMISPHERE = 0;
static const int ENV_NEE_SAMPLING_SKY_BIASED = 1;
static const int FIRST_HIT_PROBE_SAMPLING_COSINE_HEMISPHERE = 0;
static const int FIRST_HIT_PROBE_SAMPLING_SUN_BOUNCE_GUIDED = 1;
static const int FIRST_HIT_PROBE_SAMPLING_CANDIDATE_SUN_BOUNCE = 2;
static const int FIRST_HIT_PROBE_SAMPLING_CANDIDATE_AVERAGE_REFERENCE = 3;
static const int FIRST_HIT_PROBE_SAMPLING_CANDIDATE_RIS = 4;
static const int FIRST_HIT_PROBE_SAMPLING_CACHED_SECONDARY_REUSE = 5;
static const int PATH_TRACER_RESERVOIR_GI_OFF = 0;
static const int PATH_TRACER_RESERVOIR_GI_SINGLE_FRAME = 1;
static const int PATH_TRACER_RESERVOIR_GI_TEMPORAL = 2;
static const int PATH_TRACER_RESERVOIR_GI_TEMPORAL_SPATIAL = 3;
static const uint PATH_TRACER_APPLY_FIRST_HIT_PROBES_BIT = 1u << 2;
static const uint SUN_VISIBLE_CACHE_ENABLE_BIT = 1u << 16;
static const uint ADAPTIVE_CACHE_REFRESH_BIT = 1u << 17;
static const uint CACHE_WEIGHTING_MODE_SHIFT = 18;
static const uint CACHE_REFRESH_COUNT_SHIFT = 20;
static const uint CACHE_SPATIAL_TRIAL_COUNT_SHIFT = 23;
static const uint TARGETED_DIAGNOSTIC_CACHE_REFRESH_BIT = 1u << 29;
static const uint CACHE_VISIBILITY_BUDGET_SHIFT = 30;
static const int CACHE_WEIGHTING_CALIBRATED = 0;
static const int CACHE_WEIGHTING_MIS_APPROX = 1;
static const int CACHE_PROPOSAL_SPATIAL_LOCAL = 0;
static const int CACHE_PROPOSAL_GLOBAL_NON_LOCAL = 1;
static const int CACHE_PROPOSAL_CONNECTION_CACHE = 2;
static const float CACHED_REUSE_DEFAULT_CONNECTION_DISTANCE = 3.5;
static const uint CACHED_REUSE_MAX_RECORD_AGE_FRAMES = 180;
static const int CACHED_REUSE_SPATIAL_CANDIDATE_TRIALS = 32;
static const int CACHED_REUSE_GLOBAL_FALLBACK_TRIALS = 8;
static const int CACHED_REUSE_MAX_RADIUS_CELLS = 8;
static const float CACHED_REUSE_SPATIAL_CELL_SIZE = CACHED_REUSE_DEFAULT_CONNECTION_DISTANCE;
static const uint CACHED_REUSE_SPATIAL_BUCKET_SIZE = 128u;
static const uint SUN_VISIBLE_CACHE_SLOT_EMPTY = 0u;
static const uint SUN_VISIBLE_CACHE_SLOT_WRITING = 1u;
static const uint SUN_VISIBLE_CACHE_SLOT_READY = 2u;
static const uint targetWallFirstHitProbeContributionSumOffset = 88u;
static const uint targetWallBaseLuminanceSumOffset = 92u;
static const uint diagnosticTargetCacheReuseAttemptCountOffset = 96u;
static const uint diagnosticTargetCacheSelectedCountOffset = 100u;
static const uint diagnosticTargetCacheRejectDistanceCountOffset = 104u;
static const uint diagnosticTargetCacheRejectGeometryCountOffset = 108u;
static const uint diagnosticTargetCacheRejectVisibilityCountOffset = 112u;
static const uint diagnosticTargetCacheReuseAcceptedCountOffset = 116u;
static const uint diagnosticTargetCacheReuseContributionSumOffset = 120u;
static const uint cacheReusePathEntryCountOffset = 124u;
static const uint cacheReuseExtraSampleZeroCountOffset = 128u;
static const uint cacheReuseUnavailableCandidateCountOffset = 132u;
static const uint cacheReuseSelectedCountOffset = 136u;
static const uint cacheReuseSelectMissCountOffset = 140u;
static const uint cacheReuseRejectDistanceCountOffset = 144u;
static const uint cacheReuseRejectGeometryCountOffset = 148u;
static const uint cacheReuseRejectVisibilityCountOffset = 152u;
static const uint cacheReuseSelectLoadSuccessCountOffset = 156u;
static const uint cacheReuseSelectLoadMissCountOffset = 160u;
static const uint cacheReuseSelectRejectDistanceCountOffset = 164u;
static const uint cacheReuseSelectRejectGeometryCountOffset = 168u;
static const uint cacheReuseAcceptedDistanceNearCountOffset = 172u;
static const uint cacheReuseAcceptedDistanceMidCountOffset = 176u;
static const uint cacheReuseAcceptedDistanceFarCountOffset = 180u;
static const uint cacheReuseAcceptedDistanceVeryFarCountOffset = 184u;
static const uint cacheReuseAcceptedLumaDarkCountOffset = 188u;
static const uint cacheReuseAcceptedLumaDimCountOffset = 192u;
static const uint cacheReuseAcceptedLumaUsefulCountOffset = 196u;
static const uint cacheReuseAcceptedLumaBrightCountOffset = 200u;
static const uint cacheConnectionReuseAttemptsOffset = 204u;
static const uint cacheConnectionReuseAcceptedOffset = 208u;
static const uint reservoirGiCandidatesOffset = 212u;
static const uint reservoirGiAcceptedOffset = 216u;
static const uint reservoirGiLumaScaledSumOffset = 220u;
static const uint reservoirGiTemporalAcceptedOffset = 224u;
static const uint reservoirGiTemporalRejectedOffset = 228u;
static const uint reservoirGiSpatialAcceptedOffset = 232u;
static const uint reservoirGiSpatialRejectedOffset = 236u;

static const int CACHED_REUSE_REJECT_NONE = 0;
static const int CACHED_REUSE_REJECT_DISTANCE = 1;
static const int CACHED_REUSE_REJECT_GEOMETRY = 2;
static const int CACHED_REUSE_REJECT_VISIBILITY = 3;

struct FirstHitDiffuseBounceResult {
    float3 totalContribution;
    float3 secondaryDirectSunContribution;
};

struct FirstHitProbeSample {
    float3 direction;
    float  pdf;
};

struct CachedSunVisibleCandidate {
    float3 position;
    float3 normal;
    float3 albedo;
    float3 F0;
    float  roughness;
    float  metallic;
};

struct ReservoirGiRecord {
    float3 primaryPosition;
    float3 candidatePosition;
    float3 candidateNormal;
    float3 contribution;
    float3 secondaryDirectSun;
    float  weightSum;
    float  selectedWeight;
    uint   frameId;
    uint   flags;
};

RayPayload makeVisibilityPayload()
{
    RayPayload payload;
    payload.hitT        = 1.0;
    payload.albedo      = float3(0.0, 0.0, 0.0);
    payload.emission    = float3(0.0, 0.0, 0.0);
    payload.worldNormal = float3(0.0, 1.0, 0.0);
    payload.hitPos      = float3(0.0, 0.0, 0.0);
    payload.roughness   = 1.0;
    payload.metallic    = 0.0;
    payload.F0          = float3(0.0, 0.0, 0.0);
    payload.ao          = 1.0;
    payload.instanceID  = 0;
    return payload;
}

float powerHeuristic(float sampleCountA, float pdfA, float sampleCountB, float pdfB)
{
    float a = max(sampleCountA, 0.0) * max(pdfA, 0.0);
    float b = max(sampleCountB, 0.0) * max(pdfB, 0.0);
    float a2 = a * a;
    float b2 = b * b;
    float denom = a2 + b2;
    return (denom > 0.0) ? (a2 / denom) : 0.0;
}

float environmentSamplePdf(float3 envDir, float3 N)
{
    return max(dot(N, envDir), 0.0) / PI;
}

float skyBiasedEnvironmentPdf(float3 envDir)
{
    return max(envDir.y, 0.0) / PI;
}

float environmentSamplePdf(float3 envDir, float3 N, int samplingMode)
{
    if (samplingMode == ENV_NEE_SAMPLING_SKY_BIASED) {
        return skyBiasedEnvironmentPdf(envDir);
    }
    return environmentSamplePdf(envDir, N);
}

float bsdfPdfForEnvironmentDirection(float3 envDir, float3 N, float diffuseSelectionProbability)
{
    return max(diffuseSelectionProbability, 0.0) * environmentSamplePdf(envDir, N);
}

float3 sampleSkyBiasedEnvironmentDirection(float2 xi)
{
    return cosineSampleHemisphere(xi, float3(0.0, 1.0, 0.0));
}

bool environmentNeeEnabledForBounce(int bounce)
{
    return bounce == 0;
}

float pathTracerLuminance(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

void recordTargetWallLuminance(float3 color,
                               float3 firstHitProbeContribution,
                               uint primaryInstanceID,
                               bool primarySurfaceHit)
{
    if (!primarySurfaceHit || push.cascadeIndex < 0) {
        return;
    }

    uint primaryModelId = primaryInstanceID >> 14;
    if (primaryModelId != uint(push.cascadeIndex)) {
        return;
    }

    float luma = clamp(pathTracerLuminance(color), 0.0, 8.0);
    float probeLuma = clamp(pathTracerLuminance(firstHitProbeContribution), 0.0, 8.0);
    float3 baseColor = max(color - firstHitProbeContribution, float3(0.0, 0.0, 0.0));
    float baseLuma = clamp(pathTracerLuminance(baseColor), 0.0, 8.0);
    uint scaledLuma = uint(luma * 64.0 + 0.5);
    uint scaledProbeLuma = uint(probeLuma * 64.0 + 0.5);
    uint scaledBaseLuma = uint(baseLuma * 64.0 + 0.5);
    ptAnalysisCounters.InterlockedAdd(20, 1u);
    ptAnalysisCounters.InterlockedAdd(24, scaledLuma);
    ptAnalysisCounters.InterlockedAdd(targetWallFirstHitProbeContributionSumOffset, scaledProbeLuma);
    ptAnalysisCounters.InterlockedAdd(targetWallBaseLuminanceSumOffset, scaledBaseLuma);
}

void recordFirstHitProbeStats(bool surfaceHit, bool sunVisible, float3 contribution)
{
    float luma = clamp(pathTracerLuminance(contribution), 0.0, 8.0);
    uint scaledLuma = uint(luma * 64.0 + 0.5);
    ptAnalysisCounters.InterlockedAdd(32, 1u);
    if (surfaceHit) {
        ptAnalysisCounters.InterlockedAdd(36, 1u);
    }
    if (sunVisible) {
        ptAnalysisCounters.InterlockedAdd(40, 1u);
        ptAnalysisCounters.InterlockedAdd(48, scaledLuma);
    }
    ptAnalysisCounters.InterlockedAdd(44, scaledLuma);
}

uint reservoirGiRecordOffset(uint pixelIndex)
{
    return RESERVOIR_GI_CURRENT_HEADER_SIZE + pixelIndex * RESERVOIR_GI_RECORD_SIZE;
}

void storeFloat3ToReservoirGi(uint offset, float3 value)
{
    ptReservoirGiCurrent.Store(offset + 0, asuint(value.x));
    ptReservoirGiCurrent.Store(offset + 4, asuint(value.y));
    ptReservoirGiCurrent.Store(offset + 8, asuint(value.z));
}

float3 loadFloat3FromReservoirGiHistory(uint offset)
{
    return float3(asfloat(ptReservoirGiHistory.Load(offset + 0)),
                  asfloat(ptReservoirGiHistory.Load(offset + 4)),
                  asfloat(ptReservoirGiHistory.Load(offset + 8)));
}

bool isSaneReservoirGiVector(float3 value, float maxAbs)
{
    return all(abs(value) <= float3(maxAbs, maxAbs, maxAbs));
}

float3 sanitizeReservoirGiContribution(float3 value)
{
    if (!isSaneReservoirGiVector(value, 64.0)) {
        return float3(0.0, 0.0, 0.0);
    }
    return clamp(value, float3(0.0, 0.0, 0.0), float3(64.0, 64.0, 64.0));
}

void updateReservoirGi(uint2 launchID,
                       uint2 launchSize,
                       ReservoirGiRecord record)
{
    uint pixelIndex = launchID.y * launchSize.x + launchID.x;
    if (launchID.x == 0u && launchID.y == 0u) {
        ptReservoirGiCurrent.Store(0, min(launchSize.x * launchSize.y, RESERVOIR_GI_CURRENT_CAPACITY));
        ptReservoirGiCurrent.Store(4, launchSize.x);
        ptReservoirGiCurrent.Store(8, launchSize.y);
        ptReservoirGiCurrent.Store(12, record.frameId);
    }
    if (pixelIndex >= RESERVOIR_GI_CURRENT_CAPACITY) {
        return;
    }

    uint offset = reservoirGiRecordOffset(pixelIndex);

    storeFloat3ToReservoirGi(offset + 0, record.primaryPosition);
    storeFloat3ToReservoirGi(offset + 16, record.candidatePosition);
    storeFloat3ToReservoirGi(offset + 32, record.candidateNormal);
    storeFloat3ToReservoirGi(offset + 48, record.contribution);
    storeFloat3ToReservoirGi(offset + 64, record.secondaryDirectSun);
    ptReservoirGiCurrent.Store(offset + 80, asuint(record.weightSum));
    ptReservoirGiCurrent.Store(offset + 84, asuint(record.selectedWeight));
    ptReservoirGiCurrent.Store(offset + 88, record.frameId);
    ptReservoirGiCurrent.Store(offset + 92, record.flags);
}

ReservoirGiRecord loadReservoirGiHistoryRecord(uint offset)
{
    ReservoirGiRecord record;
    record.primaryPosition = loadFloat3FromReservoirGiHistory(offset + 0);
    record.candidatePosition = loadFloat3FromReservoirGiHistory(offset + 16);
    record.candidateNormal = loadFloat3FromReservoirGiHistory(offset + 32);
    record.contribution = loadFloat3FromReservoirGiHistory(offset + 48);
    record.secondaryDirectSun = loadFloat3FromReservoirGiHistory(offset + 64);
    record.weightSum = asfloat(ptReservoirGiHistory.Load(offset + 80));
    record.selectedWeight = asfloat(ptReservoirGiHistory.Load(offset + 84));
    record.frameId = ptReservoirGiHistory.Load(offset + 88);
    record.flags = ptReservoirGiHistory.Load(offset + 92);
    return record;
}

// Debug temporal path: binding 13 is the previous frame slot's reservoir buffer,
// so this deliberately does same-pixel reuse instead of motion-vector reprojection.
bool loadTemporalReservoirGi(uint2 pixel,
                             uint2 launchSize,
                             float3 currentPrimaryNormal,
                             float currentPrimaryDepth,
                             out ReservoirGiRecord record)
{
    record.primaryPosition = float3(0.0, 0.0, 0.0);
    record.candidatePosition = float3(0.0, 0.0, 0.0);
    record.candidateNormal = float3(0.0, 0.0, 0.0);
    record.contribution = float3(0.0, 0.0, 0.0);
    record.secondaryDirectSun = float3(0.0, 0.0, 0.0);
    record.weightSum = 0.0;
    record.selectedWeight = 0.0;
    record.frameId = 0u;
    record.flags = 0u;

    uint pixelIndex = pixel.y * launchSize.x + pixel.x;
    if (pixelIndex >= RESERVOIR_GI_CURRENT_CAPACITY) {
        return false;
    }

    uint historyCapacity = ptReservoirGiHistory.Load(0);
    uint historyWidth = ptReservoirGiHistory.Load(4);
    uint historyHeight = ptReservoirGiHistory.Load(8);
    if (historyCapacity == 0u ||
        historyCapacity > RESERVOIR_GI_CURRENT_CAPACITY ||
        historyWidth != launchSize.x ||
        historyHeight != launchSize.y ||
        pixelIndex >= historyCapacity) {
        return false;
    }

    uint offset = reservoirGiRecordOffset(pixelIndex);
    record = loadReservoirGiHistoryRecord(offset);
    if ((record.flags & 1u) == 0u) {
        return false;
    }
    if (record.frameId == 0u || record.frameId + 1u != ubo.frameCount) {
        return false;
    }
    if (!isSaneReservoirGiVector(record.primaryPosition, 100000.0) ||
        !isSaneReservoirGiVector(record.candidatePosition, 100000.0) ||
        !isSaneReservoirGiVector(record.candidateNormal, 1.1) ||
        !isSaneReservoirGiVector(record.contribution, 64.0) ||
        !isSaneReservoirGiVector(record.secondaryDirectSun, 64.0)) {
        return false;
    }
    if (!isFinitePositive(record.selectedWeight) ||
        !isFinitePositive(record.weightSum) ||
        pathTracerLuminance(record.contribution) <= 0.0) {
        return false;
    }

    float previousDepth = length(record.primaryPosition - ubo.cameraPos.xyz);
    if (abs(previousDepth - currentPrimaryDepth) > 0.1) {
        return false;
    }
    float3 previousPrimaryNormal = normalize(record.candidateNormal);
    float3 currentNormal = normalize(currentPrimaryNormal);
    if (!isSaneReservoirGiVector(previousPrimaryNormal, 1.1) ||
        !isSaneReservoirGiVector(currentNormal, 1.1) ||
        dot(previousPrimaryNormal, currentNormal) < 0.8) {
        return false;
    }
    return true;
}

// Spatial reuse reads neighboring pixels from binding 13, the previous frame slot's
// reservoir buffer. That avoids read-after-write hazards with ptReservoirGiCurrent.
void combineSpatialReservoirGi(uint2 pixel,
                               uint2 launchSize,
                               float3 currentPrimaryNormal,
                               float currentPrimaryDepth,
                               int neighborCount,
                               inout float3 reservoirTotal,
                               inout float3 reservoirSecondarySun,
                               inout float3 selectedPosition,
                               inout float selectedWeight,
                               inout float reservoirWeightSum,
                               inout bool selectedSurfaceHit,
                               inout bool selectedSunVisible,
                               inout bool acceptedReservoir,
                               inout uint rngState)
{
    static const int2 offsets[8] = {
        int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1),
        int2(1, 1), int2(-1, 1), int2(1, -1), int2(-1, -1)
    };
    int count = clamp(neighborCount, 1, 8);
    for (int i = 0; i < count; ++i) {
        int2 neighbor = int2(int(pixel.x), int(pixel.y)) + offsets[i];
        if (neighbor.x < 0 || neighbor.y < 0 ||
            neighbor.x >= int(launchSize.x) || neighbor.y >= int(launchSize.y)) {
            ptAnalysisCounters.InterlockedAdd(reservoirGiSpatialRejectedOffset, 1u);
            continue;
        }

        ReservoirGiRecord spatialRecord;
        if (!loadTemporalReservoirGi(uint2(uint(neighbor.x), uint(neighbor.y)), launchSize,
                                     currentPrimaryNormal, currentPrimaryDepth,
                                     spatialRecord)) {
            ptAnalysisCounters.InterlockedAdd(reservoirGiSpatialRejectedOffset, 1u);
            continue;
        }

        float spatialWeight = max(spatialRecord.selectedWeight,
                                  pathTracerLuminance(spatialRecord.contribution));
        if (!isFinitePositive(spatialWeight)) {
            ptAnalysisCounters.InterlockedAdd(reservoirGiSpatialRejectedOffset, 1u);
            continue;
        }

        ptAnalysisCounters.InterlockedAdd(reservoirGiSpatialAcceptedOffset, 1u);
        if (!acceptedReservoir) {
            reservoirTotal = sanitizeReservoirGiContribution(spatialRecord.contribution);
            reservoirSecondarySun = sanitizeReservoirGiContribution(spatialRecord.secondaryDirectSun);
            selectedPosition = spatialRecord.candidatePosition;
            selectedWeight = spatialWeight;
            reservoirWeightSum = max(spatialRecord.weightSum, spatialWeight);
            selectedSurfaceHit = (spatialRecord.flags & 2u) != 0u;
            selectedSunVisible = (spatialRecord.flags & 4u) != 0u;
            acceptedReservoir = selectedWeight > 0.0 &&
                                isFinitePositive(pathTracerLuminance(reservoirTotal));
            continue;
        }

        float combinedWeightSum = max(reservoirWeightSum + spatialWeight, 0.000001);
        if (randomFloat(rngState) < spatialWeight / combinedWeightSum) {
            reservoirTotal = sanitizeReservoirGiContribution(spatialRecord.contribution);
            reservoirSecondarySun = sanitizeReservoirGiContribution(spatialRecord.secondaryDirectSun);
            selectedPosition = spatialRecord.candidatePosition;
            selectedWeight = spatialWeight;
            selectedSurfaceHit = (spatialRecord.flags & 2u) != 0u;
            selectedSunVisible = (spatialRecord.flags & 4u) != 0u;
        }
        reservoirWeightSum = combinedWeightSum;
    }
}

uint sunVisibleCandidateRecordOffset(uint slot)
{
    return SUN_VISIBLE_CANDIDATE_CACHE_HEADER_SIZE +
           slot * SUN_VISIBLE_CANDIDATE_RECORD_SIZE;
}

uint sunVisibleCandidateBankInsertCountOffset(uint bank)
{
    return min(bank, SUN_VISIBLE_CANDIDATE_CACHE_BANK_COUNT - 1u) * 4u;
}

uint cacheWriteBank(uint currentFrame)
{
    return (currentFrame / SUN_VISIBLE_CANDIDATE_CACHE_FRAME_SLOT_COUNT) %
           SUN_VISIBLE_CANDIDATE_CACHE_BANK_COUNT;
}

uint cacheReadBank(uint currentFrame)
{
    uint frameSlotGeneration = currentFrame / SUN_VISIBLE_CANDIDATE_CACHE_FRAME_SLOT_COUNT;
    return (frameSlotGeneration + SUN_VISIBLE_CANDIDATE_CACHE_BANK_COUNT - 1u) %
           SUN_VISIBLE_CANDIDATE_CACHE_BANK_COUNT;
}

uint sunVisibleCandidateBankSlot(uint bank, uint bankSlot)
{
    return min(bank, SUN_VISIBLE_CANDIDATE_CACHE_BANK_COUNT - 1u) *
               SUN_VISIBLE_CANDIDATE_CACHE_BANK_CAPACITY +
           (bankSlot % SUN_VISIBLE_CANDIDATE_CACHE_BANK_CAPACITY);
}

int3 spatialCacheCell(float3 position)
{
    return int3(floor(position / CACHED_REUSE_SPATIAL_CELL_SIZE));
}

uint hashSpatialCacheCell(int3 cell)
{
    uint3 ucell = uint3(cell);
    uint hash = ucell.x * 73856093u;
    hash ^= ucell.y * 19349663u;
    hash ^= ucell.z * 83492791u;
    hash ^= hash >> 13;
    hash *= 1274126177u;
    return hash;
}

uint sunVisibleCandidateSpatialBankSlot(uint bank, int3 cell, uint bucketSlot)
{
    uint bucketCount = max(SUN_VISIBLE_CANDIDATE_CACHE_BANK_CAPACITY / CACHED_REUSE_SPATIAL_BUCKET_SIZE, 1u);
    uint bucket = hashSpatialCacheCell(cell) % bucketCount;
    uint bankSlot = bucket * CACHED_REUSE_SPATIAL_BUCKET_SIZE +
                    (bucketSlot % CACHED_REUSE_SPATIAL_BUCKET_SIZE);
    return sunVisibleCandidateBankSlot(bank, bankSlot);
}

uint cacheRecordFrameId(uint offset) { return ptSunVisibleCandidateCache.Load(offset + 72); }
uint cacheRecordGeneration(uint offset) { return ptSunVisibleCandidateCache.Load(offset + 76); }
uint cacheRecordSourcePdfBits(uint offset) { return ptSunVisibleCandidateCache.Load(offset + 80); }
float cacheRecordSourcePdf(uint offset) { return asfloat(ptSunVisibleCandidateCache.Load(offset + 80)); }
uint cacheRecordSlotState(uint offset) { return ptSunVisibleCandidateCache.Load(offset + 84); }

uint sunVisibleConnectionRecordOffset(uint slot)
{
    return SUN_VISIBLE_CONNECTION_CACHE_HEADER_SIZE +
           slot * SUN_VISIBLE_CONNECTION_RECORD_SIZE;
}

uint sunVisibleConnectionBankInsertCountOffset(uint bank)
{
    return min(bank, SUN_VISIBLE_CONNECTION_CACHE_BANK_COUNT - 1u) * 4u;
}

uint connectionCacheWriteBank(uint currentFrame)
{
    return (currentFrame / SUN_VISIBLE_CONNECTION_CACHE_FRAME_SLOT_COUNT) %
           SUN_VISIBLE_CONNECTION_CACHE_BANK_COUNT;
}

uint connectionCacheReadBank(uint currentFrame)
{
    uint frameSlotGeneration = currentFrame / SUN_VISIBLE_CONNECTION_CACHE_FRAME_SLOT_COUNT;
    return (frameSlotGeneration + SUN_VISIBLE_CONNECTION_CACHE_BANK_COUNT - 1u) %
           SUN_VISIBLE_CONNECTION_CACHE_BANK_COUNT;
}

uint sunVisibleConnectionBankSlot(uint bank, uint bankSlot)
{
    return min(bank, SUN_VISIBLE_CONNECTION_CACHE_BANK_COUNT - 1u) *
               SUN_VISIBLE_CONNECTION_CACHE_BANK_CAPACITY +
           (bankSlot % SUN_VISIBLE_CONNECTION_CACHE_BANK_CAPACITY);
}

uint connectionRecordFrameId(uint offset) { return ptSunVisibleConnectionCache.Load(offset + 104); }
uint connectionRecordGeneration(uint offset) { return ptSunVisibleConnectionCache.Load(offset + 108); }
uint connectionRecordSlotState(uint offset) { return ptSunVisibleConnectionCache.Load(offset + 112); }

bool isFinitePositive(float value)
{
    return value > 0.0 && abs(value) < 3.402823e+38;
}

void storeFloat3ToCache(uint offset, float3 value)
{
    ptSunVisibleCandidateCache.Store(offset + 0, asuint(value.x));
    ptSunVisibleCandidateCache.Store(offset + 4, asuint(value.y));
    ptSunVisibleCandidateCache.Store(offset + 8, asuint(value.z));
}

float3 loadFloat3FromCache(uint offset)
{
    return float3(asfloat(ptSunVisibleCandidateCache.Load(offset + 0)),
                  asfloat(ptSunVisibleCandidateCache.Load(offset + 4)),
                  asfloat(ptSunVisibleCandidateCache.Load(offset + 8)));
}

void storeFloat3ToConnectionCache(uint offset, float3 value)
{
    ptSunVisibleConnectionCache.Store(offset + 0, asuint(value.x));
    ptSunVisibleConnectionCache.Store(offset + 4, asuint(value.y));
    ptSunVisibleConnectionCache.Store(offset + 8, asuint(value.z));
}

float3 loadFloat3FromConnectionCache(uint offset)
{
    return float3(asfloat(ptSunVisibleConnectionCache.Load(offset + 0)),
                  asfloat(ptSunVisibleConnectionCache.Load(offset + 4)),
                  asfloat(ptSunVisibleConnectionCache.Load(offset + 8)));
}

void storeSunVisibleCandidateMaterial(uint offset, RayPayload payload)
{
    storeFloat3ToCache(offset + 32, payload.albedo);
    storeFloat3ToCache(offset + 48, payload.F0);
    ptSunVisibleCandidateCache.Store(offset + 60, asuint(payload.roughness));
    ptSunVisibleCandidateCache.Store(offset + 64, asuint(payload.metallic));
}

void invalidateSunVisibleCacheSlotForWrite(uint offset, uint currentFrame)
{
    ptSunVisibleCandidateCache.Store(offset + 84, SUN_VISIBLE_CACHE_SLOT_WRITING);
    ptSunVisibleCandidateCache.Store(offset + 72, currentFrame);
    ptSunVisibleCandidateCache.Store(offset + 76, 0u);
}

void invalidateSunVisibleConnectionSlotForWrite(uint offset, uint currentFrame)
{
    ptSunVisibleConnectionCache.Store(offset + 112, SUN_VISIBLE_CACHE_SLOT_WRITING);
    ptSunVisibleConnectionCache.Store(offset + 104, currentFrame);
    ptSunVisibleConnectionCache.Store(offset + 108, 0u);
}

RayPayload makeCachedSunVisiblePayload(CachedSunVisibleCandidate candidate)
{
    RayPayload payload;
    payload.albedo = candidate.albedo;
    payload.emission = float3(0.0, 0.0, 0.0);
    payload.worldNormal = candidate.normal;
    payload.hitPos = candidate.position;
    payload.roughness = candidate.roughness;
    payload.metallic = candidate.metallic;
    payload.F0 = candidate.F0;
    payload.ao = 1.0;
    payload.hitT = 0.0;
    payload.instanceID = 0;
    return payload;
}

bool recordSunVisibleCandidate(RayPayload secondaryPayload,
                               float3 referenceContribution,
                               float sourcePdf,
                               uint currentCacheGeneration)
{
    float luma = pathTracerLuminance(referenceContribution);
    if (luma <= 0.000001) {
        return false;
    }

    uint insertIndex = 0;
    uint writeBank = cacheWriteBank(ubo.frameCount);
    uint insertCountOffset = sunVisibleCandidateBankInsertCountOffset(writeBank);
    ptSunVisibleCandidateCache.InterlockedAdd(insertCountOffset, 1u, insertIndex);
    uint slot = sunVisibleCandidateSpatialBankSlot(writeBank,
                                                   spatialCacheCell(secondaryPayload.hitPos),
                                                   insertIndex);
    uint offset = sunVisibleCandidateRecordOffset(slot);

    uint previousState = 0u;
    ptSunVisibleCandidateCache.InterlockedCompareExchange(offset + 84,
                                                          SUN_VISIBLE_CACHE_SLOT_READY,
                                                          SUN_VISIBLE_CACHE_SLOT_WRITING,
                                                          previousState);
    if (previousState != SUN_VISIBLE_CACHE_SLOT_READY) {
        ptSunVisibleCandidateCache.InterlockedCompareExchange(offset + 84,
                                                              SUN_VISIBLE_CACHE_SLOT_EMPTY,
                                                              SUN_VISIBLE_CACHE_SLOT_WRITING,
                                                              previousState);
        if (previousState != SUN_VISIBLE_CACHE_SLOT_EMPTY) {
            return false;
        }
    }

    invalidateSunVisibleCacheSlotForWrite(offset, ubo.frameCount);
    storeFloat3ToCache(offset + 0, secondaryPayload.hitPos);
    storeFloat3ToCache(offset + 16, normalize(secondaryPayload.worldNormal));
    storeSunVisibleCandidateMaterial(offset, secondaryPayload);
    ptSunVisibleCandidateCache.Store(offset + 68, asuint(luma));
    ptSunVisibleCandidateCache.Store(offset + 76, currentCacheGeneration);
    ptSunVisibleCandidateCache.Store(offset + 80, asuint(max(sourcePdf, 0.000001)));
    ptSunVisibleCandidateCache.Store(offset + 72, ubo.frameCount);
    ptSunVisibleCandidateCache.Store(offset + 84, SUN_VISIBLE_CACHE_SLOT_READY);
    ptAnalysisCounters.InterlockedAdd(52, 1u);
    return true;
}

bool isSaneCachedSunVisibleCandidate(CachedSunVisibleCandidate candidate)
{
    float normalLenSq = dot(candidate.normal, candidate.normal);
    bool positionOk = all(abs(candidate.position) < float3(100000.0, 100000.0, 100000.0));
    bool normalOk = normalLenSq > 0.25 && normalLenSq < 1.25 &&
                    all(abs(candidate.normal) <= float3(1.1, 1.1, 1.1));
    bool albedoOk = all(candidate.albedo >= float3(0.0, 0.0, 0.0)) &&
                    all(candidate.albedo <= float3(8.0, 8.0, 8.0));
    bool f0Ok = all(candidate.F0 >= float3(0.0, 0.0, 0.0)) &&
                all(candidate.F0 <= float3(8.0, 8.0, 8.0));
    bool materialOk = candidate.roughness >= 0.0 && candidate.roughness <= 1.1 &&
                      candidate.metallic >= 0.0 && candidate.metallic <= 1.1;
    return positionOk && normalOk && albedoOk && f0Ok && materialOk;
}

bool recordSunVisibleConnectionCandidate(float3 hitPos,
                                         float3 N,
                                         CachedSunVisibleCandidate candidate,
                                         float sourcePdf,
                                         uint currentCacheGeneration)
{
    if (!isFinitePositive(sourcePdf) || !isSaneCachedSunVisibleCandidate(candidate)) {
        return false;
    }

    uint insertIndex = 0;
    uint writeBank = connectionCacheWriteBank(ubo.frameCount);
    uint insertCountOffset = sunVisibleConnectionBankInsertCountOffset(writeBank);
    ptSunVisibleConnectionCache.InterlockedAdd(insertCountOffset, 1u, insertIndex);
    uint slot = sunVisibleConnectionBankSlot(writeBank, insertIndex);
    uint offset = sunVisibleConnectionRecordOffset(slot);

    uint previousState = 0u;
    ptSunVisibleConnectionCache.InterlockedCompareExchange(offset + 112,
                                                           SUN_VISIBLE_CACHE_SLOT_READY,
                                                           SUN_VISIBLE_CACHE_SLOT_WRITING,
                                                           previousState);
    if (previousState != SUN_VISIBLE_CACHE_SLOT_READY) {
        ptSunVisibleConnectionCache.InterlockedCompareExchange(offset + 112,
                                                               SUN_VISIBLE_CACHE_SLOT_EMPTY,
                                                               SUN_VISIBLE_CACHE_SLOT_WRITING,
                                                               previousState);
        if (previousState != SUN_VISIBLE_CACHE_SLOT_EMPTY) {
            return false;
        }
    }

    invalidateSunVisibleConnectionSlotForWrite(offset, ubo.frameCount);
    storeFloat3ToConnectionCache(offset + 0, hitPos);
    storeFloat3ToConnectionCache(offset + 16, normalize(N));
    storeFloat3ToConnectionCache(offset + 32, candidate.position);
    storeFloat3ToConnectionCache(offset + 48, normalize(candidate.normal));
    storeFloat3ToConnectionCache(offset + 64, candidate.albedo);
    storeFloat3ToConnectionCache(offset + 80, candidate.F0);
    ptSunVisibleConnectionCache.Store(offset + 92, asuint(candidate.roughness));
    ptSunVisibleConnectionCache.Store(offset + 96, asuint(candidate.metallic));
    ptSunVisibleConnectionCache.Store(offset + 100, asuint(max(sourcePdf, 0.000001)));
    ptSunVisibleConnectionCache.Store(offset + 108, currentCacheGeneration);
    ptSunVisibleConnectionCache.Store(offset + 104, ubo.frameCount);
    ptSunVisibleConnectionCache.Store(offset + 112, SUN_VISIBLE_CACHE_SLOT_READY);
    return true;
}

bool loadSunVisibleCandidate(uint slot,
                             uint currentFrame,
                             uint currentCacheGeneration,
                             uint maxRecordAgeFrames,
                             out CachedSunVisibleCandidate candidate,
                             out float sourcePdf)
{
    uint offset = sunVisibleCandidateRecordOffset(slot);
    candidate.position = float3(0.0, 0.0, 0.0);
    candidate.normal = float3(0.0, 1.0, 0.0);
    candidate.albedo = float3(0.0, 0.0, 0.0);
    candidate.F0 = float3(0.0, 0.0, 0.0);
    candidate.roughness = 1.0;
    candidate.metallic = 0.0;
    uint initialState = cacheRecordSlotState(offset);
    if (initialState != SUN_VISIBLE_CACHE_SLOT_READY) {
        sourcePdf = 0.0;
        ptAnalysisCounters.InterlockedAdd(76, 1u);
        return false;
    }

    uint frameId = cacheRecordFrameId(offset);
    uint generation = cacheRecordGeneration(offset);
    uint sourcePdfBits = cacheRecordSourcePdfBits(offset);
    sourcePdf = asfloat(sourcePdfBits);
    bool ageOk = frameId < currentFrame && (currentFrame - frameId) <= maxRecordAgeFrames;
    bool generationOk = generation == currentCacheGeneration;
    if (!ageOk || !generationOk || !isFinitePositive(sourcePdf)) {
        ptAnalysisCounters.InterlockedAdd(76, 1u);
        return false;
    }

    candidate.position = loadFloat3FromCache(offset + 0);
    candidate.normal = loadFloat3FromCache(offset + 16);
    candidate.albedo = loadFloat3FromCache(offset + 32);
    candidate.F0 = loadFloat3FromCache(offset + 48);
    candidate.roughness = asfloat(ptSunVisibleCandidateCache.Load(offset + 60));
    candidate.metallic = asfloat(ptSunVisibleCandidateCache.Load(offset + 64));
    float luma = asfloat(ptSunVisibleCandidateCache.Load(offset + 68));
    uint finalState = cacheRecordSlotState(offset);
    uint finalFrameId = cacheRecordFrameId(offset);
    uint finalGeneration = cacheRecordGeneration(offset);
    uint finalSourcePdfBits = cacheRecordSourcePdfBits(offset);
    if (finalState != initialState ||
        finalFrameId != frameId ||
        finalGeneration != generation ||
        finalSourcePdfBits != sourcePdfBits) {
        ptAnalysisCounters.InterlockedAdd(76, 1u);
        return false;
    }

    return luma > 0.000001 && frameId < currentFrame && isSaneCachedSunVisibleCandidate(candidate);
}

bool loadSunVisibleConnection(uint slot,
                              uint currentFrame,
                              uint currentCacheGeneration,
                              uint maxRecordAgeFrames,
                              out float3 primaryPosition,
                              out float3 primaryNormal,
                              out CachedSunVisibleCandidate candidate,
                              out float sourcePdf)
{
    uint offset = sunVisibleConnectionRecordOffset(slot);
    primaryPosition = float3(0.0, 0.0, 0.0);
    primaryNormal = float3(0.0, 1.0, 0.0);
    candidate.position = float3(0.0, 0.0, 0.0);
    candidate.normal = float3(0.0, 1.0, 0.0);
    candidate.albedo = float3(0.0, 0.0, 0.0);
    candidate.F0 = float3(0.0, 0.0, 0.0);
    candidate.roughness = 1.0;
    candidate.metallic = 0.0;
    sourcePdf = 0.0;

    uint initialState = connectionRecordSlotState(offset);
    if (initialState != SUN_VISIBLE_CACHE_SLOT_READY) {
        return false;
    }

    uint frameId = connectionRecordFrameId(offset);
    uint generation = connectionRecordGeneration(offset);
    uint sourcePdfBits = ptSunVisibleConnectionCache.Load(offset + 100);
    sourcePdf = asfloat(sourcePdfBits);
    bool ageOk =
        currentFrame >= SUN_VISIBLE_CONNECTION_CACHE_FRAME_SLOT_COUNT &&
        frameId + SUN_VISIBLE_CONNECTION_CACHE_FRAME_SLOT_COUNT == currentFrame;
    bool generationOk = generation == currentCacheGeneration;
    if (!ageOk || !generationOk || !isFinitePositive(sourcePdf)) {
        return false;
    }

    primaryPosition = loadFloat3FromConnectionCache(offset + 0);
    primaryNormal = loadFloat3FromConnectionCache(offset + 16);
    candidate.position = loadFloat3FromConnectionCache(offset + 32);
    candidate.normal = loadFloat3FromConnectionCache(offset + 48);
    candidate.albedo = loadFloat3FromConnectionCache(offset + 64);
    candidate.F0 = loadFloat3FromConnectionCache(offset + 80);
    candidate.roughness = asfloat(ptSunVisibleConnectionCache.Load(offset + 92));
    candidate.metallic = asfloat(ptSunVisibleConnectionCache.Load(offset + 96));

    uint finalState = connectionRecordSlotState(offset);
    uint finalFrameId = connectionRecordFrameId(offset);
    uint finalGeneration = connectionRecordGeneration(offset);
    uint finalSourcePdfBits = ptSunVisibleConnectionCache.Load(offset + 100);
    if (finalState != initialState ||
        finalFrameId != frameId ||
        finalGeneration != generation ||
        finalSourcePdfBits != sourcePdfBits) {
        return false;
    }

    float primaryNormalLenSq = dot(primaryNormal, primaryNormal);
    bool primaryOk = all(abs(primaryPosition) < float3(100000.0, 100000.0, 100000.0)) &&
                     primaryNormalLenSq > 0.25 && primaryNormalLenSq < 1.25;
    return primaryOk && isSaneCachedSunVisibleCandidate(candidate);
}

float3 firstHitSunBounceGuideAxis(float3 N)
{
    float3 axis = normalize(ubo.lightDir.xyz);
    float axisNdot = dot(axis, N);
    if (axisNdot < 0.10) {
        float3 tangentAxis = axis - N * axisNdot;
        float tangentLenSq = dot(tangentAxis, tangentAxis);
        if (tangentLenSq > 0.0001) {
            axis = normalize(tangentAxis * rsqrt(tangentLenSq) + N * 0.35);
        } else {
            axis = N;
        }
    }
    return normalize(axis);
}

FirstHitProbeSample sampleFirstHitProbeDirection(float2 xi, float selector, float3 N, int firstHitProbeSamplingMode)
{
    FirstHitProbeSample sample;
    sample.direction = cosineSampleHemisphere(xi, N);
    sample.pdf = max(dot(N, sample.direction), 0.0) / PI;

    if (firstHitProbeSamplingMode == FIRST_HIT_PROBE_SAMPLING_SUN_BOUNCE_GUIDED) {
        float3 guideAxis = firstHitSunBounceGuideAxis(N);
        if (selector < 0.5) {
            sample.direction = cosineSampleHemisphere(xi, guideAxis);
        }

        float cosinePdf = max(dot(N, sample.direction), 0.0) / PI;
        float guidedPdf = max(dot(guideAxis, sample.direction), 0.0) / PI;
        sample.pdf = 0.5 * cosinePdf + 0.5 * guidedPdf;
    }

    return sample;
}

float3 evaluateDirectSunContribution(float3 hitPos, float3 N, float3 V, float3 throughput, RayPayload payload)
{
    float3 Ldir = normalize(-ubo.lightDir.xyz);
    float NdotL = max(dot(N, Ldir), 0.0);
    if (NdotL <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }

    RayPayload shadowPayload = makeVisibilityPayload();
    RayDesc shadowRay;
    shadowRay.Origin    = hitPos + N * 0.002;
    shadowRay.Direction = Ldir;
    shadowRay.TMin      = 0.001;
    shadowRay.TMax      = 10000.0;
    TraceRay(tlas,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
        0xFF, 0, 0, 0, shadowRay, shadowPayload);

    float shadowVisibility = (shadowPayload.hitT < 0.0) ? 1.0 : 0.0;
    if (shadowVisibility <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }

    float3 H = normalize(Ldir + V);
    float NdotV = max(dot(N, V), 0.0001);
    float VdotH = max(dot(V, H), 0.0001);
    float3 Fs = fresnelSchlick(VdotH, payload.F0);
    float D = distributionGGX(N, H, payload.roughness);
    float G = geometrySmith(N, V, Ldir, payload.roughness);
    float3 kD = (float3(1.0, 1.0, 1.0) - Fs) * (1.0 - payload.metallic);
    float3 diffuse = kD * payload.albedo / PI;
    float3 specular = (D * Fs * G) / max(4.0 * NdotV * NdotL, 0.0001);
    return throughput * (diffuse + specular) * NdotL * SUN_RADIANCE * shadowVisibility;
}

FirstHitDiffuseBounceResult sampleFirstHitDiffuseBounce(float3 hitPos, float3 N, float3 V, RayPayload payload,
                                                        int firstHitDiffuseSampleCount,
                                                        int firstHitProbeSamplingMode,
                                                        bool sunVisibleCandidateCacheEnabled,
                                                        uint currentCacheGeneration,
                                                        inout uint rngState)
{
    FirstHitDiffuseBounceResult result;
    result.totalContribution = float3(0.0, 0.0, 0.0);
    result.secondaryDirectSunContribution = float3(0.0, 0.0, 0.0);

    int extraSampleCount = clamp(firstHitDiffuseSampleCount - 1, 0, 7);
    if (extraSampleCount <= 0) {
        return result;
    }

    float3 F = fresnelSchlick(max(dot(N, V), 0.0), payload.F0);
    float3 kD = (float3(1.0, 1.0, 1.0) - F) * (1.0 - payload.metallic);
    float3 diffuseBsdf = kD * payload.albedo / PI;

    for (int i = 0; i < extraSampleCount; ++i) {
        float2 xi = float2(randomFloat(rngState), randomFloat(rngState));
        FirstHitProbeSample bounceSample =
            sampleFirstHitProbeDirection(xi, randomFloat(rngState), N, firstHitProbeSamplingMode);
        float3 bounceDir = bounceSample.direction;
        float NdotBounce = max(dot(N, bounceDir), 0.0);
        if (NdotBounce <= 0.0 || bounceSample.pdf <= 0.0) {
            recordFirstHitProbeStats(false, false, float3(0.0, 0.0, 0.0));
            continue;
        }
        float3 firstLegThroughput = diffuseBsdf * NdotBounce / max(bounceSample.pdf, 0.0001);

        RayPayload bouncePayload;
        bouncePayload.hitT = -1.0;
        RayDesc bounceRay;
        bounceRay.Origin    = hitPos + N * 0.002;
        bounceRay.Direction = bounceDir;
        bounceRay.TMin      = 0.001;
        bounceRay.TMax      = 10000.0;
        TraceRay(tlas, RAY_FLAG_NONE, 0xFF, 0, 0, 0, bounceRay, bouncePayload);

        if (bouncePayload.hitT < 0.0) {
            float3 skyContribution = firstLegThroughput * bouncePayload.emission;
            result.totalContribution += skyContribution;
            recordFirstHitProbeStats(false, false, skyContribution);
        } else {
            float3 secondaryV = -bounceDir;
            float3 emissiveContribution = firstLegThroughput * bouncePayload.emission;
            float3 secondaryDirect = evaluateDirectSunContribution(bouncePayload.hitPos,
                                                                   bouncePayload.worldNormal,
                                                                   secondaryV,
                                                                   firstLegThroughput,
                                                                   bouncePayload);
            bool secondarySunVisible = pathTracerLuminance(secondaryDirect) > 0.000001;
            if (sunVisibleCandidateCacheEnabled && secondarySunVisible) {
                recordSunVisibleCandidate(bouncePayload, secondaryDirect, bounceSample.pdf, currentCacheGeneration);
            }
            recordFirstHitProbeStats(true, secondarySunVisible, emissiveContribution + secondaryDirect);
            result.secondaryDirectSunContribution += secondaryDirect;
            result.totalContribution += emissiveContribution + secondaryDirect;
        }
    }

    result.totalContribution /= float(firstHitDiffuseSampleCount);
    result.secondaryDirectSunContribution /= float(firstHitDiffuseSampleCount);
    return result;
}

void refreshSunVisibleCandidateCache(float3 hitPos,
                                     float3 N,
                                     float3 V,
                                     RayPayload payload,
                                     int refreshCandidateCount,
                                     uint currentCacheGeneration,
                                     inout uint rngState)
{
    int candidateCount = clamp(refreshCandidateCount, 0, 4);
    if (candidateCount <= 0) {
        return;
    }

    float3 F = fresnelSchlick(max(dot(N, V), 0.0), payload.F0);
    float3 kD = (float3(1.0, 1.0, 1.0) - F) * (1.0 - payload.metallic);
    float3 diffuseBsdf = kD * payload.albedo / PI;

    for (int c = 0; c < candidateCount; ++c) {
        ptAnalysisCounters.InterlockedAdd(68, 1u);
        float2 xi = float2(randomFloat(rngState), randomFloat(rngState));
        float3 bounceDir = cosineSampleHemisphere(xi, N);
        float NdotBounce = max(dot(N, bounceDir), 0.0);
        float pdf = max(NdotBounce / PI, 0.0001);
        float3 firstLegThroughput = diffuseBsdf * NdotBounce / pdf;

        RayPayload bouncePayload;
        bouncePayload.hitT = -1.0;
        RayDesc bounceRay;
        bounceRay.Origin = hitPos + N * 0.002;
        bounceRay.Direction = bounceDir;
        bounceRay.TMin = 0.001;
        bounceRay.TMax = 10000.0;
        TraceRay(tlas, RAY_FLAG_NONE, 0xFF, 0, 0, 0, bounceRay, bouncePayload);

        if (bouncePayload.hitT >= 0.0) {
            float3 secondaryV = -bounceDir;
            float3 secondaryDirect = evaluateDirectSunContribution(bouncePayload.hitPos,
                                                                   bouncePayload.worldNormal,
                                                                   secondaryV,
                                                                   firstLegThroughput,
                                                                   bouncePayload);
            if (pathTracerLuminance(secondaryDirect) > 0.000001) {
                if (recordSunVisibleCandidate(bouncePayload, secondaryDirect, pdf, currentCacheGeneration)) {
                    ptAnalysisCounters.InterlockedAdd(72, 1u);
                }
            }
        }
    }
}

void refreshDiagnosticTargetSunVisibleCandidateCache(float3 hitPos,
                                                     float3 N,
                                                     float3 V,
                                                     RayPayload payload,
                                                     uint currentCacheGeneration,
                                                     inout uint rngState)
{
    refreshSunVisibleCandidateCache(hitPos, N, V, payload, 1, currentCacheGeneration, rngState);
}

FirstHitDiffuseBounceResult sampleFirstHitCandidateSunBounce(float3 hitPos, float3 N, float3 V, RayPayload payload,
                                                             int firstHitDiffuseSampleCount,
                                                             int candidateCount,
                                                             bool sunVisibleCandidateCacheEnabled,
                                                             uint currentCacheGeneration,
                                                             inout uint rngState)
{
    FirstHitDiffuseBounceResult result;
    result.totalContribution = float3(0.0, 0.0, 0.0);
    result.secondaryDirectSunContribution = float3(0.0, 0.0, 0.0);

    int extraSampleCount = clamp(firstHitDiffuseSampleCount - 1, 0, 7);
    if (extraSampleCount <= 0) {
        return result;
    }

    candidateCount = clamp(candidateCount, 2, 16);

    float3 F = fresnelSchlick(max(dot(N, V), 0.0), payload.F0);
    float3 kD = (float3(1.0, 1.0, 1.0) - F) * (1.0 - payload.metallic);
    float3 diffuseBsdf = kD * payload.albedo / PI;

    for (int probe = 0; probe < extraSampleCount; ++probe) {
        float3 bestTotal = float3(0.0, 0.0, 0.0);
        float3 bestSecondarySun = float3(0.0, 0.0, 0.0);
        float bestLuma = -1.0;
        bool bestSurfaceHit = false;
        bool bestSunVisible = false;

        for (int c = 0; c < candidateCount; ++c) {
            float2 xi = float2(randomFloat(rngState), randomFloat(rngState));
            float3 bounceDir = cosineSampleHemisphere(xi, N);
            float NdotBounce = max(dot(N, bounceDir), 0.0);
            float pdf = max(NdotBounce / PI, 0.0001);
            float3 firstLegThroughput = diffuseBsdf * NdotBounce / pdf;

            RayPayload bouncePayload;
            bouncePayload.hitT = -1.0;
            RayDesc bounceRay;
            bounceRay.Origin = hitPos + N * 0.002;
            bounceRay.Direction = bounceDir;
            bounceRay.TMin = 0.001;
            bounceRay.TMax = 10000.0;
            TraceRay(tlas, RAY_FLAG_NONE, 0xFF, 0, 0, 0, bounceRay, bouncePayload);

            float3 candidateTotal = float3(0.0, 0.0, 0.0);
            float3 candidateSecondarySun = float3(0.0, 0.0, 0.0);
            bool surfaceHit = bouncePayload.hitT >= 0.0;
            bool sunVisible = false;

            if (surfaceHit) {
                float3 secondaryV = -bounceDir;
                float3 emissiveContribution = firstLegThroughput * bouncePayload.emission;
                candidateSecondarySun = evaluateDirectSunContribution(bouncePayload.hitPos,
                                                                      bouncePayload.worldNormal,
                                                                      secondaryV,
                                                                      firstLegThroughput,
                                                                      bouncePayload);
                candidateTotal = emissiveContribution + candidateSecondarySun;
                sunVisible = pathTracerLuminance(candidateSecondarySun) > 0.000001;
                if (sunVisibleCandidateCacheEnabled && sunVisible) {
                    recordSunVisibleCandidate(bouncePayload, candidateSecondarySun, pdf, currentCacheGeneration);
                }
            } else {
                candidateTotal = firstLegThroughput * bouncePayload.emission;
            }

            float candidateLuma = pathTracerLuminance(candidateSecondarySun);
            if (candidateLuma > bestLuma) {
                bestLuma = candidateLuma;
                bestTotal = candidateTotal;
                bestSecondarySun = candidateSecondarySun;
                bestSurfaceHit = surfaceHit;
                bestSunVisible = sunVisible;
            }
        }

        float3 weightedBestTotal = bestTotal / float(candidateCount);
        float3 weightedBestSecondarySun = bestSecondarySun / float(candidateCount);
        recordFirstHitProbeStats(bestSurfaceHit, bestSunVisible, weightedBestTotal);
        result.totalContribution += weightedBestTotal;
        result.secondaryDirectSunContribution += weightedBestSecondarySun;
    }

    result.totalContribution /= float(firstHitDiffuseSampleCount);
    result.secondaryDirectSunContribution /= float(firstHitDiffuseSampleCount);
    return result;
}

FirstHitDiffuseBounceResult sampleFirstHitCandidateAverageReference(float3 hitPos, float3 N, float3 V, RayPayload payload,
                                                                    int firstHitDiffuseSampleCount,
                                                                    int candidateCount,
                                                                    bool sunVisibleCandidateCacheEnabled,
                                                                    uint currentCacheGeneration,
                                                                    inout uint rngState)
{
    FirstHitDiffuseBounceResult result;
    result.totalContribution = float3(0.0, 0.0, 0.0);
    result.secondaryDirectSunContribution = float3(0.0, 0.0, 0.0);

    int extraSampleCount = clamp(firstHitDiffuseSampleCount - 1, 0, 7);
    if (extraSampleCount <= 0) {
        return result;
    }

    candidateCount = clamp(candidateCount, 2, 16);

    float3 F = fresnelSchlick(max(dot(N, V), 0.0), payload.F0);
    float3 kD = (float3(1.0, 1.0, 1.0) - F) * (1.0 - payload.metallic);
    float3 diffuseBsdf = kD * payload.albedo / PI;

    for (int probe = 0; probe < extraSampleCount; ++probe) {
        float3 averagedTotal = float3(0.0, 0.0, 0.0);
        float3 averagedSecondarySun = float3(0.0, 0.0, 0.0);
        bool anySurfaceHit = false;
        bool anySunVisible = false;

        for (int c = 0; c < candidateCount; ++c) {
            float2 xi = float2(randomFloat(rngState), randomFloat(rngState));
            float3 bounceDir = cosineSampleHemisphere(xi, N);
            float NdotBounce = max(dot(N, bounceDir), 0.0);
            float pdf = max(NdotBounce / PI, 0.0001);
            float3 firstLegThroughput = diffuseBsdf * NdotBounce / pdf;

            RayPayload bouncePayload;
            bouncePayload.hitT = -1.0;
            RayDesc bounceRay;
            bounceRay.Origin = hitPos + N * 0.002;
            bounceRay.Direction = bounceDir;
            bounceRay.TMin = 0.001;
            bounceRay.TMax = 10000.0;
            TraceRay(tlas, RAY_FLAG_NONE, 0xFF, 0, 0, 0, bounceRay, bouncePayload);

            float3 candidateTotal = float3(0.0, 0.0, 0.0);
            float3 candidateSecondarySun = float3(0.0, 0.0, 0.0);
            bool surfaceHit = bouncePayload.hitT >= 0.0;
            bool sunVisible = false;

            if (surfaceHit) {
                float3 secondaryV = -bounceDir;
                float3 emissiveContribution = firstLegThroughput * bouncePayload.emission;
                candidateSecondarySun = evaluateDirectSunContribution(bouncePayload.hitPos,
                                                                      bouncePayload.worldNormal,
                                                                      secondaryV,
                                                                      firstLegThroughput,
                                                                      bouncePayload);
                candidateTotal = emissiveContribution + candidateSecondarySun;
                sunVisible = pathTracerLuminance(candidateSecondarySun) > 0.000001;
                if (sunVisibleCandidateCacheEnabled && sunVisible) {
                    recordSunVisibleCandidate(bouncePayload, candidateSecondarySun, pdf, currentCacheGeneration);
                }
            } else {
                candidateTotal = firstLegThroughput * bouncePayload.emission;
            }

            averagedTotal += candidateTotal;
            averagedSecondarySun += candidateSecondarySun;
            anySurfaceHit = anySurfaceHit || surfaceHit;
            anySunVisible = anySunVisible || sunVisible;
        }

        averagedTotal /= float(candidateCount);
        averagedSecondarySun /= float(candidateCount);
        recordFirstHitProbeStats(anySurfaceHit, anySunVisible, averagedTotal);
        result.totalContribution += averagedTotal;
        result.secondaryDirectSunContribution += averagedSecondarySun;
    }

    result.totalContribution /= float(firstHitDiffuseSampleCount);
    result.secondaryDirectSunContribution /= float(firstHitDiffuseSampleCount);
    return result;
}

FirstHitDiffuseBounceResult sampleFirstHitCandidateRis(float3 hitPos, float3 N, float3 V, RayPayload payload,
                                                       int firstHitDiffuseSampleCount,
                                                       int candidateCount,
                                                       bool sunVisibleCandidateCacheEnabled,
                                                       uint currentCacheGeneration,
                                                       inout uint rngState)
{
    FirstHitDiffuseBounceResult result;
    result.totalContribution = float3(0.0, 0.0, 0.0);
    result.secondaryDirectSunContribution = float3(0.0, 0.0, 0.0);

    int extraSampleCount = clamp(firstHitDiffuseSampleCount - 1, 0, 7);
    if (extraSampleCount <= 0) {
        return result;
    }

    candidateCount = clamp(candidateCount, 2, 16);

    float3 F = fresnelSchlick(max(dot(N, V), 0.0), payload.F0);
    float3 kD = (float3(1.0, 1.0, 1.0) - F) * (1.0 - payload.metallic);
    float3 diffuseBsdf = kD * payload.albedo / PI;

    for (int probe = 0; probe < extraSampleCount; ++probe) {
        float3 selectedTotal = float3(0.0, 0.0, 0.0);
        float3 selectedSecondarySun = float3(0.0, 0.0, 0.0);
        float selectedWeight = 0.0;
        float risWeightSum = 0.0;
        bool selectedSurfaceHit = false;
        bool selectedSunVisible = false;

        for (int c = 0; c < candidateCount; ++c) {
            float2 xi = float2(randomFloat(rngState), randomFloat(rngState));
            float3 bounceDir = cosineSampleHemisphere(xi, N);
            float NdotBounce = max(dot(N, bounceDir), 0.0);
            float pdf = max(NdotBounce / PI, 0.0001);
            float3 firstLegThroughput = diffuseBsdf * NdotBounce / pdf;

            RayPayload bouncePayload;
            bouncePayload.hitT = -1.0;
            RayDesc bounceRay;
            bounceRay.Origin = hitPos + N * 0.002;
            bounceRay.Direction = bounceDir;
            bounceRay.TMin = 0.001;
            bounceRay.TMax = 10000.0;
            TraceRay(tlas, RAY_FLAG_NONE, 0xFF, 0, 0, 0, bounceRay, bouncePayload);

            float3 candidateTotal = float3(0.0, 0.0, 0.0);
            float3 candidateSecondarySun = float3(0.0, 0.0, 0.0);
            bool surfaceHit = bouncePayload.hitT >= 0.0;
            bool sunVisible = false;

            if (surfaceHit) {
                float3 secondaryV = -bounceDir;
                float3 emissiveContribution = firstLegThroughput * bouncePayload.emission;
                candidateSecondarySun = evaluateDirectSunContribution(bouncePayload.hitPos,
                                                                      bouncePayload.worldNormal,
                                                                      secondaryV,
                                                                      firstLegThroughput,
                                                                      bouncePayload);
                candidateTotal = emissiveContribution + candidateSecondarySun;
                sunVisible = pathTracerLuminance(candidateSecondarySun) > 0.000001;
                if (sunVisibleCandidateCacheEnabled && sunVisible) {
                    recordSunVisibleCandidate(bouncePayload, candidateSecondarySun, pdf, currentCacheGeneration);
                }
            } else {
                candidateTotal = firstLegThroughput * bouncePayload.emission;
            }

            float candidateWeight = max(pathTracerLuminance(candidateTotal), 0.000001);
            risWeightSum += candidateWeight;
            if (randomFloat(rngState) < candidateWeight / risWeightSum) {
                selectedTotal = candidateTotal;
                selectedSecondarySun = candidateSecondarySun;
                selectedWeight = candidateWeight;
                selectedSurfaceHit = surfaceHit;
                selectedSunVisible = sunVisible;
            }
        }

        float selectedCandidateProbability = selectedWeight / max(risWeightSum, 0.000001);
        float risScale = 1.0 / (float(candidateCount) * max(selectedCandidateProbability, 0.000001));
        float3 risTotal = selectedTotal * risScale;
        float3 risSecondarySun = selectedSecondarySun * risScale;
        recordFirstHitProbeStats(selectedSurfaceHit, selectedSunVisible, risTotal);
        result.totalContribution += risTotal;
        result.secondaryDirectSunContribution += risSecondarySun;
    }

    result.totalContribution /= float(firstHitDiffuseSampleCount);
    result.secondaryDirectSunContribution /= float(firstHitDiffuseSampleCount);
    return result;
}

FirstHitDiffuseBounceResult sampleFirstHitReservoirGiSingleFrame(float3 hitPos, float3 N, float3 V, RayPayload payload,
                                                                 int firstHitDiffuseSampleCount,
                                                                 int candidateCount,
                                                                 bool useCandidateRis,
                                                                 int reservoirGiMode,
                                                                 int reservoirGiSpatialNeighborCount,
                                                                 uint2 launchID,
                                                                 uint2 launchSize,
                                                                 inout uint rngState)
{
    FirstHitDiffuseBounceResult result;
    result.totalContribution = float3(0.0, 0.0, 0.0);
    result.secondaryDirectSunContribution = float3(0.0, 0.0, 0.0);

    ReservoirGiRecord record;
    record.primaryPosition = float3(0.0, 0.0, 0.0);
    record.candidatePosition = float3(0.0, 0.0, 0.0);
    record.candidateNormal = float3(0.0, 0.0, 0.0);
    record.contribution = float3(0.0, 0.0, 0.0);
    record.secondaryDirectSun = float3(0.0, 0.0, 0.0);
    record.weightSum = 0.0;
    record.selectedWeight = 0.0;
    record.frameId = ubo.frameCount;
    record.flags = 0u;

    if (!isSaneReservoirGiVector(hitPos, 100000.0) ||
        !isSaneReservoirGiVector(N, 1.1) ||
        !isSaneReservoirGiVector(V, 1.1)) {
        return result;
    }

    record.primaryPosition = hitPos;
    record.candidatePosition = hitPos;
    record.candidateNormal = N;

    int extraSampleCount = clamp(firstHitDiffuseSampleCount - 1, 0, 7);
    if (extraSampleCount <= 0) {
        updateReservoirGi(launchID, launchSize, record);
        return result;
    }

    candidateCount = clamp(candidateCount, 1, 4);

    float3 F = fresnelSchlick(max(dot(N, V), 0.0), payload.F0);
    float3 kD = (float3(1.0, 1.0, 1.0) - F) * (1.0 - payload.metallic);
    float3 diffuseBsdf = kD * payload.albedo / PI;

    float3 selectedTotal = float3(0.0, 0.0, 0.0);
    float3 selectedSecondarySun = float3(0.0, 0.0, 0.0);
    float3 selectedPosition = hitPos;
    float3 selectedNormal = N;
    float selectedWeight = 0.0;
    float reservoirWeightSum = 0.0;
    bool selectedSurfaceHit = false;
    bool selectedSunVisible = false;

    for (int c = 0; c < candidateCount; ++c) {
        float2 xi = float2(randomFloat(rngState), randomFloat(rngState));
        float3 bounceDir = cosineSampleHemisphere(xi, N);
        float NdotBounce = max(dot(N, bounceDir), 0.0);
        float pdf = NdotBounce / PI;
        if (!isFinitePositive(pdf) || !isSaneReservoirGiVector(bounceDir, 1.1)) {
            continue;
        }
        pdf = max(pdf, 0.0001);
        float3 firstLegThroughput = diffuseBsdf * NdotBounce / pdf;
        if (!isSaneReservoirGiVector(firstLegThroughput, 64.0)) {
            continue;
        }

        RayPayload bouncePayload;
        bouncePayload.hitT = -1.0;
        RayDesc bounceRay;
        bounceRay.Origin = hitPos + N * 0.002;
        bounceRay.Direction = bounceDir;
        bounceRay.TMin = 0.001;
        bounceRay.TMax = 10000.0;
        TraceRay(tlas, RAY_FLAG_NONE, 0xFF, 0, 0, 0, bounceRay, bouncePayload);

        float3 candidateTotal = float3(0.0, 0.0, 0.0);
        float3 candidateSecondarySun = float3(0.0, 0.0, 0.0);
        bool surfaceHit = bouncePayload.hitT >= 0.0;
        bool sunVisible = false;
        float3 candidatePosition = hitPos;
        float3 candidateNormal = N;

        if (surfaceHit) {
            if (!isSaneReservoirGiVector(bouncePayload.hitPos, 100000.0) ||
                !isSaneReservoirGiVector(bouncePayload.worldNormal, 1.1)) {
                continue;
            }
            float3 secondaryV = -bounceDir;
            float3 emissiveContribution = firstLegThroughput * bouncePayload.emission;
            candidateSecondarySun = evaluateDirectSunContribution(bouncePayload.hitPos,
                                                                  bouncePayload.worldNormal,
                                                                  secondaryV,
                                                                  firstLegThroughput,
                                                                  bouncePayload);
            candidateTotal = emissiveContribution + candidateSecondarySun;
            candidatePosition = bouncePayload.hitPos;
            candidateNormal = bouncePayload.worldNormal;
            sunVisible = pathTracerLuminance(candidateSecondarySun) > 0.000001;
        } else {
            candidateTotal = firstLegThroughput * bouncePayload.emission;
        }

        candidateTotal = sanitizeReservoirGiContribution(candidateTotal);
        candidateSecondarySun = sanitizeReservoirGiContribution(candidateSecondarySun);
        float candidateWeight = pathTracerLuminance(candidateTotal);
        if (!isFinitePositive(candidateWeight)) {
            continue;
        }
        ptAnalysisCounters.InterlockedAdd(reservoirGiCandidatesOffset, 1u);

        if (useCandidateRis) {
            reservoirWeightSum += candidateWeight;
            if (randomFloat(rngState) < candidateWeight / max(reservoirWeightSum, 0.000001)) {
                selectedTotal = candidateTotal;
                selectedSecondarySun = candidateSecondarySun;
                selectedPosition = candidatePosition;
                selectedNormal = candidateNormal;
                selectedWeight = candidateWeight;
                selectedSurfaceHit = surfaceHit;
                selectedSunVisible = sunVisible;
            }
        } else if (candidateWeight > selectedWeight) {
            reservoirWeightSum += candidateWeight;
            selectedTotal = candidateTotal;
            selectedSecondarySun = candidateSecondarySun;
            selectedPosition = candidatePosition;
            selectedNormal = candidateNormal;
            selectedWeight = candidateWeight;
            selectedSurfaceHit = surfaceHit;
            selectedSunVisible = sunVisible;
        } else {
            reservoirWeightSum += candidateWeight;
        }
    }

    float3 reservoirTotal = selectedTotal;
    float3 reservoirSecondarySun = selectedSecondarySun;
    if (useCandidateRis) {
        float selectedProbability = selectedWeight / max(reservoirWeightSum, 0.000001);
        float risScale = 1.0 / (float(candidateCount) * max(selectedProbability, 0.000001));
        reservoirTotal *= risScale;
        reservoirSecondarySun *= risScale;
    } else {
        reservoirTotal /= float(candidateCount);
        reservoirSecondarySun /= float(candidateCount);
    }

    reservoirTotal = sanitizeReservoirGiContribution(reservoirTotal);
    reservoirSecondarySun = sanitizeReservoirGiContribution(reservoirSecondarySun);
    bool acceptedReservoir =
        selectedWeight > 0.0 && isFinitePositive(pathTracerLuminance(reservoirTotal));

    if (reservoirGiMode >= PATH_TRACER_RESERVOIR_GI_TEMPORAL) {
        ReservoirGiRecord temporalRecord;
        if (loadTemporalReservoirGi(launchID, launchSize, N, payload.hitT, temporalRecord)) {
            ptAnalysisCounters.InterlockedAdd(reservoirGiTemporalAcceptedOffset, 1u);
            float temporalWeight = max(temporalRecord.selectedWeight,
                                       pathTracerLuminance(temporalRecord.contribution));
            if (!acceptedReservoir) {
                reservoirTotal = sanitizeReservoirGiContribution(temporalRecord.contribution);
                reservoirSecondarySun = sanitizeReservoirGiContribution(temporalRecord.secondaryDirectSun);
                selectedPosition = temporalRecord.candidatePosition;
                selectedWeight = temporalWeight;
                reservoirWeightSum = max(temporalRecord.weightSum, temporalWeight);
                selectedSurfaceHit = (temporalRecord.flags & 2u) != 0u;
                selectedSunVisible = (temporalRecord.flags & 4u) != 0u;
                acceptedReservoir = selectedWeight > 0.0 &&
                                    isFinitePositive(pathTracerLuminance(reservoirTotal));
            } else {
                float combinedWeightSum = max(reservoirWeightSum + temporalWeight, 0.000001);
                if (randomFloat(rngState) < temporalWeight / combinedWeightSum) {
                    reservoirTotal = sanitizeReservoirGiContribution(temporalRecord.contribution);
                    reservoirSecondarySun = sanitizeReservoirGiContribution(temporalRecord.secondaryDirectSun);
                    selectedPosition = temporalRecord.candidatePosition;
                    selectedWeight = temporalWeight;
                    selectedSurfaceHit = (temporalRecord.flags & 2u) != 0u;
                    selectedSunVisible = (temporalRecord.flags & 4u) != 0u;
                }
                reservoirWeightSum = combinedWeightSum;
            }
        } else {
            ptAnalysisCounters.InterlockedAdd(reservoirGiTemporalRejectedOffset, 1u);
        }
    }
    if (reservoirGiMode == PATH_TRACER_RESERVOIR_GI_TEMPORAL_SPATIAL) {
        combineSpatialReservoirGi(launchID, launchSize, N, payload.hitT,
                                  reservoirGiSpatialNeighborCount,
                                  reservoirTotal, reservoirSecondarySun,
                                  selectedPosition, selectedWeight,
                                  reservoirWeightSum, selectedSurfaceHit,
                                  selectedSunVisible, acceptedReservoir,
                                  rngState);
    }

    reservoirTotal = sanitizeReservoirGiContribution(reservoirTotal);
    reservoirSecondarySun = sanitizeReservoirGiContribution(reservoirSecondarySun);
    float rawReservoirLuma = pathTracerLuminance(reservoirTotal);
    acceptedReservoir = selectedWeight > 0.0 && isFinitePositive(rawReservoirLuma);
    float reservoirLuma = acceptedReservoir ? clamp(rawReservoirLuma, 0.0, 8.0) : 0.0;
    uint scaledReservoirLuma = uint(reservoirLuma * 64.0 + 0.5);
    if (acceptedReservoir) {
        ptAnalysisCounters.InterlockedAdd(reservoirGiAcceptedOffset, 1u);
        ptAnalysisCounters.InterlockedAdd(reservoirGiLumaScaledSumOffset, scaledReservoirLuma);
        recordFirstHitProbeStats(selectedSurfaceHit, selectedSunVisible, reservoirTotal);
    } else {
        reservoirTotal = float3(0.0, 0.0, 0.0);
        reservoirSecondarySun = float3(0.0, 0.0, 0.0);
        selectedPosition = float3(0.0, 0.0, 0.0);
        selectedSurfaceHit = false;
        selectedSunVisible = false;
        selectedWeight = 0.0;
        reservoirWeightSum = 0.0;
    }

    record.candidatePosition = selectedPosition;
    record.candidateNormal = N;
    record.contribution = reservoirTotal;
    record.secondaryDirectSun = reservoirSecondarySun;
    record.weightSum = reservoirWeightSum;
    record.selectedWeight = selectedWeight;
    record.flags = (selectedWeight > 0.0 ? 1u : 0u) |
                   (selectedSurfaceHit ? 2u : 0u) |
                   (selectedSunVisible ? 4u : 0u) |
                   (useCandidateRis ? 8u : 0u) |
                   (reservoirGiMode >= PATH_TRACER_RESERVOIR_GI_TEMPORAL ? 16u : 0u);
    updateReservoirGi(launchID, launchSize, record);

    result.totalContribution = reservoirTotal / float(firstHitDiffuseSampleCount);
    result.secondaryDirectSunContribution = reservoirSecondarySun / float(firstHitDiffuseSampleCount);
    return result;
}

float3 evaluateCachedSunVisibleContribution(float3 hitPos, float3 N, float3 V,
                                             RayPayload primaryPayload,
                                             CachedSunVisibleCandidate candidate,
                                             float cacheConnectionRadius,
                                             out bool accepted,
                                             out int CachedReuseRejectReason)
{
    accepted = false;
    CachedReuseRejectReason = CACHED_REUSE_REJECT_NONE;

    float3 toCached = candidate.position - hitPos;
    float rawDistanceSq = dot(toCached, toCached);
    float maxConnectionDistanceSq = cacheConnectionRadius * cacheConnectionRadius;
    if (!(rawDistanceSq > 0.000001) || rawDistanceSq > maxConnectionDistanceSq) {
        CachedReuseRejectReason = CACHED_REUSE_REJECT_DISTANCE;
        return float3(0.0, 0.0, 0.0);
    }

    float distanceToCached = sqrt(rawDistanceSq);
    float3 bounceDir = toCached / distanceToCached;
    if (!all(abs(bounceDir) <= float3(1.1, 1.1, 1.1))) {
        CachedReuseRejectReason = CACHED_REUSE_REJECT_GEOMETRY;
        return float3(0.0, 0.0, 0.0);
    }

    float NdotBounce = max(dot(N, bounceDir), 0.0);
    float cachedNdot = max(dot(candidate.normal, -bounceDir), 0.0);
    if (NdotBounce <= 0.0 || cachedNdot <= 0.0) {
        CachedReuseRejectReason = CACHED_REUSE_REJECT_GEOMETRY;
        return float3(0.0, 0.0, 0.0);
    }

    float visibilityRayTMin = 0.001;
    float visibilityRayTMax = distanceToCached - 0.004;
    if (visibilityRayTMax <= visibilityRayTMin) {
        CachedReuseRejectReason = CACHED_REUSE_REJECT_DISTANCE;
        return float3(0.0, 0.0, 0.0);
    }

    RayPayload visibilityPayload = makeVisibilityPayload();
    RayDesc visibilityRay;
    visibilityRay.Origin = hitPos + N * 0.002;
    visibilityRay.Direction = bounceDir;
    visibilityRay.TMin = visibilityRayTMin;
    visibilityRay.TMax = visibilityRayTMax;
    TraceRay(tlas,
             RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_FORCE_OPAQUE,
             0xFF, 0, 0, 0, visibilityRay, visibilityPayload);

    if (visibilityPayload.hitT >= 0.0) {
        CachedReuseRejectReason = CACHED_REUSE_REJECT_VISIBILITY;
        return float3(0.0, 0.0, 0.0);
    }

    accepted = true;

    float3 F = fresnelSchlick(max(dot(N, V), 0.0), primaryPayload.F0);
    float3 kD = (float3(1.0, 1.0, 1.0) - F) * (1.0 - primaryPayload.metallic);
    float3 firstLegThroughput = kD * primaryPayload.albedo;
    RayPayload cachedPayload = makeCachedSunVisiblePayload(candidate);
    float3 secondaryV = -bounceDir;
    return evaluateDirectSunContribution(candidate.position,
                                         candidate.normal,
                                         secondaryV,
                                         firstLegThroughput,
                                         cachedPayload);
}

float computeCachedReuseMisWeight(float bsdfPdf,
                                  float cacheSourcePdf,
                                  int cacheWeightingMode)
{
    if (cacheWeightingMode == CACHE_WEIGHTING_CALIBRATED) {
        return 1.0;
    }

    float cachePdf = max(cacheSourcePdf, 0.000001);
    float competingPdf = max(bsdfPdf, 0.000001);
    float cachePdf2 = cachePdf * cachePdf;
    float competingPdf2 = competingPdf * competingPdf;
    return cachePdf2 / max(cachePdf2 + competingPdf2, 0.000001);
}

float computeConservativeCachedReuseMisWeight(float cacheReuseWeight,
                                              float rawMisWeight,
                                              float cacheMisStrength)
{
    float ceiling = cacheReuseWeight * max(cacheMisStrength, 0.0);
    return min(max(rawMisWeight, 0.0), ceiling);
}

bool candidateWithinConnectionRadius(float3 hitPos,
                                     CachedSunVisibleCandidate candidate,
                                     float cacheConnectionRadius,
                                     out float distanceSq)
{
    float3 delta = candidate.position - hitPos;
    distanceSq = dot(delta, delta);
    float maxConnectionDistanceSq = cacheConnectionRadius * cacheConnectionRadius;
    return distanceSq > 0.000001 && distanceSq <= maxConnectionDistanceSq;
}

bool candidateGeometryCompatible(float3 hitPos, float3 N, CachedSunVisibleCandidate candidate)
{
    float3 toCached = candidate.position - hitPos;
    float rawDistanceSq = dot(toCached, toCached);
    if (!(rawDistanceSq > 0.000001)) {
        return false;
    }

    float3 bounceDir = toCached * rsqrt(rawDistanceSq);
    float NdotBounce = max(dot(N, bounceDir), 0.0);
    float cachedNdot = max(dot(candidate.normal, -bounceDir), 0.0);
    return NdotBounce > 0.0 && cachedNdot > 0.0;
}

void considerCachedSunVisibleCandidate(float3 hitPos,
                                       float3 N,
                                       CachedSunVisibleCandidate candidate,
                                       float sourcePdf,
                                       float cacheConnectionRadius,
                                       inout bool foundCandidate,
                                       inout float bestDistanceSq,
                                       inout CachedSunVisibleCandidate selectedCandidate,
                                       inout float selectedSourcePdf)
{
    float distanceSq = 0.0;
    if (!candidateWithinConnectionRadius(hitPos, candidate, cacheConnectionRadius, distanceSq)) {
        ptAnalysisCounters.InterlockedAdd(cacheReuseSelectRejectDistanceCountOffset, 1u);
        return;
    }
    if (!candidateGeometryCompatible(hitPos, N, candidate)) {
        ptAnalysisCounters.InterlockedAdd(cacheReuseSelectRejectGeometryCountOffset, 1u);
        return;
    }

    if (!foundCandidate || distanceSq < bestDistanceSq) {
        foundCandidate = true;
        bestDistanceSq = distanceSq;
        selectedCandidate = candidate;
        selectedSourcePdf = sourcePdf;
    }
}

bool selectSpatialCachedSunVisibleCandidate(float3 hitPos,
                                            float3 N,
                                            uint available,
                                            uint readBank,
                                            int spatialCandidateTrials,
                                            float cacheConnectionRadius,
                                            uint cacheGeneration,
                                            uint cacheMaxRecordAgeFrames,
                                            out CachedSunVisibleCandidate selectedCandidate,
                                            out float selectedSourcePdf,
                                            inout uint rngState)
{
    bool foundCandidate = false;
    float bestDistanceSq = 3.402823e+38;
    selectedSourcePdf = 0.0;
    int3 centerCell = spatialCacheCell(hitPos);
    int cacheRadiusCellCount =
        clamp(int(ceil(cacheConnectionRadius / CACHED_REUSE_SPATIAL_CELL_SIZE)), 1, CACHED_REUSE_MAX_RADIUS_CELLS);
    int cellDiameter = cacheRadiusCellCount * 2 + 1;
    int cellSearchCount = cellDiameter * cellDiameter * cellDiameter;

    int trialCount = min(clamp(spatialCandidateTrials, 1, CACHED_REUSE_SPATIAL_CANDIDATE_TRIALS), int(available));
    for (int trial = 0; trial < trialCount; ++trial) {
        int neighborIndex =
            (int(randomFloat(rngState) * float(cellSearchCount)) + trial) % cellSearchCount;
        int3 neighborOffset = int3((neighborIndex % cellDiameter) - cacheRadiusCellCount,
                                   ((neighborIndex / cellDiameter) % cellDiameter) - cacheRadiusCellCount,
                                   (neighborIndex / (cellDiameter * cellDiameter)) - cacheRadiusCellCount);
        int3 candidateCell = centerCell + neighborOffset;
        uint bucketSlot = uint(randomFloat(rngState) * float(CACHED_REUSE_SPATIAL_BUCKET_SIZE));
        bucketSlot = (bucketSlot + uint(trial)) % CACHED_REUSE_SPATIAL_BUCKET_SIZE;
        uint slot = sunVisibleCandidateSpatialBankSlot(readBank, candidateCell, bucketSlot);
        CachedSunVisibleCandidate candidate;
        float sourcePdf = 0.0;
        if (!loadSunVisibleCandidate(slot, ubo.frameCount, cacheGeneration, cacheMaxRecordAgeFrames, candidate, sourcePdf)) {
            ptAnalysisCounters.InterlockedAdd(cacheReuseSelectLoadMissCountOffset, 1u);
            continue;
        }
        ptAnalysisCounters.InterlockedAdd(cacheReuseSelectLoadSuccessCountOffset, 1u);

        considerCachedSunVisibleCandidate(hitPos, N, candidate, sourcePdf, cacheConnectionRadius, foundCandidate,
                                          bestDistanceSq, selectedCandidate, selectedSourcePdf);
    }

    int fallbackTrialCount = CACHED_REUSE_GLOBAL_FALLBACK_TRIALS;
    for (int trial = 0; trial < fallbackTrialCount; ++trial) {
        uint bankSlot =
            uint(randomFloat(rngState) * float(SUN_VISIBLE_CANDIDATE_CACHE_BANK_CAPACITY)) %
            SUN_VISIBLE_CANDIDATE_CACHE_BANK_CAPACITY;
        uint slot = sunVisibleCandidateBankSlot(readBank, bankSlot);
        CachedSunVisibleCandidate candidate;
        float sourcePdf = 0.0;
        if (!loadSunVisibleCandidate(slot, ubo.frameCount, cacheGeneration, cacheMaxRecordAgeFrames, candidate, sourcePdf)) {
            ptAnalysisCounters.InterlockedAdd(cacheReuseSelectLoadMissCountOffset, 1u);
            continue;
        }
        ptAnalysisCounters.InterlockedAdd(cacheReuseSelectLoadSuccessCountOffset, 1u);

        considerCachedSunVisibleCandidate(hitPos, N, candidate, sourcePdf, cacheConnectionRadius, foundCandidate,
                                          bestDistanceSq, selectedCandidate, selectedSourcePdf);
    }

    return foundCandidate;
}

bool selectGlobalCachedSunVisibleCandidate(float3 hitPos,
                                           float3 N,
                                           uint available,
                                           uint readBank,
                                           int candidateTrials,
                                           float cacheConnectionRadius,
                                           uint cacheGeneration,
                                           uint cacheMaxRecordAgeFrames,
                                           out CachedSunVisibleCandidate selectedCandidate,
                                           out float selectedSourcePdf,
                                           inout uint rngState)
{
    bool foundCandidate = false;
    float bestDistanceSq = 1.0e30;
    selectedSourcePdf = 0.0;

    int trialCount = clamp(candidateTrials, 1, CACHED_REUSE_SPATIAL_CANDIDATE_TRIALS);
    for (int trial = 0; trial < trialCount; ++trial) {
        uint bankSlot =
            uint(randomFloat(rngState) * float(SUN_VISIBLE_CANDIDATE_CACHE_BANK_CAPACITY)) %
            SUN_VISIBLE_CANDIDATE_CACHE_BANK_CAPACITY;
        uint slot = sunVisibleCandidateBankSlot(readBank, bankSlot);

        CachedSunVisibleCandidate candidate;
        float sourcePdf = 0.0;
        if (!loadSunVisibleCandidate(slot, ubo.frameCount, cacheGeneration, cacheMaxRecordAgeFrames, candidate, sourcePdf)) {
            ptAnalysisCounters.InterlockedAdd(cacheReuseSelectLoadMissCountOffset, 1u);
            continue;
        }
        ptAnalysisCounters.InterlockedAdd(cacheReuseSelectLoadSuccessCountOffset, 1u);

        considerCachedSunVisibleCandidate(hitPos, N, candidate, sourcePdf, cacheConnectionRadius,
                                          foundCandidate, bestDistanceSq, selectedCandidate, selectedSourcePdf);
    }

    return foundCandidate;
}

bool selectCachedSunVisibleConnection(float3 hitPos,
                                      float3 N,
                                      float cacheConnectionRadius,
                                      uint cacheGeneration,
                                      uint cacheMaxRecordAgeFrames,
                                      out CachedSunVisibleCandidate selectedCandidate,
                                      out float selectedSourcePdf,
                                      inout uint rngState)
{
    selectedCandidate.position = float3(0.0, 0.0, 0.0);
    selectedCandidate.normal = float3(0.0, 1.0, 0.0);
    selectedCandidate.albedo = float3(0.0, 0.0, 0.0);
    selectedCandidate.F0 = float3(0.0, 0.0, 0.0);
    selectedCandidate.roughness = 1.0;
    selectedCandidate.metallic = 0.0;
    selectedSourcePdf = 0.0;

    uint readBank = connectionCacheReadBank(ubo.frameCount);
    uint readInsertCount = ptSunVisibleConnectionCache.Load(sunVisibleConnectionBankInsertCountOffset(readBank));
    uint available = min(readInsertCount, SUN_VISIBLE_CONNECTION_CACHE_BANK_CAPACITY);
    if (available == 0) {
        return false;
    }

    bool foundCandidate = false;
    float bestPrimaryDistanceSq = 1.0e30;
    float primaryReuseRadius = min(cacheConnectionRadius, CACHED_REUSE_DEFAULT_CONNECTION_DISTANCE);
    int trialCount = min(CACHED_REUSE_SPATIAL_CANDIDATE_TRIALS, int(available));
    for (int trial = 0; trial < trialCount; ++trial) {
        uint bankSlot =
            uint(randomFloat(rngState) * float(max(available, 1u))) % max(available, 1u);
        bankSlot = (bankSlot + uint(trial)) % max(available, 1u);
        uint slot = sunVisibleConnectionBankSlot(readBank, bankSlot);

        float3 primaryPosition;
        float3 primaryNormal;
        CachedSunVisibleCandidate candidate;
        float sourcePdf = 0.0;
        if (!loadSunVisibleConnection(slot, ubo.frameCount, cacheGeneration, cacheMaxRecordAgeFrames,
                                      primaryPosition, primaryNormal, candidate, sourcePdf)) {
            continue;
        }

        float3 primaryDelta = primaryPosition - hitPos;
        float primaryDistanceSq = dot(primaryDelta, primaryDelta);
        if (primaryDistanceSq > primaryReuseRadius * primaryReuseRadius) {
            continue;
        }
        if (dot(normalize(primaryNormal), normalize(N)) <= 0.5) {
            continue;
        }
        if (!foundCandidate || primaryDistanceSq < bestPrimaryDistanceSq) {
            foundCandidate = true;
            bestPrimaryDistanceSq = primaryDistanceSq;
            selectedCandidate = candidate;
            selectedSourcePdf = sourcePdf;
        }
    }

    return foundCandidate;
}

FirstHitDiffuseBounceResult sampleCachedSunVisibleCandidate(float3 hitPos, float3 N, float3 V,
                                                            RayPayload payload,
                                                            int firstHitDiffuseSampleCount,
                                                            float cacheReuseWeight,
                                                            float cacheMisStrength,
                                                            int cacheWeightingMode,
                                                            int cacheProposalMode,
                                                            int cacheSpatialCandidateTrials,
                                                            int cacheVisibilityValidationBudget,
                                                            float cacheConnectionRadius,
                                                            uint cacheGeneration,
                                                            uint cacheMaxRecordAgeFrames,
                                                            bool currentHitIsDiagnosticTarget,
                                                            inout uint rngState)
{
    FirstHitDiffuseBounceResult result;
    result.totalContribution = float3(0.0, 0.0, 0.0);
    result.secondaryDirectSunContribution = float3(0.0, 0.0, 0.0);
    ptAnalysisCounters.InterlockedAdd(cacheReusePathEntryCountOffset, 1u);

    int extraSampleCount = clamp(firstHitDiffuseSampleCount - 1, 0, 7);
    if (extraSampleCount <= 0) {
        ptAnalysisCounters.InterlockedAdd(cacheReuseExtraSampleZeroCountOffset, 1u);
        return result;
    }

    uint readBank = cacheReadBank(ubo.frameCount);
    uint readInsertCount = ptSunVisibleCandidateCache.Load(sunVisibleCandidateBankInsertCountOffset(readBank));
    uint available = min(readInsertCount, SUN_VISIBLE_CANDIDATE_CACHE_BANK_CAPACITY);
    if (cacheProposalMode != CACHE_PROPOSAL_CONNECTION_CACHE && available == 0) {
        ptAnalysisCounters.InterlockedAdd(cacheReuseUnavailableCandidateCountOffset, 1u);
        return result;
    }

    int visibilityValidationCount = 0;
    for (int probe = 0; probe < extraSampleCount; ++probe) {
        if (visibilityValidationCount >= cacheVisibilityValidationBudget) {
            break;
        }
        ptAnalysisCounters.InterlockedAdd(56, 1u);
        if (currentHitIsDiagnosticTarget) {
            ptAnalysisCounters.InterlockedAdd(diagnosticTargetCacheReuseAttemptCountOffset, 1u);
        }

        CachedSunVisibleCandidate candidate;
        float sourcePdf = 0.0;
        bool selectedFromCache = false;
        bool selectedConnectionProposal = false;
        if (cacheProposalMode == CACHE_PROPOSAL_CONNECTION_CACHE) {
            ptAnalysisCounters.InterlockedAdd(cacheConnectionReuseAttemptsOffset, 1u);
            selectedConnectionProposal =
                selectCachedSunVisibleConnection(hitPos, N, cacheConnectionRadius,
                                                  cacheGeneration, cacheMaxRecordAgeFrames,
                                                  candidate, sourcePdf, rngState);
            selectedFromCache = selectedConnectionProposal;
            if (!selectedFromCache && available > 0) {
                selectedFromCache =
                    selectGlobalCachedSunVisibleCandidate(hitPos, N, available, readBank,
                                                          cacheSpatialCandidateTrials,
                                                          cacheConnectionRadius,
                                                          cacheGeneration, cacheMaxRecordAgeFrames,
                                                          candidate, sourcePdf, rngState);
            }
        } else if (cacheProposalMode == CACHE_PROPOSAL_GLOBAL_NON_LOCAL) {
            selectedFromCache =
                selectGlobalCachedSunVisibleCandidate(hitPos, N, available, readBank,
                                                      cacheSpatialCandidateTrials,
                                                      cacheConnectionRadius,
                                                      cacheGeneration, cacheMaxRecordAgeFrames,
                                                      candidate, sourcePdf, rngState);
        } else {
            selectedFromCache =
                selectSpatialCachedSunVisibleCandidate(hitPos, N, available, readBank,
                                                       cacheSpatialCandidateTrials,
                                                       cacheConnectionRadius,
                                                       cacheGeneration, cacheMaxRecordAgeFrames,
                                                       candidate, sourcePdf, rngState);
        }
        if (!selectedFromCache) {
            ptAnalysisCounters.InterlockedAdd(cacheReuseSelectMissCountOffset, 1u);
            recordFirstHitProbeStats(false, false, float3(0.0, 0.0, 0.0));
            continue;
        }
        ptAnalysisCounters.InterlockedAdd(cacheReuseSelectedCountOffset, 1u);
        if (currentHitIsDiagnosticTarget) {
            ptAnalysisCounters.InterlockedAdd(diagnosticTargetCacheSelectedCountOffset, 1u);
        }
        if (visibilityValidationCount >= cacheVisibilityValidationBudget) {
            break;
        }
        ++visibilityValidationCount;

        bool accepted = false;
        int rejectReason = CACHED_REUSE_REJECT_NONE;
        float3 reusedContribution =
            evaluateCachedSunVisibleContribution(hitPos, N, V, payload, candidate, cacheConnectionRadius, accepted, rejectReason);
        if (!accepted) {
            if (rejectReason == CACHED_REUSE_REJECT_DISTANCE) {
                ptAnalysisCounters.InterlockedAdd(cacheReuseRejectDistanceCountOffset, 1u);
            } else if (rejectReason == CACHED_REUSE_REJECT_GEOMETRY) {
                ptAnalysisCounters.InterlockedAdd(cacheReuseRejectGeometryCountOffset, 1u);
            } else if (rejectReason == CACHED_REUSE_REJECT_VISIBILITY) {
                ptAnalysisCounters.InterlockedAdd(cacheReuseRejectVisibilityCountOffset, 1u);
            }
            if (currentHitIsDiagnosticTarget) {
                if (rejectReason == CACHED_REUSE_REJECT_DISTANCE) {
                    ptAnalysisCounters.InterlockedAdd(diagnosticTargetCacheRejectDistanceCountOffset, 1u);
                } else if (rejectReason == CACHED_REUSE_REJECT_GEOMETRY) {
                    ptAnalysisCounters.InterlockedAdd(diagnosticTargetCacheRejectGeometryCountOffset, 1u);
                } else if (rejectReason == CACHED_REUSE_REJECT_VISIBILITY) {
                    ptAnalysisCounters.InterlockedAdd(diagnosticTargetCacheRejectVisibilityCountOffset, 1u);
                }
            }
            recordFirstHitProbeStats(true, false, float3(0.0, 0.0, 0.0));
            continue;
        }
        recordSunVisibleConnectionCandidate(hitPos, N, candidate, sourcePdf, cacheGeneration);
        if (selectedConnectionProposal) {
            ptAnalysisCounters.InterlockedAdd(cacheConnectionReuseAcceptedOffset, 1u);
        }

        float3 cachedDirection = normalize(candidate.position - hitPos);
        float NdotBounceForPdf = max(dot(N, cachedDirection), 0.0);
        float bsdfPdf = max(NdotBounceForPdf / PI, 0.000001);
        float cacheMisWeight = computeCachedReuseMisWeight(bsdfPdf, sourcePdf, cacheWeightingMode);
        uint scaledMisWeight = uint(clamp(cacheMisWeight, 0.0, 1.0) * 65535.0 + 0.5);
        ptAnalysisCounters.InterlockedAdd(80, scaledMisWeight);
        ptAnalysisCounters.InterlockedAdd(84, 1u);

        if (cacheWeightingMode == CACHE_WEIGHTING_MIS_APPROX) {
            float conservativeMisWeight =
                computeConservativeCachedReuseMisWeight(cacheReuseWeight, cacheMisWeight, cacheMisStrength);
            reusedContribution = min(reusedContribution * conservativeMisWeight, float3(8.0, 8.0, 8.0));
        } else {
            reusedContribution = min(reusedContribution * cacheReuseWeight, float3(8.0, 8.0, 8.0));
        }
        result.totalContribution += reusedContribution;
        result.secondaryDirectSunContribution += reusedContribution;
        float acceptedDistance = length(candidate.position - hitPos);
        if (acceptedDistance <= 3.5) {
            ptAnalysisCounters.InterlockedAdd(cacheReuseAcceptedDistanceNearCountOffset, 1u);
        } else if (acceptedDistance <= 7.0) {
            ptAnalysisCounters.InterlockedAdd(cacheReuseAcceptedDistanceMidCountOffset, 1u);
        } else if (acceptedDistance <= 14.0) {
            ptAnalysisCounters.InterlockedAdd(cacheReuseAcceptedDistanceFarCountOffset, 1u);
        } else {
            ptAnalysisCounters.InterlockedAdd(cacheReuseAcceptedDistanceVeryFarCountOffset, 1u);
        }

        float acceptedLuma = pathTracerLuminance(reusedContribution);
        if (acceptedLuma <= 0.0001) {
            ptAnalysisCounters.InterlockedAdd(cacheReuseAcceptedLumaDarkCountOffset, 1u);
        } else if (acceptedLuma <= 0.001) {
            ptAnalysisCounters.InterlockedAdd(cacheReuseAcceptedLumaDimCountOffset, 1u);
        } else if (acceptedLuma <= 0.01) {
            ptAnalysisCounters.InterlockedAdd(cacheReuseAcceptedLumaUsefulCountOffset, 1u);
        } else {
            ptAnalysisCounters.InterlockedAdd(cacheReuseAcceptedLumaBrightCountOffset, 1u);
        }

        bool sunVisible = acceptedLuma > 0.000001;
        recordFirstHitProbeStats(true, sunVisible, reusedContribution);
        ptAnalysisCounters.InterlockedAdd(60, 1u);
        uint scaledLuma = uint(clamp(acceptedLuma, 0.0, 8.0) * 64.0 + 0.5);
        ptAnalysisCounters.InterlockedAdd(64, scaledLuma);
        if (currentHitIsDiagnosticTarget) {
            ptAnalysisCounters.InterlockedAdd(diagnosticTargetCacheReuseAcceptedCountOffset, 1u);
            ptAnalysisCounters.InterlockedAdd(diagnosticTargetCacheReuseContributionSumOffset, scaledLuma);
        }
        if (visibilityValidationCount >= cacheVisibilityValidationBudget) {
            break;
        }
    }

    result.totalContribution /= float(firstHitDiffuseSampleCount);
    result.secondaryDirectSunContribution /= float(firstHitDiffuseSampleCount);
    return result;
}

float3 sampleEnvironmentNEE(float3 hitPos, float3 N, float3 V, float3 throughput, RayPayload payload, float diffuseSelectionProbability, int samplingMode, inout uint rngState)
{
    float2 xi = float2(randomFloat(rngState), randomFloat(rngState));
    float3 envDir = (samplingMode == ENV_NEE_SAMPLING_SKY_BIASED)
        ? sampleSkyBiasedEnvironmentDirection(xi)
        : cosineSampleHemisphere(xi, N);
    float envPdf = environmentSamplePdf(envDir, N, samplingMode);
    float NdotEnv = max(dot(N, envDir), 0.0);
    if (envPdf <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }
    if (NdotEnv <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }

    RayPayload envPayload = makeVisibilityPayload();
    RayDesc envRay;
    envRay.Origin    = hitPos + N * 0.002;
    envRay.Direction = envDir;
    envRay.TMin      = 0.001;
    envRay.TMax      = 10000.0;
    TraceRay(tlas,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
        0xFF, 0, 0, 0, envRay, envPayload);

    if (envPayload.hitT >= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }

    float3 F = fresnelSchlick(max(dot(N, V), 0.0), payload.F0);
    float3 kD = (float3(1.0, 1.0, 1.0) - F) * (1.0 - payload.metallic);
    float3 diffuseBsdf = kD * payload.albedo / PI;
    float bsdfPdf = bsdfPdfForEnvironmentDirection(envDir, N, diffuseSelectionProbability);
    float misWeight = powerHeuristic(1.0, envPdf, 1.0, bsdfPdf);
    return throughput * diffuseBsdf * envPayload.emission * NdotEnv * misWeight / max(envPdf, 0.0001);
}

[shader("raygeneration")]
void main() {
    uint2  launchID   = DispatchRaysIndex().xy;
    uint2  launchSize = DispatchRaysDimensions().xy;

    // Per-pixel RNG seed: unique per pixel and per frame.
    uint rngState = pcgHash(launchID.x + launchID.y * launchSize.x + ubo.frameCount * launchSize.x * launchSize.y);

    // Sub-pixel jitter (disabled when jitter fields are zero; set to Halton values to enable TAA).
    float2 pixelCenter = float2(launchID) + float2(0.5 + ubo.jitter_x, 0.5 + ubo.jitter_y);
    float2 inUV        = pixelCenter / float2(launchSize);
    float2 d           = inUV * 2.0 - 1.0;

    float4 target = mul(ubo.projInverse, float4(d.x, -d.y, 1.0, 1.0));
    float3 rayDir = mul(ubo.viewInverse, float4(normalize(target.xyz / target.w), 0.0)).xyz;
    float3 origin = ubo.cameraPos.xyz;

    // ── Path tracing loop ──────────────────────────────────────────────────
    float3 radiance   = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);
    float3 debugDirectLighting = float3(0.0, 0.0, 0.0);
    float3 debugIndirectLighting = float3(0.0, 0.0, 0.0);
    float3 debugSkyContribution = float3(0.0, 0.0, 0.0);
    float3 debugThroughput = float3(1.0, 1.0, 1.0);
    float3 debugEnvironmentNeeContribution = float3(0.0, 0.0, 0.0);
    float3 debugFirstHitBounceContribution = float3(0.0, 0.0, 0.0);
    float3 debugSecondaryDirectSunContribution = float3(0.0, 0.0, 0.0);
    float3 debugBaselineContinuationContribution = float3(0.0, 0.0, 0.0);
    float  debugShadowVisibility = 0.0;
    int    debugBounceCount = 0;
    uint   primaryInstanceID = 0;
    bool   primarySurfaceHit = false;
    float  pendingSkyMissMisWeight = 1.0;
    uint   packedPathTracerFlags = push.padding3;
    bool   environmentNeeEnabled = (packedPathTracerFlags & 0x1) != 0;
    bool   applyFirstHitProbesToFinal = (push.padding3 & PATH_TRACER_APPLY_FIRST_HIT_PROBES_BIT) != 0;
    bool   sunVisibleCandidateCacheEnabled = (packedPathTracerFlags & SUN_VISIBLE_CACHE_ENABLE_BIT) != 0;
    bool   adaptiveCacheRefresh = (packedPathTracerFlags & ADAPTIVE_CACHE_REFRESH_BIT) != 0;
    bool   targetedDiagnosticCacheRefresh = (packedPathTracerFlags & TARGETED_DIAGNOSTIC_CACHE_REFRESH_BIT) != 0;
    int    environmentNeeSamplingMode = int((packedPathTracerFlags >> 3) & 0x3);
    int    firstHitDiffuseSampleCount = clamp(int((packedPathTracerFlags >> 5) & 0xF), 1, 8);
    int    candidateCount = int((packedPathTracerFlags >> 9) & 0xF) + 1;
    candidateCount = clamp(candidateCount, 2, 16);
    int    firstHitProbeSamplingMode = int((packedPathTracerFlags >> 13) & 0x7);
    float  cacheReuseWeight = clamp(push.skyData.x, 0.0, 1.0);
    float  cacheMisStrength = clamp(push.skyData.w, 0.0, 4.0);
    int    cacheWeightingMode = int((packedPathTracerFlags >> CACHE_WEIGHTING_MODE_SHIFT) & 0x3);
    int    cacheRefreshCandidateCount = int((packedPathTracerFlags >> CACHE_REFRESH_COUNT_SHIFT) & 0x7);
    int    cacheSpatialCandidateTrials = int((packedPathTracerFlags >> CACHE_SPATIAL_TRIAL_COUNT_SHIFT) & 0x3F);
    cacheSpatialCandidateTrials = clamp(cacheSpatialCandidateTrials, 1, CACHED_REUSE_SPATIAL_CANDIDATE_TRIALS);
    int    cacheVisibilityBudgetCode = int((packedPathTracerFlags >> CACHE_VISIBILITY_BUDGET_SHIFT) & 0x3);
    int    cacheVisibilityValidationBudget =
        (cacheVisibilityBudgetCode == 0) ? 1 :
        ((cacheVisibilityBudgetCode == 1) ? 2 :
        ((cacheVisibilityBudgetCode == 2) ? 4 : 7));
    float  cacheConnectionRadius = clamp(push.skyData.y, 0.5, 64.0);
    uint   packedPathTracerMaterialIndex = uint(push.materialIndex);
    int    cacheProposalMode = int(packedPathTracerMaterialIndex & 0xFFu);
    cacheProposalMode = clamp(cacheProposalMode, CACHE_PROPOSAL_SPATIAL_LOCAL, CACHE_PROPOSAL_CONNECTION_CACHE);
    uint   cacheMaxRecordAgeFrames = max((packedPathTracerMaterialIndex >> 8) & 0x3FFu, 1u); // cacheMaxRecordAgeFrames = max(packedPathTracerMaterialIndex >> 8, 1u)
    int    reservoirGiMode = int((packedPathTracerMaterialIndex >> 18) & 0x3u);
    int    reservoirGiCandidateCount = clamp(int((packedPathTracerMaterialIndex >> 20) & 0x3u) + 1, 1, 4);
    bool   reservoirGiUseCandidateRis = ((packedPathTracerMaterialIndex >> 22) & 0x1u) != 0u;
    int    reservoirGiSpatialNeighborCount = clamp(int((packedPathTracerMaterialIndex >> 23) & 0x7u) + 1, 1, 8);
    uint   cacheGeneration = uint(max(push.skyData.z, 1.0));

    RayDesc ray;
    ray.Origin    = origin;
    ray.Direction = normalize(rayDir);
    ray.TMin      = 0.001;
    ray.TMax      = 10000.0;

    // Keep a higher hard cap for enclosed/interior transport; Russian roulette below
    // keeps the average path length bounded for performance.
    static const int MAX_BOUNCES = 8;
    static const int RR_START_BOUNCE = 2;

    for (int bounce = 0; bounce < MAX_BOUNCES; ++bounce)
    {
        RayPayload payload;
        payload.hitT = -1.0;

        TraceRay(tlas, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);

        // Sky miss: accumulate background emission and terminate.
        if (payload.hitT < 0.0) {
            float3 skyContribution = throughput * payload.emission * pendingSkyMissMisWeight;
            radiance += skyContribution;
            debugSkyContribution += skyContribution;
            if (bounce > 0) {
                debugIndirectLighting += skyContribution;
                debugBaselineContinuationContribution += skyContribution;
            }
            // Write sentinel G-Buffer values so sky pixels don't leave stale/garbage data
            // that would confuse the denoiser's edge-stopping functions.
            if (bounce == 0) {
                ptAnalysisCounters.InterlockedAdd(16, 1u); // primary sky-hit count
                gBufferNormals[launchID] = float4(0.0, 0.0, 0.0, 0.0);
                gBufferDepth[launchID]   = -1.0;
                motionVectors[launchID]  = float2(0.0, 0.0);
                if (reservoirGiMode >= PATH_TRACER_RESERVOIR_GI_SINGLE_FRAME) {
                    ReservoirGiRecord emptyReservoir;
                    emptyReservoir.primaryPosition = origin;
                    emptyReservoir.candidatePosition = origin;
                    emptyReservoir.candidateNormal = float3(0.0, 0.0, 0.0);
                    emptyReservoir.contribution = float3(0.0, 0.0, 0.0);
                    emptyReservoir.secondaryDirectSun = float3(0.0, 0.0, 0.0);
                    emptyReservoir.weightSum = 0.0;
                    emptyReservoir.selectedWeight = 0.0;
                    emptyReservoir.frameId = ubo.frameCount;
                    emptyReservoir.flags = 0u;
                    updateReservoirGi(launchID, launchSize, emptyReservoir);
                }
            }
            break;
        }

        // Accumulate emissive contribution from the surface.
        float3 emissiveContribution = throughput * payload.emission;
        radiance += emissiveContribution;
        if (bounce == 0) {
            debugDirectLighting += emissiveContribution;
        } else {
            debugIndirectLighting += emissiveContribution;
            debugBaselineContinuationContribution += emissiveContribution;
        }
        debugBounceCount = bounce + 1;
        if (bounce == 0) {
            primarySurfaceHit = true;
            primaryInstanceID = payload.instanceID;
        }

        float3 N2 = payload.worldNormal;
        float3 V2 = -ray.Direction;

        if (bounce == 0 &&
            (firstHitDiffuseSampleCount > 1 ||
             reservoirGiMode >= PATH_TRACER_RESERVOIR_GI_SINGLE_FRAME)) {
            FirstHitDiffuseBounceResult firstHitBounce;
            if (reservoirGiMode >= PATH_TRACER_RESERVOIR_GI_SINGLE_FRAME) {
                firstHitBounce = sampleFirstHitReservoirGiSingleFrame(payload.hitPos, N2, V2, payload,
                                                                      max(firstHitDiffuseSampleCount, 2),
                                                                      reservoirGiCandidateCount,
                                                                      reservoirGiUseCandidateRis,
                                                                      reservoirGiMode,
                                                                      reservoirGiSpatialNeighborCount,
                                                                      launchID, launchSize, rngState);
            } else if (firstHitProbeSamplingMode == FIRST_HIT_PROBE_SAMPLING_CACHED_SECONDARY_REUSE) {
                if (sunVisibleCandidateCacheEnabled && adaptiveCacheRefresh) {
                    refreshSunVisibleCandidateCache(payload.hitPos, N2, V2, payload,
                                                    cacheRefreshCandidateCount, cacheGeneration,
                                                    rngState);
                }
                bool currentHitIsDiagnosticTarget =
                    push.cascadeIndex >= 0 &&
                    ((payload.instanceID >> 14) == uint(push.cascadeIndex));
                if (sunVisibleCandidateCacheEnabled &&
                    targetedDiagnosticCacheRefresh &&
                    currentHitIsDiagnosticTarget) {
                    refreshDiagnosticTargetSunVisibleCandidateCache(payload.hitPos, N2, V2, payload,
                                                                    cacheGeneration, rngState);
                }
                firstHitBounce = sampleCachedSunVisibleCandidate(payload.hitPos, N2, V2, payload,
                                                                 firstHitDiffuseSampleCount, cacheReuseWeight,
                                                                 cacheMisStrength,
                                                                 cacheWeightingMode,
                                                                 cacheProposalMode,
                                                                 cacheSpatialCandidateTrials,
                                                                 cacheVisibilityValidationBudget,
                                                                 cacheConnectionRadius,
                                                                 cacheGeneration, cacheMaxRecordAgeFrames,
                                                                 currentHitIsDiagnosticTarget, rngState);
            } else if (firstHitProbeSamplingMode == FIRST_HIT_PROBE_SAMPLING_CANDIDATE_SUN_BOUNCE) {
                firstHitBounce = sampleFirstHitCandidateSunBounce(payload.hitPos, N2, V2, payload,
                                                                  firstHitDiffuseSampleCount, candidateCount,
                                                                  sunVisibleCandidateCacheEnabled, cacheGeneration,
                                                                  rngState);
            } else if (firstHitProbeSamplingMode == FIRST_HIT_PROBE_SAMPLING_CANDIDATE_AVERAGE_REFERENCE) {
                firstHitBounce = sampleFirstHitCandidateAverageReference(payload.hitPos, N2, V2, payload,
                                                                         firstHitDiffuseSampleCount, candidateCount,
                                                                         sunVisibleCandidateCacheEnabled, cacheGeneration,
                                                                         rngState);
            } else if (firstHitProbeSamplingMode == FIRST_HIT_PROBE_SAMPLING_CANDIDATE_RIS) {
                firstHitBounce = sampleFirstHitCandidateRis(payload.hitPos, N2, V2, payload,
                                                            firstHitDiffuseSampleCount, candidateCount,
                                                            sunVisibleCandidateCacheEnabled, cacheGeneration,
                                                            rngState);
            } else {
                firstHitBounce = sampleFirstHitDiffuseBounce(payload.hitPos, N2, V2, payload,
                                                            firstHitDiffuseSampleCount, firstHitProbeSamplingMode,
                                                            sunVisibleCandidateCacheEnabled, cacheGeneration,
                                                            rngState);
            }
            if (applyFirstHitProbesToFinal) {
                radiance += firstHitBounce.totalContribution;
                debugIndirectLighting += firstHitBounce.totalContribution;
            }
            debugFirstHitBounceContribution += firstHitBounce.totalContribution;
            debugSecondaryDirectSunContribution += firstHitBounce.secondaryDirectSunContribution;
        }

        // ── NEE: direct lighting from directional sun light ────────────────
        // ubo.lightDir points FROM the light TOWARD the scene, so negate for surface-to-light.
        float3 Ldir  = normalize(-ubo.lightDir.xyz);
        float  NdotL = max(dot(N2, Ldir), 0.0);
        if (NdotL > 0.0)
        {
            // Cheap occlusion ray: skip ClosestHit, terminate on first accepted hit.
            RayPayload shadowPayload;
            shadowPayload.hitT        = 1.0;   // positive sentinel = "occluded"; Miss sets it to -1
            shadowPayload.albedo      = float3(0.0, 0.0, 0.0);
            shadowPayload.emission    = float3(0.0, 0.0, 0.0);
            shadowPayload.worldNormal = float3(0.0, 1.0, 0.0);
            shadowPayload.hitPos      = float3(0.0, 0.0, 0.0);
            shadowPayload.roughness   = 1.0;
            shadowPayload.metallic    = 0.0;
            shadowPayload.F0          = float3(0.0, 0.0, 0.0);
            shadowPayload.ao          = 1.0;
            shadowPayload.instanceID  = 0;

            RayDesc shadowRay;
            shadowRay.Origin    = payload.hitPos + N2 * 0.002;
            shadowRay.Direction = Ldir;
            shadowRay.TMin      = 0.001;
            shadowRay.TMax      = 10000.0;
            TraceRay(tlas,
                RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
                0xFF, 0, 0, 0, shadowRay, shadowPayload);

            // Binary hard-shadow visibility for the direct sun sample.
            float3 H     = normalize(Ldir + V2);
            float  NdotV = max(dot(N2, V2),   0.0001);
            float  VdotH = max(dot(V2, H),    0.0001);

            float3 Fs   = fresnelSchlick(VdotH, payload.F0);
            float  D    = distributionGGX(N2, H, payload.roughness);
            float  G    = geometrySmith(N2, V2, Ldir, payload.roughness);
            float3 kD   = (float3(1.0, 1.0, 1.0) - Fs) * (1.0 - payload.metallic);

            float3 diffuse  = kD * payload.albedo / PI;
            float3 specular = (D * Fs * G) / max(4.0 * NdotV * NdotL, 0.0001);

            float shadowVisibility = (shadowPayload.hitT < 0.0) ? 1.0 : 0.0;
            float3 directContribution = throughput * (diffuse + specular) * NdotL * SUN_RADIANCE * shadowVisibility;
            radiance += directContribution;
            if (bounce == 0) {
                debugDirectLighting += directContribution;
                debugShadowVisibility = shadowVisibility;
            } else {
                debugIndirectLighting += directContribution;
                debugBaselineContinuationContribution += directContribution;
            }
        }

        // ── Primary-ray G-Buffer write (bounce 0 only) ─────────────────────

        // Primary-ray G-Buffer write (bounce 0 only).
        if (bounce == 0) {
            gBufferNormals[launchID] = float4(payload.worldNormal, 0.0);
            gBufferDepth[launchID]   = payload.hitT;

            // Motion vector: re-project hit world position with previous frame VP.
            float4 prevClip = mul(ubo.prevViewProj, float4(payload.hitPos, 1.0));
            // GLM proj is Y-up NDC: prevClip.y/w = +(1 - 2*v) where v=0 at top.
            // Convert to image UV (Y-down, v=0 at top) by negating Y, matching the
            // -d.y correction used in primary ray generation.
            float2 prevNDC  = (prevClip.xy / prevClip.w) * float2(0.5, -0.5) + 0.5;
            // Use unjittered UV so TAA jitter doesn't appear as false motion in the denoiser.
            float2 unjitteredUV = (float2(launchID) + 0.5) / float2(launchSize);
            motionVectors[launchID] = unjitteredUV - prevNDC;
        }

        // ── BSDF sampling ──────────────────────────────────────────────────
        float3 N = payload.worldNormal;
        float3 V = -ray.Direction;

        float2 xi = float2(randomFloat(rngState), randomFloat(rngState));

        // Probabilistically choose diffuse vs specular lobe based on Fresnel reflectance.
        float3 F       = fresnelSchlick(max(dot(N, V), 0.0), payload.F0);
        float  fAvg    = (F.r + F.g + F.b) / 3.0;
        float  specProb = clamp(fAvg + payload.metallic * 0.5, 0.1, 0.9);
        float  diffuseSelectionProbability = 1.0 - specProb;

        bool useEnvironmentNeeOnThisBounce = environmentNeeEnabled && environmentNeeEnabledForBounce(bounce);
        if (useEnvironmentNeeOnThisBounce) {
            float3 envNeeContribution = sampleEnvironmentNEE(payload.hitPos, N, V, throughput, payload,
                                                             diffuseSelectionProbability, environmentNeeSamplingMode, rngState);
            radiance += envNeeContribution;
            debugIndirectLighting += envNeeContribution;
            debugEnvironmentNeeContribution += envNeeContribution;
            if (bounce > 0) {
                debugBaselineContinuationContribution += envNeeContribution;
            }
        }

        float3 newDir;
        float  pdf;
        float3 bsdfWeight;

        if (randomFloat(rngState) < specProb) {
            // Specular (GGX) lobe
            newDir = ggxSampleDirection(xi, N, V, payload.roughness);
            float NdotL = max(dot(N, newDir), 0.0);
            if (NdotL <= 0.0) break;
            pendingSkyMissMisWeight = 1.0;

            float3 H    = normalize(V + newDir);
            float  G    = geometrySmith(N, V, newDir, payload.roughness);
            float3 Fs   = fresnelSchlick(max(dot(H, V), 0.0), payload.F0);
            float  NdotV = max(dot(N, V),    0.0001);
            float  NdotH = max(dot(N, H),    0.0001);
            float  VdotH = max(dot(V, H),    0.0001);
            bsdfWeight   = (G * Fs * VdotH) / max(NdotV * NdotH, 0.0001);
            pdf          = specProb;
        } else {
            // Diffuse (cosine hemisphere) lobe — cosine factor cancels with PDF.
            newDir = cosineSampleHemisphere(xi, N);
            if (max(dot(N, newDir), 0.0) <= 0.0) break;
            float bsdfEnvPdf = bsdfPdfForEnvironmentDirection(newDir, N, diffuseSelectionProbability);
            float envPdf = environmentSamplePdf(newDir, N, environmentNeeSamplingMode);
            pendingSkyMissMisWeight = useEnvironmentNeeOnThisBounce ? powerHeuristic(1.0, bsdfEnvPdf, 1.0, envPdf) : 1.0;

            float3 kD  = (float3(1.0, 1.0, 1.0) - F) * (1.0 - payload.metallic);
            bsdfWeight = kD * payload.albedo;
            pdf        = diffuseSelectionProbability;
        }

        throughput *= bsdfWeight / max(pdf, 0.0001);
        debugThroughput = throughput;

        // Russian roulette: allow a few guaranteed bounces so enclosed spaces can collect
        // indirect light, then terminate stochastically based on path throughput.
        if (bounce >= RR_START_BOUNCE) {
            float maxT  = max(throughput.r, max(throughput.g, throughput.b));
            float rrProb = clamp(maxT, 0.05, 0.98); // clamp away from 0 to prevent 0/0 NaN
            if (randomFloat(rngState) > rrProb) break;
            throughput /= rrProb;
        }

        // Advance ray — offset origin along geometric normal to avoid self-intersection.
        ray.Origin    = payload.hitPos + payload.worldNormal * 0.001;
        ray.Direction = normalize(newDir);
        ray.TMin      = 0.001;
        ray.TMax      = 10000.0;
    }

    // Write noisy 1-SPP result — linear HDR, tone mapping is applied in the final A-Trous pass.
    float3 outputColor = radiance;
    if (push.padding2 == DEBUG_AOV_PATH_RAW_FINAL_COLOR) {
        outputColor = radiance;
    } else if (push.padding2 == DEBUG_AOV_PATH_DIRECT_LIGHTING) {
        outputColor = debugDirectLighting;
    } else if (push.padding2 == DEBUG_AOV_PATH_INDIRECT_LIGHTING) {
        outputColor = debugIndirectLighting;
    } else if (push.padding2 == DEBUG_AOV_PATH_SKY_CONTRIBUTION) {
        outputColor = debugSkyContribution;
    } else if (push.padding2 == DEBUG_AOV_PATH_THROUGHPUT) {
        float throughputLum = dot(debugThroughput, float3(0.2126, 0.7152, 0.0722));
        outputColor = float3(throughputLum, throughputLum, throughputLum);
    } else if (push.padding2 == DEBUG_AOV_PATH_BOUNCE_COUNT) {
        float bounceFraction = float(debugBounceCount) / float(MAX_BOUNCES);
        outputColor = float3(bounceFraction, bounceFraction, bounceFraction);
    } else if (push.padding2 == DEBUG_AOV_PATH_SHADOW_VISIBILITY) {
        outputColor = float3(debugShadowVisibility, debugShadowVisibility, debugShadowVisibility);
    } else if (push.padding2 == DEBUG_AOV_PATH_ENVIRONMENT_NEE) {
        outputColor = debugEnvironmentNeeContribution;
    } else if (push.padding2 == DEBUG_AOV_PATH_FIRST_HIT_BOUNCE) {
        outputColor = debugFirstHitBounceContribution;
    } else if (push.padding2 == DEBUG_AOV_PATH_SECONDARY_SUN) {
        outputColor = debugSecondaryDirectSunContribution;
    } else if (push.padding2 == DEBUG_AOV_PATH_BASELINE_CONTINUATION) {
        outputColor = debugBaselineContinuationContribution;
    }
    float3 appliedFirstHitProbeContribution =
        applyFirstHitProbesToFinal ? debugFirstHitBounceContribution : float3(0.0, 0.0, 0.0);
    recordTargetWallLuminance(radiance, appliedFirstHitProbeContribution, primaryInstanceID, primarySurfaceHit);
    noisyColorOutput[launchID] = float4(outputColor, 1.0);
}
