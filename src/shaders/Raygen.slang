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
[[vk::binding(12, 0)]] RWByteAddressBuffer ptReservoirGiCurrent;
[[vk::binding(13, 0)]] RWByteAddressBuffer ptReservoirGiHistory;

// Set 1 — global UBO
[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

[[vk::push_constant]] ScenePushConstants push;

static const int RESERVOIR_GI_CURRENT_BINDING = 12;
static const int RESERVOIR_GI_HISTORY_BINDING = 13;
static const uint RESERVOIR_GI_CURRENT_CAPACITY = 262144u;
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
static const int DEBUG_AOV_PATH_RESERVOIR_GI_CONTRIBUTION = 17;
static const int DEBUG_AOV_PATH_RESERVOIR_GI_ACCEPTED_LUMA = 18;
static const int DEBUG_AOV_PATH_RESERVOIR_GI_CANDIDATE_SURFACE_HIT = 19;
static const int DEBUG_AOV_PATH_RESERVOIR_GI_CANDIDATE_SUN_VISIBLE = 20;
static const int DEBUG_AOV_PATH_RESERVOIR_GI_CANDIDATE_POSITIVE_WEIGHT = 21;
static const int DEBUG_AOV_PATH_RESERVOIR_GI_SELECTED_WEIGHT = 22;
static const int ENV_NEE_SAMPLING_COSINE_HEMISPHERE = 0;
static const int ENV_NEE_SAMPLING_SKY_BIASED = 1;
static const int FIRST_HIT_PROBE_SAMPLING_COSINE_HEMISPHERE = 0;
static const int FIRST_HIT_PROBE_SAMPLING_SUN_BOUNCE_GUIDED = 1;
static const int FIRST_HIT_PROBE_SAMPLING_CANDIDATE_SUN_BOUNCE = 2;
static const int FIRST_HIT_PROBE_SAMPLING_CANDIDATE_AVERAGE_REFERENCE = 3;
static const int FIRST_HIT_PROBE_SAMPLING_CANDIDATE_RIS = 4;
static const int PATH_TRACER_RESERVOIR_GI_OFF = 0;
static const int PATH_TRACER_RESERVOIR_GI_SINGLE_FRAME = 1;
static const int PATH_TRACER_RESERVOIR_GI_TEMPORAL = 2;
static const int PATH_TRACER_RESERVOIR_GI_TEMPORAL_SPATIAL = 3;
static const int DIRECT_SUN_BOUNCE_ALL = 0;
static const int DIRECT_SUN_BOUNCE_FIRST_ONLY = 1;
static const int DIRECT_SUN_BOUNCE_FIRST_TWO = 2;
static const int RESERVOIR_GI_CANDIDATE_TRACE_ONLY = 0;
static const int RESERVOIR_GI_CANDIDATE_UNSHADOWED_SUN = 1;
static const int RESERVOIR_GI_CANDIDATE_SHADOWED_SUN = 2;
static const uint PATH_TRACER_APPLY_FIRST_HIT_PROBES_BIT = 1u << 2;
static const uint targetWallFirstHitProbeContributionSumOffset = 52u;
static const uint targetWallBaseLuminanceSumOffset = 56u;
static const uint reservoirGiCandidatesOffset = 60u;
static const uint reservoirGiAcceptedOffset = 64u;
static const uint reservoirGiLumaScaledSumOffset = 68u;
static const uint reservoirGiTemporalAcceptedOffset = 72u;
static const uint reservoirGiTemporalRejectedOffset = 76u;
static const uint reservoirGiSpatialAcceptedOffset = 80u;
static const uint reservoirGiSpatialRejectedOffset = 84u;
static const uint reservoirGiCandidateSurfaceHitsOffset = 88u;
static const uint reservoirGiCandidateSunVisibleOffset = 92u;
static const uint reservoirGiCandidatePositiveWeightOffset = 96u;
static const uint reservoirGiZeroWeightOffset = 100u;
static const uint reservoirGiSelectedWeightScaledSumOffset = 104u;

struct FirstHitDiffuseBounceResult {
    float3 totalContribution;
    float3 secondaryDirectSunContribution;
    float  reservoirCandidateSurfaceHitRatio;
    float  reservoirCandidateSunVisibleRatio;
    float  reservoirCandidatePositiveWeightRatio;
    float  reservoirSelectedWeight;
};

struct FirstHitProbeSample {
    float3 direction;
    float  pdf;
};

struct ReservoirGiRecord {
    float3 primaryPosition;
    float3 candidatePosition;
    float3 primaryNormal;
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
    storeFloat3ToReservoirGi(offset + 32, record.primaryNormal);
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
    record.primaryNormal = loadFloat3FromReservoirGiHistory(offset + 32);
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
    record.primaryNormal = float3(0.0, 0.0, 0.0);
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
        !isSaneReservoirGiVector(record.primaryNormal, 1.1) ||
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
    float3 previousPrimaryNormal = normalize(record.primaryNormal);
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

bool isFinitePositive(float value)
{
    return value > 0.0 && abs(value) < 3.402823e+38;
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

bool directSunEnabledForBounce(int bounce, int directSunBounceMode)
{
    return directSunBounceMode == DIRECT_SUN_BOUNCE_ALL ||
           (directSunBounceMode == DIRECT_SUN_BOUNCE_FIRST_ONLY && bounce == 0) ||
           (directSunBounceMode == DIRECT_SUN_BOUNCE_FIRST_TWO && bounce <= 1);
}

float3 evaluateDirectSunContributionInternal(float3 hitPos, float3 N, float3 V,
                                             float3 throughput, RayPayload payload,
                                             bool traceVisibility)
{
    float3 Ldir = normalize(-ubo.lightDir.xyz);
    float NdotL = max(dot(N, Ldir), 0.0);
    if (NdotL <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }

    float shadowVisibility = 1.0;
    if (traceVisibility) {
        RayPayload shadowPayload = makeVisibilityPayload();
        RayDesc shadowRay;
        shadowRay.Origin    = hitPos + N * 0.002;
        shadowRay.Direction = Ldir;
        shadowRay.TMin      = 0.001;
        shadowRay.TMax      = 10000.0;
        TraceRay(tlas,
            RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
            0xFF, 0, 0, 0, shadowRay, shadowPayload);

        shadowVisibility = (shadowPayload.hitT < 0.0) ? 1.0 : 0.0;
    }
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

float3 evaluateDirectSunContribution(float3 hitPos, float3 N, float3 V, float3 throughput, RayPayload payload)
{
    return evaluateDirectSunContributionInternal(hitPos, N, V, throughput, payload, true);
}

float3 evaluateUnshadowedDirectSunContribution(float3 hitPos, float3 N, float3 V,
                                               float3 throughput, RayPayload payload)
{
    return evaluateDirectSunContributionInternal(hitPos, N, V, throughput, payload, false);
}

FirstHitDiffuseBounceResult sampleFirstHitDiffuseBounce(float3 hitPos, float3 N, float3 V, RayPayload payload,
                                                        int firstHitDiffuseSampleCount,
                                                        int firstHitProbeSamplingMode,
                                                        inout uint rngState)
{
    FirstHitDiffuseBounceResult result;
    result.totalContribution = float3(0.0, 0.0, 0.0);
    result.secondaryDirectSunContribution = float3(0.0, 0.0, 0.0);
    result.reservoirCandidateSurfaceHitRatio = 0.0;
    result.reservoirCandidateSunVisibleRatio = 0.0;
    result.reservoirCandidatePositiveWeightRatio = 0.0;
    result.reservoirSelectedWeight = 0.0;

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
            recordFirstHitProbeStats(true, secondarySunVisible, emissiveContribution + secondaryDirect);
            result.secondaryDirectSunContribution += secondaryDirect;
            result.totalContribution += emissiveContribution + secondaryDirect;
        }
    }

    result.totalContribution /= float(firstHitDiffuseSampleCount);
    result.secondaryDirectSunContribution /= float(firstHitDiffuseSampleCount);
    return result;
}

FirstHitDiffuseBounceResult sampleFirstHitCandidateSunBounce(float3 hitPos, float3 N, float3 V, RayPayload payload,
                                                             int firstHitDiffuseSampleCount,
                                                             int candidateCount,
                                                             inout uint rngState)
{
    FirstHitDiffuseBounceResult result;
    result.totalContribution = float3(0.0, 0.0, 0.0);
    result.secondaryDirectSunContribution = float3(0.0, 0.0, 0.0);
    result.reservoirCandidateSurfaceHitRatio = 0.0;
    result.reservoirCandidateSunVisibleRatio = 0.0;
    result.reservoirCandidatePositiveWeightRatio = 0.0;
    result.reservoirSelectedWeight = 0.0;

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
                                                                    inout uint rngState)
{
    FirstHitDiffuseBounceResult result;
    result.totalContribution = float3(0.0, 0.0, 0.0);
    result.secondaryDirectSunContribution = float3(0.0, 0.0, 0.0);
    result.reservoirCandidateSurfaceHitRatio = 0.0;
    result.reservoirCandidateSunVisibleRatio = 0.0;
    result.reservoirCandidatePositiveWeightRatio = 0.0;
    result.reservoirSelectedWeight = 0.0;

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
                                                       inout uint rngState)
{
    FirstHitDiffuseBounceResult result;
    result.totalContribution = float3(0.0, 0.0, 0.0);
    result.secondaryDirectSunContribution = float3(0.0, 0.0, 0.0);
    result.reservoirCandidateSurfaceHitRatio = 0.0;
    result.reservoirCandidateSunVisibleRatio = 0.0;
    result.reservoirCandidatePositiveWeightRatio = 0.0;
    result.reservoirSelectedWeight = 0.0;

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
                                                                 int reservoirGiCandidateEvaluationMode,
                                                                 uint2 launchID,
                                                                 uint2 launchSize,
                                                                 inout uint rngState)
{
    FirstHitDiffuseBounceResult result;
    result.totalContribution = float3(0.0, 0.0, 0.0);
    result.secondaryDirectSunContribution = float3(0.0, 0.0, 0.0);
    result.reservoirCandidateSurfaceHitRatio = 0.0;
    result.reservoirCandidateSunVisibleRatio = 0.0;
    result.reservoirCandidatePositiveWeightRatio = 0.0;
    result.reservoirSelectedWeight = 0.0;

    ReservoirGiRecord record;
    record.primaryPosition = float3(0.0, 0.0, 0.0);
    record.candidatePosition = float3(0.0, 0.0, 0.0);
    record.primaryNormal = float3(0.0, 0.0, 0.0);
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
    record.primaryNormal = N;

    int extraSampleCount = clamp(firstHitDiffuseSampleCount - 1, 0, 7);
    if (extraSampleCount <= 0) {
        if (reservoirGiMode >= PATH_TRACER_RESERVOIR_GI_TEMPORAL) {
            updateReservoirGi(launchID, launchSize, record);
        }
        return result;
    }

    candidateCount = clamp(candidateCount, 1, 4);

    float3 F = fresnelSchlick(max(dot(N, V), 0.0), payload.F0);
    float3 kD = (float3(1.0, 1.0, 1.0) - F) * (1.0 - payload.metallic);
    float3 diffuseBsdf = kD * payload.albedo / PI;

    float3 selectedTotal = float3(0.0, 0.0, 0.0);
    float3 selectedSecondarySun = float3(0.0, 0.0, 0.0);
    float3 selectedPosition = hitPos;
    float selectedWeight = 0.0;
    float reservoirWeightSum = 0.0;
    bool selectedSurfaceHit = false;
    bool selectedSunVisible = false;
    uint candidateSurfaceHitCount = 0u;
    uint candidateSunVisibleCount = 0u;
    uint candidatePositiveWeightCount = 0u;

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
        ptAnalysisCounters.InterlockedAdd(reservoirGiCandidatesOffset, 1u);
        TraceRay(tlas, RAY_FLAG_NONE, 0xFF, 0, 0, 0, bounceRay, bouncePayload);

        float3 candidateTotal = float3(0.0, 0.0, 0.0);
        float3 candidateSecondarySun = float3(0.0, 0.0, 0.0);
        bool surfaceHit = bouncePayload.hitT >= 0.0;
        bool sunVisible = false;
        float3 candidatePosition = hitPos;

        if (surfaceHit) {
            candidateSurfaceHitCount += 1u;
            ptAnalysisCounters.InterlockedAdd(reservoirGiCandidateSurfaceHitsOffset, 1u);
            if (!isSaneReservoirGiVector(bouncePayload.hitPos, 100000.0) ||
                !isSaneReservoirGiVector(bouncePayload.worldNormal, 1.1)) {
                continue;
            }
            float3 secondaryV = -bounceDir;
            float3 emissiveContribution = firstLegThroughput * bouncePayload.emission;
            if (reservoirGiCandidateEvaluationMode == RESERVOIR_GI_CANDIDATE_UNSHADOWED_SUN) {
                candidateSecondarySun = evaluateUnshadowedDirectSunContribution(bouncePayload.hitPos,
                                                                                bouncePayload.worldNormal,
                                                                                secondaryV,
                                                                                firstLegThroughput,
                                                                                bouncePayload);
            } else if (reservoirGiCandidateEvaluationMode == RESERVOIR_GI_CANDIDATE_SHADOWED_SUN) {
                candidateSecondarySun = evaluateDirectSunContribution(bouncePayload.hitPos,
                                                                      bouncePayload.worldNormal,
                                                                      secondaryV,
                                                                      firstLegThroughput,
                                                                      bouncePayload);
            }
            candidateTotal = emissiveContribution + candidateSecondarySun;
            candidatePosition = bouncePayload.hitPos;
            sunVisible = pathTracerLuminance(candidateSecondarySun) > 0.000001;
            if (sunVisible) {
                candidateSunVisibleCount += 1u;
                ptAnalysisCounters.InterlockedAdd(reservoirGiCandidateSunVisibleOffset, 1u);
            }
        } else {
            candidateTotal = firstLegThroughput * bouncePayload.emission;
        }

        candidateTotal = sanitizeReservoirGiContribution(candidateTotal);
        candidateSecondarySun = sanitizeReservoirGiContribution(candidateSecondarySun);
        float candidateWeight = pathTracerLuminance(candidateTotal);
        if (!isFinitePositive(candidateWeight)) {
            continue;
        }
        candidatePositiveWeightCount += 1u;
        ptAnalysisCounters.InterlockedAdd(reservoirGiCandidatePositiveWeightOffset, 1u);
        if (useCandidateRis) {
            reservoirWeightSum += candidateWeight;
            if (randomFloat(rngState) < candidateWeight / max(reservoirWeightSum, 0.000001)) {
                selectedTotal = candidateTotal;
                selectedSecondarySun = candidateSecondarySun;
                selectedPosition = candidatePosition;
                selectedWeight = candidateWeight;
                selectedSurfaceHit = surfaceHit;
                selectedSunVisible = sunVisible;
            }
        } else if (candidateWeight > selectedWeight) {
            reservoirWeightSum += candidateWeight;
            selectedTotal = candidateTotal;
            selectedSecondarySun = candidateSecondarySun;
            selectedPosition = candidatePosition;
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
    float invCandidateCount = 1.0 / float(max(candidateCount, 1));
    result.reservoirCandidateSurfaceHitRatio = float(candidateSurfaceHitCount) * invCandidateCount;
    result.reservoirCandidateSunVisibleRatio = float(candidateSunVisibleCount) * invCandidateCount;
    result.reservoirCandidatePositiveWeightRatio = float(candidatePositiveWeightCount) * invCandidateCount;
    result.reservoirSelectedWeight = selectedWeight;

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
        uint scaledSelectedWeight = uint(clamp(selectedWeight, 0.0, 8.0) * 64.0 + 0.5);
        ptAnalysisCounters.InterlockedAdd(reservoirGiSelectedWeightScaledSumOffset, scaledSelectedWeight);
        recordFirstHitProbeStats(selectedSurfaceHit, selectedSunVisible, reservoirTotal);
    } else {
        ptAnalysisCounters.InterlockedAdd(reservoirGiZeroWeightOffset, 1u);
        reservoirTotal = float3(0.0, 0.0, 0.0);
        reservoirSecondarySun = float3(0.0, 0.0, 0.0);
        selectedPosition = float3(0.0, 0.0, 0.0);
        selectedSurfaceHit = false;
        selectedSunVisible = false;
        selectedWeight = 0.0;
        reservoirWeightSum = 0.0;
    }

    record.candidatePosition = selectedPosition;
    record.primaryNormal = N;
    record.contribution = reservoirTotal;
    record.secondaryDirectSun = reservoirSecondarySun;
    record.weightSum = reservoirWeightSum;
    record.selectedWeight = selectedWeight;
    record.flags = (selectedWeight > 0.0 ? 1u : 0u) |
                   (selectedSurfaceHit ? 2u : 0u) |
                   (selectedSunVisible ? 4u : 0u) |
                   (useCandidateRis ? 8u : 0u) |
                   (reservoirGiMode >= PATH_TRACER_RESERVOIR_GI_TEMPORAL ? 16u : 0u);
    if (reservoirGiMode >= PATH_TRACER_RESERVOIR_GI_TEMPORAL) {
        updateReservoirGi(launchID, launchSize, record);
    }

    float reservoirProbeScale = float(candidateCount) / float(candidateCount + 1);
    result.totalContribution = reservoirTotal * reservoirProbeScale;
    result.secondaryDirectSunContribution = reservoirSecondarySun * reservoirProbeScale;
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
    float3 debugReservoirGiContribution = float3(0.0, 0.0, 0.0);
    float  debugReservoirGiAcceptedLuma = 0.0;
    float  debugReservoirGiCandidateSurfaceHitRatio = 0.0;
    float  debugReservoirGiCandidateSunVisibleRatio = 0.0;
    float  debugReservoirGiCandidatePositiveWeightRatio = 0.0;
    float  debugReservoirGiSelectedWeight = 0.0;
    float  debugShadowVisibility = 0.0;
    int    debugBounceCount = 0;
    uint   primaryInstanceID = 0;
    bool   primarySurfaceHit = false;
    float  pendingSkyMissMisWeight = 1.0;
    uint   packedPathTracerFlags = push.padding3;
    bool   environmentNeeEnabled = (packedPathTracerFlags & 0x1) != 0;
    bool   applyFirstHitProbesToFinal = (push.padding3 & PATH_TRACER_APPLY_FIRST_HIT_PROBES_BIT) != 0;
    int    environmentNeeSamplingMode = int((packedPathTracerFlags >> 3) & 0x3);
    int    firstHitDiffuseSampleCount = clamp(int((packedPathTracerFlags >> 5) & 0xF), 1, 8);
    int    candidateCount = int((packedPathTracerFlags >> 9) & 0xF) + 1;
    candidateCount = clamp(candidateCount, 2, 16);
    int    firstHitProbeSamplingMode = int((packedPathTracerFlags >> 13) & 0x7);
    uint   packedPathTracerMaterialIndex = uint(push.materialIndex);
    int    reservoirGiMode = int((packedPathTracerMaterialIndex >> 18) & 0x3u);
    int    reservoirGiCandidateCount = clamp(int((packedPathTracerMaterialIndex >> 20) & 0x3u) + 1, 1, 4);
    bool   reservoirGiUseCandidateRis = ((packedPathTracerMaterialIndex >> 22) & 0x1u) != 0u;
    int    reservoirGiSpatialNeighborCount = clamp(int((packedPathTracerMaterialIndex >> 23) & 0x7u) + 1, 1, 8);
    int    pathTracerMaxBounceCode = int((packedPathTracerMaterialIndex >> 26) & 0x3u);
    int    pathTracerMaxBounces = (pathTracerMaxBounceCode == 0) ? 3 :
                                  ((pathTracerMaxBounceCode == 1) ? 5 : 8);
    int    directSunBounceMode = int((packedPathTracerMaterialIndex >> 28) & 0x3u);
    int    reservoirGiCandidateEvaluationMode =
        clamp(int((packedPathTracerMaterialIndex >> 30) & 0x3u),
              RESERVOIR_GI_CANDIDATE_TRACE_ONLY,
              RESERVOIR_GI_CANDIDATE_SHADOWED_SUN);

    RayDesc ray;
    ray.Origin    = origin;
    ray.Direction = normalize(rayDir);
    ray.TMin      = 0.001;
    ray.TMax      = 10000.0;

    // Keep a higher hard cap for enclosed/interior transport; Russian roulette below
    // keeps the average path length bounded for performance.
    static const int RR_START_BOUNCE = 2;

    for (int bounce = 0; bounce < pathTracerMaxBounces; ++bounce)
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
                if (reservoirGiMode >= PATH_TRACER_RESERVOIR_GI_TEMPORAL) {
                    ReservoirGiRecord emptyReservoir;
                    emptyReservoir.primaryPosition = origin;
                    emptyReservoir.candidatePosition = origin;
                    emptyReservoir.primaryNormal = float3(0.0, 0.0, 0.0);
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
            bool firstHitBounceUsesReservoirGi = false;
            if (reservoirGiMode >= PATH_TRACER_RESERVOIR_GI_SINGLE_FRAME) {
                firstHitBounce = sampleFirstHitReservoirGiSingleFrame(payload.hitPos, N2, V2, payload,
                                                                      max(firstHitDiffuseSampleCount, 2),
                                                                      reservoirGiCandidateCount,
                                                                      reservoirGiUseCandidateRis,
                                                                      reservoirGiMode,
                                                                      reservoirGiSpatialNeighborCount,
                                                                      reservoirGiCandidateEvaluationMode,
                                                                      launchID, launchSize, rngState);
                firstHitBounceUsesReservoirGi = true;
            } else if (firstHitProbeSamplingMode == FIRST_HIT_PROBE_SAMPLING_CANDIDATE_SUN_BOUNCE) {
                firstHitBounce = sampleFirstHitCandidateSunBounce(payload.hitPos, N2, V2, payload,
                                                                  firstHitDiffuseSampleCount, candidateCount,
                                                                  rngState);
            } else if (firstHitProbeSamplingMode == FIRST_HIT_PROBE_SAMPLING_CANDIDATE_AVERAGE_REFERENCE) {
                firstHitBounce = sampleFirstHitCandidateAverageReference(payload.hitPos, N2, V2, payload,
                                                                         firstHitDiffuseSampleCount, candidateCount,
                                                                         rngState);
            } else if (firstHitProbeSamplingMode == FIRST_HIT_PROBE_SAMPLING_CANDIDATE_RIS) {
                firstHitBounce = sampleFirstHitCandidateRis(payload.hitPos, N2, V2, payload,
                                                            firstHitDiffuseSampleCount, candidateCount,
                                                            rngState);
            } else {
                firstHitBounce = sampleFirstHitDiffuseBounce(payload.hitPos, N2, V2, payload,
                                                            firstHitDiffuseSampleCount, firstHitProbeSamplingMode,
                                                            rngState);
            }
            if (applyFirstHitProbesToFinal) {
                radiance += firstHitBounce.totalContribution;
                debugIndirectLighting += firstHitBounce.totalContribution;
            }
            debugFirstHitBounceContribution += firstHitBounce.totalContribution;
            debugSecondaryDirectSunContribution += firstHitBounce.secondaryDirectSunContribution;
            if (firstHitBounceUsesReservoirGi) {
                debugReservoirGiContribution += firstHitBounce.totalContribution;
                debugReservoirGiAcceptedLuma += pathTracerLuminance(firstHitBounce.totalContribution);
                debugReservoirGiCandidateSurfaceHitRatio = firstHitBounce.reservoirCandidateSurfaceHitRatio;
                debugReservoirGiCandidateSunVisibleRatio = firstHitBounce.reservoirCandidateSunVisibleRatio;
                debugReservoirGiCandidatePositiveWeightRatio = firstHitBounce.reservoirCandidatePositiveWeightRatio;
                debugReservoirGiSelectedWeight = firstHitBounce.reservoirSelectedWeight;
            }
        }

        // ── NEE: direct lighting from directional sun light ────────────────
        // ubo.lightDir points FROM the light TOWARD the scene, so negate for surface-to-light.
        float3 Ldir  = normalize(-ubo.lightDir.xyz);
        float  NdotL = max(dot(N2, Ldir), 0.0);
        if (directSunEnabledForBounce(bounce, directSunBounceMode) && NdotL > 0.0)
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
        float bounceFraction = float(debugBounceCount) / float(pathTracerMaxBounces);
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
    } else if (push.padding2 == DEBUG_AOV_PATH_RESERVOIR_GI_CONTRIBUTION) {
        outputColor = debugReservoirGiContribution;
    } else if (push.padding2 == DEBUG_AOV_PATH_RESERVOIR_GI_ACCEPTED_LUMA) {
        outputColor = float3(debugReservoirGiAcceptedLuma, debugReservoirGiAcceptedLuma, debugReservoirGiAcceptedLuma);
    } else if (push.padding2 == DEBUG_AOV_PATH_RESERVOIR_GI_CANDIDATE_SURFACE_HIT) {
        outputColor = float3(debugReservoirGiCandidateSurfaceHitRatio,
                             debugReservoirGiCandidateSurfaceHitRatio,
                             debugReservoirGiCandidateSurfaceHitRatio);
    } else if (push.padding2 == DEBUG_AOV_PATH_RESERVOIR_GI_CANDIDATE_SUN_VISIBLE) {
        outputColor = float3(debugReservoirGiCandidateSunVisibleRatio,
                             debugReservoirGiCandidateSunVisibleRatio,
                             debugReservoirGiCandidateSunVisibleRatio);
    } else if (push.padding2 == DEBUG_AOV_PATH_RESERVOIR_GI_CANDIDATE_POSITIVE_WEIGHT) {
        outputColor = float3(debugReservoirGiCandidatePositiveWeightRatio,
                             debugReservoirGiCandidatePositiveWeightRatio,
                             debugReservoirGiCandidatePositiveWeightRatio);
    } else if (push.padding2 == DEBUG_AOV_PATH_RESERVOIR_GI_SELECTED_WEIGHT) {
        float selectedWeight = clamp(debugReservoirGiSelectedWeight, 0.0, 1.0);
        outputColor = float3(selectedWeight, selectedWeight, selectedWeight);
    }
    float3 appliedFirstHitProbeContribution =
        applyFirstHitProbesToFinal ? debugFirstHitBounceContribution : float3(0.0, 0.0, 0.0);
    recordTargetWallLuminance(radiance, appliedFirstHitProbeContribution, primaryInstanceID, primarySurfaceHit);
    noisyColorOutput[launchID] = float4(outputColor, 1.0);
}
