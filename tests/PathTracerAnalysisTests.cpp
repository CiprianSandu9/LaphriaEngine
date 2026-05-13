#include "PathTracerAnalysisTests.h"

#include "../src/Core/PathTracerAnalysis.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>

namespace
{
std::string readTextFile(const std::filesystem::path &path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return {};
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

bool containsText(const std::string &haystack, const char *needle)
{
    return haystack.find(needle) != std::string::npos;
}

uint64_t packConfigKey(const Laphria::PathTracerSweepConfig &cfg)
{
    const int scaled = static_cast<int>(std::lround(cfg.resolutionScale * 100.0f));
    return (static_cast<uint64_t>(scaled & 0xFF) << 24u) |
           (static_cast<uint64_t>(cfg.denoiserIterations & 0xFF) << 16u) |
           (static_cast<uint64_t>(cfg.enableReprojection ? 1 : 0) << 8u) |
           (static_cast<uint64_t>(cfg.enableMotionAwareAccumulation ? 1 : 0));
}
}

bool testPathTracerBaselineSweepMatrix()
{
    const auto matrix = Laphria::buildPathTracerBaselineSweepMatrix();
    if (matrix.size() != 9) {
        std::cerr << "baseline sweep size mismatch: expected 9 got " << matrix.size() << "\n";
        return false;
    }

    std::unordered_set<uint64_t> uniqueKeys;
    for (const auto &cfg : matrix) {
        uniqueKeys.insert(packConfigKey(cfg));
    }
    if (uniqueKeys.size() != matrix.size()) {
        std::cerr << "baseline sweep has duplicate configurations\n";
        return false;
    }

    for (const auto &cfg : matrix) {
        if (!cfg.enableReprojection && cfg.enableMotionAwareAccumulation) {
            std::cerr << "invalid sweep config: motion-aware enabled while reprojection disabled\n";
            return false;
        }
    }
    return true;
}

bool testPathTracerPercentiles()
{
    const std::vector<float> values = {10.0f, 40.0f, 20.0f, 30.0f, 50.0f};
    const auto pct = Laphria::computePercentiles(values);
    if (std::abs(pct.p50 - 30.0f) > 0.0001f) {
        std::cerr << "p50 mismatch\n";
        return false;
    }
    if (std::abs(pct.p95 - 50.0f) > 0.0001f) {
        std::cerr << "p95 mismatch\n";
        return false;
    }
    if (std::abs(pct.p99 - 50.0f) > 0.0001f) {
        std::cerr << "p99 mismatch\n";
        return false;
    }
    return true;
}

bool testPathTracerScoreBudgetGate()
{
    Laphria::PathTracerScoreInput input{};
    input.totalFrameMsP95 = 18.0f;
    input.targetBudgetMs = 16.67f;
    input.historyRejectionRatio = 0.30f;
    input.skyHitRatio = 0.05f;
    input.fireflyClampRatio = 0.02f;
    input.visualFidelityScore = 0.85f;

    const auto score = Laphria::scorePathTracerRun(input);
    if (score.budgetPass) {
        std::cerr << "budget gate should fail when p95 exceeds target\n";
        return false;
    }
    if (score.compositeScore >= 0.80f) {
        std::cerr << "composite score unexpectedly high for budget fail case\n";
        return false;
    }
    return true;
}

bool testPathTracerHistoryClampPreservesDimIndirectHistory()
{
    Laphria::PathTracerHistoryClampInput input{};
    input.neighborhoodMinLum = 0.0f;
    input.neighborhoodMaxLum = 0.002f;
    input.historyLum = 0.035f;
    input.previousMeanLum = 0.032f;
    input.previousVariance = 0.000004f;

    const auto result = Laphria::computePathTracerHistoryClamp(input);
    if (result.clampedHistoryLum < 0.030f) {
        std::cerr << "history clamp crushed dim indirect history: "
                  << result.clampedHistoryLum << "\n";
        return false;
    }

    Laphria::PathTracerHistoryClampInput darkSpike{};
    darkSpike.neighborhoodMinLum = 0.0f;
    darkSpike.neighborhoodMaxLum = 0.002f;
    darkSpike.historyLum = 1.0f;
    darkSpike.previousMeanLum = 0.032f;
    darkSpike.previousVariance = 0.000004f;

    const auto spikeResult = Laphria::computePathTracerHistoryClamp(darkSpike);
    if (spikeResult.clampedHistoryLum > 0.050f) {
        std::cerr << "history clamp preserved an excessive dark-region spike: "
                  << spikeResult.clampedHistoryLum << "\n";
        return false;
    }

    return true;
}

bool testPathTracerPowerHeuristic()
{
    const float equal = Laphria::computePowerHeuristic(1.0f, 0.5f, 1.0f, 0.5f);
    if (std::abs(equal - 0.5f) > 0.0001f) {
        std::cerr << "equal MIS PDFs should produce 0.5 weight, got " << equal << "\n";
        return false;
    }

    const float dominant = Laphria::computePowerHeuristic(1.0f, 0.8f, 1.0f, 0.2f);
    if (std::abs(dominant - 0.9411765f) > 0.0001f) {
        std::cerr << "dominant MIS PDF mismatch: " << dominant << "\n";
        return false;
    }

    const float zeroOther = Laphria::computePowerHeuristic(1.0f, 0.5f, 1.0f, 0.0f);
    if (std::abs(zeroOther - 1.0f) > 0.0001f) {
        std::cerr << "zero competing PDF should produce full weight, got " << zeroOther << "\n";
        return false;
    }

    return true;
}

bool testPathTracerDebugAovContract()
{
    const std::filesystem::path sourceRoot =
#ifdef LAPHRIA_SOURCE_DIR
        std::filesystem::path(LAPHRIA_SOURCE_DIR);
#else
        std::filesystem::current_path();
#endif

    const std::string uiHeader = readTextFile(sourceRoot / "src" / "Core" / "UISystem.h");
    const std::string uiSource = readTextFile(sourceRoot / "src" / "Core" / "UISystem.cpp");
    const std::string engineHeader = readTextFile(sourceRoot / "src" / "Core" / "EngineCore.h");
    const std::string engineSource = readTextFile(sourceRoot / "src" / "Core" / "EngineCore.cpp");
    const std::string frameContextHeader = readTextFile(sourceRoot / "src" / "Core" / "FrameContext.h");
    const std::string frameContextSource = readTextFile(sourceRoot / "src" / "Core" / "FrameContext.cpp");
    const std::string raygen = readTextFile(sourceRoot / "src" / "shaders" / "Raygen.slang");
    const std::string miss = readTextFile(sourceRoot / "src" / "shaders" / "Miss.slang");
    const std::string denoiser = readTextFile(sourceRoot / "src" / "shaders" / "Denoiser.slang");
    const std::string anyHit = readTextFile(sourceRoot / "src" / "shaders" / "AnyHit.slang");

    if (uiHeader.empty() || uiSource.empty() || engineHeader.empty() || engineSource.empty() ||
        frameContextHeader.empty() || frameContextSource.empty() ||
        raygen.empty() || miss.empty() || denoiser.empty() || anyHit.empty()) {
        std::cerr << "failed to load path tracer debug AOV contract files\n";
        return false;
    }
    if (!containsText(uiHeader, "warmupFrames = 30") || !containsText(uiHeader, "sampleFrames = 120")) {
        std::cerr << "GI cache sweep defaults should stay short enough for interactive runs\n";
        return false;
    }
    if (!containsText(uiHeader, "cacheReuseWeight = 0.126f")) {
        std::cerr << "cached secondary reuse debug default should match the calibrated hard-bounce weight\n";
        return false;
    }

    const char *requiredLabels[] = {
        "Raw Final Color",
        "Direct Lighting",
        "Indirect Lighting",
        "Sky Contribution",
        "Throughput",
        "Bounce Count",
        "Shadow Visibility",
        "Environment NEE Contribution",
        "First-Hit Bounce Contribution",
        "Secondary Direct Sun Contribution",
        "Baseline Continuation Contribution",
        "Environment NEE",
        "First-Hit Diffuse Samples",
        "Env NEE Sampling",
        "Black Environment",
        "Apply First-Hit Probes",
        "Path Tracer Debug Lab",
        "PT Benchmark",
        "PT Stats",
        "Target Wall Avg Luma",
        "First-Hit Probe Rays",
        "First-Hit Probe Surface Hits",
        "First-Hit Probe Sun Visible",
        "First-Hit Probe Avg Luma",
        "First-Hit Probe Sun-Visible Avg Luma",
        "First-Hit Probe Sampling",
        "Naive Sun Guide",
        "Candidate Sun Bounce",
        "Candidate Average Reference",
        "Candidate RIS",
        "Cached Secondary Reuse",
        "First-Hit Candidate Count",
        "Cache Reuse Weight",
        "Run GI Cache Weight Sweep",
        "Capture Checklist",
        "Record these values after the image stabilizes",
        "Sun-Visible Cache",
        "Enable Sun-Visible Cache",
        "Clear Sun-Visible Cache",
        "Cache Refresh",
        "Adaptive Cache Refresh",
        "Cache Weighting",
        "Cache MIS Strength",
        "Calibrated Weight",
        "MIS Approximation",
        "Cache Inserts This Frame",
        "Resident Cached Candidates",
        "Cache Reuse Attempts",
        "Cache Reuse Accepted",
        "Cache Reuse Avg Luma",
        "Target Wall Base Luma",
        "Target Wall Probe Added Luma",
        "Diagnostic Target Cache Reuse Attempts",
        "Diagnostic Target Cache Selected",
        "Diagnostic Target Cache Reject Distance",
        "Diagnostic Target Cache Reject Geometry",
        "Diagnostic Target Cache Reject Visibility",
        "Diagnostic Target Cache Reuse Accepted",
        "Diagnostic Target Cache Reuse Avg Luma",
        "Cache Refresh Attempts",
        "Cache Refresh Inserts",
        "Cache Stale Rejects",
        "Cache MIS Avg Weight",
        "Light Preset",
        "Hard Bounce",
        "Medium Bounce",
        "Easy Bounce",
        "Apply Light Preset",
        "Load Indirect Bounce Test Scene"};
    for (const char *label : requiredLabels) {
        if (!containsText(uiSource, label)) {
            std::cerr << "missing path tracer debug AOV UI label: " << label << "\n";
            return false;
        }
    }

    const char *requiredEnumValues[] = {
        "PathRawFinalColor",
        "PathDirectLighting",
        "PathIndirectLighting",
        "PathSkyContribution",
        "PathThroughput",
        "PathBounceCount",
        "PathShadowVisibility",
        "PathEnvironmentNeeContribution",
        "PathFirstHitBounceContribution",
        "PathSecondaryDirectSunContribution",
        "PathBaselineContinuationContribution"};
    for (const char *enumValue : requiredEnumValues) {
        if (!containsText(uiHeader, enumValue)) {
            std::cerr << "missing path tracer debug AOV enum value: " << enumValue << "\n";
            return false;
        }
    }

    const char *requiredShaderSymbols[] = {
        "debugDirectLighting",
        "debugIndirectLighting",
        "debugSkyContribution",
        "debugThroughput",
        "debugBounceCount",
        "debugShadowVisibility",
        "sampleEnvironmentNEE",
        "debugEnvironmentNeeContribution",
        "powerHeuristic",
        "environmentSamplePdf",
        "bsdfPdfForEnvironmentDirection",
        "sampleSkyBiasedEnvironmentDirection",
        "environmentNeeEnabledForBounce",
        "ENV_NEE_SAMPLING_SKY_BIASED",
        "firstHitDiffuseSampleCount",
        "firstHitProbeSamplingMode",
        "sampleFirstHitDiffuseBounce",
        "sampleFirstHitProbeDirection",
        "debugFirstHitBounceContribution",
        "debugSecondaryDirectSunContribution",
        "debugBaselineContinuationContribution",
        "recordTargetWallLuminance",
        "recordFirstHitProbeStats",
        "FIRST_HIT_PROBE_SAMPLING_SUN_BOUNCE_GUIDED",
        "FIRST_HIT_PROBE_SAMPLING_CANDIDATE_SUN_BOUNCE",
        "FIRST_HIT_PROBE_SAMPLING_CANDIDATE_AVERAGE_REFERENCE",
        "FIRST_HIT_PROBE_SAMPLING_CANDIDATE_RIS",
        "sampleFirstHitCandidateSunBounce",
        "sampleFirstHitCandidateAverageReference",
        "sampleFirstHitCandidateRis",
        "ptSunVisibleCandidateCache",
        "SUN_VISIBLE_CANDIDATE_CACHE_BINDING",
        "SUN_VISIBLE_CANDIDATE_RECORD_SIZE",
        "SUN_VISIBLE_CANDIDATE_CACHE_BANK_COUNT",
        "SUN_VISIBLE_CANDIDATE_CACHE_BANK_CAPACITY",
        "SUN_VISIBLE_CANDIDATE_CACHE_FRAME_SLOT_COUNT",
        "currentFrame / SUN_VISIBLE_CANDIDATE_CACHE_FRAME_SLOT_COUNT",
        "cacheReadBank",
        "cacheWriteBank",
        "sunVisibleCandidateBankInsertCountOffset",
        "CACHED_REUSE_SPATIAL_CELL_SIZE",
        "CACHED_REUSE_SPATIAL_BUCKET_SIZE",
        "CACHED_REUSE_SPATIAL_BUCKET_SIZE = 128u",
        "CACHED_REUSE_SPATIAL_CANDIDATE_TRIALS = 32",
        "CACHED_REUSE_GLOBAL_FALLBACK_TRIALS",
        "candidateWithinConnectionRadius",
        "candidateGeometryCompatible",
        "considerCachedSunVisibleCandidate",
        "spatialCacheCell",
        "hashSpatialCacheCell",
        "sunVisibleCandidateSpatialBankSlot",
        "selectSpatialCachedSunVisibleCandidate",
        "CachedSunVisibleCandidate",
        "cacheRecordFrameId",
        "cacheRecordGeneration",
        "cacheRecordSourcePdf",
        "storeSunVisibleCandidateMaterial",
        "makeCachedSunVisiblePayload",
        "isSaneCachedSunVisibleCandidate",
        "evaluateCachedSunVisibleContribution",
        "rawDistanceSq > 0.000001",
        "visibilityRayTMax <= visibilityRayTMin",
        "invalidateSunVisibleCacheSlotForWrite",
        "recordSunVisibleCandidate",
        "refreshSunVisibleCandidateCache",
        "loadSunVisibleCandidate",
        "selectSpatialCachedSunVisibleCandidate",
        "sampleCachedSunVisibleCandidate",
        "SUN_VISIBLE_CACHE_SLOT_READY",
        "cacheRecordSlotState",
        "CACHED_REUSE_SPATIAL_CANDIDATE_TRIALS",
        "bestDistanceSq",
        "CACHED_REUSE_MAX_CONNECTION_DISTANCE",
        "CACHED_REUSE_MAX_CONNECTION_DISTANCE * CACHED_REUSE_MAX_CONNECTION_DISTANCE",
        "CACHE_WEIGHTING_CALIBRATED",
        "CACHE_WEIGHTING_MIS_APPROX",
        "computeCachedReuseMisWeight",
        "computeConservativeCachedReuseMisWeight",
        "cacheMisStrength",
        "CACHED_REUSE_MAX_RECORD_AGE_FRAMES",
        "targetWallFirstHitProbeContributionSum",
        "targetWallBaseLuminanceSum",
        "diagnosticTargetCacheReuseAttemptCount",
        "diagnosticTargetCacheSelectedCount",
        "diagnosticTargetCacheRejectDistanceCount",
        "diagnosticTargetCacheRejectGeometryCount",
        "diagnosticTargetCacheRejectVisibilityCount",
        "diagnosticTargetCacheReuseAcceptedCount",
        "diagnosticTargetCacheReuseContributionSum",
        "CachedReuseRejectReason",
        "CACHED_REUSE_REJECT_DISTANCE",
        "CACHED_REUSE_REJECT_GEOMETRY",
        "CACHED_REUSE_REJECT_VISIBILITY",
        "currentHitIsDiagnosticTarget",
        "recordTargetWallLuminance(radiance",
        "RAY_FLAG_FORCE_OPAQUE",
        "cacheReuseWeight",
        "FIRST_HIT_PROBE_SAMPLING_CACHED_SECONDARY_REUSE",
        "candidateCount",
        "risWeightSum",
        "selectedCandidateProbability",
        "weightedBestTotal",
        "recordFirstHitProbeStats(bestSurfaceHit, bestSunVisible, weightedBestTotal)",
        "PATH_TRACER_BLACK_ENVIRONMENT_BIT",
        "PATH_TRACER_APPLY_FIRST_HIT_PROBES_BIT",
        "applyFirstHitProbesToFinal",
        "blackEnvironmentEnabled"};
    for (const char *symbol : requiredShaderSymbols) {
        if (!containsText(raygen, symbol) && !containsText(miss, symbol) && !containsText(denoiser, symbol)) {
            std::cerr << "missing path tracer debug AOV shader symbol: " << symbol << "\n";
            return false;
        }
    }
    if (!containsText(anyHit, "float  ao;")) {
        std::cerr << "path tracer AnyHit RayPayload layout must include ao before hitT\n";
        return false;
    }

    const char *requiredEngineSymbols[] = {
        "blackEnvironment",
        "applyFirstHitProbesToFinal",
        "targetWallLuminanceAverage",
        "targetWallBaseLuminanceAverage",
        "targetWallFirstHitProbeContributionAverage",
        "diagnosticTargetCacheReuseAttemptCount",
        "diagnosticTargetCacheSelectedCount",
        "diagnosticTargetCacheRejectDistanceCount",
        "diagnosticTargetCacheRejectGeometryCount",
        "diagnosticTargetCacheRejectVisibilityCount",
        "diagnosticTargetCacheReuseAcceptedCount",
        "diagnosticTargetCacheReuseAcceptedRatio",
        "diagnosticTargetCacheReuseContributionAverage",
        "targetWallSampleCount",
        "firstHitProbeCount",
        "firstHitProbeSurfaceHitCount",
        "firstHitProbeSunVisibleCount",
        "firstHitProbeContributionAverage",
        "firstHitProbeSunVisibleContributionAverage",
        "firstHitProbeSamplingMode",
        "enableSunVisibleCandidateCache",
        "cacheReuseWeight",
        "PathTracerCacheWeightingMode",
        "adaptiveCacheRefresh",
        "cacheRefreshCandidateCount",
        "cacheSpatialCandidateTrials",
        "cacheMisStrength",
        "sunVisibleCacheGeneration",
        "PathTracerExperimentRow",
        "clearCacheBeforeRow",
        "enableSunVisibleCandidateCache = row.enableSunVisibleCandidateCache",
        "startPathTracerGiCacheWeightSweep",
        "Hard / Candidate RIS Reference",
        "Hard / Cached Reuse Trials 8",
        "Hard / Cached Reuse Trials 12",
        "Hard / Cached Reuse Trials 16",
        "Medium / Candidate RIS Reference",
        "Medium / Cached Reuse Trials 8",
        "Medium / Cached Reuse Trials 12",
        "Medium / Cached Reuse Trials 16",
        "Easy / Candidate RIS Reference",
        "Easy / Cached Reuse Trials 8",
        "Easy / Cached Reuse Trials 12",
        "Easy / Cached Reuse Trials 16",
        "experiment row reset",
        "updatePathTracerExperimentSweep",
        "logPathTracerExperimentRow",
        "clearPathTracerExperimentCache",
        "clearSunVisibleCandidateCache",
        "clearSunVisibleCacheRequested",
        "cacheRefreshAttemptCount",
        "cacheRefreshInsertCount",
        "cacheStaleRejectCount",
        "cacheMisWeightAverage",
        "targetWallBaseLuma",
        "targetWallProbeAddedLuma",
        "targetWallFinalLuma",
        "spatialTrials",
        "diagnosticTargetCacheReuseAttempts",
        "diagnosticTargetCacheSelected",
        "diagnosticTargetCacheRejectDistance",
        "diagnosticTargetCacheRejectGeometry",
        "diagnosticTargetCacheRejectVisibility",
        "diagnosticTargetCacheReuseAccepted",
        "diagnosticTargetCacheReuseAvgLuma",
        "updateSunVisibleCandidateCacheInvalidation",
        "lastSunVisibleCacheLightDirection",
        "vulkan.logicalDevice.waitIdle()",
        "debugBreakIfDebuggerAttached",
        "waitForFenceOrThrow",
        "vk::SystemError",
        "Vulkan device lost while waiting for",
        "pathTracerStorageBufferBarrier",
        "vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite",
        "sunVisibleCandidateCacheBuffers",
        "sunVisibleCandidateCacheMapped",
        "kSunVisibleCandidateCacheCapacity",
        "cachedSecondaryReuse",
        "PathTracerDebugLightPreset",
        "applyPathTracerDebugLightPreset",
        "pathTracerSettings.enableEnvironmentNEE = true",
        "pathTracerSettings.blackEnvironment = true",
        "pathTracerSettings.applyFirstHitProbesToFinal = true",
        "pathTracerSettings.enableSunVisibleCandidateCache = true",
        "pathTracerSettings.adaptiveCacheRefresh = true",
        "pathTracerSettings.environmentNeeSamplingMode = EnvironmentNeeSamplingMode::SkyBiased",
        "pathTracerSettings.firstHitProbeSamplingMode = FirstHitProbeSamplingMode::CachedSecondaryReuse",
        "pathTracerSettings.firstHitDiffuseSamples = 8",
        "pathTracerSettings.firstHitCandidateCount = 4",
        "pathTracerSettings.cacheRefreshCandidateCount = 1",
        "pathTracerSettings.cacheSpatialCandidateTrials = 8",
        "pathTracerSettings.cacheMaxRecordAgeFrames = 180",
        "pathTracerSettings.cacheWeightingMode = PathTracerCacheWeightingMode::CalibratedWeight",
        "pathTracerSettings.cacheMisStrength = 1.5f",
        "pathTracerAnalysisSettings.debugLightPreset = PathTracerDebugLightPreset::HardBounce",
        "pathTracerAnalysisSettings.applyDebugLightPreset = true",
        "pathTracerAnalysisSettings.debugAov = PathTracerDebugAov::PathRawFinalColor",
        "ptIndirectBounceTargetWallModelId",
        "loadIndirectBounceTestScene",
        "loadPathTracerIndirectBounceTestSceneIfRequested",
        "PT_IndirectBounce_Floor",
        "PT_IndirectBounce_Wall",
        "PT_IndirectBounce_Blocker",
        "PT_IndirectBounce_Ceiling",
        "PT_IndirectBounce_LeftWall",
        "PT_IndirectBounce_RightWall"};
    for (const char *symbol : requiredEngineSymbols) {
        if (!containsText(uiHeader, symbol) && !containsText(uiSource, symbol) &&
            !containsText(engineHeader, symbol) &&
            !containsText(engineSource, symbol) && !containsText(frameContextHeader, symbol) &&
            !containsText(frameContextSource, symbol)) {
            std::cerr << "missing path tracer indirect bounce scene symbol: " << symbol << "\n";
            return false;
        }
    }

    return true;
}
