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
	if (!file)
	{
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
}        // namespace

bool testPathTracerBaselineSweepMatrix()
{
	const auto matrix = Laphria::buildPathTracerBaselineSweepMatrix();
	if (matrix.size() != 9)
	{
		std::cerr << "baseline sweep size mismatch: expected 9 got " << matrix.size() << "\n";
		return false;
	}

	std::unordered_set<uint64_t> uniqueKeys;
	for (const auto &cfg : matrix)
	{
		uniqueKeys.insert(packConfigKey(cfg));
	}
	if (uniqueKeys.size() != matrix.size())
	{
		std::cerr << "baseline sweep has duplicate configurations\n";
		return false;
	}

	for (const auto &cfg : matrix)
	{
		if (!cfg.enableReprojection && cfg.enableMotionAwareAccumulation)
		{
			std::cerr << "invalid sweep config: motion-aware enabled while reprojection disabled\n";
			return false;
		}
	}
	return true;
}

bool testPathTracerPercentiles()
{
	const std::vector<float> values = {10.0f, 40.0f, 20.0f, 30.0f, 50.0f};
	const auto               pct    = Laphria::computePercentiles(values);
	if (std::abs(pct.p50 - 30.0f) > 0.0001f)
	{
		std::cerr << "p50 mismatch\n";
		return false;
	}
	if (std::abs(pct.p95 - 50.0f) > 0.0001f)
	{
		std::cerr << "p95 mismatch\n";
		return false;
	}
	if (std::abs(pct.p99 - 50.0f) > 0.0001f)
	{
		std::cerr << "p99 mismatch\n";
		return false;
	}
	return true;
}

bool testPathTracerScoreBudgetGate()
{
	Laphria::PathTracerScoreInput input{};
	input.totalFrameMsP95       = 18.0f;
	input.targetBudgetMs        = 16.67f;
	input.historyRejectionRatio = 0.30f;
	input.skyHitRatio           = 0.05f;
	input.fireflyClampRatio     = 0.02f;
	input.visualFidelityScore   = 0.85f;

	const auto score = Laphria::scorePathTracerRun(input);
	if (score.budgetPass)
	{
		std::cerr << "budget gate should fail when p95 exceeds target\n";
		return false;
	}
	if (score.compositeScore >= 0.80f)
	{
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
	input.historyLum         = 0.035f;
	input.previousMeanLum    = 0.032f;
	input.previousVariance   = 0.000004f;

	const auto result = Laphria::computePathTracerHistoryClamp(input);
	if (result.clampedHistoryLum < 0.030f)
	{
		std::cerr << "history clamp crushed dim indirect history: "
		          << result.clampedHistoryLum << "\n";
		return false;
	}

	Laphria::PathTracerHistoryClampInput darkSpike{};
	darkSpike.neighborhoodMinLum = 0.0f;
	darkSpike.neighborhoodMaxLum = 0.002f;
	darkSpike.historyLum         = 1.0f;
	darkSpike.previousMeanLum    = 0.032f;
	darkSpike.previousVariance   = 0.000004f;

	const auto spikeResult = Laphria::computePathTracerHistoryClamp(darkSpike);
	if (spikeResult.clampedHistoryLum > 0.050f)
	{
		std::cerr << "history clamp preserved an excessive dark-region spike: "
		          << spikeResult.clampedHistoryLum << "\n";
		return false;
	}

	return true;
}

bool testPathTracerPowerHeuristic()
{
	const float equal = Laphria::computePowerHeuristic(1.0f, 0.5f, 1.0f, 0.5f);
	if (std::abs(equal - 0.5f) > 0.0001f)
	{
		std::cerr << "equal MIS PDFs should produce 0.5 weight, got " << equal << "\n";
		return false;
	}

	const float dominant = Laphria::computePowerHeuristic(1.0f, 0.8f, 1.0f, 0.2f);
	if (std::abs(dominant - 0.9411765f) > 0.0001f)
	{
		std::cerr << "dominant MIS PDF mismatch: " << dominant << "\n";
		return false;
	}

	const float zeroOther = Laphria::computePowerHeuristic(1.0f, 0.5f, 1.0f, 0.0f);
	if (std::abs(zeroOther - 1.0f) > 0.0001f)
	{
		std::cerr << "zero competing PDF should produce full weight, got " << zeroOther << "\n";
		return false;
	}

	return true;
}

bool testPathTracerReservoirGiMeasurementContract()
{
	const std::filesystem::path sourceRoot =
#ifdef LAPHRIA_SOURCE_DIR
	    std::filesystem::path(LAPHRIA_SOURCE_DIR);
#else
	    std::filesystem::current_path();
#endif

	const std::string raygen          = readTextFile(sourceRoot / "src" / "shaders" / "Raygen.slang");
	const std::string engineCore      = readTextFile(sourceRoot / "src" / "Core" / "EngineCore.cpp");
	const std::string resourceManager = readTextFile(sourceRoot / "src" / "Core" / "ResourceManager.cpp");
	const std::string gltfImporter    = readTextFile(sourceRoot / "src" / "Core" / "GltfImporter.cpp");

	if (raygen.empty() || engineCore.empty() || resourceManager.empty() || gltfImporter.empty())
	{
		std::cerr << "failed to read reservoir GI measurement contract sources\n";
		return false;
	}

	const char *requiredRaygenSymbols[] = {
	    "float reservoirProbeScale = float(candidateCount) / float(candidateCount + 1)",
	    "result.totalContribution = reservoirTotal * reservoirProbeScale",
	    "result.secondaryDirectSunContribution = reservoirSecondarySun * reservoirProbeScale",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiCandidatesOffset, 1u)",
	    "primaryNormal",
	    "previousPrimaryNormal = normalize(record.primaryNormal)",
	    "record.primaryNormal = N"};
	for (const char *symbol : requiredRaygenSymbols)
	{
		if (!containsText(raygen, symbol))
		{
			std::cerr << "missing reservoir GI measurement shader contract: " << symbol << "\n";
			return false;
		}
	}

	if (containsText(raygen, "candidateNormal"))
	{
		std::cerr << "reservoir GI record still uses candidateNormal for primary-normal validation\n";
		return false;
	}

	const char *requiredEngineSymbols[] = {
	    "reservoirGiCandidateRays=%.1f",
	    "PT Experiment Row Summary:",
	    "ptExperimentCompletionLog",
	    "PT Experiment Sweep: Sponza GI perf sweep complete"};
	for (const char *symbol : requiredEngineSymbols)
	{
		if (!containsText(engineCore, symbol))
		{
			std::cerr << "missing reservoir GI measurement engine contract: " << symbol << "\n";
			return false;
		}
	}

	const char *removedNoisyEngineSymbols[] = {
	    "cacheReuseAcceptedDistanceBuckets=",
	    "diagnosticTargetCacheRejectGeometry=%.1f",
	    "Sun-Visible Cache: cleared (%s)"};
	for (const char *symbol : removedNoisyEngineSymbols)
	{
		if (containsText(engineCore, symbol))
		{
			std::cerr << "path tracer console output still contains noisy symbol: " << symbol << "\n";
			return false;
		}
	}

	const char *removedNoisyAssetLogSymbols[] = {
	    "GLTF parse: reading file bytes",
	    "GLTF parse: starting document parse",
	    "Texture upload progress:",
	    "Loading texture from URI:",
	    "Loading embedded texture",
	    "Texture path[%zu]",
	    "Texture decode progress:",
	    "Texture decode path summary:",
	    "Texture color-space summary:"};
	for (const char *symbol : removedNoisyAssetLogSymbols)
	{
		if (containsText(resourceManager, symbol) || containsText(gltfImporter, symbol))
		{
			std::cerr << "asset loading console output still contains noisy symbol: " << symbol << "\n";
			return false;
		}
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

	const std::string uiHeader              = readTextFile(sourceRoot / "src" / "Core" / "UISystem.h");
	const std::string uiSource              = readTextFile(sourceRoot / "src" / "Core" / "UISystem.cpp");
	const std::string engineAuxiliaryHeader = readTextFile(sourceRoot / "src" / "Core" / "EngineAuxiliary.h");
	const std::string engineHeader          = readTextFile(sourceRoot / "src" / "Core" / "EngineCore.h");
	const std::string engineSource          = readTextFile(sourceRoot / "src" / "Core" / "EngineCore.cpp");
	const std::string frameContextHeader    = readTextFile(sourceRoot / "src" / "Core" / "FrameContext.h");
	const std::string frameContextSource    = readTextFile(sourceRoot / "src" / "Core" / "FrameContext.cpp");
	const std::string pipelineCollection    = readTextFile(sourceRoot / "src" / "Core" / "PipelineCollection.cpp");
	const std::string raygen                = readTextFile(sourceRoot / "src" / "shaders" / "Raygen.slang");
	const std::string miss                  = readTextFile(sourceRoot / "src" / "shaders" / "Miss.slang");
	const std::string denoiser              = readTextFile(sourceRoot / "src" / "shaders" / "Denoiser.slang");
	const std::string anyHit                = readTextFile(sourceRoot / "src" / "shaders" / "AnyHit.slang");

	if (uiHeader.empty() || uiSource.empty() || engineAuxiliaryHeader.empty() ||
	    engineHeader.empty() || engineSource.empty() ||
	    frameContextHeader.empty() || frameContextSource.empty() || pipelineCollection.empty() ||
	    raygen.empty() || miss.empty() || denoiser.empty() || anyHit.empty())
	{
		std::cerr << "failed to load path tracer debug AOV contract files\n";
		return false;
	}
	if (!containsText(uiHeader, "warmupFrames = 30") || !containsText(uiHeader, "sampleFrames = 120"))
	{
		std::cerr << "GI cache sweep defaults should stay short enough for interactive runs\n";
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
	    "Reservoir GI Contribution",
	    "Reservoir GI Accepted Luma",
	    "Environment NEE",
	    "First-Hit Diffuse Samples",
	    "Env NEE Sampling",
	    "Black Environment",
	    "Apply First-Hit Probes",
	    "Path Tracer Diagnostics",
	    "Core Diagnostics",
	    "Core Metrics",
	    "Scenario: Indirect Bounce Box",
	    "Scenario: Sponza GI Validation",
	    "Benchmark Automation",
	    "Frame Stats",
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
	    "First-Hit Candidate Count",
	    "Reservoir GI",
	    "Reservoir GI Proposal",
	    "Sun Guided",
	    "Mixed Cosine + Sun Guided",
	    "Light Region Guided",
	    "Reservoir Spatial Neighbors",
	    "Reservoir Candidate Surface Hits",
	    "Reservoir Candidate Sun Visible",
	    "Reservoir Candidate Positive Weight",
	    "Reservoir GI Zero Weight",
	    "Reservoir GI Selected Weight Avg",
	    "Indirect Box Capture Checklist",
	    "Record these values after the image stabilizes",
	    "Target Wall Base Luma",
	    "Target Wall Probe Added Luma",
	    "Light Preset",
	    "Hard Bounce",
	    "Medium Bounce",
	    "Easy Bounce",
	    "Apply Light Preset",
	    "Load Indirect Bounce Test Scene",
	    "Load Sponza GI Validation Preset",
	    "Sponza Validation View",
	    "Apply Sponza Validation View",
	    "Run Sponza GI Perf Sweep"};
	for (const char *label : requiredLabels)
	{
		if (!containsText(uiSource, label))
		{
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
	    "PathBaselineContinuationContribution",
	    "PathReservoirGiContribution",
	    "PathReservoirGiAcceptedLuma",
	    "PathReservoirGiCandidateSurfaceHit",
	    "PathReservoirGiCandidateSunVisible",
	    "PathReservoirGiCandidatePositiveWeight",
	    "PathReservoirGiSelectedWeight",
	    "PathTracerSponzaValidationView"};
	for (const char *enumValue : requiredEnumValues)
	{
		if (!containsText(uiHeader, enumValue))
		{
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
	    "debugReservoirGiContribution",
	    "debugReservoirGiAcceptedLuma",
	    "debugReservoirGiCandidateSurfaceHitRatio",
	    "debugReservoirGiCandidateSunVisibleRatio",
	    "debugReservoirGiCandidatePositiveWeightRatio",
	    "debugReservoirGiSelectedWeight",
	    "DEBUG_AOV_PATH_RESERVOIR_GI_CONTRIBUTION",
	    "DEBUG_AOV_PATH_RESERVOIR_GI_ACCEPTED_LUMA",
	    "DEBUG_AOV_PATH_RESERVOIR_GI_CANDIDATE_SURFACE_HIT",
	    "DEBUG_AOV_PATH_RESERVOIR_GI_CANDIDATE_SUN_VISIBLE",
	    "DEBUG_AOV_PATH_RESERVOIR_GI_CANDIDATE_POSITIVE_WEIGHT",
	    "DEBUG_AOV_PATH_RESERVOIR_GI_SELECTED_WEIGHT",
	    "recordTargetWallLuminance",
	    "recordFirstHitProbeStats",
	    "FIRST_HIT_PROBE_SAMPLING_SUN_BOUNCE_GUIDED",
	    "FIRST_HIT_PROBE_SAMPLING_CANDIDATE_SUN_BOUNCE",
	    "FIRST_HIT_PROBE_SAMPLING_CANDIDATE_AVERAGE_REFERENCE",
	    "FIRST_HIT_PROBE_SAMPLING_CANDIDATE_RIS",
	    "sampleFirstHitCandidateSunBounce",
	    "sampleFirstHitCandidateAverageReference",
	    "sampleFirstHitCandidateRis",
	    "RESERVOIR_GI_PROPOSAL_COSINE",
	    "RESERVOIR_GI_PROPOSAL_SUN_GUIDED",
	    "RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_GUIDED",
	    "RESERVOIR_GI_PROPOSAL_LIGHT_REGION_GUIDED",
	    "sampleReservoirGiProposalDirection",
	    "sampleReservoirGiLightRegionDirection",
	    "uniformSampleCone",
	    "lightRegionTargetRadius",
	    "reservoirGiProposalPdf",
	    "reservoirGiProposalMode",
	    "ptReservoirGiCurrent",
	    "ptReservoirGiHistory",
	    "RESERVOIR_GI_CURRENT_BINDING",
	    "RESERVOIR_GI_HISTORY_BINDING",
	    "RESERVOIR_GI_CURRENT_CAPACITY",
	    "launchID.x == 0u && launchID.y == 0u",
	    "pixelIndex >= RESERVOIR_GI_CURRENT_CAPACITY",
	    "historyCapacity = ptReservoirGiHistory.Load(0)",
	    "historyWidth = ptReservoirGiHistory.Load(4)",
	    "historyHeight = ptReservoirGiHistory.Load(8)",
	    "historyWidth != launchSize.x",
	    "pixelIndex >= historyCapacity",
	    "ReservoirGiRecord",
	    "updateReservoirGi",
	    "loadTemporalReservoirGi",
	    "combineSpatialReservoirGi",
	    "isSaneReservoirGiVector",
	    "sanitizeReservoirGiContribution",
	    "targetWallFirstHitProbeContributionSum",
	    "targetWallBaseLuminanceSum",
	    "recordTargetWallLuminance(radiance",
	    "candidateCount",
	    "reservoirGiCandidates",
	    "reservoirGiAccepted",
	    "reservoirGiCandidateSurfaceHits",
	    "reservoirGiCandidateSunVisible",
	    "reservoirGiCandidatePositiveWeight",
	    "reservoirGiZeroWeight",
	    "reservoirGiSelectedWeightScaledSum",
	    "reservoirGiTemporalAccepted",
	    "reservoirGiTemporalRejected",
	    "reservoirGiSpatialAccepted",
	    "reservoirGiSpatialRejected",
	    "risWeightSum",
	    "selectedCandidateProbability",
	    "weightedBestTotal",
	    "recordFirstHitProbeStats(bestSurfaceHit, bestSunVisible, weightedBestTotal)",
	    "PATH_TRACER_BLACK_ENVIRONMENT_BIT",
	    "PATH_TRACER_APPLY_FIRST_HIT_PROBES_BIT",
	    "applyFirstHitProbesToFinal",
	    "blackEnvironmentEnabled"};
	const char *requiredDirectSunModeSymbols[] = {
	    "DIRECT_SUN_BOUNCE_FIRST_TWO",
	    "directSunBounceMode == DIRECT_SUN_BOUNCE_FIRST_TWO && bounce <= 1"};
	for (const char *symbol : requiredShaderSymbols)
	{
		if (!containsText(raygen, symbol) && !containsText(miss, symbol) && !containsText(denoiser, symbol))
		{
			std::cerr << "missing path tracer debug AOV shader symbol: " << symbol << "\n";
			return false;
		}
	}
	for (const char *symbol : requiredDirectSunModeSymbols)
	{
		if (!containsText(raygen, symbol))
		{
			std::cerr << "missing direct sun mode shader symbol: " << symbol << "\n";
			return false;
		}
	}
	if (!containsText(anyHit, "float  ao;"))
	{
		std::cerr << "path tracer AnyHit RayPayload layout must include ao before hitT\n";
		return false;
	}

	const char *requiredEngineSymbols[] = {
	    "blackEnvironment",
	    "applyFirstHitProbesToFinal",
	    "targetWallLuminanceAverage",
	    "targetWallBaseLuminanceAverage",
	    "targetWallFirstHitProbeContributionAverage",
	    "reservoirGiCandidates",
	    "reservoirGiAccepted",
	    "reservoirProposalMode=%d",
	    "reservoirGiCandidateRays=",
	    "reservoirGiAccepted=",
	    "reservoirGiAvgLuma=",
	    "reservoirGiTemporalAccepted",
	    "reservoirGiTemporalRejected",
	    "reservoirGiSpatialNeighborCount",
	    "reservoirGiSpatialAccepted",
	    "reservoirGiSpatialRejected",
	    "reservoirGiAvgLuma",
	    "reservoirGiCandidateSurfaceHitRatio",
	    "reservoirGiCandidateSunVisibleRatio",
	    "reservoirGiCandidatePositiveWeightRatio",
	    "reservoirGiSelectedWeightAverage",
	    "reservoirGiLumaScaledSum",
	    "targetWallSampleCount",
	    "firstHitProbeCount",
	    "firstHitProbeSurfaceHitCount",
	    "firstHitProbeSunVisibleCount",
	    "firstHitProbeContributionAverage",
	    "firstHitProbeSunVisibleContributionAverage",
	    "firstHitProbeSamplingMode",
	    "PathTracerReservoirGiMode",
	    "PathTracerReservoirGiProposalMode",
	    "kReservoirGiCurrentCapacity",
	    "PathTracerExperimentRow",
	    "runSponzaGiPerfSweep",
	    "startPathTracerSponzaGiPerfSweep",
	    "PathTracerSponzaValidationView",
	    "Sponza Validation View",
	    "Apply Sponza Validation View",
	    "applySponzaValidationPreset",
	    "sponzaScenarioPresetForView",
	    "scenarioName",
	    "cameraPosition",
	    "cameraPitch",
	    "cameraYaw",
	    "lightDirection",
	    "SponzaScenarioPreset",
	    "Dark Courtyard",
	    "Sunlit Courtyard Wall",
	    "Mid-Depth Interior",
	    "glm::vec3(0.0f, 12.0f, -1.5f)",
	    "glm::vec3(-1.0f, 6.5f, 3.0f)",
	    "Base 8 Sun All",
	    "Base 8 Sun First",
	    "Reservoir 1C Shadowed Sun All",
	    "Reservoir 1C Shadowed Sun First",
	    "Reservoir 1C Shadowed Sun First Sun Guided",
	    "Reservoir 1C Shadowed Sun First Mixed",
	    "Reservoir 1C Shadowed Sun First Light Region",
	    "Reservoir 2C Shadowed Sun First Mixed",
	    "Reservoir 2C Shadowed Sun First Mixed RIS",
	    "LightRegionGuided",
	    "rtPush.skyData = glm::vec4(pathTracerSettings.reservoirGiLightRegionTarget",
	    "reservoirGiLightRegionRadius",
	    "ptBenchmarkBasePosition",
	    "ui.lightDirection",
	    "makeScenarioRowName",
	    "reservoirSunFirstRow.directSunBounceMode",
	    "reservoirSunFirstRow.reservoirGiCandidateCount = 1",
	    "uint32_t padding3",
	    "rtPush.padding3 = pathTracerFlags",
	    "packedPathTracerMaterialIndex",
	    "settings.blackEnvironment",
	    "analysis.applyDebugLightPreset",
	    "updatePathTracerExperimentSweep",
	    "logPathTracerExperimentRow",
	    "clearPathTracerExperimentState",
	    "targetWallBaseLuminanceAverage",
	    "targetWallFirstHitProbeContributionAverage",
	    "targetWallLuminanceAverage",
	    "vulkan.logicalDevice.waitIdle()",
	    "debugBreakIfDebuggerAttached",
	    "waitForFenceOrThrow",
	    "vk::SystemError",
	    "Vulkan device lost while waiting for",
	    "pathTracerStorageBufferBarrier",
	    "vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite",
	    "PathTracerDebugLightPreset",
	    "applyPathTracerDebugLightPreset",
	    "pathTracerSettings.enableEnvironmentNEE",
	    "pathTracerSettings.blackEnvironment",
	    "pathTracerSettings.applyFirstHitProbesToFinal",
	    "pathTracerSettings.environmentNeeSamplingMode",
	    "pathTracerSettings.firstHitProbeSamplingMode",
	    "pathTracerSettings.firstHitDiffuseSamples",
	    "pathTracerSettings.firstHitCandidateCount",
	    "pathTracerAnalysisSettings.debugLightPreset",
	    "pathTracerAnalysisSettings.applyDebugLightPreset",
	    "pathTracerAnalysisSettings.debugAov",
	    "loadSponzaGiValidationPreset",
	    "loadPathTracerSponzaGiValidationPresetIfRequested",
	    "pathTracerAnalysisSettings.lockBenchmarkScene",
	    "pathTracerSettings.directSunBounceMode = 1",
	    "pathTracerSettings.reservoirGiCandidateCount = 1",
	    "pathTracerSettings.reservoirGiProposalMode",
	    "ui.renderMode",
	    "sponza_runtime.glb",
	    "ptIndirectBounceTargetWallModelId",
	    "loadIndirectBounceTestScene",
	    "loadPathTracerIndirectBounceTestSceneIfRequested",
	    "PT_IndirectBounce_Floor",
	    "PT_IndirectBounce_Wall",
	    "PT_IndirectBounce_Blocker",
	    "PT_IndirectBounce_Ceiling",
	    "PT_IndirectBounce_LeftWall",
	    "PT_IndirectBounce_RightWall"};
	for (const char *symbol : requiredEngineSymbols)
	{
		if (!containsText(uiHeader, symbol) && !containsText(uiSource, symbol) &&
		    !containsText(engineAuxiliaryHeader, symbol) && !containsText(engineHeader, symbol) &&
		    !containsText(engineSource, symbol) && !containsText(frameContextHeader, symbol) &&
		    !containsText(frameContextSource, symbol))
		{
			std::cerr << "missing path tracer indirect bounce scene symbol: " << symbol << "\n";
			return false;
		}
	}
	if (containsText(engineSource, "Sponza / Reservoir GI Temporal") ||
	    containsText(engineSource, "Sponza / Reservoir GI Temporal Spatial") ||
	    containsText(engineSource, "Sponza / Reservoir GI Single Frame 2 Candidates No RIS") ||
	    containsText(engineSource, "Sponza / Reservoir GI Single Frame 2 Candidates RIS") ||
	    containsText(engineSource, "Cache Chosen Radius 14 Budget 1") ||
	    containsText(engineSource, "makeCacheChosenRow"))
	{
		std::cerr << "focused Sponza reservoir sweep should only compare base and reservoir variants\n";
		return false;
	}

	const std::string activeCacheCleanupSources = uiHeader + uiSource + engineHeader + engineSource +
	                                              frameContextHeader + frameContextSource +
	                                              pipelineCollection + raygen;
	const char *forbiddenCacheSymbols[] = {
	    "Sun-Visible Cache",
	    "Enable Sun-Visible Cache",
	    "Clear Sun-Visible Cache",
	    "Cached Secondary Reuse",
	    "Cache Reuse Weight",
	    "Cache Reuse Attempts",
	    "Cache Reuse Accepted",
	    "Cache Reuse Avg Luma",
	    "Diagnostic Target Cache",
	    "PathTracerCacheWeightingMode",
	    "PathTracerCacheProposalMode",
	    "enableSunVisibleCandidateCache",
	    "cacheReuseWeight",
	    "cacheConnectionRadius",
	    "cacheProposalMode",
	    "cacheVisibilityValidationBudget",
	    "cacheRefreshCandidateCount",
	    "clearSunVisibleCandidateCache",
	    "updateSunVisibleCandidateCacheInvalidation",
	    "ptSunVisibleCandidateCache",
	    "ptSunVisibleConnectionCache",
	    "SUN_VISIBLE_CANDIDATE_CACHE_BINDING",
	    "SUN_VISIBLE_CONNECTION_CACHE_BINDING",
	    "sampleCachedSunVisibleCandidate",
	    "refreshSunVisibleCandidateCache",
	    "refreshDiagnosticTargetSunVisibleCandidateCache",
	    "sunVisibleCandidateCacheBuffers",
	    "sunVisibleConnectionCacheBuffers",
	    "cacheReusePathEntryCount",
	    "cacheRefreshAttemptCount"};
	for (const char *symbol : forbiddenCacheSymbols)
	{
		if (containsText(activeCacheCleanupSources, symbol))
		{
			std::cerr << "old sun-visible cache symbol remains in active path tracer source: " << symbol << "\n";
			return false;
		}
	}

	return true;
}
