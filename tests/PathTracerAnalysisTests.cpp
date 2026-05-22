#include "PathTracerAnalysisTests.h"

#include "../src/Core/PathTracerAnalysis.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

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

std::filesystem::path findProjectRoot()
{
#ifdef LAPHRIA_SOURCE_DIR
	return std::filesystem::path(LAPHRIA_SOURCE_DIR);
#else
	return std::filesystem::current_path();
#endif
}

std::string chars(std::initializer_list<char> values)
{
	return std::string(values.begin(), values.end());
}

bool requireNoRetiredPathTracerGiCache(const std::filesystem::path &sourceRoot)
{
	const std::vector<std::filesystem::path> activeFiles = {
	    sourceRoot / "src" / "Core" / "EngineAuxiliary.h",
	    sourceRoot / "src" / "Core" / "EngineCore.h",
	    sourceRoot / "src" / "Core" / "EngineCore.cpp",
	    sourceRoot / "src" / "Core" / "FrameContext.h",
	    sourceRoot / "src" / "Core" / "FrameContext.cpp",
	    sourceRoot / "src" / "Core" / "PipelineCollection.cpp",
	    sourceRoot / "src" / "Core" / "UISystem.h",
	    sourceRoot / "src" / "Core" / "UISystem.cpp",
	    sourceRoot / "src" / "shaders" / "Raygen.slang",
	    sourceRoot / "src" / "shaders" / "Denoiser.slang",
	    sourceRoot / "README.md"};

	const std::string lowerCache = chars({'r', 'e', 's', 'e', 'r', 'v', 'o', 'i', 'r'});
	const std::string upperCache = chars({'R', 'E', 'S', 'E', 'R', 'V', 'O', 'I', 'R'});
	const std::string titleCache = chars({'R', 'e', 's', 'e', 'r', 'v', 'o', 'i', 'r'});
	const std::vector<std::string> forbiddenSymbols = {
	    lowerCache + "Gi",
	    titleCache + "Gi",
	    upperCache + "_GI",
	    "PathTracer" + titleCache + "Gi",
	    "pt" + titleCache + "Gi",
	    "Run Sponza GI Perf Sweep",
	    "Bright " "Surfel Shadow Sweep",
	    "Bright " "Surfel Proposal Sweep"};

	for (const auto &path : activeFiles)
	{
		if (!std::filesystem::exists(path))
		{
			std::cerr << "retired path tracer GI cache guard missing required file: "
			          << path.string() << "\n";
			return false;
		}

		std::ifstream file(path, std::ios::binary);
		if (!file)
		{
			std::cerr << "retired path tracer GI cache guard cannot read required file: "
			          << path.string() << "\n";
			return false;
		}
		const std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
		for (const std::string &symbol : forbiddenSymbols)
		{
			if (text.find(symbol) != std::string::npos)
			{
				std::cerr << "retired path tracer GI cache symbol still present in "
				          << path.string() << ": " << symbol << "\n";
				return false;
			}
		}
	}
	return true;
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

bool testPathTracerRemovedGiCacheGuard()
{
	const std::filesystem::path sourceRoot = findProjectRoot();
	return requireNoRetiredPathTracerGiCache(sourceRoot);
}

bool testPathTracerDebugAovContract()
{
	const std::filesystem::path sourceRoot = findProjectRoot();

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
		std::cerr << "path tracer sweep defaults should stay short enough for interactive runs\n";
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
	    "Path Tracer Diagnostics",
	    "Core Diagnostics",
	    "Benchmark Automation",
	    "Frame Stats",
	    "First-Hit Probe Sampling",
	    "Naive Sun Guide",
	    "Candidate Sun Bounce",
	    "Candidate Average Reference",
	    "Candidate RIS",
	    "First-Hit Candidate Count",
	    "Env NEE Bounces",
	    "First Only",
	    "First Two",
	    "All Bounces"};
	for (const char *label : requiredLabels)
	{
		if (!containsText(uiSource, label))
		{
			std::cerr << "missing path tracer debug AOV UI label: " << label << "\n";
			return false;
		}
	}

	const std::string removedMetricLabels[] = {
	    std::string("Core ") + "Metrics",
	    std::string("First-Hit Probe ") + "Rays",
	    std::string("First-Hit Probe ") + "Surface Hits",
	    std::string("First-Hit Probe ") + "Sun Visible",
	    std::string("First-Hit Probe ") + "Avg Luma",
	    std::string("First-Hit Probe ") + "Sun-Visible Avg Luma"};
	for (const std::string &label : removedMetricLabels)
	{
		if (containsText(uiSource, label.c_str()) || containsText(uiHeader, label.c_str()))
		{
			std::cerr << "removed first-hit probe metric label is still present: " << label << "\n";
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
	for (const char *enumValue : requiredEnumValues)
	{
		if (!containsText(uiHeader, enumValue))
		{
			std::cerr << "missing path tracer debug AOV enum value: " << enumValue << "\n";
			return false;
		}
	}
	if (!containsText(denoiser, "selectPathTracerDebugAovOutput(pixel, color)") ||
	    !containsText(denoiser, "selectPathTracerDebugAovOutput(pixel, filtered)"))
	{
		std::cerr << "path debug AOVs must bypass temporal/denoised display in both pass-through and A-Trous paths\n";
		return false;
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
	    "ENV_NEE_BOUNCE_FIRST_ONLY",
	    "ENV_NEE_BOUNCE_FIRST_TWO",
	    "ENV_NEE_BOUNCE_ALL",
	    "bool environmentNeeEnabledForBounce(int bounce, int environmentNeeBounceMode)",
	    "environmentNeeBounceMode == ENV_NEE_BOUNCE_FIRST_TWO && bounce <= 1",
	    "environmentNeeBounceMode == ENV_NEE_BOUNCE_ALL",
	    "int    environmentNeeBounceMode",
	    "ENV_NEE_SAMPLING_SKY_BIASED",
	    "firstHitDiffuseSampleCount",
	    "firstHitProbeSamplingMode",
	    "sampleFirstHitDiffuseBounce",
	    "sampleFirstHitProbeDirection",
	    "debugFirstHitBounceContribution",
	    "debugSecondaryDirectSunContribution",
	    "debugBaselineContinuationContribution",
	    "FIRST_HIT_PROBE_SAMPLING_SUN_BOUNCE_GUIDED",
	    "FIRST_HIT_PROBE_SAMPLING_CANDIDATE_SUN_BOUNCE",
	    "FIRST_HIT_PROBE_SAMPLING_CANDIDATE_AVERAGE_REFERENCE",
	    "FIRST_HIT_PROBE_SAMPLING_CANDIDATE_RIS",
	    "sampleFirstHitCandidateSunBounce",
	    "sampleFirstHitCandidateAverageReference",
	    "sampleFirstHitCandidateRis",
	    "PT_FLAGS_ENVIRONMENT_NEE_BIT",
	    "PT_FLAGS_ENVIRONMENT_BOUNCE_SHIFT",
	    "PT_FLAGS_FIRST_HIT_PROBE_SAMPLING_SHIFT",
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
	    "packPathTracerMaterialSettings",
	    "packPathTracerFlags",
	    "kPtFlagsEnvironmentNeeBit",
	    "kPtFlagsEnvironmentBounceShift",
	    "kPtFlagsFirstHitProbeSamplingShift",
	    "environmentNeeBounceMode",
	    "Env NEE Bounces",
	    "First Only",
	    "First Two",
	    "All Bounces",
	    "firstHitProbeSamplingMode",
	    "PathTracerExperimentRow",
	    "uint32_t padding3",
	    "rtPush.padding3 = pathTracerFlags",
	    "packPathTracerMaterialSettings(ui.pathTracerSettings)",
	    "packPathTracerFlags(ui.pathTracerSettings)",
	    "settings.blackEnvironment",
	    "updatePathTracerExperimentSweep",
	    "logPathTracerExperimentRow",
	    "debugBreakIfDebuggerAttached",
	    "waitForFenceOrThrow",
	    "vk::SystemError",
	    "Vulkan device lost while waiting for",
	    "pathTracerStorageBufferBarrier",
	    "vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite",
	    "pathTracerSettings.enableEnvironmentNEE",
	    "pathTracerSettings.blackEnvironment",
	    "pathTracerSettings.applyFirstHitProbesToFinal",
	    "pathTracerSettings.environmentNeeSamplingMode",
	    "pathTracerSettings.environmentNeeBounceMode",
	    "pathTracerSettings.firstHitProbeSamplingMode",
	    "pathTracerSettings.firstHitDiffuseSamples",
	    "pathTracerSettings.firstHitCandidateCount",
	    "pathTracerAnalysisSettings.debugAov",
	    "pathTracerAnalysisSettings.lockBenchmarkScene",
	    "pathTracerSettings.directSunBounceMode",
	    "ui.renderMode"};
	for (const char *symbol : requiredEngineSymbols)
	{
		if (!containsText(uiHeader, symbol) && !containsText(uiSource, symbol) &&
		    !containsText(engineAuxiliaryHeader, symbol) && !containsText(engineHeader, symbol) &&
		    !containsText(engineSource, symbol) && !containsText(frameContextHeader, symbol) &&
		    !containsText(frameContextSource, symbol))
		{
			std::cerr << "missing path tracer analysis contract symbol: " << symbol << "\n";
			return false;
		}
	}

	const std::string activeSceneCleanupSources =
	    uiHeader + uiSource + engineHeader + engineSource + engineAuxiliaryHeader + raygen;
	const char *forbiddenSceneCleanupSymbols[] = {
	    "Scenario: Indirect " "Bounce Box",
	    "Target " "Wall Avg Luma",
	    "Indirect " "Box Capture Checklist",
	    "Record these values after the image stabilizes",
	    "Target " "Wall Base Luma",
	    "Target " "Wall Probe Added Luma",
	    "Light Preset",
	    "Hard " "Bounce",
	    "Medium " "Bounce",
	    "Easy " "Bounce",
	    "Hard" "Bounce",
	    "Medium" "Bounce",
	    "Easy" "Bounce",
	    "Apply Light Preset",
	    "Load Indirect " "Bounce Test Scene",
	    "record" "Target" "Wall" "Luminance",
	    "target" "Wall" "First" "HitProbeContributionSum",
	    "target" "Wall" "Base" "LuminanceSum",
	    "record" "Target" "Wall" "Luminance(radiance",
	    "target" "Wall" "Luminance" "Average",
	    "target" "Wall" "Base" "LuminanceAverage",
	    "target" "Wall" "First" "HitProbeContributionAverage",
	    "target" "Wall" "Sample" "Count",
	    "analysis.apply" "Debug" "LightPreset",
	    "PathTracer" "Debug" "LightPreset",
	    "applyPathTracer" "Debug" "LightPreset",
	    "pathTracerAnalysisSettings.debug" "LightPreset",
	    "pathTracerAnalysisSettings.apply" "Debug" "LightPreset",
	    "pt" "Indirect" "Bounce" "Target" "WallModelId",
	    "load" "Indirect" "Bounce" "TestScene",
	    "loadPathTracer" "Indirect" "Bounce" "TestSceneIfRequested",
	    "PT_" "Indirect" "Bounce_"};
	for (const char *symbol : forbiddenSceneCleanupSymbols)
	{
		if (containsText(activeSceneCleanupSources, symbol))
		{
			std::cerr << "retired scene diagnostic symbol remains in active path tracer source: "
			          << symbol << "\n";
			return false;
		}
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
