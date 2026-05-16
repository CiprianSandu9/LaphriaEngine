#include "PathTracerAnalysisTests.h"

#include "../src/Core/EngineAuxiliary.h"
#include "../src/Core/PathTracerAnalysis.h"

#include <cmath>
#include <cstddef>
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
	const std::string engineAuxiliaryHeader = readTextFile(sourceRoot / "src" / "Core" / "EngineAuxiliary.h");
	const std::string uiHeader = readTextFile(sourceRoot / "src" / "Core" / "UISystem.h");
	const std::string uiSource = readTextFile(sourceRoot / "src" / "Core" / "UISystem.cpp");
	const std::string frameContextHeader = readTextFile(sourceRoot / "src" / "Core" / "FrameContext.h");
	const std::string frameContextSource = readTextFile(sourceRoot / "src" / "Core" / "FrameContext.cpp");
	const std::string resourceManager = readTextFile(sourceRoot / "src" / "Core" / "ResourceManager.cpp");
	const std::string gltfImporter    = readTextFile(sourceRoot / "src" / "Core" / "GltfImporter.cpp");

	if (raygen.empty() || engineCore.empty() || engineAuxiliaryHeader.empty() ||
	    uiHeader.empty() || uiSource.empty() || frameContextHeader.empty() ||
	    frameContextSource.empty() || resourceManager.empty() || gltfImporter.empty())
	{
		std::cerr << "failed to read reservoir GI measurement contract sources\n";
		return false;
	}

	const char *requiredRaygenSymbols[] = {
	    "RESERVOIR_GI_RECORD_SIZE = 160",
	    "RESERVOIR_GI_RECORD_PRIMARY_POSITION_OFFSET = 0",
	    "RESERVOIR_GI_RECORD_PRIMARY_NORMAL_OFFSET = 16",
	    "RESERVOIR_GI_RECORD_CANDIDATE_POSITION_OFFSET = 32",
	    "RESERVOIR_GI_RECORD_CANDIDATE_NORMAL_OFFSET = 48",
	    "RESERVOIR_GI_RECORD_SUFFIX_RADIANCE_OFFSET = 64",
	    "RESERVOIR_GI_RECORD_CONTRIBUTION_OFFSET = 80",
	    "RESERVOIR_GI_RECORD_SOURCE_PDF_OFFSET = 96",
	    "RESERVOIR_GI_RECORD_TARGET_WEIGHT_OFFSET = 100",
	    "RESERVOIR_GI_RECORD_WEIGHT_SUM_OFFSET = 104",
	    "RESERVOIR_GI_RECORD_SELECTED_WEIGHT_OFFSET = 108",
	    "RESERVOIR_GI_RECORD_CONFIDENCE_M_OFFSET = 112",
	    "RESERVOIR_GI_RECORD_SOURCE_PIXEL_OFFSET = 116",
	    "RESERVOIR_GI_RECORD_SOURCE_FRAME_ID_OFFSET = 120",
	    "RESERVOIR_GI_RECORD_FRAME_ID_OFFSET = 124",
	    "RESERVOIR_GI_RECORD_FLAGS_OFFSET = 128",
	    "RESERVOIR_GI_RECORD_USED_SIZE = 132",
	    "RESERVOIR_GI_TEMPORAL_M_CLAMP",
	    "RESERVOIR_GI_SPATIAL_M_CLAMP",
	    "reservoirGiConfidenceMScaledSumOffset",
	    "clampReservoirGiConfidence",
	    "storeFloat3ToReservoirGi(offset + RESERVOIR_GI_RECORD_CANDIDATE_NORMAL_OFFSET, record.candidateNormal)",
	    "record.candidateNormal = loadFloat3FromReservoirGiHistory(offset + RESERVOIR_GI_RECORD_CANDIDATE_NORMAL_OFFSET)",
	    "ptReservoirGiCurrent.Store(offset + RESERVOIR_GI_RECORD_WEIGHT_SUM_OFFSET, asuint(record.weightSum))",
	    "ptReservoirGiCurrent.Store(offset + RESERVOIR_GI_RECORD_SELECTED_WEIGHT_OFFSET, asuint(record.selectedWeight))",
	    "ptReservoirGiCurrent.Store(offset + RESERVOIR_GI_RECORD_CONFIDENCE_M_OFFSET, asuint(record.confidenceM))",
	    "ptReservoirGiCurrent.Store(offset + RESERVOIR_GI_RECORD_FRAME_ID_OFFSET, record.frameId)",
	    "record.weightSum = asfloat(ptReservoirGiHistory.Load(offset + RESERVOIR_GI_RECORD_WEIGHT_SUM_OFFSET))",
	    "record.selectedWeight = asfloat(ptReservoirGiHistory.Load(offset + RESERVOIR_GI_RECORD_SELECTED_WEIGHT_OFFSET))",
	    "record.frameId = ptReservoirGiHistory.Load(offset + RESERVOIR_GI_RECORD_FRAME_ID_OFFSET)",
	    "record.flags = ptReservoirGiHistory.Load(offset + RESERVOIR_GI_RECORD_FLAGS_OFFSET)",
	    "float3 candidateNormal",
	    "float3 suffixRadiance",
	    "float sourcePdf",
	    "float targetWeight",
	    "float  weightSum",
	    "float  selectedWeight",
	    "float confidenceM",
	    "uint sourcePixel",
	    "uint sourceFrameId",
	    "uint   frameId",
	    "out float3 reconnectedCandidateNormal",
	    "out float3 reconnectedSuffixRadiance",
	    "out float reconnectedTargetWeight",
	    "reconnectedCandidateNormal = reconnectPayload.worldNormal",
	    "reconnectedSuffixRadiance = sanitizeReservoirGiContribution(emissiveContribution + reconnectedSecondarySunSuffix)",
	    "float3 temporalCandidateNormal",
	    "float3 temporalSuffixRadiance",
	    "float temporalTargetWeight",
	    "selectedNormal = temporalCandidateNormal",
	    "selectedSuffixRadiance = temporalSuffixRadiance",
	    "selectedTargetWeight = temporalTargetWeight",
	    "validated.confidenceM = clampReservoirGiConfidence(selected.confidenceM + 1.0, RESERVOIR_GI_TEMPORAL_M_CLAMP)",
	    "ReservoirGiRecord evaluateSpatialReservoirGiCandidate(",
	    "combineSpatialReservoirGi(launchID, launchSize, hitPos, N, V, payload",
	    "selectedSuffixRadiance = spatialCandidate.suffixRadiance",
	    "selectedTargetWeight = spatialCandidate.targetWeight",
	    "selectedConfidenceM = clampReservoirGiConfidence(spatialRecord.confidenceM, RESERVOIR_GI_SPATIAL_M_CLAMP)",
	    "makeLocalReservoirGiSample",
	    "combineReservoirGiCandidate",
	    "validateSelectedReservoirGiCandidate",
	    "reconnectTemporalReservoirGi(selected",
	    "reservoirGiSelectedLocal",
	    "reservoirGiSelectedTemporal",
	    "reservoirGiSelectedSpatial",
	    "record.sourcePdf = reservoirGiProposalPdf",
	    "record.candidateNormal = bouncePayload.worldNormal",
	    "record.suffixRadiance = candidateSuffixRadiance",
	    "evaluateReservoirGiTargetAtPrimary(hitPos",
	    "float3 evaluateReservoirGiSecondarySunSuffix",
	    "float3(1.0, 1.0, 1.0)",
	    "float3 emissiveContribution = bouncePayload.emission",
	    "float3 candidateSecondarySunSuffix = evaluateReservoirGiSecondarySunSuffix",
	    "candidateSecondarySun = sanitizeReservoirGiContribution(firstLegThroughput * candidateSecondarySunSuffix)",
	    "float3 candidateSuffixRadiance = emissiveContribution + candidateSecondarySunSuffix",
	    "reconnectedSecondarySun = sanitizeReservoirGiContribution(firstLegThroughput * reconnectedSecondarySunSuffix)",
	    "float3 selectedSuffixRadiance",
	    "ReservoirGiRecord candidateRecord = localRecord",
	    "record.suffixRadiance = selectedSuffixRadiance",
	    "reprojectReservoirHistoryPixel",
	    "mul(ubo.prevViewProj, float4(currentPrimaryPosition, 1.0))",
	    "prevPixel = uint2(uint(pixel.x), uint(pixel.y))",
	    "loadTemporalReservoirGi(temporalPixel, launchSize",
	    "float reservoirProbeScale = float(candidateCount) / float(candidateCount + 1)",
	    "result.totalContribution = reservoirTotal * reservoirProbeScale",
	    "result.secondaryDirectSunContribution = reservoirSecondarySun * reservoirProbeScale",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiCandidatesOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiConfidenceMScaledSumOffset, scaledConfidenceM)",
	    "PATH_TRACER_RESERVOIR_GI_DETAILED_DIAGNOSTICS_BIT",
	    "reservoirGiDetailedDiagnostics",
	    "ReservoirGiHistoryMetadata",
	    "loadReservoirGiHistoryMetadata",
	    "historyMetadata.valid",
	    "historyMetadata.frameId + 1u != ubo.frameCount",
	    "historyMetadata.targetWeight",
	    "historyMetadata.confidenceM",
	    "shouldAttemptTemporalReservoirGiReuse",
	    "storeAcceptedReservoirGi",
	    "updateReservoirGi(launchID, launchSize, record);",
	    "PT_FLAGS_RESERVOIR_TEMPORAL_BUDGET_SHIFT",
	    "PT_FLAGS_RESERVOIR_SPATIAL_BUDGET_SHIFT",
	    "reservoirGiTemporalBudgetDivisor",
	    "reservoirGiSpatialBudgetDivisor",
	    "if (code == 2u)",
	    "shouldRunReservoirGiBudgetedPixel",
	    "reservoirGiTemporalBudgetPass",
	    "reservoirGiSpatialBudgetPass",
	    "reservoirGiLocalSurfaceHitsOffset",
	    "reservoirGiLocalValidSamplesOffset",
	    "reservoirGiLocalShadowRaysOffset",
	    "reservoirGiTemporalReconnectRaysOffset",
	    "reservoirGiTemporalShadowRaysOffset",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiLocalSurfaceHitsOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiLocalValidSamplesOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiLocalShadowRaysOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiTemporalReconnectRaysOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiTemporalShadowRaysOffset, 1u)",
	    "float storedConfidenceLimit",
	    "selectedSource == RESERVOIR_GI_SOURCE_SPATIAL ? RESERVOIR_GI_SPATIAL_M_CLAMP : RESERVOIR_GI_TEMPORAL_M_CLAMP",
	    "record.confidenceM = acceptedReservoir ? clampReservoirGiConfidence(selectedConfidenceM, storedConfidenceLimit) : 0.0",
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
	const char *forbiddenRaygenSymbols[] = {
	    "candidateSecondarySun = targetEvaluation.suffixRadiance",
	    "selectedTargetWeight = temporalRecord.targetWeight",
	    "selectedTargetWeight = spatialRecord.targetWeight",
	    "spatialWeight = max(spatialRecord.selectedWeight,",
	    "selectedTargetWeight = spatialWeight",
	    "Milestone 2 spatial reuse stores spatialWeight as the current-domain target-weight proxy",
	    "record.suffixRadiance = reservoirSecondarySun",
	    "record.suffixRadiance = reservoirSuffixRadiance",
	    "reservoirSecondarySun = sanitizeReservoirGiContribution(spatialRecord.suffixRadiance)",
	    "reconnectedSuffixRadiance = sanitizeReservoirGiContribution(emissiveContribution + reconnectedSecondarySun)",
	    "reservoirSuffixRadiance *= risScale",
	    "reservoirSuffixRadiance /= float(candidateCount)",
	    "reservoirSuffixRadiance = temporalContribution",
	    "ReservoirGiRecord emptyReservoir",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiTemporalRejectGeometryOffset, 1u)"};
	for (const char *symbol : forbiddenRaygenSymbols)
	{
		if (containsText(raygen, symbol))
		{
			std::cerr << "stale reservoir GI source contract remains: " << symbol << "\n";
			return false;
		}
	}
	const std::size_t localSamplePos = raygen.find("makeLocalReservoirGiSample(hitPos");
	const std::size_t localValidGuardPos = raygen.find("if (!localSurfaceHit || !localValid) {", localSamplePos);
	const std::size_t sunVisibleCounterPos = raygen.find("candidateSunVisibleCount += 1u;", localSamplePos);
	if (localSamplePos == std::string::npos ||
	    localValidGuardPos == std::string::npos ||
	    sunVisibleCounterPos == std::string::npos ||
	    sunVisibleCounterPos < localValidGuardPos)
	{
		std::cerr << "reservoir GI sun-visible counter must run after local sample validation\n";
		return false;
	}
	const std::size_t temporalSamplePos =
	    raygen.find("if (reservoirGiMode >= PATH_TRACER_RESERVOIR_GI_TEMPORAL)");
	const std::size_t spatialSamplePos =
	    raygen.find("if (reservoirGiMode == PATH_TRACER_RESERVOIR_GI_TEMPORAL_SPATIAL)", temporalSamplePos);
	if (temporalSamplePos == std::string::npos || spatialSamplePos == std::string::npos)
	{
		std::cerr << "reservoir GI temporal/spatial sample function markers are missing\n";
		return false;
	}
	const std::string temporalBranch = raygen.substr(temporalSamplePos, spatialSamplePos - temporalSamplePos);
	if (containsText(temporalBranch, "loadTemporalReservoirGi(launchID, launchSize"))
	{
		std::cerr << "reservoir GI temporal branch still uses same-pixel history lookup\n";
		return false;
	}
	if (containsText(temporalBranch, "reconnectTemporalReservoirGi("))
	{
		std::cerr << "reservoir GI temporal branch must combine loaded candidates before selected-candidate reconnect\n";
		return false;
	}
	const std::size_t localReservoirSamplePos = raygen.find("ReservoirGiRecord makeLocalReservoirGiSample(");
	const std::size_t temporalReconnectPos = raygen.find("bool reconnectTemporalReservoirGi(");
	if (localReservoirSamplePos == std::string::npos ||
	    temporalReconnectPos == std::string::npos ||
	    localReservoirSamplePos > temporalReconnectPos)
	{
		std::cerr << "reservoir GI local sample function markers are missing\n";
		return false;
	}
	const std::string localReservoirSampleFunction =
	    raygen.substr(localReservoirSamplePos, temporalReconnectPos - localReservoirSamplePos);
	if (containsText(localReservoirSampleFunction, "float3 emissiveContribution = firstLegThroughput * bouncePayload.emission") ||
	    containsText(localReservoirSampleFunction, "float3 candidateSuffixRadiance = emissiveContribution + secondaryDirectSun") ||
	    containsText(localReservoirSampleFunction, "candidateSecondarySun = evaluateUnshadowedDirectSunContribution") ||
	    containsText(localReservoirSampleFunction, "candidateSecondarySun = evaluateDirectSunContributionInternal"))
	{
		std::cerr << "reservoir GI local sun evaluation must derive weighted output from one suffix sample\n";
		return false;
	}
	const std::size_t postTemporalReconnectPos =
	    raygen.find("FirstHitDiffuseBounceResult sampleFirstHitDiffuseBounce", temporalReconnectPos);
	if (postTemporalReconnectPos == std::string::npos)
	{
		std::cerr << "reservoir GI temporal reconnect function markers are missing\n";
		return false;
	}
	const std::string temporalReconnectFunction =
	    raygen.substr(temporalReconnectPos, postTemporalReconnectPos - temporalReconnectPos);
	const char *requiredTemporalReconnectSymbols[] = {
	    "float sourcePdf = isFinitePositive(temporalRecord.sourcePdf)",
	    "evaluateReservoirGiTargetAtPrimary(currentPrimaryPosition",
	    "reconnectedSuffixRadiance, sourcePdf",
	    "reconnectedContribution = targetEvaluation.contribution",
	    "reconnectedTargetWeight = targetEvaluation.targetWeight",
	    "reconnectedWeight = targetEvaluation.targetWeight"};
	for (const char *symbol : requiredTemporalReconnectSymbols)
	{
		if (!containsText(temporalReconnectFunction, symbol))
		{
			std::cerr << "missing reservoir GI temporal reconnect target contract: " << symbol << "\n";
			return false;
		}
	}
	const char *forbiddenTemporalReconnectSymbols[] = {
	    "reconnectedContribution = reconnectedSuffixRadiance",
	    "reconnectedWeight = pathTracerLuminance(reconnectedContribution)",
	    "reconnectedTargetWeight = reconnectedWeight",
	    "float3 emissiveContribution = firstLegThroughput * reconnectPayload.emission",
	    "reconnectedSuffixRadiance = sanitizeReservoirGiContribution(emissiveContribution + reconnectedSecondarySun)",
	    "reconnectedSecondarySun = evaluateUnshadowedDirectSunContribution",
	    "reconnectedSecondarySun = evaluateDirectSunContributionInternal"};
	for (const char *symbol : forbiddenTemporalReconnectSymbols)
	{
		if (containsText(temporalReconnectFunction, symbol))
		{
			std::cerr << "stale reservoir GI temporal reconnect contract remains: " << symbol << "\n";
			return false;
		}
	}
	const std::size_t temporalTargetEvalPos =
	    temporalReconnectFunction.find("ReservoirGiTargetEvaluation targetEvaluation");
	const std::size_t temporalTargetLightRejectPos =
	    temporalReconnectFunction.find("reservoirGiTemporalRejectLightOffset", temporalTargetEvalPos);
	if (temporalTargetEvalPos == std::string::npos ||
	    temporalTargetLightRejectPos == std::string::npos)
	{
	    std::cerr << "reservoir GI temporal reconnect invalid target evaluation must reject as light\n";
	    return false;
	}
	const std::size_t targetEvalFunctionPos = raygen.find("ReservoirGiTargetEvaluation evaluateReservoirGiTargetAtPrimary(");
	const std::size_t postTargetEvalFunctionPos = raygen.find("void updateReservoirGi", targetEvalFunctionPos);
	if (targetEvalFunctionPos == std::string::npos || postTargetEvalFunctionPos == std::string::npos)
	{
		std::cerr << "reservoir GI target evaluation function markers are missing\n";
		return false;
	}
	const std::string targetEvalFunction =
	    raygen.substr(targetEvalFunctionPos, postTargetEvalFunctionPos - targetEvalFunctionPos);
	if (!containsText(targetEvalFunction, "float directionalWeight = primaryCosine / max(sourcePdf, 0.0001)") ||
	    !containsText(targetEvalFunction, "candidateCosine > 0.0001") ||
	    !containsText(targetEvalFunction, "bsdfAtPrimary * evaluation.suffixRadiance * directionalWeight") ||
	    containsText(targetEvalFunction, "(primaryCosine * candidateCosine) / max(distanceSquared") ||
	    containsText(targetEvalFunction, "geometryTerm"))
	{
		std::cerr << "reservoir GI target evaluation must use directional-PDF weighting while retaining candidate backface gating\n";
		return false;
	}
	const std::size_t temporalReprojectPos = temporalBranch.find("reprojectReservoirHistoryPixel(hitPos, launchSize, temporalPixel)");
	const std::size_t temporalMetadataGatePos =
	    temporalBranch.find("shouldAttemptTemporalReservoirGiReuse(temporalPixel, launchSize, historyMetadata)");
	const std::size_t temporalAttemptPos =
	    temporalBranch.find("ptAnalysisCounters.InterlockedAdd(reservoirGiTemporalReuseAttemptsOffset, 1u)",
	                        temporalMetadataGatePos);
	if (temporalReprojectPos == std::string::npos ||
	    temporalMetadataGatePos == std::string::npos ||
	    temporalAttemptPos == std::string::npos ||
	    temporalReprojectPos > temporalMetadataGatePos ||
	    temporalMetadataGatePos > temporalAttemptPos)
	{
		std::cerr << "reservoir GI temporal branch must count reuse attempts only after reprojection and metadata validation\n";
		return false;
	}
	const std::size_t loaderPos = raygen.find("bool loadTemporalReservoirGi(");
	const std::size_t spatialCombinerPos = raygen.find("void combineSpatialReservoirGi", loaderPos);
	if (loaderPos == std::string::npos || spatialCombinerPos == std::string::npos)
	{
		std::cerr << "reservoir GI temporal loader markers are missing\n";
		return false;
	}
	const std::string temporalLoader = raygen.substr(loaderPos, spatialCombinerPos - loaderPos);
	if (containsText(temporalLoader, "reservoirGiTemporalReuseAttemptsOffset"))
	{
		std::cerr << "reservoir GI temporal loader must not count attempts for spatial neighbor loads\n";
		return false;
	}
	if (!containsText(temporalLoader, "bool countTemporalRejects") ||
	    !containsText(temporalLoader, "bool reservoirGiDetailedDiagnostics") ||
	    !containsText(temporalLoader, "countTemporalReservoirGiReject(reservoirGiDetailedDiagnostics, countTemporalRejects, reservoirGiTemporalRejectGeometryOffset") ||
	    !containsText(temporalLoader, "countTemporalReservoirGiReject(reservoirGiDetailedDiagnostics, countTemporalRejects, reservoirGiTemporalRejectLightOffset") ||
	    containsText(temporalLoader, "ptAnalysisCounters.InterlockedAdd(reservoirGiTemporalReject"))
	{
		std::cerr << "reservoir GI temporal loader reject counters must be gated by diagnostics and countTemporalRejects\n";
		return false;
	}
	const std::size_t temporalBudgetGatePos =
	    temporalBranch.find("reservoirGiTemporalBudgetPass");
	const std::size_t temporalBudgetedLoadPos =
	    temporalBranch.find("loadTemporalReservoirGi(temporalPixel, launchSize");
	if (temporalBudgetGatePos == std::string::npos ||
	    temporalBudgetedLoadPos == std::string::npos ||
	    temporalBudgetGatePos > temporalBudgetedLoadPos)
	{
		std::cerr << "temporal budget gate must run before full temporal reservoir load\n";
		return false;
	}
	const std::size_t spatialEvaluatorPos =
	    raygen.find("ReservoirGiRecord evaluateSpatialReservoirGiCandidate(", loaderPos);
	const std::size_t postSpatialEvaluatorPos =
	    raygen.find("void combineSpatialReservoirGi", spatialEvaluatorPos);
	if (spatialEvaluatorPos == std::string::npos || postSpatialEvaluatorPos == std::string::npos)
	{
		std::cerr << "reservoir GI spatial candidate evaluator markers are missing\n";
		return false;
	}
	const std::string spatialEvaluator =
	    raygen.substr(spatialEvaluatorPos, postSpatialEvaluatorPos - spatialEvaluatorPos);
	if (!containsText(spatialEvaluator, "evaluateReservoirGiTargetAtPrimary(currentPrimaryPosition") ||
	    !containsText(spatialEvaluator, "spatialCandidate.contribution = targetEvaluation.contribution") ||
	    !containsText(spatialEvaluator, "spatialCandidate.targetWeight = targetEvaluation.targetWeight") ||
	    !containsText(spatialEvaluator, "spatialCandidate.selectedWeight = targetEvaluation.targetWeight") ||
	    !containsText(spatialEvaluator, "spatialCandidate.weightSum = targetEvaluation.targetWeight"))
	{
		std::cerr << "reservoir GI spatial candidate evaluator must use the current-domain target evaluator\n";
		return false;
	}
	const std::size_t spatialCombinerEndPos = raygen.find("bool isFinitePositive(float value)", spatialCombinerPos);
	const std::string spatialCombiner =
	    raygen.substr(spatialCombinerPos, spatialCombinerEndPos - spatialCombinerPos);
	if (!containsText(spatialCombiner, "float3 currentPrimaryPosition") ||
	    !containsText(spatialCombiner, "RayPayload primaryPayload") ||
	    !containsText(spatialCombiner, "evaluateSpatialReservoirGiCandidate(currentPrimaryPosition") ||
	    !containsText(spatialCombiner, "float spatialWeight = spatialCandidate.targetWeight") ||
	    containsText(spatialCombiner, "max(spatialRecord.selectedWeight") ||
	    containsText(spatialCombiner, "pathTracerLuminance(spatialRecord.contribution)"))
	{
		std::cerr << "reservoir GI spatial combiner must select using current-domain evaluated target weights\n";
		return false;
	}
	const std::size_t spatialBudgetGatePos =
	    spatialCombiner.find("reservoirGiSpatialBudgetPass");
	const std::size_t spatialBudgetedLoadPos =
	    spatialCombiner.find("loadTemporalReservoirGi(uint2(uint(neighbor.x), uint(neighbor.y))");
	if (spatialBudgetGatePos == std::string::npos ||
	    spatialBudgetedLoadPos == std::string::npos ||
	    spatialBudgetGatePos > spatialBudgetedLoadPos)
	{
		std::cerr << "spatial budget gate must run before full spatial neighbor reservoir load\n";
		return false;
	}
	const std::size_t temporalReprojectFailPos =
	    temporalBranch.find("if (!reprojectReservoirHistoryPixel(hitPos, launchSize, temporalPixel))");
	const std::size_t temporalLoadPos = temporalBranch.find("loadTemporalReservoirGi(temporalPixel, launchSize");
	if (temporalReprojectFailPos == std::string::npos ||
	    temporalLoadPos == std::string::npos ||
	    temporalReprojectFailPos > temporalLoadPos)
	{
		std::cerr << "reservoir GI temporal branch must handle reprojection failure before loading history\n";
		return false;
	}
	const std::string reprojectFailureBranch = temporalBranch.substr(temporalReprojectFailPos,
	                                                                temporalLoadPos - temporalReprojectFailPos);
	if (!containsText(reprojectFailureBranch, "reservoirGiDetailedDiagnostics") ||
	    !containsText(reprojectFailureBranch, "reservoirGiTemporalRejectGeometryOffset") ||
	    !containsText(reprojectFailureBranch, "reservoirGiTemporalRejectedOffset"))
	{
		std::cerr << "reservoir GI temporal reprojection failure must count rejected buckets and gate geometry diagnostics\n";
		return false;
	}
	const std::string temporalLoadCall = temporalBranch.substr(temporalLoadPos, 180);
	if (!containsText(temporalLoadCall, "reservoirGiDetailedDiagnostics") ||
	    !containsText(temporalLoadCall, "true") ||
	    !containsText(temporalLoadCall, "temporalRecord"))
	{
		std::cerr << "reservoir GI temporal history load must explicitly enable temporal reject counters\n";
		return false;
	}
	const std::size_t spatialLoaderCallPos =
	    raygen.find("loadTemporalReservoirGi(uint2(uint(neighbor.x), uint(neighbor.y))", spatialCombinerPos);
	if (spatialLoaderCallPos == std::string::npos)
	{
		std::cerr << "reservoir GI spatial combiner history load is missing\n";
		return false;
	}
	const std::string spatialLoadCall = raygen.substr(spatialLoaderCallPos, 260);
	if (!containsText(spatialLoadCall, "reservoirGiDetailedDiagnostics") ||
	    !containsText(spatialLoadCall, "false,") ||
	    containsText(spatialLoadCall, "reservoirGiTemporalRejectGeometryOffset") ||
	    containsText(spatialLoadCall, "reservoirGiTemporalRejectLightOffset") ||
	    containsText(spatialLoadCall, "reservoirGiTemporalRejectVisibilityOffset"))
	{
		std::cerr << "reservoir GI spatial neighbor load must suppress temporal reject counters\n";
		return false;
	}

	const char *requiredCppSymbols[] = {
	    "kReservoirGiRecordSize = 160"};
	for (const char *symbol : requiredCppSymbols)
	{
		if (!containsText(frameContextHeader, symbol) && !containsText(frameContextSource, symbol))
		{
			std::cerr << "missing reservoir GI measurement C++ contract: " << symbol << "\n";
			return false;
		}
	}
	if (containsText(frameContextHeader, "reservoirGiConfidenceM") ||
	    containsText(frameContextSource, "reservoirGiConfidenceM"))
	{
		std::cerr << "reservoir GI confidence should remain shader record state, not dead CPU state\n";
		return false;
	}

	struct CounterOffsetExpectation
	{
		const char *name;
		std::size_t offset;
		std::size_t expectedOffset;
	};
	const CounterOffsetExpectation counterOffsets[] = {
	    {"reservoirGiTargetWeightScaledSum",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiTargetWeightScaledSum), 124u},
	    {"reservoirGiSelectedLocal",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiSelectedLocal), 128u},
	    {"reservoirGiSelectedTemporal",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiSelectedTemporal), 132u},
	    {"reservoirGiSelectedSpatial",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiSelectedSpatial), 136u},
	    {"reservoirGiConfidenceMScaledSum",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiConfidenceMScaledSum), 140u},
	    {"reservoirGiLocalSurfaceHits",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiLocalSurfaceHits), 144u},
	    {"reservoirGiLocalValidSamples",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiLocalValidSamples), 148u},
	    {"reservoirGiLocalShadowRays",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiLocalShadowRays), 152u},
	    {"reservoirGiTemporalReconnectRays",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiTemporalReconnectRays), 156u},
	    {"reservoirGiTemporalShadowRays",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiTemporalShadowRays), 160u}};
	for (const auto &counterOffset : counterOffsets)
	{
		if (counterOffset.offset != counterOffset.expectedOffset)
		{
			std::cerr << "reservoir GI analysis counter offset mismatch for "
			          << counterOffset.name << ": expected " << counterOffset.expectedOffset
			          << ", got " << counterOffset.offset << "\n";
			return false;
		}
	}

	const char *requiredEngineSymbols[] = {
	    "reservoirGiCandidateRays=%.1f",
	    "localSurfaceHits=%.1f",
	    "localValid=%.1f",
	    "localShadowRays=%.1f",
	    "temporalReconnectRays=%.1f",
	    "temporalShadowRays=%.1f",
	    "reservoirGiConfidenceMAvg=%.5f",
	    "reservoirGiConfidenceMAvg",
	    "Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N",
	    "reservoirMixedTemporalSpatialTwoNeighborRow.reservoirGiMode = UISystem::PathTracerReservoirGiMode::TemporalSpatial",
	    "reservoirMixedTemporalSpatialTwoNeighborRow.reservoirGiSpatialNeighborCount = 2",
	    "PT Experiment Row Summary:",
	    "ptExperimentCompletionLog",
	    "PT Experiment Sweep: Sponza PT/GI audit sweep complete"};
	for (const char *symbol : requiredEngineSymbols)
	{
		if (!containsText(engineCore, symbol))
		{
			std::cerr << "missing reservoir GI measurement engine contract: " << symbol << "\n";
			return false;
		}
	}

	const char *requiredCounterAndUiSymbols[] = {
	    "uint32_t reservoirGiConfidenceMScaledSum = 0",
	    "uint32_t reservoirGiLocalSurfaceHits = 0",
	    "uint32_t reservoirGiLocalValidSamples = 0",
	    "uint32_t reservoirGiLocalShadowRays = 0",
	    "uint32_t reservoirGiTemporalReconnectRays = 0",
	    "uint32_t reservoirGiTemporalShadowRays = 0",
	    "float reservoirGiConfidenceMAvg = 0.0f",
	    "float reservoirGiLocalValidRatio = 0.0f",
	    "Reservoir GI Confidence M Avg",
	    "Reservoir GI Local Valid",
	    "Reservoir GI Local Shadow Rays",
	    "Reservoir GI Temporal Reconnect Rays"};
	for (const char *symbol : requiredCounterAndUiSymbols)
	{
		if (!containsText(engineAuxiliaryHeader, symbol) &&
		    !containsText(uiHeader, symbol) &&
		    !containsText(uiSource, symbol))
		{
			std::cerr << "missing reservoir GI confidence M counter/UI contract: " << symbol << "\n";
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
	    "Reservoir Spatial Neighbors",
	    "Reservoir Candidate Surface Hits",
	    "Reservoir Candidate Sun Visible",
	    "Reservoir Candidate Positive Weight",
	    "Reservoir GI Zero Weight",
	    "Reservoir GI Selected Weight Avg",
	    "Reservoir GI Target Weight Avg",
	    "Reservoir GI Selected Local",
	    "Reservoir GI Selected Temporal",
	    "Reservoir GI Selected Spatial",
	    "Reservoir GI Accepted Avg Luma",
	    "Reservoir GI Accepted Luma Sum",
	    "Reservoir GI Temporal Reuse",
	    "geometry",
	    "visibility",
	    "light",
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
	    "ReservoirGiTargetEvaluation",
	    "evaluateReservoirGiTargetAtPrimary",
	    "directionalWeight = primaryCosine / max(sourcePdf, 0.0001)",
	    "bsdfAtPrimary =",
	    "targetWeight = pathTracerLuminance",
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
	    "PT_MATERIAL_RESERVOIR_PROPOSAL_SHIFT",
	    "PT_MATERIAL_RESERVOIR_PROPOSAL_MASK",
	    "PT_MATERIAL_RESERVOIR_MODE_SHIFT",
	    "PT_FLAGS_ENVIRONMENT_NEE_BIT",
	    "PT_FLAGS_FIRST_HIT_PROBE_SAMPLING_SHIFT",
	    "sampleReservoirGiProposalDirection",
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
	    "reconnectTemporalReservoirGi",
	    "TraceRay(tlas, RAY_FLAG_NONE",
	    "evaluateReservoirGiSecondarySunSuffix(reconnectPayload.hitPos",
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
	    "reservoirGiTargetWeightScaledSum",
	    "reservoirGiTemporalReuseAttemptsOffset",
	    "reservoirGiTemporalRejectGeometryOffset",
	    "reservoirGiTemporalRejectVisibilityOffset",
	    "reservoirGiTemporalRejectLightOffset",
	    "reservoirGiTemporalAccepted",
	    "reservoirGiTemporalRejected",
	    "reservoirGiTemporalReuseAttempts",
	    "reservoirGiTemporalRejectGeometry",
	    "reservoirGiTemporalRejectVisibility",
	    "reservoirGiTemporalRejectLight",
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
	    "reservoirGiAcceptedAvgLuma=",
	    "reservoirGiAcceptedLumaSum=",
	    "temporalAccepted=%.1f",
	    "reservoirGiTemporalAccepted",
	    "reservoirGiTemporalRejected",
	    "reservoirGiTemporalReuseAttempts",
	    "reservoirGiTemporalRejectGeometry",
	    "reservoirGiTemporalRejectVisibility",
	    "reservoirGiTemporalRejectLight",
	    "reservoirGiSpatialNeighborCount",
	    "reservoirGiSpatialAccepted",
	    "reservoirGiSpatialRejected",
	    "reservoirGiAcceptedAvgLuma",
	    "reservoirGiAcceptedLumaSum",
	    "reservoirGiCandidateSurfaceHitRatio",
	    "reservoirGiCandidateSunVisibleRatio",
	    "reservoirGiCandidatePositiveWeightRatio",
	    "reservoirGiSelectedWeightAverage",
	    "reservoirGiTargetWeightAverage",
	    "reservoirGiSelectedLocal",
	    "reservoirGiSelectedTemporal",
	    "reservoirGiSelectedSpatial",
	    "reservoirGiSelectedLocal=",
	    "reservoirGiTargetWeightAvg=",
	    "reservoirGiLumaScaledSum",
	    "normalizeReservoirGiProposalMode",
	    "packPathTracerMaterialSettings",
	    "packPathTracerFlags",
	    "kPtMaterialReservoirProposalShift",
	    "kPtMaterialReservoirProposalMask",
	    "kPtFlagsEnvironmentNeeBit",
	    "kPtFlagsReservoirGiDetailedDiagnosticsBit",
	    "kPtFlagsReservoirTemporalBudgetShift",
	    "kPtFlagsReservoirSpatialBudgetShift",
	    "if (divisor <= 3)",
	    "kPtFlagsFirstHitProbeSamplingShift",
	    "reservoirGiDetailedDiagnostics",
	    "reservoirGiTemporalBudgetDivisor",
	    "reservoirGiSpatialBudgetDivisor",
	    "Temporal Reuse Budget Divisor",
	    "Spatial Reuse Budget Divisor",
	    "Detailed Reservoir Diagnostics",
	    "settings.reservoirGiDetailedDiagnostics = false",
	    "settings.reservoirGiTemporalBudgetDivisor",
	    "settings.reservoirGiSpatialBudgetDivisor",
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
	    "Base 3 Sun First",
	    "Base 5 Sun First",
	    "Base 8 Sun First",
	    "Base 8 Sun All",
	    "Reservoir 1C Shadowed Sun First Mixed",
	    "Reservoir 1C Shadowed Sun First Mixed Temporal",
	    "Reservoir 1C Shadowed Sun First Mixed Temporal Budget 2",
	    "Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N",
	    "Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2",
	    "reservoirTemporalBudget=%d",
	    "reservoirSpatialBudget=%d",
	    "reservoirMixedTemporalBudget2Row.reservoirGiTemporalBudgetDivisor = 2",
	    "reservoirMixedTemporalSpatialBudget2Row.reservoirGiSpatialBudgetDivisor = 2",
	    "ptBenchmarkBasePosition",
	    "ui.lightDirection",
	    "makeScenarioRowName",
	    "base3SunFirstRow.pathTracerMaxBounces = 3",
	    "base5SunFirstRow.pathTracerMaxBounces = 5",
	    "base8SunFirstRow.pathTracerMaxBounces = 8",
	    "base8SunAllRow.pathTracerMaxBounces = 8",
	    "uint32_t padding3",
	    "rtPush.padding3 = pathTracerFlags",
	    "packPathTracerMaterialSettings(ui.pathTracerSettings)",
	    "packPathTracerFlags(ui.pathTracerSettings)",
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
	    containsText(engineSource, "Reservoir 1C Shadowed Sun All") ||
	    containsText(engineSource, "Reservoir 1C Shadowed Sun First\", 1") ||
	    containsText(engineSource, "Reservoir 1C Shadowed Sun First Sun Guided") ||
	    containsText(engineSource, "Reservoir 2C Shadowed Sun First Mixed") ||
	    containsText(engineSource, "Reservoir 2C Shadowed Sun First Mixed RIS") ||
	    containsText(engineSource, "Reservoir 1C Shadowed Sun First Light Region") ||
	    containsText(engineSource, "Cache Chosen Radius 14 Budget 1") ||
	    containsText(engineSource, "makeCacheChosenRow"))
	{
		std::cerr << "focused Sponza PT/GI audit sweep should only run base audit and current ReSTIR variants\n";
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

	const char *forbiddenLightRegionGuidedSymbols[] = {
	    "Light Region Guided",
	    "LightRegionGuided",
	    "reservoirGiLightRegionTarget",
	    "reservoirGiLightRegionRadius",
	    "lightRegionTarget",
	    "lightRegionRadius",
	    "RESERVOIR_GI_PROPOSAL_LIGHT_REGION_GUIDED",
	    "sampleReservoirGiLightRegionDirection",
	    "uniformSampleCone",
	    "lightRegionTargetRadius",
	    "rtPush.skyData = glm::vec4(pathTracerSettings.reservoirGiLightRegionTarget"};
	for (const char *symbol : forbiddenLightRegionGuidedSymbols)
	{
		if (containsText(activeCacheCleanupSources, symbol))
		{
			std::cerr << "fixed light-region guided proposal path remains in active source: " << symbol << "\n";
			return false;
		}
	}

	const char *forbiddenTemporalLumaGateSymbols[] = {
	    "RESERVOIR_GI_TEMPORAL_REUSE_LUMA_THRESHOLD"};
	for (const char *symbol : forbiddenTemporalLumaGateSymbols)
	{
		if (containsText(raygen, symbol))
		{
			std::cerr << "heuristic temporal luma gate remains in reservoir shader: " << symbol << "\n";
			return false;
		}
	}

	return true;
}
