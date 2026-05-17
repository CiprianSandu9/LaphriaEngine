#include "PathTracerAnalysisTests.h"

#include "../src/Core/EngineAuxiliary.h"
#include "../src/Core/PathTracerAnalysis.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
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

std::optional<unsigned> extractUnsignedAssignment(const std::string &source,
                                                  const char *name)
{
	const std::regex assignmentRegex(
	    std::string(R"(\b)") + name + R"(\s*=\s*([0-9]+)\s*u?\s*;)");
	std::smatch match;
	if (!std::regex_search(source, match, assignmentRegex))
	{
		return std::nullopt;
	}
	return static_cast<unsigned>(std::stoul(match[1].str()));
}

std::optional<std::pair<int, int>> extractSurfelEvalCandidateSliderRange(
    const std::string &source)
{
	const std::regex sliderRegex(
	    R"(SliderInt\(\s*"Surfel Eval Candidates"\s*,\s*&pathTracerSettings\.surfelGiMaxEvalCandidates\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*\))");
	std::smatch match;
	if (!std::regex_search(source, match, sliderRegex))
	{
		return std::nullopt;
	}
	return std::pair<int, int>{std::stoi(match[1].str()), std::stoi(match[2].str())};
}

std::string extractFunctionBody(const std::string &source, const char *signature)
{
	const size_t signaturePos = source.find(signature);
	if (signaturePos == std::string::npos)
	{
		return {};
	}
	const size_t bodyStart = source.find('{', signaturePos);
	if (bodyStart == std::string::npos)
	{
		return {};
	}

	int depth = 0;
	for (size_t i = bodyStart; i < source.size(); ++i)
	{
		if (source[i] == '{')
		{
			++depth;
		}
		else if (source[i] == '}')
		{
			--depth;
			if (depth == 0)
			{
				return source.substr(bodyStart, i - bodyStart + 1u);
			}
		}
	}
	return {};
}

std::string stripComments(const std::string &source)
{
	std::string stripped;
	stripped.reserve(source.size());

	bool inLineComment = false;
	bool inBlockComment = false;
	for (size_t i = 0; i < source.size(); ++i)
	{
		if (inLineComment)
		{
			if (source[i] == '\n')
			{
				inLineComment = false;
				stripped.push_back(source[i]);
			}
			continue;
		}
		if (inBlockComment)
		{
			if (source[i] == '*' && i + 1u < source.size() && source[i + 1u] == '/')
			{
				inBlockComment = false;
				++i;
			}
			continue;
		}
		if (source[i] == '/' && i + 1u < source.size() && source[i + 1u] == '/')
		{
			inLineComment = true;
			++i;
			continue;
		}
		if (source[i] == '/' && i + 1u < source.size() && source[i + 1u] == '*')
		{
			inBlockComment = true;
			++i;
			continue;
		}
		stripped.push_back(source[i]);
	}

	return stripped;
}

bool brightSurfelCombineUsesTargetWeight(const std::string &raygen)
{
	const std::string reservoirSampling =
	    extractFunctionBody(raygen, "FirstHitDiffuseBounceResult sampleFirstHitReservoirGiSingleFrame(");
	if (reservoirSampling.empty())
	{
		return false;
	}
	const std::size_t sourcePos =
	    reservoirSampling.find("RESERVOIR_GI_SOURCE_BRIGHT_SURFEL");
	if (sourcePos == std::string::npos)
	{
		return false;
	}
	const std::size_t combinePos =
	    reservoirSampling.rfind("combineReservoirGiCandidate(surfelRecord,", sourcePos);
	if (combinePos == std::string::npos)
	{
		return false;
	}
	const std::string combineSnippet =
	    reservoirSampling.substr(combinePos, sourcePos - combinePos + sizeof("RESERVOIR_GI_SOURCE_BRIGHT_SURFEL"));
	return containsText(combineSnippet, "surfelRecord.targetWeight");
}

bool requireBrightSurfelProposalDisabledForSweeps(const std::string &raygen,
                                                  const std::string &engineCore)
{
	if (!containsText(raygen, "static const int RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_BRIGHT_SURFEL"))
		return false;

	const std::string reservoirMain =
	    extractFunctionBody(raygen, "FirstHitDiffuseBounceResult sampleFirstHitReservoirGiSingleFrame(");
	if (reservoirMain.empty())
		return false;

	if (!containsText(reservoirMain, "const bool enableBrightSurfelProposal = false"))
		return false;

	if (!containsText(reservoirMain,
	                  "enableBrightSurfelProposal &&\n"
	                  "        reservoirGiProposalMode == RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_BRIGHT_SURFEL"))
		return false;

	if (containsText(engineCore, "reservoirMixedTemporalSpatialBudget2SunReceiverBrightSurfelRow") ||
	    containsText(engineCore, "reservoirMixedSingleFrameSunReceiverBrightSurfelRow"))
		return false;

	return true;
}

bool requirePersistentSurfelCounterLayout(const std::string &engineAuxiliaryHeader,
                                          const std::string &surfelCommon,
                                          const std::string &cmakeLists)
{
	const char *requiredCounters[] = {
	    "surfelGiClearDispatches",
	    "surfelGiGenerateAttempts",
	    "surfelGiGenerated",
	    "surfelGiGenerateRejectInvalid",
	    "surfelGiGenerateRejectCoverage",
	    "surfelGiCellInsertAttempts",
	    "surfelGiCellInserted",
	    "surfelGiCellOverflow",
	    "surfelGiEvalAttempts",
	    "surfelGiEvalCellEmpty",
	    "surfelGiEvalCandidates",
	    "surfelGiEvalAccepted"};

	for (const char *counter : requiredCounters)
	{
		if (!containsText(engineAuxiliaryHeader, counter))
			return false;
	}

	const char *requiredOffsets[] = {
	    "surfelGiClearDispatchesOffset",
	    "surfelGiGenerateAttemptsOffset",
	    "surfelGiGeneratedOffset",
	    "surfelGiGenerateRejectInvalidOffset",
	    "surfelGiGenerateRejectCoverageOffset",
	    "surfelGiCellInsertAttemptsOffset",
	    "surfelGiCellInsertedOffset",
	    "surfelGiCellOverflowOffset",
	    "surfelGiEvalAttemptsOffset",
	    "surfelGiEvalCellEmptyOffset",
	    "surfelGiEvalCandidatesOffset",
	    "surfelGiEvalAcceptedOffset"};

	for (const char *offset : requiredOffsets)
	{
		if (!containsText(surfelCommon, offset))
			return false;
	}

	return containsText(cmakeLists, "SURFEL_SHADER_INCLUDE_DEPS") &&
	       containsText(cmakeLists, "SurfelCommon.slang") &&
	       containsText(cmakeLists, "ShaderCommon.slang");
}

bool requirePersistentSurfelFrameResources(const std::string &frameContextHeader,
                                           const std::string &frameContextSource)
{
	const char *headerSymbols[] = {
	    "kSurfelGiMaxSurfels",
	    "kSurfelGiGridDim",
	    "kSurfelGiCellSlotCount",
	    "surfelGiRecordBuffers",
	    "surfelGiCellBuffers",
	    "surfelGiCellSlotBuffers",
	    "surfelGiCounterBuffers",
	    "surfelGiDebugImages",
	    "surfelGiDebugImageViews",
	    "createSurfelGiBuffers"};

	for (const char *symbol : headerSymbols)
	{
		if (!containsText(frameContextHeader, symbol))
			return false;
	}

	const char *sourceSymbols[] = {
	    "void FrameContext::createSurfelGiBuffers",
	    "vk::MemoryPropertyFlagBits::eDeviceLocal",
	    "vk::BufferUsageFlagBits::eStorageBuffer",
	    "vk::ImageUsageFlagBits::eStorage",
	    "surfelGiRecordBuffers.clear()",
	    "surfelGiCellSlotBuffers.clear()",
	    "surfelGiDebugImages.clear()"};

	for (const char *symbol : sourceSymbols)
	{
		if (!containsText(frameContextSource, symbol))
			return false;
	}

	const std::string cleanupSwapchain =
	    extractFunctionBody(frameContextSource, "void FrameContext::cleanupSwapChainDependents(");
	if (cleanupSwapchain.empty())
		return false;

	const std::size_t debugViewClearPos = cleanupSwapchain.find("surfelGiDebugImageViews.clear()");
	const std::size_t debugImageDestroyPos =
	    cleanupSwapchain.find("destroyImagesAndReleaseAllocations(surfelGiDebugImages)");
	if (debugViewClearPos == std::string::npos || debugImageDestroyPos == std::string::npos ||
	    debugViewClearPos > debugImageDestroyPos)
		return false;

	const char *resizePersistentBufferDestroys[] = {
	    "destroyBuffersAndReleaseAllocations(surfelGiRecordBuffers)",
	    "destroyBuffersAndReleaseAllocations(surfelGiCellBuffers)",
	    "destroyBuffersAndReleaseAllocations(surfelGiCellSlotBuffers)",
	    "destroyBuffersAndReleaseAllocations(surfelGiCounterBuffers)"};

	for (const char *destroyCall : resizePersistentBufferDestroys)
	{
		if (containsText(cleanupSwapchain, destroyCall))
			return false;
	}

	const std::string createSurfelBuffers =
	    extractFunctionBody(frameContextSource, "void FrameContext::createSurfelGiBuffers(");
	if (createSurfelBuffers.empty())
		return false;

	if (!containsText(createSurfelBuffers, "const bool createFixedBuffers") ||
	    !containsText(createSurfelBuffers, "if (createFixedBuffers)"))
		return false;

	return true;
}

bool requireSurfelClearPassContracts(const std::string &pipelineHeader,
                                     const std::string &pipelineSource,
                                     const std::string &engineHeader,
                                     const std::string &engineCore,
                                     const std::string &surfelClear)
{
	const char *pipelineSymbols[] = {
	    "surfelGiDescriptorSetLayout",
	    "surfelGiPipelineLayout",
	    "surfelGiClearPipeline",
	    "createSurfelGiDescriptorSetLayout",
	    "createSurfelGiClearPipeline"};

	for (const char *symbol : pipelineSymbols)
	{
		if (!containsText(pipelineHeader, symbol) && !containsText(pipelineSource, symbol))
			return false;
	}

	const char *engineSymbols[] = {
	    "createSurfelGiDescriptorSets",
	    "recordSurfelGiClearPass"};

	for (const char *symbol : engineSymbols)
	{
		if (!containsText(engineHeader, symbol) && !containsText(engineCore, symbol))
			return false;
	}

	return containsText(surfelClear, "void surfelClearMain") &&
	       containsText(surfelClear, "surfelGiClearDispatchesOffset");
}

bool requireSurfelGeneratePassContracts(const std::string &cmakeLists,
                                        const std::string &pipelineHeader,
                                        const std::string &pipelineSource,
                                        const std::string &engineHeader,
                                        const std::string &engineCore,
                                        const std::string &surfelGenerate)
{
	const char *requiredSymbols[] = {
	    "surfelGiGeneratePipeline",
	    "createSurfelGiGeneratePipeline",
	    "void surfelGenerateMain",
	    "surfelGiGenerateAttemptsOffset",
	    "surfelGiGeneratedOffset",
	    "rtGBufferNormalsViews",
	    "rtGBufferDepthViews"};

	for (const char *symbol : requiredSymbols)
	{
		if (!containsText(cmakeLists, symbol) &&
		    !containsText(pipelineHeader, symbol) &&
		    !containsText(pipelineSource, symbol) &&
		    !containsText(engineHeader, symbol) &&
		    !containsText(engineCore, symbol) &&
		    !containsText(surfelGenerate, symbol))
			return false;
	}

	return containsText(cmakeLists, "SurfelGenerate.slang|surfelGenerateMain") &&
	       containsText(surfelGenerate, "[shader(\"compute\")]") &&
	       containsText(surfelGenerate, "[numthreads(8, 8, 1)]") &&
	       containsText(surfelGenerate, "SurfelGiPushConstants") &&
	       containsText(surfelGenerate, "push.renderWidth") &&
	       containsText(surfelGenerate, "push.renderHeight") &&
	       containsText(engineCore, "commandBuffer.pushConstants<SurfelGiPushConstants>") &&
	       containsText(pipelineSource, "sizeof(SurfelGiPushConstants)") &&
	       containsText(surfelGenerate, "surfelGiGenerateRejectInvalidOffset") &&
	       containsText(surfelGenerate, "surfelGiGenerateRejectCoverageOffset");
}

bool requireSurfelBuildCellsPassContracts(const std::string &cmakeLists,
                                          const std::string &pipelineHeader,
                                          const std::string &pipelineSource,
                                          const std::string &engineHeader,
                                          const std::string &engineCore,
                                          const std::string &surfelBuildCells)
{
	const char *requiredSymbols[] = {
	    "surfelGiBuildCellsPipeline",
	    "createSurfelGiBuildCellsPipeline",
	    "void surfelBuildCellsMain",
	    "surfelGiCellInsertAttemptsOffset",
	    "surfelGiCellInsertedOffset",
	    "surfelGiCellOverflowOffset"};

	for (const char *symbol : requiredSymbols)
	{
		if (!containsText(cmakeLists, symbol) &&
		    !containsText(pipelineHeader, symbol) &&
		    !containsText(pipelineSource, symbol) &&
		    !containsText(engineHeader, symbol) &&
		    !containsText(engineCore, symbol) &&
		    !containsText(surfelBuildCells, symbol))
			return false;
	}

	const std::string buildCellsMain =
	    stripComments(extractFunctionBody(surfelBuildCells, "void surfelBuildCellsMain("));
	if (buildCellsMain.empty())
		return false;

	return containsText(cmakeLists, "SurfelBuildCells.slang|surfelBuildCellsMain") &&
	       containsText(surfelBuildCells, "[shader(\"compute\")]") &&
	       containsText(surfelBuildCells, "[numthreads(128, 1, 1)]") &&
	       !containsText(surfelBuildCells, "Contract anchor for Task 6") &&
	       !containsText(surfelBuildCells, "// Contract anchor") &&
	       containsText(buildCellsMain, "InterlockedAdd(cell.count");
}

bool requireSurfelEvaluatePassContracts(const std::string &cmakeLists,
                                        const std::string &pipelineHeader,
                                        const std::string &pipelineSource,
                                        const std::string &engineHeader,
                                        const std::string &engineCore,
                                        const std::string &denoiser,
                                        const std::string &surfelEvaluate)
{
	const char *requiredSymbols[] = {
	    "surfelGiEvaluatePipeline",
	    "createSurfelGiEvaluatePipeline",
	    "void surfelEvaluateMain",
	    "SURFEL_GI_MAX_EVAL_CANDIDATES",
	    "surfelGiEvalAttemptsOffset",
	    "surfelGiEvalCellEmptyOffset",
	    "surfelGiEvalCandidatesOffset",
	    "surfelGiEvalAcceptedOffset"};

	for (const char *symbol : requiredSymbols)
	{
		if (!containsText(cmakeLists, symbol) &&
		    !containsText(pipelineHeader, symbol) &&
		    !containsText(pipelineSource, symbol) &&
		    !containsText(engineHeader, symbol) &&
		    !containsText(engineCore, symbol) &&
		    !containsText(surfelEvaluate, symbol))
			return false;
	}

	return containsText(cmakeLists, "SurfelEvaluate.slang|surfelEvaluateMain") &&
	       containsText(surfelEvaluate, "[shader(\"compute\")]") &&
	       containsText(surfelEvaluate, "[numthreads(8, 8, 1)]") &&
	       containsText(surfelEvaluate, "SurfelGiPushConstants") &&
	       containsText(surfelEvaluate, "push.renderWidth") &&
	       containsText(surfelEvaluate, "push.renderHeight") &&
	       containsText(surfelEvaluate, "push.maxEvalCandidates") &&
	       containsText(surfelEvaluate, "clamp(push.maxEvalCandidates, 1u, SURFEL_GI_MAX_EVAL_CANDIDATES)") &&
	       containsText(engineCore, "recordSurfelGiEvaluatePass") &&
	       containsText(engineCore, "commandBuffer.pushConstants<SurfelGiPushConstants>") &&
	       containsText(engineCore, "ui.pathTracerSettings.surfelGiMaxEvalCandidates") &&
	       containsText(denoiser, "surfelGiDebugView") &&
	       containsText(denoiser, "DEBUG_AOV_SURFEL_GI_OCCUPANCY") &&
	       containsText(denoiser, "DEBUG_AOV_SURFEL_GI_GATHER") &&
	       containsText(pipelineSource, "binding = 15") &&
	       containsText(engineCore, "dstBinding      = 15") &&
	       containsText(engineCore, "frames.surfelGiDebugImageViews[i]");
}

std::string extractDebugAovBranch(const std::string &denoiser,
                                  const char *debugAovSymbol)
{
	const std::string selector =
	    extractFunctionBody(denoiser, "float3 selectPathTracerDebugAovOutput(");
	if (selector.empty())
	{
		return {};
	}

	const std::size_t symbolPos = selector.find(debugAovSymbol);
	if (symbolPos == std::string::npos)
	{
		return {};
	}

	const std::size_t branchStart = selector.rfind("} else if", symbolPos);
	if (branchStart == std::string::npos)
	{
		return {};
	}
	const std::size_t branchBodyStart = selector.find('{', branchStart);
	if (branchBodyStart == std::string::npos)
	{
		return {};
	}
	const std::size_t nextBranch = selector.find("} else if", branchBodyStart + 1u);
	if (nextBranch == std::string::npos)
	{
		const std::size_t selectorReturn = selector.find("return outColor", branchBodyStart);
		return selectorReturn == std::string::npos
		           ? selector.substr(branchStart)
		           : selector.substr(branchStart, selectorReturn - branchStart);
	}
	return selector.substr(branchStart, nextBranch - branchStart);
}

bool requireSurfelGiDebugAovChannelSplit(const std::string &denoiser)
{
	const std::string occupancyBranch =
	    stripComments(extractDebugAovBranch(denoiser, "DEBUG_AOV_SURFEL_GI_OCCUPANCY"));
	const std::string gatherBranch =
	    stripComments(extractDebugAovBranch(denoiser, "DEBUG_AOV_SURFEL_GI_GATHER"));
	if (occupancyBranch.empty() || gatherBranch.empty())
	{
		std::cerr << "missing surfel GI debug AOV denoiser branches\n";
		return false;
	}

	const bool occupancyUsesBlue =
	    containsText(occupancyBranch, "surfelGiDebugView[pixel].b") ||
	    containsText(occupancyBranch, "surfelGiDebugView[pixel].z") ||
	    containsText(occupancyBranch, ".b") ||
	    containsText(occupancyBranch, ".z");
	const bool gatherUsesRed =
	    containsText(gatherBranch, "surfelGiDebugView[pixel].r") ||
	    containsText(gatherBranch, "surfelGiDebugView[pixel].x") ||
	    containsText(gatherBranch, ".r") ||
	    containsText(gatherBranch, ".x");
	const bool occupancyLooksScalar =
	    containsText(occupancyBranch, "float3(") &&
	    !containsText(occupancyBranch, "surfelGiDebugView[pixel].rgb") &&
	    !containsText(occupancyBranch, "surfelGiDebugView[pixel].xyz");
	const bool gatherLooksScalar =
	    containsText(gatherBranch, "float3(") &&
	    !containsText(gatherBranch, "surfelGiDebugView[pixel].rgb") &&
	    !containsText(gatherBranch, "surfelGiDebugView[pixel].xyz");

	if (!occupancyUsesBlue || !gatherUsesRed || !occupancyLooksScalar || !gatherLooksScalar)
	{
		std::cerr << "surfel GI occupancy and gather AOVs must present distinct scalar channels\n";
		return false;
	}
	return true;
}

bool requireSurfelGiSlotCapacityAlignment(const std::string &frameContextHeader,
                                          const std::string &uiSource,
                                          const std::string &surfelCommon,
                                          const std::string &surfelEvaluate)
{
	const auto shaderSlotCount = extractUnsignedAssignment(surfelCommon, "SURFEL_GI_CELL_SLOT_COUNT");
	const auto shaderMaxCandidates =
	    extractUnsignedAssignment(surfelCommon, "SURFEL_GI_MAX_EVAL_CANDIDATES");
	const auto frameSlotCount = extractUnsignedAssignment(frameContextHeader, "kSurfelGiCellSlotCount");
	const auto sliderRange = extractSurfelEvalCandidateSliderRange(uiSource);
	if (!shaderSlotCount || !shaderMaxCandidates || !frameSlotCount || !sliderRange)
	{
		std::cerr << "missing surfel GI slot/candidate capacity declarations\n";
		return false;
	}

	if (*shaderSlotCount != 16u || *shaderMaxCandidates != 16u ||
	    *frameSlotCount != *shaderSlotCount || sliderRange->first != 1 ||
	    sliderRange->second != static_cast<int>(*shaderMaxCandidates))
	{
		std::cerr << "surfel GI slot storage, shader candidate max, and UI range must align at 1..16\n";
		return false;
	}

	const std::string evaluateMain =
	    stripComments(extractFunctionBody(surfelEvaluate, "void surfelEvaluateMain("));
	if (evaluateMain.empty())
	{
		std::cerr << "missing surfel GI evaluate main body\n";
		return false;
	}
	return containsText(evaluateMain, "clamp(push.maxEvalCandidates, 1u, SURFEL_GI_MAX_EVAL_CANDIDATES)") &&
	       containsText(evaluateMain, "min(count, configuredCandidateCount)") &&
	       containsText(evaluateMain, "slot < boundedCandidateCount") &&
	       containsText(evaluateMain, "cellIndex * SURFEL_GI_CELL_SLOT_COUNT + slot");
}

bool requireIndexedBrightSurfelShaderContracts(const std::string &raygen)
{
	const std::string brightSurfelStore =
	    extractFunctionBody(raygen, "bool storeReservoirGiBrightSurfelRecord(");
	if (brightSurfelStore.empty())
	{
		std::cerr << "missing bright surfel store function body\n";
		return false;
	}
	if (!containsText(brightSurfelStore, "reservoirGiBrightSurfelIndexedStoreIndex("))
	{
		std::cerr << "bright surfel store must use indexed store lookup\n";
		return false;
	}
	if (containsText(brightSurfelStore, "reservoirGiBrightSurfelGlobalStoreIndex"))
	{
		std::cerr << "bright surfel store must not use old global store lookup\n";
		return false;
	}

	const std::string brightSurfelEvaluator =
	    extractFunctionBody(raygen, "bool evaluateBrightReceiverSurfelReservoirGiCandidate(");
	if (brightSurfelEvaluator.empty())
	{
		std::cerr << "missing bright surfel candidate evaluator function body\n";
		return false;
	}
	if (!containsText(brightSurfelEvaluator, "selectWeightedIndexedBrightReceiverSurfelRecord("))
	{
		std::cerr << "bright surfel candidate evaluator must call indexed selector\n";
		return false;
	}
	if (containsText(brightSurfelEvaluator, "selectWeightedGlobalBrightReceiverSurfelRecord"))
	{
		std::cerr << "bright surfel candidate evaluator must not call old global selector\n";
		return false;
	}

	const std::string brightSurfelSelector =
	    extractFunctionBody(raygen, "bool selectWeightedIndexedBrightReceiverSurfelRecord");
	if (brightSurfelSelector.empty())
	{
		std::cerr << "missing indexed bright surfel selector function body\n";
		return false;
	}
	const char *forbiddenSelectorSymbols[] = {
	    "RESERVOIR_GI_BRIGHT_SURFEL_GLOBAL_SCAN_COUNT",
	    "reservoirGiBrightSurfelGlobalIndex",
	    "reservoirGiBrightSurfelGlobalStoreIndex"};
	for (const char *symbol : forbiddenSelectorSymbols)
	{
		if (containsText(brightSurfelSelector, symbol))
		{
			std::cerr << "indexed bright surfel selector still uses old global lookup: "
			          << symbol << "\n";
			return false;
		}
	}
	const char *requiredSelectorCollisionSymbols[] = {
	    "precheckReservoirGiBrightSurfelHistoryRecord(surfelIndex, surfelCapacity, surfelHistoryFrameId",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelIndexedEmptyOffset, 1u)"};
	for (const char *symbol : requiredSelectorCollisionSymbols)
	{
		if (!containsText(brightSurfelSelector, symbol))
		{
			std::cerr << "missing indexed bright surfel selector collision guard: "
			          << symbol << "\n";
			return false;
		}
	}
	const std::size_t collisionPrecheckPos =
	    brightSurfelSelector.find("precheckReservoirGiBrightSurfelHistoryRecord(surfelIndex, surfelCapacity, surfelHistoryFrameId");
	const std::size_t collisionEmptyCountPos =
	    brightSurfelSelector.find("ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelIndexedEmptyOffset, 1u)",
	                              collisionPrecheckPos);
	const std::size_t collisionContinuePos = brightSurfelSelector.find("continue;", collisionPrecheckPos);
	if (collisionPrecheckPos == std::string::npos ||
	    collisionEmptyCountPos == std::string::npos ||
	    collisionContinuePos == std::string::npos ||
	    collisionEmptyCountPos > collisionContinuePos)
	{
		std::cerr << "indexed bright surfel selector collision/precheck rejection must count IndexedEmpty\n";
		return false;
	}

	const std::string brightSurfelPrecheck =
	    extractFunctionBody(raygen, "bool precheckReservoirGiBrightSurfelHistoryRecord");
	if (brightSurfelPrecheck.empty())
	{
		std::cerr << "missing indexed bright surfel cheap precheck helper body\n";
		return false;
	}
	const char *requiredPrecheckSymbols[] = {
	    "surfelIndex >= surfelCapacity",
	    "precheck.frameId > surfelHistoryFrameId",
	    "precheck.flags = ptReservoirGiBrightSurfelHistory.Load(",
	    "precheck.frameId = ptReservoirGiBrightSurfelHistory.Load(",
	    "precheck.targetWeight = asfloat(ptReservoirGiBrightSurfelHistory.Load(",
	    "precheck.position = loadFloat3FromReservoirGiBrightSurfelHistory(",
	    "reservoirGiBrightSurfelMatchesIndexedQueryCell(precheck.position, receiverPosition, cellOffsetIndex)"};
	for (const char *symbol : requiredPrecheckSymbols)
	{
		if (!containsText(brightSurfelPrecheck, symbol))
		{
			std::cerr << "missing indexed bright surfel cheap precheck contract: "
			          << symbol << "\n";
			return false;
		}
	}
	const char *forbiddenPrecheckSymbols[] = {
	    "RESERVOIR_GI_BRIGHT_SURFEL_NORMAL_OFFSET",
	    "RESERVOIR_GI_BRIGHT_SURFEL_RADIANCE_OFFSET",
	    "RESERVOIR_GI_BRIGHT_SURFEL_RADIUS_OFFSET",
	    "RESERVOIR_GI_BRIGHT_SURFEL_CONFIDENCE_OFFSET"};
	for (const char *symbol : forbiddenPrecheckSymbols)
	{
		if (containsText(brightSurfelPrecheck, symbol))
		{
			std::cerr << "indexed bright surfel cheap precheck must not load full record field: "
			          << symbol << "\n";
			return false;
		}
	}
	return true;
}

struct BrightSurfelDiagnosticContract
{
	const char *counterName;
	const char *rowFieldName;
	const char *uiLabel;
};

constexpr BrightSurfelDiagnosticContract kBrightSurfelIndexedDiagnostics[] = {
    {"reservoirGiBrightSurfelIndexedQuery",
     "brightSurfelIndexedQuery",
     "Reservoir GI Bright Surfel Indexed Query"},
    {"reservoirGiBrightSurfelIndexedEmpty",
     "brightSurfelIndexedEmpty",
     "Reservoir GI Bright Surfel Indexed Empty"},
    {"reservoirGiBrightSurfelIndexedProbe",
     "brightSurfelIndexedProbe",
     "Reservoir GI Bright Surfel Indexed Probe"},
    {"reservoirGiBrightSurfelSelectorRejectDistance",
     "brightSurfelSelectorRejectDistance",
     "Reservoir GI Bright Surfel Selector Reject Distance"},
    {"reservoirGiBrightSurfelSelectorRejectReceiverHemisphere",
     "brightSurfelSelectorRejectReceiverHemisphere",
     "Reservoir GI Bright Surfel Selector Reject Receiver Hemisphere"},
    {"reservoirGiBrightSurfelSelectorRejectSurfelHemisphere",
     "brightSurfelSelectorRejectSurfelHemisphere",
     "Reservoir GI Bright Surfel Selector Reject Surfel Hemisphere"},
    {"reservoirGiBrightSurfelSelectorRejectInvalidVector",
     "brightSurfelSelectorRejectInvalidVector",
     "Reservoir GI Bright Surfel Selector Reject Invalid Vector"}};

bool requireIndexedBrightSurfelDiagnosticPlumbing(const std::string &engineAuxiliaryHeader,
                                                  const std::string &uiHeader,
                                                  const std::string &uiSource,
                                                  const std::string &engineCore)
{
	for (const auto &diagnostic : kBrightSurfelIndexedDiagnostics)
	{
		const std::string counterField =
		    std::string("uint32_t ") + diagnostic.counterName + " = 0";
		if (!containsText(engineAuxiliaryHeader, counterField.c_str()))
		{
			std::cerr << "missing indexed bright surfel CPU counter field: "
			          << diagnostic.counterName << "\n";
			return false;
		}
		if (!containsText(uiHeader, counterField.c_str()))
		{
			std::cerr << "missing indexed bright surfel UI stat field: "
			          << diagnostic.counterName << "\n";
			return false;
		}
		if (!containsText(uiSource, diagnostic.uiLabel))
		{
			std::cerr << "missing indexed bright surfel UI label: "
			          << diagnostic.uiLabel << "\n";
			return false;
		}

		const std::string uiCopyTarget =
		    std::string("ui.pathTracerPerfStats.") + diagnostic.counterName + " =";
		const std::string uiCopySource =
		    std::string("counters->") + diagnostic.counterName;
		if (!containsText(engineCore, uiCopyTarget.c_str()) ||
		    !containsText(engineCore, uiCopySource.c_str()))
		{
			std::cerr << "missing indexed bright surfel UI counter copy path: "
			          << diagnostic.counterName << "\n";
			return false;
		}

		const std::string accumulationTarget =
		    std::string("ptExperimentAccum.") + diagnostic.rowFieldName + " +=";
		const std::string accumulationSource =
		    std::string("stats.") + diagnostic.counterName;
		if (!containsText(engineCore, accumulationTarget.c_str()) ||
		    !containsText(engineCore, accumulationSource.c_str()))
		{
			std::cerr << "missing indexed bright surfel accumulation path: "
			          << diagnostic.rowFieldName << "\n";
			return false;
		}

		const std::string rowFormat =
		    std::string(diagnostic.rowFieldName) + "=%.1f";
		const std::string rowArgument =
		    std::string("accum.") + diagnostic.rowFieldName + " * invSamples";
		if (!containsText(engineCore, rowFormat.c_str()) ||
		    !containsText(engineCore, rowArgument.c_str()))
		{
			std::cerr << "missing indexed bright surfel row-summary format/argument: "
			          << diagnostic.rowFieldName << "\n";
			return false;
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

bool testSurfelGiDiagnosticRatios();

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

	return testSurfelGiDiagnosticRatios();
}

bool testSurfelGiDiagnosticRatios()
{
	Laphria::SurfelGiDiagnosticCounters counters{};
	counters.generated = 80;
	counters.cellInserted = 60;
	counters.cellOverflow = 20;
	counters.evalCandidates = 40;
	counters.evalAccepted = 10;
	counters.evalCellEmpty = 5;

	const auto ratios = Laphria::computeSurfelGiDiagnosticRatios(counters);
	if (std::abs(ratios.cellInsertRatio - 0.75f) > 0.0001f)
	{
		std::cerr << "surfel GI cell insert ratio mismatch\n";
		return false;
	}
	if (std::abs(ratios.cellOverflowRatio - 0.25f) > 0.0001f)
	{
		std::cerr << "surfel GI cell overflow ratio mismatch\n";
		return false;
	}
	if (std::abs(ratios.evalAcceptedRatio - 0.25f) > 0.0001f)
	{
		std::cerr << "surfel GI eval accepted ratio mismatch\n";
		return false;
	}
	if (std::abs(ratios.evalCellEmptyRatio - 0.125f) > 0.0001f)
	{
		std::cerr << "surfel GI eval empty ratio mismatch\n";
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
	const std::string surfelCommon = readTextFile(sourceRoot / "src" / "shaders" / "SurfelCommon.slang");
	const std::string cmakeLists = readTextFile(sourceRoot / "CMakeLists.txt");
	const std::string uiHeader = readTextFile(sourceRoot / "src" / "Core" / "UISystem.h");
	const std::string uiSource = readTextFile(sourceRoot / "src" / "Core" / "UISystem.cpp");
	const std::string frameContextHeader = readTextFile(sourceRoot / "src" / "Core" / "FrameContext.h");
	const std::string frameContextSource = readTextFile(sourceRoot / "src" / "Core" / "FrameContext.cpp");
	const std::string pipelineHeader = readTextFile(sourceRoot / "src" / "Core" / "PipelineCollection.h");
	const std::string pipelineSource = readTextFile(sourceRoot / "src" / "Core" / "PipelineCollection.cpp");
	const std::string engineHeader = readTextFile(sourceRoot / "src" / "Core" / "EngineCore.h");
	const std::string surfelClear = readTextFile(sourceRoot / "src" / "shaders" / "SurfelClear.slang");
	const std::string surfelGenerate = readTextFile(sourceRoot / "src" / "shaders" / "SurfelGenerate.slang");
	const std::string surfelBuildCells = readTextFile(sourceRoot / "src" / "shaders" / "SurfelBuildCells.slang");
	const std::string surfelEvaluate = readTextFile(sourceRoot / "src" / "shaders" / "SurfelEvaluate.slang");
	const std::string denoiser = readTextFile(sourceRoot / "src" / "shaders" / "Denoiser.slang");
	const std::string resourceManager = readTextFile(sourceRoot / "src" / "Core" / "ResourceManager.cpp");
	const std::string gltfImporter    = readTextFile(sourceRoot / "src" / "Core" / "GltfImporter.cpp");

	if (raygen.empty() || engineCore.empty() || engineAuxiliaryHeader.empty() ||
	    uiHeader.empty() || uiSource.empty() || frameContextHeader.empty() ||
	    frameContextSource.empty() || pipelineHeader.empty() || pipelineSource.empty() ||
	    engineHeader.empty() || denoiser.empty() || resourceManager.empty() || gltfImporter.empty())
	{
		std::cerr << "failed to read reservoir GI measurement contract sources\n";
		return false;
	}

	if (!requirePersistentSurfelCounterLayout(engineAuxiliaryHeader, surfelCommon, cmakeLists))
	{
		std::cerr << "persistent surfel GI counter layout contract is incomplete\n";
		return false;
	}

	if (!requirePersistentSurfelFrameResources(frameContextHeader, frameContextSource))
	{
		std::cerr << "persistent surfel GI frame resource contract is incomplete\n";
		return false;
	}

	if (!requireSurfelClearPassContracts(pipelineHeader, pipelineSource, engineHeader, engineCore, surfelClear))
	{
		std::cerr << "persistent surfel GI clear pass contract is incomplete\n";
		return false;
	}

	if (!requireSurfelGeneratePassContracts(cmakeLists, pipelineHeader, pipelineSource,
	                                        engineHeader, engineCore, surfelGenerate))
	{
		std::cerr << "persistent surfel GI generate pass contract is incomplete\n";
		return false;
	}

	if (!requireSurfelBuildCellsPassContracts(cmakeLists, pipelineHeader, pipelineSource,
	                                          engineHeader, engineCore, surfelBuildCells))
	{
		std::cerr << "persistent surfel GI build-cells pass contract is incomplete\n";
		return false;
	}

	if (!requireSurfelEvaluatePassContracts(cmakeLists, pipelineHeader, pipelineSource,
	                                        engineHeader, engineCore, denoiser, surfelEvaluate))
	{
		std::cerr << "persistent surfel GI evaluate pass contract is incomplete\n";
		return false;
	}
	if (!requireSurfelGiDebugAovChannelSplit(denoiser))
	{
		return false;
	}
	if (!requireSurfelGiSlotCapacityAlignment(frameContextHeader, uiSource, surfelCommon, surfelEvaluate))
	{
		return false;
	}

	if (!requireBrightSurfelProposalDisabledForSweeps(raygen, engineCore))
	{
		std::cerr << "Bright surfel reservoir proposal must be disabled for sweeps before persistent surfel cache work\n";
		return false;
	}

	const std::string recordRayTracingSource =
	    extractFunctionBody(engineCore, "void EngineCore::recordRayTracingCommandBuffer(");
	if (!containsText(recordRayTracingSource, "const bool     surfelGiDebugAovSelected") ||
	    !containsText(recordRayTracingSource, "PathTracerDebugAov::SurfelGiOccupancy") ||
	    !containsText(recordRayTracingSource, "PathTracerDebugAov::SurfelGiGather"))
	{
		std::cerr << "surfel GI debug AOV selection must include occupancy and gather\n";
		return false;
	}
	const char *requiredSurfelPassOrder[] = {
	    "recordSurfelGiClearPass(commandBuffer, fi);",
	    "recordSurfelGiGeneratePass(commandBuffer, fi);",
	    "recordSurfelGiBuildCellsPass(commandBuffer, fi);",
	    "recordSurfelGiEvaluatePass(commandBuffer, fi);"};
	std::size_t surfelPassSearchPos = recordRayTracingSource.find(requiredSurfelPassOrder[0]);
	if (surfelPassSearchPos == std::string::npos)
	{
		std::cerr << "surfel GI pass gate missing first ordered call\n";
		return false;
	}
	const std::size_t gateWindowStart = surfelPassSearchPos > 500 ? surfelPassSearchPos - 500 : 0;
	const std::string gateWindow = recordRayTracingSource.substr(gateWindowStart, surfelPassSearchPos - gateWindowStart);
	if (!containsText(gateWindow, "ui.pathTracerSettings.enableSurfelGi") ||
	    !containsText(gateWindow, "ui.pathTracerSettings.surfelGiDebug") ||
	    !containsText(gateWindow, "surfelGiDebugAovSelected"))
	{
		std::cerr << "surfel GI passes must run for cache, debug toggle, or surfel debug AOV selection\n";
		return false;
	}
	for (const char *passCall : requiredSurfelPassOrder)
	{
		const std::size_t passPos = recordRayTracingSource.find(passCall, surfelPassSearchPos);
		if (passPos == std::string::npos)
		{
			std::cerr << "surfel GI pass gate missing ordered call: " << passCall << "\n";
			return false;
		}
		surfelPassSearchPos = passPos + std::string(passCall).size();
	}

	const char *requiredTask8Symbols[] = {
	    "enableSurfelGi",
	    "surfelGiDebug",
	    "Surfel GI Occupancy",
	    "Surfel GI Gather",
	    "surfelGiGenerated",
	    "surfelGiCellOverflow",
	    "surfelGiEvalAccepted"};
	for (const char *symbol : requiredTask8Symbols)
	{
		if (!containsText(uiHeader, symbol) &&
		    !containsText(uiSource, symbol) &&
		    !containsText(engineCore, symbol) &&
		    !containsText(engineHeader, symbol) &&
		    !containsText(engineAuxiliaryHeader, symbol) &&
		    !containsText(surfelCommon, symbol) &&
		    !containsText(surfelGenerate, symbol) &&
		    !containsText(surfelBuildCells, symbol) &&
		    !containsText(surfelEvaluate, symbol))
		{
			std::cerr << "missing Task 8 surfel GI UI/diagnostic symbol: " << symbol << "\n";
			return false;
		}
	}

	const char *requiredSurfelUiAndSweepSymbols[] = {
	    "bool enableSurfelGi = false",
	    "bool surfelGiDebug = false",
	    "int surfelGiMaxEvalCandidates = 8",
	    "ImGui::Checkbox(\"Surfel GI Cache\", &pathTracerSettings.enableSurfelGi)",
	    "ImGui::Checkbox(\"Surfel GI Debug\", &pathTracerSettings.surfelGiDebug)",
	    "ImGui::SliderInt(\"Surfel Eval Candidates\", &pathTracerSettings.surfelGiMaxEvalCandidates, 1, 16)",
	    "Surfel GI Generated",
	    "Surfel GI Cell Insert Attempts",
	    "Surfel GI Cell Overflow",
	    "Surfel GI Eval Attempts",
	    "Surfel GI Eval Accepted",
	    "SurfelGiOccupancy = 25",
	    "SurfelGiGather = 26",
	    "PathTracerDebugAov::SurfelGiOccupancy",
	    "PathTracerDebugAov::SurfelGiGather",
	    "surfelGiGenerated=%.1f",
	    "surfelGiCellInsertAttempts=%.1f",
	    "surfelGiCellInserted=%.1f",
	    "surfelGiCellOverflow=%.1f",
	    "surfelGiEvalAttempts=%.1f",
	    "surfelGiEvalCandidates=%.1f",
	    "surfelGiEvalAccepted=%.1f",
	    "surfelGiEvalCellEmpty=%.1f",
	    "stats.surfelGiGenerated",
	    "stats.surfelGiCellInsertAttempts",
	    "stats.surfelGiCellInserted",
	    "stats.surfelGiCellOverflow",
	    "stats.surfelGiEvalAttempts",
	    "stats.surfelGiEvalCandidates",
	    "stats.surfelGiEvalAccepted",
	    "stats.surfelGiEvalCellEmpty"};
	for (const char *symbol : requiredSurfelUiAndSweepSymbols)
	{
		if (!containsText(uiHeader, symbol) &&
		    !containsText(uiSource, symbol) &&
		    !containsText(engineCore, symbol) &&
		    !containsText(engineHeader, symbol))
		{
			std::cerr << "missing Task 8 surfel GI UI/sweep contract: " << symbol << "\n";
			return false;
		}
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
	    "float3 baseSuffixRadiance = emissiveContribution + candidateSecondarySunSuffix",
	    "shouldAttemptReservoirGiCacheContinuation",
	    "RESERVOIR_GI_CACHE_CONTINUATION_BUDGET_DIVISOR",
	    "RESERVOIR_GI_CACHE_CONTINUATION_BASE_LUMA_THRESHOLD",
	    "float3 candidateSuffixRadiance = baseSuffixRadiance + cacheContinuationSuffixRadiance",
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
	    "reservoirGiLocalMissCandidatesOffset",
	    "reservoirGiLocalMissPositiveWeightOffset",
	    "reservoirGiLocalSurfaceInvalidOffset",
	    "reservoirGiLocalRejectGeometryOffset",
	    "reservoirGiLocalRejectNoLightOffset",
	    "reservoirGiLocalRejectZeroTargetOffset",
	    "reservoirGiLocalRejectBadPdfOffset",
	    "RESERVOIR_GI_LOCAL_REJECT_GEOMETRY",
	    "RESERVOIR_GI_LOCAL_REJECT_NO_LIGHT",
	    "RESERVOIR_GI_LOCAL_REJECT_ZERO_TARGET",
	    "RESERVOIR_GI_LOCAL_REJECT_BAD_PDF",
	    "reservoirGiAcceptedLocalSurfaceOffset",
	    "reservoirGiAcceptedLocalMissOffset",
	    "reservoirGiLocalShadowRaysOffset",
	    "reservoirGiTemporalReconnectRaysOffset",
	    "reservoirGiTemporalShadowRaysOffset",
	    "ptReservoirGiBrightSurfelCurrent",
	    "ptReservoirGiBrightSurfelHistory",
	    "ReservoirGiBrightSurfelRecord",
	    "RESERVOIR_GI_BRIGHT_SURFEL_CURRENT_BINDING",
	    "RESERVOIR_GI_BRIGHT_SURFEL_HISTORY_BINDING",
	    "RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_BRIGHT_SURFEL",
	    "RESERVOIR_GI_SOURCE_BRIGHT_SURFEL",
	    "RESERVOIR_GI_BRIGHT_SURFEL_ATTEMPT_STRIDE",
	    "RESERVOIR_GI_BRIGHT_SURFEL_CANDIDATE_MIN_TARGET_WEIGHT",
	    "RESERVOIR_GI_BRIGHT_SURFEL_INDEX_CELL_SIZE",
	    "RESERVOIR_GI_BRIGHT_SURFEL_INDEX_CELL_SLOTS",
	    "RESERVOIR_GI_BRIGHT_SURFEL_INDEX_RADIUS_CELLS",
	    "storeReservoirGiBrightSurfelRecord",
	    "loadReservoirGiBrightSurfelHistoryRecord",
	    "loadReservoirGiBrightSurfelHistoryHeader",
	    "ReservoirGiBrightSurfelPrecheck",
	    "precheckReservoirGiBrightSurfelHistoryRecord",
	    "loadPrecheckedReservoirGiBrightSurfelHistoryRecord",
	    "shouldAttemptBrightReceiverSurfel",
	    "shouldTrainBrightReceiverSurfel",
	    "reservoirGiBrightSurfelIndexedStoreIndex",
	    "reservoirGiBrightSurfelIndexedQueryIndex",
	    "tryStoreBrightSurfelTrainingCandidate",
	    "selectWeightedIndexedBrightReceiverSurfelRecord",
	    "surfelSelectionPdf",
	    "Scan-local selector diagnostic only; do not feed it into target sourcePdf or reservoir weight.",
	    "BrightSurfelTargetEstimateResult",
	    "BRIGHT_SURFEL_TARGET_REJECT_NONE",
	    "BRIGHT_SURFEL_TARGET_REJECT_GEOMETRY",
	    "BRIGHT_SURFEL_TARGET_REJECT_TARGET",
	    "BRIGHT_SURFEL_TARGET_REJECT_DISTANCE",
	    "BRIGHT_SURFEL_TARGET_REJECT_RECEIVER_HEMISPHERE",
	    "BRIGHT_SURFEL_TARGET_REJECT_SURFEL_HEMISPHERE",
	    "BRIGHT_SURFEL_TARGET_REJECT_INVALID_VECTOR",
	    "estimateBrightSurfelTargetForReceiver",
	    "selectedTargetEvaluation",
	    "evaluateBrightReceiverSurfelReservoirGiCandidate",
	    "reservoirGiBrightSurfelStoreOffset",
	    "reservoirGiBrightSurfelAttemptOffset",
	    "reservoirGiBrightSurfelHitOffset",
	    "reservoirGiBrightSurfelMissOffset",
	    "reservoirGiBrightSurfelRejectVisibilityOffset",
	    "reservoirGiBrightSurfelRejectGeometryOffset",
	    "reservoirGiBrightSurfelRejectTargetOffset",
	    "reservoirGiBrightSurfelAcceptedOffset",
	    "reservoirGiSelectedBrightSurfelOffset",
	    "reservoirGiBrightSurfelPrecheckRejectTargetOffset",
	    "reservoirGiBrightSurfelTrainingAttemptOffset",
	    "reservoirGiBrightSurfelTrainingStoreOffset",
	    "reservoirGiBrightSurfelTrainingRejectGeometryOffset",
	    "reservoirGiBrightSurfelTrainingRejectTargetOffset",
	    "reservoirGiBrightSurfelSelectorRejectGeometryOffset",
	    "reservoirGiBrightSurfelSelectorRejectTargetOffset",
	    "reservoirGiBrightSurfelSelectorViableOffset",
	    "reservoirGiBrightSurfelIndexedQueryOffset",
	    "reservoirGiBrightSurfelIndexedEmptyOffset",
	    "reservoirGiBrightSurfelIndexedProbeOffset",
	    "reservoirGiBrightSurfelSelectorRejectDistanceOffset",
	    "reservoirGiBrightSurfelSelectorRejectReceiverHemisphereOffset",
	    "reservoirGiBrightSurfelSelectorRejectSurfelHemisphereOffset",
	    "reservoirGiBrightSurfelSelectorRejectInvalidVectorOffset",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiLocalSurfaceHitsOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiLocalValidSamplesOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiLocalMissCandidatesOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiLocalSurfaceInvalidOffset, 1u)",
	    "recordReservoirGiLocalReject(localRejectReason)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiAcceptedLocalSurfaceOffset, 1u)",
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
	if (containsText(raygen, "estimateBrightSurfelTargetBeforeVisibility"))
	{
		std::cerr
		    << "old bright surfel target estimator must be replaced by estimateBrightSurfelTargetForReceiver\n";
		return false;
	}
	if (!requireIndexedBrightSurfelShaderContracts(raygen))
	{
		return false;
	}
	const std::string brightSurfelSelector =
	    extractFunctionBody(raygen, "bool selectWeightedIndexedBrightReceiverSurfelRecord");
	if (brightSurfelSelector.empty())
	{
		std::cerr << "missing indexed bright surfel selector function body\n";
		return false;
	}
	const char *requiredBrightSurfelSelectorSymbols[] = {
	    "loadReservoirGiBrightSurfelHistoryHeader(surfelCapacity, surfelHistoryFrameId)",
	    "reservoirGiBrightSurfelIndexedQueryIndex(hitPos, cellOffset, slot)",
	    "precheckReservoirGiBrightSurfelHistoryRecord(surfelIndex, surfelCapacity, surfelHistoryFrameId",
	    "loadPrecheckedReservoirGiBrightSurfelHistoryRecord(surfelIndex, precheck, surfel)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelIndexedQueryOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelIndexedEmptyOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelIndexedProbeOffset, 1u)",
	    "estimateBrightSurfelTargetForReceiver(hitPos, N, V, primaryPayload",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorRejectDistanceOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorRejectReceiverHemisphereOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorRejectSurfelHemisphereOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorRejectInvalidVectorOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorRejectGeometryOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorRejectTargetOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorViableOffset, 1u)",
	    "selectedTargetEvaluation = estimate.targetEvaluation",
	    "float score = estimate.targetEvaluation.targetWeight"};
	for (const char *symbol : requiredBrightSurfelSelectorSymbols)
	{
		if (!containsText(brightSurfelSelector, symbol))
		{
			std::cerr << "missing bright surfel selector-local shader contract: " << symbol << "\n";
			return false;
		}
	}
	if (!brightSurfelCombineUsesTargetWeight(raygen))
	{
		std::cerr << "bright surfel reservoir combine must use surfelRecord.targetWeight\n";
		return false;
	}
	const char *forbiddenRaygenSymbols[] = {
	    "selectWeightedGlobalBrightReceiverSurfelRecord",
	    "reservoirGiBrightSurfelGlobalIndex",
	    "reservoirGiBrightSurfelGlobalStoreIndex",
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
	const std::size_t reservoirSamplingFunctionPos =
	    raygen.find("FirstHitDiffuseBounceResult sampleFirstHitReservoirGiSingleFrame(");
	const std::size_t postReservoirSamplingFunctionPos =
	    raygen.find("float3 debugReservoirGiContribution", reservoirSamplingFunctionPos);
	if (reservoirSamplingFunctionPos == std::string::npos ||
	    postReservoirSamplingFunctionPos == std::string::npos)
	{
		std::cerr << "reservoir GI sampling function markers are missing\n";
		return false;
	}
	const std::string reservoirSamplingFunction =
	    raygen.substr(reservoirSamplingFunctionPos, postReservoirSamplingFunctionPos - reservoirSamplingFunctionPos);
	const char *requiredReservoirMissSkipSymbols[] = {
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiLocalMissCandidatesOffset, 1u)",
	    "continue; // Local ReSTIR GI candidates must be secondary surface samples."};
	for (const char *symbol : requiredReservoirMissSkipSymbols)
	{
		if (!containsText(reservoirSamplingFunction, symbol))
		{
			std::cerr << "missing reservoir GI miss-skip contract: " << symbol << "\n";
			return false;
		}
	}
	const char *requiredHistoryGuidedReservoirSymbols[] = {
	    "bool usesReservoirGiHistoryBuffer = reservoirGiMode >= PATH_TRACER_RESERVOIR_GI_TEMPORAL ||",
	    "reservoirGiProposalMode == RESERVOIR_GI_PROPOSAL_MIXED_COSINE_HISTORY_GUIDED",
	    "if (usesReservoirGiHistoryBuffer) {",
	    "updateReservoirGiHeader(launchID, launchSize, ubo.frameCount)",
	    "bool useHistoryGuidedExtraCandidate = reservoirGiProposalMode == RESERVOIR_GI_PROPOSAL_MIXED_COSINE_HISTORY_GUIDED",
	    "int totalLocalCandidateCount = candidateCount + (useHistoryGuidedExtraCandidate ? 1 : 0)",
	    "int localProposalMode = useHistoryGuidedExtraCandidate ?",
	    "RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_GUIDED :",
	    "(useReceiverCacheReconnect ?",
	    "RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_GUIDED :",
	    "reservoirGiProposalMode)",
	    "if (useHistoryGuidedExtraCandidate && c == candidateCount)",
	    "trySampleHistoryGuidedReservoirGiProposalDirection(",
	    "storeAcceptedReservoirGiForGuidedProposal"};
	for (const char *symbol : requiredHistoryGuidedReservoirSymbols)
	{
		if (!containsText(reservoirSamplingFunction, symbol))
		{
			std::cerr << "missing history-guided reservoir buffer contract: " << symbol << "\n";
			return false;
		}
	}
	const char *requiredHistoryGuideShaderSymbols[] = {
	    "RESERVOIR_GI_HISTORY_GUIDE_MIN_TARGET_WEIGHT",
	    "RESERVOIR_GI_HISTORY_GUIDE_REJECT_REPROJECTION",
	    "RESERVOIR_GI_HISTORY_GUIDE_REJECT_LOAD",
	    "RESERVOIR_GI_HISTORY_GUIDE_REJECT_GEOMETRY",
	    "bool loadHistoryGuidedReservoirGiProposalRecord(",
	    "static const int2 historyGuideNeighborOffsets[5]",
	    "bestHistoryTargetWeight",
	    "historyGuideNeighborOffsets[i]",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiHistoryGuideNeighborSearchesOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiHistoryGuideNeighborHitsOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiHistoryGuideNeighborMissesOffset, 1u)",
	    "bool trySampleHistoryGuidedReservoirGiProposalDirection(",
	    "loadReservoirGiHistoryMetadata(historyPixel, launchSize, historyMetadata)",
	    "!isFinitePositive(historyRecord.targetWeight) ||",
	    "historyRecord.targetWeight < RESERVOIR_GI_HISTORY_GUIDE_MIN_TARGET_WEIGHT",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiHistoryGuideUsedOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiHistoryGuideRejectedLowWeightOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiHistoryGuideFallbackCosineOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiHistoryGuideRejectReprojectionOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiHistoryGuideRejectLoadOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiHistoryGuideRejectGeometryOffset, 1u)"};
	for (const char *symbol : requiredHistoryGuideShaderSymbols)
	{
		if (!containsText(raygen, symbol))
		{
			std::cerr << "missing history-guided proposal gate contract: " << symbol << "\n";
			return false;
		}
	}
	const std::size_t historyGuideAxisPos = raygen.find("bool tryGetHistoryGuidedReservoirGiAxis(");
	const std::size_t historyGuideSamplePos =
	    raygen.find("FirstHitProbeSample sampleHistoryGuidedReservoirGiProposalDirection(", historyGuideAxisPos);
	if (historyGuideAxisPos == std::string::npos || historyGuideSamplePos == std::string::npos)
	{
		std::cerr << "history-guided proposal function markers are missing\n";
		return false;
	}
	const std::string historyGuideAxisFunction =
	    raygen.substr(historyGuideAxisPos, historyGuideSamplePos - historyGuideAxisPos);
	const std::size_t historyGuideReprojectPos =
	    historyGuideAxisFunction.find("reprojectReservoirHistoryPixel(hitPos, launchSize, historyPixel)");
	const std::size_t historyGuideLoadPos =
	    historyGuideAxisFunction.find("loadHistoryGuidedReservoirGiProposalRecord(neighborPixel, launchSize");
	if (historyGuideReprojectPos == std::string::npos ||
	    historyGuideLoadPos == std::string::npos ||
	    historyGuideReprojectPos > historyGuideLoadPos ||
	    containsText(historyGuideAxisFunction, "loadTemporalReservoirGi("))
	{
		std::cerr << "history-guided proposal must reproject and use its lightweight proposal loader\n";
		return false;
	}
	const char *forbiddenReservoirMissFallbackSymbols[] = {
	    "candidateTotal = firstLegThroughput * bouncePayload.emission",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiLocalMissPositiveWeightOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiAcceptedLocalMissOffset, 1u)",
	    "candidateRecord = record"};
	for (const char *symbol : forbiddenReservoirMissFallbackSymbols)
	{
		if (containsText(reservoirSamplingFunction, symbol))
		{
			std::cerr << "reservoir GI local miss fallback still participates in candidate selection: "
			          << symbol << "\n";
			return false;
		}
	}
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
	    {"reservoirGiLocalMissCandidates",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiLocalMissCandidates), 152u},
	    {"reservoirGiLocalMissPositiveWeight",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiLocalMissPositiveWeight), 156u},
	    {"reservoirGiLocalSurfaceInvalid",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiLocalSurfaceInvalid), 160u},
	    {"reservoirGiAcceptedLocalSurface",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiAcceptedLocalSurface), 164u},
	    {"reservoirGiAcceptedLocalMiss",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiAcceptedLocalMiss), 168u},
	    {"reservoirGiLocalShadowRays",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiLocalShadowRays), 172u},
	    {"reservoirGiTemporalReconnectRays",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiTemporalReconnectRays), 176u},
	    {"reservoirGiTemporalShadowRays",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiTemporalShadowRays), 180u},
	    {"reservoirGiHistoryGuideUsed",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiHistoryGuideUsed), 184u},
	    {"reservoirGiHistoryGuideRejectedLowWeight",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiHistoryGuideRejectedLowWeight), 188u},
	    {"reservoirGiHistoryGuideFallbackCosine",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiHistoryGuideFallbackCosine), 192u},
	    {"reservoirGiHistoryGuideRejectReprojection",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiHistoryGuideRejectReprojection), 196u},
	    {"reservoirGiHistoryGuideRejectLoad",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiHistoryGuideRejectLoad), 200u},
	    {"reservoirGiHistoryGuideRejectGeometry",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiHistoryGuideRejectGeometry), 204u},
	    {"reservoirGiHistoryGuideNeighborSearches",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiHistoryGuideNeighborSearches), 208u},
	    {"reservoirGiHistoryGuideNeighborHits",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiHistoryGuideNeighborHits), 212u},
	    {"reservoirGiHistoryGuideNeighborMisses",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiHistoryGuideNeighborMisses), 216u},
	    {"reservoirGiLocalRejectGeometry",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiLocalRejectGeometry), 220u},
	    {"reservoirGiLocalRejectNoLight",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiLocalRejectNoLight), 224u},
	    {"reservoirGiLocalRejectZeroTarget",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiLocalRejectZeroTarget), 228u},
	    {"reservoirGiLocalRejectBadPdf",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiLocalRejectBadPdf), 232u},
	    {"reservoirGiReceiverCacheStore",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiReceiverCacheStore), 236u},
	    {"reservoirGiReceiverCacheAttempt",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiReceiverCacheAttempt), 240u},
	    {"reservoirGiReceiverCacheHit",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiReceiverCacheHit), 244u},
	    {"reservoirGiReceiverCacheMiss",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiReceiverCacheMiss), 248u},
	    {"reservoirGiReceiverCacheRejectNoLight",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiReceiverCacheRejectNoLight), 252u},
	    {"reservoirGiReceiverCacheAccepted",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiReceiverCacheAccepted), 256u},
	    {"reservoirGiSelectedCache",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiSelectedCache), 260u},
	    {"reservoirGiReceiverReconnectAttempt",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiReceiverReconnectAttempt), 264u},
	    {"reservoirGiReceiverReconnectHit",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiReceiverReconnectHit), 268u},
	    {"reservoirGiReceiverReconnectMiss",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiReceiverReconnectMiss), 272u},
	    {"reservoirGiReceiverReconnectRejectVisibility",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiReceiverReconnectRejectVisibility), 276u},
	    {"reservoirGiReceiverReconnectRejectTarget",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiReceiverReconnectRejectTarget), 280u},
	    {"reservoirGiReceiverReconnectAccepted",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiReceiverReconnectAccepted), 284u},
	    {"reservoirGiSelectedCacheReconnect",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiSelectedCacheReconnect), 288u},
	    {"reservoirGiReceiverCacheContinuationAttempt",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiReceiverCacheContinuationAttempt), 292u},
	    {"reservoirGiReceiverCacheContinuationHit",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiReceiverCacheContinuationHit), 296u},
	    {"reservoirGiReceiverCacheContinuationMiss",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiReceiverCacheContinuationMiss), 300u},
	    {"reservoirGiReceiverCacheContinuationAccepted",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiReceiverCacheContinuationAccepted), 304u},
	    {"reservoirGiBrightSurfelStore",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelStore), 308u},
	    {"reservoirGiBrightSurfelAttempt",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelAttempt), 312u},
	    {"reservoirGiBrightSurfelHit",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelHit), 316u},
	    {"reservoirGiBrightSurfelMiss",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelMiss), 320u},
	    {"reservoirGiBrightSurfelRejectVisibility",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelRejectVisibility), 324u},
	    {"reservoirGiBrightSurfelRejectGeometry",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelRejectGeometry), 328u},
	    {"reservoirGiBrightSurfelRejectTarget",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelRejectTarget), 332u},
	    {"reservoirGiBrightSurfelAccepted",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelAccepted), 336u},
	    {"reservoirGiSelectedBrightSurfel",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiSelectedBrightSurfel), 340u},
	    {"reservoirGiBrightSurfelPrecheckRejectTarget",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelPrecheckRejectTarget), 344u},
	    {"reservoirGiBrightSurfelTrainingAttempt",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelTrainingAttempt), 348u},
	    {"reservoirGiBrightSurfelTrainingStore",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelTrainingStore), 352u},
	    {"reservoirGiBrightSurfelTrainingRejectGeometry",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelTrainingRejectGeometry), 356u},
	    {"reservoirGiBrightSurfelTrainingRejectTarget",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelTrainingRejectTarget), 360u},
	    {"reservoirGiBrightSurfelSelectorRejectGeometry",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelSelectorRejectGeometry), 364u},
	    {"reservoirGiBrightSurfelSelectorRejectTarget",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelSelectorRejectTarget), 368u},
	    {"reservoirGiBrightSurfelSelectorViable",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelSelectorViable), 372u},
	    {"reservoirGiBrightSurfelIndexedQuery",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelIndexedQuery), 376u},
	    {"reservoirGiBrightSurfelIndexedEmpty",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelIndexedEmpty), 380u},
	    {"reservoirGiBrightSurfelIndexedProbe",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelIndexedProbe), 384u},
	    {"reservoirGiBrightSurfelSelectorRejectDistance",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelSelectorRejectDistance), 388u},
	    {"reservoirGiBrightSurfelSelectorRejectReceiverHemisphere",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelSelectorRejectReceiverHemisphere), 392u},
	    {"reservoirGiBrightSurfelSelectorRejectSurfelHemisphere",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelSelectorRejectSurfelHemisphere), 396u},
	    {"reservoirGiBrightSurfelSelectorRejectInvalidVector",
	     offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelSelectorRejectInvalidVector), 400u}};
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
	    "localMiss=%.1f",
	    "localMissPositive=%.1f",
	    "localSurfaceInvalid=%.1f",
	    "localRejectGeometry=%.1f",
	    "localRejectNoLight=%.1f",
	    "localRejectZeroTarget=%.1f",
	    "localRejectBadPdf=%.1f",
	    "acceptedLocalSurface=%.1f",
	    "acceptedLocalMiss=%.1f",
	    "localShadowRays=%.1f",
	    "temporalReconnectRays=%.1f",
	    "temporalShadowRays=%.1f",
	    "historyGuideUsed=%.1f",
	    "historyGuideRejectedLowWeight=%.1f",
	    "historyGuideFallbackCosine=%.1f",
	    "historyGuideRejectReprojection=%.1f",
	    "historyGuideRejectLoad=%.1f",
	    "historyGuideRejectGeometry=%.1f",
	    "historyGuideNeighborSearches=%.1f",
	    "historyGuideNeighborHits=%.1f",
	    "historyGuideNeighborMisses=%.1f",
	    "reservoirGiConfidenceMAvg=%.5f",
	    "reservoirGiConfidenceMAvg",
	    "reservoirMixedTemporalSpatialBudget2Row.reservoirGiMode = UISystem::PathTracerReservoirGiMode::TemporalSpatial",
	    "reservoirMixedTemporalSpatialBudget2Row.reservoirGiSpatialNeighborCount = 2",
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

	const char *requiredBrightSurfelRowSummaryFields[] = {
	    "brightSurfelPrecheckRejectTarget",
	    "brightSurfelTrainingAttempt",
	    "brightSurfelTrainingStore",
	    "brightSurfelTrainingRejectGeometry",
	    "brightSurfelTrainingRejectTarget",
	    "brightSurfelSelectorRejectGeometry",
	    "brightSurfelSelectorRejectTarget",
	    "brightSurfelSelectorViable",
	    "brightSurfelIndexedQuery",
	    "brightSurfelIndexedEmpty",
	    "brightSurfelIndexedProbe",
	    "brightSurfelSelectorRejectDistance",
	    "brightSurfelSelectorRejectReceiverHemisphere",
	    "brightSurfelSelectorRejectSurfelHemisphere",
	    "brightSurfelSelectorRejectInvalidVector"};
	for (const char *fieldName : requiredBrightSurfelRowSummaryFields)
	{
		if (!containsText(engineCore, fieldName))
		{
			std::cerr << "missing reservoir GI bright surfel row-summary log field: "
			          << fieldName << "\n";
			return false;
		}
	}
	if (!requireIndexedBrightSurfelDiagnosticPlumbing(
	        engineAuxiliaryHeader, uiHeader, uiSource, engineCore))
	{
		return false;
	}

	const char *requiredSponzaAuditRows[] = {
	    "Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2",
	    "Reservoir 1C Shadowed Sun First Mixed Single Frame Sun Receiver",
	    "Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Sun Receiver",
	    "Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Sun Receiver Surfel Cache Debug",
	    "Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Sun Receiver Env First Two",
	    "Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Sun Receiver Env First Two Cache Continuation"};
	const std::size_t sponzaSweepStartPos =
	    engineCore.find("void EngineCore::startPathTracerSponzaGiPerfSweep()");
	const std::size_t sponzaSweepEndPos =
	    sponzaSweepStartPos == std::string::npos
	        ? std::string::npos
	        : engineCore.find("\nvoid EngineCore::", sponzaSweepStartPos + 1u);
	if (sponzaSweepStartPos == std::string::npos)
	{
		std::cerr << "focused Sponza PT/GI audit sweep function is missing\n";
		return false;
	}
	const std::string sponzaSweepSource =
	    engineCore.substr(sponzaSweepStartPos, sponzaSweepEndPos - sponzaSweepStartPos);
	for (const char *rowName : requiredSponzaAuditRows)
	{
		const std::string quotedRowName = std::string("\"") + rowName + "\"";
		if (sponzaSweepSource.find(quotedRowName) == std::string::npos)
		{
			std::cerr << "focused Sponza PT/GI audit sweep is missing exact quoted row literal: "
			          << rowName << "\n";
			return false;
		}
	}
	const char *requiredSponzaAuditRowSchedule[] = {
	    "ptExperimentRows.push_back(reservoirMixedTemporalSpatialBudget2Row);",
	    "ptExperimentRows.push_back(reservoirMixedTemporalSpatialBudget2SunReceiverRow);",
	    "ptExperimentRows.push_back(reservoirMixedTemporalSpatialBudget2SunReceiverSurfelCacheDebugRow);",
	    "ptExperimentRows.push_back(reservoirMixedSingleFrameSunReceiverRow);",
	    "ptExperimentRows.push_back(reservoirMixedTemporalSpatialBudget2SunReceiverEnvFirstTwoRow);",
	    "ptExperimentRows.push_back(reservoirMixedTemporalSpatialBudget2SunReceiverEnvFirstTwoCacheContinuationRow);"};
	const std::size_t expectedSponzaAuditRowScheduleCount =
	    sizeof(requiredSponzaAuditRowSchedule) / sizeof(requiredSponzaAuditRowSchedule[0]);
	const char *sponzaAuditPushBackNeedle = "ptExperimentRows.push_back(";
	std::size_t sponzaAuditPushBackCount = 0;
	std::size_t sponzaAuditPushBackSearchPos = 0;
	while ((sponzaAuditPushBackSearchPos =
	            sponzaSweepSource.find(sponzaAuditPushBackNeedle, sponzaAuditPushBackSearchPos)) !=
	       std::string::npos)
	{
		++sponzaAuditPushBackCount;
		sponzaAuditPushBackSearchPos += std::string(sponzaAuditPushBackNeedle).size();
	}
	if (sponzaAuditPushBackCount != expectedSponzaAuditRowScheduleCount)
	{
		std::cerr << "focused Sponza PT/GI audit sweep should schedule exactly the expected audit rows: expected "
		          << expectedSponzaAuditRowScheduleCount << " got " << sponzaAuditPushBackCount << "\n";
		return false;
	}
	std::size_t sponzaAuditRowScheduleSearchPos = 0;
	for (const char *scheduledRow : requiredSponzaAuditRowSchedule)
	{
		const std::size_t scheduledRowPos =
		    sponzaSweepSource.find(scheduledRow, sponzaAuditRowScheduleSearchPos);
		if (scheduledRowPos == std::string::npos)
		{
			std::cerr << "focused Sponza PT/GI audit sweep is missing scheduled row in order: "
			          << scheduledRow << "\n";
			return false;
		}
		sponzaAuditRowScheduleSearchPos = scheduledRowPos + std::string(scheduledRow).size();
	}
	const char *requiredSurfelDiagnosticRowSymbols[] = {
	    "bool enableSurfelGi = false",
	    "bool surfelGiDebug = false",
	    "settings.enableSurfelGi",
	    "settings.surfelGiDebug",
	    "auto reservoirMixedTemporalSpatialBudget2SunReceiverSurfelCacheDebugRow =",
	    "reservoirMixedTemporalSpatialBudget2SunReceiverSurfelCacheDebugRow.enableSurfelGi = true;",
	    "reservoirMixedTemporalSpatialBudget2SunReceiverSurfelCacheDebugRow.surfelGiDebug = true;"};
	for (const char *symbol : requiredSurfelDiagnosticRowSymbols)
	{
		if (!containsText(engineCore, symbol) && !containsText(engineHeader, symbol))
		{
			std::cerr << "missing diagnostic-only Sponza surfel cache sweep contract: "
			          << symbol << "\n";
			return false;
		}
	}
	const std::size_t surfelDiagnosticRowStart =
	    sponzaSweepSource.find("reservoirMixedTemporalSpatialBudget2SunReceiverSurfelCacheDebugRow");
	const std::size_t surfelDiagnosticRowSchedule =
	    sponzaSweepSource.find("ptExperimentRows.push_back(reservoirMixedTemporalSpatialBudget2SunReceiverSurfelCacheDebugRow);");
	if (surfelDiagnosticRowStart == std::string::npos ||
	    surfelDiagnosticRowSchedule == std::string::npos ||
	    surfelDiagnosticRowSchedule <= surfelDiagnosticRowStart)
	{
		std::cerr << "focused Sponza PT/GI audit sweep is missing the diagnostic-only surfel cache row setup\n";
		return false;
	}
	const std::string surfelDiagnosticRowSource =
	    sponzaSweepSource.substr(surfelDiagnosticRowStart,
	                             surfelDiagnosticRowSchedule - surfelDiagnosticRowStart);
	if (!containsText(surfelDiagnosticRowSource, "MixedCosineSunReceiverGuided") ||
	    containsText(surfelDiagnosticRowSource, "MixedCosineSunReceiverBrightSurfel") ||
	    containsText(surfelDiagnosticRowSource, "settings.enableSurfelGi") ||
	    containsText(surfelDiagnosticRowSource, "settings.surfelGiDebug"))
	{
		std::cerr << "diagnostic-only Sponza surfel cache row must keep the Sun Receiver proposal and only set row flags\n";
		return false;
	}
	const std::size_t experimentSweepUpdatePos =
	    engineCore.find("void EngineCore::updatePathTracerExperimentSweep()");
	const std::size_t postExperimentSweepUpdatePos =
	    engineCore.find("void EngineCore::ensurePathTracerSanityScene()", experimentSweepUpdatePos);
	const std::size_t rowTransitionWaitIdlePos =
	    engineCore.find("vulkan.logicalDevice.waitIdle();", experimentSweepUpdatePos);
	const std::size_t rowTransitionClearStatePos =
	    engineCore.find("clearPathTracerExperimentState();", rowTransitionWaitIdlePos);
	const std::size_t rowTransitionApplyRowPos =
	    engineCore.find("applyPathTracerExperimentRow(ptExperimentRows[ptExperimentRowIndex]);", rowTransitionClearStatePos);
	if (experimentSweepUpdatePos == std::string::npos ||
	    postExperimentSweepUpdatePos == std::string::npos ||
	    rowTransitionWaitIdlePos == std::string::npos ||
	    rowTransitionClearStatePos == std::string::npos ||
	    rowTransitionApplyRowPos == std::string::npos ||
	    rowTransitionApplyRowPos >= postExperimentSweepUpdatePos)
	{
		std::cerr << "focused Sponza PT/GI audit sweep should clear experiment state before advancing rows\n";
		return false;
	}
	if (containsText(engineCore, "Sponza / Reservoir GI Temporal") ||
	    containsText(engineCore, "Sponza / Reservoir GI Temporal Spatial") ||
	    containsText(engineCore, "Sponza / Reservoir GI Single Frame 2 Candidates No RIS") ||
	    containsText(engineCore, "Sponza / Reservoir GI Single Frame 2 Candidates RIS") ||
	    containsText(engineCore, "Base 3 Sun First") ||
	    containsText(engineCore, "Base 5 Sun First") ||
	    containsText(engineCore, "Base 8 Sun First") ||
	    containsText(engineCore, "Base 8 Sun All") ||
	    containsText(engineCore, "Reservoir 1C Shadowed Sun First Mixed\",") ||
	    containsText(engineCore, "Reservoir 1C Shadowed Sun First Mixed Temporal Budget 2") ||
	    containsText(engineCore, "Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N\"") ||
	    containsText(engineCore, "Reservoir 1C Shadowed Sun First Mixed Temporal\")") ||
	    containsText(engineCore, "Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Env First Two") ||
	    containsText(engineCore, "Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Receiver Reconnect") ||
	    containsText(engineCore, "Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 1N Budget") ||
	    containsText(engineCore, "Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 3") ||
	    containsText(engineCore, "Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Dual Sun") ||
	    containsText(engineCore, "Reservoir 1C Shadowed Sun First Mixed History Guide") ||
	    containsText(engineCore, "Reservoir 1C Shadowed Sun All") ||
	    containsText(engineCore, "Reservoir 1C Shadowed Sun First\", 1") ||
	    containsText(engineCore, "Reservoir 1C Shadowed Sun First Sun Guided") ||
	    containsText(engineCore, "Reservoir 2C Shadowed Sun First Mixed") ||
	    containsText(engineCore, "Reservoir 2C Shadowed Sun First Mixed RIS") ||
	    containsText(engineCore, "Reservoir 1C Shadowed Sun First Light Region") ||
	    containsText(engineCore, "Cache Chosen Radius 14 Budget 1") ||
	    containsText(engineCore, "makeCacheChosenRow"))
	{
		std::cerr << "focused Sponza PT/GI audit sweep should only run base audit and current ReSTIR variants\n";
		return false;
	}

	const char *requiredCounterAndUiSymbols[] = {
	    "uint32_t reservoirGiConfidenceMScaledSum = 0",
	    "uint32_t reservoirGiLocalSurfaceHits = 0",
	    "uint32_t reservoirGiLocalValidSamples = 0",
	    "uint32_t reservoirGiLocalMissCandidates = 0",
	    "uint32_t reservoirGiLocalMissPositiveWeight = 0",
	    "uint32_t reservoirGiLocalSurfaceInvalid = 0",
	    "uint32_t reservoirGiLocalRejectGeometry = 0",
	    "uint32_t reservoirGiLocalRejectNoLight = 0",
	    "uint32_t reservoirGiLocalRejectZeroTarget = 0",
	    "uint32_t reservoirGiLocalRejectBadPdf = 0",
	    "uint32_t reservoirGiAcceptedLocalSurface = 0",
	    "uint32_t reservoirGiAcceptedLocalMiss = 0",
	    "uint32_t reservoirGiLocalShadowRays = 0",
	    "uint32_t reservoirGiTemporalReconnectRays = 0",
	    "uint32_t reservoirGiTemporalShadowRays = 0",
	    "uint32_t reservoirGiHistoryGuideUsed = 0",
	    "uint32_t reservoirGiHistoryGuideRejectedLowWeight = 0",
	    "uint32_t reservoirGiHistoryGuideFallbackCosine = 0",
	    "uint32_t reservoirGiHistoryGuideRejectReprojection = 0",
	    "uint32_t reservoirGiHistoryGuideRejectLoad = 0",
	    "uint32_t reservoirGiHistoryGuideRejectGeometry = 0",
	    "uint32_t reservoirGiHistoryGuideNeighborSearches = 0",
	    "uint32_t reservoirGiHistoryGuideNeighborHits = 0",
	    "uint32_t reservoirGiHistoryGuideNeighborMisses = 0",
	    "reservoirGiBrightSurfelStore",
	    "reservoirGiBrightSurfelAttempt",
	    "reservoirGiBrightSurfelHit",
	    "reservoirGiBrightSurfelMiss",
	    "reservoirGiBrightSurfelRejectVisibility",
	    "reservoirGiBrightSurfelRejectGeometry",
	    "reservoirGiBrightSurfelRejectTarget",
	    "reservoirGiBrightSurfelAccepted",
	    "reservoirGiSelectedBrightSurfel",
	    "reservoirGiBrightSurfelSelectorRejectGeometry",
	    "reservoirGiBrightSurfelSelectorRejectTarget",
	    "reservoirGiBrightSurfelSelectorViable",
	    "float reservoirGiConfidenceMAvg = 0.0f",
	    "float reservoirGiLocalValidRatio = 0.0f",
	    "Reservoir GI Confidence M Avg",
	    "Reservoir GI Local Valid",
	    "Reservoir GI Local Miss",
	    "Reservoir GI Accepted Local Surface",
	    "Reservoir GI Accepted Local Miss",
	    "Reservoir GI Local Shadow Rays",
	    "Reservoir GI Temporal Reconnect Rays",
	    "Reservoir GI History Guide Used",
	    "Reservoir GI History Guide Rejected Low Weight",
	    "Reservoir GI History Guide Fallback Cosine",
	    "Reservoir GI History Guide Reject Reprojection",
	    "Reservoir GI History Guide Reject Load",
	    "Reservoir GI History Guide Reject Geometry",
	    "Reservoir GI History Guide Neighbor Searches",
	    "Reservoir GI History Guide Neighbor Hits",
	    "Reservoir GI History Guide Neighbor Misses",
	    "Reservoir GI Selected Bright Surfel",
	    "Reservoir GI Bright Surfel Accepted",
	    "Reservoir GI Bright Surfel Precheck Reject Target",
	    "Reservoir GI Bright Surfel Training Attempts",
	    "Reservoir GI Bright Surfel Training Stores",
	    "Reservoir GI Bright Surfel Training Reject Geometry",
	    "Reservoir GI Bright Surfel Training Reject Target",
	    "Reservoir GI Bright Surfel Selector Reject Geometry",
	    "Reservoir GI Bright Surfel Selector Reject Target",
	    "Reservoir GI Bright Surfel Selector Viable",
	    "Reservoir GI Bright Surfel Indexed Query",
	    "Reservoir GI Bright Surfel Indexed Empty",
	    "Reservoir GI Bright Surfel Indexed Probe",
	    "Reservoir GI Bright Surfel Selector Reject Distance",
	    "Reservoir GI Bright Surfel Selector Reject Receiver Hemisphere",
	    "Reservoir GI Bright Surfel Selector Reject Surfel Hemisphere",
	    "Reservoir GI Bright Surfel Selector Reject Invalid Vector"};
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
	if (!containsText(uiHeader, "sponzaGiSweepWarmupFrames = 8") ||
	    !containsText(uiHeader, "sponzaGiSweepSampleFrames = 32") ||
	    !containsText(engineSource, "analysis.sponzaGiSweepWarmupFrames") ||
	    !containsText(engineSource, "analysis.sponzaGiSweepSampleFrames"))
	{
		std::cerr << "Sponza PT/GI audit sweep should use dedicated fast frame counts\n";
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
	    "Sponza Sweep Warmup",
	    "Sponza Sweep Samples",
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
	    "Mixed Cosine + Sun Receiver Guide",
	    "Mixed Cosine + Dual Sun Guide",
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
	if (!containsText(denoiser, "selectPathTracerDebugAovOutput(pixel, color)") ||
	    !containsText(denoiser, "selectPathTracerDebugAovOutput(pixel, filtered)"))
	{
		std::cerr << "path debug AOVs must bypass temporal/denoised display in both pass-through and A-Trous paths\n";
		return false;
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
	    "debugReservoirGiContribution",
	    "debugReservoirGiAcceptedLuma",
	    "debugReservoirGiCandidateSurfaceHitRatio",
	    "debugReservoirGiCandidateSunVisibleRatio",
	    "debugReservoirGiCandidatePositiveWeightRatio",
	    "debugReservoirGiSelectedWeight",
	    "debugReservoirGiLocalNoLightRatio",
	    "debugReservoirGiSelectedSource",
	    "DEBUG_AOV_PATH_RESERVOIR_GI_CONTRIBUTION",
	    "DEBUG_AOV_PATH_RESERVOIR_GI_ACCEPTED_LUMA",
	    "DEBUG_AOV_PATH_RESERVOIR_GI_CANDIDATE_SURFACE_HIT",
	    "DEBUG_AOV_PATH_RESERVOIR_GI_CANDIDATE_SUN_VISIBLE",
	    "DEBUG_AOV_PATH_RESERVOIR_GI_CANDIDATE_POSITIVE_WEIGHT",
	    "DEBUG_AOV_PATH_RESERVOIR_GI_SELECTED_WEIGHT",
	    "DEBUG_AOV_PATH_RESERVOIR_GI_LOCAL_NO_LIGHT",
	    "DEBUG_AOV_PATH_RESERVOIR_GI_SELECTED_SOURCE",
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
	    "RESERVOIR_GI_PROPOSAL_MIXED_COSINE_HISTORY_GUIDED",
	    "RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_GUIDED",
	    "RESERVOIR_GI_PROPOSAL_MIXED_COSINE_DUAL_SUN_GUIDED",
	    "RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_CACHE_GUIDED",
	    "RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_CACHE_RECONNECT",
	    "RESERVOIR_GI_RECEIVER_CACHE_CURRENT_BINDING",
	    "RESERVOIR_GI_RECEIVER_CACHE_HISTORY_BINDING",
	    "ptReservoirGiReceiverCacheCurrent",
	    "ptReservoirGiReceiverCacheHistory",
	    "ReservoirGiReceiverCacheRecord",
	    "trySampleReceiverCacheGuidedReservoirGiProposalDirection",
	    "evaluateReceiverCacheReconnectReservoirGiCandidate",
	    "storeReservoirGiReceiverCacheRecord",
	    "reservoirGiReceiverCacheAttemptOffset",
	    "reservoirGiReceiverCacheHitOffset",
	    "reservoirGiReceiverCacheMissOffset",
	    "reservoirGiReceiverCacheRejectNoLightOffset",
	    "reservoirGiReceiverCacheAcceptedOffset",
	    "reservoirGiReceiverCacheStoreOffset",
	    "reservoirGiSelectedCacheOffset",
	    "reservoirGiReceiverReconnectAttemptOffset",
	    "reservoirGiReceiverReconnectHitOffset",
	    "reservoirGiReceiverReconnectMissOffset",
	    "reservoirGiReceiverReconnectRejectVisibilityOffset",
	    "reservoirGiReceiverReconnectRejectTargetOffset",
	    "reservoirGiReceiverReconnectAcceptedOffset",
	    "reservoirGiSelectedCacheReconnectOffset",
	    "reservoirGiReceiverCacheContinuationAttemptOffset",
	    "reservoirGiReceiverCacheContinuationHitOffset",
	    "reservoirGiReceiverCacheContinuationMissOffset",
	    "reservoirGiReceiverCacheContinuationAcceptedOffset",
	    "ptReservoirGiBrightSurfelCurrent",
	    "ptReservoirGiBrightSurfelHistory",
	    "ReservoirGiBrightSurfelRecord",
	    "RESERVOIR_GI_BRIGHT_SURFEL_CURRENT_BINDING",
	    "RESERVOIR_GI_BRIGHT_SURFEL_HISTORY_BINDING",
	    "RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_BRIGHT_SURFEL",
	    "RESERVOIR_GI_SOURCE_BRIGHT_SURFEL",
	    "RESERVOIR_GI_BRIGHT_SURFEL_ATTEMPT_STRIDE",
	    "RESERVOIR_GI_BRIGHT_SURFEL_CANDIDATE_MIN_TARGET_WEIGHT",
	    "RESERVOIR_GI_BRIGHT_SURFEL_INDEX_CELL_SIZE",
	    "RESERVOIR_GI_BRIGHT_SURFEL_INDEX_CELL_SLOTS",
	    "RESERVOIR_GI_BRIGHT_SURFEL_INDEX_RADIUS_CELLS",
	    "storeReservoirGiBrightSurfelRecord",
	    "loadReservoirGiBrightSurfelHistoryRecord",
	    "loadReservoirGiBrightSurfelHistoryHeader",
	    "ReservoirGiBrightSurfelPrecheck",
	    "precheckReservoirGiBrightSurfelHistoryRecord",
	    "loadPrecheckedReservoirGiBrightSurfelHistoryRecord",
	    "shouldAttemptBrightReceiverSurfel",
	    "shouldTrainBrightReceiverSurfel",
	    "reservoirGiBrightSurfelIndexedStoreIndex",
	    "reservoirGiBrightSurfelIndexedQueryIndex",
	    "tryStoreBrightSurfelTrainingCandidate",
	    "selectWeightedIndexedBrightReceiverSurfelRecord",
	    "surfelSelectionPdf",
	    "Scan-local selector diagnostic only; do not feed it into target sourcePdf or reservoir weight.",
	    "BrightSurfelTargetEstimateResult",
	    "BRIGHT_SURFEL_TARGET_REJECT_NONE",
	    "BRIGHT_SURFEL_TARGET_REJECT_GEOMETRY",
	    "BRIGHT_SURFEL_TARGET_REJECT_TARGET",
	    "BRIGHT_SURFEL_TARGET_REJECT_DISTANCE",
	    "BRIGHT_SURFEL_TARGET_REJECT_RECEIVER_HEMISPHERE",
	    "BRIGHT_SURFEL_TARGET_REJECT_SURFEL_HEMISPHERE",
	    "BRIGHT_SURFEL_TARGET_REJECT_INVALID_VECTOR",
	    "estimateBrightSurfelTargetForReceiver",
	    "selectedTargetEvaluation",
	    "evaluateBrightReceiverSurfelReservoirGiCandidate",
	    "reservoirGiBrightSurfelStoreOffset",
	    "reservoirGiBrightSurfelAttemptOffset",
	    "reservoirGiBrightSurfelHitOffset",
	    "reservoirGiBrightSurfelMissOffset",
	    "reservoirGiBrightSurfelRejectVisibilityOffset",
	    "reservoirGiBrightSurfelRejectGeometryOffset",
	    "reservoirGiBrightSurfelRejectTargetOffset",
	    "reservoirGiBrightSurfelAcceptedOffset",
	    "reservoirGiSelectedBrightSurfelOffset",
	    "reservoirGiBrightSurfelPrecheckRejectTargetOffset",
	    "reservoirGiBrightSurfelTrainingAttemptOffset",
	    "reservoirGiBrightSurfelTrainingStoreOffset",
	    "reservoirGiBrightSurfelTrainingRejectGeometryOffset",
	    "reservoirGiBrightSurfelTrainingRejectTargetOffset",
	    "reservoirGiBrightSurfelSelectorRejectGeometryOffset",
	    "reservoirGiBrightSurfelSelectorRejectTargetOffset",
	    "reservoirGiBrightSurfelSelectorViableOffset",
	    "reservoirGiBrightSurfelIndexedQueryOffset",
	    "reservoirGiBrightSurfelIndexedEmptyOffset",
	    "reservoirGiBrightSurfelIndexedProbeOffset",
	    "reservoirGiBrightSurfelSelectorRejectDistanceOffset",
	    "reservoirGiBrightSurfelSelectorRejectReceiverHemisphereOffset",
	    "reservoirGiBrightSurfelSelectorRejectSurfelHemisphereOffset",
	    "reservoirGiBrightSurfelSelectorRejectInvalidVectorOffset",
	    "RESERVOIR_GI_CANDIDATE_SHADOWED_SUN_CACHE_CONTINUATION",
	    "tryEvaluateReservoirGiReceiverCacheContinuation",
	    "RESERVOIR_GI_CACHE_CONTINUATION_FLAG",
	    "RESERVOIR_GI_SOURCE_CACHE_RECONNECT",
	    "PT_MATERIAL_RESERVOIR_PROPOSAL_SHIFT",
	    "PT_MATERIAL_RESERVOIR_PROPOSAL_MASK",
	    "PT_MATERIAL_RESERVOIR_MODE_SHIFT",
	    "PT_FLAGS_ENVIRONMENT_NEE_BIT",
	    "PT_FLAGS_ENVIRONMENT_BOUNCE_SHIFT",
	    "PT_FLAGS_FIRST_HIT_PROBE_SAMPLING_SHIFT",
	    "sampleReservoirGiProposalDirection",
	    "sunReceiverBounceGuideAxis",
	    "reservoirGiProposalMode == RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_GUIDED",
	    "reservoirGiProposalMode == RESERVOIR_GI_PROPOSAL_MIXED_COSINE_DUAL_SUN_GUIDED",
	    "reservoirGiProposalMode == RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_CACHE_GUIDED",
	    "reservoirGiProposalMode == RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_CACHE_RECONNECT",
	    "RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_CACHE_RECONNECT);",
	    "reconnectRecord.confidenceM = 1.0",
	    "bool canPersistSelectedReservoirGi = selectedSource != RESERVOIR_GI_SOURCE_CACHE_RECONNECT",
	    "if (acceptedReservoir && canPersistSelectedReservoirGi) {",
	    "storeAcceptedReservoirGiForGuidedProposal) &&",
	    "acceptedReservoir && canPersistSelectedReservoirGi) {",
	    "0.50 * cosinePdf + 0.25 * sunPdf + 0.25 * receiverPdf",
	    "sampleHistoryGuidedReservoirGiProposalDirection",
	    "reservoirGiProposalMode == RESERVOIR_GI_PROPOSAL_MIXED_COSINE_HISTORY_GUIDED",
	    "storeAcceptedReservoirGiForGuidedProposal",
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
	if (containsText(raygen, "estimateBrightSurfelTargetBeforeVisibility"))
	{
		std::cerr
		    << "old bright surfel target estimator must be replaced by estimateBrightSurfelTargetForReceiver\n";
		return false;
	}
	if (!requireIndexedBrightSurfelShaderContracts(raygen))
	{
		return false;
	}
	const std::string brightSurfelSelector =
	    extractFunctionBody(raygen, "bool selectWeightedIndexedBrightReceiverSurfelRecord");
	if (brightSurfelSelector.empty())
	{
		std::cerr << "missing indexed bright surfel selector function body\n";
		return false;
	}
	const char *requiredBrightSurfelSelectorSymbols[] = {
	    "loadReservoirGiBrightSurfelHistoryHeader(surfelCapacity, surfelHistoryFrameId)",
	    "reservoirGiBrightSurfelIndexedQueryIndex(hitPos, cellOffset, slot)",
	    "precheckReservoirGiBrightSurfelHistoryRecord(surfelIndex, surfelCapacity, surfelHistoryFrameId",
	    "loadPrecheckedReservoirGiBrightSurfelHistoryRecord(surfelIndex, precheck, surfel)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelIndexedQueryOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelIndexedEmptyOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelIndexedProbeOffset, 1u)",
	    "estimateBrightSurfelTargetForReceiver(hitPos, N, V, primaryPayload",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorRejectDistanceOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorRejectReceiverHemisphereOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorRejectSurfelHemisphereOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorRejectInvalidVectorOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorRejectGeometryOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorRejectTargetOffset, 1u)",
	    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorViableOffset, 1u)",
	    "selectedTargetEvaluation = estimate.targetEvaluation",
	    "float score = estimate.targetEvaluation.targetWeight"};
	for (const char *symbol : requiredBrightSurfelSelectorSymbols)
	{
		if (!containsText(brightSurfelSelector, symbol))
		{
			std::cerr << "missing bright surfel selector-local shader contract: " << symbol << "\n";
			return false;
		}
	}
	if (!brightSurfelCombineUsesTargetWeight(raygen))
	{
		std::cerr << "bright surfel reservoir combine must use surfelRecord.targetWeight\n";
		return false;
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
	    "kPtFlagsEnvironmentBounceShift",
	    "kPtFlagsReservoirGiDetailedDiagnosticsBit",
	    "kPtFlagsReservoirTemporalBudgetShift",
	    "kPtFlagsReservoirSpatialBudgetShift",
	    "if (divisor <= 3)",
	    "kPtFlagsFirstHitProbeSamplingShift",
	    "reservoirGiDetailedDiagnostics",
	    "reservoirGiTemporalBudgetDivisor",
	    "reservoirGiSpatialBudgetDivisor",
	    "environmentNeeBounceMode",
	    "Env NEE Bounces",
	    "First Only",
	    "First Two",
	    "All Bounces",
	    "Temporal Reuse Budget Divisor",
	    "Spatial Reuse Budget Divisor",
	    "Detailed Reservoir Diagnostics",
	    "Reservoir GI Local No Light",
	    "Reservoir GI Selected Source",
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
	    "kReservoirGiReceiverCacheCapacity",
	    "kReservoirGiBrightSurfelCapacity",
	    "reservoirGiReceiverCacheBuffers",
	    "reservoirGiReceiverCacheMapped",
	    "createReservoirGiReceiverCacheBuffers",
	    "reservoirGiBrightSurfelBuffers",
	    "reservoirGiBrightSurfelMapped",
	    "createReservoirGiBrightSurfelBuffers",
	    "RESERVOIR_GI_RECEIVER_CACHE_CURRENT_BINDING",
	    "RESERVOIR_GI_RECEIVER_CACHE_HISTORY_BINDING",
	    "reservoirGiReceiverCacheStore",
	    "reservoirGiReceiverCacheAttempt",
	    "reservoirGiReceiverCacheHit",
	    "reservoirGiReceiverCacheMiss",
	    "reservoirGiReceiverCacheRejectNoLight",
	    "reservoirGiReceiverCacheAccepted",
	    "reservoirGiSelectedCache",
	    "reservoirGiReceiverReconnectAttempt",
	    "reservoirGiReceiverReconnectHit",
	    "reservoirGiReceiverReconnectMiss",
	    "reservoirGiReceiverReconnectRejectVisibility",
	    "reservoirGiReceiverReconnectRejectTarget",
	    "reservoirGiReceiverReconnectAccepted",
	    "reservoirGiSelectedCacheReconnect",
	    "reservoirGiReceiverCacheContinuationAttempt",
	    "reservoirGiReceiverCacheContinuationHit",
	    "reservoirGiReceiverCacheContinuationMiss",
	    "reservoirGiReceiverCacheContinuationAccepted",
	    "reservoirGiBrightSurfelStore",
	    "reservoirGiBrightSurfelAttempt",
	    "reservoirGiBrightSurfelHit",
	    "reservoirGiBrightSurfelMiss",
	    "reservoirGiBrightSurfelRejectVisibility",
	    "reservoirGiBrightSurfelRejectGeometry",
	    "reservoirGiBrightSurfelRejectTarget",
	    "reservoirGiBrightSurfelAccepted",
	    "reservoirGiSelectedBrightSurfel",
	    "brightSurfelPrecheckRejectTarget",
	    "brightSurfelTrainingAttempt",
	    "brightSurfelTrainingStore",
	    "brightSurfelTrainingRejectGeometry",
	    "brightSurfelTrainingRejectTarget",
	    "brightSurfelIndexedQuery",
	    "brightSurfelIndexedEmpty",
	    "brightSurfelIndexedProbe",
	    "brightSurfelSelectorRejectDistance",
	    "brightSurfelSelectorRejectReceiverHemisphere",
	    "brightSurfelSelectorRejectSurfelHemisphere",
	    "brightSurfelSelectorRejectInvalidVector",
	    "MixedCosineSunReceiverCacheGuided",
	    "Mixed Cosine + Sun Receiver + Cache Guide",
	    "MixedCosineSunReceiverCacheReconnect",
	    "Mixed Cosine + Sun Receiver + Cache Reconnect",
	    "MixedCosineSunReceiverBrightSurfel",
	    "Reservoir GI Selected Cache Reconnect",
	    "Reservoir GI Selected Bright Surfel",
	    "Reservoir GI Bright Surfel Accepted",
	    "Reservoir GI Bright Surfel Precheck Reject Target",
	    "Reservoir GI Bright Surfel Training Attempts",
	    "Reservoir GI Bright Surfel Training Stores",
	    "Reservoir GI Bright Surfel Training Reject Geometry",
	    "Reservoir GI Bright Surfel Training Reject Target",
	    "Reservoir GI Bright Surfel Indexed Query",
	    "Reservoir GI Bright Surfel Indexed Empty",
	    "Reservoir GI Bright Surfel Indexed Probe",
	    "Reservoir GI Bright Surfel Selector Reject Distance",
	    "Reservoir GI Bright Surfel Selector Reject Receiver Hemisphere",
	    "Reservoir GI Bright Surfel Selector Reject Surfel Hemisphere",
	    "Reservoir GI Bright Surfel Selector Reject Invalid Vector",
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
	    "Shadowed Sun + Cache Continuation",
	    "environmentNeeMode=%d",
	    "reservoirTemporalBudget=%d",
	    "reservoirSpatialBudget=%d",
	    "reservoirMixedTemporalSpatialBudget2Row.reservoirGiTemporalBudgetDivisor = 2",
	    "reservoirMixedTemporalSpatialBudget2Row.reservoirGiSpatialBudgetDivisor = 2",
	    "pathTracerSettings.reservoirGiMode            = UISystem::PathTracerReservoirGiMode::TemporalSpatial",
	    "pathTracerSettings.reservoirGiSpatialNeighborCount = 2",
	    "pathTracerSettings.environmentNeeBounceMode = 0",
	    "pathTracerSettings.reservoirGiProposalMode    = UISystem::PathTracerReservoirGiProposalMode::MixedCosineSunReceiverGuided",
	    "pathTracerSettings.reservoirGiTemporalBudgetDivisor = 2",
	    "pathTracerSettings.reservoirGiSpatialBudgetDivisor = 2",
	    "ptBenchmarkBasePosition",
	    "ui.lightDirection",
	    "makeScenarioRowName",
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
	    "pathTracerSettings.environmentNeeBounceMode",
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
	    "MixedCosineDualSunGuided",
	    "MixedCosineSunReceiverGuided",
	    "MixedCosineHistoryGuided",
	    "Mixed Cosine + History Guide",
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

	const std::string activeCacheCleanupSources = uiHeader + uiSource + engineHeader + engineSource +
	                                              frameContextHeader + frameContextSource +
	                                              pipelineCollection + raygen;
	const char *forbiddenOppositeGuideSymbols[] = {
	    "MixedCosineOppositeSunGuided",
	    "Mixed Cosine + Opposite Sun Guide",
	    "RESERVOIR_GI_PROPOSAL_MIXED_COSINE_OPPOSITE_SUN_GUIDED",
	    "oppositeSunBounceGuideAxis"};
	for (const char *symbol : forbiddenOppositeGuideSymbols)
	{
		if (containsText(activeCacheCleanupSources, symbol))
		{
			std::cerr << "retired opposite-guide proposal path remains in active source: " << symbol << "\n";
			return false;
		}
	}

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
