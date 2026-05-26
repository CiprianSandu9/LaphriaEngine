#include "SurfelPathTracerPipelineTests.h"

#include "../src/Core/SurfelPathTracerResources.h"

#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace
{
std::filesystem::path sourceRoot()
{
#ifdef LAPHRIA_SOURCE_DIR
	return std::filesystem::path(LAPHRIA_SOURCE_DIR);
#else
	return std::filesystem::current_path();
#endif
}

std::string readTextFile(const std::filesystem::path &path, bool &ok)
{
	std::ifstream file(path, std::ios::in | std::ios::binary);
	if (!file)
	{
		std::cerr << "missing SurfelPathTracer contract file: " << path.string() << '\n';
		ok = false;
		return {};
	}

	std::ostringstream stream;
	stream << file.rdbuf();
	return stream.str();
}

std::string normalizeContractText(std::string_view text)
{
	std::string normalized;
	normalized.reserve(text.size());
	bool inWhitespace = false;
	for (unsigned char ch : text)
	{
		if (std::isspace(ch))
		{
			inWhitespace = true;
			continue;
		}
		if (inWhitespace && !normalized.empty())
		{
			normalized.push_back(' ');
		}
		normalized.push_back(static_cast<char>(ch));
		inWhitespace = false;
	}
	return normalized;
}

bool containsNeedle(std::string_view haystack, std::string_view needle)
{
	return normalizeContractText(haystack).find(normalizeContractText(needle)) != std::string::npos;
}

std::string extractStructBlock(std::string_view haystack, std::string_view structName, std::string_view label, bool &ok)
{
	const std::string marker = "struct " + std::string(structName);
	const size_t start = haystack.find(marker);
	if (start == std::string_view::npos)
	{
		std::cerr << "missing SurfelPathTracer struct contract in " << label << ": " << marker << '\n';
		ok = false;
		return {};
	}

	const size_t openBrace = haystack.find('{', start);
	if (openBrace == std::string_view::npos)
	{
		std::cerr << "missing SurfelPathTracer struct opening brace in " << label << ": " << marker << '\n';
		ok = false;
		return {};
	}

	size_t closeBrace = std::string_view::npos;
	int braceDepth = 0;
	for (size_t i = openBrace; i < haystack.size(); ++i)
	{
		if (haystack[i] == '{')
		{
			++braceDepth;
		}
		else if (haystack[i] == '}')
		{
			--braceDepth;
			if (braceDepth == 0)
			{
				closeBrace = i;
				break;
			}
		}
	}
	if (closeBrace == std::string_view::npos)
	{
		std::cerr << "missing SurfelPathTracer struct closing brace in " << label << ": " << marker << '\n';
		ok = false;
		return {};
	}

	const size_t semicolon = haystack.find(';', closeBrace);
	if (semicolon == std::string_view::npos)
	{
		std::cerr << "missing SurfelPathTracer struct semicolon in " << label << ": " << marker << '\n';
		ok = false;
		return {};
	}
	return std::string(haystack.substr(start, semicolon - start + 1u));
}

bool containsAllNeedles(std::string_view haystack, std::initializer_list<std::string_view> needles)
{
	bool ok = true;
	for (std::string_view needle : needles)
	{
		if (!containsNeedle(haystack, needle))
		{
			std::cerr << "missing SurfelPathTracer contract: " << needle << '\n';
			ok = false;
		}
	}
	return ok;
}

bool appearsBefore(std::string_view haystack, std::string_view first, std::string_view second)
{
	const auto firstPos = haystack.find(first);
	const auto secondPos = haystack.find(second);
	if (firstPos == std::string_view::npos || secondPos == std::string_view::npos || firstPos >= secondPos)
	{
		std::cerr << "SurfelPathTracer pass-order contract failed: " << first << " before " << second << '\n';
		return false;
	}
	return true;
}

bool occursAtLeast(std::string_view haystack, std::string_view needle, size_t expectedCount)
{
	size_t count = 0;
	size_t offset = 0;
	while ((offset = haystack.find(needle, offset)) != std::string_view::npos)
	{
		++count;
		offset += needle.size();
	}
	if (count < expectedCount)
	{
		std::cerr << "SurfelPathTracer contract expected at least " << expectedCount
		          << " occurrences of: " << needle << '\n';
		return false;
	}
	return true;
}

bool testSurfelPathTracerPipelineContractFiles()
{
	const std::filesystem::path root = sourceRoot();
	const std::array<std::filesystem::path, 33> contractFiles = {
	    root / "CMakeLists.txt",
	    root / "src" / "Core" / "EngineAuxiliary.h",
	    root / "src" / "Core" / "UISystem.h",
	    root / "src" / "Core" / "UISystem.cpp",
	    root / "src" / "Core" / "EngineCore.cpp",
	    root / "src" / "SceneManagement" / "SceneNode.h",
	    root / "src" / "Core" / "SurfelPathTracerPipelines.h",
	    root / "src" / "Core" / "SurfelPathTracerPipelines.cpp",
	    root / "src" / "Core" / "SurfelPathTracerResources.h",
	    root / "src" / "Core" / "SurfelPathTracerResources.cpp",
	    root / "src" / "Core" / "SurfelPathTracerPasses.h",
	    root / "src" / "Core" / "SurfelPathTracerPasses.cpp",
	    root / "src" / "shaders" / "SurfelPathTracerCommon.slang",
	    root / "src" / "shaders" / "SurfelPathTracerGBuffer.slang",
	    root / "src" / "shaders" / "SurfelPathTracerGBufferMiss.slang",
	    root / "src" / "shaders" / "SurfelPathTracerGBufferClosestHit.slang",
	    root / "src" / "shaders" / "SurfelPathTracerGBufferAnyHit.slang",
	    root / "src" / "shaders" / "SurfelPathTracerPrepare.slang",
	    root / "src" / "shaders" / "SurfelPathTracerUpdate.slang",
	    root / "src" / "shaders" / "SurfelPathTracerCellInfo.slang",
	    root / "src" / "shaders" / "SurfelPathTracerCellToSurfel.slang",
	    root / "src" / "shaders" / "SurfelPathTracerRaygen.slang",
	    root / "src" / "shaders" / "SurfelPathTracerMiss.slang",
	    root / "src" / "shaders" / "SurfelPathTracerClosestHit.slang",
	    root / "src" / "shaders" / "SurfelPathTracerAnyHit.slang",
	    root / "src" / "shaders" / "SurfelPathTracerIntegrate.slang",
	    root / "src" / "shaders" / "SurfelPathTracerEvaluate.slang",
	    root / "src" / "shaders" / "SurfelPathTracerReflection.slang",
	    root / "src" / "shaders" / "SurfelPathTracerReference.slang",
	    root / "src" / "shaders" / "SurfelPathTracerReflectionFilter.slang",
	    root / "src" / "shaders" / "SurfelPathTracerBilateral.slang",
	    root / "src" / "shaders" / "SurfelPathTracerLightIntegrate.slang",
	    root / "src" / "shaders" / "SurfelPathTracerTaa.slang",
	};

	std::string combined;
	bool filesOk = true;
	for (const auto &file : contractFiles)
	{
		combined += readTextFile(file, filesOk);
		combined += '\n';
	}

	const std::initializer_list<std::string_view> needles = {
	    "RenderMode::SurfelPathTracer",
	    "SurfelPathTracerSettings",
	    "SurfelPathTracerStats",
	    "class SurfelPathTracerPipelines",
	    "class SurfelPathTracerResources",
	    "class SurfelPathTracerPasses",
	    "struct SurfelPathTracerSurfel",
	    "struct SurfelPathTracerCellInfo",
	    "struct SurfelPathTracerCounters",
	    "struct SurfelPathTracerSource",
	    "struct SurfelPathTracerPixelSource",
	    "struct SurfelPathTracerSourceInstance",
	    "struct SurfelPathTracerSourceTransform",
	    "SURFEL_PT_SOURCE_FLAG_VALID",
	    "SURFEL_PT_SOURCE_FLAG_REFRESH_FAILED",
	    "lastSeenFrame",
	    "lastReferencedFrame",
	    "sleepState",
	    "materialKey",
	    "sourceNodeId",
	    "sourceInstanceId",
	    "sourceInstanceCustomIndex",
	    "sourcePrimitiveIndex",
	    "sourceBarycentrics",
	    "sourceObjectPosition",
	    "sourceObjectNormal",
	    "surfelSourceNodeId",
	    "sourceTransformCount",
	    "varianceAndInconsistency",
	    "recycledSurfels",
	    "spawnedSurfels",
	    "removedSurfels",
	    "guidedRays",
	    "cosineRays",
	    "surfelTerminatedPaths",
	    "pathMisses",
	    "cellAddressForPosition",
	    "cameraRelativeCellAddressForPosition",
	    "cameraRelativeCellIndexForPosition",
	    "cameraRelativeCellCoord",
	    "flattenCameraRelativeCellCoord",
	    "coord.x >= -halfDim",
	    "coord.x < upperDim",
	    "for (int z = -1; z <= 1; ++z)",
	    "isSurfelIntersectCell(surfel",
	    "shouldRecycleSurfel",
	    "pushDeadSurfel",
	    "surfelRadius",
	    "lockSurfels",
	    "lastReferencedFrame = counters.Load(SURFEL_PT_COUNTER_FRAME_INDEX_OFFSET)",
	    "lastSeenFrame = counters.Load(SURFEL_PT_COUNTER_FRAME_INDEX_OFFSET)",
	    "needsPersistentReset",
	    "~SurfelPathTracerResources",
	    "SurfelPathTracerResources(const SurfelPathTracerResources &) = delete",
	    "kMinCellDimension",
	    "kMaxCellDimension",
	    "const uint32_t dim = std::clamp(cellDimension, kMinCellDimension, kMaxCellDimension)",
	    "const glm::dvec3 clampedCell",
	    "static_cast<int>(clampedCell.x)",
	    "const uint64_t flat",
	    "irradianceAtlasWidth",
	    "irradianceAtlasHeight",
	    "minRaysPerSurfel",
	    "maxRaysPerSurfel",
	    "rayBudgetScale",
	    "activeMaxDepth",
	    "sleepingMaxDepth",
	    "placementThreshold",
	    "removalThreshold",
	    "varianceSensitivity",
	    "surfelTargetArea",
	    "surfelMinRadius",
	    "surfelMaxRadiusScale",
	    "maxSurfelSamplesPerQuery",
	    "maxRadianceSharingSamples",
	    "enableGuidedSampling",
	    "enableSurfelTermination",
	    "enableRadianceSharing",
	    "enableSurfelPlacement",
	    "enableSurfelRemoval",
	    "enableReferenceValidation",
	    "atlasTileSize",
	    "SurfelCoverage = 10",
	    "ReferenceColor = 11",
	    "ReferenceDifference = 12",
	    "kMaxSurfelPathTracerDebugView",
	    "SurfelPathTracerDebugView::SunVisibility",
	    "SurfelPathTracerSky.slang|main",
	    "SurfelPathTracerGBuffer.slang|main",
	    "SurfelPathTracerGBufferMiss.slang|main",
	    "SurfelPathTracerGBufferClosestHit.slang|main",
	    "SurfelPathTracerGBufferAnyHit.slang|main",
	    "SurfelPathTracerPrepare.slang|main",
	    "SurfelPathTracerUpdate.slang|main",
	    "SurfelPathTracerCellInfo.slang|main",
	    "SurfelPathTracerCellToSurfel.slang|main",
	    "SurfelPathTracerRaygen.slang|main",
	    "SurfelPathTracerMiss.slang|main",
	    "SurfelPathTracerClosestHit.slang|main",
	    "SurfelPathTracerAnyHit.slang|main",
	    "SurfelPathTracerIntegrate.slang|main",
	    "SurfelPathTracerEvaluate.slang|main",
	    "SurfelPathTracerReflection.slang|main",
	    "SurfelPathTracerReference.slang|main",
	    "SurfelPathTracerReflectionFilter.slang|main",
	    "SurfelPathTracerBilateral.slang|main",
	    "SurfelPathTracerLightIntegrate.slang|main",
	    "SurfelPathTracerTaa.slang|main",
	    "recordGBufferPass",
	    "recordPreparePass",
	    "recordUpdatePass",
	    "recordCellInfoPass",
	    "recordCellToSurfelPass",
	    "recordSurfelRayTracePass",
	    "recordIntegratePass",
	    "recordEvaluatePass",
	    "recordReflectionPass",
	    "recordReferencePass",
	    "recordReflectionFilterPass",
	    "recordBilateralPass",
	    "recordLightIntegratePass",
	    "recordTaaPass",
	    "SurfelPathTracerEvaluateMode::Generate",
	    "SurfelPathTracerEvaluateMode::Resolve",
	    "recordImageBarrierGBufferToCompute",
	    "recordStorageBarrierComputeToCompute",
	    "recordStorageBarrierComputeToRt",
	    "recordStorageBarrierRtToCompute",
	    "gBufferRayTracingPipeline",
	    "referenceRayTracingPipeline",
	    "rayTracingDescriptorSetLayout",
	    "surfelPathTracerStorageDescriptorSets",
	    "surfelPathTracerRtDescriptorSets",
	    "gBufferSbt",
	    "referenceSbt",
	    "traceRaysKHR",
	    "SurfelPathTracerGBuffer.slang",
	    "SurfelPathTracerGBufferMiss.slang",
	    "SurfelPathTracerGBufferClosestHit.slang",
	    "SurfelPathTracerGBufferAnyHit.slang",
	    "decodeGBufferSurface",
	    "gBufferMotionMaterial",
	    "gBufferNormal",
	    "gBufferDepth",
	    "lightingImages",
	    "taaHistoryImages",
	    "filteredReflectionHistoryImages",
	    "filteredReflectionHistoryViews",
	    "taaHistoryViews",
	    "RaytracingAccelerationStructure tlas",
	    "node->modelId >= static_cast<int>(Laphria::EngineConfig::kBindlessModelCapacity)",
	    "vk::Format::eR32G32B32A32Sfloat, gBufferMotionMaterialImages",
	    "uint modelId;",
	    "float4(motion, float(payload.modelId), float(payload.materialIndex))",
	    "createComputePipeline(dev, *computePipelineLayout, \"Shaders/SurfelPathTracerPrepare.slang.spv\", \"main\")",
	    "createComputePipeline(dev, *computePipelineLayout, \"Shaders/SurfelPathTracerUpdate.slang.spv\", \"main\")",
	    "createComputePipeline(dev, *computePipelineLayout, \"Shaders/SurfelPathTracerCellInfo.slang.spv\", \"main\")",
	    "createComputePipeline(dev, *computePipelineLayout, \"Shaders/SurfelPathTracerCellToSurfel.slang.spv\", \"main\")",
	    "markPersistentResetConsumed",
	    "needsPersistentResourceRecreate",
	    "waitForSurfelPathTracerIdle",
	    "refreshSurfelPathTracerPersistentResources",
	    "maxSurfelsCapacity",
	    "cellDimensionCapacity",
	    "perCellSurfelLimitCapacity",
	    "surfel path tracer stats fence",
	    "surfel path tracer persistent resource fence",
	    "surfel.flags = 0",
	    "resetPersistent",
	    "perCellSurfelLimit",
	    "rejectedStores",
	    "InterlockedAdd",
	    "updateMsme",
	    "makeSurfelMaterialKey",
	    "packNormalOctahedral",
	    "unpackNormalOctahedral",
	    "SURFEL_PT_RAY_BIAS",
	    "SURFEL_PT_ATLAS_TILE_SIZE",
	    "SURFEL_PT_SLEEP_AWAKE",
	    "SURFEL_PT_SLEEP_SLEEPING",
	    "SURFEL_PT_STATUS_LAST_SEEN",
	    "SURFEL_PT_STATUS_LAST_REFERENCED",
	    "SURFEL_PT_COUNTER_RECYCLED_SURFELS_OFFSET",
	    "SURFEL_PT_COUNTER_SPAWNED_SURFELS_OFFSET",
	    "SURFEL_PT_COUNTER_REMOVED_SURFELS_OFFSET",
	    "SURFEL_PT_COUNTER_GUIDED_RAYS_OFFSET",
	    "SURFEL_PT_COUNTER_COSINE_RAYS_OFFSET",
	    "SURFEL_PT_COUNTER_SURFEL_TERMINATED_PATHS_OFFSET",
	    "SURFEL_PT_COUNTER_PATH_MISSES_OFFSET",
	    "clampLuminance",
	    "ggxSampleDirection",
	    "applyAcesTonemap",
	    "SurfelPathTracerDebugView::ReflectionFiltered",
	    "const uint32_t halfWidth = std::max((width + 1u) / 2u, 1u)",
	    "const uint32_t halfHeight = std::max((height + 1u) / 2u, 1u)",
	    "makeReflectionHistoryKey",
	    "reflectionHistoryKeyMatches",
	    "float currentKey = makeReflectionHistoryKey(centerDepth, centerNormal, centerMaterial)",
	    "bool historyUsable = push.resetHistory == 0u &&",
	    "reflectionHistoryKeyMatches(previousFiltered.a, currentKey)",
	    "float4 filteredValue = float4(clampLuminance(filtered, SURFEL_PT_MAX_RADIANCE_LUMINANCE), currentKey)",
	    "filteredReflectionImages[pixel] = filteredValue",
	    "currentFilteredReflectionHistory[pixel] = filteredValue",
	    "push.resetHistory == 0u ? previousFilteredReflectionHistory[pixel] : float4(0.0)",
	    "gBufferSourceBuffer",
	    "surfelSourceBuffer",
	    "sourceInstanceBuffer",
	    "sourceTransformBuffer",
	    "writeSurfelSource",
	    "refreshAnchoredSurfel",
	    "[[vk::binding(16, 0)]] RWTexture2D<float4> reflectionImages",
	    "[[vk::binding(17, 0)]] RWTexture2D<float4> filteredReflectionImages",
	    "[[vk::binding(23, 0)]] RWTexture2D<float4> previousFilteredReflectionHistory",
	    "[[vk::binding(24, 0)]] RWTexture2D<float4> currentFilteredReflectionHistory",
	    "float4 sampleValue = filteredReflectionImages[samplePixel]",
	    "reflectionImages[pixel] = float4(clampLuminance(filtered, SURFEL_PT_MAX_RADIANCE_LUMINANCE), centerValue.a)",
	    "makeTaaHistoryKey",
	    "taaHistoryKeyMatches",
	    "[[vk::binding(1, 0)]] RWTexture2D<float4> gBufferNormal",
	    "float3 normalRaw = gBufferNormal[pixel].xyz",
	    "float currentKey = makeTaaHistoryKey(depth, motionMaterial, normal)",
	    "uint2 previousPixel = min(uint2(previousUv * float2(push.width, push.height))",
	    "if (taaHistoryKeyMatches(storedHistory.a, currentKey))",
	    "[[vk::binding(25, 0)]] RWTexture2D<float4> previousTaaHistory",
	    "[[vk::binding(26, 0)]] RWTexture2D<float4> currentTaaHistory",
	    "currentTaaHistory[pixel] = float4(blended, currentKey)",
	    "outputImage[pixel] = float4(applyAcesTonemap(blended, ubo.exposure), 1.0)",
	    "SurfelPathTracerSource source{}",
	    "VmaBuffer surfelSourceBuffer",
	    "VmaBuffer sourceInstanceBuffer",
	    "VmaBuffer sourceTransformBuffer",
	    "std::vector<VmaBuffer> gBufferSourceBuffers",
	    "createBuffer(byteSize(maxSurfels, sizeof(SurfelPathTracerSource))",
	    "createBuffer(byteSize(maxSourceInstances, sizeof(SurfelPathTracerSourceInstance))",
	    "createBuffer(byteSize(maxSourceTransforms, sizeof(SurfelPathTracerSourceTransform))",
	    "SurfelPathTracer persistent resource creation failed",
	    "SurfelPathTracer extent resource creation failed",
	    "surfelPathTracerStaticSettings",
	    "makeSurfelPathTracerStaticSettingsSnapshot",
	    "refreshSurfelPathTracerRuntimeResources",
	    "transitionSurfelOutputForBlit",
	    "transitionSwapchainForBlit",
	    "transitionSwapchainForUi",
	    "recordSurfelFinalBlit",
	    "recordSurfelPathTracerNoSceneFallback",
	    "if (!surfelSettings.enabled)",
	    "if (!resourceManager || resourceManager->getModelCount() == 0)",
	    "commandBuffer.dispatch(groupCount16(push.width), groupCount16(push.height), 1)",
	    "const bool updateSurfels = !surfelSettings.lockSurfels || resetPersistent",
	    "const bool historyReady = !ptForceHistoryReset && surfelPathTracerTemporalHistoryValid[fi]",
	    "updateSurfelPathTracerHistoryDescriptors(fi)",
	    "else\n\t{\n\t\tsurfelPathTracerPasses.recordStorageBarrierComputeToCompute(commandBuffer);",
	    "vk::PipelineStageFlagBits2::eComputeShader |\n\t\t                    vk::PipelineStageFlagBits2::eRayTracingShaderKHR",
	    "const bool preserveReflectionDebug =",
	    "SurfelPathTracerDebugView::ReflectionFiltered",
	    "const bool taaHistoryEnabled =",
	    "historyReady &&",
	    "surfelSettings.debugView == UISystem::SurfelPathTracerDebugView::FinalColor",
	    "if (taaHistoryEnabled)",
	    "surfelPathTracerTemporalHistoryValid.fill(false)",
	    "std::swap(surfelPathTracerPreviousHistoryIndex[fi], surfelPathTracerCurrentHistoryIndex[fi])",
	    "if (push.enabled == 0u)",
	    "outputImage[pixel] = float4(applyAcesTonemap(currentLighting, ubo.exposure), 1.0)",
	    "!isFinite3(rawNormal) || length(rawNormal) <= 0.001",
	    "!isFinite3(sampleNormalRaw) || length(sampleNormalRaw) <= 0.001",
	    "bool validNormal = isFinite3(normalRaw) && length(normalRaw) > 0.001",
	    "[[vk::binding(28, 0)]]",
	    "[[vk::binding(28, 1)]]",
	    "[[vk::binding(29, 0)]]",
	    "[[vk::binding(30, 1)]]",
	    "[[vk::binding(31, 0)]]",
	    "std::array<vk::DescriptorSetLayoutBinding, 32>",
	    "halfWidth,\n\t                           halfHeight,\n\t                           1)",
	};

	const bool ok = containsAllNeedles(combined, needles);
	const std::string updateShader = readTextFile(root / "src" / "shaders" / "SurfelPathTracerUpdate.slang", filesOk);
	const std::string evaluateShader =
	    readTextFile(root / "src" / "shaders" / "SurfelPathTracerEvaluate.slang", filesOk);
	const std::string cellToSurfelShader =
	    readTextFile(root / "src" / "shaders" / "SurfelPathTracerCellToSurfel.slang", filesOk);
	const std::string commonShader =
	    readTextFile(root / "src" / "shaders" / "SurfelPathTracerCommon.slang", filesOk);
	const std::string raygenShader = readTextFile(root / "src" / "shaders" / "SurfelPathTracerRaygen.slang", filesOk);
	const std::string gBufferShader =
	    readTextFile(root / "src" / "shaders" / "SurfelPathTracerGBuffer.slang", filesOk);
	const std::string gBufferClosestHitShader =
	    readTextFile(root / "src" / "shaders" / "SurfelPathTracerGBufferClosestHit.slang", filesOk);
	const std::string gBufferMissShader =
	    readTextFile(root / "src" / "shaders" / "SurfelPathTracerGBufferMiss.slang", filesOk);
	const std::string gBufferAnyHitShader =
	    readTextFile(root / "src" / "shaders" / "SurfelPathTracerGBufferAnyHit.slang", filesOk);
	const std::string surfelClosestHitShader =
	    readTextFile(root / "src" / "shaders" / "SurfelPathTracerClosestHit.slang", filesOk);
	const std::string integrateShader =
	    readTextFile(root / "src" / "shaders" / "SurfelPathTracerIntegrate.slang", filesOk);
	const std::string reflectionShader =
	    readTextFile(root / "src" / "shaders" / "SurfelPathTracerReflection.slang", filesOk);
	const std::string referenceShader =
	    readTextFile(root / "src" / "shaders" / "SurfelPathTracerReference.slang", filesOk);
	const std::string engineCore = readTextFile(root / "src" / "Core" / "EngineCore.cpp", filesOk);
	const std::string uiSystemHeader = readTextFile(root / "src" / "Core" / "UISystem.h", filesOk);
	const std::string passesHeader = readTextFile(root / "src" / "Core" / "SurfelPathTracerPasses.h", filesOk);
	const std::string passesCpp = readTextFile(root / "src" / "Core" / "SurfelPathTracerPasses.cpp", filesOk);
	auto findAfter = [&](std::string_view needle, size_t start) {
		const auto pos = engineCore.find(needle, start);
		if (pos == std::string::npos)
		{
			std::cerr << "missing SurfelPathTracer pass marker: " << needle << '\n';
		}
		return pos;
	};
	auto before = [&](size_t first, size_t second, std::string_view label) {
		if (first == std::string::npos || second == std::string::npos || first >= second)
		{
			std::cerr << "SurfelPathTracer pass-order contract failed: " << label << '\n';
			return false;
		}
		return true;
	};

	const auto gbuffer = findAfter("recordGBufferPass", 0);
	const auto gbufferToCompute = findAfter("recordImageBarrierGBufferToCompute", gbuffer);
	const auto prepare = findAfter("recordPreparePass", gbufferToCompute);
	const auto evaluateGenerateCall = findAfter("recordEvaluatePass", prepare);
	const auto evaluateGenerate = findAfter("SurfelPathTracerEvaluateMode::Generate", evaluateGenerateCall);
	const auto update = findAfter("recordUpdatePass", evaluateGenerate);
	const auto cellInfo = findAfter("recordCellInfoPass", update);
	const auto cellToSurfel = findAfter("recordCellToSurfelPass", cellInfo);
	const auto cellToSurfelToRt = findAfter("recordStorageBarrierComputeToRt", cellToSurfel);
	const auto surfelRayTrace = findAfter("recordSurfelRayTracePass", cellToSurfelToRt);
	const auto surfelRtToCompute = findAfter("recordStorageBarrierRtToCompute", surfelRayTrace);
	const auto integrate = findAfter("recordIntegratePass", surfelRtToCompute);
	const auto evaluateResolveCall = findAfter("recordEvaluatePass", integrate);
	const auto evaluateResolve = findAfter("SurfelPathTracerEvaluateMode::Resolve", evaluateResolveCall);
	const auto resolveToRt = findAfter("recordStorageBarrierComputeToRt", evaluateResolve);
	const auto reflection = findAfter("recordReflectionPass", resolveToRt);
	const auto reflectionRtToCompute = findAfter("recordStorageBarrierRtToCompute", reflection);
	const auto reflectionFilter = findAfter("recordReflectionFilterPass", reflectionRtToCompute);
	const auto bilateral = findAfter("recordBilateralPass", reflectionFilter);
	const auto referenceGate = findAfter("if (surfelSettings.enableReferenceValidation)", bilateral);
	const auto referenceComputeToRt = findAfter("recordStorageBarrierComputeToRt", referenceGate);
	const auto referenceRecord = findAfter("recordReferencePass", referenceComputeToRt);
	const auto referenceRtToCompute = findAfter("recordStorageBarrierRtToCompute", referenceRecord);
	const auto referenceClear = findAfter("clearColorImage(*surfelPathTracerResources.referenceImages[fi]", referenceGate);
	const auto lightIntegrate = findAfter("recordLightIntegratePass", referenceGate);
	const auto lightIntegrateBarrier = findAfter("recordStorageBarrierComputeToCompute", lightIntegrate);
	const auto taa = findAfter("recordTaaPass", lightIntegrateBarrier);

	bool task17Ok = before(gbuffer, prepare, "GBuffer before Prepare") &&
	    before(gbuffer, gbufferToCompute, "GBuffer before GBuffer-to-compute barrier") &&
	    before(gbufferToCompute, prepare, "GBuffer-to-compute barrier before Prepare") &&
	    before(evaluateGenerateCall, evaluateGenerate, "Generate Evaluate call before Generate mode") &&
	    before(prepare, evaluateGenerate, "Prepare before Generate Evaluate") &&
	    before(evaluateGenerate, update, "Generate Evaluate before Update") &&
	    before(update, cellInfo, "Update before CellInfo") &&
	    before(cellInfo, cellToSurfel, "CellInfo before CellToSurfel") &&
	    before(cellToSurfel, cellToSurfelToRt, "CellToSurfel before compute-to-RT barrier") &&
	    before(cellToSurfelToRt, surfelRayTrace, "compute-to-RT barrier before Surfel RayTrace") &&
	    before(cellToSurfel, surfelRayTrace, "CellToSurfel before Surfel RayTrace") &&
	    before(surfelRayTrace, surfelRtToCompute, "Surfel RayTrace before RT-to-compute barrier") &&
	    before(surfelRtToCompute, integrate, "RT-to-compute barrier before Integrate") &&
	    before(surfelRayTrace, integrate, "Surfel RayTrace before Integrate") &&
	    before(evaluateResolveCall, evaluateResolve, "Resolve Evaluate call before Resolve mode") &&
	    before(integrate, evaluateResolve, "Integrate before Resolve Evaluate") &&
	    before(evaluateResolve, resolveToRt, "Resolve Evaluate before reflection compute-to-RT barrier") &&
	    before(resolveToRt, reflection, "reflection compute-to-RT barrier before Reflection") &&
	    before(evaluateResolve, reflection, "Resolve Evaluate before Reflection") &&
	    before(reflection, reflectionRtToCompute, "Reflection before RT-to-compute barrier") &&
	    before(reflectionRtToCompute, reflectionFilter, "RT-to-compute barrier before ReflectionFilter") &&
	    before(reflection, reflectionFilter, "Reflection before ReflectionFilter") &&
	    before(reflectionFilter, bilateral, "ReflectionFilter before Bilateral") &&
	    before(bilateral, referenceGate, "Bilateral before reference validation branch") &&
	    before(referenceGate, referenceComputeToRt, "reference branch before compute-to-RT barrier") &&
	    before(referenceComputeToRt, referenceRecord, "reference compute-to-RT barrier before Reference") &&
	    before(referenceRecord, referenceRtToCompute, "Reference before RT-to-compute barrier") &&
	    before(referenceRtToCompute, lightIntegrate, "reference RT-to-compute barrier before LightIntegrate") &&
	    before(referenceGate, referenceClear, "reference branch before disabled-reference clear") &&
	    before(referenceClear, lightIntegrate, "disabled-reference clear before LightIntegrate") &&
	    before(referenceRecord, lightIntegrate, "Reference before LightIntegrate") &&
	    before(bilateral, lightIntegrate, "Bilateral before LightIntegrate") &&
	    before(lightIntegrate, lightIntegrateBarrier, "LightIntegrate before compute barrier") &&
	    before(lightIntegrateBarrier, taa, "LightIntegrate compute barrier before TAA") &&
	    before(lightIntegrate, taa, "LightIntegrate before TAA") &&
	    containsAllNeedles(engineCore,
	                       {"recordImageBarrierGBufferToCompute",
	                        "recordStorageBarrierComputeToCompute",
	                        "recordStorageBarrierComputeToRt",
	                        "recordStorageBarrierRtToCompute",
	                        "transitionSurfelOutputForBlit"}) &&
	    containsAllNeedles(passesCpp,
	                       {"recordImageBarrierGBufferToCompute",
	                        "recordStorageBarrierComputeToCompute",
	                        "recordStorageBarrierComputeToRt",
	                        "recordStorageBarrierRtToCompute"}) &&
	    containsAllNeedles(uiSystemHeader,
	                       {"uint32_t maxSurfels = 150000;",
	                        "uint32_t maxRaysPerFrame = 150000 * 16;",
	                        "uint32_t minRaysPerSurfel = 4;",
	                        "uint32_t maxRaysPerSurfel = 64;",
	                        "uint32_t activeMaxDepth = 3;",
	                        "uint32_t sleepingMaxDepth = 5;",
	                        "uint32_t maxSurfelSamplesPerQuery = 32;",
	                        "uint32_t maxRadianceSharingSamples = 32;"});
	const bool task5Ok =
	    containsAllNeedles(updateShader,
	                       {"[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;",
	                        "for (int z = -1; z <= 1; ++z)",
	                        "!isSurfelIntersectCell(surfel, cellCoord, ubo.cameraPos.xyz, push.cellSize)",
	                        "cellCounterBuffer.InterlockedAdd(cellCountOffset, 1u);"}) &&
	    containsAllNeedles(cellToSurfelShader,
	                       {"[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;",
	                        "for (int z = -1; z <= 1; ++z)",
	                        "!isSurfelIntersectCell(surfel, cellCoord, ubo.cameraPos.xyz, push.cellSize)",
	                        "cellToSurfelBuffer[cellInfo.surfelOffset + cellSlot] = surfelIndex;",
	                        "counters.InterlockedAdd(SURFEL_PT_COUNTER_REJECTED_STORES_OFFSET, 1u);"}) &&
	    occursAtLeast(passesCpp, "const std::array descriptorSets = {imageSet, globalSet};", 4);
	bool task6Ok =
	    appearsBefore(updateShader, "if (pushDeadSurfel(surfelIndex, push.maxSurfels))", "surfel.flags = 0u;") &&
	    containsAllNeedles(updateShader,
	                       {"push.lockSurfels == 0u && shouldRecycleSurfel",
	                        "counters.InterlockedAdd(SURFEL_PT_COUNTER_RECYCLED_SURFELS_OFFSET, 1u);"}) &&
	    containsAllNeedles(evaluateShader,
	                       {"if (stampClosest && closestSurfelIndex != SURFEL_PT_INVALID_INDEX)",
	                        "surfel.lastReferencedFrame = counters.Load(SURFEL_PT_COUNTER_FRAME_INDEX_OFFSET)",
	                        "surfel.lastSeenFrame = counters.Load(SURFEL_PT_COUNTER_FRAME_INDEX_OFFSET)"});
	if (containsNeedle(updateShader, "SURFEL_PT_COUNTER_ALIVE_SURFELS_OFFSET"))
	{
		std::cerr << "SurfelPathTracer lifecycle contract must not decrement aliveSurfels in Update\n";
		task6Ok = false;
	}
	if (containsNeedle(evaluateShader, "SURFEL_PT_COUNTER_ALIVE_SURFELS_OFFSET") ||
	    containsNeedle(evaluateShader, "aliveBuffer[aliveSlot]"))
	{
		std::cerr << "SurfelPathTracer allocation contract must not accumulate aliveSurfels across frames\n";
		task6Ok = false;
	}
	task6Ok = task6Ok &&
	    containsAllNeedles(readTextFile(root / "src" / "Core" / "SurfelPathTracerResources.cpp", filesOk),
	                       {"const uint32_t deadSurfels = std::min(counters->deadSurfels, settings_.maxSurfels);",
	                        "stats.deadSurfels = deadSurfels;",
	                        "stats.aliveSurfels = settings_.maxSurfels - deadSurfels;"});
	bool task7Ok =
	    containsAllNeedles(commonShader,
	                       {"SURFEL_PT_SURFEL_FLAG_ACTIVE",
	                        "SURFEL_PT_SURFEL_FLAG_PENDING_FREE",
	                        "bool isActiveSurfel(SurfelPathTracerSurfel surfel)"}) &&
	    containsAllNeedles(updateShader,
	                       {"if ((surfel.flags & SURFEL_PT_SURFEL_FLAG_PENDING_FREE) != 0u)",
	                        "if (!isActiveSurfel(surfel))",
	                        "counters.InterlockedAdd(SURFEL_PT_COUNTER_REMOVED_SURFELS_OFFSET, 1u);"}) &&
	    appearsBefore(updateShader,
	                  "if ((surfel.flags & SURFEL_PT_SURFEL_FLAG_PENDING_FREE) != 0u)",
	                  "if (!isActiveSurfel(surfel))") &&
	    containsAllNeedles(evaluateShader,
	                       {"float placementThreshold;",
	                        "float removalThreshold;",
	                        "float surfelTargetArea;",
	                        "float surfelMinRadius;",
	                        "float surfelMaxRadiusScale;",
	                        "uint enablePlacement;",
	                        "uint enableRemoval;",
	                        "coverage += weight;",
	                        "push.enablePlacement != 0u",
	                        "push.enableRemoval != 0u",
	                        "surfel.flags = SURFEL_PT_SURFEL_FLAG_ACTIVE",
	                        "SURFEL_PT_SURFEL_FLAG_PENDING_FREE",
	                        "push.cellSize * push.surfelMaxRadiusScale",
	                        "uint frameIndex = counters.Load(SURFEL_PT_COUNTER_FRAME_INDEX_OFFSET);",
	                        "estimateCoverage(position, normal, closestSurfelIndex, false)",
	                        "estimateCoverage(position, normal, closestSurfelIndex, true)",
	                        "allocateSurfel(pixel, position, normal, radius, resolveSurfelRadiance"}) &&
	    containsAllNeedles(passesCpp,
	                       {"placementThreshold",
	                        "removalThreshold",
	                        "surfelTargetArea",
	                        "surfelMinRadius",
	                        "surfelMaxRadiusScale"}) &&
	    containsAllNeedles(engineCore,
	                       {"surfelSettings.placementThreshold",
	                        "surfelSettings.removalThreshold",
	                        "surfelSettings.surfelTargetArea",
	                        "surfelSettings.surfelMinRadius",
	                        "surfelSettings.surfelMaxRadiusScale",
	                        "enableSurfelPlacement",
	                        "enableSurfelRemoval"});
	for (std::string_view shader : {updateShader,
	                                evaluateShader,
	                                cellToSurfelShader,
	                                raygenShader,
	                                integrateShader,
	                                reflectionShader})
	{
		if (!containsNeedle(shader, "isActiveSurfel(surfel)"))
		{
			std::cerr << "SurfelPathTracer active-surfel contract missing in shader\n";
			task7Ok = false;
		}
	}
	if (containsNeedle(evaluateShader, "pushDeadSurfel"))
	{
		std::cerr << "SurfelPathTracer Evaluate must only mark pending frees, not push dead surfels\n";
		task7Ok = false;
	}
	bool task8Ok =
	    containsAllNeedles(commonShader,
	                       {"float2 dirToOctUv(float3 direction)",
	                        "uint2 atlasTileBase(uint surfelIndex, uint atlasWidth)",
	                        "uint2 atlasDirectionCoord(float3 localDirection)",
	                        "SURFEL_PT_RAY_LOCAL_DIRECTION"}) &&
	    containsAllNeedles(integrateShader,
	                       {"float3 loadRayLocalDirection(uint rayIndex)",
	                        "void writeAtlas(uint surfelIndex, float3 localDirection, float3 radiance, float depth, uint width, uint height)",
	                        "uint2 coord = atlasTileBase(surfelIndex, width) + atlasDirectionCoord(localDirection)",
	                        "irradianceAtlas.GetDimensions(atlasWidth, atlasHeight)",
	                        "surfelDepthAtlas[coord] = depth",
	                        "writeAtlas(surfelIndex, localDirection, rayRadiance * integrationWeight, rayDepth, atlasWidth, atlasHeight)"}) &&
	    containsAllNeedles(raygenShader,
	                       {"worldToLocalDirection(rayDir, normal)",
	                        "packRayLocalDirection(localRayDir)",
	                        "rayBuffer[base + SURFEL_PT_RAY_LOCAL_DIRECTION]"}) &&
	    containsAllNeedles(readTextFile(root / "src" / "Core" / "SurfelPathTracerResources.cpp", filesOk),
	                       {"const uint64_t tileCount = static_cast<uint64_t>(settings_.maxSurfels)",
	                        "const uint32_t atlasTileSize = std::max(settings_.atlasTileSize, 1u)",
	                        "const uint64_t tilesPerRow = std::max<uint64_t>(atlasWidth / atlasTileSize, 1u)",
	                        "const uint64_t requiredHeight = requiredRows * atlasTileSize",
	                        "SurfelPathTracer irradiance atlas is too small for maxSurfels"});
	bool task9Ok =
	    containsAllNeedles(commonShader,
	                       {"uint adaptiveRayCountForSurfel(SurfelPathTracerSurfel surfel",
	                        "uint minRays",
	                        "uint maxRays",
	                        "float varianceSensitivity",
	                        "uint ageSinceSeen = currentFrame >= surfel.lastSeenFrame ? currentFrame - surfel.lastSeenFrame : 0u;",
	                        "surfel.sleepState == SURFEL_PT_SLEEP_SLEEPING",
	                        "return clamp(count, safeMinRays, safeMaxRays);"}) &&
	    containsAllNeedles(updateShader,
	                       {"uint minRaysPerSurfel;",
	                        "uint maxRaysPerSurfel;",
	                        "float varianceSensitivity;",
	                        "uint rayRequestCount = adaptiveRayCountForSurfel(surfel",
	                        "push.minRaysPerSurfel",
	                        "push.maxRaysPerSurfel",
	                        "push.varianceSensitivity",
	                        "surfel.sleepState = SURFEL_PT_SLEEP_SLEEPING;",
	                        "surfel.sleepState = SURFEL_PT_SLEEP_AWAKE;",
	                        "counters.InterlockedAdd(SURFEL_PT_COUNTER_REQUESTED_RAYS_OFFSET, rayRequestCount, rayOffset)",
	                        "if (writableRayCount < rayRequestCount)"}) &&
	    containsAllNeedles(passesCpp,
	                       {"minRaysPerSurfel",
	                        "maxRaysPerSurfel",
	                        "varianceSensitivity"}) &&
	    containsAllNeedles(integrateShader,
	                       {"updateMsme(sampleRadiance, surfel, shortAlpha)"}) &&
	    containsAllNeedles(engineCore,
	                       {"surfelSettings.minRaysPerSurfel",
	                        "surfelSettings.maxRaysPerSurfel",
	                        "surfelSettings.varianceSensitivity",
	                        "surfelPathTracerResources.maxRaysPerFrameCapacity()"});
	bool task10Ok =
	    containsAllNeedles(raygenShader,
	                       {"uint enableGuidedSampling;",
	                        "uint irradianceAtlasWidth;",
	                        "[[vk::binding(14, 1)]] RWTexture2D<float4> irradianceAtlas;",
	                        "float luminanceAtAtlasCoord(uint2 coord, RWTexture2D<float4> atlas)",
	                        "bool hasGuidedSurfelHistory(SurfelPathTracerSurfel surfel)",
	                        "return surfel.shortMeanAndLife.w > 2.0;",
	                        "float3 cosineSampleHemisphereLocal(float2 xi)",
	                        "bool sampleGuidedSurfelDirection(uint surfelIndex",
	                        "if (texelDirection.z > 0.0)",
	                        "if (centeredDirection.z <= 0.0)",
	                        "if (localDirection.z <= 0.0)",
	                        "return false;",
	                        "pdf = max(weight / total, 1e-5);",
	                        "push.enableGuidedSampling != 0u",
	                        "hasGuidedSurfelHistory(surfel)",
	                        "sampleGuidedSurfelDirection(surfelIndex, guideXi, guideJitter, irradianceAtlas, push.irradianceAtlasWidth, localDir, pdf)",
	                        "localDir = cosineSampleHemisphereLocal(xi);",
	                        "pdf = max(localDir.z / PI, 1e-5);",
	                        "counters.InterlockedAdd(SURFEL_PT_COUNTER_COSINE_RAYS_OFFSET, 1u);",
	                        "counters.InterlockedAdd(SURFEL_PT_COUNTER_GUIDED_RAYS_OFFSET, 1u);",
	                        "float3 rayDir = localToWorldDirection(localDir, normal);",
	                        "rayBuffer[base + SURFEL_PT_RAY_LOCAL_DIRECTION] = packRayLocalDirection(localRayDir);"}) &&
	    containsAllNeedles(passesHeader,
	                       {"uint32_t rayCount,\n\t                              bool enableGuidedSampling,\n\t                              uint32_t irradianceAtlasWidth,"}) &&
	    containsAllNeedles(passesCpp,
	                       {"uint32_t enableGuidedSampling = 0;",
	                        "uint32_t irradianceAtlasWidth = 1;",
	                        "uint32_t rayCount,\n                                                      bool enableGuidedSampling,\n                                                      uint32_t irradianceAtlasWidth,",
	                        ".enableGuidedSampling = enableGuidedSampling ? 1u : 0u",
	                        ".irradianceAtlasWidth = std::max(irradianceAtlasWidth, 1u)"}) &&
	    containsAllNeedles(engineCore,
	                       {"surfelPathTracerResources.maxRaysPerFrameCapacity(),\n\t\t                                                surfelSettings.enableGuidedSampling,\n\t\t                                                surfelSettings.irradianceAtlasWidth"});
	const std::string sceneNodeHeader =
	    readTextFile(root / "src" / "SceneManagement" / "SceneNode.h", filesOk);
	const std::string engineCoreHeader =
	    readTextFile(root / "src" / "Core" / "EngineCore.h", filesOk);
	bool sourceInstanceTablesOk =
	    containsAllNeedles(sceneNodeHeader,
	                       {"#include <cstdint>",
	                        "uint32_t surfelSourceNodeId = UINT32_MAX;"}) &&
	    containsAllNeedles(engineCoreHeader,
	                       {"mutable std::vector<Laphria::SurfelPathTracerSourceInstance> surfelSourceInstances;",
	                        "mutable std::vector<Laphria::SurfelPathTracerSourceTransform> surfelSourceTransforms;",
	                        "mutable uint32_t currentSurfelSourceInstanceCount = 0u;",
	                        "mutable uint32_t currentSurfelSourceTransformCount = 0u;",
	                        "mutable uint32_t nextSurfelSourceNodeId = 0u;",
	                        "mutable std::array<Laphria::VulkanUtils::VmaBuffer, MAX_FRAMES_IN_FLIGHT> surfelSourceInstanceStagingBuffers;",
	                        "mutable std::array<Laphria::VulkanUtils::VmaBuffer, MAX_FRAMES_IN_FLIGHT> surfelSourceTransformStagingBuffers;",
	                        "mutable std::array<void *, MAX_FRAMES_IN_FLIGHT> surfelSourceInstanceStagingMapped{};",
	                        "mutable std::array<void *, MAX_FRAMES_IN_FLIGHT> surfelSourceTransformStagingMapped{};",
	                        "mutable std::array<vk::DeviceSize, MAX_FRAMES_IN_FLIGHT> surfelSourceInstanceStagingSizes{};",
	                        "mutable std::array<vk::DeviceSize, MAX_FRAMES_IN_FLIGHT> surfelSourceTransformStagingSizes{};"}) &&
	    containsAllNeedles(engineCore,
	                       {"surfelSourceInstances.clear();",
	                        "surfelSourceTransforms.clear();",
	                        "currentSurfelSourceInstanceCount = 0u;",
	                        "currentSurfelSourceTransformCount = 0u;",
	                        "uint32_t nextSourceNodeId = 0u;",
	                        "for (const auto &node : scene->getAllNodes())",
	                        "if (node->surfelSourceNodeId != UINT32_MAX)",
	                        "nextSourceNodeId = std::max(nextSourceNodeId, node->surfelSourceNodeId + 1u);",
	                        "if (node->surfelSourceNodeId == UINT32_MAX)",
	                        "node->surfelSourceNodeId = nextSourceNodeId++;",
	                        "const uint32_t sourceNodeId = node->surfelSourceNodeId;",
	                        "sourceNodeId >= surfelPathTracerResources.maxSourceTransforms",
	                        "exceeds SurfelPathTracer source transform capacity",
	                        "surfelSourceTransforms.resize(sourceNodeId + 1u);",
	                        "sourceTransform.objectToWorld = node->getWorldTransform();",
	                        "sourceTransform.worldToObject = glm::inverse(sourceTransform.objectToWorld);",
	                        "sourceTransform.flags = Laphria::SURFEL_PT_SOURCE_FLAG_VALID;",
	                        "surfelSourceTransforms[sourceNodeId] = sourceTransform;",
	                        "surfelSourceInstances.size() >= surfelPathTracerResources.maxSourceInstances",
	                        "exceeds SurfelPathTracer source instance capacity",
	                        "const uint32_t sourceInstanceId = static_cast<uint32_t>(surfelSourceInstances.size());",
	                        "sourceInstanceId > 0x00FFFFFFu",
	                        "exceeds Vulkan 24-bit instance custom index range",
	                        "sourceInstance.sourceNodeId = sourceNodeId;",
	                        "sourceInstance.modelId = static_cast<uint32_t>(node->modelId);",
	                        "sourceInstance.primitiveOffset = primitiveOffset;",
	                        "sourceInstance.flags = Laphria::SURFEL_PT_SOURCE_FLAG_VALID;",
	                        "surfelSourceInstances.push_back(sourceInstance);",
	                        "const uint32_t legacyCustomIndex = (static_cast<uint32_t>(node->modelId) << 14u) | (primitiveOffset & 0x3FFFu);",
	                        "instance.instanceCustomIndex = legacyCustomIndex;",
	                        "nextSurfelSourceNodeId = nextSourceNodeId;",
	                        "auto &instanceStagingBuffer = surfelSourceInstanceStagingBuffers[frames.frameIndex];",
	                        "auto &instanceStagingMapped = surfelSourceInstanceStagingMapped[frames.frameIndex];",
	                        "auto &instanceStagingSize = surfelSourceInstanceStagingSizes[frames.frameIndex];",
	                        "auto &transformStagingBuffer = surfelSourceTransformStagingBuffers[frames.frameIndex];",
	                        "auto &transformStagingMapped = surfelSourceTransformStagingMapped[frames.frameIndex];",
	                        "auto &transformStagingSize = surfelSourceTransformStagingSizes[frames.frameIndex];",
	                        "commandBuffer.copyBuffer(*instanceStagingBuffer, *surfelPathTracerResources.sourceInstanceBuffer,",
	                        "commandBuffer.copyBuffer(*transformStagingBuffer, *surfelPathTracerResources.sourceTransformBuffer,",
	                        "currentSurfelSourceInstanceCount = static_cast<uint32_t>(surfelSourceInstances.size());",
	                        "currentSurfelSourceTransformCount = static_cast<uint32_t>(surfelSourceTransforms.size());"});
	sourceInstanceTablesOk = sourceInstanceTablesOk &&
	                         appearsBefore(engineCore,
	                                       "surfelSourceInstances.clear();",
	                                       "recordSurfelPathTracerCommandBuffer(commandBuffer, imageIndex)") &&
	                         !containsNeedle(engineCore, "surfelSourceInstanceStagingBuffer;") &&
	                         !containsNeedle(engineCore, "surfelSourceTransformStagingBuffer;") &&
	                         !containsNeedle(engineCore, "instance.instanceCustomIndex = sourceInstanceId;") &&
	                         !containsNeedle(engineCore,
	                                         "uint32_t customIndex = (node->modelId << 14) | (primitiveOffset & 0x3FFF);");
	bool rtPushConstantStagesOk =
	    containsAllNeedles(passesCpp,
	                       {"kSurfelRtPushStages",
	                        "vk::ShaderStageFlagBits::eRaygenKHR |",
	                        "vk::ShaderStageFlagBits::eClosestHitKHR |",
	                        "vk::ShaderStageFlagBits::eMissKHR |",
	                        "vk::ShaderStageFlagBits::eAnyHitKHR",
	                        "pushConstants<SurfelRayTracePushConstants>(*pipelines.rayTracingPipelineLayout,\n\t                                                         kSurfelRtPushStages",
	                        "pushConstants<SurfelReflectionPushConstants>(*pipelines.rayTracingPipelineLayout,\n\t                                                           kSurfelRtPushStages"});
	bool materialAlbedoOk =
	    containsAllNeedles(combined,
	                       {"gBufferAlbedo",
	                        "gBufferAlbedoImages",
	                        "gBufferAlbedoViews",
	                        "gBufferMaterialImages",
	                        "gBufferMaterialViews",
	                        "gBufferEmissiveImages",
	                        "gBufferEmissiveViews",
	                        "vk::Format::eR16G16B16A16Sfloat, gBufferAlbedoImages",
	                        "vk::Format::eR16G16B16A16Sfloat, gBufferMaterialImages",
	                        "vk::Format::eR16G16B16A16Sfloat, gBufferEmissiveImages",
	                        "vk::DescriptorSetLayoutBinding{.binding = 20",
	                        "vk::DescriptorSetLayoutBinding{.binding = 21",
	                        "vk::DescriptorSetLayoutBinding{.binding = 22",
	                        "vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 18 * MAX_FRAMES_IN_FLIGHT}",
	                        "[[vk::binding(20, 0)]] RWTexture2D<float4> gBufferAlbedo",
	                        "[[vk::binding(20, 1)]] RWTexture2D<float4> gBufferAlbedo",
	                        "[[vk::binding(21, 0)]] RWTexture2D<float4> gBufferMaterial",
	                        "[[vk::binding(22, 0)]] RWTexture2D<float4> gBufferEmissive",
	                        "[[vk::binding(21, 1)]] RWTexture2D<float4> gBufferMaterial",
	                        "[[vk::binding(22, 1)]] RWTexture2D<float4> gBufferEmissive",
	                        "[[vk::binding(8, 0)]] Sampler2D globalTextures[]",
	                        "baseColor.rgb *= decodeColorSample(sampled.rgb, ubo.textureColorSpaceModel)",
	                        "roughness *= mr.g",
	                        "metallic *= mr.b",
	                        "emissive *= decodeColorSample",
	                        "ao = 1.0 + mat.occlusionStrength * (aoSample - 1.0)",
	                        "dielectricSpec *= globalTextures[NonUniformResourceIndex(mat.specularTextureIndex + mat.globalTextureOffset)].SampleLevel(uv, 0.0).a",
	                        "lerp(float3(0.08 * dielectricSpec), baseColor.rgb, metallic)",
	                        "payload.baseColor",
	                        "gBufferAlbedo[launchID] = float4(payload.baseColor, 1.0)",
	                        "gBufferMaterial[launchID] = float4(payload.roughness, payload.metallic, payload.dielectricSpec, payload.ao)",
	                        "gBufferEmissive[launchID] = float4(payload.emissive, 0.0)",
	                        "float3 albedo = max(gBufferAlbedo[pixel].rgb, float3(0.0))",
	                        "float3 material = gBufferMaterial[pixel].rgb",
	                        "float3 emissive = max(gBufferEmissive[pixel].rgb, float3(0.0))",
	                        "float3 evaluatePrimaryDirectLighting(float3 viewDir",
	                        "float3 directLighting = evaluatePrimaryDirectLighting(viewDir",
	                        "directLighting + diffuseGi + reflection + emissive"});
	const std::string lightIntegrateShader =
	    readTextFile(root / "src" / "shaders" / "SurfelPathTracerLightIntegrate.slang", filesOk);
	bool surfelPrimaryLightingUnitsOk =
	    containsAllNeedles(lightIntegrateShader,
	                       {"float3 diffuseBsdf = albedo",
	                        "float3 F = fresnelSchlick(max(dot(H, V), 0.0), f0)",
	                        "float D = distributionGGX(normal, H, roughness)",
	                        "float G = geometrySmith(normal, V, L, roughness)",
	                        "float3 kD = (float3(1.0) - F) * (1.0 - metallic)",
	                        "float3 diffuse = kD * baseColor / PI",
	                        "float3 specular = (D * G * F) / max(4.0 * max(dot(normal, V), 0.0001) * directSun, 0.0001)",
	                        "diffuseGi = push.enableDiffuseGi != 0u ? diffuseBsdf * rawSurfelRadiance * SURFEL_PT_DIFFUSE_GI_SCALE : float3(0.0)"}) &&
	    !containsNeedle(lightIntegrateShader, "float3 diffuseLighting = albedo * directLighting") &&
	    !containsNeedle(lightIntegrateShader, "float3 directLighting = SUN_RADIANCE * directSun * sunVisibility") &&
	    !containsNeedle(lightIntegrateShader, "albedo * rawSurfelRadiance");
	bool surfelDiffuseGiOk =
	    containsAllNeedles(evaluateShader,
	                       {"float3 sampleWeightedSurfelRadiance(float3 position, float3 normal)",
	                        "uint maxSamples = max(push.maxSurfelSamplesPerQuery, 1u)",
	                        "for (int shell = 0; shell <= 1; ++shell)",
	                        "if (max(abs(x), max(abs(y), abs(z))) != shell)",
	                        "int3 cellCoord = centerCoord + int3(x, y, z)",
	                        "if (samplesVisited >= maxSamples)",
	                        "weightedRadiance += clampLuminance(surfel.radiance, SURFEL_PT_MAX_RADIANCE_LUMINANCE) * weight",
	                        "float influenceRadius = max(surfel.radius * 2.0, push.cellSize * 0.75)",
	                        "length(position - surfel.position) / max(influenceRadius, 0.0001)",
	                        "return weightSum > 0.0001 ? weightedRadiance / weightSum : float3(0.0)",
	                        "float3 resolveSurfelRadiance(float3 position, float3 normal)",
	                        "return sampleWeightedSurfelRadiance(position, normal)",
	                        "outputImage[pixel] = float4(resolveSurfelRadiance(position, normal), coverage)"}) &&
	    containsAllNeedles(lightIntegrateShader,
	                       {"static const float SURFEL_PT_DIFFUSE_GI_SCALE = 1.00",
	                        "float3 rawSurfelRadiance = max(outputImage[pixel].rgb, float3(0.0))",
	                        "diffuseGi = push.enableDiffuseGi != 0u ? diffuseBsdf * rawSurfelRadiance * SURFEL_PT_DIFFUSE_GI_SCALE : float3(0.0)"});
	bool surfelIncidentRadianceOk =
	    containsAllNeedles(surfelClosestHitShader,
	                       {"payload.radiance = float3(0.0)",
	                        "payload.hitKind = 1u",
	                        "payload.hitT = RayTCurrent()",
	                        "payload.hitPosition = WorldRayOrigin() + WorldRayDirection() * RayTCurrent()",
	                        "float tangentLengthSq = dot(T, T)",
	                        "if (tangentLengthSq > 0.0001)",
	                        "T = T * rsqrt(tangentLengthSq)",
	                        "T = T - N * dot(N, T)",
	                        "payload.f0 = lerp(float3(0.08 * dielectricSpec), payload.baseColor, payload.metallic)"}) &&
	    containsAllNeedles(gBufferClosestHitShader,
	                       {"float tangentLengthSq = dot(T, T)",
	                        "if (tangentLengthSq > 0.0001)",
	                        "T = T * rsqrt(tangentLengthSq)",
	                        "T = T - worldNormal * dot(worldNormal, T)"}) &&
	    containsAllNeedles(combined,
	                       {"SurfelPathTracerPayload makeEmptySurfelPayload()",
	                        "SurfelPathTracerPayload makeVisibilitySurfelPayload()",
	                        "struct SurfelPathTracerPayload",
	                        "float3 hitPosition",
	                        "float3 hitNormal",
	                        "float3 baseColor",
	                        "float3 emission",
	                        "float3 f0",
	                        "float ao",
	                        "float metallic",
	                        "float roughness",
	                        "uint hitKind",
	                        "uint instanceID",
	                        "float traceSunVisibility(float3 position, float3 normal, float3 sunDir)",
	                        "float3 evaluateShadowedDirectLighting(float3 viewDir",
	                        "float3 terminatePathWithSurfels(float3 position",
	                        "float3 traceSurfelPath(RayDesc initialRay, uint maxDepth, uint surfelIndex, uint seed, out float firstHitT)",
	                        "payload.hitKind == 0u",
	                        "radiance += throughput * payload.radiance",
	                        "radiance += throughput * payload.emission",
	                        "push.enableSurfelTermination != 0u",
	                        "counters.InterlockedAdd(SURFEL_PT_COUNTER_SURFEL_TERMINATED_PATHS_OFFSET, 1u)",
	                        "throughput *= bsdfWeight / max(pdf, 0.0001)",
	                        "[[vk::binding(11, 1)]] RWStructuredBuffer<SurfelPathTracerCellInfo> cellInfoBuffer",
	                        "[[vk::binding(13, 1)]] RWStructuredBuffer<uint> cellToSurfelBuffer"}) &&
	    containsAllNeedles(readTextFile(root / "src" / "shaders" / "SurfelPathTracerMiss.slang", filesOk),
	                       {"payload.radiance = clampLuminance(evalSkyColor",
	                        "payload.emission = float3(0.0)",
	                        "payload.hitKind = 0u",
	                        "payload.hitT = -1.0"}) &&
	    containsAllNeedles(readTextFile(root / "src" / "shaders" / "SurfelPathTracerReflection.slang", filesOk),
	                       {"uint enableSurfelTermination;",
	                        "uint maxSurfelSamplesPerQuery;",
	                        "SurfelPathTracerPayload payload = makeEmptySurfelPayload()",
	                        "terminatePathWithSurfels(payload.hitPosition",
	                        "push.enableSurfelTermination != 0u"});
	if (containsNeedle(surfelClosestHitShader, "incidentLighting") ||
	    containsNeedle(surfelClosestHitShader, "evalSkyColor(worldNormal") ||
	    containsNeedle(surfelClosestHitShader, "SUN_RADIANCE * direct"))
	{
		std::cerr << "SurfelPathTracer closest hit must expose material data, not incident lighting\n";
		surfelIncidentRadianceOk = false;
	}
	const std::initializer_list<std::string_view> gBufferPayloadFields = {
	    "struct SurfelPathTracerGBufferPayload",
	    "float hitT;",
	    "uint modelId;",
	    "uint materialIndex;",
	    "float3 worldNormal;",
	    "float3 baseColor;",
	    "float roughness;",
	    "float metallic;",
	    "float dielectricSpec;",
	    "float ao;",
	    "float3 emissive;"};
	bool gBufferPayloadBlocksOk = true;
	const std::string gBufferPayloadBlock =
	    extractStructBlock(gBufferShader, "SurfelPathTracerGBufferPayload", "GBuffer raygen", gBufferPayloadBlocksOk);
	const std::string gBufferClosestHitPayloadBlock =
	    extractStructBlock(gBufferClosestHitShader, "SurfelPathTracerGBufferPayload", "GBuffer closest-hit", gBufferPayloadBlocksOk);
	const std::string gBufferAnyHitPayloadBlock =
	    extractStructBlock(gBufferAnyHitShader, "SurfelPathTracerGBufferPayload", "GBuffer any-hit", gBufferPayloadBlocksOk);
	const std::string gBufferMissPayloadBlock =
	    extractStructBlock(gBufferMissShader, "SurfelPathTracerGBufferPayload", "GBuffer miss", gBufferPayloadBlocksOk);
	const std::string normalizedGBufferPayloadBlock = normalizeContractText(gBufferPayloadBlock);
	auto gBufferPayloadBlockMatches = [&](std::string_view label, const std::string &payloadBlock) {
		if (normalizeContractText(payloadBlock) != normalizedGBufferPayloadBlock)
		{
			std::cerr << "SurfelPathTracer GBuffer payload ABI mismatch in " << label << '\n';
			return false;
		}
		return true;
	};
	const bool gBufferClosestHitPayloadMatches =
	    gBufferPayloadBlockMatches("closest-hit", gBufferClosestHitPayloadBlock);
	const bool gBufferAnyHitPayloadMatches = gBufferPayloadBlockMatches("any-hit", gBufferAnyHitPayloadBlock);
	const bool gBufferMissPayloadMatches = gBufferPayloadBlockMatches("miss", gBufferMissPayloadBlock);
	gBufferPayloadBlocksOk =
	    gBufferPayloadBlocksOk &&
	    gBufferClosestHitPayloadMatches &&
	    gBufferAnyHitPayloadMatches &&
	    gBufferMissPayloadMatches;
	bool gBufferPayloadAbiOk =
	    gBufferPayloadBlocksOk &&
	    containsAllNeedles(gBufferShader, gBufferPayloadFields) &&
	    containsAllNeedles(gBufferClosestHitShader, gBufferPayloadFields) &&
	    containsAllNeedles(gBufferMissShader, gBufferPayloadFields) &&
	    containsAllNeedles(gBufferAnyHitShader, gBufferPayloadFields) &&
	    containsAllNeedles(gBufferShader,
	                       {"uint sourceNodeId;",
	                        "uint sourceInstanceId;",
	                        "uint sourceInstanceCustomIndex;",
	                        "uint sourcePrimitiveIndex;",
	                        "float2 sourceBarycentrics;",
	                        "float3 sourceObjectPosition;",
	                        "float3 sourceObjectNormal;"}) &&
	    containsAllNeedles(gBufferClosestHitShader,
	                       {"uint sourceNodeId;",
	                        "uint sourceInstanceId;",
	                        "uint sourceInstanceCustomIndex;",
	                        "uint sourcePrimitiveIndex;",
	                        "float2 sourceBarycentrics;",
	                        "float3 sourceObjectPosition;",
	                        "float3 sourceObjectNormal;",
	                        "uint sourceInstanceId = InstanceIndex()",
	                        "uint sourceInstanceCustomIndex = instanceId",
	                        "SurfelPathTracerSourceInstance sourceInstance = sourceInstanceBuffer[sourceInstanceId]",
	                        "payload.sourceObjectPosition = surface.sourceObjectPosition",
	                        "payload.sourceObjectNormal = surface.sourceObjectNormal"}) &&
	    containsAllNeedles(gBufferAnyHitShader,
	                       {"[[vk::binding(30, 1)]] StructuredBuffer<SurfelPathTracerSourceInstance> sourceInstanceBuffer",
	                        "uint sourceInstanceId = InstanceIndex()",
	                        "SurfelPathTracerSourceInstance sourceInstance = sourceInstanceBuffer[sourceInstanceId]",
	                        "uint modelId = sourceInstance.modelId",
	                        "uint primitiveOffset = sourceInstance.primitiveOffset",
	                        "uint materialIndex = primitiveOffset + GeometryIndex()",
	                        "MaterialData mat = globalMaterials[NonUniformResourceIndex(modelId)][materialIndex]"}) &&
	    containsAllNeedles(gBufferShader,
	                       {"SurfelPathTracerGBufferPayload makeVisibilityGBufferPayload()",
	                        "payload.hitT = 1.0;",
	                        "SurfelPathTracerGBufferPayload shadowPayload = makeVisibilityGBufferPayload()",
	                        "RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER"}) &&
	    containsAllNeedles(gBufferMissShader,
	                       {"payload.roughness = 1.0;",
	                        "payload.metallic = 0.0;",
	                        "payload.dielectricSpec = 1.0;",
	                        "payload.ao = 1.0;",
	                        "payload.emissive = float3(0.0)"});
	if (containsNeedle(gBufferShader, "TraceRay(tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,"))
	{
		std::cerr << "SurfelPathTracer GBuffer visibility rays must skip closest-hit and start occluded\n";
		gBufferPayloadAbiOk = false;
	}
	bool gBufferSourceWriteOk =
	    containsAllNeedles(gBufferShader,
	                       {"void writeSurfelSource(uint2 pixel, uint2 launchSize, SurfelPathTracerGBufferPayload payload)",
	                        "bool isValidSourceNormal(float3 normal)",
	                        "source.sourceFlags = 0u",
	                        "source.sourceFlags = SURFEL_PT_SOURCE_FLAG_VALID",
	                        "source.sourceObjectNormal = packNormalOctahedral(payload.sourceObjectNormal)",
	                        "source.sourceMaterialKey = makeSurfelMaterialKey(payload.modelId, payload.materialIndex)",
	                        "gBufferSourceBuffer[pixelIndex] = source",
	                        "clearSurfelSource(launchID, launchSize)",
	                        "writeSurfelSource(launchID, launchSize, payload)"}) &&
	    appearsBefore(gBufferShader,
	                  "if (!isValidSourceNormal(payload.sourceObjectNormal))",
	                  "source.sourceObjectNormal = packNormalOctahedral(payload.sourceObjectNormal)");
	bool reflectionSurfelTerminationOk =
	    containsAllNeedles(reflectionShader,
	                       {"float3 cachedIncident = terminatePathWithSurfels(payload.hitPosition",
	                        "payload.hitNormal",
	                        "(1.0 - payload.metallic) * payload.baseColor / PI * cachedIncident",
	                        "float3 reflected = payload.radiance;"});
	if (containsNeedle(reflectionShader,
	                   "float3 cachedIncident = terminatePathWithSurfels(position,\n                                                             normal,"))
	{
		std::cerr << "SurfelPathTracer reflection hit termination must sample the reflected-hit endpoint\n";
		reflectionSurfelTerminationOk = false;
	}
	if (containsNeedle(reflectionShader, "terminatePathWithSurfels(position") ||
	    containsNeedle(reflectionShader, "} else if (push.enableSurfelTermination != 0u) {"))
	{
		std::cerr << "SurfelPathTracer reflection miss branch must not terminate against the primary surface cache\n";
		reflectionSurfelTerminationOk = false;
	}
	bool reflectionRisOk =
	    containsAllNeedles(commonShader,
	                       {"float ggxPdf(float3 L, float3 N, float3 V, float roughness)",
	                        "float3 ggxSpecularBrdf(float3 L, float3 N, float3 V, float3 f0, float roughness)",
	                        "struct SurfelPtReservoir",
	                        "void addReflectionCandidate(inout SurfelPtReservoir reservoir"}) &&
	    containsAllNeedles(reflectionShader,
	                       {"float4 material = gBufferMaterial[fullPixel]",
	                        "const uint candidateCount = 16u",
	                        "ggxSampleDirection(candidateXi, normal, viewDir, roughness)",
	                        "ggxSpecularBrdf(candidateDir, normal, viewDir, f0, roughness)",
	                        "ggxPdf(candidateDir, normal, viewDir, roughness)",
	                        "addReflectionCandidate(reservoir",
	                        "reservoir.weightSum",
	                        "bool reservoirFinite = isFiniteScalar(reservoir.weightSum) &&",
	                        "isFiniteScalar(reservoir.pdf) &&",
	                        "isFinite3(reservoir.brdfWeight)",
	                        "!reservoirFinite || reservoir.weightSum <= 1e-5",
	                        "float risWeight = reservoir.weightSum / max(luminance(reservoir.brdfWeight), 1e-5)",
	                        "reservoir.brdfWeight",
	                        "float nDotSelected = max(dot(reservoir.direction, normal), 0.0)"});
	if (containsNeedle(reflectionShader, "float3 rayDir = normalize(ggxSampleDirection("))
	{
		std::cerr << "SurfelPathTracer reflection must use RIS candidates, not the old single GGX sample ray\n";
		reflectionRisOk = false;
	}
	bool guidedRayIntegrationOk =
	    containsAllNeedles(integrateShader,
	                       {"float loadRayPdf(uint rayIndex)",
	                        "float3 loadRayLocalDirection(uint rayIndex)",
	                        "float pdf = max(loadRayPdf(rayIndex), 1e-5)",
	                        "float cosine = max(localDirection.z, 0.0)",
	                        "float integrationWeight = cosine / pdf",
	                        "float3 weightedRadiance = rayRadiance * integrationWeight",
	                        "accumulatedRadiance += weightedRadiance",
	                        "writeAtlas(surfelIndex, localDirection, rayRadiance * integrationWeight, rayDepth, atlasWidth, atlasHeight)"});
	if (containsNeedle(integrateShader, "accumulatedRadiance += rayRadiance;"))
	{
		std::cerr << "SurfelPathTracer guided rays must be integrated with cosine/pdf weighting\n";
		guidedRayIntegrationOk = false;
	}
	bool msmRadianceSharingOk =
	    containsAllNeedles(commonShader,
	                       {"float3 updateMsme(float3 sampleRadiance, inout SurfelPathTracerSurfel surfel, float shortBlend)",
	                        "float3 previousMean = surfel.meanAndVariance.xyz",
	                        "float3 previousShort = surfel.shortMeanAndLife.xyz",
	                        "float3 shortMean = lerp(previousShort, sampleRadiance, saturate(shortBlend))",
	                        "float3 longMean = lerp(previousMean, sampleRadiance, saturate(shortBlend * 0.25))",
	                        "float variance = lerp(surfel.meanAndVariance.w, dot(delta, delta), saturate(shortBlend * 0.5))",
	                        "surfel.varianceAndInconsistency = float4(abs(shortMean - longMean), variance)",
	                        "surfel.radiance = longMean",
	                        "return longMean"}) &&
	    !containsNeedle(commonShader, "float3 msmeBlend(") &&
	    containsAllNeedles(integrateShader,
	                       {"uint enableRadianceSharing;",
	                        "uint maxRadianceSharingSamples;",
	                        "uint cellDimension;",
	                        "float cellSize;",
	                        "[[vk::binding(11, 0)]] RWStructuredBuffer<SurfelPathTracerCellInfo> cellInfoBuffer;",
	                        "[[vk::binding(13, 0)]] RWStructuredBuffer<uint> cellToSurfelBuffer;",
	                        "[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;",
	                        "float3 terminatePathWithSurfels(float3 position",
	                        "uint cellIndex = cameraRelativeCellIndexForPosition(position, cameraPosition, cellSize, cellDimension)",
	                        "sum += surfel.radiance * weight",
	                        "if (push.enableRadianceSharing != 0u)",
	                        "float3 shared = terminatePathWithSurfels(surfel.position",
	                        "unpackNormalOctahedral(surfel.packedNormal)",
	                        "push.maxRadianceSharingSamples",
	                        "push.cellSize",
	                        "push.cellDimension",
	                        "ubo.cameraPos.xyz",
	                        "sampleRadiance = lerp(sampleRadiance, shared, 0.25)",
	                        "float3 longMean = updateMsme(sampleRadiance, surfel, shortAlpha)"}) &&
	    containsAllNeedles(passesHeader,
	                       {"void recordIntegratePass(const vk::raii::CommandBuffer &commandBuffer,",
	                        "vk::DescriptorSet globalSet",
	                        "bool enableRadianceSharing",
	                        "uint32_t maxRadianceSharingSamples",
	                        "uint32_t cellDimension",
	                        "float cellSize"}) &&
	    containsAllNeedles(passesCpp,
	                       {"uint32_t enableRadianceSharing = 0;",
	                        "uint32_t maxRadianceSharingSamples = 1;",
	                        "uint32_t cellDimension = 1;",
	                        "float cellSize = 1.0f;",
	                        "const std::array descriptorSets = {imageSet, globalSet};",
	                        ".enableRadianceSharing = enableRadianceSharing ? 1u : 0u",
	                        ".maxRadianceSharingSamples = std::clamp(maxRadianceSharingSamples, 1u, 128u)",
	                        ".cellDimension = std::max(cellDimension, 1u)",
	                        ".cellSize = std::max(cellSize, 0.0001f)"}) &&
	    containsAllNeedles(engineCore,
	                       {"surfelSettings.enableRadianceSharing",
	                        "surfelSettings.maxRadianceSharingSamples",
	                        "surfelPathTracerResources.cellDimensionCapacity()",
	                        "surfelSettings.cellSize"});
	bool debugViewsOk =
	    containsAllNeedles(combined,
	                       {"GBufferAlbedo = 13",
	                        "DiffuseGi = 14",
	                        "SunVisibility = 15",
	                        "kMaxSurfelPathTracerDebugView =\n        SurfelPathTracerDebugView::SunVisibility",
	                        "\"GBuffer Albedo\"",
	                        "\"Diffuse GI\"",
	                        "\"Sun Visibility\"",
	                        "static_cast<int>(SurfelPathTracerDebugView::SunVisibility)",
	                        "static const uint SURFEL_DEBUG_SURFEL_RADIUS = 4u",
	                        "static const uint SURFEL_DEBUG_SURFEL_VARIANCE = 6u",
	                        "static const uint SURFEL_DEBUG_GBUFFER_ALBEDO = 13u",
	                        "static const uint SURFEL_DEBUG_DIFFUSE_GI = 14u",
	                        "static const uint SURFEL_DEBUG_SUN_VISIBILITY = 15u",
	                        "static const uint SURFEL_DEBUG_SURFEL_COVERAGE = 10u",
	                        "static const uint SURFEL_DEBUG_REFERENCE_COLOR = 11u",
	                        "static const uint SURFEL_DEBUG_REFERENCE_DIFFERENCE = 12u",
	                        "SurfelPathTracerSurfel surfel;",
	                        "bool loadSelectedContributingSurfel(float3 worldPosition",
	                        "int3 centerCoord = cameraRelativeCellCoord(worldPosition, ubo.cameraPos.xyz, push.cellSize)",
	                        "float bestWeight = 0.0",
	                        "for (int shell = 0; shell <= 1; ++shell)",
	                        "int3 cellCoord = centerCoord + int3(x, y, z)",
	                        "float normalWeight = saturate(dot(normal, surfelNormal))",
	                        "float distanceWeight = saturate(1.0 - length(worldPosition - surfel.position) / max(surfel.radius, 0.0001))",
	                        "float weight = normalWeight * distanceWeight",
	                        "if (weight > bestWeight)",
	                        "loadSelectedContributingSurfel(worldPosition, normal, surfel)",
	                        "surfel.radius / max(push.cellSize * push.surfelMaxRadiusScale, 1e-4)",
	                        "surfel.varianceAndInconsistency.xyz",
	                        "float surfelCoverage = max(outputImage[pixel].a, 0.0)",
	                        "lightingImages[pixel] = float4(albedo, 1.0)",
	                        "lightingImages[pixel] = float4(rawSurfelRadiance, 1.0)",
	                        "push.debugView == SURFEL_DEBUG_REFERENCE_COLOR",
	                        "push.debugView == SURFEL_DEBUG_REFERENCE_DIFFERENCE",
	                        "lightingImages[pixel] = float4(referenceColor",
	                        "lightingImages[pixel] = float4(referenceDifference",
	                        "lightingImages[pixel] = float4(diffuseGi, 1.0)",
	                        "lightingImages[pixel] = float4(sunVisibility.xxx, 1.0)",
	                        "lightingImages[pixel] = float4(surfelCoverage.xxx, 1.0)",
	                        "outputImage[pixel] = float4(resolveSurfelRadiance(position, normal), coverage)",
	                        "ImGui::Text(\"Recycled Surfels: %u\", surfelPathTracerStats.recycledSurfels)",
	                        "ImGui::Text(\"Spawned Surfels: %u\", surfelPathTracerStats.spawnedSurfels)",
	                        "ImGui::Text(\"Removed Surfels: %u\", surfelPathTracerStats.removedSurfels)",
	                        "ImGui::Text(\"Guided Rays: %u\", surfelPathTracerStats.guidedRays)",
	                        "ImGui::Text(\"Cosine Rays: %u\", surfelPathTracerStats.cosineRays)",
	                        "ImGui::Text(\"Surfel-Terminated Paths: %u\", surfelPathTracerStats.surfelTerminatedPaths)"});
	const bool referenceDifferenceUsesReferenceLighting =
	    containsNeedle(lightIntegrateShader, "referenceDifference") &&
	    (containsNeedle(lightIntegrateShader, "referenceColor - lighting") ||
	     containsNeedle(lightIntegrateShader, "lighting - referenceColor"));
	if (!referenceDifferenceUsesReferenceLighting)
	{
		std::cerr << "SurfelPathTracer ReferenceDifference must compare referenceColor against current lighting\n";
		debugViewsOk = false;
	}
	if (containsNeedle(lightIntegrateShader, "bool loadSelectedSurfel(float3 worldPosition") ||
	    containsNeedle(lightIntegrateShader, "lightingImages[pixel] = float4(max(abs(referenceColor - rawSurfelRadiance), float3(0.002)), 1.0)") ||
	    containsNeedle(lightIntegrateShader, "abs(referenceColor - rawSurfelRadiance)"))
	{
		std::cerr << "SurfelPathTracer debug views must use weighted selected surfel and reference/current-lighting difference\n";
		debugViewsOk = false;
	}
	const bool referenceDifferenceZeroWhenUnavailable =
	    containsNeedle(lightIntegrateShader, "referenceSample.a") &&
	    containsNeedle(lightIntegrateShader, "abs(referenceColor - lighting)") &&
	    (containsNeedle(lightIntegrateShader, "referenceSample.a > 0.0 ? abs(referenceColor - lighting)") ||
	     containsNeedle(lightIntegrateShader, "referenceSample.a <= 0.0 ? float3(0.0)"));
	if (!referenceDifferenceZeroWhenUnavailable ||
	    containsNeedle(lightIntegrateShader, "float3(0.002)") ||
	    containsNeedle(lightIntegrateShader, "referenceSample.a > 0.0 ? float3(0.0)"))
	{
		std::cerr << "SurfelPathTracer ReferenceDifference must be zero when reference alpha is 0 and real abs(reference-lighting) when alpha is valid\n";
		debugViewsOk = false;
	}
	bool cellOccupancyDebugOk =
	    containsAllNeedles(readTextFile(root / "src" / "shaders" / "SurfelPathTracerLightIntegrate.slang", filesOk),
	                       {"static const uint SURFEL_DEBUG_CELL_OCCUPANCY = 7u",
	                        "[[vk::binding(11, 0)]] RWStructuredBuffer<SurfelPathTracerCellInfo> cellInfoBuffer",
	                        "[[vk::binding(12, 0)]] RWByteAddressBuffer cellCounterBuffer",
	                        "uint cellIndex = cameraRelativeCellIndexForPosition(worldPosition",
	                        "uint rawCount = cellCounterBuffer.Load((1u + cellIndex) * 4u)",
	                        "float overflow = saturate(float(rawCount - storedCount) / float(perCellLimit))",
	                        "float3 occupancyColor = float3(overflow, occupancy, 1.0 - occupancy)",
	                        "lightingImages[pixel] = float4(occupancyColor, 1.0)"}) &&
	    containsAllNeedles(passesCpp,
	                       {"float cellSize = 1.0f;",
	                        "uint32_t cellDimension = 1;",
	                        "uint32_t perCellSurfelLimit = 1;",
	                        ".cellSize = std::max(cellSize, 0.0001f)",
	                        ".cellDimension = std::max(cellDimension, 1u)",
	                        ".perCellSurfelLimit = std::max(perCellSurfelLimit, 1u)"}) &&
	    containsAllNeedles(engineCore,
	                       {"surfelSettings.cellSize,",
	                        "surfelPathTracerResources.cellDimensionCapacity(),",
	                        "surfelPathTracerResources.perCellSurfelLimitCapacity(),"});
	bool primarySunVisibilityOk =
	    containsAllNeedles(gBufferShader,
	                       {"SurfelPathTracerGBufferPayload makeEmptyGBufferPayload()",
	                        "SurfelPathTracerGBufferPayload makeVisibilityGBufferPayload()",
	                        "float traceSunVisibility(float3 position, float3 normal, float3 sunDir)",
	                        "RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER",
	                        "float sunVisibility = directSun > 0.0 ? traceSunVisibility(hitPos, normal, sunDir) : 0.0",
	                        "gBufferNormal[launchID] = float4(normal, sunVisibility)"}) &&
	    containsAllNeedles(readTextFile(root / "src" / "shaders" / "SurfelPathTracerLightIntegrate.slang", filesOk),
	                       {"float sunVisibility = saturate(gBufferNormal[pixel].w)",
	                        "float3 directLighting = evaluatePrimaryDirectLighting(viewDir"}) &&
	    containsAllNeedles(gBufferMissShader,
	                       {"float3 baseColor",
	                        "payload.baseColor = float3(0.0, 0.0, 0.0)"}) &&
	    containsAllNeedles(gBufferAnyHitShader,
	                       {"float3 baseColor"});
	if (containsNeedle(readTextFile(root / "src" / "shaders" / "SurfelPathTracerLightIntegrate.slang", filesOk),
	                   "skyAmbient"))
	{
		std::cerr << "SurfelPathTracer final color must not use fake skyAmbient fill lighting\n";
		primarySunVisibilityOk = false;
	}
	bool neighborGatherOk =
	    containsAllNeedles(evaluateShader,
	                       {"uint maxSurfelSamplesPerQuery;",
	                        "uint samplesVisited = 0u;",
	                        "samplesVisited += 1u;",
	                        "if (!isCellCoordValid(cellCoord, dim))",
	                        "uint cellIndex = flattenCameraRelativeCellCoord(cellCoord, dim);"}) &&
	    containsAllNeedles(passesCpp,
	                       {"uint32_t maxSurfelSamplesPerQuery = 1;",
	                        "uint32_t maxSurfelSamplesPerQuery,",
	                        ".maxSurfelSamplesPerQuery = std::clamp(maxSurfelSamplesPerQuery, 1u, 128u)"}) &&
	    containsAllNeedles(engineCore,
	                       {"surfelSettings.maxSurfelSamplesPerQuery,"});
	bool representativeCellOverflowOk =
	    containsAllNeedles(cellToSurfelShader,
	                       {"uint reservoirSlotForCellSurfel(uint surfelIndex, uint cellIndex, uint cellSlot, uint cellCapacity)",
	                        "uint streamLength = cellSlot + 1u;",
	                        "uint candidate = pcgHash(surfelIndex ^ (cellIndex * 1664525u) ^ cellSlot) % streamLength;",
	                        "return candidate < cellCapacity ? candidate : SURFEL_PT_INVALID_INDEX;",
	                        "uint representativeSlotForCellSurfel(SurfelPathTracerSurfel surfel",
	                        "float3 localPosition = saturate((surfel.position - cellMin) / safeCellSize)",
	                        "uint spatialBucket = localBucket.x + localBucket.y * 2u + localBucket.z * 4u",
	                        "uint normalBucket = dominantNormalBucket(unpackNormalOctahedral(surfel.packedNormal))",
	                        "uint bucket = spatialBucket + normalBucket * 8u",
	                        "return bucket % cellCapacity",
	                        "uint replaceSlot = reservoirSlotForCellSurfel(surfelIndex, cellIndex, cellSlot, cellInfo.surfelCount);",
	                        "uint representativeSlot = representativeSlotForCellSurfel(surfel",
	                        "replaceSlot = representativeSlot;",
	                        "cellToSurfelBuffer[cellInfo.surfelOffset + replaceSlot] = surfelIndex;",
	                        "if (replaceSlot == SURFEL_PT_INVALID_INDEX)"});
	bool temporalHistoryPingPongOk =
	    containsAllNeedles(combined,
	                       {"std::array<std::vector<VulkanUtils::VmaImage>, 2> filteredReflectionHistoryImages;",
	                        "std::array<std::vector<vk::raii::ImageView>, 2> filteredReflectionHistoryViews;",
	                        "std::array<std::vector<VulkanUtils::VmaImage>, 2> taaHistoryImages;",
	                        "std::array<std::vector<vk::raii::ImageView>, 2> taaHistoryViews;",
	                        "surfelPathTracerPreviousHistoryIndex.fill(0);",
	                        "surfelPathTracerCurrentHistoryIndex.fill(1);",
	                        "surfelPathTracerTemporalHistoryValid.fill(false);",
	                        "vk::DescriptorSetLayoutBinding{.binding = 23",
	                        "vk::DescriptorSetLayoutBinding{.binding = 24",
	                        "vk::DescriptorSetLayoutBinding{.binding = 25",
	                        "vk::DescriptorSetLayoutBinding{.binding = 26",
	                        "const std::array<uint32_t, 5> imageBindings = {19, 23, 24, 25, 26};",
	                        "if (surfelPathTracerTemporalHistoryValid[frameIndex] && previousHistoryIndex == currentHistoryIndex)",
	                        "updateSurfelPathTracerHistoryDescriptors(fi);",
	                        "const bool historyReady = !ptForceHistoryReset && surfelPathTracerTemporalHistoryValid[fi];",
	                        "std::swap(surfelPathTracerPreviousHistoryIndex[fi], surfelPathTracerCurrentHistoryIndex[fi]);",
	                        "surfelPathTracerTemporalHistoryValid[fi] = true;",
	                        "float4 previousFiltered = push.resetHistory == 0u ? previousFilteredReflectionHistory[pixel] : float4(0.0);",
	                        "float4 filteredValue = float4(clampLuminance(filtered, SURFEL_PT_MAX_RADIANCE_LUMINANCE), currentKey);",
	                        "filteredReflectionImages[pixel] = filteredValue;",
	                        "currentFilteredReflectionHistory[pixel] = filteredValue;",
	                        "float4 storedHistory = previousTaaHistory[previousPixel];",
	                        "currentTaaHistory[pixel] = float4(blended, currentKey);"}) &&
	    !containsNeedle(combined, "const bool historyReady = false") &&
	    !containsNeedle(combined, "currentFilteredReflectionHistory[pixel] = filteredReflectionImages[pixel]") &&
	    !containsNeedle(combined, "samePixelFallback");
	bool referenceValidationOk =
	    containsAllNeedles(combined,
	                       {"std::vector<VulkanUtils::VmaImage> referenceImages;",
	                        "std::vector<vk::raii::ImageView> referenceViews;",
	                        "clearViewsThenDestroyImages(referenceViews, referenceImages);",
	                        "createStorageImageSet(dev, width, height, vk::Format::eR16G16B16A16Sfloat, referenceImages, referenceViews);",
	                        "std::array<vk::DescriptorSetLayoutBinding, 32> storageBindings",
	                        "vk::DescriptorSetLayoutBinding{.binding = 27",
	                        "vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 18 * MAX_FRAMES_IN_FLIGHT}",
	                        "referenceImages",
	                        "referenceViews[i]",
	                        "const std::array<uint32_t, 13> imageBindings = {0, 1, 2, 3, 14, 15, 16, 17, 18, 20, 21, 22, 27};",
	                        "createReferenceRayTracingPipeline",
	                        "createReferenceShaderBindingTable",
	                        "referenceRayTracingPipeline",
	                        "referenceSbt",
	                        "SurfelPathTracerReference.slang.spv",
	                        "recordReferencePass",
	                        "enableReferenceValidation"}) &&
	    containsAllNeedles(referenceShader,
	                       {"[shader(\"raygeneration\")]",
	                        "[[vk::binding(27, 1)]] RWTexture2D<float4> referenceImage;",
	                        "[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;",
	                        "[[vk::binding(0, 2)]] ConstantBuffer<UniformBuffer> ubo;",
	                        "struct SurfelReferencePushConstants",
	                        "uint maxDepth;",
	                        "uint enabled;",
	                        "SurfelPathTracerPayload payload = makeEmptySurfelPayload()",
	                        "SurfelPathTracerPayload shadowPayload = makeVisibilitySurfelPayload()",
	                        "float traceSunVisibility(float3 position, float3 normal, float3 sunDir)",
	                        "float3 evaluateShadowedDirectLighting",
	                        "TraceRay(tlas",
	                        "referenceImage[pixel]"}) &&
	    containsAllNeedles(lightIntegrateShader,
	                       {"[[vk::binding(27, 0)]] RWTexture2D<float4> referenceImage",
	                        "float3 referenceColor = referenceImage[pixel].rgb",
	                        "float3 referenceDifference",
	                        "lightingImages[pixel] = float4(referenceColor",
	                        "lightingImages[pixel] = float4(referenceDifference"});
	if (containsNeedle(referenceShader, "[numthreads(") || containsNeedle(referenceShader, "RWStructuredBuffer<SurfelPathTracerSurfel> surfelBuffer") ||
	    containsNeedle(lightIntegrateShader, "float3 referenceColor = lighting"))
	{
		std::cerr << "SurfelPathTracer reference validation must be a real RT path and not a placeholder comparison\n";
		referenceValidationOk = false;
	}
	const size_t referenceGatePos = engineCore.find("if (surfelSettings.enableReferenceValidation)");
	const size_t referenceRecordPos = engineCore.find("surfelPathTracerPasses.recordReferencePass", referenceGatePos);
	const size_t referenceElsePos = engineCore.find("else", referenceRecordPos);
	const size_t referenceClearPos =
	    engineCore.find("clearColorImage(*surfelPathTracerResources.referenceImages[fi]", referenceElsePos);
	const bool referenceRtIsGated =
	    referenceGatePos != std::string::npos &&
	    referenceRecordPos != std::string::npos &&
	    referenceElsePos != std::string::npos &&
	    referenceClearPos != std::string::npos &&
	    referenceGatePos < referenceRecordPos &&
	    referenceRecordPos < referenceElsePos &&
	    referenceElsePos < referenceClearPos &&
	    containsAllNeedles(engineCore,
	                       {"const vk::ClearColorValue referenceClearColor(0.0f, 0.0f, 0.0f, 0.0f)",
	                        "vk::ImageLayout::eGeneral",
	                        "vk::AccessFlagBits2::eTransferWrite",
	                        "vk::PipelineStageFlagBits2::eTransfer",
	                        "vk::AccessFlagBits2::eShaderRead",
	                        "vk::PipelineStageFlagBits2::eComputeShader"});
	if (!referenceRtIsGated)
	{
		std::cerr << "SurfelPathTracer reference RT pass must be gated by enableReferenceValidation and clear referenceImages when disabled\n";
		referenceValidationOk = false;
	}

	return filesOk && ok && task5Ok && task6Ok && task7Ok && task8Ok && task9Ok && task10Ok && sourceInstanceTablesOk && task17Ok &&
	       rtPushConstantStagesOk && materialAlbedoOk && surfelPrimaryLightingUnitsOk && surfelDiffuseGiOk && surfelIncidentRadianceOk &&
	       gBufferPayloadAbiOk && gBufferSourceWriteOk && reflectionSurfelTerminationOk && guidedRayIntegrationOk && msmRadianceSharingOk &&
	       debugViewsOk && neighborGatherOk && cellOccupancyDebugOk && primarySunVisibilityOk &&
	       representativeCellOverflowOk && reflectionRisOk && temporalHistoryPingPongOk && referenceValidationOk;
}

bool testSurfelPathTracerCellAddressBounds()
{
	const auto atCenter = Laphria::SurfelPathTracerResources::cellAddressForPosition(glm::vec3(0.0f), 1.0f, 64);
	const auto atFar = Laphria::SurfelPathTracerResources::cellAddressForPosition(glm::vec3(100000.0f), 1.0f, 64);
	const glm::vec3 cameraB(100000.0f, 20.0f, -300.0f);
	const glm::vec3 position = cameraB + glm::vec3(1.0f, 0.0f, 0.0f);
	const auto atB =
	    Laphria::SurfelPathTracerResources::cameraRelativeCellAddressForPosition(position, cameraB, 1.0f, 64);
	const auto atNegativeEdge =
	    Laphria::SurfelPathTracerResources::cameraRelativeCellAddressForPosition(cameraB + glm::vec3(-32.0f, 0.0f, 0.0f),
	                                                                             cameraB,
	                                                                             1.0f,
	                                                                             64);
	const auto atPositiveEdge =
	    Laphria::SurfelPathTracerResources::cameraRelativeCellAddressForPosition(cameraB + glm::vec3(31.0f, 0.0f, 0.0f),
	                                                                             cameraB,
	                                                                             1.0f,
	                                                                             64);

	if (atCenter.flatIndex >= 64u * 64u * 64u || atFar.flatIndex >= 64u * 64u * 64u)
	{
		std::cerr << "surfel cell address escaped valid bounds\n";
		return false;
	}
	if (std::abs(atB.coord.x - 33) > 1)
	{
		std::cerr << "camera-relative surfel cell address did not stay near the grid center\n";
		return false;
	}
	if (atNegativeEdge.coord.x != 0 || atPositiveEdge.coord.x != 63)
	{
		std::cerr << "camera-relative surfel cell address lost an even-dimension boundary cell\n";
		return false;
	}
	return true;
}
} // namespace

bool testSurfelPathTracerPipelineContracts()
{
	return testSurfelPathTracerPipelineContractFiles() && testSurfelPathTracerCellAddressBounds();
}
