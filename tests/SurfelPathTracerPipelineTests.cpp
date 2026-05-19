#include "SurfelPathTracerPipelineTests.h"

#include "../src/Core/SurfelPathTracerResources.h"

#include <array>
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

bool containsNeedle(std::string_view haystack, std::string_view needle)
{
	return haystack.find(needle) != std::string::npos;
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
	const std::array<std::filesystem::path, 31> contractFiles = {
	    root / "CMakeLists.txt",
	    root / "src" / "Core" / "EngineAuxiliary.h",
	    root / "src" / "Core" / "UISystem.h",
	    root / "src" / "Core" / "UISystem.cpp",
	    root / "src" / "Core" / "EngineCore.cpp",
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
	    "lastSeenFrame",
	    "lastReferencedFrame",
	    "sleepState",
	    "materialKey",
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
	    "SurfelPathTracerDebugView::DiffuseGi",
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
	    "rayTracingDescriptorSetLayout",
	    "surfelPathTracerStorageDescriptorSets",
	    "surfelPathTracerRtDescriptorSets",
	    "gBufferSbt",
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
	    "msmeBlend",
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
	    "filteredReflectionImages[pixel] = float4(clampLuminance(filtered, SURFEL_PT_MAX_RADIANCE_LUMINANCE), currentKey)",
	    "push.resetHistory == 0u ? filteredReflectionImages[pixel] : float4(0.0)",
	    "[[vk::binding(16, 0)]] RWTexture2D<float4> reflectionImages",
	    "[[vk::binding(17, 0)]] RWTexture2D<float4> filteredReflectionImages",
	    "float4 sampleValue = filteredReflectionImages[samplePixel]",
	    "reflectionImages[pixel] = float4(clampLuminance(filtered, SURFEL_PT_MAX_RADIANCE_LUMINANCE), centerValue.a)",
	    "makeTaaHistoryKey",
	    "taaHistoryKeyMatches",
	    "[[vk::binding(1, 0)]] RWTexture2D<float4> gBufferNormal",
	    "float3 normalRaw = gBufferNormal[pixel].xyz",
	    "float currentKey = makeTaaHistoryKey(depth, motionMaterial, normal)",
	    "uint2 previousPixel = min(uint2(previousUv * float2(push.width, push.height))",
	    "bool samePixelFallback = all(abs(int2(previousPixel) - int2(pixel)) <= int2(0))",
	    "if (samePixelFallback && taaHistoryKeyMatches(storedHistory.a, currentKey))",
	    "taaHistoryImages[pixel] = float4(blended, currentKey)",
	    "outputImage[pixel] = float4(applyAcesTonemap(blended, ubo.exposure), 1.0)",
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
	    "Current descriptors expose one history image per frame-in-flight",
	    "const bool historyReady = false",
	    "else\n\t{\n\t\tsurfelPathTracerPasses.recordStorageBarrierComputeToCompute(commandBuffer);",
	    "vk::PipelineStageFlagBits2::eComputeShader |\n\t\t                    vk::PipelineStageFlagBits2::eRayTracingShaderKHR",
	    "const bool preserveReflectionDebug =",
	    "SurfelPathTracerDebugView::ReflectionFiltered",
	    "const bool taaHistoryEnabled =",
	    "historyReady &&",
	    "surfelSettings.debugView == UISystem::SurfelPathTracerDebugView::FinalColor",
	    "if (taaHistoryEnabled)",
	    "surfelPathTracerHistoryValid.fill(false)",
	    "if (push.enabled == 0u)",
	    "outputImage[pixel] = float4(applyAcesTonemap(currentLighting, ubo.exposure), 1.0)",
	    "!isFinite3(rawNormal) || length(rawNormal) <= 0.001",
	    "!isFinite3(sampleNormalRaw) || length(sampleNormalRaw) <= 0.001",
	    "bool validNormal = isFinite3(normalRaw) && length(normalRaw) > 0.001",
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
	const std::string surfelClosestHitShader =
	    readTextFile(root / "src" / "shaders" / "SurfelPathTracerClosestHit.slang", filesOk);
	const std::string integrateShader =
	    readTextFile(root / "src" / "shaders" / "SurfelPathTracerIntegrate.slang", filesOk);
	const std::string reflectionShader =
	    readTextFile(root / "src" / "shaders" / "SurfelPathTracerReflection.slang", filesOk);
	const std::string engineCore = readTextFile(root / "src" / "Core" / "EngineCore.cpp", filesOk);
	const std::string passesCpp = readTextFile(root / "src" / "Core" / "SurfelPathTracerPasses.cpp", filesOk);
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
	                        "uint enablePlacement;",
	                        "uint enableRemoval;",
	                        "coverage += weight;",
	                        "push.enablePlacement != 0u",
	                        "push.enableRemoval != 0u",
	                        "surfel.flags = SURFEL_PT_SURFEL_FLAG_ACTIVE",
	                        "SURFEL_PT_SURFEL_FLAG_PENDING_FREE",
	                        "uint frameIndex = counters.Load(SURFEL_PT_COUNTER_FRAME_INDEX_OFFSET);",
	                        "estimateCoverage(position, normal, closestSurfelIndex, false)",
	                        "estimateCoverage(position, normal, closestSurfelIndex, true)",
	                        "allocateSurfel(pixel, position, normal, radius, resolveSurfelRadiance"}) &&
	    containsAllNeedles(passesCpp,
	                       {"placementThreshold",
	                        "removalThreshold",
	                        "surfelTargetArea",
	                        "surfelMinRadius"}) &&
	    containsAllNeedles(engineCore,
	                       {"surfelSettings.placementThreshold",
	                        "surfelSettings.removalThreshold",
	                        "surfelSettings.surfelTargetArea",
	                        "surfelSettings.surfelMinRadius",
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
	                        "writeAtlas(surfelIndex, localDirection, rayRadiance, rayDepth, atlasWidth, atlasHeight)"}) &&
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
	                       {"surfel.varianceAndInconsistency = float4(variance.xxx, surfel.varianceAndInconsistency.w);"}) &&
	    containsAllNeedles(engineCore,
	                       {"surfelSettings.minRaysPerSurfel",
	                        "surfelSettings.maxRaysPerSurfel",
	                        "surfelSettings.varianceSensitivity",
	                        "surfelPathTracerResources.maxRaysPerFrameCapacity())"});
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
	                        "vk::Format::eR16G16B16A16Sfloat, gBufferAlbedoImages",
	                        "vk::DescriptorSetLayoutBinding{.binding = 20",
	                        "vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 11 * MAX_FRAMES_IN_FLIGHT}",
	                        "[[vk::binding(20, 0)]] RWTexture2D<float4> gBufferAlbedo",
	                        "[[vk::binding(20, 1)]] RWTexture2D<float4> gBufferAlbedo",
	                        "[[vk::binding(8, 0)]] Sampler2D globalTextures[]",
	                        "baseColor.rgb *= decodeColorSample(sampled.rgb, ubo.textureColorSpaceModel)",
	                        "payload.baseColor",
	                        "gBufferAlbedo[launchID] = float4(payload.baseColor, 1.0)",
	                        "float3 albedo = max(gBufferAlbedo[pixel].rgb, float3(0.0))",
	                        "float3 diffuseLighting = albedo * directLighting",
	                        "diffuseLighting + diffuseGi + reflection"});
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
	                        "outputImage[pixel] = float4(resolveSurfelRadiance(position, normal), 1.0)"}) &&
	    containsAllNeedles(readTextFile(root / "src" / "shaders" / "SurfelPathTracerLightIntegrate.slang", filesOk),
	                       {"static const float SURFEL_PT_DIFFUSE_GI_SCALE = 0.35",
	                        "float3 rawSurfelRadiance = max(outputImage[pixel].rgb, float3(0.0))",
	                        "float3 diffuseGi = push.enableDiffuseGi != 0u ? albedo * rawSurfelRadiance * SURFEL_PT_DIFFUSE_GI_SCALE : float3(0.0)"});
	bool surfelIncidentRadianceOk =
	    containsAllNeedles(surfelClosestHitShader,
	                       {"float3 incidentLighting = sky + SUN_RADIANCE * direct",
	                        "payload.radiance = clampLuminance(incidentLighting, SURFEL_PT_MAX_RADIANCE_LUMINANCE)"});
	if (containsNeedle(surfelClosestHitShader, "baseColor * (sky + SUN_RADIANCE * direct)") ||
	    containsNeedle(surfelClosestHitShader, "baseColor * incidentLighting"))
	{
		std::cerr << "SurfelPathTracer surfel cache must store incident lighting, not hit-surface outgoing albedo\n";
		surfelIncidentRadianceOk = false;
	}
	bool debugViewsOk =
	    containsAllNeedles(combined,
	                       {"GBufferAlbedo = 13",
	                        "DiffuseGi = 14",
	                        "kMaxSurfelPathTracerDebugView =\n        SurfelPathTracerDebugView::DiffuseGi",
	                        "\"GBuffer Albedo\"",
	                        "\"Diffuse GI\"",
	                        "static_cast<int>(SurfelPathTracerDebugView::DiffuseGi)",
	                        "static const uint SURFEL_DEBUG_GBUFFER_ALBEDO = 13u",
	                        "static const uint SURFEL_DEBUG_DIFFUSE_GI = 14u",
	                        "lightingImages[pixel] = float4(albedo, 1.0)",
	                        "lightingImages[pixel] = float4(rawSurfelRadiance, 1.0)",
	                        "lightingImages[pixel] = float4(diffuseGi, 1.0)"});
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

	return filesOk && ok && task5Ok && task6Ok && task7Ok && task8Ok && task9Ok &&
	       rtPushConstantStagesOk && materialAlbedoOk && surfelDiffuseGiOk && surfelIncidentRadianceOk &&
	       debugViewsOk && neighborGatherOk && cellOccupancyDebugOk && representativeCellOverflowOk;
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
