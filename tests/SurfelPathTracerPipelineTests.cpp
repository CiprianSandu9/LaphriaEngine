#include "SurfelPathTracerPipelineTests.h"

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

bool containsNeedle(const std::string &haystack, std::string_view needle)
{
	return haystack.find(needle) != std::string::npos;
}
} // namespace

bool testSurfelPathTracerPipelineContracts()
{
	const std::filesystem::path root = sourceRoot();
	const std::array<std::filesystem::path, 30> contractFiles = {
	    root / "CMakeLists.txt",
	    root / "src" / "Core" / "EngineAuxiliary.h",
	    root / "src" / "Core" / "UISystem.h",
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

	const std::vector<std::string_view> needles = {
	    "RenderMode::SurfelPathTracer",
	    "SurfelPathTracerSettings",
	    "SurfelPathTracerStats",
	    "class SurfelPathTracerPipelines",
	    "class SurfelPathTracerResources",
	    "class SurfelPathTracerPasses",
	    "struct SurfelPathTracerSurfel",
	    "struct SurfelPathTracerCellInfo",
	    "struct SurfelPathTracerCounters",
	    "cellAddressForPosition",
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

	bool ok = true;
	for (std::string_view needle : needles)
	{
		if (!containsNeedle(combined, needle))
		{
			std::cerr << "missing SurfelPathTracer contract: " << needle << '\n';
			ok = false;
		}
	}

	return filesOk && ok;
}
