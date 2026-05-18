#include "SurfelPathTracerPipelineTests.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

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
	const std::array<std::filesystem::path, 19> contractFiles = {
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
	};

	std::string combined;
	bool filesOk = true;
	for (const auto &file : contractFiles)
	{
		combined += readTextFile(file, filesOk);
		combined += '\n';
	}

	const std::array<std::string_view, 84> needles = {
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
	    "recordStorageBarrierComputeToCompute",
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
