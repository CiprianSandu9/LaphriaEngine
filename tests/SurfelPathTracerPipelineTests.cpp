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
	const std::array<std::filesystem::path, 7> contractFiles = {
	    root / "CMakeLists.txt",
	    root / "src" / "Core" / "EngineAuxiliary.h",
	    root / "src" / "Core" / "UISystem.h",
	    root / "src" / "Core" / "SurfelPathTracerPipelines.h",
	    root / "src" / "Core" / "SurfelPathTracerResources.h",
	    root / "src" / "Core" / "SurfelPathTracerPasses.h",
	    root / "src" / "shaders" / "SurfelPathTracerCommon.slang",
	};

	std::string combined;
	bool filesOk = true;
	for (const auto &file : contractFiles)
	{
		combined += readTextFile(file, filesOk);
		combined += '\n';
	}

	const std::array<std::string_view, 29> needles = {
	    "RenderMode::SurfelPathTracer",
	    "SurfelPathTracerSettings",
	    "SurfelPathTracerStats",
	    "class SurfelPathTracerPipelines",
	    "class SurfelPathTracerResources",
	    "class SurfelPathTracerPasses",
	    "struct SurfelPathTracerSurfel",
	    "struct SurfelPathTracerCellInfo",
	    "struct SurfelPathTracerCounters",
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
