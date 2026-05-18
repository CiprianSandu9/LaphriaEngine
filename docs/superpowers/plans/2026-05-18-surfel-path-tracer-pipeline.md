# Surfel Path Tracer Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new native `SurfelPathTracer` rendering backend beside the existing `PathTracer`, adapting the full SurfelGI/SurfelPlus architecture into Laphria's Vulkan/Slang renderer.

**Architecture:** Add a separate backend with its own resources, pipelines, pass recorder, shaders, UI controls, and structural tests. Reuse existing TLAS/BLAS, bindless scene descriptors, swapchain blit, CMake shader compilation, and editor UI composition.

**Tech Stack:** C++20, Vulkan-Hpp RAII, VMA helpers, Slang SPIR-V shaders, CMake, ImGui, existing `LaphriaEngineUnitTests`.

---

## File Structure

Create:

- `src/Core/SurfelPathTracerResources.h`: Host-shared surfel structs, settings, stats, and resource owner declaration.
- `src/Core/SurfelPathTracerResources.cpp`: Buffer/image allocation, cleanup, resize, reset, and CPU helper functions.
- `src/Core/SurfelPathTracerPipelines.h`: Descriptor layout, pipeline layout, compute pipeline, RT pipeline, and SBT owner declaration.
- `src/Core/SurfelPathTracerPipelines.cpp`: Shader loading through the existing pipeline style and SBT creation for the new backend.
- `src/Core/SurfelPathTracerPasses.h`: Command recording API for each pass.
- `src/Core/SurfelPathTracerPasses.cpp`: Per-pass command recording, barriers, dispatches, trace calls, and final blit.
- `src/shaders/SurfelPathTracerCommon.slang`: Shared shader structs/constants/helpers.
- `src/shaders/SurfelPathTracerSky.slang`: First skeleton compute output.
- `src/shaders/SurfelPathTracerGBuffer.slang`: Primary visibility/GBuffer output.
- `src/shaders/SurfelPathTracerPrepare.slang`: Counter clear and transient preparation.
- `src/shaders/SurfelPathTracerUpdate.slang`: Surfel aging, recycling, ray budget, and cell occupancy.
- `src/shaders/SurfelPathTracerCellInfo.slang`: Cell offset accumulation.
- `src/shaders/SurfelPathTracerCellToSurfel.slang`: Cell-to-surfel population.
- `src/shaders/SurfelPathTracerRaygen.slang`: Surfel ray tracing raygen.
- `src/shaders/SurfelPathTracerMiss.slang`: Surfel/reflection miss shader.
- `src/shaders/SurfelPathTracerClosestHit.slang`: Surface payload for surfel/reflection rays.
- `src/shaders/SurfelPathTracerAnyHit.slang`: Alpha cutout handling for surfel/reflection rays.
- `src/shaders/SurfelPathTracerIntegrate.slang`: MSME/radiance integration.
- `src/shaders/SurfelPathTracerEvaluate.slang`: Screen-space diffuse GI evaluation and surfel generation/removal.
- `src/shaders/SurfelPathTracerReflection.slang`: Reduced-resolution glossy reflection trace.
- `src/shaders/SurfelPathTracerReflectionFilter.slang`: Reflection temporal/spatial filter.
- `src/shaders/SurfelPathTracerBilateral.slang`: Bilateral cleanup.
- `src/shaders/SurfelPathTracerLightIntegrate.slang`: Direct + indirect + reflection composition.
- `src/shaders/SurfelPathTracerTaa.slang`: Final TAA/tone map.
- `tests/SurfelPathTracerPipelineTests.h`: Test entrypoint declaration.
- `tests/SurfelPathTracerPipelineTests.cpp`: Structural shader/C++ contract tests.

Modify:

- `CMakeLists.txt`: Add new engine sources, shader entries, shader include dependency, and test source.
- `src/Core/EngineAuxiliary.h`: Add `RenderMode::SurfelPathTracer` and host-shared push constants/stats only when they are genuinely shared.
- `src/Core/EngineCore.h`: Add backend members and command-recording/descriptor lifecycle hooks.
- `src/Core/EngineCore.cpp`: Initialize resources/pipelines, recreate descriptors on model/resize changes, route render mode, and collect stats.
- `src/Core/PipelineCollection.h`: Own the `SurfelPathTracerPipelines` helper.
- `src/Core/PipelineCollection.cpp`: Initialize the helper from existing pipeline creation order.
- `src/Core/UISystem.h`: Add `SurfelPathTracerSettings`, `SurfelPathTracerStats`, and debug enums.
- `src/Core/UISystem.cpp`: Add render-mode radio button and controls.
- `tests/EngineUnitTestsMain.cpp`: Invoke surfel path tracer tests.

---

### Task 1: Add Structural Tests First

**Files:**

- Create: `tests/SurfelPathTracerPipelineTests.h`
- Create: `tests/SurfelPathTracerPipelineTests.cpp`
- Modify: `tests/EngineUnitTestsMain.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the failing test header**

Create `tests/SurfelPathTracerPipelineTests.h`:

```cpp
#ifndef LAPHRIAENGINE_SURFELPATHTRACERPIPELINETESTS_H
#define LAPHRIAENGINE_SURFELPATHTRACERPIPELINETESTS_H

bool testSurfelPathTracerPipelineContracts();

#endif
```

- [ ] **Step 2: Write the failing structural tests**

Create `tests/SurfelPathTracerPipelineTests.cpp`:

```cpp
#include "SurfelPathTracerPipelineTests.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace
{
std::string readText(const std::filesystem::path &path)
{
	std::ifstream file(path, std::ios::binary);
	if (!file)
	{
		return {};
	}
	std::ostringstream out;
	out << file.rdbuf();
	return out.str();
}

bool containsText(const std::string &text, const char *needle)
{
	return text.find(needle) != std::string::npos;
}

std::filesystem::path sourceRoot()
{
#ifdef LAPHRIA_SOURCE_DIR
	return std::filesystem::path(LAPHRIA_SOURCE_DIR);
#else
	return std::filesystem::current_path();
#endif
}
} // namespace

bool testSurfelPathTracerPipelineContracts()
{
	const auto root = sourceRoot();
	const std::string cmake = readText(root / "CMakeLists.txt");
	const std::string engineAux = readText(root / "src/Core/EngineAuxiliary.h");
	const std::string uiHeader = readText(root / "src/Core/UISystem.h");
	const std::string pipelineHeader = readText(root / "src/Core/SurfelPathTracerPipelines.h");
	const std::string resourceHeader = readText(root / "src/Core/SurfelPathTracerResources.h");
	const std::string passesHeader = readText(root / "src/Core/SurfelPathTracerPasses.h");
	const std::string shaderCommon = readText(root / "src/shaders/SurfelPathTracerCommon.slang");

	const char *required[] = {
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
	    "SurfelPathTracerTaa.slang|main"};

	const std::string combined =
	    cmake + engineAux + uiHeader + pipelineHeader + resourceHeader + passesHeader + shaderCommon;
	for (const char *needle : required)
	{
		if (!containsText(combined, needle))
		{
			std::cerr << "missing SurfelPathTracer contract: " << needle << "\n";
			return false;
		}
	}
	return true;
}
```

- [ ] **Step 3: Wire the failing test into the unit-test binary**

Modify `tests/EngineUnitTestsMain.cpp`:

```cpp
#include "SurfelPathTracerPipelineTests.h"
```

Add near the end of `main()`:

```cpp
	const bool okSurfelPathTracer = testSurfelPathTracerPipelineContracts();
	return (okTransform && okFrustum && okBroadphase && okPtSweep && okPtPercentiles &&
	        okPtScore && okPtAovContract && okPtReservoirGiMeasurement &&
	        okPtHistoryClamp && okPtPowerHeuristic && okSurfelPathTracer) ? 0 : 1;
```

- [ ] **Step 4: Add the test source to CMake**

Modify the `LaphriaEngineUnitTests` source list in `CMakeLists.txt`:

```cmake
        tests/SurfelPathTracerPipelineTests.cpp
```

- [ ] **Step 5: Run the test and verify it fails for missing contracts**

Run:

```powershell
cmd /c call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests
.\cmake-build-debug\LaphriaEngine\Debug\LaphriaEngineUnitTests.exe
```

Expected: build succeeds, test executable fails with `missing SurfelPathTracer contract`.

- [ ] **Step 6: Commit**

```powershell
git add CMakeLists.txt tests/EngineUnitTestsMain.cpp tests/SurfelPathTracerPipelineTests.h tests/SurfelPathTracerPipelineTests.cpp
git commit -m "test: add surfel path tracer pipeline contracts"
```

---

### Task 2: Add Render Mode, UI Settings, And Stats

**Files:**

- Modify: `src/Core/EngineAuxiliary.h`
- Modify: `src/Core/UISystem.h`
- Modify: `src/Core/UISystem.cpp`

- [ ] **Step 1: Add the render mode**

Modify `RenderMode` in `src/Core/EngineAuxiliary.h`:

```cpp
enum class RenderMode
{
	Rasterizer,
	RayTracer,
	PathTracer,
	SurfelPathTracer,
};
```

- [ ] **Step 2: Add UI-facing settings and stats**

Add to `UISystem` in `src/Core/UISystem.h`:

```cpp
    enum class SurfelPathTracerDebugView
    {
        FinalColor = 0,
        GBufferNormal = 1,
        GBufferDepth = 2,
        SurfelId = 3,
        SurfelRadius = 4,
        SurfelRadiance = 5,
        SurfelVariance = 6,
        CellOccupancy = 7,
        ReflectionRaw = 8,
        ReflectionFiltered = 9
    };

    struct SurfelPathTracerSettings
    {
        bool enabled = true;
        bool lockSurfels = false;
        bool resetSurfels = false;
        bool enableDiffuseGi = true;
        bool enableReflections = true;
        bool enableReflectionFilter = true;
        bool enableBilateralCleanup = true;
        bool enableTaa = true;
        float resolutionScale = 1.0f;
        uint32_t maxSurfels = 150000;
        uint32_t maxRaysPerFrame = 150000 * 64;
        float cellSize = 2.0f;
        uint32_t cellDimension = 64;
        uint32_t perCellSurfelLimit = 64;
        SurfelPathTracerDebugView debugView = SurfelPathTracerDebugView::FinalColor;
    };

    struct SurfelPathTracerStats
    {
        uint32_t aliveSurfels = 0;
        uint32_t deadSurfels = 0;
        uint32_t dirtySurfels = 0;
        uint32_t requestedRays = 0;
        uint32_t filledCells = 0;
        uint32_t rejectedStores = 0;
        float totalFrameMs = 0.0f;
    };
```

Add public fields:

```cpp
    SurfelPathTracerSettings surfelPathTracerSettings;
    SurfelPathTracerStats surfelPathTracerStats;
```

- [ ] **Step 3: Add the UI radio button and controls**

Modify `UISystem::drawPhysicsUI()` in `src/Core/UISystem.cpp`:

```cpp
    ImGui::SameLine();
    if (ImGui::RadioButton("Surfel PT", renderMode == RenderMode::SurfelPathTracer))
        renderMode = RenderMode::SurfelPathTracer;
```

Add a collapsed section after the path tracer controls:

```cpp
    if (ImGui::CollapsingHeader("Surfel Path Tracer##settings")) {
        auto &settings = surfelPathTracerSettings;
        settings.resolutionScale = std::clamp(settings.resolutionScale, 0.5f, 1.0f);
        settings.maxSurfels = std::clamp(settings.maxSurfels, 1024u, 500000u);
        settings.maxRaysPerFrame = std::clamp(settings.maxRaysPerFrame, 1024u, settings.maxSurfels * 64u);
        settings.cellSize = std::clamp(settings.cellSize, 0.05f, 64.0f);
        settings.cellDimension = std::clamp(settings.cellDimension, 8u, 128u);
        settings.perCellSurfelLimit = std::clamp(settings.perCellSurfelLimit, 4u, 256u);

        ImGui::Checkbox("Lock Surfels", &settings.lockSurfels);
        if (ImGui::Button("Reset Surfels")) {
            settings.resetSurfels = true;
        }
        ImGui::Checkbox("Diffuse GI", &settings.enableDiffuseGi);
        ImGui::Checkbox("Reflections", &settings.enableReflections);
        ImGui::Checkbox("Reflection Filter", &settings.enableReflectionFilter);
        ImGui::Checkbox("Bilateral Cleanup", &settings.enableBilateralCleanup);
        ImGui::Checkbox("TAA", &settings.enableTaa);
        ImGui::SliderFloat("Surfel PT Resolution", &settings.resolutionScale, 0.5f, 1.0f, "%.2f");
        int perCellLimit = static_cast<int>(settings.perCellSurfelLimit);
        ImGui::SliderInt("Per Cell Limit", &perCellLimit, 4, 256);
        settings.perCellSurfelLimit = static_cast<uint32_t>(perCellLimit);
        const char *debugViews[] = {
            "Final Color", "GBuffer Normal", "GBuffer Depth", "Surfel ID", "Surfel Radius",
            "Surfel Radiance", "Surfel Variance", "Cell Occupancy", "Reflection Raw",
            "Reflection Filtered"};
        int debugView = static_cast<int>(settings.debugView);
        ImGui::Combo("Debug View", &debugView, debugViews, IM_ARRAYSIZE(debugViews));
        settings.debugView = static_cast<SurfelPathTracerDebugView>(debugView);

        ImGui::Text("Surfels: %u alive / %u dead",
                    surfelPathTracerStats.aliveSurfels,
                    surfelPathTracerStats.deadSurfels);
        ImGui::Text("Rays: %u | Cells: %u | Rejected Stores: %u",
                    surfelPathTracerStats.requestedRays,
                    surfelPathTracerStats.filledCells,
                    surfelPathTracerStats.rejectedStores);
    }
```

- [ ] **Step 4: Run tests and verify the contract still fails only on missing backend files/shaders**

Run:

```powershell
cmd /c call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests
.\cmake-build-debug\LaphriaEngine\Debug\LaphriaEngineUnitTests.exe
```

Expected: build succeeds; test still fails for missing `SurfelPathTracer*` resources/pipelines/shaders.

- [ ] **Step 5: Commit**

```powershell
git add src/Core/EngineAuxiliary.h src/Core/UISystem.h src/Core/UISystem.cpp
git commit -m "feat: add surfel path tracer ui mode"
```

---

### Task 3: Add Resource Owner And CPU Cell Helpers

**Files:**

- Create: `src/Core/SurfelPathTracerResources.h`
- Create: `src/Core/SurfelPathTracerResources.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add the resource-owner header**

Create `src/Core/SurfelPathTracerResources.h`:

```cpp
#ifndef LAPHRIAENGINE_SURFELPATHTRACERRESOURCES_H
#define LAPHRIAENGINE_SURFELPATHTRACERRESOURCES_H

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "SwapchainManager.h"
#include "UISystem.h"
#include "VulkanDevice.h"
#include "VulkanUtils.h"

namespace Laphria
{
struct SurfelPathTracerSurfel
{
	glm::vec3 position{0.0f};
	float radius = 0.0f;
	glm::vec3 radiance{0.0f};
	uint32_t packedNormal = 0;
	uint32_t rayOffset = 0;
	uint32_t rayCount = 0;
	uint32_t irradianceAtlasOffset = 0;
	uint32_t flags = 0;
	glm::vec4 meanAndVariance{0.0f};
	glm::vec4 shortMeanAndLife{0.0f};
};

struct SurfelPathTracerCellInfo
{
	uint32_t surfelOffset = 0;
	uint32_t surfelCount = 0;
};

struct SurfelPathTracerCounters
{
	uint32_t aliveSurfels = 0;
	uint32_t deadSurfels = 0;
	uint32_t dirtySurfels = 0;
	uint32_t requestedRays = 0;
	uint32_t filledCells = 0;
	uint32_t rejectedStores = 0;
	uint32_t frameIndex = 0;
	uint32_t pad0 = 0;
};

struct SurfelPathTracerCellAddress
{
	glm::ivec3 coord{0};
	uint32_t flatIndex = 0;
};

class SurfelPathTracerResources
{
  public:
	void init(const VulkanDevice &dev, const SwapchainManager &swapchain,
	          const UISystem::SurfelPathTracerSettings &settings);
	void cleanupSwapchainResources();
	void recreateSwapchainResources(const VulkanDevice &dev, const SwapchainManager &swapchain);
	void resetPersistentResources(const VulkanDevice &dev,
	                              const UISystem::SurfelPathTracerSettings &settings);
	void destroy();

	[[nodiscard]] bool initialized() const { return initialized_; }
	[[nodiscard]] uint32_t cellCount() const { return cellCount_; }
	[[nodiscard]] UISystem::SurfelPathTracerStats readStats() const;

	static SurfelPathTracerCellAddress cellAddressForPosition(const glm::vec3 &position,
	                                                          float cellSize,
	                                                          uint32_t cellDimension);

	VulkanUtils::VmaBuffer countersBuffer;
	VulkanUtils::VmaBuffer surfelBuffer;
	VulkanUtils::VmaBuffer aliveBuffer;
	VulkanUtils::VmaBuffer deadBuffer;
	VulkanUtils::VmaBuffer dirtyBuffer;
	VulkanUtils::VmaBuffer recycleBuffer;
	VulkanUtils::VmaBuffer rayBuffer;
	VulkanUtils::VmaBuffer cellInfoBuffer;
	VulkanUtils::VmaBuffer cellCounterBuffer;
	VulkanUtils::VmaBuffer cellToSurfelBuffer;
	void *mappedCounters = nullptr;

	std::vector<VulkanUtils::VmaImage> outputImages;
	std::vector<vk::raii::ImageView> outputImageViews;
	std::vector<VulkanUtils::VmaImage> gBufferNormalImages;
	std::vector<vk::raii::ImageView> gBufferNormalViews;
	std::vector<VulkanUtils::VmaImage> gBufferDepthImages;
	std::vector<vk::raii::ImageView> gBufferDepthViews;
	std::vector<VulkanUtils::VmaImage> reflectionImages;
	std::vector<vk::raii::ImageView> reflectionViews;
	std::vector<VulkanUtils::VmaImage> filteredReflectionImages;
	std::vector<vk::raii::ImageView> filteredReflectionViews;
	std::vector<VulkanUtils::VmaImage> irradianceAtlasImages;
	std::vector<vk::raii::ImageView> irradianceAtlasViews;
	std::vector<VulkanUtils::VmaImage> surfelDepthAtlasImages;
	std::vector<vk::raii::ImageView> surfelDepthAtlasViews;

  private:
	void createPersistentBuffers(const VulkanDevice &dev,
	                             const UISystem::SurfelPathTracerSettings &settings);
	void createExtentImages(const VulkanDevice &dev, const SwapchainManager &swapchain);
	void destroyPersistentBuffers();

	bool initialized_ = false;
	uint32_t cellCount_ = 0;
	UISystem::SurfelPathTracerSettings settings_{};
};
} // namespace Laphria

#endif
```

- [ ] **Step 2: Add the resource implementation**

Create `src/Core/SurfelPathTracerResources.cpp`:

```cpp
#include "SurfelPathTracerResources.h"

#include <algorithm>
#include <cstring>

using namespace Laphria;

namespace
{
void destroyBuffers(std::initializer_list<VulkanUtils::VmaBuffer *> buffers)
{
	for (VulkanUtils::VmaBuffer *buffer : buffers)
	{
		if (buffer)
		{
			buffer->reset();
		}
	}
}

void destroyImages(std::vector<VulkanUtils::VmaImage> &images)
{
	for (auto &image : images)
	{
		image.reset();
	}
	images.clear();
}

void createStorageImageSet(const VulkanDevice &dev,
                           uint32_t width,
                           uint32_t height,
                           vk::Format format,
                           std::vector<VulkanUtils::VmaImage> &images,
                           std::vector<vk::raii::ImageView> &views)
{
	images.clear();
	views.clear();
	images.reserve(MAX_FRAMES_IN_FLIGHT);
	views.reserve(MAX_FRAMES_IN_FLIGHT);
	for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
	{
		VulkanUtils::VmaImage image{};
		VulkanUtils::createImage(dev.logicalDevice, dev.physicalDevice, width, height, format,
		                         vk::ImageTiling::eOptimal,
		                         vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,
		                         vk::MemoryPropertyFlagBits::eDeviceLocal, image);
		images.push_back(std::move(image));
		views.push_back(VulkanUtils::createImageView(dev.logicalDevice, *images.back(), format,
		                                             vk::ImageAspectFlagBits::eColor));
	}
}
} // namespace

void SurfelPathTracerResources::init(const VulkanDevice &dev,
                                     const SwapchainManager &swapchain,
                                     const UISystem::SurfelPathTracerSettings &settings)
{
	settings_ = settings;
	createPersistentBuffers(dev, settings);
	createExtentImages(dev, swapchain);
	initialized_ = true;
}

void SurfelPathTracerResources::cleanupSwapchainResources()
{
	destroyImages(outputImages);
	destroyImages(gBufferNormalImages);
	destroyImages(gBufferDepthImages);
	destroyImages(reflectionImages);
	destroyImages(filteredReflectionImages);
	destroyImages(irradianceAtlasImages);
	destroyImages(surfelDepthAtlasImages);
	outputImageViews.clear();
	gBufferNormalViews.clear();
	gBufferDepthViews.clear();
	reflectionViews.clear();
	filteredReflectionViews.clear();
	irradianceAtlasViews.clear();
	surfelDepthAtlasViews.clear();
}

void SurfelPathTracerResources::recreateSwapchainResources(const VulkanDevice &dev,
                                                           const SwapchainManager &swapchain)
{
	cleanupSwapchainResources();
	createExtentImages(dev, swapchain);
}

void SurfelPathTracerResources::resetPersistentResources(
    const VulkanDevice &dev,
    const UISystem::SurfelPathTracerSettings &settings)
{
	destroyPersistentBuffers();
	settings_ = settings;
	createPersistentBuffers(dev, settings);
}

void SurfelPathTracerResources::destroy()
{
	cleanupSwapchainResources();
	destroyPersistentBuffers();
	initialized_ = false;
}

UISystem::SurfelPathTracerStats SurfelPathTracerResources::readStats() const
{
	UISystem::SurfelPathTracerStats stats{};
	if (!mappedCounters)
	{
		return stats;
	}
	const auto *counters = static_cast<const SurfelPathTracerCounters *>(mappedCounters);
	stats.aliveSurfels = counters->aliveSurfels;
	stats.deadSurfels = counters->deadSurfels;
	stats.dirtySurfels = counters->dirtySurfels;
	stats.requestedRays = counters->requestedRays;
	stats.filledCells = counters->filledCells;
	stats.rejectedStores = counters->rejectedStores;
	return stats;
}

SurfelPathTracerCellAddress SurfelPathTracerResources::cellAddressForPosition(
    const glm::vec3 &position, float cellSize, uint32_t cellDimension)
{
	const float safeCellSize = std::max(cellSize, 0.0001f);
	const int dim = static_cast<int>(std::max(cellDimension, 1u));
	const glm::ivec3 coord = glm::clamp(glm::ivec3(glm::floor(position / safeCellSize)) + dim / 2,
	                                    glm::ivec3(0), glm::ivec3(dim - 1));
	const uint32_t flat = static_cast<uint32_t>(coord.x + coord.y * dim + coord.z * dim * dim);
	return {coord, flat};
}

void SurfelPathTracerResources::createPersistentBuffers(
    const VulkanDevice &dev,
    const UISystem::SurfelPathTracerSettings &settings)
{
	const uint32_t maxSurfels = std::max(settings.maxSurfels, 1u);
	const uint32_t maxRays = std::max(settings.maxRaysPerFrame, 1u);
	cellCount_ = settings.cellDimension * settings.cellDimension * settings.cellDimension;
	const uint32_t cellToSurfelCount = cellCount_ * std::max(settings.perCellSurfelLimit, 1u);

	auto createBuffer = [&](vk::DeviceSize size, VulkanUtils::VmaBuffer &buffer,
	                        vk::MemoryPropertyFlags memoryFlags) {
		VulkanUtils::createBuffer(dev.logicalDevice, dev.physicalDevice, size,
		                          vk::BufferUsageFlagBits::eStorageBuffer,
		                          memoryFlags, buffer);
	};

	createBuffer(sizeof(SurfelPathTracerCounters), countersBuffer,
	             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	mappedCounters = countersBuffer.memory.mapMemory(0, sizeof(SurfelPathTracerCounters));
	std::memset(mappedCounters, 0, sizeof(SurfelPathTracerCounters));

	createBuffer(sizeof(SurfelPathTracerSurfel) * maxSurfels, surfelBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
	createBuffer(sizeof(uint32_t) * maxSurfels, aliveBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
	createBuffer(sizeof(uint32_t) * maxSurfels, deadBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
	createBuffer(sizeof(uint32_t) * maxSurfels, dirtyBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
	createBuffer(sizeof(uint32_t) * maxSurfels * 4u, recycleBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
	createBuffer(sizeof(uint32_t) * maxRays * 8u, rayBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
	createBuffer(sizeof(SurfelPathTracerCellInfo) * cellCount_, cellInfoBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
	createBuffer(sizeof(uint32_t) * 4u, cellCounterBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
	createBuffer(sizeof(uint32_t) * cellToSurfelCount, cellToSurfelBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
}

void SurfelPathTracerResources::createExtentImages(const VulkanDevice &dev,
                                                   const SwapchainManager &swapchain)
{
	const uint32_t width = std::max(swapchain.extent.width, 1u);
	const uint32_t height = std::max(swapchain.extent.height, 1u);
	createStorageImageSet(dev, width, height, vk::Format::eR16G16B16A16Sfloat, outputImages, outputImageViews);
	createStorageImageSet(dev, width, height, vk::Format::eR16G16B16A16Sfloat, gBufferNormalImages, gBufferNormalViews);
	createStorageImageSet(dev, width, height, vk::Format::eR32Sfloat, gBufferDepthImages, gBufferDepthViews);
	createStorageImageSet(dev, std::max(width / 2u, 1u), std::max(height / 2u, 1u),
	                      vk::Format::eR16G16B16A16Sfloat, reflectionImages, reflectionViews);
	createStorageImageSet(dev, std::max(width / 2u, 1u), std::max(height / 2u, 1u),
	                      vk::Format::eR16G16B16A16Sfloat, filteredReflectionImages, filteredReflectionViews);
	createStorageImageSet(dev, 3840u, 2160u, vk::Format::eR16G16B16A16Sfloat, irradianceAtlasImages, irradianceAtlasViews);
	createStorageImageSet(dev, 3840u, 2160u, vk::Format::eR32Sfloat, surfelDepthAtlasImages, surfelDepthAtlasViews);
}

void SurfelPathTracerResources::destroyPersistentBuffers()
{
	mappedCounters = nullptr;
	destroyBuffers({&countersBuffer, &surfelBuffer, &aliveBuffer, &deadBuffer, &dirtyBuffer,
	                &recycleBuffer, &rayBuffer, &cellInfoBuffer, &cellCounterBuffer,
	                &cellToSurfelBuffer});
	cellCount_ = 0;
}
```

- [ ] **Step 3: Add sources to CMake**

Add to `LAPHRIA_ENGINE_SOURCES`:

```cmake
        src/Core/SurfelPathTracerResources.cpp
        src/Core/SurfelPathTracerResources.h
```

Add to `LaphriaEngineUnitTests` because this task adds CPU helper assertions that call `SurfelPathTracerResources::cellAddressForPosition()`:

```cmake
        src/Core/SurfelPathTracerResources.cpp
```

- [ ] **Step 4: Add CPU helper assertions to the structural test**

In `tests/SurfelPathTracerPipelineTests.cpp`, include the header and add:

```cpp
#include "../src/Core/SurfelPathTracerResources.h"
```

Inside `testSurfelPathTracerPipelineContracts()`:

```cpp
	const auto center = Laphria::SurfelPathTracerResources::cellAddressForPosition(
	    glm::vec3(0.0f), 2.0f, 64u);
	if (center.flatIndex >= 64u * 64u * 64u)
	{
		std::cerr << "surfel path tracer center cell out of range\n";
		return false;
	}
	const auto clamped = Laphria::SurfelPathTracerResources::cellAddressForPosition(
	    glm::vec3(1000000.0f), 2.0f, 64u);
	if (clamped.coord.x != 63 || clamped.coord.y != 63 || clamped.coord.z != 63)
	{
		std::cerr << "surfel path tracer cell address must clamp positive overflow\n";
		return false;
	}
```

- [ ] **Step 5: Run tests**

Run:

```powershell
cmd /c call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests
.\cmake-build-debug\LaphriaEngine\Debug\LaphriaEngineUnitTests.exe
```

Expected: build succeeds; contract test still fails for missing pipeline/pass/shader files.

- [ ] **Step 6: Commit**

```powershell
git add CMakeLists.txt src/Core/SurfelPathTracerResources.h src/Core/SurfelPathTracerResources.cpp tests/SurfelPathTracerPipelineTests.cpp
git commit -m "feat: add surfel path tracer resource owner"
```

---

### Task 4: Add Pipeline Owner And Shader Skeletons

**Files:**

- Create: `src/Core/SurfelPathTracerPipelines.h`
- Create: `src/Core/SurfelPathTracerPipelines.cpp`
- Modify: `src/Core/PipelineCollection.h`
- Modify: `src/Core/PipelineCollection.cpp`
- Modify: `CMakeLists.txt`
- Create: all `src/shaders/SurfelPathTracer*.slang` files listed in File Structure.

- [ ] **Step 1: Add shared shader definitions**

Create `src/shaders/SurfelPathTracerCommon.slang`:

```hlsl
#ifndef SURFEL_PATH_TRACER_COMMON_SLANG
#define SURFEL_PATH_TRACER_COMMON_SLANG

#include "ShaderCommon.slang"

static const uint SURFEL_PT_MAX_LIFE = 1200u;
static const uint SURFEL_PT_INVALID_INDEX = 0xFFFFFFFFu;
static const float SURFEL_PT_RAY_BIAS = 0.002;

struct SurfelPathTracerSurfel {
    float3 position;
    float radius;
    float3 radiance;
    uint packedNormal;
    uint rayOffset;
    uint rayCount;
    uint irradianceAtlasOffset;
    uint flags;
    float4 meanAndVariance;
    float4 shortMeanAndLife;
};

struct SurfelPathTracerCellInfo {
    uint surfelOffset;
    uint surfelCount;
};

struct SurfelPathTracerCounters {
    uint aliveSurfels;
    uint deadSurfels;
    uint dirtySurfels;
    uint requestedRays;
    uint filledCells;
    uint rejectedStores;
    uint frameIndex;
    uint pad0;
};

uint packNormalOctahedral(float3 n) {
    float3 safeN = normalize(n);
    float2 p = safeN.xy / (abs(safeN.x) + abs(safeN.y) + abs(safeN.z) + 1e-6);
    if (safeN.z < 0.0) {
        p = (1.0 - abs(p.yx)) * (p.xy >= 0.0 ? 1.0 : -1.0);
    }
    uint2 packed = uint2(clamp(p * 0.5 + 0.5, 0.0, 1.0) * 65535.0 + 0.5);
    return packed.x | (packed.y << 16);
}

float3 unpackNormalOctahedral(uint packedNormal) {
    float2 f = float2(packedNormal & 0xFFFFu, packedNormal >> 16) / 65535.0 * 2.0 - 1.0;
    float3 n = float3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = clamp(-n.z, 0.0, 1.0);
    n.xy += n.xy >= 0.0 ? -t : t;
    return normalize(n);
}

bool isFinite3(float3 v) {
    return all(abs(v) < float3(3.402823e30, 3.402823e30, 3.402823e30));
}

#endif
```

- [ ] **Step 2: Add skeleton shader entry points**

Create `src/shaders/SurfelPathTracerSky.slang`:

```hlsl
#include "SurfelPathTracerCommon.slang"

[[vk::binding(0, 0)]] RWTexture2D<float4> outputImage;
[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint2 size;
    outputImage.GetDimensions(size.x, size.y);
    if (tid.x >= size.x || tid.y >= size.y) {
        return;
    }
    float2 uv = (float2(tid.xy) + 0.5) / float2(size);
    float2 d = uv * 2.0 - 1.0;
    float4 target = mul(ubo.projInverse, float4(d.x, -d.y, 1.0, 1.0));
    float3 rayDir = normalize(mul(ubo.viewInverse, float4(normalize(target.xyz / target.w), 0.0)).xyz);
    outputImage[tid.xy] = float4(evalSkyColor(rayDir, normalize(-ubo.lightDir.xyz)), 1.0);
}
```

Create this exact compile-valid skeleton in each compute shader that is not listed with custom code in this task: `SurfelPathTracerGBuffer.slang`, `SurfelPathTracerPrepare.slang`, `SurfelPathTracerUpdate.slang`, `SurfelPathTracerCellInfo.slang`, `SurfelPathTracerCellToSurfel.slang`, `SurfelPathTracerIntegrate.slang`, `SurfelPathTracerEvaluate.slang`, `SurfelPathTracerReflection.slang`, `SurfelPathTracerReflectionFilter.slang`, `SurfelPathTracerBilateral.slang`, `SurfelPathTracerLightIntegrate.slang`, and `SurfelPathTracerTaa.slang`.

```hlsl
#include "SurfelPathTracerCommon.slang"

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
}
```

For `SurfelPathTracerRaygen.slang`, `SurfelPathTracerMiss.slang`, `SurfelPathTracerClosestHit.slang`, and `SurfelPathTracerAnyHit.slang`, use RT-stage skeletons:

```hlsl
#include "SurfelPathTracerCommon.slang"

struct SurfelPathTracerPayload {
    float3 radiance;
    float hitT;
};

[shader("raygeneration")]
void main()
{
}
```

```hlsl
#include "SurfelPathTracerCommon.slang"

struct SurfelPathTracerPayload {
    float3 radiance;
    float hitT;
};

[shader("miss")]
void main(inout SurfelPathTracerPayload payload)
{
    payload.radiance = float3(0.0, 0.0, 0.0);
    payload.hitT = -1.0;
}
```

```hlsl
#include "SurfelPathTracerCommon.slang"

struct SurfelPathTracerPayload {
    float3 radiance;
    float hitT;
};

[shader("closesthit")]
void main(inout SurfelPathTracerPayload payload, BuiltInTriangleIntersectionAttributes attribs)
{
    payload.hitT = RayTCurrent();
}
```

```hlsl
#include "SurfelPathTracerCommon.slang"

[shader("anyhit")]
void main()
{
}
```

- [ ] **Step 3: Add shader entries and dependencies in CMake**

Append to `SHADER_SOURCES`:

```cmake
        "SurfelPathTracerSky.slang|main"
        "SurfelPathTracerGBuffer.slang|main"
        "SurfelPathTracerPrepare.slang|main"
        "SurfelPathTracerUpdate.slang|main"
        "SurfelPathTracerCellInfo.slang|main"
        "SurfelPathTracerCellToSurfel.slang|main"
        "SurfelPathTracerRaygen.slang|main"
        "SurfelPathTracerMiss.slang|main"
        "SurfelPathTracerClosestHit.slang|main"
        "SurfelPathTracerAnyHit.slang|main"
        "SurfelPathTracerIntegrate.slang|main"
        "SurfelPathTracerEvaluate.slang|main"
        "SurfelPathTracerReflection.slang|main"
        "SurfelPathTracerReflectionFilter.slang|main"
        "SurfelPathTracerBilateral.slang|main"
        "SurfelPathTracerLightIntegrate.slang|main"
        "SurfelPathTracerTaa.slang|main"
```

Append to `SHADER_INCLUDE_DEPS`:

```cmake
        "${SHADER_SOURCE_ROOT}/SurfelPathTracerCommon.slang"
```

- [ ] **Step 4: Add the pipeline owner header**

Create `src/Core/SurfelPathTracerPipelines.h`:

```cpp
#ifndef LAPHRIAENGINE_SURFELPATHTRACERPIPELINES_H
#define LAPHRIAENGINE_SURFELPATHTRACERPIPELINES_H

#include "VulkanDevice.h"
#include "VulkanUtils.h"

namespace Laphria
{
class SurfelPathTracerPipelines
{
  public:
	void createDescriptorSetLayouts(const VulkanDevice &dev);
	void createPipelineLayouts(const VulkanDevice &dev, vk::DescriptorSetLayout globalLayout);
	void createComputePipelines(const VulkanDevice &dev);
	void createRayTracingPipelines(const VulkanDevice &dev);
	void createShaderBindingTables(const VulkanDevice &dev);

	vk::raii::DescriptorSetLayout imageDescriptorSetLayout{nullptr};
	vk::raii::PipelineLayout computePipelineLayout{nullptr};
	vk::raii::Pipeline skyPipeline{nullptr};
	vk::raii::Pipeline gBufferPipeline{nullptr};
	vk::raii::Pipeline preparePipeline{nullptr};
	vk::raii::Pipeline updatePipeline{nullptr};
	vk::raii::Pipeline cellInfoPipeline{nullptr};
	vk::raii::Pipeline cellToSurfelPipeline{nullptr};
	vk::raii::Pipeline integratePipeline{nullptr};
	vk::raii::Pipeline evaluatePipeline{nullptr};
	vk::raii::Pipeline reflectionPipeline{nullptr};
	vk::raii::Pipeline reflectionFilterPipeline{nullptr};
	vk::raii::Pipeline bilateralPipeline{nullptr};
	vk::raii::Pipeline lightIntegratePipeline{nullptr};
	vk::raii::Pipeline taaPipeline{nullptr};

	vk::raii::PipelineLayout rayTracingPipelineLayout{nullptr};
	vk::raii::Pipeline surfelRayTracingPipeline{nullptr};
	vk::raii::Pipeline glossyReflectionRayTracingPipeline{nullptr};
	VulkanUtils::VmaBuffer raygenSBTBuffer{};
	VulkanUtils::VmaBuffer missSBTBuffer{};
	VulkanUtils::VmaBuffer hitSBTBuffer{};
	vk::StridedDeviceAddressRegionKHR raygenRegion{};
	vk::StridedDeviceAddressRegionKHR missRegion{};
	vk::StridedDeviceAddressRegionKHR hitRegion{};
};
} // namespace Laphria

#endif
```

- [ ] **Step 5: Implement compute pipeline creation**

Create `src/Core/SurfelPathTracerPipelines.cpp` with a `createComputePipeline()` helper patterned after `PipelineCollection::createDenoiserPipelines()`. The first pass can instantiate only `skyPipeline`; the other members must be created in this task so shader compilation catches bad entry points.

```cpp
#include "SurfelPathTracerPipelines.h"

#include "PipelineCollection.h"

using namespace Laphria;

void SurfelPathTracerPipelines::createDescriptorSetLayouts(const VulkanDevice &dev)
{
	std::array<vk::DescriptorSetLayoutBinding, 8> bindings = {
	    vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
	    vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
	    vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
	    vk::DescriptorSetLayoutBinding{3, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
	    vk::DescriptorSetLayoutBinding{4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
	    vk::DescriptorSetLayoutBinding{5, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
	    vk::DescriptorSetLayoutBinding{6, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
	    vk::DescriptorSetLayoutBinding{7, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}};
	vk::DescriptorSetLayoutCreateInfo info{.bindingCount = static_cast<uint32_t>(bindings.size()),
	                                       .pBindings = bindings.data()};
	imageDescriptorSetLayout = vk::raii::DescriptorSetLayout(dev.logicalDevice, info);
}
```

- [ ] **Step 6: Wire the helper into `PipelineCollection`**

Modify `src/Core/PipelineCollection.h`:

```cpp
#include "SurfelPathTracerPipelines.h"
```

Add a public member:

```cpp
	Laphria::SurfelPathTracerPipelines surfelPathTracerPipelines;
```

Modify `PipelineCollection::createDescriptorSetLayouts()`:

```cpp
	surfelPathTracerPipelines.createDescriptorSetLayouts(dev);
```

Modify `EngineCore::initVulkan()` after existing pipeline creation:

```cpp
	pipelines.surfelPathTracerPipelines.createPipelineLayouts(vulkan, *pipelines.descriptorSetLayoutGlobal);
	pipelines.surfelPathTracerPipelines.createComputePipelines(vulkan);
	pipelines.surfelPathTracerPipelines.createRayTracingPipelines(vulkan);
	pipelines.surfelPathTracerPipelines.createShaderBindingTables(vulkan);
```

- [ ] **Step 7: Run shader build and unit tests**

Run:

```powershell
cmd /c call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests
.\cmake-build-debug\LaphriaEngine\Debug\LaphriaEngineUnitTests.exe
```

Expected: shader compilation succeeds; contract test still fails until pass owner exists.

- [ ] **Step 8: Commit**

```powershell
git add CMakeLists.txt src/Core/PipelineCollection.h src/Core/PipelineCollection.cpp src/Core/EngineCore.cpp src/Core/SurfelPathTracerPipelines.h src/Core/SurfelPathTracerPipelines.cpp src/shaders/SurfelPathTracer*.slang
git commit -m "feat: add surfel path tracer pipeline skeleton"
```

---

### Task 5: Add Pass Recorder And Sky-Only Backend

**Files:**

- Create: `src/Core/SurfelPathTracerPasses.h`
- Create: `src/Core/SurfelPathTracerPasses.cpp`
- Modify: `src/Core/EngineCore.h`
- Modify: `src/Core/EngineCore.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add the pass recorder header**

Create `src/Core/SurfelPathTracerPasses.h`:

```cpp
#ifndef LAPHRIAENGINE_SURFELPATHTRACERPASSES_H
#define LAPHRIAENGINE_SURFELPATHTRACERPASSES_H

#include "FrameContext.h"
#include "SurfelPathTracerPipelines.h"
#include "SurfelPathTracerResources.h"

namespace Laphria
{
class SurfelPathTracerPasses
{
  public:
	void recordSkyPass(const vk::raii::CommandBuffer &commandBuffer,
	                   const SurfelPathTracerPipelines &pipelines,
	                   const SurfelPathTracerResources &resources,
	                   vk::DescriptorSet imageSet,
	                   vk::DescriptorSet globalSet,
	                   uint32_t frameIndex,
	                   vk::Extent2D extent) const;
	void recordFinalBlit(const vk::raii::CommandBuffer &commandBuffer,
	                     const SurfelPathTracerResources &resources,
	                     vk::Image swapchainImage,
	                     uint32_t frameIndex,
	                     vk::Extent2D extent) const;
};
} // namespace Laphria

#endif
```

- [ ] **Step 2: Implement sky pass and blit**

Create `src/Core/SurfelPathTracerPasses.cpp`:

```cpp
#include "SurfelPathTracerPasses.h"

using namespace Laphria;

void SurfelPathTracerPasses::recordSkyPass(const vk::raii::CommandBuffer &commandBuffer,
                                           const SurfelPathTracerPipelines &pipelines,
                                           const SurfelPathTracerResources &resources,
                                           vk::DescriptorSet imageSet,
                                           vk::DescriptorSet globalSet,
                                           uint32_t frameIndex,
                                           vk::Extent2D extent) const
{
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelines.skyPipeline);
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelines.computePipelineLayout,
	                                 0, {imageSet, globalSet}, nullptr);
	commandBuffer.dispatch((extent.width + 15u) / 16u, (extent.height + 15u) / 16u, 1u);
}

void SurfelPathTracerPasses::recordFinalBlit(const vk::raii::CommandBuffer &commandBuffer,
                                             const SurfelPathTracerResources &resources,
                                             vk::Image swapchainImage,
                                             uint32_t frameIndex,
                                             vk::Extent2D extent) const
{
	vk::ImageBlit blit{
	    .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
	    .srcOffsets = {{vk::Offset3D{0, 0, 0},
	                    vk::Offset3D{static_cast<int32_t>(extent.width),
	                                 static_cast<int32_t>(extent.height), 1}}},
	    .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
	    .dstOffsets = {{vk::Offset3D{0, 0, 0},
	                    vk::Offset3D{static_cast<int32_t>(extent.width),
	                                 static_cast<int32_t>(extent.height), 1}}}};
	commandBuffer.blitImage(*resources.outputImages[frameIndex], vk::ImageLayout::eTransferSrcOptimal,
	                        swapchainImage, vk::ImageLayout::eTransferDstOptimal,
	                        blit, vk::Filter::eLinear);
}
```

- [ ] **Step 3: Add backend members to `EngineCore`**

Modify `src/Core/EngineCore.h`:

```cpp
#include "SurfelPathTracerPasses.h"
#include "SurfelPathTracerResources.h"
```

Add members:

```cpp
	Laphria::SurfelPathTracerResources surfelPathTracerResources;
	Laphria::SurfelPathTracerPasses surfelPathTracerPasses;
	vk::raii::DescriptorPool surfelPathTracerDescriptorPool{nullptr};
	std::vector<vk::raii::DescriptorSet> surfelPathTracerDescriptorSets;
```

Add private methods:

```cpp
	void createSurfelPathTracerDescriptorSets();
	void recordSurfelPathTracerCommandBuffer(const vk::raii::CommandBuffer &commandBuffer,
	                                         uint32_t imageIndex) const;
```

- [ ] **Step 4: Initialize resources and route render mode**

In `EngineCore::initVulkan()` after `frames.init(...)` and pipeline creation, call:

```cpp
	surfelPathTracerResources.init(vulkan, swapchain, ui.surfelPathTracerSettings);
	createSurfelPathTracerDescriptorSets();
```

In `EngineCore::recordCommandBuffer()`:

```cpp
	else if (ui.renderMode == RenderMode::SurfelPathTracer)
	{
		recordSurfelPathTracerCommandBuffer(commandBuffer, imageIndex);
	}
```

In render-mode switch handling, reset surfel history when entering or leaving this mode:

```cpp
	if (ui.renderMode == RenderMode::SurfelPathTracer ||
	    lastSubmittedRenderMode == RenderMode::SurfelPathTracer)
	{
		ui.surfelPathTracerSettings.resetSurfels = true;
	}
```

- [ ] **Step 5: Implement descriptors and command recording**

Add `EngineCore::createSurfelPathTracerDescriptorSets()`:

```cpp
void EngineCore::createSurfelPathTracerDescriptorSets()
{
	surfelPathTracerDescriptorSets.clear();
	if (*surfelPathTracerDescriptorPool)
	{
		surfelPathTracerDescriptorPool = nullptr;
	}
	std::array<vk::DescriptorPoolSize, 2> poolSizes = {
	    vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 8u * MAX_FRAMES_IN_FLIGHT},
	    vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 8u * MAX_FRAMES_IN_FLIGHT}};
	vk::DescriptorPoolCreateInfo poolInfo{.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
	                                      .maxSets = MAX_FRAMES_IN_FLIGHT,
	                                      .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
	                                      .pPoolSizes = poolSizes.data()};
	surfelPathTracerDescriptorPool = vk::raii::DescriptorPool(vulkan.logicalDevice, poolInfo);
	std::vector<vk::DescriptorSetLayout> layouts(
	    MAX_FRAMES_IN_FLIGHT,
	    *pipelines.surfelPathTracerPipelines.imageDescriptorSetLayout);
	vk::DescriptorSetAllocateInfo alloc{.descriptorPool = *surfelPathTracerDescriptorPool,
	                                    .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
	                                    .pSetLayouts = layouts.data()};
	surfelPathTracerDescriptorSets = vulkan.logicalDevice.allocateDescriptorSets(alloc);

	for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
	{
		vk::DescriptorImageInfo output{nullptr, *surfelPathTracerResources.outputImageViews[i],
		                               vk::ImageLayout::eGeneral};
		vk::WriteDescriptorSet write{.dstSet = *surfelPathTracerDescriptorSets[i],
		                             .dstBinding = 0,
		                             .descriptorCount = 1,
		                             .descriptorType = vk::DescriptorType::eStorageImage,
		                             .pImageInfo = &output};
		vulkan.logicalDevice.updateDescriptorSets(write, {});
	}
}
```

Add `EngineCore::recordSurfelPathTracerCommandBuffer()`:

```cpp
void EngineCore::recordSurfelPathTracerCommandBuffer(const vk::raii::CommandBuffer &commandBuffer,
                                                     uint32_t imageIndex) const
{
	const uint32_t fi = frames.frameIndex;
	transition_image_layout(*surfelPathTracerResources.outputImages[fi],
	                        vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
	                        {}, vk::AccessFlagBits2::eShaderWrite,
	                        vk::PipelineStageFlagBits2::eTopOfPipe,
	                        vk::PipelineStageFlagBits2::eComputeShader,
	                        vk::ImageAspectFlagBits::eColor);
	surfelPathTracerPasses.recordSkyPass(commandBuffer, pipelines.surfelPathTracerPipelines,
	                                     surfelPathTracerResources,
	                                     *surfelPathTracerDescriptorSets[fi],
	                                     *descriptorSets[fi], fi, swapchain.extent);
	transition_image_layout(*surfelPathTracerResources.outputImages[fi],
	                        vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal,
	                        vk::AccessFlagBits2::eShaderWrite, vk::AccessFlagBits2::eTransferRead,
	                        vk::PipelineStageFlagBits2::eComputeShader,
	                        vk::PipelineStageFlagBits2::eTransfer,
	                        vk::ImageAspectFlagBits::eColor);
	transition_image_layout(swapchain.images[imageIndex],
	                        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
	                        {}, vk::AccessFlagBits2::eTransferWrite,
	                        vk::PipelineStageFlagBits2::eTopOfPipe,
	                        vk::PipelineStageFlagBits2::eTransfer,
	                        vk::ImageAspectFlagBits::eColor);
	surfelPathTracerPasses.recordFinalBlit(commandBuffer, surfelPathTracerResources,
	                                       swapchain.images[imageIndex], fi, swapchain.extent);
	transition_image_layout(swapchain.images[imageIndex],
	                        vk::ImageLayout::eTransferDstOptimal,
	                        vk::ImageLayout::eColorAttachmentOptimal,
	                        vk::AccessFlagBits2::eTransferWrite,
	                        vk::AccessFlagBits2::eColorAttachmentWrite |
	                            vk::AccessFlagBits2::eColorAttachmentRead,
	                        vk::PipelineStageFlagBits2::eTransfer,
	                        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
	                        vk::ImageAspectFlagBits::eColor);
}
```

- [ ] **Step 6: Run tests**

Run:

```powershell
cmd /c call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests
.\cmake-build-debug\LaphriaEngine\Debug\LaphriaEngineUnitTests.exe
```

Expected: `LaphriaEngineUnitTests` passes.

- [ ] **Step 7: Build editor**

Run:

```powershell
cmd /c call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEditor
```

Expected: `LaphriaEditor` builds, and selecting `Surfel PT` shows sky output.

- [ ] **Step 8: Commit**

```powershell
git add CMakeLists.txt src/Core/EngineCore.h src/Core/EngineCore.cpp src/Core/SurfelPathTracerPasses.h src/Core/SurfelPathTracerPasses.cpp
git commit -m "feat: render surfel path tracer sky backend"
```

---

### Task 6: Implement GBuffer/VBuffer Stage

**Files:**

- Modify: `src/Core/SurfelPathTracerResources.h`
- Modify: `src/Core/SurfelPathTracerResources.cpp`
- Modify: `src/Core/SurfelPathTracerPasses.h`
- Modify: `src/Core/SurfelPathTracerPasses.cpp`
- Modify: `src/Core/EngineCore.cpp`
- Modify: `src/shaders/SurfelPathTracerGBuffer.slang`

- [ ] **Step 1: Extend descriptors for GBuffer images and TLAS/scene data**

Modify `SurfelPathTracerPipelines::createDescriptorSetLayouts()` so binding `0` is output color, `1` normal, `2` depth, `3` motion/material/debug image, and bindings `4..7` are buffers. Add a separate RT descriptor set layout for TLAS plus bindless scene arrays matching the existing RT descriptor set pattern.

- [ ] **Step 2: Write GBuffer shader output contract**

Replace `SurfelPathTracerGBuffer.slang` with a raygen shader that writes:

```hlsl
[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(1, 0)]] RWTexture2D<float4> outputImage;
[[vk::binding(2, 0)]] RWTexture2D<float4> gBufferNormal;
[[vk::binding(3, 0)]] RWTexture2D<float> gBufferDepth;
[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;
```

The first implementation should trace one primary ray, store sky on miss, store normal/depth on hit, and output debug view based on a push constant.

- [ ] **Step 3: Add pass recorder call**

Add `recordGBufferPass()` before `recordSkyPass()` and route final output to sky, normal, or depth based on `ui.surfelPathTracerSettings.debugView`.

- [ ] **Step 4: Update structural tests**

Add required test needles:

```cpp
	    "recordGBufferPass",
	    "gBufferNormal",
	    "gBufferDepth",
	    "RaytracingAccelerationStructure tlas"
```

- [ ] **Step 5: Run tests and editor build**

Run:

```powershell
cmd /c call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests
.\cmake-build-debug\LaphriaEngine\Debug\LaphriaEngineUnitTests.exe
cmd /c call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEditor
```

Expected: unit tests pass; editor can show final/normal/depth debug views.

- [ ] **Step 6: Commit**

```powershell
git add src/Core/SurfelPathTracerResources.* src/Core/SurfelPathTracerPasses.* src/Core/EngineCore.cpp src/shaders/SurfelPathTracerGBuffer.slang tests/SurfelPathTracerPipelineTests.cpp
git commit -m "feat: add surfel path tracer gbuffer pass"
```

---

### Task 7: Implement Persistent Surfel Preparation, Update, And Cell Grid

**Files:**

- Modify: `src/shaders/SurfelPathTracerPrepare.slang`
- Modify: `src/shaders/SurfelPathTracerUpdate.slang`
- Modify: `src/shaders/SurfelPathTracerCellInfo.slang`
- Modify: `src/shaders/SurfelPathTracerCellToSurfel.slang`
- Modify: `src/Core/SurfelPathTracerPasses.*`
- Modify: `src/Core/SurfelPathTracerResources.*`
- Modify: `tests/SurfelPathTracerPipelineTests.cpp`

- [ ] **Step 1: Implement prepare pass**

`SurfelPathTracerPrepare.slang` should clear transient counters while preserving persistent surfel data:

```hlsl
#include "SurfelPathTracerCommon.slang"

[[vk::binding(4, 0)]] RWStructuredBuffer<SurfelPathTracerCounters> counters;

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    counters[0].dirtySurfels = 0;
    counters[0].requestedRays = 0;
    counters[0].filledCells = 0;
    counters[0].rejectedStores = 0;
    counters[0].frameIndex += 1;
}
```

- [ ] **Step 2: Implement cell occupancy update**

`SurfelPathTracerUpdate.slang` should iterate surfels, skip inactive records, age active records, compute a cell from position, increment cell count, and allocate ray offsets using `InterlockedAdd` on `requestedRays`.

- [ ] **Step 3: Implement cell info accumulation**

`SurfelPathTracerCellInfo.slang` should convert per-cell counts into offsets using an atomic running total in `cellCounterBuffer`. It must clamp each cell to `perCellSurfelLimit`.

- [ ] **Step 4: Implement cell-to-surfel population**

`SurfelPathTracerCellToSurfel.slang` should write active surfel indices into each cell's segment, incrementing a per-cell write cursor and rejecting stores that exceed the cell segment.

- [ ] **Step 5: Add pass recorder methods**

Add and call:

```cpp
	void recordPreparePass(const vk::raii::CommandBuffer &commandBuffer,
	                       const SurfelPathTracerPipelines &pipelines,
	                       vk::DescriptorSet imageSet,
	                       vk::Extent2D extent) const;
	void recordUpdatePass(const vk::raii::CommandBuffer &commandBuffer,
	                      const SurfelPathTracerPipelines &pipelines,
	                      vk::DescriptorSet imageSet,
	                      uint32_t maxSurfels) const;
	void recordCellInfoPass(const vk::raii::CommandBuffer &commandBuffer,
	                        const SurfelPathTracerPipelines &pipelines,
	                        vk::DescriptorSet imageSet,
	                        uint32_t cellCount) const;
	void recordCellToSurfelPass(const vk::raii::CommandBuffer &commandBuffer,
	                            const SurfelPathTracerPipelines &pipelines,
	                            vk::DescriptorSet imageSet,
	                            uint32_t maxSurfels) const;
```

The pass order is:

```cpp
recordPreparePass();
recordUpdatePass();
recordCellInfoPass();
recordCellToSurfelPass();
recordGBufferPass();
```

- [ ] **Step 6: Add tests for required symbols**

Add required contract needles:

```cpp
	    "recordPreparePass",
	    "recordUpdatePass",
	    "recordCellInfoPass",
	    "recordCellToSurfelPass",
	    "InterlockedAdd",
	    "perCellSurfelLimit",
	    "rejectedStores"
```

- [ ] **Step 7: Run tests and editor build**

Run:

```powershell
cmd /c call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests
.\cmake-build-debug\LaphriaEngine\Debug\LaphriaEngineUnitTests.exe
cmd /c call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEditor
```

Expected: tests pass; UI counters remain stable in sky/GBuffer mode.

- [ ] **Step 8: Commit**

```powershell
git add src/Core/SurfelPathTracerPasses.* src/Core/SurfelPathTracerResources.* src/shaders/SurfelPathTracerPrepare.slang src/shaders/SurfelPathTracerUpdate.slang src/shaders/SurfelPathTracerCellInfo.slang src/shaders/SurfelPathTracerCellToSurfel.slang tests/SurfelPathTracerPipelineTests.cpp
git commit -m "feat: add surfel path tracer cell grid"
```

---

### Task 8: Implement Surfel Generation, Ray Tracing, And Radiance Integration

**Files:**

- Modify: `src/shaders/SurfelPathTracerEvaluate.slang`
- Modify: `src/shaders/SurfelPathTracerRaygen.slang`
- Modify: `src/shaders/SurfelPathTracerMiss.slang`
- Modify: `src/shaders/SurfelPathTracerClosestHit.slang`
- Modify: `src/shaders/SurfelPathTracerAnyHit.slang`
- Modify: `src/shaders/SurfelPathTracerIntegrate.slang`
- Modify: `src/Core/SurfelPathTracerPasses.*`
- Modify: `tests/SurfelPathTracerPipelineTests.cpp`

- [ ] **Step 1: Generate surfels from under-covered GBuffer pixels**

`SurfelPathTracerEvaluate.slang` should:

- read GBuffer normal/depth,
- reconstruct world position,
- query neighboring cells,
- estimate coverage from surfel radius and normal alignment,
- allocate a dead/free surfel when coverage is below threshold,
- initialize position, packed normal, radius, radiance, and life.

- [ ] **Step 2: Trace surfel rays**

`SurfelPathTracerRaygen.slang` should dispatch over `requestedRays`, load the owning surfel, sample a cosine hemisphere or guided direction, trace against TLAS, write radiance and PDF into `rayBuffer`, and clamp luminance before store.

- [ ] **Step 3: Integrate ray results with MSME**

`SurfelPathTracerIntegrate.slang` should:

- load rays for each surfel,
- update long mean, short mean, and variance,
- write surfel radiance,
- update irradiance/depth atlas texels for directional reuse.

Use this helper in `SurfelPathTracerCommon.slang`:

```hlsl
float3 msmeBlend(float3 previousMean, float3 sampleRadiance, float alpha)
{
    return lerp(previousMean, sampleRadiance, clamp(alpha, 0.0, 1.0));
}
```

- [ ] **Step 4: Record ray trace and integrate pass**

Add pass methods:

```cpp
	void recordSurfelRayTracePass(const vk::raii::CommandBuffer &commandBuffer,
	                              const SurfelPathTracerPipelines &pipelines,
	                              const SurfelPathTracerResources &resources,
	                              vk::DescriptorSet rayTracingSet,
	                              uint32_t rayCount) const;
	void recordIntegratePass(const vk::raii::CommandBuffer &commandBuffer,
	                         const SurfelPathTracerPipelines &pipelines,
	                         vk::DescriptorSet imageSet,
	                         uint32_t maxSurfels) const;
	void recordEvaluatePass(const vk::raii::CommandBuffer &commandBuffer,
	                        const SurfelPathTracerPipelines &pipelines,
	                        vk::DescriptorSet imageSet,
	                        vk::Extent2D extent) const;
```

Order:

```cpp
recordEvaluatePass();
recordSurfelRayTracePass();
recordIntegratePass();
recordEvaluatePass();
```

The first evaluate call can generate surfels; the second can output diffuse GI/debug.

- [ ] **Step 5: Add tests for required symbols**

Add needles:

```cpp
	    "recordSurfelRayTracePass",
	    "recordIntegratePass",
	    "recordEvaluatePass",
	    "msmeBlend",
	    "packNormalOctahedral",
	    "unpackNormalOctahedral",
	    "SURFEL_PT_RAY_BIAS",
	    "clampLuminance"
```

- [ ] **Step 6: Run tests and editor build**

Run:

```powershell
cmd /c call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests
.\cmake-build-debug\LaphriaEngine\Debug\LaphriaEngineUnitTests.exe
cmd /c call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEditor
```

Expected: tests pass; surfel debug views show surfel IDs/radius/radiance after camera frames accumulate.

- [ ] **Step 7: Commit**

```powershell
git add src/Core/SurfelPathTracerPasses.* src/shaders/SurfelPathTracerCommon.slang src/shaders/SurfelPathTracerEvaluate.slang src/shaders/SurfelPathTracerRaygen.slang src/shaders/SurfelPathTracerMiss.slang src/shaders/SurfelPathTracerClosestHit.slang src/shaders/SurfelPathTracerAnyHit.slang src/shaders/SurfelPathTracerIntegrate.slang tests/SurfelPathTracerPipelineTests.cpp
git commit -m "feat: integrate surfel path tracer radiance"
```

---

### Task 9: Add Glossy Reflections, Filters, Light Integrate, And TAA

**Files:**

- Modify: `src/shaders/SurfelPathTracerReflection.slang`
- Modify: `src/shaders/SurfelPathTracerReflectionFilter.slang`
- Modify: `src/shaders/SurfelPathTracerBilateral.slang`
- Modify: `src/shaders/SurfelPathTracerLightIntegrate.slang`
- Modify: `src/shaders/SurfelPathTracerTaa.slang`
- Modify: `src/Core/SurfelPathTracerPasses.*`
- Modify: `src/Core/UISystem.*`
- Modify: `tests/SurfelPathTracerPipelineTests.cpp`

- [ ] **Step 1: Implement reduced-resolution reflection trace**

`SurfelPathTracerReflection.slang` should trace half-resolution glossy rays, evaluate GGX direction selection, use surfel radiance for indirect termination when a ray reaches max depth without direct light, and write raw reflection plus variance.

- [ ] **Step 2: Implement temporal/spatial reflection filter**

`SurfelPathTracerReflectionFilter.slang` should combine current reflection with previous filtered reflection using depth/normal/material consistency and neighborhood reconstruction.

- [ ] **Step 3: Implement bilateral cleanup**

`SurfelPathTracerBilateral.slang` should preserve edges using spatial distance, depth similarity, normal similarity, and reflection variance.

- [ ] **Step 4: Implement light integrate**

`SurfelPathTracerLightIntegrate.slang` should combine:

- direct sun/sky lighting,
- diffuse surfel indirect lighting,
- filtered reflection,
- debug view overrides.

- [ ] **Step 5: Implement final TAA/tone map**

`SurfelPathTracerTaa.slang` should reproject final lighting using previous VP and GBuffer depth, clamp history with a 3x3 neighborhood, blend by camera motion/reset state, apply ACES tone mapping, and write `outputImage`.

- [ ] **Step 6: Wire pass order**

Final pass order:

```cpp
recordPreparePass();
recordGBufferPass();
recordUpdatePass();
recordCellInfoPass();
recordCellToSurfelPass();
recordSurfelRayTracePass();
recordIntegratePass();
recordEvaluatePass();
recordReflectionPass();
recordReflectionFilterPass();
recordBilateralPass();
recordLightIntegratePass();
recordTaaPass();
recordFinalBlit();
```

- [ ] **Step 7: Add tests for required symbols**

Add needles:

```cpp
	    "recordReflectionPass",
	    "recordReflectionFilterPass",
	    "recordBilateralPass",
	    "recordLightIntegratePass",
	    "recordTaaPass",
	    "ggxSampleDirection",
	    "applyAcesTonemap",
	    "SurfelPathTracerDebugView::ReflectionFiltered"
```

- [ ] **Step 8: Run tests and editor build**

Run:

```powershell
cmd /c call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests
.\cmake-build-debug\LaphriaEngine\Debug\LaphriaEngineUnitTests.exe
cmd /c call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEditor
```

Expected: tests pass; editor shows diffuse GI, reflection raw/filter debug, and final tone-mapped output.

- [ ] **Step 9: Commit**

```powershell
git add src/Core/SurfelPathTracerPasses.* src/Core/UISystem.* src/shaders/SurfelPathTracerReflection.slang src/shaders/SurfelPathTracerReflectionFilter.slang src/shaders/SurfelPathTracerBilateral.slang src/shaders/SurfelPathTracerLightIntegrate.slang src/shaders/SurfelPathTracerTaa.slang tests/SurfelPathTracerPipelineTests.cpp
git commit -m "feat: complete surfel path tracer lighting stack"
```

---

### Task 10: Add Runtime Validation And Polish

**Files:**

- Modify: `src/Core/SurfelPathTracerResources.*`
- Modify: `src/Core/SurfelPathTracerPasses.*`
- Modify: `src/Core/EngineCore.cpp`
- Modify: `src/Core/UISystem.*`
- Modify: `README.md`

- [ ] **Step 1: Add allocation failure labels**

Wrap every resource creation in `SurfelPathTracerResources` with a clear failure context:

```cpp
try
{
	createPersistentBuffers(dev, settings);
}
catch (const std::exception &error)
{
	LOGE("SurfelPathTracer persistent resource creation failed: %s", error.what());
	throw;
}
```

- [ ] **Step 2: Add no-scene fallback**

In `recordSurfelPathTracerCommandBuffer()`, before GBuffer/ray tracing:

```cpp
	if (!resourceManager || resourceManager->getModelCount() == 0)
	{
		surfelPathTracerPasses.recordSkyPass(commandBuffer, pipelines.surfelPathTracerPipelines,
		                                     surfelPathTracerResources,
		                                     *surfelPathTracerDescriptorSets[fi],
		                                     *descriptorSets[fi], fi, swapchain.extent);
		return;
	}
```

- [ ] **Step 3: Reset on static setting changes**

Track previous `SurfelPathTracerSettings` capacity/cell fields in `EngineCore`; when they change, call `vulkan.logicalDevice.waitIdle()`, `resetPersistentResources()`, and `createSurfelPathTracerDescriptorSets()`.

- [ ] **Step 4: Update README**

Add a short Rendering bullet:

```markdown
- Surfel path tracer backend with persistent surfel GI cache, glossy reflections,
  temporal/spatial filtering, bilateral cleanup, and TAA debug controls.
```

- [ ] **Step 5: Run full verification**

Run:

```powershell
cmd /c call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests
.\cmake-build-debug\LaphriaEngine\Debug\LaphriaEngineUnitTests.exe
cmd /c call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEditor
```

Expected: tests pass, editor builds, `Surfel PT` can be selected, reset/lock controls work, resize does not crash, and Sponza displays a stable final output.

- [ ] **Step 6: Commit**

```powershell
git add README.md src/Core/EngineCore.cpp src/Core/UISystem.* src/Core/SurfelPathTracerResources.* src/Core/SurfelPathTracerPasses.*
git commit -m "fix: harden surfel path tracer runtime behavior"
```

---

## Self-Review Checklist

- Every file in the approved design has an implementation task.
- The existing `PathTracer` remains intact and separately selectable.
- New symbols use `SurfelPathTracer`, not the old forbidden standalone `SurfelGi` names.
- First executable slice is sky-only, so risky Vulkan/RT work is introduced incrementally.
- Each task has a verification command and expected result.
- Structural tests guard render mode, shader entries, resource owner, pipeline owner, pass owner, and core shader helper contracts.
