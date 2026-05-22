#if defined(_WIN32) && !defined(NOMINMAX)
#	define NOMINMAX
#endif

#include "EngineCore.h"
#include "VmaContext.h"
#include "VulkanUtils.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#	include <Windows.h>
#endif

#include "../SceneManagement/Scene.h"
#include "EngineAuxiliary.h"
#include "EngineConfig.h"
#include "PathTracerAnalysis.h"
#include "ResourceManager.h"

using namespace Laphria;

namespace
{
constexpr uint32_t    kPtTimestampQueryCountPerFrame         = 8;
constexpr uint32_t    kPtMaterialMaxBounceShift              = 26u;
constexpr uint32_t    kPtMaterialMaxBounceMask               = 0x3u;
constexpr uint32_t    kPtMaterialDirectSunShift              = 28u;
constexpr uint32_t    kPtMaterialDirectSunMask               = 0x3u;
constexpr uint32_t    kPtFlagsEnvironmentNeeBit              = 1u << 0u;
constexpr uint32_t    kPtFlagsBlackEnvironmentBit            = 1u << 1u;
constexpr uint32_t    kPtFlagsApplyFirstHitProbesBit         = 1u << 2u;
constexpr uint32_t    kPtFlagsEnvironmentSamplingShift       = 3u;
constexpr uint32_t    kPtFlagsEnvironmentSamplingMask        = 0x3u;
constexpr uint32_t    kPtFlagsFirstHitDiffuseSamplesShift    = 5u;
constexpr uint32_t    kPtFlagsFirstHitDiffuseSamplesMask     = 0xFu;
constexpr uint32_t    kPtFlagsFirstHitCandidateCountShift    = 9u;
constexpr uint32_t    kPtFlagsFirstHitCandidateCountMask     = 0xFu;
constexpr uint32_t    kPtFlagsFirstHitProbeSamplingShift     = 13u;
constexpr uint32_t    kPtFlagsFirstHitProbeSamplingMask      = 0x7u;
constexpr uint32_t    kPtFlagsEnvironmentBounceShift         = 21u;
constexpr uint32_t    kPtFlagsEnvironmentBounceMask          = 0x3u;
constexpr int         PATH_TRACER_BLACK_ENVIRONMENT_BIT      = static_cast<int>(kPtFlagsBlackEnvironmentBit);
constexpr int         PATH_TRACER_APPLY_FIRST_HIT_PROBES_BIT = static_cast<int>(kPtFlagsApplyFirstHitProbesBit);
constexpr double      kWindowTitleUpdateIntervalSeconds      = 0.5;
constexpr const char *kPtBenchmarkCsvFileName                = "path_tracer_benchmark_results.csv";
constexpr const char *kPtBacklogCsvFileName                  = "path_tracer_backlog.csv";

uint32_t packPathTracerBits(uint32_t value, uint32_t shift, uint32_t mask)
{
	return (value & mask) << shift;
}

uint32_t encodePathTracerMaxBounceCode(int maxBounces)
{
	if (maxBounces <= 3)
	{
		return 0u;
	}
	if (maxBounces <= 5)
	{
		return 1u;
	}
	return 2u;
}

uint32_t packPathTracerMaterialSettings(const UISystem::PathTracerSettings &settings)
{
	return packPathTracerBits(static_cast<uint32_t>(std::clamp(settings.directSunBounceMode, 0, 2)),
	                          kPtMaterialDirectSunShift,
	                          kPtMaterialDirectSunMask) |
	       packPathTracerBits(encodePathTracerMaxBounceCode(settings.pathTracerMaxBounces),
	                          kPtMaterialMaxBounceShift,
	                          kPtMaterialMaxBounceMask);
}

uint32_t packPathTracerFlags(const UISystem::PathTracerSettings &settings)
{
	const uint32_t candidateCountMinusOne =
	    static_cast<uint32_t>(std::clamp(settings.firstHitCandidateCount, 2, 16) - 1);
	return (settings.enableEnvironmentNEE ? kPtFlagsEnvironmentNeeBit : 0u) |
	       (settings.blackEnvironment ? kPtFlagsBlackEnvironmentBit : 0u) |
	       (settings.applyFirstHitProbesToFinal ? kPtFlagsApplyFirstHitProbesBit : 0u) |
	       packPathTracerBits(static_cast<uint32_t>(settings.environmentNeeSamplingMode),
	                          kPtFlagsEnvironmentSamplingShift,
	                          kPtFlagsEnvironmentSamplingMask) |
	       packPathTracerBits(static_cast<uint32_t>(std::clamp(settings.firstHitDiffuseSamples, 1, 8)),
	                          kPtFlagsFirstHitDiffuseSamplesShift,
	                          kPtFlagsFirstHitDiffuseSamplesMask) |
	       packPathTracerBits(candidateCountMinusOne,
	                          kPtFlagsFirstHitCandidateCountShift,
	                          kPtFlagsFirstHitCandidateCountMask) |
	       packPathTracerBits(static_cast<uint32_t>(settings.firstHitProbeSamplingMode),
	                          kPtFlagsFirstHitProbeSamplingShift,
	                          kPtFlagsFirstHitProbeSamplingMask) |
	       packPathTracerBits(static_cast<uint32_t>(std::clamp(settings.environmentNeeBounceMode, 0, 2)),
	                          kPtFlagsEnvironmentBounceShift,
	                          kPtFlagsEnvironmentBounceMask);
}

std::filesystem::path resolveProjectRootPath()
{
	namespace fs = std::filesystem;
	std::error_code ec;
	fs::path        current = fs::current_path(ec);
	if (ec)
	{
		return fs::path(".");
	}

	for (int i = 0; i < 8; ++i)
	{
		if (fs::exists(current / "CMakeLists.txt", ec) && !ec)
		{
			return current;
		}
		if (!current.has_parent_path())
		{
			break;
		}
		current = current.parent_path();
	}
	return fs::current_path();
}

std::filesystem::path resolveAnalysisOutputPath(const char *fileName)
{
	namespace fs = std::filesystem;
	std::error_code ec;
	const fs::path  outputDir = resolveProjectRootPath() / "build";
	fs::create_directories(outputDir, ec);
	return outputDir / fileName;
}

void debugBreakIfDebuggerAttached()
{
#if defined(_WIN32)
	if (::IsDebuggerPresent() != 0)
	{
		::DebugBreak();
	}
#endif
}

vk::Result waitForFencesOrThrow(const vk::raii::Device         &device,
                                vk::ArrayProxy<const vk::Fence> fences,
                                const char                     *waitLabel)
{
	try
	{
		return device.waitForFences(fences, vk::True, UINT64_MAX);
	}
	catch (const vk::SystemError &error)
	{
		if (error.code().value() == static_cast<int>(vk::Result::eErrorDeviceLost))
		{
			LOGE("Vulkan device lost while waiting for %s: %s", waitLabel, error.what());
		}
		else
		{
			LOGE("Vulkan fence wait failed while waiting for %s: %s", waitLabel, error.what());
		}
		debugBreakIfDebuggerAttached();
		throw;
	}
}

vk::Result waitForFenceOrThrow(const vk::raii::Device &device, vk::Fence fence, const char *waitLabel)
{
	return waitForFencesOrThrow(device, std::array<vk::Fence, 1>{fence}, waitLabel);
}

enum PtTimestampSlot : uint32_t
{
	kPtTS_TlasStart         = 0,
	kPtTS_TlasEnd           = 1,
	kPtTS_RayTraceStart     = 2,
	kPtTS_RayTraceEnd       = 3,
	kPtTS_ReprojectionStart = 4,
	kPtTS_ReprojectionEnd   = 5,
	kPtTS_DenoiserStart     = 6,
	kPtTS_DenoiserEnd       = 7
};

struct SponzaScenarioPreset
{
	const char *name;
	glm::vec3   cameraPosition;
	float       cameraPitch;
	float       cameraYaw;
	glm::vec3   lightDirection;
};

const std::array<SponzaScenarioPreset, 3> &sponzaScenarioPresets()
{
	static const std::array<SponzaScenarioPreset, 3> presets = {
	    SponzaScenarioPreset{
	        "Dark Courtyard",
	        glm::vec3(0.0f, 1.2f, 0.0f),
	        glm::radians(-8.0f),
	        glm::radians(180.0f),
	        glm::normalize(glm::vec3(-0.45f, -1.0f, 0.65f))},
	    SponzaScenarioPreset{
	        "Sunlit Courtyard Wall",
	        glm::vec3(0.0f, 12.0f, -1.5f),
	        glm::radians(-4.0f),
	        glm::radians(180.0f),
	        glm::normalize(glm::vec3(-0.45f, -1.0f, 0.65f))},
	    SponzaScenarioPreset{
	        "Mid-Depth Interior",
	        glm::vec3(-1.0f, 6.5f, 3.0f),
	        glm::radians(-8.0f),
	        glm::radians(180.0f),
	        glm::normalize(glm::vec3(-0.35f, -1.0f, 0.45f))}};
	return presets;
}

SurfelPathTracerStaticSettingsSnapshot makeSurfelPathTracerStaticSettingsSnapshot(
    const UISystem::SurfelPathTracerSettings &settings)
{
	return SurfelPathTracerStaticSettingsSnapshot{
	    .cellSize = std::max(settings.cellSize, 0.0001f),
	    .maxSurfels = std::max(settings.maxSurfels, 1u),
	    .maxRaysPerFrame = std::max(settings.maxRaysPerFrame, 1u),
	    .cellDimension = std::clamp(settings.cellDimension, 8u, 128u),
	    .perCellSurfelLimit = std::clamp(settings.perCellSurfelLimit, 1u, 256u),
	    .irradianceAtlasWidth = std::clamp(settings.irradianceAtlasWidth, 512u, 4096u),
	    .irradianceAtlasHeight = std::clamp(settings.irradianceAtlasHeight, 512u, 4096u)};
}

bool surfelPathTracerStaticSettingsMatch(
    const SurfelPathTracerStaticSettingsSnapshot &lhs,
    const SurfelPathTracerStaticSettingsSnapshot &rhs)
{
	return lhs.cellSize == rhs.cellSize &&
	       lhs.maxSurfels == rhs.maxSurfels &&
	       lhs.maxRaysPerFrame == rhs.maxRaysPerFrame &&
	       lhs.cellDimension == rhs.cellDimension &&
	       lhs.perCellSurfelLimit == rhs.perCellSurfelLimit &&
	       lhs.irradianceAtlasWidth == rhs.irradianceAtlasWidth &&
	       lhs.irradianceAtlasHeight == rhs.irradianceAtlasHeight;
}

bool surfelPathTracerAtlasSettingsChanged(
    const SurfelPathTracerStaticSettingsSnapshot &lhs,
    const SurfelPathTracerStaticSettingsSnapshot &rhs)
{
	return lhs.irradianceAtlasWidth != rhs.irradianceAtlasWidth ||
	       lhs.irradianceAtlasHeight != rhs.irradianceAtlasHeight;
}
}        // namespace

EngineCore::EngineCore(EngineHostOptions optionsIn, EngineHostCallbacks callbacksIn) : options(std::move(optionsIn)), callbacks(std::move(callbacksIn))
{
}

void EngineCore::run()
{
	try
	{
		VulkanUtils::resetAllocationCounter();
		initWindow();
		initInput();
		initVulkan();
		initImgui();
		invokeInitializeCallback();
		// Game initialization may load models/maps after Vulkan init. Rebuild RT descriptor
		// sets here so first-time RT/PT switching never uses stale pre-init bindings.
		if (resourceManager)
		{
			createRayTracingDescriptorSets();
			createSurfelPathTracerRtDescriptorSets();
		}
		mainLoop();
		const auto vmaStats = Laphria::VmaContext::getStats();
		LOGI("VMA stats: blocks=%u allocations=%u allocationBytes=%llu",
		     vmaStats.blockCount,
		     vmaStats.allocationCount,
		     static_cast<unsigned long long>(vmaStats.allocationBytes));
		LOGI("Tracked Vulkan allocations: %llu", static_cast<unsigned long long>(VulkanUtils::getAllocationCounter()));
		invokeShutdownCallback();
		cleanup();
	}
	catch (...)
	{
		try
		{
			if (vulkanInitialized)
			{
				vulkan.logicalDevice.waitIdle();
			}
		}
		catch (...)
		{
			// Best-effort cleanup path.
		}

		try
		{
			cleanup();
		}
		catch (...)
		{
			// Suppress cleanup exceptions while propagating the original failure.
		}

		throw;
	}
}

EngineServices EngineCore::buildServices()
{
	return EngineServices{
	    .window          = window,
	    .camera          = camera,
	    .scene           = *scene,
	    .physics         = *physicsSystem,
	    .resourceManager = *resourceManager,
	    .ui              = ui,
	    .loadSceneAsset =
	        [this](const std::string &path) {
		        scene->loadScene(path, *resourceManager, *pipelines.descriptorSetLayoutMaterial);
	        },
	    .saveSceneAsset =
	        [this](const std::string &path) {
		        scene->saveScene(path, *resourceManager);
	        },
	    .loadModelAsset =
	        [this](const std::string &path, const SceneNode::Ptr &parent) {
		        scene->loadModel(path, *resourceManager, *pipelines.descriptorSetLayoutMaterial, parent);
	        },
	    .createCubePrimitive =
	        [this](float size) {
		        return resourceManager->createCubeModel(size, *pipelines.descriptorSetLayoutMaterial);
	        },
	    .createCubePrimitiveWithMaterial =
	        [this](float size, const Laphria::MaterialData &material) {
		        return resourceManager->createCubeModel(size, *pipelines.descriptorSetLayoutMaterial, material);
	        },
	    .createSpherePrimitive =
	        [this](float radius, int slices, int stacks) {
		        return resourceManager->createSphereModel(radius, slices, stacks, *pipelines.descriptorSetLayoutMaterial);
	        },
	    .createCylinderPrimitive =
	        [this](float radius, float height, int slices) {
		        return resourceManager->createCylinderModel(radius, height, slices, *pipelines.descriptorSetLayoutMaterial);
	        }};
}

void EngineCore::invokeInitializeCallback()
{
	ui.showEditorPanels = options.showEditorPanels;
	if (callbacks.initialize && scene && physicsSystem && resourceManager)
	{
		auto services = buildServices();
		callbacks.initialize(services);
	}
}

void EngineCore::invokeShutdownCallback()
{
	if (callbacks.shutdown && scene && physicsSystem && resourceManager)
	{
		auto services = buildServices();
		callbacks.shutdown(services);
	}
}

void EngineCore::initWindow()
{
	if (glfwInit() != GLFW_TRUE)
	{
		throw std::runtime_error("failed to initialize GLFW");
	}
	windowInitialized = true;

	// GLFW_NO_API: we manage the Vulkan surface ourselves, not via an OpenGL context.
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

	window = glfwCreateWindow(WIDTH, HEIGHT, options.windowTitle.c_str(), nullptr, nullptr);
	if (!window)
	{
		throw std::runtime_error("failed to create GLFW window");
	}
	lastFrameTime = std::chrono::high_resolution_clock::now();
}

void EngineCore::initInput()
{
	input.init(window, camera, swapchain.framebufferResized, options.enableDefaultCameraInput);
}

void EngineCore::initVulkan()
{
	// Ordering matters here:
	//  1. vulkan → swapchain: surface must exist before swapchain creation.
	//  2. frames → createDescriptorPool: commandPool must exist before ResourceManager.
	//  3. ResourceManager receives a reference to descriptorPool, so the pool must be alive
	//     for the entire lifetime of the ResourceManager.
	//  4. Descriptor set layouts must precede pipeline creation.
	//  5. Descriptor sets must be written after both pool and uniform buffers/images exist.
	vulkan.init(window);
	vulkanInitialized = true;
	swapchain.init(vulkan, window);
	imagesInFlight.assign(swapchain.images.size(), vk::Fence{});
	frames.init(vulkan, swapchain);
	createDescriptorPool();

	resourceManager        = std::make_unique<ResourceManager>(vulkan.logicalDevice, vulkan.physicalDevice, frames.commandPool, vulkan.queue,
	                                                           descriptorPool);
	scene                  = std::make_unique<Scene>();
	constexpr float bounds = Laphria::EngineConfig::kDefaultSceneBoundsExtent;
	scene->init({{-bounds, -bounds, -bounds}, {bounds, bounds, bounds}});

	physicsSystem = std::make_unique<PhysicsSystem>();

	pipelines.createDescriptorSetLayouts(vulkan);
	pipelines.surfelPathTracerPipelines.createPipelineLayouts(vulkan, *pipelines.descriptorSetLayoutGlobal);
	pipelines.surfelPathTracerPipelines.createComputePipelines(vulkan);
	pipelines.surfelPathTracerPipelines.createGBufferRayTracingPipeline(vulkan);
	pipelines.surfelPathTracerPipelines.createGBufferShaderBindingTable(vulkan);
	pipelines.surfelPathTracerPipelines.createSurfelRayTracingPipeline(vulkan);
	pipelines.surfelPathTracerPipelines.createSurfelShaderBindingTable(vulkan);
	pipelines.surfelPathTracerPipelines.createReflectionRayTracingPipeline(vulkan);
	pipelines.surfelPathTracerPipelines.createReflectionShaderBindingTable(vulkan);

	// Pipeline creation order matches dependency on the descriptor set layouts above.
	pipelines.createGraphicsPipeline(vulkan, swapchain.surfaceFormat.format, vulkan.findDepthFormat());
	pipelines.createShadowPipeline(vulkan);
	pipelines.createComputePipeline(vulkan);
	pipelines.createSkinningPipeline(vulkan);
	pipelines.createPhysicsPipeline(vulkan);
	pipelines.createRayTracingPipeline(vulkan);
	pipelines.createShaderBindingTable(vulkan);
	pipelines.createDenoiserPipelines(vulkan);
	pipelines.createClassicRTPipeline(vulkan);
	pipelines.createClassicRTShaderBindingTable(vulkan);

	resourceManager->setSkinningDescriptorSetLayout(*pipelines.skinningDescriptorSetLayout);

	createDescriptorSets();
	createComputeDescriptorSets();
	createPhysicsDescriptorSets();
	createRayTracingDescriptorSets();
	createDenoiserDescriptorSets();
	surfelPathTracerResources.init(vulkan, swapchain, ui.surfelPathTracerSettings);
	surfelPathTracerStaticSettings =
	    makeSurfelPathTracerStaticSettingsSnapshot(ui.surfelPathTracerSettings);
	surfelPathTracerStaticSettingsInitialized = true;
	surfelPathTracerPersistentImageLayoutsInitialized = false;
	surfelPathTracerHistoryValid.fill(false);
	createSurfelPathTracerSkyDescriptorSets();
	createSurfelPathTracerStorageDescriptorSets();
	createSurfelPathTracerRtDescriptorSets();
	createTimestampQueryPool();
}

void EngineCore::initImgui()
{
	ui.init(vulkan, window, swapchain.surfaceFormat.format, vulkan.findDepthFormat());
	imguiInitialized = true;
}

void EngineCore::mainLoop()
{
	size_t prevModelCount = resourceManager->getModelCount();

	while (!glfwWindowShouldClose(window))
	{
		// Delta Time calculation
		auto  currentTime = std::chrono::high_resolution_clock::now();
		float deltaTime   = std::chrono::duration<float>(currentTime - lastFrameTime).count();
		lastFrameTime     = currentTime;
		updatePerformanceWindowTitle(deltaTime);

		glfwPollEvents();
		loadPathTracerBenchmarkSceneIfNeeded();
		updatePathTracerBenchmark(deltaTime);
		updatePathTracerPhysicalSanityChecks(deltaTime);
		camera.update(deltaTime);

		std::optional<EngineServices> services;
		if (scene && physicsSystem && resourceManager)
		{
			services.emplace(buildServices());
		}

		if (callbacks.updateFrame && services.has_value())
		{
			auto &servicesRef = *services;
			callbacks.updateFrame(servicesRef, deltaTime);
		}
		if (resourceManager)
		{
			resourceManager->setTextureColorSpaceModel(ui.textureColorSpaceModel);
		}
		if (scene && resourceManager)
		{
			scene->update(deltaTime, *resourceManager);
		}

		// Physics Update
		if (options.runPhysicsSimulation && ui.simulationRunning && physicsSystem)
		{
			auto start = std::chrono::high_resolution_clock::now();

			if (ui.useGPUPhysics)
			{
				auto cmd = VulkanUtils::beginSingleTimeCommands(vulkan.logicalDevice, frames.commandPool);
				physicsSystem->updateGPU(scene->getAllNodes(), deltaTime, cmd, pipelines.physicsPipelineLayout, pipelines.physicsPipeline, physicsDescriptorSet);
				cmd.end();

				vk::raii::Fence physicsFence(vulkan.logicalDevice, vk::FenceCreateInfo{});
				vk::SubmitInfo  submitInfo{};
				submitInfo.commandBufferCount = 1;
				submitInfo.pCommandBuffers    = &*cmd;
				vulkan.queue.submit(submitInfo, *physicsFence);

				const vk::Result waitResult = vulkan.logicalDevice.waitForFences(*physicsFence, vk::True, UINT64_MAX);
				if (waitResult != vk::Result::eSuccess)
				{
					throw std::runtime_error("failed to wait for GPU physics fence");
				}

				// Readback immediately
				physicsSystem->syncFromGPU(scene->getAllNodes());
			}
			else
			{
				physicsSystem->updateCPU(scene->getAllNodes(), deltaTime);
			}

			auto end       = std::chrono::high_resolution_clock::now();
			ui.physicsTime = std::chrono::duration<float, std::milli>(end - start).count();
		}

		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ui.draw(window, *scene, *physicsSystem, *resourceManager, *pipelines.descriptorSetLayoutMaterial, camera);
		if (callbacks.drawUi && services.has_value())
		{
			auto &servicesRef = *services;
			callbacks.drawUi(servicesRef);
		}
		if (scene)
		{
			scene->syncSpatialIndex();
		}

		// If models were loaded during the UI frame, the RT descriptor sets (bindings 5-8:
		// vertex/index/material/texture arrays) must be rebuilt to include the new buffers.
		// buildBLAS already called queue.waitIdle(), so the queue is idle here.
		size_t currentModelCount = resourceManager->getModelCount();
		if (currentModelCount != prevModelCount)
		{
			prevModelCount = currentModelCount;
			createRayTracingDescriptorSets();
			createSurfelPathTracerRtDescriptorSets();
		}

		ImGui::Render();

		drawFrame();
	}

	vulkan.logicalDevice.waitIdle();
}

void EngineCore::updatePerformanceWindowTitle(float deltaTimeSeconds)
{
	if (!window || deltaTimeSeconds <= 0.0f)
	{
		return;
	}

	titleStatsAccumSeconds += static_cast<double>(deltaTimeSeconds);
	++titleStatsFrameCount;

	if (titleStatsAccumSeconds < kWindowTitleUpdateIntervalSeconds || titleStatsFrameCount == 0)
	{
		return;
	}

	const double averageFrameTimeSeconds = titleStatsAccumSeconds / static_cast<double>(titleStatsFrameCount);
	const double fps                     = 1.0 / averageFrameTimeSeconds;
	const double frameTimeMs             = averageFrameTimeSeconds * 1000.0;

	char titleBuffer[256];
	std::snprintf(titleBuffer, sizeof(titleBuffer), "%s | %.1f FPS | %.2f ms",
	              options.windowTitle.c_str(), fps, frameTimeMs);
	glfwSetWindowTitle(window, titleBuffer);

	titleStatsAccumSeconds = 0.0;
	titleStatsFrameCount   = 0;
}

void EngineCore::cleanupSwapChain()
{
	surfelPathTracerSkyDescriptorSets.clear();
	surfelPathTracerSkyDescriptorPool = nullptr;
	surfelPathTracerStorageDescriptorSets.clear();
	surfelPathTracerStorageDescriptorPool = nullptr;
	surfelPathTracerRtDescriptorSets.clear();
	surfelPathTracerRtDescriptorPool = nullptr;
	surfelPathTracerResources.cleanupSwapchainResources();
	surfelPathTracerPersistentImageLayoutsInitialized = false;
	surfelPathTracerHistoryValid.fill(false);
	swapchain.cleanup();
	frames.cleanupSwapChainDependents();
}

void EngineCore::cleanup()
{
	if (imguiInitialized)
	{
		ui.cleanup();
		imguiInitialized = false;
	}

	if (windowInitialized && window)
	{
		glfwDestroyWindow(window);
		window = nullptr;
	}
	if (windowInitialized)
	{
		glfwTerminate();
		windowInitialized = false;
	}
}

void EngineCore::recreateSwapChain()
{
	// A zero-sized framebuffer means the window is minimized; block here until it is restored.
	int width = 0, height = 0;
	glfwGetFramebufferSize(window, &width, &height);
	while (width == 0 || height == 0)
	{
		glfwGetFramebufferSize(window, &width, &height);
		glfwWaitEvents();
	}

	vulkan.logicalDevice.waitIdle();

	cleanupSwapChain();
	swapchain.init(vulkan, window);
	imagesInFlight.assign(swapchain.images.size(), vk::Fence{});
	frames.recreate(vulkan, swapchain);
	// Compute, RT, and denoiser descriptor sets reference images that are recreated above
	// (storageImages, rayTracingOutputImages, and G-Buffer images are extent-dependent),
	// so all three must be rewritten after frames.recreate().
	createComputeDescriptorSets();
	createRayTracingDescriptorSets();
	createDenoiserDescriptorSets();
	if (surfelPathTracerResources.initialized())
	{
		surfelPathTracerResources.recreateSwapchainResources(vulkan, swapchain);
		surfelPathTracerPersistentImageLayoutsInitialized = false;
		surfelPathTracerHistoryValid.fill(false);
		createSurfelPathTracerSkyDescriptorSets();
		createSurfelPathTracerStorageDescriptorSets();
		createSurfelPathTracerRtDescriptorSets();
	}
	ptForceHistoryReset = true;
}

void EngineCore::createPhysicsDescriptorSets()
{
	vk::DescriptorPoolSize       poolSize{vk::DescriptorType::eStorageBuffer, 1};
	vk::DescriptorPoolCreateInfo poolInfo{
	    .flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
	    .maxSets       = 1,
	    .poolSizeCount = 1,
	    .pPoolSizes    = &poolSize};
	physicsDescriptorPool = vk::raii::DescriptorPool(vulkan.logicalDevice, poolInfo);

	vk::DescriptorSetAllocateInfo allocInfo{
	    .descriptorPool     = *physicsDescriptorPool,
	    .descriptorSetCount = 1,
	    .pSetLayouts        = &*pipelines.physicsDescriptorSetLayout};

	physicsDescriptorSet = std::move(vulkan.logicalDevice.allocateDescriptorSets(allocInfo)[0]);

	// Create SSBO
	constexpr size_t maxObjects = Laphria::EngineConfig::kMaxPhysicsObjects;
	physicsSystem->createSSBO(vulkan.logicalDevice, vulkan.physicalDevice, maxObjects * sizeof(PhysicsObject));

	// Bind SSBO to Set
	vk::DescriptorBufferInfo bufferInfo{
	    .buffer = *physicsSystem->getSSBOBuffer(),
	    .offset = 0,
	    .range  = maxObjects * sizeof(PhysicsObject)};

	vk::WriteDescriptorSet writeDescriptorSet{
	    .dstSet          = *physicsDescriptorSet,
	    .dstBinding      = 0,
	    .dstArrayElement = 0,
	    .descriptorCount = 1,
	    .descriptorType  = vk::DescriptorType::eStorageBuffer,
	    .pBufferInfo     = &bufferInfo};

	vulkan.logicalDevice.updateDescriptorSets(writeDescriptorSet, nullptr);
}

void EngineCore::createComputeDescriptorSets()
{
	// One set per Frame In Flight (matching storage images)
	std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *pipelines.computeDescriptorSetLayout);

	vk::DescriptorSetAllocateInfo allocInfo{
	    .descriptorPool     = *descriptorPool,        // Use the same global pool
	    .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
	    .pSetLayouts        = layouts.data()};

	computeDescriptorSets.clear();
	computeDescriptorSets = vulkan.logicalDevice.allocateDescriptorSets(allocInfo);

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		vk::DescriptorImageInfo imageInfo{
		    .imageView   = *frames.storageImageViews[i],
		    .imageLayout = vk::ImageLayout::eGeneral        // Compute shader writes to General layout
		};

		vk::WriteDescriptorSet storageImageWrite{
		    .dstSet          = *computeDescriptorSets[i],
		    .dstBinding      = 0,
		    .dstArrayElement = 0,
		    .descriptorCount = 1,
		    .descriptorType  = vk::DescriptorType::eStorageImage,
		    .pImageInfo      = &imageInfo};

		vulkan.logicalDevice.updateDescriptorSets(storageImageWrite, {});
	}
}

void EngineCore::createRayTracingDescriptorSets()
{
	// One set per frame in flight; bindings shifted to accommodate the new G-Buffer images.
	// RT set bindings: 0 = TLAS, 1 = noisy colour, 2 = normals, 3 = depth, 4 = motion vectors,
	//                  5 = vertex arrays, 6 = index arrays, 7 = material arrays, 8 = texture array.
	std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *pipelines.rayTracingDescriptorSetLayout);

	std::vector<uint32_t>                                variableDescCounts(MAX_FRAMES_IN_FLIGHT, Laphria::EngineConfig::kBindlessModelCapacity);
	vk::DescriptorSetVariableDescriptorCountAllocateInfo variableDescCountInfo{
	    .descriptorSetCount = MAX_FRAMES_IN_FLIGHT,
	    .pDescriptorCounts  = variableDescCounts.data()};

	vk::DescriptorSetAllocateInfo allocInfo{
	    .pNext              = &variableDescCountInfo,
	    .descriptorPool     = *descriptorPool,
	    .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
	    .pSetLayouts        = layouts.data()};

	rtDescriptorSets.clear();
	rtDescriptorSets = vulkan.logicalDevice.allocateDescriptorSets(allocInfo);

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		// Binding 0 — TLAS.
		// The TLAS write requires a WriteDescriptorSetAccelerationStructureKHR in pNext;
		// it cannot use pBufferInfo or pImageInfo like every other descriptor type.
		vk::WriteDescriptorSetAccelerationStructureKHR tlasInfo{
		    .accelerationStructureCount = 1,
		    .pAccelerationStructures    = &*frames.tlas[i]};

		vk::WriteDescriptorSet tlasWrite{
		    .pNext           = &tlasInfo,
		    .dstSet          = *rtDescriptorSets[i],
		    .dstBinding      = 0,
		    .dstArrayElement = 0,
		    .descriptorCount = 1,
		    .descriptorType  = vk::DescriptorType::eAccelerationStructureKHR};

		// Binding 1 — noisy colour output (written by the raygen shader in General layout).
		vk::DescriptorImageInfo rtOutputImageInfo{
		    .imageView   = *frames.rayTracingOutputImageViews[i],
		    .imageLayout = vk::ImageLayout::eGeneral};
		vk::WriteDescriptorSet rtOutputWrite{
		    .dstSet = *rtDescriptorSets[i], .dstBinding = 1, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &rtOutputImageInfo};

		// Binding 2 — G-Buffer world normals.
		vk::DescriptorImageInfo normalsInfo{.imageView = *frames.rtGBufferNormalsViews[i], .imageLayout = vk::ImageLayout::eGeneral};
		vk::WriteDescriptorSet  normalsWrite{
		     .dstSet = *rtDescriptorSets[i], .dstBinding = 2, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &normalsInfo};

		// Binding 3 — G-Buffer linear depth.
		vk::DescriptorImageInfo depthInfo{.imageView = *frames.rtGBufferDepthViews[i], .imageLayout = vk::ImageLayout::eGeneral};
		vk::WriteDescriptorSet  depthWrite{
		     .dstSet = *rtDescriptorSets[i], .dstBinding = 3, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &depthInfo};

		// Binding 4 — motion vectors.
		vk::DescriptorImageInfo mvInfo{.imageView = *frames.rtMotionVectorsViews[i], .imageLayout = vk::ImageLayout::eGeneral};
		vk::WriteDescriptorSet  mvWrite{
		     .dstSet = *rtDescriptorSets[i], .dstBinding = 4, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &mvInfo};
		vk::DescriptorBufferInfo analysisCounterInfo{
		    .buffer = *frames.ptAnalysisCounterBuffers[i],
		    .offset = 0,
		    .range  = sizeof(Laphria::PathTracerAnalysisCounters)};
		vk::WriteDescriptorSet analysisCounterWrite{
		    .dstSet = *rtDescriptorSets[i], .dstBinding = 9, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &analysisCounterInfo};

		std::vector<vk::WriteDescriptorSet> descriptorWrites;
		descriptorWrites.push_back(tlasWrite);
		descriptorWrites.push_back(rtOutputWrite);
		descriptorWrites.push_back(normalsWrite);
		descriptorWrites.push_back(depthWrite);
		descriptorWrites.push_back(mvWrite);
		descriptorWrites.push_back(analysisCounterWrite);

		// Now we extract ALL global vertices, indices, materials, and textures
		// across all Scene Nodes that have been uploaded into VRAM by ResourceManager
		std::vector<vk::DescriptorBufferInfo> vertexInfos;
		std::vector<vk::DescriptorBufferInfo> indexInfos;
		std::vector<vk::DescriptorBufferInfo> materialInfos;
		std::vector<vk::DescriptorImageInfo>  textureInfos;

		// Since our ResourceManager stores ModelResource objects linearly in ID...
		// In a production engine, this would be an iterative flat map or array
		constexpr int totalModels = static_cast<int>(Laphria::EngineConfig::kBindlessModelCapacity);
		for (int modelId = 0; modelId < totalModels; ++modelId)
		{
			if (ModelResource *model = resourceManager->getModelResource(modelId))
			{
				// Writing a null VkBuffer into a descriptor is invalid even with ePartiallyBound.
				if (!*model->vertexBuffer || !*model->indexBuffer || !*model->materialBuffer)
					throw std::runtime_error("RT descriptor: model " + std::to_string(modelId) + " has null buffer(s)");

				// 1. Accumulate Vertex Buffers (use skinned stream for RT/PT when available)
				const vk::Buffer rtVertexBuffer = (model->hasRuntimeSkinning && *model->skinnedVertexBuffer) ? *model->skinnedVertexBuffer : *model->vertexBuffer;
				vertexInfos.push_back({rtVertexBuffer, 0, VK_WHOLE_SIZE});

				// 2. Accumulate Index Buffers
				indexInfos.push_back({*model->indexBuffer, 0, VK_WHOLE_SIZE});

				// 3. Accumulate Material Buffers
				materialInfos.push_back({*model->materialBuffer, 0, VK_WHOLE_SIZE});

				// 4. Accumulate Textures — pair each view with its own sampler.
				for (size_t texIdx = 0; texIdx < model->textureImageViews.size(); ++texIdx)
				{
					textureInfos.push_back({*model->textureSamplers[texIdx], *model->textureImageViews[texIdx], vk::ImageLayout::eShaderReadOnlyOptimal});
				}
			}
			else
			{
				break;        // Stop at the first empty ID
			}
		}

		if (!vertexInfos.empty())
		{
			descriptorWrites.push_back(vk::WriteDescriptorSet{
			    .dstSet          = *rtDescriptorSets[i],
			    .dstBinding      = 5,
			    .dstArrayElement = 0,
			    .descriptorCount = static_cast<uint32_t>(vertexInfos.size()),
			    .descriptorType  = vk::DescriptorType::eStorageBuffer,
			    .pBufferInfo     = vertexInfos.data()});
		}

		if (!indexInfos.empty())
		{
			descriptorWrites.push_back(vk::WriteDescriptorSet{
			    .dstSet          = *rtDescriptorSets[i],
			    .dstBinding      = 6,
			    .dstArrayElement = 0,
			    .descriptorCount = static_cast<uint32_t>(indexInfos.size()),
			    .descriptorType  = vk::DescriptorType::eStorageBuffer,
			    .pBufferInfo     = indexInfos.data()});
		}

		if (!materialInfos.empty())
		{
			descriptorWrites.push_back(vk::WriteDescriptorSet{
			    .dstSet          = *rtDescriptorSets[i],
			    .dstBinding      = 7,
			    .dstArrayElement = 0,
			    .descriptorCount = static_cast<uint32_t>(materialInfos.size()),
			    .descriptorType  = vk::DescriptorType::eStorageBuffer,
			    .pBufferInfo     = materialInfos.data()});
		}

		if (!textureInfos.empty())
		{
			descriptorWrites.push_back(vk::WriteDescriptorSet{
			    .dstSet          = *rtDescriptorSets[i],
			    .dstBinding      = 8,
			    .dstArrayElement = 0,
			    .descriptorCount = static_cast<uint32_t>(textureInfos.size()),
			    .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
			    .pImageInfo      = textureInfos.data()});
		}

		vulkan.logicalDevice.updateDescriptorSets(descriptorWrites, {});
	}
}

void EngineCore::createSurfelPathTracerSkyDescriptorSets()
{
	surfelPathTracerSkyDescriptorSets.clear();
	if (*surfelPathTracerSkyDescriptorPool)
	{
		surfelPathTracerSkyDescriptorPool = nullptr;
	}

	if (!surfelPathTracerResources.initialized())
	{
		return;
	}
	if (surfelPathTracerResources.outputImageViews.size() < MAX_FRAMES_IN_FLIGHT)
	{
		throw std::runtime_error("Surfel path tracer sky descriptors require one output image view per frame");
	}

	vk::DescriptorPoolSize poolSize{vk::DescriptorType::eStorageImage, MAX_FRAMES_IN_FLIGHT};
	vk::DescriptorPoolCreateInfo poolInfo{
	    .flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
	    .maxSets       = MAX_FRAMES_IN_FLIGHT,
	    .poolSizeCount = 1,
	    .pPoolSizes    = &poolSize};
	surfelPathTracerSkyDescriptorPool = vk::raii::DescriptorPool(vulkan.logicalDevice, poolInfo);

	std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,
	                                             *pipelines.surfelPathTracerPipelines.skyDescriptorSetLayout);
	vk::DescriptorSetAllocateInfo allocInfo{
	    .descriptorPool     = *surfelPathTracerSkyDescriptorPool,
	    .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
	    .pSetLayouts        = layouts.data()};
	surfelPathTracerSkyDescriptorSets = vulkan.logicalDevice.allocateDescriptorSets(allocInfo);

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
	{
		vk::DescriptorImageInfo outputInfo{
		    .imageView   = *surfelPathTracerResources.outputImageViews[i],
		    .imageLayout = vk::ImageLayout::eGeneral};
		vk::WriteDescriptorSet outputWrite{
		    .dstSet          = *surfelPathTracerSkyDescriptorSets[i],
		    .dstBinding      = 0,
		    .dstArrayElement = 0,
		    .descriptorCount = 1,
		    .descriptorType  = vk::DescriptorType::eStorageImage,
		    .pImageInfo      = &outputInfo};
		vulkan.logicalDevice.updateDescriptorSets(outputWrite, {});
	}
}

void EngineCore::createSurfelPathTracerStorageDescriptorSets()
{
	surfelPathTracerStorageDescriptorSets.clear();
	if (*surfelPathTracerStorageDescriptorPool)
	{
		surfelPathTracerStorageDescriptorPool = nullptr;
	}

	std::array<vk::DescriptorPoolSize, 2> poolSizes = {
	    vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 11 * MAX_FRAMES_IN_FLIGHT},
	    vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 10 * MAX_FRAMES_IN_FLIGHT}};
	vk::DescriptorPoolCreateInfo poolInfo{
	    .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
	    .maxSets = MAX_FRAMES_IN_FLIGHT,
	    .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
	    .pPoolSizes = poolSizes.data()};
	surfelPathTracerStorageDescriptorPool = vk::raii::DescriptorPool(vulkan.logicalDevice, poolInfo);

	std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,
	                                             *pipelines.surfelPathTracerPipelines.storageDescriptorSetLayout);
	vk::DescriptorSetAllocateInfo allocInfo{
	    .descriptorPool = *surfelPathTracerStorageDescriptorPool,
	    .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
	    .pSetLayouts = layouts.data()};
	surfelPathTracerStorageDescriptorSets = vulkan.logicalDevice.allocateDescriptorSets(allocInfo);

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
	{
		const std::array<vk::DescriptorImageInfo, 11> imageInfos = {
		    vk::DescriptorImageInfo{.imageView = *surfelPathTracerResources.outputImageViews[i], .imageLayout = vk::ImageLayout::eGeneral},
		    vk::DescriptorImageInfo{.imageView = *surfelPathTracerResources.gBufferNormalViews[i], .imageLayout = vk::ImageLayout::eGeneral},
		    vk::DescriptorImageInfo{.imageView = *surfelPathTracerResources.gBufferDepthViews[i], .imageLayout = vk::ImageLayout::eGeneral},
		    vk::DescriptorImageInfo{.imageView = *surfelPathTracerResources.gBufferMotionMaterialViews[i], .imageLayout = vk::ImageLayout::eGeneral},
		    vk::DescriptorImageInfo{.imageView = *surfelPathTracerResources.irradianceAtlasViews[i], .imageLayout = vk::ImageLayout::eGeneral},
		    vk::DescriptorImageInfo{.imageView = *surfelPathTracerResources.surfelDepthAtlasViews[i], .imageLayout = vk::ImageLayout::eGeneral},
		    vk::DescriptorImageInfo{.imageView = *surfelPathTracerResources.reflectionViews[i], .imageLayout = vk::ImageLayout::eGeneral},
		    vk::DescriptorImageInfo{.imageView = *surfelPathTracerResources.filteredReflectionViews[i], .imageLayout = vk::ImageLayout::eGeneral},
		    vk::DescriptorImageInfo{.imageView = *surfelPathTracerResources.lightingViews[i], .imageLayout = vk::ImageLayout::eGeneral},
		    vk::DescriptorImageInfo{.imageView = *surfelPathTracerResources.taaHistoryViews[i], .imageLayout = vk::ImageLayout::eGeneral},
		    vk::DescriptorImageInfo{.imageView = *surfelPathTracerResources.gBufferAlbedoViews[i], .imageLayout = vk::ImageLayout::eGeneral}};
		const std::array<uint32_t, 11> imageBindings = {0, 1, 2, 3, 14, 15, 16, 17, 18, 19, 20};

		const std::array<vk::DescriptorBufferInfo, 10> bufferInfos = {
		    vk::DescriptorBufferInfo{.buffer = *surfelPathTracerResources.countersBuffer, .offset = 0, .range = VK_WHOLE_SIZE},
		    vk::DescriptorBufferInfo{.buffer = *surfelPathTracerResources.surfelBuffer, .offset = 0, .range = VK_WHOLE_SIZE},
		    vk::DescriptorBufferInfo{.buffer = *surfelPathTracerResources.aliveBuffer, .offset = 0, .range = VK_WHOLE_SIZE},
		    vk::DescriptorBufferInfo{.buffer = *surfelPathTracerResources.deadBuffer, .offset = 0, .range = VK_WHOLE_SIZE},
		    vk::DescriptorBufferInfo{.buffer = *surfelPathTracerResources.dirtyBuffer, .offset = 0, .range = VK_WHOLE_SIZE},
		    vk::DescriptorBufferInfo{.buffer = *surfelPathTracerResources.recycleBuffer, .offset = 0, .range = VK_WHOLE_SIZE},
		    vk::DescriptorBufferInfo{.buffer = *surfelPathTracerResources.rayBuffer, .offset = 0, .range = VK_WHOLE_SIZE},
		    vk::DescriptorBufferInfo{.buffer = *surfelPathTracerResources.cellInfoBuffer, .offset = 0, .range = VK_WHOLE_SIZE},
		    vk::DescriptorBufferInfo{.buffer = *surfelPathTracerResources.cellCounterBuffer, .offset = 0, .range = VK_WHOLE_SIZE},
		    vk::DescriptorBufferInfo{.buffer = *surfelPathTracerResources.cellToSurfelBuffer, .offset = 0, .range = VK_WHOLE_SIZE}};

		std::vector<vk::WriteDescriptorSet> writes;
		writes.reserve(20);
		for (size_t imageIndex = 0; imageIndex < imageInfos.size(); ++imageIndex)
		{
			writes.push_back(vk::WriteDescriptorSet{
			    .dstSet = *surfelPathTracerStorageDescriptorSets[i],
			    .dstBinding = imageBindings[imageIndex],
			    .dstArrayElement = 0,
			    .descriptorCount = 1,
			    .descriptorType = vk::DescriptorType::eStorageImage,
			    .pImageInfo = &imageInfos[imageIndex]});
		}
		for (uint32_t binding = 4; binding <= 13; ++binding)
		{
			writes.push_back(vk::WriteDescriptorSet{
			    .dstSet = *surfelPathTracerStorageDescriptorSets[i],
			    .dstBinding = binding,
			    .dstArrayElement = 0,
			    .descriptorCount = 1,
			    .descriptorType = vk::DescriptorType::eStorageBuffer,
			    .pBufferInfo = &bufferInfos[binding - 4]});
		}

		vulkan.logicalDevice.updateDescriptorSets(writes, {});
	}
}

void EngineCore::createSurfelPathTracerRtDescriptorSets()
{
	surfelPathTracerRtDescriptorSets.clear();
	if (*surfelPathTracerRtDescriptorPool)
	{
		surfelPathTracerRtDescriptorPool = nullptr;
	}

	constexpr uint32_t bindlessCapacity = Laphria::EngineConfig::kBindlessModelCapacity;
	std::array<vk::DescriptorPoolSize, 3> poolSizes = {
	    vk::DescriptorPoolSize{vk::DescriptorType::eAccelerationStructureKHR, MAX_FRAMES_IN_FLIGHT},
	    vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 3 * bindlessCapacity * MAX_FRAMES_IN_FLIGHT},
	    vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, bindlessCapacity * MAX_FRAMES_IN_FLIGHT}};
	vk::DescriptorPoolCreateInfo poolInfo{
	    .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet |
	             vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind,
	    .maxSets = MAX_FRAMES_IN_FLIGHT,
	    .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
	    .pPoolSizes = poolSizes.data()};
	surfelPathTracerRtDescriptorPool = vk::raii::DescriptorPool(vulkan.logicalDevice, poolInfo);

	std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,
	                                             *pipelines.surfelPathTracerPipelines.rayTracingDescriptorSetLayout);
	vk::DescriptorSetAllocateInfo allocInfo{
	    .descriptorPool = *surfelPathTracerRtDescriptorPool,
	    .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
	    .pSetLayouts = layouts.data()};
	surfelPathTracerRtDescriptorSets = vulkan.logicalDevice.allocateDescriptorSets(allocInfo);

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
	{
		vk::WriteDescriptorSetAccelerationStructureKHR tlasInfo{
		    .accelerationStructureCount = 1,
		    .pAccelerationStructures = &*frames.tlas[i]};
		vk::WriteDescriptorSet tlasWrite{
		    .pNext = &tlasInfo,
		    .dstSet = *surfelPathTracerRtDescriptorSets[i],
		    .dstBinding = 0,
		    .dstArrayElement = 0,
		    .descriptorCount = 1,
		    .descriptorType = vk::DescriptorType::eAccelerationStructureKHR};

		std::vector<vk::WriteDescriptorSet> descriptorWrites;
		descriptorWrites.push_back(tlasWrite);

		std::vector<vk::DescriptorBufferInfo> vertexInfos;
		std::vector<vk::DescriptorBufferInfo> indexInfos;
		std::vector<vk::DescriptorBufferInfo> materialInfos;
		std::vector<vk::DescriptorImageInfo> textureInfos;

		for (int modelId = 0; modelId < static_cast<int>(bindlessCapacity); ++modelId)
		{
			ModelResource *model = resourceManager ? resourceManager->getModelResource(modelId) : nullptr;
			if (!model)
			{
				break;
			}
			if (!*model->vertexBuffer || !*model->indexBuffer || !*model->materialBuffer)
			{
				throw std::runtime_error("Surfel GBuffer RT descriptor: model " + std::to_string(modelId) + " has null buffer(s)");
			}

			const vk::Buffer rtVertexBuffer =
			    (model->hasRuntimeSkinning && *model->skinnedVertexBuffer) ? *model->skinnedVertexBuffer : *model->vertexBuffer;
			vertexInfos.push_back({rtVertexBuffer, 0, VK_WHOLE_SIZE});
			indexInfos.push_back({*model->indexBuffer, 0, VK_WHOLE_SIZE});
			materialInfos.push_back({*model->materialBuffer, 0, VK_WHOLE_SIZE});

			for (size_t texIdx = 0; texIdx < model->textureImageViews.size(); ++texIdx)
			{
				textureInfos.push_back({*model->textureSamplers[texIdx],
				                        *model->textureImageViews[texIdx],
				                        vk::ImageLayout::eShaderReadOnlyOptimal});
			}
		}

		if (!vertexInfos.empty())
		{
			descriptorWrites.push_back(vk::WriteDescriptorSet{
			    .dstSet = *surfelPathTracerRtDescriptorSets[i],
			    .dstBinding = 5,
			    .dstArrayElement = 0,
			    .descriptorCount = static_cast<uint32_t>(vertexInfos.size()),
			    .descriptorType = vk::DescriptorType::eStorageBuffer,
			    .pBufferInfo = vertexInfos.data()});
		}
		if (!indexInfos.empty())
		{
			descriptorWrites.push_back(vk::WriteDescriptorSet{
			    .dstSet = *surfelPathTracerRtDescriptorSets[i],
			    .dstBinding = 6,
			    .dstArrayElement = 0,
			    .descriptorCount = static_cast<uint32_t>(indexInfos.size()),
			    .descriptorType = vk::DescriptorType::eStorageBuffer,
			    .pBufferInfo = indexInfos.data()});
		}
		if (!materialInfos.empty())
		{
			descriptorWrites.push_back(vk::WriteDescriptorSet{
			    .dstSet = *surfelPathTracerRtDescriptorSets[i],
			    .dstBinding = 7,
			    .dstArrayElement = 0,
			    .descriptorCount = static_cast<uint32_t>(materialInfos.size()),
			    .descriptorType = vk::DescriptorType::eStorageBuffer,
			    .pBufferInfo = materialInfos.data()});
		}
		if (!textureInfos.empty())
		{
			descriptorWrites.push_back(vk::WriteDescriptorSet{
			    .dstSet = *surfelPathTracerRtDescriptorSets[i],
			    .dstBinding = 8,
			    .dstArrayElement = 0,
			    .descriptorCount = static_cast<uint32_t>(textureInfos.size()),
			    .descriptorType = vk::DescriptorType::eCombinedImageSampler,
			    .pImageInfo = textureInfos.data()});
		}

		vulkan.logicalDevice.updateDescriptorSets(descriptorWrites, {});
	}
}

void EngineCore::createDenoiserDescriptorSets()
{
	// One set per frame in flight. Bindings include storage images + one storage buffer.
	// Free old sets before replacing the pool; each RAII DescriptorSet stores its parent pool handle.
	denoiserDescriptorSets.clear();
	if (*denoiserDescriptorPool)
	{
		denoiserDescriptorPool = nullptr;
	}

	std::vector<vk::DescriptorPoolSize> poolSizes = {
	    {vk::DescriptorType::eStorageImage, 14 * MAX_FRAMES_IN_FLIGHT},
	    {vk::DescriptorType::eStorageBuffer, MAX_FRAMES_IN_FLIGHT}};
	vk::DescriptorPoolCreateInfo poolInfo{
	    .flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
	    .maxSets       = MAX_FRAMES_IN_FLIGHT,
	    .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
	    .pPoolSizes    = poolSizes.data()};
	denoiserDescriptorPool = vk::raii::DescriptorPool(vulkan.logicalDevice, poolInfo);

	std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *pipelines.denoiserDescriptorSetLayout);
	vk::DescriptorSetAllocateInfo        allocInfo{
	           .descriptorPool     = *denoiserDescriptorPool,
	           .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
	           .pSetLayouts        = layouts.data()};
	denoiserDescriptorSets = vulkan.logicalDevice.allocateDescriptorSets(allocInfo);

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		size_t       prevSlot   = (i - 1 + MAX_FRAMES_IN_FLIGHT) % MAX_FRAMES_IN_FLIGHT;
		const size_t atrousBase = i * 2;

		// Build image infos in binding order.
		vk::DescriptorImageInfo infos[14] = {
		    {.imageView = *frames.rayTracingOutputImageViews[i], .imageLayout = vk::ImageLayout::eGeneral},          // 0: noisy colour
		    {.imageView = *frames.rtGBufferNormalsViews[i], .imageLayout = vk::ImageLayout::eGeneral},               // 1: current normals
		    {.imageView = *frames.rtGBufferDepthViews[i], .imageLayout = vk::ImageLayout::eGeneral},                 // 2: current depth
		    {.imageView = *frames.rtMotionVectorsViews[i], .imageLayout = vk::ImageLayout::eGeneral},                // 3: motion vectors
		    {.imageView = *frames.historyColorViews[prevSlot], .imageLayout = vk::ImageLayout::eGeneral},            // 4: history colour read
		    {.imageView = *frames.historyColorViews[i], .imageLayout = vk::ImageLayout::eGeneral},                   // 5: history colour write
		    {.imageView = *frames.historyMomentsViews[prevSlot], .imageLayout = vk::ImageLayout::eGeneral},          // 6: history moments read
		    {.imageView = *frames.historyMomentsViews[i], .imageLayout = vk::ImageLayout::eGeneral},                 // 7: history moments write
		    {.imageView = *frames.atrousTempViews[atrousBase + 0], .imageLayout = vk::ImageLayout::eGeneral},        // 8: A-Trous buffer A
		    {.imageView = *frames.atrousTempViews[atrousBase + 1], .imageLayout = vk::ImageLayout::eGeneral},        // 9: A-Trous buffer B
		    {.imageView = *frames.rayTracingOutputImageViews[i], .imageLayout = vk::ImageLayout::eGeneral},          // 10: final denoised output (reuses slot 0 image)
		    {.imageView = *frames.rtGBufferNormalsViews[prevSlot], .imageLayout = vk::ImageLayout::eGeneral},        // 11: previous-frame normals
		    {.imageView = *frames.rtGBufferDepthViews[prevSlot], .imageLayout = vk::ImageLayout::eGeneral},          // 12: previous-frame depth
		    {.imageView = *frames.ptReprojectionDebugViews[i], .imageLayout = vk::ImageLayout::eGeneral},            // 13: reprojection debug channels
		};

		vk::DescriptorBufferInfo analysisCounterInfo{
		    .buffer = *frames.ptAnalysisCounterBuffers[i],
		    .offset = 0,
		    .range  = sizeof(Laphria::PathTracerAnalysisCounters)};

		std::vector<vk::WriteDescriptorSet> writes;
		writes.reserve(15);
		for (uint32_t b = 0; b < 14; ++b)
		{
			writes.push_back(vk::WriteDescriptorSet{
			    .dstSet          = *denoiserDescriptorSets[i],
			    .dstBinding      = b,
			    .dstArrayElement = 0,
			    .descriptorCount = 1,
			    .descriptorType  = vk::DescriptorType::eStorageImage,
			    .pImageInfo      = &infos[b]});
		}
		writes.push_back(vk::WriteDescriptorSet{
		    .dstSet          = *denoiserDescriptorSets[i],
		    .dstBinding      = 14,
		    .dstArrayElement = 0,
		    .descriptorCount = 1,
		    .descriptorType  = vk::DescriptorType::eStorageBuffer,
		    .pBufferInfo     = &analysisCounterInfo});
		vulkan.logicalDevice.updateDescriptorSets(writes, {});
	}
}

void EngineCore::recordComputeCommandBuffer(const vk::raii::CommandBuffer &commandBuffer, uint32_t imageIndex) const
{
	// 1. Execution Barrier — General Layout for Compute Write
	// eGeneral→eGeneral: no content discard; waits for the previous frame's TRANSFER_SRC→eGeneral
	// restore (or the one-time creation pre-transition) before the compute shader writes.
	transition_image_layout(
	    *frames.storageImages[frames.frameIndex],
	    vk::ImageLayout::eGeneral,
	    vk::ImageLayout::eGeneral,
	    {},
	    vk::AccessFlagBits2::eShaderWrite,
	    vk::PipelineStageFlagBits2::eTransfer,        // Wait for the previous frame's restore
	    vk::PipelineStageFlagBits2::eComputeShader,
	    vk::ImageAspectFlagBits::eColor);

	// 2. Compute Dispatch
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelines.computePipeline);

	// Bind Set 0 (storage image) — the simplified layout only exposes this one set.
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelines.computePipelineLayout, 0,
	                                 *computeDescriptorSets[frames.frameIndex], nullptr);

	Laphria::ScenePushConstants push{};
	push.skyData = glm::vec4(0.01f, 0.03f, 0.1f, 0.99f);

	commandBuffer.pushConstants<Laphria::ScenePushConstants>(*pipelines.computePipelineLayout,
	                                                         vk::ShaderStageFlagBits::eCompute,
	                                                         0, push);

	// Dispatch
	// Workgroup size is 16x16.
	uint32_t groupCountX = (swapchain.extent.width + 15) / 16;
	uint32_t groupCountY = (swapchain.extent.height + 15) / 16;
	commandBuffer.dispatch(groupCountX, groupCountY, 1);

	// 3. Blit Storage Image -> SwapChain Image

	// Transition Storage Image: General -> TransferSrc
	transition_image_layout(
	    *frames.storageImages[frames.frameIndex],
	    vk::ImageLayout::eGeneral,
	    vk::ImageLayout::eTransferSrcOptimal,
	    vk::AccessFlagBits2::eShaderWrite,
	    vk::AccessFlagBits2::eTransferRead,
	    vk::PipelineStageFlagBits2::eComputeShader,
	    vk::PipelineStageFlagBits2::eTransfer,
	    vk::ImageAspectFlagBits::eColor);

	// Transition SwapChain Image: Undefined -> TransferDst
	transition_image_layout(
	    swapchain.images[imageIndex],
	    vk::ImageLayout::eUndefined,
	    vk::ImageLayout::eTransferDstOptimal,
	    {},
	    vk::AccessFlagBits2::eTransferWrite,
	    vk::PipelineStageFlagBits2::eTopOfPipe,
	    vk::PipelineStageFlagBits2::eTransfer,
	    vk::ImageAspectFlagBits::eColor);

	// Blit
	vk::ImageBlit blitRegion{
	    .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
	    .srcOffsets     = {{vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<int32_t>(swapchain.extent.width), static_cast<int32_t>(swapchain.extent.height), 1}}},
	    .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
	    .dstOffsets     = {{vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<int32_t>(swapchain.extent.width), static_cast<int32_t>(swapchain.extent.height), 1}}}};

	commandBuffer.blitImage(*frames.storageImages[frames.frameIndex], vk::ImageLayout::eTransferSrcOptimal,
	                        swapchain.images[imageIndex], vk::ImageLayout::eTransferDstOptimal,
	                        blitRegion, vk::Filter::eLinear);

	// 3b. Restore storage image to eGeneral so it always matches the layout declared in
	// computeDescriptorSets. This prevents VUID-vkCmdDraw-None-09600 when the rasterizer's
	// draw commands follow in the same command buffer.
	transition_image_layout(
	    *frames.storageImages[frames.frameIndex],
	    vk::ImageLayout::eTransferSrcOptimal,
	    vk::ImageLayout::eGeneral,
	    vk::AccessFlagBits2::eTransferRead,
	    {},
	    vk::PipelineStageFlagBits2::eTransfer,
	    vk::PipelineStageFlagBits2::eBottomOfPipe,
	    vk::ImageAspectFlagBits::eColor);

	// 4. Transition SwapChain to Color Attachment for Rendering
	transition_image_layout(
	    swapchain.images[imageIndex],
	    vk::ImageLayout::eTransferDstOptimal,
	    vk::ImageLayout::eColorAttachmentOptimal,
	    vk::AccessFlagBits2::eTransferWrite,
	    vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentRead,
	    vk::PipelineStageFlagBits2::eTransfer,
	    vk::PipelineStageFlagBits2::eColorAttachmentOutput,
	    vk::ImageAspectFlagBits::eColor);
}

void EngineCore::recordSkinningPass(const vk::raii::CommandBuffer &commandBuffer) const
{
	std::unordered_map<int, const SceneNode *> instanceRootsByModel;
	for (const auto &node : scene->getAllNodes())
	{
		if (!node || node->modelId < 0)
		{
			continue;
		}
		ModelResource *modelRes = resourceManager->getModelResource(node->modelId);
		if (!modelRes || !modelRes->hasRuntimeSkinning || !*modelRes->skinningDescriptorSet || !modelRes->skinningJointMatricesMapped)
		{
			continue;
		}
		const SceneNode *parent         = node->getParent();
		const bool       isInstanceRoot = (parent == nullptr || parent->modelId != node->modelId);
		if (isInstanceRoot && !instanceRootsByModel.contains(node->modelId))
		{
			instanceRootsByModel.emplace(node->modelId, node.get());
		}
	}

	if (instanceRootsByModel.empty())
	{
		return;
	}

	vk::MemoryBarrier2 hostToComputeBarrier{
	    .srcStageMask  = vk::PipelineStageFlagBits2::eHost,
	    .srcAccessMask = vk::AccessFlagBits2::eHostWrite,
	    .dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
	    .dstAccessMask = vk::AccessFlagBits2::eShaderRead};
	vk::DependencyInfo hostToComputeDependency{
	    .memoryBarrierCount = 1,
	    .pMemoryBarriers    = &hostToComputeBarrier};
	commandBuffer.pipelineBarrier2(hostToComputeDependency);

	commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelines.skinningPipeline);

	for (const auto &[modelId, rootNode] : instanceRootsByModel)
	{
		ModelResource *modelRes = resourceManager->getModelResource(modelId);
		if (!modelRes || modelRes->skinningJointMatrixCount == 0 || modelRes->skinningVertexCount == 0)
		{
			continue;
		}

		std::unordered_map<int, const SceneNode *> nodesBySourceIndex;
		std::vector<const SceneNode *>             stack{rootNode};
		while (!stack.empty())
		{
			const SceneNode *current = stack.back();
			stack.pop_back();
			if (!current || current->modelId != modelId)
			{
				continue;
			}
			if (current->sourceNodeIndex >= 0 && !nodesBySourceIndex.contains(current->sourceNodeIndex))
			{
				nodesBySourceIndex.emplace(current->sourceNodeIndex, current);
			}
			for (const auto &child : current->getChildren())
			{
				if (child)
				{
					stack.push_back(child.get());
				}
			}
		}

		std::vector<glm::mat4> jointPalette(modelRes->skinningJointMatrixCount, glm::mat4(1.0f));
		for (const auto &skin : modelRes->skins)
		{
			for (size_t jointIndex = 0; jointIndex < skin.jointSourceNodeIndices.size(); ++jointIndex)
			{
				const uint32_t paletteIndex = skin.jointMatrixOffset + static_cast<uint32_t>(jointIndex);
				if (paletteIndex >= jointPalette.size())
				{
					continue;
				}
				const auto nodeIt = nodesBySourceIndex.find(skin.jointSourceNodeIndices[jointIndex]);
				if (nodeIt == nodesBySourceIndex.end() || !nodeIt->second)
				{
					continue;
				}
				jointPalette[paletteIndex] = nodeIt->second->getWorldTransform() * skin.inverseBindMatrices[jointIndex];
			}
		}

		memcpy(modelRes->skinningJointMatricesMapped, jointPalette.data(), sizeof(glm::mat4) * jointPalette.size());

		commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelines.skinningPipelineLayout, 0, {*modelRes->skinningDescriptorSet}, nullptr);

		Laphria::SkinningPushConstants push{};
		push.vertexCount       = modelRes->skinningVertexCount;
		push.jointMatrixOffset = 0;
		push.jointCount        = modelRes->skinningJointMatrixCount;
		commandBuffer.pushConstants<Laphria::SkinningPushConstants>(*pipelines.skinningPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, push);

		const uint32_t groupCountX = (modelRes->skinningVertexCount + 63u) / 64u;
		commandBuffer.dispatch(groupCountX, 1, 1);
	}

	vk::MemoryBarrier2 skinningToConsumerBarrier{
	    .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
	    .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
	    .dstStageMask  = vk::PipelineStageFlagBits2::eVertexInput | vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
	    .dstAccessMask = vk::AccessFlagBits2::eVertexAttributeRead | vk::AccessFlagBits2::eAccelerationStructureReadKHR};
	vk::DependencyInfo skinningToConsumerDependency{
	    .memoryBarrierCount = 1,
	    .pMemoryBarriers    = &skinningToConsumerBarrier};
	commandBuffer.pipelineBarrier2(skinningToConsumerDependency);

	if (ui.renderMode != RenderMode::Rasterizer)
	{
		resourceManager->recordSkinnedBLASRefit(commandBuffer);
	}
}

void EngineCore::recordClassicRTCommandBuffer(const vk::raii::CommandBuffer &commandBuffer, uint32_t imageIndex) const
{
	const uint32_t fi = frames.frameIndex;

	// 1. Transition RT Output Image to General Layout for Writing
	transition_image_layout(
	    *frames.rayTracingOutputImages[fi],
	    vk::ImageLayout::eUndefined,
	    vk::ImageLayout::eGeneral,
	    {},
	    vk::AccessFlagBits2::eShaderWrite,
	    vk::PipelineStageFlagBits2::eTopOfPipe,
	    vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
	    vk::ImageAspectFlagBits::eColor);

	// 2. Bind Classic RT Pipeline and Descriptor Sets
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *pipelines.classicRTPipeline);
	commandBuffer.bindDescriptorSets(
	    vk::PipelineBindPoint::eRayTracingKHR,
	    *pipelines.rayTracingPipelineLayout,
	    0,
	    {*rtDescriptorSets[fi], *descriptorSets[fi]},
	    nullptr);

	ScenePushConstants pushConstants{};
	pushConstants.modelMatrix = glm::mat4(1.0f);
	commandBuffer.pushConstants<ScenePushConstants>(
	    *pipelines.rayTracingPipelineLayout,
	    vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eMissKHR,
	    0,
	    pushConstants);

	// 3. Dispatch Rays
	vk::StridedDeviceAddressRegionKHR callableRegion{};
	commandBuffer.traceRaysKHR(
	    pipelines.classicRTRaygenRegion,
	    pipelines.classicRTMissRegion,
	    pipelines.classicRTHitRegion,
	    callableRegion,
	    swapchain.extent.width,
	    swapchain.extent.height,
	    1);

	// 4. Transition RT Output Image for Blit (General → TransferSrcOptimal)
	transition_image_layout(
	    *frames.rayTracingOutputImages[fi],
	    vk::ImageLayout::eGeneral,
	    vk::ImageLayout::eTransferSrcOptimal,
	    vk::AccessFlagBits2::eShaderWrite,
	    vk::AccessFlagBits2::eTransferRead,
	    vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
	    vk::PipelineStageFlagBits2::eTransfer,
	    vk::ImageAspectFlagBits::eColor);

	// 5. Transition SwapChain Image for Blit
	transition_image_layout(
	    swapchain.images[imageIndex],
	    vk::ImageLayout::eUndefined,
	    vk::ImageLayout::eTransferDstOptimal,
	    {},
	    vk::AccessFlagBits2::eTransferWrite,
	    vk::PipelineStageFlagBits2::eTopOfPipe,
	    vk::PipelineStageFlagBits2::eTransfer,
	    vk::ImageAspectFlagBits::eColor);

	// 6. Blit RT Output to SwapChain Image
	vk::ImageBlit blitRegion{
	    .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
	    .srcOffsets     = {{vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<int32_t>(swapchain.extent.width), static_cast<int32_t>(swapchain.extent.height), 1}}},
	    .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
	    .dstOffsets     = {{vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<int32_t>(swapchain.extent.width), static_cast<int32_t>(swapchain.extent.height), 1}}}};
	commandBuffer.blitImage(
	    *frames.rayTracingOutputImages[fi], vk::ImageLayout::eTransferSrcOptimal,
	    swapchain.images[imageIndex], vk::ImageLayout::eTransferDstOptimal,
	    blitRegion, vk::Filter::eLinear);

	// 6b. Restore RT output image to eGeneral so it always matches the layout declared in
	// rtDescriptorSets and denoiserDescriptorSets (prevents VUID-vkCmdDraw-None-09600 if
	// the render mode is switched back to Rasterizer in a subsequent frame).
	transition_image_layout(
	    *frames.rayTracingOutputImages[fi],
	    vk::ImageLayout::eTransferSrcOptimal,
	    vk::ImageLayout::eGeneral,
	    vk::AccessFlagBits2::eTransferRead,
	    {},
	    vk::PipelineStageFlagBits2::eTransfer,
	    vk::PipelineStageFlagBits2::eBottomOfPipe,
	    vk::ImageAspectFlagBits::eColor);

	// 7. Transition SwapChain to Color Attachment for UI Rendering
	transition_image_layout(
	    swapchain.images[imageIndex],
	    vk::ImageLayout::eTransferDstOptimal,
	    vk::ImageLayout::eColorAttachmentOptimal,
	    vk::AccessFlagBits2::eTransferWrite,
	    vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentRead,
	    vk::PipelineStageFlagBits2::eTransfer,
	    vk::PipelineStageFlagBits2::eColorAttachmentOutput,
	    vk::ImageAspectFlagBits::eColor);
}

void EngineCore::recordRayTracingCommandBuffer(const vk::raii::CommandBuffer &commandBuffer, uint32_t imageIndex) const
{
	const uint32_t fi         = frames.frameIndex;
	const uint32_t queryBase  = getPathTracerQueryBase(fi);
	const size_t   atrousBase = static_cast<size_t>(fi) * 2;
	const size_t   atrousA    = atrousBase + 0;
	const size_t   atrousB    = atrousBase + 1;

	const float    clampedScale         = std::clamp(ui.pathTracerSettings.resolutionScale, 0.5f, 1.0f);
	const float    secondaryScale       = ui.pathTracerSettings.reduceSecondaryEffects ? 0.90f : 1.0f;
	const float    effectiveScale       = std::clamp(clampedScale * secondaryScale, 0.5f, 1.0f);
	const uint32_t rtWidth              = std::max(1u, static_cast<uint32_t>(static_cast<float>(swapchain.extent.width) * effectiveScale));
	const uint32_t rtHeight             = std::max(1u, static_cast<uint32_t>(static_cast<float>(swapchain.extent.height) * effectiveScale));
	const uint32_t gx                   = (rtWidth + 15) / 16;
	const uint32_t gy                   = (rtHeight + 15) / 16;
	const bool     analysisEnabled      = ui.pathTracerAnalysisSettings.enableAnalysisMode;
	const int      debugAov             = analysisEnabled ? static_cast<int>(ui.pathTracerAnalysisSettings.debugAov) : 0;
	const int      debugAtrousIteration = analysisEnabled ? std::clamp(ui.pathTracerAnalysisSettings.debugAtrousIteration, 0, 4) : 0;
	if (fi < frames.ptAnalysisCounterMapped.size() && frames.ptAnalysisCounterMapped[fi])
	{
		std::memset(frames.ptAnalysisCounterMapped[fi], 0, sizeof(Laphria::PathTracerAnalysisCounters));
	}

	// PT analysis buffers are host-cleared and also persist shader writes across
	// submissions. Make those writes visible before the next raygen reads or updates them.
	vk::MemoryBarrier2 pathTracerStorageBufferBarrier{
	    .srcStageMask  = vk::PipelineStageFlagBits2::eHost | vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
	    .srcAccessMask = vk::AccessFlagBits2::eHostWrite | vk::AccessFlagBits2::eShaderStorageWrite,
	    .dstStageMask  = vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
	    .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite};
	vk::DependencyInfo pathTracerStorageBufferDependency{
	    .memoryBarrierCount = 1,
	    .pMemoryBarriers    = &pathTracerStorageBufferBarrier};
	commandBuffer.pipelineBarrier2(pathTracerStorageBufferDependency);

	// 1. Transition all PT images to general layout for writing.
	auto transitionToGeneral = [&](vk::Image img) {
		transition_image_layout(img, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
		                        {}, vk::AccessFlagBits2::eShaderWrite,
		                        vk::PipelineStageFlagBits2::eTopOfPipe, vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
		                        vk::ImageAspectFlagBits::eColor);
	};
	transitionToGeneral(*frames.rayTracingOutputImages[fi]);
	transitionToGeneral(*frames.rtGBufferNormals[fi]);
	transitionToGeneral(*frames.rtGBufferDepth[fi]);
	transitionToGeneral(*frames.rtMotionVectors[fi]);
	transitionToGeneral(*frames.atrousTemp[atrousA]);
	transitionToGeneral(*frames.atrousTemp[atrousB]);
	transitionToGeneral(*frames.ptReprojectionDebug[fi]);

	// 2. Ray tracing dispatch.
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *pipelines.rayTracingPipeline);
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR,
	                                 *pipelines.rayTracingPipelineLayout, 0,
	                                 {*rtDescriptorSets[fi], *descriptorSets[fi]}, nullptr);

	ScenePushConstants rtPush{};
	rtPush.modelMatrix                       = glm::mat4(1.0f);
	rtPush.cascadeIndex                      = -1;
	const uint32_t packedPathTracerMaterialIndex = packPathTracerMaterialSettings(ui.pathTracerSettings);
	rtPush.materialIndex = static_cast<int>(packedPathTracerMaterialIndex);
	rtPush.padding2      = debugAov;
	const uint32_t pathTracerFlags = packPathTracerFlags(ui.pathTracerSettings);
	rtPush.padding3 = pathTracerFlags;
	commandBuffer.pushConstants<ScenePushConstants>(*pipelines.rayTracingPipelineLayout,
	                                                vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eMissKHR,
	                                                0, rtPush);

	if (*ptTimestampQueryPool)
	{
		commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eRayTracingShaderKHR, *ptTimestampQueryPool, queryBase + kPtTS_RayTraceStart);
	}
	vk::StridedDeviceAddressRegionKHR callableRegion{};
	commandBuffer.traceRaysKHR(pipelines.raygenRegion, pipelines.missRegion, pipelines.hitRegion,
	                           callableRegion, rtWidth, rtHeight, 1);
	if (*ptTimestampQueryPool)
	{
		commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eRayTracingShaderKHR, *ptTimestampQueryPool, queryBase + kPtTS_RayTraceEnd);
	}

	// 3. Barrier: RT writes -> compute reads.
	auto barrierRTtoCompute = [&](vk::Image img) {
		transition_image_layout(img, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
		                        vk::AccessFlagBits2::eShaderWrite, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
		                        vk::PipelineStageFlagBits2::eRayTracingShaderKHR, vk::PipelineStageFlagBits2::eComputeShader,
		                        vk::ImageAspectFlagBits::eColor);
	};
	barrierRTtoCompute(*frames.rayTracingOutputImages[fi]);
	barrierRTtoCompute(*frames.rtGBufferNormals[fi]);
	barrierRTtoCompute(*frames.rtGBufferDepth[fi]);
	barrierRTtoCompute(*frames.rtMotionVectors[fi]);

	vk::BufferMemoryBarrier2 ptAnalysisRtToComputeBarrier{
	    .srcStageMask  = vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
	    .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
	    .dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
	    .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
	    .buffer        = *frames.ptAnalysisCounterBuffers[fi],
	    .offset        = 0,
	    .size          = sizeof(Laphria::PathTracerAnalysisCounters)};
	vk::DependencyInfo ptAnalysisRtToComputeDependency{
	    .bufferMemoryBarrierCount = 1,
	    .pBufferMemoryBarriers    = &ptAnalysisRtToComputeBarrier};
	commandBuffer.pipelineBarrier2(ptAnalysisRtToComputeDependency);

	// 4. Reprojection pass.
	if (ui.pathTracerSettings.enableReprojection)
	{
		commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelines.reprojectionPipeline);
		commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
		                                 *pipelines.denoiserPipelineLayout, 0, *denoiserDescriptorSets[fi], nullptr);

		constexpr float kStaticAlpha = 0.08f;
		float           historyAlpha = kStaticAlpha;
		if (ui.pathTracerSettings.enableMotionAwareAccumulation)
		{
			const float motionAlphaMin = std::clamp(ui.pathTracerSettings.motionAlphaMin, 0.05f, 0.40f);
			const float motionAlphaMax = std::clamp(ui.pathTracerSettings.motionAlphaMax, 0.20f, 1.00f);
			const float minAlpha       = std::min(motionAlphaMin, motionAlphaMax);
			const float maxAlpha       = std::max(motionAlphaMin, motionAlphaMax);

			if (ptForceHistoryReset)
			{
				historyAlpha = 1.0f;
			}
			else if (ptCameraMoved)
			{
				historyAlpha = glm::mix(minAlpha, maxAlpha, std::clamp(ptSmoothedMotion, 0.0f, 1.0f));
			}
		}
		else
		{
			historyAlpha = ptCameraMoved ? 1.0f : 0.1f;
		}

		DenoisePushConstants reproPush{
		    .stepSize             = 0,
		    .isLastPass           = 0,
		    .phiColor             = historyAlpha,
		    .phiNormal            = 128.0f,
		    .exposureScale        = ui.exposure,
		    .useRawInput          = 0,
		    .debugAov             = debugAov,
		    .debugAtrousIteration = debugAtrousIteration};
		commandBuffer.pushConstants<DenoisePushConstants>(*pipelines.denoiserPipelineLayout,
		                                                  vk::ShaderStageFlagBits::eCompute, 0, reproPush);
		if (*ptTimestampQueryPool)
		{
			commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, *ptTimestampQueryPool, queryBase + kPtTS_ReprojectionStart);
		}
		commandBuffer.dispatch(gx, gy, 1);
		if (*ptTimestampQueryPool)
		{
			commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, *ptTimestampQueryPool, queryBase + kPtTS_ReprojectionEnd);
		}
	}

	auto barrierCompute = [&](vk::Image img) {
		transition_image_layout(img, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
		                        vk::AccessFlagBits2::eShaderWrite, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
		                        vk::PipelineStageFlagBits2::eComputeShader, vk::PipelineStageFlagBits2::eComputeShader,
		                        vk::ImageAspectFlagBits::eColor);
	};
	barrierCompute(*frames.atrousTemp[atrousA]);
	barrierCompute(*frames.historyMoments[fi]);
	barrierCompute(*frames.ptReprojectionDebug[fi]);

	// 5. A-Trous denoiser.
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelines.atrousPipeline);
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
	                                 *pipelines.denoiserPipelineLayout, 0, *denoiserDescriptorSets[fi], nullptr);

	int atrousIterations = ui.pathTracerSettings.enableDenoiser ? std::clamp(ui.pathTracerSettings.denoiserIterations, 1, 5) : 0;
	if (ui.pathTracerSettings.enableMotionAwareAccumulation && atrousIterations > 0 &&
	    std::clamp(ptSmoothedMotion, 0.0f, 1.0f) > 0.35f)
	{
		atrousIterations = std::min(5, atrousIterations + 1);
	}
	if (analysisEnabled &&
	    ui.pathTracerAnalysisSettings.debugAov == UISystem::PathTracerDebugAov::AtrousIteration &&
	    atrousIterations > 0)
	{
		atrousIterations = std::clamp(debugAtrousIteration + 1, 1, atrousIterations);
	}
	const int useRawInput = ui.pathTracerSettings.enableReprojection ? 0 : 1;

	if (*ptTimestampQueryPool)
	{
		commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, *ptTimestampQueryPool, queryBase + kPtTS_DenoiserStart);
	}

	if (atrousIterations == 0)
	{
		// Pass-through tonemapping
		DenoisePushConstants atrousPush{
		    .stepSize             = 0,
		    .isLastPass           = 1,
		    .phiColor             = 1.0f,
		    .phiNormal            = 128.0f,
		    .exposureScale        = ui.exposure,
		    .useRawInput          = useRawInput,
		    .debugAov             = debugAov,
		    .debugAtrousIteration = debugAtrousIteration};
		commandBuffer.pushConstants<DenoisePushConstants>(*pipelines.denoiserPipelineLayout,
		                                                  vk::ShaderStageFlagBits::eCompute, 0, atrousPush);
		commandBuffer.dispatch(gx, gy, 1);
	}
	else
	{
		for (int iter = 0; iter < atrousIterations; ++iter)
		{
			const int32_t        stepSize   = 1 << iter;
			const int32_t        isLastPass = (iter == atrousIterations - 1) ? 1 : 0;
			DenoisePushConstants atrousPush{
			    .stepSize             = stepSize,
			    .isLastPass           = isLastPass,
			    .phiColor             = 1.0f,
			    .phiNormal            = 128.0f,
			    .exposureScale        = ui.exposure,
			    .useRawInput          = useRawInput,
			    .debugAov             = debugAov,
			    .debugAtrousIteration = debugAtrousIteration};
			commandBuffer.pushConstants<DenoisePushConstants>(*pipelines.denoiserPipelineLayout,
			                                                  vk::ShaderStageFlagBits::eCompute, 0, atrousPush);
			commandBuffer.dispatch(gx, gy, 1);

			if (!isLastPass)
			{
				const int writeBuf = iter % 2;
				barrierCompute(*frames.atrousTemp[(writeBuf == 0) ? atrousB : atrousA]);
			}
		}
	}

	if (*ptTimestampQueryPool)
	{
		commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, *ptTimestampQueryPool, queryBase + kPtTS_DenoiserEnd);
	}

	// 6. Blit denoised image to swapchain.
	transition_image_layout(*frames.rayTracingOutputImages[fi],
	                        vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal,
	                        vk::AccessFlagBits2::eShaderWrite, vk::AccessFlagBits2::eTransferRead,
	                        vk::PipelineStageFlagBits2::eComputeShader, vk::PipelineStageFlagBits2::eTransfer,
	                        vk::ImageAspectFlagBits::eColor);

	transition_image_layout(swapchain.images[imageIndex],
	                        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
	                        {}, vk::AccessFlagBits2::eTransferWrite,
	                        vk::PipelineStageFlagBits2::eTopOfPipe, vk::PipelineStageFlagBits2::eTransfer,
	                        vk::ImageAspectFlagBits::eColor);

	vk::ImageBlit blitRegion{
	    .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
	    .srcOffsets     = {{vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<int32_t>(rtWidth), static_cast<int32_t>(rtHeight), 1}}},
	    .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
	    .dstOffsets     = {{vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<int32_t>(swapchain.extent.width), static_cast<int32_t>(swapchain.extent.height), 1}}}};
	commandBuffer.blitImage(*frames.rayTracingOutputImages[fi], vk::ImageLayout::eTransferSrcOptimal,
	                        swapchain.images[imageIndex], vk::ImageLayout::eTransferDstOptimal, blitRegion, vk::Filter::eLinear);

	transition_image_layout(*frames.rayTracingOutputImages[fi],
	                        vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral,
	                        vk::AccessFlagBits2::eTransferRead, {},
	                        vk::PipelineStageFlagBits2::eTransfer, vk::PipelineStageFlagBits2::eBottomOfPipe,
	                        vk::ImageAspectFlagBits::eColor);

	transition_image_layout(swapchain.images[imageIndex],
	                        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eColorAttachmentOptimal,
	                        vk::AccessFlagBits2::eTransferWrite, vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentRead,
	                        vk::PipelineStageFlagBits2::eTransfer, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
	                        vk::ImageAspectFlagBits::eColor);
}

void EngineCore::transitionPersistentSurfelImagesToGeneral(uint32_t frameIndex) const
{
	(void)frameIndex;

	auto transitionImageSet = [&](const std::vector<VulkanUtils::VmaImage> &images) {
		for (const auto &image : images)
		{
			transition_image_layout(*image,
			                        vk::ImageLayout::eUndefined,
			                        vk::ImageLayout::eGeneral,
			                        {},
			                        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
			                        vk::PipelineStageFlagBits2::eTopOfPipe,
			                        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
			                        vk::ImageAspectFlagBits::eColor);
		}
	};

	transitionImageSet(surfelPathTracerResources.gBufferNormalImages);
	transitionImageSet(surfelPathTracerResources.gBufferDepthImages);
	transitionImageSet(surfelPathTracerResources.gBufferMotionMaterialImages);
	transitionImageSet(surfelPathTracerResources.gBufferAlbedoImages);
	transitionImageSet(surfelPathTracerResources.reflectionImages);
	transitionImageSet(surfelPathTracerResources.filteredReflectionImages);
	transitionImageSet(surfelPathTracerResources.lightingImages);
	transitionImageSet(surfelPathTracerResources.taaHistoryImages);
	transitionImageSet(surfelPathTracerResources.irradianceAtlasImages);
	transitionImageSet(surfelPathTracerResources.surfelDepthAtlasImages);
}

void EngineCore::recordSurfelPathTracerCommandBuffer(const vk::raii::CommandBuffer &commandBuffer, uint32_t imageIndex) const
{
	const uint32_t fi = frames.frameIndex;
	if (!surfelPathTracerResources.initialized())
	{
		throw std::runtime_error("Surfel path tracer resources are not initialized");
	}
	auto &mutableUi = const_cast<UISystem &>(ui);
	auto &mutableSurfelPathTracerResources =
	    const_cast<Laphria::SurfelPathTracerResources &>(surfelPathTracerResources);

	if (fi >= surfelPathTracerResources.outputImages.size() ||
	    fi >= surfelPathTracerSkyDescriptorSets.size() ||
	    fi >= descriptorSets.size())
	{
		throw std::runtime_error("Surfel path tracer frame resources are incomplete");
	}

	if (!surfelPathTracerPersistentImageLayoutsInitialized)
	{
		transitionPersistentSurfelImagesToGeneral(fi);
		surfelPathTracerPersistentImageLayoutsInitialized = true;
	}

	transition_image_layout(*surfelPathTracerResources.outputImages[fi],
	                        vk::ImageLayout::eUndefined,
	                        vk::ImageLayout::eGeneral,
	                        {},
	                        vk::AccessFlagBits2::eShaderWrite,
	                        vk::PipelineStageFlagBits2::eTopOfPipe,
	                        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
	                        vk::ImageAspectFlagBits::eColor);

	auto transitionSurfelOutputForBlit = [&]() {
		transition_image_layout(*surfelPathTracerResources.outputImages[fi],
		                        vk::ImageLayout::eGeneral,
		                        vk::ImageLayout::eTransferSrcOptimal,
		                        vk::AccessFlagBits2::eShaderWrite,
		                        vk::AccessFlagBits2::eTransferRead,
		                        vk::PipelineStageFlagBits2::eComputeShader,
		                        vk::PipelineStageFlagBits2::eTransfer,
		                        vk::ImageAspectFlagBits::eColor);
	};
	auto transitionSwapchainForBlit = [&]() {
		transition_image_layout(swapchain.images[imageIndex],
		                        vk::ImageLayout::eUndefined,
		                        vk::ImageLayout::eTransferDstOptimal,
		                        {},
		                        vk::AccessFlagBits2::eTransferWrite,
		                        vk::PipelineStageFlagBits2::eTopOfPipe,
		                        vk::PipelineStageFlagBits2::eTransfer,
		                        vk::ImageAspectFlagBits::eColor);
	};
	auto recordSurfelFinalBlit = [&]() {
		surfelPathTracerPasses.recordFinalBlit(commandBuffer,
		                                       surfelPathTracerResources,
		                                       swapchain.images[imageIndex],
		                                       fi,
		                                       swapchain.extent);
	};
	auto transitionSwapchainForUi = [&]() {
		transition_image_layout(swapchain.images[imageIndex],
		                        vk::ImageLayout::eTransferDstOptimal,
		                        vk::ImageLayout::eColorAttachmentOptimal,
		                        vk::AccessFlagBits2::eTransferWrite,
		                        vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentRead,
		                        vk::PipelineStageFlagBits2::eTransfer,
		                        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		                        vk::ImageAspectFlagBits::eColor);
	};
	auto recordSurfelPathTracerNoSceneFallback = [&]() {
		surfelPathTracerPasses.recordSkyPass(commandBuffer,
		                                     pipelines.surfelPathTracerPipelines,
		                                     surfelPathTracerResources,
		                                     *surfelPathTracerSkyDescriptorSets[fi],
		                                     *descriptorSets[fi],
		                                     fi,
		                                     swapchain.extent);
		transitionSurfelOutputForBlit();
		transitionSwapchainForBlit();
		recordSurfelFinalBlit();
		transitionSwapchainForUi();
		surfelPathTracerHistoryValid.fill(false);
	};

	const auto &surfelSettings = mutableUi.surfelPathTracerSettings;
	if (!surfelSettings.enabled)
	{
		recordSurfelPathTracerNoSceneFallback();
		return;
	}

	if (!resourceManager || resourceManager->getModelCount() == 0)
	{
		recordSurfelPathTracerNoSceneFallback();
		return;
	}

	if (fi >= surfelPathTracerResources.gBufferNormalImages.size() ||
	    fi >= surfelPathTracerResources.gBufferDepthImages.size() ||
	    fi >= surfelPathTracerResources.gBufferMotionMaterialImages.size() ||
	    fi >= surfelPathTracerResources.gBufferAlbedoImages.size() ||
	    fi >= surfelPathTracerStorageDescriptorSets.size() ||
	    fi >= surfelPathTracerRtDescriptorSets.size())
	{
		throw std::runtime_error("Surfel path tracer scene frame resources are incomplete");
	}

	auto transitionSurfelGBufferToRtWrite = [&](vk::Image image) {
		transition_image_layout(image,
		                        vk::ImageLayout::eGeneral,
		                        vk::ImageLayout::eGeneral,
		                        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
		                        vk::AccessFlagBits2::eShaderWrite,
		                        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
		                        vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
		                        vk::ImageAspectFlagBits::eColor);
	};
	transitionSurfelGBufferToRtWrite(*surfelPathTracerResources.gBufferNormalImages[fi]);
	transitionSurfelGBufferToRtWrite(*surfelPathTracerResources.gBufferDepthImages[fi]);
	transitionSurfelGBufferToRtWrite(*surfelPathTracerResources.gBufferMotionMaterialImages[fi]);
	transitionSurfelGBufferToRtWrite(*surfelPathTracerResources.gBufferAlbedoImages[fi]);

	surfelPathTracerPasses.recordGBufferPass(commandBuffer,
	                                         pipelines.surfelPathTracerPipelines,
	                                         *surfelPathTracerRtDescriptorSets[fi],
	                                         *surfelPathTracerStorageDescriptorSets[fi],
	                                         *descriptorSets[fi],
	                                         swapchain.extent);
	surfelPathTracerPasses.recordImageBarrierGBufferToCompute(commandBuffer, surfelPathTracerResources, fi);

	const bool resetPersistent = surfelSettings.resetSurfels ||
	                             surfelPathTracerResources.needsPersistentReset();
	const bool updateSurfels = !surfelSettings.lockSurfels || resetPersistent;
	// Current descriptors expose one history image per frame-in-flight, so a frame cannot
	// safely sample the previous completed frame while writing the current history image.
	const bool historyReady = false;
	if (updateSurfels)
	{
		surfelPathTracerPasses.recordPreparePass(commandBuffer,
		                                         pipelines.surfelPathTracerPipelines,
		                                         *surfelPathTracerStorageDescriptorSets[fi],
		                                         surfelPathTracerResources.maxSurfelsCapacity(),
		                                         resetPersistent,
		                                         surfelPathTracerResources.cellCount(),
		                                         surfelPathTracerResources.perCellSurfelLimitCapacity());
		mutableSurfelPathTracerResources.markPersistentResetConsumed();
		mutableUi.surfelPathTracerSettings.resetSurfels = false;

		surfelPathTracerPasses.recordStorageBarrierComputeToCompute(commandBuffer);
		surfelPathTracerPasses.recordEvaluatePass(commandBuffer,
		                                          pipelines.surfelPathTracerPipelines,
		                                          *surfelPathTracerStorageDescriptorSets[fi],
		                                          *descriptorSets[fi],
		                                          Laphria::SurfelPathTracerEvaluateMode::Generate,
		                                          surfelSettings.cellSize,
		                                          surfelPathTracerResources.cellDimensionCapacity(),
		                                          surfelPathTracerResources.maxSurfelsCapacity(),
		                                          surfelSettings.placementThreshold,
		                                          surfelSettings.removalThreshold,
		                                          surfelSettings.surfelTargetArea,
		                                          surfelSettings.surfelMinRadius,
		                                          fi,
		                                          surfelSettings.lockSurfels,
		                                          surfelSettings.enableSurfelPlacement,
		                                          surfelSettings.enableSurfelRemoval,
		                                          surfelSettings.maxSurfelSamplesPerQuery,
		                                          swapchain.extent);
		surfelPathTracerPasses.recordStorageBarrierComputeToCompute(commandBuffer);
		surfelPathTracerPasses.recordUpdatePass(commandBuffer,
		                                        pipelines.surfelPathTracerPipelines,
		                                        *surfelPathTracerStorageDescriptorSets[fi],
		                                        *descriptorSets[fi],
		                                        surfelPathTracerResources.maxSurfelsCapacity(),
		                                        surfelPathTracerResources.maxRaysPerFrameCapacity(),
		                                        surfelSettings.cellSize,
		                                        surfelPathTracerResources.cellDimensionCapacity(),
		                                        surfelSettings.minRaysPerSurfel,
		                                        surfelSettings.maxRaysPerSurfel,
		                                        surfelSettings.varianceSensitivity,
		                                        surfelSettings.lockSurfels);
		surfelPathTracerPasses.recordStorageBarrierComputeToCompute(commandBuffer);
		surfelPathTracerPasses.recordCellInfoPass(commandBuffer,
		                                          pipelines.surfelPathTracerPipelines,
		                                          *surfelPathTracerStorageDescriptorSets[fi],
		                                          surfelPathTracerResources.cellCount(),
		                                          surfelPathTracerResources.perCellSurfelLimitCapacity());
		surfelPathTracerPasses.recordStorageBarrierComputeToCompute(commandBuffer);
		surfelPathTracerPasses.recordCellToSurfelPass(commandBuffer,
		                                              pipelines.surfelPathTracerPipelines,
		                                              *surfelPathTracerStorageDescriptorSets[fi],
		                                              *descriptorSets[fi],
		                                              surfelPathTracerResources.maxSurfelsCapacity(),
		                                              surfelSettings.cellSize,
		                                              surfelPathTracerResources.cellDimensionCapacity(),
		                                              surfelPathTracerResources.perCellSurfelLimitCapacity());
		surfelPathTracerPasses.recordStorageBarrierComputeToRt(commandBuffer);
		surfelPathTracerPasses.recordSurfelRayTracePass(commandBuffer,
		                                                pipelines.surfelPathTracerPipelines,
		                                                surfelPathTracerResources,
		                                                *surfelPathTracerRtDescriptorSets[fi],
		                                                *surfelPathTracerStorageDescriptorSets[fi],
		                                                *descriptorSets[fi],
		                                                surfelPathTracerResources.maxRaysPerFrameCapacity());
		surfelPathTracerPasses.recordStorageBarrierRtToCompute(commandBuffer);
		surfelPathTracerPasses.recordIntegratePass(commandBuffer,
		                                           pipelines.surfelPathTracerPipelines,
		                                           *surfelPathTracerStorageDescriptorSets[fi],
		                                           surfelPathTracerResources.maxSurfelsCapacity());
		surfelPathTracerPasses.recordStorageBarrierComputeToCompute(commandBuffer);
	}
	else
	{
		surfelPathTracerPasses.recordStorageBarrierComputeToCompute(commandBuffer);
	}
	surfelPathTracerPasses.recordEvaluatePass(commandBuffer,
	                                          pipelines.surfelPathTracerPipelines,
	                                          *surfelPathTracerStorageDescriptorSets[fi],
	                                          *descriptorSets[fi],
	                                          Laphria::SurfelPathTracerEvaluateMode::Resolve,
	                                          surfelSettings.cellSize,
	                                          surfelPathTracerResources.cellDimensionCapacity(),
	                                          surfelPathTracerResources.maxSurfelsCapacity(),
	                                          surfelSettings.placementThreshold,
	                                          surfelSettings.removalThreshold,
	                                          surfelSettings.surfelTargetArea,
	                                          surfelSettings.surfelMinRadius,
	                                          fi,
	                                          surfelSettings.lockSurfels,
	                                          false,
	                                          false,
	                                          surfelSettings.maxSurfelSamplesPerQuery,
	                                          swapchain.extent);

	surfelPathTracerPasses.recordStorageBarrierComputeToRt(commandBuffer);
	surfelPathTracerPasses.recordReflectionPass(commandBuffer,
	                                            pipelines.surfelPathTracerPipelines,
	                                            surfelPathTracerResources,
	                                            *surfelPathTracerRtDescriptorSets[fi],
	                                            *surfelPathTracerStorageDescriptorSets[fi],
	                                            *descriptorSets[fi],
	                                            surfelSettings.enableReflections,
	                                            surfelSettings.cellSize,
	                                            surfelPathTracerResources.cellDimensionCapacity(),
	                                            swapchain.extent,
	                                            fi);
	surfelPathTracerPasses.recordStorageBarrierRtToCompute(commandBuffer);
	surfelPathTracerPasses.recordReflectionFilterPass(commandBuffer,
	                                                  pipelines.surfelPathTracerPipelines,
	                                                  *surfelPathTracerStorageDescriptorSets[fi],
	                                                  surfelSettings.enableReflections && surfelSettings.enableReflectionFilter,
	                                                  !historyReady,
	                                                  swapchain.extent,
	                                                  fi);
	surfelPathTracerPasses.recordStorageBarrierComputeToCompute(commandBuffer);
	const bool preserveReflectionDebug =
	    surfelSettings.debugView == UISystem::SurfelPathTracerDebugView::ReflectionRaw ||
	    surfelSettings.debugView == UISystem::SurfelPathTracerDebugView::ReflectionFiltered;
	const bool useBilateralReflection =
	    surfelSettings.enableReflections &&
	    surfelSettings.enableBilateralCleanup &&
	    !preserveReflectionDebug;
	surfelPathTracerPasses.recordBilateralPass(commandBuffer,
	                                           pipelines.surfelPathTracerPipelines,
	                                           *surfelPathTracerStorageDescriptorSets[fi],
	                                           useBilateralReflection,
	                                           swapchain.extent);
	surfelPathTracerPasses.recordStorageBarrierComputeToCompute(commandBuffer);
	surfelPathTracerPasses.recordLightIntegratePass(commandBuffer,
	                                                pipelines.surfelPathTracerPipelines,
	                                                *surfelPathTracerStorageDescriptorSets[fi],
	                                                *descriptorSets[fi],
	                                                surfelSettings.enableDiffuseGi,
	                                                surfelSettings.enableReflections,
	                                                useBilateralReflection,
	                                                surfelSettings.cellSize,
	                                                surfelPathTracerResources.cellDimensionCapacity(),
	                                                surfelPathTracerResources.perCellSurfelLimitCapacity(),
	                                                static_cast<uint32_t>(surfelSettings.debugView),
	                                                swapchain.extent);
	surfelPathTracerPasses.recordStorageBarrierComputeToCompute(commandBuffer);
	const bool taaHistoryEnabled =
	    historyReady &&
	    surfelSettings.enableTaa &&
	    surfelSettings.debugView == UISystem::SurfelPathTracerDebugView::FinalColor;
	surfelPathTracerPasses.recordTaaPass(commandBuffer,
	                                     pipelines.surfelPathTracerPipelines,
	                                     *surfelPathTracerStorageDescriptorSets[fi],
	                                     *descriptorSets[fi],
	                                     taaHistoryEnabled,
	                                     !historyReady,
	                                     swapchain.extent,
	                                     fi);
	if (taaHistoryEnabled)
	{
		surfelPathTracerHistoryValid[fi] = true;
	}
	else
	{
		surfelPathTracerHistoryValid.fill(false);
	}

	transitionSurfelOutputForBlit();
	transitionSwapchainForBlit();
	recordSurfelFinalBlit();
	transitionSwapchainForUi();
}

void EngineCore::createDescriptorPool()
{
	// Generous pool sizes to accommodate an arbitrary number of loaded models.
	// eSampledImage / eSampler are separate because the shadow map binding uses them
	// as distinct descriptor types (binding 1 and 2 in the global layout).
	constexpr uint32_t                    poolScale = Laphria::EngineConfig::kDescriptorPoolScale;
	std::array<vk::DescriptorPoolSize, 7> poolSizes = {
	    vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, poolScale},
	    // 1000 per loaded model (material textures) + 2×1000 for the two RT descriptor sets.
	    vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 5 * poolScale},
	    vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, poolScale},
	    vk::DescriptorPoolSize{vk::DescriptorType::eSampler, poolScale},
	    // 1000 for materials + vertex/index arrays + PT analysis counter buffers.
	    vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 16 * poolScale},
	    vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, poolScale},
	    vk::DescriptorPoolSize{vk::DescriptorType::eAccelerationStructureKHR, MAX_FRAMES_IN_FLIGHT}};

	vk::DescriptorPoolCreateInfo poolInfo{
	    // eFreeDescriptorSet: allows individual sets to be freed (needed by ResourceManager).
	    // eUpdateAfterBind: required for bindless descriptor indexing (VK_EXT_descriptor_indexing).
	    .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet |
	             vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind,
	    .maxSets       = poolScale * MAX_FRAMES_IN_FLIGHT,
	    .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
	    .pPoolSizes    = poolSizes.data()};
	descriptorPool = vk::raii::DescriptorPool(vulkan.logicalDevice, poolInfo);
}

void EngineCore::createDescriptorSets()
{
	std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *pipelines.descriptorSetLayoutGlobal);

	vk::DescriptorSetAllocateInfo allocInfo{
	    .descriptorPool     = *descriptorPool,
	    .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
	    .pSetLayouts        = layouts.data()};

	descriptorSets.clear();
	descriptorSets = vulkan.logicalDevice.allocateDescriptorSets(allocInfo);

	// Global descriptor set layout (Set 0):
	//   binding 0 → UniformBufferObject  (view/proj/light/cascade matrices, camera pos)
	//   binding 1 → shadow depth array   (sampled, ShaderReadOnlyOptimal)
	//   binding 2 → shadow PCF sampler   (comparison sampler)
	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		vk::DescriptorBufferInfo bufferInfo{
		    .buffer = *frames.uniformBuffers[i],
		    .offset = 0,
		    .range  = sizeof(Laphria::UniformBufferObject)};

		vk::WriteDescriptorSet uboWrite{
		    .dstSet          = *descriptorSets[i],
		    .dstBinding      = 0,
		    .dstArrayElement = 0,
		    .descriptorCount = 1,
		    .descriptorType  = vk::DescriptorType::eUniformBuffer,
		    .pBufferInfo     = &bufferInfo};

		// The shadow array image starts in eUndefined; we use eShaderReadOnlyOptimal
		// as the declared layout here because the first frame's shadow pass will
		// transition it via eUndefined → eDepthAttachmentOptimal → eShaderReadOnlyOptimal
		// before the main pass samples it.
		vk::DescriptorImageInfo shadowImageInfo{
		    .imageView   = *frames.shadowArrayViews[i],
		    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};

		vk::WriteDescriptorSet shadowImageWrite{
		    .dstSet          = *descriptorSets[i],
		    .dstBinding      = 1,
		    .dstArrayElement = 0,
		    .descriptorCount = 1,
		    .descriptorType  = vk::DescriptorType::eSampledImage,
		    .pImageInfo      = &shadowImageInfo};

		vk::DescriptorImageInfo shadowSamplerInfo{
		    .sampler = *frames.shadowSampler};

		vk::WriteDescriptorSet shadowSamplerWrite{
		    .dstSet          = *descriptorSets[i],
		    .dstBinding      = 2,
		    .dstArrayElement = 0,
		    .descriptorCount = 1,
		    .descriptorType  = vk::DescriptorType::eSampler,
		    .pImageInfo      = &shadowSamplerInfo};

		std::array<vk::WriteDescriptorSet, 3> writes = {uboWrite, shadowImageWrite, shadowSamplerWrite};
		vulkan.logicalDevice.updateDescriptorSets(writes, {});
	}
}

void EngineCore::createTimestampQueryPool()
{
	vk::QueryPoolCreateInfo queryPoolInfo{
	    .queryType  = vk::QueryType::eTimestamp,
	    .queryCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * kPtTimestampQueryCountPerFrame};
	ptTimestampQueryPool = vk::raii::QueryPool(vulkan.logicalDevice, queryPoolInfo);
	timestampPeriodNs    = vulkan.physicalDevice.getProperties().limits.timestampPeriod;
}

uint32_t EngineCore::getPathTracerQueryBase(uint32_t frameSlot) const
{
	return frameSlot * kPtTimestampQueryCountPerFrame;
}

void EngineCore::collectPathTracerTimings(uint32_t frameSlot)
{
	if (!*ptTimestampQueryPool || !ptTimestampsValid[frameSlot])
	{
		return;
	}

	std::array<uint64_t, kPtTimestampQueryCountPerFrame> timestamps{};
	const VkResult                                       queryResult = vkGetQueryPoolResults(
        static_cast<VkDevice>(*vulkan.logicalDevice),
        static_cast<VkQueryPool>(*ptTimestampQueryPool),
        getPathTracerQueryBase(frameSlot),
        kPtTimestampQueryCountPerFrame,
        sizeof(timestamps),
        timestamps.data(),
        sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT);

	if (queryResult != VK_SUCCESS)
	{
		return;
	}

	auto toMs = [this](uint64_t start, uint64_t end) -> float {
		if (end <= start)
		{
			return 0.0f;
		}
		const double deltaTicks = static_cast<double>(end - start);
		const double deltaNs    = deltaTicks * static_cast<double>(timestampPeriodNs);
		return static_cast<float>(deltaNs * 1e-6);
	};

	ui.pathTracerPerfStats.tlasBuildMs    = toMs(timestamps[kPtTS_TlasStart], timestamps[kPtTS_TlasEnd]);
	ui.pathTracerPerfStats.rayTraceMs     = toMs(timestamps[kPtTS_RayTraceStart], timestamps[kPtTS_RayTraceEnd]);
	ui.pathTracerPerfStats.reprojectionMs = toMs(timestamps[kPtTS_ReprojectionStart], timestamps[kPtTS_ReprojectionEnd]);
	ui.pathTracerPerfStats.denoiserMs     = toMs(timestamps[kPtTS_DenoiserStart], timestamps[kPtTS_DenoiserEnd]);

	const uint64_t totalStart           = (timestamps[kPtTS_TlasStart] != 0) ? timestamps[kPtTS_TlasStart] : timestamps[kPtTS_RayTraceStart];
	const uint64_t totalEnd             = (timestamps[kPtTS_DenoiserEnd] != 0) ? timestamps[kPtTS_DenoiserEnd] : timestamps[kPtTS_RayTraceEnd];
	ui.pathTracerPerfStats.totalFrameMs = toMs(totalStart, totalEnd);

	constexpr size_t kRollingWindow = 300;
	ptRollingTotalMs.push_back(ui.pathTracerPerfStats.totalFrameMs);
	ptRollingRayTraceMs.push_back(ui.pathTracerPerfStats.rayTraceMs);
	ptRollingDenoiserMs.push_back(ui.pathTracerPerfStats.denoiserMs);
	if (ptRollingTotalMs.size() > kRollingWindow)
	{
		ptRollingTotalMs.erase(ptRollingTotalMs.begin());
	}
	if (ptRollingRayTraceMs.size() > kRollingWindow)
	{
		ptRollingRayTraceMs.erase(ptRollingRayTraceMs.begin());
	}
	if (ptRollingDenoiserMs.size() > kRollingWindow)
	{
		ptRollingDenoiserMs.erase(ptRollingDenoiserMs.begin());
	}

	updatePathTracerTimingPercentiles();
}

void EngineCore::updatePathTracerTimingPercentiles()
{
	const auto totalPct                        = computePercentiles(ptRollingTotalMs);
	const auto rayPct                          = computePercentiles(ptRollingRayTraceMs);
	const auto denoisePct                      = computePercentiles(ptRollingDenoiserMs);
	ui.pathTracerPerfStats.totalFrameP50Ms     = totalPct.p50;
	ui.pathTracerPerfStats.totalFrameP95Ms     = totalPct.p95;
	ui.pathTracerPerfStats.totalFrameP99Ms     = totalPct.p99;
	ui.pathTracerPerfStats.rayTraceP95Ms       = rayPct.p95;
	ui.pathTracerPerfStats.denoiserP95Ms       = denoisePct.p95;
	ui.pathTracerPerfStats.analysisSampleCount = static_cast<uint32_t>(ptRollingTotalMs.size());
}

void EngineCore::collectPathTracerAnalysisCounters(uint32_t frameSlot)
{
	if (frameSlot >= frames.ptAnalysisCounterMapped.size() || !frames.ptAnalysisCounterMapped[frameSlot])
	{
		return;
	}

	const auto *counters                         = reinterpret_cast<const Laphria::PathTracerAnalysisCounters *>(frames.ptAnalysisCounterMapped[frameSlot]);
	ui.pathTracerPerfStats.historyAcceptedCount  = counters->historyAcceptedCount;
	ui.pathTracerPerfStats.historyRejectedCount  = counters->historyRejectedCount;
	ui.pathTracerPerfStats.skyHitCount           = counters->skyHitCount;
	ui.pathTracerPerfStats.fireflyClampCount     = counters->fireflyClampCount;
	ui.pathTracerPerfStats.pixelSampleCount      = counters->pixelCount;

	const float historyTotal                      = static_cast<float>(counters->historyAcceptedCount + counters->historyRejectedCount);
	ui.pathTracerPerfStats.historyAcceptanceRatio = (historyTotal > 0.0f) ? static_cast<float>(counters->historyAcceptedCount) / historyTotal : 0.0f;
	ui.pathTracerPerfStats.historyRejectionRatio  = (historyTotal > 0.0f) ? static_cast<float>(counters->historyRejectedCount) / historyTotal : 0.0f;

	const float pixelTotal                   = static_cast<float>(std::max(counters->pixelCount, 1u));
	ui.pathTracerPerfStats.skyHitRatio       = static_cast<float>(counters->skyHitCount) / pixelTotal;
	ui.pathTracerPerfStats.fireflyClampRatio = static_cast<float>(counters->fireflyClampCount) / pixelTotal;

	auto &analysis = ui.pathTracerAnalysisSettings;
	if (!(analysis.enableAnalysisMode && analysis.benchmarkActive && analysis.runBaselineSweep))
	{
		return;
	}

	if (ptSweepConfigs.empty())
	{
		ptSweepConfigs         = buildPathTracerBaselineSweepMatrix();
		ptSweepConfigIndex     = 0;
		ptSweepWarmupRemaining = analysis.warmupFrames;
		ptSweepSampleRemaining = analysis.sampleFrames;
		ptSweepScores.clear();
		ptBacklogItems.clear();
		ptSampleTotalMs.clear();
		ptSampleRayTraceMs.clear();
		ptSampleDenoiserMs.clear();
		analysis.recommendationManual.clear();
		analysis.recommendationAutoBalanced.clear();
		analysis.recommendationAutoAggressive.clear();
		analysis.backlogSummary.clear();
		const auto benchmarkCsvPath                          = resolveAnalysisOutputPath(kPtBenchmarkCsvFileName);
		ui.pathTracerAnalysisSettings.benchmarkCsvOutputPath = benchmarkCsvPath.string();
		std::ofstream out(benchmarkCsvPath, std::ios::trunc);
		if (out.is_open())
		{
			out << "config_index,resolution_scale,denoiser_iterations,reprojection,motion_aware,reduce_secondary,"
			       "total_p50_ms,total_p95_ms,total_p99_ms,ray_p95_ms,denoiser_p95_ms,"
			       "history_accept_ratio,history_reject_ratio,sky_hit_ratio,firefly_clamp_ratio,"
			       "budget_pass,composite_score\n";
		}
		else
		{
			LOGE("Failed to open benchmark CSV for writing: %s", benchmarkCsvPath.string().c_str());
		}
	}

	if (ptSweepConfigIndex >= ptSweepConfigs.size())
	{
		analysis.benchmarkActive  = false;
		analysis.runBaselineSweep = false;
		return;
	}

	if (ptSweepWarmupRemaining > 0)
	{
		--ptSweepWarmupRemaining;
		return;
	}

	if (ptSweepSampleRemaining > 0)
	{
		ptSampleTotalMs.push_back(ui.pathTracerPerfStats.totalFrameMs);
		ptSampleRayTraceMs.push_back(ui.pathTracerPerfStats.rayTraceMs);
		ptSampleDenoiserMs.push_back(ui.pathTracerPerfStats.denoiserMs);
		--ptSweepSampleRemaining;

		if (analysis.adaptiveSampling &&
		    static_cast<int>(ptSampleTotalMs.size()) >= analysis.minSampleFrames)
		{
			const int window = std::min(analysis.convergenceWindowFrames,
			                            static_cast<int>(ptSampleTotalMs.size()) / 2);
			if (window >= 20)
			{
				const auto beginPrev = ptSampleTotalMs.end() - (window * 2);
				const auto endPrev   = ptSampleTotalMs.end() - window;
				const auto beginCurr = endPrev;
				const auto endCurr   = ptSampleTotalMs.end();

				std::vector<float> prevWindow(beginPrev, endPrev);
				std::vector<float> currWindow(beginCurr, endCurr);
				const auto         prevPct     = computePercentiles(prevWindow);
				const auto         currPct     = computePercentiles(currWindow);
				const float        denom       = std::max(prevPct.p95, 0.001f);
				const float        relP95Delta = std::abs(currPct.p95 - prevPct.p95) / denom;
				if (relP95Delta <= analysis.p95ConvergenceThreshold)
				{
					ptSweepSampleRemaining = 0;
				}
			}
		}
	}

	if (ptSweepSampleRemaining == 0 && !ptSampleTotalMs.empty())
	{
		const auto           totalPct   = computePercentiles(ptSampleTotalMs);
		const auto           rayPct     = computePercentiles(ptSampleRayTraceMs);
		const auto           denoisePct = computePercentiles(ptSampleDenoiserMs);
		PathTracerScoreInput scoreInput{};
		scoreInput.totalFrameMsP95       = totalPct.p95;
		scoreInput.targetBudgetMs        = 16.67f;
		scoreInput.historyRejectionRatio = ui.pathTracerPerfStats.historyRejectionRatio;
		scoreInput.skyHitRatio           = ui.pathTracerPerfStats.skyHitRatio;
		scoreInput.fireflyClampRatio     = ui.pathTracerPerfStats.fireflyClampRatio;
		scoreInput.visualFidelityScore   = analysis.benchmarkVisualFidelityScore;
		const auto runScore              = scorePathTracerRun(scoreInput);
		ptSweepScores.push_back(runScore);
		const auto &cfg = ptSweepConfigs[ptSweepConfigIndex];

		const auto benchmarkCsvPath                          = resolveAnalysisOutputPath(kPtBenchmarkCsvFileName);
		ui.pathTracerAnalysisSettings.benchmarkCsvOutputPath = benchmarkCsvPath.string();
		std::ofstream out(benchmarkCsvPath, std::ios::app);
		if (out.is_open())
		{
			out << ptSweepConfigIndex << ','
			    << cfg.resolutionScale << ','
			    << cfg.denoiserIterations << ','
			    << (cfg.enableReprojection ? 1 : 0) << ','
			    << (cfg.enableMotionAwareAccumulation ? 1 : 0) << ','
			    << (cfg.reduceSecondaryEffects ? 1 : 0) << ','
			    << totalPct.p50 << ','
			    << totalPct.p95 << ','
			    << totalPct.p99 << ','
			    << rayPct.p95 << ','
			    << denoisePct.p95 << ','
			    << ui.pathTracerPerfStats.historyAcceptanceRatio << ','
			    << ui.pathTracerPerfStats.historyRejectionRatio << ','
			    << ui.pathTracerPerfStats.skyHitRatio << ','
			    << ui.pathTracerPerfStats.fireflyClampRatio << ','
			    << (runScore.budgetPass ? 1 : 0) << ','
			    << runScore.compositeScore << '\n';
		}
		else
		{
			LOGE("Failed to append benchmark CSV: %s", benchmarkCsvPath.string().c_str());
		}

		++ptSweepConfigIndex;
		ptSweepWarmupRemaining = analysis.warmupFrames;
		ptSweepSampleRemaining = analysis.sampleFrames;
		ptSampleTotalMs.clear();
		ptSampleRayTraceMs.clear();
		ptSampleDenoiserMs.clear();

		if (ptSweepConfigIndex >= ptSweepConfigs.size())
		{
			analysis.benchmarkActive  = false;
			analysis.runBaselineSweep = false;
			if (!ptSweepScores.empty() && ptSweepScores.size() == ptSweepConfigs.size())
			{
				size_t bestIndex       = 0;
				bool   foundBudgetPass = false;
				for (size_t i = 0; i < ptSweepScores.size(); ++i)
				{
					const auto &candidate = ptSweepScores[i];
					const auto &best      = ptSweepScores[bestIndex];
					if (!foundBudgetPass && candidate.budgetPass)
					{
						bestIndex       = i;
						foundBudgetPass = true;
						continue;
					}
					if (foundBudgetPass)
					{
						if (candidate.budgetPass && candidate.compositeScore > best.compositeScore)
						{
							bestIndex = i;
						}
					}
					else if (candidate.compositeScore > best.compositeScore)
					{
						bestIndex = i;
					}
				}

				const auto &bestCfg = ptSweepConfigs[bestIndex];
				char        manualPreset[256];
				std::snprintf(manualPreset, sizeof(manualPreset),
				              "scale=%.2f, denoise=%d, reproj=%s, motionAware=%s",
				              bestCfg.resolutionScale,
				              bestCfg.denoiserIterations,
				              bestCfg.enableReprojection ? "on" : "off",
				              bestCfg.enableMotionAwareAccumulation ? "on" : "off");
				analysis.recommendationManual = manualPreset;

				const float balancedScale   = std::max(0.50f, bestCfg.resolutionScale - 0.05f);
				const int   balancedDenoise = std::max(1, bestCfg.denoiserIterations - 1);
				char        balancedPreset[256];
				std::snprintf(balancedPreset, sizeof(balancedPreset),
				              "scale=%.2f, denoise=%d, reproj=on, motionAware=on",
				              balancedScale, balancedDenoise);
				analysis.recommendationAutoBalanced = balancedPreset;

				const float aggressiveScale   = std::max(0.50f, bestCfg.resolutionScale - 0.10f);
				const int   aggressiveDenoise = std::max(1, bestCfg.denoiserIterations - 2);
				char        aggressivePreset[256];
				std::snprintf(aggressivePreset, sizeof(aggressivePreset),
				              "scale=%.2f, denoise=%d, reproj=on, motionAware=on, reduceSecondary=on",
				              aggressiveScale, aggressiveDenoise);
				analysis.recommendationAutoAggressive = aggressivePreset;

				ptBacklogItems = buildDefaultFidelityBacklog(
				    ui.pathTracerPerfStats.rayTraceP95Ms,
				    ui.pathTracerPerfStats.reprojectionMs,
				    ui.pathTracerPerfStats.denoiserP95Ms,
				    ui.pathTracerPerfStats.totalFrameP95Ms,
				    16.67f);
				writePathTracerBacklogCsv();
				if (!ptBacklogItems.empty())
				{
					analysis.backlogSummary = ptBacklogItems[0].name;
					if (ptBacklogItems.size() > 1)
					{
						analysis.backlogSummary += "; ";
						analysis.backlogSummary += ptBacklogItems[1].name;
					}
				}
			}
		}
	}
}

void EngineCore::resetPathTracerAnalysisCounters(uint32_t frameSlot)
{
	if (frameSlot >= frames.ptAnalysisCounterMapped.size() || !frames.ptAnalysisCounterMapped[frameSlot])
	{
		return;
	}
	std::memset(frames.ptAnalysisCounterMapped[frameSlot], 0, sizeof(Laphria::PathTracerAnalysisCounters));
}

void EngineCore::applyPathTracerExperimentRow(const PathTracerExperimentRow &row)
{
	auto &settings                         = ui.pathTracerSettings;
	auto &analysis                         = ui.pathTracerAnalysisSettings;
	settings.enableEnvironmentNEE          = true;
	settings.environmentNeeBounceMode      = row.environmentNeeBounceMode;
	settings.blackEnvironment              = row.blackEnvironment;
	settings.applyFirstHitProbesToFinal    = true;
	settings.environmentNeeSamplingMode    = UISystem::EnvironmentNeeSamplingMode::SkyBiased;
	settings.firstHitProbeSamplingMode     = row.probeMode;
	settings.firstHitDiffuseSamples        = row.firstHitDiffuseSamples;
	settings.firstHitCandidateCount        = row.firstHitCandidateCount;
	settings.pathTracerMaxBounces          = row.pathTracerMaxBounces;
	settings.directSunBounceMode           = row.directSunBounceMode;
	ptBenchmarkBasePosition                = row.cameraPosition;
	ptBenchmarkBasePitch                   = row.cameraPitch;
	ptBenchmarkBaseYaw                     = row.cameraYaw;
	camera.position                        = row.cameraPosition;
	camera.pitch                           = row.cameraPitch;
	camera.yaw                             = row.cameraYaw;
	camera.processInput(0.0f, 0.0f, 0.0f);
	ptPrevCameraPos               = camera.position;
	ptPrevPitch                   = camera.pitch;
	ptPrevYaw                     = camera.yaw;
	ui.lightDirection             = glm::normalize(row.lightDirection);
	ptForceHistoryReset           = true;
	analysis.debugAov             = row.debugAov;
	analysis.debugAtrousIteration = 0;
	analysis.enableAnalysisMode   = true;
}
void EngineCore::logPathTracerExperimentRow(const PathTracerExperimentRow         &row,
                                            const PathTracerExperimentAccumulator &accum) const
{
	const double invSamples = (accum.sampleCount > 0) ? (1.0 / static_cast<double>(accum.sampleCount)) : 0.0;
	LOGI("PT Experiment Row Summary: name=\"%s\", samples=%u, mode=%d, "
	     "maxBounces=%d, directSunMode=%d, environmentNeeMode=%d, "
	     "rayTraceMs=%.3f, totalMs=%.3f",
	     row.name.c_str(),
	     accum.sampleCount,
	     static_cast<int>(row.probeMode),
	     row.pathTracerMaxBounces,
	     row.directSunBounceMode,
	     row.environmentNeeBounceMode,
	     accum.rayTraceMs * invSamples,
	     accum.totalFrameMs * invSamples);
}
void EngineCore::updatePathTracerExperimentSweep()
{
	if (!ptExperimentSweepActive)
	{
		return;
	}

	if (ptExperimentRowIndex >= ptExperimentRows.size())
	{
		ptExperimentSweepActive = false;
		return;
	}

	if (ptExperimentWarmupRemaining > 0)
	{
		--ptExperimentWarmupRemaining;
		return;
	}

	const auto &stats = ui.pathTracerPerfStats;
	ptExperimentAccum.rayTraceMs += stats.rayTraceMs;
	ptExperimentAccum.totalFrameMs += stats.totalFrameMs;
	++ptExperimentAccum.sampleCount;

	if (ptExperimentSampleRemaining > 0)
	{
		--ptExperimentSampleRemaining;
	}
	if (ptExperimentSampleRemaining > 0)
	{
		return;
	}

	logPathTracerExperimentRow(ptExperimentRows[ptExperimentRowIndex], ptExperimentAccum);
	++ptExperimentRowIndex;
	if (ptExperimentRowIndex >= ptExperimentRows.size())
	{
		ptExperimentSweepActive = false;
		LOGI("%s", ptExperimentCompletionLog.c_str());
		return;
	}

	ptExperimentWarmupRemaining = std::max(1, ptExperimentWarmupFrames);
	ptExperimentSampleRemaining = std::max(1, ptExperimentSampleFrames);
	ptExperimentAccum           = {};
	applyPathTracerExperimentRow(ptExperimentRows[ptExperimentRowIndex]);
}
void EngineCore::ensurePathTracerSanityScene()
{
	if (ptSanitySceneCreated || !scene || !resourceManager)
	{
		return;
	}

	Laphria::MaterialData whiteDiffuse{};
	whiteDiffuse.baseColorFactor = glm::vec4(0.95f, 0.95f, 0.95f, 1.0f);
	whiteDiffuse.metallicFactor  = 0.0f;
	whiteDiffuse.roughnessFactor = 0.75f;
	whiteDiffuse.emissiveFactor  = glm::vec3(0.0f);

	Laphria::MaterialData roughMetal{};
	roughMetal.baseColorFactor = glm::vec4(0.90f, 0.90f, 0.92f, 1.0f);
	roughMetal.metallicFactor  = 1.0f;
	roughMetal.roughnessFactor = 0.80f;
	roughMetal.emissiveFactor  = glm::vec3(0.0f);

	Laphria::MaterialData emissivePatch{};
	emissivePatch.baseColorFactor = glm::vec4(0.25f, 0.25f, 0.25f, 1.0f);
	emissivePatch.metallicFactor  = 0.0f;
	emissivePatch.roughnessFactor = 0.60f;
	emissivePatch.emissiveFactor  = glm::vec3(6.0f, 5.2f, 4.8f);

	ptSanityWhiteDiffuseNode = resourceManager->createCubeModel(1.0f, *pipelines.descriptorSetLayoutMaterial, whiteDiffuse);
	ptSanityRoughMetalNode   = resourceManager->createCubeModel(1.0f, *pipelines.descriptorSetLayoutMaterial, roughMetal);
	ptSanityEmissiveNode     = resourceManager->createCubeModel(0.7f, *pipelines.descriptorSetLayoutMaterial, emissivePatch);

	if (ptSanityWhiteDiffuseNode)
	{
		ptSanityWhiteDiffuseNode->name = "PT_Sanity_WhiteDiffuse";
		ptSanityWhiteDiffuseNode->setPosition(ptBenchmarkBasePosition + glm::vec3(-1.4f, -0.2f, -2.5f));
		scene->addNode(ptSanityWhiteDiffuseNode, scene->getRoot());
	}
	if (ptSanityRoughMetalNode)
	{
		ptSanityRoughMetalNode->name = "PT_Sanity_RoughMetal";
		ptSanityRoughMetalNode->setPosition(ptBenchmarkBasePosition + glm::vec3(0.0f, -0.2f, -2.5f));
		scene->addNode(ptSanityRoughMetalNode, scene->getRoot());
	}
	if (ptSanityEmissiveNode)
	{
		ptSanityEmissiveNode->name = "PT_Sanity_Emissive";
		ptSanityEmissiveNode->setPosition(ptBenchmarkBasePosition + glm::vec3(1.4f, -0.2f, -2.5f));
		scene->addNode(ptSanityEmissiveNode, scene->getRoot());
	}

	ptSanitySceneCreated = true;
}

void EngineCore::updatePathTracerPhysicalSanityChecks(float /*deltaTimeSeconds*/)
{
	auto &analysis = ui.pathTracerAnalysisSettings;
	if (!(analysis.enableAnalysisMode && analysis.runPhysicalSanityChecks))
	{
		analysis.physicalSanityActive = false;
		return;
	}

	if (!analysis.physicalSanityActive && ptSanityPhase >= 2)
	{
		ptSanityPhase                      = 0;
		ptSanityFramesRemaining            = 0;
		ptSanityBaselineCaptured           = false;
		ptSanityDriftMetric                = 0.0f;
		analysis.physicalSanityDriftMetric = 0.0f;
		analysis.physicalSanityPassed      = false;
	}

	ensurePathTracerSanityScene();
	analysis.physicalSanityActive = true;

	// Use two controlled exposure phases and compare temporal stability counters.
	if (!ptSanityBaselineCaptured && ptSanityPhase == 0)
	{
		ui.exposure = ptSanityBaselineExposure;
		if (ptSanityFramesRemaining <= 0)
		{
			ptSanityFramesRemaining = 90;
		}
		--ptSanityFramesRemaining;
		if (ptSanityFramesRemaining <= 0)
		{
			ptSanityBaselineRejectRatio  = ui.pathTracerPerfStats.historyRejectionRatio;
			ptSanityBaselineFireflyRatio = ui.pathTracerPerfStats.fireflyClampRatio;
			ptSanityBaselineSkyRatio     = ui.pathTracerPerfStats.skyHitRatio;
			ptSanityBaselineCaptured     = true;
			ptSanityPhase                = 1;
			ptSanityFramesRemaining      = 90;
		}
		return;
	}

	if (ptSanityPhase == 1)
	{
		ui.exposure = 2.0f;
		--ptSanityFramesRemaining;
		if (ptSanityFramesRemaining <= 0)
		{
			const float rejectDrift            = std::abs(ui.pathTracerPerfStats.historyRejectionRatio - ptSanityBaselineRejectRatio);
			const float fireflyDrift           = std::abs(ui.pathTracerPerfStats.fireflyClampRatio - ptSanityBaselineFireflyRatio);
			const float skyDrift               = std::abs(ui.pathTracerPerfStats.skyHitRatio - ptSanityBaselineSkyRatio);
			ptSanityDriftMetric                = rejectDrift + fireflyDrift + 0.5f * skyDrift;
			analysis.physicalSanityDriftMetric = ptSanityDriftMetric;
			analysis.physicalSanityPassed      = (ptSanityDriftMetric < 0.20f);
			analysis.physicalSanityActive      = false;
			analysis.runPhysicalSanityChecks   = false;
			ptSanityPhase                      = 2;
			ui.exposure                        = 1.0f;
		}
	}
}

void EngineCore::writePathTracerBacklogCsv()
{
	const auto backlogCsvPath                          = resolveAnalysisOutputPath(kPtBacklogCsvFileName);
	ui.pathTracerAnalysisSettings.backlogCsvOutputPath = backlogCsvPath.string();
	std::ofstream out(backlogCsvPath, std::ios::trunc);
	if (!out.is_open())
	{
		LOGE("Failed to open backlog CSV for writing: %s", backlogCsvPath.string().c_str());
		return;
	}

	out << "priority,name,expected_impact,estimated_ms,measured_ms,budget_pass\n";
	for (const auto &item : ptBacklogItems)
	{
		const char *priority = "Medium";
		if (item.priority == Laphria::PathTracerBacklogPriority::High)
		{
			priority = "High";
		}
		else if (item.priority == Laphria::PathTracerBacklogPriority::Low)
		{
			priority = "Low";
		}
		out << priority << ','
		    << '"' << item.name << '"' << ','
		    << '"' << item.expectedArtifactImpact << '"' << ','
		    << item.estimatedMsCost << ','
		    << item.measuredMsCost << ','
		    << (item.budgetPass ? 1 : 0) << '\n';
	}
}

void EngineCore::loadPathTracerBenchmarkSceneIfNeeded()
{
	auto &analysis = ui.pathTracerAnalysisSettings;
	if (!(analysis.enableAnalysisMode && analysis.lockBenchmarkScene && scene && resourceManager))
	{
		return;
	}
	if (ptBenchmarkSceneLoaded)
	{
		return;
	}
	namespace fs                             = std::filesystem;
	const fs::path                directPath = fs::path("Assets") / "sponza_runtime.glb";
	const std::array<fs::path, 4> candidates = {
	    directPath,
	    fs::path("..") / directPath,
	    fs::path("..") / fs::path("..") / directPath,
	    fs::path("..") / fs::path("..") / fs::path("..") / directPath};

	std::string resolvedPath = directPath.generic_string();
	for (const auto &candidate : candidates)
	{
		std::error_code ec;
		if (fs::exists(candidate, ec) && !ec)
		{
			resolvedPath = candidate.generic_string();
			break;
		}
	}

	scene->loadModel(resolvedPath, *resourceManager, *pipelines.descriptorSetLayoutMaterial, scene->getRoot());
	auto &pathTracerSettings                 = ui.pathTracerSettings;
	pathTracerSettings.directSunBounceMode   = 1;
	ptBenchmarkSceneLoaded = true;
}

void EngineCore::updatePathTracerBenchmark(float deltaTimeSeconds)
{
	auto &analysis = ui.pathTracerAnalysisSettings;
	if (!(analysis.enableAnalysisMode && analysis.benchmarkActive))
	{
		return;
	}

	if (analysis.runBaselineSweep)
	{
		if (ptSweepConfigs.empty())
		{
			ptSweepConfigs         = buildPathTracerBaselineSweepMatrix();
			ptSweepConfigIndex     = 0;
			ptSweepWarmupRemaining = analysis.warmupFrames;
			ptSweepSampleRemaining = analysis.sampleFrames;
			ptSweepScores.clear();
			ptBacklogItems.clear();
			ptSampleTotalMs.clear();
			ptSampleRayTraceMs.clear();
			ptSampleDenoiserMs.clear();
			analysis.recommendationManual.clear();
			analysis.recommendationAutoBalanced.clear();
			analysis.recommendationAutoAggressive.clear();
			analysis.backlogSummary.clear();
			const auto benchmarkCsvPath                          = resolveAnalysisOutputPath(kPtBenchmarkCsvFileName);
			ui.pathTracerAnalysisSettings.benchmarkCsvOutputPath = benchmarkCsvPath.string();
			std::ofstream out(benchmarkCsvPath, std::ios::trunc);
			if (out.is_open())
			{
				out << "config_index,resolution_scale,denoiser_iterations,reprojection,motion_aware,reduce_secondary,"
				       "total_p50_ms,total_p95_ms,total_p99_ms,ray_p95_ms,denoiser_p95_ms,"
				       "history_accept_ratio,history_reject_ratio,sky_hit_ratio,firefly_clamp_ratio,"
				       "budget_pass,composite_score\n";
			}
			else
			{
				LOGE("Failed to open benchmark CSV for writing: %s", benchmarkCsvPath.string().c_str());
			}
		}
		if (ptSweepConfigIndex < ptSweepConfigs.size())
		{
			const auto &cfg                                     = ptSweepConfigs[ptSweepConfigIndex];
			ui.pathTracerSettings.resolutionScale               = cfg.resolutionScale;
			ui.pathTracerSettings.denoiserIterations            = cfg.denoiserIterations;
			ui.pathTracerSettings.enableReprojection            = cfg.enableReprojection;
			ui.pathTracerSettings.enableMotionAwareAccumulation = cfg.enableMotionAwareAccumulation;
			ui.pathTracerSettings.reduceSecondaryEffects        = cfg.reduceSecondaryEffects;
		}
	}

	if (analysis.freezeCameraInputDuringBenchmark)
	{
		camera.processInput(0.0f, 0.0f, 0.0f);
	}

	ptBenchmarkClockSeconds += std::max(0.0f, deltaTimeSeconds);
	ptBenchmarkTeleportClockSeconds += std::max(0.0f, deltaTimeSeconds);

	switch (analysis.cameraPath)
	{
		case UISystem::PathTracerBenchmarkCameraPath::Static:
			camera.position = ptBenchmarkBasePosition;
			camera.pitch    = ptBenchmarkBasePitch;
			camera.yaw      = ptBenchmarkBaseYaw;
			break;
		case UISystem::PathTracerBenchmarkCameraPath::SlowPan:
			camera.position = ptBenchmarkBasePosition;
			camera.pitch    = ptBenchmarkBasePitch;
			camera.yaw      = ptBenchmarkBaseYaw + glm::radians(8.0f) * ptBenchmarkClockSeconds;
			break;
		case UISystem::PathTracerBenchmarkCameraPath::FastPan:
			camera.position = ptBenchmarkBasePosition;
			camera.pitch    = ptBenchmarkBasePitch;
			camera.yaw      = ptBenchmarkBaseYaw + glm::radians(36.0f) * ptBenchmarkClockSeconds;
			break;
		case UISystem::PathTracerBenchmarkCameraPath::Teleport:
			if (ptBenchmarkTeleportClockSeconds >= 1.5f)
			{
				ptBenchmarkTeleportClockSeconds = 0.0f;
			}
			if (ptBenchmarkTeleportClockSeconds < 0.75f)
			{
				camera.position = ptBenchmarkBasePosition;
				camera.pitch    = ptBenchmarkBasePitch;
				camera.yaw      = ptBenchmarkBaseYaw;
			}
			else
			{
				camera.position = ptBenchmarkBasePosition + glm::vec3(3.5f, 0.4f, -2.5f);
				camera.pitch    = ptBenchmarkBasePitch + glm::radians(6.0f);
				camera.yaw      = ptBenchmarkBaseYaw + glm::radians(45.0f);
			}
			break;
	}
}

void EngineCore::updateAdaptivePathTracerSettings()
{
	if (ui.pathTracerAnalysisSettings.enableAnalysisMode && ui.pathTracerAnalysisSettings.benchmarkActive)
	{
		return;
	}
	if (ui.pathTracerSettings.qualityMode == UISystem::PathTracerQualityMode::Manual)
	{
		return;
	}

	const float frameMs = ui.pathTracerPerfStats.totalFrameMs;
	if (frameMs <= 0.0f)
	{
		return;
	}

	const bool  aggressive  = (ui.pathTracerSettings.qualityMode == UISystem::PathTracerQualityMode::AutoAggressive);
	const float targetMs    = std::max(8.0f, ui.pathTracerSettings.targetFrameMs);
	const float dropMargin  = aggressive ? 0.25f : 0.75f;
	const float raiseMargin = aggressive ? 2.5f : 1.5f;

	if (frameMs > targetMs + dropMargin)
	{
		if (ui.pathTracerSettings.denoiserIterations > 1)
		{
			ui.pathTracerSettings.denoiserIterations -= 1;
		}
		else
		{
			ui.pathTracerSettings.resolutionScale = std::max(0.50f, ui.pathTracerSettings.resolutionScale - 0.05f);
		}
		return;
	}

	if (frameMs < targetMs - raiseMargin)
	{
		if (ui.pathTracerSettings.resolutionScale < 1.0f)
		{
			ui.pathTracerSettings.resolutionScale = std::min(1.0f, ui.pathTracerSettings.resolutionScale + 0.05f);
		}
		else if (ui.pathTracerSettings.denoiserIterations < 5)
		{
			ui.pathTracerSettings.denoiserIterations += 1;
		}
	}
}

void EngineCore::recordCommandBuffer(uint32_t imageIndex) const
{
	auto          &commandBuffer = frames.commandBuffers[frames.frameIndex];
	const uint32_t queryBase     = getPathTracerQueryBase(frames.frameIndex);
	if (*ptTimestampQueryPool)
	{
		commandBuffer.resetQueryPool(*ptTimestampQueryPool, queryBase, kPtTimestampQueryCountPerFrame);
	}

	vk::ClearValue clearColor = vk::ClearColorValue(0.02f, 0.02f, 0.02f, 1.0f);
	if (ui.renderMode == RenderMode::Rasterizer)
	{
		// V1.3: raster path uses direct atmospheric clear color (no compute sky prepass).
		clearColor = vk::ClearColorValue(0.60f, 0.64f, 0.72f, 1.0f);
	}

	recordSkinningPass(commandBuffer);

	// --- Build TLAS ---
	if (ui.renderMode != RenderMode::Rasterizer)
	{
		std::vector<vk::AccelerationStructureInstanceKHR> tlasInstances;
		for (const auto &node : scene->getAllNodes())
		{
			if (node->modelId < 0)
			{
				continue;
			}
			if (node->modelId >= static_cast<int>(Laphria::EngineConfig::kBindlessModelCapacity))
			{
				throw std::runtime_error(
				    "TLAS custom-index modelId " + std::to_string(node->modelId) +
				    " exceeds bindless RT descriptor capacity " +
				    std::to_string(Laphria::EngineConfig::kBindlessModelCapacity));
			}

			ModelResource *modelRes = resourceManager->getModelResource(node->modelId);
			if (!modelRes || modelRes->blasElements.empty())
			{
				continue;
			}

			glm::mat4 transform = node->getWorldTransform();

			// Convert to vk::TransformMatrixKHR (3x4 row-major array)
			vk::TransformMatrixKHR transformMatrix;
			for (int r = 0; r < 3; ++r)
			{
				for (int c = 0; c < 4; ++c)
				{
					transformMatrix.matrix[r][c] = transform[c][r];        // GLM is column-major
				}
			}

			for (int meshIdx : node->getMeshIndices())
			{
				if (meshIdx < 0 || meshIdx >= modelRes->blasElements.size())
				{
					continue;
				}

				auto &blas = modelRes->blasElements[meshIdx];

				uint32_t primitiveOffset = 0;
				for (int i = 0; i < meshIdx; ++i)
				{
					primitiveOffset += modelRes->meshes[i].primitives.size();
				}

				// Encode modelId in top bits, primitiveOffset in bottom 14 bits
				// InstanceCustomIndex is exactly 24 bit in size.
				uint32_t customIndex = (node->modelId << 14) | (primitiveOffset & 0x3FFF);

				vk::AccelerationStructureDeviceAddressInfoKHR addressInfo{};
				addressInfo.accelerationStructure = *blas;
				vk::DeviceAddress blasAddress     = vulkan.logicalDevice.getAccelerationStructureAddressKHR(addressInfo);

				vk::AccelerationStructureInstanceKHR instance{};
				instance.transform                              = transformMatrix;
				instance.instanceCustomIndex                    = customIndex;
				instance.mask                                   = 0xFF;        // All rays hit
				instance.instanceShaderBindingTableRecordOffset = 0;
				instance.flags                                  = static_cast<uint32_t>(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);
				instance.accelerationStructureReference         = blasAddress;

				tlasInstances.push_back(instance);
			}
		}

		if (tlasInstances.size() > frames.MAX_TLAS_INSTANCES)
		{
			throw std::runtime_error(
			    "TLAS instance count (" + std::to_string(tlasInstances.size()) +
			    ") exceeds MAX_TLAS_INSTANCES (" + std::to_string(frames.MAX_TLAS_INSTANCES) + ")");
		}

		// Only copy instance data when there is something to copy; building with
		// primitiveCount = 0 is valid and produces a traversable empty TLAS.
		if (!tlasInstances.empty())
		{
			size_t dataSize = tlasInstances.size() * sizeof(vk::AccelerationStructureInstanceKHR);
			memcpy(frames.tlasInstanceBuffersMapped[frames.frameIndex], tlasInstances.data(), dataSize);
		}

		// Memory barrier to ensure host writes to the instance buffer are visible to the AS builder
		vk::MemoryBarrier2 hostToDeviceBarrier{
		    .srcStageMask  = vk::PipelineStageFlagBits2::eHost,
		    .srcAccessMask = vk::AccessFlagBits2::eHostWrite,
		    .dstStageMask  = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
		    .dstAccessMask = vk::AccessFlagBits2::eAccelerationStructureReadKHR};

		vk::DependencyInfo dependencyInfo{
		    .memoryBarrierCount = 1,
		    .pMemoryBarriers    = &hostToDeviceBarrier};
		commandBuffer.pipelineBarrier2(dependencyInfo);

		// Build TLAS — always, even when the scene is empty (primitiveCount = 0 is valid).
		vk::AccelerationStructureGeometryInstancesDataKHR instancesData{};
		instancesData.arrayOfPointers    = vk::False;
		instancesData.data.deviceAddress = frames.tlasInstanceAddresses[frames.frameIndex];

		vk::AccelerationStructureGeometryKHR tlasGeometry{};
		tlasGeometry.geometryType       = vk::GeometryTypeKHR::eInstances;
		tlasGeometry.geometry.instances = instancesData;

		vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{};
		buildInfo.type                      = vk::AccelerationStructureTypeKHR::eTopLevel;
		buildInfo.flags                     = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
		buildInfo.mode                      = vk::BuildAccelerationStructureModeKHR::eBuild;
		buildInfo.geometryCount             = 1;
		buildInfo.pGeometries               = &tlasGeometry;
		buildInfo.dstAccelerationStructure  = *frames.tlas[frames.frameIndex];
		buildInfo.scratchData.deviceAddress = frames.tlasScratchAddresses[frames.frameIndex];

		vk::AccelerationStructureBuildRangeInfoKHR buildRange{};
		buildRange.primitiveCount  = static_cast<uint32_t>(tlasInstances.size());
		buildRange.primitiveOffset = 0;
		buildRange.firstVertex     = 0;
		buildRange.transformOffset = 0;

		const vk::AccelerationStructureBuildRangeInfoKHR *pBuildRange = &buildRange;
		if (*ptTimestampQueryPool)
		{
			commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR, *ptTimestampQueryPool, queryBase + kPtTS_TlasStart);
		}
		commandBuffer.buildAccelerationStructuresKHR(buildInfo, pBuildRange);
		if (*ptTimestampQueryPool)
		{
			commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR, *ptTimestampQueryPool, queryBase + kPtTS_TlasEnd);
		}

		// Memory barrier to ensure TLAS build finishes before the ray tracing shader reads it
		vk::MemoryBarrier2 asBuildToRayTracingBarrier{
		    .srcStageMask  = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
		    .srcAccessMask = vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
		    .dstStageMask  = vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
		    .dstAccessMask = vk::AccessFlagBits2::eAccelerationStructureReadKHR};

		vk::DependencyInfo asDependencyInfo{
		    .memoryBarrierCount = 1,
		    .pMemoryBarriers    = &asBuildToRayTracingBarrier};
		commandBuffer.pipelineBarrier2(asDependencyInfo);
	}
	// --- End TLAS Build ---

	// ── Cascaded Shadow Map Pass ──────────────────────────────────────────────
	// Only run for the raster path; both RT pipelines handle their own shadowing.
	if (ui.renderMode == RenderMode::Rasterizer)
	{
		vk::Image shadowImg = *frames.shadowImages[frames.frameIndex];

		// Transition all 4 cascade layers: eUndefined → eDepthAttachmentOptimal.
		// We always use eUndefined as the old layout so the previous frame's contents
		// are discarded — the depth buffer is cleared at the start of each cascade render.
		vk::ImageMemoryBarrier2 shadowToWrite{
		    .srcStageMask        = vk::PipelineStageFlagBits2::eTopOfPipe,
		    .srcAccessMask       = {},
		    .dstStageMask        = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
		    .dstAccessMask       = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		    .oldLayout           = vk::ImageLayout::eUndefined,
		    .newLayout           = vk::ImageLayout::eDepthAttachmentOptimal,
		    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .image               = shadowImg,
		    .subresourceRange    = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, NUM_SHADOW_CASCADES}};
		vk::DependencyInfo shadowWriteDep{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &shadowToWrite};
		commandBuffer.pipelineBarrier2(shadowWriteDep);

		// Render each cascade into its own layer of the shadow array image.
		commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines.shadowPipeline);
		commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
		                                 *pipelines.shadowPipelineLayout, 0,
		                                 *descriptorSets[frames.frameIndex], nullptr);

		vk::Viewport shadowViewport{
		    0.0f, 0.0f,
		    static_cast<float>(SHADOW_MAP_DIM), static_cast<float>(SHADOW_MAP_DIM),
		    0.0f, 1.0f};
		vk::Rect2D shadowScissor{{0, 0}, {SHADOW_MAP_DIM, SHADOW_MAP_DIM}};

		for (uint32_t cascadeIdx = 0; cascadeIdx < NUM_SHADOW_CASCADES; cascadeIdx++)
		{
			uint32_t viewIdx = frames.frameIndex * NUM_SHADOW_CASCADES + cascadeIdx;

			vk::RenderingAttachmentInfo cascadeDepthAttachment{
			    .imageView   = *frames.shadowCascadeViews[viewIdx],
			    .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
			    .loadOp      = vk::AttachmentLoadOp::eClear,
			    .storeOp     = vk::AttachmentStoreOp::eStore,
			    .clearValue  = vk::ClearDepthStencilValue{1.0f, 0}};

			vk::RenderingInfo cascadeRenderingInfo{
			    .renderArea           = {{0, 0}, {SHADOW_MAP_DIM, SHADOW_MAP_DIM}},
			    .layerCount           = 1,
			    .colorAttachmentCount = 0,
			    .pDepthAttachment     = &cascadeDepthAttachment};

			commandBuffer.beginRendering(cascadeRenderingInfo);
			commandBuffer.setViewport(0, shadowViewport);
			commandBuffer.setScissor(0, shadowScissor);

			// Draw all scene nodes into this cascade.
			for (const auto &node : scene->getAllNodes())
			{
				if (node->modelId < 0)
					continue;
				auto *modelRes = resourceManager->getModelResource(node->modelId);
				if (!modelRes)
					continue;

				resourceManager->bindResources(commandBuffer, node->modelId, modelRes->hasRuntimeSkinning);
				glm::mat4 worldTransform = node->getWorldTransform();

				if (*modelRes->descriptorSet)
				{
					commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelines.shadowPipelineLayout, 1, {*modelRes->descriptorSet}, nullptr);
				}

				for (int meshIdx : node->getMeshIndices())
				{
					if (meshIdx < 0 || meshIdx >= static_cast<int>(modelRes->meshes.size()))
						continue;
					for (const auto &prim : modelRes->meshes[meshIdx].primitives)
					{
						Laphria::ScenePushConstants pc{};
						pc.modelMatrix   = worldTransform;
						pc.cascadeIndex  = static_cast<int>(cascadeIdx);
						pc.materialIndex = prim.flatPrimitiveIndex;
						commandBuffer.pushConstants<Laphria::ScenePushConstants>(
						    *pipelines.shadowPipelineLayout,
						    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
						    0, pc);
						commandBuffer.drawIndexed(prim.indexCount, 1, prim.firstIndex, prim.vertexOffset, 0);
					}
				}
			}

			commandBuffer.endRendering();
		}

		// Transition shadow image: eDepthAttachmentOptimal → eShaderReadOnlyOptimal
		// so the main fragment shader can sample it.
		vk::ImageMemoryBarrier2 shadowToRead{
		    .srcStageMask        = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
		    .srcAccessMask       = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		    .dstStageMask        = vk::PipelineStageFlagBits2::eFragmentShader,
		    .dstAccessMask       = vk::AccessFlagBits2::eShaderRead,
		    .oldLayout           = vk::ImageLayout::eDepthAttachmentOptimal,
		    .newLayout           = vk::ImageLayout::eShaderReadOnlyOptimal,
		    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .image               = shadowImg,
		    .subresourceRange    = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, NUM_SHADOW_CASCADES}};
		vk::DependencyInfo shadowReadDep{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &shadowToRead};
		commandBuffer.pipelineBarrier2(shadowReadDep);
		// V1.3: remove compute sky from raster path; render directly into a cleared color target.

		transition_image_layout(
		    swapchain.images[imageIndex],
		    vk::ImageLayout::eUndefined,
		    vk::ImageLayout::eColorAttachmentOptimal,
		    {},
		    vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentRead,
		    vk::PipelineStageFlagBits2::eTopOfPipe,
		    vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		    vk::ImageAspectFlagBits::eColor);
	}

	if (ui.renderMode == RenderMode::PathTracer)
	{
		recordRayTracingCommandBuffer(commandBuffer, imageIndex);
	}
	else if (ui.renderMode == RenderMode::RayTracer)
	{
		recordClassicRTCommandBuffer(commandBuffer, imageIndex);
	}
	else if (ui.renderMode == RenderMode::SurfelPathTracer)
	{
		recordSurfelPathTracerCommandBuffer(commandBuffer, imageIndex);
	}

	transition_image_layout(
	    *frames.depthImages[imageIndex],
	    vk::ImageLayout::eUndefined,
	    vk::ImageLayout::eDepthAttachmentOptimal,
	    {},
	    vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
	    vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
	    vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
	    vk::ImageAspectFlagBits::eDepth);

	vk::RenderingAttachmentInfo attachmentInfo = {
	    .imageView   = *swapchain.imageViews[imageIndex],
	    .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
	    .loadOp      = (ui.renderMode == RenderMode::Rasterizer) ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
	    .storeOp     = vk::AttachmentStoreOp::eStore,
	    .clearValue  = clearColor};

	vk::RenderingAttachmentInfo depthAttachmentInfo{
	    .imageView   = *frames.depthImageViews[imageIndex],
	    .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
	    .loadOp      = vk::AttachmentLoadOp::eClear,
	    .storeOp     = vk::AttachmentStoreOp::eStore,
	    .clearValue  = vk::ClearDepthStencilValue{1.0f, 0}};

	vk::RenderingInfo renderingInfo = {
	    .renderArea           = {.offset = {0, 0}, .extent = swapchain.extent},
	    .layerCount           = 1,
	    .colorAttachmentCount = 1,
	    .pColorAttachments    = &attachmentInfo,
	    .pDepthAttachment     = &depthAttachmentInfo};

	commandBuffer.beginRendering(renderingInfo);

	if (ui.renderMode == RenderMode::Rasterizer)
	{
		commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines.graphicsPipeline);

		// Y starts at height and height is negative: this flips the Vulkan NDC Y-axis so that
		// +Y points up in clip space, matching GLM's convention (which was designed for OpenGL).
		vk::Viewport viewport{
		    0.0f, static_cast<float>(swapchain.extent.height),
		    static_cast<float>(swapchain.extent.width),
		    -static_cast<float>(swapchain.extent.height), 0.0f, 1.0f};
		commandBuffer.setViewport(0, viewport);
		commandBuffer.setScissor(0, vk::Rect2D({0, 0}, swapchain.extent));

		// Global UBO Binding (Set 0)
		commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelines.graphicsPipelineLayout, 0,
		                                 *descriptorSets[frames.frameIndex], nullptr);

		const float     aspectRatio = static_cast<float>(swapchain.extent.width) / static_cast<float>(swapchain.extent.height);
		const glm::mat4 view        = camera.getViewMatrix();
		const glm::mat4 proj        = glm::perspective(
            glm::radians(Laphria::EngineConfig::kMainCameraFovDegrees),
            aspectRatio,
            Laphria::EngineConfig::kMainCameraNearPlane,
            Laphria::EngineConfig::kMainCameraFarPlane);
		const glm::mat4 viewProjection    = proj * view;
		const glm::mat4 invViewProjection = glm::inverse(viewProjection);

		const Laphria::Frustum frustum    = Laphria::Frustum::fromViewProjection(viewProjection);
		Laphria::AABB          cullBounds = Laphria::Frustum::computeAABB(invViewProjection);
		// Expand query bounds so close-up objects whose origins are just outside
		// the near plane are still submitted in raster mode.
		constexpr float kRasterCullMargin = 2.0f;
		cullBounds.min -= glm::vec3(kRasterCullMargin);
		cullBounds.max += glm::vec3(kRasterCullMargin);
		scene->draw(commandBuffer, pipelines.graphicsPipelineLayout, *resourceManager, cullBounds, frustum);
	}

	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *commandBuffer);

	commandBuffer.endRendering();

	// Transition SwapChain to Present Layout
	transition_image_layout(
	    swapchain.images[imageIndex],
	    vk::ImageLayout::eColorAttachmentOptimal,
	    vk::ImageLayout::ePresentSrcKHR,
	    vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentRead,
	    {},
	    vk::PipelineStageFlagBits2::eColorAttachmentOutput,
	    vk::PipelineStageFlagBits2::eBottomOfPipe,
	    vk::ImageAspectFlagBits::eColor);
}

// Inline Synchronization2 image barrier recorded into the current frame's command buffer.
// Unlike VulkanUtils::recordImageLayoutTransition (which uses Vulkan 1.0 pipelineBarrier),
// this version accepts explicit stage/access masks for fine-grained GPU dependency control.
void EngineCore::transition_image_layout(
    vk::Image               image,
    vk::ImageLayout         old_layout,
    vk::ImageLayout         new_layout,
    vk::AccessFlags2        src_access_mask,
    vk::AccessFlags2        dst_access_mask,
    vk::PipelineStageFlags2 src_stage_mask,
    vk::PipelineStageFlags2 dst_stage_mask,
    vk::ImageAspectFlags    image_aspect_flags) const
{
	vk::ImageMemoryBarrier2 barrier = {
	    .srcStageMask        = src_stage_mask,
	    .srcAccessMask       = src_access_mask,
	    .dstStageMask        = dst_stage_mask,
	    .dstAccessMask       = dst_access_mask,
	    .oldLayout           = old_layout,
	    .newLayout           = new_layout,
	    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .image               = image,
	    .subresourceRange    = {
	           .aspectMask     = image_aspect_flags,
	           .baseMipLevel   = 0,
	           .levelCount     = 1,
	           .baseArrayLayer = 0,
	           .layerCount     = 1}};
	vk::DependencyInfo dependency_info = {
	    .dependencyFlags         = {},
	    .imageMemoryBarrierCount = 1,
	    .pImageMemoryBarriers    = &barrier};
	frames.commandBuffers[frames.frameIndex].pipelineBarrier2(dependency_info);
}

void EngineCore::drawFrame()
{
	if (!renderModeInitialized)
	{
		lastSubmittedRenderMode = ui.renderMode;
		renderModeInitialized   = true;
	}
	else if (ui.renderMode != lastSubmittedRenderMode)
	{
		// Renderer switches can otherwise overlap in-flight GPU work that uses different
		// pipeline/resource access patterns (especially PT denoiser scratch buffers).
		vulkan.logicalDevice.waitIdle();
		if (ui.renderMode == RenderMode::SurfelPathTracer ||
		    lastSubmittedRenderMode == RenderMode::SurfelPathTracer)
		{
			// Task 7's prepare pass consumes this to reset persistent surfel state.
			ui.surfelPathTracerSettings.resetSurfels = true;
		}
		ptForceHistoryReset     = true;
		lastSubmittedRenderMode = ui.renderMode;
	}

	// Note: inFlightFences, presentCompleteSemaphores, and commandBuffers are indexed by frameIndex,
	//       while renderFinishedSemaphores is indexed by imageIndex
	auto fenceResult = waitForFenceOrThrow(
	    vulkan.logicalDevice, *frames.inFlightFences[frames.frameIndex], "frame in-flight fence");
	if (fenceResult != vk::Result::eSuccess)
	{
		throw std::runtime_error("failed to wait for fence!");
	}

	auto refreshSurfelPathTracerRuntimeResources = [&]() {
		if (!surfelPathTracerResources.initialized())
		{
			return;
		}

		// Extends the earlier refreshSurfelPathTracerPersistentResources path to cover atlas images.
		auto waitForSurfelPathTracerIdle = [&](const char *waitLabel) {
			for (const auto &fence : frames.inFlightFences)
			{
				const auto surfelFenceResult = waitForFenceOrThrow(
				    vulkan.logicalDevice, *fence, waitLabel);
				if (surfelFenceResult != vk::Result::eSuccess)
				{
					throw std::runtime_error("failed to wait for surfel path tracer in-flight work");
				}
			}
		};

		const auto nextStaticSettings =
		    makeSurfelPathTracerStaticSettingsSnapshot(ui.surfelPathTracerSettings);
		if (!surfelPathTracerStaticSettingsInitialized)
		{
			surfelPathTracerStaticSettings = nextStaticSettings;
			surfelPathTracerStaticSettingsInitialized = true;
		}

		const bool staticSettingsChanged =
		    !surfelPathTracerStaticSettingsMatch(surfelPathTracerStaticSettings, nextStaticSettings);
		const bool atlasSettingsChanged =
		    surfelPathTracerAtlasSettingsChanged(surfelPathTracerStaticSettings, nextStaticSettings);
		const bool needsResourceRecreate =
		    staticSettingsChanged ||
		    surfelPathTracerResources.needsPersistentResourceRecreate(ui.surfelPathTracerSettings);

		waitForSurfelPathTracerIdle(needsResourceRecreate
		                                ? "surfel path tracer persistent resource fence"
		                                : "surfel path tracer stats fence");

		if (needsResourceRecreate)
		{
			surfelPathTracerResources.resetPersistentResources(vulkan, ui.surfelPathTracerSettings);
			if (atlasSettingsChanged)
			{
				surfelPathTracerResources.recreateSwapchainResources(vulkan, swapchain);
				surfelPathTracerPersistentImageLayoutsInitialized = false;
				createSurfelPathTracerSkyDescriptorSets();
			}
			createSurfelPathTracerStorageDescriptorSets();
			createSurfelPathTracerRtDescriptorSets();
			surfelPathTracerStaticSettings = nextStaticSettings;
			surfelPathTracerStaticSettingsInitialized = true;
			surfelPathTracerHistoryValid.fill(false);
			ptForceHistoryReset = true;
		}

		ui.surfelPathTracerStats = surfelPathTracerResources.readStats();
	};
	if (ui.renderMode == RenderMode::SurfelPathTracer)
	{
		refreshSurfelPathTracerRuntimeResources();
	}

	// Runtime skinned BLAS refit currently reuses per-model AS buffers across frames.
	// Serialize in-flight submissions in this mode to avoid cross-frame AS write hazards.
	if (resourceManager && resourceManager->hasRuntimeSkinnedModels())
	{
		for (size_t i = 0; i < frames.inFlightFences.size(); ++i)
		{
			if (i == frames.frameIndex)
			{
				continue;
			}
			const auto syncResult = waitForFenceOrThrow(
			    vulkan.logicalDevice, *frames.inFlightFences[i], "runtime skinned BLAS synchronization fence");
			if (syncResult != vk::Result::eSuccess)
			{
				throw std::runtime_error("failed to synchronize runtime skinned BLAS updates");
			}
		}
	}

	if (submittedRenderModes[frames.frameIndex] == RenderMode::PathTracer)
	{
		collectPathTracerTimings(frames.frameIndex);
		collectPathTracerAnalysisCounters(frames.frameIndex);
		updatePathTracerExperimentSweep();
		updateAdaptivePathTracerSettings();
	}

	auto [result, imageIndex] = swapchain.swapChain.acquireNextImage(
	    UINT64_MAX, *frames.presentCompleteSemaphores[frames.frameIndex], nullptr);

	if (result == vk::Result::eErrorOutOfDateKHR)
	{
		recreateSwapChain();
		return;
	}
	if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
	{
		assert(result == vk::Result::eTimeout || result == vk::Result::eNotReady);
		throw std::runtime_error("failed to acquire swap chain image!");
	}

	// If this swapchain image is still associated with an older in-flight frame, wait for it.
	if (imageIndex < imagesInFlight.size() && imagesInFlight[imageIndex] != VK_NULL_HANDLE)
	{
		const auto imageFenceResult = waitForFenceOrThrow(
		    vulkan.logicalDevice, imagesInFlight[imageIndex], "swapchain image in-flight fence");
		if (imageFenceResult != vk::Result::eSuccess)
		{
			throw std::runtime_error("failed to wait for in-flight swapchain image fence");
		}
	}

	frames.updateUniformBuffer(frames.frameIndex, camera, swapchain.extent, ui.lightDirection, ui.exposure, ui.textureColorSpaceModel);

	// Camera motion metric for motion-aware temporal blending.
	const float translationDelta = glm::distance(camera.position, ptPrevCameraPos);
	const float pitchDelta       = std::abs(camera.pitch - ptPrevPitch);
	const float yawDelta         = std::abs(camera.yaw - ptPrevYaw);
	const float angularDelta     = std::max(pitchDelta, yawDelta);

	constexpr float kTranslationForFullMotion = 0.05f;
	constexpr float kRotationForFullMotion    = glm::radians(1.5f);
	const float     normalizedTranslation     = translationDelta / kTranslationForFullMotion;
	const float     normalizedRotation        = angularDelta / kRotationForFullMotion;
	const float     rawMotion                 = std::max(normalizedTranslation, normalizedRotation);
	const float     rawMotion01               = std::clamp(rawMotion, 0.0f, 1.0f);
	ptSmoothedMotion                          = glm::mix(ptSmoothedMotion, rawMotion01, 0.2f);
	ptCameraMoved                             = rawMotion01 > 1e-4f;

	// Keep hard history reset only for large jumps/teleports and explicit reset events.
	const bool largeJump = (translationDelta > std::max(0.25f, ui.pathTracerSettings.historyResetMotionThreshold)) ||
	                       (angularDelta > glm::radians(75.0f));
	ptForceHistoryReset                       = ptForceHistoryReset || largeJump;
	ui.pathTracerPerfStats.cameraMotionFactor = std::clamp(ptSmoothedMotion, 0.0f, 1.0f);

	ptPrevCameraPos = camera.position;
	ptPrevPitch     = camera.pitch;
	ptPrevYaw       = camera.yaw;

	// Only reset the fence if we are submitting work
	vulkan.logicalDevice.resetFences(*frames.inFlightFences[frames.frameIndex]);

	frames.commandBuffers[frames.frameIndex].reset();
	vk::raii::CommandBuffer &commandBuffer = frames.commandBuffers[frames.frameIndex];
	commandBuffer.begin(vk::CommandBufferBeginInfo{});

	// 2. Main Pass
	recordCommandBuffer(imageIndex);
	if (ui.renderMode == RenderMode::PathTracer)
	{
		ptForceHistoryReset = false;
	}
	else if (ui.renderMode == RenderMode::SurfelPathTracer)
	{
		ptForceHistoryReset = false;
	}
	submittedRenderModes[frames.frameIndex] = ui.renderMode;
	ptTimestampsValid[frames.frameIndex]    = (ui.renderMode == RenderMode::PathTracer);

	// The swapchain image is accessed at eColorAttachmentOutput (main/ImGui pass) and at
	// eTransfer (blit in compute and RT paths). Both stages must wait for vkAcquireNextImage.
	vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eAllCommands);
	const vk::SubmitInfo   submitInfo{
	      .waitSemaphoreCount   = 1,
	      .pWaitSemaphores      = &*frames.presentCompleteSemaphores[frames.frameIndex],
	      .pWaitDstStageMask    = &waitDestinationStageMask,
	      .commandBufferCount   = 1,
	      .pCommandBuffers      = &*frames.commandBuffers[frames.frameIndex],
	      .signalSemaphoreCount = 1,
	      .pSignalSemaphores    = &*frames.renderFinishedSemaphores[imageIndex]};

	commandBuffer.end();

	vulkan.queue.submit(submitInfo, *frames.inFlightFences[frames.frameIndex]);
	if (imageIndex < imagesInFlight.size())
	{
		imagesInFlight[imageIndex] = *frames.inFlightFences[frames.frameIndex];
	}

	const vk::PresentInfoKHR presentInfoKHR{
	    .waitSemaphoreCount = 1,
	    .pWaitSemaphores    = &*frames.renderFinishedSemaphores[imageIndex],
	    .swapchainCount     = 1,
	    .pSwapchains        = &*swapchain.swapChain,
	    .pImageIndices      = &imageIndex};

	// VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS is defined so presentKHR should return
	// eErrorOutOfDateKHR as a value rather than throw, but behaviour is inconsistent across
	// loader/driver versions. The try/catch ensures resize detection is never silently lost.
	try
	{
		result = vulkan.queue.presentKHR(presentInfoKHR);
	}
	catch (vk::OutOfDateKHRError &)
	{
		result = vk::Result::eErrorOutOfDateKHR;
	}
	catch (vk::SurfaceLostKHRError &)
	{
		result = vk::Result::eErrorOutOfDateKHR;
	}

	if ((result == vk::Result::eSuboptimalKHR) || (result == vk::Result::eErrorOutOfDateKHR) ||
	    swapchain.framebufferResized)
	{
		swapchain.framebufferResized = false;
		recreateSwapChain();
	}
	else
	{
		assert(result == vk::Result::eSuccess);
	}
	frames.frameIndex = (frames.frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
}
