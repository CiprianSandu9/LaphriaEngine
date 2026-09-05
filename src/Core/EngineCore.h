#ifndef LAPHRIAENGINE_ENGINECORE_H
#define LAPHRIAENGINE_ENGINECORE_H

#include <chrono>
#include <array>
#include <string>
#include <memory>
#include <vector>

#include "../Physics/PhysicsSystem.h"
#include "../SceneManagement/Scene.h"
#include "Camera.h"
#include "FrameContext.h"
#include "InputSystem.h"
#include "PipelineCollection.h"
#include "ResourceManager.h"
#include "SurfelPathTracerPasses.h"
#include "SurfelPathTracerResources.h"
#include "SwapchainManager.h"
#include "EngineHost.h"
#include <array>
#include "UISystem.h"
#include "VulkanDevice.h"

struct SurfelPathTracerStaticSettingsSnapshot
{
	float cellSize = 0.0f;
	uint32_t maxSurfels = 0;
	uint32_t maxRaysPerFrame = 0;
	uint32_t cellDimension = 0;
	uint32_t perCellSurfelLimit = 0;
	uint32_t irradianceAtlasWidth = 0;
	uint32_t irradianceAtlasHeight = 0;
};

class EngineCore
{
  public:
	EngineCore() = default;
	EngineCore(EngineHostOptions options, EngineHostCallbacks callbacks);

	void run();

  private:
	EngineHostOptions   options;
	EngineHostCallbacks callbacks;
	GLFWwindow *window{nullptr};
	Camera      camera;
	InputSystem input;

	VulkanDevice       vulkan;
	UISystem           ui;
	SwapchainManager   swapchain;
	PipelineCollection pipelines;
	FrameContext       frames;
	Laphria::SurfelPathTracerResources surfelPathTracerResources;
	Laphria::SurfelPathTracerPasses    surfelPathTracerPasses;

	// Auto-exposure luminance probe: the HDR image is blitted down to a small image and read
	// back on the CPU (per frame slot), where the log-average luminance drives ui.exposure.
	static constexpr uint32_t kLuminanceProbeWidth  = 96;
	static constexpr uint32_t kLuminanceProbeHeight = 54;
	static constexpr vk::DeviceSize kPixelProbeBytes = 128;
	std::array<Laphria::VulkanUtils::VmaImage, MAX_FRAMES_IN_FLIGHT>  luminanceProbeImages;
	std::array<Laphria::VulkanUtils::VmaBuffer, MAX_FRAMES_IN_FLIGHT> luminanceReadbackBuffers;
	std::array<void *, MAX_FRAMES_IN_FLIGHT>                           luminanceReadbackMapped{};
	mutable std::array<bool, MAX_FRAMES_IN_FLIGHT>                     luminanceReadbackValid{};
	mutable std::array<bool, MAX_FRAMES_IN_FLIGHT>                     luminanceProbeImageInitialized{};
	// Pixel probe (SurfelPathTracer only): one texel of several HDR/GBuffer images copied to a host buffer.
	std::array<Laphria::VulkanUtils::VmaBuffer, MAX_FRAMES_IN_FLIGHT> pixelProbeReadbackBuffers;
	std::array<void *, MAX_FRAMES_IN_FLIGHT>                           pixelProbeReadbackMapped{};
	mutable std::array<bool, MAX_FRAMES_IN_FLIGHT>                     pixelProbeReadbackValid{};
	mutable std::array<glm::uvec2, MAX_FRAMES_IN_FLIGHT>               pixelProbeReadbackPixel{};
	float lastDeltaTimeSeconds{0.0f};

	// Global UBO sets
	vk::raii::DescriptorPool             descriptorPool{nullptr};
	std::vector<vk::raii::DescriptorSet> descriptorSets;

	// Physics Pipeline
	vk::raii::DescriptorPool physicsDescriptorPool{nullptr};
	vk::raii::DescriptorSet  physicsDescriptorSet{nullptr};

	// Compute Resources
	vk::raii::DescriptorPool             computeDescriptorPool{nullptr};
	std::vector<vk::raii::DescriptorSet> computeDescriptorSets;

	// Ray Tracing Resources
	std::vector<vk::raii::DescriptorSet> rtDescriptorSets;

	// Surfel Path Tracer Resources
	vk::raii::DescriptorPool             surfelPathTracerSkyDescriptorPool{nullptr};
	std::vector<vk::raii::DescriptorSet> surfelPathTracerSkyDescriptorSets;
	vk::raii::DescriptorPool             surfelPathTracerStorageDescriptorPool{nullptr};
	std::vector<vk::raii::DescriptorSet> surfelPathTracerStorageDescriptorSets;
	vk::raii::DescriptorPool             surfelPathTracerRtDescriptorPool{nullptr};
	std::vector<vk::raii::DescriptorSet> surfelPathTracerRtDescriptorSets;
	mutable bool                         surfelPathTracerPersistentImageLayoutsInitialized = false;
	mutable uint32_t surfelPathTracerPreviousHistoryFrameIndex = 0;
	mutable bool surfelPathTracerTemporalHistoryValid = false;
	mutable std::vector<Laphria::SurfelPathTracerSourceInstance> surfelSourceInstances;
	mutable std::vector<Laphria::SurfelPathTracerSourceTransform> surfelSourceTransforms;
	mutable uint32_t currentSurfelSourceInstanceCount = 0u;
	mutable uint32_t currentSurfelSourceTransformCount = 0u;
	mutable uint32_t nextSurfelSourceNodeId = 0u;
	mutable std::array<Laphria::VulkanUtils::VmaBuffer, MAX_FRAMES_IN_FLIGHT> surfelSourceInstanceStagingBuffers;
	mutable std::array<Laphria::VulkanUtils::VmaBuffer, MAX_FRAMES_IN_FLIGHT> surfelSourceTransformStagingBuffers;
	mutable std::array<void *, MAX_FRAMES_IN_FLIGHT> surfelSourceInstanceStagingMapped{};
	mutable std::array<void *, MAX_FRAMES_IN_FLIGHT> surfelSourceTransformStagingMapped{};
	mutable std::array<vk::DeviceSize, MAX_FRAMES_IN_FLIGHT> surfelSourceInstanceStagingSizes{};
	mutable std::array<vk::DeviceSize, MAX_FRAMES_IN_FLIGHT> surfelSourceTransformStagingSizes{};
	SurfelPathTracerStaticSettingsSnapshot surfelPathTracerStaticSettings{};
	bool surfelPathTracerStaticSettingsInitialized = false;

	// Denoiser Resources (one set per frame in flight)
	vk::raii::DescriptorPool             denoiserDescriptorPool{nullptr};
	std::vector<vk::raii::DescriptorSet> denoiserDescriptorSets;

	vk::raii::QueryPool                  gpuTimestampQueryPool{nullptr};
	float                                timestampPeriodNs = 1.0f;
	std::array<bool, MAX_FRAMES_IN_FLIGHT>       gpuTimestampsValid{};
	std::array<RenderMode, MAX_FRAMES_IN_FLIGHT> submittedRenderModes{};
	std::vector<vk::Fence>                       imagesInFlight;

	// Scene System
	std::unique_ptr<Scene>           scene;
	std::unique_ptr<ResourceManager> resourceManager;
	std::unique_ptr<PhysicsSystem>   physicsSystem;

	// Path tracer camera motion tracking (for motion-aware accumulation and reset events).
	glm::vec3 ptPrevCameraPos{0.f};
	float     ptPrevPitch{0.f};
	float     ptPrevYaw{0.f};
	bool      ptCameraMoved{false};
	float     ptSmoothedMotion{0.0f};
	bool      ptForceHistoryReset{true};
	vk::Extent2D ptActiveRenderExtent{};
	std::vector<float>                 ptRollingTotalMs;
	std::vector<float>                 ptRollingRayTraceMs;
	std::vector<float>                 ptRollingDenoiserMs;
	// Surfel backend rolling window: GBuffer, Cache update, Surfel rays, Integrate,
	// Evaluate, Reflections, Post, Total (same spans as the thesis timing table).
	static constexpr size_t            kSurfelRollingSpanCount = 8;
	std::array<std::vector<float>, kSurfelRollingSpanCount> surfelRollingMs;
	// Same window for the per-frame counters: alive, rays traced, ray demand, filled cells,
	// rejected stores, spawned, removed, recycled, guided rays, cosine rays,
	// termination attempts, termination hits.
	static constexpr size_t            kSurfelRollingCounterCount = 12;
	std::array<std::vector<float>, kSurfelRollingCounterCount> surfelRollingCounters;
	RenderMode lastSubmittedRenderMode{RenderMode::Rasterizer};
	bool       renderModeInitialized{false};
	std::chrono::high_resolution_clock::time_point lastFrameTime{};
	double    titleStatsAccumSeconds{0.0};
	uint32_t  titleStatsFrameCount{0};
	bool      windowInitialized{false};
	bool      imguiInitialized{false};
	bool      vulkanInitialized{false};

	EngineServices buildServices();
	void           invokeInitializeCallback();
	void           invokeShutdownCallback();

	void initWindow();

	void initInput();

	void initVulkan();

	void initImgui();

	void mainLoop();
	void updatePerformanceWindowTitle(float deltaTimeSeconds);

	void cleanup();

	void cleanupSwapChain();

	void recreateSwapChain();

	void createPhysicsDescriptorSets();

	void createComputeDescriptorSets();

	void createRayTracingDescriptorSets();
	void createSurfelPathTracerSkyDescriptorSets();
	void createSurfelPathTracerStorageDescriptorSets();
	void createSurfelPathTracerRtDescriptorSets();
	void createDenoiserDescriptorSets();
	void resetSurfelPathTracerTemporalHistory() const;
	void updateSurfelPathTracerHistoryDescriptors(uint32_t frameIndex) const;

	void recordSkinningPass(const vk::raii::CommandBuffer &commandBuffer) const;
	void recordClassicRTCommandBuffer(const vk::raii::CommandBuffer &commandBuffer, uint32_t imageIndex) const;
	void recordRayTracingCommandBuffer(const vk::raii::CommandBuffer &commandBuffer, uint32_t imageIndex) const;
	void transitionPersistentSurfelImagesToGeneral(uint32_t frameIndex) const;
	void recordSurfelPathTracerCommandBuffer(const vk::raii::CommandBuffer &commandBuffer, uint32_t imageIndex) const;

	void createDescriptorPool();

	void createDescriptorSets();
	void createTimestampQueryPool();
	void collectPathTracerTimings(uint32_t frameSlot);
	void collectSurfelPathTracerTimings(uint32_t frameSlot);
	void updatePathTracerTimingPercentiles();
	void createExposureProbeResources();
	void destroyExposureProbeResources();
	void recordLuminanceProbe(const vk::raii::CommandBuffer &commandBuffer, vk::Image hdrImage,
	                          vk::Extent2D hdrExtent, uint32_t frameSlot) const;
	void recordSurfelPixelProbe(const vk::raii::CommandBuffer &commandBuffer, uint32_t frameSlot) const;
	void updateAutoExposure(uint32_t frameSlot);
	void readPixelProbe(uint32_t frameSlot);
	void updateAdaptivePathTracerSettings();

	[[nodiscard]] uint32_t getGpuTimestampQueryBase(uint32_t frameSlot) const;

	void recordCommandBuffer(uint32_t imageIndex) const;

	void transition_image_layout(vk::Image image, vk::ImageLayout old_layout, vk::ImageLayout new_layout, vk::AccessFlags2 src_access_mask, vk::AccessFlags2 dst_access_mask,
	                             vk::PipelineStageFlags2 src_stage_mask, vk::PipelineStageFlags2 dst_stage_mask, vk::ImageAspectFlags image_aspect_flags) const;

	void drawFrame();
};

#endif        // LAPHRIAENGINE_ENGINECORE_H
