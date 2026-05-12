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
#include "PathTracerAnalysis.h"
#include "ResourceManager.h"
#include "SwapchainManager.h"
#include "EngineHost.h"
#include "UISystem.h"
#include "VulkanDevice.h"

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

	// Denoiser Resources (one set per frame in flight)
	vk::raii::DescriptorPool             denoiserDescriptorPool{nullptr};
	std::vector<vk::raii::DescriptorSet> denoiserDescriptorSets;
	vk::raii::QueryPool                  ptTimestampQueryPool{nullptr};
	float                                timestampPeriodNs = 1.0f;
	std::array<bool, MAX_FRAMES_IN_FLIGHT>       ptTimestampsValid{};
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
	bool      ptBenchmarkSceneLoaded{false};
	glm::vec3 ptBenchmarkBasePosition{0.0f, 1.2f, 0.0f};
	float     ptBenchmarkBasePitch{glm::radians(-8.0f)};
	float     ptBenchmarkBaseYaw{glm::radians(180.0f)};
	float     ptBenchmarkClockSeconds{0.0f};
	float     ptBenchmarkTeleportClockSeconds{0.0f};
	size_t    ptSweepConfigIndex{0};
	int       ptSweepWarmupRemaining{0};
	int       ptSweepSampleRemaining{0};
	std::vector<Laphria::PathTracerSweepConfig> ptSweepConfigs;
	std::vector<float>                 ptRollingTotalMs;
	std::vector<float>                 ptRollingRayTraceMs;
	std::vector<float>                 ptRollingDenoiserMs;
	std::vector<float>                 ptSampleTotalMs;
	std::vector<float>                 ptSampleRayTraceMs;
	std::vector<float>                 ptSampleDenoiserMs;
	std::vector<Laphria::PathTracerRunScore>    ptSweepScores;
	std::vector<Laphria::PathTracerBacklogItem> ptBacklogItems;
	bool      ptSanitySceneCreated{false};
	bool      ptSanityBaselineCaptured{false};
	float     ptSanityBaselineExposure{1.0f};
	float     ptSanityBaselineRejectRatio{0.0f};
	float     ptSanityBaselineFireflyRatio{0.0f};
	float     ptSanityBaselineSkyRatio{0.0f};
	float     ptSanityDriftMetric{0.0f};
	int       ptSanityPhase{0};
	int       ptSanityFramesRemaining{0};
	SceneNode::Ptr ptSanityWhiteDiffuseNode;
	SceneNode::Ptr ptSanityRoughMetalNode;
	SceneNode::Ptr ptSanityEmissiveNode;
	int            ptIndirectBounceTargetWallModelId{-1};
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
	void createDenoiserDescriptorSets();

	void recordComputeCommandBuffer(const vk::raii::CommandBuffer &commandBuffer, uint32_t imageIndex) const;
	void recordSkinningPass(const vk::raii::CommandBuffer &commandBuffer) const;
	void recordClassicRTCommandBuffer(const vk::raii::CommandBuffer &commandBuffer, uint32_t imageIndex) const;
	void recordRayTracingCommandBuffer(const vk::raii::CommandBuffer &commandBuffer, uint32_t imageIndex) const;

	void createDescriptorPool();

	void createDescriptorSets();
	void createTimestampQueryPool();
	void collectPathTracerTimings(uint32_t frameSlot);
	void updatePathTracerTimingPercentiles();
	void resetPathTracerAnalysisCounters(uint32_t frameSlot);
	void collectPathTracerAnalysisCounters(uint32_t frameSlot);
	void ensurePathTracerSanityScene();
	void loadPathTracerIndirectBounceTestSceneIfRequested();
	void updatePathTracerPhysicalSanityChecks(float deltaTimeSeconds);
	void writePathTracerBacklogCsv();
	void updatePathTracerBenchmark(float deltaTimeSeconds);
	void loadPathTracerBenchmarkSceneIfNeeded();
	void updateAdaptivePathTracerSettings();

	[[nodiscard]] uint32_t getPathTracerQueryBase(uint32_t frameSlot) const;

	void recordCommandBuffer(uint32_t imageIndex) const;

	void transition_image_layout(vk::Image image, vk::ImageLayout old_layout, vk::ImageLayout new_layout, vk::AccessFlags2 src_access_mask, vk::AccessFlags2 dst_access_mask,
	                             vk::PipelineStageFlags2 src_stage_mask, vk::PipelineStageFlags2 dst_stage_mask, vk::ImageAspectFlags image_aspect_flags) const;

	void drawFrame();
};

#endif        // LAPHRIAENGINE_ENGINECORE_H
