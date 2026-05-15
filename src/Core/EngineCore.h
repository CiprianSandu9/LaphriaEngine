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
	struct PathTracerExperimentRow
	{
		std::string name;
		UISystem::FirstHitProbeSamplingMode probeMode = UISystem::FirstHitProbeSamplingMode::CosineHemisphere;
		float cacheReuseWeight = 1.0f;
		float cacheConnectionRadius = 3.5f;
		float cacheMisStrength = 1.5f;
		UISystem::PathTracerCacheWeightingMode cacheWeightingMode =
		    UISystem::PathTracerCacheWeightingMode::CalibratedWeight;
		UISystem::PathTracerCacheProposalMode cacheProposalMode =
		    UISystem::PathTracerCacheProposalMode::SpatialLocal;
		bool adaptiveCacheRefresh = true;
		bool targetedDiagnosticCacheRefresh = false;
		bool enableSunVisibleCandidateCache = true;
		bool blackEnvironment = true;
		bool applyDebugLightPreset = true;
		bool clearCacheBeforeRow = true;
		int cacheRefreshCandidateCount = 1;
		int cacheSpatialCandidateTrials = 16;
		int cacheVisibilityValidationBudget = 7;
		int firstHitDiffuseSamples = 8;
		int firstHitCandidateCount = 4;
		UISystem::PathTracerDebugLightPreset lightPreset = UISystem::PathTracerDebugLightPreset::HardBounce;
		UISystem::PathTracerDebugAov debugAov = UISystem::PathTracerDebugAov::PathRawFinalColor;
	};
	struct PathTracerExperimentAccumulator
	{
		uint32_t sampleCount = 0;
		double targetWallLuma = 0.0;
		double targetWallBaseLuma = 0.0;
		double targetWallProbeAddedLuma = 0.0;
		double firstHitProbeAvgLuma = 0.0;
		double firstHitProbeSunVisibleAvgLuma = 0.0;
		double residentCachedCandidates = 0.0;
		double cacheReadBank = 0.0;
		double cacheWriteBank = 0.0;
		double cacheReadBankInserts = 0.0;
		double cacheWriteBankInserts = 0.0;
		double cacheReadBankResidents = 0.0;
		double cacheWriteBankResidents = 0.0;
		double cacheReusePathEntries = 0.0;
		double cacheReuseExtraSampleZero = 0.0;
		double cacheReuseUnavailable = 0.0;
		double cacheReuseAttempts = 0.0;
		double cacheReuseSelected = 0.0;
		double cacheReuseSelectMiss = 0.0;
		double cacheReuseRejectDistance = 0.0;
		double cacheReuseRejectGeometry = 0.0;
		double cacheReuseRejectVisibility = 0.0;
		double cacheReuseSelectLoaded = 0.0;
		double cacheReuseSelectLoadMiss = 0.0;
		double cacheReuseSelectRejectDistance = 0.0;
		double cacheReuseSelectRejectGeometry = 0.0;
		double cacheReuseAcceptedDistanceNear = 0.0;
		double cacheReuseAcceptedDistanceMid = 0.0;
		double cacheReuseAcceptedDistanceFar = 0.0;
		double cacheReuseAcceptedDistanceVeryFar = 0.0;
		double cacheReuseAcceptedLumaDark = 0.0;
		double cacheReuseAcceptedLumaDim = 0.0;
		double cacheReuseAcceptedLumaUseful = 0.0;
		double cacheReuseAcceptedLumaBright = 0.0;
		double cacheConnectionReuseAttempts = 0.0;
		double cacheConnectionReuseAccepted = 0.0;
		double cacheReuseAcceptedRatio = 0.0;
		double cacheReuseAvgLuma = 0.0;
		double diagnosticTargetCacheReuseAttempts = 0.0;
		double diagnosticTargetCacheSelected = 0.0;
		double diagnosticTargetCacheRejectDistance = 0.0;
		double diagnosticTargetCacheRejectGeometry = 0.0;
		double diagnosticTargetCacheRejectVisibility = 0.0;
		double diagnosticTargetCacheReuseAccepted = 0.0;
		double diagnosticTargetCacheReuseAvgLuma = 0.0;
		double cacheRefreshAttempts = 0.0;
		double cacheRefreshInserts = 0.0;
		double cacheMisAvgWeight = 0.0;
		double rayTraceMs = 0.0;
		double totalFrameMs = 0.0;
	};
	bool ptExperimentSweepActive{false};
	size_t ptExperimentRowIndex{0};
	int ptExperimentWarmupRemaining{0};
	int ptExperimentSampleRemaining{0};
	std::vector<PathTracerExperimentRow> ptExperimentRows;
	PathTracerExperimentAccumulator ptExperimentAccum;
	glm::vec3 lastSunVisibleCacheLightDirection{0.0f};
	uint32_t  sunVisibleCacheGeneration{1};
	bool      sunVisibleCacheLightDirectionInitialized{false};
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
	void startPathTracerGiCacheWeightSweep();
	void startPathTracerSponzaGiPerfSweep();
	void clearPathTracerExperimentCache();
	void clearSunVisibleCandidateCache(const char *reason);
	void updateSunVisibleCandidateCacheInvalidation();
	void updatePathTracerExperimentSweep();
	void applyPathTracerExperimentRow(const PathTracerExperimentRow &row);
	void logPathTracerExperimentRow(const PathTracerExperimentRow &row,
	                                const PathTracerExperimentAccumulator &accum) const;
	void ensurePathTracerSanityScene();
	void applyPathTracerDebugLightPreset();
	void loadPathTracerIndirectBounceTestSceneIfRequested();
	void loadPathTracerSponzaGiValidationPresetIfRequested();
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
