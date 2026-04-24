#ifndef LAPHRIAENGINE_FRAMECONTEXT_H
#define LAPHRIAENGINE_FRAMECONTEXT_H

#include <glm/glm.hpp>
#include <vector>
#include <vulkan/vulkan_raii.hpp>

#include "Camera.h"
#include "EngineAuxiliary.h"
#include "EngineConfig.h"
#include "SwapchainManager.h"
#include "UISystem.h"
#include "VulkanUtils.h"

// Owns all per-frame GPU resources: command buffers, sync objects, depth/storage/shadow images, UBOs.
// Call init() once after the swapchain is ready; call recreate() after swapchain recreation.
class FrameContext
{
  public:
	~FrameContext();

	void init(VulkanDevice &dev, SwapchainManager &swapchain);
	void cleanupSwapChainDependents();
	void recreate(VulkanDevice &dev, SwapchainManager &swapchain);
	void updateUniformBuffer(uint32_t frameIdx, const Camera &camera, vk::Extent2D extent, glm::vec3 lightDirection,
	                         const UISystem::VisualsV1Settings &visualsV1);

	// ── CSM Shadow resources (extent-independent, NOT cleaned on swapchain resize) ──
	// One depth array image per frame-in-flight; each has NUM_SHADOW_CASCADES layers at SHADOW_MAP_DIM x SHADOW_MAP_DIM.
	std::vector<Laphria::VulkanUtils::VmaImage> shadowImages;
	// Per-layer 2D views for rendering into each cascade (size = MAX_FRAMES_IN_FLIGHT * NUM_SHADOW_CASCADES).
	// Access: shadowCascadeViews[frameIndex * NUM_SHADOW_CASCADES + cascadeIndex]
	std::vector<vk::raii::ImageView>    shadowCascadeViews;
	// Full 2D_ARRAY views for sampling in the main pass (size = MAX_FRAMES_IN_FLIGHT).
	std::vector<vk::raii::ImageView>    shadowArrayViews;
	// Comparison sampler (shared across frames and cascades).
	vk::raii::Sampler                   shadowSampler{nullptr};

	uint32_t frameIndex = 0;

	// ── Command resources ─────────────────────────────────────────────────
	vk::raii::CommandPool                commandPool{nullptr};
	std::vector<vk::raii::CommandBuffer> commandBuffers;

	// ── Synchronization ───────────────────────────────────────────────────
	std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
	std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
	std::vector<vk::raii::Fence>     inFlightFences;

	// ── Depth images (per swapchain image) ───────────────────────────────
	std::vector<Laphria::VulkanUtils::VmaImage> depthImages;
	std::vector<vk::raii::ImageView>            depthImageViews;

	// ── Storage images for compute starfield (per frame in flight) ────────
	std::vector<Laphria::VulkanUtils::VmaImage> storageImages;
	std::vector<vk::raii::ImageView>            storageImageViews;

	// ── RT output images (per frame in flight) ────────────────────────────
	// Noisy 1-SPP path tracer output. After denoising, the final denoised result is
	// written back here so the existing swapchain blit path remains unchanged.
	std::vector<Laphria::VulkanUtils::VmaImage> rayTracingOutputImages;
	std::vector<vk::raii::ImageView>            rayTracingOutputImageViews;

	// ── G-Buffer images written by the Raygen shader (per frame in flight) ──
	// All are swapchain-extent-dependent and recreated on resize.
	std::vector<Laphria::VulkanUtils::VmaImage> rtGBufferNormals;        // R16G16B16A16_SFLOAT world normals
	std::vector<vk::raii::ImageView>            rtGBufferNormalsViews;

	std::vector<Laphria::VulkanUtils::VmaImage> rtGBufferDepth;          // R32_SFLOAT linear depth (ray hit t)
	std::vector<vk::raii::ImageView>            rtGBufferDepthViews;

	std::vector<Laphria::VulkanUtils::VmaImage> rtMotionVectors;         // R16G16_SFLOAT screen-space motion
	std::vector<vk::raii::ImageView>            rtMotionVectorsViews;

	// ── Temporal accumulation history buffers (per frame in flight) ─────────
	// historyColor[i] stores the blended reprojection output from frame slot i,
	// read by frame slot (i+1)%2 as "previous frame" and written by frame slot i.
	std::vector<Laphria::VulkanUtils::VmaImage> historyColor;            // R16G16B16A16_SFLOAT
	std::vector<vk::raii::ImageView>            historyColorViews;

	std::vector<Laphria::VulkanUtils::VmaImage> historyMoments;          // R16G16_SFLOAT (mean, variance)
	std::vector<vk::raii::ImageView>            historyMomentsViews;

	// ── A-Trous ping-pong buffers (per frame slot) ─────────────────────────
	// Layout: atrousTemp[frameIndex*2 + 0] and atrousTemp[frameIndex*2 + 1].
	// This avoids cross-frame hazards and removes the need to serialize PT frames.
	std::vector<Laphria::VulkanUtils::VmaImage> atrousTemp;              // MAX_FRAMES_IN_FLIGHT * 2 × R16G16B16A16_SFLOAT
	std::vector<vk::raii::ImageView>            atrousTempViews;

	// ── Temporal tracking (updated each frame by updateUniformBuffer) ────────
	glm::mat4 prevViewProj{1.0f};   // VP matrix of the last submitted frame
	uint32_t  frameCount = 0;       // monotonically increasing, seeds per-pixel RNG

	// ── Uniform buffers (per frame in flight) ─────────────────────────────
	std::vector<Laphria::VulkanUtils::VmaBuffer> uniformBuffers;
	std::vector<void *>                          uniformBuffersMapped;

	// ── Ray Tracing TLAS (per frame in flight) ────────────────────────────
	static constexpr uint32_t                       MAX_TLAS_INSTANCES = Laphria::EngineConfig::kMaxTLASInstances;
	std::vector<vk::raii::AccelerationStructureKHR> tlas;
	std::vector<Laphria::VulkanUtils::VmaBuffer>    tlasBuffers;

	std::vector<Laphria::VulkanUtils::VmaBuffer> tlasScratchBuffers;
	std::vector<vk::DeviceAddress>               tlasScratchAddresses;

	std::vector<Laphria::VulkanUtils::VmaBuffer> tlasInstanceBuffers;
	std::vector<void *>                          tlasInstanceBuffersMapped;
	std::vector<vk::DeviceAddress>               tlasInstanceAddresses;

  private:
	void createCommandPool(const VulkanDevice &dev);
	void createCommandBuffers(const VulkanDevice &dev);
	void createSyncObjects(VulkanDevice &dev, uint32_t imageCount);
	void createDepthResources(const VulkanDevice &dev, const SwapchainManager &swapchain);
	void createStorageResources(VulkanDevice &dev, SwapchainManager &swapchain);
	void createRayTracingOutputImages(const VulkanDevice &dev, const SwapchainManager &swapchain);
	void createGBufferResources(const VulkanDevice &dev, const SwapchainManager &swapchain);
	void createHistoryResources(const VulkanDevice &dev, const SwapchainManager &swapchain);
	void createAtrousResources(const VulkanDevice &dev, const SwapchainManager &swapchain);

	void createUniformBuffers(const VulkanDevice &dev);
	void createTLASResources(VulkanDevice &dev);
	void createShadowResources(const VulkanDevice &dev);
};

#endif        // LAPHRIAENGINE_FRAMECONTEXT_H
