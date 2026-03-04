#ifndef LAPHRIAENGINE_FRAMECONTEXT_H
#define LAPHRIAENGINE_FRAMECONTEXT_H

#include <glm/glm.hpp>
#include <vector>
#include <vulkan/vulkan_raii.hpp>

#include "Camera.h"
#include "EngineAuxiliary.h"
#include "SwapchainManager.h"
#include "VulkanDevice.h"

// Owns all per-frame GPU resources: command buffers, sync objects, depth/storage/shadow images, UBOs.
// Call init() once after the swapchain is ready; call recreate() after swapchain recreation.
class FrameContext
{
  public:
	void init(VulkanDevice &dev, SwapchainManager &swapchain);
	void cleanupSwapChainDependents();
	void recreate(VulkanDevice &dev, SwapchainManager &swapchain);
	void updateUniformBuffer(uint32_t frameIdx, const Camera &camera, vk::Extent2D extent, glm::vec3 lightDirection);

	// ── CSM Shadow resources (extent-independent, NOT cleaned on swapchain resize) ──
	// One depth array image per frame-in-flight; each has NUM_SHADOW_CASCADES layers at SHADOW_MAP_DIM x SHADOW_MAP_DIM.
	std::vector<vk::raii::Image>        shadowImages;
	std::vector<vk::raii::DeviceMemory> shadowImagesMemory;
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
	std::vector<vk::raii::Image>        depthImages;
	std::vector<vk::raii::DeviceMemory> depthImagesMemory;
	std::vector<vk::raii::ImageView>    depthImageViews;

	// ── Storage images for compute starfield (per frame in flight) ────────
	std::vector<vk::raii::Image>        storageImages;
	std::vector<vk::raii::DeviceMemory> storageImagesMemory;
	std::vector<vk::raii::ImageView>    storageImageViews;

	// ── RT output images (per frame in flight) ────────────────────────────
	// Noisy 1-SPP path tracer output. After denoising, the final denoised result is
	// written back here so the existing swapchain blit path remains unchanged.
	std::vector<vk::raii::Image>        rayTracingOutputImages;
	std::vector<vk::raii::DeviceMemory> rayTracingOutputImagesMemory;
	std::vector<vk::raii::ImageView>    rayTracingOutputImageViews;

	// ── G-Buffer images written by the Raygen shader (per frame in flight) ──
	// All are swapchain-extent-dependent and recreated on resize.
	std::vector<vk::raii::Image>        rtGBufferNormals;        // R16G16B16A16_SFLOAT world normals
	std::vector<vk::raii::DeviceMemory> rtGBufferNormalsMemory;
	std::vector<vk::raii::ImageView>    rtGBufferNormalsViews;

	std::vector<vk::raii::Image>        rtGBufferDepth;          // R32_SFLOAT linear depth (ray hit t)
	std::vector<vk::raii::DeviceMemory> rtGBufferDepthMemory;
	std::vector<vk::raii::ImageView>    rtGBufferDepthViews;

	std::vector<vk::raii::Image>        rtMotionVectors;         // R16G16_SFLOAT screen-space motion
	std::vector<vk::raii::DeviceMemory> rtMotionVectorsMemory;
	std::vector<vk::raii::ImageView>    rtMotionVectorsViews;

	// ── Temporal accumulation history buffers (per frame in flight) ─────────
	// historyColor[i] stores the blended reprojection output from frame slot i,
	// read by frame slot (i+1)%2 as "previous frame" and written by frame slot i.
	std::vector<vk::raii::Image>        historyColor;            // R16G16B16A16_SFLOAT
	std::vector<vk::raii::DeviceMemory> historyColorMemory;
	std::vector<vk::raii::ImageView>    historyColorViews;

	std::vector<vk::raii::Image>        historyMoments;          // R16G16_SFLOAT (mean, variance)
	std::vector<vk::raii::DeviceMemory> historyMomentsMemory;
	std::vector<vk::raii::ImageView>    historyMomentsViews;

	// ── A-Trous ping-pong buffers (2, shared across frames — not per-slot) ──
	// atrousTemp[0] receives the reprojection output; iterations alternate between [0] and [1].
	std::vector<vk::raii::Image>        atrousTemp;              // 2 × R16G16B16A16_SFLOAT
	std::vector<vk::raii::DeviceMemory> atrousTempMemory;
	std::vector<vk::raii::ImageView>    atrousTempViews;

	// ── Temporal tracking (updated each frame by updateUniformBuffer) ────────
	glm::mat4 prevViewProj{1.0f};   // VP matrix of the last submitted frame
	uint32_t  frameCount = 0;       // monotonically increasing, seeds per-pixel RNG

	// ── Uniform buffers (per frame in flight) ─────────────────────────────
	std::vector<vk::raii::Buffer>       uniformBuffers;
	std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
	std::vector<void *>                 uniformBuffersMapped;

	// ── Ray Tracing TLAS (per frame in flight) ────────────────────────────
	const uint32_t                                  MAX_TLAS_INSTANCES = 10000;
	std::vector<vk::raii::AccelerationStructureKHR> tlas;
	std::vector<vk::raii::Buffer>                   tlasBuffers;
	std::vector<vk::raii::DeviceMemory>             tlasMemories;

	std::vector<vk::raii::Buffer>       tlasScratchBuffers;
	std::vector<vk::raii::DeviceMemory> tlasScratchMemories;
	std::vector<vk::DeviceAddress>      tlasScratchAddresses;

	std::vector<vk::raii::Buffer>       tlasInstanceBuffers;
	std::vector<vk::raii::DeviceMemory> tlasInstanceMemories;
	std::vector<void *>                 tlasInstanceBuffersMapped;
	std::vector<vk::DeviceAddress>      tlasInstanceAddresses;

  private:
	void createCommandPool(VulkanDevice &dev);
	void createCommandBuffers(VulkanDevice &dev);
	void createSyncObjects(VulkanDevice &dev, uint32_t imageCount);
	void createDepthResources(VulkanDevice &dev, SwapchainManager &swapchain);
	void createStorageResources(VulkanDevice &dev, SwapchainManager &swapchain);
	void createRayTracingOutputImages(VulkanDevice &dev, SwapchainManager &swapchain);
	void createGBufferResources(VulkanDevice &dev, SwapchainManager &swapchain);
	void createHistoryResources(VulkanDevice &dev, SwapchainManager &swapchain);
	void createAtrousResources(VulkanDevice &dev, SwapchainManager &swapchain);

	void createUniformBuffers(VulkanDevice &dev);
	void createTLASResources(VulkanDevice &dev);
	void createShadowResources(VulkanDevice &dev);
};

#endif        // LAPHRIAENGINE_FRAMECONTEXT_H
