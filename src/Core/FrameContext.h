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
	void updateUniformBuffer(uint32_t frameIdx, const Camera &camera, vk::Extent2D extent, glm::vec3 lightDirection) const;

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
	// Separate from the compute storage images: the RT pipeline writes here,
	// then these are blitted to the swapchain independently of the starfield.
	std::vector<vk::raii::Image>        rayTracingOutputImages;
	std::vector<vk::raii::DeviceMemory> rayTracingOutputImagesMemory;
	std::vector<vk::raii::ImageView>    rayTracingOutputImageViews;

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

	void createUniformBuffers(VulkanDevice &dev);
	void createTLASResources(VulkanDevice &dev);
	void createShadowResources(VulkanDevice &dev);
};

#endif        // LAPHRIAENGINE_FRAMECONTEXT_H
