#ifndef LAPHRIAENGINE_SURFELPATHTRACERRESOURCES_H
#define LAPHRIAENGINE_SURFELPATHTRACERRESOURCES_H

#include <algorithm>
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
	uint32_t lastSeenFrame = 0;
	uint32_t lastReferencedFrame = 0;
	uint32_t sleepState = 0;
	uint32_t materialKey = 0;
	glm::vec4 varianceAndInconsistency{1.0f};
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
	uint32_t recycledSurfels = 0;
	uint32_t spawnedSurfels = 0;
	uint32_t removedSurfels = 0;
	uint32_t guidedRays = 0;
	uint32_t cosineRays = 0;
	uint32_t surfelTerminatedPaths = 0;
	uint32_t pathMisses = 0;
	uint32_t pad1 = 0;
};

struct SurfelPathTracerCellAddress
{
	glm::ivec3 coord{0};
	uint32_t flatIndex = 0;
};

class SurfelPathTracerResources
{
  public:
	SurfelPathTracerResources() = default;
	~SurfelPathTracerResources();

	SurfelPathTracerResources(const SurfelPathTracerResources &) = delete;
	SurfelPathTracerResources &operator=(const SurfelPathTracerResources &) = delete;
	SurfelPathTracerResources(SurfelPathTracerResources &&) = delete;
	SurfelPathTracerResources &operator=(SurfelPathTracerResources &&) = delete;

	void init(const VulkanDevice &dev, const SwapchainManager &swapchain,
	          const UISystem::SurfelPathTracerSettings &settings);
	void cleanupSwapchainResources();
	void recreateSwapchainResources(const VulkanDevice &dev, const SwapchainManager &swapchain);
	void resetPersistentResources(const VulkanDevice &dev,
	                              const UISystem::SurfelPathTracerSettings &settings);
	void destroy();

	[[nodiscard]] bool initialized() const { return initialized_; }
	[[nodiscard]] uint32_t cellCount() const { return cellCount_; }
	[[nodiscard]] uint32_t maxSurfelsCapacity() const { return settings_.maxSurfels; }
	[[nodiscard]] uint32_t maxRaysPerFrameCapacity() const { return settings_.maxRaysPerFrame; }
	[[nodiscard]] uint32_t cellDimensionCapacity() const { return settings_.cellDimension; }
	[[nodiscard]] uint32_t perCellSurfelLimitCapacity() const { return settings_.perCellSurfelLimit; }
	[[nodiscard]] bool needsPersistentResourceRecreate(
	    const UISystem::SurfelPathTracerSettings &settings) const;
	[[nodiscard]] bool needsPersistentReset() const { return needsPersistentReset_; }
	void markPersistentResetConsumed() { needsPersistentReset_ = false; }
	[[nodiscard]] UISystem::SurfelPathTracerStats readStats() const;

	static SurfelPathTracerCellAddress cellAddressForPosition(const glm::vec3 &position,
	                                                          float cellSize,
	                                                          uint32_t cellDimension)
	{
		const float safeCellSize = std::max(cellSize, 0.0001f);
		const uint32_t dim = std::clamp(cellDimension, kMinCellDimension, kMaxCellDimension);
		const double halfDim = static_cast<double>(dim) * 0.5;
		const double maxCoord = static_cast<double>(dim - 1u);
		const glm::dvec3 centeredCell = glm::floor(glm::dvec3(position) / static_cast<double>(safeCellSize)) +
		                                glm::dvec3(halfDim);
		const glm::dvec3 clampedCell = glm::clamp(centeredCell, glm::dvec3(0.0), glm::dvec3(maxCoord));
		const glm::ivec3 coord{
		    static_cast<int>(clampedCell.x),
		    static_cast<int>(clampedCell.y),
		    static_cast<int>(clampedCell.z)};
		const uint64_t flat = static_cast<uint64_t>(coord.x) +
		                      static_cast<uint64_t>(coord.y) * dim +
		                      static_cast<uint64_t>(coord.z) * dim * dim;
		return {coord, static_cast<uint32_t>(flat)};
	}
	static SurfelPathTracerCellAddress cameraRelativeCellAddressForPosition(const glm::vec3 &position,
	                                                                        const glm::vec3 &cameraPosition,
	                                                                        float cellSize,
	                                                                        uint32_t cellDimension)
	{
		return cellAddressForPosition(position - cameraPosition, cellSize, cellDimension);
	}

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
	std::vector<VulkanUtils::VmaImage> gBufferMotionMaterialImages;
	std::vector<vk::raii::ImageView> gBufferMotionMaterialViews;
	std::vector<VulkanUtils::VmaImage> gBufferAlbedoImages;
	std::vector<vk::raii::ImageView> gBufferAlbedoViews;
	std::vector<VulkanUtils::VmaImage> reflectionImages;
	std::vector<vk::raii::ImageView> reflectionViews;
	std::vector<VulkanUtils::VmaImage> filteredReflectionImages;
	std::vector<vk::raii::ImageView> filteredReflectionViews;
	std::vector<VulkanUtils::VmaImage> lightingImages;
	std::vector<vk::raii::ImageView> lightingViews;
	std::vector<VulkanUtils::VmaImage> taaHistoryImages;
	std::vector<vk::raii::ImageView> taaHistoryViews;
	std::vector<VulkanUtils::VmaImage> irradianceAtlasImages;
	std::vector<vk::raii::ImageView> irradianceAtlasViews;
	std::vector<VulkanUtils::VmaImage> surfelDepthAtlasImages;
	std::vector<vk::raii::ImageView> surfelDepthAtlasViews;

  private:
	static constexpr uint32_t kMinCellDimension = 8;
	static constexpr uint32_t kMaxCellDimension = 128;
	static constexpr uint32_t kMinPerCellSurfelLimit = 1;
	static constexpr uint32_t kMaxPerCellSurfelLimit = 256;

	void createPersistentBuffers(const VulkanDevice &dev,
	                             const UISystem::SurfelPathTracerSettings &settings);
	void createExtentImages(const VulkanDevice &dev, const SwapchainManager &swapchain);
	void destroyPersistentBuffers();

	bool initialized_ = false;
	bool needsPersistentReset_ = true;
	uint32_t cellCount_ = 0;
	UISystem::SurfelPathTracerSettings settings_{};
};
} // namespace Laphria

#endif // LAPHRIAENGINE_SURFELPATHTRACERRESOURCES_H
