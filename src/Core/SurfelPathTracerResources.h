#ifndef LAPHRIAENGINE_SURFELPATHTRACERRESOURCES_H
#define LAPHRIAENGINE_SURFELPATHTRACERRESOURCES_H

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "SwapchainManager.h"
#include "UISystem.h"
#include "VulkanDevice.h"
#include "VulkanUtils.h"

namespace Laphria
{
static_assert(UISystem::SurfelPathTracerSettings::atlasCapacity(
                  UISystem::SurfelPathTracerSettings::kMaxAtlasDimension,
                  UISystem::SurfelPathTracerSettings::kMaxAtlasDimension) <
              (1u << 19u),
              "Packed surfel cell entries reserve 19 bits for the surfel index");

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

static constexpr uint32_t SURFEL_PT_SOURCE_FLAG_VALID = 1u << 0u;
static constexpr uint32_t SURFEL_PT_SOURCE_FLAG_REFRESH_FAILED = 1u << 1u;

struct SurfelPathTracerSource
{
	uint32_t sourceNodeId = UINT32_MAX;
	uint32_t sourceInstanceId = UINT32_MAX;
	uint32_t sourceInstanceCustomIndex = UINT32_MAX;
	uint32_t sourcePrimitiveIndex = UINT32_MAX;
	glm::vec2 sourceBarycentrics{0.0f};
	uint32_t sourceMaterialKey = 0u;
	uint32_t sourceFlags = 0u;
	glm::vec3 sourceObjectPosition{0.0f};
	uint32_t sourceObjectNormal = 0u;
};

struct SurfelPathTracerPixelSource
{
	glm::vec3 sourceObjectPosition{0.0f};
	uint32_t sourceObjectNormal = 0u;
	glm::vec2 sourceBarycentrics{0.0f};
	uint32_t sourcePrimitiveIndex = UINT32_MAX;
	uint32_t sourceInstanceId = UINT32_MAX;
	uint32_t sourceNodeId = UINT32_MAX;
	uint32_t sourceInstanceCustomIndex = UINT32_MAX;
	uint32_t sourceMaterialKey = 0u;
	uint32_t sourceFlags = 0u;
};

struct SurfelPathTracerSourceInstance
{
	uint32_t sourceNodeId = UINT32_MAX;
	uint32_t modelId = 0u;
	uint32_t primitiveOffset = 0u;
	uint32_t flags = 0u;
};

struct SurfelPathTracerSourceTransform
{
	glm::mat4 objectToWorld{1.0f};
	glm::mat4 worldToObject{1.0f};
	uint32_t flags = 0u;
	uint32_t padding0 = 0u;
	uint32_t padding1 = 0u;
	uint32_t padding2 = 0u;
};

static_assert(sizeof(SurfelPathTracerSource) == 48u);
static_assert(offsetof(SurfelPathTracerSource, sourceBarycentrics) == 16u);
static_assert(offsetof(SurfelPathTracerSource, sourceObjectPosition) == 32u);
static_assert(offsetof(SurfelPathTracerSource, sourceObjectNormal) == 44u);

static_assert(sizeof(SurfelPathTracerPixelSource) == 48u);
static_assert(offsetof(SurfelPathTracerPixelSource, sourceObjectPosition) == 0u);
static_assert(offsetof(SurfelPathTracerPixelSource, sourceObjectNormal) == 12u);
static_assert(offsetof(SurfelPathTracerPixelSource, sourceBarycentrics) == 16u);

static_assert(sizeof(SurfelPathTracerSourceInstance) == 16u);
static_assert(offsetof(SurfelPathTracerSourceInstance, flags) == 12u);

static_assert(sizeof(SurfelPathTracerSourceTransform) == 144u);
static_assert(offsetof(SurfelPathTracerSourceTransform, worldToObject) == 64u);
static_assert(offsetof(SurfelPathTracerSourceTransform, flags) == 128u);

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
	uint32_t surfelTerminationAttempts = 0;
	uint32_t pathMisses = 0;
	uint32_t surfelTerminationHits = 0;
	uint32_t indirectRayWidth = 0;
	uint32_t indirectRayHeight = 1;
	uint32_t indirectRayDepth = 1;
	uint32_t demandedRays = 0;        // unclamped ray demand (SURFEL_PT_COUNTER_DEMANDED_RAYS_OFFSET)
};

static_assert(offsetof(SurfelPathTracerCounters, indirectRayWidth) == 64u);
static_assert(sizeof(SurfelPathTracerCounters) == 80u);

struct SurfelPathTracerCellAddress
{
	glm::ivec3 coord{0};
	uint32_t flatIndex = 0;
};

class SurfelPathTracerResources
{
  public:
	using VmaBuffer = VulkanUtils::VmaBuffer;

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
	[[nodiscard]] uint32_t cellToSurfelCapacity() const { return cellToSurfelCapacity_; }
	[[nodiscard]] vk::DeviceAddress rayDispatchIndirectAddress() const { return rayDispatchIndirectAddress_; }
	[[nodiscard]] bool needsPersistentResourceRecreate(
	    const UISystem::SurfelPathTracerSettings &settings) const;
	[[nodiscard]] bool needsPersistentReset() const { return needsPersistentReset_; }
	void markPersistentResetConsumed() { needsPersistentReset_ = false; }
	void recordStatsReadback(const vk::raii::CommandBuffer &commandBuffer, uint32_t frameIndex) const;
	[[nodiscard]] UISystem::SurfelPathTracerStats readStats(uint32_t frameIndex) const;

	static SurfelPathTracerCellAddress cellAddressForPosition(const glm::vec3 &position,
	                                                          float cellSize,
	                                                          uint32_t cellDimension)
	{
		const float safeCellSize = std::max(cellSize, 0.0001f);
		const uint32_t dim = std::clamp(cellDimension, kMinCellDimension, kMaxCellDimension);
		const double halfDim = static_cast<double>(dim) * 0.5;
		const double maxCoord = static_cast<double>(dim - 1u);
		const glm::dvec3 centeredCell = glm::round(glm::dvec3(position) / static_cast<double>(safeCellSize)) +
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
		const float safeCellSize = std::max(cellSize, 0.0001f);
		const glm::dvec3 worldCell = glm::round(glm::dvec3(position) /
		                                        static_cast<double>(safeCellSize));
		const glm::dvec3 cameraCell = glm::round(glm::dvec3(cameraPosition) /
		                                         static_cast<double>(safeCellSize));
		return cellAddressForPosition(glm::vec3(worldCell - cameraCell),
		                              1.0f,
		                              cellDimension);
	}

	VmaBuffer countersBuffer;
	VmaBuffer surfelBuffer;
	VmaBuffer surfelSourceBuffer;
	VmaBuffer sourceInstanceBuffer;
	VmaBuffer sourceTransformBuffer;
	VmaBuffer deadBuffer;
	VmaBuffer rayBuffer;
	VmaBuffer cellInfoBuffer;
	VmaBuffer cellCounterBuffer;
	VmaBuffer cellToSurfelBuffer;
	vk::DeviceAddress rayDispatchIndirectAddress_ = 0;
	uint32_t maxSourceInstances = 65536u;
	uint32_t maxSourceTransforms = 65536u;

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
	std::vector<VulkanUtils::VmaImage> gBufferMaterialImages;
	std::vector<vk::raii::ImageView> gBufferMaterialViews;
	std::vector<VulkanUtils::VmaImage> gBufferEmissiveImages;
	std::vector<vk::raii::ImageView> gBufferEmissiveViews;
	std::vector<VmaBuffer> gBufferSourceBuffers;
	std::vector<VulkanUtils::VmaImage> reflectionImages;
	std::vector<vk::raii::ImageView> reflectionViews;
	std::vector<VulkanUtils::VmaImage> filteredReflectionImages;
	std::vector<vk::raii::ImageView> filteredReflectionViews;
	std::array<std::vector<VulkanUtils::VmaImage>, 2> filteredReflectionHistoryImages;
	std::array<std::vector<vk::raii::ImageView>, 2> filteredReflectionHistoryViews;
	std::vector<VulkanUtils::VmaImage> lightingImages;
	std::vector<vk::raii::ImageView> lightingViews;
	std::vector<VulkanUtils::VmaImage> referenceImages;
	std::vector<vk::raii::ImageView> referenceViews;
	std::array<std::vector<VulkanUtils::VmaImage>, 2> taaHistoryImages;
	std::array<std::vector<vk::raii::ImageView>, 2> taaHistoryViews;
	std::vector<VulkanUtils::VmaImage> irradianceAtlasImages;
	std::vector<vk::raii::ImageView> irradianceAtlasViews;

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
	uint32_t cellToSurfelCapacity_ = 0;
	UISystem::SurfelPathTracerSettings settings_{};
	std::array<VmaBuffer, MAX_FRAMES_IN_FLIGHT> statsReadbackBuffers_;
	std::array<void *, MAX_FRAMES_IN_FLIGHT> statsReadbackMapped_{};
	mutable std::array<bool, MAX_FRAMES_IN_FLIGHT> statsReadbackValid_{};
};
} // namespace Laphria

#endif // LAPHRIAENGINE_SURFELPATHTRACERRESOURCES_H
