#include "SurfelPathTracerResources.h"

#include <algorithm>
#include <cstring>
#include <initializer_list>

using namespace Laphria;

namespace
{
vk::DeviceSize byteSize(uint64_t elementCount, uint64_t elementSize)
{
	return static_cast<vk::DeviceSize>(std::max<uint64_t>(elementCount * elementSize, 1u));
}

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

UISystem::SurfelPathTracerSettings persistentCapacitySettings(
    const UISystem::SurfelPathTracerSettings &settings)
{
	UISystem::SurfelPathTracerSettings capacity = settings;
	capacity.maxSurfels = std::max(capacity.maxSurfels, 1u);
	capacity.maxRaysPerFrame = std::max(capacity.maxRaysPerFrame, 1u);
	capacity.cellDimension = std::clamp(capacity.cellDimension, 8u, 128u);
	capacity.perCellSurfelLimit = std::clamp(capacity.perCellSurfelLimit, 1u, 256u);
	return capacity;
}

void destroyImages(std::vector<VulkanUtils::VmaImage> &images)
{
	for (auto &image : images)
	{
		image.reset();
	}
	images.clear();
}

void clearViewsThenDestroyImages(std::vector<vk::raii::ImageView> &views,
                                 std::vector<VulkanUtils::VmaImage> &images)
{
	views.clear();
	destroyImages(images);
}

void createStorageImageSet(const VulkanDevice &dev,
                           uint32_t width,
                           uint32_t height,
                           vk::Format format,
                           std::vector<VulkanUtils::VmaImage> &images,
                           std::vector<vk::raii::ImageView> &views)
{
	views.clear();
	destroyImages(images);
	images.reserve(MAX_FRAMES_IN_FLIGHT);
	views.reserve(MAX_FRAMES_IN_FLIGHT);
	for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
	{
		VulkanUtils::VmaImage image{};
		VulkanUtils::createImage(dev.logicalDevice, dev.physicalDevice, width, height, format,
		                         vk::ImageTiling::eOptimal,
		                         vk::ImageUsageFlagBits::eStorage |
		                             vk::ImageUsageFlagBits::eTransferSrc |
		                             vk::ImageUsageFlagBits::eTransferDst,
		                         vk::MemoryPropertyFlagBits::eDeviceLocal, image);
		images.push_back(std::move(image));
		views.push_back(VulkanUtils::createImageView(dev.logicalDevice, *images.back(), format,
		                                             vk::ImageAspectFlagBits::eColor));
	}
}
} // namespace

SurfelPathTracerResources::~SurfelPathTracerResources()
{
	destroy();
}

void SurfelPathTracerResources::init(const VulkanDevice &dev,
                                     const SwapchainManager &swapchain,
                                     const UISystem::SurfelPathTracerSettings &settings)
{
	destroy();
	settings_ = settings;
	createPersistentBuffers(dev, settings);
	createExtentImages(dev, swapchain);
	initialized_ = true;
}

void SurfelPathTracerResources::cleanupSwapchainResources()
{
	clearViewsThenDestroyImages(outputImageViews, outputImages);
	clearViewsThenDestroyImages(gBufferNormalViews, gBufferNormalImages);
	clearViewsThenDestroyImages(gBufferDepthViews, gBufferDepthImages);
	clearViewsThenDestroyImages(gBufferMotionMaterialViews, gBufferMotionMaterialImages);
	clearViewsThenDestroyImages(reflectionViews, reflectionImages);
	clearViewsThenDestroyImages(filteredReflectionViews, filteredReflectionImages);
	clearViewsThenDestroyImages(lightingViews, lightingImages);
	clearViewsThenDestroyImages(taaHistoryViews, taaHistoryImages);
	clearViewsThenDestroyImages(irradianceAtlasViews, irradianceAtlasImages);
	clearViewsThenDestroyImages(surfelDepthAtlasViews, surfelDepthAtlasImages);
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

bool SurfelPathTracerResources::needsPersistentResourceRecreate(
    const UISystem::SurfelPathTracerSettings &settings) const
{
	const UISystem::SurfelPathTracerSettings capacity = persistentCapacitySettings(settings);
	return capacity.maxSurfels != settings_.maxSurfels ||
	       capacity.maxRaysPerFrame != settings_.maxRaysPerFrame ||
	       capacity.cellDimension != settings_.cellDimension ||
	       capacity.perCellSurfelLimit != settings_.perCellSurfelLimit;
}

SurfelPathTracerCellAddress SurfelPathTracerResources::cellAddressForPosition(
    const glm::vec3 &position,
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

void SurfelPathTracerResources::createPersistentBuffers(
    const VulkanDevice &dev,
    const UISystem::SurfelPathTracerSettings &settings)
{
	settings_ = persistentCapacitySettings(settings);
	const uint32_t maxSurfels = settings_.maxSurfels;
	const uint32_t maxRays = settings_.maxRaysPerFrame;
	const uint32_t cellDimension = settings_.cellDimension;
	const uint32_t perCellSurfelLimit = settings_.perCellSurfelLimit;
	settings_.cellDimension = cellDimension;
	settings_.perCellSurfelLimit = perCellSurfelLimit;

	const uint64_t cellDimension64 = cellDimension;
	const uint64_t cellCount = cellDimension64 * cellDimension64 * cellDimension64;
	cellCount_ = static_cast<uint32_t>(cellCount);
	const uint64_t cellToSurfelCount = cellCount * perCellSurfelLimit;
	const uint64_t cellCounterCount = 1u + cellCount * 2u;

	auto createBuffer = [&](vk::DeviceSize size,
	                        VulkanUtils::VmaBuffer &buffer,
	                        vk::MemoryPropertyFlags memoryFlags) {
		VulkanUtils::createBuffer(dev.logicalDevice, dev.physicalDevice, size,
		                          vk::BufferUsageFlagBits::eStorageBuffer,
		                          memoryFlags, buffer);
	};

	createBuffer(sizeof(SurfelPathTracerCounters), countersBuffer,
	             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	mappedCounters = countersBuffer.memory.mapMemory(0, sizeof(SurfelPathTracerCounters));
	std::memset(mappedCounters, 0, sizeof(SurfelPathTracerCounters));
	auto *initialCounters = static_cast<SurfelPathTracerCounters *>(mappedCounters);
	initialCounters->aliveSurfels = 0;
	initialCounters->deadSurfels = maxSurfels;

	createBuffer(byteSize(maxSurfels, sizeof(SurfelPathTracerSurfel)), surfelBuffer,
	             vk::MemoryPropertyFlagBits::eDeviceLocal);
	createBuffer(byteSize(maxSurfels, sizeof(uint32_t)), aliveBuffer,
	             vk::MemoryPropertyFlagBits::eDeviceLocal);
	createBuffer(byteSize(maxSurfels, sizeof(uint32_t)), deadBuffer,
	             vk::MemoryPropertyFlagBits::eDeviceLocal);
	createBuffer(byteSize(maxSurfels, sizeof(uint32_t)), dirtyBuffer,
	             vk::MemoryPropertyFlagBits::eDeviceLocal);
	createBuffer(byteSize(static_cast<uint64_t>(maxSurfels) * 4u, sizeof(uint32_t)), recycleBuffer,
	             vk::MemoryPropertyFlagBits::eDeviceLocal);
	createBuffer(byteSize(static_cast<uint64_t>(maxRays) * 8u, sizeof(uint32_t)), rayBuffer,
	             vk::MemoryPropertyFlagBits::eDeviceLocal);
	createBuffer(byteSize(cellCount, sizeof(SurfelPathTracerCellInfo)), cellInfoBuffer,
	             vk::MemoryPropertyFlagBits::eDeviceLocal);
	createBuffer(byteSize(cellCounterCount, sizeof(uint32_t)), cellCounterBuffer,
	             vk::MemoryPropertyFlagBits::eDeviceLocal);
	createBuffer(byteSize(cellToSurfelCount, sizeof(uint32_t)), cellToSurfelBuffer,
	             vk::MemoryPropertyFlagBits::eDeviceLocal);
	needsPersistentReset_ = true;
}

void SurfelPathTracerResources::createExtentImages(const VulkanDevice &dev,
                                                   const SwapchainManager &swapchain)
{
	const uint32_t width = std::max(swapchain.extent.width, 1u);
	const uint32_t height = std::max(swapchain.extent.height, 1u);
	createStorageImageSet(dev, width, height, vk::Format::eR16G16B16A16Sfloat, outputImages, outputImageViews);
	createStorageImageSet(dev, width, height, vk::Format::eR16G16B16A16Sfloat, gBufferNormalImages, gBufferNormalViews);
	createStorageImageSet(dev, width, height, vk::Format::eR32Sfloat, gBufferDepthImages, gBufferDepthViews);
	createStorageImageSet(dev, width, height, vk::Format::eR32G32B32A32Sfloat, gBufferMotionMaterialImages, gBufferMotionMaterialViews);
	const uint32_t halfWidth = std::max((width + 1u) / 2u, 1u);
	const uint32_t halfHeight = std::max((height + 1u) / 2u, 1u);
	createStorageImageSet(dev, halfWidth, halfHeight,
	                      vk::Format::eR16G16B16A16Sfloat, reflectionImages, reflectionViews);
	createStorageImageSet(dev, halfWidth, halfHeight,
	                      vk::Format::eR16G16B16A16Sfloat, filteredReflectionImages, filteredReflectionViews);
	createStorageImageSet(dev, width, height, vk::Format::eR16G16B16A16Sfloat, lightingImages, lightingViews);
	createStorageImageSet(dev, width, height, vk::Format::eR16G16B16A16Sfloat, taaHistoryImages, taaHistoryViews);

	const uint32_t atlasWidth = std::clamp(settings_.irradianceAtlasWidth, 512u, 4096u);
	const uint32_t atlasHeight = std::clamp(settings_.irradianceAtlasHeight, 512u, 4096u);
	createStorageImageSet(dev, atlasWidth, atlasHeight, vk::Format::eR16G16B16A16Sfloat,
	                      irradianceAtlasImages, irradianceAtlasViews);
	createStorageImageSet(dev, atlasWidth, atlasHeight, vk::Format::eR32Sfloat,
	                      surfelDepthAtlasImages, surfelDepthAtlasViews);
}

void SurfelPathTracerResources::destroyPersistentBuffers()
{
	if (mappedCounters)
	{
		countersBuffer.memory.unmapMemory();
	}
	mappedCounters = nullptr;
	destroyBuffers({&countersBuffer, &surfelBuffer, &aliveBuffer, &deadBuffer, &dirtyBuffer,
	                &recycleBuffer, &rayBuffer, &cellInfoBuffer, &cellCounterBuffer,
	                &cellToSurfelBuffer});
	cellCount_ = 0;
	needsPersistentReset_ = true;
}
