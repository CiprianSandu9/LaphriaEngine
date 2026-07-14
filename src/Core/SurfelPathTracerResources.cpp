#include "SurfelPathTracerResources.h"

#include "EngineAuxiliary.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <stdexcept>
#include <string>

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

void destroyBufferSet(std::vector<VulkanUtils::VmaBuffer> &buffers)
{
	for (auto &buffer : buffers)
	{
		buffer.reset();
	}
	buffers.clear();
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
	try
	{
		createPersistentBuffers(dev, settings);
	}
	catch (const std::exception &error)
	{
		LOGE("SurfelPathTracer persistent resource creation failed: %s", error.what());
		destroyPersistentBuffers();
		initialized_ = false;
		throw;
	}
	try
	{
		createExtentImages(dev, swapchain);
	}
	catch (const std::exception &error)
	{
		LOGE("SurfelPathTracer extent resource creation failed: %s", error.what());
		cleanupSwapchainResources();
		destroyPersistentBuffers();
		initialized_ = false;
		throw;
	}
	initialized_ = true;
}

void SurfelPathTracerResources::cleanupSwapchainResources()
{
	clearViewsThenDestroyImages(outputImageViews, outputImages);
	clearViewsThenDestroyImages(gBufferNormalViews, gBufferNormalImages);
	clearViewsThenDestroyImages(gBufferDepthViews, gBufferDepthImages);
	clearViewsThenDestroyImages(gBufferMotionMaterialViews, gBufferMotionMaterialImages);
	clearViewsThenDestroyImages(gBufferAlbedoViews, gBufferAlbedoImages);
	clearViewsThenDestroyImages(gBufferMaterialViews, gBufferMaterialImages);
	clearViewsThenDestroyImages(gBufferEmissiveViews, gBufferEmissiveImages);
	destroyBufferSet(gBufferSourceBuffers);
	clearViewsThenDestroyImages(reflectionViews, reflectionImages);
	clearViewsThenDestroyImages(filteredReflectionViews, filteredReflectionImages);
	for (size_t bank = 0; bank < filteredReflectionHistoryImages.size(); ++bank)
	{
		clearViewsThenDestroyImages(filteredReflectionHistoryViews[bank], filteredReflectionHistoryImages[bank]);
	}
	clearViewsThenDestroyImages(lightingViews, lightingImages);
	clearViewsThenDestroyImages(referenceViews, referenceImages);
	for (size_t bank = 0; bank < taaHistoryImages.size(); ++bank)
	{
		clearViewsThenDestroyImages(taaHistoryViews[bank], taaHistoryImages[bank]);
	}
	clearViewsThenDestroyImages(irradianceAtlasViews, irradianceAtlasImages);
	clearViewsThenDestroyImages(surfelDepthAtlasViews, surfelDepthAtlasImages);
}

void SurfelPathTracerResources::recreateSwapchainResources(const VulkanDevice &dev,
                                                           const SwapchainManager &swapchain)
{
	cleanupSwapchainResources();
	try
	{
		createExtentImages(dev, swapchain);
	}
	catch (const std::exception &error)
	{
		LOGE("SurfelPathTracer extent resource creation failed: %s", error.what());
		cleanupSwapchainResources();
		initialized_ = false;
		throw;
	}
	initialized_ = true;
}

void SurfelPathTracerResources::resetPersistentResources(
    const VulkanDevice &dev,
    const UISystem::SurfelPathTracerSettings &settings)
{
	destroyPersistentBuffers();
	initialized_ = false;
	settings_ = settings;
	try
	{
		createPersistentBuffers(dev, settings);
	}
	catch (const std::exception &error)
	{
		LOGE("SurfelPathTracer persistent resource creation failed: %s", error.what());
		destroyPersistentBuffers();
		initialized_ = false;
		throw;
	}
	initialized_ = true;
}

void SurfelPathTracerResources::destroy()
{
	cleanupSwapchainResources();
	destroyPersistentBuffers();
	initialized_ = false;
}

void SurfelPathTracerResources::recordStatsReadback(
    const vk::raii::CommandBuffer &commandBuffer,
    uint32_t frameIndex) const
{
	if (frameIndex >= MAX_FRAMES_IN_FLIGHT || !statsReadbackBuffers_[frameIndex].valid())
	{
		return;
	}

	vk::BufferMemoryBarrier2 countersToTransfer{
	    .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader |
	                    vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
	    .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
	    .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
	    .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
	    .buffer = *countersBuffer,
	    .offset = 0,
	    .size = sizeof(SurfelPathTracerCounters)};
	vk::DependencyInfo countersToTransferDependency{
	    .bufferMemoryBarrierCount = 1,
	    .pBufferMemoryBarriers = &countersToTransfer};
	commandBuffer.pipelineBarrier2(countersToTransferDependency);

	vk::BufferCopy copyRegion{.size = sizeof(SurfelPathTracerCounters)};
	commandBuffer.copyBuffer(*countersBuffer, *statsReadbackBuffers_[frameIndex], copyRegion);

	std::array<vk::BufferMemoryBarrier2, 2> copyCompletionBarriers = {
	    vk::BufferMemoryBarrier2{
	    .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
	    .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
	    .dstStageMask = vk::PipelineStageFlagBits2::eHost,
	    .dstAccessMask = vk::AccessFlagBits2::eHostRead,
	    .buffer = *statsReadbackBuffers_[frameIndex],
	    .offset = 0,
	    .size = sizeof(SurfelPathTracerCounters)},
	    vk::BufferMemoryBarrier2{
	    .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
	    .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
	    .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader |
	                    vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
	    .dstAccessMask = vk::AccessFlagBits2::eShaderRead |
	                     vk::AccessFlagBits2::eShaderWrite,
	    .buffer = *countersBuffer,
	    .offset = 0,
	    .size = sizeof(SurfelPathTracerCounters)}};
	vk::DependencyInfo transferToHostDependency{
	    .bufferMemoryBarrierCount = static_cast<uint32_t>(copyCompletionBarriers.size()),
	    .pBufferMemoryBarriers = copyCompletionBarriers.data()};
	commandBuffer.pipelineBarrier2(transferToHostDependency);
	statsReadbackValid_[frameIndex] = true;
}

UISystem::SurfelPathTracerStats SurfelPathTracerResources::readStats(uint32_t frameIndex) const
{
	UISystem::SurfelPathTracerStats stats{};
	if (frameIndex >= MAX_FRAMES_IN_FLIGHT || !statsReadbackValid_[frameIndex] ||
	    !statsReadbackMapped_[frameIndex])
	{
		return stats;
	}

	const auto *counters = static_cast<const SurfelPathTracerCounters *>(statsReadbackMapped_[frameIndex]);
	const uint32_t deadSurfels = std::min(counters->deadSurfels, settings_.maxSurfels);
	stats.deadSurfels = deadSurfels;
	stats.aliveSurfels = settings_.maxSurfels - deadSurfels;
	stats.dirtySurfels = counters->dirtySurfels;
	stats.requestedRays = counters->requestedRays;
	stats.rayBudget = settings_.maxRaysPerFrame;
	stats.filledCells = counters->filledCells;
	stats.rejectedStores = counters->rejectedStores;
	stats.recycledSurfels = counters->recycledSurfels;
	stats.spawnedSurfels = counters->spawnedSurfels;
	stats.removedSurfels = counters->removedSurfels;
	stats.guidedRays = counters->guidedRays;
	stats.cosineRays = counters->cosineRays;
	stats.surfelTerminationAttempts = counters->surfelTerminationAttempts;
	stats.surfelTerminationHits = counters->surfelTerminationHits;
	stats.pathMisses = counters->pathMisses;
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
	                        vk::MemoryPropertyFlags memoryFlags,
	                        vk::BufferUsageFlags usageFlags,
	                        const char *resourceName) {
		try
		{
			VulkanUtils::createBuffer(dev.logicalDevice, dev.physicalDevice, size,
			                          usageFlags,
			                          memoryFlags, buffer);
		}
		catch (const std::exception &ex)
		{
			throw std::runtime_error(std::string(resourceName) + ": " + ex.what());
		}
	};

	createBuffer(sizeof(SurfelPathTracerCounters), countersBuffer,
	             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
	             vk::BufferUsageFlagBits::eStorageBuffer |
	                 vk::BufferUsageFlagBits::eTransferSrc |
	                 vk::BufferUsageFlagBits::eIndirectBuffer |
	                 vk::BufferUsageFlagBits::eShaderDeviceAddress,
	             "SurfelPathTracer.CountersBuffer");
	rayDispatchIndirectAddress_ = VulkanUtils::getBufferDeviceAddress(dev.logicalDevice, countersBuffer) +
	                              offsetof(SurfelPathTracerCounters, indirectRayWidth);
	mappedCounters = countersBuffer.memory.mapMemory(0, sizeof(SurfelPathTracerCounters));
	std::memset(mappedCounters, 0, sizeof(SurfelPathTracerCounters));
	auto *initialCounters = static_cast<SurfelPathTracerCounters *>(mappedCounters);
	initialCounters->aliveSurfels = 0;
	initialCounters->deadSurfels = maxSurfels;
	initialCounters->indirectRayHeight = 1;
	initialCounters->indirectRayDepth = 1;
	for (uint32_t frameIndex = 0; frameIndex < MAX_FRAMES_IN_FLIGHT; ++frameIndex)
	{
		createBuffer(sizeof(SurfelPathTracerCounters), statsReadbackBuffers_[frameIndex],
		             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		             vk::BufferUsageFlagBits::eTransferDst,
		             "SurfelPathTracer.StatsReadbackBuffer");
		statsReadbackMapped_[frameIndex] = statsReadbackBuffers_[frameIndex].memory.mapMemory(
		    0, sizeof(SurfelPathTracerCounters));
		std::memset(statsReadbackMapped_[frameIndex], 0, sizeof(SurfelPathTracerCounters));
		statsReadbackValid_[frameIndex] = false;
	}

	createBuffer(byteSize(maxSurfels, sizeof(SurfelPathTracerSurfel)), surfelBuffer,
	             vk::MemoryPropertyFlagBits::eDeviceLocal,
	             vk::BufferUsageFlagBits::eStorageBuffer,
	             "SurfelPathTracer.SurfelBuffer");
	createBuffer(byteSize(maxSurfels, sizeof(SurfelPathTracerSource)), surfelSourceBuffer,
	             vk::MemoryPropertyFlagBits::eDeviceLocal,
	             vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
	             "SurfelPathTracer.SourceBuffer");
	createBuffer(byteSize(maxSourceInstances, sizeof(SurfelPathTracerSourceInstance)), sourceInstanceBuffer,
	             vk::MemoryPropertyFlagBits::eDeviceLocal,
	             vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
	             "SurfelPathTracer.SourceInstanceBuffer");
	createBuffer(byteSize(maxSourceTransforms, sizeof(SurfelPathTracerSourceTransform)), sourceTransformBuffer,
	             vk::MemoryPropertyFlagBits::eDeviceLocal,
	             vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
	             "SurfelPathTracer.SourceTransformBuffer");
	createBuffer(byteSize(maxSurfels, sizeof(uint32_t)), aliveBuffer,
	             vk::MemoryPropertyFlagBits::eDeviceLocal,
	             vk::BufferUsageFlagBits::eStorageBuffer,
	             "SurfelPathTracer.AliveBuffer");
	createBuffer(byteSize(maxSurfels, sizeof(uint32_t)), deadBuffer,
	             vk::MemoryPropertyFlagBits::eDeviceLocal,
	             vk::BufferUsageFlagBits::eStorageBuffer,
	             "SurfelPathTracer.DeadBuffer");
	createBuffer(byteSize(maxSurfels, sizeof(uint32_t)), dirtyBuffer,
	             vk::MemoryPropertyFlagBits::eDeviceLocal,
	             vk::BufferUsageFlagBits::eStorageBuffer,
	             "SurfelPathTracer.DirtyBuffer");
	createBuffer(byteSize(static_cast<uint64_t>(maxSurfels) * 4u, sizeof(uint32_t)), recycleBuffer,
	             vk::MemoryPropertyFlagBits::eDeviceLocal,
	             vk::BufferUsageFlagBits::eStorageBuffer,
	             "SurfelPathTracer.RecycleBuffer");
	createBuffer(byteSize(static_cast<uint64_t>(maxRays) * 8u, sizeof(uint32_t)), rayBuffer,
	             vk::MemoryPropertyFlagBits::eDeviceLocal,
	             vk::BufferUsageFlagBits::eStorageBuffer,
	             "SurfelPathTracer.RayBuffer");
	createBuffer(byteSize(cellCount, sizeof(SurfelPathTracerCellInfo)), cellInfoBuffer,
	             vk::MemoryPropertyFlagBits::eDeviceLocal,
	             vk::BufferUsageFlagBits::eStorageBuffer,
	             "SurfelPathTracer.CellInfoBuffer");
	createBuffer(byteSize(cellCounterCount, sizeof(uint32_t)), cellCounterBuffer,
	             vk::MemoryPropertyFlagBits::eDeviceLocal,
	             vk::BufferUsageFlagBits::eStorageBuffer,
	             "SurfelPathTracer.CellCounterBuffer");
	createBuffer(byteSize(cellToSurfelCount, sizeof(uint32_t)), cellToSurfelBuffer,
	             vk::MemoryPropertyFlagBits::eDeviceLocal,
	             vk::BufferUsageFlagBits::eStorageBuffer,
	             "SurfelPathTracer.CellToSurfelBuffer");
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
	createStorageImageSet(dev, width, height, vk::Format::eR16G16B16A16Sfloat, gBufferAlbedoImages, gBufferAlbedoViews);
	createStorageImageSet(dev, width, height, vk::Format::eR16G16B16A16Sfloat, gBufferMaterialImages, gBufferMaterialViews);
	createStorageImageSet(dev, width, height, vk::Format::eR16G16B16A16Sfloat, gBufferEmissiveImages, gBufferEmissiveViews);
	destroyBufferSet(gBufferSourceBuffers);
	gBufferSourceBuffers.reserve(MAX_FRAMES_IN_FLIGHT);
	for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
	{
		VmaBuffer gBufferSourceBuffer{};
		VulkanUtils::createBuffer(dev.logicalDevice,
		                          dev.physicalDevice,
		                          byteSize(static_cast<uint64_t>(width) * height,
		                                   sizeof(SurfelPathTracerPixelSource)),
		                          vk::BufferUsageFlagBits::eStorageBuffer |
		                              vk::BufferUsageFlagBits::eTransferDst,
		                          vk::MemoryPropertyFlagBits::eDeviceLocal,
		                          gBufferSourceBuffer);
		gBufferSourceBuffers.push_back(std::move(gBufferSourceBuffer));
	}
	const uint32_t halfWidth = std::max((width + 1u) / 2u, 1u);
	const uint32_t halfHeight = std::max((height + 1u) / 2u, 1u);
	createStorageImageSet(dev, halfWidth, halfHeight,
	                      vk::Format::eR16G16B16A16Sfloat, reflectionImages, reflectionViews);
	createStorageImageSet(dev, halfWidth, halfHeight,
	                      vk::Format::eR16G16B16A16Sfloat, filteredReflectionImages, filteredReflectionViews);
	for (size_t bank = 0; bank < filteredReflectionHistoryImages.size(); ++bank)
	{
		createStorageImageSet(dev, halfWidth, halfHeight, vk::Format::eR16G16B16A16Sfloat,
		                      filteredReflectionHistoryImages[bank], filteredReflectionHistoryViews[bank]);
	}
	createStorageImageSet(dev, width, height, vk::Format::eR16G16B16A16Sfloat, lightingImages, lightingViews);
	createStorageImageSet(dev, width, height, vk::Format::eR16G16B16A16Sfloat, referenceImages, referenceViews);
	for (size_t bank = 0; bank < taaHistoryImages.size(); ++bank)
	{
		createStorageImageSet(dev, width, height, vk::Format::eR16G16B16A16Sfloat,
		                      taaHistoryImages[bank], taaHistoryViews[bank]);
	}

	const uint32_t atlasWidth = std::clamp(settings_.irradianceAtlasWidth, 512u, 4096u);
	const uint32_t atlasHeight = std::clamp(settings_.irradianceAtlasHeight, 512u, 4096u);
	const uint32_t atlasTileSize = std::max(settings_.atlasTileSize, 1u);
	const uint64_t tileCount = static_cast<uint64_t>(settings_.maxSurfels);
	const uint64_t tilesPerRow = std::max<uint64_t>(atlasWidth / atlasTileSize, 1u);
	const uint64_t requiredRows = (tileCount + tilesPerRow - 1u) / tilesPerRow;
	const uint64_t requiredHeight = requiredRows * atlasTileSize;
	if (requiredHeight > atlasHeight)
	{
		throw std::runtime_error("SurfelPathTracer irradiance atlas is too small for maxSurfels");
	}

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
	rayDispatchIndirectAddress_ = 0;
	for (uint32_t frameIndex = 0; frameIndex < MAX_FRAMES_IN_FLIGHT; ++frameIndex)
	{
		if (statsReadbackMapped_[frameIndex])
		{
			statsReadbackBuffers_[frameIndex].memory.unmapMemory();
		}
		statsReadbackMapped_[frameIndex] = nullptr;
		statsReadbackValid_[frameIndex] = false;
		statsReadbackBuffers_[frameIndex].reset();
	}
	destroyBuffers({&countersBuffer, &surfelBuffer, &aliveBuffer, &deadBuffer, &dirtyBuffer,
	                &recycleBuffer, &rayBuffer, &cellInfoBuffer, &cellCounterBuffer,
	                &cellToSurfelBuffer, &surfelSourceBuffer, &sourceInstanceBuffer,
	                &sourceTransformBuffer});
	cellCount_ = 0;
	needsPersistentReset_ = true;
}
