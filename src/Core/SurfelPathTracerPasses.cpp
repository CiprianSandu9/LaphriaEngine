#include "SurfelPathTracerPasses.h"

#include <algorithm>
#include <array>

using namespace Laphria;

namespace
{
constexpr uint32_t kLinearWorkgroupSize = 64;

struct SurfelPreparePushConstants
{
	uint32_t maxSurfels = 0;
	uint32_t resetPersistent = 0;
	uint32_t cellCount = 0;
	uint32_t perCellSurfelLimit = 0;
};

struct SurfelUpdatePushConstants
{
	uint32_t maxSurfels = 0;
	float cellSize = 1.0f;
	uint32_t cellDimension = 1;
	uint32_t pad0 = 0;
};

struct SurfelCellInfoPushConstants
{
	uint32_t cellCount = 0;
	uint32_t perCellSurfelLimit = 0;
	uint32_t pad0 = 0;
	uint32_t pad1 = 0;
};

struct SurfelCellToSurfelPushConstants
{
	uint32_t maxSurfels = 0;
	float cellSize = 1.0f;
	uint32_t cellDimension = 1;
	uint32_t perCellSurfelLimit = 0;
};

uint32_t linearGroupCount(uint64_t workItemCount)
{
	return static_cast<uint32_t>((std::max<uint64_t>(workItemCount, 1u) + kLinearWorkgroupSize - 1u) /
	                             kLinearWorkgroupSize);
}

void bindComputeStorageSet(const vk::raii::CommandBuffer &commandBuffer,
                           const SurfelPathTracerPipelines &pipelines,
                           const vk::raii::Pipeline &pipeline,
                           vk::DescriptorSet imageSet)
{
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
	const std::array descriptorSets = {imageSet};
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
	                                 *pipelines.computePipelineLayout,
	                                 0,
	                                 descriptorSets,
	                                 nullptr);
}
} // namespace

void SurfelPathTracerPasses::recordGBufferPass(const vk::raii::CommandBuffer &commandBuffer,
                                               const SurfelPathTracerPipelines &pipelines,
                                               vk::DescriptorSet rtSet,
                                               vk::DescriptorSet storageSet,
                                               vk::DescriptorSet globalSet,
                                               vk::Extent2D extent) const
{
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *pipelines.gBufferRayTracingPipeline);
	const std::array descriptorSets = {rtSet, storageSet, globalSet};
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR,
	                                 *pipelines.rayTracingPipelineLayout,
	                                 0,
	                                 descriptorSets,
	                                 nullptr);

	vk::StridedDeviceAddressRegionKHR callableRegion{};
	commandBuffer.traceRaysKHR(pipelines.gBufferSbt.raygenRegion,
	                           pipelines.gBufferSbt.missRegion,
	                           pipelines.gBufferSbt.hitRegion,
	                           callableRegion,
	                           extent.width,
	                           extent.height,
	                           1);
}

void SurfelPathTracerPasses::recordPreparePass(const vk::raii::CommandBuffer &commandBuffer,
                                               const SurfelPathTracerPipelines &pipelines,
                                               vk::DescriptorSet imageSet,
                                               uint32_t maxSurfels,
                                               bool resetPersistent,
                                               uint32_t cellCount,
                                               uint32_t perCellSurfelLimit) const
{
	bindComputeStorageSet(commandBuffer, pipelines, pipelines.preparePipeline, imageSet);

	const SurfelPreparePushConstants push{
	    .maxSurfels = std::max(maxSurfels, 1u),
	    .resetPersistent = resetPersistent ? 1u : 0u,
	    .cellCount = cellCount,
	    .perCellSurfelLimit = std::max(perCellSurfelLimit, 1u)};
	commandBuffer.pushConstants<SurfelPreparePushConstants>(*pipelines.computePipelineLayout,
	                                                        vk::ShaderStageFlagBits::eCompute,
	                                                        0,
	                                                        push);

	const uint64_t cellEntryCount = static_cast<uint64_t>(push.cellCount) * push.perCellSurfelLimit;
	const uint64_t cellCounterCount = 1ull + static_cast<uint64_t>(push.cellCount) * 2ull;
	const uint64_t resetEntryCount = static_cast<uint64_t>(push.maxSurfels) * 4ull;
	const uint64_t workItemCount = std::max({static_cast<uint64_t>(push.maxSurfels),
	                                         cellEntryCount,
	                                         cellCounterCount,
	                                         resetEntryCount});
	commandBuffer.dispatch(linearGroupCount(workItemCount), 1, 1);
}

void SurfelPathTracerPasses::recordUpdatePass(const vk::raii::CommandBuffer &commandBuffer,
                                              const SurfelPathTracerPipelines &pipelines,
                                              vk::DescriptorSet imageSet,
                                              uint32_t maxSurfels,
                                              float cellSize,
                                              uint32_t cellDimension) const
{
	bindComputeStorageSet(commandBuffer, pipelines, pipelines.updatePipeline, imageSet);

	const SurfelUpdatePushConstants push{
	    .maxSurfels = std::max(maxSurfels, 1u),
	    .cellSize = std::max(cellSize, 0.0001f),
	    .cellDimension = std::max(cellDimension, 1u),
	    .pad0 = 0};
	commandBuffer.pushConstants<SurfelUpdatePushConstants>(*pipelines.computePipelineLayout,
	                                                       vk::ShaderStageFlagBits::eCompute,
	                                                       0,
	                                                       push);
	commandBuffer.dispatch(linearGroupCount(push.maxSurfels), 1, 1);
}

void SurfelPathTracerPasses::recordCellInfoPass(const vk::raii::CommandBuffer &commandBuffer,
                                                const SurfelPathTracerPipelines &pipelines,
                                                vk::DescriptorSet imageSet,
                                                uint32_t cellCount,
                                                uint32_t perCellSurfelLimit) const
{
	bindComputeStorageSet(commandBuffer, pipelines, pipelines.cellInfoPipeline, imageSet);

	const SurfelCellInfoPushConstants push{
	    .cellCount = cellCount,
	    .perCellSurfelLimit = std::max(perCellSurfelLimit, 1u),
	    .pad0 = 0,
	    .pad1 = 0};
	commandBuffer.pushConstants<SurfelCellInfoPushConstants>(*pipelines.computePipelineLayout,
	                                                         vk::ShaderStageFlagBits::eCompute,
	                                                         0,
	                                                         push);
	commandBuffer.dispatch(linearGroupCount(push.cellCount), 1, 1);
}

void SurfelPathTracerPasses::recordCellToSurfelPass(const vk::raii::CommandBuffer &commandBuffer,
                                                    const SurfelPathTracerPipelines &pipelines,
                                                    vk::DescriptorSet imageSet,
                                                    uint32_t maxSurfels,
                                                    float cellSize,
                                                    uint32_t cellDimension,
                                                    uint32_t perCellSurfelLimit) const
{
	bindComputeStorageSet(commandBuffer, pipelines, pipelines.cellToSurfelPipeline, imageSet);

	const SurfelCellToSurfelPushConstants push{
	    .maxSurfels = std::max(maxSurfels, 1u),
	    .cellSize = std::max(cellSize, 0.0001f),
	    .cellDimension = std::max(cellDimension, 1u),
	    .perCellSurfelLimit = std::max(perCellSurfelLimit, 1u)};
	commandBuffer.pushConstants<SurfelCellToSurfelPushConstants>(*pipelines.computePipelineLayout,
	                                                             vk::ShaderStageFlagBits::eCompute,
	                                                             0,
	                                                             push);
	commandBuffer.dispatch(linearGroupCount(push.maxSurfels), 1, 1);
}

void SurfelPathTracerPasses::recordStorageBarrierComputeToCompute(
    const vk::raii::CommandBuffer &commandBuffer) const
{
	vk::MemoryBarrier2 storageBufferBarrier{
	    .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
	    .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
	    .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
	    .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite};
	vk::DependencyInfo dependency{
	    .memoryBarrierCount = 1,
	    .pMemoryBarriers = &storageBufferBarrier};
	commandBuffer.pipelineBarrier2(dependency);
}

void SurfelPathTracerPasses::recordSkyPass(const vk::raii::CommandBuffer &commandBuffer,
                                           const SurfelPathTracerPipelines &pipelines,
                                           const SurfelPathTracerResources &resources,
                                           vk::DescriptorSet imageSet,
                                           vk::DescriptorSet globalSet,
                                           uint32_t frameIndex,
                                           vk::Extent2D extent) const
{
	(void)resources;
	(void)frameIndex;

	commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelines.skyPipeline);
	const std::array descriptorSets = {imageSet, globalSet};
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
	                                 *pipelines.skyPipelineLayout,
	                                 0,
	                                 descriptorSets,
	                                 nullptr);

	const uint32_t groupCountX = (extent.width + 15u) / 16u;
	const uint32_t groupCountY = (extent.height + 15u) / 16u;
	commandBuffer.dispatch(groupCountX, groupCountY, 1);
}

void SurfelPathTracerPasses::recordFinalBlit(const vk::raii::CommandBuffer &commandBuffer,
                                             const SurfelPathTracerResources &resources,
                                             vk::Image swapchainImage,
                                             uint32_t frameIndex,
                                             vk::Extent2D extent) const
{
	vk::ImageBlit blitRegion{
	    .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
	    .srcOffsets     = {{vk::Offset3D{0, 0, 0},
	                        vk::Offset3D{static_cast<int32_t>(extent.width),
	                                     static_cast<int32_t>(extent.height),
	                                     1}}},
	    .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
	    .dstOffsets     = {{vk::Offset3D{0, 0, 0},
	                        vk::Offset3D{static_cast<int32_t>(extent.width),
	                                     static_cast<int32_t>(extent.height),
	                                     1}}}};

	commandBuffer.blitImage(*resources.outputImages[frameIndex],
	                        vk::ImageLayout::eTransferSrcOptimal,
	                        swapchainImage,
	                        vk::ImageLayout::eTransferDstOptimal,
	                        blitRegion,
	                        vk::Filter::eLinear);
}
