#include "SurfelPathTracerPasses.h"

#include <array>

using namespace Laphria;

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
