#ifndef LAPHRIAENGINE_SURFELPATHTRACERPASSES_H
#define LAPHRIAENGINE_SURFELPATHTRACERPASSES_H

#include <cstdint>

#include <vulkan/vulkan_raii.hpp>

#include "SurfelPathTracerPipelines.h"
#include "SurfelPathTracerResources.h"

namespace Laphria
{
class SurfelPathTracerPasses
{
  public:
	void recordSkyPass(const vk::raii::CommandBuffer &commandBuffer,
	                   const SurfelPathTracerPipelines &pipelines,
	                   const SurfelPathTracerResources &resources,
	                   vk::DescriptorSet imageSet,
	                   vk::DescriptorSet globalSet,
	                   uint32_t frameIndex,
	                   vk::Extent2D extent) const;

	void recordFinalBlit(const vk::raii::CommandBuffer &commandBuffer,
	                     const SurfelPathTracerResources &resources,
	                     vk::Image swapchainImage,
	                     uint32_t frameIndex,
	                     vk::Extent2D extent) const;
};
} // namespace Laphria

#endif // LAPHRIAENGINE_SURFELPATHTRACERPASSES_H
