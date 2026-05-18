#ifndef LAPHRIAENGINE_SURFELPATHTRACERPASSES_H
#define LAPHRIAENGINE_SURFELPATHTRACERPASSES_H

#include <cstdint>

#include <vulkan/vulkan_raii.hpp>

#include "SurfelPathTracerPipelines.h"
#include "SurfelPathTracerResources.h"

namespace Laphria
{
enum class SurfelPathTracerEvaluateMode : uint32_t
{
	Generate = 0,
	Resolve = 1
};

class SurfelPathTracerPasses
{
  public:
	void recordGBufferPass(const vk::raii::CommandBuffer &commandBuffer,
	                       const SurfelPathTracerPipelines &pipelines,
	                       vk::DescriptorSet rtSet,
	                       vk::DescriptorSet storageSet,
	                       vk::DescriptorSet globalSet,
	                       vk::Extent2D extent) const;

	void recordPreparePass(const vk::raii::CommandBuffer &commandBuffer,
	                       const SurfelPathTracerPipelines &pipelines,
	                       vk::DescriptorSet imageSet,
	                       uint32_t maxSurfels,
	                       bool resetPersistent,
	                       uint32_t cellCount,
	                       uint32_t perCellSurfelLimit) const;

	void recordUpdatePass(const vk::raii::CommandBuffer &commandBuffer,
	                      const SurfelPathTracerPipelines &pipelines,
	                      vk::DescriptorSet imageSet,
	                      uint32_t maxSurfels,
	                      uint32_t maxRays,
	                      float cellSize,
	                      uint32_t cellDimension) const;

	void recordCellInfoPass(const vk::raii::CommandBuffer &commandBuffer,
	                        const SurfelPathTracerPipelines &pipelines,
	                        vk::DescriptorSet imageSet,
	                        uint32_t cellCount,
	                        uint32_t perCellSurfelLimit) const;

	void recordCellToSurfelPass(const vk::raii::CommandBuffer &commandBuffer,
	                            const SurfelPathTracerPipelines &pipelines,
	                            vk::DescriptorSet imageSet,
	                            uint32_t maxSurfels,
	                            float cellSize,
	                            uint32_t cellDimension,
	                            uint32_t perCellSurfelLimit) const;

	void recordSurfelRayTracePass(const vk::raii::CommandBuffer &commandBuffer,
	                              const SurfelPathTracerPipelines &pipelines,
	                              const SurfelPathTracerResources &resources,
	                              vk::DescriptorSet rayTracingSet,
	                              vk::DescriptorSet storageSet,
	                              vk::DescriptorSet globalSet,
	                              uint32_t rayCount) const;

	void recordIntegratePass(const vk::raii::CommandBuffer &commandBuffer,
	                         const SurfelPathTracerPipelines &pipelines,
	                         vk::DescriptorSet imageSet,
	                         uint32_t maxSurfels) const;

	void recordEvaluatePass(const vk::raii::CommandBuffer &commandBuffer,
	                        const SurfelPathTracerPipelines &pipelines,
	                        vk::DescriptorSet imageSet,
	                        vk::DescriptorSet globalSet,
	                        SurfelPathTracerEvaluateMode mode,
	                        float cellSize,
	                        uint32_t cellDimension,
	                        uint32_t maxSurfels,
	                        vk::Extent2D extent) const;

	void recordImageBarrierGBufferToCompute(const vk::raii::CommandBuffer &commandBuffer,
	                                        const SurfelPathTracerResources &resources,
	                                        uint32_t frameIndex) const;

	void recordStorageBarrierComputeToCompute(const vk::raii::CommandBuffer &commandBuffer) const;

	void recordStorageBarrierComputeToRt(const vk::raii::CommandBuffer &commandBuffer) const;

	void recordStorageBarrierRtToCompute(const vk::raii::CommandBuffer &commandBuffer) const;

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
