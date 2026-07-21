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
	                      vk::DescriptorSet globalSet,
	                      uint32_t maxSurfels,
	                      uint32_t maxRays,
	                      float cellSize,
	                      uint32_t cellDimension,
	                      uint32_t sourceTransformCount,
	                      uint32_t minRaysPerSurfel,
	                      uint32_t maxRaysPerSurfel,
	                      float varianceSensitivity,
	                      bool lockSurfels) const;

	void recordCellInfoPass(const vk::raii::CommandBuffer &commandBuffer,
	                        const SurfelPathTracerPipelines &pipelines,
	                        vk::DescriptorSet imageSet,
	                        uint32_t cellCount,
	                        uint32_t perCellSurfelLimit) const;

	void recordCellToSurfelPass(const vk::raii::CommandBuffer &commandBuffer,
	                            const SurfelPathTracerPipelines &pipelines,
	                            vk::DescriptorSet imageSet,
	                            vk::DescriptorSet globalSet,
	                            uint32_t maxSurfels,
	                            float cellSize,
	                            uint32_t cellDimension,
	                            uint32_t perCellSurfelLimit) const;

	void recordRaySchedulePass(const vk::raii::CommandBuffer &commandBuffer,
	                           const SurfelPathTracerPipelines &pipelines,
	                           vk::DescriptorSet imageSet,
	                           uint32_t cellCount,
	                           uint32_t maxSurfels,
	                           uint32_t maxRays,
	                           uint32_t minRaysPerSurfel,
	                           uint32_t maxRaysPerSurfel,
	                           float varianceSensitivity) const;

	void recordSurfelRayTracePass(const vk::raii::CommandBuffer &commandBuffer,
	                              const SurfelPathTracerPipelines &pipelines,
	                              const SurfelPathTracerResources &resources,
	                              vk::DescriptorSet rayTracingSet,
	                              vk::DescriptorSet storageSet,
	                              vk::DescriptorSet globalSet,
	                              uint32_t rayCount,
	                              bool useIndirectDispatch,
	                              bool enableGuidedSampling,
	                              uint32_t irradianceAtlasWidth,
	                              uint32_t activeMaxDepth,
	                              uint32_t sleepingMaxDepth,
	                              bool enableSurfelTermination,
	                              bool useOriginalStyleGiNormalization,
	                              uint32_t maxSurfelSamplesPerQuery,
	                              float cellSize,
	                              uint32_t cellDimension) const;

	void recordIntegratePass(const vk::raii::CommandBuffer &commandBuffer,
	                         const SurfelPathTracerPipelines &pipelines,
	                         vk::DescriptorSet imageSet,
	                         vk::DescriptorSet globalSet,
	                         uint32_t maxSurfels,
	                         bool enableRadianceSharing,
	                         uint32_t maxRadianceSharingSamples,
	                         uint32_t cellDimension,
	                         float cellSize,
	                         bool enableGuidedSampling) const;

	void recordEvaluatePass(const vk::raii::CommandBuffer &commandBuffer,
	                        const SurfelPathTracerPipelines &pipelines,
	                        vk::DescriptorSet imageSet,
	                        vk::DescriptorSet globalSet,
	                        SurfelPathTracerEvaluateMode mode,
	                        float cellSize,
	                        uint32_t cellDimension,
	                        uint32_t maxSurfels,
	                        float placementThreshold,
	                        float removalThreshold,
	                        float surfelTargetArea,
	                        float surfelMinRadius,
	                        float surfelMaxRadiusScale,
	                        uint32_t frameIndex,
	                        bool lockSurfels,
	                        bool enableSurfelPlacement,
	                        bool enableSurfelRemoval,
	                        uint32_t maxSurfelSamplesPerQuery,
	                        uint32_t perCellSurfelLimit,
	                        vk::Extent2D extent) const;

	void recordReflectionPass(const vk::raii::CommandBuffer &commandBuffer,
	                          const SurfelPathTracerPipelines &pipelines,
	                          const SurfelPathTracerResources &resources,
	                          vk::DescriptorSet rayTracingSet,
	                          vk::DescriptorSet storageSet,
	                          vk::DescriptorSet globalSet,
	                          bool enabled,
	                          float cellSize,
	                          uint32_t cellDimension,
	                          vk::Extent2D extent,
	                          uint32_t frameIndex,
	                          bool enableSurfelTermination,
	                          bool useOriginalStyleGiNormalization,
	                          uint32_t maxSurfelSamplesPerQuery) const;

	void recordReferencePass(const vk::raii::CommandBuffer &commandBuffer,
	                         const SurfelPathTracerPipelines &pipelines,
	                         vk::DescriptorSet rayTracingSet,
	                         vk::DescriptorSet storageSet,
	                         vk::DescriptorSet globalSet,
	                         bool enableReferenceValidation,
	                         uint32_t maxDepth,
	                         vk::Extent2D extent) const;

	void recordReflectionFilterPass(const vk::raii::CommandBuffer &commandBuffer,
	                                const SurfelPathTracerPipelines &pipelines,
	                                vk::DescriptorSet imageSet,
	                                bool enabled,
	                                bool resetHistory,
	                                vk::Extent2D extent,
	                                uint32_t frameIndex) const;

	void recordBilateralPass(const vk::raii::CommandBuffer &commandBuffer,
	                         const SurfelPathTracerPipelines &pipelines,
	                         vk::DescriptorSet imageSet,
	                         bool enabled,
	                         vk::Extent2D extent) const;

	void recordLightIntegratePass(const vk::raii::CommandBuffer &commandBuffer,
	                              const SurfelPathTracerPipelines &pipelines,
	                              vk::DescriptorSet imageSet,
	                              vk::DescriptorSet globalSet,
	                              bool enableDiffuseGi,
	                              bool enableReflections,
	                              bool useBilateralReflection,
	                              bool useOriginalStyleGiNormalization,
	                              float cellSize,
	                              float surfelMaxRadiusScale,
	                              uint32_t cellDimension,
	                              uint32_t perCellSurfelLimit,
	                              uint32_t debugView,
	                              vk::Extent2D extent) const;

	void recordTaaPass(const vk::raii::CommandBuffer &commandBuffer,
	                   const SurfelPathTracerPipelines &pipelines,
	                   vk::DescriptorSet imageSet,
	                   vk::DescriptorSet globalSet,
	                   bool enabled,
	                   bool resetHistory,
	                   vk::Extent2D extent,
	                   uint32_t frameIndex) const;

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
