#include "SurfelPathTracerPasses.h"

#include <algorithm>
#include <array>

using namespace Laphria;

namespace
{
constexpr uint32_t kLinearWorkgroupSize = 64;
constexpr vk::ShaderStageFlags kSurfelRtPushStages = vk::ShaderStageFlagBits::eRaygenKHR |
                                                     vk::ShaderStageFlagBits::eClosestHitKHR |
                                                     vk::ShaderStageFlagBits::eMissKHR |
                                                     vk::ShaderStageFlagBits::eAnyHitKHR;

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
	uint32_t maxRays = 0;
	uint32_t lockSurfels = 0;
	uint32_t minRaysPerSurfel = 1;
	uint32_t maxRaysPerSurfel = 1;
	float varianceSensitivity = 1.0f;
	uint32_t sourceTransformCount = 0;
	uint32_t perCellSurfelLimit = 1;
	uint32_t pad1 = 0;
	uint32_t pad2 = 0;
};

struct SurfelCellInfoPushConstants
{
	uint32_t cellCount = 0;
	uint32_t perCellSurfelLimit = 0;
	uint32_t mapCapacity = 0;
	uint32_t pad1 = 0;
};

struct SurfelCellToSurfelPushConstants
{
	uint32_t maxSurfels = 0;
	float cellSize = 1.0f;
	uint32_t cellDimension = 1;
	uint32_t perCellSurfelLimit = 0;
};

struct SurfelRaySchedulePushConstants
{
	uint32_t cellCount = 0;
	uint32_t maxSurfels = 0;
	uint32_t maxRays = 0;
	uint32_t minRaysPerSurfel = 1;
	uint32_t maxRaysPerSurfel = 1;
	float varianceSensitivity = 1.0f;
	uint32_t offscreenRayInterval = 4;
	float surfelSupportRadius = 0.25f;
};

struct SurfelRayTracePushConstants
{
	uint32_t rayCount = 0;
	uint32_t maxSurfels = 0;
	uint32_t enableGuidedSampling = 0;
	uint32_t irradianceAtlasWidth = 1;
	uint32_t activeMaxDepth = 1;
	uint32_t sleepingMaxDepth = 1;
	uint32_t enableSurfelTermination = 0;
	uint32_t maxSurfelSamplesPerQuery = 1;
	uint32_t cellDimension = 1;
	float cellSize = 1.0f;
	float surfelSupportRadius = 0.25f;
	uint32_t useOriginalStyleGiNormalization = 0;
};

struct SurfelIntegratePushConstants
{
	uint32_t maxSurfels = 0;
	uint32_t enableRadianceSharing = 0;
	uint32_t maxRadianceSharingSamples = 1;
	uint32_t cellDimension = 1;
	float cellSize = 1.0f;
	uint32_t enableGuidedSampling = 0;
	float surfelSupportRadius = 0.25f;
	uint32_t pad1 = 0;
};

struct SurfelEvaluatePushConstants
{
	uint32_t mode = 0;
	uint32_t width = 0;
	uint32_t height = 0;
	float cellSize = 1.0f;
	uint32_t cellDimension = 1;
	uint32_t maxSurfels = 0;
	float placementThreshold = 0.35f;
	float removalThreshold = 12.0f;
	float surfelTargetArea = 16.0f;
	float surfelMinRadius = 0.05f;
	float surfelMaxRadiusScale = 2.0f;
	uint32_t frameIndex = 0;
	uint32_t lockSurfels = 0;
	uint32_t enablePlacement = 1;
	uint32_t enableRemoval = 1;
	uint32_t maxSurfelSamplesPerQuery = 1;
	uint32_t perCellSurfelLimit = 1;
	float surfelSupportRadius = 0.25f;
};

struct SurfelReflectionPushConstants
{
	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t fullWidth = 0;
	uint32_t fullHeight = 0;
	uint32_t maxSurfels = 0;
	uint32_t cellDimension = 1;
	float cellSize = 1.0f;
	uint32_t frameIndex = 0;
	uint32_t enabled = 1;
	uint32_t enableSurfelTermination = 0;
	uint32_t maxSurfelSamplesPerQuery = 1;
	uint32_t useOriginalStyleGiNormalization = 0;
	float roughReflectionStart = 0.5f;
	float roughReflectionEnd = 0.7f;
	float surfelSupportRadius = 0.25f;
};

struct SurfelReflectionFilterPushConstants
{
	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t fullWidth = 0;
	uint32_t fullHeight = 0;
	uint32_t enabled = 1;
	uint32_t resetHistory = 1;
	uint32_t frameIndex = 0;
	uint32_t pad0 = 0;
};

struct SurfelReferencePushConstants
{
	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t maxDepth = 1;
	uint32_t enabled = 0;
};

struct SurfelBilateralPushConstants
{
	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t fullWidth = 0;
	uint32_t fullHeight = 0;
	uint32_t enabled = 1;
	uint32_t pad0 = 0;
	uint32_t pad1 = 0;
	uint32_t pad2 = 0;
};

struct SurfelLightIntegratePushConstants
{
	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t enableDiffuseGi = 1;
	uint32_t enableReflections = 1;
	uint32_t useBilateralReflection = 0;
	uint32_t debugView = 0;
	float cellSize = 1.0f;
	float surfelMaxRadiusScale = 2.0f;
	uint32_t cellDimension = 1;
	uint32_t perCellSurfelLimit = 1;
	uint32_t useOriginalStyleGiNormalization = 0;
	float roughReflectionStart = 0.5f;
	float roughReflectionEnd = 0.7f;
	float surfelSupportRadius = 0.25f;
};

struct SurfelTaaPushConstants
{
	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t enabled = 1;
	uint32_t resetHistory = 1;
	uint32_t frameIndex = 0;
	uint32_t pad0 = 0;
	uint32_t pad1 = 0;
	uint32_t pad2 = 0;
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

uint32_t groupCount16(uint32_t value)
{
	return (std::max(value, 1u) + 15u) / 16u;
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

	const uint64_t cellCounterCount = 1ull + static_cast<uint64_t>(push.cellCount) * 2ull;
	uint64_t workItemCount = cellCounterCount;
	if (resetPersistent)
	{
		workItemCount = std::max(static_cast<uint64_t>(push.maxSurfels), cellCounterCount);
	}
	commandBuffer.dispatch(linearGroupCount(workItemCount), 1, 1);
}

void SurfelPathTracerPasses::recordUpdatePass(const vk::raii::CommandBuffer &commandBuffer,
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
                                              bool lockSurfels,
                                              uint32_t perCellSurfelLimit) const
{
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelines.updatePipeline);
	const std::array descriptorSets = {imageSet, globalSet};
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
	                                 *pipelines.computePipelineLayout,
	                                 0,
	                                 descriptorSets,
	                                 nullptr);

	const SurfelUpdatePushConstants push{
	    .maxSurfels = std::max(maxSurfels, 1u),
	    .cellSize = std::max(cellSize, 0.0001f),
	    .cellDimension = std::max(cellDimension, 1u),
	    .maxRays = std::max(maxRays, 1u),
	    .lockSurfels = lockSurfels ? 1u : 0u,
	    .minRaysPerSurfel = std::max(minRaysPerSurfel, 1u),
	    .maxRaysPerSurfel = std::max(maxRaysPerSurfel, std::max(minRaysPerSurfel, 1u)),
	    .varianceSensitivity = std::max(varianceSensitivity, 0.0f),
	    .sourceTransformCount = sourceTransformCount,
	    .perCellSurfelLimit = std::max(perCellSurfelLimit, 1u)};
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
                                                uint32_t perCellSurfelLimit,
                                                uint32_t cellMapCapacity) const
{
	bindComputeStorageSet(commandBuffer, pipelines, pipelines.cellInfoPipeline, imageSet);

	const SurfelCellInfoPushConstants push{
	    .cellCount = cellCount,
	    .perCellSurfelLimit = std::max(perCellSurfelLimit, 1u),
	    .mapCapacity = std::max(cellMapCapacity, 1u),
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
                                                    vk::DescriptorSet globalSet,
                                                    uint32_t maxSurfels,
                                                    float cellSize,
                                                    uint32_t cellDimension,
                                                    uint32_t perCellSurfelLimit) const
{
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelines.cellToSurfelPipeline);
	const std::array descriptorSets = {imageSet, globalSet};
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
	                                 *pipelines.computePipelineLayout,
	                                 0,
	                                 descriptorSets,
	                                 nullptr);

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

void SurfelPathTracerPasses::recordRaySchedulePass(const vk::raii::CommandBuffer &commandBuffer,
                                                   const SurfelPathTracerPipelines &pipelines,
                                                   vk::DescriptorSet imageSet,
                                                   vk::DescriptorSet globalSet,
                                                   uint32_t cellCount,
                                                   uint32_t maxSurfels,
                                                   uint32_t maxRays,
                                                   uint32_t minRaysPerSurfel,
                                                   uint32_t maxRaysPerSurfel,
                                                   float varianceSensitivity,
                                                   uint32_t offscreenRayInterval,
                                                   float surfelSupportRadius) const
{
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelines.raySchedulePipeline);
	const std::array descriptorSets = {imageSet, globalSet};
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
	                                 *pipelines.computePipelineLayout,
	                                 0,
	                                 descriptorSets,
	                                 nullptr);
	const SurfelRaySchedulePushConstants push{
	    .cellCount = cellCount,
	    .maxSurfels = std::max(maxSurfels, 1u),
	    .maxRays = std::max(maxRays, 1u),
	    .minRaysPerSurfel = std::max(minRaysPerSurfel, 1u),
	    .maxRaysPerSurfel = std::max(maxRaysPerSurfel, std::max(minRaysPerSurfel, 1u)),
	    .varianceSensitivity = std::max(varianceSensitivity, 0.0f),
	    .offscreenRayInterval = std::max(offscreenRayInterval, 1u),
	    .surfelSupportRadius = std::max(surfelSupportRadius, 0.0001f)};
	commandBuffer.pushConstants<SurfelRaySchedulePushConstants>(*pipelines.computePipelineLayout,
	                                                            vk::ShaderStageFlagBits::eCompute,
	                                                            0,
	                                                            push);
	commandBuffer.dispatch(linearGroupCount(push.cellCount), 1, 1);
}

void SurfelPathTracerPasses::recordSurfelRayTracePass(const vk::raii::CommandBuffer &commandBuffer,
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
                                                      float surfelSupportRadius,
                                                      uint32_t cellDimension) const
{
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *pipelines.surfelRayTracingPipeline);
	const std::array descriptorSets = {rayTracingSet, storageSet, globalSet};
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR,
	                                 *pipelines.rayTracingPipelineLayout,
	                                 0,
	                                 descriptorSets,
	                                 nullptr);

	const SurfelRayTracePushConstants push{
	    .rayCount = std::max(rayCount, 1u),
	    .maxSurfels = std::max(resources.maxSurfelsCapacity(), 1u),
	    .enableGuidedSampling = enableGuidedSampling ? 1u : 0u,
	    .irradianceAtlasWidth = std::max(irradianceAtlasWidth, 1u),
	    .activeMaxDepth = std::clamp(activeMaxDepth, 1u, 8u),
	    .sleepingMaxDepth = std::clamp(sleepingMaxDepth, 1u, 8u),
	    .enableSurfelTermination = enableSurfelTermination ? 1u : 0u,
	    .maxSurfelSamplesPerQuery = std::clamp(maxSurfelSamplesPerQuery, 1u, 128u),
	    .cellDimension = std::max(cellDimension, 1u),
	    .cellSize = std::max(cellSize, 0.0001f),
	    .surfelSupportRadius = std::clamp(surfelSupportRadius, 0.0001f, std::max(cellSize, 0.0001f)),
	    .useOriginalStyleGiNormalization = useOriginalStyleGiNormalization ? 1u : 0u};
	commandBuffer.pushConstants<SurfelRayTracePushConstants>(*pipelines.rayTracingPipelineLayout,
	                                                         kSurfelRtPushStages,
	                                                         0,
	                                                         push);

	vk::StridedDeviceAddressRegionKHR callableRegion{};
	if (useIndirectDispatch && resources.rayDispatchIndirectAddress() != 0)
	{
		commandBuffer.traceRaysIndirectKHR(pipelines.surfelSbt.raygenRegion,
		                                   pipelines.surfelSbt.missRegion,
		                                   pipelines.surfelSbt.hitRegion,
		                                   callableRegion,
		                                   resources.rayDispatchIndirectAddress());
	}
	else
	{
		commandBuffer.traceRaysKHR(pipelines.surfelSbt.raygenRegion,
		                           pipelines.surfelSbt.missRegion,
		                           pipelines.surfelSbt.hitRegion,
		                           callableRegion,
		                           push.rayCount,
		                           1,
		                           1);
	}
}

void SurfelPathTracerPasses::recordIntegratePass(const vk::raii::CommandBuffer &commandBuffer,
                                                 const SurfelPathTracerPipelines &pipelines,
                                                 vk::DescriptorSet imageSet,
                                                 vk::DescriptorSet globalSet,
                                                 uint32_t maxSurfels,
                                                 bool enableRadianceSharing,
                                                 uint32_t maxRadianceSharingSamples,
                                                 uint32_t cellDimension,
                                                 float cellSize,
                                                 float surfelSupportRadius,
                                                 bool enableGuidedSampling) const
{
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelines.integratePipeline);
	const std::array descriptorSets = {imageSet, globalSet};
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
	                                 *pipelines.computePipelineLayout,
	                                 0,
	                                 descriptorSets,
	                                 nullptr);

	const SurfelIntegratePushConstants push{
	    .maxSurfels = std::max(maxSurfels, 1u),
	    .enableRadianceSharing = enableRadianceSharing ? 1u : 0u,
	    .maxRadianceSharingSamples = std::clamp(maxRadianceSharingSamples, 1u, 128u),
	    .cellDimension = std::max(cellDimension, 1u),
	    .cellSize = std::max(cellSize, 0.0001f),
	    .enableGuidedSampling = enableGuidedSampling ? 1u : 0u,
	    .surfelSupportRadius = std::clamp(surfelSupportRadius, 0.0001f, std::max(cellSize, 0.0001f))};
	commandBuffer.pushConstants<SurfelIntegratePushConstants>(*pipelines.computePipelineLayout,
	                                                          vk::ShaderStageFlagBits::eCompute,
	                                                          0,
	                                                          push);
	commandBuffer.dispatch(linearGroupCount(push.maxSurfels), 1, 1);
}

void SurfelPathTracerPasses::recordEvaluatePass(const vk::raii::CommandBuffer &commandBuffer,
                                                const SurfelPathTracerPipelines &pipelines,
                                                vk::DescriptorSet imageSet,
                                                vk::DescriptorSet globalSet,
                                                SurfelPathTracerEvaluateMode mode,
                                                float cellSize,
                                                float surfelSupportRadius,
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
                                                vk::Extent2D extent) const
{
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelines.evaluatePipeline);
	const std::array descriptorSets = {imageSet, globalSet};
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
	                                 *pipelines.computePipelineLayout,
	                                 0,
	                                 descriptorSets,
	                                 nullptr);

	const SurfelEvaluatePushConstants push{
	    .mode = static_cast<uint32_t>(mode),
	    .width = std::max(extent.width, 1u),
	    .height = std::max(extent.height, 1u),
	    .cellSize = std::max(cellSize, 0.0001f),
	    .cellDimension = std::max(cellDimension, 1u),
	    .maxSurfels = std::max(maxSurfels, 1u),
	    .placementThreshold = std::max(placementThreshold, 0.0f),
	    .removalThreshold = std::max(removalThreshold, placementThreshold),
	    .surfelTargetArea = std::max(surfelTargetArea, 1.0f),
	    .surfelMinRadius = std::max(surfelMinRadius, 0.0001f),
	    .surfelMaxRadiusScale = std::max(surfelMaxRadiusScale, 0.25f),
	    .frameIndex = frameIndex,
	    .lockSurfels = lockSurfels ? 1u : 0u,
	    .enablePlacement = enableSurfelPlacement ? 1u : 0u,
	    .enableRemoval = enableSurfelRemoval ? 1u : 0u,
	    .maxSurfelSamplesPerQuery = std::clamp(maxSurfelSamplesPerQuery, 1u, 128u),
	    .perCellSurfelLimit = std::max(perCellSurfelLimit, 1u),
	    .surfelSupportRadius = std::clamp(surfelSupportRadius, 0.0001f, std::max(cellSize, 0.0001f))};
	commandBuffer.pushConstants<SurfelEvaluatePushConstants>(*pipelines.computePipelineLayout,
	                                                         vk::ShaderStageFlagBits::eCompute,
	                                                         0,
	                                                         push);

	// Generation evaluates one temporal sample per 8x8 pixel tile. Dispatch the
	// compact candidate grid so every wave contains useful lanes; the shader maps
	// those lanes back to the current full-resolution temporal phase.
	const bool generationPass = mode == SurfelPathTracerEvaluateMode::Generate;
	const uint32_t dispatchWidth = generationPass ? (push.width + 7u) / 8u : push.width;
	const uint32_t dispatchHeight = generationPass ? (push.height + 7u) / 8u : push.height;
	const uint32_t groupCountX = (dispatchWidth + 15u) / 16u;
	const uint32_t groupCountY = (dispatchHeight + 15u) / 16u;
	commandBuffer.dispatch(groupCountX, groupCountY, 1);
}

void SurfelPathTracerPasses::recordReflectionPass(const vk::raii::CommandBuffer &commandBuffer,
                                                  const SurfelPathTracerPipelines &pipelines,
                                                  const SurfelPathTracerResources &resources,
                                                  vk::DescriptorSet rayTracingSet,
                                                  vk::DescriptorSet storageSet,
                                                  vk::DescriptorSet globalSet,
                                                  bool enabled,
                                                  float cellSize,
                                                  float surfelSupportRadius,
                                                  uint32_t cellDimension,
                                                  vk::Extent2D extent,
                                                  uint32_t frameIndex,
                                                  bool enableSurfelTermination,
	                                              bool useOriginalStyleGiNormalization,
                                                  uint32_t maxSurfelSamplesPerQuery,
                                                float roughReflectionStart,
                                                float roughReflectionEnd) const
{
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *pipelines.reflectionRayTracingPipeline);
	const std::array descriptorSets = {rayTracingSet, storageSet, globalSet};
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR,
	                                 *pipelines.rayTracingPipelineLayout,
	                                 0,
	                                 descriptorSets,
	                                 nullptr);

	const uint32_t halfWidth = std::max((extent.width + 1u) / 2u, 1u);
	const uint32_t halfHeight = std::max((extent.height + 1u) / 2u, 1u);
	const SurfelReflectionPushConstants push{
	    .width = halfWidth,
	    .height = halfHeight,
	    .fullWidth = std::max(extent.width, 1u),
	    .fullHeight = std::max(extent.height, 1u),
	    .maxSurfels = std::max(resources.maxSurfelsCapacity(), 1u),
	    .cellDimension = std::max(cellDimension, 1u),
	    .cellSize = std::max(cellSize, 0.0001f),
	    .frameIndex = frameIndex,
	    .enabled = enabled ? 1u : 0u,
	    .enableSurfelTermination = enableSurfelTermination ? 1u : 0u,
	    .maxSurfelSamplesPerQuery = std::clamp(maxSurfelSamplesPerQuery, 1u, 128u),
	    .useOriginalStyleGiNormalization = useOriginalStyleGiNormalization ? 1u : 0u,
	    .roughReflectionStart = std::clamp(roughReflectionStart, 0.0f, 1.0f),
	    .roughReflectionEnd = std::clamp(std::max(roughReflectionEnd, roughReflectionStart + 0.01f), 0.0f, 1.01f),
	    .surfelSupportRadius = std::clamp(surfelSupportRadius, 0.0001f, std::max(cellSize, 0.0001f))};
	commandBuffer.pushConstants<SurfelReflectionPushConstants>(*pipelines.rayTracingPipelineLayout,
	                                                           kSurfelRtPushStages,
	                                                           0,
	                                                           push);

	vk::StridedDeviceAddressRegionKHR callableRegion{};
	commandBuffer.traceRaysKHR(pipelines.reflectionSbt.raygenRegion,
	                           pipelines.reflectionSbt.missRegion,
	                           pipelines.reflectionSbt.hitRegion,
	                           callableRegion,
	                           halfWidth,
	                           halfHeight,
	                           1);
}

void SurfelPathTracerPasses::recordReferencePass(const vk::raii::CommandBuffer &commandBuffer,
                                                 const SurfelPathTracerPipelines &pipelines,
                                                 vk::DescriptorSet rayTracingSet,
                                                 vk::DescriptorSet storageSet,
                                                 vk::DescriptorSet globalSet,
                                                 bool enableReferenceValidation,
                                                 uint32_t maxDepth,
                                                 vk::Extent2D extent) const
{
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *pipelines.referenceRayTracingPipeline);
	const std::array descriptorSets = {rayTracingSet, storageSet, globalSet};
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR,
	                                 *pipelines.rayTracingPipelineLayout,
	                                 0,
	                                 descriptorSets,
	                                 nullptr);

	const SurfelReferencePushConstants push{
	    .width = std::max(extent.width, 1u),
	    .height = std::max(extent.height, 1u),
	    .maxDepth = std::clamp(maxDepth, 1u, 8u),
	    .enabled = enableReferenceValidation ? 1u : 0u};
	commandBuffer.pushConstants<SurfelReferencePushConstants>(*pipelines.rayTracingPipelineLayout,
	                                                          kSurfelRtPushStages,
	                                                          0,
	                                                          push);

	vk::StridedDeviceAddressRegionKHR callableRegion{};
	commandBuffer.traceRaysKHR(pipelines.referenceSbt.raygenRegion,
	                           pipelines.referenceSbt.missRegion,
	                           pipelines.referenceSbt.hitRegion,
	                           callableRegion,
	                           push.width,
	                           push.height,
	                           1);
}

void SurfelPathTracerPasses::recordReflectionFilterPass(const vk::raii::CommandBuffer &commandBuffer,
                                                        const SurfelPathTracerPipelines &pipelines,
                                                        vk::DescriptorSet imageSet,
                                                        bool enabled,
                                                        bool resetHistory,
                                                        vk::Extent2D extent,
                                                        uint32_t frameIndex) const
{
	bindComputeStorageSet(commandBuffer, pipelines, pipelines.reflectionFilterPipeline, imageSet);

	const SurfelReflectionFilterPushConstants push{
	    .width = std::max((extent.width + 1u) / 2u, 1u),
	    .height = std::max((extent.height + 1u) / 2u, 1u),
	    .fullWidth = std::max(extent.width, 1u),
	    .fullHeight = std::max(extent.height, 1u),
	    .enabled = enabled ? 1u : 0u,
	    .resetHistory = resetHistory ? 1u : 0u,
	    .frameIndex = frameIndex};
	commandBuffer.pushConstants<SurfelReflectionFilterPushConstants>(*pipelines.computePipelineLayout,
	                                                                 vk::ShaderStageFlagBits::eCompute,
	                                                                 0,
	                                                                 push);
	commandBuffer.dispatch(groupCount16(push.width), groupCount16(push.height), 1);
}

void SurfelPathTracerPasses::recordBilateralPass(const vk::raii::CommandBuffer &commandBuffer,
                                                 const SurfelPathTracerPipelines &pipelines,
                                                 vk::DescriptorSet imageSet,
                                                 bool enabled,
                                                 vk::Extent2D extent) const
{
	bindComputeStorageSet(commandBuffer, pipelines, pipelines.bilateralPipeline, imageSet);

	const SurfelBilateralPushConstants push{
	    .width = std::max((extent.width + 1u) / 2u, 1u),
	    .height = std::max((extent.height + 1u) / 2u, 1u),
	    .fullWidth = std::max(extent.width, 1u),
	    .fullHeight = std::max(extent.height, 1u),
	    .enabled = enabled ? 1u : 0u};
	commandBuffer.pushConstants<SurfelBilateralPushConstants>(*pipelines.computePipelineLayout,
	                                                          vk::ShaderStageFlagBits::eCompute,
	                                                          0,
	                                                          push);
	commandBuffer.dispatch(groupCount16(push.width), groupCount16(push.height), 1);
}

void SurfelPathTracerPasses::recordLightIntegratePass(const vk::raii::CommandBuffer &commandBuffer,
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
                                                      vk::Extent2D extent,
                                                      float roughReflectionStart,
                                                      float roughReflectionEnd,
                                                      float surfelSupportRadius) const
{
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelines.lightIntegratePipeline);
	const std::array descriptorSets = {imageSet, globalSet};
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
	                                 *pipelines.computePipelineLayout,
	                                 0,
	                                 descriptorSets,
	                                 nullptr);

	const SurfelLightIntegratePushConstants push{
	    .width = std::max(extent.width, 1u),
	    .height = std::max(extent.height, 1u),
	    .enableDiffuseGi = enableDiffuseGi ? 1u : 0u,
	    .enableReflections = enableReflections ? 1u : 0u,
	    .useBilateralReflection = useBilateralReflection ? 1u : 0u,
	    .debugView = debugView,
	    .cellSize = std::max(cellSize, 0.0001f),
	    .surfelMaxRadiusScale = std::max(surfelMaxRadiusScale, 0.25f),
	    .cellDimension = std::max(cellDimension, 1u),
	    .perCellSurfelLimit = std::max(perCellSurfelLimit, 1u),
	    .useOriginalStyleGiNormalization = useOriginalStyleGiNormalization ? 1u : 0u,
	    .roughReflectionStart = std::clamp(roughReflectionStart, 0.0f, 1.0f),
	    .roughReflectionEnd = std::clamp(std::max(roughReflectionEnd, roughReflectionStart + 0.01f), 0.0f, 1.01f),
	    .surfelSupportRadius = std::clamp(surfelSupportRadius, 0.0001f, std::max(cellSize, 0.0001f))};
	commandBuffer.pushConstants<SurfelLightIntegratePushConstants>(*pipelines.computePipelineLayout,
	                                                               vk::ShaderStageFlagBits::eCompute,
	                                                               0,
	                                                               push);
	commandBuffer.dispatch(groupCount16(push.width), groupCount16(push.height), 1);
}

void SurfelPathTracerPasses::recordTaaPass(const vk::raii::CommandBuffer &commandBuffer,
                                           const SurfelPathTracerPipelines &pipelines,
                                           vk::DescriptorSet imageSet,
                                           vk::DescriptorSet globalSet,
                                           bool enabled,
                                           bool resetHistory,
                                           vk::Extent2D extent,
                                           uint32_t frameIndex) const
{
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelines.taaPipeline);
	const std::array descriptorSets = {imageSet, globalSet};
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
	                                 *pipelines.computePipelineLayout,
	                                 0,
	                                 descriptorSets,
	                                 nullptr);

	const SurfelTaaPushConstants push{
	    .width = std::max(extent.width, 1u),
	    .height = std::max(extent.height, 1u),
	    .enabled = enabled ? 1u : 0u,
	    .resetHistory = resetHistory ? 1u : 0u,
	    .frameIndex = frameIndex};
	commandBuffer.pushConstants<SurfelTaaPushConstants>(*pipelines.computePipelineLayout,
	                                                    vk::ShaderStageFlagBits::eCompute,
	                                                    0,
	                                                    push);
	commandBuffer.dispatch(groupCount16(push.width), groupCount16(push.height), 1);
}

void SurfelPathTracerPasses::recordImageBarrierGBufferToCompute(
    const vk::raii::CommandBuffer &commandBuffer,
    const SurfelPathTracerResources &resources,
    uint32_t frameIndex) const
{
	const std::array images = {
	    static_cast<vk::Image>(*resources.gBufferNormalImages[frameIndex]),
	    static_cast<vk::Image>(*resources.gBufferDepthImages[frameIndex]),
	    static_cast<vk::Image>(*resources.gBufferMotionMaterialImages[frameIndex]),
	    static_cast<vk::Image>(*resources.gBufferAlbedoImages[frameIndex]),
	    static_cast<vk::Image>(*resources.gBufferMaterialImages[frameIndex]),
	    static_cast<vk::Image>(*resources.gBufferEmissiveImages[frameIndex])};

	std::array<vk::ImageMemoryBarrier2, 6> barriers{};
	for (size_t i = 0; i < images.size(); ++i)
	{
		barriers[i] = vk::ImageMemoryBarrier2{
		    .srcStageMask = vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
		    .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
		    .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader |
		                    vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
		    .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead,
		    .oldLayout = vk::ImageLayout::eGeneral,
		    .newLayout = vk::ImageLayout::eGeneral,
		    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .image = images[i],
		    .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
	}

	vk::BufferMemoryBarrier2 sourceBufferBarrier{
	    .srcStageMask = vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
	    .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
	    .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
	    .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead,
	    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .buffer = *resources.gBufferSourceBuffers[frameIndex],
	    .offset = 0,
	    .size = VK_WHOLE_SIZE,
	};

	vk::DependencyInfo dependency{
	    .bufferMemoryBarrierCount = 1,
	    .pBufferMemoryBarriers = &sourceBufferBarrier,
	    .imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
	    .pImageMemoryBarriers = barriers.data(),
	};
	commandBuffer.pipelineBarrier2(dependency);
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

void SurfelPathTracerPasses::recordStorageBarrierComputeToRt(
    const vk::raii::CommandBuffer &commandBuffer) const
{
	vk::MemoryBarrier2 storageBufferBarrier{
	    .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
	    .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
	    .dstStageMask = vk::PipelineStageFlagBits2::eDrawIndirect |
	                    vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
	    .dstAccessMask = vk::AccessFlagBits2::eIndirectCommandRead |
	                     vk::AccessFlagBits2::eShaderStorageRead |
	                     vk::AccessFlagBits2::eShaderStorageWrite};
	vk::DependencyInfo dependency{
	    .memoryBarrierCount = 1,
	    .pMemoryBarriers = &storageBufferBarrier};
	commandBuffer.pipelineBarrier2(dependency);
}

void SurfelPathTracerPasses::recordStorageBarrierRtToCompute(
    const vk::raii::CommandBuffer &commandBuffer) const
{
	vk::MemoryBarrier2 storageBufferBarrier{
	    .srcStageMask = vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
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
