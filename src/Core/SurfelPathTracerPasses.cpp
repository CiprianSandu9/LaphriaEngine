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
	uint32_t maxRays = 0;
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

struct SurfelRayTracePushConstants
{
	uint32_t rayCount = 0;
	uint32_t maxSurfels = 0;
	uint32_t pad0 = 0;
	uint32_t pad1 = 0;
};

struct SurfelIntegratePushConstants
{
	uint32_t maxSurfels = 0;
	uint32_t pad0 = 0;
	uint32_t pad1 = 0;
	uint32_t pad2 = 0;
};

struct SurfelEvaluatePushConstants
{
	uint32_t mode = 0;
	uint32_t width = 0;
	uint32_t height = 0;
	float cellSize = 1.0f;
	uint32_t cellDimension = 1;
	uint32_t maxSurfels = 0;
	uint32_t pad1 = 0;
	uint32_t pad2 = 0;
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
	uint32_t pad0 = 0;
	uint32_t pad1 = 0;
	uint32_t pad2 = 0;
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
	uint32_t pad0 = 0;
	uint32_t pad1 = 0;
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
                                              vk::DescriptorSet globalSet,
                                              uint32_t maxSurfels,
                                              uint32_t maxRays,
                                              float cellSize,
                                              uint32_t cellDimension) const
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
	    .maxRays = std::max(maxRays, 1u)};
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

void SurfelPathTracerPasses::recordSurfelRayTracePass(const vk::raii::CommandBuffer &commandBuffer,
                                                      const SurfelPathTracerPipelines &pipelines,
                                                      const SurfelPathTracerResources &resources,
                                                      vk::DescriptorSet rayTracingSet,
                                                      vk::DescriptorSet storageSet,
                                                      vk::DescriptorSet globalSet,
                                                      uint32_t rayCount) const
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
	    .pad0 = 0,
	    .pad1 = 0};
	commandBuffer.pushConstants<SurfelRayTracePushConstants>(*pipelines.rayTracingPipelineLayout,
	                                                         vk::ShaderStageFlagBits::eRaygenKHR,
	                                                         0,
	                                                         push);

	vk::StridedDeviceAddressRegionKHR callableRegion{};
	commandBuffer.traceRaysKHR(pipelines.surfelSbt.raygenRegion,
	                           pipelines.surfelSbt.missRegion,
	                           pipelines.surfelSbt.hitRegion,
	                           callableRegion,
	                           push.rayCount,
	                           1,
	                           1);
}

void SurfelPathTracerPasses::recordIntegratePass(const vk::raii::CommandBuffer &commandBuffer,
                                                 const SurfelPathTracerPipelines &pipelines,
                                                 vk::DescriptorSet imageSet,
                                                 uint32_t maxSurfels) const
{
	bindComputeStorageSet(commandBuffer, pipelines, pipelines.integratePipeline, imageSet);

	const SurfelIntegratePushConstants push{
	    .maxSurfels = std::max(maxSurfels, 1u),
	    .pad0 = 0,
	    .pad1 = 0,
	    .pad2 = 0};
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
                                                uint32_t cellDimension,
                                                uint32_t maxSurfels,
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
	    .pad1 = 0,
	    .pad2 = 0};
	commandBuffer.pushConstants<SurfelEvaluatePushConstants>(*pipelines.computePipelineLayout,
	                                                         vk::ShaderStageFlagBits::eCompute,
	                                                         0,
	                                                         push);

	const uint32_t groupCountX = (push.width + 15u) / 16u;
	const uint32_t groupCountY = (push.height + 15u) / 16u;
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
                                                  uint32_t cellDimension,
                                                  vk::Extent2D extent,
                                                  uint32_t frameIndex) const
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
	    .enabled = enabled ? 1u : 0u};
	commandBuffer.pushConstants<SurfelReflectionPushConstants>(*pipelines.rayTracingPipelineLayout,
	                                                           vk::ShaderStageFlagBits::eRaygenKHR,
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
                                                      uint32_t debugView,
                                                      vk::Extent2D extent) const
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
	    .debugView = debugView};
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
	    static_cast<vk::Image>(*resources.gBufferMotionMaterialImages[frameIndex])};

	std::array<vk::ImageMemoryBarrier2, 3> barriers{};
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

	vk::DependencyInfo dependency{
	    .imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
	    .pImageMemoryBarriers = barriers.data()};
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
	    .dstStageMask = vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
	    .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite};
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
