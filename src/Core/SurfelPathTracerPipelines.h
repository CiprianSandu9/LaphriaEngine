#ifndef LAPHRIAENGINE_SURFELPATHTRACERPIPELINES_H
#define LAPHRIAENGINE_SURFELPATHTRACERPIPELINES_H

#include "VulkanDevice.h"
#include "VulkanUtils.h"

namespace Laphria
{
struct SurfelPathTracerSbtRegions
{
	VulkanUtils::VmaBuffer raygenSBTBuffer{};
	vk::StridedDeviceAddressRegionKHR raygenRegion{};

	VulkanUtils::VmaBuffer missSBTBuffer{};
	vk::StridedDeviceAddressRegionKHR missRegion{};

	VulkanUtils::VmaBuffer hitSBTBuffer{};
	vk::StridedDeviceAddressRegionKHR hitRegion{};
};

class SurfelPathTracerPipelines
{
  public:
	SurfelPathTracerPipelines() = default;
	~SurfelPathTracerPipelines() = default;

	void createDescriptorSetLayouts(const VulkanDevice &dev);
	void createPipelineLayouts(const VulkanDevice &dev, vk::DescriptorSetLayout globalDescriptorSetLayout);
	void createComputePipelines(const VulkanDevice &dev);
	void createGBufferRayTracingPipeline(const VulkanDevice &dev);
	void createSurfelRayTracingPipeline(const VulkanDevice &dev);
	void createReflectionRayTracingPipeline(const VulkanDevice &dev);
	void createReferenceRayTracingPipeline(const VulkanDevice &dev);
	void createGBufferShaderBindingTable(const VulkanDevice &dev);
	void createSurfelShaderBindingTable(const VulkanDevice &dev);
	void createReflectionShaderBindingTable(const VulkanDevice &dev);
	void createReferenceShaderBindingTable(const VulkanDevice &dev);

	vk::raii::DescriptorSetLayout skyDescriptorSetLayout{nullptr};
	vk::raii::DescriptorSetLayout storageDescriptorSetLayout{nullptr};
	vk::raii::DescriptorSetLayout rayTracingDescriptorSetLayout{nullptr};

	vk::raii::PipelineLayout skyPipelineLayout{nullptr};
	vk::raii::PipelineLayout computePipelineLayout{nullptr};
	vk::raii::PipelineLayout rayTracingPipelineLayout{nullptr};

	vk::raii::Pipeline skyPipeline{nullptr};
	vk::raii::Pipeline preparePipeline{nullptr};
	vk::raii::Pipeline updatePipeline{nullptr};
	vk::raii::Pipeline cellInfoPipeline{nullptr};
	vk::raii::Pipeline cellToSurfelPipeline{nullptr};
	vk::raii::Pipeline integratePipeline{nullptr};
	vk::raii::Pipeline evaluatePipeline{nullptr};
	vk::raii::Pipeline reflectionFilterPipeline{nullptr};
	vk::raii::Pipeline bilateralPipeline{nullptr};
	vk::raii::Pipeline lightIntegratePipeline{nullptr};
	vk::raii::Pipeline taaPipeline{nullptr};

	vk::raii::Pipeline gBufferRayTracingPipeline{nullptr};
	vk::raii::Pipeline surfelRayTracingPipeline{nullptr};
	vk::raii::Pipeline reflectionRayTracingPipeline{nullptr};
	vk::raii::Pipeline referenceRayTracingPipeline{nullptr};

	SurfelPathTracerSbtRegions gBufferSbt{};
	SurfelPathTracerSbtRegions surfelSbt{};
	SurfelPathTracerSbtRegions reflectionSbt{};
	SurfelPathTracerSbtRegions referenceSbt{};
};
} // namespace Laphria

#endif // LAPHRIAENGINE_SURFELPATHTRACERPIPELINES_H
