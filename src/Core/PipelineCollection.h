#ifndef LAPHRIAENGINE_PIPELINECOLLECTION_H
#define LAPHRIAENGINE_PIPELINECOLLECTION_H

#include <string>
#include <vector>

#include "EngineAuxiliary.h"
#include "VulkanDevice.h"

// Owns all descriptor set layouts, pipelines, and pipeline layouts.
// Call createDescriptorSetLayouts() first, then each createXxxPipeline() in order.
class PipelineCollection
{
  public:
	void createDescriptorSetLayouts(VulkanDevice &dev);
	void createGraphicsPipeline(VulkanDevice &dev, vk::Format colorFormat, vk::Format depthFormat);
	void createShadowPipeline(VulkanDevice &dev);

	void createComputePipeline(VulkanDevice &dev);
	void createPhysicsPipeline(VulkanDevice &dev);
	void createRayTracingPipeline(VulkanDevice &dev);
	void createShaderBindingTable(VulkanDevice &dev);

	// ── Descriptor Set Layouts ────────────────────────────────────────────
	vk::raii::DescriptorSetLayout descriptorSetLayoutGlobal{nullptr};
	vk::raii::DescriptorSetLayout descriptorSetLayoutMaterial{nullptr};
	vk::raii::DescriptorSetLayout computeDescriptorSetLayout{nullptr};
	vk::raii::DescriptorSetLayout physicsDescriptorSetLayout{nullptr};
	vk::raii::DescriptorSetLayout rayTracingDescriptorSetLayout{nullptr};

	// ── Pipelines ─────────────────────────────────────────────────────────
	vk::raii::Pipeline graphicsPipeline{nullptr};
	vk::raii::Pipeline shadowPipeline{nullptr};
	vk::raii::Pipeline computePipeline{nullptr};
	vk::raii::Pipeline physicsPipeline{nullptr};

	vk::raii::Pipeline rayTracingPipeline{nullptr};

	// ── Pipeline Layouts ──────────────────────────────────────────────────
	vk::raii::PipelineLayout graphicsPipelineLayout{nullptr};
	vk::raii::PipelineLayout shadowPipelineLayout{nullptr};
	vk::raii::PipelineLayout computePipelineLayout{nullptr};
	vk::raii::PipelineLayout physicsPipelineLayout{nullptr};

	vk::raii::PipelineLayout rayTracingPipelineLayout{nullptr};

	// ── Shader Binding Table (SBT) ────────────────────────────────────────
	vk::raii::Buffer                  raygenSBTBuffer{nullptr};
	vk::raii::DeviceMemory            raygenSBTMemory{nullptr};
	vk::StridedDeviceAddressRegionKHR raygenRegion{};

	vk::raii::Buffer                  missSBTBuffer{nullptr};
	vk::raii::DeviceMemory            missSBTMemory{nullptr};
	vk::StridedDeviceAddressRegionKHR missRegion{};

	vk::raii::Buffer                  hitSBTBuffer{nullptr};
	vk::raii::DeviceMemory            hitSBTMemory{nullptr};
	vk::StridedDeviceAddressRegionKHR hitRegion{};

  private:
	void createGlobalDescriptorSetLayout(VulkanDevice &dev);
	void createMaterialDescriptorSetLayout(VulkanDevice &dev);
	void createComputeDescriptorSetLayout(VulkanDevice &dev);
	void createPhysicsDescriptorSetLayout(VulkanDevice &dev);
	void createRayTracingDescriptorSetLayout(VulkanDevice &dev);
	void createGraphicsPipelineLayout(VulkanDevice &dev);
	void createShadowPipelineLayout(VulkanDevice &dev);
	void createComputePipelineLayout(VulkanDevice &dev);
	void createPhysicsPipelineLayout(VulkanDevice &dev);
	void createRayTracingPipelineLayout(VulkanDevice &dev);

	[[nodiscard]] vk::raii::ShaderModule createShaderModule(VulkanDevice            &dev,
	                                                        const std::vector<char> &code) const;
	static std::vector<char>             readFile(const std::string &filename);
};

#endif        // LAPHRIAENGINE_PIPELINECOLLECTION_H
