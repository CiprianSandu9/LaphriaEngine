#ifndef LAPHRIAENGINE_PIPELINECOLLECTION_H
#define LAPHRIAENGINE_PIPELINECOLLECTION_H

#include <string>
#include <vector>

#include "EngineAuxiliary.h"
#include "VulkanDevice.h"
#include "VulkanUtils.h"

// Owns all descriptor set layouts, pipelines, and pipeline layouts.
// Call createDescriptorSetLayouts() first, then each createXxxPipeline() in order.
class PipelineCollection
{
  public:
	~PipelineCollection() = default;

	void createDescriptorSetLayouts(const VulkanDevice &dev);
	void createGraphicsPipeline(VulkanDevice &dev, vk::Format colorFormat, vk::Format depthFormat);
	void createShadowPipeline(VulkanDevice &dev);

	void createComputePipeline(const VulkanDevice &dev);
	void createSkinningPipeline(const VulkanDevice &dev);
	void createPhysicsPipeline(const VulkanDevice &dev);
	void createRayTracingPipeline(const VulkanDevice &dev);
	void createShaderBindingTable(const VulkanDevice &dev);
	void createDenoiserPipelines(const VulkanDevice &dev);
	void createClassicRTPipeline(const VulkanDevice &dev);
	void createClassicRTShaderBindingTable(const VulkanDevice &dev);

	// ── Descriptor Set Layouts ────────────────────────────────────────────
	vk::raii::DescriptorSetLayout descriptorSetLayoutGlobal{nullptr};
	vk::raii::DescriptorSetLayout descriptorSetLayoutMaterial{nullptr};
	vk::raii::DescriptorSetLayout computeDescriptorSetLayout{nullptr};
	vk::raii::DescriptorSetLayout skinningDescriptorSetLayout{nullptr};
	vk::raii::DescriptorSetLayout physicsDescriptorSetLayout{nullptr};
	vk::raii::DescriptorSetLayout rayTracingDescriptorSetLayout{nullptr};
	vk::raii::DescriptorSetLayout denoiserDescriptorSetLayout{nullptr};

	// ── Pipelines ─────────────────────────────────────────────────────────
	vk::raii::Pipeline graphicsPipeline{nullptr};
	vk::raii::Pipeline shadowPipeline{nullptr};
	vk::raii::Pipeline computePipeline{nullptr};
	vk::raii::Pipeline skinningPipeline{nullptr};
	vk::raii::Pipeline physicsPipeline{nullptr};

	vk::raii::Pipeline rayTracingPipeline{nullptr};   // path tracer
	vk::raii::Pipeline classicRTPipeline{nullptr};    // classic ray tracer (direct illumination)

	// Denoiser: two compute pipelines (temporal reprojection + spatial A-Trous).
	vk::raii::Pipeline reprojectionPipeline{nullptr};
	vk::raii::Pipeline atrousPipeline{nullptr};

	// ── Pipeline Layouts ──────────────────────────────────────────────────
	vk::raii::PipelineLayout graphicsPipelineLayout{nullptr};
	vk::raii::PipelineLayout shadowPipelineLayout{nullptr};
	vk::raii::PipelineLayout computePipelineLayout{nullptr};
	vk::raii::PipelineLayout skinningPipelineLayout{nullptr};
	vk::raii::PipelineLayout physicsPipelineLayout{nullptr};

	vk::raii::PipelineLayout rayTracingPipelineLayout{nullptr};
	vk::raii::PipelineLayout denoiserPipelineLayout{nullptr};

	// ── Shader Binding Table (SBT) — Path Tracer ─────────────────────────
	Laphria::VulkanUtils::VmaBuffer   raygenSBTBuffer{};
	vk::StridedDeviceAddressRegionKHR raygenRegion{};

	Laphria::VulkanUtils::VmaBuffer   missSBTBuffer{};
	vk::StridedDeviceAddressRegionKHR missRegion{};

	Laphria::VulkanUtils::VmaBuffer   hitSBTBuffer{};
	vk::StridedDeviceAddressRegionKHR hitRegion{};

	// ── Shader Binding Table (SBT) — Classic Ray Tracer ──────────────────
	Laphria::VulkanUtils::VmaBuffer   classicRTRaygenSBTBuffer{};
	vk::StridedDeviceAddressRegionKHR classicRTRaygenRegion{};

	Laphria::VulkanUtils::VmaBuffer   classicRTMissSBTBuffer{};
	vk::StridedDeviceAddressRegionKHR classicRTMissRegion{};

	Laphria::VulkanUtils::VmaBuffer   classicRTHitSBTBuffer{};
	vk::StridedDeviceAddressRegionKHR classicRTHitRegion{};

  private:
	void createGlobalDescriptorSetLayout(const VulkanDevice &dev);
	void createMaterialDescriptorSetLayout(const VulkanDevice &dev);
	void createComputeDescriptorSetLayout(const VulkanDevice &dev);
	void createSkinningDescriptorSetLayout(const VulkanDevice &dev);
	void createPhysicsDescriptorSetLayout(const VulkanDevice &dev);
	void createRayTracingDescriptorSetLayout(const VulkanDevice &dev);
	void createDenoiserDescriptorSetLayout(const VulkanDevice &dev);
	void createDenoiserPipelineLayout(const VulkanDevice &dev);
	void createGraphicsPipelineLayout(const VulkanDevice &dev);
	void createShadowPipelineLayout(const VulkanDevice &dev);
	void createComputePipelineLayout(const VulkanDevice &dev);
	void createSkinningPipelineLayout(const VulkanDevice &dev);
	void createPhysicsPipelineLayout(const VulkanDevice &dev);
	void createRayTracingPipelineLayout(const VulkanDevice &dev);

	[[nodiscard]] vk::raii::ShaderModule createShaderModule(const VulkanDevice            &dev,
	                                                        const std::vector<char> &code) const;
	static std::vector<char>             readFile(const std::string &filename);
};

#endif        // LAPHRIAENGINE_PIPELINECOLLECTION_H
