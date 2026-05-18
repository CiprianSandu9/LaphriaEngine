#include "SurfelPathTracerPipelines.h"

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace Laphria;

namespace
{
std::filesystem::path getExecutableDirectory()
{
#ifdef _WIN32
	std::array<char, MAX_PATH> buffer{};
	const DWORD length = GetModuleFileNameA(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
	if (length > 0 && length < buffer.size())
	{
		return std::filesystem::path(std::string(buffer.data(), length)).parent_path();
	}
#endif
	return std::filesystem::current_path();
}

std::vector<std::filesystem::path> buildShaderSearchCandidates(const std::string &filename)
{
	const std::filesystem::path relativePath(filename);
	const std::filesystem::path executableDir = getExecutableDirectory();

	return {
	    executableDir / relativePath,
	    std::filesystem::current_path() / relativePath};
}

std::vector<char> readFile(const std::string &filename)
{
	std::ifstream file;
	std::filesystem::path resolvedPath;
	for (const auto &candidate : buildShaderSearchCandidates(filename))
	{
		file.open(candidate, std::ios::ate | std::ios::binary);
		if (file.is_open())
		{
			resolvedPath = candidate;
			break;
		}
		file.clear();
	}
	if (!file.is_open())
	{
		throw std::runtime_error("failed to open file: " + filename);
	}
	const size_t fileSize = static_cast<size_t>(file.tellg());
	std::vector<char> buffer(fileSize);
	file.seekg(0);
	file.read(buffer.data(), fileSize);
	if (file.fail())
	{
		throw std::runtime_error("failed to read file: " + resolvedPath.string());
	}
	return buffer;
}

vk::raii::ShaderModule createShaderModule(const VulkanDevice &dev, const std::vector<char> &code)
{
	if (code.empty() || code.size() % 4 != 0)
	{
		throw std::runtime_error("invalid SPIR-V shader: code must be non-empty and a multiple of 4 bytes");
	}

	vk::ShaderModuleCreateInfo createInfo{
	    .codeSize = code.size(),
	    .pCode = reinterpret_cast<const uint32_t *>(code.data())};
	return vk::raii::ShaderModule{dev.logicalDevice, createInfo};
}

vk::raii::Pipeline createComputePipeline(const VulkanDevice &dev,
                                         vk::PipelineLayout layout,
                                         const std::string &shaderPath,
                                         const char *entryPoint)
{
	vk::raii::ShaderModule shaderModule = createShaderModule(dev, readFile(shaderPath));
	vk::PipelineShaderStageCreateInfo computeShaderStageInfo{
	    .stage = vk::ShaderStageFlagBits::eCompute,
	    .module = *shaderModule,
	    .pName = entryPoint};
	vk::ComputePipelineCreateInfo pipelineInfo{
	    .stage = computeShaderStageInfo,
	    .layout = layout};
	return vk::raii::Pipeline(dev.logicalDevice, nullptr, pipelineInfo);
}

void createShaderBindingTableForPipeline(const VulkanDevice &dev,
                                         const vk::raii::Pipeline &pipeline,
                                         SurfelPathTracerSbtRegions &sbt)
{
	const uint32_t handleSize      = dev.rayTracingProperties.shaderGroupHandleSize;
	const uint32_t handleAlignment = dev.rayTracingProperties.shaderGroupHandleAlignment;
	const uint32_t baseAlignment   = dev.rayTracingProperties.shaderGroupBaseAlignment;

	const uint32_t handleSizeAligned = VulkanUtils::alignUp(handleSize, handleAlignment);
	const uint32_t raygenSBTSize     = VulkanUtils::alignUp(handleSizeAligned, baseAlignment);
	const uint32_t missSBTSize       = VulkanUtils::alignUp(handleSizeAligned, baseAlignment);
	const uint32_t hitSBTSize        = VulkanUtils::alignUp(handleSizeAligned, baseAlignment);

	constexpr uint32_t groupCount = 3;
	const uint32_t sbtSize = groupCount * handleSize;
	std::vector<uint8_t> handles = pipeline.getRayTracingShaderGroupHandlesKHR<uint8_t>(0, groupCount, sbtSize);

	auto createSBTBuffer = [&](VulkanUtils::VmaBuffer &buffer, uint32_t size, const void *data, uint32_t handleOffset) {
		VulkanUtils::createBuffer(
		    dev.logicalDevice, dev.physicalDevice, size,
		    vk::BufferUsageFlagBits::eShaderBindingTableKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
		    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		    buffer);

		void *mapped = buffer.memory.mapMemory(0, size);
		std::memcpy(mapped, static_cast<const uint8_t *>(data) + handleOffset, handleSize);
		buffer.memory.unmapMemory();
	};

	createSBTBuffer(sbt.raygenSBTBuffer, raygenSBTSize, handles.data(), 0);
	createSBTBuffer(sbt.missSBTBuffer, missSBTSize, handles.data(), handleSize);
	createSBTBuffer(sbt.hitSBTBuffer, hitSBTSize, handles.data(), handleSize * 2);

	vk::BufferDeviceAddressInfo raygenInfo{.buffer = *sbt.raygenSBTBuffer};
	sbt.raygenRegion.deviceAddress = dev.logicalDevice.getBufferAddress(raygenInfo);
	sbt.raygenRegion.stride = raygenSBTSize;
	sbt.raygenRegion.size = raygenSBTSize;

	vk::BufferDeviceAddressInfo missInfo{.buffer = *sbt.missSBTBuffer};
	sbt.missRegion.deviceAddress = dev.logicalDevice.getBufferAddress(missInfo);
	sbt.missRegion.stride = handleSizeAligned;
	sbt.missRegion.size = missSBTSize;

	vk::BufferDeviceAddressInfo hitInfo{.buffer = *sbt.hitSBTBuffer};
	sbt.hitRegion.deviceAddress = dev.logicalDevice.getBufferAddress(hitInfo);
	sbt.hitRegion.stride = handleSizeAligned;
	sbt.hitRegion.size = hitSBTSize;
}
} // namespace

void SurfelPathTracerPipelines::createDescriptorSetLayouts(const VulkanDevice &dev)
{
	vk::DescriptorSetLayoutBinding skyOutputBinding{
	    .binding = 0,
	    .descriptorType = vk::DescriptorType::eStorageImage,
	    .descriptorCount = 1,
	    .stageFlags = vk::ShaderStageFlagBits::eCompute};
	vk::DescriptorSetLayoutCreateInfo skyLayoutInfo{
	    .bindingCount = 1,
	    .pBindings = &skyOutputBinding};
	skyDescriptorSetLayout = vk::raii::DescriptorSetLayout(dev.logicalDevice, skyLayoutInfo);

	const vk::ShaderStageFlags storageStages =
	    vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eRaygenKHR |
	    vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eMissKHR |
	    vk::ShaderStageFlagBits::eAnyHitKHR;
	std::array<vk::DescriptorSetLayoutBinding, 20> storageBindings = {
	    vk::DescriptorSetLayoutBinding{.binding = 0, .descriptorType = vk::DescriptorType::eStorageImage, .descriptorCount = 1, .stageFlags = storageStages},
	    vk::DescriptorSetLayoutBinding{.binding = 1, .descriptorType = vk::DescriptorType::eStorageImage, .descriptorCount = 1, .stageFlags = storageStages},
	    vk::DescriptorSetLayoutBinding{.binding = 2, .descriptorType = vk::DescriptorType::eStorageImage, .descriptorCount = 1, .stageFlags = storageStages},
	    vk::DescriptorSetLayoutBinding{.binding = 3, .descriptorType = vk::DescriptorType::eStorageImage, .descriptorCount = 1, .stageFlags = storageStages},
	    vk::DescriptorSetLayoutBinding{.binding = 4, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = storageStages},
	    vk::DescriptorSetLayoutBinding{.binding = 5, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = storageStages},
	    vk::DescriptorSetLayoutBinding{.binding = 6, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = storageStages},
	    vk::DescriptorSetLayoutBinding{.binding = 7, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = storageStages},
	    vk::DescriptorSetLayoutBinding{.binding = 8, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = storageStages},
	    vk::DescriptorSetLayoutBinding{.binding = 9, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = storageStages},
	    vk::DescriptorSetLayoutBinding{.binding = 10, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = storageStages},
	    vk::DescriptorSetLayoutBinding{.binding = 11, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = storageStages},
	    vk::DescriptorSetLayoutBinding{.binding = 12, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = storageStages},
	    vk::DescriptorSetLayoutBinding{.binding = 13, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = storageStages},
	    vk::DescriptorSetLayoutBinding{.binding = 14, .descriptorType = vk::DescriptorType::eStorageImage, .descriptorCount = 1, .stageFlags = storageStages},
	    vk::DescriptorSetLayoutBinding{.binding = 15, .descriptorType = vk::DescriptorType::eStorageImage, .descriptorCount = 1, .stageFlags = storageStages},
	    vk::DescriptorSetLayoutBinding{.binding = 16, .descriptorType = vk::DescriptorType::eStorageImage, .descriptorCount = 1, .stageFlags = storageStages},
	    vk::DescriptorSetLayoutBinding{.binding = 17, .descriptorType = vk::DescriptorType::eStorageImage, .descriptorCount = 1, .stageFlags = storageStages},
	    vk::DescriptorSetLayoutBinding{.binding = 18, .descriptorType = vk::DescriptorType::eStorageImage, .descriptorCount = 1, .stageFlags = storageStages},
	    vk::DescriptorSetLayoutBinding{.binding = 19, .descriptorType = vk::DescriptorType::eStorageImage, .descriptorCount = 1, .stageFlags = storageStages}};
	vk::DescriptorSetLayoutCreateInfo storageLayoutInfo{
	    .bindingCount = static_cast<uint32_t>(storageBindings.size()),
	    .pBindings = storageBindings.data()};
	storageDescriptorSetLayout = vk::raii::DescriptorSetLayout(dev.logicalDevice, storageLayoutInfo);

	std::array<vk::DescriptorSetLayoutBinding, 5> rtBindings = {
	    vk::DescriptorSetLayoutBinding{.binding = 0, .descriptorType = vk::DescriptorType::eAccelerationStructureKHR, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR},
	    vk::DescriptorSetLayoutBinding{.binding = 5, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1000, .stageFlags = vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eAnyHitKHR},
	    vk::DescriptorSetLayoutBinding{.binding = 6, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1000, .stageFlags = vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eAnyHitKHR},
	    vk::DescriptorSetLayoutBinding{.binding = 7, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1000, .stageFlags = vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eAnyHitKHR},
	    vk::DescriptorSetLayoutBinding{.binding = 8, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1000, .stageFlags = vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eAnyHitKHR}};
	std::array<vk::DescriptorBindingFlags, 5> rtBindingFlags = {
	    vk::DescriptorBindingFlags{},
	    vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind,
	    vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind,
	    vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind,
	    vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind};
	vk::DescriptorSetLayoutBindingFlagsCreateInfo rtFlagsInfo{
	    .bindingCount = static_cast<uint32_t>(rtBindingFlags.size()),
	    .pBindingFlags = rtBindingFlags.data()};
	vk::DescriptorSetLayoutCreateInfo rtLayoutInfo{
	    .pNext = &rtFlagsInfo,
	    .flags = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool,
	    .bindingCount = static_cast<uint32_t>(rtBindings.size()),
	    .pBindings = rtBindings.data()};
	rayTracingDescriptorSetLayout = vk::raii::DescriptorSetLayout(dev.logicalDevice, rtLayoutInfo);
}

void SurfelPathTracerPipelines::createPipelineLayouts(const VulkanDevice &dev,
                                                       vk::DescriptorSetLayout globalDescriptorSetLayout)
{
	vk::PushConstantRange computePushRange{
	    .stageFlags = vk::ShaderStageFlagBits::eCompute,
	    .offset = 0,
	    .size = 128};
	std::array skyLayouts = {*skyDescriptorSetLayout, globalDescriptorSetLayout};
	vk::PipelineLayoutCreateInfo skyLayoutInfo{
	    .setLayoutCount = static_cast<uint32_t>(skyLayouts.size()),
	    .pSetLayouts = skyLayouts.data(),
	    .pushConstantRangeCount = 1,
	    .pPushConstantRanges = &computePushRange};
	skyPipelineLayout = vk::raii::PipelineLayout(dev.logicalDevice, skyLayoutInfo);

	std::array computeLayouts = {*storageDescriptorSetLayout, globalDescriptorSetLayout};
	vk::PipelineLayoutCreateInfo computeLayoutInfo{
	    .setLayoutCount = static_cast<uint32_t>(computeLayouts.size()),
	    .pSetLayouts = computeLayouts.data(),
	    .pushConstantRangeCount = 1,
	    .pPushConstantRanges = &computePushRange};
	computePipelineLayout = vk::raii::PipelineLayout(dev.logicalDevice, computeLayoutInfo);

	vk::PushConstantRange rtPushRange{
	    .stageFlags = vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR |
	                  vk::ShaderStageFlagBits::eMissKHR | vk::ShaderStageFlagBits::eAnyHitKHR,
	    .offset = 0,
	    .size = 128};
	std::array rtLayouts = {*rayTracingDescriptorSetLayout, *storageDescriptorSetLayout, globalDescriptorSetLayout};
	vk::PipelineLayoutCreateInfo rtLayoutInfo{
	    .setLayoutCount = static_cast<uint32_t>(rtLayouts.size()),
	    .pSetLayouts = rtLayouts.data(),
	    .pushConstantRangeCount = 1,
	    .pPushConstantRanges = &rtPushRange};
	rayTracingPipelineLayout = vk::raii::PipelineLayout(dev.logicalDevice, rtLayoutInfo);
}

void SurfelPathTracerPipelines::createComputePipelines(const VulkanDevice &dev)
{
	skyPipeline = createComputePipeline(dev, *skyPipelineLayout, "Shaders/SurfelPathTracerSky.slang.spv", "main");
	preparePipeline = createComputePipeline(dev, *computePipelineLayout, "Shaders/SurfelPathTracerPrepare.slang.spv", "main");
	updatePipeline = createComputePipeline(dev, *computePipelineLayout, "Shaders/SurfelPathTracerUpdate.slang.spv", "main");
	cellInfoPipeline = createComputePipeline(dev, *computePipelineLayout, "Shaders/SurfelPathTracerCellInfo.slang.spv", "main");
	cellToSurfelPipeline = createComputePipeline(dev, *computePipelineLayout, "Shaders/SurfelPathTracerCellToSurfel.slang.spv", "main");
}

void SurfelPathTracerPipelines::createGBufferRayTracingPipeline(const VulkanDevice &dev)
{
	vk::raii::ShaderModule rgenModule = createShaderModule(dev, readFile("Shaders/SurfelPathTracerGBuffer.slang.spv"));
	vk::raii::ShaderModule rmissModule = createShaderModule(dev, readFile("Shaders/SurfelPathTracerGBufferMiss.slang.spv"));
	vk::raii::ShaderModule rchitModule = createShaderModule(dev, readFile("Shaders/SurfelPathTracerGBufferClosestHit.slang.spv"));
	vk::raii::ShaderModule ranyModule = createShaderModule(dev, readFile("Shaders/SurfelPathTracerGBufferAnyHit.slang.spv"));

	std::array<vk::PipelineShaderStageCreateInfo, 4> stages = {
	    vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eRaygenKHR, .module = *rgenModule, .pName = "main"},
	    vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eMissKHR, .module = *rmissModule, .pName = "main"},
	    vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eClosestHitKHR, .module = *rchitModule, .pName = "main"},
	    vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eAnyHitKHR, .module = *ranyModule, .pName = "main"}};

	std::array<vk::RayTracingShaderGroupCreateInfoKHR, 3> groups = {
	    vk::RayTracingShaderGroupCreateInfoKHR{
	        .type = vk::RayTracingShaderGroupTypeKHR::eGeneral,
	        .generalShader = 0,
	        .closestHitShader = VK_SHADER_UNUSED_KHR,
	        .anyHitShader = VK_SHADER_UNUSED_KHR,
	        .intersectionShader = VK_SHADER_UNUSED_KHR},
	    vk::RayTracingShaderGroupCreateInfoKHR{
	        .type = vk::RayTracingShaderGroupTypeKHR::eGeneral,
	        .generalShader = 1,
	        .closestHitShader = VK_SHADER_UNUSED_KHR,
	        .anyHitShader = VK_SHADER_UNUSED_KHR,
	        .intersectionShader = VK_SHADER_UNUSED_KHR},
	    vk::RayTracingShaderGroupCreateInfoKHR{
	        .type = vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
	        .generalShader = VK_SHADER_UNUSED_KHR,
	        .closestHitShader = 2,
	        .anyHitShader = 3,
	        .intersectionShader = VK_SHADER_UNUSED_KHR}};

	vk::RayTracingPipelineCreateInfoKHR pipelineInfo{
	    .stageCount = static_cast<uint32_t>(stages.size()),
	    .pStages = stages.data(),
	    .groupCount = static_cast<uint32_t>(groups.size()),
	    .pGroups = groups.data(),
	    .maxPipelineRayRecursionDepth = 1,
	    .layout = *rayTracingPipelineLayout};

	gBufferRayTracingPipeline = dev.logicalDevice.createRayTracingPipelineKHR(nullptr, nullptr, pipelineInfo);
}

void SurfelPathTracerPipelines::createSurfelRayTracingPipeline(const VulkanDevice &)
{
	// Task 6 will create the first non-null RT pipeline.
}

void SurfelPathTracerPipelines::createReflectionRayTracingPipeline(const VulkanDevice &)
{
	// Task 6 will create the first non-null RT pipeline.
}

void SurfelPathTracerPipelines::createGBufferShaderBindingTable(const VulkanDevice &dev)
{
	createShaderBindingTableForPipeline(dev, gBufferRayTracingPipeline, gBufferSbt);
}

void SurfelPathTracerPipelines::createSurfelShaderBindingTable(const VulkanDevice &)
{
	// SBT allocation is deferred until the corresponding RT pipeline exists.
}

void SurfelPathTracerPipelines::createReflectionShaderBindingTable(const VulkanDevice &)
{
	// SBT allocation is deferred until the corresponding RT pipeline exists.
}
