#include "GpuResourceRegistry.h"

#include "ResourceManager.h"
#include "VulkanUtils.h"

#include <algorithm>
#include <array>
#include <fastgltf/tools.hpp>

GpuResourceRegistry::GpuResourceRegistry(vk::raii::Device &device, vk::raii::PhysicalDevice &physicalDevice, vk::raii::CommandPool &commandPool, vk::raii::Queue &queue,
                                         vk::raii::DescriptorPool &descriptorPool)
    : device(device), physicalDevice(physicalDevice), commandPool(commandPool), queue(queue), descriptorPool(descriptorPool)
{
}

void GpuResourceRegistry::setSkinningDescriptorSetLayout(vk::DescriptorSetLayout layout)
{
	skinningDescriptorSetLayout = layout;
}

void GpuResourceRegistry::uploadModelBuffers(ModelResource &modelResource, const std::vector<Laphria::Vertex> &vertices, const std::vector<uint32_t> &indices) const
{
	modelResource.vertexCount = static_cast<uint32_t>(vertices.size());
	modelResource.indexCount = static_cast<uint32_t>(indices.size());

	if (vertices.empty())
	{
		return;
	}

	constexpr vk::BufferUsageFlags vertexUsage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
	                                             vk::BufferUsageFlagBits::eShaderDeviceAddress |
	                                             vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
	Laphria::VulkanUtils::createDeviceLocalBufferFromData(device, physicalDevice, commandPool, queue,
	                                                      vertices.data(), sizeof(Laphria::Vertex) * vertices.size(), vertexUsage,
	                                                      modelResource.vertexBuffer);

	constexpr vk::BufferUsageFlags indexUsage = vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
	                                            vk::BufferUsageFlagBits::eShaderDeviceAddress |
	                                            vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
	Laphria::VulkanUtils::createDeviceLocalBufferFromData(device, physicalDevice, commandPool, queue,
	                                                      indices.data(), sizeof(uint32_t) * indices.size(), indexUsage,
	                                                      modelResource.indexBuffer);
}

void GpuResourceRegistry::uploadMaterialBuffer(ModelResource &modelResource, const std::vector<Laphria::MaterialData> &materials) const
{
	if (materials.empty())
	{
		return;
	}

	const vk::DeviceSize bufferSize = sizeof(Laphria::MaterialData) * materials.size();
	Laphria::VulkanUtils::createDeviceLocalBufferFromData(device, physicalDevice, commandPool, queue,
	                                                      materials.data(), bufferSize, vk::BufferUsageFlagBits::eStorageBuffer,
	                                                      modelResource.materialBuffer);
}

void GpuResourceRegistry::uploadMaterialBuffer(ModelResource &modelResource, const Laphria::MaterialData &material) const
{
	Laphria::VulkanUtils::createDeviceLocalBufferFromData(device, physicalDevice, commandPool, queue,
	                                                      &material, sizeof(Laphria::MaterialData), vk::BufferUsageFlagBits::eStorageBuffer,
	                                                      modelResource.materialBuffer);
}

void GpuResourceRegistry::createSkinningResources(const fastgltf::Asset &gltf, ModelResource &modelResource, const std::vector<Laphria::Vertex> &vertices,
                                                  const std::vector<ModelResource::SkinningInfluence> &skinningInfluences, const std::vector<int> &nodeSkinIndices) const
{
	if (!modelResource.hasSkins || gltf.skins.empty() || vertices.empty())
	{
		return;
	}
	if (!skinningDescriptorSetLayout)
	{
		LOGW("Skinning descriptor set layout not configured; skipping GPU skinning resources for %s", modelResource.name.c_str());
		return;
	}

	modelResource.skinningVertexCount = static_cast<uint32_t>(vertices.size());
	modelResource.meshNodeSkinBySourceNode.clear();
	for (size_t sourceNodeIndex = 0; sourceNodeIndex < nodeSkinIndices.size(); ++sourceNodeIndex)
	{
		if (nodeSkinIndices[sourceNodeIndex] >= 0)
		{
			modelResource.meshNodeSkinBySourceNode[static_cast<int>(sourceNodeIndex)] = nodeSkinIndices[sourceNodeIndex];
		}
	}

	modelResource.skins.clear();
	uint32_t totalJointMatrixCount = 0;
	for (size_t skinIndex = 0; skinIndex < gltf.skins.size(); ++skinIndex)
	{
		const auto &skin = gltf.skins[skinIndex];

		ModelResource::SkinData skinData;
		skinData.name = skin.name.empty() ? ("skin_" + std::to_string(skinIndex)) : std::string(skin.name.c_str());
		skinData.jointMatrixOffset = totalJointMatrixCount;
		skinData.jointSourceNodeIndices.reserve(skin.joints.size());
		for (size_t jointNodeIndex : skin.joints)
		{
			skinData.jointSourceNodeIndices.push_back(static_cast<int>(jointNodeIndex));
		}

		skinData.inverseBindMatrices.assign(skinData.jointSourceNodeIndices.size(), glm::mat4(1.0f));
		if (skin.inverseBindMatrices.has_value())
		{
			const size_t ibmAccessorIndex = skin.inverseBindMatrices.value();
			if (ibmAccessorIndex < gltf.accessors.size())
			{
				const auto &accessor = gltf.accessors[ibmAccessorIndex];
				const size_t writeCount = std::min<size_t>(accessor.count, skinData.inverseBindMatrices.size());
				fastgltf::iterateAccessorWithIndex<fastgltf::math::fmat4x4>(gltf, accessor, [&](fastgltf::math::fmat4x4 m, size_t i) {
					if (i >= writeCount)
					{
						return;
					}
					glm::mat4 matrix;
					memcpy(&matrix, m.data(), sizeof(glm::mat4));
					skinData.inverseBindMatrices[i] = matrix;
				});
			}
		}

		totalJointMatrixCount += static_cast<uint32_t>(skinData.jointSourceNodeIndices.size());
		modelResource.skins.push_back(std::move(skinData));
	}

	modelResource.skinningJointMatrixCount = totalJointMatrixCount;
	if (modelResource.skinningJointMatrixCount == 0)
	{
		LOGW("Skinning detected but joint palette is empty for %s", modelResource.name.c_str());
		return;
	}

	std::vector<ModelResource::SkinningInfluence> influences = skinningInfluences;
	if (influences.size() != vertices.size())
	{
		influences.assign(vertices.size(), ModelResource::SkinningInfluence{});
	}
	for (auto &influence : influences)
	{
		uint32_t skinIndex = influence.skinIndex;
		if (skinIndex >= modelResource.skins.size())
		{
			skinIndex = 0;
		}
		const uint32_t jointOffset = (skinIndex < modelResource.skins.size()) ? modelResource.skins[skinIndex].jointMatrixOffset : 0u;
		influence.joints += glm::uvec4(jointOffset, jointOffset, jointOffset, jointOffset);
	}

	Laphria::VulkanUtils::createDeviceLocalBufferFromData(
	    device, physicalDevice, commandPool, queue,
	    influences.data(), sizeof(ModelResource::SkinningInfluence) * influences.size(),
	    vk::BufferUsageFlagBits::eStorageBuffer,
	    modelResource.skinningInfluenceBuffer);

	constexpr vk::BufferUsageFlags skinnedVertexUsage = vk::BufferUsageFlagBits::eVertexBuffer |
	                                                    vk::BufferUsageFlagBits::eStorageBuffer |
	                                                    vk::BufferUsageFlagBits::eShaderDeviceAddress |
	                                                    vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
	Laphria::VulkanUtils::createDeviceLocalBufferFromData(
	    device, physicalDevice, commandPool, queue,
	    vertices.data(), sizeof(Laphria::Vertex) * vertices.size(), skinnedVertexUsage,
	    modelResource.skinnedVertexBuffer);

	const vk::DeviceSize jointPaletteBufferSize = sizeof(glm::mat4) * modelResource.skinningJointMatrixCount;
	Laphria::VulkanUtils::createBuffer(
	    device, physicalDevice, jointPaletteBufferSize,
	    vk::BufferUsageFlagBits::eStorageBuffer,
	    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
	    modelResource.skinningJointMatrixBuffer);
	modelResource.skinningJointMatricesMapped = modelResource.skinningJointMatrixBuffer.memory.mapMemory(0, jointPaletteBufferSize);
	std::vector<glm::mat4> identityPalette(modelResource.skinningJointMatrixCount, glm::mat4(1.0f));
	memcpy(modelResource.skinningJointMatricesMapped, identityPalette.data(), jointPaletteBufferSize);

	vk::DescriptorSetAllocateInfo allocInfo{
	    .descriptorPool = *descriptorPool,
	    .descriptorSetCount = 1,
	    .pSetLayouts = &skinningDescriptorSetLayout};
	modelResource.skinningDescriptorSet = std::move(vk::raii::DescriptorSets(device, allocInfo).front());

	vk::DescriptorBufferInfo sourceVerticesInfo{
	    .buffer = *modelResource.vertexBuffer,
	    .offset = 0,
	    .range = VK_WHOLE_SIZE};
	vk::DescriptorBufferInfo skinnedVerticesInfo{
	    .buffer = *modelResource.skinnedVertexBuffer,
	    .offset = 0,
	    .range = VK_WHOLE_SIZE};
	vk::DescriptorBufferInfo influenceInfo{
	    .buffer = *modelResource.skinningInfluenceBuffer,
	    .offset = 0,
	    .range = VK_WHOLE_SIZE};
	vk::DescriptorBufferInfo jointMatricesInfo{
	    .buffer = *modelResource.skinningJointMatrixBuffer,
	    .offset = 0,
	    .range = VK_WHOLE_SIZE};

	std::array<vk::WriteDescriptorSet, 4> writes = {
	    vk::WriteDescriptorSet{
	        .dstSet = *modelResource.skinningDescriptorSet,
	        .dstBinding = 0,
	        .dstArrayElement = 0,
	        .descriptorCount = 1,
	        .descriptorType = vk::DescriptorType::eStorageBuffer,
	        .pBufferInfo = &sourceVerticesInfo},
	    vk::WriteDescriptorSet{
	        .dstSet = *modelResource.skinningDescriptorSet,
	        .dstBinding = 1,
	        .dstArrayElement = 0,
	        .descriptorCount = 1,
	        .descriptorType = vk::DescriptorType::eStorageBuffer,
	        .pBufferInfo = &skinnedVerticesInfo},
	    vk::WriteDescriptorSet{
	        .dstSet = *modelResource.skinningDescriptorSet,
	        .dstBinding = 2,
	        .dstArrayElement = 0,
	        .descriptorCount = 1,
	        .descriptorType = vk::DescriptorType::eStorageBuffer,
	        .pBufferInfo = &influenceInfo},
	    vk::WriteDescriptorSet{
	        .dstSet = *modelResource.skinningDescriptorSet,
	        .dstBinding = 3,
	        .dstArrayElement = 0,
	        .descriptorCount = 1,
	        .descriptorType = vk::DescriptorType::eStorageBuffer,
	        .pBufferInfo = &jointMatricesInfo}};
	device.updateDescriptorSets(writes, nullptr);

	modelResource.hasRuntimeSkinning = true;
}

void GpuResourceRegistry::createModelDescriptorSet(ModelResource &modelResource, vk::DescriptorSetLayout layout) const
{
	uint32_t variableDescCounts[] = {1000};
	vk::DescriptorSetVariableDescriptorCountAllocateInfo variableDescriptorCountAllocInfo;
	variableDescriptorCountAllocInfo.descriptorSetCount = 1;
	variableDescriptorCountAllocInfo.pDescriptorCounts = variableDescCounts;

	vk::DescriptorSetAllocateInfo allocInfo{};
	allocInfo.pNext = &variableDescriptorCountAllocInfo;
	allocInfo.descriptorPool = *descriptorPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &layout;
	modelResource.descriptorSet = std::move(vk::raii::DescriptorSets(device, allocInfo).front());

	std::vector<vk::WriteDescriptorSet> writes;

	vk::DescriptorBufferInfo matBufferInfo{};
	if (*modelResource.materialBuffer)
	{
		matBufferInfo.buffer = *modelResource.materialBuffer;
		matBufferInfo.offset = 0;
		matBufferInfo.range = VK_WHOLE_SIZE;
	}
	vk::WriteDescriptorSet matWrite{};
	matWrite.dstSet = *modelResource.descriptorSet;
	matWrite.dstBinding = 0;
	matWrite.dstArrayElement = 0;
	matWrite.descriptorType = vk::DescriptorType::eStorageBuffer;
	matWrite.descriptorCount = 1;
	matWrite.pBufferInfo = &matBufferInfo;
	if (*modelResource.materialBuffer)
	{
		writes.push_back(matWrite);
	}

	std::vector<vk::DescriptorImageInfo> imageInfos;
	if (!modelResource.textureImageViews.empty())
	{
		imageInfos.reserve(modelResource.textureImageViews.size());
		for (size_t i = 0; i < modelResource.textureImageViews.size(); ++i)
		{
			imageInfos.push_back({
			    *modelResource.textureSamplers[i],
			    *modelResource.textureImageViews[i],
			    vk::ImageLayout::eShaderReadOnlyOptimal});
		}
		vk::WriteDescriptorSet texWrite{};
		texWrite.dstSet = *modelResource.descriptorSet;
		texWrite.dstBinding = 1;
		texWrite.dstArrayElement = 0;
		texWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
		texWrite.descriptorCount = static_cast<uint32_t>(imageInfos.size());
		texWrite.pImageInfo = imageInfos.data();
		writes.push_back(texWrite);
	}

	device.updateDescriptorSets(writes, nullptr);
}

void GpuResourceRegistry::buildBLAS(ModelResource &modelResource, const std::vector<Laphria::Vertex> &vertices, const std::vector<uint32_t> &indices) const
{
	if (modelResource.meshes.empty() || !*modelResource.vertexBuffer || !*modelResource.indexBuffer)
	{
		return;
	}

	vk::DeviceAddress vertexAddress = Laphria::VulkanUtils::getBufferDeviceAddress(device, modelResource.vertexBuffer);
	vk::DeviceAddress indexAddress = Laphria::VulkanUtils::getBufferDeviceAddress(device, modelResource.indexBuffer);

	for (const auto &mesh : modelResource.meshes)
	{
		std::vector<vk::AccelerationStructureGeometryKHR> geometries;
		std::vector<vk::AccelerationStructureBuildRangeInfoKHR> buildRanges;
		std::vector<uint32_t> maxPrimitiveCounts;

		for (size_t primIdx = 0; primIdx < mesh.primitives.size(); ++primIdx)
		{
			const auto &prim = mesh.primitives[primIdx];

			vk::AccelerationStructureGeometryKHR geometry{};
			geometry.geometryType = vk::GeometryTypeKHR::eTriangles;
			auto &triangles = geometry.geometry.triangles;
			triangles.vertexFormat = vk::Format::eR32G32B32Sfloat;
			triangles.vertexData.deviceAddress = vertexAddress;
			triangles.vertexStride = sizeof(Laphria::Vertex);

			const uint32_t nextVertexOffset = (primIdx + 1 < mesh.primitives.size()) ? mesh.primitives[primIdx + 1].vertexOffset : static_cast<uint32_t>(vertices.size());
			triangles.maxVertex = nextVertexOffset - prim.vertexOffset - 1;

			triangles.indexType = vk::IndexType::eUint32;
			triangles.indexData.deviceAddress = indexAddress;
			triangles.transformData.deviceAddress = 0;

			geometries.push_back(geometry);

			vk::AccelerationStructureBuildRangeInfoKHR range{};
			range.primitiveCount = prim.indexCount / 3;
			range.primitiveOffset = prim.firstIndex * sizeof(uint32_t);
			range.firstVertex = prim.vertexOffset;
			range.transformOffset = 0;

			buildRanges.push_back(range);
			maxPrimitiveCounts.push_back(range.primitiveCount);
		}

		if (geometries.empty())
		{
			continue;
		}

		vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{};
		buildInfo.type = vk::AccelerationStructureTypeKHR::eBottomLevel;
		buildInfo.flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
		if (modelResource.hasRuntimeSkinning)
		{
			buildInfo.flags |= vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate;
		}
		buildInfo.mode = vk::BuildAccelerationStructureModeKHR::eBuild;
		buildInfo.geometryCount = geometries.size();
		buildInfo.pGeometries = geometries.data();

		vk::AccelerationStructureBuildSizesInfoKHR sizeInfo = device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo, maxPrimitiveCounts);

		Laphria::VulkanUtils::VmaBuffer blasBuffer{};
		Laphria::VulkanUtils::createBuffer(device, physicalDevice, sizeInfo.accelerationStructureSize,
		                                   vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
		                                   vk::MemoryPropertyFlagBits::eDeviceLocal, blasBuffer);

		modelResource.blasBuffers.push_back(std::move(blasBuffer));

		vk::AccelerationStructureCreateInfoKHR createInfo{};
		createInfo.buffer = *modelResource.blasBuffers.back();
		createInfo.size = sizeInfo.accelerationStructureSize;
		createInfo.type = vk::AccelerationStructureTypeKHR::eBottomLevel;

		vk::raii::AccelerationStructureKHR blas = vk::raii::AccelerationStructureKHR(device, createInfo);

		Laphria::VulkanUtils::VmaBuffer scratchBuffer{};
		const vk::DeviceSize scratchSize = std::max(sizeInfo.buildScratchSize, sizeInfo.updateScratchSize);
		Laphria::VulkanUtils::createBuffer(device, physicalDevice, scratchSize,
		                                   vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
		                                   vk::MemoryPropertyFlagBits::eDeviceLocal, scratchBuffer);

		buildInfo.dstAccelerationStructure = *blas;
		buildInfo.scratchData.deviceAddress = Laphria::VulkanUtils::getBufferDeviceAddress(device, scratchBuffer);

		auto cmd = Laphria::VulkanUtils::beginSingleTimeCommands(device, commandPool);
		const vk::AccelerationStructureBuildRangeInfoKHR *pBuildRanges = buildRanges.data();
		cmd.buildAccelerationStructuresKHR(buildInfo, pBuildRanges);
		Laphria::VulkanUtils::endSingleTimeCommands(device, queue, commandPool, cmd);

		modelResource.blasElements.push_back(std::move(blas));
		modelResource.blasScratchBuffers.push_back(std::move(scratchBuffer));
	}
}
