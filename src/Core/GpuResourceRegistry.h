#ifndef LAPHRIAENGINE_GPURESOURCEREGISTRY_H
#define LAPHRIAENGINE_GPURESOURCEREGISTRY_H

#include "EngineAuxiliary.h"
#include "ResourceManager.h"

#include <fastgltf/types.hpp>
#include <vector>

class GpuResourceRegistry
{
  public:
	GpuResourceRegistry(vk::raii::Device &device, vk::raii::PhysicalDevice &physicalDevice, vk::raii::CommandPool &commandPool, vk::raii::Queue &queue,
	                    vk::raii::DescriptorPool &descriptorPool);

	void uploadModelBuffers(ModelResource &modelResource, const std::vector<Laphria::Vertex> &vertices, const std::vector<uint32_t> &indices) const;
	void uploadMaterialBuffer(ModelResource &modelResource, const std::vector<Laphria::MaterialData> &materials) const;
	void uploadMaterialBuffer(ModelResource &modelResource, const Laphria::MaterialData &material) const;
	void setSkinningDescriptorSetLayout(vk::DescriptorSetLayout layout);
	void createSkinningResources(const fastgltf::Asset &gltf, ModelResource &modelResource, const std::vector<Laphria::Vertex> &vertices,
	                             const std::vector<ModelResource::SkinningInfluence> &skinningInfluences, const std::vector<int> &nodeSkinIndices) const;
	void createModelDescriptorSet(ModelResource &modelResource, vk::DescriptorSetLayout layout) const;
	void buildBLAS(ModelResource &modelResource, const std::vector<Laphria::Vertex> &vertices, const std::vector<uint32_t> &indices) const;

  private:
	vk::raii::Device         &device;
	vk::raii::PhysicalDevice &physicalDevice;
	vk::raii::CommandPool    &commandPool;
	vk::raii::Queue          &queue;
	vk::raii::DescriptorPool &descriptorPool;
	vk::DescriptorSetLayout   skinningDescriptorSetLayout = nullptr;
};

#endif // LAPHRIAENGINE_GPURESOURCEREGISTRY_H
