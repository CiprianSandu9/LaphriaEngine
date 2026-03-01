#ifndef LAPHRIAENGINE_RESOURCEMANAGER_H
#define LAPHRIAENGINE_RESOURCEMANAGER_H

#include "../SceneManagement/SceneNode.h"
#include "EngineAuxiliary.h"
#include <fastgltf/types.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan_raii.hpp>

using namespace Laphria;

// Struct to hold GPU resources for a loaded model
struct ModelResource
{
	std::string name;
	std::string path;
	int         globalTextureOffset = 0;

	// One buffer per model for now
	vk::raii::Buffer       vertexBuffer{nullptr};
	vk::raii::DeviceMemory vertexBufferMemory{nullptr};
	vk::raii::Buffer       indexBuffer{nullptr};
	vk::raii::DeviceMemory indexBufferMemory{nullptr};

	// CPU side info to map mesh primitives to buffer offsets
	std::vector<LoadedMesh> meshes;

	// Materials associated with this model
	std::vector<PBRMaterial> materials;
	vk::raii::Buffer         materialBuffer{nullptr};
	vk::raii::DeviceMemory   materialBufferMemory{nullptr};

	std::vector<vk::raii::Image>        textureImages;
	std::vector<vk::raii::DeviceMemory> textureImageMemories;
	std::vector<vk::raii::ImageView>    textureImageViews;
	std::vector<vk::raii::Sampler>      textureSamplers;

	// Resource Binding
	vk::raii::DescriptorSet descriptorSet{nullptr};        // Set 1: Materials + Textures

	// Ray Tracing
	std::vector<vk::raii::AccelerationStructureKHR> blasElements;
	std::vector<vk::raii::Buffer>                   blasBuffers;
	std::vector<vk::raii::DeviceMemory>             blasMemories;

	// Prototype for caching (Scene Graph Hierarchy)
	SceneNode::Ptr prototype{nullptr};
};

class ResourceManager
{
  public:
	ResourceManager(vk::raii::Device &device, vk::raii::PhysicalDevice &physicalDevice, vk::raii::CommandPool &commandPool, vk::raii::Queue &queue, vk::raii::DescriptorPool &descriptorPool);

	// Load a GLTF model and return the root node of the constructed hierarchy
	SceneNode::Ptr loadGltfModel(const std::string &path, vk::DescriptorSetLayout layout);

	// Primitives
	SceneNode::Ptr createSphereModel(float radius, int slices, int stacks, vk::DescriptorSetLayout layout);

	SceneNode::Ptr createCubeModel(float size, vk::DescriptorSetLayout layout);

	SceneNode::Ptr createCylinderModel(float radius, float height, int slices, vk::DescriptorSetLayout layout);

	// Get resource by internal ID (SceneNode will store these ID)
	[[nodiscard]] ModelResource *getModelResource(int id) const;

	[[nodiscard]] size_t getModelCount() const
	{
		return models.size();
	}

	// Helpers for rendering
	void bindResources(const vk::raii::CommandBuffer &cmd, int modelId) const;

  public:
	vk::raii::Device         &device;
	vk::raii::PhysicalDevice &physicalDevice;
	vk::raii::CommandPool    &commandPool;
	vk::raii::Queue          &queue;
	vk::raii::DescriptorPool &descriptorPool;

  private:
	std::vector<std::unique_ptr<ModelResource>> models;

  public:
  private:
	void loadTextures(const fastgltf::Asset &gltf, const std::filesystem::path &modelDir, ModelResource *modelRes);

	bool prepareKTXFromMemory(const unsigned char *data, size_t length, vk::raii::Image &outImage, vk::raii::DeviceMemory &outMem, uint32_t &width, uint32_t &height,
	                          vk::Format &format) const;

	void prepareTextureFromPixels(const unsigned char *pixels, int width, int height, vk::raii::Image &outImage, vk::raii::DeviceMemory &outMem, vk::Format &format) const;

	void loadMaterials(const fastgltf::Asset &gltf, ModelResource *modelRes);

	SceneNode::Ptr processSceneNodes(const fastgltf::Asset &gltf, ModelResource *modelRes, std::vector<Vertex> &vertices, std::vector<uint32_t> &indices);

	SceneNode::Ptr processGltfNode(const fastgltf::Asset &gltf, const fastgltf::Node &node, ModelResource *modelRes, std::vector<Vertex> &vertices, std::vector<uint32_t> &indices);

	void uploadModelBuffers(ModelResource *modelRes, const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices);

	void createModelDescriptorSet(ModelResource *modelRes, vk::DescriptorSetLayout layout);

	void buildBLAS(ModelResource *modelRes, const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices);

	void finalizeProceduralModel(ModelResource *modelRes, const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices, vk::DescriptorSetLayout layout,
	                             const std::string &meshName);

  private:
	std::unordered_map<std::string, int> loadedModels;
};        // End of ResourceManager class

#endif        // LAPHRIAENGINE_RESOURCEMANAGER_H
