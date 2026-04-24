#ifndef LAPHRIAENGINE_RESOURCEMANAGER_H
#define LAPHRIAENGINE_RESOURCEMANAGER_H

#include "../SceneManagement/SceneNode.h"
#include "EngineAuxiliary.h"
#include "VulkanUtils.h"
#include <fastgltf/types.hpp>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan_raii.hpp>

// Struct to hold GPU resources for a loaded model
struct ModelResource
{
	~ModelResource() = default;

	enum class AnimationInterpolationMode
	{
		Linear,
		Step
	};

	struct AnimationTrackVec3
	{
		std::vector<float>     keyTimes;
		std::vector<glm::vec3> keyValues;
		AnimationInterpolationMode interpolation = AnimationInterpolationMode::Linear;
	};

	struct AnimationTrackQuat
	{
		std::vector<float>     keyTimes;
		std::vector<glm::quat> keyValues;
		AnimationInterpolationMode interpolation = AnimationInterpolationMode::Linear;
	};

	struct AnimationNodeTracks
	{
		std::optional<AnimationTrackVec3> translation;
		std::optional<AnimationTrackQuat> rotation;
		std::optional<AnimationTrackVec3> scale;
	};

	struct AnimationClip
	{
		std::string                                 id;
		float                                       durationSeconds = 0.0f;
		std::unordered_map<int, AnimationNodeTracks> nodeTracks;
	};

	struct SkinningInfluence
	{
		glm::uvec4 joints{0u, 0u, 0u, 0u};
		glm::vec4  weights{1.0f, 0.0f, 0.0f, 0.0f};
		uint32_t   skinIndex = 0;
		glm::uvec3 _pad{0u, 0u, 0u};
	};

	struct SkinData
	{
		std::string      name;
		std::vector<int> jointSourceNodeIndices;
		std::vector<glm::mat4> inverseBindMatrices;
		uint32_t         jointMatrixOffset = 0;
	};

	std::string name;
	std::string path;
	int         globalTextureOffset = 0;
	bool        hasAnimations = false;
	bool        hasSkins = false;
	bool        dynamicGeometry = false;
	bool        hasRuntimeSkinning = false;
	uint32_t    skinningVertexCount = 0;
	uint32_t    skinningJointMatrixCount = 0;
	uint32_t    vertexCount = 0;
	uint32_t    indexCount = 0;
	std::vector<std::string> animationClipNames;
	std::vector<AnimationClip> animationClips;
	std::vector<SkinData> skins;
	std::unordered_map<int, int> meshNodeSkinBySourceNode;

	// One buffer per model for now
	Laphria::VulkanUtils::VmaBuffer vertexBuffer;
	Laphria::VulkanUtils::VmaBuffer indexBuffer;
	Laphria::VulkanUtils::VmaBuffer skinnedVertexBuffer;
	Laphria::VulkanUtils::VmaBuffer skinningInfluenceBuffer;
	Laphria::VulkanUtils::VmaBuffer skinningJointMatrixBuffer;
	void                   *skinningJointMatricesMapped = nullptr;

	// CPU side info to map mesh primitives to buffer offsets
	std::vector<Laphria::LoadedMesh> meshes;

	// Materials associated with this model
	std::vector<Laphria::PBRMaterial> materials;
	Laphria::VulkanUtils::VmaBuffer materialBuffer;

	std::vector<Laphria::VulkanUtils::VmaImage> textureImages;
	std::vector<vk::raii::ImageView>            textureImageViews;
	std::vector<vk::raii::Sampler>              textureSamplers;

	// Resource Binding
	vk::raii::DescriptorSet descriptorSet{nullptr};        // Set 1: Materials + Textures
	vk::raii::DescriptorSet skinningDescriptorSet{nullptr};

	// Ray Tracing
	std::vector<vk::raii::AccelerationStructureKHR> blasElements;
	std::vector<Laphria::VulkanUtils::VmaBuffer>    blasBuffers;
	std::vector<Laphria::VulkanUtils::VmaBuffer>    blasScratchBuffers;

	// Prototype for caching (Scene Graph Hierarchy)
	SceneNode::Ptr prototype{nullptr};
};

struct ModelImportReport
{
	std::string              modelPath;
	std::vector<std::string> warnings;
	std::vector<std::string> errors;
	std::vector<std::string> supportedFeatures;
	bool                     hasAnimations = false;
	bool                     hasSkins = false;
};

class GltfImporter;
class GpuResourceRegistry;

class ResourceManager
{
  public:
	ResourceManager(vk::raii::Device &device, vk::raii::PhysicalDevice &physicalDevice, vk::raii::CommandPool &commandPool, vk::raii::Queue &queue, vk::raii::DescriptorPool &descriptorPool);
	~ResourceManager();

	// Load a GLTF model and return the root node of the constructed hierarchy
	SceneNode::Ptr loadGltfModel(const std::string &path, vk::DescriptorSetLayout layout);
	void setSkinningDescriptorSetLayout(vk::DescriptorSetLayout layout) const;

	// Primitives
	SceneNode::Ptr createSphereModel(float radius, int slices, int stacks, vk::DescriptorSetLayout layout);

	SceneNode::Ptr createCubeModel(float size, vk::DescriptorSetLayout layout);
	SceneNode::Ptr createCubeModel(float size, vk::DescriptorSetLayout layout, const Laphria::MaterialData &materialOverride);

	SceneNode::Ptr createCylinderModel(float radius, float height, int slices, vk::DescriptorSetLayout layout);

	// Get resource by internal ID (SceneNode will store these ID)
	[[nodiscard]] ModelResource *getModelResource(int id) const;
	[[nodiscard]] const ModelImportReport *getLastImportReport() const;
	[[nodiscard]] const ModelResource::AnimationClip *findAnimationClip(int modelId, const std::string &clipId) const;
	[[nodiscard]] float getAnimationClipDurationSeconds(int modelId, const std::string &clipId) const;
	[[nodiscard]] bool hasRuntimeSkinnedModels() const;

	[[nodiscard]] size_t getModelCount() const
	{
		return models.size();
	}

	// Helpers for rendering
	void bindResources(const vk::raii::CommandBuffer &cmd, int modelId, bool useSkinnedVertices = false) const;
	void recordSkinnedBLASRefit(const vk::raii::CommandBuffer &cmd) const;

  private:
	vk::raii::Device         &device;
	vk::raii::PhysicalDevice &physicalDevice;
	vk::raii::CommandPool    &commandPool;
	vk::raii::Queue          &queue;
	vk::raii::DescriptorPool &descriptorPool;

	std::vector<std::unique_ptr<ModelResource>> models;
	std::optional<ModelImportReport>            lastImportReport;
	std::unique_ptr<GltfImporter>               gltfImporter;
	std::unique_ptr<GpuResourceRegistry>        gpuResourceRegistry;

	void loadTextures(const fastgltf::Asset &gltf, const std::filesystem::path &modelDir, ModelResource *modelRes) const;

	bool prepareKTXFromMemory(const unsigned char *data, size_t length, Laphria::VulkanUtils::VmaImage &outImage, uint32_t &width, uint32_t &height,
	                          vk::Format &format) const;

	void prepareTextureFromPixels(const unsigned char *pixels, int width, int height, Laphria::VulkanUtils::VmaImage &outImage, vk::Format &format) const;

	void finalizeProceduralModel(ModelResource *modelRes, const std::vector<Laphria::Vertex> &vertices, const std::vector<uint32_t> &indices,
	                             vk::DescriptorSetLayout layout, const std::string &meshName,
	                             const std::optional<Laphria::MaterialData> &materialOverride = std::nullopt) const;

	std::unordered_map<std::string, int> loadedModels;
};        // End of ResourceManager class

#endif        // LAPHRIAENGINE_RESOURCEMANAGER_H
