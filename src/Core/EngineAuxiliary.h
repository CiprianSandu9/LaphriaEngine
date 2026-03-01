#ifndef LAPHRIAENGINE_ENGINEAUXILIARY_H
#define LAPHRIAENGINE_ENGINEAUXILIARY_H

#ifndef GLM_FORCE_RADIANS
#	define GLM_FORCE_RADIANS
#endif
#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE
#	define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif
#ifndef GLM_ENABLE_EXPERIMENTAL
#	define GLM_ENABLE_EXPERIMENTAL
#endif
#ifndef GLM_FORCE_CXX11
#	define GLM_FORCE_CXX11
#endif

#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

// Include KTX library for texture loading
#include <ktx.h>

// Include stb_image for decoding embedded PNG/JPEG in glTF
// #define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// Define AAssetManager type for non-Android platforms
typedef void AssetManagerType;
// Desktop-specific includes
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

// Define logging macros for Desktop
#define LOGI(...)        \
	printf(__VA_ARGS__); \
	printf("\n")
#define LOGW(...)        \
	printf(__VA_ARGS__); \
	printf("\n")
#define LOGE(...)                 \
	fprintf(stderr, __VA_ARGS__); \
	fprintf(stderr, "\n")

constexpr uint32_t WIDTH                = 1920;
constexpr uint32_t HEIGHT               = 1080;
constexpr int      MAX_FRAMES_IN_FLIGHT = 2;
constexpr uint32_t NUM_SHADOW_CASCADES  = 4;
constexpr uint32_t SHADOW_MAP_DIM       = 2048;

namespace Laphria
{
#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

struct Vertex
{
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec4 tangent;
	glm::vec2 texCoord;
	glm::vec3 color;

	static vk::VertexInputBindingDescription getBindingDescription()
	{
		return {0, sizeof(Vertex), vk::VertexInputRate::eVertex};
	}

	static std::array<vk::VertexInputAttributeDescription, 5> getAttributeDescriptions()
	{
		return {
		    vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)),
		    vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, normal)),
		    vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Vertex, tangent)),
		    vk::VertexInputAttributeDescription(3, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord)),
		    vk::VertexInputAttributeDescription(4, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color))};
	}

	bool operator==(const Vertex &other) const
	{
		return pos == other.pos && normal == other.normal && tangent == other.tangent &&
		       texCoord == other.texCoord && color == other.color;
	}
};

struct UniformBufferObject
{
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;
	alignas(16) glm::vec4 cameraPos;        // vec4 to ensure exact 16-byte alignment without float padding ambiguities

	alignas(16) glm::vec4 lightDir;

	alignas(16) glm::mat4 viewInverse;
	alignas(16) glm::mat4 projInverse;

	// Cascaded Shadow Map data
	// cascadeSplits: far-plane depth for each cascade in view space (positive, in camera units)
	alignas(16) glm::vec4 cascadeSplits;
	alignas(16) glm::mat4 cascadeViewProj[NUM_SHADOW_CASCADES];
};

struct ScenePushConstants
{
	alignas(16) glm::mat4 modelMatrix;
	alignas(4) int materialIndex;
	alignas(4) int cascadeIndex;        // which CSM cascade is being rendered (shadow pass); padding for main pass
	alignas(4) int padding2;
	alignas(4) int padding3;
	alignas(16) glm::vec4 skyData;        // xyz = color, w = threshold
};

struct MaterialPushConstants
{
	// Texture indices (-1 means no texture) - 8 x int32 = 32 bytes
	int32_t baseColorIndex;
	int32_t metallicRoughnessIndex;
	int32_t normalIndex;
	int32_t occlusionIndex;
	int32_t emissiveIndex;
	int32_t padding1;
	int32_t padding2;
	int32_t padding3;

	// Material factors
	glm::vec4 baseColorFactor;          // 16 bytes (offset 32)
	float     metallicFactor;           // 4 bytes (offset 48)
	float     roughnessFactor;          // 4 bytes (offset 52)
	float     normalScale;              // 4 bytes (offset 56)
	float     occlusionStrength;        // 4 bytes (offset 60)
	glm::vec3 emissiveFactor;           // 12 bytes (offset 64)
	float     alphaCutoff;              // 4 bytes (offset 76)
	                                    // Total: 80 bytes
};

// PBR Material data to be sent to shaders (SSBO equivalent)
struct MaterialData
{
	int32_t  baseColorIndex         = -1;
	int32_t  metallicRoughnessIndex = -1;
	int32_t  normalIndex            = -1;
	int32_t  occlusionIndex         = -1;
	int32_t  emissiveIndex          = -1;
	int32_t  specularTextureIndex   = -1;
	uint32_t firstIndex             = 0;
	uint32_t vertexOffset           = 0;
	int32_t  globalTextureOffset    = 0;

	alignas(16) glm::vec4 baseColorFactor = glm::vec4(1.0f);
	alignas(4) float metallicFactor       = 1.0f;
	alignas(4) float roughnessFactor      = 1.0f;
	alignas(4) float normalScale          = 1.0f;
	alignas(4) float occlusionStrength    = 1.0f;
	alignas(16) glm::vec3 emissiveFactor  = glm::vec3(0.0f);
	alignas(4) float specularFactor       = 1.0f;
	alignas(4) float alphaCutoff          = 0.5f;
};

// CPU-side material description
struct PBRMaterial
{
	MaterialData data;
	int32_t      baseColorTextureIndex         = -1;
	int32_t      metallicRoughnessTextureIndex = -1;
	int32_t      normalTextureIndex            = -1;
	int32_t      occlusionTextureIndex         = -1;
	int32_t      emissiveTextureIndex          = -1;
	int32_t      specularTextureIndex          = -1;
};

// Mesh primitive with material reference
struct MeshPrimitive
{
	uint32_t firstIndex;
	uint32_t indexCount;
	uint32_t vertexOffset;
	int32_t  materialIndex      = -1;
	uint32_t flatPrimitiveIndex = 0;
};

struct LoadedMesh
{
	std::string                name;
	std::vector<MeshPrimitive> primitives;
};
}        // namespace Laphria

template <>
struct std::hash<Laphria::Vertex>
{
	size_t operator()(Laphria::Vertex const &vertex) const noexcept
	{
		return ((std::hash<glm::vec3>()(vertex.pos) ^ (std::hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ (std::hash<glm::vec2>()(vertex.texCoord) << 1);
	}
};

#endif        // LAPHRIAENGINE_ENGINEAUXILIARY_H
