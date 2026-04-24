#ifndef LAPHRIAENGINE_ENGINEAUXILIARY_H
#define LAPHRIAENGINE_ENGINEAUXILIARY_H

#include <cstdio>
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
#define LOGI(...)                                                                 \
	do                                                                            \
	{                                                                             \
		std::fprintf(stdout, "[INFO] ");                                          \
		std::fprintf(stdout, __VA_ARGS__);                                        \
		std::fprintf(stdout, "\n");                                               \
	} while (0)
#define LOGW(...)                                                                 \
	do                                                                            \
	{                                                                             \
		std::fprintf(stdout, "[WARN] ");                                          \
		std::fprintf(stdout, __VA_ARGS__);                                        \
		std::fprintf(stdout, "\n");                                               \
	} while (0)
#define LOGE(...)                                                                 \
	do                                                                            \
	{                                                                             \
		std::fprintf(stderr, "[ERROR] ");                                         \
		std::fprintf(stderr, __VA_ARGS__);                                        \
		std::fprintf(stderr, "\n");                                               \
	} while (0)

constexpr uint32_t WIDTH                = 1920;
constexpr uint32_t HEIGHT               = 1080;
constexpr int      MAX_FRAMES_IN_FLIGHT = 2;
constexpr uint32_t NUM_SHADOW_CASCADES  = 4;
constexpr uint32_t SHADOW_MAP_DIM       = 2048;

// Selects which rendering backend is active.
enum class RenderMode
{
	Rasterizer,   // shadow + starfield compute + raster graphics pipeline
	RayTracer,    // classic RT: direct illumination with RT shadows, tone-mapped in ClosestHit
	PathTracer,   // path tracer with temporal reprojection and A-Trous denoiser
};

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

	// Path tracer temporal fields
	alignas(16) glm::mat4 prevViewProj;   // previous frame VP for motion vector reprojection
	alignas(4)  uint32_t  frameCount;     // monotonically increasing; seeds per-pixel RNG in Raygen
	alignas(4)  float     jitter_x;       // sub-pixel x jitter in NDC (Halton sequence, zero when TAA disabled)
	alignas(4)  float     jitter_y;       // sub-pixel y jitter in NDC
	alignas(4)  uint32_t  _pad0;          // padding for 16-byte struct alignment
	alignas(16) glm::vec4 gameplayVisuals = glm::vec4(1.0f); // x=sun, y=fill, z=ambient, w=exposure multipliers
};

struct DenoisePushConstants
{
	int32_t stepSize;    // A-Trous step size: 1, 2, 4, 8, 16 for iterations 0-4; unused in reprojection pass
	int32_t isLastPass;  // 1 on the final A-Trous iteration: triggers tone mapping + history copy
	float   phiColor;    // luminance edge-stopping weight (typical: 10.0)
	float   phiNormal;   // normal edge-stopping exponent (typical: 128.0)
	float   exposureScale; // gameplay exposure multiplier applied on final denoise pass
};

struct SkinningPushConstants
{
	alignas(4) uint32_t vertexCount = 0;
	alignas(4) uint32_t jointMatrixOffset = 0;
	alignas(4) uint32_t jointCount = 0;
	alignas(4) uint32_t _pad = 0;
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
