#include "ResourceManager.h"
#include "GltfImporter.h"
#include "GpuResourceRegistry.h"
#include "VulkanUtils.h"

#include <fastgltf/types.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ktx.h>

using namespace Laphria;
using Laphria::LoadedMesh;
using Laphria::MaterialData;
using Laphria::PBRMaterial;
using Laphria::Vertex;

namespace
{
static_assert(sizeof(Vertex) == 60, "Skinning shader expects Vertex stride of 60 bytes.");
static_assert(sizeof(ModelResource::SkinningInfluence) == 48, "Skinning shader expects SkinningInfluence stride of 48 bytes.");

enum class ImportTextureRole : uint8_t
{
	Color,
	Linear
};

std::vector<unsigned char> readBinaryFile(const std::filesystem::path &path)
{
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file)
	{
		return {};
	}
	const std::streamsize fileSize = static_cast<std::streamsize>(file.tellg());
	if (fileSize <= 0)
	{
		return {};
	}
	std::vector<unsigned char> bytes(static_cast<size_t>(fileSize));
	file.seekg(0, std::ios::beg);
	file.read(reinterpret_cast<char *>(bytes.data()), fileSize);
	if (!file)
	{
		return {};
	}
	return bytes;
}

bool supportsSampledImageFormat(const vk::raii::PhysicalDevice &physicalDevice, vk::Format format)
{
	const vk::FormatProperties properties = physicalDevice.getFormatProperties(format);
	return static_cast<bool>(properties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImage);
}

bool isSrgbFormat(vk::Format format)
{
	switch (format)
	{
	case vk::Format::eR8G8B8A8Srgb:
	case vk::Format::eB8G8R8A8Srgb:
	case vk::Format::eBc1RgbSrgbBlock:
	case vk::Format::eBc1RgbaSrgbBlock:
	case vk::Format::eBc2SrgbBlock:
	case vk::Format::eBc3SrgbBlock:
	case vk::Format::eBc7SrgbBlock: return true;
	default: return false;
	}
}

vk::Format srgbTwin(vk::Format format)
{
	switch (format)
	{
	case vk::Format::eR8G8B8A8Unorm: return vk::Format::eR8G8B8A8Srgb;
	case vk::Format::eB8G8R8A8Unorm: return vk::Format::eB8G8R8A8Srgb;
	case vk::Format::eBc1RgbUnormBlock: return vk::Format::eBc1RgbSrgbBlock;
	case vk::Format::eBc1RgbaUnormBlock: return vk::Format::eBc1RgbaSrgbBlock;
	case vk::Format::eBc2UnormBlock: return vk::Format::eBc2SrgbBlock;
	case vk::Format::eBc3UnormBlock: return vk::Format::eBc3SrgbBlock;
	case vk::Format::eBc7UnormBlock: return vk::Format::eBc7SrgbBlock;
	default: return vk::Format::eUndefined;
	}
}

vk::Format unormTwin(vk::Format format)
{
	switch (format)
	{
	case vk::Format::eR8G8B8A8Srgb: return vk::Format::eR8G8B8A8Unorm;
	case vk::Format::eB8G8R8A8Srgb: return vk::Format::eB8G8R8A8Unorm;
	case vk::Format::eBc1RgbSrgbBlock: return vk::Format::eBc1RgbUnormBlock;
	case vk::Format::eBc1RgbaSrgbBlock: return vk::Format::eBc1RgbaUnormBlock;
	case vk::Format::eBc2SrgbBlock: return vk::Format::eBc2UnormBlock;
	case vk::Format::eBc3SrgbBlock: return vk::Format::eBc3UnormBlock;
	case vk::Format::eBc7SrgbBlock: return vk::Format::eBc7UnormBlock;
	default: return vk::Format::eUndefined;
	}
}

int resolveTextureImageIndex(const fastgltf::Asset &gltf, size_t textureIndex)
{
	if (textureIndex >= gltf.textures.size())
	{
		return -1;
	}
	const auto &texture = gltf.textures[textureIndex];
	if (texture.imageIndex.has_value())
	{
		return static_cast<int>(texture.imageIndex.value());
	}
	if (texture.basisuImageIndex.has_value())
	{
		return static_cast<int>(texture.basisuImageIndex.value());
	}
	return -1;
}

std::vector<ImportTextureRole> buildTextureRoles(const fastgltf::Asset &gltf, uint32_t &mixedUsageCount)
{
	constexpr uint8_t kUsageColor = 1u << 0u;
	constexpr uint8_t kUsageLinear = 1u << 1u;
	std::vector<uint8_t> usage(gltf.images.size(), 0u);

	auto markTextureUsage = [&](size_t textureIndex, uint8_t usageFlag) {
		const int imageIndex = resolveTextureImageIndex(gltf, textureIndex);
		if (imageIndex >= 0 && static_cast<size_t>(imageIndex) < usage.size())
		{
			usage[static_cast<size_t>(imageIndex)] |= usageFlag;
		}
	};

	for (const auto &material : gltf.materials)
	{
		if (material.pbrData.baseColorTexture.has_value())
		{
			markTextureUsage(material.pbrData.baseColorTexture->textureIndex, kUsageColor);
		}
		if (material.emissiveTexture.has_value())
		{
			markTextureUsage(material.emissiveTexture->textureIndex, kUsageColor);
		}
		if (material.normalTexture.has_value())
		{
			markTextureUsage(material.normalTexture->textureIndex, kUsageLinear);
		}
		if (material.pbrData.metallicRoughnessTexture.has_value())
		{
			markTextureUsage(material.pbrData.metallicRoughnessTexture->textureIndex, kUsageLinear);
		}
		if (material.occlusionTexture.has_value())
		{
			markTextureUsage(material.occlusionTexture->textureIndex, kUsageLinear);
		}
		if (material.specular != nullptr && material.specular->specularTexture.has_value())
		{
			markTextureUsage(material.specular->specularTexture->textureIndex, kUsageLinear);
		}
	}

	std::vector<ImportTextureRole> roles(gltf.images.size(), ImportTextureRole::Linear);
	for (size_t i = 0; i < usage.size(); ++i)
	{
		const bool hasColor = (usage[i] & kUsageColor) != 0u;
		const bool hasLinear = (usage[i] & kUsageLinear) != 0u;
		if (hasColor && hasLinear)
		{
			++mixedUsageCount;
			LOGW("Texture image %zu is used by both color and linear slots; treating as color (SRGB-preferred path).", i);
			roles[i] = ImportTextureRole::Color;
		}
		else if (hasColor)
		{
			roles[i] = ImportTextureRole::Color;
		}
		else
		{
			roles[i] = ImportTextureRole::Linear;
		}
	}
	return roles;
}

void buildSingleMipRgbaPayload(const unsigned char *pixels, uint32_t width, uint32_t height, vk::Format format,
                               VulkanUtils::TextureUploadPayload &payload)
{
	payload.data.assign(pixels, pixels + (static_cast<size_t>(width) * static_cast<size_t>(height) * 4u));
	payload.copyRegions.clear();
	payload.copyRegions.push_back(vk::BufferImageCopy{
	    .bufferOffset = 0,
	    .bufferRowLength = 0,
	    .bufferImageHeight = 0,
	    .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
	    .imageOffset = {0, 0, 0},
	    .imageExtent = {width, height, 1}});
	payload.format = format;
	payload.width = width;
	payload.height = height;
	payload.mipLevels = 1;
	payload.isCompressed = false;
}
}

ResourceManager::ResourceManager(vk::raii::Device &device, vk::raii::PhysicalDevice &physicalDevice, vk::raii::CommandPool &commandPool, vk::raii::Queue &queue,
                                 vk::raii::DescriptorPool &descriptorPool) : device(device),
                                                                             physicalDevice(physicalDevice),
                                                                             commandPool(commandPool),
                                                                             queue(queue),
                                                                             descriptorPool(descriptorPool),
                                                                             gltfImporter(std::make_unique<GltfImporter>()),
                                                                             gpuResourceRegistry(std::make_unique<GpuResourceRegistry>(device, physicalDevice, commandPool, queue, descriptorPool)) {
}

ResourceManager::~ResourceManager() = default;

void ResourceManager::setSkinningDescriptorSetLayout(vk::DescriptorSetLayout layout) const {
    gpuResourceRegistry->setSkinningDescriptorSetLayout(layout);
}

void ResourceManager::setTextureColorSpaceModel(TextureColorSpaceModel model) {
    if (textureColorSpaceModel == model) {
        return;
    }
    textureColorSpaceModel = model;
    loadedModels.clear();
    LOGI("Texture color-space model switched to %s. Model cache cleared; reload assets to apply new texture formats.",
         textureColorSpaceModel == TextureColorSpaceModel::HardwareSrgb ? "HardwareSrgb" : "LegacyManual");
}

// Internal helper struct for texture batching

// Texture Helpers

bool ResourceManager::prepareKTXFromMemory(const unsigned char *data, size_t length, TextureSemanticRole role,
                                           VulkanUtils::TextureUploadPayload &outPayload, std::string &outPathTag,
                                           TextureLoadStats &stats) const {
    static const unsigned char ktx2Magic[12] = {
        0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A
    };
    if (length < 12 || memcmp(data, ktx2Magic, 12) != 0)
        return false;

    ktxTexture2 *ktx2{nullptr};
    if (ktxTexture2_CreateFromMemory(data, length, KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktx2) != KTX_SUCCESS)
        return false;

    auto cleanup = [&]() {
        if (ktx2 != nullptr) {
            ktxTexture_Destroy(ktxTexture(ktx2));
        }
    };

    outPathTag = "native-ktx-vkformat";

    const bool useSrgbForColorRole = (role == TextureSemanticRole::Color) &&
                                     (textureColorSpaceModel == TextureColorSpaceModel::HardwareSrgb);

    if (ktxTexture2_NeedsTranscoding(ktx2)) {
        struct Candidate {
            ktx_transcode_fmt_e target;
            vk::Format format;
            const char *tag;
        };
        const Candidate colorCandidates[] = {
            {KTX_TTF_BC7_RGBA, vk::Format::eBc7SrgbBlock, "basisu->bc7-srgb"},
            {KTX_TTF_BC3_RGBA, vk::Format::eBc3SrgbBlock, "basisu->bc3-srgb"},
            {KTX_TTF_BC1_RGB, vk::Format::eBc1RgbSrgbBlock, "basisu->bc1-srgb"},
            {KTX_TTF_RGBA32, vk::Format::eR8G8B8A8Srgb, "rgba-fallback-srgb"}
        };
        const Candidate linearCandidates[] = {
            {KTX_TTF_BC7_RGBA, vk::Format::eBc7UnormBlock, "basisu->bc7-unorm"},
            {KTX_TTF_BC3_RGBA, vk::Format::eBc3UnormBlock, "basisu->bc3-unorm"},
            {KTX_TTF_BC1_RGB, vk::Format::eBc1RgbUnormBlock, "basisu->bc1-unorm"},
            {KTX_TTF_RGBA32, vk::Format::eR8G8B8A8Unorm, "rgba-fallback-unorm"}
        };
        const Candidate *candidates = useSrgbForColorRole ? colorCandidates : linearCandidates;
        const size_t candidateCount = useSrgbForColorRole ? (sizeof(colorCandidates) / sizeof(colorCandidates[0]))
                                                          : (sizeof(linearCandidates) / sizeof(linearCandidates[0]));

        bool transcoded = false;
        for (size_t candidateIndex = 0; candidateIndex < candidateCount; ++candidateIndex) {
            const auto &candidate = candidates[candidateIndex];
            if (candidate.target != KTX_TTF_RGBA32 && !supportsSampledImageFormat(physicalDevice, candidate.format)) {
                continue;
            }
            if (ktxTexture2_TranscodeBasis(ktx2, candidate.target, 0) == KTX_SUCCESS) {
                outPathTag = candidate.tag;
                transcoded = true;
                break;
            }
        }
        if (!transcoded) {
            cleanup();
            return false;
        }
    }

    vk::Format vkFormat = static_cast<vk::Format>(ktx2->vkFormat);
    if (vkFormat == vk::Format::eUndefined) {
        vkFormat = useSrgbForColorRole ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm;
        outPathTag = "rgba-fallback";
    }
    const bool sourceIsSrgb = isSrgbFormat(vkFormat);
    if (sourceIsSrgb != useSrgbForColorRole) {
        const vk::Format twin = useSrgbForColorRole ? srgbTwin(vkFormat) : unormTwin(vkFormat);
        if (twin == vk::Format::eUndefined) {
            LOGW("Texture format %d mismatched for role; no SRGB/UNORM twin exists. Keeping original format.", static_cast<int>(vkFormat));
        } else {
            ++stats.forcedRemapCount;
            LOGI("Texture format remapped for role: %d -> %d", static_cast<int>(vkFormat), static_cast<int>(twin));
            vkFormat = twin;
        }
    }

    const uint32_t baseWidth = ktx2->baseWidth;
    const uint32_t baseHeight = ktx2->baseHeight;
    const uint32_t mipLevels = std::max<uint32_t>(1, ktx2->numLevels);
    const bool isCompressed = (vkFormat != vk::Format::eR8G8B8A8Unorm);

    const auto *textureData = ktxTexture_GetData(ktxTexture(ktx2));
    const ktx_size_t textureSize = ktxTexture_GetDataSize(ktxTexture(ktx2));
    if (textureData == nullptr || textureSize == 0) {
        cleanup();
        return false;
    }

    outPayload.data.assign(textureData, textureData + textureSize);
    outPayload.copyRegions.clear();
    outPayload.copyRegions.reserve(mipLevels);
    outPayload.format = vkFormat;
    outPayload.width = baseWidth;
    outPayload.height = baseHeight;
    outPayload.mipLevels = mipLevels;
    outPayload.isCompressed = isCompressed;

    ktx_size_t sequentialOffset = 0;
    for (uint32_t mipLevel = 0; mipLevel < mipLevels; ++mipLevel) {
        ktx_size_t imageOffset = 0;
        const bool hasExplicitOffset = (ktxTexture_GetImageOffset(ktxTexture(ktx2), mipLevel, 0, 0, &imageOffset) == KTX_SUCCESS);
        if (!hasExplicitOffset) {
            imageOffset = sequentialOffset;
        }

        const uint32_t mipWidth = std::max<uint32_t>(1, baseWidth >> mipLevel);
        const uint32_t mipHeight = std::max<uint32_t>(1, baseHeight >> mipLevel);
        outPayload.copyRegions.push_back(vk::BufferImageCopy{
            .bufferOffset = imageOffset,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {vk::ImageAspectFlagBits::eColor, mipLevel, 0, 1},
            .imageOffset = {0, 0, 0},
            .imageExtent = {mipWidth, mipHeight, 1}});

        const ktx_size_t imageSize = ktxTexture_GetImageSize(ktxTexture(ktx2), mipLevel);
        sequentialOffset += imageSize;
    }

    cleanup();
    return true;
}

void ResourceManager::loadTextures(const fastgltf::Asset &gltf, const std::filesystem::path &modelDir, ModelResource *modelRes, TextureLoadStats &stats) const {
    const auto textureSources = gltfImporter->buildTextureImportSources(gltf, modelDir);
    if (textureSources.empty()) {
        return;
    }
    const auto textureRoles = buildTextureRoles(gltf, stats.mixedUsageCount);
    LOGI("Texture import: %zu image(s) detected", textureSources.size());

    modelRes->textureImages.reserve(textureSources.size());
    modelRes->textureImageViews.reserve(textureSources.size());
    modelRes->textureSamplers.reserve(textureSources.size());

    constexpr size_t maxBatchTextures = 16;
    constexpr size_t maxBatchBytes = 256ull * 1024ull * 1024ull; // 256 MiB of staging per submit.

    vk::raii::CommandBuffer uploadCommandBuffer{nullptr};
    std::vector<vk::raii::Buffer> stagingBuffers;
    std::vector<vk::raii::DeviceMemory> stagingMemories;
    stagingBuffers.reserve(maxBatchTextures);
    stagingMemories.reserve(maxBatchTextures);
    size_t batchTextureCount = 0;
    size_t batchBytes = 0;
    size_t submittedTextureCount = 0;

    auto beginBatch = [&]() {
        uploadCommandBuffer = VulkanUtils::beginSingleTimeCommands(device, commandPool);
    };
    auto flushBatch = [&]() {
        if (batchTextureCount == 0) {
            return;
        }
        const auto uploadSubmitStart = std::chrono::high_resolution_clock::now();
        VulkanUtils::endSingleTimeCommands(device, queue, commandPool, uploadCommandBuffer);
        const auto uploadSubmitEnd = std::chrono::high_resolution_clock::now();
        stats.uploadMs += std::chrono::duration<double, std::milli>(uploadSubmitEnd - uploadSubmitStart).count();

        submittedTextureCount += batchTextureCount;
        LOGI("Texture upload progress: %zu/%zu textures submitted", submittedTextureCount, textureSources.size());

        stagingBuffers.clear();
        stagingMemories.clear();
        batchTextureCount = 0;
        batchBytes = 0;
    };
    beginBatch();

    for (size_t i = 0; i < textureSources.size(); ++i) {
        const auto decodeStart = std::chrono::high_resolution_clock::now();
        const auto &source = textureSources[i];
        const TextureSemanticRole role =
            (i < textureRoles.size() && textureRoles[i] == ImportTextureRole::Color) ? TextureSemanticRole::Color : TextureSemanticRole::Linear;
        const bool useSrgbForColorRole = (role == TextureSemanticRole::Color) &&
                                         (textureColorSpaceModel == TextureColorSpaceModel::HardwareSrgb);

        VulkanUtils::VmaImage img{};
        VulkanUtils::TextureUploadPayload payload{};
        bool success = false;
        std::string decodePathTag = "rgba-fallback";

        if (source.kind == GltfImporter::TextureImportSource::Kind::Uri) {
            LOGI("Loading texture from URI: %s", source.uriPath.string().c_str());
			const std::string extension = source.uriPath.extension().string();
			if (extension == ".ktx2" || extension == ".KTX2")
			{
				const auto bytes = readBinaryFile(source.uriPath);
				if (!bytes.empty())
				{
					success = prepareKTXFromMemory(bytes.data(), bytes.size(), role, payload, decodePathTag, stats);
				}
				if (!success)
				{
					LOGE("Failed to load KTX2 texture from file: %s", source.uriPath.string().c_str());
				}
			}
			else
			{
				int w, h, c;
				if (unsigned char *px = stbi_load(source.uriPath.string().c_str(), &w, &h, &c, STBI_rgb_alpha))
				{
					buildSingleMipRgbaPayload(px, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
					                          useSrgbForColorRole ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm, payload);
					success = true;
					decodePathTag = useSrgbForColorRole ? "rgba-fallback-srgb" : "rgba-fallback-unorm";
					stbi_image_free(px);
				}
				else
				{
					LOGE("Failed to load texture from file: %s", source.uriPath.string().c_str());
				}
			}
        } else if (source.kind == GltfImporter::TextureImportSource::Kind::Bytes) {
            LOGI("Loading embedded texture (size: %zu)", source.bytesLength);
            const auto *data = source.bytesData;
            const size_t len = source.bytesLength;

            if (!prepareKTXFromMemory(data, len, role, payload, decodePathTag, stats)) {
                int w, h, c;
                if (unsigned char *px = stbi_load_from_memory(data, static_cast<int>(len), &w, &h, &c, STBI_rgb_alpha)) {
                    buildSingleMipRgbaPayload(px, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                                              useSrgbForColorRole ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm, payload);
                    success = true;
                    decodePathTag = useSrgbForColorRole ? "rgba-fallback-srgb" : "rgba-fallback-unorm";
                    stbi_image_free(px);
                } else {
                    LOGE("Failed to decode embedded texture (Index: %zu)", i);
                }
            } else {
                success = true;
            }
        } else {
            LOGW("Unsupported texture source for image index %zu", i);
        }

        if (!success) {
            LOGW("Texture invalid, using white placeholder.");
            static const unsigned char white[] = {255, 255, 255, 255};
            buildSingleMipRgbaPayload(white, 1, 1,
                                      useSrgbForColorRole ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm, payload);
            decodePathTag = useSrgbForColorRole ? "rgba-fallback-srgb" : "rgba-fallback-unorm";
        }

        if (decodePathTag.find("basisu->bc7") != std::string::npos) {
            ++stats.basisuBc7Count;
        } else if (decodePathTag.find("basisu->bc3") != std::string::npos) {
            ++stats.basisuBc3Count;
        } else if (decodePathTag.find("basisu->bc1") != std::string::npos) {
            ++stats.basisuBc1Count;
        } else if (decodePathTag == "native-ktx-vkformat") {
            ++stats.nativeKtxCount;
        } else {
            ++stats.rgbaFallbackCount;
        }
        if (role == TextureSemanticRole::Color && isSrgbFormat(payload.format)) {
            ++stats.srgbColorCount;
        }
        if (role == TextureSemanticRole::Linear && !isSrgbFormat(payload.format)) {
            ++stats.unormLinearCount;
        }
        LOGI("Texture path[%zu]: %s", i, decodePathTag.c_str());

        const auto decodeEnd = std::chrono::high_resolution_clock::now();
        stats.decodeMs += std::chrono::duration<double, std::milli>(decodeEnd - decodeStart).count();

        const auto uploadRecordStart = std::chrono::high_resolution_clock::now();
        VulkanUtils::createTextureImageFromPayloadBatched(device, physicalDevice, uploadCommandBuffer,
                                                          stagingBuffers, stagingMemories, payload, img);
        const auto uploadRecordEnd = std::chrono::high_resolution_clock::now();
        stats.uploadMs += std::chrono::duration<double, std::milli>(uploadRecordEnd - uploadRecordStart).count();
        ++batchTextureCount;
        batchBytes += payload.data.size();

        vk::ImageViewCreateInfo viewInfo{};
        viewInfo.image = *img;
        viewInfo.viewType = vk::ImageViewType::e2D;
        viewInfo.format = payload.format;
        viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        viewInfo.subresourceRange.levelCount = payload.mipLevels;
        viewInfo.subresourceRange.layerCount = 1;
        modelRes->textureImages.push_back(std::move(img));
        modelRes->textureImageViews.emplace_back(device, viewInfo);

        vk::SamplerCreateInfo samplerInfo{};
        samplerInfo.magFilter = vk::Filter::eLinear;
        samplerInfo.minFilter = vk::Filter::eLinear;
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
        samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
        samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
        samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
        samplerInfo.anisotropyEnable = vk::True;
        samplerInfo.maxAnisotropy = physicalDevice.getProperties().limits.maxSamplerAnisotropy;
        samplerInfo.maxLod = static_cast<float>(payload.mipLevels);
        modelRes->textureSamplers.emplace_back(device, samplerInfo);

        if (((i + 1) % 8) == 0 || (i + 1) == textureSources.size()) {
            LOGI("Texture decode progress: %zu/%zu", i + 1, textureSources.size());
        }
        if (batchTextureCount >= maxBatchTextures || batchBytes >= maxBatchBytes) {
            flushBatch();
            if ((i + 1) < textureSources.size()) {
                beginBatch();
            }
        }
    }

    flushBatch();
    LOGI("Texture decode path summary: bc7=%u bc3=%u bc1=%u nativeKtx=%u rgbaFallback=%u",
         stats.basisuBc7Count, stats.basisuBc3Count, stats.basisuBc1Count, stats.nativeKtxCount,
         stats.rgbaFallbackCount);
    LOGI("Texture color-space summary: srgbColor=%u unormLinear=%u mixedUsage=%u forcedRemap=%u model=%s",
         stats.srgbColorCount, stats.unormLinearCount, stats.mixedUsageCount, stats.forcedRemapCount,
         textureColorSpaceModel == TextureColorSpaceModel::HardwareSrgb ? "HardwareSrgb" : "LegacyManual");
}

SceneNode::Ptr ResourceManager::loadGltfModel(const std::string &path, vk::DescriptorSetLayout layout) {
    const auto importStart = std::chrono::high_resolution_clock::now();
    ModelImportReport report{};
    report.modelPath = path;

    auto it = loadedModels.find(path);
    if (it != loadedModels.end()) {
        if (const auto *cachedModel = getModelResource(it->second); cachedModel && cachedModel->hasRuntimeSkinning) {
            // Skinned models keep per-model mutable GPU output buffers. Reusing them via cache would
            // couple multiple scene instances to the same animated pose.
            loadedModels.erase(it);
        } else {
            LOGI("Loading GLTF from cache: %s", path.c_str());
            report.supportedFeatures.push_back("cached_model_instance");
            if (const auto *cachedModel = getModelResource(it->second)) {
                report.hasAnimations = cachedModel->hasAnimations;
                report.hasSkins = cachedModel->hasSkins;
                if (!cachedModel->animationClipNames.empty()) {
                    report.supportedFeatures.push_back("animation_clips");
                }
                if (!cachedModel->animationClips.empty()) {
                    report.supportedFeatures.push_back("runtime_animation_playback");
                }
                if (cachedModel->hasRuntimeSkinning) {
                    report.supportedFeatures.push_back("gpu_skinning_raster");
                }
            }
            lastImportReport = std::move(report);
            return models[it->second]->prototype->clone();
        }
    }

    LOGI("Loading GLTF: %s", path.c_str());

    const auto parseStart = std::chrono::high_resolution_clock::now();
    const GltfImporter::ParsedAsset parsedAsset = gltfImporter->parseAsset(path);
    const auto parseEnd = std::chrono::high_resolution_clock::now();
    report.parseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();
    const auto &gltf = parsedAsset.asset;
    const std::filesystem::path &modelDir = parsedAsset.modelDirectory;
    report.hasAnimations = !gltf.animations.empty();
    report.hasSkins = !gltf.skins.empty();
    report.supportedFeatures.push_back("meshes");
    report.supportedFeatures.push_back("materials");
    if (report.hasAnimations) {
        report.supportedFeatures.push_back("animations");
    }
    if (report.hasSkins) {
        report.supportedFeatures.push_back("skins");
    }
    const bool hasSkinningAttributes = parsedAsset.hasSkinningAttributes;

    // Create a new ModelResource
    auto modelRes = std::make_unique<ModelResource>();
    modelRes->name = std::filesystem::path(path).filename().string();
    modelRes->path = path;
    modelRes->hasAnimations = report.hasAnimations;
    modelRes->hasSkins = report.hasSkins;
    modelRes->dynamicGeometry = report.hasSkins;
    gltfImporter->populateAnimationClips(gltf, *modelRes, report);
    if (!modelRes->animationClipNames.empty()) {
        report.supportedFeatures.push_back("animation_clips");
    }

    int totalTexturesLoaded = 0;
    for (const auto &m: models) {
        totalTexturesLoaded += m->textureImageViews.size();
    }
    modelRes->globalTextureOffset = totalTexturesLoaded;

    // 1. Textures
    TextureLoadStats textureStats{};
    loadTextures(gltf, modelDir, modelRes.get(), textureStats);
    report.textureDecodeMs = textureStats.decodeMs;
    report.textureUploadMs = textureStats.uploadMs;
    report.supportedFeatures.push_back("texture_decode_path_bc7:" + std::to_string(textureStats.basisuBc7Count));
    report.supportedFeatures.push_back("texture_decode_path_bc3:" + std::to_string(textureStats.basisuBc3Count));
    report.supportedFeatures.push_back("texture_decode_path_bc1:" + std::to_string(textureStats.basisuBc1Count));
    report.supportedFeatures.push_back("texture_decode_path_native_ktx:" + std::to_string(textureStats.nativeKtxCount));
    report.supportedFeatures.push_back("texture_decode_path_rgba_fallback:" + std::to_string(textureStats.rgbaFallbackCount));
    report.supportedFeatures.push_back("texture_srgb_color_count:" + std::to_string(textureStats.srgbColorCount));
    report.supportedFeatures.push_back("texture_unorm_linear_count:" + std::to_string(textureStats.unormLinearCount));
    report.supportedFeatures.push_back("texture_mixed_usage_count:" + std::to_string(textureStats.mixedUsageCount));
    report.supportedFeatures.push_back("texture_forced_remap_count:" + std::to_string(textureStats.forcedRemapCount));
    report.supportedFeatures.push_back(
        std::string("texture_color_space_model:") +
        (textureColorSpaceModel == TextureColorSpaceModel::HardwareSrgb ? "HardwareSrgb" : "LegacyManual"));

    // 2. Materials
    gltfImporter->populateMaterials(gltf, *modelRes);

    // 3. Meshes & Scene Graph
    const auto meshStart = std::chrono::high_resolution_clock::now();
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<ModelResource::SkinningInfluence> skinningInfluences;
    std::vector<int> nodeSkinIndices(gltf.nodes.size(), -1);
    SceneNode::Ptr rootNode = gltfImporter->buildSceneNodes(gltf, *modelRes, vertices, indices, skinningInfluences, nodeSkinIndices);
    const auto meshEnd = std::chrono::high_resolution_clock::now();
    report.meshExtractionMs = std::chrono::duration<double, std::milli>(meshEnd - meshStart).count();
    if (report.hasAnimations && !modelRes->animationClips.empty()) {
        report.supportedFeatures.push_back("runtime_animation_playback");
    } else if (report.hasAnimations && modelRes->animationClips.empty()) {
        report.warnings.push_back("Animation clips were found, but no runtime-supported TRS channels were imported.");
    }
    if (hasSkinningAttributes && !report.hasSkins) {
        report.warnings.push_back("JOINTS_0/WEIGHTS_0 attributes were found without a skin block.");
    }

    // 4. Build flattened Material Buffer specifically sized per-primitive
    std::vector<MaterialData> perPrimitiveMaterials = gltfImporter->buildPerPrimitiveMaterials(*modelRes);

    if (!perPrimitiveMaterials.empty()) {
        const auto bufferUploadStart = std::chrono::high_resolution_clock::now();
        auto bufferUploadCommandBuffer = VulkanUtils::beginSingleTimeCommands(device, commandPool);
        std::vector<vk::raii::Buffer> bufferStagingBuffers;
        std::vector<vk::raii::DeviceMemory> bufferStagingMemories;
        GpuResourceRegistry::UploadBatchContext uploadBatchContext{
            .commandBuffer = &bufferUploadCommandBuffer,
            .stagingBuffers = &bufferStagingBuffers,
            .stagingMemories = &bufferStagingMemories};

        gpuResourceRegistry->uploadMaterialBuffer(*modelRes, perPrimitiveMaterials, &uploadBatchContext);

        // 5. Upload Geometry
        gpuResourceRegistry->uploadModelBuffers(*modelRes, vertices, indices, &uploadBatchContext);
        gpuResourceRegistry->createSkinningResources(gltf, *modelRes, vertices, skinningInfluences, nodeSkinIndices, &uploadBatchContext);

        VulkanUtils::endSingleTimeCommands(device, queue, commandPool, bufferUploadCommandBuffer);
        const auto bufferUploadEnd = std::chrono::high_resolution_clock::now();
        report.bufferUploadMs = std::chrono::duration<double, std::milli>(bufferUploadEnd - bufferUploadStart).count();
    } else {
        const auto bufferUploadStart = std::chrono::high_resolution_clock::now();
        auto bufferUploadCommandBuffer = VulkanUtils::beginSingleTimeCommands(device, commandPool);
        std::vector<vk::raii::Buffer> bufferStagingBuffers;
        std::vector<vk::raii::DeviceMemory> bufferStagingMemories;
        GpuResourceRegistry::UploadBatchContext uploadBatchContext{
            .commandBuffer = &bufferUploadCommandBuffer,
            .stagingBuffers = &bufferStagingBuffers,
            .stagingMemories = &bufferStagingMemories};

        gpuResourceRegistry->uploadModelBuffers(*modelRes, vertices, indices, &uploadBatchContext);
        gpuResourceRegistry->createSkinningResources(gltf, *modelRes, vertices, skinningInfluences, nodeSkinIndices, &uploadBatchContext);

        VulkanUtils::endSingleTimeCommands(device, queue, commandPool, bufferUploadCommandBuffer);
        const auto bufferUploadEnd = std::chrono::high_resolution_clock::now();
        report.bufferUploadMs = std::chrono::duration<double, std::milli>(bufferUploadEnd - bufferUploadStart).count();
    }

    if (report.hasSkins && modelRes->hasRuntimeSkinning) {
        report.supportedFeatures.push_back("gpu_skinning_raster");
    } else if (report.hasSkins) {
        report.warnings.push_back("Skinning data detected, but GPU skinning setup is incomplete. Mesh will render in bind pose.");
    }

    // 6. Build BLAS (requires vertex/index buffers to be on the GPU)
    const auto blasStart = std::chrono::high_resolution_clock::now();
    gpuResourceRegistry->buildBLAS(*modelRes, vertices, indices);
    const auto blasEnd = std::chrono::high_resolution_clock::now();
    report.blasBuildMs = std::chrono::duration<double, std::milli>(blasEnd - blasStart).count();

    // Store model resource
    models.push_back(std::move(modelRes));
    int modelId = models.size() - 1;
    ModelResource *res = models.back().get();

    // 5. Descriptor Set
    gpuResourceRegistry->createModelDescriptorSet(*res, layout);

    // Fix up SceneNodes to point to this modelID
    std::function<void(SceneNode::Ptr)> fixNodes = [&](const SceneNode::Ptr &node) {
        node->modelId = modelId;
        node->assetRef.path = path;
        node->assetRef.variant = "default";
        if (res->hasAnimations) {
            node->animation.enabled = true;
            if (!res->animationClipNames.empty()) {
                node->animation.clipId = res->animationClipNames.front();
            }
        }
        for (auto &child: node->getChildren())
            fixNodes(child);
    };
    fixNodes(rootNode);

    LOGI("Loaded Model. Vertices: %zu, Indices: %zu", vertices.size(), indices.size());
    const auto importEnd = std::chrono::high_resolution_clock::now();
    report.totalMs = std::chrono::duration<double, std::milli>(importEnd - importStart).count();
    LOGI("Import timings (ms) | parse=%.2f decode=%.2f texUpload=%.2f mesh=%.2f bufUpload=%.2f blas=%.2f total=%.2f",
         report.parseMs.value_or(0.0), report.textureDecodeMs.value_or(0.0), report.textureUploadMs.value_or(0.0),
         report.meshExtractionMs.value_or(0.0), report.bufferUploadMs.value_or(0.0), report.blasBuildMs.value_or(0.0), report.totalMs.value_or(0.0));

    models.back()->prototype = rootNode;
    if (!res->hasRuntimeSkinning) {
        loadedModels[path] = modelId;
    }
    lastImportReport = std::move(report);

    return rootNode->clone();
}

ModelResource *ResourceManager::getModelResource(int id) const {
    if (id >= 0 && static_cast<size_t>(id) < models.size())
        return models[id].get();
    return nullptr;
}

const ModelImportReport *ResourceManager::getLastImportReport() const {
    return lastImportReport ? &*lastImportReport : nullptr;
}

const ModelResource::AnimationClip *ResourceManager::findAnimationClip(int modelId, const std::string &clipId) const {
    const auto *resource = getModelResource(modelId);
    if (!resource || resource->animationClips.empty()) {
        return nullptr;
    }

    if (!clipId.empty()) {
        for (const auto &clip: resource->animationClips) {
            if (clip.id == clipId) {
                return &clip;
            }
        }
    }
    return &resource->animationClips.front();
}

float ResourceManager::getAnimationClipDurationSeconds(int modelId, const std::string &clipId) const {
    const auto *clip = findAnimationClip(modelId, clipId);
    if (!clip) {
        return 0.0f;
    }
    return clip->durationSeconds;
}

bool ResourceManager::hasRuntimeSkinnedModels() const {
    for (const auto &model : models) {
        if (model && model->hasRuntimeSkinning) {
            return true;
        }
    }
    return false;
}

void ResourceManager::bindResources(const vk::raii::CommandBuffer &cmd, int modelId, bool useSkinnedVertices) const {
    if (modelId >= 0 && static_cast<size_t>(modelId) < models.size()) {
        auto &res = models[modelId];
        const bool bindSkinned = useSkinnedVertices && res->hasRuntimeSkinning && *res->skinnedVertexBuffer;
        if (const vk::Buffer vertexBufferHandle = bindSkinned ? *res->skinnedVertexBuffer : *res->vertexBuffer) {
            vk::DeviceSize offsets[] = {0};
            cmd.bindVertexBuffers(0, vertexBufferHandle, offsets);
            cmd.bindIndexBuffer(*res->indexBuffer, 0, vk::IndexType::eUint32);
        }
    }
}

void ResourceManager::recordSkinnedBLASRefit(const vk::raii::CommandBuffer &cmd) const {
    const vk::DeviceSize scratchAlignment =
        VulkanUtils::getAccelerationStructureScratchAlignment(physicalDevice);

    for (auto &model : models) {
        if (!model || !model->hasRuntimeSkinning) {
            continue;
        }
        if (!*model->skinnedVertexBuffer || !*model->indexBuffer) {
            continue;
        }
        if (model->blasElements.empty() || model->blasElements.size() != model->meshes.size()) {
            continue;
        }
        if (model->blasScratchBuffers.size() != model->meshes.size()) {
            continue;
        }

        const vk::DeviceAddress vertexAddress = VulkanUtils::getBufferDeviceAddress(device, model->skinnedVertexBuffer);
        const vk::DeviceAddress indexAddress = VulkanUtils::getBufferDeviceAddress(device, model->indexBuffer);

        for (size_t meshIndex = 0; meshIndex < model->meshes.size(); ++meshIndex) {
            const auto &mesh = model->meshes[meshIndex];
            if (mesh.primitives.empty()) {
                continue;
            }

            std::vector<vk::AccelerationStructureGeometryKHR> geometries;
            std::vector<vk::AccelerationStructureBuildRangeInfoKHR> buildRanges;
            geometries.reserve(mesh.primitives.size());
            buildRanges.reserve(mesh.primitives.size());

            for (size_t primIdx = 0; primIdx < mesh.primitives.size(); ++primIdx) {
                const auto &prim = mesh.primitives[primIdx];
                if (prim.indexCount == 0) {
                    continue;
                }

                vk::AccelerationStructureGeometryKHR geometry{};
                geometry.geometryType = vk::GeometryTypeKHR::eTriangles;
                auto &triangles = geometry.geometry.triangles;
                triangles.vertexFormat = vk::Format::eR32G32B32Sfloat;
                triangles.vertexData.deviceAddress = vertexAddress;
                triangles.vertexStride = sizeof(Vertex);
                uint32_t nextVertexOffset = (primIdx + 1 < mesh.primitives.size())
                                                ? mesh.primitives[primIdx + 1].vertexOffset
                                                : model->vertexCount;
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
            }

            if (geometries.empty()) {
                continue;
            }

            vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{};
            buildInfo.type = vk::AccelerationStructureTypeKHR::eBottomLevel;
            buildInfo.flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace |
                              vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate;
            buildInfo.mode = vk::BuildAccelerationStructureModeKHR::eUpdate;
            buildInfo.srcAccelerationStructure = *model->blasElements[meshIndex];
            buildInfo.dstAccelerationStructure = *model->blasElements[meshIndex];
            buildInfo.geometryCount = static_cast<uint32_t>(geometries.size());
            buildInfo.pGeometries = geometries.data();
            const vk::DeviceAddress baseScratchAddress =
                VulkanUtils::getBufferDeviceAddress(device, model->blasScratchBuffers[meshIndex]);
            buildInfo.scratchData.deviceAddress =
                VulkanUtils::alignDeviceAddress(baseScratchAddress, scratchAlignment);

            const vk::AccelerationStructureBuildRangeInfoKHR *pBuildRanges = buildRanges.data();
            cmd.buildAccelerationStructuresKHR(buildInfo, pBuildRanges);
        }
    }

    vk::MemoryBarrier2 refitBarrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
        .srcAccessMask = vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
        .dstStageMask = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR | vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
        .dstAccessMask = vk::AccessFlagBits2::eAccelerationStructureReadKHR};
    vk::DependencyInfo dep{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &refitBarrier};
    cmd.pipelineBarrier2(dep);
}

SceneNode::Ptr ResourceManager::createSphereModel(float radius, int slices, int stacks, vk::DescriptorSetLayout layout) {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    for (int i = 0; i <= stacks; ++i) {
        float V = static_cast<float>(i) / static_cast<float>(stacks);
        float phi = V * glm::pi<float>();

        for (int j = 0; j <= slices; ++j) {
            float U = static_cast<float>(j) / static_cast<float>(slices);
            float theta = U * (glm::pi<float>() * 2);

            float x = cos(theta) * sin(phi);
            float y = cos(phi);
            float z = sin(theta) * sin(phi);

            Vertex vert{};
            vert.pos = glm::vec3(x, y, z) * radius;
            vert.normal = glm::vec3(x, y, z); // Normalized if radius=1, else normalize
            if (radius != 1.0f && radius != 0.0f)
                vert.normal = glm::normalize(vert.normal);
            vert.texCoord = glm::vec2(U, V);

            // Calculate Tangent: Derivative of pos w.r.t theta(U)
            // (-sin(theta), 0, cos(theta))
            glm::vec3 tanVec(-sin(theta), 0.0f, cos(theta));
            vert.tangent = glm::vec4(glm::normalize(tanVec), 1.0f);

            vert.color = glm::vec3(1.0f);
            vertices.push_back(vert);
        }
    }

    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            uint32_t a = (slices + 1) * i + j;
            uint32_t b = (slices + 1) * i + (j + 1);
            uint32_t c = (slices + 1) * (i + 1) + (j + 1);
            uint32_t d = (slices + 1) * (i + 1) + j;

            indices.push_back(a);
            indices.push_back(b);
            indices.push_back(d);

            indices.push_back(b);
            indices.push_back(c);
            indices.push_back(d);
        }
    }

    auto modelRes = std::make_unique<ModelResource>();
    modelRes->name = "ProceduralSphere";
    modelRes->path = "";

    finalizeProceduralModel(modelRes.get(), vertices, indices, layout, "SphereMesh");

    models.push_back(std::move(modelRes));
    int modelId = models.size() - 1;

    SceneNode::Ptr node = std::make_shared<SceneNode>("Sphere");
    node->modelId = modelId;
    node->addMeshIndex(0);
    models.back()->prototype = node;

    return node->clone();
}

SceneNode::Ptr ResourceManager::createCubeModel(float size, vk::DescriptorSetLayout layout) {
    return createCubeModel(size, layout, MaterialData{});
}

SceneNode::Ptr ResourceManager::createCubeModel(float size, vk::DescriptorSetLayout layout, const MaterialData &materialOverride) {
    float h = size * 0.5f;
    glm::vec3 positions[] = {
        {-h, -h, h}, {h, -h, h}, {h, h, h}, {-h, h, h}, {h, -h, -h}, {-h, -h, -h}, {-h, h, -h}, {h, h, -h}, {-h, h, h}, {h, h, h}, {h, h, -h}, {-h, h, -h}, {-h, -h, -h}, {h, -h, -h},
        {h, -h, h}, {-h, -h, h}, {h, -h, h}, {h, -h, -h}, {h, h, -h}, {h, h, h}, {-h, -h, -h}, {-h, -h, h}, {-h, h, h}, {-h, h, -h}
    };
    glm::vec3 normals[] = {
        {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, -1, 0}, {0, -1, 0}, {0, -1, 0},
        {0, -1, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0}
    };
    std::vector<uint32_t> indices;
    for (int i = 0; i < 6; i++) {
        uint32_t base = i * 4;
        indices.push_back(base);
        indices.push_back(base + 1);
        indices.push_back(base + 2);
        indices.push_back(base + 2);
        indices.push_back(base + 3);
        indices.push_back(base);
    }

    std::vector<Vertex> vertices;
    for (int i = 0; i < 24; i++) {
        Vertex v{};
        v.pos = positions[i];
        v.normal = normals[i];
        v.color = glm::vec3(1.0f);

        // UVs based on face index
        int face = i / 4;
        int corner = i % 4;

        if (corner == 0)
            v.texCoord = glm::vec2(0, 1);
        else if (corner == 1)
            v.texCoord = glm::vec2(1, 1);
        else if (corner == 2)
            v.texCoord = glm::vec2(1, 0);
        else if (corner == 3)
            v.texCoord = glm::vec2(0, 0);

        // Tangents
        // Face 0 (Z+): Tangent X+ (1,0,0)
        // Face 1 (Z-): Tangent X- (-1,0,0) ?
        // Face 2 (Y+): Tangent X+ (1,0,0)
        // Face 3 (Y-): Tangent X+ (1,0,0)
        // Face 4 (X+): Tangent Z- (0,0,-1) ?
        // Face 5 (X-): Tangent Z+ (0,0,1) ?

        glm::vec3 t(1, 0, 0);
        if (face == 0)
            t = glm::vec3(1, 0, 0);
        else if (face == 1)
            t = glm::vec3(-1, 0, 0); // or 1,0,0 depends on UV dir
        else if (face == 2)
            t = glm::vec3(1, 0, 0);
        else if (face == 3)
            t = glm::vec3(1, 0, 0);
        else if (face == 4)
            t = glm::vec3(0, 0, -1);
        else if (face == 5)
            t = glm::vec3(0, 0, 1);

        v.tangent = glm::vec4(t, 1.0f);

        vertices.push_back(v);
    }

    auto modelRes = std::make_unique<ModelResource>();
    modelRes->name = "ProceduralCube";
    modelRes->path = "";

    finalizeProceduralModel(modelRes.get(), vertices, indices, layout, "CubeMesh", materialOverride);

    models.push_back(std::move(modelRes));
    int modelId = models.size() - 1;

    SceneNode::Ptr node = std::make_shared<SceneNode>("Cube");
    node->modelId = modelId;
    node->addMeshIndex(0);
    models.back()->prototype = node;

    return node->clone();
}

SceneNode::Ptr ResourceManager::createCylinderModel(float radius, float height, int slices, vk::DescriptorSetLayout layout) {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    float halfH = height * 0.5f;

    // Side vertices
    for (int i = 0; i <= slices; ++i) {
        float U = i / static_cast<float>(slices);
        float theta = U * glm::pi<float>() * 2.0f;

        float x = cos(theta) * radius;
        float z = sin(theta) * radius;

        // Tangent for sides: (-sin(theta), 0, cos(theta)) -> converted to vec4
        glm::vec4 tangent = glm::vec4(-sin(theta), 0.0f, cos(theta), 1.0f);

        // Top edge
        vertices.push_back({
            glm::vec3(x, halfH, z),
            glm::vec3(cos(theta), 0.0f, sin(theta)),
            tangent,
            glm::vec2(U, 0.0f),
            glm::vec3(1.0f)
        });

        // Bottom edge
        vertices.push_back({
            glm::vec3(x, -halfH, z),
            glm::vec3(cos(theta), 0.0f, sin(theta)),
            tangent,
            glm::vec2(U, 1.0f),
            glm::vec3(1.0f)
        });
    }

    // Side indices -- FIXED WINDING
    for (int i = 0; i < slices; ++i) {
        uint32_t top1 = i * 2;
        uint32_t bot1 = i * 2 + 1;
        uint32_t top2 = (i + 1) * 2;
        uint32_t bot2 = (i + 1) * 2 + 1;

        indices.push_back(top1);
        indices.push_back(top2);
        indices.push_back(bot1);

        indices.push_back(bot1);
        indices.push_back(top2);
        indices.push_back(bot2);
    }

    // Top Cap
    uint32_t topCenterIdx = vertices.size();
    vertices.push_back({
        glm::vec3(0, halfH, 0),
        glm::vec3(0, 1, 0),
        glm::vec4(1, 0, 0, 1), // Tangent
        glm::vec2(0.5f),
        glm::vec3(1.0f)
    });

    for (int i = 0; i <= slices; ++i) {
        float U = i / static_cast<float>(slices);
        float theta = U * glm::pi<float>() * 2.0f;
        float x = cos(theta) * radius;
        float z = sin(theta) * radius;

        vertices.push_back({
            glm::vec3(x, halfH, z),
            glm::vec3(0, 1, 0),
            glm::vec4(1, 0, 0, 1),
            glm::vec2(x / radius * 0.5f + 0.5f, z / radius * 0.5f + 0.5f),
            glm::vec3(1.0f)
        });
    }

    for (int i = 0; i < slices; ++i) {
        indices.push_back(topCenterIdx);
        indices.push_back(topCenterIdx + 1 + i + 1);
        indices.push_back(topCenterIdx + 1 + i);
    }

    // Bottom Cap
    uint32_t botCenterIdx = vertices.size();
    vertices.push_back({
        glm::vec3(0, -halfH, 0),
        glm::vec3(0, -1, 0),
        glm::vec4(1, 0, 0, 1),
        glm::vec2(0.5f),
        glm::vec3(1.0f)
    });

    for (int i = 0; i <= slices; ++i) {
        float U = static_cast<float>(i) / static_cast<float>(slices);
        float theta = U * glm::pi<float>() * 2.0f;
        float x = cos(theta) * radius;
        float z = sin(theta) * radius;

        vertices.push_back({
            glm::vec3(x, -halfH, z),
            glm::vec3(0, -1, 0),
            glm::vec4(1, 0, 0, 1),
            glm::vec2(x / radius * 0.5f + 0.5f, z / radius * 0.5f + 0.5f),
            glm::vec3(1.0f)
        });
    }

    for (int i = 0; i < slices; ++i) {
        indices.push_back(botCenterIdx);
        indices.push_back(botCenterIdx + 1 + i);
        indices.push_back(botCenterIdx + 1 + i + 1);
    }

    auto modelRes = std::make_unique<ModelResource>();
    modelRes->name = "ProceduralCylinder";
    modelRes->path = "";

    finalizeProceduralModel(modelRes.get(), vertices, indices, layout, "CylinderMesh");

    models.push_back(std::move(modelRes));
    int modelId = models.size() - 1;

    SceneNode::Ptr node = std::make_shared<SceneNode>("Cylinder");
    node->modelId = modelId;
    node->addMeshIndex(0);
    models.back()->prototype = node;

    return node->clone();
}

void ResourceManager::finalizeProceduralModel(ModelResource *modelRes, const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices, vk::DescriptorSetLayout layout,
                                              const std::string &meshName, const std::optional<MaterialData> &materialOverride) const {
    vk::DeviceSize vSize = sizeof(Vertex) * vertices.size();
    vk::DeviceSize iSize = sizeof(uint32_t) * indices.size();

    vk::raii::Buffer vStaging{nullptr}, iStaging{nullptr};
    vk::raii::DeviceMemory vStagingMem{nullptr}, iStagingMem{nullptr};

    VulkanUtils::createBuffer(device, physicalDevice, vSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                              vStaging, vStagingMem);
    VulkanUtils::createBuffer(device, physicalDevice, iSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                              iStaging, iStagingMem);

    void *data = vStagingMem.mapMemory(0, vSize);
    memcpy(data, vertices.data(), vSize);
    vStagingMem.unmapMemory();

    data = iStagingMem.mapMemory(0, iSize);
    memcpy(data, indices.data(), iSize);
    iStagingMem.unmapMemory();

    vk::BufferUsageFlags vFlags = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
                                  vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
    VulkanUtils::createBuffer(device, physicalDevice, vSize, vFlags, vk::MemoryPropertyFlagBits::eDeviceLocal,
                              modelRes->vertexBuffer);

    vk::BufferUsageFlags iFlags = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
                                  vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
    VulkanUtils::createBuffer(device, physicalDevice, iSize, iFlags, vk::MemoryPropertyFlagBits::eDeviceLocal,
                              modelRes->indexBuffer);

    VulkanUtils::copyBuffer(device, commandPool, queue, vStaging, modelRes->vertexBuffer, vSize);
    VulkanUtils::copyBuffer(device, commandPool, queue, iStaging, modelRes->indexBuffer, iSize);

    // Default Material
    PBRMaterial defaultMat;
    if (materialOverride.has_value()) {
        defaultMat.data = *materialOverride;
    }
    modelRes->materials.push_back(defaultMat);

    // Material Buffer
    {
        gpuResourceRegistry->uploadMaterialBuffer(*modelRes, defaultMat.data);
    }

    // Create Descriptor
    gpuResourceRegistry->createModelDescriptorSet(*modelRes, layout);

    // Add Mesh entry
    LoadedMesh mesh;
    mesh.name = meshName;
    MeshPrimitive prim;
    prim.firstIndex = 0;
    prim.indexCount = indices.size();
    prim.vertexOffset = 0;
    prim.materialIndex = 0;
    mesh.primitives.push_back(prim);
    modelRes->meshes.push_back(mesh);

    // Build BLAS
    gpuResourceRegistry->buildBLAS(*modelRes, vertices, indices);
}
