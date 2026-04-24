#include "ResourceManager.h"
#include "GltfImporter.h"
#include "GpuResourceRegistry.h"
#include "VulkanUtils.h"

#include <fastgltf/types.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
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

// Internal helper struct for texture batching

// Texture Helpers

bool ResourceManager::prepareKTXFromMemory(const unsigned char *data, size_t length, VulkanUtils::VmaImage &outImage, uint32_t &width, uint32_t &height,
                                           vk::Format &format) const {
    static const unsigned char ktx2Magic[12] = {
        0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A
    };
    if (length < 12 || memcmp(data, ktx2Magic, 12) != 0)
        return false;

    ktxTexture2 *ktx2{nullptr};
    if (ktxTexture2_CreateFromMemory(data, length, KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktx2) != KTX_SUCCESS)
        return false;

    if (ktxTexture2_NeedsTranscoding(ktx2)) {
        // Simply use RGBA32 for now or BC7 if available. simplified for this file.
        ktxTexture2_TranscodeBasis(ktx2, KTX_TTF_RGBA32, 0);
    }

    auto vkFormat = static_cast<vk::Format>(ktx2->vkFormat);
    if (vkFormat == vk::Format::eUndefined)
        vkFormat = vk::Format::eR8G8B8A8Unorm;

    ktx_size_t offset;
    ktxTexture_GetImageOffset(ktxTexture(ktx2), 0, 0, 0, &offset);
    ktx_uint8_t *textureData = ktxTexture_GetData(ktxTexture(ktx2)) + offset;
    ktx_size_t textureSize = ktxTexture_GetImageSize(ktxTexture(ktx2), 0);
    width = ktx2->baseWidth;
    height = ktx2->baseHeight;

    VulkanUtils::createTextureImageFromData(device, physicalDevice, commandPool, queue,
                                            textureData, textureSize, width, height, vkFormat,
                                            outImage);

    // width/height are already set above
    format = vkFormat;

    ktxTexture_Destroy(ktxTexture(ktx2));
    return true;
}

void ResourceManager::prepareTextureFromPixels(const unsigned char *pixels, int width, int height, VulkanUtils::VmaImage &outImage, vk::Format &format) const {
    vk::DeviceSize size = width * height * 4;
    format = vk::Format::eR8G8B8A8Unorm;

    VulkanUtils::createTextureImageFromData(device, physicalDevice, commandPool, queue,
                                            pixels, size, width, height, format,
                                            outImage);
}

void ResourceManager::loadTextures(const fastgltf::Asset &gltf, const std::filesystem::path &modelDir, ModelResource *modelRes) const {
    const auto textureSources = gltfImporter->buildTextureImportSources(gltf, modelDir);

    modelRes->textureImages.reserve(textureSources.size());
    modelRes->textureImageViews.reserve(textureSources.size());
    modelRes->textureSamplers.reserve(textureSources.size());

    for (size_t i = 0; i < textureSources.size(); ++i) {
        const auto &source = textureSources[i];

        VulkanUtils::VmaImage img{};
        uint32_t width = 0, height = 0;
        vk::Format format = vk::Format::eUndefined;
        bool success = false;

        if (source.kind == GltfImporter::TextureImportSource::Kind::Uri) {
            LOGI("Loading texture from URI: %s", source.uriPath.string().c_str());
            int w, h, c;
            if (unsigned char *px = stbi_load(source.uriPath.string().c_str(), &w, &h, &c, STBI_rgb_alpha)) {
                prepareTextureFromPixels(px, w, h, img, format);
                success = true;
                stbi_image_free(px);
            } else {
                LOGE("Failed to load texture from file: %s", source.uriPath.string().c_str());
            }
        } else if (source.kind == GltfImporter::TextureImportSource::Kind::Bytes) {
            LOGI("Loading embedded texture (size: %zu)", source.bytes.size());
            const auto *data = source.bytes.data();
            const size_t len = source.bytes.size();

            if (!prepareKTXFromMemory(data, len, img, width, height, format)) {
                int w, h, c;
                if (unsigned char *px = stbi_load_from_memory(data, static_cast<int>(len), &w, &h, &c, STBI_rgb_alpha)) {
                    prepareTextureFromPixels(px, w, h, img, format);
                    success = true;
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
            unsigned char white[] = {255, 255, 255, 255};
            prepareTextureFromPixels(white, 1, 1, img, format);
        }

        vk::ImageViewCreateInfo viewInfo{};
        viewInfo.image = *img;
        viewInfo.viewType = vk::ImageViewType::e2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        viewInfo.subresourceRange.levelCount = 1;
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
        modelRes->textureSamplers.emplace_back(device, samplerInfo);
    }
}

SceneNode::Ptr ResourceManager::loadGltfModel(const std::string &path, vk::DescriptorSetLayout layout) {
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

    const GltfImporter::ParsedAsset parsedAsset = gltfImporter->parseAsset(path);
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
    loadTextures(gltf, modelDir, modelRes.get());

    // 2. Materials
    gltfImporter->populateMaterials(gltf, *modelRes);

    // 3. Meshes & Scene Graph
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<ModelResource::SkinningInfluence> skinningInfluences;
    std::vector<int> nodeSkinIndices(gltf.nodes.size(), -1);
    SceneNode::Ptr rootNode = gltfImporter->buildSceneNodes(gltf, *modelRes, vertices, indices, skinningInfluences, nodeSkinIndices);
    if (report.hasAnimations && !modelRes->animationClips.empty()) {
        report.supportedFeatures.push_back("runtime_animation_playback");
    } else if (report.hasAnimations && modelRes->animationClips.empty()) {
        report.warnings.push_back("Animation clips were found, but no runtime-supported TRS channels were imported.");
    }
    if (report.hasSkins && modelRes->hasRuntimeSkinning) {
        report.supportedFeatures.push_back("gpu_skinning_raster");
    } else if (report.hasSkins) {
        report.warnings.push_back("Skinning data detected, but GPU skinning setup is incomplete. Mesh will render in bind pose.");
    }
    if (hasSkinningAttributes && !report.hasSkins) {
        report.warnings.push_back("JOINTS_0/WEIGHTS_0 attributes were found without a skin block.");
    }

    // 4. Build flattened Material Buffer specifically sized per-primitive
    std::vector<MaterialData> perPrimitiveMaterials = gltfImporter->buildPerPrimitiveMaterials(*modelRes);

    if (!perPrimitiveMaterials.empty()) {
        gpuResourceRegistry->uploadMaterialBuffer(*modelRes, perPrimitiveMaterials);
    }

    // 5. Upload Geometry
    gpuResourceRegistry->uploadModelBuffers(*modelRes, vertices, indices);
    gpuResourceRegistry->createSkinningResources(gltf, *modelRes, vertices, skinningInfluences, nodeSkinIndices);

    // 6. Build BLAS (requires vertex/index buffers to be on the GPU)
    gpuResourceRegistry->buildBLAS(*modelRes, vertices, indices);

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
            buildInfo.scratchData.deviceAddress = VulkanUtils::getBufferDeviceAddress(device, model->blasScratchBuffers[meshIndex]);

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
