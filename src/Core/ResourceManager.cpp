#include "ResourceManager.h"
#include "VulkanUtils.h"

#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#include <filesystem>
#include <iostream>
#include <ktx.h>

// We only need ktx.h for loading data, vulkan creation is done manually.
#include <glm/gtx/matrix_decompose.hpp>

ResourceManager::ResourceManager(vk::raii::Device &device, vk::raii::PhysicalDevice &physicalDevice, vk::raii::CommandPool &commandPool, vk::raii::Queue &queue,
                                 vk::raii::DescriptorPool &descriptorPool) : device(device),
                                                                             physicalDevice(physicalDevice),
                                                                             commandPool(commandPool),
                                                                             queue(queue),
                                                                             descriptorPool(descriptorPool)
{
}

// Internal helper struct for texture batching

// Texture Helpers

bool ResourceManager::prepareKTXFromMemory(const unsigned char *data, size_t length, vk::raii::Image &outImage, vk::raii::DeviceMemory &outMem, uint32_t &width, uint32_t &height,
                                           vk::Format &format) const
{
	static const unsigned char ktx2Magic[12] = {
	    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};
	if (length < 12 || memcmp(data, ktx2Magic, 12) != 0)
		return false;

	ktxTexture2 *ktx2{nullptr};
	if (ktxTexture2_CreateFromMemory(data, length, KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktx2) != KTX_SUCCESS)
		return false;

	if (ktxTexture2_NeedsTranscoding(ktx2))
	{
		// Simply use RGBA32 for now or BC7 if available. simplified for this file.
		ktxTexture2_TranscodeBasis(ktx2, KTX_TTF_RGBA32, 0);
	}

	auto vkFormat = static_cast<vk::Format>(ktx2->vkFormat);
	if (vkFormat == vk::Format::eUndefined)
		vkFormat = vk::Format::eR8G8B8A8Unorm;

	ktx_size_t offset;
	ktxTexture_GetImageOffset(ktxTexture(ktx2), 0, 0, 0, &offset);
	ktx_uint8_t *textureData = ktxTexture_GetData(ktxTexture(ktx2)) + offset;
	ktx_size_t   textureSize = ktxTexture_GetImageSize(ktxTexture(ktx2), 0);
	width                    = ktx2->baseWidth;
	height                   = ktx2->baseHeight;

	vk::raii::Image        tempImage{nullptr};
	vk::raii::DeviceMemory tempMem{nullptr};

	VulkanUtils::createTextureImageFromData(device, physicalDevice, commandPool, queue,
	                                        textureData, textureSize, width, height, vkFormat,
	                                        outImage, outMem);

	// width/height are already set above
	format = vkFormat;

	ktxTexture_Destroy(ktxTexture(ktx2));
	return true;
}

void ResourceManager::prepareTextureFromPixels(const unsigned char *pixels, int width, int height, vk::raii::Image &outImage, vk::raii::DeviceMemory &outMem, vk::Format &format) const
{
	vk::DeviceSize size = width * height * 4;
	format              = vk::Format::eR8G8B8A8Unorm;

	VulkanUtils::createTextureImageFromData(device, physicalDevice, commandPool, queue,
	                                        pixels, size, width, height, format,
	                                        outImage, outMem);
}

void ResourceManager::loadTextures(const fastgltf::Asset &gltf, const std::filesystem::path &modelDir, ModelResource *modelRes)
{
	modelRes->textureImages.reserve(gltf.images.size());
	modelRes->textureImageMemories.reserve(gltf.images.size());
	modelRes->textureImageViews.reserve(gltf.images.size());
	modelRes->textureSamplers.reserve(gltf.images.size());

	for (size_t i = 0; i < gltf.images.size(); ++i)
	{
		const auto &image = gltf.images[i];

		vk::raii::Image        img{nullptr};
		vk::raii::DeviceMemory mem{nullptr};
		uint32_t               width = 0, height = 0;
		vk::Format             format  = vk::Format::eUndefined;
		bool                   success = false;

		std::visit(fastgltf::visitor{
		               [&](const fastgltf::sources::URI &uri) {
			               std::filesystem::path imgPath = modelDir / uri.uri.fspath();
			               LOGI("Loading texture from URI: %s", imgPath.string().c_str());
			               int w, h, c;
			               if (unsigned char *px = stbi_load(imgPath.string().c_str(), &w, &h, &c, STBI_rgb_alpha))
			               {
				               prepareTextureFromPixels(px, w, h, img, mem, format);
				               success = true;
				               stbi_image_free(px);
			               }
			               else
			               {
				               LOGE("Failed to load texture from file: %s", imgPath.string().c_str());
			               }
		               },
		               [&](const fastgltf::sources::Array &array) {
			               LOGI("Loading embedded texture (size: %zu)", array.bytes.size());
			               if (!prepareKTXFromMemory(reinterpret_cast<const unsigned char *>(array.bytes.data()), array.bytes.size(), img, mem, width, height, format))
			               {
				               // Fallback to STBI
				               int w, h, c;
				               if (unsigned char *px = stbi_load_from_memory(reinterpret_cast<const unsigned char *>(array.bytes.data()), array.bytes.size(), &w, &h, &c, STBI_rgb_alpha))
				               {
					               prepareTextureFromPixels(px, w, h, img, mem, format);
					               success = true;
					               stbi_image_free(px);
				               }
				               else
				               {
					               LOGE("Failed to load embedded texture (Index: %zu)", i);
				               }
			               }
			               else
			               {
				               success = true;
			               }
		               },
		               [&](const fastgltf::sources::BufferView &view) {
			               auto &bufferView = gltf.bufferViews[view.bufferViewIndex];
			               auto &buffer     = gltf.buffers[bufferView.bufferIndex];

			               std::visit(fastgltf::visitor{
			                              [&](const fastgltf::sources::Array &array) {
				                              const unsigned char *data = reinterpret_cast<const unsigned char *>(array.bytes.data()) + bufferView.byteOffset;
				                              size_t               len  = bufferView.byteLength;

				                              if (!prepareKTXFromMemory(data, len, img, mem, width, height, format))
				                              {
					                              int w, h, c;
					                              if (unsigned char *px = stbi_load_from_memory(data, len, &w, &h, &c, STBI_rgb_alpha))
					                              {
						                              prepareTextureFromPixels(px, w, h, img, mem, format);
						                              success = true;
						                              stbi_image_free(px);
					                              }
					                              else
					                              {
						                              LOGE("Failed to load texture from BufferView index %zu", view.bufferViewIndex);
					                              }
				                              }
				                              else
				                              {
					                              success = true;
				                              }
			                              },
			                              [&](auto &) { LOGE("Unsupported buffer type for texture BufferView"); }},
			                          buffer.data);
		               },
		               [&](auto &) {
		               }        // Others ignored for brevity
		           },
		           image.data);

		if (!success)
		{
			LOGW("Texture invalid, using white placeholder.");
			unsigned char white[] = {255, 255, 255, 255};
			prepareTextureFromPixels(white, 1, 1, img, mem, format);
		}

		// Create Views/Samplers
		vk::ImageViewCreateInfo viewInfo{};
		viewInfo.image                       = *img;
		viewInfo.viewType                    = vk::ImageViewType::e2D;
		viewInfo.format                      = format;
		viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.layerCount = 1;
		modelRes->textureImages.push_back(std::move(img));
		modelRes->textureImageMemories.push_back(std::move(mem));
		modelRes->textureImageViews.emplace_back(device, viewInfo);

		vk::SamplerCreateInfo samplerInfo{};
		samplerInfo.magFilter        = vk::Filter::eLinear;
		samplerInfo.minFilter        = vk::Filter::eLinear;
		samplerInfo.mipmapMode       = vk::SamplerMipmapMode::eLinear;
		samplerInfo.addressModeU     = vk::SamplerAddressMode::eRepeat;
		samplerInfo.addressModeV     = vk::SamplerAddressMode::eRepeat;
		samplerInfo.addressModeW     = vk::SamplerAddressMode::eRepeat;
		samplerInfo.anisotropyEnable = vk::True;
		samplerInfo.maxAnisotropy    = physicalDevice.getProperties().limits.maxSamplerAnisotropy;
		modelRes->textureSamplers.emplace_back(device, samplerInfo);
	}
}

void ResourceManager::loadMaterials(const fastgltf::Asset &gltf, ModelResource *modelRes)
{
	std::vector<MaterialData> materialDataList;
	for (const auto &mat : gltf.materials)
	{
		PBRMaterial pbrMat;
		pbrMat.data.baseColorFactor = glm::vec4(mat.pbrData.baseColorFactor[0], mat.pbrData.baseColorFactor[1], mat.pbrData.baseColorFactor[2], mat.pbrData.baseColorFactor[3]);
		pbrMat.data.metallicFactor  = mat.pbrData.metallicFactor;
		pbrMat.data.roughnessFactor = mat.pbrData.roughnessFactor;

		if (mat.pbrData.baseColorTexture.has_value())
			pbrMat.baseColorTextureIndex = gltf.textures[mat.pbrData.baseColorTexture.value().textureIndex].imageIndex.value();
		if (mat.pbrData.metallicRoughnessTexture.has_value())
			pbrMat.metallicRoughnessTextureIndex = gltf.textures[mat.pbrData.metallicRoughnessTexture.value().textureIndex].imageIndex.value();
		if (mat.normalTexture.has_value())
			pbrMat.normalTextureIndex = gltf.textures[mat.normalTexture.value().textureIndex].imageIndex.value();
		if (mat.occlusionTexture.has_value())
			pbrMat.occlusionTextureIndex = gltf.textures[mat.occlusionTexture.value().textureIndex].imageIndex.value();
		if (mat.emissiveTexture.has_value())
			pbrMat.emissiveTextureIndex = gltf.textures[mat.emissiveTexture.value().textureIndex].imageIndex.value();

		if (mat.specular != nullptr)
		{
			pbrMat.data.specularFactor = mat.specular->specularFactor;
			if (mat.specular->specularTexture.has_value())
			{
				pbrMat.specularTextureIndex = gltf.textures[mat.specular->specularTexture.value().textureIndex].imageIndex.value();
			}
		}

		pbrMat.data.baseColorIndex         = pbrMat.baseColorTextureIndex;
		pbrMat.data.metallicRoughnessIndex = pbrMat.metallicRoughnessTextureIndex;
		pbrMat.data.normalIndex            = pbrMat.normalTextureIndex;
		pbrMat.data.occlusionIndex         = pbrMat.occlusionTextureIndex;
		pbrMat.data.emissiveIndex          = pbrMat.emissiveTextureIndex;
		pbrMat.data.specularTextureIndex   = pbrMat.specularTextureIndex;

		modelRes->materials.push_back(pbrMat);
	}
}

SceneNode::Ptr ResourceManager::processGltfNode(const fastgltf::Asset &gltf, const fastgltf::Node &node, ModelResource *modelRes, std::vector<Vertex> &vertices,
                                                std::vector<uint32_t> &indices)
{
	auto newNode = std::make_shared<SceneNode>(std::string(node.name));

	// Transform
	if (auto *trs = std::get_if<fastgltf::TRS>(&node.transform))
	{
		newNode->setPosition(glm::vec3(trs->translation[0], trs->translation[1], trs->translation[2]));
		newNode->setRotation(glm::quat(trs->rotation[3], trs->rotation[0], trs->rotation[1], trs->rotation[2]));
		newNode->setScale(glm::vec3(trs->scale[0], trs->scale[1], trs->scale[2]));
	}
	else if (auto *mat = std::get_if<fastgltf::math::fmat4x4>(&node.transform))
	{
		glm::mat4 m;
		memcpy(&m, mat->data(), sizeof(glm::mat4));
		glm::vec3 scale, translation, skew;
		glm::vec4 perspective;
		glm::quat rotation;
		glm::decompose(m, scale, rotation, translation, skew, perspective);
		newNode->setPosition(translation);
		newNode->setRotation(rotation);
		newNode->setScale(scale);
	}

	// Mesh
	if (node.meshIndex.has_value())
	{
		const auto &mesh = gltf.meshes[node.meshIndex.value()];
		LoadedMesh  loadedMesh;
		loadedMesh.name = mesh.name;

		for (const auto &primitive : mesh.primitives)
		{
			Laphria::MeshPrimitive meshPrim;
			meshPrim.vertexOffset  = vertices.size();
			meshPrim.firstIndex    = indices.size();
			meshPrim.materialIndex = primitive.materialIndex.has_value() ? primitive.materialIndex.value() : -1;

			auto posIt = primitive.findAttribute("POSITION");
			if (posIt != primitive.attributes.end())
			{
				auto  &posAcc = gltf.accessors[posIt->accessorIndex];
				size_t vCount = vertices.size();
				vertices.resize(vCount + posAcc.count);

				fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(gltf, posAcc, [&](fastgltf::math::fvec3 p, size_t i) {
					vertices[vCount + i].pos      = glm::vec3(p.x(), p.y(), p.z());
					vertices[vCount + i].color    = glm::vec3(1.0f);
					vertices[vCount + i].normal   = glm::vec3(0, 1, 0);        // Default
					vertices[vCount + i].texCoord = glm::vec2(0.0f);           // Default; overwritten below if TEXCOORD_0 present
				});
			}

			// Normal
			auto normIt = primitive.findAttribute("NORMAL");
			if (normIt != primitive.attributes.end())
			{
				fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(gltf, gltf.accessors[normIt->accessorIndex], [&](fastgltf::math::fvec3 n, size_t i) {
					vertices[meshPrim.vertexOffset + i].normal = glm::vec3(n.x(), n.y(), n.z());
				});
			}
			// TexCoord
			auto texIt = primitive.findAttribute("TEXCOORD_0");
			if (texIt != primitive.attributes.end())
			{
				fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec2>(gltf, gltf.accessors[texIt->accessorIndex], [&](fastgltf::math::fvec2 uv, size_t i) {
					vertices[meshPrim.vertexOffset + i].texCoord = glm::vec2(uv.x(), uv.y());
				});
			}

			// Indices
			if (primitive.indicesAccessor.has_value())
			{
				auto &idxAcc        = gltf.accessors[primitive.indicesAccessor.value()];
				meshPrim.indexCount = idxAcc.count;
				fastgltf::iterateAccessor<uint32_t>(gltf, idxAcc, [&](uint32_t idx) {
					indices.push_back(idx);        // Relative to buffer!
				});
			}
			else
			{
				// Generate indices
				meshPrim.indexCount = vertices.size() - meshPrim.vertexOffset;
				for (uint32_t i = 0; i < meshPrim.indexCount; ++i)
					indices.push_back(i);
			}

			loadedMesh.primitives.push_back(meshPrim);
		}

		// Register mesh in ModelResource and get index
		modelRes->meshes.push_back(loadedMesh);
		newNode->addMeshIndex(modelRes->meshes.size() - 1);
	}

	for (auto childIdx : node.children)
	{
		newNode->addChild(processGltfNode(gltf, gltf.nodes[childIdx], modelRes, vertices, indices));
	}

	return newNode;
}

SceneNode::Ptr ResourceManager::processSceneNodes(const fastgltf::Asset &gltf, ModelResource *modelRes, std::vector<Vertex> &vertices, std::vector<uint32_t> &indices)
{
	SceneNode::Ptr rootNode = std::make_shared<SceneNode>(modelRes->name);

	if (gltf.scenes.empty())
	{
		if (!gltf.nodes.empty())
			rootNode->addChild(processGltfNode(gltf, gltf.nodes[0], modelRes, vertices, indices));
	}
	else
	{
		const auto &scene = gltf.scenes[gltf.defaultScene.value_or(0)];
		for (auto nodeIdx : scene.nodeIndices)
		{
			rootNode->addChild(processGltfNode(gltf, gltf.nodes[nodeIdx], modelRes, vertices, indices));
		}
	}
	return rootNode;
}

void ResourceManager::uploadModelBuffers(ModelResource *modelRes, const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices)
{
	if (!vertices.empty())
	{
		vk::BufferUsageFlags vFlags = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
		VulkanUtils::createDeviceLocalBufferFromData(device, physicalDevice, commandPool, queue,
		                                             vertices.data(), sizeof(Vertex) * vertices.size(), vFlags,
		                                             modelRes->vertexBuffer, modelRes->vertexBufferMemory);

		vk::BufferUsageFlags iFlags = vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
		VulkanUtils::createDeviceLocalBufferFromData(device, physicalDevice, commandPool, queue,
		                                             indices.data(), sizeof(uint32_t) * indices.size(), iFlags,
		                                             modelRes->indexBuffer, modelRes->indexBufferMemory);
	}
}

void ResourceManager::createModelDescriptorSet(ModelResource *modelRes, vk::DescriptorSetLayout layout)
{
	// Allocate Descriptor Set
	uint32_t                                             variableDescCounts[] = {1000};
	vk::DescriptorSetVariableDescriptorCountAllocateInfo variableDescriptorCountAllocInfo;
	variableDescriptorCountAllocInfo.descriptorSetCount = 1;
	variableDescriptorCountAllocInfo.pDescriptorCounts  = variableDescCounts;

	vk::DescriptorSetAllocateInfo allocInfo{};
	allocInfo.pNext              = &variableDescriptorCountAllocInfo;
	allocInfo.descriptorPool     = *descriptorPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts        = &layout;
	modelRes->descriptorSet      = std::move(vk::raii::DescriptorSets(device, allocInfo).front());

	// Update Descriptor Set
	std::vector<vk::WriteDescriptorSet> writes;

	// Binding 0: Material Buffer
	vk::DescriptorBufferInfo matBufferInfo{};
	if (*modelRes->materialBuffer)
	{
		matBufferInfo.buffer = *modelRes->materialBuffer;
		matBufferInfo.offset = 0;
		matBufferInfo.range  = VK_WHOLE_SIZE;
	}
	vk::WriteDescriptorSet matWrite{};
	matWrite.dstSet          = *modelRes->descriptorSet;
	matWrite.dstBinding      = 0;
	matWrite.dstArrayElement = 0;
	matWrite.descriptorType  = vk::DescriptorType::eStorageBuffer;
	matWrite.descriptorCount = 1;
	matWrite.pBufferInfo     = &matBufferInfo;        // Pointer must remain valid until update
	if (*modelRes->materialBuffer)
		writes.push_back(matWrite);

	// Binding 1: Textures
	std::vector<vk::DescriptorImageInfo> imageInfos;
	if (!modelRes->textureImageViews.empty())
	{
		imageInfos.reserve(modelRes->textureImageViews.size());
		for (size_t i = 0; i < modelRes->textureImageViews.size(); ++i)
		{
			imageInfos.push_back({*modelRes->textureSamplers[i],
			                      *modelRes->textureImageViews[i],
			                      vk::ImageLayout::eShaderReadOnlyOptimal});
		}
		vk::WriteDescriptorSet texWrite{};
		texWrite.dstSet          = *modelRes->descriptorSet;
		texWrite.dstBinding      = 1;
		texWrite.dstArrayElement = 0;
		texWrite.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
		texWrite.descriptorCount = static_cast<uint32_t>(imageInfos.size());
		texWrite.pImageInfo      = imageInfos.data();
		writes.push_back(texWrite);
	}

	device.updateDescriptorSets(writes, nullptr);
}

SceneNode::Ptr ResourceManager::loadGltfModel(const std::string &path, vk::DescriptorSetLayout layout)
{
	auto it = loadedModels.find(path);
	if (it != loadedModels.end())
	{
		LOGI("Loading GLTF from cache: %s", path.c_str());
		return models[it->second]->prototype->clone();
	}

	LOGI("Loading GLTF: %s", path.c_str());

	fastgltf::Parser parser;
	auto             data = fastgltf::GltfDataBuffer::FromPath(path);
	if (data.error() != fastgltf::Error::None)
	{
		throw std::runtime_error("Failed to load glTF file");
	}

	constexpr auto gltfOptions = fastgltf::Options::LoadExternalBuffers | fastgltf::Options::LoadExternalImages | fastgltf::Options::GenerateMeshIndices |
	                             fastgltf::Options::DecomposeNodeMatrices;

	std::filesystem::path modelDir = std::filesystem::path(path).parent_path();
	auto                  asset    = parser.loadGltf(data.get(), modelDir, gltfOptions);
	if (asset.error() != fastgltf::Error::None)
	{
		throw std::runtime_error("Failed to parse glTF");
	}

	const auto &gltf = asset.get();

	// Create a new ModelResource
	auto modelRes  = std::make_unique<ModelResource>();
	modelRes->name = std::filesystem::path(path).filename().string();
	modelRes->path = path;

	int totalTexturesLoaded = 0;
	for (const auto &m : models)
	{
		totalTexturesLoaded += m->textureImageViews.size();
	}
	modelRes->globalTextureOffset = totalTexturesLoaded;

	// 1. Textures
	loadTextures(gltf, modelDir, modelRes.get());

	// 2. Materials
	loadMaterials(gltf, modelRes.get());

	// 3. Meshes & Scene Graph
	std::vector<Vertex>   vertices;
	std::vector<uint32_t> indices;
	SceneNode::Ptr        rootNode = processSceneNodes(gltf, modelRes.get(), vertices, indices);

	uint32_t currentFlatPrimitiveIndex = 0;
	// 4. Build flattened Material Buffer specifically sized per-primitive
	std::vector<MaterialData> perPrimitiveMaterials;
	for (auto &mesh : modelRes->meshes)
	{
		for (auto &prim : mesh.primitives)
		{
			MaterialData primMat{};
			if (prim.materialIndex >= 0 && prim.materialIndex < modelRes->materials.size())
				primMat = modelRes->materials[prim.materialIndex].data;

			prim.flatPrimitiveIndex = currentFlatPrimitiveIndex++;

			primMat.firstIndex          = prim.firstIndex;
			primMat.vertexOffset        = prim.vertexOffset;
			primMat.globalTextureOffset = modelRes->globalTextureOffset;

			perPrimitiveMaterials.push_back(primMat);
		}
	}

	if (!perPrimitiveMaterials.empty())
	{
		vk::DeviceSize bufferSize = sizeof(MaterialData) * perPrimitiveMaterials.size();
		VulkanUtils::createDeviceLocalBufferFromData(device, physicalDevice, commandPool, queue,
		                                             perPrimitiveMaterials.data(), bufferSize, vk::BufferUsageFlagBits::eStorageBuffer,
		                                             modelRes->materialBuffer, modelRes->materialBufferMemory);
	}

	// 5. Upload Geometry
	uploadModelBuffers(modelRes.get(), vertices, indices);

	// 6. Build BLAS (requires vertex/index buffers to be on the GPU)
	buildBLAS(modelRes.get(), vertices, indices);

	// Store model resource
	models.push_back(std::move(modelRes));
	int            modelId = models.size() - 1;
	ModelResource *res     = models.back().get();

	// 5. Descriptor Set
	createModelDescriptorSet(res, layout);

	// Fix up SceneNodes to point to this modelID
	std::function<void(SceneNode::Ptr)> fixNodes = [&](const SceneNode::Ptr &node) {
		node->modelId = modelId;
		for (auto &child : node->getChildren())
			fixNodes(child);
	};
	fixNodes(rootNode);

	LOGI("Loaded Model. Vertices: %zu, Indices: %zu", vertices.size(), indices.size());

	models.back()->prototype = rootNode;
	loadedModels[path]       = modelId;

	return rootNode->clone();
}

ModelResource *ResourceManager::getModelResource(int id) const
{
	if (id >= 0 && static_cast<size_t>(id) < models.size())
		return models[id].get();
	return nullptr;
}

void ResourceManager::bindResources(const vk::raii::CommandBuffer &cmd, int modelId) const
{
	if (modelId >= 0 && static_cast<size_t>(modelId) < models.size())
	{
		auto &res = models[modelId];
		if (*res->vertexBuffer)
		{
			vk::DeviceSize offsets[] = {0};
			cmd.bindVertexBuffers(0, *res->vertexBuffer, offsets);
			cmd.bindIndexBuffer(*res->indexBuffer, 0, vk::IndexType::eUint32);
		}
	}
}

SceneNode::Ptr ResourceManager::createSphereModel(float radius, int slices, int stacks, vk::DescriptorSetLayout layout)
{
	std::vector<Vertex>   vertices;
	std::vector<uint32_t> indices;

	for (int i = 0; i <= stacks; ++i)
	{
		float V   = static_cast<float>(i) / static_cast<float>(stacks);
		float phi = V * glm::pi<float>();

		for (int j = 0; j <= slices; ++j)
		{
			float U     = static_cast<float>(j) / static_cast<float>(slices);
			float theta = U * (glm::pi<float>() * 2);

			float x = cos(theta) * sin(phi);
			float y = cos(phi);
			float z = sin(theta) * sin(phi);

			Vertex vert{};
			vert.pos    = glm::vec3(x, y, z) * radius;
			vert.normal = glm::vec3(x, y, z);        // Normalized if radius=1, else normalize
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

	for (int i = 0; i < stacks; ++i)
	{
		for (int j = 0; j < slices; ++j)
		{
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

	auto modelRes  = std::make_unique<ModelResource>();
	modelRes->name = "ProceduralSphere";
	modelRes->path = "";

	finalizeProceduralModel(modelRes.get(), vertices, indices, layout, "SphereMesh");

	models.push_back(std::move(modelRes));
	int modelId = models.size() - 1;

	SceneNode::Ptr node = std::make_shared<SceneNode>("Sphere");
	node->modelId       = modelId;
	node->addMeshIndex(0);
	models.back()->prototype = node;

	return node->clone();
}

SceneNode::Ptr ResourceManager::createCubeModel(float size, vk::DescriptorSetLayout layout)
{
	float     h           = size * 0.5f;
	glm::vec3 positions[] = {
	    {-h, -h, h}, {h, -h, h}, {h, h, h}, {-h, h, h}, {h, -h, -h}, {-h, -h, -h}, {-h, h, -h}, {h, h, -h}, {-h, h, h}, {h, h, h}, {h, h, -h}, {-h, h, -h}, {-h, -h, -h}, {h, -h, -h}, {h, -h, h}, {-h, -h, h}, {h, -h, h}, {h, -h, -h}, {h, h, -h}, {h, h, h}, {-h, -h, -h}, {-h, -h, h}, {-h, h, h}, {-h, h, -h}};
	glm::vec3 normals[] = {
	    {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, -1, 0}, {0, -1, 0}, {0, -1, 0}, {0, -1, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0}};
	std::vector<uint32_t> indices;
	for (int i = 0; i < 6; i++)
	{
		uint32_t base = i * 4;
		indices.push_back(base);
		indices.push_back(base + 1);
		indices.push_back(base + 2);
		indices.push_back(base + 2);
		indices.push_back(base + 3);
		indices.push_back(base);
	}

	std::vector<Vertex> vertices;
	for (int i = 0; i < 24; i++)
	{
		Vertex v{};
		v.pos    = positions[i];
		v.normal = normals[i];
		v.color  = glm::vec3(1.0f);

		// UVs based on face index
		int face   = i / 4;
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
			t = glm::vec3(-1, 0, 0);        // or 1,0,0 depends on UV dir
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

	auto modelRes  = std::make_unique<ModelResource>();
	modelRes->name = "ProceduralCube";
	modelRes->path = "";

	finalizeProceduralModel(modelRes.get(), vertices, indices, layout, "CubeMesh");

	models.push_back(std::move(modelRes));
	int modelId = models.size() - 1;

	SceneNode::Ptr node = std::make_shared<SceneNode>("Cube");
	node->modelId       = modelId;
	node->addMeshIndex(0);
	models.back()->prototype = node;

	return node->clone();
}

SceneNode::Ptr ResourceManager::createCylinderModel(float radius, float height, int slices, vk::DescriptorSetLayout layout)
{
	std::vector<Vertex>   vertices;
	std::vector<uint32_t> indices;

	float halfH = height * 0.5f;

	// Side vertices
	for (int i = 0; i <= slices; ++i)
	{
		float U     = i / static_cast<float>(slices);
		float theta = U * glm::pi<float>() * 2.0f;

		float x = cos(theta) * radius;
		float z = sin(theta) * radius;

		// Tangent for sides: (-sin(theta), 0, cos(theta)) -> converted to vec4
		glm::vec4 tangent = glm::vec4(-sin(theta), 0.0f, cos(theta), 1.0f);

		// Top edge
		vertices.push_back({glm::vec3(x, halfH, z),
		                    glm::vec3(cos(theta), 0.0f, sin(theta)),
		                    tangent,
		                    glm::vec2(U, 0.0f),
		                    glm::vec3(1.0f)});

		// Bottom edge
		vertices.push_back({glm::vec3(x, -halfH, z),
		                    glm::vec3(cos(theta), 0.0f, sin(theta)),
		                    tangent,
		                    glm::vec2(U, 1.0f),
		                    glm::vec3(1.0f)});
	}

	// Side indices -- FIXED WINDING
	for (int i = 0; i < slices; ++i)
	{
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
	vertices.push_back({glm::vec3(0, halfH, 0),
	                    glm::vec3(0, 1, 0),
	                    glm::vec4(1, 0, 0, 1),        // Tangent
	                    glm::vec2(0.5f),
	                    glm::vec3(1.0f)});

	for (int i = 0; i <= slices; ++i)
	{
		float U     = i / static_cast<float>(slices);
		float theta = U * glm::pi<float>() * 2.0f;
		float x     = cos(theta) * radius;
		float z     = sin(theta) * radius;

		vertices.push_back({glm::vec3(x, halfH, z),
		                    glm::vec3(0, 1, 0),
		                    glm::vec4(1, 0, 0, 1),
		                    glm::vec2(x / radius * 0.5f + 0.5f, z / radius * 0.5f + 0.5f),
		                    glm::vec3(1.0f)});
	}

	for (int i = 0; i < slices; ++i)
	{
		indices.push_back(topCenterIdx);
		indices.push_back(topCenterIdx + 1 + i + 1);
		indices.push_back(topCenterIdx + 1 + i);
	}

	// Bottom Cap
	uint32_t botCenterIdx = vertices.size();
	vertices.push_back({glm::vec3(0, -halfH, 0),
	                    glm::vec3(0, -1, 0),
	                    glm::vec4(1, 0, 0, 1),
	                    glm::vec2(0.5f),
	                    glm::vec3(1.0f)});

	for (int i = 0; i <= slices; ++i)
	{
		float U     = static_cast<float>(i) / static_cast<float>(slices);
		float theta = U * glm::pi<float>() * 2.0f;
		float x     = cos(theta) * radius;
		float z     = sin(theta) * radius;

		vertices.push_back({glm::vec3(x, -halfH, z),
		                    glm::vec3(0, -1, 0),
		                    glm::vec4(1, 0, 0, 1),
		                    glm::vec2(x / radius * 0.5f + 0.5f, z / radius * 0.5f + 0.5f),
		                    glm::vec3(1.0f)});
	}

	for (int i = 0; i < slices; ++i)
	{
		indices.push_back(botCenterIdx);
		indices.push_back(botCenterIdx + 1 + i);
		indices.push_back(botCenterIdx + 1 + i + 1);
	}

	auto modelRes  = std::make_unique<ModelResource>();
	modelRes->name = "ProceduralCylinder";
	modelRes->path = "";

	finalizeProceduralModel(modelRes.get(), vertices, indices, layout, "CylinderMesh");

	models.push_back(std::move(modelRes));
	int modelId = models.size() - 1;

	SceneNode::Ptr node = std::make_shared<SceneNode>("Cylinder");
	node->modelId       = modelId;
	node->addMeshIndex(0);
	models.back()->prototype = node;

	return node->clone();
}

void ResourceManager::finalizeProceduralModel(ModelResource *modelRes, const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices, vk::DescriptorSetLayout layout,
                                              const std::string &meshName)
{
	vk::DeviceSize vSize = sizeof(Vertex) * vertices.size();
	vk::DeviceSize iSize = sizeof(uint32_t) * indices.size();

	vk::raii::Buffer       vStaging{nullptr}, iStaging{nullptr};
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

	vk::BufferUsageFlags vFlags = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
	VulkanUtils::createBuffer(device, physicalDevice, vSize, vFlags, vk::MemoryPropertyFlagBits::eDeviceLocal,
	                          modelRes->vertexBuffer,
	                          modelRes->vertexBufferMemory);

	vk::BufferUsageFlags iFlags = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
	VulkanUtils::createBuffer(device, physicalDevice, iSize, iFlags, vk::MemoryPropertyFlagBits::eDeviceLocal,
	                          modelRes->indexBuffer,
	                          modelRes->indexBufferMemory);

	VulkanUtils::copyBuffer(device, commandPool, queue, vStaging, modelRes->vertexBuffer, vSize);
	VulkanUtils::copyBuffer(device, commandPool, queue, iStaging, modelRes->indexBuffer, iSize);

	// Default Material
	PBRMaterial defaultMat;
	modelRes->materials.push_back(defaultMat);

	// Material Buffer
	{
		VulkanUtils::createDeviceLocalBufferFromData(device, physicalDevice, commandPool, queue,
		                                             &defaultMat.data, sizeof(MaterialData), vk::BufferUsageFlagBits::eStorageBuffer,
		                                             modelRes->materialBuffer, modelRes->materialBufferMemory);
	}

	// Create Descriptor
	createModelDescriptorSet(modelRes, layout);

	// Add Mesh entry
	LoadedMesh mesh;
	mesh.name = meshName;
	MeshPrimitive prim;
	prim.firstIndex    = 0;
	prim.indexCount    = indices.size();
	prim.vertexOffset  = 0;
	prim.materialIndex = 0;
	mesh.primitives.push_back(prim);
	modelRes->meshes.push_back(mesh);

	// Build BLAS
	buildBLAS(modelRes, vertices, indices);
}

void ResourceManager::buildBLAS(ModelResource *modelRes, const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices)
{
	if (modelRes->meshes.empty() || !*modelRes->vertexBuffer)
		return;

	vk::DeviceAddress vertexAddress = VulkanUtils::getBufferDeviceAddress(device, modelRes->vertexBuffer);
	vk::DeviceAddress indexAddress  = VulkanUtils::getBufferDeviceAddress(device, modelRes->indexBuffer);

	for (const auto &mesh : modelRes->meshes)
	{
		std::vector<vk::AccelerationStructureGeometryKHR>       geometries;
		std::vector<vk::AccelerationStructureBuildRangeInfoKHR> buildRanges;
		std::vector<uint32_t>                                   maxPrimitiveCounts;

		for (size_t primIdx = 0; primIdx < mesh.primitives.size(); ++primIdx)
		{
			const auto &prim = mesh.primitives[primIdx];

			vk::AccelerationStructureGeometryKHR geometry{};
			geometry.geometryType = vk::GeometryTypeKHR::eTriangles;

			auto &triangles                    = geometry.geometry.triangles;
			triangles.vertexFormat             = vk::Format::eR32G32B32Sfloat;
			triangles.vertexData.deviceAddress = vertexAddress;
			triangles.vertexStride             = sizeof(Vertex);

			// maxVertex is the highest vertex index reachable within this geometry's vertex range.
			// Primitives are packed contiguously, so the range ends at the next primitive's
			// vertexOffset (or the end of the buffer for the last primitive).
			uint32_t nextVertexOffset = (primIdx + 1 < mesh.primitives.size()) ? mesh.primitives[primIdx + 1].vertexOffset : static_cast<uint32_t>(vertices.size());
			triangles.maxVertex       = nextVertexOffset - prim.vertexOffset - 1;

			triangles.indexType                   = vk::IndexType::eUint32;
			triangles.indexData.deviceAddress     = indexAddress;
			triangles.transformData.deviceAddress = 0;

			geometries.push_back(geometry);

			vk::AccelerationStructureBuildRangeInfoKHR range{};
			range.primitiveCount  = prim.indexCount / 3;
			range.primitiveOffset = prim.firstIndex * sizeof(uint32_t);
			range.firstVertex     = prim.vertexOffset;
			range.transformOffset = 0;

			buildRanges.push_back(range);
			maxPrimitiveCounts.push_back(range.primitiveCount);
		}

		if (geometries.empty())
			continue;

		vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{};
		buildInfo.type          = vk::AccelerationStructureTypeKHR::eBottomLevel;
		buildInfo.flags         = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
		buildInfo.mode          = vk::BuildAccelerationStructureModeKHR::eBuild;
		buildInfo.geometryCount = geometries.size();
		buildInfo.pGeometries   = geometries.data();

		vk::AccelerationStructureBuildSizesInfoKHR sizeInfo = device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo, maxPrimitiveCounts);

		vk::raii::Buffer       blasBuffer{nullptr};
		vk::raii::DeviceMemory blasMemory{nullptr};
		VulkanUtils::createBuffer(device, physicalDevice, sizeInfo.accelerationStructureSize,
		                          vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
		                          vk::MemoryPropertyFlagBits::eDeviceLocal, blasBuffer, blasMemory);

		modelRes->blasBuffers.push_back(std::move(blasBuffer));
		modelRes->blasMemories.push_back(std::move(blasMemory));

		vk::AccelerationStructureCreateInfoKHR createInfo{};
		createInfo.buffer = *modelRes->blasBuffers.back();
		createInfo.size   = sizeInfo.accelerationStructureSize;
		createInfo.type   = vk::AccelerationStructureTypeKHR::eBottomLevel;

		vk::raii::AccelerationStructureKHR blas = vk::raii::AccelerationStructureKHR(device, createInfo);

		vk::raii::Buffer       scratchBuffer{nullptr};
		vk::raii::DeviceMemory scratchMemory{nullptr};
		VulkanUtils::createBuffer(device, physicalDevice, sizeInfo.buildScratchSize,
		                          vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
		                          vk::MemoryPropertyFlagBits::eDeviceLocal, scratchBuffer, scratchMemory);

		buildInfo.dstAccelerationStructure  = *blas;
		buildInfo.scratchData.deviceAddress = VulkanUtils::getBufferDeviceAddress(device, scratchBuffer);

		auto                                              cmd          = VulkanUtils::beginSingleTimeCommands(device, commandPool);
		const vk::AccelerationStructureBuildRangeInfoKHR *pBuildRanges = buildRanges.data();
		cmd.buildAccelerationStructuresKHR(buildInfo, pBuildRanges);
		VulkanUtils::endSingleTimeCommands(device, queue, commandPool, cmd);

		modelRes->blasElements.push_back(std::move(blas));
	}
}
