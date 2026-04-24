#include "GltfImporter.h"

#include "ResourceManager.h"

#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>

#include <algorithm>
#include <cmath>
#include <glm/gtx/matrix_decompose.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
ModelResource::AnimationInterpolationMode toInterpolationMode(fastgltf::AnimationInterpolation interpolation)
{
	return interpolation == fastgltf::AnimationInterpolation::Step
	           ? ModelResource::AnimationInterpolationMode::Step
	           : ModelResource::AnimationInterpolationMode::Linear;
}

bool readAccessorTimes(const fastgltf::Asset &gltf, size_t accessorIndex, std::vector<float> &outTimes)
{
	if (accessorIndex >= gltf.accessors.size())
	{
		return false;
	}
	const auto &accessor = gltf.accessors[accessorIndex];
	outTimes.assign(accessor.count, 0.0f);
	fastgltf::iterateAccessorWithIndex<float>(gltf, accessor, [&](float value, size_t idx) {
		if (idx < outTimes.size())
		{
			outTimes[idx] = value;
		}
	});
	return true;
}

bool readAccessorVec3(const fastgltf::Asset &gltf, size_t accessorIndex, std::vector<glm::vec3> &outValues)
{
	if (accessorIndex >= gltf.accessors.size())
	{
		return false;
	}
	const auto &accessor = gltf.accessors[accessorIndex];
	outValues.assign(accessor.count, glm::vec3(0.0f));
	fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(gltf, accessor, [&](fastgltf::math::fvec3 value, size_t idx) {
		if (idx < outValues.size())
		{
			outValues[idx] = glm::vec3(value.x(), value.y(), value.z());
		}
	});
	return true;
}

bool readAccessorQuat(const fastgltf::Asset &gltf, size_t accessorIndex, std::vector<glm::quat> &outValues)
{
	if (accessorIndex >= gltf.accessors.size())
	{
		return false;
	}
	const auto &accessor = gltf.accessors[accessorIndex];
	outValues.assign(accessor.count, glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
	fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec4>(gltf, accessor, [&](fastgltf::math::fvec4 value, size_t idx) {
		if (idx < outValues.size())
		{
			outValues[idx] = glm::normalize(glm::quat(value.w(), value.x(), value.y(), value.z()));
		}
	});
	return true;
}
} // namespace

SceneNode::Ptr GltfImporter::processGltfNode(const fastgltf::Asset &gltf, size_t nodeIndex, ModelResource &modelResource, std::vector<Laphria::Vertex> &vertices,
                                             std::vector<uint32_t> &indices, std::vector<ModelResource::SkinningInfluence> &skinningInfluences,
                                             std::vector<int> &nodeSkinIndices) const
{
	const auto &node = gltf.nodes[nodeIndex];
	auto        newNode = std::make_shared<SceneNode>(std::string(node.name));
	newNode->sourceNodeIndex = static_cast<int>(nodeIndex);
	if (node.skinIndex.has_value())
	{
		nodeSkinIndices[nodeIndex] = static_cast<int>(node.skinIndex.value());
	}

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

	if (node.meshIndex.has_value())
	{
		const auto &mesh = gltf.meshes[node.meshIndex.value()];
		Laphria::LoadedMesh loadedMesh;
		loadedMesh.name = mesh.name;

		for (const auto &primitive : mesh.primitives)
		{
			Laphria::MeshPrimitive meshPrim;
			meshPrim.vertexOffset = vertices.size();
			meshPrim.firstIndex = indices.size();
			meshPrim.materialIndex = primitive.materialIndex.has_value() ? primitive.materialIndex.value() : -1;
			const uint32_t primitiveSkinIndex = (nodeSkinIndices[nodeIndex] >= 0) ? static_cast<uint32_t>(nodeSkinIndices[nodeIndex]) : 0u;

			auto posIt = primitive.findAttribute("POSITION");
			if (posIt != primitive.attributes.end())
			{
				auto  &posAcc = gltf.accessors[posIt->accessorIndex];
				size_t vCount = vertices.size();
				vertices.resize(vCount + posAcc.count);
				skinningInfluences.resize(vCount + posAcc.count, ModelResource::SkinningInfluence{});
				for (size_t i = 0; i < posAcc.count; ++i)
				{
					skinningInfluences[vCount + i].skinIndex = primitiveSkinIndex;
				}

				fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(gltf, posAcc, [&](fastgltf::math::fvec3 p, size_t i) {
					vertices[vCount + i].pos = glm::vec3(p.x(), p.y(), p.z());
					vertices[vCount + i].color = glm::vec3(1.0f);
					vertices[vCount + i].normal = glm::vec3(0, 1, 0);
					vertices[vCount + i].texCoord = glm::vec2(0.0f);
				});
			}

			auto normIt = primitive.findAttribute("NORMAL");
			if (normIt != primitive.attributes.end())
			{
				fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(gltf, gltf.accessors[normIt->accessorIndex], [&](fastgltf::math::fvec3 n, size_t i) {
					vertices[meshPrim.vertexOffset + i].normal = glm::vec3(n.x(), n.y(), n.z());
				});
			}

			auto texIt = primitive.findAttribute("TEXCOORD_0");
			if (texIt != primitive.attributes.end())
			{
				fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec2>(gltf, gltf.accessors[texIt->accessorIndex], [&](fastgltf::math::fvec2 uv, size_t i) {
					vertices[meshPrim.vertexOffset + i].texCoord = glm::vec2(uv.x(), uv.y());
				});
			}

			auto jointsIt = primitive.findAttribute("JOINTS_0");
			if (jointsIt != primitive.attributes.end())
			{
				fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec4>(gltf, gltf.accessors[jointsIt->accessorIndex], [&](fastgltf::math::fvec4 j, size_t i) {
					if (meshPrim.vertexOffset + i >= skinningInfluences.size())
					{
						return;
					}
					auto &influence = skinningInfluences[meshPrim.vertexOffset + i];
					influence.joints = glm::uvec4(
					    static_cast<uint32_t>(std::max(0, static_cast<int>(std::lround(j.x())))),
					    static_cast<uint32_t>(std::max(0, static_cast<int>(std::lround(j.y())))),
					    static_cast<uint32_t>(std::max(0, static_cast<int>(std::lround(j.z())))),
					    static_cast<uint32_t>(std::max(0, static_cast<int>(std::lround(j.w())))));
				});
			}

			auto weightsIt = primitive.findAttribute("WEIGHTS_0");
			if (weightsIt != primitive.attributes.end())
			{
				fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec4>(gltf, gltf.accessors[weightsIt->accessorIndex], [&](fastgltf::math::fvec4 w, size_t i) {
					if (meshPrim.vertexOffset + i >= skinningInfluences.size())
					{
						return;
					}
					auto      &influence = skinningInfluences[meshPrim.vertexOffset + i];
					glm::vec4  weights(w.x(), w.y(), w.z(), w.w());
					const float sum = weights.x + weights.y + weights.z + weights.w;
					influence.weights = sum > 1e-6f ? (weights / sum) : glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
				});
			}

			auto tanIt = primitive.findAttribute("TANGENT");
			if (tanIt != primitive.attributes.end())
			{
				fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec4>(gltf, gltf.accessors[tanIt->accessorIndex], [&](fastgltf::math::fvec4 t, size_t i) {
					vertices[meshPrim.vertexOffset + i].tangent = glm::vec4(t.x(), t.y(), t.z(), t.w());
				});
			}

			if (primitive.indicesAccessor.has_value())
			{
				auto &idxAcc = gltf.accessors[primitive.indicesAccessor.value()];
				meshPrim.indexCount = idxAcc.count;
				fastgltf::iterateAccessor<uint32_t>(gltf, idxAcc, [&](uint32_t idx) { indices.push_back(idx); });
			}
			else
			{
				meshPrim.indexCount = vertices.size() - meshPrim.vertexOffset;
				for (uint32_t i = 0; i < meshPrim.indexCount; ++i)
				{
					indices.push_back(i);
				}
			}

			if (tanIt == primitive.attributes.end())
			{
				const size_t primVertexCount = vertices.size() - meshPrim.vertexOffset;
				for (size_t t = 0; t < meshPrim.indexCount / 3; ++t)
				{
					uint32_t i0 = indices[meshPrim.firstIndex + t * 3 + 0];
					uint32_t i1 = indices[meshPrim.firstIndex + t * 3 + 1];
					uint32_t i2 = indices[meshPrim.firstIndex + t * 3 + 2];

					auto &v0 = vertices[meshPrim.vertexOffset + i0];
					auto &v1 = vertices[meshPrim.vertexOffset + i1];
					auto &v2 = vertices[meshPrim.vertexOffset + i2];

					glm::vec3 edge1 = v1.pos - v0.pos;
					glm::vec3 edge2 = v2.pos - v0.pos;
					glm::vec2 duv1 = v1.texCoord - v0.texCoord;
					glm::vec2 duv2 = v2.texCoord - v0.texCoord;

					const float denom = duv1.x * duv2.y - duv2.x * duv1.y;
					if (std::abs(denom) < 1e-6f)
					{
						continue;
					}
					const float f = 1.0f / denom;

					glm::vec3 T(f * (duv2.y * edge1 - duv1.y * edge2));
					v0.tangent += glm::vec4(T, 0.0f);
					v1.tangent += glm::vec4(T, 0.0f);
					v2.tangent += glm::vec4(T, 0.0f);
				}

				for (size_t i = 0; i < primVertexCount; ++i)
				{
					auto      &v = vertices[meshPrim.vertexOffset + i];
					glm::vec3  N = glm::normalize(v.normal);
					glm::vec3  T = glm::vec3(v.tangent);
					const float len = glm::length(T);
					if (len < 1e-6f)
					{
						glm::vec3 up = std::abs(N.y) < 0.99f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
						T = glm::normalize(glm::cross(up, N));
					}
					else
					{
						T = glm::normalize(T - N * glm::dot(N, T));
					}
					v.tangent = glm::vec4(T, 1.0f);
				}
			}

			loadedMesh.primitives.push_back(meshPrim);
		}

		modelResource.meshes.push_back(loadedMesh);
		newNode->addMeshIndex(modelResource.meshes.size() - 1);
	}

	for (auto childIdx : node.children)
	{
		newNode->addChild(processGltfNode(gltf, childIdx, modelResource, vertices, indices, skinningInfluences, nodeSkinIndices));
	}

	return newNode;
}

SceneNode::Ptr GltfImporter::buildSceneNodes(const fastgltf::Asset &gltf, ModelResource &modelResource, std::vector<Laphria::Vertex> &vertices,
                                             std::vector<uint32_t> &indices, std::vector<ModelResource::SkinningInfluence> &skinningInfluences,
                                             std::vector<int> &nodeSkinIndices) const
{
	SceneNode::Ptr rootNode = std::make_shared<SceneNode>(modelResource.name);
	if (gltf.scenes.empty())
	{
		if (!gltf.nodes.empty())
		{
			rootNode->addChild(processGltfNode(gltf, 0, modelResource, vertices, indices, skinningInfluences, nodeSkinIndices));
		}
	}
	else
	{
		const auto &scene = gltf.scenes[gltf.defaultScene.value_or(0)];
		for (auto nodeIdx : scene.nodeIndices)
		{
			rootNode->addChild(processGltfNode(gltf, nodeIdx, modelResource, vertices, indices, skinningInfluences, nodeSkinIndices));
		}
	}
	return rootNode;
}

GltfImporter::ParsedAsset GltfImporter::parseAsset(const std::string &path) const
{
	fastgltf::Parser parser;
	auto data = fastgltf::GltfDataBuffer::FromPath(path);
	if (data.error() != fastgltf::Error::None)
	{
		throw std::runtime_error("Failed to load glTF file");
	}

	constexpr auto gltfOptions = fastgltf::Options::LoadExternalBuffers | fastgltf::Options::LoadExternalImages |
	                             fastgltf::Options::GenerateMeshIndices | fastgltf::Options::DecomposeNodeMatrices;

	std::filesystem::path modelDir = std::filesystem::path(path).parent_path();
	auto asset = parser.loadGltf(data.get(), modelDir, gltfOptions);
	if (asset.error() != fastgltf::Error::None)
	{
		throw std::runtime_error("Failed to parse glTF");
	}

	const bool hasSkinning = detectSkinningAttributes(asset.get());
	return ParsedAsset{
	    .asset = std::move(asset.get()),
	    .modelDirectory = std::move(modelDir),
	    .hasSkinningAttributes = hasSkinning};
}

std::vector<GltfImporter::TextureImportSource> GltfImporter::buildTextureImportSources(const fastgltf::Asset &gltf, const std::filesystem::path &modelDirectory) const
{
	std::vector<TextureImportSource> sources;
	sources.reserve(gltf.images.size());

	for (const auto &image : gltf.images)
	{
		TextureImportSource source;

		std::visit(fastgltf::visitor{
		               [&](const fastgltf::sources::URI &uri) {
			               source.kind = TextureImportSource::Kind::Uri;
			               source.uriPath = modelDirectory / uri.uri.fspath();
		               },
		               [&](const fastgltf::sources::Array &array) {
			               source.kind = TextureImportSource::Kind::Bytes;
			               const auto *data = reinterpret_cast<const unsigned char *>(array.bytes.data());
			               source.bytes.assign(data, data + array.bytes.size());
		               },
		               [&](const fastgltf::sources::BufferView &view) {
			               if (view.bufferViewIndex >= gltf.bufferViews.size())
			               {
				               source.kind = TextureImportSource::Kind::Unsupported;
				               return;
			               }
			               const auto &bufferView = gltf.bufferViews[view.bufferViewIndex];
			               if (bufferView.bufferIndex >= gltf.buffers.size())
			               {
				               source.kind = TextureImportSource::Kind::Unsupported;
				               return;
			               }
			               const auto &buffer = gltf.buffers[bufferView.bufferIndex];

			               std::visit(fastgltf::visitor{
			                              [&](const fastgltf::sources::Array &array) {
				                              const auto bufferSize = array.bytes.size();
				                              if (bufferView.byteOffset + bufferView.byteLength > bufferSize)
				                              {
					                              source.kind = TextureImportSource::Kind::Unsupported;
					                              return;
				                              }

				                              source.kind = TextureImportSource::Kind::Bytes;
				                              const auto *data = reinterpret_cast<const unsigned char *>(array.bytes.data()) + bufferView.byteOffset;
				                              source.bytes.assign(data, data + bufferView.byteLength);
			                              },
			                              [&](auto &) {
				                              source.kind = TextureImportSource::Kind::Unsupported;
			                              }},
			                          buffer.data);
		               },
		               [&](auto &) {
			               source.kind = TextureImportSource::Kind::Unsupported;
		               }},
		           image.data);

		sources.push_back(std::move(source));
	}

	return sources;
}

void GltfImporter::populateAnimationClips(const fastgltf::Asset &gltf, ModelResource &modelResource, ModelImportReport &report) const
{
	modelResource.animationClips.clear();
	modelResource.animationClipNames.clear();

	for (size_t animationIndex = 0; animationIndex < gltf.animations.size(); ++animationIndex)
	{
		const auto &animation = gltf.animations[animationIndex];
		ModelResource::AnimationClip clip;
		clip.id = !animation.name.empty() ? std::string(animation.name.c_str()) : ("clip_" + std::to_string(animationIndex));

		for (const auto &channel : animation.channels)
		{
			if (!channel.nodeIndex.has_value())
			{
				continue;
			}
			if (channel.samplerIndex >= animation.samplers.size())
			{
				report.warnings.push_back("Animation clip '" + clip.id + "' has an out-of-range sampler reference.");
				continue;
			}

			const auto &sampler = animation.samplers[channel.samplerIndex];
			if (sampler.interpolation == fastgltf::AnimationInterpolation::CubicSpline)
			{
				report.warnings.push_back("Animation clip '" + clip.id + "' uses CubicSpline interpolation, which is not yet supported. Channel skipped.");
				continue;
			}

			std::vector<float> keyTimes;
			if (!readAccessorTimes(gltf, sampler.inputAccessor, keyTimes) || keyTimes.empty())
			{
				report.warnings.push_back("Animation clip '" + clip.id + "' has invalid keyframe time data. Channel skipped.");
				continue;
			}

			clip.durationSeconds = std::max(clip.durationSeconds, keyTimes.back());
			auto &nodeTracks = clip.nodeTracks[static_cast<int>(channel.nodeIndex.value())];
			const auto interpolation = toInterpolationMode(sampler.interpolation);

			switch (channel.path)
			{
				case fastgltf::AnimationPath::Translation:
				{
					std::vector<glm::vec3> values;
					if (!readAccessorVec3(gltf, sampler.outputAccessor, values) || values.size() != keyTimes.size())
					{
						report.warnings.push_back("Animation clip '" + clip.id + "' has mismatched translation keyframe data. Channel skipped.");
						continue;
					}
					nodeTracks.translation = ModelResource::AnimationTrackVec3{
					    .keyTimes = std::move(keyTimes),
					    .keyValues = std::move(values),
					    .interpolation = interpolation};
					break;
				}
				case fastgltf::AnimationPath::Rotation:
				{
					std::vector<glm::quat> values;
					if (!readAccessorQuat(gltf, sampler.outputAccessor, values) || values.size() != keyTimes.size())
					{
						report.warnings.push_back("Animation clip '" + clip.id + "' has mismatched rotation keyframe data. Channel skipped.");
						continue;
					}
					nodeTracks.rotation = ModelResource::AnimationTrackQuat{
					    .keyTimes = std::move(keyTimes),
					    .keyValues = std::move(values),
					    .interpolation = interpolation};
					break;
				}
				case fastgltf::AnimationPath::Scale:
				{
					std::vector<glm::vec3> values;
					if (!readAccessorVec3(gltf, sampler.outputAccessor, values) || values.size() != keyTimes.size())
					{
						report.warnings.push_back("Animation clip '" + clip.id + "' has mismatched scale keyframe data. Channel skipped.");
						continue;
					}
					nodeTracks.scale = ModelResource::AnimationTrackVec3{
					    .keyTimes = std::move(keyTimes),
					    .keyValues = std::move(values),
					    .interpolation = interpolation};
					break;
				}
				case fastgltf::AnimationPath::Weights:
				default:
					report.warnings.push_back("Animation clip '" + clip.id + "' contains weights animation, which is not yet supported. Channel skipped.");
					break;
			}
		}

		modelResource.animationClipNames.push_back(clip.id);
		modelResource.animationClips.push_back(std::move(clip));
	}
}

void GltfImporter::populateMaterials(const fastgltf::Asset &gltf, ModelResource &modelResource) const
{
	modelResource.materials.clear();
	modelResource.materials.reserve(gltf.materials.size());

	for (const auto &mat : gltf.materials)
	{
		Laphria::PBRMaterial pbrMat;
		pbrMat.data.baseColorFactor = glm::vec4(mat.pbrData.baseColorFactor[0], mat.pbrData.baseColorFactor[1], mat.pbrData.baseColorFactor[2], mat.pbrData.baseColorFactor[3]);
		pbrMat.data.metallicFactor = mat.pbrData.metallicFactor;
		pbrMat.data.roughnessFactor = mat.pbrData.roughnessFactor;

		if (mat.pbrData.baseColorTexture.has_value())
		{
			pbrMat.baseColorTextureIndex = gltf.textures[mat.pbrData.baseColorTexture.value().textureIndex].imageIndex.value();
		}
		if (mat.pbrData.metallicRoughnessTexture.has_value())
		{
			pbrMat.metallicRoughnessTextureIndex = gltf.textures[mat.pbrData.metallicRoughnessTexture.value().textureIndex].imageIndex.value();
		}
		if (mat.normalTexture.has_value())
		{
			pbrMat.normalTextureIndex = gltf.textures[mat.normalTexture.value().textureIndex].imageIndex.value();
			pbrMat.data.normalScale = mat.normalTexture.value().scale;
		}
		if (mat.occlusionTexture.has_value())
		{
			pbrMat.occlusionTextureIndex = gltf.textures[mat.occlusionTexture.value().textureIndex].imageIndex.value();
			pbrMat.data.occlusionStrength = mat.occlusionTexture.value().strength;
		}
		if (mat.emissiveTexture.has_value())
		{
			pbrMat.emissiveTextureIndex = gltf.textures[mat.emissiveTexture.value().textureIndex].imageIndex.value();
		}

		if (mat.specular != nullptr)
		{
			pbrMat.data.specularFactor = mat.specular->specularFactor;
			if (mat.specular->specularTexture.has_value())
			{
				pbrMat.specularTextureIndex = gltf.textures[mat.specular->specularTexture.value().textureIndex].imageIndex.value();
			}
		}

		pbrMat.data.baseColorIndex = pbrMat.baseColorTextureIndex;
		pbrMat.data.metallicRoughnessIndex = pbrMat.metallicRoughnessTextureIndex;
		pbrMat.data.normalIndex = pbrMat.normalTextureIndex;
		pbrMat.data.occlusionIndex = pbrMat.occlusionTextureIndex;
		pbrMat.data.emissiveIndex = pbrMat.emissiveTextureIndex;
		pbrMat.data.specularTextureIndex = pbrMat.specularTextureIndex;

		modelResource.materials.push_back(pbrMat);
	}
}

std::vector<Laphria::MaterialData> GltfImporter::buildPerPrimitiveMaterials(ModelResource &modelResource) const
{
	uint32_t currentFlatPrimitiveIndex = 0;
	std::vector<Laphria::MaterialData> perPrimitiveMaterials;

	for (auto &mesh : modelResource.meshes)
	{
		for (auto &prim : mesh.primitives)
		{
			Laphria::MaterialData primMat{};
			if (prim.materialIndex >= 0 && prim.materialIndex < modelResource.materials.size())
			{
				primMat = modelResource.materials[prim.materialIndex].data;
			}

			prim.flatPrimitiveIndex = currentFlatPrimitiveIndex++;
			primMat.firstIndex = prim.firstIndex;
			primMat.vertexOffset = prim.vertexOffset;
			primMat.globalTextureOffset = modelResource.globalTextureOffset;

			perPrimitiveMaterials.push_back(primMat);
		}
	}

	return perPrimitiveMaterials;
}

bool GltfImporter::detectSkinningAttributes(const fastgltf::Asset &gltf)
{
	for (const auto &mesh : gltf.meshes)
	{
		for (const auto &primitive : mesh.primitives)
		{
			if (primitive.findAttribute("JOINTS_0") != primitive.attributes.end() ||
			    primitive.findAttribute("WEIGHTS_0") != primitive.attributes.end())
			{
				return true;
			}
		}
	}
	return false;
}
