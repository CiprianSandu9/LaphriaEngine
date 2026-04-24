#ifndef LAPHRIAENGINE_GLTFIMPORTER_H
#define LAPHRIAENGINE_GLTFIMPORTER_H

#include "ResourceManager.h"

#include <fastgltf/types.hpp>
#include <filesystem>
#include <string>
#include <vector>

class GltfImporter
{
  public:
	struct TextureImportSource
	{
		enum class Kind
		{
			Uri,
			Bytes,
			Unsupported
		};

		Kind                  kind = Kind::Unsupported;
		std::filesystem::path uriPath;
		std::vector<unsigned char> bytes;
	};

	struct ParsedAsset
	{
		fastgltf::Asset      asset;
		std::filesystem::path modelDirectory;
		bool                  hasSkinningAttributes = false;
	};

	[[nodiscard]] ParsedAsset parseAsset(const std::string &path) const;
	[[nodiscard]] std::vector<TextureImportSource> buildTextureImportSources(const fastgltf::Asset &gltf, const std::filesystem::path &modelDirectory) const;
	void populateAnimationClips(const fastgltf::Asset &gltf, ModelResource &modelResource, ModelImportReport &report) const;
	void populateMaterials(const fastgltf::Asset &gltf, ModelResource &modelResource) const;
	[[nodiscard]] std::vector<Laphria::MaterialData> buildPerPrimitiveMaterials(ModelResource &modelResource) const;
	SceneNode::Ptr buildSceneNodes(const fastgltf::Asset &gltf, ModelResource &modelResource, std::vector<Laphria::Vertex> &vertices,
	                               std::vector<uint32_t> &indices, std::vector<ModelResource::SkinningInfluence> &skinningInfluences,
	                               std::vector<int> &nodeSkinIndices) const;

  private:
	SceneNode::Ptr processGltfNode(const fastgltf::Asset &gltf, size_t nodeIndex, ModelResource &modelResource, std::vector<Laphria::Vertex> &vertices,
	                               std::vector<uint32_t> &indices, std::vector<ModelResource::SkinningInfluence> &skinningInfluences,
	                               std::vector<int> &nodeSkinIndices) const;
	[[nodiscard]] static bool detectSkinningAttributes(const fastgltf::Asset &gltf);
};

#endif // LAPHRIAENGINE_GLTFIMPORTER_H
