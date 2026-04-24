#ifndef LAPHRIAENGINE_EDITORPROJECT_H
#define LAPHRIAENGINE_EDITORPROJECT_H

#include <string>
#include <vector>

namespace LaphriaEditor
{
struct ImportSettings
{
	bool importAnimations = true;
	bool importMaterials = true;
	bool importSkins = true;
	bool strictValidation = false;
};

struct EditorProject
{
	std::string              name = "Laphria Project";
	std::vector<std::string> assetRoots;
	std::string              sceneOutputPath = "scene.json";
	ImportSettings           importSettings{};

	static bool loadFromFile(const std::string &path, EditorProject &outProject, std::string *errorMessage = nullptr);
	static bool saveToFile(const std::string &path, const EditorProject &project, std::string *errorMessage = nullptr);
};
}        // namespace LaphriaEditor

#endif        // LAPHRIAENGINE_EDITORPROJECT_H
