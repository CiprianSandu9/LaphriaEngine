#include "EditorProject.h"

#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

namespace LaphriaEditor
{
namespace
{
using json = nlohmann::json;

void setError(std::string *errorMessage, const std::string &text)
{
	if (errorMessage)
	{
		*errorMessage = text;
	}
}
}        // namespace

bool EditorProject::loadFromFile(const std::string &path, EditorProject &outProject, std::string *errorMessage)
{
	std::ifstream stream(path);
	if (!stream.is_open())
	{
		setError(errorMessage, "Failed to open project file: " + path);
		return false;
	}

	json payload;
	try
	{
		stream >> payload;
	}
	catch (const std::exception &e)
	{
		setError(errorMessage, "Invalid project JSON: " + std::string(e.what()));
		return false;
	}

	if (!payload.is_object())
	{
		setError(errorMessage, "Project file must be a JSON object.");
		return false;
	}

	EditorProject project{};
	project.name = payload.value("name", "Laphria Project");
	project.sceneOutputPath = payload.value("scene_output_path", "scene.json");

	if (payload.contains("asset_roots") && payload["asset_roots"].is_array())
	{
		for (const auto &entry : payload["asset_roots"])
		{
			if (entry.is_string())
			{
				project.assetRoots.push_back(entry.get<std::string>());
			}
		}
	}

	if (payload.contains("import_settings") && payload["import_settings"].is_object())
	{
		const auto &settings = payload["import_settings"];
		project.importSettings.importAnimations = settings.value("import_animations", true);
		project.importSettings.importMaterials = settings.value("import_materials", true);
		project.importSettings.importSkins = settings.value("import_skins", true);
		project.importSettings.strictValidation = settings.value("strict_validation", false);
	}

	if (project.assetRoots.empty())
	{
		// Reasonable default for first run.
		project.assetRoots.emplace_back("Assets");
	}

	outProject = std::move(project);
	return true;
}

bool EditorProject::saveToFile(const std::string &path, const EditorProject &project, std::string *errorMessage)
{
	json payload;
	payload["name"] = project.name;
	payload["asset_roots"] = project.assetRoots;
	payload["scene_output_path"] = project.sceneOutputPath;
	payload["import_settings"] = {
	    {"import_animations", project.importSettings.importAnimations},
	    {"import_materials", project.importSettings.importMaterials},
	    {"import_skins", project.importSettings.importSkins},
	    {"strict_validation", project.importSettings.strictValidation}};

	const std::filesystem::path outputPath(path);
	if (!outputPath.parent_path().empty())
	{
		std::error_code ec;
		std::filesystem::create_directories(outputPath.parent_path(), ec);
		if (ec)
		{
			setError(errorMessage, "Failed to create project directory: " + ec.message());
			return false;
		}
	}

	std::ofstream stream(path);
	if (!stream.is_open())
	{
		setError(errorMessage, "Failed to write project file: " + path);
		return false;
	}

	stream << payload.dump(4) << '\n';
	return true;
}
}        // namespace LaphriaEditor
