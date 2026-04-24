#include "EditorValidation.h"

#include <filesystem>
#include <fstream>
#include <unordered_set>

#include <nlohmann/json.hpp>

namespace LaphriaEditor
{
namespace
{
using json = nlohmann::json;

std::filesystem::path resolveProjectRelativePath(const std::string &projectPath, const std::string &candidatePath)
{
    std::filesystem::path candidate(candidatePath);
    if (candidate.is_absolute() || projectPath.empty())
    {
        return candidate;
    }

    const std::filesystem::path projectFilePath(projectPath);
    if (projectFilePath.parent_path().empty())
    {
        return candidate;
    }

    return projectFilePath.parent_path() / candidate;
}

bool loadJsonFile(const std::string &path, json &outPayload, ValidationReport &report)
{
    std::ifstream stream(path);
    if (!stream.is_open())
    {
        report.addError(path, "$", "Failed to open file.");
        return false;
    }

    try
    {
        stream >> outPayload;
    }
    catch (const std::exception &e)
    {
        report.addError(path, "$", std::string("Invalid JSON: ") + e.what());
        return false;
    }

    return true;
}

void validateImportSettingBool(const json &settings,
                               const std::string &file,
                               const std::string &fieldPath,
                               const char *key,
                               ValidationReport &report)
{
    if (!settings.contains(key))
    {
        return;
    }

    if (!settings[key].is_boolean())
    {
        report.addError(file, fieldPath + "." + key, "Expected a boolean.");
    }
}

void validateProjectPayload(const json &payload, const std::string &file, ValidationReport &report)
{
    if (!payload.is_object())
    {
        report.addError(file, "$", "Project file must be a JSON object.");
        return;
    }

    const std::unordered_set<std::string> knownFields = {
        "name",
        "asset_roots",
        "scene_output_path",
        "import_settings"};

    for (const auto &item : payload.items())
    {
        if (!knownFields.contains(item.key()))
        {
            report.addWarning(file, "$." + item.key(), "Unknown project field.");
        }
    }

    if (payload.contains("name"))
    {
        if (!payload["name"].is_string())
        {
            report.addError(file, "$.name", "Expected a string.");
        }
        else if (payload["name"].get<std::string>().empty())
        {
            report.addWarning(file, "$.name", "Project name is empty.");
        }
    }

    if (!payload.contains("asset_roots"))
    {
        report.addWarning(file, "$.asset_roots", "Missing field. Default root 'Assets' will be used.");
    }
    else if (!payload["asset_roots"].is_array())
    {
        report.addError(file, "$.asset_roots", "Expected an array of strings.");
    }
    else
    {
        const auto &assetRoots = payload["asset_roots"];
        if (assetRoots.empty())
        {
            report.addWarning(file, "$.asset_roots", "No asset roots configured.");
        }
        for (size_t i = 0; i < assetRoots.size(); ++i)
        {
            const std::string path = "$.asset_roots[" + std::to_string(i) + "]";
            if (!assetRoots[i].is_string())
            {
                report.addError(file, path, "Expected a string.");
                continue;
            }

            const std::string root = assetRoots[i].get<std::string>();
            if (root.empty())
            {
                report.addWarning(file, path, "Asset root path is empty.");
                continue;
            }

            std::error_code ec;
            const std::filesystem::path resolvedRoot = resolveProjectRelativePath(file, root);
            if (!std::filesystem::exists(resolvedRoot, ec) || ec)
            {
                report.addWarning(file, path, "Asset root does not exist on disk: " + resolvedRoot.string());
            }
        }
    }

    if (!payload.contains("scene_output_path"))
    {
        report.addWarning(file, "$.scene_output_path", "Missing field. Default 'scene.json' will be used.");
    }
    else if (!payload["scene_output_path"].is_string())
    {
        report.addError(file, "$.scene_output_path", "Expected a string.");
    }
    else
    {
        const std::string sceneOutput = payload["scene_output_path"].get<std::string>();
        if (sceneOutput.empty())
        {
            report.addWarning(file, "$.scene_output_path", "Scene output path is empty.");
        }
    }

    if (payload.contains("import_settings"))
    {
        if (!payload["import_settings"].is_object())
        {
            report.addError(file, "$.import_settings", "Expected an object.");
        }
        else
        {
            const auto &settings = payload["import_settings"];
            validateImportSettingBool(settings, file, "$.import_settings", "import_animations", report);
            validateImportSettingBool(settings, file, "$.import_settings", "import_materials", report);
            validateImportSettingBool(settings, file, "$.import_settings", "import_skins", report);
            validateImportSettingBool(settings, file, "$.import_settings", "strict_validation", report);
        }
    }
}

void validateSceneNode(const json &node,
                       const std::string &file,
                       const std::string &fieldPath,
                       ValidationReport &report)
{
    if (!node.is_object())
    {
        report.addError(file, fieldPath, "Scene node must be an object.");
        return;
    }

    if (node.contains("children"))
    {
        if (!node["children"].is_array())
        {
            report.addError(file, fieldPath + ".children", "Expected an array.");
            return;
        }

        const auto &children = node["children"];
        for (size_t i = 0; i < children.size(); ++i)
        {
            validateSceneNode(children[i], file, fieldPath + ".children[" + std::to_string(i) + "]", report);
        }
    }
}

std::string inferScenePathFromProject(const std::string &projectPath, ValidationReport &report)
{
    json projectPayload;
    if (!loadJsonFile(projectPath, projectPayload, report))
    {
        return {};
    }

    if (!projectPayload.is_object())
    {
        return {};
    }

    if (!projectPayload.contains("scene_output_path") || !projectPayload["scene_output_path"].is_string())
    {
        return {};
    }

    const std::string sceneFromProject = projectPayload["scene_output_path"].get<std::string>();
    if (sceneFromProject.empty())
    {
        return {};
    }

    return resolveProjectRelativePath(projectPath, sceneFromProject).string();
}
}        // namespace

void ValidationReport::add(ValidationSeverity severity, const std::string &file, const std::string &fieldPath, const std::string &message)
{
    messages.push_back(ValidationMessage{.severity = severity, .file = file, .fieldPath = fieldPath, .message = message});
}

void ValidationReport::addError(const std::string &file, const std::string &fieldPath, const std::string &message)
{
    add(ValidationSeverity::Error, file, fieldPath, message);
}

void ValidationReport::addWarning(const std::string &file, const std::string &fieldPath, const std::string &message)
{
    add(ValidationSeverity::Warning, file, fieldPath, message);
}

void ValidationReport::append(const ValidationReport &other)
{
    messages.insert(messages.end(), other.messages.begin(), other.messages.end());
}

size_t ValidationReport::errorCount() const
{
    size_t count = 0;
    for (const auto &message : messages)
    {
        if (message.severity == ValidationSeverity::Error)
        {
            ++count;
        }
    }
    return count;
}

size_t ValidationReport::warningCount() const
{
    size_t count = 0;
    for (const auto &message : messages)
    {
        if (message.severity == ValidationSeverity::Warning)
        {
            ++count;
        }
    }
    return count;
}

bool ValidationReport::hasErrors() const
{
    return errorCount() > 0;
}

const char *validationSeverityToString(ValidationSeverity severity)
{
    return (severity == ValidationSeverity::Error) ? "error" : "warning";
}

ValidationReport EditorValidator::validateProjectFile(const std::string &path)
{
    ValidationReport report;

    json payload;
    if (!loadJsonFile(path, payload, report))
    {
        return report;
    }

    validateProjectPayload(payload, path, report);

    if (payload.is_object() && payload.contains("scene_output_path") && payload["scene_output_path"].is_string())
    {
        const std::string sceneOutputPath = payload["scene_output_path"].get<std::string>();
        if (!sceneOutputPath.empty())
        {
            std::error_code ec;
            const std::filesystem::path resolvedScenePath = resolveProjectRelativePath(path, sceneOutputPath);
            if (!std::filesystem::exists(resolvedScenePath, ec) || ec)
            {
                report.addWarning(path, "$.scene_output_path", "Referenced scene file does not exist yet: " + resolvedScenePath.string());
            }
        }
    }

    return report;
}

ValidationReport EditorValidator::validateSceneFile(const std::string &path)
{
    ValidationReport report;

    json payload;
    if (!loadJsonFile(path, payload, report))
    {
        return report;
    }

    if (!payload.is_object())
    {
        report.addError(path, "$", "Scene file must be a JSON object.");
        return report;
    }

    validateSceneNode(payload, path, "$", report);
    return report;
}

ValidationReport EditorValidator::validateProjectAndScene(const std::string &projectPath, const std::string &scenePath)
{
    ValidationReport report;
    report.append(validateProjectFile(projectPath));

    std::string resolvedScenePath = scenePath;
    if (resolvedScenePath.empty())
    {
        resolvedScenePath = inferScenePathFromProject(projectPath, report);
    }

    if (resolvedScenePath.empty())
    {
        report.addWarning(projectPath, "$.scene_output_path", "No scene path available; scene validation skipped.");
        return report;
    }

    if (!std::filesystem::path(resolvedScenePath).is_absolute())
    {
        resolvedScenePath = resolveProjectRelativePath(projectPath, resolvedScenePath).string();
    }

    report.append(validateSceneFile(resolvedScenePath));
    return report;
}
}        // namespace LaphriaEditor
