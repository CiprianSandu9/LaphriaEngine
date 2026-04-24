#ifndef LAPHRIAENGINE_EDITORVALIDATION_H
#define LAPHRIAENGINE_EDITORVALIDATION_H

#include <string>
#include <vector>

namespace LaphriaEditor
{
enum class ValidationSeverity
{
    Error,
    Warning
};

struct ValidationMessage
{
    ValidationSeverity severity = ValidationSeverity::Error;
    std::string        file;
    std::string        fieldPath;
    std::string        message;
};

struct ValidationReport
{
    std::vector<ValidationMessage> messages;

    void add(ValidationSeverity severity, const std::string &file, const std::string &fieldPath, const std::string &message);
    void addError(const std::string &file, const std::string &fieldPath, const std::string &message);
    void addWarning(const std::string &file, const std::string &fieldPath, const std::string &message);
    void append(const ValidationReport &other);

    [[nodiscard]] size_t errorCount() const;
    [[nodiscard]] size_t warningCount() const;
    [[nodiscard]] bool   hasErrors() const;
};

const char *validationSeverityToString(ValidationSeverity severity);

class EditorValidator
{
  public:
    static ValidationReport validateProjectFile(const std::string &path);
    static ValidationReport validateSceneFile(const std::string &path);
    static ValidationReport validateProjectAndScene(const std::string &projectPath, const std::string &scenePath = "");
};
}        // namespace LaphriaEditor

#endif        // LAPHRIAENGINE_EDITORVALIDATION_H
