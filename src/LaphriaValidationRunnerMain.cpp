#include "Core/EditorValidation.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
struct Options
{
    std::string projectPath = "project.laphria_project.json";
    std::string scenePath = "scene.json";
    bool        validateProject = false;
    bool        validateScene = false;
    bool        sceneExplicitlyProvided = false;
};

void printUsage()
{
    std::cout << "LaphriaValidationRunner usage:\n"
              << "  --project <path>        Project file path (.laphria_project.json)\n"
              << "  --scene <path>          Scene file path (.json)\n"
              << "  --validate-project      Validate only project content\n"
              << "  --validate-scene        Validate only scene content\n"
              << "  --help                  Show this help\n";
}

Options parseArgs(int argc, char **argv)
{
    Options options;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--project" && i + 1 < argc)
        {
            options.projectPath = argv[++i];
        }
        else if (arg == "--scene" && i + 1 < argc)
        {
            options.scenePath = argv[++i];
            options.sceneExplicitlyProvided = true;
        }
        else if (arg == "--validate-project")
        {
            options.validateProject = true;
        }
        else if (arg == "--validate-scene")
        {
            options.validateScene = true;
        }
        else if (arg == "--help")
        {
            printUsage();
            std::exit(EXIT_SUCCESS);
        }
        else
        {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (!options.validateProject && !options.validateScene)
    {
        options.validateProject = true;
        options.validateScene = true;
    }

    return options;
}

void printReport(const LaphriaEditor::ValidationReport &report)
{
    std::cout << "Validation summary: errors=" << report.errorCount() << ", warnings=" << report.warningCount() << '\n';

    for (const auto &message : report.messages)
    {
        std::cout << '[' << LaphriaEditor::validationSeverityToString(message.severity) << "] "
                  << message.file << " | " << message.fieldPath << " | " << message.message << '\n';
    }
}
}        // namespace

int main(int argc, char **argv)
{
    try
    {
        const Options options = parseArgs(argc, argv);
        LaphriaEditor::ValidationReport report;

        if (options.validateProject && options.validateScene)
        {
            report = LaphriaEditor::EditorValidator::validateProjectAndScene(
                options.projectPath,
                options.sceneExplicitlyProvided ? options.scenePath : std::string());
        }
        else if (options.validateProject)
        {
            report = LaphriaEditor::EditorValidator::validateProjectFile(options.projectPath);
        }
        else
        {
            report = LaphriaEditor::EditorValidator::validateSceneFile(options.scenePath);
        }

        printReport(report);
        return report.hasErrors() ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
