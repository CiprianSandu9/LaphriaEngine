#ifndef LAPHRIAENGINE_UISYSTEM_H
#define LAPHRIAENGINE_UISYSTEM_H

#include <random>
#include <string>
#include <vector>

#include "../Physics/PhysicsSystem.h"
#include "../SceneManagement/Scene.h"
#include "EditorValidation.h"
#include "EditorProject.h"
#include "Camera.h"
#include "EngineAuxiliary.h"
#include "VulkanDevice.h"

// Owns ImGui lifecycle, all editor draw calls, and UI-driven simulation state.
class UISystem {
public:
    struct VisualsV1Settings
    {
        float sunIntensity = 1.0f;
        float fillIntensity = 1.0f;
        float ambientBoost = 1.0f;
        float exposure = 1.0f;

        void reset()
        {
            sunIntensity = 1.0f;
            fillIntensity = 1.0f;
            ambientBoost = 1.0f;
            exposure = 1.0f;
        }
    };

    enum class PathTracerQualityMode
    {
        Manual = 0,
        AutoBalanced = 1,
        AutoAggressive = 2
    };

    struct PathTracerSettings
    {
        float                 resolutionScale = 1.0f;
        int                   denoiserIterations = 5;
        PathTracerQualityMode qualityMode = PathTracerQualityMode::AutoBalanced;
        bool                  reduceSecondaryEffects = false;
        float                 targetFrameMs = 16.6f;
    };

    struct PathTracerPerfStats
    {
        float tlasBuildMs = 0.0f;
        float rayTraceMs = 0.0f;
        float reprojectionMs = 0.0f;
        float denoiserMs = 0.0f;
        float totalFrameMs = 0.0f;
    };

    // Call after the swapchain has been created (needs colorFormat / depthFormat).
    void init(VulkanDevice &dev, GLFWwindow *window,
              vk::Format colorFormat, vk::Format depthFormat);

    // Record one ImGui frame worth of widgets.
    // Must be called between ImGui::NewFrame() and ImGui::Render() in EngineCore.
    void draw(GLFWwindow *window, Scene &scene, PhysicsSystem &physics,
              ResourceManager &rm, vk::DescriptorSetLayout matLayout, Camera &camera);

    void cleanup();

    // State shared with EngineCore's main loop.
    bool useGPUPhysics = false;
    RenderMode renderMode = RenderMode::Rasterizer;
    bool simulationRunning = false;
    float physicsTime = 0.0f; // updated by EngineCore after each tick
    glm::vec3 lightDirection = glm::vec3(-0.30f, -1.0f, -0.20f);
    VisualsV1Settings visualsV1;
    PathTracerSettings pathTracerSettings;
    PathTracerPerfStats pathTracerPerfStats;
    bool showEditorPanels = true;

private:
    enum class TransformGizmoMode
    {
        None = 0,
        Translate = 1,
        Rotate = 2,
        Scale = 3
    };

    vk::raii::DescriptorPool imguiDescriptorPool{nullptr};

    // Editor state
    SceneNode::Ptr selectedNode{nullptr};
    std::vector<SceneNode::Ptr> nodesPendingDeletion;
    bool showModelLoadDialog = false;
    char modelLoadPath[512] = "assets/paladin.glb";
    bool showSceneSaveDialog = false;
    bool showSceneLoadDialog = false;
    bool showProjectLoadDialog = false;
    bool showProjectSaveDialog = false;
    char scenePath[512] = "scene.json";
    char projectPath[512] = "project.laphria_project.json";
    char newAssetRootPath[512] = "Assets";
    bool hasLoadedProject = false;
    LaphriaEditor::EditorProject project;
    bool assetListDirty = true;
    std::vector<std::string> cachedAssetFiles;
    std::string selectedAssetPath;
    std::vector<std::string> lastImportMessages;
    LaphriaEditor::ValidationReport lastValidationReport;
    bool hasValidationReport = false;
    SceneNode::Ptr nodePendingReparent{nullptr};
    std::mt19937 rng{std::random_device{}()};
    TransformGizmoMode transformGizmoMode = TransformGizmoMode::Translate;
    int activeTransformAxis = -1;
    bool transformGizmoDragging = false;
    glm::vec3 transformDragStartPosition{0.0f};
    glm::vec3 transformDragStartEuler{0.0f};
    glm::vec3 transformDragStartScale{1.0f};
    glm::vec2 transformDragStartMouse{0.0f};

    void drawMainMenuBar(GLFWwindow *window);

    void drawSceneHierarchy(Scene &scene);

    void drawSceneNode(const SceneNode::Ptr &node, Scene &scene);

    void drawInspector(ResourceManager &rm);

    void drawAssetBrowser(Scene &scene, ResourceManager &rm, vk::DescriptorSetLayout matLayout);
    void drawValidationPanel();

    void refreshAssetCache();

    static bool isDescendant(const SceneNode::Ptr &node, const SceneNode::Ptr &candidateParent);

    void drawPhysicsUI(Scene &scene, PhysicsSystem &physics,
                       ResourceManager &rm, vk::DescriptorSetLayout matLayout);

    void drawSelectedNodeTransformGizmo(Camera &camera);
};

#endif        // LAPHRIAENGINE_UISYSTEM_H
