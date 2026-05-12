#ifndef LAPHRIAENGINE_UISYSTEM_H
#define LAPHRIAENGINE_UISYSTEM_H

#include <random>
#include <string>
#include <cstdint>
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
    enum class PathTracerBenchmarkCameraPath
    {
        Static = 0,
        SlowPan = 1,
        FastPan = 2,
        Teleport = 3
    };

    enum class PathTracerDebugAov
    {
        FinalColor = 0,
        ReprojectionValidity = 1,
        HistoryAlpha = 2,
        MotionMagnitude = 3,
        TemporalVariance = 4,
        AtrousIteration = 5,
        PathRawFinalColor = 6,
        PathDirectLighting = 7,
        PathIndirectLighting = 8,
        PathSkyContribution = 9,
        PathThroughput = 10,
        PathBounceCount = 11,
        PathShadowVisibility = 12,
        PathEnvironmentNeeContribution = 13,
        PathFirstHitBounceContribution = 14,
        PathSecondaryDirectSunContribution = 15,
        PathBaselineContinuationContribution = 16
    };

    enum class PathTracerQualityMode
    {
        Manual = 0,
        AutoBalanced = 1,
        AutoAggressive = 2
    };

    enum class EnvironmentNeeSamplingMode
    {
        CosineHemisphere = 0,
        SkyBiased = 1
    };

    enum class FirstHitProbeSamplingMode
    {
        CosineHemisphere = 0,
        SunBounceGuided = 1
    };

    struct PathTracerSettings
    {
        float                 resolutionScale = 1.0f;
        int                   denoiserIterations = 1;
        PathTracerQualityMode qualityMode = PathTracerQualityMode::Manual;
        bool                  reduceSecondaryEffects = false;
        bool                  enableEnvironmentNEE = true;
        bool                  blackEnvironment = false;
        bool                  applyFirstHitProbesToFinal = false;
        EnvironmentNeeSamplingMode environmentNeeSamplingMode = EnvironmentNeeSamplingMode::SkyBiased;
        FirstHitProbeSamplingMode firstHitProbeSamplingMode = FirstHitProbeSamplingMode::CosineHemisphere;
        int                   firstHitDiffuseSamples = 1;
        float                 targetFrameMs = 16.6f;
        bool                  enableReprojection = true;
        bool                  enableDenoiser = true;
        bool                  enableMotionAwareAccumulation = true;
        float                 motionAlphaMin = 0.14f;
        float                 motionAlphaMax = 0.55f;
        float                 historyResetMotionThreshold = 1.5f;
    };

    struct PathTracerPerfStats
    {
        float tlasBuildMs = 0.0f;
        float rayTraceMs = 0.0f;
        float reprojectionMs = 0.0f;
        float denoiserMs = 0.0f;
        float totalFrameMs = 0.0f;
        float totalFrameP50Ms = 0.0f;
        float totalFrameP95Ms = 0.0f;
        float totalFrameP99Ms = 0.0f;
        float rayTraceP95Ms = 0.0f;
        float denoiserP95Ms = 0.0f;
        uint32_t analysisSampleCount = 0;
        float historyAcceptanceRatio = 0.0f;
        float historyRejectionRatio = 0.0f;
        float skyHitRatio = 0.0f;
        float fireflyClampRatio = 0.0f;
        uint32_t historyAcceptedCount = 0;
        uint32_t historyRejectedCount = 0;
        uint32_t skyHitCount = 0;
        uint32_t fireflyClampCount = 0;
        uint32_t pixelSampleCount = 0;
        uint32_t targetWallSampleCount = 0;
        float targetWallLuminanceAverage = 0.0f;
        uint32_t firstHitProbeCount = 0;
        uint32_t firstHitProbeSurfaceHitCount = 0;
        uint32_t firstHitProbeSunVisibleCount = 0;
        float firstHitProbeSurfaceHitRatio = 0.0f;
        float firstHitProbeSunVisibleRatio = 0.0f;
        float firstHitProbeContributionAverage = 0.0f;
        float firstHitProbeSunVisibleContributionAverage = 0.0f;
        float cameraMotionFactor = 0.0f;
    };

    struct PathTracerAnalysisSettings
    {
        bool                         enableAnalysisMode = false;
        bool                         lockBenchmarkScene = false;
        bool                         benchmarkActive = false;
        bool                         runBaselineSweep = false;
        bool                         loadIndirectBounceTestScene = false;
        bool                         freezeCameraInputDuringBenchmark = true;
        PathTracerBenchmarkCameraPath cameraPath = PathTracerBenchmarkCameraPath::SlowPan;
        bool                         adaptiveSampling = true;
        int                          minSampleFrames = 120;
        int                          convergenceWindowFrames = 60;
        float                        p95ConvergenceThreshold = 0.02f;
        PathTracerDebugAov           debugAov = PathTracerDebugAov::FinalColor;
        int                          debugAtrousIteration = 0;
        int                          warmupFrames = 60;
        int                          sampleFrames = 240;
        float                        benchmarkVisualFidelityScore = 0.80f;
        bool                         runPhysicalSanityChecks = false;
        bool                         physicalSanityActive = false;
        bool                         physicalSanityPassed = false;
        float                        physicalSanityDriftMetric = 0.0f;
        std::string                  recommendationManual;
        std::string                  recommendationAutoBalanced;
        std::string                  recommendationAutoAggressive;
        std::string                  backlogSummary;
        std::string                  benchmarkCsvOutputPath;
        std::string                  backlogCsvOutputPath;
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
    TextureColorSpaceModel textureColorSpaceModel = TextureColorSpaceModel::HardwareSrgb;
    bool simulationRunning = false;
    float physicsTime = 0.0f; // updated by EngineCore after each tick
    glm::vec3 lightDirection = glm::vec3(-0.30f, -1.0f, -0.20f);
    float exposure = 1.0f;
    PathTracerSettings pathTracerSettings;
    PathTracerAnalysisSettings pathTracerAnalysisSettings;
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
    void drawPathTracerMainControls();
    void drawPathTracerDebugLab();
    void drawPathTracerBenchmarkControls();
    void drawPathTracerStats();

    void refreshAssetCache();

    static bool isDescendant(const SceneNode::Ptr &node, const SceneNode::Ptr &candidateParent);

    void drawPhysicsUI(Scene &scene, PhysicsSystem &physics,
                       ResourceManager &rm, vk::DescriptorSetLayout matLayout);

    void drawSelectedNodeTransformGizmo(Camera &camera);
};

#endif        // LAPHRIAENGINE_UISYSTEM_H
