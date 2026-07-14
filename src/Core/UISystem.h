#ifndef LAPHRIAENGINE_UISYSTEM_H
#define LAPHRIAENGINE_UISYSTEM_H

#include <random>
#include <string>
#include <cstdint>
#include <vector>

#include "../Physics/PhysicsSystem.h"
#include "../SceneManagement/Scene.h"
#include "EditorProject.h"
#include "Camera.h"
#include "EngineAuxiliary.h"
#include "VulkanDevice.h"

// Owns ImGui lifecycle, all editor draw calls, and UI-driven simulation state.
class UISystem {
public:
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

    enum class SurfelPathTracerDebugView
    {
        FinalColor = 0,
        GBufferNormal = 1,
        GBufferDepth = 2,
        SurfelId = 3,
        SurfelRadius = 4,
        SurfelRadiance = 5,
        SurfelVariance = 6,
        CellOccupancy = 7,
        ReflectionRaw = 8,
        ReflectionFiltered = 9,
        SurfelCoverage = 10,
        ReferenceColor = 11,
        ReferenceDifference = 12,
        GBufferAlbedo = 13,
        DiffuseGi = 14,
        SunVisibility = 15,
        AmbientOcclusion = 16,
        DiffuseGiBeforeAo = 17
    };
    static constexpr SurfelPathTracerDebugView kMaxSurfelPathTracerDebugView =
        SurfelPathTracerDebugView::DiffuseGiBeforeAo;

    struct PathTracerSettings
    {
        float                 resolutionScale = 1.0f;
        int                   denoiserIterations = 1;
        PathTracerQualityMode qualityMode = PathTracerQualityMode::Manual;
        bool                  reduceSecondaryEffects = false;
        bool                  enableEnvironmentNEE = true;
        // 0 = first bounce only, 1 = first 2 bounces, 2 = all bounces.
        int                   environmentNeeBounceMode = 0;
        bool                  blackEnvironment = false;
        EnvironmentNeeSamplingMode environmentNeeSamplingMode = EnvironmentNeeSamplingMode::SkyBiased;
        int                   pathTracerMaxBounces = 8;
        // 0 = all bounces, 1 = first bounce only, 2 = first 2 bounces.
        int                   directSunBounceMode = 0;
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
        float cameraMotionFactor = 0.0f;
    };

    struct SurfelPathTracerSettings
    {
        bool enabled = true;
        bool lockSurfels = false;
        bool resetSurfels = false;
        bool enableDiffuseGi = true;
        bool enableReflections = true;
        bool enableReflectionFilter = true;
        bool enableBilateralCleanup = true;
        bool enableTaa = true;
        uint32_t maxSurfels = 150000;
        uint32_t maxRaysPerFrame = 150000 * 16;
        float cellSize = 2.0f;
        uint32_t cellDimension = 64;
        uint32_t perCellSurfelLimit = 64;
        uint32_t irradianceAtlasWidth = 2048;
        uint32_t irradianceAtlasHeight = 4096;
        uint32_t minRaysPerSurfel = 4;
        uint32_t maxRaysPerSurfel = 64;
        uint32_t rayBudgetScale = 16;
        uint32_t activeMaxDepth = 3;
        uint32_t sleepingMaxDepth = 5;
        float placementThreshold = 0.35f;
        float removalThreshold = 4.0f;
        float varianceSensitivity = 1.2f;
        float surfelTargetArea = 16.0f;
        float surfelMinRadius = 0.05f;
        float surfelMaxRadiusScale = 2.0f;
        uint32_t maxSurfelSamplesPerQuery = 64;
        uint32_t maxRadianceSharingSamples = 32;
        uint32_t atlasTileSize = 6;
        bool enableGuidedSampling = false;
        bool enableSurfelTermination = true;
        bool useOriginalStyleGiNormalization = true;
        bool enableRadianceSharing = true;
        bool enableSurfelPlacement = true;
        bool enableSurfelRemoval = true;
        bool enableReferenceValidation = false;
        SurfelPathTracerDebugView debugView = SurfelPathTracerDebugView::FinalColor;
    };

    struct SurfelPathTracerStats
    {
        uint32_t aliveSurfels = 0;
        uint32_t deadSurfels = 0;
        uint32_t dirtySurfels = 0;
        uint32_t requestedRays = 0;
        uint32_t rayBudget = 0;
        uint32_t filledCells = 0;
        uint32_t rejectedStores = 0;
        uint32_t recycledSurfels = 0;
        uint32_t spawnedSurfels = 0;
        uint32_t removedSurfels = 0;
        uint32_t guidedRays = 0;
        uint32_t cosineRays = 0;
        uint32_t surfelTerminationAttempts = 0;
        uint32_t surfelTerminationHits = 0;
        uint32_t pathMisses = 0;
        float gBufferMs = 0.0f;
        float cacheUpdateMs = 0.0f;
        float prepareMs = 0.0f;
        float generateMs = 0.0f;
        float updateMs = 0.0f;
        float cellInfoMs = 0.0f;
        float cellMapMs = 0.0f;
        float rayScheduleMs = 0.0f;
        float surfelRayTraceMs = 0.0f;
        float integrateMs = 0.0f;
        float evaluateMs = 0.0f;
        float reflectionMs = 0.0f;
        float postProcessMs = 0.0f;
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
    TextureColorSpaceModel textureColorSpaceModel = TextureColorSpaceModel::HardwareSrgb;
    bool simulationRunning = false;
    float physicsTime = 0.0f; // updated by EngineCore after each tick
    glm::vec3 lightDirection = glm::vec3(-0.30f, -1.0f, -0.20f);
    float exposure = 1.0f;
    PathTracerSettings pathTracerSettings;
    PathTracerPerfStats pathTracerPerfStats;
    SurfelPathTracerSettings surfelPathTracerSettings;
    SurfelPathTracerStats surfelPathTracerStats;
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
    void drawPathTracerMainControls();
    void drawPathTracerAdvancedLightingControls();
    void drawPathTracerStats();

    void refreshAssetCache();

    static bool isDescendant(const SceneNode::Ptr &node, const SceneNode::Ptr &candidateParent);

    void drawPhysicsUI(Scene &scene, PhysicsSystem &physics,
                       ResourceManager &rm, vk::DescriptorSetLayout matLayout);

    void drawSelectedNodeTransformGizmo(Camera &camera);
};

#endif        // LAPHRIAENGINE_UISYSTEM_H
