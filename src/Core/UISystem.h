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
        static constexpr uint32_t kAtlasTileSize = 6;
        static constexpr uint32_t kMinMaxSurfels = 1024;
        static constexpr uint32_t kMaxAtlasDimension = 4096;
        static constexpr uint64_t kMaxStorageBufferBytes = 128ull * 1024ull * 1024ull;
        static constexpr uint32_t kRayRecordBytes = 8u * sizeof(uint32_t);
        static constexpr uint32_t kMaxRayCapacity =
            static_cast<uint32_t>(kMaxStorageBufferBytes / kRayRecordBytes);

        static constexpr uint32_t atlasCapacity(uint32_t width, uint32_t height)
        {
            return (width / kAtlasTileSize) * (height / kAtlasTileSize);
        }

        // The compact cell map is packed (CellInfo assigns each cell a range from a running
        // total), so it is sized per surfel, not per cell: grid resolution and per-cell limit
        // are independent. Each surfel is inserted into every cell it overlaps, which with the
        // default radii is one to a few cells; 8 entries per surfel leaves generous headroom.
        static constexpr uint32_t kCellMapEntriesPerSurfel = 8;
        static constexpr uint32_t kMaxPerCellLimit = 256;
        static constexpr uint64_t cellMapEntryCount(uint32_t dimension, uint32_t perCellLimit, uint32_t maxSurfels)
        {
            const uint64_t safeDimension = dimension > 0u ? dimension : 1u;
            const uint64_t cellCount = safeDimension * safeDimension * safeDimension;
            const uint64_t worstCase = cellCount * (perCellLimit > 0u ? perCellLimit : 1u);
            const uint64_t perSurfel = static_cast<uint64_t>(maxSurfels) * kCellMapEntriesPerSurfel;
            return worstCase < perSurfel ? worstCase : perSurfel;
        }

        bool enabled = true;
        bool lockSurfels = false;
        bool resetSurfels = false;
        bool enableDiffuseGi = true;
        bool enableReflections = true;
        bool enableReflectionFilter = true;
        bool enableBilateralCleanup = true;
        bool enableTaa = true;
        // 400k surfels keep Sponza's foliage-driven population below the 85% pressure
        // threshold, so placement never waits for the 480-frame eviction clock.
        uint32_t maxSurfels = 400000;
        uint32_t maxRaysPerFrame = 400000 * 8; // 3.2M rays, 102 MiB ray buffer (cap 128 MiB)
        // 0.75 m: Cell Size is the lookup window/index granularity only (48 m span at
        // 64^3); the effective surfel size is surfelSupportRadius below (clamped to Cell
        // Size). At 2.0 m a cell held far more surfels than the 64-entry compact map
        // kept, starving ray scheduling and the resolve.
        float cellSize = 0.75f;
        uint32_t cellDimension = 64;
        uint32_t perCellSurfelLimit = 128;      // compact-map slots per cell; the map itself is sized per surfel (cellMapEntryCount)
        uint32_t irradianceAtlasWidth = 4096;   // 4096x4096 tiles of 6x6 = 465 124 surfels of capacity
        uint32_t irradianceAtlasHeight = 4096;
        uint32_t minRaysPerSurfel = 4;
        uint32_t maxRaysPerSurfel = 64;
        // Represented surfels whose support sphere is outside the view frustum trace
        // only every N-th frame (staggered per surfel). 1 disables the skip. Surfels in
        // warm-up always trace.
        uint32_t offscreenRayInterval = 4;
        uint32_t activeMaxDepth = 3;
        uint32_t sleepingMaxDepth = 5;
        float placementThreshold = 0.35f;
        // Removal at 12 keeps roughly a dozen overlapping supports per point; 4.0 left
        // the cache too sparse once coverage measured the real support radius.
        float removalThreshold = 12.0f;
        float varianceSensitivity = 1.2f;
        float surfelTargetArea = 16.0f;
        float surfelMinRadius = 0.05f;
        float surfelMaxRadiusScale = 2.0f;
        // World-space support radius shared by resolve, coverage, path termination and
        // radiance sharing: each surfel weights a point with max(own radius, this).
        // Decoupled from Cell Size (it used to be 0.75 x Cell Size); clamped to Cell
        // Size so the fixed +/-1 cell lookup neighborhood stays complete.
        float surfelSupportRadius = 0.25f;
        uint32_t maxSurfelSamplesPerQuery = 64;
        uint32_t maxRadianceSharingSamples = 32;
        bool enableGuidedSampling = false;
        bool enableSurfelTermination = true;
        // Surfels store irradiance (Integrate weights each ray by cos/pdf), so the
        // physically consistent diffuse consumption is albedo / pi, matching the
        // direct-lighting BRDF. "Original style" (albedo only) over-brightens all
        // indirect light by a factor of pi; kept as a toggle for comparison only.
        bool useOriginalStyleGiNormalization = false;
        bool enableRadianceSharing = true;
        bool enableSurfelPlacement = true;
        bool enableSurfelRemoval = true;
        bool enableReferenceValidation = false;
        // Roughness band over which the traced glossy reflection fades into the specular
        // term evaluated from the surfel cache (env-BRDF x E/pi). At or above the end value
        // no reflection ray is traced at all.
        float roughReflectionStart = 0.5f;
        float roughReflectionEnd = 0.7f;
        SurfelPathTracerDebugView debugView = SurfelPathTracerDebugView::FinalColor;
    };

    struct SurfelPathTracerStats
    {
        uint32_t aliveSurfels = 0;
        uint32_t deadSurfels = 0;
        uint32_t dirtySurfels = 0;
        uint32_t requestedRays = 0;
        uint32_t demandedRays = 0;
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
        // Rolling-window statistics over the last N surfel frames (N = rollingSamples),
        // filled by EngineCore::collectSurfelPathTracerTimings. Reset with
        // resetSurfelPathTracerStatsWindow after warm-up to measure a steady state.
        uint32_t rollingSamples = 0;
        float avgGBufferMs = 0.0f;
        float avgCacheUpdateMs = 0.0f;
        float avgSurfelRayTraceMs = 0.0f;
        float avgIntegrateMs = 0.0f;
        float avgEvaluateMs = 0.0f;
        float avgReflectionMs = 0.0f;
        float avgPostProcessMs = 0.0f;
        float avgTotalFrameMs = 0.0f;
        float totalP50Ms = 0.0f;
        float totalP95Ms = 0.0f;
        // Rolling means of the per-frame counters over the same window.
        float avgAliveSurfels = 0.0f;
        float avgRequestedRays = 0.0f;
        float avgDemandedRays = 0.0f;
        float avgFilledCells = 0.0f;
        float avgRejectedStores = 0.0f;
        float avgSpawnedSurfels = 0.0f;
        float avgRemovedSurfels = 0.0f;
        float avgRecycledSurfels = 0.0f;
        float avgGuidedRays = 0.0f;
        float avgCosineRays = 0.0f;
        float avgTerminationAttempts = 0.0f;
        float avgTerminationHits = 0.0f;
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
    // Auto-exposure: EngineCore measures the log-average luminance of the HDR image and, when
    // enabled, writes the adapted exposure into `exposure`. Disabling it locks the last value,
    // which is how both backends are captured at an identical exposure.
    struct AutoExposureSettings
    {
        bool  enabled = false;
        float key = 0.18f;            // target mid-grey for the log-average luminance
        float minExposure = 0.05f;
        float maxExposure = 8.0f;
        float adaptationSpeed = 0.7f; // 1/s, applied in log2 (stops); ~1.4 s time constant
        float measuredLogMeanLuminance = 0.0f;
        float targetExposure = 0.0f;
    };
    AutoExposureSettings autoExposure;
    // Pixel probe (SurfelPathTracer): linear values of one pixel, read back by EngineCore.
    struct PixelProbe
    {
        bool     enabled = false;
        bool     freeze = false;
        uint32_t x = 0, y = 0;               // requested pixel (follows the mouse unless frozen)
        uint32_t sampledX = 0, sampledY = 0; // pixel the current values were read from
        bool     valid = false;
        glm::vec3 lighting{0.0f};            // linear composite before tonemapping
        glm::vec3 resolvedIrradiance{0.0f};  // surfel resolve output (E)
        float     coverage = 0.0f;
        glm::vec3 albedo{0.0f};
        glm::vec3 normal{0.0f};
        float     sunVisibility = 0.0f;
        float     ao = 0.0f;
        float     depth = 0.0f;
        glm::vec3 reference{0.0f};           // 1 spp reference probe, if enabled
        bool      referenceValid = false;
    };
    PixelProbe pixelProbe;
    PathTracerSettings pathTracerSettings;
    PathTracerPerfStats pathTracerPerfStats;
    SurfelPathTracerSettings surfelPathTracerSettings;
    SurfelPathTracerStats surfelPathTracerStats;
    // Set by the panel's "Reset stats window" button; consumed by EngineCore.
    bool resetSurfelPathTracerStatsWindow = false;
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

    SurfelPathTracerSettings surfelResourceSettingsDraft;
    bool surfelResourceSettingsDraftInitialized = false;

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
