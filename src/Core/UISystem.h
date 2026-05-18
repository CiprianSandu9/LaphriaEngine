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
        PathBaselineContinuationContribution = 16,
        PathReservoirGiContribution = 17,
        PathReservoirGiAcceptedLuma = 18,
        PathReservoirGiCandidateSurfaceHit = 19,
        PathReservoirGiCandidateSunVisible = 20,
        PathReservoirGiCandidatePositiveWeight = 21,
        PathReservoirGiSelectedWeight = 22,
        PathReservoirGiLocalNoLight = 23,
        PathReservoirGiSelectedSource = 24
    };

    enum class PathTracerQualityMode
    {
        Manual = 0,
        AutoBalanced = 1,
        AutoAggressive = 2
    };

    enum class PathTracerDebugLightPreset
    {
        HardBounce = 0,
        MediumBounce = 1,
        EasyBounce = 2
    };

    enum class EnvironmentNeeSamplingMode
    {
        CosineHemisphere = 0,
        SkyBiased = 1
    };

    enum class FirstHitProbeSamplingMode
    {
        CosineHemisphere = 0,
        SunBounceGuided = 1,
        CandidateSunBounce = 2,
        CandidateAverageReference = 3,
        CandidateRis = 4
    };

    enum class PathTracerReservoirGiMode
    {
        Off = 0,
        SingleFrame = 1,
        Temporal = 2,
        TemporalSpatial = 3
    };

    enum class PathTracerReservoirGiEstimatorAuditMode
    {
        Off = 0,
        Current = 1,
        NoProbeScale = 2
    };

    enum class PathTracerReservoirGiProposalMode
    {
        Cosine = 0,
        SunGuided = 1,
        MixedCosineSunGuided = 2,
        MixedCosineHistoryGuided = 3,
        MixedCosineSunReceiverGuided = 4,
        MixedCosineDualSunGuided = 5,
        MixedCosineSunReceiverCacheGuided = 6,
        MixedCosineSunReceiverCacheReconnect = 7,
        MixedCosineSunReceiverBrightSurfel = 8
    };

    enum class PathTracerSponzaValidationView
    {
        DarkCourtyard = 0,
        SunlitCourtyardWall = 1,
        MidDepthInterior = 2
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
        ReferenceDifference = 12
    };
    static constexpr SurfelPathTracerDebugView kMaxSurfelPathTracerDebugView =
        SurfelPathTracerDebugView::ReferenceDifference;

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
        bool                  applyFirstHitProbesToFinal = false;
        EnvironmentNeeSamplingMode environmentNeeSamplingMode = EnvironmentNeeSamplingMode::SkyBiased;
        FirstHitProbeSamplingMode firstHitProbeSamplingMode = FirstHitProbeSamplingMode::CosineHemisphere;
        int                   firstHitDiffuseSamples = 1;
        int                   firstHitCandidateCount = 4;
        PathTracerReservoirGiMode reservoirGiMode = PathTracerReservoirGiMode::Off;
        PathTracerReservoirGiProposalMode reservoirGiProposalMode = PathTracerReservoirGiProposalMode::Cosine;
        int                   reservoirGiCandidateCount = 2;
        int                   reservoirGiSpatialNeighborCount = 4;
        bool                  reservoirGiUseCandidateRis = true;
        bool                  reservoirGiDetailedDiagnostics = true;
        bool                  reservoirGiBrightSurfelShadowOnly = false;
        PathTracerReservoirGiEstimatorAuditMode reservoirGiEstimatorAuditMode =
            PathTracerReservoirGiEstimatorAuditMode::Off;
        int                   reservoirGiTemporalBudgetDivisor = 1;
        int                   reservoirGiSpatialBudgetDivisor = 1;
        int                   pathTracerMaxBounces = 8;
        // 0 = all bounces, 1 = first bounce only, 2 = first 2 bounces.
        int                   directSunBounceMode = 0;
        int                   reservoirGiCandidateEvaluationMode = 2;
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
        float targetWallBaseLuminanceAverage = 0.0f;
        float targetWallFirstHitProbeContributionAverage = 0.0f;
        uint32_t firstHitProbeCount = 0;
        uint32_t firstHitProbeSurfaceHitCount = 0;
        uint32_t firstHitProbeSunVisibleCount = 0;
        float firstHitProbeSurfaceHitRatio = 0.0f;
        float firstHitProbeSunVisibleRatio = 0.0f;
        float firstHitProbeContributionAverage = 0.0f;
        float firstHitProbeSunVisibleContributionAverage = 0.0f;
        uint32_t reservoirGiCandidates = 0;
        uint32_t reservoirGiAccepted = 0;
        uint32_t reservoirGiCandidateSurfaceHits = 0;
        uint32_t reservoirGiCandidateSunVisible = 0;
        uint32_t reservoirGiCandidatePositiveWeight = 0;
        uint32_t reservoirGiZeroWeight = 0;
        uint32_t reservoirGiTemporalAccepted = 0;
        uint32_t reservoirGiTemporalRejected = 0;
        uint32_t reservoirGiTemporalReuseAttempts = 0;
        uint32_t reservoirGiTemporalRejectGeometry = 0;
        uint32_t reservoirGiTemporalRejectVisibility = 0;
        uint32_t reservoirGiTemporalRejectLight = 0;
        uint32_t reservoirGiSpatialAccepted = 0;
        uint32_t reservoirGiSpatialRejected = 0;
        uint32_t reservoirGiSelectedLocal = 0;
        uint32_t reservoirGiSelectedTemporal = 0;
        uint32_t reservoirGiSelectedSpatial = 0;
        uint32_t reservoirGiSelectedCache = 0;
        uint32_t reservoirGiSelectedCacheReconnect = 0;
        uint32_t reservoirGiLocalSurfaceHits = 0;
        uint32_t reservoirGiLocalValidSamples = 0;
        uint32_t reservoirGiLocalMissCandidates = 0;
        uint32_t reservoirGiLocalMissPositiveWeight = 0;
        uint32_t reservoirGiLocalSurfaceInvalid = 0;
        uint32_t reservoirGiLocalRejectGeometry = 0;
        uint32_t reservoirGiLocalRejectNoLight = 0;
        uint32_t reservoirGiLocalRejectZeroTarget = 0;
        uint32_t reservoirGiLocalRejectBadPdf = 0;
        uint32_t reservoirGiAcceptedLocalSurface = 0;
        uint32_t reservoirGiAcceptedLocalMiss = 0;
        uint32_t reservoirGiLocalShadowRays = 0;
        uint32_t reservoirGiTemporalReconnectRays = 0;
        uint32_t reservoirGiTemporalShadowRays = 0;
        uint32_t reservoirGiHistoryGuideUsed = 0;
        uint32_t reservoirGiHistoryGuideRejectedLowWeight = 0;
        uint32_t reservoirGiHistoryGuideFallbackCosine = 0;
        uint32_t reservoirGiHistoryGuideRejectReprojection = 0;
        uint32_t reservoirGiHistoryGuideRejectLoad = 0;
        uint32_t reservoirGiHistoryGuideRejectGeometry = 0;
        uint32_t reservoirGiHistoryGuideNeighborSearches = 0;
        uint32_t reservoirGiHistoryGuideNeighborHits = 0;
        uint32_t reservoirGiHistoryGuideNeighborMisses = 0;
        uint32_t reservoirGiReceiverCacheStore = 0;
        uint32_t reservoirGiReceiverCacheAttempt = 0;
        uint32_t reservoirGiReceiverCacheHit = 0;
        uint32_t reservoirGiReceiverCacheMiss = 0;
        uint32_t reservoirGiReceiverCacheRejectNoLight = 0;
        uint32_t reservoirGiReceiverCacheAccepted = 0;
        uint32_t reservoirGiReceiverReconnectAttempt = 0;
        uint32_t reservoirGiReceiverReconnectHit = 0;
        uint32_t reservoirGiReceiverReconnectMiss = 0;
        uint32_t reservoirGiReceiverReconnectRejectVisibility = 0;
        uint32_t reservoirGiReceiverReconnectRejectTarget = 0;
        uint32_t reservoirGiReceiverReconnectAccepted = 0;
        uint32_t reservoirGiReceiverCacheContinuationAttempt = 0;
        uint32_t reservoirGiReceiverCacheContinuationHit = 0;
        uint32_t reservoirGiReceiverCacheContinuationMiss = 0;
        uint32_t reservoirGiReceiverCacheContinuationAccepted = 0;
        uint32_t reservoirGiBrightSurfelStore = 0;
        uint32_t reservoirGiBrightSurfelAttempt = 0;
        uint32_t reservoirGiBrightSurfelHit = 0;
        uint32_t reservoirGiBrightSurfelMiss = 0;
        uint32_t reservoirGiBrightSurfelRejectVisibility = 0;
        uint32_t reservoirGiBrightSurfelRejectGeometry = 0;
        uint32_t reservoirGiBrightSurfelRejectTarget = 0;
        uint32_t reservoirGiBrightSurfelAccepted = 0;
        uint32_t reservoirGiSelectedBrightSurfel = 0;
        uint32_t reservoirGiBrightSurfelPrecheckRejectTarget = 0;
        uint32_t reservoirGiBrightSurfelTrainingAttempt = 0;
        uint32_t reservoirGiBrightSurfelTrainingStore = 0;
        uint32_t reservoirGiBrightSurfelTrainingRejectGeometry = 0;
        uint32_t reservoirGiBrightSurfelTrainingRejectTarget = 0;
        uint32_t reservoirGiBrightSurfelSelectorRejectGeometry = 0;
        uint32_t reservoirGiBrightSurfelSelectorRejectTarget = 0;
        uint32_t reservoirGiBrightSurfelSelectorViable = 0;
        uint32_t reservoirGiBrightSurfelIndexedQuery = 0;
        uint32_t reservoirGiBrightSurfelIndexedEmpty = 0;
        uint32_t reservoirGiBrightSurfelIndexedProbe = 0;
        uint32_t reservoirGiBrightSurfelSelectorRejectDistance = 0;
        uint32_t reservoirGiBrightSurfelSelectorRejectReceiverHemisphere = 0;
        uint32_t reservoirGiBrightSurfelSelectorRejectSurfelHemisphere = 0;
        uint32_t reservoirGiBrightSurfelSelectorRejectInvalidVector = 0;
        float reservoirGiAcceptedAvgLuma = 0.0f;
        float reservoirGiAcceptedLumaSum = 0.0f;
        float reservoirGiCandidateSurfaceHitRatio = 0.0f;
        float reservoirGiCandidateSunVisibleRatio = 0.0f;
        float reservoirGiCandidatePositiveWeightRatio = 0.0f;
        float reservoirGiLocalValidRatio = 0.0f;
        float reservoirGiSelectedWeightAverage = 0.0f;
        float reservoirGiTargetWeightAverage = 0.0f;
        float reservoirGiConfidenceMAvg = 0.0f;
        float reservoirGiAuditCurrentLuma = 0.0f;
        float reservoirGiAuditReferenceLuma = 0.0f;
        float reservoirGiAuditRelativeErrorPct = 0.0f;
        float reservoirGiAuditProbeScale = 0.0f;
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
        uint32_t maxRaysPerFrame = 150000 * 64;
        float cellSize = 2.0f;
        uint32_t cellDimension = 64;
        uint32_t perCellSurfelLimit = 64;
        uint32_t irradianceAtlasWidth = 2048;
        uint32_t irradianceAtlasHeight = 2048;
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
        uint32_t maxSurfelSamplesPerQuery = 32;
        uint32_t maxRadianceSharingSamples = 32;
        uint32_t atlasTileSize = 6;
        bool enableGuidedSampling = true;
        bool enableSurfelTermination = true;
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
        uint32_t filledCells = 0;
        uint32_t rejectedStores = 0;
        uint32_t recycledSurfels = 0;
        uint32_t spawnedSurfels = 0;
        uint32_t removedSurfels = 0;
        uint32_t guidedRays = 0;
        uint32_t cosineRays = 0;
        uint32_t surfelTerminatedPaths = 0;
        uint32_t pathMisses = 0;
        float totalFrameMs = 0.0f;
    };

    struct PathTracerAnalysisSettings
    {
        bool                         enableAnalysisMode = false;
        bool                         lockBenchmarkScene = false;
        bool                         benchmarkActive = false;
        bool                         runBaselineSweep = false;
        bool                         runSponzaGiPerfSweep = false;
        bool                         runBrightSurfelShadowEvaluationSweep = false;
        bool                         runBrightSurfelProposalEvaluationSweep = false;
        bool                         loadIndirectBounceTestScene = false;
        bool                         loadSponzaGiValidationPreset = false;
        PathTracerSponzaValidationView sponzaValidationView = PathTracerSponzaValidationView::DarkCourtyard;
        bool                         applySponzaValidationView = false;
        PathTracerDebugLightPreset   debugLightPreset = PathTracerDebugLightPreset::HardBounce;
        bool                         applyDebugLightPreset = false;
        bool                         freezeCameraInputDuringBenchmark = true;
        PathTracerBenchmarkCameraPath cameraPath = PathTracerBenchmarkCameraPath::SlowPan;
        bool                         adaptiveSampling = true;
        int                          minSampleFrames = 120;
        int                          convergenceWindowFrames = 60;
        float                        p95ConvergenceThreshold = 0.02f;
        PathTracerDebugAov           debugAov = PathTracerDebugAov::FinalColor;
        int                          debugAtrousIteration = 0;
        int                          warmupFrames = 30;
        int                          sampleFrames = 120;
        int                          sponzaGiSweepWarmupFrames = 8;
        int                          sponzaGiSweepSampleFrames = 32;
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
