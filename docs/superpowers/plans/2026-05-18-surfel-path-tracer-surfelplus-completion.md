# Surfel Path Tracer SurfelPlus Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the existing `SurfelPathTracer` backend as a real-time, SurfelGI/SurfelPlus-informed path-traced GI pipeline, using surfels as a dynamic radiance cache and acceleration structure rather than replacing the renderer with an offline path tracer.

**Architecture:** Keep the current pass graph and Vulkan/Slang integration, then upgrade the simplified passes toward SurfelPlus behavior: camera-relative surfel cells, lifecycle/recycling, adaptive ray budgets, directional irradiance/depth atlas, guided surfel rays, surfel-terminated transport, MSME integration, radiance sharing, RIS reflections, and correct ping-pong temporal history. Each task leaves `RenderMode::SurfelPathTracer` runnable and biased toward real-time stability.

**Tech Stack:** C++20, Vulkan-Hpp RAII, VMA helpers, Slang SPIR-V shaders, CMake shader compilation, ImGui, existing `LaphriaEngineUnitTests`, local references under `.codex_refs/SurfelGI` and `.codex_refs/SurfelPlus`.

---

## Design Direction

This plan corrects the post-review gaps without changing the original intent. The target is not an unbiased offline renderer. The target is a real-time path-traced GI pipeline in the style of SurfelGI and SurfelPlus:

- sparse path tracing from surfels and glossy reflection points;
- surfel cache termination for diffuse indirect transport;
- adaptive ray allocation based on variance, visibility, age, and sleep state;
- 6x6 directional irradiance/depth atlas per surfel for guided sampling;
- direct lighting visibility as a first-class contract; unshadowed sun is allowed only as a temporary debug fallback;
- dynamic surfel placement and removal based on screen coverage;
- temporal/spatial filtering to hide low sample counts;
- debug/reference modes to measure bias and convergence, not to drive the main frame.

The current backend already has the right host shape: `SurfelPathTracerResources`, `SurfelPathTracerPipelines`, `SurfelPathTracerPasses`, a separate GBuffer RT pipeline, surfel ray RT pipeline, reflection RT pipeline, compute filters, and UI controls. The work below upgrades the shader semantics and resource contracts inside that architecture.

## Reference Mapping

Use these local reference points when implementing:

- `.codex_refs/SurfelPlus/docs/index.md`: pass overview and real-time feature descriptions.
- `.codex_refs/SurfelPlus/shaders/surfel_update.comp`: lifecycle, adaptive radius, adaptive ray allocation, alive/dead recycle.
- `.codex_refs/SurfelPlus/shaders/surfel_generation_pass.comp`: coverage evaluation, surfel spawn, over-coverage removal.
- `.codex_refs/SurfelPlus/shaders/surfel_raytrace.comp`: guided sampling using a 6x6 irradiance patch, fallback cosine sampling, path termination.
- `.codex_refs/SurfelPlus/shaders/surfel_integrate.comp`: MSME, 6x6 irradiance/depth atlas updates, radiance sharing.
- `.codex_refs/SurfelPlus/shaders/reflection_generation.comp`: RIS reflection candidate selection and surfel-assisted termination.
- `.codex_refs/SurfelPlus/shaders/taa_pass.comp`: ping-pong history and reprojection.
- `.codex_refs/SurfelGI/RenderPasses/Surfel/SurfelGI/SurfelUtils.slang`: camera-relative cell position, radius, cell intersection.
- `.codex_refs/SurfelGI/RenderPasses/Surfel/SurfelGI/SurfelUpdatePass.cs.slang`: neighbor-cell insertion and MSME-variance ray allocation.
- `.codex_refs/SurfelGI/RenderPasses/Surfel/SurfelGI/SurfelRayTrace.rt.slang`: surfel-terminated ray paths.
- `.codex_refs/SurfelGI/RenderPasses/Surfel/SurfelGI/SurfelIntegratePass.cs.slang`: irradiance atlas and radiance sharing.

## File Structure

Modify:

- `src/Core/SurfelPathTracerResources.h`: Extend host-shared surfel/counter structs and helper APIs for camera-relative cells, lifecycle fields, atlas tile constants, and previous/current history indices.
- `src/Core/SurfelPathTracerResources.cpp`: Allocate enlarged surfel buffers, 6x6 atlas storage, ping-pong history images, and CPU cell-address tests.
- `src/Core/SurfelPathTracerPasses.h`: Add pass parameters for camera position, adaptive ray limits, debug/reference toggles, history source/destination frame indices, and reference path-trace dispatch.
- `src/Core/SurfelPathTracerPasses.cpp`: Record the updated pass order, push constants, barriers, and ping-pong history bindings.
- `src/Core/SurfelPathTracerPipelines.h`: Add optional reference/validation RT pipeline handles when introduced.
- `src/Core/SurfelPathTracerPipelines.cpp`: Create new or renamed pipelines and keep SBT ownership per RT family.
- `src/Core/EngineCore.cpp`: Refresh resource recreation contracts, pass sequencing, history ping-pong state, stats readback, and UI setting consumption.
- `src/Core/EngineCore.h`: Store previous/current Surfel PT history indices and static settings snapshots.
- `src/Core/UISystem.h`: Add settings for real-time budgets, guided sampling, lifecycle, reference validation, and debug views.
- `src/Core/UISystem.cpp`: Add bounded controls for new settings and expose debug views without marketing text.
- `src/shaders/SurfelPathTracerCommon.slang`: Shared surfel structs, MSME, BSDF/path helpers, atlas mapping, cell mapping, and sampling utilities.
- `src/shaders/SurfelPathTracerPrepare.slang`: Initialize transient counters and reset persistent lifecycle/atlas metadata.
- `src/shaders/SurfelPathTracerUpdate.slang`: Lifecycle, radius, camera-relative cells, neighbor-cell counts, and adaptive ray allocation.
- `src/shaders/SurfelPathTracerCellInfo.slang`: Preserve offset allocation but verify capacity and rejection counters for multi-cell insertion.
- `src/shaders/SurfelPathTracerCellToSurfel.slang`: Insert surfels into every intersecting neighbor cell.
- `src/shaders/SurfelPathTracerRaygen.slang`: Use guided/cosine sampling, multi-bounce path transport, surfel termination, and per-ray PDF storage.
- `src/shaders/SurfelPathTracerClosestHit.slang`: Return material/surface data required by reusable path transport, not only direct sun/sky radiance.
- `src/shaders/SurfelPathTracerMiss.slang`: Return sky/environment radiance for path transport.
- `src/shaders/SurfelPathTracerIntegrate.slang`: MSME radiance aggregation, directional atlas update, depth moments, and radiance sharing.
- `src/shaders/SurfelPathTracerEvaluate.slang`: Screen-space diffuse resolve, placement, last-seen marking, over-coverage removal, and debug modes.
- `src/shaders/SurfelPathTracerReflection.slang`: RIS reflection sampling and surfel-assisted one-bounce reflection.
- `src/shaders/SurfelPathTracerReflectionFilter.slang`: Re-enable temporal accumulation using previous/current history images.
- `src/shaders/SurfelPathTracerTaa.slang`: Re-enable TAA using previous/current history images.
- `src/shaders/SurfelPathTracerLightIntegrate.slang`: Compose direct, surfel diffuse, reflection, optional reference diff, and debug views.
- `tests/SurfelPathTracerPipelineTests.cpp`: Expand structural and CPU contract tests for the upgraded architecture.

No new top-level backend is created. Any new helper shader should be introduced only when it replaces duplicated logic in at least two existing shaders.

## Non-Goals

- Do not build an offline/unbiased path tracer as the main Surfel PT mode.
- Do not remove the existing `PathTracer` backend.
- Do not introduce dependency downloads or external render graph frameworks.
- Do not implement non-uniform SurfelPlus cells before the camera-relative uniform grid is stable and tested.
- Do not re-enable temporal history by reading and writing the same image in one frame.

---

### Task 1: Add Incremental Contract Test Helpers

**Files:**

- Modify: `tests/SurfelPathTracerPipelineTests.cpp`

- [ ] **Step 1: Add reusable contract helpers without adding future requirements**

Add these helpers near the existing `containsNeedle` helper. They must be used by later tasks as each feature lands; do not add future-facing required needles in this task.

```cpp
bool containsAllNeedles(std::string_view haystack, std::initializer_list<std::string_view> needles)
{
    bool ok = true;
    for (std::string_view needle : needles)
    {
        if (!containsNeedle(haystack, needle))
        {
            std::cerr << "missing SurfelPathTracer contract: " << needle << '\n';
            ok = false;
        }
    }
    return ok;
}

bool appearsBefore(std::string_view haystack, std::string_view first, std::string_view second)
{
    const auto firstPos = haystack.find(first);
    const auto secondPos = haystack.find(second);
    if (firstPos == std::string_view::npos || secondPos == std::string_view::npos || firstPos >= secondPos)
    {
        std::cerr << "SurfelPathTracer pass-order contract failed: " << first << " before " << second << '\n';
        return false;
    }
    return true;
}
```

- [ ] **Step 2: Add a passing CPU helper contract for current cell helpers**

Add a new test function in the same file:

```cpp
bool testSurfelPathTracerCellAddressBounds()
{
    const auto atCenter = Laphria::SurfelPathTracerResources::cellAddressForPosition(glm::vec3(0.0f), 1.0f, 64);
    const auto atFar = Laphria::SurfelPathTracerResources::cellAddressForPosition(glm::vec3(100000.0f), 1.0f, 64);

    if (atCenter.flatIndex >= 64u * 64u * 64u || atFar.flatIndex >= 64u * 64u * 64u)
    {
        std::cerr << "surfel cell address escaped valid bounds\n";
        return false;
    }
    return true;
}
```

Update the exported test to return both `testSurfelPathTracerPipelineContracts()` and `testSurfelPathTracerCellAddressBounds()`. If the existing header exposes only one function, keep the public function name and call the helper internally.

- [ ] **Step 3: Run the test**

Run:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests"
.\cmake-build-debug\LaphriaEngineUnitTests.exe
```

Expected: build succeeds and the unit executable exits `0`.

- [ ] **Step 4: Commit the passing helpers**

```powershell
git add tests/SurfelPathTracerPipelineTests.cpp
git commit -m "test: add surfel path tracer contract helpers"
```

### Task 2: Extend Host-Shared Settings, Stats, And Debug Views

**Files:**

- Modify: `src/Core/UISystem.h`
- Modify: `src/Core/UISystem.cpp`
- Modify: `tests/SurfelPathTracerPipelineTests.cpp`

- [ ] **Step 1: Add settings fields**

In `UISystem::SurfelPathTracerSettings`, add:

```cpp
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
```

Extend `SurfelPathTracerDebugView` with:

```cpp
SurfelCoverage,
SurfelVariance,
SurfelRadius,
ReferenceColor,
ReferenceDifference,
```

Update `kMaxSurfelPathTracerDebugView` to `ReferenceDifference`.

- [ ] **Step 2: Add UI clamps and controls**

In the Surfel Path Tracer UI block in `UISystem.cpp`, clamp:

```cpp
settings.minRaysPerSurfel = std::clamp(settings.minRaysPerSurfel, 1u, 64u);
settings.maxRaysPerSurfel = std::clamp(settings.maxRaysPerSurfel, settings.minRaysPerSurfel, 128u);
settings.rayBudgetScale = std::clamp(settings.rayBudgetScale, 1u, 64u);
settings.activeMaxDepth = std::clamp(settings.activeMaxDepth, 1u, 8u);
settings.sleepingMaxDepth = std::clamp(settings.sleepingMaxDepth, settings.activeMaxDepth, 8u);
settings.placementThreshold = std::clamp(settings.placementThreshold, 0.05f, 4.0f);
settings.removalThreshold = std::clamp(settings.removalThreshold, settings.placementThreshold, 16.0f);
settings.varianceSensitivity = std::clamp(settings.varianceSensitivity, 0.01f, 16.0f);
settings.surfelTargetArea = std::clamp(settings.surfelTargetArea, 1.0f, 256.0f);
settings.surfelMinRadius = std::clamp(settings.surfelMinRadius, 0.001f, 1.0f);
settings.surfelMaxRadiusScale = std::clamp(settings.surfelMaxRadiusScale, 0.25f, 8.0f);
settings.maxSurfelSamplesPerQuery = std::clamp(settings.maxSurfelSamplesPerQuery, 1u, 128u);
settings.maxRadianceSharingSamples = std::clamp(settings.maxRadianceSharingSamples, 1u, 128u);
settings.atlasTileSize = 6u;
settings.irradianceAtlasWidth = std::clamp(settings.irradianceAtlasWidth, 512u, 4096u);
const uint32_t tilesPerRow = std::max(settings.irradianceAtlasWidth / settings.atlasTileSize, 1u);
const uint32_t requiredRows = (settings.maxSurfels + tilesPerRow - 1u) / tilesPerRow;
settings.irradianceAtlasHeight = std::clamp(std::max(settings.irradianceAtlasHeight,
                                                     requiredRows * settings.atlasTileSize),
                                            512u,
                                            4096u);
const uint32_t atlasSurfelsCapacity = tilesPerRow * std::max(settings.irradianceAtlasHeight / settings.atlasTileSize, 1u);
settings.maxSurfels = std::min(settings.maxSurfels, atlasSurfelsCapacity);
settings.maxRaysPerFrame = std::clamp(settings.maxRaysPerFrame,
                                      1024u,
                                      settings.maxSurfels * 64u);
settings.maxRaysPerFrame = std::max(settings.maxRaysPerFrame,
                                    settings.maxSurfels * settings.rayBudgetScale);
```

Add matching ImGui sliders/checkboxes near existing Surfel PT controls. Use existing style; do not add explanatory text blocks.

- [ ] **Step 3: Add contract needles**

Add test strings:

```cpp
"minRaysPerSurfel",
"maxRaysPerSurfel",
"rayBudgetScale",
"activeMaxDepth",
"sleepingMaxDepth",
"placementThreshold",
"removalThreshold",
"varianceSensitivity",
"enableGuidedSampling",
"enableSurfelTermination",
"enableRadianceSharing",
"enableSurfelPlacement",
"enableSurfelRemoval",
"enableReferenceValidation",
"atlasTileSize",
```

- [ ] **Step 4: Verify**

Run:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests"
.\cmake-build-debug\LaphriaEngineUnitTests.exe
```

Expected: build succeeds and the unit executable exits `0`.

- [ ] **Step 5: Commit**

```powershell
git add src/Core/UISystem.h src/Core/UISystem.cpp tests/SurfelPathTracerPipelineTests.cpp
git commit -m "feat: add surfel path tracer completion controls"
```

### Task 3: Expand Surfel State And Counters

**Files:**

- Modify: `src/Core/SurfelPathTracerResources.h`
- Modify: `src/Core/SurfelPathTracerResources.cpp`
- Modify: `src/shaders/SurfelPathTracerCommon.slang`
- Modify: `src/shaders/SurfelPathTracerPrepare.slang`
- Modify: `tests/SurfelPathTracerPipelineTests.cpp`

- [ ] **Step 1: Extend surfel structs in C++ and Slang**

Add these fields after the current surfel fields in both host and shader structs, preserving 16-byte alignment:

```cpp
uint32_t lastSeenFrame = 0;
uint32_t lastReferencedFrame = 0;
uint32_t sleepState = 0;
uint32_t materialKey = 0;
glm::vec4 varianceAndInconsistency{1.0f, 1.0f, 1.0f, 1.0f};
```

Slang equivalent:

```hlsl
uint lastSeenFrame;
uint lastReferencedFrame;
uint sleepState;
uint materialKey;
float4 varianceAndInconsistency;
```

Add constants in `SurfelPathTracerCommon.slang`:

```hlsl
static const uint SURFEL_PT_ATLAS_TILE_SIZE = 6u;
static const uint SURFEL_PT_SLEEP_AWAKE = 0u;
static const uint SURFEL_PT_SLEEP_SLEEPING = 1u;
static const uint SURFEL_PT_STATUS_LAST_SEEN = 0x2u;
static const uint SURFEL_PT_STATUS_LAST_REFERENCED = 0x4u;
```

Add counter offsets matching the extended C++ counter layout. Keep the offsets in bytes and update every shader that uses the old counter constants:

```hlsl
static const uint SURFEL_PT_COUNTER_RECYCLED_SURFELS_OFFSET = 32u;
static const uint SURFEL_PT_COUNTER_SPAWNED_SURFELS_OFFSET = 36u;
static const uint SURFEL_PT_COUNTER_REMOVED_SURFELS_OFFSET = 40u;
static const uint SURFEL_PT_COUNTER_GUIDED_RAYS_OFFSET = 44u;
static const uint SURFEL_PT_COUNTER_COSINE_RAYS_OFFSET = 48u;
static const uint SURFEL_PT_COUNTER_SURFEL_TERMINATED_PATHS_OFFSET = 52u;
static const uint SURFEL_PT_COUNTER_PATH_MISSES_OFFSET = 56u;
```

- [ ] **Step 2: Extend counters**

Add fields to `SurfelPathTracerCounters`:

```cpp
uint32_t recycledSurfels = 0;
uint32_t spawnedSurfels = 0;
uint32_t removedSurfels = 0;
uint32_t guidedRays = 0;
uint32_t cosineRays = 0;
uint32_t surfelTerminatedPaths = 0;
uint32_t pathMisses = 0;
uint32_t pad1 = 0;
```

Update `readStats()` and UI stats with the same names.

- [ ] **Step 3: Initialize new fields in prepare**

When `resetPersistent != 0u`, initialize the new fields:

```hlsl
surfel.lastSeenFrame = 0u;
surfel.lastReferencedFrame = 0u;
surfel.sleepState = SURFEL_PT_SLEEP_AWAKE;
surfel.materialKey = 0u;
surfel.varianceAndInconsistency = float4(1.0, 1.0, 1.0, 1.0);
```

Clear new transient counters at `index == 0u`.

```hlsl
counters.Store(SURFEL_PT_COUNTER_RECYCLED_SURFELS_OFFSET, 0u);
counters.Store(SURFEL_PT_COUNTER_SPAWNED_SURFELS_OFFSET, 0u);
counters.Store(SURFEL_PT_COUNTER_REMOVED_SURFELS_OFFSET, 0u);
counters.Store(SURFEL_PT_COUNTER_GUIDED_RAYS_OFFSET, 0u);
counters.Store(SURFEL_PT_COUNTER_COSINE_RAYS_OFFSET, 0u);
counters.Store(SURFEL_PT_COUNTER_SURFEL_TERMINATED_PATHS_OFFSET, 0u);
counters.Store(SURFEL_PT_COUNTER_PATH_MISSES_OFFSET, 0u);
```

- [ ] **Step 4: Verify**

Run the unit target and executable. Expected: build succeeds and the unit executable exits `0`.

- [ ] **Step 5: Commit**

```powershell
git add src/Core/SurfelPathTracerResources.h src/Core/SurfelPathTracerResources.cpp src/shaders/SurfelPathTracerCommon.slang src/shaders/SurfelPathTracerPrepare.slang tests/SurfelPathTracerPipelineTests.cpp
git commit -m "feat: extend surfel path tracer state"
```

### Task 4: Make The Cell Grid Camera-Relative

**Files:**

- Modify: `src/Core/SurfelPathTracerResources.h`
- Modify: `src/Core/SurfelPathTracerResources.cpp`
- Modify: `src/shaders/SurfelPathTracerCommon.slang`
- Modify: `src/shaders/SurfelPathTracerEvaluate.slang`
- Modify: `src/shaders/SurfelPathTracerUpdate.slang`
- Modify: `src/shaders/SurfelPathTracerCellToSurfel.slang`
- Modify: `src/shaders/SurfelPathTracerReflection.slang`
- Modify: `tests/SurfelPathTracerPipelineTests.cpp`

- [ ] **Step 1: Add C++ helper overload**

Keep the existing helper for compatibility and add:

```cpp
static SurfelPathTracerCellAddress cameraRelativeCellAddressForPosition(
    const glm::vec3 &position,
    const glm::vec3 &cameraPosition,
    float cellSize,
    uint32_t cellDimension)
{
    return cellAddressForPosition(position - cameraPosition, cellSize, cellDimension);
}
```

- [ ] **Step 2: Add Slang helpers**

In `SurfelPathTracerCommon.slang`, add:

```hlsl
int3 cameraRelativeCellCoord(float3 position, float3 cameraPosition, float cellSize)
{
    return int3(round((position - cameraPosition) / max(cellSize, 0.0001)));
}

bool isCellCoordValid(int3 coord, uint cellDimension)
{
    int halfDim = int(max(cellDimension, 1u)) / 2;
    return abs(coord.x) < halfDim && abs(coord.y) < halfDim && abs(coord.z) < halfDim;
}

uint flattenCameraRelativeCellCoord(int3 coord, uint cellDimension)
{
    uint dim = max(cellDimension, 1u);
    int halfDim = int(dim) / 2;
    uint3 unsignedCoord = uint3(coord + int3(halfDim, halfDim, halfDim));
    return unsignedCoord.x + unsignedCoord.y * dim + unsignedCoord.z * dim * dim;
}

uint cameraRelativeCellIndexForPosition(float3 position, float3 cameraPosition, float cellSize, uint cellDimension)
{
    int3 coord = cameraRelativeCellCoord(position, cameraPosition, cellSize);
    if (!isCellCoordValid(coord, cellDimension))
    {
        return SURFEL_PT_INVALID_INDEX;
    }
    return flattenCameraRelativeCellCoord(coord, cellDimension);
}
```

- [ ] **Step 3: Replace fixed-origin cell lookups**

In Evaluate, Update, CellToSurfel, and Reflection, replace local `cellIndexForPosition(position)` helpers with calls to:

```hlsl
uint cellIndex = cameraRelativeCellIndexForPosition(position, ubo.cameraPos.xyz, push.cellSize, push.cellDimension);
if (cellIndex == SURFEL_PT_INVALID_INDEX) { return; }
```

For functions that cannot return directly, return zero coverage/radiance when invalid.

- [ ] **Step 4: Update the CPU test from Task 1**

Change the test to use `cameraRelativeCellAddressForPosition(position, camera, ...)` and assert that a point near `cameraB` lands near the grid center:

```cpp
const auto atB = Laphria::SurfelPathTracerResources::cameraRelativeCellAddressForPosition(position, cameraB, 1.0f, 64);
if (std::abs(atB.coord.x - 33) > 1) { return false; }
```

Add Task 4-specific contract needles:

```cpp
"cameraRelativeCellAddressForPosition",
"cameraRelativeCellIndexForPosition",
"cameraRelativeCellCoord",
"flattenCameraRelativeCellCoord",
```

- [ ] **Step 5: Verify and commit**

Run unit build, unit executable, and editor build. Commit:

```powershell
git add src/Core/SurfelPathTracerResources.h src/Core/SurfelPathTracerResources.cpp src/shaders/SurfelPathTracerCommon.slang src/shaders/SurfelPathTracerEvaluate.slang src/shaders/SurfelPathTracerUpdate.slang src/shaders/SurfelPathTracerCellToSurfel.slang src/shaders/SurfelPathTracerReflection.slang tests/SurfelPathTracerPipelineTests.cpp
git commit -m "feat: make surfel cells camera relative"
```

### Task 5: Insert Surfels Into Every Intersecting Neighbor Cell

**Files:**

- Modify: `src/shaders/SurfelPathTracerCommon.slang`
- Modify: `src/shaders/SurfelPathTracerUpdate.slang`
- Modify: `src/shaders/SurfelPathTracerCellToSurfel.slang`
- Modify: `src/Core/SurfelPathTracerPasses.h`
- Modify: `src/Core/SurfelPathTracerPasses.cpp`
- Modify: `src/Core/EngineCore.cpp`
- Modify: `tests/SurfelPathTracerPipelineTests.cpp`

- [ ] **Step 1: Add cell intersection helper**

In common:

```hlsl
bool isSurfelIntersectCell(SurfelPathTracerSurfel surfel, int3 cellCoord, float3 cameraPosition, float cellSize)
{
    if (surfel.flags == 0u || surfel.radius <= 0.0 || !isFinite3(surfel.position))
    {
        return false;
    }

    float3 cellCenter = float3(cellCoord) * max(cellSize, 0.0001) + cameraPosition;
    float3 halfExtent = float3(max(cellSize, 0.0001) * 0.5);
    float3 closest = clamp(surfel.position, cellCenter - halfExtent, cellCenter + halfExtent);
    return length(closest - surfel.position) <= surfel.radius;
}
```

- [ ] **Step 2: Update `SurfelPathTracerUpdate.slang` occupancy counts**

Replace single-cell count with:

```hlsl
int3 centerCoord = cameraRelativeCellCoord(surfel.position, ubo.cameraPos.xyz, push.cellSize);
for (int z = -1; z <= 1; ++z)
for (int y = -1; y <= 1; ++y)
for (int x = -1; x <= 1; ++x)
{
    int3 cellCoord = centerCoord + int3(x, y, z);
    if (!isCellCoordValid(cellCoord, push.cellDimension) ||
        !isSurfelIntersectCell(surfel, cellCoord, ubo.cameraPos.xyz, push.cellSize))
    {
        continue;
    }
    uint cellIndex = flattenCameraRelativeCellCoord(cellCoord, push.cellDimension);
    uint cellCountOffset = (1u + cellIndex) * 4u;
    cellCounterBuffer.InterlockedAdd(cellCountOffset, 1u);
}
```

Add `[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;` to Update and adjust `recordUpdatePass` descriptor bindings so set `1` is bound alongside storage set `0`.

This binding change is required, not optional. Update the C++ signature:

```cpp
void recordUpdatePass(const vk::raii::CommandBuffer &commandBuffer,
                      const SurfelPathTracerPipelines &pipelines,
                      vk::DescriptorSet imageSet,
                      vk::DescriptorSet globalSet,
                      uint32_t maxSurfels,
                      uint32_t maxRays,
                      float cellSize,
                      uint32_t cellDimension) const;
```

Inside `recordUpdatePass`, bind both descriptor sets:

```cpp
std::array descriptorSets = {imageSet, globalSet};
commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                 *pipelines.computePipelineLayout,
                                 0,
                                 descriptorSets,
                                 {});
```

Update the `EngineCore.cpp` call site to pass `*descriptorSets[fi]`.

- [ ] **Step 3: Update `SurfelPathTracerCellToSurfel.slang` population**

Use the same neighbor loop and write into `cellToSurfelBuffer` for each intersecting cell. Increment `COUNTER_REJECTED_STORES_OFFSET` if a cell overflows.

- [ ] **Step 4: Add contract needles**

Add:

```cpp
"for (int z = -1; z <= 1; ++z)",
"isSurfelIntersectCell(surfel",
"flattenCameraRelativeCellCoord",
```

- [ ] **Step 5: Verify and commit**

Run unit build, unit executable, editor build. Commit:

```powershell
git add src/shaders/SurfelPathTracerCommon.slang src/shaders/SurfelPathTracerUpdate.slang src/shaders/SurfelPathTracerCellToSurfel.slang src/Core/SurfelPathTracerPasses.cpp src/Core/SurfelPathTracerPasses.h src/Core/EngineCore.cpp tests/SurfelPathTracerPipelineTests.cpp
git commit -m "feat: insert surfels into intersecting cells"
```

### Task 6: Add Lifecycle, Sleep, And Recycling

**Files:**

- Modify: `src/shaders/SurfelPathTracerUpdate.slang`
- Modify: `src/shaders/SurfelPathTracerEvaluate.slang`
- Modify: `src/shaders/SurfelPathTracerCommon.slang`
- Modify: `src/Core/SurfelPathTracerPasses.h`
- Modify: `src/Core/SurfelPathTracerPasses.cpp`
- Modify: `src/Core/EngineCore.cpp`
- Modify: `tests/SurfelPathTracerPipelineTests.cpp`

- [ ] **Step 1: Add lifecycle helpers**

In common:

```hlsl
bool shouldRecycleSurfel(SurfelPathTracerSurfel surfel, uint frameIndex, float distanceToCamera, float aliveRatio)
{
    if (surfel.flags == 0u)
    {
        return false;
    }
    if (surfel.radius <= 0.0 || !isFinite3(surfel.position))
    {
        return true;
    }

    uint ageSinceSeen = frameIndex >= surfel.lastSeenFrame ? frameIndex - surfel.lastSeenFrame : 0u;
    uint ageSinceRef = frameIndex >= surfel.lastReferencedFrame ? frameIndex - surfel.lastReferencedFrame : 0u;
    bool stale = ageSinceSeen > 240u && ageSinceRef > 120u;
    bool distant = distanceToCamera > 160.0;
    bool pressure = aliveRatio > 0.85;
    return stale && (distant || pressure);
}

float surfelRadius(float distanceToCamera, float targetArea, float minRadius, float maxRadius)
{
    float radius = max(distanceToCamera * sqrt(max(targetArea, 1.0)) * 0.0007, minRadius);
    return min(radius, maxRadius);
}
```

- [ ] **Step 2: Recycle in update**

Do not compact `aliveBuffer` in this task. The current update path scans `0..maxSurfels`, so the surfel `flags` field is the authoritative liveness source. To avoid corrupting the append-only alive stats/free-list, reinterpret the counters as follows before editing shader code:

- `aliveSurfels` is the number of successful spawns since the last persistent reset and is clamped to capacity for UI display.
- `deadSurfels` is the current free-list count.
- Active surfel iteration must continue to scan `surfelBuffer[0..maxSurfels)`.
- Recycled surfels are made inactive with `flags = 0` and pushed to the dead free-list with a compare-exchange loop.
- Do not decrement `aliveSurfels` until a separate compacted alive-list pass exists.
- Update is the only pass allowed to return indices to `deadBuffer`. Evaluate may allocate from the dead list, so it must not also push newly removed surfels into that list in the same dispatch.

In Update, before cell counting:

```hlsl
bool pushDeadSurfel(uint surfelIndex, uint maxSurfels)
{
    uint deadCount = counters.Load(SURFEL_PT_COUNTER_DEAD_SURFELS_OFFSET);
    uint previousDeadCount = deadCount;
    for (uint attempt = 0u; attempt < 16u; ++attempt)
    {
        if (deadCount >= maxSurfels)
        {
            return false;
        }

        uint desiredDeadCount = deadCount + 1u;
        counters.InterlockedCompareExchange(SURFEL_PT_COUNTER_DEAD_SURFELS_OFFSET,
                                            deadCount,
                                            desiredDeadCount,
                                            previousDeadCount);
        if (previousDeadCount == deadCount)
        {
            deadBuffer[deadCount] = surfelIndex;
            return true;
        }
        deadCount = previousDeadCount;
    }
    return false;
}

uint frameIndex = counters.Load(SURFEL_PT_COUNTER_FRAME_INDEX_OFFSET);
uint deadCount = counters.Load(SURFEL_PT_COUNTER_DEAD_SURFELS_OFFSET);
float aliveRatio = 1.0 - saturate(float(deadCount) / float(max(push.maxSurfels, 1u)));
float distanceToCamera = length(surfel.position - ubo.cameraPos.xyz);
if (!push.lockSurfels && shouldRecycleSurfel(surfel, frameIndex, distanceToCamera, aliveRatio))
{
    surfel.flags = 0u;
    surfel.radius = 0.0;
    surfelBuffer[surfelIndex] = surfel;
    if (pushDeadSurfel(surfelIndex, push.maxSurfels))
    {
        counters.InterlockedAdd(SURFEL_PT_COUNTER_RECYCLED_SURFELS_OFFSET, 1u);
    }
    return;
}
```

Add `lockSurfels` to Update push constants so the debug freeze does not recycle. Update `recordUpdatePass(...)` in `SurfelPathTracerPasses.{h,cpp}` and its `EngineCore.cpp` call site to pass `surfelSettings.lockSurfels`.

- [ ] **Step 3: Mark last-seen in evaluate**

When a surfel contributes to coverage, set:

```hlsl
surfel.lastReferencedFrame = counters.Load(SURFEL_PT_COUNTER_FRAME_INDEX_OFFSET);
surfelBuffer[surfelIndex] = surfel;
```

When it is the chosen closest surfel for screen coverage, set `lastSeenFrame` as well.

- [ ] **Step 4: Verify and commit**

Run unit build, unit executable, editor build. Commit:

```powershell
git add src/shaders/SurfelPathTracerUpdate.slang src/shaders/SurfelPathTracerEvaluate.slang src/shaders/SurfelPathTracerCommon.slang src/Core/SurfelPathTracerPasses.cpp src/Core/SurfelPathTracerPasses.h src/Core/EngineCore.cpp tests/SurfelPathTracerPipelineTests.cpp
git commit -m "feat: add surfel lifecycle recycling"
```

### Task 7: Implement Coverage-Based Placement And Removal

**Files:**

- Modify: `src/shaders/SurfelPathTracerEvaluate.slang`
- Modify: `src/shaders/SurfelPathTracerUpdate.slang`
- Modify: `src/shaders/SurfelPathTracerCommon.slang`
- Modify: `src/Core/SurfelPathTracerPasses.h`
- Modify: `src/Core/SurfelPathTracerPasses.cpp`
- Modify: `src/Core/EngineCore.cpp`
- Modify: `tests/SurfelPathTracerPipelineTests.cpp`

- [ ] **Step 0: Add pending-free surfel flag**

In `SurfelPathTracerCommon.slang`, replace raw flag literals with named bits:

```hlsl
static const uint SURFEL_PT_SURFEL_FLAG_ACTIVE = 1u;
static const uint SURFEL_PT_SURFEL_FLAG_PENDING_FREE = 2u;

bool isActiveSurfel(SurfelPathTracerSurfel surfel)
{
    return (surfel.flags & SURFEL_PT_SURFEL_FLAG_ACTIVE) != 0u &&
           (surfel.flags & SURFEL_PT_SURFEL_FLAG_PENDING_FREE) == 0u &&
           surfel.radius > 0.0 &&
           isFinite3(surfel.position);
}
```

Use `isActiveSurfel(...)` in Update, Evaluate, CellToSurfel, Raygen, Integrate, and Reflection instead of checking only `surfel.flags == 0u`. New surfels set:

```hlsl
surfel.flags = SURFEL_PT_SURFEL_FLAG_ACTIVE;
```

In Update, before the active-surfel early-out, consume pending frees:

```hlsl
if ((surfel.flags & SURFEL_PT_SURFEL_FLAG_PENDING_FREE) != 0u)
{
    surfel.flags = 0u;
    surfel.radius = 0.0;
    surfelBuffer[surfelIndex] = surfel;
    if (pushDeadSurfel(surfelIndex, push.maxSurfels))
    {
        counters.InterlockedAdd(SURFEL_PT_COUNTER_REMOVED_SURFELS_OFFSET, 1u);
    }
    return;
}
```

This keeps free-list mutation phase-separated: Evaluate can allocate and mark removals, while Update is the only pass that appends returned indices to `deadBuffer`.

- [ ] **Step 1: Add push constants**

Extend Evaluate push constants:

```hlsl
float placementThreshold;
float removalThreshold;
float surfelTargetArea;
float surfelMinRadius;
uint frameIndex;
uint lockSurfels;
uint enablePlacement;
uint enableRemoval;
```

Wire these from `recordEvaluatePass`.

Update `recordEvaluatePass(...)` in `SurfelPathTracerPasses.{h,cpp}` and its `EngineCore.cpp` call sites to pass `surfelSettings.placementThreshold`, `surfelSettings.removalThreshold`, `surfelSettings.surfelTargetArea`, `surfelSettings.surfelMinRadius`, `surfelSettings.lockSurfels`, `surfelSettings.enableSurfelPlacement`, and `surfelSettings.enableSurfelRemoval`. Generate mode uses these controls for spawn/removal; Resolve mode still uses the same thresholds only for debug/coverage calculations and must not allocate or remove surfels.

- [ ] **Step 2: Initialize new surfels with current indirect estimate**

In `allocateSurfel`, change:

```hlsl
surfel.radiance = seedRadiance;
surfel.meanAndVariance = float4(seedRadiance, 1.0);
surfel.shortMeanAndLife = float4(seedRadiance, 1.0);
surfel.varianceAndInconsistency = float4(1.0, 1.0, 1.0, 1.0);
surfel.lastSeenFrame = push.frameIndex;
surfel.lastReferencedFrame = push.frameIndex;
surfel.sleepState = SURFEL_PT_SLEEP_AWAKE;
```

Change the signature to:

```hlsl
void allocateSurfel(uint2 pixel, float3 position, float3 normal, float radius, float3 seedRadiance, uint materialKey)
```

- [ ] **Step 3: Use placement/removal thresholds**

In Generate mode:

```hlsl
if (push.lockSurfels == 0u && push.enablePlacement != 0u && coverage < push.placementThreshold)
{
    float radius = surfelRadius(depth, push.surfelTargetArea, push.surfelMinRadius, push.cellSize * 2.0);
    allocateSurfel(pixel, position, normal, radius, resolveSurfelRadiance(pixel, depth, normal), 0u);
}

if (push.lockSurfels == 0u && push.enableRemoval != 0u &&
    coverage > push.removalThreshold && closestSurfelIndex != SURFEL_PT_INVALID_INDEX)
{
    SurfelPathTracerSurfel overCovered = surfelBuffer[closestSurfelIndex];
    overCovered.flags |= SURFEL_PT_SURFEL_FLAG_PENDING_FREE;
    surfelBuffer[closestSurfelIndex] = overCovered;
}
```

Do not add `pushDeadSurfel` to Evaluate. Marking `SURFEL_PT_SURFEL_FLAG_PENDING_FREE` defers the free-list append to the next Update pass and avoids racing Evaluate allocation against Evaluate removal.

- [ ] **Step 4: Verify and commit**

Run unit build, unit executable, editor build. Commit:

```powershell
git add src/shaders/SurfelPathTracerEvaluate.slang src/shaders/SurfelPathTracerUpdate.slang src/shaders/SurfelPathTracerCommon.slang src/Core/SurfelPathTracerPasses.cpp src/Core/SurfelPathTracerPasses.h src/Core/EngineCore.cpp tests/SurfelPathTracerPipelineTests.cpp
git commit -m "feat: adapt surfel placement and removal"
```

### Task 8: Add 6x6 Directional Irradiance And Depth Atlas Mapping

**Files:**

- Modify: `src/Core/SurfelPathTracerResources.cpp`
- Modify: `src/shaders/SurfelPathTracerCommon.slang`
- Modify: `src/shaders/SurfelPathTracerIntegrate.slang`
- Modify: `tests/SurfelPathTracerPipelineTests.cpp`

- [ ] **Step 1: Add atlas helpers**

In common:

```hlsl
uint2 atlasTileBase(uint surfelIndex, uint atlasWidth)
{
    uint tilesPerRow = max(atlasWidth / SURFEL_PT_ATLAS_TILE_SIZE, 1u);
    uint tileX = surfelIndex % tilesPerRow;
    uint tileY = surfelIndex / tilesPerRow;
    return uint2(tileX, tileY) * SURFEL_PT_ATLAS_TILE_SIZE;
}

uint2 atlasDirectionCoord(float3 localDirection)
{
    float2 uv = dirToOctUv(localDirection);
    int2 offset = int2(round(clamp(uv, float2(-1.0), float2(1.0)) * 2.5 + float2(2.5)));
    return uint2(clamp(offset, int2(0), int2(5)));
}
```

If `dirToOctUv` does not exist yet, add the octahedral direction encoding/decoding helpers beside `packNormalOctahedral`.

- [ ] **Step 2: Update resource sizing**

In `createExtentImages`, validate that `irradianceAtlasWidth` and `irradianceAtlasHeight` can contain `maxSurfels * 36` texels:

```cpp
const uint64_t tileCount = static_cast<uint64_t>(settings_.maxSurfels);
const uint64_t tilesPerRow = std::max<uint64_t>(settings_.irradianceAtlasWidth / 6u, 1u);
const uint64_t requiredRows = (tileCount + tilesPerRow - 1u) / tilesPerRow;
const uint64_t requiredHeight = requiredRows * 6u;
if (requiredHeight > settings_.irradianceAtlasHeight)
{
    throw std::runtime_error("SurfelPathTracer irradiance atlas is too small for maxSurfels");
}
```

- [ ] **Step 3: Write directional atlas entries in integrate**

Replace one-texel `writeAtlas` with:

```hlsl
void writeAtlas(uint surfelIndex, float3 localDirection, float3 radiance, float depth)
{
    uint width = 0u;
    uint height = 0u;
    irradianceAtlas.GetDimensions(width, height);
    if (width == 0u || height == 0u)
    {
        return;
    }

    uint2 coord = atlasTileBase(surfelIndex, width) + atlasDirectionCoord(localDirection);
    if (coord.x >= width || coord.y >= height)
    {
        return;
    }

    float oldLuma = irradianceAtlas[coord].a;
    float newLuma = luminance(radiance);
    float blendedLuma = lerp(oldLuma, newLuma, 0.2);
    irradianceAtlas[coord] = float4(radiance, max(blendedLuma, 1e-5));
    surfelDepthAtlas[coord] = depth;
}
```

Store ray local direction in the ray record, or reconstruct it from the sampled world direction and surfel tangent frame. Use unused record slot `base + 7u` for packed local direction until a typed ray struct is introduced.

- [ ] **Step 4: Verify and commit**

Run unit build, unit executable, editor build. Commit:

```powershell
git add src/Core/SurfelPathTracerResources.cpp src/shaders/SurfelPathTracerCommon.slang src/shaders/SurfelPathTracerIntegrate.slang tests/SurfelPathTracerPipelineTests.cpp
git commit -m "feat: add surfel directional atlas mapping"
```

### Task 9: Add Adaptive Ray Allocation

**Files:**

- Modify: `src/shaders/SurfelPathTracerCommon.slang`
- Modify: `src/shaders/SurfelPathTracerUpdate.slang`
- Modify: `src/Core/SurfelPathTracerPasses.h`
- Modify: `src/Core/SurfelPathTracerPasses.cpp`
- Modify: `src/Core/EngineCore.cpp`
- Modify: `tests/SurfelPathTracerPipelineTests.cpp`

- [ ] **Step 1: Add helper**

In common:

```hlsl
uint adaptiveRayCountForSurfel(SurfelPathTracerSurfel surfel,
                               uint minRays,
                               uint maxRays,
                               float varianceSensitivity,
                               uint currentFrame)
{
    float variance = length(max(surfel.varianceAndInconsistency.xyz, float3(0.0)));
    float ageBoost = currentFrame - surfel.lastSeenFrame < 20u ? 1.0 : 0.0;
    float varianceT = saturate(variance * varianceSensitivity + ageBoost);
    uint count = uint(round(lerp(float(minRays), float(maxRays), varianceT)));
    if (surfel.sleepState == SURFEL_PT_SLEEP_SLEEPING)
    {
        count = max(minRays, count / 4u);
    }
    return clamp(count, minRays, maxRays);
}
```

- [ ] **Step 2: Extend Update push constants**

Add `minRaysPerSurfel`, `maxRaysPerSurfel`, and `varianceSensitivity` to the C++ and Slang Update push constants.

Update `recordUpdatePass(...)` and the `EngineCore.cpp` call site to pass `surfelSettings.minRaysPerSurfel`, `surfelSettings.maxRaysPerSurfel`, and `surfelSettings.varianceSensitivity`.

- [ ] **Step 3: Replace fixed `surfel.rayCount` allocation**

In Update:

```hlsl
uint frameIndex = counters.Load(SURFEL_PT_COUNTER_FRAME_INDEX_OFFSET);
uint rayRequestCount = adaptiveRayCountForSurfel(surfel,
                                                 push.minRaysPerSurfel,
                                                 push.maxRaysPerSurfel,
                                                 push.varianceSensitivity,
                                                 frameIndex);
```

Allocate `rayRequestCount`, not the previous `surfel.rayCount`.

- [ ] **Step 4: Verify and commit**

Run unit build, unit executable, editor build. Commit:

```powershell
git add src/shaders/SurfelPathTracerCommon.slang src/shaders/SurfelPathTracerUpdate.slang src/Core/SurfelPathTracerPasses.h src/Core/SurfelPathTracerPasses.cpp src/Core/EngineCore.cpp tests/SurfelPathTracerPipelineTests.cpp
git commit -m "feat: allocate surfel rays adaptively"
```

### Task 9.5: Shadow Primary Direct Lighting

**Files:**

- Modify: `src/shaders/SurfelPathTracerGBuffer.slang`
- Modify: `src/shaders/SurfelPathTracerGBufferMiss.slang`
- Modify: `src/shaders/SurfelPathTracerGBufferAnyHit.slang`
- Modify: `src/shaders/SurfelPathTracerLightIntegrate.slang`
- Modify: `tests/SurfelPathTracerPipelineTests.cpp`

- [ ] **Step 1: Add a failing contract test for direct sun visibility**

Add needles that require the GBuffer ray-generation shader to write a primary-surface sun visibility term into `gBufferNormal.w`, and require `SurfelPathTracerLightIntegrate.slang` to multiply the sun term by that visibility:

```cpp
bool primarySunVisibilityOk =
    containsAllNeedles(readTextFile(root / "src" / "shaders" / "SurfelPathTracerGBuffer.slang", filesOk),
                       {"SurfelPathTracerGBufferPayload makeEmptyGBufferPayload()",
                        "float traceSunVisibility(float3 position, float3 normal, float3 sunDir)",
                        "TraceRay(tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH",
                        "float sunVisibility = directSun > 0.0 ? traceSunVisibility(hitPos, normal, sunDir) : 0.0",
                        "gBufferNormal[launchID] = float4(normal, sunVisibility)"}) &&
    containsAllNeedles(readTextFile(root / "src" / "shaders" / "SurfelPathTracerLightIntegrate.slang", filesOk),
                       {"float sunVisibility = saturate(gBufferNormal[pixel].w)",
                        "SUN_RADIANCE * directSun * sunVisibility + skyAmbient"}) &&
    containsAllNeedles(readTextFile(root / "src" / "shaders" / "SurfelPathTracerGBufferMiss.slang", filesOk),
                       {"float3 baseColor",
                        "payload.baseColor = float3(0.0, 0.0, 0.0)"}) &&
    containsAllNeedles(readTextFile(root / "src" / "shaders" / "SurfelPathTracerGBufferAnyHit.slang", filesOk),
                       {"float3 baseColor"});
```

Include `primarySunVisibilityOk` in the final contract return value.

- [ ] **Step 2: Verify the test fails**

Run:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests"
.\cmake-build-debug\LaphriaEngineUnitTests.exe
```

Expected: unit executable exits non-zero and reports the missing sun-visibility contract needles.

- [ ] **Step 3: Trace primary sun visibility in GBuffer raygen**

Add a helper that traces from the primary hit point toward the sun using the existing GBuffer hit group and alpha any-hit handling:

```hlsl
SurfelPathTracerGBufferPayload makeEmptyGBufferPayload()
{
    SurfelPathTracerGBufferPayload payload;
    payload.hitT = -1.0;
    payload.modelId = 0u;
    payload.materialIndex = 0u;
    payload.worldNormal = float3(0.0, 0.0, 0.0);
    payload.baseColor = float3(0.0, 0.0, 0.0);
    return payload;
}

float traceSunVisibility(float3 position, float3 normal, float3 sunDir)
{
    RayDesc shadowRay;
    float bias = max(SURFEL_PT_RAY_BIAS * 4.0, 0.005);
    shadowRay.Origin = position + normal * bias;
    shadowRay.Direction = normalize(sunDir);
    shadowRay.TMin = bias;
    shadowRay.TMax = 10000.0;

    SurfelPathTracerGBufferPayload shadowPayload = makeEmptyGBufferPayload();
    TraceRay(tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, 0, 0, 0, shadowRay, shadowPayload);
    return shadowPayload.hitT < 0.0 ? 1.0 : 0.0;
}
```

In the primary hit path, compute:

```hlsl
float3 sunDir = normalize(-ubo.lightDir.xyz);
float directSun = max(dot(normal, sunDir), 0.0);
float sunVisibility = directSun > 0.0 ? traceSunVisibility(hitPos, normal, sunDir) : 0.0;
gBufferNormal[launchID] = float4(normal, sunVisibility);
```

Keep miss pixels at `gBufferNormal.w = 0.0`.

- [ ] **Step 4: Keep GBuffer payload layouts identical**

Add `float3 baseColor;` to `SurfelPathTracerGBufferMiss.slang` and `SurfelPathTracerGBufferAnyHit.slang`. The miss shader must initialize it to black:

```hlsl
payload.baseColor = float3(0.0, 0.0, 0.0);
```

This avoids payload ABI drift between raygen, closest-hit, miss, and any-hit shaders.

- [ ] **Step 5: Apply visibility in light integration**

In `SurfelPathTracerLightIntegrate.slang`, read:

```hlsl
float sunVisibility = saturate(gBufferNormal[pixel].w);
```

Then compose direct light as:

```hlsl
float3 directLighting = SUN_RADIANCE * directSun * sunVisibility + skyAmbient;
```

Sky ambient remains unshadowed in this first slice. The reusable path transport in Task 11 should introduce a shared shadowed direct-light helper for surfel rays, reflections, and the reference path.

- [ ] **Step 6: Verify and commit**

Run:

```powershell
git diff --check
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests"
.\cmake-build-debug\LaphriaEngineUnitTests.exe
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build cmake-build-debug --config Debug --target LaphriaEditor"
```

Expected: diff check returns `0`, unit target builds, unit executable exits `0`, and editor target builds. Manual expected result: Sponza interior no longer receives full sun everywhere in Final Color; GBuffer Albedo remains unchanged.

```powershell
git add docs/superpowers/plans/2026-05-18-surfel-path-tracer-surfelplus-completion.md src/shaders/SurfelPathTracerGBuffer.slang src/shaders/SurfelPathTracerGBufferMiss.slang src/shaders/SurfelPathTracerGBufferAnyHit.slang src/shaders/SurfelPathTracerLightIntegrate.slang tests/SurfelPathTracerPipelineTests.cpp
git commit -m "fix: shadow surfel path tracer direct lighting"
```

### Task 10: Implement Guided Surfel Ray Sampling

**Files:**

- Modify: `src/shaders/SurfelPathTracerCommon.slang`
- Modify: `src/shaders/SurfelPathTracerRaygen.slang`
- Modify: `src/Core/SurfelPathTracerPasses.h`
- Modify: `src/Core/SurfelPathTracerPasses.cpp`
- Modify: `src/Core/EngineCore.cpp`
- Modify: `tests/SurfelPathTracerPipelineTests.cpp`

- [ ] **Step 1: Add guided sampling helper**

In `SurfelPathTracerRaygen.slang`, add the resource-bound atlas helper locally:

```hlsl
float luminanceAtAtlasCoord(uint2 coord, RWTexture2D<float4> atlas)
{
    return max(atlas[coord].a, 0.0);
}
```

Do not move resource-bound helpers into `SurfelPathTracerCommon.slang`; keep common limited to pure math helpers such as `atlasTileBase`, `atlasDirectionCoord`, `octUvToDir`, and tangent-frame construction. Then add:

```hlsl
float3 cosineSampleHemisphereLocal(float2 xi)
{
    float phi = 2.0 * PI * xi.x;
    float cosTheta = sqrt(max(1.0 - xi.y, 0.0));
    float sinTheta = sqrt(max(xi.y, 0.0));
    return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

bool sampleGuidedSurfelDirection(uint surfelIndex,
                                 float2 cdfXi,
                                 float2 jitterXi,
                                 RWTexture2D<float4> atlas,
                                 uint atlasWidth,
                                 out float3 localDirection,
                                 out float pdf)
{
    uint2 base = atlasTileBase(surfelIndex, atlasWidth);
    float total = 0.0;
    for (uint y = 0u; y < SURFEL_PT_ATLAS_TILE_SIZE; ++y)
    for (uint x = 0u; x < SURFEL_PT_ATLAS_TILE_SIZE; ++x)
    {
        total += max(atlas[base + uint2(x, y)].a, 0.0);
    }

    if (total <= 1e-5)
    {
        localDirection = float3(0.0, 0.0, 1.0);
        pdf = 0.0;
        return false;
    }

    float threshold = cdfXi.x * total;
    float cumulative = 0.0;
    for (uint y = 0u; y < SURFEL_PT_ATLAS_TILE_SIZE; ++y)
    for (uint x = 0u; x < SURFEL_PT_ATLAS_TILE_SIZE; ++x)
    {
        float weight = max(atlas[base + uint2(x, y)].a, 0.0);
        cumulative += weight;
        if (cumulative >= threshold)
        {
            float2 texelJitter = jitterXi - 0.5;
            float2 texelCenter = float2(x, y) + 0.5 + texelJitter;
            float2 oct = clamp(texelCenter / float(SURFEL_PT_ATLAS_TILE_SIZE) * 2.0 - 1.0,
                               float2(-1.0),
                               float2(1.0));
            localDirection = octUvToDir(oct);
            pdf = max(weight / total, 1e-5);
            return true;
        }
    }

    localDirection = float3(0.0, 0.0, 1.0);
    pdf = 1.0;
    return true;
}
```

- [ ] **Step 2: Use guided sampling in Raygen**

Bind `irradianceAtlas` at the existing storage descriptor binding and extend the surfel ray trace push constants in both Slang and C++:

```hlsl
uint enableGuidedSampling;
uint irradianceAtlasWidth;
```

Update `recordSurfelRayTracePass(...)` to accept `bool enableGuidedSampling` and the atlas width, then pass `surfelSettings.enableGuidedSampling` and `surfelSettings.irradianceAtlasWidth` from `EngineCore.cpp`. In Raygen:

```hlsl
float2 guideXi = float2(randomFloat(seed), randomFloat(seed));
float2 guideJitter = float2(randomFloat(seed), randomFloat(seed));
bool guided = push.enableGuidedSampling != 0u &&
              surfel.rayCount > 16u &&
              sampleGuidedSurfelDirection(surfelIndex, guideXi, guideJitter, irradianceAtlas, push.irradianceAtlasWidth, localDir, pdf);
if (!guided)
{
    localDir = cosineSampleHemisphereLocal(xi);
    pdf = max(localDir.z / PI, 1e-5);
    counters.InterlockedAdd(SURFEL_PT_COUNTER_COSINE_RAYS_OFFSET, 1u);
}
else
{
    counters.InterlockedAdd(SURFEL_PT_COUNTER_GUIDED_RAYS_OFFSET, 1u);
}
```

Convert `localDir` to world using the surfel normal tangent basis.

- [ ] **Step 3: Store local direction in ray records**

Use record slot `base + 7u`:

```hlsl
rayBuffer[base + 7u] = packNormalOctahedral(localDirection);
```

- [ ] **Step 4: Verify and commit**

Run unit build, unit executable, editor build. Commit:

```powershell
git add src/shaders/SurfelPathTracerCommon.slang src/shaders/SurfelPathTracerRaygen.slang src/Core/SurfelPathTracerPasses.h src/Core/SurfelPathTracerPasses.cpp src/Core/EngineCore.cpp tests/SurfelPathTracerPipelineTests.cpp
git commit -m "feat: guide surfel ray sampling"
```

### Task 11: Add Reusable Real-Time Path Transport And Surfel Termination

**Files:**

- Modify: `src/shaders/SurfelPathTracerCommon.slang`
- Modify: `src/shaders/SurfelPathTracerRaygen.slang`
- Modify: `src/shaders/SurfelPathTracerClosestHit.slang`
- Modify: `src/shaders/SurfelPathTracerAnyHit.slang`
- Modify: `src/shaders/SurfelPathTracerMiss.slang`
- Modify: `src/shaders/SurfelPathTracerReflection.slang`
- Modify: `src/Core/SurfelPathTracerPasses.h`
- Modify: `src/Core/SurfelPathTracerPasses.cpp`
- Modify: `src/Core/EngineCore.cpp`
- Modify: `tests/SurfelPathTracerPipelineTests.cpp`

- [ ] **Step 1: Change the ray payload contract**

Replace direct-radiance-only payload with a surface-aware payload. This is required for real path transport; the raygen loop cannot choose a bounce direction or apply a BSDF from `radiance/hitT` alone.

```hlsl
struct SurfelPathTracerPayload {
    float3 radiance;
    float hitT;
    float3 hitPosition;
    float3 hitNormal;
    float3 baseColor;
    float metallic;
    float roughness;
    uint hitKind;
};
```

Add `makeEmptySurfelPayload()` in each raygen shader that initializes `hitT = -1.0`, `hitKind = 0u`, zero radiance, a safe up normal, black base color, and roughness `1.0`. Closest-hit sets `hitKind = 1u`, fills surface data, and may fill local/direct/emissive radiance. Miss fills sky radiance and leaves `hitKind = 0u`. Any-hit must declare the same payload layout even if it only performs alpha rejection.

- [ ] **Step 1.5: Extend pass bindings and push constants**

`traceSurfelPath` needs cell data, surfel termination settings, and active/sleeping depth settings. Extend `SurfelRayTracePushConstants` in Slang and C++ with:

```hlsl
uint activeMaxDepth;
uint sleepingMaxDepth;
uint enableSurfelTermination;
uint maxSurfelSamplesPerQuery;
uint cellDimension;
float cellSize;
```

Bind these existing storage resources in `SurfelPathTracerRaygen.slang`:

```hlsl
[[vk::binding(11, 1)]] RWStructuredBuffer<SurfelPathTracerCellInfo> cellInfoBuffer;
[[vk::binding(13, 1)]] RWStructuredBuffer<uint> cellToSurfelBuffer;
```

Update `recordSurfelRayTracePass(...)` and its `EngineCore.cpp` call site to pass `surfelSettings.activeMaxDepth`, `surfelSettings.sleepingMaxDepth`, `surfelSettings.enableSurfelTermination`, `surfelSettings.maxSurfelSamplesPerQuery`, `surfelPathTracerResources.cellDimensionCapacity()`, and `surfelSettings.cellSize`.

Reflection uses the same termination helper at one bounce, so extend `ReflectionPushConstants`, `recordReflectionPass(...)`, and the `EngineCore.cpp` call site with `enableSurfelTermination` and `maxSurfelSamplesPerQuery` as well. The reflection pass already receives `cellSize`, `cellDimension`, `cellInfoBuffer`, and `cellToSurfelBuffer`; verify those bindings remain present after the payload change.

- [ ] **Step 1.6: Factor reusable shadowed direct lighting**

Move direct sun evaluation behind a shared helper that can be copied locally or included once it is resource-independent:

```hlsl
float3 evaluateShadowedDirectLighting(float3 position,
                                      float3 normal,
                                      float3 sunDir,
                                      float sunVisibility)
{
    float directSun = max(dot(normal, sunDir), 0.0);
    float3 skyAmbient = evalSkyColor(normal, sunDir) * 0.25 + evalGroundColor(sunDir) * 0.10;
    return SUN_RADIANCE * directSun * sunVisibility + skyAmbient;
}
```

For Task 11 raygen/reflection/reference paths, compute `sunVisibility` with a shadow ray from the current path hit before applying direct sun. Do not reintroduce an unshadowed `SUN_RADIANCE * directSun` production path.

- [ ] **Step 2: Add surfel termination helper**

In `SurfelPathTracerRaygen.slang` and `SurfelPathTracerReflection.slang`, keep the resource-bound version shader-local:

```hlsl
float3 terminatePathWithSurfels(float3 position,
                                float3 normal,
                                uint maxSamples,
                                uint maxSurfels,
                                float cellSize,
                                uint cellDimension,
                                float3 cameraPosition,
                                RWStructuredBuffer<SurfelPathTracerSurfel> surfels,
                                RWStructuredBuffer<SurfelPathTracerCellInfo> cells,
                                RWStructuredBuffer<uint> cellToSurfels)
{
    uint cellIndex = cameraRelativeCellIndexForPosition(position, cameraPosition, cellSize, cellDimension);
    if (cellIndex == SURFEL_PT_INVALID_INDEX)
    {
        return float3(0.0);
    }

    SurfelPathTracerCellInfo cellInfo = cells[cellIndex];
    uint count = min(cellInfo.surfelCount, maxSamples);
    float3 sum = float3(0.0);
    float weightSum = 0.0;
    for (uint i = 0u; i < count; ++i)
    {
        uint surfelIndex = cellToSurfels[cellInfo.surfelOffset + i];
        if (surfelIndex == SURFEL_PT_INVALID_INDEX || surfelIndex >= maxSurfels)
        {
            continue;
        }

        SurfelPathTracerSurfel surfel = surfels[surfelIndex];
        if (!isActiveSurfel(surfel))
        {
            continue;
        }

        float3 surfelNormal = unpackNormalOctahedral(surfel.packedNormal);
        float distanceWeight = saturate(1.0 - length(position - surfel.position) / max(surfel.radius, 1e-4));
        float normalWeight = saturate(dot(normal, surfelNormal));
        float weight = distanceWeight * normalWeight;
        sum += surfel.radiance * weight;
        weightSum += weight;
    }
    return weightSum > 1e-5 ? sum / weightSum : float3(0.0);
}
```

Only pure math helpers used by `terminatePathWithSurfels`, such as distance/normal weighting and cell coordinate math, belong in `SurfelPathTracerCommon.slang`.

- [ ] **Step 3: Add `traceSurfelPath` loop in Raygen**

Implement a small real-time path loop:

```hlsl
float3 traceSurfelPath(RayDesc initialRay, uint maxDepth, uint surfelIndex, uint seed)
{
    float3 throughput = float3(1.0);
    float3 radiance = float3(0.0);
    RayDesc ray = initialRay;
    for (uint depth = 0u; depth < maxDepth; ++depth)
    {
        SurfelPathTracerPayload payload = makeEmptySurfelPayload();
        TraceRay(tlas, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);
        if (payload.hitKind == 0u)
        {
            radiance += throughput * payload.radiance;
            break;
        }

        radiance += throughput * payload.radiance;

        if (push.enableSurfelTermination != 0u && depth + 1u >= maxDepth)
        {
            radiance += throughput * terminatePathWithSurfels(payload.hitPosition,
                                                              payload.hitNormal,
                                                              push.maxSurfelSamplesPerQuery,
                                                              push.maxSurfels,
                                                              push.cellSize,
                                                              push.cellDimension,
                                                              ubo.cameraPos.xyz,
                                                              surfelBuffer,
                                                              cellInfoBuffer,
                                                              cellToSurfelBuffer);
            counters.InterlockedAdd(SURFEL_PT_COUNTER_SURFEL_TERMINATED_PATHS_OFFSET, 1u);
            break;
        }

        float2 xi = float2(randomFloat(seed), randomFloat(seed));
        float3 nextDir = cosineSampleHemisphere(xi, payload.hitNormal);
        float cosTheta = max(dot(payload.hitNormal, nextDir), 0.0);
        if (cosTheta <= 0.0)
        {
            break;
        }
        throughput *= saturate(payload.baseColor);
        ray.Origin = payload.hitPosition + payload.hitNormal * SURFEL_PT_RAY_BIAS;
        ray.Direction = normalize(nextDir);
    }
    return clampLuminance(radiance, SURFEL_PT_MAX_RADIANCE_LUMINANCE);
}
```

Use active/sleeping max depth based on `surfel.sleepState`.

- [ ] **Step 4: Use the helper in reflections**

Reflection pass should call the same surfel termination helper at one bounce when no direct hit or when max reflection depth is reached.

- [ ] **Step 5: Verify and commit**

Run unit build, unit executable, editor build. Commit:

```powershell
git add src/shaders/SurfelPathTracerCommon.slang src/shaders/SurfelPathTracerRaygen.slang src/shaders/SurfelPathTracerClosestHit.slang src/shaders/SurfelPathTracerAnyHit.slang src/shaders/SurfelPathTracerMiss.slang src/shaders/SurfelPathTracerReflection.slang src/Core/SurfelPathTracerPasses.h src/Core/SurfelPathTracerPasses.cpp src/Core/EngineCore.cpp tests/SurfelPathTracerPipelineTests.cpp
git commit -m "feat: add surfel-terminated path transport"
```

### Task 12: Implement MSME Integration And Radiance Sharing

**Files:**

- Modify: `src/shaders/SurfelPathTracerCommon.slang`
- Modify: `src/shaders/SurfelPathTracerIntegrate.slang`
- Modify: `src/Core/SurfelPathTracerPasses.h`
- Modify: `src/Core/SurfelPathTracerPasses.cpp`
- Modify: `src/Core/EngineCore.cpp`
- Modify: `tests/SurfelPathTracerPipelineTests.cpp`

- [ ] **Step 1: Replace simple `msmeBlend`**

Add an MSME helper based on current struct fields:

```hlsl
float3 updateMsme(float3 sampleRadiance, inout SurfelPathTracerSurfel surfel, float shortBlend)
{
    float3 previousMean = surfel.meanAndVariance.xyz;
    float3 previousShort = surfel.shortMeanAndLife.xyz;
    float3 shortMean = lerp(previousShort, sampleRadiance, saturate(shortBlend));
    float3 longMean = lerp(previousMean, sampleRadiance, saturate(shortBlend * 0.25));
    float3 delta = sampleRadiance - longMean;
    float variance = lerp(surfel.meanAndVariance.w, dot(delta, delta), saturate(shortBlend * 0.5));
    surfel.meanAndVariance = float4(longMean, variance);
    surfel.shortMeanAndLife = float4(shortMean, surfel.shortMeanAndLife.w);
    surfel.varianceAndInconsistency = float4(abs(shortMean - longMean), variance);
    surfel.radiance = longMean;
    return longMean;
}
```

- [ ] **Step 2: Apply PDF-weighted ray integration**

In Integrate:

```hlsl
float pdf = max(asfloat(rayBuffer[rayBase + SURFEL_PT_RAY_PDF]), 1e-5);
float3 localDir = unpackNormalOctahedral(rayBuffer[rayBase + 7u]);
float cosine = max(localDir.z, 0.0);
accumulatedRadiance += loadRayRadiance(rayIndex) * cosine / pdf;
```

- [ ] **Step 3: Add radiance sharing**

Extend Integrate push constants in Slang and C++ before using radiance sharing:

```hlsl
uint enableRadianceSharing;
uint maxRadianceSharingSamples;
uint cellDimension;
float cellSize;
```

Bind the global UBO and cell buffers in `SurfelPathTracerIntegrate.slang`:

```hlsl
[[vk::binding(11, 0)]] RWStructuredBuffer<SurfelPathTracerCellInfo> cellInfoBuffer;
[[vk::binding(13, 0)]] RWStructuredBuffer<uint> cellToSurfelBuffer;
[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;
```

Update `recordIntegratePass(...)` to bind both `{imageSet, globalSet}` and pass `surfelSettings.enableRadianceSharing`, `surfelSettings.maxRadianceSharingSamples`, `surfelPathTracerResources.cellDimensionCapacity()`, and `surfelSettings.cellSize` from `EngineCore.cpp`.

Add a shader-local `terminatePathWithSurfels` wrapper to `SurfelPathTracerIntegrate.slang` using the same body from Task 11, because it reads `surfelBuffer`, `cellInfoBuffer`, and `cellToSurfelBuffer`. After direct ray aggregation, if `push.enableRadianceSharing != 0u`, sample up to `push.maxRadianceSharingSamples` surfels from the current cell and blend:

```hlsl
float3 shared = terminatePathWithSurfels(surfel.position,
                                         unpackNormalOctahedral(surfel.packedNormal),
                                         push.maxRadianceSharingSamples,
                                         push.maxSurfels,
                                         push.cellSize,
                                         push.cellDimension,
                                         ubo.cameraPos.xyz,
                                         surfelBuffer,
                                         cellInfoBuffer,
                                         cellToSurfelBuffer);
if (luminance(shared) > 1e-5)
{
    sampleRadiance = lerp(sampleRadiance, shared, 0.25);
}
```

- [ ] **Step 4: Update atlas from the packed local direction**

Call `writeAtlas(surfelIndex, localDir, integratedRadiance, hitDepth)` for each valid ray.

- [ ] **Step 5: Verify and commit**

Run unit build, unit executable, editor build. Commit:

```powershell
git add src/shaders/SurfelPathTracerCommon.slang src/shaders/SurfelPathTracerIntegrate.slang src/Core/SurfelPathTracerPasses.h src/Core/SurfelPathTracerPasses.cpp src/Core/EngineCore.cpp tests/SurfelPathTracerPipelineTests.cpp
git commit -m "feat: integrate surfel radiance with msme"
```

### Task 13: Add RIS Glossy Reflection Sampling

**Files:**

- Modify: `src/shaders/SurfelPathTracerReflection.slang`
- Modify: `src/shaders/SurfelPathTracerCommon.slang`
- Modify: `tests/SurfelPathTracerPipelineTests.cpp`

- [ ] **Step 1: Add reservoir helper**

In common:

```hlsl
float ggxPdf(float3 L, float3 N, float3 V, float roughness)
{
    float3 H = normalize(L + V);
    float nDotH = max(dot(N, H), 0.0);
    float vDotH = max(dot(V, H), 1e-5);
    if (nDotH <= 0.0)
    {
        return 0.0;
    }

    float D = distributionGGX(N, H, max(roughness, MIN_ROUGHNESS));
    return max(D * nDotH / (4.0 * vDotH), 1e-5);
}

float3 ggxSpecularBrdf(float3 L, float3 N, float3 V, float3 f0, float roughness)
{
    float nDotL = max(dot(N, L), 0.0);
    float nDotV = max(dot(N, V), 0.0);
    if (nDotL <= 0.0 || nDotV <= 0.0)
    {
        return float3(0.0);
    }

    float3 H = normalize(L + V);
    float D = distributionGGX(N, H, max(roughness, MIN_ROUGHNESS));
    float G = geometrySmith(N, V, L, max(roughness, MIN_ROUGHNESS));
    float3 F = fresnelSchlickRoughness(saturate(dot(H, V)), f0, roughness);
    return (D * G * F) / max(4.0 * nDotV * nDotL, 1e-5);
}

struct SurfelPtReservoir {
    float3 direction;
    float3 brdfWeight;
    float pdf;
    float weightSum;
    float selectedWeight;
};

void addReflectionCandidate(inout SurfelPtReservoir reservoir,
                            float3 direction,
                            float3 brdfWeight,
                            float pdf,
                            float candidateWeight,
                            float randomValue)
{
    reservoir.weightSum += candidateWeight;
    if (reservoir.weightSum > 0.0 && randomValue < candidateWeight / reservoir.weightSum)
    {
        reservoir.direction = direction;
        reservoir.brdfWeight = brdfWeight;
        reservoir.pdf = max(pdf, 1e-5);
        reservoir.selectedWeight = candidateWeight;
    }
}
```

- [ ] **Step 2: Replace single GGX sample**

In Reflection:

```hlsl
SurfelPtReservoir reservoir;
reservoir.direction = reflect(-viewDir, normal);
reservoir.brdfWeight = float3(1.0);
reservoir.pdf = 1.0;
reservoir.weightSum = 0.0;
reservoir.selectedWeight = 0.0;

const uint candidateCount = 16u;
for (uint i = 0u; i < candidateCount; ++i)
{
    float2 candidateXi = float2(randomFloat(seed), randomFloat(seed));
    float3 candidateDir = normalize(ggxSampleDirection(candidateXi, normal, viewDir, roughness));
    float nDotL = max(dot(candidateDir, normal), 0.0);
    if (nDotL <= 0.0)
    {
        continue;
    }

    float3 brdf = ggxSpecularBrdf(candidateDir, normal, viewDir, f0, roughness);
    float pdf = max(ggxPdf(candidateDir, normal, viewDir, roughness), 1e-5);
    float candidateWeight = (1.0 / float(candidateCount)) * luminance(brdf) / pdf;
    addReflectionCandidate(reservoir, candidateDir, brdf, pdf, candidateWeight, randomFloat(seed));
}
```

If `reservoir.weightSum <= 1e-5`, `reservoir.pdf <= 1e-5`, or `luminance(reservoir.brdfWeight) <= 1e-5`, write zero reflection. Otherwise trace `reservoir.direction` and weight radiance like the SurfelPlus RIS estimator:

```hlsl
float risWeight = reservoir.weightSum / max(luminance(reservoir.brdfWeight), 1e-5);
float nDotSelected = max(dot(reservoir.direction, normal), 0.0);
radiance *= risWeight * reservoir.brdfWeight * nDotSelected;
```

Use the existing luminance clamp after applying the selected BRDF and cosine factor.

- [ ] **Step 3: Keep half-resolution reflection path**

Do not change reflection dispatch dimensions in this task. Real-time cost stays bounded.

- [ ] **Step 4: Verify and commit**

Run unit build, unit executable, editor build. Commit:

```powershell
git add src/shaders/SurfelPathTracerReflection.slang src/shaders/SurfelPathTracerCommon.slang tests/SurfelPathTracerPipelineTests.cpp
git commit -m "feat: sample reflections with ris"
```

### Task 14: Add Ping-Pong Temporal History Resources

**Files:**

- Modify: `src/Core/SurfelPathTracerResources.h`
- Modify: `src/Core/SurfelPathTracerResources.cpp`
- Modify: `src/Core/EngineCore.h`
- Modify: `src/Core/EngineCore.cpp`
- Modify: `src/Core/SurfelPathTracerPipelines.cpp`
- Modify: `src/Core/SurfelPathTracerPasses.h`
- Modify: `src/Core/SurfelPathTracerPasses.cpp`
- Modify: `src/shaders/SurfelPathTracerReflectionFilter.slang`
- Modify: `src/shaders/SurfelPathTracerTaa.slang`
- Modify: `tests/SurfelPathTracerPipelineTests.cpp`

- [ ] **Step 1: Add double history images**

Replace single history vectors with previous/current-capable resources:

```cpp
std::array<std::vector<VulkanUtils::VmaImage>, 2> filteredReflectionHistoryImages;
std::array<std::vector<vk::raii::ImageView>, 2> filteredReflectionHistoryViews;
std::array<std::vector<VulkanUtils::VmaImage>, 2> taaHistoryImages;
std::array<std::vector<vk::raii::ImageView>, 2> taaHistoryViews;
```

Keep the current filtered/current images as working images for existing passes, but history reads must bind source history and writes must bind destination history.

- [ ] **Step 2: Track indices**

In `EngineCore.h`:

```cpp
std::array<uint32_t, MAX_FRAMES_IN_FLIGHT> surfelPathTracerPreviousHistoryIndex{};
std::array<uint32_t, MAX_FRAMES_IN_FLIGHT> surfelPathTracerCurrentHistoryIndex{};
std::array<bool, MAX_FRAMES_IN_FLIGHT> surfelPathTracerTemporalHistoryValid{};
```

Initialize each frame so previous is bank `0` and current is bank `1`:

```cpp
surfelPathTracerPreviousHistoryIndex.fill(0u);
surfelPathTracerCurrentHistoryIndex.fill(1u);
surfelPathTracerTemporalHistoryValid.fill(false);
```

After successful TAA/final pass for frame-in-flight `fi`, swap only that frame's bank indices:

```cpp
std::swap(surfelPathTracerPreviousHistoryIndex[fi], surfelPathTracerCurrentHistoryIndex[fi]);
surfelPathTracerTemporalHistoryValid[fi] = true;
```

On resize/resource recreate/debug non-final/reset, set all valid flags false.

- [ ] **Step 3: Expand descriptor layout**

Keep history bindings as storage images to match the existing storage descriptor set and avoid adding sampler/layout complexity in the same task. Add bindings for:

```cpp
previousFilteredReflectionHistory
currentFilteredReflectionHistory
previousTaaHistory
currentTaaHistory
```

Use fixed storage bindings:

```cpp
20 = previousFilteredReflectionHistory
21 = currentFilteredReflectionHistory
22 = previousTaaHistory
23 = currentTaaHistory
```

Update `storageBindings` from 20 entries to 24 entries. Static descriptor writes are not enough because previous/current banks swap over time. Add a per-frame storage descriptor set variant for every `(previousBank, currentBank)` pair, or update the frame's storage descriptor set before command recording. The command buffer must bind descriptors where:

```cpp
previousFilteredReflectionHistory = filteredReflectionHistoryViews[previousHistoryIndex][fi]
currentFilteredReflectionHistory = filteredReflectionHistoryViews[currentHistoryIndex][fi]
previousTaaHistory = taaHistoryViews[previousHistoryIndex][fi]
currentTaaHistory = taaHistoryViews[currentHistoryIndex][fi]
```

Never bind the same image view to a previous/current pair when history is valid.

- [ ] **Step 4: Update shaders**

Reflection filter:

```hlsl
[[vk::binding(20, 0)]] RWTexture2D<float4> previousFilteredReflectionHistory;
[[vk::binding(21, 0)]] RWTexture2D<float4> currentFilteredReflectionHistory;
```

TAA:

```hlsl
[[vk::binding(22, 0)]] RWTexture2D<float4> previousTaaHistory;
[[vk::binding(23, 0)]] RWTexture2D<float4> currentTaaHistory;
```

Read previous at the reprojected pixel and write current at the current pixel. The previous and current bindings must point to different underlying images whenever `historyReady` is true. Add test needles for `previousFilteredReflectionHistory`, `currentFilteredReflectionHistory`, `previousTaaHistory`, `currentTaaHistory`, and either the per-bank descriptor set variants or the descriptor rewrites immediately before recording Surfel PT commands.

- [ ] **Step 5: Re-enable historyReady**

In `recordSurfelPathTracerCommandBuffer`, replace:

```cpp
const bool historyReady = false;
```

with:

```cpp
const bool historyReady = !ptForceHistoryReset && surfelPathTracerTemporalHistoryValid[fi];
```

Pass `surfelPathTracerPreviousHistoryIndex[fi]` and `surfelPathTracerCurrentHistoryIndex[fi]` into the descriptor selection/update path before recording ReflectionFilter and TAA. Only swap the indices after the current frame has successfully recorded the TAA/final pass.

- [ ] **Step 6: Verify and commit**

Run unit build, unit executable, editor build. Commit:

```powershell
git add src/Core/SurfelPathTracerResources.h src/Core/SurfelPathTracerResources.cpp src/Core/EngineCore.h src/Core/EngineCore.cpp src/Core/SurfelPathTracerPipelines.cpp src/Core/SurfelPathTracerPasses.h src/Core/SurfelPathTracerPasses.cpp src/shaders/SurfelPathTracerReflectionFilter.slang src/shaders/SurfelPathTracerTaa.slang tests/SurfelPathTracerPipelineTests.cpp
git commit -m "feat: add surfel path tracer ping-pong history"
```

### Task 15: Add Real-Time Validation And Debug Views

**Files:**

- Modify: `src/shaders/SurfelPathTracerLightIntegrate.slang`
- Modify: `src/shaders/SurfelPathTracerEvaluate.slang`
- Modify: `src/Core/UISystem.h`
- Modify: `src/Core/UISystem.cpp`
- Modify: `tests/SurfelPathTracerPipelineTests.cpp`

- [ ] **Step 1: Add debug outputs**

Map debug views:

```hlsl
static const uint SURFEL_DEBUG_SURFEL_COVERAGE = 10u;
static const uint SURFEL_DEBUG_SURFEL_VARIANCE = 11u;
static const uint SURFEL_DEBUG_SURFEL_RADIUS = 12u;
static const uint SURFEL_DEBUG_REFERENCE_COLOR = 13u;
static const uint SURFEL_DEBUG_REFERENCE_DIFFERENCE = 14u;
```

Coverage: output `coverage.xxx`.

Variance: output selected surfel `varianceAndInconsistency.xyz`.

Radius: output `surfel.radius / max(push.cellSize * 2.0, 1e-4)`.

Reference views can initially use the current lighting as reference until Task 16 adds the optional reference path. Do not display zeros because that hides wiring bugs.

- [ ] **Step 2: Add stats display**

Display new counters in `drawPathTracerStats()` or the Surfel PT stats block using existing ImGui style:

```cpp
ImGui::Text("Recycled Surfels: %u", surfelPathTracerStats.recycledSurfels);
ImGui::Text("Spawned Surfels: %u", surfelPathTracerStats.spawnedSurfels);
ImGui::Text("Removed Surfels: %u", surfelPathTracerStats.removedSurfels);
ImGui::Text("Guided Rays: %u", surfelPathTracerStats.guidedRays);
ImGui::Text("Cosine Rays: %u", surfelPathTracerStats.cosineRays);
ImGui::Text("Surfel-Terminated Paths: %u", surfelPathTracerStats.surfelTerminatedPaths);
```

- [ ] **Step 3: Verify and commit**

Run unit build, unit executable, editor build. Commit:

```powershell
git add src/shaders/SurfelPathTracerLightIntegrate.slang src/shaders/SurfelPathTracerEvaluate.slang src/Core/UISystem.h src/Core/UISystem.cpp tests/SurfelPathTracerPipelineTests.cpp
git commit -m "feat: add surfel path tracer validation views"
```

### Task 16: Add Optional Low-Sample Reference Mode

**Files:**

- Modify: `src/Core/SurfelPathTracerResources.h`
- Modify: `src/Core/SurfelPathTracerResources.cpp`
- Modify: `src/Core/SurfelPathTracerPipelines.h`
- Modify: `src/Core/SurfelPathTracerPipelines.cpp`
- Modify: `src/Core/SurfelPathTracerPasses.h`
- Modify: `src/Core/SurfelPathTracerPasses.cpp`
- Modify: `src/Core/EngineCore.cpp`
- Create: `src/shaders/SurfelPathTracerReference.slang`
- Modify: `src/shaders/SurfelPathTracerLightIntegrate.slang`
- Modify: `CMakeLists.txt`
- Modify: `tests/SurfelPathTracerPipelineTests.cpp`

Reference mode is a validation path only. It must use the same material decode, sky, and shadowed direct-light conventions as the production path so its difference view measures surfel-cache bias rather than inconsistent lighting equations.

- [ ] **Step 1: Add reference output image**

Add `referenceImages` and `referenceViews` sized to swapchain extent. Bind as storage image at storage descriptor binding `24`, after the Task 14 history bindings. Update `storageBindings` from 24 entries to 25 entries.

- [ ] **Step 2: Add shader entry**

Add to `CMakeLists.txt` shader list:

```cmake
"SurfelPathTracerReference.slang|main"
```

- [ ] **Step 3: Add a real RT reference pipeline**

Do not implement the reference path as a compute shader that calls `TraceRay`; `TraceRay` belongs in ray-generation shaders. Add a fourth RT pipeline/SBT family named `referenceRayTracingPipeline`/`referenceSbt` that uses:

- raygen: `SurfelPathTracerReference.slang`
- miss: `SurfelPathTracerMiss.slang`
- closest-hit: `SurfelPathTracerClosestHit.slang`
- any-hit: `SurfelPathTracerAnyHit.slang`

Add `createReferenceRayTracingPipeline()` and `createReferenceShaderBindingTable()` beside the existing GBuffer/surfel/reflection pipeline creation functions. Use the existing `rayTracingPipelineLayout`.

- [ ] **Step 4: Add reference raygen shader**

Create `src/shaders/SurfelPathTracerReference.slang` as a ray-generation shader that writes one low-sample validation path per pixel when enabled:

```hlsl
#include "SurfelPathTracerCommon.slang"

[[vk::binding(24, 0)]] RWTexture2D<float4> referenceImage;
[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(0, 2)]] ConstantBuffer<UniformBuffer> ubo;

struct SurfelPathTracerPayload {
    float3 radiance;
    float hitT;
    float3 hitPosition;
    float3 hitNormal;
    float3 baseColor;
    float metallic;
    float roughness;
    uint hitKind;
};

struct ReferencePushConstants {
    uint width;
    uint height;
    uint maxDepth;
    uint enabled;
};

[[vk::push_constant]] ReferencePushConstants push;

float3 referencePrimaryRayDirection(uint2 pixel)
{
    float2 uv = (float2(pixel) + 0.5) / float2(max(push.width, 1u), max(push.height, 1u));
    float2 d = uv * 2.0 - 1.0;
    float4 target = mul(ubo.projInverse, float4(d.x, -d.y, 1.0, 1.0));
    return normalize(mul(ubo.viewInverse, float4(normalize(target.xyz / target.w), 0.0)).xyz);
}

SurfelPathTracerPayload makeEmptySurfelPayload()
{
    SurfelPathTracerPayload payload;
    payload.radiance = float3(0.0);
    payload.hitT = -1.0;
    payload.hitPosition = float3(0.0);
    payload.hitNormal = float3(0.0, 1.0, 0.0);
    payload.baseColor = float3(0.0);
    payload.metallic = 0.0;
    payload.roughness = 1.0;
    payload.hitKind = 0u;
    return payload;
}

float3 traceReferencePath(RayDesc initialRay, uint maxDepth, uint seed)
{
    RayDesc ray = initialRay;
    float3 throughput = float3(1.0);
    float3 radiance = float3(0.0);
    for (uint depth = 0u; depth < max(maxDepth, 1u); ++depth)
    {
        SurfelPathTracerPayload payload = makeEmptySurfelPayload();
        TraceRay(tlas, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);
        radiance += throughput * payload.radiance;
        if (payload.hitKind == 0u)
        {
            break;
        }

        float2 xi = float2(randomFloat(seed), randomFloat(seed));
        float3 nextDir = cosineSampleHemisphere(xi, payload.hitNormal);
        float cosTheta = max(dot(payload.hitNormal, nextDir), 0.0);
        if (cosTheta <= 0.0)
        {
            break;
        }
        throughput *= saturate(payload.baseColor);
        ray.Origin = payload.hitPosition + payload.hitNormal * SURFEL_PT_RAY_BIAS;
        ray.Direction = normalize(nextDir);
    }
    return clampLuminance(radiance, SURFEL_PT_MAX_RADIANCE_LUMINANCE);
}

[shader("raygeneration")]
void main()
{
    uint2 pixel = DispatchRaysIndex().xy;
    if (pixel.x >= push.width || pixel.y >= push.height)
    {
        return;
    }
    if (push.enabled == 0u)
    {
        referenceImage[pixel] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    RayDesc ray;
    ray.Origin = ubo.cameraPos.xyz;
    ray.Direction = referencePrimaryRayDirection(pixel);
    ray.TMin = SURFEL_PT_RAY_BIAS;
    ray.TMax = 10000.0;

    uint seed = pcgHash(pixel.x ^ (pixel.y * 1664525u));
    referenceImage[pixel] = float4(traceReferencePath(ray, push.maxDepth, seed), 1.0);
}
```

This is a low-sample validation view, not the production renderer. It must be gated by `enableReferenceValidation`, record no work when disabled except clearing/stabilizing the reference image, and expose contract strings that prevent the reference path from being mistaken for the production Surfel PT result:

```cpp
"SurfelPathTracerReference.slang|main",
"createReferenceRayTracingPipeline",
"referenceSbt",
"[shader(\"raygeneration\")]",
"TraceRay(tlas",
"referenceImage[pixel]",
"enableReferenceValidation",
```

- [ ] **Step 5: Integrate reference difference view**

Light integrate reads `referenceImage` and outputs `abs(reference - lighting)` for `ReferenceDifference`.

- [ ] **Step 6: Verify and commit**

Run unit build, unit executable, editor build. Commit:

```powershell
git add CMakeLists.txt src/Core/SurfelPathTracerResources.h src/Core/SurfelPathTracerResources.cpp src/Core/SurfelPathTracerPipelines.h src/Core/SurfelPathTracerPipelines.cpp src/Core/SurfelPathTracerPasses.h src/Core/SurfelPathTracerPasses.cpp src/Core/EngineCore.cpp src/shaders/SurfelPathTracerReference.slang src/shaders/SurfelPathTracerLightIntegrate.slang tests/SurfelPathTracerPipelineTests.cpp
git commit -m "feat: add surfel path tracer reference view"
```

### Task 17: Final Pass-Order, Barrier, And Performance Review

**Files:**

- Modify: `src/Core/EngineCore.cpp`
- Modify: `src/Core/SurfelPathTracerPasses.cpp`
- Modify: `tests/SurfelPathTracerPipelineTests.cpp`

- [ ] **Step 1: Assert final pass order in tests**

Add a helper that checks relative substring order in `EngineCore.cpp`:

```cpp
auto findAfter = [&](std::string_view needle, size_t start) {
    const auto pos = engineCore.find(needle, start);
    if (pos == std::string::npos)
    {
        std::cerr << "missing SurfelPathTracer pass marker: " << needle << '\n';
    }
    return pos;
};
auto before = [&](size_t first, size_t second, std::string_view label) {
    if (first == std::string::npos || second == std::string::npos || first >= second)
    {
        std::cerr << "SurfelPathTracer pass-order contract failed: " << label << '\n';
        return false;
    }
    return true;
};

const auto gbuffer = findAfter("recordGBufferPass", 0);
const auto prepare = findAfter("recordPreparePass", gbuffer);
const auto evaluateGenerate = findAfter("SurfelPathTracerEvaluateMode::Generate", prepare);
const auto update = findAfter("recordUpdatePass", evaluateGenerate);
const auto cellInfo = findAfter("recordCellInfoPass", update);
const auto cellToSurfel = findAfter("recordCellToSurfelPass", cellInfo);
const auto surfelRayTrace = findAfter("recordSurfelRayTracePass", cellToSurfel);
const auto integrate = findAfter("recordIntegratePass", surfelRayTrace);
const auto evaluateResolve = findAfter("SurfelPathTracerEvaluateMode::Resolve", integrate);
const auto reflection = findAfter("recordReflectionPass", evaluateResolve);
const auto reflectionFilter = findAfter("recordReflectionFilterPass", reflection);
const auto bilateral = findAfter("recordBilateralPass", reflectionFilter);
const auto lightIntegrate = findAfter("recordLightIntegratePass", bilateral);
const auto taa = findAfter("recordTaaPass", lightIntegrate);

ok = ok && before(gbuffer, prepare, "GBuffer before Prepare");
ok = ok && before(prepare, evaluateGenerate, "Prepare before Generate Evaluate");
ok = ok && before(evaluateGenerate, update, "Generate Evaluate before Update");
ok = ok && before(update, cellInfo, "Update before CellInfo");
ok = ok && before(cellInfo, cellToSurfel, "CellInfo before CellToSurfel");
ok = ok && before(cellToSurfel, surfelRayTrace, "CellToSurfel before Surfel RayTrace");
ok = ok && before(surfelRayTrace, integrate, "Surfel RayTrace before Integrate");
ok = ok && before(integrate, evaluateResolve, "Integrate before Resolve Evaluate");
ok = ok && before(evaluateResolve, reflection, "Resolve Evaluate before Reflection");
ok = ok && before(reflection, reflectionFilter, "Reflection before ReflectionFilter");
ok = ok && before(reflectionFilter, bilateral, "ReflectionFilter before Bilateral");
ok = ok && before(bilateral, lightIntegrate, "Bilateral before LightIntegrate");
ok = ok && before(lightIntegrate, taa, "LightIntegrate before TAA");
```

This test must distinguish the Generate and Resolve Evaluate calls. A single substring search for `recordEvaluatePass` is not sufficient because the renderer intentionally records Evaluate twice.

Do not use this older ambiguous pattern:

```cpp
auto before = [&](std::string_view first, std::string_view second) {
    const auto a = engineCore.find(first);
    const auto b = engineCore.find(second);
    return a != std::string::npos && b != std::string::npos && a < b;
};
```

- [ ] **Step 2: Review barriers**

Ensure these barriers exist:

```cpp
recordImageBarrierGBufferToCompute
recordStorageBarrierComputeToCompute
recordStorageBarrierComputeToRt
recordStorageBarrierRtToCompute
transitionSurfelOutputForBlit
```

Add test needles for every barrier helper.

- [ ] **Step 3: Add bounded real-time defaults**

Set default settings to:

```cpp
maxSurfels = 150000;
maxRaysPerFrame = 150000 * 16;
minRaysPerSurfel = 4;
maxRaysPerSurfel = 64;
maxSurfelSamplesPerQuery = 32;
maxRadianceSharingSamples = 32;
activeMaxDepth = 3;
sleepingMaxDepth = 5;
```

If GPU memory pressure is observed during editor build/run, reduce `maxRaysPerFrame` default to `150000 * 8` and document the reason in a code comment near the default.

- [ ] **Step 4: Verify**

Run:

```powershell
git diff --check
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests"
.\cmake-build-debug\LaphriaEngineUnitTests.exe
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build cmake-build-debug --config Debug --target LaphriaEditor"
```

Expected:

- `git diff --check` returns exit 0, ignoring CRLF warnings.
- Unit target builds.
- Unit executable exits 0.
- Editor target builds.

- [ ] **Step 5: Commit**

```powershell
git add src/Core/EngineCore.cpp src/Core/SurfelPathTracerPasses.cpp tests/SurfelPathTracerPipelineTests.cpp src/Core/UISystem.h
git commit -m "fix: harden surfel path tracer pass graph"
```

### Task 18: Final Review And Handoff

**Files:**

- Modify only if review finds issues.

- [ ] **Step 1: Review against local references**

Compare local implementation against:

```powershell
rg -n "rayRequestCnt|surfelIrradiance|MSME|sharedRadiance|sample_PrevTAASampler|Resevior|surfelRefelctionTrace" .codex_refs\SurfelPlus\shaders
rg -n "getCellPos|isSurfelIntersectCell|gVarianceSensitivity|gIrradianceMap|MSME" .codex_refs\SurfelGI\RenderPasses\Surfel
```

Confirm local equivalents exist:

```powershell
rg -n "adaptiveRayCountForSurfel|sampleGuidedSurfelDirection|updateMsme|terminatePathWithSurfels|cameraRelativeCellIndexForPosition|isSurfelIntersectCell|previousHistoryIndex|currentHistoryIndex" src\shaders src\Core
```

- [ ] **Step 2: Request code review**

Ask a reviewer to focus on:

- same-image history hazards;
- descriptor binding completeness;
- pass-order and barrier correctness;
- ray budget bounds;
- atlas bounds;
- surfel recycling free-list correctness;
- whether the main path remains real-time-biased rather than offline-oriented.

- [ ] **Step 3: Fix Important or Critical review findings**

For each accepted finding, patch, verify with the full command set from Task 17, and commit with a focused message.

- [ ] **Step 4: Final verification**

Run:

```powershell
git diff --check
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests"
.\cmake-build-debug\LaphriaEngineUnitTests.exe
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build cmake-build-debug --config Debug --target LaphriaEditor"
```

- [ ] **Step 5: Handoff summary**

Summarize:

- implemented SurfelPlus/SurfelGI features;
- intentionally biased real-time choices;
- debug/reference validation modes;
- remaining limitations, especially non-uniform outer grid if not implemented;
- verification commands and results.

---

## Self-Review

- Spec coverage: The plan covers every issue from the review: real-time path transport, camera-relative grid, lifecycle/recycling, directional atlas, guided sampling, adaptive ray budgets, surfel termination, MSME/radiance sharing, RIS reflections, temporal ping-pong, and validation views.
- Gap scan: There are no unresolved markers or open-ended implementation gaps. Task 16 intentionally introduces a gated reference image wiring step and names the exact contract it must satisfy before use.
- Type consistency: Settings names, helper names, and test needles are kept consistent across tasks.
- Scope check: Non-uniform SurfelPlus outer grid is explicitly deferred until the camera-relative uniform grid is correct. That keeps this plan achievable while still aligned with the real-time Surfel repos.
- Execution risk: Tasks 10-14 are the highest risk and should use subagents plus focused review. Do not batch them without verification.
