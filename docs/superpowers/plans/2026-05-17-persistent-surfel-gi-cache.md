# Persistent Surfel GI Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a diagnostic-first persistent surfel GI cache that replaces the failed per-receiver bright-surfel proposal experiment with bounded surfel generation, cell indexing, and receiver gather plumbing.

**Architecture:** Add surfels as a first-class cache subsystem rather than a reservoir proposal mode. The MVP uses fixed-capacity cells and compute passes to generate and index surfels from path-tracer first-hit data, then exposes bounded gather diagnostics to `Raygen.slang`; lighting contribution remains disabled until counters prove the cache is populated and bounded.

**Tech Stack:** Vulkan RAII, Slang compute/raygen shaders, existing `FrameContext`, `PipelineCollection`, `EngineCore`, `PathTracerAnalysisCounters`, CMake unit tests.

---

## File Structure

- Modify `src/Core/EngineAuxiliary.h`
  - Add surfel constants and diagnostic counter fields after the current bright-surfel diagnostics.
  - Keep counter layout append-only to avoid breaking existing shader offsets.
- Modify `CMakeLists.txt`
  - Register every new Slang entry point in `SHADER_SOURCES`.
  - Add shared include dependencies so changes to `ShaderCommon.slang` or `SurfelCommon.slang` rebuild dependent surfel shaders.
- Modify `src/Core/FrameContext.h`
  - Add persistent surfel buffers, fixed cell buffers, surfel debug images, and helper constants.
  - Keep old `reservoirGiBrightSurfelBuffers` intact for now, but the plan disables their expensive proposal use.
- Modify `src/Core/FrameContext.cpp`
  - Create/destroy surfel buffers as device-local or host-visible depending on whether CPU readback is required.
  - MVP counters remain in `ptAnalysisCounterBuffers`; surfel data itself should not be host-visible.
- Modify `src/Core/PipelineCollection.h`
  - Add surfel descriptor set layout, pipeline layout, and compute pipelines.
- Modify `src/Core/PipelineCollection.cpp`
  - Create the surfel descriptor layout and compute pipelines for clear/generate/build-cell/debug-evaluate passes.
- Modify `src/Core/EngineCore.h`
  - Add surfel descriptor pool/set members and helper declarations.
- Modify `src/Core/EngineCore.cpp`
  - Allocate/update surfel descriptors.
  - Insert surfel compute passes in the path-tracer frame before main ray tracing.
  - Disable the old bright-surfel proposal mode in automated sweeps.
- Modify `src/Core/UISystem.h`
  - Add surfel GI settings and debug AOV options.
- Modify `src/Core/UISystem.cpp`
  - Add UI controls for surfel enable, gather debug, and caps.
- Create `src/shaders/SurfelCommon.slang`
  - Shared surfel structs, constants, cell helpers, and diagnostic counter offsets.
- Create `src/shaders/SurfelClear.slang`
  - Clears per-frame surfel counters and fixed-cell headers.
- Create `src/shaders/SurfelGenerate.slang`
  - Generates surfels from first-hit G-buffer data.
- Create `src/shaders/SurfelBuildCells.slang`
  - Inserts live surfels into fixed-capacity cell slots.
- Create `src/shaders/SurfelEvaluate.slang`
  - Computes bounded receiver gather diagnostics into counters/debug channels.
- Modify `src/shaders/Raygen.slang`
  - Remove the old bright-surfel proposal from normal execution.
  - Do not read surfel gather data from raygen in this diagnostic plan; raygen integration belongs in the later lighting-contribution plan.
- Modify `src/shaders/Denoiser.slang`
  - Add debug AOV display path for surfel occupancy/evaluation if we store it in an existing debug image.
- Modify `tests/PathTracerAnalysisTests.cpp`
  - Replace tests requiring indexed bright-surfel lookup with tests that require the old path to be disabled and surfel cache contracts to exist.

---

## Re-Review Corrections

These constraints are mandatory for execution agents:

- Surfel compute shaders must include `ShaderCommon.slang` before using `UniformBuffer`.
- The current UBO field names are `viewInverse`, `projInverse`, and `cameraPos`, not `invView`, `invProj`, or `cameraPosition`.
- `gBufferDepth` stores primary ray hit distance (`payload.hitT`), not clip-space depth. Reconstruct world position as `ubo.cameraPos.xyz + primaryRayDir * hitT`.
- Existing G-buffer images are bound as storage images and existing compute shaders read them as `RWTexture2D`; do not use sampled `Texture2D` unless the image usage and descriptor types are also changed.
- Surfel passes should run after the main path tracing dispatch and before reprojection/denoising for this diagnostic MVP. That gives them current-frame G-buffer data and avoids previous-frame camera reconstruction errors.
- `surfelGiPipelineLayout` must bind both the surfel descriptor set and the global UBO descriptor set. `descriptorSetLayoutGlobal` binding 0 must include `vk::ShaderStageFlagBits::eCompute`.
- The first diagnostic cache is frame-local data stored in persistent GPU buffers. True multi-frame surfel lifetime/free-list/recycling is a follow-up plan after the fixed-cell diagnostics pass its gates.
- Do not reuse `ptReprojectionDebug` for surfel debug output. Reprojection writes that image after surfel evaluation and would erase the surfel debug AOV. Add a separate `surfelGiDebug` image and bind it to both the surfel and denoiser descriptor sets.
- The clear pass must clear stale surfel record flags as well as cells/counters. Otherwise persistent GPU buffers will retain prior-frame `flags = 1` records and `SurfelBuildCells` will index ghost surfels.

---

## Task 1: Disable The Failed Bright-Surfel Proposal From Normal Runs

**Files:**
- Modify: `src/shaders/Raygen.slang`
- Modify: `src/Core/EngineCore.cpp`
- Modify: `tests/PathTracerAnalysisTests.cpp`

- [ ] **Step 1: Write the failing contract test**

Add this helper near the existing bright-surfel contract helpers in `tests/PathTracerAnalysisTests.cpp`:

```cpp
bool requireBrightSurfelProposalDisabledForSweeps(const std::string &raygen,
                                                  const std::string &engineCore)
{
    if (!containsText(raygen, "static const int RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_BRIGHT_SURFEL"))
        return false;

    const std::string reservoirMain =
        extractFunctionBody(raygen, "FirstHitDiffuseBounceResult sampleFirstHitReservoirGiSingleFrame(");
    if (reservoirMain.empty())
        return false;

    if (!containsText(reservoirMain, "const bool enableBrightSurfelProposal = false"))
        return false;

    if (containsText(engineCore, "reservoirMixedTemporalSpatialBudget2SunReceiverBrightSurfelRow") ||
        containsText(engineCore, "reservoirMixedSingleFrameSunReceiverBrightSurfelRow"))
        return false;

    return true;
}
```

Call it from the existing path tracer measurement contract after reading `raygen` and `engineCore`:

```cpp
if (!requireBrightSurfelProposalDisabledForSweeps(raygen, engineCore))
{
    std::cerr << "Bright surfel reservoir proposal must be disabled for sweeps before persistent surfel cache work\n";
    return false;
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
cmake --build build --config Debug --target LaphriaEngineUnitTests
ctest --test-dir build -C Debug --output-on-failure -R LaphriaEngineUnitTests
```

Expected: `LaphriaEngineUnitTests` fails with the new bright-surfel proposal disable message.

- [ ] **Step 3: Disable the old proposal path in shader control flow**

In `src/shaders/Raygen.slang`, inside `evaluateReservoirGiForFirstHit`, define:

```hlsl
const bool enableBrightSurfelProposal = false;
```

Replace checks like:

```hlsl
reservoirGiProposalMode == RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_BRIGHT_SURFEL
```

with:

```hlsl
enableBrightSurfelProposal &&
reservoirGiProposalMode == RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_BRIGHT_SURFEL
```

- [ ] **Step 4: Remove bright-surfel rows from the automated Sponza sweep**

In `src/Core/EngineCore.cpp`, remove or comment out the rows named:

```cpp
reservoirMixedTemporalSpatialBudget2SunReceiverBrightSurfelRow
reservoirMixedSingleFrameSunReceiverBrightSurfelRow
```

Keep the enum and UI value available for manual archaeology; the automated sweep must not select it.

- [ ] **Step 5: Run the contract test**

Run:

```powershell
cmake --build build --config Debug --target LaphriaEngineUnitTests
ctest --test-dir build -C Debug --output-on-failure -R LaphriaEngineUnitTests
```

Expected: `LaphriaEngineUnitTests` passes or fails only on older bright-surfel tests that Task 2 will replace.

- [ ] **Step 6: Commit**

```powershell
git add src/shaders/Raygen.slang src/Core/EngineCore.cpp tests/PathTracerAnalysisTests.cpp
git commit -m "chore: disable failed bright surfel proposal path"
```

---

## Task 2: Add Surfel Constants And Diagnostic Counters

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `src/Core/EngineAuxiliary.h`
- Modify: `tests/PathTracerAnalysisTests.cpp`
- Create: `src/shaders/SurfelCommon.slang`

- [ ] **Step 1: Write the failing counter-layout test**

Append a new contract helper in `tests/PathTracerAnalysisTests.cpp`:

```cpp
bool requirePersistentSurfelCounterLayout(const std::string &engineAuxiliaryHeader,
                                          const std::string &surfelCommon,
                                          const std::string &cmakeLists)
{
    const char *requiredCounters[] = {
        "surfelGiClearDispatches",
        "surfelGiGenerateAttempts",
        "surfelGiGenerated",
        "surfelGiGenerateRejectInvalid",
        "surfelGiGenerateRejectCoverage",
        "surfelGiCellInsertAttempts",
        "surfelGiCellInserted",
        "surfelGiCellOverflow",
        "surfelGiEvalAttempts",
        "surfelGiEvalCellEmpty",
        "surfelGiEvalCandidates",
        "surfelGiEvalAccepted"
    };

    for (const char *counter : requiredCounters)
    {
        if (!containsText(engineAuxiliaryHeader, counter))
            return false;
    }

    const char *requiredOffsets[] = {
        "surfelGiClearDispatchesOffset",
        "surfelGiGenerateAttemptsOffset",
        "surfelGiGeneratedOffset",
        "surfelGiCellOverflowOffset",
        "surfelGiEvalCandidatesOffset",
        "surfelGiEvalAcceptedOffset"
    };

    for (const char *offset : requiredOffsets)
    {
        if (!containsText(surfelCommon, offset))
            return false;
    }

    return containsText(cmakeLists, "SURFEL_SHADER_INCLUDE_DEPS") &&
           containsText(cmakeLists, "SurfelCommon.slang") &&
           containsText(cmakeLists, "ShaderCommon.slang");
}
```

Read `SurfelCommon.slang` and `CMakeLists.txt` in the test fixture and call the helper.

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
cmake --build build --config Debug --target LaphriaEngineUnitTests
ctest --test-dir build -C Debug --output-on-failure -R LaphriaEngineUnitTests
```

Expected: failure because `SurfelCommon.slang`, surfel counters, and shader include dependencies do not exist.

- [ ] **Step 3: Append surfel counters**

In `src/Core/EngineAuxiliary.h`, append these fields at the end of `PathTracerAnalysisCounters`:

```cpp
uint32_t surfelGiClearDispatches = 0;
uint32_t surfelGiGenerateAttempts = 0;
uint32_t surfelGiGenerated = 0;
uint32_t surfelGiGenerateRejectInvalid = 0;
uint32_t surfelGiGenerateRejectCoverage = 0;
uint32_t surfelGiCellInsertAttempts = 0;
uint32_t surfelGiCellInserted = 0;
uint32_t surfelGiCellOverflow = 0;
uint32_t surfelGiEvalAttempts = 0;
uint32_t surfelGiEvalCellEmpty = 0;
uint32_t surfelGiEvalCandidates = 0;
uint32_t surfelGiEvalAccepted = 0;
```

- [ ] **Step 4: Create shared shader definitions**

Create `src/shaders/SurfelCommon.slang`:

```hlsl
#ifndef LAPHRIA_SURFEL_COMMON_SLANG
#define LAPHRIA_SURFEL_COMMON_SLANG

#include "ShaderCommon.slang"

static const uint SURFEL_GI_MAX_SURFELS = 32768u;
static const uint SURFEL_GI_GRID_DIM = 32u;
static const uint SURFEL_GI_CELL_COUNT = SURFEL_GI_GRID_DIM * SURFEL_GI_GRID_DIM * SURFEL_GI_GRID_DIM;
static const uint SURFEL_GI_CELL_SLOT_COUNT = 4u;
static const uint SURFEL_GI_MAX_EVAL_CANDIDATES = 8u;
static const float SURFEL_GI_CELL_SIZE = 1.5f;
static const float SURFEL_GI_MIN_RADIUS = 0.25f;
static const float SURFEL_GI_MAX_RADIUS = 2.5f;

static const uint surfelGiClearDispatchesOffset = 404u;
static const uint surfelGiGenerateAttemptsOffset = 408u;
static const uint surfelGiGeneratedOffset = 412u;
static const uint surfelGiGenerateRejectInvalidOffset = 416u;
static const uint surfelGiGenerateRejectCoverageOffset = 420u;
static const uint surfelGiCellInsertAttemptsOffset = 424u;
static const uint surfelGiCellInsertedOffset = 428u;
static const uint surfelGiCellOverflowOffset = 432u;
static const uint surfelGiEvalAttemptsOffset = 436u;
static const uint surfelGiEvalCellEmptyOffset = 440u;
static const uint surfelGiEvalCandidatesOffset = 444u;
static const uint surfelGiEvalAcceptedOffset = 448u;

struct SurfelGiRecord
{
    float4 positionRadius;
    float4 normalConfidence;
    float4 radianceAge;
    uint flags;
    uint lastSeenFrame;
    uint cellIndex;
    uint pad0;
};

struct SurfelGiCell
{
    uint count;
    uint overflow;
    uint pad0;
    uint pad1;
};

uint flattenSurfelGiCell(int3 cell)
{
    return uint(cell.x) +
           uint(cell.y) * SURFEL_GI_GRID_DIM +
           uint(cell.z) * SURFEL_GI_GRID_DIM * SURFEL_GI_GRID_DIM;
}

bool isSurfelGiCellValid(int3 cell)
{
    return all(cell >= int3(0, 0, 0)) &&
           all(cell < int3(int(SURFEL_GI_GRID_DIM), int(SURFEL_GI_GRID_DIM), int(SURFEL_GI_GRID_DIM)));
}

int3 calcSurfelGiCell(float3 worldPos, float3 cameraPos)
{
    float3 local = worldPos - cameraPos;
    float3 grid = local / SURFEL_GI_CELL_SIZE + float3(SURFEL_GI_GRID_DIM, SURFEL_GI_GRID_DIM, SURFEL_GI_GRID_DIM) * 0.5f;
    return int3(floor(grid));
}

float3 calcSurfelGiPrimaryRayDirection(uint2 pixel,
                                       uint2 extent,
                                       float4x4 projInverse,
                                       float4x4 viewInverse)
{
    float2 pixelCenter = float2(pixel) + float2(0.5, 0.5);
    float2 inUV = pixelCenter / float2(extent);
    float2 d = inUV * 2.0f - 1.0f;
    float4 target = mul(projInverse, float4(d.x, -d.y, 1.0f, 1.0f));
    return mul(viewInverse, float4(normalize(target.xyz / target.w), 0.0f)).xyz;
}

float3 reconstructSurfelGiFirstHitWorldPosition(uint2 pixel,
                                                uint2 extent,
                                                float hitT,
                                                float3 cameraPos,
                                                float4x4 projInverse,
                                                float4x4 viewInverse)
{
    return cameraPos + calcSurfelGiPrimaryRayDirection(pixel, extent, projInverse, viewInverse) * hitT;
}

#endif
```

In `CMakeLists.txt`, before the shader compile loop, add:

```cmake
set(SURFEL_SHADER_INCLUDE_DEPS
        "${SHADER_SOURCE_ROOT}/ShaderCommon.slang"
        "${SHADER_SOURCE_ROOT}/SurfelCommon.slang"
)
```

In the `add_custom_command` for shader compilation, change:

```cmake
DEPENDS ${SHADER_SOURCE}
```

to:

```cmake
DEPENDS ${SHADER_SOURCE} ${SURFEL_SHADER_INCLUDE_DEPS}
```

Each task that creates a new surfel shader must also add that shader's own `SHADER_SOURCES` entry after the file exists. Do not register a shader before its source file is created.

- [ ] **Step 5: Run the unit tests**

Run:

```powershell
cmake --build build --config Debug --target LaphriaEngineUnitTests
ctest --test-dir build -C Debug --output-on-failure -R LaphriaEngineUnitTests
```

Expected: counter layout test passes.

- [ ] **Step 6: Commit**

```powershell
git add CMakeLists.txt src/Core/EngineAuxiliary.h src/shaders/SurfelCommon.slang tests/PathTracerAnalysisTests.cpp
git commit -m "feat: add persistent surfel diagnostics contract"
```

---

## Task 3: Add Surfel GPU Resources

**Files:**
- Modify: `src/Core/FrameContext.h`
- Modify: `src/Core/FrameContext.cpp`
- Modify: `tests/PathTracerAnalysisTests.cpp`

- [ ] **Step 1: Write the failing resource contract test**

Add a test helper requiring these symbols in `FrameContext.h` and `FrameContext.cpp`:

```cpp
bool requirePersistentSurfelFrameResources(const std::string &frameContextHeader,
                                           const std::string &frameContextSource)
{
    const char *headerSymbols[] = {
        "kSurfelGiMaxSurfels",
        "kSurfelGiGridDim",
        "kSurfelGiCellSlotCount",
        "surfelGiRecordBuffers",
        "surfelGiCellBuffers",
        "surfelGiCellSlotBuffers",
        "surfelGiCounterBuffers",
        "surfelGiDebugImages",
        "surfelGiDebugImageViews",
        "createSurfelGiBuffers"
    };

    for (const char *symbol : headerSymbols)
    {
        if (!containsText(frameContextHeader, symbol))
            return false;
    }

    const char *sourceSymbols[] = {
        "void FrameContext::createSurfelGiBuffers",
        "vk::MemoryPropertyFlagBits::eDeviceLocal",
        "vk::BufferUsageFlagBits::eStorageBuffer",
        "vk::ImageUsageFlagBits::eStorage",
        "surfelGiRecordBuffers.clear()",
        "surfelGiCellSlotBuffers.clear()",
        "surfelGiDebugImages.clear()"
    };

    for (const char *symbol : sourceSymbols)
    {
        if (!containsText(frameContextSource, symbol))
            return false;
    }

    return true;
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run the unit test command from Task 2. Expected: missing surfel frame resources.

- [ ] **Step 3: Add resource declarations**

In `src/Core/FrameContext.h`, add:

```cpp
static constexpr uint32_t kSurfelGiMaxSurfels = 32768;
static constexpr uint32_t kSurfelGiGridDim = 32;
static constexpr uint32_t kSurfelGiCellCount = kSurfelGiGridDim * kSurfelGiGridDim * kSurfelGiGridDim;
static constexpr uint32_t kSurfelGiCellSlotCount = 4;
static constexpr vk::DeviceSize kSurfelGiRecordSize = 64;
static constexpr vk::DeviceSize kSurfelGiCellSize = 16;
static constexpr vk::DeviceSize kSurfelGiCellSlotSize = 4;
static constexpr vk::DeviceSize kSurfelGiCounterSize = 64;

std::vector<Laphria::VulkanUtils::VmaBuffer> surfelGiRecordBuffers;
std::vector<Laphria::VulkanUtils::VmaBuffer> surfelGiCellBuffers;
std::vector<Laphria::VulkanUtils::VmaBuffer> surfelGiCellSlotBuffers;
std::vector<Laphria::VulkanUtils::VmaBuffer> surfelGiCounterBuffers;
std::vector<Laphria::VulkanUtils::VmaImage> surfelGiDebugImages;
std::vector<vk::raii::ImageView> surfelGiDebugImageViews;
```

Add private declaration:

```cpp
void createSurfelGiBuffers(const VulkanDevice &dev, const SwapchainManager &swapchain);
```

- [ ] **Step 4: Create buffers**

In `FrameContext::init`, call `createSurfelGiBuffers(dev, swapchain)` after `createPathTracerAnalysisBuffers(dev)`.

In `FrameContext` cleanup paths, release the new buffer vectors using `destroyBuffersAndReleaseAllocations`, release `surfelGiDebugImages` using `destroyImagesAndReleaseAllocations`, and clear `surfelGiDebugImageViews`.

Implement:

```cpp
void FrameContext::createSurfelGiBuffers(const VulkanDevice &dev, const SwapchainManager &swapchain)
{
    surfelGiRecordBuffers.clear();
    surfelGiCellBuffers.clear();
    surfelGiCellSlotBuffers.clear();
    surfelGiCounterBuffers.clear();
    surfelGiDebugImages.clear();
    surfelGiDebugImageViews.clear();

    surfelGiRecordBuffers.reserve(MAX_FRAMES_IN_FLIGHT);
    surfelGiCellBuffers.reserve(MAX_FRAMES_IN_FLIGHT);
    surfelGiCellSlotBuffers.reserve(MAX_FRAMES_IN_FLIGHT);
    surfelGiCounterBuffers.reserve(MAX_FRAMES_IN_FLIGHT);
    surfelGiDebugImages.reserve(MAX_FRAMES_IN_FLIGHT);
    surfelGiDebugImageViews.reserve(MAX_FRAMES_IN_FLIGHT);

    const vk::DeviceSize recordBytes = kSurfelGiMaxSurfels * kSurfelGiRecordSize;
    const vk::DeviceSize cellBytes = kSurfelGiCellCount * kSurfelGiCellSize;
    const vk::DeviceSize cellSlotBytes = kSurfelGiCellCount * kSurfelGiCellSlotCount * kSurfelGiCellSlotSize;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        Laphria::VulkanUtils::VmaBuffer records{};
        Laphria::VulkanUtils::VmaBuffer cells{};
        Laphria::VulkanUtils::VmaBuffer slots{};
        Laphria::VulkanUtils::VmaBuffer counters{};

        VulkanUtils::createBuffer(dev.logicalDevice, dev.physicalDevice, recordBytes,
                                  vk::BufferUsageFlagBits::eStorageBuffer,
                                  vk::MemoryPropertyFlagBits::eDeviceLocal, records);
        VulkanUtils::createBuffer(dev.logicalDevice, dev.physicalDevice, cellBytes,
                                  vk::BufferUsageFlagBits::eStorageBuffer,
                                  vk::MemoryPropertyFlagBits::eDeviceLocal, cells);
        VulkanUtils::createBuffer(dev.logicalDevice, dev.physicalDevice, cellSlotBytes,
                                  vk::BufferUsageFlagBits::eStorageBuffer,
                                  vk::MemoryPropertyFlagBits::eDeviceLocal, slots);
        VulkanUtils::createBuffer(dev.logicalDevice, dev.physicalDevice, kSurfelGiCounterSize,
                                  vk::BufferUsageFlagBits::eStorageBuffer,
                                  vk::MemoryPropertyFlagBits::eDeviceLocal, counters);

        surfelGiRecordBuffers.push_back(std::move(records));
        surfelGiCellBuffers.push_back(std::move(cells));
        surfelGiCellSlotBuffers.push_back(std::move(slots));
        surfelGiCounterBuffers.push_back(std::move(counters));

        Laphria::VulkanUtils::VmaImage debugImage{};
        VulkanUtils::createImage(dev.logicalDevice, dev.physicalDevice,
                                 swapchain.extent.width, swapchain.extent.height,
                                 vk::Format::eR16G16B16A16Sfloat, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eStorage,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal, debugImage);
        surfelGiDebugImages.push_back(std::move(debugImage));
        surfelGiDebugImageViews.push_back(VulkanUtils::createImageView(
            dev.logicalDevice, *surfelGiDebugImages.back(),
            vk::Format::eR16G16B16A16Sfloat, vk::ImageAspectFlagBits::eColor));
    }

    {
        auto cmd = VulkanUtils::beginSingleTimeCommands(dev.logicalDevice, commandPool);
        for (auto &img : surfelGiDebugImages)
            VulkanUtils::recordImageLayoutTransition(cmd, *img,
                                                     vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
        VulkanUtils::endSingleTimeCommands(dev.logicalDevice, dev.queue, commandPool, cmd);
    }
}
```

- [ ] **Step 5: Run unit tests**

Run:

```powershell
cmake --build build --config Debug --target LaphriaEngineUnitTests
ctest --test-dir build -C Debug --output-on-failure -R LaphriaEngineUnitTests
```

Expected: resource contract passes.

- [ ] **Step 6: Commit**

```powershell
git add src/Core/FrameContext.h src/Core/FrameContext.cpp tests/PathTracerAnalysisTests.cpp
git commit -m "feat: allocate persistent surfel GI buffers"
```

---

## Task 4: Add Surfel Descriptor Layout And Clear Pass

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `src/Core/PipelineCollection.h`
- Modify: `src/Core/PipelineCollection.cpp`
- Modify: `src/Core/EngineCore.h`
- Modify: `src/Core/EngineCore.cpp`
- Create: `src/shaders/SurfelClear.slang`
- Modify: `tests/PathTracerAnalysisTests.cpp`

- [ ] **Step 1: Write the failing pipeline contract test**

Require these symbols:

```cpp
bool requireSurfelClearPassContracts(const std::string &pipelineHeader,
                                     const std::string &pipelineSource,
                                     const std::string &engineHeader,
                                     const std::string &engineCore,
                                     const std::string &surfelClear)
{
    const char *symbols[] = {
        "surfelGiDescriptorSetLayout",
        "surfelGiPipelineLayout",
        "surfelGiClearPipeline",
        "createSurfelGiDescriptorSetLayout",
        "createSurfelGiClearPipeline",
        "createSurfelGiDescriptorSets",
        "recordSurfelGiClearPass"
    };

    for (const char *symbol : symbols)
    {
        if (!containsText(pipelineHeader + pipelineSource + engineHeader + engineCore, symbol))
            return false;
    }

    return containsText(surfelClear, "void surfelClearMain") &&
           containsText(surfelClear, "surfelGiClearDispatchesOffset");
}
```

- [ ] **Step 2: Run the test to verify it fails**

Expected: missing surfel clear pass symbols.

- [ ] **Step 3: Create descriptor layout and pipeline declarations**

After Step 5 creates `src/shaders/SurfelClear.slang`, add this entry to `SHADER_SOURCES` in `CMakeLists.txt`:

```cmake
"SurfelClear.slang|surfelClearMain"
```

In `PipelineCollection.h`, add RAII members:

```cpp
vk::raii::DescriptorSetLayout surfelGiDescriptorSetLayout{nullptr};
vk::raii::PipelineLayout surfelGiPipelineLayout{nullptr};
vk::raii::Pipeline surfelGiClearPipeline{nullptr};
```

Add methods:

```cpp
void createSurfelGiDescriptorSetLayout(const VulkanDevice &dev);
void createSurfelGiPipelineLayout(const VulkanDevice &dev);
void createSurfelGiClearPipeline(const VulkanDevice &dev);
```

Call these from existing pipeline setup.

- [ ] **Step 4: Implement surfel descriptor layout and pipeline layout**

In `createGlobalDescriptorSetLayout`, add `vk::ShaderStageFlagBits::eCompute` to binding 0. Surfel compute shaders bind `UniformBuffer` from global descriptor set 1.

Use binding layout:

```cpp
std::array<vk::DescriptorSetLayoutBinding, 6> bindings = {
    vk::DescriptorSetLayoutBinding{.binding = 0, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eRaygenKHR},
    vk::DescriptorSetLayoutBinding{.binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eRaygenKHR},
    vk::DescriptorSetLayoutBinding{.binding = 2, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eRaygenKHR},
    vk::DescriptorSetLayoutBinding{.binding = 3, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eRaygenKHR},
    vk::DescriptorSetLayoutBinding{.binding = 4, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eRaygenKHR},
    vk::DescriptorSetLayoutBinding{.binding = 5, .descriptorType = vk::DescriptorType::eStorageImage,  .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute}
};
```

Meanings:

- 0: `SurfelGiRecord[]`
- 1: `SurfelGiCell[]`
- 2: `uint CellSlots[]`
- 3: surfel internal counters
- 4: `PathTracerAnalysisCounters`
- 5: surfel debug image, `frames.surfelGiDebugImageViews[i]`

Create `surfelGiPipelineLayout` with both surfel resources and the global UBO:

```cpp
std::array layouts = {*surfelGiDescriptorSetLayout, *descriptorSetLayoutGlobal};
vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
    .setLayoutCount = static_cast<uint32_t>(layouts.size()),
    .pSetLayouts = layouts.data()};
surfelGiPipelineLayout = vk::raii::PipelineLayout(dev.logicalDevice, pipelineLayoutInfo);
```

- [ ] **Step 5: Create clear shader**

Create `src/shaders/SurfelClear.slang`:

```hlsl
#include "SurfelCommon.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<SurfelGiRecord> surfelGiRecords;
[[vk::binding(1, 0)]] RWStructuredBuffer<SurfelGiCell> surfelGiCells;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> surfelGiCellSlots;
[[vk::binding(3, 0)]] RWByteAddressBuffer surfelGiCounters;
[[vk::binding(4, 0)]] RWByteAddressBuffer ptAnalysisCounters;

[numthreads(128, 1, 1)]
void surfelClearMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint index = dispatchThreadID.x;

    if (index == 0u)
    {
        surfelGiCounters.Store(0u, 0u);
        surfelGiCounters.Store(4u, 0u);
        surfelGiCounters.Store(8u, 0u);
        surfelGiCounters.Store(12u, 0u);
        ptAnalysisCounters.InterlockedAdd(surfelGiClearDispatchesOffset, 1u);
    }

    if (index < SURFEL_GI_MAX_SURFELS)
    {
        SurfelGiRecord emptyRecord;
        emptyRecord.positionRadius = float4(0.0f, 0.0f, 0.0f, 0.0f);
        emptyRecord.normalConfidence = float4(0.0f, 1.0f, 0.0f, 0.0f);
        emptyRecord.radianceAge = float4(0.0f, 0.0f, 0.0f, 0.0f);
        emptyRecord.flags = 0u;
        emptyRecord.lastSeenFrame = 0u;
        emptyRecord.cellIndex = 0xffffffffu;
        emptyRecord.pad0 = 0u;
        surfelGiRecords[index] = emptyRecord;
    }

    if (index < SURFEL_GI_CELL_COUNT)
    {
        SurfelGiCell cell;
        cell.count = 0u;
        cell.overflow = 0u;
        cell.pad0 = 0u;
        cell.pad1 = 0u;
        surfelGiCells[index] = cell;

        uint baseSlot = index * SURFEL_GI_CELL_SLOT_COUNT;
        [unroll]
        for (uint slot = 0u; slot < SURFEL_GI_CELL_SLOT_COUNT; ++slot)
        {
            surfelGiCellSlots[baseSlot + slot] = 0xffffffffu;
        }
    }
}
```

- [ ] **Step 6: Allocate/update surfel descriptor sets**

In `EngineCore.h`, add:

```cpp
vk::raii::DescriptorPool surfelGiDescriptorPool{nullptr};
std::vector<vk::raii::DescriptorSet> surfelGiDescriptorSets;
void createSurfelGiDescriptorSets();
void recordSurfelGiClearPass(const vk::raii::CommandBuffer &commandBuffer, uint32_t frameIndex);
```

Implement descriptor writes for bindings 0-5 using current frame buffers, `frames.ptAnalysisCounterBuffers[i]`, and `frames.surfelGiDebugImageViews[i]`.

Create a dedicated `surfelGiDescriptorPool` with enough descriptors for `MAX_FRAMES_IN_FLIGHT * 5` storage buffers and `MAX_FRAMES_IN_FLIGHT` storage images. Do not allocate surfel descriptors from the existing RT or denoiser pools.

- [ ] **Step 7: Record clear pass after main path tracing and before surfel generate**

In the path tracer recording function, after the RT dispatch and the existing RT-to-compute G-buffer barrier, call:

```cpp
recordSurfelGiClearPass(commandBuffer, fi);
```

Dispatch:

```cpp
const uint32_t clearItems = std::max(FrameContext::kSurfelGiCellCount,
                                     FrameContext::kSurfelGiMaxSurfels);
const uint32_t groups = (clearItems + 127u) / 128u;
commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelines.surfelGiClearPipeline);
commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                 *pipelines.surfelGiPipelineLayout,
                                 0, {*surfelGiDescriptorSets[frameIndex], *descriptorSets[frameIndex]}, nullptr);
commandBuffer.dispatch(groups, 1, 1);
```

Add a compute-to-compute buffer memory barrier after the pass.

- [ ] **Step 8: Build and test**

Run:

```powershell
cmake --build build --config Debug --target LaphriaEditor
cmake --build build --config Debug --target LaphriaEngineUnitTests
ctest --test-dir build -C Debug --output-on-failure -R LaphriaEngineUnitTests
```

Expected: editor builds, tests pass.

- [ ] **Step 9: Commit**

```powershell
git add CMakeLists.txt src/Core/PipelineCollection.h src/Core/PipelineCollection.cpp src/Core/EngineCore.h src/Core/EngineCore.cpp src/shaders/SurfelClear.slang tests/PathTracerAnalysisTests.cpp
git commit -m "feat: add surfel GI clear pass"
```

---

## Task 5: Generate Surfels From First-Hit G-Buffer

**Files:**
- Modify: `CMakeLists.txt`
- Create: `src/shaders/SurfelGenerate.slang`
- Modify: `src/Core/PipelineCollection.h`
- Modify: `src/Core/PipelineCollection.cpp`
- Modify: `src/Core/EngineCore.cpp`
- Modify: `tests/PathTracerAnalysisTests.cpp`

- [ ] **Step 1: Write the failing generate-pass contract test**

Require these concrete symbols in the UI, analysis, and engine summary files:

```cpp
"surfelGiGeneratePipeline"
"createSurfelGiGeneratePipeline"
"void surfelGenerateMain"
"surfelGiGenerateAttemptsOffset"
"surfelGiGeneratedOffset"
"rtGBufferNormalsViews"
"rtGBufferDepthViews"
```

- [ ] **Step 2: Run tests to verify failure**

Run unit tests. Expected: missing generate pass.

- [ ] **Step 3: Create generate shader**

In `CMakeLists.txt`, add this entry to `SHADER_SOURCES` after creating `src/shaders/SurfelGenerate.slang`:

```cmake
"SurfelGenerate.slang|surfelGenerateMain"
```

Create `src/shaders/SurfelGenerate.slang`:

```hlsl
#include "SurfelCommon.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<SurfelGiRecord> surfelGiRecords;
[[vk::binding(1, 0)]] RWStructuredBuffer<SurfelGiCell> surfelGiCells;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> surfelGiCellSlots;
[[vk::binding(3, 0)]] RWByteAddressBuffer surfelGiCounters;
[[vk::binding(4, 0)]] RWByteAddressBuffer ptAnalysisCounters;
[[vk::binding(5, 0)]] RWTexture2D<float4> surfelGiDebug;

[[vk::binding(6, 0)]] RWTexture2D<float4> gBufferNormalsRead;
[[vk::binding(7, 0)]] RWTexture2D<float> gBufferDepthRead;
[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

[numthreads(8, 8, 1)]
void surfelGenerateMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;
    uint width;
    uint height;
    gBufferDepthRead.GetDimensions(width, height);

    if (pixel.x >= width || pixel.y >= height)
        return;

    float depth = gBufferDepthRead[pixel];
    float3 normal = gBufferNormalsRead[pixel].xyz;

    ptAnalysisCounters.InterlockedAdd(surfelGiGenerateAttemptsOffset, 1u);

    if (depth <= 0.0f || length(normal) < 0.5f)
    {
        ptAnalysisCounters.InterlockedAdd(surfelGiGenerateRejectInvalidOffset, 1u);
        return;
    }

    if (((pixel.x + pixel.y) & 15u) != 0u)
    {
        ptAnalysisCounters.InterlockedAdd(surfelGiGenerateRejectCoverageOffset, 1u);
        return;
    }

    uint surfelIndex;
    surfelGiCounters.InterlockedAdd(0u, 1u, surfelIndex);
    if (surfelIndex >= SURFEL_GI_MAX_SURFELS)
        return;

    float3 worldPos = reconstructSurfelGiFirstHitWorldPosition(
        pixel, uint2(width, height), depth, ubo.cameraPos.xyz, ubo.projInverse, ubo.viewInverse);

    SurfelGiRecord record;
    record.positionRadius = float4(worldPos, SURFEL_GI_MIN_RADIUS);
    record.normalConfidence = float4(normalize(normal), 0.1f);
    record.radianceAge = float4(0.0f, 0.0f, 0.0f, 0.0f);
    record.flags = 1u;
    record.lastSeenFrame = ubo.frameCount;
    record.cellIndex = 0xffffffffu;
    record.pad0 = 0u;
    surfelGiRecords[surfelIndex] = record;

    ptAnalysisCounters.InterlockedAdd(surfelGiGeneratedOffset, 1u);
}
```

Do not use projection-depth reconstruction here. `gBufferDepth` stores `payload.hitT`.

- [ ] **Step 4: Extend descriptor layout for G-buffer reads**

Add bindings 6 and 7 to `surfelGiDescriptorSetLayout` as `vk::DescriptorType::eStorageImage` for normals and depth with compute stage flags. The existing G-buffer images already have storage-image usage and existing compute shaders read them as `RWTexture2D`.

- [ ] **Step 5: Add and record generate pipeline**

Create pipeline `surfelGiGeneratePipeline` with entry point `surfelGenerateMain`.

Record after RT writes the current-frame G-buffer and after the existing RT-to-compute G-buffer barrier:

```cpp
recordSurfelGiGeneratePass(commandBuffer, fi);
```

Use current frame index `fi` for G-buffer image descriptors.

- [ ] **Step 6: Build and test**

Run editor and unit test builds. Expected: successful compilation and tests.

- [ ] **Step 7: Commit**

```powershell
git add CMakeLists.txt src/shaders/SurfelGenerate.slang src/Core/PipelineCollection.h src/Core/PipelineCollection.cpp src/Core/EngineCore.cpp tests/PathTracerAnalysisTests.cpp
git commit -m "feat: generate diagnostic surfels from first-hit data"
```

---

## Task 6: Build Fixed-Capacity Cell Lists

**Files:**
- Modify: `CMakeLists.txt`
- Create: `src/shaders/SurfelBuildCells.slang`
- Modify: `src/Core/PipelineCollection.h`
- Modify: `src/Core/PipelineCollection.cpp`
- Modify: `src/Core/EngineCore.cpp`
- Modify: `tests/PathTracerAnalysisTests.cpp`

- [ ] **Step 1: Write the failing cell-build contract test**

Require:

```cpp
"surfelGiBuildCellsPipeline"
"createSurfelGiBuildCellsPipeline"
"void surfelBuildCellsMain"
"surfelGiCellInsertAttemptsOffset"
"surfelGiCellInsertedOffset"
"surfelGiCellOverflowOffset"
"InterlockedAdd(cell.count"
```

- [ ] **Step 2: Run tests to verify failure**

Expected: missing build-cells pass.

- [ ] **Step 3: Create build-cells shader**

In `CMakeLists.txt`, add this entry to `SHADER_SOURCES` after creating `src/shaders/SurfelBuildCells.slang`:

```cmake
"SurfelBuildCells.slang|surfelBuildCellsMain"
```

Create `src/shaders/SurfelBuildCells.slang`:

```hlsl
#include "SurfelCommon.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<SurfelGiRecord> surfelGiRecords;
[[vk::binding(1, 0)]] RWStructuredBuffer<SurfelGiCell> surfelGiCells;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> surfelGiCellSlots;
[[vk::binding(3, 0)]] RWByteAddressBuffer surfelGiCounters;
[[vk::binding(4, 0)]] RWByteAddressBuffer ptAnalysisCounters;
[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

[numthreads(128, 1, 1)]
void surfelBuildCellsMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint surfelIndex = dispatchThreadID.x;
    if (surfelIndex >= SURFEL_GI_MAX_SURFELS)
        return;

    SurfelGiRecord record = surfelGiRecords[surfelIndex];
    if ((record.flags & 1u) == 0u)
        return;

    ptAnalysisCounters.InterlockedAdd(surfelGiCellInsertAttemptsOffset, 1u);

    int3 cellCoord = calcSurfelGiCell(record.positionRadius.xyz, ubo.cameraPos.xyz);
    if (!isSurfelGiCellValid(cellCoord))
    {
        ptAnalysisCounters.InterlockedAdd(surfelGiCellOverflowOffset, 1u);
        return;
    }

    uint cellIndex = flattenSurfelGiCell(cellCoord);
    uint slotIndex;
    InterlockedAdd(surfelGiCells[cellIndex].count, 1u, slotIndex);

    if (slotIndex >= SURFEL_GI_CELL_SLOT_COUNT)
    {
        surfelGiCells[cellIndex].overflow = 1u;
        ptAnalysisCounters.InterlockedAdd(surfelGiCellOverflowOffset, 1u);
        return;
    }

    surfelGiCellSlots[cellIndex * SURFEL_GI_CELL_SLOT_COUNT + slotIndex] = surfelIndex;
    record.cellIndex = cellIndex;
    surfelGiRecords[surfelIndex] = record;
    ptAnalysisCounters.InterlockedAdd(surfelGiCellInsertedOffset, 1u);
}
```

- [ ] **Step 4: Add and record build-cells pipeline**

Record after generate:

```cpp
recordSurfelGiBuildCellsPass(commandBuffer, fi);
```

Dispatch:

```cpp
const uint32_t groups = (FrameContext::kSurfelGiMaxSurfels + 127u) / 128u;
commandBuffer.dispatch(groups, 1, 1);
```

Add a buffer memory barrier from compute writes to compute reads after the pass.

- [ ] **Step 5: Build and test**

Run editor and unit tests. Expected: success.

- [ ] **Step 6: Commit**

```powershell
git add CMakeLists.txt src/shaders/SurfelBuildCells.slang src/Core/PipelineCollection.h src/Core/PipelineCollection.cpp src/Core/EngineCore.cpp tests/PathTracerAnalysisTests.cpp
git commit -m "feat: build fixed surfel cell lists"
```

---

## Task 7: Add Bounded Surfel Evaluation Diagnostics

**Files:**
- Modify: `CMakeLists.txt`
- Create: `src/shaders/SurfelEvaluate.slang`
- Modify: `src/Core/PipelineCollection.h`
- Modify: `src/Core/PipelineCollection.cpp`
- Modify: `src/Core/EngineCore.cpp`
- Modify: `src/shaders/Denoiser.slang`
- Modify: `tests/PathTracerAnalysisTests.cpp`

- [ ] **Step 1: Write the failing evaluation contract test**

Require:

```cpp
"surfelGiEvaluatePipeline"
"createSurfelGiEvaluatePipeline"
"void surfelEvaluateMain"
"SURFEL_GI_MAX_EVAL_CANDIDATES"
"surfelGiEvalAttemptsOffset"
"surfelGiEvalCellEmptyOffset"
"surfelGiEvalCandidatesOffset"
"surfelGiEvalAcceptedOffset"
```

- [ ] **Step 2: Run tests to verify failure**

Expected: missing evaluation pass.

- [ ] **Step 3: Create evaluate shader**

In `CMakeLists.txt`, add this entry to `SHADER_SOURCES` after creating `src/shaders/SurfelEvaluate.slang`:

```cmake
"SurfelEvaluate.slang|surfelEvaluateMain"
```

Create `src/shaders/SurfelEvaluate.slang`:

```hlsl
#include "SurfelCommon.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<SurfelGiRecord> surfelGiRecords;
[[vk::binding(1, 0)]] RWStructuredBuffer<SurfelGiCell> surfelGiCells;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> surfelGiCellSlots;
[[vk::binding(3, 0)]] RWByteAddressBuffer surfelGiCounters;
[[vk::binding(4, 0)]] RWByteAddressBuffer ptAnalysisCounters;
[[vk::binding(5, 0)]] RWTexture2D<float4> surfelGiDebug;
[[vk::binding(6, 0)]] RWTexture2D<float4> gBufferNormalsRead;
[[vk::binding(7, 0)]] RWTexture2D<float> gBufferDepthRead;
[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

[numthreads(8, 8, 1)]
void surfelEvaluateMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;
    uint width;
    uint height;
    gBufferDepthRead.GetDimensions(width, height);

    if (pixel.x >= width || pixel.y >= height)
        return;

    float depth = gBufferDepthRead[pixel];
    float3 normal = gBufferNormalsRead[pixel].xyz;
    ptAnalysisCounters.InterlockedAdd(surfelGiEvalAttemptsOffset, 1u);

    if (depth <= 0.0f || length(normal) < 0.5f)
    {
        surfelGiDebug[pixel] = float4(0.0f, 0.0f, 0.0f, 1.0f);
        return;
    }

    float3 worldPos = reconstructSurfelGiFirstHitWorldPosition(
        pixel, uint2(width, height), depth, ubo.cameraPos.xyz, ubo.projInverse, ubo.viewInverse);

    int3 cellCoord = calcSurfelGiCell(worldPos, ubo.cameraPos.xyz);
    if (!isSurfelGiCellValid(cellCoord))
    {
        ptAnalysisCounters.InterlockedAdd(surfelGiEvalCellEmptyOffset, 1u);
        surfelGiDebug[pixel] = float4(0.0f, 0.0f, 0.0f, 1.0f);
        return;
    }

    uint cellIndex = flattenSurfelGiCell(cellCoord);
    uint count = min(surfelGiCells[cellIndex].count, SURFEL_GI_CELL_SLOT_COUNT);
    if (count == 0u)
    {
        ptAnalysisCounters.InterlockedAdd(surfelGiEvalCellEmptyOffset, 1u);
        surfelGiDebug[pixel] = float4(0.0f, 0.0f, 0.0f, 1.0f);
        return;
    }

    uint accepted = 0u;
    uint candidateLimit = min(count, SURFEL_GI_MAX_EVAL_CANDIDATES);
    for (uint i = 0u; i < candidateLimit; ++i)
    {
        uint surfelIndex = surfelGiCellSlots[cellIndex * SURFEL_GI_CELL_SLOT_COUNT + i];
        if (surfelIndex == 0xffffffffu)
            continue;

        ptAnalysisCounters.InterlockedAdd(surfelGiEvalCandidatesOffset, 1u);
        SurfelGiRecord surfel = surfelGiRecords[surfelIndex];
        float3 toSurfel = surfel.positionRadius.xyz - worldPos;
        float dist2 = dot(toSurfel, toSurfel);
        if (dist2 <= surfel.positionRadius.w * surfel.positionRadius.w * 16.0f &&
            dot(normalize(normal), surfel.normalConfidence.xyz) > 0.1f)
        {
            accepted += 1u;
        }
    }

    if (accepted > 0u)
        ptAnalysisCounters.InterlockedAdd(surfelGiEvalAcceptedOffset, accepted);

    surfelGiDebug[pixel] = float4(float(accepted) / float(SURFEL_GI_MAX_EVAL_CANDIDATES), 0.0f, float(count) / float(SURFEL_GI_CELL_SLOT_COUNT), 1.0f);
}
```

- [ ] **Step 4: Add debug AOV display**

In `Denoiser.slang`, add a storage image binding for the surfel debug image:

```hlsl
[[vk::binding(15, 0)]] RWTexture2D<float4> surfelGiDebugView;
```

Update `createDenoiserDescriptorSetLayout` and `createDenoiserDescriptorSets` so binding 15 points at `frames.surfelGiDebugImageViews[i]`.

Add a debug AOV branch that displays `surfelGiDebugView[pixel].rgb` for the surfel mode. The color convention is:

- red: accepted gather ratio
- blue: cell occupancy ratio

- [ ] **Step 5: Record evaluate pass after build-cells**

Record:

```cpp
recordSurfelGiEvaluatePass(commandBuffer, fi);
```

after RT and before reprojection using the current-frame G-buffer.

- [ ] **Step 6: Build and test**

Run editor and unit tests. Expected: success.

- [ ] **Step 7: Commit**

```powershell
git add CMakeLists.txt src/shaders/SurfelEvaluate.slang src/shaders/Denoiser.slang src/Core/PipelineCollection.h src/Core/PipelineCollection.cpp src/Core/EngineCore.cpp tests/PathTracerAnalysisTests.cpp
git commit -m "feat: add bounded surfel gather diagnostics"
```

---

## Task 8: Surface UI And Sweep Diagnostics

**Files:**
- Modify: `src/Core/UISystem.h`
- Modify: `src/Core/UISystem.cpp`
- Modify: `src/Core/EngineCore.cpp`
- Modify: `src/Core/PathTracerAnalysis.h`
- Modify: `src/Core/PathTracerAnalysis.cpp`
- Modify: `tests/PathTracerAnalysisTests.cpp`

- [ ] **Step 1: Write the failing UI/summary contract test**

Require:

```cpp
"enableSurfelGi"
"surfelGiDebug"
"Surfel GI Occupancy"
"surfelGiGenerated"
"surfelGiCellOverflow"
"surfelGiEvalAccepted"
```

Look for those strings in `src/Core/UISystem.h`, `src/Core/UISystem.cpp`,
`src/Core/PathTracerAnalysis.h`, `src/Core/PathTracerAnalysis.cpp`, and
`src/Core/EngineCore.cpp`.

- [ ] **Step 2: Run tests to verify failure**

Expected: missing UI and summary strings.

- [ ] **Step 3: Add settings**

In `UISystem.h`, add to path tracer settings:

```cpp
bool enableSurfelGi = false;
bool surfelGiDebug = false;
int surfelGiMaxEvalCandidates = 8;
```

Add debug AOV enum:

```cpp
SurfelGiOccupancy,
SurfelGiGather
```

- [ ] **Step 4: Add UI controls**

In `UISystem.cpp`, add controls near the ReSTIR GI section:

```cpp
ImGui::Checkbox("Surfel GI Cache", &pathTracerSettings.enableSurfelGi);
ImGui::Checkbox("Surfel GI Debug", &pathTracerSettings.surfelGiDebug);
ImGui::SliderInt("Surfel Eval Candidates", &pathTracerSettings.surfelGiMaxEvalCandidates, 1, 16);
```

- [ ] **Step 5: Gate pass recording**

In `EngineCore.cpp`, only record surfel passes after the main ray tracing dispatch and RT-to-compute G-buffer barrier, before reprojection:

```cpp
if (ui.pathTracerSettings.enableSurfelGi || ui.pathTracerSettings.surfelGiDebug)
{
    recordSurfelGiClearPass(commandBuffer, fi);
    recordSurfelGiGeneratePass(commandBuffer, fi);
    recordSurfelGiBuildCellsPass(commandBuffer, fi);
    recordSurfelGiEvaluatePass(commandBuffer, fi);
}
```

- [ ] **Step 6: Add summary fields**

Extend path tracer analysis accumulation and row summary printing with:

```text
surfelGiGenerated
surfelGiCellInserted
surfelGiCellOverflow
surfelGiEvalCandidates
surfelGiEvalAccepted
surfelGiEvalCellEmpty
```

- [ ] **Step 7: Run tests**

Run:

```powershell
cmake --build build --config Debug --target LaphriaEngineUnitTests
ctest --test-dir build -C Debug --output-on-failure -R LaphriaEngineUnitTests
```

Expected: UI/summary contract passes.

- [ ] **Step 8: Commit**

```powershell
git add src/Core/UISystem.h src/Core/UISystem.cpp src/Core/EngineCore.cpp src/Core/PathTracerAnalysis.h src/Core/PathTracerAnalysis.cpp tests/PathTracerAnalysisTests.cpp
git commit -m "feat: expose surfel GI diagnostics"
```

---

## Task 9: Add First Verification Sweep And Kill Gates

**Files:**
- Modify: `src/Core/EngineCore.cpp`
- Modify: `docs/architecture/restir-gi-sponza-handoff.md`
- Modify: `tests/PathTracerAnalysisTests.cpp`

- [ ] **Step 1: Write the failing sweep contract test**

Require one diagnostic-only row named:

```text
Sponza / ... / Sun Receiver Surfel Cache Debug
```

The row must have:

```cpp
settings.enableSurfelGi = true;
settings.surfelGiDebug = true;
```

and must not use `RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_BRIGHT_SURFEL`.

- [ ] **Step 2: Run tests to verify failure**

Expected: missing surfel cache debug row.

- [ ] **Step 3: Add one diagnostic row**

Add a single row after the current Sun Receiver comparison row. Keep it diagnostic-only and do not enable surfel lighting contribution.

- [ ] **Step 4: Add kill gates to docs**

Append this section to `docs/architecture/restir-gi-sponza-handoff.md`:

```markdown
### Persistent Surfel Cache Diagnostic Gates

The surfel cache remains diagnostic-only unless all of these are true:

- `surfelGiGenerated` is non-zero in all three Sponza validation views.
- `surfelGiEvalCandidates / surfelGiEvalAttempts` is bounded below 16 candidates per valid pixel.
- `surfelGiCellOverflow` is below 10% of `surfelGiCellInsertAttempts`.
- Enabling surfel diagnostics does not increase `totalMs` by more than 25% over Sun Receiver.
- Debug AOVs show coherent local coverage rather than sparse isolated points.
```

- [ ] **Step 5: Build and run the focused test**

Run:

```powershell
cmake --build build --config Debug --target LaphriaEngineUnitTests
ctest --test-dir build -C Debug --output-on-failure -R LaphriaEngineUnitTests
```

Expected: pass.

- [ ] **Step 6: Manual validation command**

Run the editor and trigger the focused Sponza sweep from the UI.

Expected first diagnostic read:

```text
surfelGiGenerated > 0
surfelGiEvalCandidates > 0
surfelGiCellOverflow not dominant
totalMs does not explode like the old bright-surfel indexed lookup
```

- [ ] **Step 7: Commit**

```powershell
git add src/Core/EngineCore.cpp docs/architecture/restir-gi-sponza-handoff.md tests/PathTracerAnalysisTests.cpp
git commit -m "test: add surfel GI diagnostic sweep gates"
```

---

## Task 10: Prepare For Per-Surfel Radiance Update Without Enabling Lighting

**Files:**
- Modify: `CMakeLists.txt`
- Create: `src/shaders/SurfelIntegrate.slang`
- Modify: `src/shaders/SurfelCommon.slang`
- Modify: `src/Core/PipelineCollection.h`
- Modify: `src/Core/PipelineCollection.cpp`
- Modify: `src/Core/EngineCore.cpp`
- Modify: `tests/PathTracerAnalysisTests.cpp`

- [ ] **Step 1: Write the failing integrate contract test**

Require:

```cpp
"surfelGiIntegratePipeline"
"void surfelIntegrateMain"
"radianceAge"
"normalConfidence"
"lastSeenFrame"
```

- [ ] **Step 2: Run tests to verify failure**

Expected: missing integrate pass.

- [ ] **Step 3: Create metadata-only integrate pass**

In `CMakeLists.txt`, add this entry to `SHADER_SOURCES` after creating `src/shaders/SurfelIntegrate.slang`:

```cmake
"SurfelIntegrate.slang|surfelIntegrateMain"
```

Create `src/shaders/SurfelIntegrate.slang`:

```hlsl
#include "SurfelCommon.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<SurfelGiRecord> surfelGiRecords;
[[vk::binding(3, 0)]] RWByteAddressBuffer surfelGiCounters;
[[vk::binding(4, 0)]] RWByteAddressBuffer ptAnalysisCounters;
[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

[numthreads(128, 1, 1)]
void surfelIntegrateMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint surfelIndex = dispatchThreadID.x;
    if (surfelIndex >= SURFEL_GI_MAX_SURFELS)
        return;

    SurfelGiRecord record = surfelGiRecords[surfelIndex];
    if ((record.flags & 1u) == 0u)
        return;

    float age = record.radianceAge.w + 1.0f;
    float confidence = min(record.normalConfidence.w + 0.01f, 1.0f);

    record.radianceAge = float4(record.radianceAge.xyz, age);
    record.normalConfidence = float4(record.normalConfidence.xyz, confidence);
    record.lastSeenFrame = ubo.frameCount;
    surfelGiRecords[surfelIndex] = record;
}
```

This pass intentionally does not add lighting. It proves pass scheduling and persistent surfel mutation before ray-budget work.

- [ ] **Step 4: Record integrate after generate and before build-cells**

Order:

```text
clear cells
generate surfels
integrate surfel metadata
build cells
evaluate diagnostics
```

- [ ] **Step 5: Build and test**

Run editor and unit tests. Expected: success.

- [ ] **Step 6: Commit**

```powershell
git add CMakeLists.txt src/shaders/SurfelIntegrate.slang src/shaders/SurfelCommon.slang src/Core/PipelineCollection.h src/Core/PipelineCollection.cpp src/Core/EngineCore.cpp tests/PathTracerAnalysisTests.cpp
git commit -m "feat: add surfel GI integrate pass scaffold"
```

---

## Implementation Notes

- Do not enable surfel lighting contribution in this plan.
- Do not couple surfel gather to reservoir candidate selection in this plan.
- Use current-frame G-buffer for MVP generation/evaluation after the main RT dispatch and before reprojection.
- Keep all gather loops bounded by `SURFEL_GI_MAX_EVAL_CANDIDATES`.
- Treat `cellOverflow` as a design signal, not an error.
- Keep old bright-surfel buffers until the new cache has passed diagnostic sweeps; delete old experiment plumbing in a later cleanup plan.

## Verification Checklist

- [ ] `cmake --build build --config Debug --target LaphriaEngineUnitTests`
- [ ] `ctest --test-dir build -C Debug --output-on-failure -R LaphriaEngineUnitTests`
- [ ] `cmake --build build --config Debug --target LaphriaEditor`
- [ ] Manual Sponza diagnostic row shows `surfelGiGenerated > 0`.
- [ ] Manual Sponza diagnostic row shows bounded `surfelGiEvalCandidates`.
- [ ] Manual Sponza diagnostic row does not reproduce the old `brightSurfelIndexedQuery ~= 28M` failure mode.

## Self-Review

- Spec coverage: the plan disables the failed receiver-side surfel proposal, adds persistent surfel resources, creates fixed-capacity cell indexing, adds bounded evaluation diagnostics, exposes UI/sweep counters, and prepares a later radiance update path.
- Placeholder scan: no task depends on an undefined later task; lighting contribution is explicitly out of scope.
- Type consistency: `SurfelGiRecord`, `SurfelGiCell`, `surfelGi*` counters, and pipeline names are consistent across tasks.
