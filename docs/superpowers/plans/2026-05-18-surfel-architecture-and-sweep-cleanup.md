# Reservoir Surfel Cleanup And Proposal Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the standalone `Surfel*.slang` compute GI prototype, reduce Sponza sweeps to the reservoir rows we actually need, and then evaluate the remaining bright-surfel reservoir proposal as a possible foundation for a future reservoir-owned receiver cache.

**Architecture:** Keep ReSTIR GI/reservoir shading as the only active indirect-lighting estimator. Treat the current compute surfel implementation as legacy scaffolding and remove it from build, runtime, UI, diagnostics, tests, and sweeps. After that cleanup, evaluate the `Raygen.slang` bright-surfel reservoir proposal separately against the reservoir estimator contract before deciding whether to keep, rename, refactor, or delete it.

**Tech Stack:** C++ engine/UI code, Vulkan ray tracing and compute pipeline setup, Slang shaders, source-contract unit tests in `tests/PathTracerAnalysisTests.cpp`, Sponza PT/GI audit sweeps.

---

## Architectural Decision

The reference surfel repos describe a persistent world-space cache with lifecycle, allocation, indexing, radiance update, coverage, and bounded query behavior. The current Laphria `Surfel*.slang` compute path is not the right basis for that architecture: it is a parallel debug GI experiment with its own pass graph, buffers, debug AOVs, dense-cell behavior, and sweep rows.

Remove that compute subsystem first so future planning is not biased by its names or failure modes.

Keep the reservoir-side bright-surfel proposal temporarily because it is already embedded in the reservoir candidate path. Do not treat "bright surfel" as the architecture we are committing to: the architecture is useful receiver/cache evidence feeding reservoir candidate generation, with brightness only one possible training signal. It must be evaluated after the compute surfel removal with stricter criteria:

- Does it store useful receiver/radiance evidence produced by the reservoir/path tracer?
- Does it prove a general receiver-cache direction, or only a high-luma heuristic that should not become an abstraction?
- Does it reduce `localRejectNoLight` or increase useful accepted candidates?
- Does it preserve audit mean through the reservoir estimator?
- Is its naming/shape still misleading enough that refactoring would be more expensive than replacement?

---

## Reference Architecture Notes

Reference repos used:

- `.codex_refs/SurfelPlus`
- `.codex_refs/SurfelGI`

Useful ideas to carry forward later:

- Persistent records with explicit lifecycle.
- Per-frame cell index rebuild over persistent records.
- Bounded query over dense cells rather than dense-cell rejection.
- Coverage, age, radius, and rejection diagnostics.
- Clear ownership: cache proposes or supplies reconnectable receiver evidence; reservoir remains the final estimator.

Ideas intentionally not carried into this cleanup:

- Full radiance atlas/MSME integration.
- Direct surfel lighting contribution to final color.
- Reusing the existing `SurfelEvaluate.slang` output path.

---

## Files To Modify

Remove standalone compute surfel implementation:

- Delete: `src/shaders/SurfelClear.slang`
- Delete: `src/shaders/SurfelGenerate.slang`
- Delete: `src/shaders/SurfelCountCells.slang`
- Delete: `src/shaders/SurfelAllocateCells.slang`
- Delete: `src/shaders/SurfelIntegrate.slang`
- Delete: `src/shaders/SurfelBuildCells.slang`
- Delete: `src/shaders/SurfelEvaluate.slang`
- Delete or keep only if still needed by no remaining shader: `src/shaders/SurfelCommon.slang`

Remove compute surfel build and runtime plumbing:

- Modify: `CMakeLists.txt`
- Modify: `src/Core/PipelineCollection.h`
- Modify: `src/Core/PipelineCollection.cpp`
- Modify: `src/Core/EngineCore.h`
- Modify: `src/Core/EngineCore.cpp`
- Modify: `src/Core/FrameContext.h`
- Modify: `src/Core/FrameContext.cpp`
- Modify: `src/Core/UISystem.h`
- Modify: `src/Core/UISystem.cpp`
- Modify: `src/Core/EngineAuxiliary.h`
- Modify: `src/Core/PathTracerAnalysis.h`
- Modify: `src/Core/PathTracerAnalysis.cpp`
- Modify: `src/shaders/Denoiser.slang`
- Modify: `tests/PathTracerAnalysisTests.cpp`

Evaluate remaining reservoir bright-surfel proposal:

- Modify: `src/shaders/Raygen.slang`
- Modify: `src/Core/EngineCore.cpp`
- Modify: `src/Core/EngineCore.h`
- Modify: `src/Core/UISystem.h`
- Modify: `src/Core/UISystem.cpp`
- Modify: `src/Core/EngineAuxiliary.h`
- Modify if derived analysis helpers become necessary: `src/Core/PathTracerAnalysis.h`
- Modify if derived analysis helpers become necessary: `src/Core/PathTracerAnalysis.cpp`
- Modify: `tests/PathTracerAnalysisTests.cpp`
- Optional docs: `docs/architecture/reservoir-gi-estimator-contract.md`
- Optional docs: `docs/architecture/restir-gi-approach-decision.md`
- Optional docs: `docs/architecture/restir-gi-sponza-handoff.md`

---

## Task 1: Freeze Current Reservoir Estimator Result

**Files:**

- Modify: `tests/PathTracerAnalysisTests.cpp`
- Modify: `src/shaders/Raygen.slang`

- [ ] **Step 1: Verify current reservoir-only diff is intentional**

Run:

```powershell
git diff -- src/shaders/Raygen.slang tests/PathTracerAnalysisTests.cpp
```

Expected:

- `Raygen.slang` computes `currentEstimatorLuma` from `reservoirTotal`.
- `recordReservoirGiEstimatorAudit(..., reservoirProbeScale)` still records the old diagnostic scale.
- `result.totalContribution = reservoirTotal;`
- `result.secondaryDirectSunContribution = reservoirSecondarySun;`
- `tests/PathTracerAnalysisTests.cpp` expects the unscaled estimator path.
- No `Surfel*.slang` behavior changes are present in this diff.

- [ ] **Step 2: Build tests**

Run:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests'
```

Expected: exit code `0`.

- [ ] **Step 3: Run tests**

Run:

```powershell
.\cmake-build-debug\LaphriaEngineUnitTests.exe
```

Expected: exit code `0`.

- [ ] **Step 4: Build editor/shaders**

Run:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEditor'
```

Expected: exit code `0`, including `Raygen.slang` compile.

- [ ] **Step 5: Commit reservoir estimator result**

Run:

```powershell
git add src/shaders/Raygen.slang tests/PathTracerAnalysisTests.cpp
git commit -m "fix: remove reservoir estimator probe damping"
```

Expected: commit succeeds and does not include `.idea`, `.codex_refs`, or unrelated docs.

---

## Task 2: Remove Standalone Surfel Compute Contracts

**Files:**

- Modify: `tests/PathTracerAnalysisTests.cpp`

- [ ] **Step 1: Replace positive surfel-compute contracts with removal contracts**

In `tests/PathTracerAnalysisTests.cpp`, remove tests/helpers whose purpose is to require the old compute surfel path:

```cpp
extractSurfelEvalCandidateSliderRange
requirePersistentSurfelCounterLayout
requirePersistentSurfelFrameResources
requireCompactSurfelCellIndexListBuffers
requireSurfelClearPassContracts
requireSurfelGeneratePassContracts
requireSurfelBuildCellsPassContracts
requireSurfelIntegratePassContracts
requireSurfelEvaluatePassContracts
requireSurfelGiDebugAovChannelSplit
requireSurfelGiSlotCapacityAlignment
requireCompactSurfelGridDiagnosticPlumbing
testSurfelGiDiagnosticRatios
```

Remove all source reads for deleted surfel compute shaders before adding the new removal contract:

```cpp
const std::string surfelCommon = readTextFile(sourceRoot / "src" / "shaders" / "SurfelCommon.slang");
const std::string surfelClear = readTextFile(sourceRoot / "src" / "shaders" / "SurfelClear.slang");
const std::string surfelGenerate = readTextFile(sourceRoot / "src" / "shaders" / "SurfelGenerate.slang");
const std::string surfelIntegrate = readTextFile(sourceRoot / "src" / "shaders" / "SurfelIntegrate.slang");
const std::string surfelBuildCells = readTextFile(sourceRoot / "src" / "shaders" / "SurfelBuildCells.slang");
const std::string surfelEvaluate = readTextFile(sourceRoot / "src" / "shaders" / "SurfelEvaluate.slang");
```

Keep reading `src/shaders/Denoiser.slang`, because the removal contract must verify its old surfel debug AOV dependency is gone.

Remove surfel-specific assertions from remaining tests that still stay valid after the cleanup, especially the path tracer debug AOV contract. That test should continue to prove the remaining debug AOVs are selectable, but it must no longer require `SurfelGiOccupancy`, `SurfelGiGather`, or `surfelGiDebugView`.

Add a new helper:

```cpp
bool requireStandaloneSurfelComputeRemoved(const std::string &cmakeLists,
                                           const std::string &pipelineHeader,
                                           const std::string &pipelineSource,
                                           const std::string &engineHeader,
                                           const std::string &engineCore,
                                           const std::string &frameContextHeader,
                                           const std::string &frameContextSource,
                                           const std::string &uiHeader,
                                           const std::string &uiSource,
                                           const std::string &denoiser,
                                           const std::string &engineAuxiliaryHeader,
                                           const std::string &pathTracerAnalysisHeader,
                                           const std::string &pathTracerAnalysisSource)
{
    const char *forbidden[] = {
        "SurfelClear.slang",
        "SurfelGenerate.slang",
        "SurfelCountCells.slang",
        "SurfelAllocateCells.slang",
        "SurfelIntegrate.slang",
        "SurfelBuildCells.slang",
        "SurfelEvaluate.slang",
        "SurfelCommon.slang",
        "createSurfelGiDescriptorSetLayout",
        "createSurfelGiPipelineLayout",
        "createSurfelGiClearPipeline",
        "createSurfelGiGeneratePipeline",
        "createSurfelGiCountCellsPipeline",
        "createSurfelGiAllocateCellsPipeline",
        "createSurfelGiIntegratePipeline",
        "createSurfelGiBuildCellsPipeline",
        "createSurfelGiEvaluatePipeline",
        "createSurfelGiDescriptorSets",
        "recordSurfelGiClearPass",
        "recordSurfelGiGeneratePass",
        "recordSurfelGiCountCellsPass",
        "recordSurfelGiAllocateCellsPass",
        "recordSurfelGiIntegratePass",
        "recordSurfelGiBuildCellsPass",
        "recordSurfelGiEvaluatePass",
        "resetSurfelGiRecordBuffers",
        "createSurfelGiBuffers",
        "surfelGiRecordBuffers",
        "surfelGiCellBuffers",
        "surfelGiCellSlotBuffers",
        "surfelGiCounterBuffers",
        "surfelGiDebugImages",
        "surfelGiDebugImageViews",
        "surfelGiDebugView",
        "enableSurfelGi",
        "surfelGiDebug",
        "surfelGiMaxEvalCandidates",
        "SurfelGiOccupancy",
        "SurfelGiGather",
        "surfelGiGenerateAttempts",
        "surfelGiEvalAttempts",
        "Surfel GI Cache",
        "Surfel GI Debug",
        "Surfel Eval Candidates"
    };

    const std::string combined = cmakeLists + pipelineHeader + pipelineSource + engineHeader +
                                 engineCore + frameContextHeader + frameContextSource + uiHeader +
                                 uiSource + denoiser + engineAuxiliaryHeader + pathTracerAnalysisHeader +
                                 pathTracerAnalysisSource;

    for (const char *symbol : forbidden)
    {
        if (containsText(combined, symbol))
        {
            std::cerr << "standalone surfel compute path must be removed; found " << symbol << "\n";
            return false;
        }
    }

    return true;
}
```

Call this helper from the existing source-contract test after loading the same files it already reads.

- [ ] **Step 2: Keep bright-surfel reservoir contracts in place**

Do not remove these helpers yet:

```cpp
brightSurfelCombineUsesTargetWeight
requireBrightSurfelProposalDisabledForSweeps
requireIndexedBrightSurfelShaderContracts
requireIndexedBrightSurfelDiagnosticPlumbing
```

Expected: this preserves the reservoir-side proposal until Task 6. Task 6 intentionally replaces `requireBrightSurfelProposalDisabledForSweeps` when it adds explicit shadow/proposal evaluation sweeps.

- [ ] **Step 3: Run tests to confirm RED**

Run:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests'
.\cmake-build-debug\LaphriaEngineUnitTests.exe
```

Expected: source-contract test fails with `standalone surfel compute path must be removed`.

---

## Task 3: Remove Standalone Surfel Compute Runtime

**Files:**

- Delete: `src/shaders/SurfelClear.slang`
- Delete: `src/shaders/SurfelGenerate.slang`
- Delete: `src/shaders/SurfelCountCells.slang`
- Delete: `src/shaders/SurfelAllocateCells.slang`
- Delete: `src/shaders/SurfelIntegrate.slang`
- Delete: `src/shaders/SurfelBuildCells.slang`
- Delete: `src/shaders/SurfelEvaluate.slang`
- Delete: `src/shaders/SurfelCommon.slang`
- Modify: `src/shaders/Denoiser.slang`
- Modify: `CMakeLists.txt`
- Modify: `src/Core/PipelineCollection.h`
- Modify: `src/Core/PipelineCollection.cpp`
- Modify: `src/Core/EngineCore.h`
- Modify: `src/Core/EngineCore.cpp`
- Modify: `src/Core/FrameContext.h`
- Modify: `src/Core/FrameContext.cpp`
- Modify: `src/Core/UISystem.h`
- Modify: `src/Core/UISystem.cpp`
- Modify: `src/Core/EngineAuxiliary.h`
- Modify: `src/Core/PathTracerAnalysis.h`
- Modify: `src/Core/PathTracerAnalysis.cpp`

- [ ] **Step 1: Remove surfel shader compilation**

In `CMakeLists.txt`, remove these entries from `SHADER_SOURCES`:

```cmake
"SurfelClear.slang|surfelClearMain"
"SurfelGenerate.slang|surfelGenerateMain"
"SurfelCountCells.slang|surfelCountCellsMain"
"SurfelAllocateCells.slang|surfelAllocateCellsMain"
"SurfelIntegrate.slang|surfelIntegrateMain"
"SurfelBuildCells.slang|surfelBuildCellsMain"
"SurfelEvaluate.slang|surfelEvaluateMain"
```

Remove `SurfelCommon.slang` from `SURFEL_SHADER_INCLUDE_DEPS`. If the variable becomes generic or empty, rename it to `SHADER_INCLUDE_DEPS` and keep only includes needed by remaining shaders.

- [ ] **Step 2: Delete surfel shader files**

Delete:

```text
src/shaders/SurfelClear.slang
src/shaders/SurfelGenerate.slang
src/shaders/SurfelCountCells.slang
src/shaders/SurfelAllocateCells.slang
src/shaders/SurfelIntegrate.slang
src/shaders/SurfelBuildCells.slang
src/shaders/SurfelEvaluate.slang
src/shaders/SurfelCommon.slang
```

- [ ] **Step 3: Remove surfel compute pipeline creation**

In `src/Core/PipelineCollection.h`, remove:

```cpp
vk::raii::DescriptorSetLayout surfelGiDescriptorSetLayout{nullptr};
vk::raii::PipelineLayout surfelGiPipelineLayout{nullptr};
vk::raii::Pipeline surfelGiClearPipeline{nullptr};
vk::raii::Pipeline surfelGiGeneratePipeline{nullptr};
vk::raii::Pipeline surfelGiCountCellsPipeline{nullptr};
vk::raii::Pipeline surfelGiAllocateCellsPipeline{nullptr};
vk::raii::Pipeline surfelGiIntegratePipeline{nullptr};
vk::raii::Pipeline surfelGiBuildCellsPipeline{nullptr};
vk::raii::Pipeline surfelGiEvaluatePipeline{nullptr};
```

Remove declarations:

```cpp
void createSurfelGiDescriptorSetLayout(const VulkanDevice &dev);
void createSurfelGiPipelineLayout(const VulkanDevice &dev);
void createSurfelGiClearPipeline(const VulkanDevice &dev);
void createSurfelGiGeneratePipeline(const VulkanDevice &dev);
void createSurfelGiCountCellsPipeline(const VulkanDevice &dev);
void createSurfelGiAllocateCellsPipeline(const VulkanDevice &dev);
void createSurfelGiIntegratePipeline(const VulkanDevice &dev);
void createSurfelGiBuildCellsPipeline(const VulkanDevice &dev);
void createSurfelGiEvaluatePipeline(const VulkanDevice &dev);
```

In `src/Core/PipelineCollection.cpp`, delete the implementations of those functions and remove calls to them from descriptor set layout creation or pipeline setup.

Also remove the global descriptor layout binding for the surfel debug image from `src/Core/PipelineCollection.cpp`:

```cpp
vk::DescriptorSetLayoutBinding{.binding = 15, .descriptorType = vk::DescriptorType::eStorageImage, ...} // surfel GI debug view
```

Remove the matching descriptor image info/write from `src/Core/EngineCore.cpp`:

```cpp
{.imageView = *frames.surfelGiDebugImageViews[i], .imageLayout = vk::ImageLayout::eGeneral}
```

- [ ] **Step 4: Remove surfel compute frame resources**

In `src/Core/FrameContext.h`, remove:

```cpp
kSurfelGiMaxSurfels
kSurfelGiGridDim
kSurfelGiCellCount
kSurfelGiMaxCellMembershipsPerSurfel
kSurfelGiCellToSurfelCapacity
kSurfelGiRecordSize
kSurfelGiCellSize
kSurfelGiCellToSurfelIndexSize
kSurfelGiCounterSize
surfelGiRecordBuffers
surfelGiCellBuffers
surfelGiCellSlotBuffers
surfelGiCounterBuffers
surfelGiDebugImages
surfelGiDebugImageViews
createSurfelGiBuffers
```

In `src/Core/FrameContext.cpp`, remove `createSurfelGiBuffers`, calls to it, swapchain cleanup of surfel images/views, and destruction of surfel buffers.

- [ ] **Step 5: Remove surfel compute descriptors and passes from EngineCore**

In `src/Core/EngineCore.h`, remove:

```cpp
vk::raii::DescriptorPool surfelGiDescriptorPool{nullptr};
std::vector<vk::raii::DescriptorSet> surfelGiDescriptorSets;
void createSurfelGiDescriptorSets();
void recordSurfelGiClearPass(...);
void recordSurfelGiGeneratePass(...);
void recordSurfelGiCountCellsPass(...);
void recordSurfelGiAllocateCellsPass(...);
void recordSurfelGiIntegratePass(...);
void recordSurfelGiBuildCellsPass(...);
void recordSurfelGiEvaluatePass(...);
void resetSurfelGiRecordBuffers();
```

In `src/Core/EngineCore.cpp`, remove:

```cpp
pipelines.createSurfelGiClearPipeline(vulkan);
pipelines.createSurfelGiGeneratePipeline(vulkan);
pipelines.createSurfelGiCountCellsPipeline(vulkan);
pipelines.createSurfelGiAllocateCellsPipeline(vulkan);
pipelines.createSurfelGiIntegratePipeline(vulkan);
pipelines.createSurfelGiBuildCellsPipeline(vulkan);
pipelines.createSurfelGiEvaluatePipeline(vulkan);
createSurfelGiDescriptorSets();
recordSurfelGiClearPass(commandBuffer, fi);
recordSurfelGiGeneratePass(commandBuffer, fi);
recordSurfelGiCountCellsPass(commandBuffer, fi);
recordSurfelGiAllocateCellsPass(commandBuffer, fi);
recordSurfelGiBuildCellsPass(commandBuffer, fi);
recordSurfelGiIntegratePass(commandBuffer, fi);
recordSurfelGiEvaluatePass(commandBuffer, fi);
resetSurfelGiRecordBuffers();
```

Delete the implementations of the surfel descriptor-set and record-pass functions.

- [ ] **Step 6: Remove surfel compute UI and debug AOVs**

In `src/Core/UISystem.h`, remove:

```cpp
PathTracerDebugAov::SurfelGiOccupancy
PathTracerDebugAov::SurfelGiGather
bool enableSurfelGi
bool surfelGiDebug
int surfelGiMaxEvalCandidates
```

Remove surfel compute perf fields:

```cpp
surfelGiGenerateAttempts
surfelGiGenerated
surfelGiGenerateRejectInvalid
surfelGiGenerateRejectCoverage
surfelGiCellInsertAttempts
surfelGiCellInserted
surfelGiCellOverflow
surfelGiCellTotalMemberships
surfelGiCellAllocatedMemberships
surfelGiCellAllocationOverflow
surfelGiCellNonEmpty
surfelGiCellMaxPopulation
surfelGiEvalDenseCell
surfelGiEvalDenseCellSkipped
surfelGiEvalAttempts
surfelGiEvalCandidates
surfelGiEvalAccepted
surfelGiEvalCellEmpty
```

In `src/Core/UISystem.cpp`, remove:

```cpp
"Surfel GI Occupancy"
"Surfel GI Gather"
ImGui::Checkbox("Surfel GI Cache", ...)
ImGui::Checkbox("Surfel GI Debug", ...)
ImGui::SliderInt("Surfel Eval Candidates", ...)
ImGui::Text("Surfel GI ...", ...)
```

Keep bright-surfel UI diagnostics for Task 6.

In `src/shaders/Denoiser.slang`, remove:

```slang
[[vk::binding(15, 0)]] RWTexture2D<float4> surfelGiDebugView;
surfelGiDebugView[pixel]
```

Remove the `SurfelGiOccupancy` and `SurfelGiGather` cases from `selectPathTracerDebugAovOutput`.

- [ ] **Step 7: Remove surfel compute counters and ratio helpers**

In `src/Core/EngineAuxiliary.h`, remove surfel compute counter fields:

```cpp
surfelGiClearDispatches
surfelGiGenerateAttempts
surfelGiGenerated
surfelGiGenerateRejectInvalid
surfelGiGenerateRejectCoverage
surfelGiCellInsertAttempts
surfelGiCellInserted
surfelGiCellOverflow
surfelGiCellTotalMemberships
surfelGiCellAllocatedMemberships
surfelGiCellAllocationOverflow
surfelGiCellNonEmpty
surfelGiCellMaxPopulation
surfelGiEvalDenseCell
surfelGiEvalDenseCellSkipped
surfelGiEvalAttempts
surfelGiEvalCandidates
surfelGiEvalAccepted
surfelGiEvalCellEmpty
```

In `src/Core/PathTracerAnalysis.h` and `src/Core/PathTracerAnalysis.cpp`, remove `SurfelGiDiagnosticCounters`, `SurfelGiDiagnosticRatios`, and `computeSurfelGiDiagnosticRatios` if they only serve the deleted compute path.

In `src/Core/EngineCore.cpp`, remove collection, accumulation, and row-summary logging for the same `surfelGi...` fields.

- [ ] **Step 8: Remove surfel compute sweep row**

In `src/Core/EngineCore.cpp`, remove:

```cpp
reservoirMixedTemporalSpatialBudget2SunReceiverSurfelCacheDebugRow
ptExperimentRows.push_back(reservoirMixedTemporalSpatialBudget2SunReceiverSurfelCacheDebugRow);
```

Do not add a new surfel validation sweep yet. That comes only after Task 6 decides whether the bright-surfel proposal survives.

- [ ] **Step 9: Run tests to confirm GREEN**

Run:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests'
.\cmake-build-debug\LaphriaEngineUnitTests.exe
```

Expected: exit code `0`.

- [ ] **Step 10: Build editor**

Run:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEditor'
```

Expected: exit code `0`; no `Surfel*.slang` shader compile steps appear.

- [ ] **Step 11: Commit compute surfel removal**

Run:

```powershell
git add CMakeLists.txt src/Core/PipelineCollection.h src/Core/PipelineCollection.cpp src/Core/EngineCore.h src/Core/EngineCore.cpp src/Core/FrameContext.h src/Core/FrameContext.cpp src/Core/UISystem.h src/Core/UISystem.cpp src/Core/EngineAuxiliary.h src/Core/PathTracerAnalysis.h src/Core/PathTracerAnalysis.cpp src/shaders/Denoiser.slang tests/PathTracerAnalysisTests.cpp
git add -u src/shaders
git commit -m "refactor: remove standalone surfel gi compute path"
```

Expected: commit succeeds.

---

## Task 4: Clean Sponza Sweeps To Reservoir Rows Only

**Files:**

- Modify: `tests/PathTracerAnalysisTests.cpp`
- Modify: `src/Core/EngineCore.cpp`

- [ ] **Step 1: Write/update source contract for lean default sweep**

In `tests/PathTracerAnalysisTests.cpp`, replace the current Sponza sweep row contract with required rows only:

```cpp
const char *requiredDefaultRows[] = {
    "ptExperimentRows.push_back(reservoirMixedTemporalSpatialBudget2Row);",
    "ptExperimentRows.push_back(reservoirMixedTemporalSpatialBudget2SunReceiverRow);",
    "ptExperimentRows.push_back(reservoirMixedSingleFrameSunReceiverRow);",
    "ptExperimentRows.push_back(reservoirAuditSingleFrame1cCurrentRow);",
    "ptExperimentRows.push_back(reservoirAuditTemporalSpatialStaticRow);"
};
```

Forbid:

```cpp
const char *forbiddenDefaultRows[] = {
    "reservoirMixedTemporalSpatialBudget2SunReceiverSurfelCacheDebugRow",
    "reservoirMixedTemporalSpatialBudget2SunReceiverEnvFirstTwoRow",
    "reservoirMixedTemporalSpatialBudget2SunReceiverEnvFirstTwoCacheContinuationRow",
    "reservoirAuditSingleFrame1cNoProbeScaleRow",
    "reservoirAuditSingleFrame2cCurrentRow",
    "reservoirAuditSingleFrame2cRisRow",
    "reservoirAuditTemporalStaticRow"
};
```

Expected: the test fails until the sweep is trimmed.

- [ ] **Step 2: Trim row construction**

In `src/Core/EngineCore.cpp`, keep only these pushes per Sponza scenario:

```cpp
ptExperimentRows.push_back(reservoirMixedTemporalSpatialBudget2Row);
ptExperimentRows.push_back(reservoirMixedTemporalSpatialBudget2SunReceiverRow);
ptExperimentRows.push_back(reservoirMixedSingleFrameSunReceiverRow);
ptExperimentRows.push_back(reservoirAuditSingleFrame1cCurrentRow);
ptExperimentRows.push_back(reservoirAuditTemporalSpatialStaticRow);
```

Delete construction of rows that are now forbidden by Step 1.

- [ ] **Step 3: Run tests and editor build**

Run:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests'
.\cmake-build-debug\LaphriaEngineUnitTests.exe
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEditor'
```

Expected: all exit code `0`.

- [ ] **Step 4: Commit sweep cleanup**

Run:

```powershell
git add src/Core/EngineCore.cpp tests/PathTracerAnalysisTests.cpp
git commit -m "test: trim sponza reservoir audit sweep"
```

Expected: commit succeeds.

---

## Task 5: Update Architecture Docs After Compute Surfel Removal

**Files:**

- Modify: `docs/architecture/reservoir-gi-estimator-contract.md`
- Modify: `docs/architecture/restir-gi-approach-decision.md`
- Optional modify: `docs/architecture/restir-gi-sponza-handoff.md`

- [ ] **Step 1: Update estimator contract**

In `docs/architecture/reservoir-gi-estimator-contract.md`, change `Current Implementation Questions` so the removed damping is no longer listed as a current question. Replace:

```markdown
- All returned reservoir contribution is additionally multiplied by `candidateCount / (candidateCount + 1)`.
```

with:

```markdown
- The old `candidateCount / (candidateCount + 1)` reservoir output damping was removed after audit rows showed it was artificial estimator darkening.
```

- [ ] **Step 2: Update approach decision note**

In `docs/architecture/restir-gi-approach-decision.md`, append:

```markdown
## 2026-05-18 Surfel Cleanup Update

The standalone compute `Surfel*.slang` GI prototype is no longer the basis for the sparse receiver/radiance cache direction. It was removed from the active plan to avoid confusing a parallel surfel-lighting experiment with a reservoir-owned proposal/cache system.

The remaining bright-surfel reservoir proposal in `Raygen.slang` will be evaluated separately. It should be kept only if it behaves like useful receiver evidence for reservoir candidate generation and preserves the reservoir estimator audit contract.
```

- [ ] **Step 3: Run docs grep sanity check**

Run:

```powershell
rg "Surfel\\*.slang|Surfel GI Cache|surfel compute|dense surfel" docs
```

Expected: remaining mentions are historical notes or this cleanup decision, not implementation instructions to continue the deleted compute path.

- [ ] **Step 4: Commit docs update**

Run:

```powershell
git add docs/architecture/reservoir-gi-estimator-contract.md docs/architecture/restir-gi-approach-decision.md docs/architecture/restir-gi-sponza-handoff.md
git commit -m "docs: clarify reservoir-owned surfel cache direction"
```

Expected: commit succeeds if touched docs changed.

---

## Task 6: Evaluate Bright-Surfel Reservoir Proposal

**Files:**

- Modify: `tests/PathTracerAnalysisTests.cpp`
- Modify: `src/shaders/Raygen.slang`
- Modify: `src/Core/EngineCore.cpp`
- Modify: `src/Core/EngineCore.h`
- Modify: `src/Core/UISystem.h`
- Modify: `src/Core/UISystem.cpp`
- Modify: `src/Core/EngineAuxiliary.h`
- Modify if derived analysis helpers become necessary: `src/Core/PathTracerAnalysis.h`
- Modify if derived analysis helpers become necessary: `src/Core/PathTracerAnalysis.cpp`

- [ ] **Step 1: Add explicit review and estimator-contract checks**

Before enabling the proposal, inspect `src/shaders/Raygen.slang` and verify the bright-surfel path still behaves as a reservoir candidate source, not as a second lighting estimator.

Run:

```powershell
rg -n "brightSurfel|BrightSurfel|enableBrightSurfelProposal|totalContribution|reservoirTotal|selectedBrightSurfel" src/shaders/Raygen.slang
```

Expected:

- No bright-surfel radiance is added directly to `result.totalContribution`.
- No bright-surfel-only scale, clamp, or damping is applied after reservoir contribution is computed.
- Bright-surfel contribution can reach final lighting only through the same reservoir candidate selection/contribution path as other proposal sources.
- If this review finds a direct lighting shortcut, stop Task 6 and replace it with a follow-up delete/refactor plan for the bright-surfel path.

In `tests/PathTracerAnalysisTests.cpp`, replace the old `requireBrightSurfelProposalDisabledForSweeps` call with a helper that forces the bright-surfel path to stay out of defaults while allowing only explicit evaluation sweeps to enable it:

```cpp
bool requireBrightSurfelOnlyInExplicitEvaluationRows(const std::string &engineCore,
                                                     const std::string &uiHeader,
                                                     const std::string &raygen)
{
    if (containsText(raygen, "const bool enableBrightSurfelProposal = false"))
    {
        std::cerr << "bright-surfel evaluation cannot work while the shader hard-disables the proposal\n";
        return false;
    }

    const std::string defaultSweepSource =
        extractFunctionBody(engineCore, "void EngineCore::startPathTracerSponzaGiPerfSweep(");
    const std::string shadowSweepSource =
        extractFunctionBody(engineCore, "void EngineCore::startPathTracerBrightSurfelShadowEvaluationSweep(");
    const std::string proposalSweepSource =
        extractFunctionBody(engineCore, "void EngineCore::startPathTracerBrightSurfelProposalEvaluationSweep(");

    if (defaultSweepSource.empty() || shadowSweepSource.empty() || proposalSweepSource.empty())
    {
        std::cerr << "bright-surfel evaluation must be isolated in explicit shadow and proposal sweeps\n";
        return false;
    }

    if (containsText(defaultSweepSource, "MixedCosineSunReceiverBrightSurfel"))
    {
        std::cerr << "default Sponza sweep must not contain bright-surfel proposal rows\n";
        return false;
    }

    if (containsText(uiHeader, "reservoirGiProposalMode = PathTracerReservoirGiProposalMode::MixedCosineSunReceiverBrightSurfel"))
    {
        std::cerr << "bright-surfel proposal mode must not be the default UI proposal mode\n";
        return false;
    }

    if (!containsText(shadowSweepSource, "MixedCosineSunReceiverBrightSurfel") ||
        !containsText(proposalSweepSource, "MixedCosineSunReceiverBrightSurfel"))
    {
        std::cerr << "explicit evaluation sweeps must contain bright-surfel proposal rows\n";
        return false;
    }

    const char *requiredShadowRows[] = {
        "Bright Surfel Evaluation / Baseline Sun Receiver",
        "Bright Surfel Evaluation / Baseline Static Audit",
        "Bright Surfel Evaluation / Shadow Diagnostics"};
    for (const char *rowName : requiredShadowRows)
    {
        if (!containsText(shadowSweepSource, rowName))
        {
            std::cerr << "explicit shadow sweep missing row: " << rowName << "\n";
            return false;
        }
    }

    const char *requiredProposalRows[] = {
        "Bright Surfel Evaluation / Baseline Static Audit",
        "Bright Surfel Evaluation / Proposal Enabled",
        "Bright Surfel Evaluation / Proposal Enabled Static Audit"};
    for (const char *rowName : requiredProposalRows)
    {
        if (!containsText(proposalSweepSource, rowName))
        {
            std::cerr << "explicit proposal sweep missing row: " << rowName << "\n";
            return false;
        }
    }

    if (!containsText(shadowSweepSource, "reservoirGiBrightSurfelShadowOnly = true") ||
        containsText(proposalSweepSource, "reservoirGiBrightSurfelShadowOnly = true"))
    {
        std::cerr << "bright-surfel shadow-only mode belongs only in the shadow sweep\n";
        return false;
    }

    if (!containsText(shadowSweepSource, "PathTracerReservoirGiEstimatorAuditMode::Current") ||
        !containsText(proposalSweepSource, "PathTracerReservoirGiEstimatorAuditMode::Current"))
    {
        std::cerr << "bright-surfel evaluation must include paired static audit rows\n";
        return false;
    }

    if (!containsText(raygen, "allowBrightSurfelSelection") ||
        !containsText(raygen, "useBrightSurfelProposal && !reservoirGiBrightSurfelShadowOnly"))
    {
        std::cerr << "shadow-only bright-surfel diagnostics must be gated out of reservoir selection\n";
        return false;
    }

    return true;
}
```

Call it from the source-contract test after removing the old `requireBrightSurfelProposalDisabledForSweeps` call.

- [ ] **Step 2: Add shadow-only diagnostic controls**

Add a shadow-only switch so the engine can train/query/probe the bright-surfel cache and record candidate viability without allowing that candidate to enter reservoir selection.

In `UISystem::PathTracerSettings` and `EngineCore::PathTracerExperimentRow`, add:

```cpp
bool reservoirGiBrightSurfelShadowOnly = false;
```

In `EngineCore::applyPathTracerExperimentRow`, copy `row.reservoirGiBrightSurfelShadowOnly` into `settings.reservoirGiBrightSurfelShadowOnly`.

Thread this setting through the existing path tracer push-constant flag path used by reservoir GI settings:

```cpp
constexpr uint32_t kPtFlagsReservoirBrightSurfelShadowOnlyBit = 1u << 25u;
```

Pack the bit in `packPathTracerFlags()` when `settings.reservoirGiBrightSurfelShadowOnly` is true. Mirror it in `src/shaders/Raygen.slang`:

```slang
static const uint PT_FLAGS_RESERVOIR_BRIGHT_SURFEL_SHADOW_ONLY_BIT = 1u << 25;
```

Decode it near the other path tracer flags:

```slang
bool reservoirGiBrightSurfelShadowOnly =
    (packedPathTracerFlags & PT_FLAGS_RESERVOIR_BRIGHT_SURFEL_SHADOW_ONLY_BIT) != 0;
```

In `src/shaders/Raygen.slang`, use the flag so shadow-only mode executes the bright-surfel training, indexed query, probe, geometry/normal/hemisphere checks, target-weight check, and visibility check, but does not insert or select a bright-surfel reservoir candidate:

```slang
const bool allowBrightSurfelSelection =
    useBrightSurfelProposal && !reservoirGiBrightSurfelShadowOnly;
```

Use `allowBrightSurfelSelection` only at the point where the candidate can enter reservoir selection. Diagnostics before that point must still run when `useBrightSurfelProposal` is true.

Do not gate shadow-only mode before `evaluateBrightReceiverSurfelReservoirGiCandidate`, because that would skip target and visibility diagnostics. Split or parameterize that function so shadow-only mode runs through indexed selection, geometry/hemisphere/target checks, visibility, and reject counters, then returns before candidate insertion and before incrementing `reservoirGiBrightSurfelAcceptedOffset`:

```slang
bool evaluateBrightReceiverSurfelReservoirGiCandidate(float3 hitPos,
                                                      float3 N,
                                                      float3 V,
                                                      RayPayload primaryPayload,
                                                      uint2 launchID,
                                                      uint2 launchSize,
                                                      inout uint rngState,
                                                      bool allowSelection,
                                                      out ReservoirGiRecord candidateRecord)
{
    // Existing train/query/probe/target/visibility diagnostic path stays above this point.
    // All existing reject counters still fire before this guard.

    if (!allowSelection) {
        return false;
    }

    candidateRecord.candidatePosition = surfel.position;
    candidateRecord.candidateNormal = surfel.normal;
    candidateRecord.suffixRadiance = targetEvaluation.suffixRadiance;
    candidateRecord.contribution = targetEvaluation.contribution;
    candidateRecord.sourcePdf = targetEvaluation.sourcePdf;
    candidateRecord.targetWeight = targetEvaluation.targetWeight;
    candidateRecord.weightSum = targetEvaluation.targetWeight / max(targetEvaluation.sourcePdf, 0.000001);
    candidateRecord.selectedWeight = candidateRecord.weightSum;
    candidateRecord.confidenceM = max(surfel.confidence, 1.0);
    candidateRecord.sourcePixel = surfel.sourcePixel;
    candidateRecord.sourceFrameId = surfel.frameId;
    candidateRecord.frameId = ubo.frameCount;
    candidateRecord.flags = 1u | 2u |
                            (pathTracerLuminance(targetEvaluation.suffixRadiance) > 0.000001 ? 4u : 0u) |
                            16u;

    ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelAcceptedOffset, 1u);
    return true;
}
```

At the call site, pass `allowBrightSurfelSelection`:

```slang
if (useBrightReceiverSurfel) {
    ReservoirGiRecord surfelRecord;
    if (evaluateBrightReceiverSurfelReservoirGiCandidate(hitPos, N, V, payload,
                                                         launchID, launchSize,
                                                         rngState,
                                                         allowBrightSurfelSelection,
                                                         surfelRecord)) {
        combineReservoirGiCandidate(surfelRecord,
                                    surfelRecord.targetWeight,
                                    RESERVOIR_GI_SOURCE_BRIGHT_SURFEL,
                                    float3(0.0, 0.0, 0.0),
                                    selectedRecord, selectedTargetWeight,
                                    selectedSource, selectedSecondarySun,
                                    reservoirWeightSum,
                                    hasSelectedReservoirCandidate, rngState);
    }
}
```

Expected: shadow-only rows can still show visibility and target failures, but cannot increment `brightSurfelAccepted`, cannot call `combineReservoirGiCandidate` for `RESERVOIR_GI_SOURCE_BRIGHT_SURFEL`, and cannot increment `reservoirGiSelectedBrightSurfel`.

Use the existing row-summary funnel counters first. Do not add new positive/pass counters unless the existing reject-oriented counters cannot identify the failing stage:

```text
brightSurfelTrainingStore
brightSurfelIndexedQuery
brightSurfelIndexedEmpty
brightSurfelIndexedProbe
brightSurfelSelectorRejectDistance
brightSurfelSelectorRejectInvalidVector
brightSurfelSelectorRejectReceiverHemisphere
brightSurfelSelectorRejectSurfelHemisphere
brightSurfelSelectorRejectGeometry
brightSurfelSelectorRejectTarget
brightSurfelSelectorViable
brightSurfelRejectVisibility
brightSurfelAccepted
reservoirGiSelectedBrightSurfel
```

Expected: shadow-only rows can report all pre-selection counters, while `brightSurfelAccepted` and `reservoirGiSelectedBrightSurfel` stay `0`.

- [ ] **Step 3: Add explicit shadow and proposal evaluation sweeps**

In `src/Core/EngineCore.h`, add:

```cpp
void startPathTracerBrightSurfelShadowEvaluationSweep();
void startPathTracerBrightSurfelProposalEvaluationSweep();
```

In `src/Core/UISystem.h`, add:

```cpp
bool runBrightSurfelShadowEvaluationSweep = false;
bool runBrightSurfelProposalEvaluationSweep = false;
```

In `src/Core/UISystem.cpp`, near the Sponza sweep button, add:

```cpp
if (ImGui::Button("Run Bright Surfel Shadow Sweep"))
{
    pathTracerAnalysisSettings.runBrightSurfelShadowEvaluationSweep = true;
}

if (ImGui::Button("Run Bright Surfel Proposal Sweep"))
{
    pathTracerAnalysisSettings.runBrightSurfelProposalEvaluationSweep = true;
}
```

In `src/Core/EngineCore.cpp`, consume the flag next to `runSponzaGiPerfSweep`:

```cpp
if (ui.pathTracerAnalysisSettings.runBrightSurfelShadowEvaluationSweep)
{
    startPathTracerBrightSurfelShadowEvaluationSweep();
}

if (ui.pathTracerAnalysisSettings.runBrightSurfelProposalEvaluationSweep)
{
    startPathTracerBrightSurfelProposalEvaluationSweep();
}
```

Reset each flag inside its matching sweep starter. The split is intentional: the proposal sweep should be run only after the shadow sweep shows a live train/query/probe/viability funnel.

- [ ] **Step 4: Build staged evaluation rows**

Implement `startPathTracerBrightSurfelShadowEvaluationSweep()` by copying the Sponza scenario loop but using only these rows per scenario:

```text
Bright Surfel Evaluation / Baseline Sun Receiver
Bright Surfel Evaluation / Baseline Static Audit
Bright Surfel Evaluation / Shadow Diagnostics
```

Implement `startPathTracerBrightSurfelProposalEvaluationSweep()` by copying the same scenario loop but using only these rows per scenario:

```text
Bright Surfel Evaluation / Baseline Static Audit
Bright Surfel Evaluation / Proposal Enabled
Bright Surfel Evaluation / Proposal Enabled Static Audit
```

Row settings:

```cpp
baseline.reservoirGiProposalMode =
    UISystem::PathTracerReservoirGiProposalMode::MixedCosineSunReceiver;

baselineStaticAudit = baseline;
baselineStaticAudit.name = makeScenarioRowName(scenario, "Bright Surfel Evaluation / Baseline Static Audit");
baselineStaticAudit.reservoirGiMode = UISystem::PathTracerReservoirGiMode::TemporalSpatial;
baselineStaticAudit.reservoirGiTemporalBudgetDivisor = 1;
baselineStaticAudit.reservoirGiSpatialBudgetDivisor = 1;
baselineStaticAudit.reservoirGiEstimatorAuditMode =
    UISystem::PathTracerReservoirGiEstimatorAuditMode::Current;

shadow = baseline;
shadow.name = makeScenarioRowName(scenario, "Bright Surfel Evaluation / Shadow Diagnostics");
shadow.reservoirGiProposalMode =
    UISystem::PathTracerReservoirGiProposalMode::MixedCosineSunReceiverBrightSurfel;
shadow.reservoirGiBrightSurfelShadowOnly = true;

proposal = baseline;
proposal.name = makeScenarioRowName(scenario, "Bright Surfel Evaluation / Proposal Enabled");
proposal.reservoirGiProposalMode =
    UISystem::PathTracerReservoirGiProposalMode::MixedCosineSunReceiverBrightSurfel;
proposal.reservoirGiBrightSurfelShadowOnly = false;

staticAudit = proposal;
staticAudit.name = makeScenarioRowName(scenario, "Bright Surfel Evaluation / Proposal Enabled Static Audit");
staticAudit.reservoirGiMode = UISystem::PathTracerReservoirGiMode::TemporalSpatial;
staticAudit.reservoirGiTemporalBudgetDivisor = 1;
staticAudit.reservoirGiSpatialBudgetDivisor = 1;
staticAudit.reservoirGiEstimatorAuditMode =
    UISystem::PathTracerReservoirGiEstimatorAuditMode::Current;
```

Push rows in this order:

```cpp
// Shadow sweep:
ptExperimentRows.push_back(baseline);
ptExperimentRows.push_back(baselineStaticAudit);
ptExperimentRows.push_back(shadow);

// Proposal sweep:
ptExperimentRows.push_back(baselineStaticAudit);
ptExperimentRows.push_back(proposal);
ptExperimentRows.push_back(staticAudit);
```

The duplicate `Baseline Static Audit` row in the proposal sweep is intentional. Proposal audit comparisons must use a baseline captured in the same sweep run and scenario, not a baseline from a previous shadow sweep.

In `src/shaders/Raygen.slang`, remove the hard-disable constant:

```slang
const bool enableBrightSurfelProposal = false;
```

Replace the selection gate with proposal-mode ownership:

```slang
const bool useBrightSurfelProposal =
    reservoirGiProposalMode == RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_BRIGHT_SURFEL;
```

Use `useBrightSurfelProposal` wherever the current code combines `enableBrightSurfelProposal` with the bright-surfel proposal mode. Default behavior stays disabled because normal UI defaults and default sweeps do not choose the bright-surfel proposal mode.

- [ ] **Step 5: Run shadow diagnostics first**

Use `Run Bright Surfel Shadow Sweep` and record row summaries. Treat shadow diagnostics as a gate before running proposal-enabled output.

Shadow-only acceptance criteria:

- `brightSurfelTrainingStore > 0` after warmup in at least one scenario.
- `brightSurfelIndexedQuery > 0` and `brightSurfelIndexedProbe > 0` in shadow rows.
- Geometry is not the dominant rejection stage: `brightSurfelSelectorRejectGeometry` is low relative to `brightSurfelIndexedProbe`.
- Hemisphere checks are not the dominant rejection stage: `brightSurfelSelectorRejectReceiverHemisphere` and `brightSurfelSelectorRejectSurfelHemisphere` are low relative to `brightSurfelIndexedProbe`.
- Target checks are not the dominant rejection stage: `brightSurfelSelectorRejectTarget` is low relative to `brightSurfelIndexedProbe`.
- `brightSurfelSelectorViable > 0` in at least one scenario.
- `brightSurfelAccepted == 0` and `reservoirGiSelectedBrightSurfel == 0` in all shadow rows.

If any pre-selection counter stays near zero, do not run `Run Bright Surfel Proposal Sweep` yet. Decide between delete/refactor based on the failed funnel stage.

- [ ] **Step 6: Run proposal-enabled evaluation sweep**

Use `Run Bright Surfel Proposal Sweep` only after Step 5 passes. Record row summaries.

Acceptance criteria to keep bright-surfel proposal:

- `brightSurfelTrainingStore > 0` after warmup in at least one scenario.
- `brightSurfelIndexedQuery > 0` and `brightSurfelIndexedProbe > 0` in proposal rows.
- `brightSurfelSelectorViable > 0` in at least one scenario.
- `brightSurfelAccepted > 0` in at least one scenario.
- `reservoirGiSelectedBrightSurfel > 0` in at least one scenario.
- `reservoirGiAuditRelativeErrorPct <= 5.00` in the proposal static audit row.
- For each scenario, proposal static audit `reservoirGiAuditRelativeErrorPct <= baselineStaticAudit.reservoirGiAuditRelativeErrorPct + 1.00`. This is a percentage-point threshold, not a relative multiplier.
- A known reservoir failure mode improves, preferably lower `localRejectNoLight` or more useful accepted candidates.
- `totalMs` increase over baseline is under 25 percent unless Dark Courtyard visibly improves.

Kill criteria:

- Training stores occur but indexed query/probe is near zero.
- Query/probe occurs but selector viability is near zero, or geometry, receiver-hemisphere, surfel-hemisphere, target, distance, invalid-vector, or visibility rejects dominate the probed candidates.
- Selected bright-surfel rows shift audit mean by more than 5 percent.
- Selected candidates only work because of a bright-surfel-only shortcut, scale, clamp, or reservoir bypass.
- Runtime increase exceeds 25 percent without clear Dark Courtyard improvement.
- Code shape cannot be explained as reservoir-owned receiver evidence without misleading naming or special-case bias.
- The only success signal is that high-luma records are easier to rediscover, with no evidence that the mechanism generalizes to useful receiver-cache candidate discovery.

- [ ] **Step 7: Decide keep/refactor/delete/not-proven**

If keep:

- Rename/refactor future work toward `ReservoirGiReceiverCache`; treat brightness as one possible training feature, not the cache identity.
- Keep existing code disabled by default.
- Plan a refactor that separates storage, indexed query, target evaluation, and reservoir combination into clearly named helpers.

If not proven:

- Record the exact failed or inconclusive funnel stage.
- Keep the measured lesson in docs.
- Do not polish the current bright-surfel path.
- Write a cleaner reservoir receiver-cache plan if the shadow diagnostics show that storage/query is useful but current selection is not.
- If the only positive signal is "bright things are easier to find," do not promote it into a receiver-cache abstraction.

If delete:

- Write a follow-up plan to remove:

```text
ptReservoirGiBrightSurfelCurrent
ptReservoirGiBrightSurfelHistory
ReservoirGiBrightSurfelRecord
MixedCosineSunReceiverBrightSurfel
reservoirGiBrightSurfel...
brightSurfel...
```

If refactor before deciding:

- Keep only training/query diagnostics and disable selection, then rerun the shadow diagnostic sweep before considering the proposal sweep again.

- [ ] **Step 8: Commit evaluation harness**

Run verification first:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests'
.\cmake-build-debug\LaphriaEngineUnitTests.exe
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEditor'
```

Expected: all exit code `0`.

Run:

```powershell
git add src/shaders/Raygen.slang src/Core/EngineAuxiliary.h src/Core/EngineCore.cpp src/Core/EngineCore.h src/Core/UISystem.cpp src/Core/UISystem.h src/Core/PathTracerAnalysis.h src/Core/PathTracerAnalysis.cpp tests/PathTracerAnalysisTests.cpp
git commit -m "test: isolate bright surfel reservoir evaluation sweeps"
```

Expected: commit succeeds.

---

## Task 7: Record Final Decision

**Files:**

- Modify: `docs/architecture/restir-gi-approach-decision.md`
- Modify: `docs/architecture/restir-gi-sponza-handoff.md`

- [ ] **Step 1: Add decision note**

Append:

```markdown
## 2026-05-18 Bright-Surfel Evaluation Decision

After removing the standalone compute surfel GI path, the remaining bright-surfel reservoir proposal was evaluated as a possible reservoir-owned receiver-cache proposal source. The concept under evaluation was receiver/cache evidence for reservoir candidate discovery; "bright surfel" remains a temporary implementation name.
```

If keeping or refactoring the proposal, append:

```markdown
The proposal is being kept as a disabled-by-default evaluation path because the shadow/proposal sweeps showed positive training stores, positive indexed probes, positive viable candidates, at least one selected bright-surfel candidate, audit relative error below 5 percent, runtime within the 25 percent budget, and evidence that the result is more than a high-luma rediscovery heuristic.

The next implementation plan is a reservoir receiver cache refactor plan that renames the surviving pieces away from bright-surfel terminology.
```

In the same decision section, add a prose metrics paragraph with exact values for `brightSurfelTrainingStore`, `brightSurfelIndexedQuery`, `brightSurfelIndexedEmpty`, `brightSurfelIndexedProbe`, `brightSurfelSelectorRejectDistance`, `brightSurfelSelectorRejectInvalidVector`, `brightSurfelSelectorRejectReceiverHemisphere`, `brightSurfelSelectorRejectSurfelHemisphere`, `brightSurfelSelectorRejectGeometry`, `brightSurfelSelectorRejectTarget`, `brightSurfelRejectVisibility`, `brightSurfelSelectorViable`, `brightSurfelAccepted`, `reservoirGiSelectedBrightSurfel`, audit relative error, and runtime delta.

If the proposal is not proven, append:

```markdown
The proposal is not being kept as a production candidate path yet. The staged shadow/proposal evaluation showed that at least one part of the train -> indexed query -> probe -> reject checks -> viable -> accepted -> selected funnel was missing, too weak, too noisy, too expensive, or only proved that high-luma records are easy to rediscover.

The next implementation plan is either a cleaner reservoir receiver-cache design or a bright-surfel removal plan, depending on which funnel stage failed.
```

In the same decision section, add a prose metrics paragraph with exact values for `brightSurfelTrainingStore`, `brightSurfelIndexedQuery`, `brightSurfelIndexedEmpty`, `brightSurfelIndexedProbe`, `brightSurfelSelectorRejectDistance`, `brightSurfelSelectorRejectInvalidVector`, `brightSurfelSelectorRejectReceiverHemisphere`, `brightSurfelSelectorRejectSurfelHemisphere`, `brightSurfelSelectorRejectGeometry`, `brightSurfelSelectorRejectTarget`, `brightSurfelRejectVisibility`, `brightSurfelSelectorViable`, `brightSurfelAccepted`, `reservoirGiSelectedBrightSurfel`, audit relative error, and runtime delta.

If deleting the proposal, append:

```markdown
The proposal is being deleted because at least one keep criterion failed.

The next implementation plan is a bright-surfel removal plan.
```

In the same decision section, add a prose metrics paragraph with exact values for `brightSurfelTrainingStore`, `brightSurfelIndexedQuery`, `brightSurfelIndexedEmpty`, `brightSurfelIndexedProbe`, `brightSurfelSelectorRejectDistance`, `brightSurfelSelectorRejectInvalidVector`, `brightSurfelSelectorRejectReceiverHemisphere`, `brightSurfelSelectorRejectSurfelHemisphere`, `brightSurfelSelectorRejectGeometry`, `brightSurfelSelectorRejectTarget`, `brightSurfelRejectVisibility`, `brightSurfelSelectorViable`, `brightSurfelAccepted`, `reservoirGiSelectedBrightSurfel`, audit relative error, and runtime delta.

- [ ] **Step 2: Commit decision**

Run:

```powershell
git add docs/architecture/restir-gi-approach-decision.md docs/architecture/restir-gi-sponza-handoff.md
git commit -m "docs: record bright surfel reservoir decision"
```

Expected: commit succeeds if docs changed.

---

## Success Criteria

- `Surfel*.slang` files are gone.
- `CMakeLists.txt` no longer compiles standalone surfel compute shaders.
- No `surfelGi...` compute buffers, pipelines, descriptor sets, UI toggles, debug AOVs, denoiser bindings, counters, ratio helpers, row-summary fields, or sweep rows remain.
- Reservoir estimator audit remains green.
- Default Sponza sweep contains only the lean reservoir regression rows.
- Bright-surfel reservoir proposal remains isolated and disabled by default until explicit evaluation.
- The final decision is documented as keep, refactor, delete, or not proven.

---

## Self-Review

- Spec coverage: includes compute surfel removal, sweep cleanup, docs, and separate shadow/proposal evaluation of the reservoir-side bright-surfel path as a possible receiver-cache candidate source.
- Placeholder scan: no placeholder values remain; final decision wording requires exact measured values in prose before commit.
- Type consistency: standalone compute surfel names use `surfelGi...`; reservoir-side proposal names use `brightSurfel...`/`reservoirGiBrightSurfel...`.
- Scope check: no new cache architecture is implemented here. This plan removes confusion first, then evaluates whether the one remaining candidate path proves a general receiver-cache concept rather than just a high-luma heuristic.
