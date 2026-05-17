# Sparse Bright Surfel Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current bright surfel proposal into a sparse, explicitly trained virtual-light candidate path that is cheaper to evaluate, less dependent on the final selected reservoir sample, and easier to validate against temporal artifacts.

**Architecture:** Keep the existing bright surfel buffers and proposal mode, but split the path into producer and consumer phases. The producer stores qualified local ReSTIR GI candidates through a sparse training gate, while the consumer samples the global surfel pool with a weighted reservoir-style selector, rejects bad target candidates before visibility rays, and reports separate counters for precheck, training, and selected surfel contribution. The Sponza sweep clears sparse GI state at row boundaries and gains matching single-frame Sun Receiver and bright surfel rows so display/postprocess issues can be separated from estimator history.

**Tech Stack:** C++17 engine/UI/analysis plumbing, Slang ray generation shader, Vulkan storage buffers, `tests/PathTracerAnalysisTests.cpp` string-contract tests, CMake/CTest verification, in-app Sponza PT/GI audit sweep.

---

## Current State

The bright surfel prototype already has storage buffers, UI counters, a proposal mode, and Sponza sweep rows. The latest focused rows show that the surfel path can improve luma, but still spends too many rays and fails to select useful surfels in Dark Courtyard and Mid-depth Interior.

```text
Sunlit Courtyard Wall Bright Surfel:
  firstHitProbeAvgLuma=0.06534 vs Sun Receiver 0.05290
  brightSurfelStore=59182.7
  brightSurfelAttempt=518314.2
  brightSurfelAccepted=652.5
  reservoirGiSelectedBrightSurfel=607.1
  totalMs=230.011 vs Sun Receiver 169.737

Mid-Depth Interior Bright Surfel:
  firstHitProbeAvgLuma=0.04387 vs Sun Receiver 0.03666
  brightSurfelStore=28330.8
  brightSurfelAttempt=518370.8
  brightSurfelHit=95992.9
  brightSurfelRejectTarget=90279.8
  brightSurfelAccepted=0.0
  reservoirGiSelectedBrightSurfel=0.0
  totalMs=198.068 vs Sun Receiver 162.174
```

The plan optimizes for three outcomes:

```text
1. Fewer visibility rays for candidates that cannot produce nonzero target weight.
2. More useful surfel records in the global pool, especially when the final selected reservoir sample is not surfel-worthy.
3. Cleaner diagnosis of temporal artifacts by comparing temporal-spatial rows against matching single-frame Sun Receiver and bright surfel rows.
```

## File Structure

- Modify `tests/PathTracerAnalysisTests.cpp`
  - Add contract checks for new counters, shader helper names, sweep rows, and counter offsets.
- Modify `src/Core/EngineAuxiliary.h`
  - Append bright surfel precheck and training counters to `PathTracerAnalysisCounters`.
- Modify `src/Core/UISystem.h`
  - Mirror new counter fields in `PathTracerPerfStats`.
- Modify `src/Core/UISystem.cpp`
  - Display new bright surfel counters in the Reservoir GI diagnostics panel.
- Modify `src/Core/EngineCore.h`
  - Add accumulator fields for the new sweep metrics.
- Modify `src/Core/EngineCore.cpp`
  - Clear sparse reservoir/surfel state at sweep row transitions, copy counters to UI stats, accumulate them in the sweep, log them in row summaries, and add matching single-frame Sun Receiver control rows.
- Modify `src/shaders/Raygen.slang`
  - Add pre-visibility target precheck, sparse producer training gate, weighted global surfel selection, and conservative source PDF handling.
- Modify `docs/superpowers/plans/2026-05-17-global-bright-surfel-sampling.md`
  - Append a short result note after the next sweep to link the previous global sampler plan to this training plan.

## Counter Layout

Append these fields after `reservoirGiSelectedBrightSurfel` so existing offsets remain stable:

```cpp
uint32_t reservoirGiBrightSurfelPrecheckRejectTarget = 0; // offset 344
uint32_t reservoirGiBrightSurfelTrainingAttempt = 0;      // offset 348
uint32_t reservoirGiBrightSurfelTrainingStore = 0;        // offset 352
uint32_t reservoirGiBrightSurfelTrainingRejectGeometry = 0; // offset 356
uint32_t reservoirGiBrightSurfelTrainingRejectTarget = 0; // offset 360
```

Use matching shader counter offsets:

```slang
static const uint reservoirGiBrightSurfelPrecheckRejectTargetOffset = 344u;
static const uint reservoirGiBrightSurfelTrainingAttemptOffset = 348u;
static const uint reservoirGiBrightSurfelTrainingStoreOffset = 352u;
static const uint reservoirGiBrightSurfelTrainingRejectGeometryOffset = 356u;
static const uint reservoirGiBrightSurfelTrainingRejectTargetOffset = 360u;
```

---

### Task 0: Isolate Sparse GI State Between Sweep Rows

**Files:**
- Modify: `tests/PathTracerAnalysisTests.cpp`
- Modify: `src/Core/EngineCore.cpp`

- [ ] **Step 1: Add a row-transition isolation contract**

In the focused Sponza sweep contract test, add a required source string that proves row transitions clear sparse GI state after the device is idle and before applying the next row:

```cpp
"vulkan.logicalDevice.waitIdle();\n\tclearPathTracerExperimentState();\n\tapplyPathTracerExperimentRow(ptExperimentRows[ptExperimentRowIndex]);",
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" >nul && cmake --build cmake-build-debug --target LaphriaEngineUnitTests --config Debug && cmake-build-debug\LaphriaEngineUnitTests.exe"
```

Expected: `LaphriaEngineUnitTests` fails because `updatePathTracerExperimentSweep` does not clear reservoir/surfel buffers before applying the next row.

- [ ] **Step 3: Clear state at row transitions**

In `EngineCore::updatePathTracerExperimentSweep`, replace the row transition tail:

```cpp
ptExperimentWarmupRemaining = std::max(1, ptExperimentWarmupFrames);
ptExperimentSampleRemaining = std::max(1, ptExperimentSampleFrames);
ptExperimentAccum           = {};
vulkan.logicalDevice.waitIdle();
applyPathTracerExperimentRow(ptExperimentRows[ptExperimentRowIndex]);
```

with:

```cpp
ptExperimentWarmupRemaining = std::max(1, ptExperimentWarmupFrames);
ptExperimentSampleRemaining = std::max(1, ptExperimentSampleFrames);
ptExperimentAccum           = {};
vulkan.logicalDevice.waitIdle();
clearPathTracerExperimentState();
applyPathTracerExperimentRow(ptExperimentRows[ptExperimentRowIndex]);
```

This prevents sparse bright surfel records, receiver cache records, and reservoir history from one Sponza row contaminating the next row.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" >nul && cmake --build cmake-build-debug --target LaphriaEngineUnitTests --config Debug && cmake-build-debug\LaphriaEngineUnitTests.exe"
```

Expected: the row-transition isolation contract passes. Later tests may still fail after Task 1 adds new counter and shader contracts.

---

### Task 1: Add Counter And Contract Tests

**Files:**
- Modify: `tests/PathTracerAnalysisTests.cpp`

- [ ] **Step 1: Extend the counter offset table test**

In the `PathTracerAnalysisCounters` offset table, append these entries immediately after the existing `reservoirGiSelectedBrightSurfel` entry:

```cpp
{"reservoirGiBrightSurfelPrecheckRejectTarget",
 offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelPrecheckRejectTarget), 344u},
{"reservoirGiBrightSurfelTrainingAttempt",
 offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelTrainingAttempt), 348u},
{"reservoirGiBrightSurfelTrainingStore",
 offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelTrainingStore), 352u},
{"reservoirGiBrightSurfelTrainingRejectGeometry",
 offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelTrainingRejectGeometry), 356u},
{"reservoirGiBrightSurfelTrainingRejectTarget",
 offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelTrainingRejectTarget), 360u}
```

- [ ] **Step 2: Add required shader symbols**

In both bright surfel shader contract lists, add these exact strings:

```cpp
"reservoirGiBrightSurfelPrecheckRejectTargetOffset",
"reservoirGiBrightSurfelTrainingAttemptOffset",
"reservoirGiBrightSurfelTrainingStoreOffset",
"reservoirGiBrightSurfelTrainingRejectGeometryOffset",
"reservoirGiBrightSurfelTrainingRejectTargetOffset",
"shouldTrainBrightReceiverSurfel",
"tryStoreBrightSurfelTrainingCandidate",
"estimateBrightSurfelTargetBeforeVisibility",
"selectWeightedGlobalBrightReceiverSurfelRecord",
"surfelSelectionPdf",
```

Remove this old deterministic selector requirement from both lists:

```cpp
"selectGlobalBrightReceiverSurfelRecord",
```

- [ ] **Step 3: Add required UI/log labels**

Add these exact strings to the required metrics and UI label checks:

```cpp
"Reservoir GI Bright Surfel Precheck Reject Target",
"Reservoir GI Bright Surfel Training Attempts",
"Reservoir GI Bright Surfel Training Stores",
"Reservoir GI Bright Surfel Training Reject Geometry",
"Reservoir GI Bright Surfel Training Reject Target",
"brightSurfelPrecheckRejectTarget",
"brightSurfelTrainingAttempt",
"brightSurfelTrainingStore",
"brightSurfelTrainingRejectGeometry",
"brightSurfelTrainingRejectTarget",
```

- [ ] **Step 4: Add required sweep row**

In the focused Sponza sweep row-name checks, add:

```cpp
"Reservoir 1C Shadowed Sun First Mixed Single Frame Sun Receiver",
"Reservoir 1C Shadowed Sun First Mixed Single Frame Sun Receiver Bright Surfel",
```

- [ ] **Step 5: Run tests and verify RED**

Run:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" >nul && cmake --build cmake-build-debug --target LaphriaEngineUnitTests --config Debug && cmake-build-debug\LaphriaEngineUnitTests.exe"
```

Expected: the unit binary fails with a missing symbol, missing counter offset, or missing sweep row message from `PathTracerAnalysisTests.cpp`.

---

### Task 2: Add Counter Plumbing

**Files:**
- Modify: `src/Core/EngineAuxiliary.h`
- Modify: `src/Core/UISystem.h`
- Modify: `src/Core/UISystem.cpp`
- Modify: `src/Core/EngineCore.h`
- Modify: `src/Core/EngineCore.cpp`

- [ ] **Step 1: Add analysis counter fields**

In `PathTracerAnalysisCounters`, immediately after `reservoirGiSelectedBrightSurfel`, add:

```cpp
uint32_t reservoirGiBrightSurfelPrecheckRejectTarget = 0;
uint32_t reservoirGiBrightSurfelTrainingAttempt = 0;
uint32_t reservoirGiBrightSurfelTrainingStore = 0;
uint32_t reservoirGiBrightSurfelTrainingRejectGeometry = 0;
uint32_t reservoirGiBrightSurfelTrainingRejectTarget = 0;
```

- [ ] **Step 2: Add UI perf stat fields**

In `UISystem::PathTracerPerfStats`, immediately after `reservoirGiSelectedBrightSurfel`, add:

```cpp
uint32_t reservoirGiBrightSurfelPrecheckRejectTarget = 0;
uint32_t reservoirGiBrightSurfelTrainingAttempt = 0;
uint32_t reservoirGiBrightSurfelTrainingStore = 0;
uint32_t reservoirGiBrightSurfelTrainingRejectGeometry = 0;
uint32_t reservoirGiBrightSurfelTrainingRejectTarget = 0;
```

- [ ] **Step 3: Display the new counters**

In the Reservoir GI diagnostics block after the existing bright surfel lines, add:

```cpp
ImGui::Text("Reservoir GI Bright Surfel Precheck Reject Target: %u",
            pathTracerPerfStats.reservoirGiBrightSurfelPrecheckRejectTarget);
ImGui::Text("Reservoir GI Bright Surfel Training Attempts: %u",
            pathTracerPerfStats.reservoirGiBrightSurfelTrainingAttempt);
ImGui::Text("Reservoir GI Bright Surfel Training Stores: %u",
            pathTracerPerfStats.reservoirGiBrightSurfelTrainingStore);
ImGui::Text("Reservoir GI Bright Surfel Training Reject Geometry: %u",
            pathTracerPerfStats.reservoirGiBrightSurfelTrainingRejectGeometry);
ImGui::Text("Reservoir GI Bright Surfel Training Reject Target: %u",
            pathTracerPerfStats.reservoirGiBrightSurfelTrainingRejectTarget);
```

- [ ] **Step 4: Copy mapped counters to UI stats**

In `EngineCore::collectPathTracerAnalysisCounters`, after copying `reservoirGiSelectedBrightSurfel`, add:

```cpp
ui.pathTracerPerfStats.reservoirGiBrightSurfelPrecheckRejectTarget =
    counters->reservoirGiBrightSurfelPrecheckRejectTarget;
ui.pathTracerPerfStats.reservoirGiBrightSurfelTrainingAttempt =
    counters->reservoirGiBrightSurfelTrainingAttempt;
ui.pathTracerPerfStats.reservoirGiBrightSurfelTrainingStore =
    counters->reservoirGiBrightSurfelTrainingStore;
ui.pathTracerPerfStats.reservoirGiBrightSurfelTrainingRejectGeometry =
    counters->reservoirGiBrightSurfelTrainingRejectGeometry;
ui.pathTracerPerfStats.reservoirGiBrightSurfelTrainingRejectTarget =
    counters->reservoirGiBrightSurfelTrainingRejectTarget;
```

- [ ] **Step 5: Add sweep accumulator fields**

In `PathTracerExperimentAccumulator`, immediately after `reservoirGiSelectedBrightSurfel`, add:

```cpp
double reservoirGiBrightSurfelPrecheckRejectTarget = 0.0;
double reservoirGiBrightSurfelTrainingAttempt = 0.0;
double reservoirGiBrightSurfelTrainingStore = 0.0;
double reservoirGiBrightSurfelTrainingRejectGeometry = 0.0;
double reservoirGiBrightSurfelTrainingRejectTarget = 0.0;
```

- [ ] **Step 6: Accumulate the new sweep stats**

In the sweep accumulation block, after `ptExperimentAccum.reservoirGiSelectedBrightSurfel`, add:

```cpp
ptExperimentAccum.reservoirGiBrightSurfelPrecheckRejectTarget +=
    static_cast<double>(stats.reservoirGiBrightSurfelPrecheckRejectTarget);
ptExperimentAccum.reservoirGiBrightSurfelTrainingAttempt +=
    static_cast<double>(stats.reservoirGiBrightSurfelTrainingAttempt);
ptExperimentAccum.reservoirGiBrightSurfelTrainingStore +=
    static_cast<double>(stats.reservoirGiBrightSurfelTrainingStore);
ptExperimentAccum.reservoirGiBrightSurfelTrainingRejectGeometry +=
    static_cast<double>(stats.reservoirGiBrightSurfelTrainingRejectGeometry);
ptExperimentAccum.reservoirGiBrightSurfelTrainingRejectTarget +=
    static_cast<double>(stats.reservoirGiBrightSurfelTrainingRejectTarget);
```

- [ ] **Step 7: Log the new row summary fields**

In `logPathTracerExperimentRow`, append this text to the format string after `reservoirGiSelectedBrightSurfel=%.1f`:

```cpp
"brightSurfelPrecheckRejectTarget=%.1f, brightSurfelTrainingAttempt=%.1f, "
"brightSurfelTrainingStore=%.1f, brightSurfelTrainingRejectGeometry=%.1f, "
"brightSurfelTrainingRejectTarget=%.1f, "
```

Add these arguments in the matching order:

```cpp
accum.reservoirGiBrightSurfelPrecheckRejectTarget * invSamples,
accum.reservoirGiBrightSurfelTrainingAttempt * invSamples,
accum.reservoirGiBrightSurfelTrainingStore * invSamples,
accum.reservoirGiBrightSurfelTrainingRejectGeometry * invSamples,
accum.reservoirGiBrightSurfelTrainingRejectTarget * invSamples,
```

- [ ] **Step 8: Run tests and verify partial GREEN**

Run:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" >nul && cmake --build cmake-build-debug --target LaphriaEngineUnitTests --config Debug && cmake-build-debug\LaphriaEngineUnitTests.exe"
```

Expected: counter offset and UI/log label checks pass. The test still fails on missing shader helper names and the missing single-frame sweep rows.

---

### Task 3: Move Bright Surfel Target Precheck Before Visibility

**Files:**
- Modify: `src/shaders/Raygen.slang`

- [ ] **Step 1: Add shader counter offsets**

After `reservoirGiSelectedBrightSurfelOffset`, add:

```slang
static const uint reservoirGiBrightSurfelPrecheckRejectTargetOffset = 344u;
static const uint reservoirGiBrightSurfelTrainingAttemptOffset = 348u;
static const uint reservoirGiBrightSurfelTrainingStoreOffset = 352u;
static const uint reservoirGiBrightSurfelTrainingRejectGeometryOffset = 356u;
static const uint reservoirGiBrightSurfelTrainingRejectTargetOffset = 360u;
```

- [ ] **Step 2: Add precheck helper**

Near `evaluateBrightReceiverSurfelReservoirGiCandidate`, add:

```slang
bool estimateBrightSurfelTargetBeforeVisibility(float3 hitPos,
                                                float3 N,
                                                float3 V,
                                                RayPayload primaryPayload,
                                                ReservoirGiBrightSurfelRecord surfel,
                                                float dist2,
                                                out ReservoirGiTargetEvaluation targetEvaluation)
{
    float areaProxy = PI * max(surfel.radius * surfel.radius, 0.0001);
    float3 toSurfel = surfel.position - hitPos;
    float3 wi = toSurfel * rsqrt(max(dist2, 0.0001));
    float surfelCos = max(dot(surfel.normal, -wi), 0.0);
    float geometry = surfelCos * areaProxy / max(dist2, 0.0001);
    float3 suffixRadiance = sanitizeReservoirGiContribution(
        max(surfel.radiance, float3(0.0, 0.0, 0.0)) * geometry);
    float sourcePdf = 1.0;
    targetEvaluation = evaluateReservoirGiTargetAtPrimary(hitPos, N, V, primaryPayload,
                                                          surfel.position, surfel.normal,
                                                          suffixRadiance, sourcePdf);
    return targetEvaluation.validGeometry &&
           targetEvaluation.validLight &&
           targetEvaluation.targetWeight > RESERVOIR_GI_BRIGHT_SURFEL_CANDIDATE_MIN_TARGET_WEIGHT;
}
```

- [ ] **Step 3: Call the precheck before `TraceRay`**

In `evaluateBrightReceiverSurfelReservoirGiCandidate`, after geometry validation and before constructing `visibilityPayload`, add:

```slang
ReservoirGiTargetEvaluation targetEvaluation;
if (!estimateBrightSurfelTargetBeforeVisibility(hitPos, N, V, primaryPayload,
                                                surfel, dist2, targetEvaluation)) {
    ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelPrecheckRejectTargetOffset, 1u);
    return false;
}
```

- [ ] **Step 4: Reuse the prechecked target result after visibility**

Remove the second `ReservoirGiTargetEvaluation targetEvaluation = evaluateReservoirGiTargetAtPrimary(...)` call after visibility and keep the existing record fill:

```slang
candidateRecord.suffixRadiance = targetEvaluation.suffixRadiance;
candidateRecord.contribution = targetEvaluation.contribution;
candidateRecord.sourcePdf = targetEvaluation.sourcePdf;
candidateRecord.targetWeight = targetEvaluation.targetWeight;
candidateRecord.weightSum = targetEvaluation.targetWeight / max(targetEvaluation.sourcePdf, 0.000001);
candidateRecord.selectedWeight = candidateRecord.weightSum;
```

- [ ] **Step 5: Preserve post-visibility target rejection for safety**

Keep this check after visibility and before record fill:

```slang
if (!targetEvaluation.validGeometry ||
    !targetEvaluation.validLight ||
    targetEvaluation.targetWeight <= RESERVOIR_GI_BRIGHT_SURFEL_CANDIDATE_MIN_TARGET_WEIGHT) {
    ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelRejectTargetOffset, 1u);
    return false;
}
```

- [ ] **Step 6: Run shader build and tests**

Run:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" >nul && cmake --build cmake-build-debug --target LaphriaEditor --config Debug && ctest --test-dir cmake-build-debug --output-on-failure"
```

Expected: shader compilation passes. Tests still fail only if the training helper names or single-frame sweep rows are not implemented yet.

---

### Task 4: Add Sparse Bright Surfel Training Producer

**Files:**
- Modify: `src/shaders/Raygen.slang`

- [ ] **Step 1: Add sparse training gate**

Near `shouldAttemptBrightReceiverSurfel`, add:

```slang
bool shouldTrainBrightReceiverSurfel(uint2 launchID, uint2 launchSize)
{
    uint sourcePixel = launchID.y * launchSize.x + launchID.x;
    uint hash = pcgHash(sourcePixel * 747796405u ^
                        ubo.frameCount * 2891336453u ^
                        uint(launchSize.x) * 277803737u ^
                        uint(launchSize.y) * 2246822519u);
    return (hash & 7u) == 0u;
}
```

- [ ] **Step 2: Make surfel storage report whether it stored**

Change the signature of `storeReservoirGiBrightSurfelRecord` from:

```slang
void storeReservoirGiBrightSurfelRecord(uint2 launchID,
                                        uint2 launchSize,
                                        ReservoirGiRecord selectedRecord)
```

to:

```slang
bool storeReservoirGiBrightSurfelRecord(uint2 launchID,
                                        uint2 launchSize,
                                        ReservoirGiRecord selectedRecord)
```

For every existing early `return;` in that function, return `false` instead. At the end, after:

```slang
ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelStoreOffset, 1u);
```

add:

```slang
return true;
```

Existing call sites may ignore the boolean return value.

- [ ] **Step 3: Add training store helper**

Near `storeReservoirGiBrightSurfelRecord`, add:

```slang
bool tryStoreBrightSurfelTrainingCandidate(uint2 launchID,
                                           uint2 launchSize,
                                           ReservoirGiRecord candidateRecord)
{
    ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelTrainingAttemptOffset, 1u);
    float normalLengthSquared = dot(candidateRecord.candidateNormal, candidateRecord.candidateNormal);
    if ((candidateRecord.flags & 2u) == 0u ||
        !isSaneReservoirGiVector(candidateRecord.candidatePosition, 100000.0) ||
        !isSaneReservoirGiVector(candidateRecord.candidateNormal, 1.1) ||
        !isFinitePositive(normalLengthSquared) ||
        normalLengthSquared <= 0.0001) {
        ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelTrainingRejectGeometryOffset, 1u);
        return false;
    }
    if (!isFinitePositive(candidateRecord.targetWeight) ||
        candidateRecord.targetWeight < RESERVOIR_GI_BRIGHT_SURFEL_MIN_TARGET_WEIGHT ||
        pathTracerLuminance(max(candidateRecord.suffixRadiance, float3(0.0, 0.0, 0.0))) <
            RESERVOIR_GI_BRIGHT_SURFEL_MIN_LUMA) {
        ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelTrainingRejectTargetOffset, 1u);
        return false;
    }
    if (!storeReservoirGiBrightSurfelRecord(launchID, launchSize, candidateRecord)) {
        ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelTrainingRejectTargetOffset, 1u);
        return false;
    }
    ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelTrainingStoreOffset, 1u);
    return true;
}
```

- [ ] **Step 4: Train from valid local candidates**

In the local candidate loop inside `sampleFirstHitReservoirGiSingleFrame`, immediately after `candidatePositiveWeightCount += 1u;`, add:

```slang
if (useBrightReceiverSurfel &&
    shouldTrainBrightReceiverSurfel(launchID, launchSize)) {
    tryStoreBrightSurfelTrainingCandidate(launchID, launchSize, localRecord);
}
```

This trains the surfel pool from qualified local candidates before reservoir selection, without adding another path trace ray.

- [ ] **Step 5: Keep final selected-record storage as a fallback**

Leave the existing end-of-function `storeReservoirGiBrightSurfelRecord(launchID, launchSize, record);` path in place. It remains useful when the selected record is stronger than training candidates.

- [ ] **Step 6: Run shader build and tests**

Run:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" >nul && cmake --build cmake-build-debug --target LaphriaEditor --config Debug && ctest --test-dir cmake-build-debug --output-on-failure"
```

Expected: shader compilation passes. Tests still fail only if weighted selection or the sweep row is not implemented yet.

---

### Task 5: Make Global Surfel Selection Weighted With A Conservative PDF Proxy

**Files:**
- Modify: `src/shaders/Raygen.slang`

- [ ] **Step 1: Rename selector signature**

Replace:

```slang
bool selectGlobalBrightReceiverSurfelRecord(float3 hitPos,
                                            float3 N,
                                            uint2 launchID,
                                            uint2 launchSize,
                                            out ReservoirGiBrightSurfelRecord selectedSurfel)
```

with:

```slang
bool selectWeightedGlobalBrightReceiverSurfelRecord(float3 hitPos,
                                                    float3 N,
                                                    uint2 launchID,
                                                    uint2 launchSize,
                                                    inout uint rngState,
                                                    out ReservoirGiBrightSurfelRecord selectedSurfel,
                                                    out float surfelSelectionPdf)
```

Initialize `surfelSelectionPdf` to `0.0`.

- [ ] **Step 2: Replace best-score selection with weighted reservoir selection**

Inside the scan loop, replace deterministic `bestScore` selection with:

```slang
float scoreSum = 0.0;
float selectedScore = 0.0;
bool found = false;

[unroll]
for (uint slot = 0u; slot < RESERVOIR_GI_BRIGHT_SURFEL_GLOBAL_SCAN_COUNT; ++slot) {
    uint surfelIndex = reservoirGiBrightSurfelGlobalIndex(launchID, launchSize, slot);
    ReservoirGiBrightSurfelRecord surfel;
    if (!loadReservoirGiBrightSurfelHistoryRecord(surfelIndex, surfel)) {
        continue;
    }

    float3 toSurfel = surfel.position - hitPos;
    float dist2 = dot(toSurfel, toSurfel);
    if (dist2 <= 0.0001 || !isFinitePositive(dist2)) {
        continue;
    }

    float3 wi = toSurfel * rsqrt(dist2);
    float receiverCos = dot(N, wi);
    float surfelCos = dot(surfel.normal, -wi);
    if (receiverCos <= 0.0 ||
        surfelCos <= RESERVOIR_GI_BRIGHT_SURFEL_COS_THETA_MAX ||
        !isSaneReservoirGiVector(wi, 1.1)) {
        continue;
    }

    float score = pathTracerLuminance(max(surfel.radiance, float3(0.0, 0.0, 0.0))) *
                  max(receiverCos, 0.0) *
                  max(surfelCos, 0.0) /
                  max(dist2, 0.0001);
    if (!isFinitePositive(score)) {
        continue;
    }

    scoreSum += score;
    if (!found || randomFloat(rngState) < score / max(scoreSum, 0.000001)) {
        selectedSurfel = surfel;
        selectedScore = score;
        found = true;
    }
}

if (!found || !isFinitePositive(scoreSum) || !isFinitePositive(selectedScore)) {
    ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelMissOffset, 1u);
    return false;
}

surfelSelectionPdf = selectedScore / max(scoreSum, 0.000001);
ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelHitOffset, 1u);
return true;
```

- [ ] **Step 3: Thread `rngState` and PDF through candidate evaluation**

Change `evaluateBrightReceiverSurfelReservoirGiCandidate` to accept `inout uint rngState`:

```slang
bool evaluateBrightReceiverSurfelReservoirGiCandidate(float3 hitPos,
                                                      float3 N,
                                                      float3 V,
                                                      RayPayload primaryPayload,
                                                      uint2 launchID,
                                                      uint2 launchSize,
                                                      inout uint rngState,
                                                      out ReservoirGiRecord candidateRecord)
```

Replace the selector call with:

```slang
float surfelSelectionPdf = 0.0;
ReservoirGiBrightSurfelRecord surfel;
if (!selectWeightedGlobalBrightReceiverSurfelRecord(hitPos, N, launchID, launchSize,
                                                    rngState, surfel, surfelSelectionPdf)) {
    return false;
}
```

- [ ] **Step 4: Keep target evaluation on the conservative PDF proxy**

Do not feed `surfelSelectionPdf` directly into `evaluateReservoirGiTargetAtPrimary` in this pass. `surfelSelectionPdf` is the conditional probability inside the small scanned subset, not the absolute probability of proposing this surfel from the full pool. Using it as `sourcePdf` would divide contribution by a too-small probability and can over-brighten the estimator.

Keep the target-evaluation source PDF conservative:

```slang
// The weighted selector reports conditional selection probability for diagnostics.
// Keep the source PDF proxy conservative until the virtual-light proposal has a
// full absolute sampling PDF over the surfel pool.
float sourcePdf = 1.0;
targetEvaluation = evaluateReservoirGiTargetAtPrimary(hitPos, N, V, primaryPayload,
                                                      surfel.position, surfel.normal,
                                                      suffixRadiance, sourcePdf);
```

Leave `surfelSelectionPdf` available in the function so a later virtual-light reservoir pass can convert it into a physically meaningful proposal PDF.

- [ ] **Step 5: Update call site**

In `sampleFirstHitReservoirGiSingleFrame`, change the call to:

```slang
if (evaluateBrightReceiverSurfelReservoirGiCandidate(hitPos, N, V, payload,
                                                     launchID, launchSize,
                                                     rngState,
                                                     surfelRecord)) {
```

- [ ] **Step 6: Run shader build and tests**

Run:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" >nul && cmake --build cmake-build-debug --target LaphriaEditor --config Debug && ctest --test-dir cmake-build-debug --output-on-failure"
```

Expected: shader compilation passes. Tests still fail only if the single-frame sweep rows are not implemented yet.

---

### Task 6: Add Single-Frame Sun Receiver Control Rows

**Files:**
- Modify: `src/Core/EngineCore.cpp`
- Modify: `tests/PathTracerAnalysisTests.cpp`

- [ ] **Step 1: Create single-frame Sun Receiver control row**

In `startPathTracerSponzaGiPerfSweep`, after `reservoirMixedTemporalSpatialBudget2SunReceiverRow`, create:

```cpp
auto reservoirMixedSingleFrameSunReceiverRow =
    makeReservoirRow(scenario,
                     "Reservoir 1C Shadowed Sun First Mixed Single Frame Sun Receiver",
                     1,
                     UISystem::PathTracerReservoirGiProposalMode::MixedCosineSunReceiverGuided);
reservoirMixedSingleFrameSunReceiverRow.reservoirGiMode =
    UISystem::PathTracerReservoirGiMode::SingleFrame;
reservoirMixedSingleFrameSunReceiverRow.reservoirGiTemporalBudgetDivisor = 1;
reservoirMixedSingleFrameSunReceiverRow.reservoirGiSpatialBudgetDivisor = 1;
reservoirMixedSingleFrameSunReceiverRow.reservoirGiCandidateEvaluationMode = 2;
```

- [ ] **Step 2: Create single-frame bright surfel row**

In `startPathTracerSponzaGiPerfSweep`, after `reservoirMixedTemporalSpatialBudget2SunReceiverBrightSurfelRow`, create:

```cpp
auto reservoirMixedSingleFrameSunReceiverBrightSurfelRow =
    makeReservoirRow(scenario,
                     "Reservoir 1C Shadowed Sun First Mixed Single Frame Sun Receiver Bright Surfel",
                     1,
                     UISystem::PathTracerReservoirGiProposalMode::MixedCosineSunReceiverBrightSurfel);
reservoirMixedSingleFrameSunReceiverBrightSurfelRow.reservoirGiMode =
    UISystem::PathTracerReservoirGiMode::SingleFrame;
reservoirMixedSingleFrameSunReceiverBrightSurfelRow.reservoirGiTemporalBudgetDivisor = 1;
reservoirMixedSingleFrameSunReceiverBrightSurfelRow.reservoirGiSpatialBudgetDivisor = 1;
reservoirMixedSingleFrameSunReceiverBrightSurfelRow.reservoirGiCandidateEvaluationMode = 2;
```

- [ ] **Step 3: Push both rows into the focused sweep**

Push the non-surfel single-frame control after the temporal-spatial Sun Receiver row, and push the single-frame bright surfel row immediately after the temporal-spatial bright surfel row:

```cpp
ptExperimentRows.push_back(reservoirMixedTemporalSpatialBudget2SunReceiverRow);
ptExperimentRows.push_back(reservoirMixedSingleFrameSunReceiverRow);
ptExperimentRows.push_back(reservoirMixedTemporalSpatialBudget2SunReceiverBrightSurfelRow);
ptExperimentRows.push_back(reservoirMixedSingleFrameSunReceiverBrightSurfelRow);
```

Keep the existing env-first-two and cache-continuation rows after these controls.

- [ ] **Step 4: Update row-count-sensitive tests**

If `tests/PathTracerAnalysisTests.cpp` checks the focused row set by exact names, include:

```cpp
"Reservoir 1C Shadowed Sun First Mixed Single Frame Sun Receiver",
"Reservoir 1C Shadowed Sun First Mixed Single Frame Sun Receiver Bright Surfel",
```

If it checks the number of rows per scenario, increase that expectation by two rows per scenario.

- [ ] **Step 5: Run full verification**

Run:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" >nul && cmake --build cmake-build-debug --target LaphriaEditor --config Debug && ctest --test-dir cmake-build-debug --output-on-failure && git diff --check -- src\Core\EngineAuxiliary.h src\Core\UISystem.h src\Core\UISystem.cpp src\Core\EngineCore.h src\Core\EngineCore.cpp src\shaders\Raygen.slang tests\PathTracerAnalysisTests.cpp"
```

Expected:

```text
LaphriaEditor builds successfully, including Raygen.slang.
100% tests passed, 0 tests failed out of 5.
git diff --check reports no whitespace errors.
```

---

### Task 7: Run Focused Sweep And Record Results

**Files:**
- Modify: `docs/superpowers/plans/2026-05-17-global-bright-surfel-sampling.md`

- [ ] **Step 1: Run the Sponza PT/GI audit sweep**

Use the in-app `Run Sponza GI Perf Sweep` control with `Sponza Sweep Warmup = 8` and `Sponza Sweep Samples = 32`.

- [ ] **Step 2: Compare the key rows**

Record these rows for `Dark Courtyard`, `Sunlit Courtyard Wall`, and `Mid-Depth Interior`:

```text
Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2
Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Sun Receiver
Reservoir 1C Shadowed Sun First Mixed Single Frame Sun Receiver
Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Sun Receiver Bright Surfel
Reservoir 1C Shadowed Sun First Mixed Single Frame Sun Receiver Bright Surfel
Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Sun Receiver Env First Two
```

- [ ] **Step 3: Decide pass/follow-up using numeric gates**

Treat the implementation as a useful step if all of these are true:

```text
No row uses stale surfels from a previous Sponza scenario or variant.
brightSurfelAttempt stays at or below the current sparse global level.
brightSurfelPrecheckRejectTarget is nonzero in Mid-depth.
Sunlit Bright Surfel totalMs decreases relative to the previous 230 ms result.
Sunlit Bright Surfel firstHitProbeAvgLuma stays within 10% of the previous 0.06534 result.
Mid-depth Single Frame Sun Receiver is stable enough to use as a control.
Mid-depth Single Frame Bright Surfel separates ReSTIR temporal/spatial artifacting from surfel-specific history.
```

Treat it as needing another surfel-producer iteration if either of these is true:

```text
brightSurfelTrainingStore remains near zero in Dark Courtyard and Mid-depth.
reservoirGiSelectedBrightSurfel remains zero in Mid-depth while brightSurfelHit is high.
```

- [ ] **Step 4: Append result note to the previous surfel plan**

Append a `## Follow-Up: Sparse Training Result` section to `docs/superpowers/plans/2026-05-17-global-bright-surfel-sampling.md`.

The section must include the exact row logs from the app output for the five rows listed in Step 2, grouped by scenario. Do not summarize the numeric row summaries before saving the note.

End the section with exactly one of these decision lines:

```text
Continue with surfel producer quality tuning.
Continue with many-light-compatible virtual light reservoir integration.
Pause surfel work and investigate temporal estimator artifacting.
```

---

## Self-Review Checklist

- Every new counter has matching shader offset, C++ counter field, UI stat field, UI label, sweep accumulator field, log field, and offset test.
- Sponza sweep row transitions clear sparse GI state before the next row starts.
- The precheck runs before the visibility `TraceRay` in `evaluateBrightReceiverSurfelReservoirGiCandidate`.
- The sparse training producer stores local valid candidates before reservoir selection and does not add extra path trace rays.
- The weighted surfel selector reports `surfelSelectionPdf`, but target evaluation keeps `sourcePdf = 1.0` until an absolute surfel-pool PDF is implemented.
- The single-frame Sun Receiver and bright surfel rows both use `reservoirGiMode = SingleFrame`.
- Full verification includes `LaphriaEditor`, CTest, and `git diff --check`.
