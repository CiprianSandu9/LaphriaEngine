# Target-Aware Bright Surfel Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make bright surfel reuse receiver-relevant by scoring scanned surfels with the same target estimate used for acceptance before selecting and tracing visibility.

**Architecture:** Keep the sparse surfel producer and existing global pool, but change the consumer from brightness-weighted selection followed by target rejection into target-aware selection. The selector will estimate receiver target weight for every scanned surfel, reject target-useless surfels before selection, select among only viable target candidates, and return the chosen target evaluation so the caller does not recompute it. Add diagnostics that distinguish selector target rejects from post-selection/pre-visibility rejects.

**Tech Stack:** C++17 counter/UI/sweep plumbing, Slang ray generation shader, Vulkan counter buffer layout, `tests/PathTracerAnalysisTests.cpp` string-contract tests, CMake/CTest verification, in-app Sponza PT/GI audit sweep.

---

## Current Evidence

The sparse training sweep shows the surfel path is wired but not receiver-useful in two scenarios:

```text
Dark Courtyard Bright Surfel:
  brightSurfelHit=6.6
  brightSurfelPrecheckRejectTarget=6.6
  brightSurfelAccepted=0.0
  reservoirGiSelectedBrightSurfel=0.0

Mid-Depth Interior Bright Surfel:
  brightSurfelHit=96224.6
  brightSurfelPrecheckRejectTarget=96224.6
  brightSurfelAccepted=0.0
  reservoirGiSelectedBrightSurfel=0.0

Sunlit Courtyard Wall Bright Surfel:
  brightSurfelHit=298432.0
  brightSurfelPrecheckRejectTarget=297776.3
  brightSurfelAccepted=650.4
  reservoirGiSelectedBrightSurfel=603.4
```

Root cause hypothesis for this plan:

```text
The current selector chooses bright/near-ish surfels, then most selected surfels fail receiver target evaluation.
The selector should score by receiver target weight before visibility, not by brightness alone.
```

Success for this plan is not "more surfels." Success is either:

```text
1. Mid-depth bright surfel accepted/selected counts become nonzero without excessive cost, or
2. new selector-reject counters prove the scanned pool has no target-viable candidates for Mid-depth, which means the next fix must be producer/index quality, not selector scoring.
```

## File Structure

- Modify `tests/PathTracerAnalysisTests.cpp`
  - Add counter offsets, required shader helper symbols, UI labels, and row-summary log fields for selector target diagnostics.
- Modify `src/Core/EngineAuxiliary.h`
  - Append bright surfel selector reject counters to `PathTracerAnalysisCounters`.
- Modify `src/Core/UISystem.h`
  - Mirror selector reject counters in `PathTracerPerfStats`.
- Modify `src/Core/UISystem.cpp`
  - Display selector reject counters in the path tracer diagnostics panel.
- Modify `src/Core/EngineCore.h`
  - Add sweep accumulator fields for selector reject counters.
- Modify `src/Core/EngineCore.cpp`
  - Copy, accumulate, and log selector reject counters.
- Modify `src/shaders/Raygen.slang`
  - Refactor target estimation into a reusable reason-coded helper.
  - Make `selectWeightedGlobalBrightReceiverSurfelRecord` target-aware.
  - Return the selected `ReservoirGiTargetEvaluation` to the caller.
- Modify `docs/superpowers/plans/2026-05-17-global-bright-surfel-sampling.md`
  - Append a short result note after the follow-up sweep.

## Counter Layout

Append these fields after `reservoirGiBrightSurfelTrainingRejectTarget` so existing offsets remain stable:

```cpp
uint32_t reservoirGiBrightSurfelSelectorRejectGeometry = 0; // offset 364
uint32_t reservoirGiBrightSurfelSelectorRejectTarget = 0;   // offset 368
uint32_t reservoirGiBrightSurfelSelectorViable = 0;         // offset 372
```

Use matching shader offsets:

```slang
static const uint reservoirGiBrightSurfelSelectorRejectGeometryOffset = 364u;
static const uint reservoirGiBrightSurfelSelectorRejectTargetOffset = 368u;
static const uint reservoirGiBrightSurfelSelectorViableOffset = 372u;
```

`SelectorRejectGeometry` means the scanned surfel could not be used as a receiver candidate because of invalid distance, direction, cosine, or vector sanity.

`SelectorRejectTarget` means geometry was sane but `evaluateReservoirGiTargetAtPrimary` produced invalid/no-light/too-small target weight.

`SelectorViable` counts scanned surfels that passed target estimation and entered weighted selection.

---

### Task 1: Add Failing Counter And Shader Contract Tests

**Files:**
- Modify: `tests/PathTracerAnalysisTests.cpp`

- [ ] **Step 1: Add counter offset expectations**

In the `counterOffsets` table, immediately after:

```cpp
{"reservoirGiBrightSurfelTrainingRejectTarget",
 offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelTrainingRejectTarget), 360u}
```

replace the table tail with:

```cpp
{"reservoirGiBrightSurfelTrainingRejectTarget",
 offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelTrainingRejectTarget), 360u},
{"reservoirGiBrightSurfelSelectorRejectGeometry",
 offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelSelectorRejectGeometry), 364u},
{"reservoirGiBrightSurfelSelectorRejectTarget",
 offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelSelectorRejectTarget), 368u},
{"reservoirGiBrightSurfelSelectorViable",
 offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelSelectorViable), 372u}};
```

- [ ] **Step 2: Add required shader symbols**

In both bright surfel shader symbol lists, add:

```cpp
"reservoirGiBrightSurfelSelectorRejectGeometryOffset",
"reservoirGiBrightSurfelSelectorRejectTargetOffset",
"reservoirGiBrightSurfelSelectorViableOffset",
"BrightSurfelTargetEstimateResult",
"BRIGHT_SURFEL_TARGET_REJECT_NONE",
"BRIGHT_SURFEL_TARGET_REJECT_GEOMETRY",
"BRIGHT_SURFEL_TARGET_REJECT_TARGET",
"estimateBrightSurfelTargetForReceiver",
"selectedTargetEvaluation",
```

In both bright surfel shader symbol lists, remove the old helper name:

```cpp
"estimateBrightSurfelTargetBeforeVisibility",
```

Keep the existing symbols:

```cpp
"selectWeightedGlobalBrightReceiverSurfelRecord",
"surfelSelectionPdf",
```

- [ ] **Step 3: Add UI/counter required strings**

In `requiredCounterAndUiSymbols`, add the C++ field and UI labels:

```cpp
"reservoirGiBrightSurfelSelectorRejectGeometry",
"reservoirGiBrightSurfelSelectorRejectTarget",
"reservoirGiBrightSurfelSelectorViable",
"Reservoir GI Bright Surfel Selector Reject Geometry",
"Reservoir GI Bright Surfel Selector Reject Target",
"Reservoir GI Bright Surfel Selector Viable",
```

- [ ] **Step 4: Add row-summary log-field required strings**

In the focused row-summary field contract that checks `engineCore`, add:

```cpp
"brightSurfelSelectorRejectGeometry",
"brightSurfelSelectorRejectTarget",
"brightSurfelSelectorViable",
```

- [ ] **Step 5: Run red test**

Run:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" >nul && cmake --build cmake-build-debug --target LaphriaEngineUnitTests --config Debug"
```

Expected:

```text
FAIL: reservoirGiBrightSurfelSelectorRejectGeometry is not a member of Laphria::PathTracerAnalysisCounters
```

- [ ] **Step 6: Check whitespace**

Run:

```powershell
git diff --check -- tests\PathTracerAnalysisTests.cpp
```

Expected:

```text
No whitespace errors. LF-to-CRLF warnings are acceptable.
```

---

### Task 2: Add Counter Plumbing

**Files:**
- Modify: `src/Core/EngineAuxiliary.h`
- Modify: `src/Core/UISystem.h`
- Modify: `src/Core/UISystem.cpp`
- Modify: `src/Core/EngineCore.h`
- Modify: `src/Core/EngineCore.cpp`

- [ ] **Step 1: Add C++ counter fields**

In `src/Core/EngineAuxiliary.h`, after `reservoirGiBrightSurfelTrainingRejectTarget`, add:

```cpp
uint32_t reservoirGiBrightSurfelSelectorRejectGeometry = 0;
uint32_t reservoirGiBrightSurfelSelectorRejectTarget = 0;
uint32_t reservoirGiBrightSurfelSelectorViable = 0;
```

- [ ] **Step 2: Add UI perf stats**

In `src/Core/UISystem.h`, after `reservoirGiBrightSurfelTrainingRejectTarget`, add:

```cpp
uint32_t reservoirGiBrightSurfelSelectorRejectGeometry = 0;
uint32_t reservoirGiBrightSurfelSelectorRejectTarget = 0;
uint32_t reservoirGiBrightSurfelSelectorViable = 0;
```

- [ ] **Step 3: Add UI labels**

In `src/Core/UISystem.cpp`, after the existing bright surfel training reject target text, add:

```cpp
ImGui::Text("Reservoir GI Bright Surfel Selector Reject Geometry: %u",
            pathTracerPerfStats.reservoirGiBrightSurfelSelectorRejectGeometry);
ImGui::Text("Reservoir GI Bright Surfel Selector Reject Target: %u",
            pathTracerPerfStats.reservoirGiBrightSurfelSelectorRejectTarget);
ImGui::Text("Reservoir GI Bright Surfel Selector Viable: %u",
            pathTracerPerfStats.reservoirGiBrightSurfelSelectorViable);
```

- [ ] **Step 4: Add sweep accumulator fields**

In `src/Core/EngineCore.h`, after `brightSurfelTrainingRejectTarget`, add:

```cpp
double brightSurfelSelectorRejectGeometry = 0.0;
double brightSurfelSelectorRejectTarget = 0.0;
double brightSurfelSelectorViable = 0.0;
```

- [ ] **Step 5: Copy mapped counters**

In `EngineCore::collectPathTracerAnalysisCounters`, after copying `reservoirGiBrightSurfelTrainingRejectTarget`, add:

```cpp
ui.pathTracerPerfStats.reservoirGiBrightSurfelSelectorRejectGeometry =
    counters->reservoirGiBrightSurfelSelectorRejectGeometry;
ui.pathTracerPerfStats.reservoirGiBrightSurfelSelectorRejectTarget =
    counters->reservoirGiBrightSurfelSelectorRejectTarget;
ui.pathTracerPerfStats.reservoirGiBrightSurfelSelectorViable =
    counters->reservoirGiBrightSurfelSelectorViable;
```

- [ ] **Step 6: Accumulate sweep fields**

In `EngineCore::updatePathTracerExperimentSweep`, after accumulating `brightSurfelTrainingRejectTarget`, add:

```cpp
ptExperimentAccum.brightSurfelSelectorRejectGeometry +=
    static_cast<double>(stats.reservoirGiBrightSurfelSelectorRejectGeometry);
ptExperimentAccum.brightSurfelSelectorRejectTarget +=
    static_cast<double>(stats.reservoirGiBrightSurfelSelectorRejectTarget);
ptExperimentAccum.brightSurfelSelectorViable +=
    static_cast<double>(stats.reservoirGiBrightSurfelSelectorViable);
```

- [ ] **Step 7: Log sweep fields**

In `EngineCore::logPathTracerExperimentRow`, append this text after `brightSurfelTrainingRejectTarget=%.1f`:

```cpp
"brightSurfelSelectorRejectGeometry=%.1f, brightSurfelSelectorRejectTarget=%.1f, "
"brightSurfelSelectorViable=%.1f, "
```

Append these arguments after `accum.brightSurfelTrainingRejectTarget * invSamples`:

```cpp
accum.brightSurfelSelectorRejectGeometry * invSamples,
accum.brightSurfelSelectorRejectTarget * invSamples,
accum.brightSurfelSelectorViable * invSamples,
```

- [ ] **Step 8: Run tests**

Run:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" >nul && cmake --build cmake-build-debug --target LaphriaEngineUnitTests --config Debug && cmake-build-debug\LaphriaEngineUnitTests.exe"
```

Expected:

```text
FAIL: missing shader symbol BrightSurfelTargetEstimateResult
```

- [ ] **Step 9: Check whitespace**

Run:

```powershell
git diff --check -- src\Core\EngineAuxiliary.h src\Core\UISystem.h src\Core\UISystem.cpp src\Core\EngineCore.h src\Core\EngineCore.cpp
```

Expected:

```text
No whitespace errors. LF-to-CRLF warnings are acceptable.
```

---

### Task 3: Refactor Bright Surfel Target Estimation With Reject Reasons

**Files:**
- Modify: `src/shaders/Raygen.slang`

- [ ] **Step 1: Add shader counter offsets**

After `reservoirGiBrightSurfelTrainingRejectTargetOffset`, add:

```slang
static const uint reservoirGiBrightSurfelSelectorRejectGeometryOffset = 364u;
static const uint reservoirGiBrightSurfelSelectorRejectTargetOffset = 368u;
static const uint reservoirGiBrightSurfelSelectorViableOffset = 372u;
```

- [ ] **Step 2: Add reason constants and result struct**

Near `ReservoirGiBrightSurfelRecord`, add:

```slang
static const uint BRIGHT_SURFEL_TARGET_REJECT_NONE = 0u;
static const uint BRIGHT_SURFEL_TARGET_REJECT_GEOMETRY = 1u;
static const uint BRIGHT_SURFEL_TARGET_REJECT_TARGET = 2u;

struct BrightSurfelTargetEstimateResult {
    ReservoirGiTargetEvaluation targetEvaluation;
    uint rejectReason;
};
```

- [ ] **Step 3: Replace target estimator with reason-coded helper**

Replace `estimateBrightSurfelTargetBeforeVisibility(...)` with:

```slang
bool estimateBrightSurfelTargetForReceiver(float3 hitPos,
                                           float3 N,
                                           float3 V,
                                           RayPayload primaryPayload,
                                           ReservoirGiBrightSurfelRecord surfel,
                                           out BrightSurfelTargetEstimateResult estimate)
{
    estimate.targetEvaluation = evaluateReservoirGiTargetAtPrimary(
        hitPos, N, V, primaryPayload,
        hitPos, float3(0.0, 1.0, 0.0),
        float3(0.0, 0.0, 0.0), 1.0);
    estimate.rejectReason = BRIGHT_SURFEL_TARGET_REJECT_GEOMETRY;

    float3 toSurfel = surfel.position - hitPos;
    float dist2 = dot(toSurfel, toSurfel);
    if (dist2 <= 0.0001 || !isFinitePositive(dist2)) {
        return false;
    }

    float3 wi = toSurfel * rsqrt(dist2);
    float receiverCos = dot(N, wi);
    float surfelCos = dot(surfel.normal, -wi);
    if (receiverCos <= 0.0 ||
        surfelCos <= RESERVOIR_GI_BRIGHT_SURFEL_COS_THETA_MAX ||
        !isSaneReservoirGiVector(wi, 1.1)) {
        return false;
    }

    float areaProxy = PI * max(surfel.radius * surfel.radius, 0.0001);
    float geometry = max(surfelCos, 0.0) * areaProxy / max(dist2, 0.0001);
    float3 suffixRadiance = sanitizeReservoirGiContribution(
        max(surfel.radiance, float3(0.0, 0.0, 0.0)) * geometry);
    // The scan-space selection PDF is diagnostic only; it is not an absolute PDF over the surfel pool.
    float sourcePdf = 1.0;

    estimate.targetEvaluation = evaluateReservoirGiTargetAtPrimary(hitPos, N, V, primaryPayload,
                                                                   surfel.position, surfel.normal,
                                                                   suffixRadiance, sourcePdf);
    estimate.rejectReason = BRIGHT_SURFEL_TARGET_REJECT_TARGET;
    if (!estimate.targetEvaluation.validGeometry ||
        !estimate.targetEvaluation.validLight ||
        estimate.targetEvaluation.targetWeight <= RESERVOIR_GI_BRIGHT_SURFEL_CANDIDATE_MIN_TARGET_WEIGHT) {
        return false;
    }

    estimate.rejectReason = BRIGHT_SURFEL_TARGET_REJECT_NONE;
    return true;
}
```

- [ ] **Step 4: Preserve existing caller behavior**

In `evaluateBrightReceiverSurfelReservoirGiCandidate`, replace:

```slang
ReservoirGiTargetEvaluation targetEvaluation;
if (!estimateBrightSurfelTargetBeforeVisibility(hitPos, N, V, primaryPayload,
                                                surfel, targetEvaluation)) {
    ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelPrecheckRejectTargetOffset, 1u);
    return false;
}
```

with:

```slang
BrightSurfelTargetEstimateResult selectedTargetEstimate;
if (!estimateBrightSurfelTargetForReceiver(hitPos, N, V, primaryPayload,
                                           surfel, selectedTargetEstimate)) {
    ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelPrecheckRejectTargetOffset, 1u);
    return false;
}
ReservoirGiTargetEvaluation targetEvaluation = selectedTargetEstimate.targetEvaluation;
```

- [ ] **Step 5: Run shader build**

Run:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" >nul && cmake --build cmake-build-debug --target LaphriaEditor --config Debug"
```

Expected:

```text
Raygen.slang compiles.
```

- [ ] **Step 6: Run unit tests**

Run:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" >nul && cmake --build cmake-build-debug --target LaphriaEngineUnitTests --config Debug && cmake-build-debug\LaphriaEngineUnitTests.exe"
```

Expected:

```text
FAIL: missing shader symbol selectedTargetEvaluation
```

---

### Task 4: Make The Selector Target-Aware

**Files:**
- Modify: `src/shaders/Raygen.slang`

- [ ] **Step 1: Change selector signature**

Change `selectWeightedGlobalBrightReceiverSurfelRecord` to:

```slang
bool selectWeightedGlobalBrightReceiverSurfelRecord(float3 hitPos,
                                                    float3 N,
                                                    float3 V,
                                                    RayPayload primaryPayload,
                                                    uint2 launchID,
                                                    uint2 launchSize,
                                                    inout uint rngState,
                                                    out ReservoirGiBrightSurfelRecord selectedSurfel,
                                                    out ReservoirGiTargetEvaluation selectedTargetEvaluation,
                                                    out float surfelSelectionPdf)
```

Initialize `selectedTargetEvaluation` the same way as the current zero target estimate:

```slang
selectedTargetEvaluation = evaluateReservoirGiTargetAtPrimary(
    hitPos, N, V, primaryPayload,
    hitPos, float3(0.0, 1.0, 0.0),
    float3(0.0, 0.0, 0.0), 1.0);
```

- [ ] **Step 2: Score scanned surfels by target estimate**

Inside the scan loop, replace the per-record geometry gates and brightness score block:

```slang
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
```

with:

```slang
BrightSurfelTargetEstimateResult estimate;
if (!estimateBrightSurfelTargetForReceiver(hitPos, N, V, primaryPayload,
                                           surfel, estimate)) {
    if (estimate.rejectReason == BRIGHT_SURFEL_TARGET_REJECT_GEOMETRY) {
        ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorRejectGeometryOffset, 1u);
    } else {
        ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorRejectTargetOffset, 1u);
    }
    continue;
}

float score = estimate.targetEvaluation.targetWeight;
if (!isFinitePositive(score)) {
    ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorRejectTargetOffset, 1u);
    continue;
}
ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorViableOffset, 1u);
```

When selecting, also save the target evaluation:

```slang
if (randomFloat(rngState) < score / max(scoreSum, 0.000001)) {
    selectedScore = score;
    selectedSurfel = surfel;
    selectedTargetEvaluation = estimate.targetEvaluation;
    found = true;
}
```

- [ ] **Step 3: Update evaluator call**

In `evaluateBrightReceiverSurfelReservoirGiCandidate`, replace:

```slang
float surfelSelectionPdf;
if (!selectWeightedGlobalBrightReceiverSurfelRecord(hitPos, N, launchID, launchSize,
                                                   rngState, surfel, surfelSelectionPdf)) {
    return false;
}
```

with:

```slang
ReservoirGiTargetEvaluation selectedTargetEvaluation;
float surfelSelectionPdf;
if (!selectWeightedGlobalBrightReceiverSurfelRecord(hitPos, N, V, primaryPayload,
                                                   launchID, launchSize,
                                                   rngState, surfel, selectedTargetEvaluation,
                                                   surfelSelectionPdf)) {
    return false;
}
```

- [ ] **Step 4: Remove duplicate precheck**

Replace the post-selector precheck block:

```slang
BrightSurfelTargetEstimateResult selectedTargetEstimate;
if (!estimateBrightSurfelTargetForReceiver(hitPos, N, V, primaryPayload,
                                           surfel, selectedTargetEstimate)) {
    ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelPrecheckRejectTargetOffset, 1u);
    return false;
}
ReservoirGiTargetEvaluation targetEvaluation = selectedTargetEstimate.targetEvaluation;
```

with:

```slang
ReservoirGiTargetEvaluation targetEvaluation = selectedTargetEvaluation;
```

Keep the existing post-visibility safety guard:

```slang
if (!targetEvaluation.validGeometry ||
    !targetEvaluation.validLight ||
    targetEvaluation.targetWeight <= RESERVOIR_GI_BRIGHT_SURFEL_CANDIDATE_MIN_TARGET_WEIGHT) {
    ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelRejectTargetOffset, 1u);
    return false;
}
```

- [ ] **Step 5: Run shader build**

Run:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" >nul && cmake --build cmake-build-debug --target LaphriaEditor --config Debug"
```

Expected:

```text
Raygen.slang compiles.
```

- [ ] **Step 6: Run unit tests**

Run:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" >nul && cmake --build cmake-build-debug --target LaphriaEngineUnitTests --config Debug && cmake-build-debug\LaphriaEngineUnitTests.exe"
```

Expected:

```text
PASS.
```

---

### Task 5: Full Verification And Sweep

**Files:**
- Modify: `docs/superpowers/plans/2026-05-17-global-bright-surfel-sampling.md`

- [ ] **Step 1: Run full verification**

Run:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" >nul && cmake --build cmake-build-debug --target LaphriaEditor --config Debug && ctest --test-dir cmake-build-debug --output-on-failure && git diff --check -- src\Core\EngineAuxiliary.h src\Core\UISystem.h src\Core\UISystem.cpp src\Core\EngineCore.h src\Core\EngineCore.cpp src\shaders\Raygen.slang tests\PathTracerAnalysisTests.cpp"
```

Expected:

```text
LaphriaEditor builds.
100% tests passed, 0 tests failed out of 5.
git diff --check reports no whitespace errors.
```

- [ ] **Step 2: Run focused Sponza sweep**

Use the in-app `Run Sponza GI Perf Sweep` control:

```text
Sponza Sweep Warmup = 8
Sponza Sweep Samples = 32
```

Record these rows for all scenarios:

```text
Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Sun Receiver
Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Sun Receiver Bright Surfel
Reservoir 1C Shadowed Sun First Mixed Single Frame Sun Receiver
Reservoir 1C Shadowed Sun First Mixed Single Frame Sun Receiver Bright Surfel
```

- [ ] **Step 3: Interpret the numeric gates**

Treat target-aware selection as an improvement if:

```text
Mid-Depth Bright Surfel reservoirGiSelectedBrightSurfel > 0
Mid-Depth Bright Surfel brightSurfelAccepted > 0
Mid-Depth Bright Surfel brightSurfelSelectorViable > 0
Mid-Depth Bright Surfel totalMs is not more than 1.25x Sun Receiver totalMs
Sunlit Bright Surfel firstHitProbeAvgLuma remains within 10% of Sun Receiver luma
Dark Courtyard does not gain nonzero bright surfel contribution with visible leaks or pinpricks
```

Treat the result as "producer/index failure, not selector failure" if:

```text
Mid-Depth brightSurfelSelectorViable remains zero while brightSurfelTrainingStore is nonzero
Mid-Depth brightSurfelSelectorRejectTarget dominates scanned candidates
```

- [ ] **Step 4: Append result note**

Append this section to `docs/superpowers/plans/2026-05-17-global-bright-surfel-sampling.md`:

```markdown
## Follow-Up: Target-Aware Selection Result

Local run date: 2026-05-17.
Build identity: local Debug build after target-aware bright surfel selection.

Add a result table with these exact columns: `Scenario`, `Row`, `Luma`, `selector viable`, `selector reject target`, `accepted bright`, `selected bright`, and `total ms`.
Include rows for Dark Courtyard, Sunlit Courtyard Wall, and Mid-Depth Interior, with both the Sun Receiver control and Bright Surfel variant for each scenario.
Copy every numeric value directly from the app row summaries generated by the sweep.

Decision: use one of the decision lines listed below.
```

Use exactly one decision line:

```text
Continue with target-aware surfel selection.
Move to producer/index quality.
Pause surfel work and prioritize many-light virtual reservoir integration.
```

---

## Self-Review Checklist

- New counters are appended after existing bright surfel counters; no existing counter offsets change.
- Selector target reject counters are distinct from post-selection precheck counters.
- `selectWeightedGlobalBrightReceiverSurfelRecord` scores by `targetEvaluation.targetWeight`, not luminance-only brightness.
- `surfelSelectionPdf` remains diagnostic and is not used as `sourcePdf`.
- Visibility rays are traced only after target-aware selection chooses a viable surfel.
- The post-visibility safety target guard remains in place.
- Full verification includes `LaphriaEditor`, CTest, and `git diff --check`.
