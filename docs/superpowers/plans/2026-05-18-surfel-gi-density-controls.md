# Surfel GI Density Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SurfelGI-style density controls so the compact surfel grid stops producing a few giant cells and millions of pathological evaluation candidates in the Sponza debug sweep.

**Architecture:** Keep the current compact count/allocate/build/evaluate cell-list pipeline, but add the missing original-repo safeguards around it: explicit dense-cell diagnostics, a dense-cell lookup bailout, and a finer radius/cell relationship. This is diagnostic-first: it should make dense cells visible and cheap before we attempt full persistent surfel lifecycle or coverage-based replacement.

**Tech Stack:** C++17 engine/tests, Slang compute shaders, Vulkan storage buffers, existing `PathTracerAnalysisTests` text-contract test harness.

---

## File Structure

- Modify `src/shaders/SurfelCommon.slang`
  - Add density constants.
  - Add new analysis counter offsets.
  - Add a shared projected-radius helper.
  - Tune grid/radius constants to reduce clustering.

- Modify `src/shaders/SurfelGenerate.slang`
  - Store projected surfel radius instead of fixed `SURFEL_GI_MIN_RADIUS`.

- Modify `src/shaders/SurfelEvaluate.slang`
  - Add SurfelGI-style dense-cell bailout before sampling membership lists.
  - Write a distinct debug color for dense-cell rejection.
  - Increment dense-cell counters.

- Modify `src/Core/FrameContext.h`
  - Keep host-side cell count aligned with shader grid dimension.
  - Preserve the previous camera-relative grid coverage while increasing resolution.

- Modify `src/Core/EngineAuxiliary.h`, `src/Core/EngineCore.h`, `src/Core/EngineCore.cpp`, `src/Core/UISystem.h`, `src/Core/UISystem.cpp`
  - Thread the new counters through stats, UI, accumulation, and experiment row logging.

- Modify `tests/PathTracerAnalysisTests.cpp`
  - Extend counter-layout tests.
  - Add shader contract tests for dense bailout and projected radius.
  - Update grid-dimension expectations.

**Unit-test command convention:** Every step below that says to run unit tests must rebuild the test target first:

```powershell
cmd /c "call \"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat\" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests"
.\cmake-build-debug\tests\Debug\LaphriaEngineUnitTests.exe
```

---

### Task 1: Add Density Constants And Counter Contracts

**Files:**
- Modify: `src/shaders/SurfelCommon.slang`
- Modify: `src/Core/EngineAuxiliary.h`
- Modify: `src/Core/EngineCore.h`
- Modify: `src/Core/UISystem.h`
- Test: `tests/PathTracerAnalysisTests.cpp`

- [ ] **Step 1: Write the failing test**

Add the following expectations to the existing counter-layout/summary tests in `tests/PathTracerAnalysisTests.cpp` near the current `surfelGiCellMaxPopulation` assertions:

```cpp
if (!containsText(surfelCommon, "SURFEL_GI_DENSE_CELL_LIMIT = 64u") ||
    !containsText(surfelCommon, "surfelGiEvalDenseCellOffset = 472u") ||
    !containsText(surfelCommon, "surfelGiEvalDenseCellSkippedOffset = 476u"))
{
    std::cerr << "surfel GI dense-cell constants and counters are required\n";
    return false;
}
```

Extend any counter-name arrays that currently end at `surfelGiCellMaxPopulation` with:

```cpp
"surfelGiEvalDenseCell",
"surfelGiEvalDenseCellSkipped"
```

Extend the offset mapping table with:

```cpp
{"surfelGiEvalDenseCell",
 offsetof(Laphria::PathTracerAnalysisCounters, surfelGiEvalDenseCell), 472u},
{"surfelGiEvalDenseCellSkipped",
 offsetof(Laphria::PathTracerAnalysisCounters, surfelGiEvalDenseCellSkipped), 476u}
```

- [ ] **Step 2: Run the unit tests and verify failure**

Run:

```powershell
.\cmake-build-debug\tests\Debug\LaphriaEngineUnitTests.exe
```

Expected: FAIL with a message mentioning missing dense-cell constants/counters.

- [ ] **Step 3: Add shader constants and offsets**

In `src/shaders/SurfelCommon.slang`, update the constant block to:

```cpp
static const uint SURFEL_GI_MAX_SURFELS = 32768u;
static const uint SURFEL_GI_GRID_DIM = 64u;
static const uint SURFEL_GI_CELL_COUNT = SURFEL_GI_GRID_DIM * SURFEL_GI_GRID_DIM * SURFEL_GI_GRID_DIM;
static const uint SURFEL_GI_MAX_CELL_MEMBERSHIPS_PER_SURFEL = 27u;
static const uint SURFEL_GI_CELL_TO_SURFEL_CAPACITY =
    SURFEL_GI_MAX_SURFELS * SURFEL_GI_MAX_CELL_MEMBERSHIPS_PER_SURFEL;
static const uint SURFEL_GI_MAX_EVAL_CANDIDATES = 16u;
static const uint SURFEL_GI_DENSE_CELL_LIMIT = 64u;
static const float SURFEL_GI_CELL_SIZE = 0.75f;
static const float SURFEL_GI_MIN_RADIUS = 0.08f;
static const float SURFEL_GI_MAX_RADIUS = 0.75f;
```

Append the new offsets after `surfelGiCellMaxPopulationOffset`:

```cpp
static const uint surfelGiCellMaxPopulationOffset = 468u;
static const uint surfelGiEvalDenseCellOffset = 472u;
static const uint surfelGiEvalDenseCellSkippedOffset = 476u;
```

- [ ] **Step 4: Add C++ counter fields**

In `src/Core/EngineAuxiliary.h`, append to `PathTracerAnalysisCounters` after `surfelGiCellMaxPopulation`:

```cpp
uint32_t surfelGiEvalDenseCell = 0;
uint32_t surfelGiEvalDenseCellSkipped = 0;
```

In `src/Core/EngineCore.h`, append to the path-tracer experiment accumulator after `surfelGiCellMaxPopulation`:

```cpp
double surfelGiEvalDenseCell = 0.0;
double surfelGiEvalDenseCellSkipped = 0.0;
```

In `src/Core/UISystem.h`, append to `PathTracerPerfStats` after `surfelGiCellMaxPopulation`:

```cpp
uint32_t surfelGiEvalDenseCell = 0;
uint32_t surfelGiEvalDenseCellSkipped = 0;
```

- [ ] **Step 5: Run the unit tests and verify the remaining failures**

Run:

```powershell
.\cmake-build-debug\tests\Debug\LaphriaEngineUnitTests.exe
```

Expected: FAIL only in logging/UI/accumulation contract checks that still need the new fields threaded through.

---

### Task 2: Thread Dense-Cell Counters Through UI And Sweep Logging

**Files:**
- Modify: `src/Core/EngineCore.cpp`
- Modify: `src/Core/UISystem.cpp`
- Test: `tests/PathTracerAnalysisTests.cpp`

- [ ] **Step 1: Extend the failing test expectations**

In `tests/PathTracerAnalysisTests.cpp`, extend UI/logging required strings near existing `surfelGiCellMaxPopulation` checks:

```cpp
"surfelGiEvalDenseCell",
"surfelGiEvalDenseCellSkipped"
```

Add row-summary format expectations:

```cpp
"surfelGiEvalDenseCell=%.1f",
"surfelGiEvalDenseCellSkipped=%.1f"
```

- [ ] **Step 2: Run the unit tests and verify failure**

Run:

```powershell
.\cmake-build-debug\tests\Debug\LaphriaEngineUnitTests.exe
```

Expected: FAIL because the fields are not copied, accumulated, or printed yet.

- [ ] **Step 3: Copy counters into UI stats**

In `src/Core/EngineCore.cpp`, near the existing `ui.pathTracerPerfStats.surfelGiCellMaxPopulation` assignment, add:

```cpp
ui.pathTracerPerfStats.surfelGiEvalDenseCell =
    counters->surfelGiEvalDenseCell;
ui.pathTracerPerfStats.surfelGiEvalDenseCellSkipped =
    counters->surfelGiEvalDenseCellSkipped;
```

- [ ] **Step 4: Accumulate counters for experiment rows**

In `src/Core/EngineCore.cpp`, near the existing `ptExperimentAccum.surfelGiCellMaxPopulation +=` block, add:

```cpp
ptExperimentAccum.surfelGiEvalDenseCell +=
    static_cast<double>(stats.surfelGiEvalDenseCell);
ptExperimentAccum.surfelGiEvalDenseCellSkipped +=
    static_cast<double>(stats.surfelGiEvalDenseCellSkipped);
```

- [ ] **Step 5: Print counters in the Sponza row summary**

In `src/Core/EngineCore.cpp`, extend the row-summary format string after `surfelGiCellMaxPopulation=%.1f`:

```cpp
"surfelGiCellMaxPopulation=%.1f, "
"surfelGiEvalDenseCell=%.1f, surfelGiEvalDenseCellSkipped=%.1f, "
```

Add matching arguments immediately after `accum.surfelGiCellMaxPopulation * invSamples`:

```cpp
accum.surfelGiEvalDenseCell * invSamples,
accum.surfelGiEvalDenseCellSkipped * invSamples,
```

- [ ] **Step 6: Show counters in the UI stats panel**

In `src/Core/UISystem.cpp`, near the existing max population text, add:

```cpp
ImGui::Text("Surfel GI dense cells: %u", pathTracerPerfStats.surfelGiEvalDenseCell);
ImGui::Text("Surfel GI dense cells skipped: %u", pathTracerPerfStats.surfelGiEvalDenseCellSkipped);
```

- [ ] **Step 7: Run the unit tests and verify pass for counter plumbing**

Run:

```powershell
.\cmake-build-debug\tests\Debug\LaphriaEngineUnitTests.exe
```

Expected: PASS for counter-layout/logging/UI checks, with shader behavior tests still failing until Task 3.

---

### Task 3: Add SurfelGI-Style Dense-Cell Bailout In Evaluation

**Files:**
- Modify: `src/shaders/SurfelEvaluate.slang`
- Test: `tests/PathTracerAnalysisTests.cpp`

- [ ] **Step 1: Write the failing shader contract test**

In `tests/PathTracerAnalysisTests.cpp`, extend `requireSurfelEvaluatePassContracts` with checks against `evaluateMain`:

```cpp
if (!containsText(evaluateMain, "cell.count > SURFEL_GI_DENSE_CELL_LIMIT") ||
    !containsText(evaluateMain, "surfelGiEvalDenseCellOffset") ||
    !containsText(evaluateMain, "surfelGiEvalDenseCellSkippedOffset") ||
    !containsText(evaluateMain, "return;") ||
    !containsText(evaluateMain, "float4(0.0f, 0.5f, 1.0f, 1.0f)"))
{
    std::cerr << "surfel GI evaluation must bail out of dense cells with diagnostics\n";
    return false;
}
```

- [ ] **Step 2: Run the unit tests and verify failure**

Run:

```powershell
.\cmake-build-debug\tests\Debug\LaphriaEngineUnitTests.exe
```

Expected: FAIL with the dense-cell evaluation diagnostic message.

- [ ] **Step 3: Implement the dense-cell bailout**

In `src/shaders/SurfelEvaluate.slang`, immediately after the existing empty-cell check:

```cpp
if (cell.count == 0u || cell.offset == 0xffffffffu)
{
    ptAnalysisCounters.InterlockedAdd(surfelGiEvalCellEmptyOffset, 1u);
    surfelGiDebug[pixel] = float4(0.0f, 0.0f, 0.0f, 1.0f);
    return;
}
```

add:

```cpp
if (cell.count > SURFEL_GI_DENSE_CELL_LIMIT)
{
    ptAnalysisCounters.InterlockedAdd(surfelGiEvalDenseCellOffset, 1u);
    ptAnalysisCounters.InterlockedAdd(surfelGiEvalDenseCellSkippedOffset, 1u);
    surfelGiDebug[pixel] = float4(0.0f, 0.5f, 1.0f, 1.0f);
    return;
}
```

- [ ] **Step 4: Run the unit tests and verify pass**

Run:

```powershell
.\cmake-build-debug\tests\Debug\LaphriaEngineUnitTests.exe
```

Expected: PASS for the dense-cell bailout contract.

---

### Task 4: Replace Fixed Surfel Radius With Projected Radius

**Files:**
- Modify: `src/shaders/SurfelCommon.slang`
- Modify: `src/shaders/SurfelGenerate.slang`
- Test: `tests/PathTracerAnalysisTests.cpp`

- [ ] **Step 1: Write the failing shader contract test**

In `tests/PathTracerAnalysisTests.cpp`, extend the generate/common surfel contract checks:

```cpp
if (!containsText(surfelCommon, "float calcSurfelGiRadius(") ||
    !containsText(surfelCommon, "SURFEL_GI_CELL_SIZE * 0.5f") ||
    !containsText(surfelGenerate, "calcSurfelGiRadius(depth, uint2(width, height))") ||
    containsText(surfelGenerate, "record.positionRadius = float4(worldPos, SURFEL_GI_MIN_RADIUS)"))
{
    std::cerr << "surfel GI generation must use projected radius instead of fixed min radius\n";
    return false;
}
```

- [ ] **Step 2: Run the unit tests and verify failure**

Run:

```powershell
.\cmake-build-debug\tests\Debug\LaphriaEngineUnitTests.exe
```

Expected: FAIL with the projected-radius diagnostic.

- [ ] **Step 3: Add the projected radius helper**

In `src/shaders/SurfelCommon.slang`, after `calcSurfelGiCell`, add:

```cpp
float calcSurfelGiRadius(float hitT, uint2 extent)
{
    const float shortestAxis = max(float(min(extent.x, extent.y)), 1.0f);
    const float projectedRadius = (hitT / shortestAxis) * 4.0f;
    return clamp(projectedRadius,
                 SURFEL_GI_MIN_RADIUS,
                 min(SURFEL_GI_MAX_RADIUS, SURFEL_GI_CELL_SIZE * 0.5f));
}
```

Note: scale before clamping so `SURFEL_GI_MIN_RADIUS` remains the actual minimum stored radius. Clamping before the `4.0f` projection scale would turn the effective minimum into `0.32f`, which is not what the constant name or density experiment intends.

- [ ] **Step 4: Use projected radius during generation**

In `src/shaders/SurfelGenerate.slang`, replace:

```cpp
record.positionRadius = float4(worldPos, SURFEL_GI_MIN_RADIUS);
```

with:

```cpp
record.positionRadius = float4(worldPos, calcSurfelGiRadius(depth, uint2(width, height)));
```

- [ ] **Step 5: Run the unit tests and verify pass**

Run:

```powershell
.\cmake-build-debug\tests\Debug\LaphriaEngineUnitTests.exe
```

Expected: PASS for projected-radius contracts.

---

### Task 5: Align Host Cell Count With Finer Grid

**Files:**
- Modify: `src/Core/FrameContext.h`
- Test: `tests/PathTracerAnalysisTests.cpp`

- [ ] **Step 1: Write the failing alignment test**

In `tests/PathTracerAnalysisTests.cpp`, extend `requireSurfelGiSlotCapacityAlignment`:

```cpp
const auto shaderGridDim =
    extractUnsignedAssignment(surfelCommon, "SURFEL_GI_GRID_DIM");
const auto frameGridDim =
    extractUnsignedAssignment(frameContextHeader, "kSurfelGiGridDim");
if (!shaderGridDim || !frameGridDim || *shaderGridDim != *frameGridDim)
{
    std::cerr << "surfel GI shader and host grid dimensions must match\n";
    return false;
}
if (*shaderGridDim != 64u || !containsText(surfelCommon, "SURFEL_GI_CELL_SIZE = 0.75f"))
{
    std::cerr << "surfel GI density experiment expects 64^3 cells at 0.75m to preserve coverage\n";
    return false;
}
```

- [ ] **Step 2: Run the unit tests and verify failure**

Run:

```powershell
.\cmake-build-debug\tests\Debug\LaphriaEngineUnitTests.exe
```

Expected: FAIL until the host grid dimension is updated.

- [ ] **Step 3: Update the host grid dimension**

In `src/Core/FrameContext.h`, change:

```cpp
static constexpr uint32_t       kSurfelGiGridDim = 32;
```

to:

```cpp
static constexpr uint32_t       kSurfelGiGridDim = 64;
```

- [ ] **Step 4: Run the unit tests and verify pass**

Run:

```powershell
.\cmake-build-debug\tests\Debug\LaphriaEngineUnitTests.exe
```

Expected: PASS for host/shader capacity alignment.

---

### Task 6: Build And Run A Focused Sponza Diagnostic Sweep

**Files:**
- No source edits unless verification exposes a failure.

- [ ] **Step 1: Build the editor**

Run:

```powershell
cmd /c "call \"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat\" && cmake --build cmake-build-debug --config Debug --target LaphriaEditor"
```

Expected: build exits with code `0`.

- [ ] **Step 2: Run the automatic Sponza PT/GI audit sweep**

Run the same editor/sweep command used for the previous Sponza PT/GI audit. Keep the existing automatic rows, especially:

```text
Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Sun Receiver Surfel Cache Debug
```

Expected: sweep completes and prints `PT Experiment Sweep: Sponza PT/GI audit sweep complete`.

- [ ] **Step 3: Validate dense-cell behavior from the sweep output**

For each Surfel Cache Debug row, check these acceptance targets:

```text
surfelGiCellAllocationOverflow = 0.0
surfelGiCellOverflow = 0.0
surfelGiEvalCandidates significantly below the previous 6.8M-16.6M range
surfelGiEvalDenseCell > 0.0 if max cell population is still above 64
surfelGiEvalDenseCellSkipped == surfelGiEvalDenseCell
```

If `surfelGiCellMaxPopulation` remains in the thousands while `surfelGiEvalCandidates` is low, this task is still a success: the dense-cell failure is now isolated and cheap. If `surfelGiCellNonEmpty` rises materially above the previous `16-36` range, the grid/radius tuning is also improving distribution.

Also check that `surfelGiCellInserted` does not collapse relative to the previous debug rows. A large collapse would suggest the grid became too small or the radius became too strict rather than genuinely reducing dense-cell pressure.

- [ ] **Step 4: Commit**

Run:

```powershell
git add src/shaders/SurfelCommon.slang src/shaders/SurfelGenerate.slang src/shaders/SurfelEvaluate.slang src/Core/FrameContext.h src/Core/EngineAuxiliary.h src/Core/EngineCore.h src/Core/EngineCore.cpp src/Core/UISystem.h src/Core/UISystem.cpp tests/PathTracerAnalysisTests.cpp docs/superpowers/plans/2026-05-18-surfel-gi-density-controls.md
git commit -m "feat: add surfel gi density controls"
```

Expected: commit succeeds.

---

## Self-Review

**Spec coverage:** The plan covers the original-repo behaviors we are ready to port now: dense-cell rejection, bounded lookup preservation, finer grid/radius relationship, diagnostics, UI/logging, and sweep validation. It intentionally does not implement persistent lifecycle, sleeping surfels, depth maps, or coverage-based removal yet. The grid resolution changes from `32 x 1.5m` to `64 x 0.75m`, preserving the previous approximate +/-24m camera-relative extent while splitting each old cell into eight smaller cells. The projected-radius helper scales the footprint before clamping so the configured `0.08f` minimum remains literal rather than becoming an accidental `0.32f` post-scale floor.

**Placeholder scan:** No task uses unspecified TODOs. Each code-changing step includes the concrete symbols and snippets to add.

**Type consistency:** New counters are consistently named `surfelGiEvalDenseCell` and `surfelGiEvalDenseCellSkipped` across shader offsets, C++ structs, UI stats, accumulation, and row logging.

**Known follow-up:** If the sweep shows dense skips are common but non-empty cell count remains low, the next plan should add generation-side density feedback: a per-cell spawn budget or coverage removal pass, closer to `SurfelGenerationPass.cs.slang` in the SurfelGI reference.
