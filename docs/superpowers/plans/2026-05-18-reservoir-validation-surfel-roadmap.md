# Reservoir Validation And Surfel GI Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the ReSTIR GI reservoir estimator contract before making surfels a lighting or proposal dependency, then redesign surfels around the verified estimator.

**Architecture:** Split the work into two explicit gates. Gate 1 adds a reservoir audit harness that compares the current estimator against an RTXDI-style reference contract and simple invariance tests. Gate 2 only starts after the estimator is classified as correct, deliberately damped, or needing repair; surfels then become either a proposal source for the verified reservoir or, if the reservoir is rejected, a separate cache estimator with its own validation.

**Tech Stack:** C++17 engine code, Slang raygen/compute shaders, Vulkan storage buffers, existing path tracer analysis counters, source-inspection unit tests in `tests/PathTracerAnalysisTests.cpp`, manual Sponza PT/GI sweeps, NVIDIA RTXDI ReSTIR GI reference docs/code.

---

## File Structure

- Modify `C:\Dev\Dizertatie\LaphriaEngine\docs\architecture\restir-gi-sponza-handoff.md`
  - Add a short estimator-audit section so future sweep reads do not treat current reservoir output as proven ground truth.
- Create `C:\Dev\Dizertatie\LaphriaEngine\docs\architecture\reservoir-gi-estimator-contract.md`
  - Document the RTXDI baseline contract, current implementation differences, and acceptance gates.
- Modify `C:\Dev\Dizertatie\LaphriaEngine\tests\PathTracerAnalysisTests.cpp`
  - Add source-inspection gates for audit modes, suspicious heuristic toggles, and row-summary diagnostics.
- Modify `C:\Dev\Dizertatie\LaphriaEngine\src\Core\UISystem.h`
  - Add reservoir audit mode/settings and perf-stat fields.
- Modify `C:\Dev\Dizertatie\LaphriaEngine\src\Core\UISystem.cpp`
  - Expose audit controls and diagnostics in the path tracer debug UI.
- Modify `C:\Dev\Dizertatie\LaphriaEngine\src\Core\EngineAuxiliary.h`
  - Append audit counters without moving existing counter offsets.
- Modify `C:\Dev\Dizertatie\LaphriaEngine\src\Core\EngineCore.h`
  - Add sweep accumulators for audit metrics.
- Modify `C:\Dev\Dizertatie\LaphriaEngine\src\Core\EngineCore.cpp`
  - Thread audit counters into UI, experiment rows, and focused validation sweeps.
- Modify `C:\Dev\Dizertatie\LaphriaEngine\src\shaders\Raygen.slang`
  - Add diagnostic-only estimator modes and counters. Do not change the production estimator until audit data proves the root cause.
- Later modify surfel files only after Gate 1:
  - `C:\Dev\Dizertatie\LaphriaEngine\src\shaders\SurfelCommon.slang`
  - `C:\Dev\Dizertatie\LaphriaEngine\src\shaders\SurfelGenerate.slang`
  - `C:\Dev\Dizertatie\LaphriaEngine\src\shaders\SurfelIntegrate.slang`
  - `C:\Dev\Dizertatie\LaphriaEngine\src\shaders\SurfelEvaluate.slang`
  - `C:\Dev\Dizertatie\LaphriaEngine\src\shaders\SurfelCountCells.slang`
  - `C:\Dev\Dizertatie\LaphriaEngine\src\shaders\SurfelAllocateCells.slang`
  - `C:\Dev\Dizertatie\LaphriaEngine\src\shaders\SurfelBuildCells.slang`

---

## Task 1: Write The Reservoir Estimator Contract

**Files:**
- Create: `C:\Dev\Dizertatie\LaphriaEngine\docs\architecture\reservoir-gi-estimator-contract.md`
- Modify: `C:\Dev\Dizertatie\LaphriaEngine\docs\architecture\restir-gi-sponza-handoff.md`

- [ ] **Step 1: Create the contract document**

Add `docs/architecture/reservoir-gi-estimator-contract.md` with this structure:

```markdown
# Reservoir GI Estimator Contract

Date: 2026-05-18

## Reference Model

The audit baseline is NVIDIA RTXDI ReSTIR GI:

- `RTXDI_MakeGIReservoir(samplePos, sampleNormal, sampleRadiance, samplePdf)` stores radiance and starts `weightSum` as inverse sample PDF.
- `RTXDI_CombineGIReservoirs` streams normalized reservoirs using `targetPdf * newReservoir.weightSum * newReservoir.M`.
- `RTXDI_FinalizeGIResampling` converts the streamed RIS weight into final reservoir weight.
- Final shading uses the selected secondary sample as an indirect sample:
  `primaryBRDF * reservoir.radiance * reservoir.weightSum`.

Reference sources:

- https://github.com/NVIDIA-RTX/RTXDI/blob/main/Doc/RestirGI.md
- https://raw.githubusercontent.com/NVIDIA-RTX/RTXDI-Library/main/Include/Rtxdi/GI/Reservoir.hlsli

## Current Implementation Questions

- `Raygen.slang` stores pre-shaded `ReservoirGiRecord::contribution`, not only secondary radiance plus final reservoir weight.
- Non-RIS local candidates are divided by `candidateCount`.
- All returned reservoir contribution is additionally multiplied by `candidateCount / (candidateCount + 1)`.
- Temporal/spatial reuse reconnects samples, but stored `targetWeight`, `selectedWeight`, `weightSum`, and `confidenceM` need a single written semantic contract.

## Audit Acceptance Gates

- Single-frame one-candidate audit output must match the equivalent plain first-hit diffuse probe within 2 percent average luminance on a static debug scene.
- Increasing local candidate count from 1 to 2 to 4 must not systematically darken or brighten the mean by more than 5 percent in audit rows.
- Toggling candidate RIS must preserve mean within 5 percent while changing variance/selection diagnostics.
- Temporal reuse on a static camera must reduce noise or increase accepted reuse without shifting mean more than 5 percent after warmup.
- Spatial reuse on a static camera must not shift mean more than 5 percent relative to temporal-only after warmup.

## Decision Gate

- If the estimator passes, surfels may be designed as a proposal/source for ReSTIR GI.
- If the estimator fails only because of explicit damping heuristics, isolate those heuristics behind audit toggles and decide whether to keep them as artistic/stability bias.
- If the estimator fails core normalization, fix it before using surfels for lighting.
```

- [ ] **Step 2: Update the Sponza handoff**

Append this section to `docs/architecture/restir-gi-sponza-handoff.md`:

```markdown
## Reservoir Estimator Audit Gate

The current ReSTIR GI reservoir should be treated as plausible but not yet proven. Before surfels become a lighting contributor or a trusted proposal source, the estimator must pass the audit gates in `docs/architecture/reservoir-gi-estimator-contract.md`.

In particular, the current `candidateCount / (candidateCount + 1)` reservoir output scale and the local non-RIS `candidateCount` division are audit targets, not assumed-correct normalization.
```

- [ ] **Step 3: Commit documentation**

Run:

```powershell
git add docs/architecture/reservoir-gi-estimator-contract.md docs/architecture/restir-gi-sponza-handoff.md
git commit -m "docs: define reservoir gi estimator audit contract"
```

Expected: commit succeeds with only documentation changes.

---

## Task 2: Add Source Contracts For Audit Controls

**Files:**
- Modify: `C:\Dev\Dizertatie\LaphriaEngine\tests\PathTracerAnalysisTests.cpp`

- [ ] **Step 1: Write failing source-inspection checks**

In `testPathTracerReservoirGiMeasurementContract()`, after the existing reservoir GI source-contract checks, add:

```cpp
const std::array<const char *, 10> estimatorAuditSymbols{
    "enum class PathTracerReservoirGiEstimatorAuditMode",
    "reservoirGiEstimatorAuditMode",
    "ReservoirEstimatorAuditOff",
    "ReservoirEstimatorAuditCurrent",
    "ReservoirEstimatorAuditSingleCandidateReference",
    "ReservoirEstimatorAuditNoProbeScale",
    "reservoirGiAuditCurrentLuma",
    "reservoirGiAuditReferenceLuma",
    "reservoirGiAuditRelativeErrorPct",
    "reservoirGiAuditProbeScale",
};
for (const char *symbol : estimatorAuditSymbols)
{
    if (!containsText(uiHeader, symbol) &&
        !containsText(uiSource, symbol) &&
        !containsText(engineAuxiliaryHeader, symbol) &&
        !containsText(engineCore, symbol) &&
        !containsText(raygen, symbol))
    {
        std::cerr << "reservoir estimator audit missing symbol: " << symbol << "\n";
        return false;
    }
}

if (!containsText(raygen, "float reservoirProbeScale =") ||
    !containsText(raygen, "reservoirGiAuditProbeScaleOffset"))
{
    std::cerr << "reservoir estimator audit must instrument reservoirProbeScale\n";
    return false;
}
```

- [ ] **Step 2: Run the unit test and verify failure**

Run:

```powershell
cmd /c "call \"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat\" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests"
.\cmake-build-debug\tests\Debug\LaphriaEngineUnitTests.exe
```

Expected: FAIL with `reservoir estimator audit missing symbol`.

- [ ] **Step 3: Commit the failing contract**

Run:

```powershell
git add tests/PathTracerAnalysisTests.cpp
git commit -m "test: require reservoir estimator audit controls"
```

Expected: commit succeeds with a deliberately failing source-inspection test.

---

## Task 3: Add Audit Mode State, Counters, And UI

**Files:**
- Modify: `C:\Dev\Dizertatie\LaphriaEngine\src\Core\UISystem.h`
- Modify: `C:\Dev\Dizertatie\LaphriaEngine\src\Core\UISystem.cpp`
- Modify: `C:\Dev\Dizertatie\LaphriaEngine\src\Core\EngineAuxiliary.h`
- Modify: `C:\Dev\Dizertatie\LaphriaEngine\src\Core\EngineCore.h`
- Modify: `C:\Dev\Dizertatie\LaphriaEngine\src\Core\EngineCore.cpp`
- Modify: `C:\Dev\Dizertatie\LaphriaEngine\src\shaders\Raygen.slang`

- [ ] **Step 1: Add UI enum and setting**

In `UISystem.h`, near `PathTracerReservoirGiMode`, add:

```cpp
enum class PathTracerReservoirGiEstimatorAuditMode
{
    Off = 0,
    Current = 1,
    SingleCandidateReference = 2,
    NoProbeScale = 3
};
```

In `PathTracerSettings`, add:

```cpp
PathTracerReservoirGiEstimatorAuditMode reservoirGiEstimatorAuditMode =
    PathTracerReservoirGiEstimatorAuditMode::Off;
```

- [ ] **Step 2: Add CPU perf fields**

In `PathTracerPerfStats`, append:

```cpp
float reservoirGiAuditCurrentLuma = 0.0f;
float reservoirGiAuditReferenceLuma = 0.0f;
float reservoirGiAuditRelativeErrorPct = 0.0f;
float reservoirGiAuditProbeScale = 0.0f;
```

In `EngineCore.h`, append matching doubles to the path tracer experiment accumulator:

```cpp
double reservoirGiAuditCurrentLuma = 0.0;
double reservoirGiAuditReferenceLuma = 0.0;
double reservoirGiAuditRelativeErrorPct = 0.0;
double reservoirGiAuditProbeScale = 0.0;
```

- [ ] **Step 3: Append analysis counters**

In `EngineAuxiliary.h`, append counters after the current surfel GI counters:

```cpp
uint32_t reservoirGiAuditCurrentLumaScaledSum = 0;
uint32_t reservoirGiAuditReferenceLumaScaledSum = 0;
uint32_t reservoirGiAuditRelativeErrorScaledSum = 0;
uint32_t reservoirGiAuditProbeScaleScaledSum = 0;
uint32_t reservoirGiAuditSampleCount = 0;
```

In `Raygen.slang`, add matching append-only byte offsets after the last current counter offset. Use the next free 4-byte-aligned offsets and keep all existing offsets unchanged:

```hlsl
static const uint reservoirGiAuditCurrentLumaScaledSumOffset = 480u;
static const uint reservoirGiAuditReferenceLumaScaledSumOffset = 484u;
static const uint reservoirGiAuditRelativeErrorScaledSumOffset = 488u;
static const uint reservoirGiAuditProbeScaleScaledSumOffset = 492u;
static const uint reservoirGiAuditSampleCountOffset = 496u;
```

- [ ] **Step 4: Add UI controls**

In `UISystem.cpp`, inside the Reservoir GI controls block, add:

```cpp
const char *reservoirAuditModes[] = {
    "Off",
    "Current",
    "Single Candidate Reference",
    "No Probe Scale"
};
int reservoirAuditMode = static_cast<int>(pathTracerSettings.reservoirGiEstimatorAuditMode);
if (ImGui::Combo("Reservoir Estimator Audit",
                 &reservoirAuditMode,
                 reservoirAuditModes,
                 IM_ARRAYSIZE(reservoirAuditModes)))
{
    pathTracerSettings.reservoirGiEstimatorAuditMode =
        static_cast<PathTracerReservoirGiEstimatorAuditMode>(reservoirAuditMode);
}
```

In the Reservoir GI diagnostics block, add:

```cpp
ImGui::Text("Reservoir GI Audit Current Luma: %.5f", pathTracerPerfStats.reservoirGiAuditCurrentLuma);
ImGui::Text("Reservoir GI Audit Reference Luma: %.5f", pathTracerPerfStats.reservoirGiAuditReferenceLuma);
ImGui::Text("Reservoir GI Audit Relative Error: %.2f%%", pathTracerPerfStats.reservoirGiAuditRelativeErrorPct);
ImGui::Text("Reservoir GI Audit Probe Scale: %.5f", pathTracerPerfStats.reservoirGiAuditProbeScale);
```

- [ ] **Step 5: Thread counters through collection and row logging**

In `EngineCore.cpp`, where path tracer counters are copied into `ui.pathTracerPerfStats`, compute:

```cpp
const float auditInvSamples =
    counters->reservoirGiAuditSampleCount > 0
        ? 1.0f / static_cast<float>(counters->reservoirGiAuditSampleCount)
        : 0.0f;
ui.pathTracerPerfStats.reservoirGiAuditCurrentLuma =
    static_cast<float>(counters->reservoirGiAuditCurrentLumaScaledSum) * auditInvSamples / 64.0f;
ui.pathTracerPerfStats.reservoirGiAuditReferenceLuma =
    static_cast<float>(counters->reservoirGiAuditReferenceLumaScaledSum) * auditInvSamples / 64.0f;
ui.pathTracerPerfStats.reservoirGiAuditRelativeErrorPct =
    static_cast<float>(counters->reservoirGiAuditRelativeErrorScaledSum) * auditInvSamples / 64.0f;
ui.pathTracerPerfStats.reservoirGiAuditProbeScale =
    static_cast<float>(counters->reservoirGiAuditProbeScaleScaledSum) * auditInvSamples / 64.0f;
```

Add the same four fields to experiment accumulation and to the row summary format string:

```cpp
"reservoirGiAuditCurrentLuma=%.5f, "
"reservoirGiAuditReferenceLuma=%.5f, "
"reservoirGiAuditRelativeErrorPct=%.2f, "
"reservoirGiAuditProbeScale=%.5f, "
```

- [ ] **Step 6: Run tests**

Run:

```powershell
cmd /c "call \"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat\" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests"
.\cmake-build-debug\tests\Debug\LaphriaEngineUnitTests.exe
```

Expected: PASS for the audit control source contract, or fail only on shader behavior that Task 4 implements.

- [ ] **Step 7: Commit**

Run:

```powershell
git add src/Core/UISystem.h src/Core/UISystem.cpp src/Core/EngineAuxiliary.h src/Core/EngineCore.h src/Core/EngineCore.cpp src/shaders/Raygen.slang tests/PathTracerAnalysisTests.cpp
git commit -m "feat: add reservoir estimator audit controls"
```

---

## Task 4: Instrument Current Estimator Without Changing It

**Files:**
- Modify: `C:\Dev\Dizertatie\LaphriaEngine\src\shaders\Raygen.slang`
- Modify: `C:\Dev\Dizertatie\LaphriaEngine\tests\PathTracerAnalysisTests.cpp`

- [ ] **Step 1: Add a failing shader contract**

In `PathTracerAnalysisTests.cpp`, add checks that `sampleFirstHitReservoirGiSingleFrame` records audit metrics after computing `reservoirProbeScale`:

```cpp
const std::string reservoirMain =
    stripComments(extractFunctionBody(raygen, "FirstHitDiffuseBounceResult sampleFirstHitReservoirGiSingleFrame("));
if (!containsText(reservoirMain, "recordReservoirGiEstimatorAudit(") ||
    !containsText(reservoirMain, "rawReservoirLuma") ||
    !containsText(reservoirMain, "reservoirProbeScale"))
{
    std::cerr << "reservoir estimator audit must record current raw/scaled luma and probe scale\n";
    return false;
}
```

- [ ] **Step 2: Run the unit test and verify failure**

Run:

```powershell
.\cmake-build-debug\tests\Debug\LaphriaEngineUnitTests.exe
```

Expected: FAIL with the new audit recording message.

- [ ] **Step 3: Add shader helper**

In `Raygen.slang`, add:

```hlsl
void recordReservoirGiEstimatorAudit(float currentLuma,
                                     float referenceLuma,
                                     float reservoirProbeScale)
{
    float safeReference = max(referenceLuma, 0.0001);
    float relativeErrorPct = abs(currentLuma - referenceLuma) / safeReference * 100.0;
    ptAnalysisCounters.InterlockedAdd(reservoirGiAuditCurrentLumaScaledSumOffset,
                                      uint(clamp(currentLuma, 0.0, 8.0) * 64.0 + 0.5));
    ptAnalysisCounters.InterlockedAdd(reservoirGiAuditReferenceLumaScaledSumOffset,
                                      uint(clamp(referenceLuma, 0.0, 8.0) * 64.0 + 0.5));
    ptAnalysisCounters.InterlockedAdd(reservoirGiAuditRelativeErrorScaledSumOffset,
                                      uint(clamp(relativeErrorPct, 0.0, 1024.0) * 64.0 + 0.5));
    ptAnalysisCounters.InterlockedAdd(reservoirGiAuditProbeScaleScaledSumOffset,
                                      uint(clamp(reservoirProbeScale, 0.0, 8.0) * 64.0 + 0.5));
    ptAnalysisCounters.InterlockedAdd(reservoirGiAuditSampleCountOffset, 1u);
}
```

- [ ] **Step 4: Record current estimator metrics**

In `sampleFirstHitReservoirGiSingleFrame`, after:

```hlsl
float reservoirProbeScale = float(candidateCount) / float(candidateCount + 1);
```

add:

```hlsl
float currentEstimatorLuma = pathTracerLuminance(reservoirTotal * reservoirProbeScale);
float referenceEstimatorLuma = pathTracerLuminance(reservoirTotal);
if (acceptedReservoir) {
    recordReservoirGiEstimatorAudit(currentEstimatorLuma,
                                    referenceEstimatorLuma,
                                    reservoirProbeScale);
}
```

This task intentionally records a provisional reference of “same selected reservoir without final probe scale.” It does not claim RTXDI correctness yet; it isolates the effect of the known damping term.

- [ ] **Step 5: Run unit tests and editor build**

Run:

```powershell
cmd /c "call \"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat\" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests"
.\cmake-build-debug\tests\Debug\LaphriaEngineUnitTests.exe
cmd /c "call \"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat\" && cmake --build cmake-build-debug --config Debug --target LaphriaEditor"
```

Expected: unit tests pass and `LaphriaEditor` builds.

- [ ] **Step 6: Commit**

Run:

```powershell
git add src/shaders/Raygen.slang tests/PathTracerAnalysisTests.cpp
git commit -m "feat: instrument reservoir estimator damping"
```

---

## Task 5: Add Focused Reservoir Validation Sweep Rows

**Files:**
- Modify: `C:\Dev\Dizertatie\LaphriaEngine\src\Core\EngineCore.cpp`
- Modify: `C:\Dev\Dizertatie\LaphriaEngine\tests\PathTracerAnalysisTests.cpp`

- [ ] **Step 1: Add source-contract expectations**

Add test checks requiring these row names in the focused Sponza sweep:

```cpp
const std::array<const char *, 6> reservoirAuditRows{
    "Reservoir Audit / Single Frame 1C Current",
    "Reservoir Audit / Single Frame 1C No Probe Scale",
    "Reservoir Audit / Single Frame 2C Current",
    "Reservoir Audit / Single Frame 2C RIS",
    "Reservoir Audit / Temporal Static",
    "Reservoir Audit / Temporal Spatial Static"
};
for (const char *row : reservoirAuditRows)
{
    if (!containsText(engineCore, row))
    {
        std::cerr << "missing reservoir audit sweep row: " << row << "\n";
        return false;
    }
}
```

- [ ] **Step 2: Run the unit test and verify failure**

Run:

```powershell
.\cmake-build-debug\tests\Debug\LaphriaEngineUnitTests.exe
```

Expected: FAIL with `missing reservoir audit sweep row`.

- [ ] **Step 3: Add audit rows after existing Sponza rows**

In `EngineCore.cpp`, where focused Sponza experiment rows are created, add rows with these settings:

```cpp
// Reservoir Audit / Single Frame 1C Current
reservoirGiMode = SingleFrame;
reservoirGiCandidateCount = 1;
reservoirGiUseCandidateRis = false;
reservoirGiEstimatorAuditMode = Current;

// Reservoir Audit / Single Frame 1C No Probe Scale
reservoirGiMode = SingleFrame;
reservoirGiCandidateCount = 1;
reservoirGiUseCandidateRis = false;
reservoirGiEstimatorAuditMode = NoProbeScale;

// Reservoir Audit / Single Frame 2C Current
reservoirGiMode = SingleFrame;
reservoirGiCandidateCount = 2;
reservoirGiUseCandidateRis = false;
reservoirGiEstimatorAuditMode = Current;

// Reservoir Audit / Single Frame 2C RIS
reservoirGiMode = SingleFrame;
reservoirGiCandidateCount = 2;
reservoirGiUseCandidateRis = true;
reservoirGiEstimatorAuditMode = Current;

// Reservoir Audit / Temporal Static
reservoirGiMode = Temporal;
reservoirGiCandidateCount = 1;
reservoirGiUseCandidateRis = false;
reservoirGiTemporalBudgetDivisor = 1;
reservoirGiEstimatorAuditMode = Current;

// Reservoir Audit / Temporal Spatial Static
reservoirGiMode = TemporalSpatial;
reservoirGiCandidateCount = 1;
reservoirGiUseCandidateRis = false;
reservoirGiTemporalBudgetDivisor = 1;
reservoirGiSpatialBudgetDivisor = 1;
reservoirGiSpatialNeighborCount = 2;
reservoirGiEstimatorAuditMode = Current;
```

Keep all other Sponza validation settings identical to the existing Sun Receiver validation preset.

- [ ] **Step 4: Run tests and editor build**

Run:

```powershell
cmd /c "call \"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat\" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests"
.\cmake-build-debug\tests\Debug\LaphriaEngineUnitTests.exe
cmd /c "call \"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat\" && cmake --build cmake-build-debug --config Debug --target LaphriaEditor"
```

Expected: all pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/Core/EngineCore.cpp tests/PathTracerAnalysisTests.cpp
git commit -m "test: add reservoir estimator audit sweep rows"
```

---

## Task 6: Manual Reservoir Audit Gate

**Files:**
- No code changes unless the sweep exposes a bug.
- Update: `C:\Dev\Dizertatie\LaphriaEngine\docs\architecture\reservoir-gi-estimator-contract.md`

- [ ] **Step 1: Run the focused Sponza audit sweep manually**

Run the Sponza PT/GI audit sweep in the editor with the new audit rows enabled.

Expected row-summary fields to inspect:

```text
reservoirGiAuditCurrentLuma
reservoirGiAuditReferenceLuma
reservoirGiAuditRelativeErrorPct
reservoirGiAuditProbeScale
reservoirGiAcceptedAvgLuma
reservoirGiSelectedWeightAvg
reservoirGiTargetWeightAvg
reservoirGiConfidenceMAvg
reservoirGiSelectedLocal
reservoirGiSelectedTemporal
reservoirGiSelectedSpatial
```

- [ ] **Step 2: Classify the estimator**

Append one of these classifications to `reservoir-gi-estimator-contract.md`:

```markdown
## Audit Result

Classification: Pass

The reservoir estimator preserves mean sufficiently for the tested modes. Surfel GI may proceed as a proposal/cache source for ReSTIR GI.
```

or:

```markdown
## Audit Result

Classification: Deliberately Damped

The reservoir estimator is internally plausible, but final output scale is affected by explicit damping terms. Surfels may proceed only if the damping is treated as a named quality/stability heuristic, not as an estimator normalization term.
```

or:

```markdown
## Audit Result

Classification: Fails Normalization

The reservoir estimator changes mean as candidate count, RIS, temporal reuse, or spatial reuse changes. Surfel work is blocked until estimator normalization is fixed.
```

- [ ] **Step 3: Commit the audit result**

Run:

```powershell
git add docs/architecture/reservoir-gi-estimator-contract.md
git commit -m "docs: record reservoir estimator audit result"
```

---

## Task 7: Decide And Plan Surfel Architecture From The Audit Result

**Files:**
- Create: `C:\Dev\Dizertatie\LaphriaEngine\docs\architecture\surfel-gi-architecture.md`
- Later plan files under `C:\Dev\Dizertatie\LaphriaEngine\docs\superpowers\plans\`

- [ ] **Step 1: If estimator passes or is deliberately damped, write the surfel-as-proposal architecture**

Create `docs/architecture/surfel-gi-architecture.md` with:

```markdown
# Surfel GI Architecture

## Role

Surfels are a bounded proposal and cache source for the verified ReSTIR GI estimator. They do not directly replace final GI until their own validation gates pass.

## Data Flow

1. Path tracer produces first-hit and accepted-reservoir data.
2. Surfel generation uses coverage feedback to place or update surfels.
3. Surfel integration maintains radiance, age, confidence, and last-seen state.
4. Compact cell indexing exposes bounded local surfel candidates.
5. Raygen evaluates selected surfels as candidate records with target weight and source PDF.
6. The reservoir estimator performs final selection and scaling.

## Required Gates

- Cell max population stays below the dense-cell bailout threshold in all Sponza audit views.
- Dense-cell skipped count trends toward zero after warmup.
- Surfel-selected candidates have nonzero accepted count in all Sponza views.
- Enabling surfel proposals changes mean less than 5 percent unless accepted as an intentional quality improvement backed by image comparison.
```

- [ ] **Step 2: If estimator fails normalization, write the estimator-fix blocker**

Instead create:

```markdown
# Surfel GI Architecture

## Status

Surfel GI lighting and proposal integration are blocked by reservoir estimator normalization failures.

## Allowed Work

- Diagnostic surfel visualization.
- Coverage and lifecycle experiments that do not feed final lighting.
- CPU/source tests for surfel data structure invariants.

## Blocked Work

- Surfels as ReSTIR GI candidates.
- Surfels as direct GI contribution.
- Sponza quality claims involving surfel lighting.
```

- [ ] **Step 3: Commit**

Run:

```powershell
git add docs/architecture/surfel-gi-architecture.md
git commit -m "docs: define surfel gi architecture gate"
```

---

## Task 8: Only Then Write The Next Implementation Plan

**Files:**
- Create one of:
  - `C:\Dev\Dizertatie\LaphriaEngine\docs\superpowers\plans\YYYY-MM-DD-reservoir-estimator-normalization.md`
  - `C:\Dev\Dizertatie\LaphriaEngine\docs\superpowers\plans\YYYY-MM-DD-surfel-coverage-lifecycle.md`

- [ ] **Step 1: Choose the next plan based on Gate 1**

If the audit classification is `Fails Normalization`, write a reservoir normalization plan first.

If the audit classification is `Pass` or `Deliberately Damped`, write the surfel coverage/lifecycle plan first.

- [ ] **Step 2: Keep the next plan narrow**

The next surfel plan must include only:

- coverage-aware generation,
- over-coverage removal or replacement,
- persistent age/confidence/last-seen lifecycle,
- bounded cell population gates,
- no direct lighting contribution until diagnostics pass.

- [ ] **Step 3: Get user approval before executing**

Do not implement the next plan until the user reviews the audit result and the selected follow-up plan.

---

## Verification Checklist

Run before claiming this roadmap is ready for execution:

```powershell
git diff --check
cmd /c "call \"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat\" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests"
.\cmake-build-debug\tests\Debug\LaphriaEngineUnitTests.exe
```

Expected:

- `git diff --check` reports no whitespace errors.
- Unit tests pass after each implementation task except the explicitly failing-test commit in Task 2.
