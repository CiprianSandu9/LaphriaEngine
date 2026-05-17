# Diagnostic Indexed Bright Surfel Lookup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current global random bright-surfel scan with a diagnostic-first receiver-local indexed lookup that tells us whether bright surfels can be found cheaply and geometrically plausibly.

**Architecture:** Reuse the existing `ptReservoirGiBrightSurfelCurrent` / `ptReservoirGiBrightSurfelHistory` buffers. Store bright surfels into spatially hashed cells, query nearby receiver cells with a bounded probe count, keep the existing target-aware evaluation and reservoir combine path, and add split diagnostics so the next sweep explains whether misses come from empty cells, distance, receiver hemisphere, surfel hemisphere, invalid vectors, or target rejection.

**Tech Stack:** C++17 engine counters/UI/logging, Slang raygen shader, existing CTest text-contract analysis tests, existing Vulkan buffer bindings.

---

## Current Evidence

The target-aware selector is no longer the main unknown. The latest sweep showed:

- Dark Courtyard bright surfel rows: `brightSurfelSelectorViable=0.0`, accepted/selected `0.0`.
- Mid-Depth bright surfel rows: `brightSurfelSelectorViable=0.0`, accepted/selected `0.0`, with millions of selector geometry rejects.
- Sunlit Wall bright surfel rows: viable surfels exist and select, but total frame time nearly doubles.

This points at producer/index quality. The global scan spends too much work evaluating random records that are not geometrically relevant to the receiver. The next pass should prove whether a cheap local index can produce viable candidates before we invest in a full SHaRC/RTXGI-like cache.

## File Structure

- Modify `src/shaders/Raygen.slang`
  - Add indexed lookup constants and helper functions.
  - Change bright surfel storage from global random placement to spatial hashed placement.
  - Replace `selectWeightedGlobalBrightReceiverSurfelRecord` with a receiver-local indexed selector.
  - Split selector geometry diagnostics into distance, receiver hemisphere, surfel hemisphere, and invalid vector counters.

- Modify `src/Core/EngineAuxiliary.h`
  - Add new `PathTracerAnalysisCounters` fields matching the shader counter byte offsets.

- Modify `src/Core/UISystem.h`
  - Add UI stats fields for the new diagnostics.

- Modify `src/Core/UISystem.cpp`
  - Display the new diagnostics in the existing path tracer analysis panel.

- Modify `src/Core/EngineCore.h`
  - Add experiment accumulation fields for the new diagnostics.

- Modify `src/Core/EngineCore.cpp`
  - Copy new counters into UI stats.
  - Include new fields in sweep accumulation and row summaries.

- Modify `tests/PathTracerAnalysisTests.cpp`
  - Add/update shader contract checks for indexed lookup.
  - Add/update counter offset checks for new diagnostics.
  - Keep guardrails that prevent using selector-local PDF as reservoir source PDF.

- Modify `docs/superpowers/plans/2026-05-17-global-bright-surfel-sampling.md`
  - Append a short follow-up note recording that the next experiment is diagnostic indexed lookup.

---

### Task 1: Add Counter Contract Tests First

**Files:**
- Modify: `tests/PathTracerAnalysisTests.cpp`

- [ ] **Step 1: Update both required shader symbol lists**

`tests/PathTracerAnalysisTests.cpp` currently has two bright-surfel shader contract blocks: one in the reservoir GI measurement contract and one in the broader path tracer debug/AOV contract. Update both blocks the same way.

Remove these old global-scan required symbols from both lists:

```cpp
"RESERVOIR_GI_BRIGHT_SURFEL_GLOBAL_SCAN_COUNT",
"reservoirGiBrightSurfelGlobalIndex",
"reservoirGiBrightSurfelGlobalStoreIndex",
"selectWeightedGlobalBrightReceiverSurfelRecord",
```

Keep these existing symbols because the attempt cadence and diagnostic PDF still matter:

```cpp
"RESERVOIR_GI_BRIGHT_SURFEL_ATTEMPT_STRIDE",
"RESERVOIR_GI_BRIGHT_SURFEL_CANDIDATE_MIN_TARGET_WEIGHT",
"surfelSelectionPdf",
```

Add the new indexed constants and helper names to both lists:

```cpp
"RESERVOIR_GI_BRIGHT_SURFEL_INDEX_CELL_SIZE",
"RESERVOIR_GI_BRIGHT_SURFEL_INDEX_CELL_SLOTS",
"RESERVOIR_GI_BRIGHT_SURFEL_INDEX_RADIUS_CELLS",
"reservoirGiBrightSurfelIndexedStoreIndex",
"reservoirGiBrightSurfelIndexedQueryIndex",
"selectWeightedIndexedBrightReceiverSurfelRecord",
"reservoirGiBrightSurfelIndexedQueryOffset",
"reservoirGiBrightSurfelIndexedEmptyOffset",
"reservoirGiBrightSurfelIndexedProbeOffset",
"reservoirGiBrightSurfelSelectorRejectDistanceOffset",
"reservoirGiBrightSurfelSelectorRejectReceiverHemisphereOffset",
"reservoirGiBrightSurfelSelectorRejectSurfelHemisphereOffset",
"reservoirGiBrightSurfelSelectorRejectInvalidVectorOffset",
"BRIGHT_SURFEL_TARGET_REJECT_DISTANCE",
"BRIGHT_SURFEL_TARGET_REJECT_RECEIVER_HEMISPHERE",
"BRIGHT_SURFEL_TARGET_REJECT_SURFEL_HEMISPHERE",
"BRIGHT_SURFEL_TARGET_REJECT_INVALID_VECTOR",
```

- [ ] **Step 2: Require the new selector body in both contract blocks**

In both test blocks, replace the current selector body extraction:

```cpp
const std::string brightSurfelSelector =
    extractFunctionBody(raygen, "bool selectWeightedGlobalBrightReceiverSurfelRecord");
```

with:

```cpp
const std::string brightSurfelSelector =
    extractFunctionBody(raygen, "bool selectWeightedIndexedBrightReceiverSurfelRecord");
```

Then update both failure messages to:

```cpp
std::cerr << "missing indexed bright surfel selector function body\n";
```

- [ ] **Step 3: Require indexed lookup and split reject counters inside both selector checks**

Update both `requiredBrightSurfelSelectorSymbols` arrays to include:

```cpp
const char *requiredBrightSurfelSelectorSymbols[] = {
    "reservoirGiBrightSurfelIndexedQueryIndex(hitPos, cellOffset, slot)",
    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelIndexedQueryOffset, 1u)",
    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelIndexedEmptyOffset, 1u)",
    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelIndexedProbeOffset, 1u)",
    "estimateBrightSurfelTargetForReceiver(hitPos, N, V, primaryPayload",
    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorRejectDistanceOffset, 1u)",
    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorRejectReceiverHemisphereOffset, 1u)",
    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorRejectSurfelHemisphereOffset, 1u)",
    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorRejectInvalidVectorOffset, 1u)",
    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorRejectGeometryOffset, 1u)",
    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorRejectTargetOffset, 1u)",
    "ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorViableOffset, 1u)",
    "selectedTargetEvaluation = estimate.targetEvaluation",
    "float score = estimate.targetEvaluation.targetWeight"};
```

- [ ] **Step 4: Add a forbidden global selector guard**

Add this symbol to the existing `forbiddenRaygenSymbols` list:

```cpp
"selectWeightedGlobalBrightReceiverSurfelRecord",
```

Keep `reservoirGiBrightSurfelGlobalIndex` out of the forbidden list until Task 3 removes the old query helper. This makes the first failing test point at the intended selector replacement instead of failing on helper cleanup.

- [ ] **Step 5: Extend the counter offset table**

In the `PathTracerAnalysisCounters` offset contract table, append these entries after `reservoirGiBrightSurfelSelectorViable`:

```cpp
{"reservoirGiBrightSurfelIndexedQuery",
 offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelIndexedQuery), 376u},
{"reservoirGiBrightSurfelIndexedEmpty",
 offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelIndexedEmpty), 380u},
{"reservoirGiBrightSurfelIndexedProbe",
 offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelIndexedProbe), 384u},
{"reservoirGiBrightSurfelSelectorRejectDistance",
 offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelSelectorRejectDistance), 388u},
{"reservoirGiBrightSurfelSelectorRejectReceiverHemisphere",
 offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelSelectorRejectReceiverHemisphere), 392u},
{"reservoirGiBrightSurfelSelectorRejectSurfelHemisphere",
 offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelSelectorRejectSurfelHemisphere), 396u},
{"reservoirGiBrightSurfelSelectorRejectInvalidVector",
 offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelSelectorRejectInvalidVector), 400u}
```

- [ ] **Step 6: Extend row summary/UI field tests**

Where tests list required row summary fields and UI labels, add:

```cpp
"brightSurfelIndexedQuery",
"brightSurfelIndexedEmpty",
"brightSurfelIndexedProbe",
"brightSurfelSelectorRejectDistance",
"brightSurfelSelectorRejectReceiverHemisphere",
"brightSurfelSelectorRejectSurfelHemisphere",
"brightSurfelSelectorRejectInvalidVector",
```

and UI labels:

```cpp
"Reservoir GI Bright Surfel Indexed Query",
"Reservoir GI Bright Surfel Indexed Empty",
"Reservoir GI Bright Surfel Indexed Probe",
"Reservoir GI Bright Surfel Selector Reject Distance",
"Reservoir GI Bright Surfel Selector Reject Receiver Hemisphere",
"Reservoir GI Bright Surfel Selector Reject Surfel Hemisphere",
"Reservoir GI Bright Surfel Selector Reject Invalid Vector",
```

- [ ] **Step 7: Run the focused tests and verify failure**

Run:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" >nul && cmake --build cmake-build-debug --target LaphriaEngineUnitTests --config Debug && ctest --test-dir cmake-build-debug --output-on-failure"
```

Expected: tests fail because the new shader functions, counters, UI fields, and row summary fields do not exist yet.

---

### Task 2: Add CPU Counter Plumbing

**Files:**
- Modify: `src/Core/EngineAuxiliary.h`
- Modify: `src/Core/UISystem.h`
- Modify: `src/Core/UISystem.cpp`
- Modify: `src/Core/EngineCore.h`
- Modify: `src/Core/EngineCore.cpp`

- [ ] **Step 1: Add engine-side analysis counter fields**

In `src/Core/EngineAuxiliary.h`, append these fields immediately after `reservoirGiBrightSurfelSelectorViable`:

```cpp
uint32_t reservoirGiBrightSurfelIndexedQuery = 0;
uint32_t reservoirGiBrightSurfelIndexedEmpty = 0;
uint32_t reservoirGiBrightSurfelIndexedProbe = 0;
uint32_t reservoirGiBrightSurfelSelectorRejectDistance = 0;
uint32_t reservoirGiBrightSurfelSelectorRejectReceiverHemisphere = 0;
uint32_t reservoirGiBrightSurfelSelectorRejectSurfelHemisphere = 0;
uint32_t reservoirGiBrightSurfelSelectorRejectInvalidVector = 0;
```

- [ ] **Step 2: Add UI stat fields**

In `src/Core/UISystem.h`, append the same fields immediately after `reservoirGiBrightSurfelSelectorViable`:

```cpp
uint32_t reservoirGiBrightSurfelIndexedQuery = 0;
uint32_t reservoirGiBrightSurfelIndexedEmpty = 0;
uint32_t reservoirGiBrightSurfelIndexedProbe = 0;
uint32_t reservoirGiBrightSurfelSelectorRejectDistance = 0;
uint32_t reservoirGiBrightSurfelSelectorRejectReceiverHemisphere = 0;
uint32_t reservoirGiBrightSurfelSelectorRejectSurfelHemisphere = 0;
uint32_t reservoirGiBrightSurfelSelectorRejectInvalidVector = 0;
```

- [ ] **Step 3: Display the new UI stats**

In `src/Core/UISystem.cpp`, add these lines immediately after the existing selector viable display:

```cpp
ImGui::Text("Reservoir GI Bright Surfel Indexed Query: %u",
            pathTracerPerfStats.reservoirGiBrightSurfelIndexedQuery);
ImGui::Text("Reservoir GI Bright Surfel Indexed Empty: %u",
            pathTracerPerfStats.reservoirGiBrightSurfelIndexedEmpty);
ImGui::Text("Reservoir GI Bright Surfel Indexed Probe: %u",
            pathTracerPerfStats.reservoirGiBrightSurfelIndexedProbe);
ImGui::Text("Reservoir GI Bright Surfel Selector Reject Distance: %u",
            pathTracerPerfStats.reservoirGiBrightSurfelSelectorRejectDistance);
ImGui::Text("Reservoir GI Bright Surfel Selector Reject Receiver Hemisphere: %u",
            pathTracerPerfStats.reservoirGiBrightSurfelSelectorRejectReceiverHemisphere);
ImGui::Text("Reservoir GI Bright Surfel Selector Reject Surfel Hemisphere: %u",
            pathTracerPerfStats.reservoirGiBrightSurfelSelectorRejectSurfelHemisphere);
ImGui::Text("Reservoir GI Bright Surfel Selector Reject Invalid Vector: %u",
            pathTracerPerfStats.reservoirGiBrightSurfelSelectorRejectInvalidVector);
```

- [ ] **Step 4: Add experiment accumulation fields**

In `src/Core/EngineCore.h`, append these `double` fields immediately after `brightSurfelSelectorViable`:

```cpp
double brightSurfelIndexedQuery = 0.0;
double brightSurfelIndexedEmpty = 0.0;
double brightSurfelIndexedProbe = 0.0;
double brightSurfelSelectorRejectDistance = 0.0;
double brightSurfelSelectorRejectReceiverHemisphere = 0.0;
double brightSurfelSelectorRejectSurfelHemisphere = 0.0;
double brightSurfelSelectorRejectInvalidVector = 0.0;
```

- [ ] **Step 5: Copy counters into UI stats**

In `src/Core/EngineCore.cpp`, after copying `reservoirGiBrightSurfelSelectorViable`, add:

```cpp
ui.pathTracerPerfStats.reservoirGiBrightSurfelIndexedQuery =
    counters->reservoirGiBrightSurfelIndexedQuery;
ui.pathTracerPerfStats.reservoirGiBrightSurfelIndexedEmpty =
    counters->reservoirGiBrightSurfelIndexedEmpty;
ui.pathTracerPerfStats.reservoirGiBrightSurfelIndexedProbe =
    counters->reservoirGiBrightSurfelIndexedProbe;
ui.pathTracerPerfStats.reservoirGiBrightSurfelSelectorRejectDistance =
    counters->reservoirGiBrightSurfelSelectorRejectDistance;
ui.pathTracerPerfStats.reservoirGiBrightSurfelSelectorRejectReceiverHemisphere =
    counters->reservoirGiBrightSurfelSelectorRejectReceiverHemisphere;
ui.pathTracerPerfStats.reservoirGiBrightSurfelSelectorRejectSurfelHemisphere =
    counters->reservoirGiBrightSurfelSelectorRejectSurfelHemisphere;
ui.pathTracerPerfStats.reservoirGiBrightSurfelSelectorRejectInvalidVector =
    counters->reservoirGiBrightSurfelSelectorRejectInvalidVector;
```

- [ ] **Step 6: Extend sweep row summary formatting**

In the sweep row summary format string, after `brightSurfelSelectorViable=%.1f, `, add:

```cpp
"brightSurfelIndexedQuery=%.1f, brightSurfelIndexedEmpty=%.1f, "
"brightSurfelIndexedProbe=%.1f, brightSurfelSelectorRejectDistance=%.1f, "
"brightSurfelSelectorRejectReceiverHemisphere=%.1f, "
"brightSurfelSelectorRejectSurfelHemisphere=%.1f, "
"brightSurfelSelectorRejectInvalidVector=%.1f, "
```

Then add the matching arguments:

```cpp
accum.brightSurfelIndexedQuery * invSamples,
accum.brightSurfelIndexedEmpty * invSamples,
accum.brightSurfelIndexedProbe * invSamples,
accum.brightSurfelSelectorRejectDistance * invSamples,
accum.brightSurfelSelectorRejectReceiverHemisphere * invSamples,
accum.brightSurfelSelectorRejectSurfelHemisphere * invSamples,
accum.brightSurfelSelectorRejectInvalidVector * invSamples,
```

- [ ] **Step 7: Accumulate the new fields**

In the experiment accumulation function, after accumulating `brightSurfelSelectorViable`, add:

```cpp
ptExperimentAccum.brightSurfelIndexedQuery +=
    static_cast<double>(stats.reservoirGiBrightSurfelIndexedQuery);
ptExperimentAccum.brightSurfelIndexedEmpty +=
    static_cast<double>(stats.reservoirGiBrightSurfelIndexedEmpty);
ptExperimentAccum.brightSurfelIndexedProbe +=
    static_cast<double>(stats.reservoirGiBrightSurfelIndexedProbe);
ptExperimentAccum.brightSurfelSelectorRejectDistance +=
    static_cast<double>(stats.reservoirGiBrightSurfelSelectorRejectDistance);
ptExperimentAccum.brightSurfelSelectorRejectReceiverHemisphere +=
    static_cast<double>(stats.reservoirGiBrightSurfelSelectorRejectReceiverHemisphere);
ptExperimentAccum.brightSurfelSelectorRejectSurfelHemisphere +=
    static_cast<double>(stats.reservoirGiBrightSurfelSelectorRejectSurfelHemisphere);
ptExperimentAccum.brightSurfelSelectorRejectInvalidVector +=
    static_cast<double>(stats.reservoirGiBrightSurfelSelectorRejectInvalidVector);
```

- [ ] **Step 8: Build tests and verify CPU plumbing still fails only on shader work**

Run:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" >nul && cmake --build cmake-build-debug --target LaphriaEngineUnitTests --config Debug && ctest --test-dir cmake-build-debug --output-on-failure"
```

Expected: counter offset/UI/row summary tests should advance; shader selector tests still fail because Task 3 is not implemented.

---

### Task 3: Implement Indexed Store And Query In Raygen

**Files:**
- Modify: `src/shaders/Raygen.slang`

- [ ] **Step 1: Add shader counter offsets**

After `reservoirGiBrightSurfelSelectorViableOffset`, add:

```slang
static const uint reservoirGiBrightSurfelIndexedQueryOffset = 376u;
static const uint reservoirGiBrightSurfelIndexedEmptyOffset = 380u;
static const uint reservoirGiBrightSurfelIndexedProbeOffset = 384u;
static const uint reservoirGiBrightSurfelSelectorRejectDistanceOffset = 388u;
static const uint reservoirGiBrightSurfelSelectorRejectReceiverHemisphereOffset = 392u;
static const uint reservoirGiBrightSurfelSelectorRejectSurfelHemisphereOffset = 396u;
static const uint reservoirGiBrightSurfelSelectorRejectInvalidVectorOffset = 400u;
```

- [ ] **Step 2: Add indexed lookup constants**

Near the existing bright surfel constants, add:

```slang
static const float RESERVOIR_GI_BRIGHT_SURFEL_INDEX_CELL_SIZE = 0.75;
static const uint RESERVOIR_GI_BRIGHT_SURFEL_INDEX_CELL_SLOTS = 2u;
static const uint RESERVOIR_GI_BRIGHT_SURFEL_INDEX_RADIUS_CELLS = 1u;
static const uint RESERVOIR_GI_BRIGHT_SURFEL_INDEX_CELL_COUNT =
    (RESERVOIR_GI_BRIGHT_SURFEL_INDEX_RADIUS_CELLS * 2u + 1u) *
    (RESERVOIR_GI_BRIGHT_SURFEL_INDEX_RADIUS_CELLS * 2u + 1u) *
    (RESERVOIR_GI_BRIGHT_SURFEL_INDEX_RADIUS_CELLS * 2u + 1u);
```

- [ ] **Step 3: Add cell and offset helpers**

Replace or supplement the old bright surfel spatial/global index helpers with:

```slang
int3 reservoirGiBrightSurfelIndexCell(float3 position)
{
    return int3(floor(position / RESERVOIR_GI_BRIGHT_SURFEL_INDEX_CELL_SIZE));
}

int3 reservoirGiBrightSurfelCellOffset(uint offsetIndex)
{
    uint span = RESERVOIR_GI_BRIGHT_SURFEL_INDEX_RADIUS_CELLS * 2u + 1u;
    uint z = offsetIndex / (span * span);
    uint rem = offsetIndex - z * span * span;
    uint y = rem / span;
    uint x = rem - y * span;
    int radius = int(RESERVOIR_GI_BRIGHT_SURFEL_INDEX_RADIUS_CELLS);
    return int3(int(x) - radius, int(y) - radius, int(z) - radius);
}

uint reservoirGiBrightSurfelIndexHash(int3 cell, uint slotSalt)
{
    uint hash = pcgHash(uint(cell.x) * 73856093u ^
                        uint(cell.y) * 19349663u ^
                        uint(cell.z) * 83492791u ^
                        slotSalt * 2654435761u);
    return hash % RESERVOIR_GI_BRIGHT_SURFEL_CAPACITY;
}

uint reservoirGiBrightSurfelIndexedStoreIndex(float3 position, uint slotSalt)
{
    int3 cell = reservoirGiBrightSurfelIndexCell(position);
    return reservoirGiBrightSurfelIndexHash(cell, slotSalt);
}

uint reservoirGiBrightSurfelIndexedQueryIndex(float3 receiverPosition,
                                              uint cellOffsetIndex,
                                              uint slot)
{
    int3 receiverCell = reservoirGiBrightSurfelIndexCell(receiverPosition);
    int3 cellOffset = reservoirGiBrightSurfelCellOffset(cellOffsetIndex);
    // Store and query must use the same per-cell slot salt. The offset changes the cell,
    // not the slot namespace; otherwise a stored surfel cannot be found from a neighbor query.
    return reservoirGiBrightSurfelIndexHash(receiverCell + cellOffset, slot);
}
```

- [ ] **Step 4: Store surfels into indexed cells**

In `storeReservoirGiBrightSurfelRecord`, replace the current `surfelIndex` assignment:

```slang
uint surfelIndex = reservoirGiBrightSurfelGlobalStoreIndex(
    launchID,
    launchSize,
    selectedRecord);
```

with:

```slang
uint storeSalt = pcgHash(sourcePixel ^
                         selectedRecord.sourcePixel * 1013904223u ^
                         ubo.frameCount * 747796405u) %
                 RESERVOIR_GI_BRIGHT_SURFEL_INDEX_CELL_SLOTS;
uint surfelIndex = reservoirGiBrightSurfelIndexedStoreIndex(
    selectedRecord.candidatePosition,
    storeSalt);
```

- [ ] **Step 5: Split geometric reject reasons in the estimator**

Replace the existing three bright surfel reject constants near `BrightSurfelTargetEstimateResult` with this single block. Do not add a second duplicate block:

```slang
static const uint BRIGHT_SURFEL_TARGET_REJECT_NONE = 0u;
static const uint BRIGHT_SURFEL_TARGET_REJECT_GEOMETRY = 1u;
static const uint BRIGHT_SURFEL_TARGET_REJECT_TARGET = 2u;
static const uint BRIGHT_SURFEL_TARGET_REJECT_DISTANCE = 3u;
static const uint BRIGHT_SURFEL_TARGET_REJECT_RECEIVER_HEMISPHERE = 4u;
static const uint BRIGHT_SURFEL_TARGET_REJECT_SURFEL_HEMISPHERE = 5u;
static const uint BRIGHT_SURFEL_TARGET_REJECT_INVALID_VECTOR = 6u;
```

In `estimateBrightSurfelTargetForReceiver`, change the early geometry checks to set specific reasons:

```slang
float3 toSurfel = surfel.position - hitPos;
float dist2 = dot(toSurfel, toSurfel);
if (dist2 <= 0.0001 || !isFinitePositive(dist2)) {
    estimate.rejectReason = BRIGHT_SURFEL_TARGET_REJECT_DISTANCE;
    return false;
}

float3 wi = toSurfel * rsqrt(dist2);
if (!isSaneReservoirGiVector(wi, 1.1)) {
    estimate.rejectReason = BRIGHT_SURFEL_TARGET_REJECT_INVALID_VECTOR;
    return false;
}

float receiverCos = dot(N, wi);
if (receiverCos <= 0.0) {
    estimate.rejectReason = BRIGHT_SURFEL_TARGET_REJECT_RECEIVER_HEMISPHERE;
    return false;
}

float surfelCos = dot(surfel.normal, -wi);
if (surfelCos <= RESERVOIR_GI_BRIGHT_SURFEL_COS_THETA_MAX) {
    estimate.rejectReason = BRIGHT_SURFEL_TARGET_REJECT_SURFEL_HEMISPHERE;
    return false;
}
```

Keep the later `validGeometry` failure mapped to `BRIGHT_SURFEL_TARGET_REJECT_GEOMETRY`, and keep light/weight failures mapped to `BRIGHT_SURFEL_TARGET_REJECT_TARGET`.

- [ ] **Step 6: Add the indexed selector**

Replace `selectWeightedGlobalBrightReceiverSurfelRecord` with:

```slang
bool selectWeightedIndexedBrightReceiverSurfelRecord(float3 hitPos,
                                                     float3 N,
                                                     float3 V,
                                                     RayPayload primaryPayload,
                                                     uint2 launchID,
                                                     uint2 launchSize,
                                                     inout uint rngState,
                                                     out ReservoirGiBrightSurfelRecord selectedSurfel,
                                                     out ReservoirGiTargetEvaluation selectedTargetEvaluation,
                                                     out float surfelSelectionPdf)
{
    selectedSurfel.position = float3(0.0, 0.0, 0.0);
    selectedSurfel.normal = float3(0.0, 1.0, 0.0);
    selectedSurfel.radiance = float3(0.0, 0.0, 0.0);
    selectedSurfel.targetWeight = 0.0;
    selectedSurfel.confidence = 0.0;
    selectedSurfel.radius = 0.0;
    selectedSurfel.frameId = 0u;
    selectedSurfel.flags = 0u;
    selectedSurfel.sourcePixel = 0u;
    selectedTargetEvaluation.contribution = float3(0.0, 0.0, 0.0);
    selectedTargetEvaluation.suffixRadiance = float3(0.0, 0.0, 0.0);
    selectedTargetEvaluation.targetWeight = 0.0;
    selectedTargetEvaluation.sourcePdf = 0.0;
    selectedTargetEvaluation.validGeometry = false;
    selectedTargetEvaluation.validLight = false;
    surfelSelectionPdf = 0.0;

    ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelAttemptOffset, 1u);

    float scoreSum = 0.0;
    float selectedScore = 0.0;
    bool found = false;
    uint surfelCapacity;
    uint surfelHistoryFrameId;
    if (!loadReservoirGiBrightSurfelHistoryHeader(surfelCapacity, surfelHistoryFrameId)) {
        ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelMissOffset, 1u);
        return false;
    }

    [unroll]
    for (uint cellOffset = 0u; cellOffset < RESERVOIR_GI_BRIGHT_SURFEL_INDEX_CELL_COUNT; ++cellOffset) {
        [unroll]
        for (uint slot = 0u; slot < RESERVOIR_GI_BRIGHT_SURFEL_INDEX_CELL_SLOTS; ++slot) {
            ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelIndexedQueryOffset, 1u);
            uint surfelIndex = reservoirGiBrightSurfelIndexedQueryIndex(hitPos, cellOffset, slot);
            ReservoirGiBrightSurfelPrecheck precheck;
            if (!precheckReservoirGiBrightSurfelHistoryRecord(surfelIndex, surfelCapacity, surfelHistoryFrameId,
                                                              hitPos, cellOffset, precheck)) {
                ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelIndexedEmptyOffset, 1u);
                continue;
            }

            ReservoirGiBrightSurfelRecord surfel;
            if (!loadPrecheckedReservoirGiBrightSurfelHistoryRecord(surfelIndex, precheck, surfel)) {
                ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelIndexedEmptyOffset, 1u);
                continue;
            }

            ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelIndexedProbeOffset, 1u);
            BrightSurfelTargetEstimateResult estimate;
            if (!estimateBrightSurfelTargetForReceiver(hitPos, N, V, primaryPayload,
                                                       surfel, estimate)) {
                if (estimate.rejectReason == BRIGHT_SURFEL_TARGET_REJECT_DISTANCE) {
                    ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorRejectDistanceOffset, 1u);
                } else if (estimate.rejectReason == BRIGHT_SURFEL_TARGET_REJECT_RECEIVER_HEMISPHERE) {
                    ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorRejectReceiverHemisphereOffset, 1u);
                } else if (estimate.rejectReason == BRIGHT_SURFEL_TARGET_REJECT_SURFEL_HEMISPHERE) {
                    ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorRejectSurfelHemisphereOffset, 1u);
                } else if (estimate.rejectReason == BRIGHT_SURFEL_TARGET_REJECT_INVALID_VECTOR) {
                    ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelSelectorRejectInvalidVectorOffset, 1u);
                } else if (estimate.rejectReason == BRIGHT_SURFEL_TARGET_REJECT_GEOMETRY) {
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
            scoreSum += score;
            if (randomFloat(rngState) < score / max(scoreSum, 0.000001)) {
                selectedScore = score;
                selectedSurfel = surfel;
                selectedTargetEvaluation = estimate.targetEvaluation;
                found = true;
            }
        }
    }

    if (!found) {
        ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelMissOffset, 1u);
        return false;
    }

    surfelSelectionPdf = selectedScore / max(scoreSum, 0.000001);
    ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelHitOffset, 1u);
    return true;
}
```

- [ ] **Step 7: Call the indexed selector from the candidate evaluator**

In `evaluateBrightReceiverSurfelReservoirGiCandidate`, replace:

```slang
if (!selectWeightedGlobalBrightReceiverSurfelRecord(hitPos, N, V, primaryPayload,
                                                   launchID, launchSize,
                                                   rngState, surfel, selectedTargetEvaluation,
                                                   surfelSelectionPdf)) {
```

with:

```slang
if (!selectWeightedIndexedBrightReceiverSurfelRecord(hitPos, N, V, primaryPayload,
                                                    launchID, launchSize,
                                                    rngState, surfel, selectedTargetEvaluation,
                                                    surfelSelectionPdf)) {
```

- [ ] **Step 8: Remove old global query helpers after callers are gone**

Delete `reservoirGiBrightSurfelGlobalIndex` and `reservoirGiBrightSurfelGlobalStoreIndex` if no call sites remain. Keep `reservoirGiBrightSurfelSpatialIndex` only if it is still used; otherwise delete it too.

- [ ] **Step 9: Build shader/tests**

Run:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" >nul && cmake --build cmake-build-debug --target LaphriaEditor --config Debug && ctest --test-dir cmake-build-debug --output-on-failure"
```

Expected: Slang compilation succeeds and tests pass or fail only on row-summary/contract strings that Task 4 will finish.

---

### Task 4: Finalize Diagnostics In Tests And Docs

**Files:**
- Modify: `tests/PathTracerAnalysisTests.cpp`
- Modify: `docs/superpowers/plans/2026-05-17-global-bright-surfel-sampling.md`
- Modify: `docs/superpowers/plans/2026-05-17-diagnostic-indexed-bright-surfel-lookup.md`

- [ ] **Step 1: Forbid old global helpers once Task 3 removes them**

Add these to `forbiddenRaygenSymbols`:

```cpp
"reservoirGiBrightSurfelGlobalIndex",
"reservoirGiBrightSurfelGlobalStoreIndex",
```

Do not forbid `surfelSelectionPdf`; the candidate evaluator intentionally keeps it as a diagnostic-only value and must not use it as `sourcePdf`.

- [ ] **Step 2: Confirm guardrail for selector-local PDF remains**

Keep this comment required in the shader contract:

```cpp
"Scan-local selector diagnostic only; do not feed it into target sourcePdf or reservoir weight.",
```

If the exact wording changed in Task 3, update the test to the exact final comment so future edits preserve the intent.

- [ ] **Step 3: Append the follow-up note to the prior surfel plan**

Append this section to `docs/superpowers/plans/2026-05-17-global-bright-surfel-sampling.md`:

```markdown
## Follow-Up: Diagnostic Indexed Lookup

The target-aware bright surfel selector proved that the reservoir combine path can accept surfels on the Sunlit Wall, but Dark Courtyard and Mid-Depth produced zero viable surfels while spending heavily on global selector probes. The next experiment is diagnostic indexed lookup: reuse the current bright surfel buffer, store records into spatial cells, query the receiver neighborhood, and split selector rejects into empty-cell, distance, receiver hemisphere, surfel hemisphere, invalid-vector, geometry, and target categories.

Success criteria for the next sweep:
- Mid-Depth bright surfel rows report nonzero `brightSurfelSelectorViable`.
- Sunlit Wall still reports nonzero `brightSurfelAccepted` and `reservoirGiSelectedBrightSurfel`.
- Bright surfel rows avoid the current multi-million geometry reject pattern.
- Total time moves toward the Sun Receiver baseline enough to justify deeper cache work.
```

- [ ] **Step 4: Mark this plan's implementation notes section**

At the bottom of this plan, add:

```markdown
## Implementation Notes

- This pass intentionally does not add a new GPU buffer.
- The indexed selector still uses target-aware evaluation before visibility.
- The indexed selector now loads the bright-surfel history header once per attempt and cheaply prechecks slot flags, frame, target weight, and indexed cell before loading full normal/radiance/radius fields.
- The selector-local PDF remains diagnostic and must not replace `targetEvaluation.sourcePdf`.
- If indexed lookup still produces zero Mid-Depth viable surfels, the next likely step is producer-side training quality rather than more selector math.
```

- [ ] **Step 5: Run all tests**

Run:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" >nul && cmake --build cmake-build-debug --target LaphriaEditor --config Debug && ctest --test-dir cmake-build-debug --output-on-failure"
```

Expected: all CTest tests pass.

---

### Task 5: Run The Audit Sweep And Interpret The New Counters

**Files:**
- No planned source edits unless the sweep exposes a compile/runtime issue.

- [ ] **Step 1: Run the existing Sponza PT/GI audit sweep**

Run `LaphriaEditor`, open the path tracer analysis panel, set:

```text
Sponza Sweep Warmup = 8
Sponza Sweep Samples = 32
```

Then click:

```text
Run Sponza GI Perf Sweep
```

Wait for this completion log:

```text
PT Experiment Sweep: Sponza PT/GI audit sweep complete
```

Capture the row summaries for:

```text
Sponza / Dark Courtyard / ... Sun Receiver Bright Surfel
Sponza / Dark Courtyard / ... Single Frame Sun Receiver Bright Surfel
Sponza / Sunlit Courtyard Wall / ... Sun Receiver Bright Surfel
Sponza / Sunlit Courtyard Wall / ... Single Frame Sun Receiver Bright Surfel
Sponza / Mid-Depth Interior / ... Sun Receiver Bright Surfel
Sponza / Mid-Depth Interior / ... Single Frame Sun Receiver Bright Surfel
```

- [ ] **Step 2: Interpret indexed lookup health**

Use these thresholds for the first read:

```text
Healthy indexed lookup:
- brightSurfelIndexedQuery is bounded near attempt count * 54.
- brightSurfelIndexedProbe is much lower than query count.
- brightSurfelSelectorRejectGeometry no longer reaches multi-million scale.
- brightSurfelSelectorViable is nonzero in Sunlit Wall and ideally nonzero in Mid-Depth.

Likely empty producer/index:
- brightSurfelIndexedEmpty dominates query count.
- brightSurfelSelectorViable remains 0 in Mid-Depth.

Likely bad local-neighborhood assumption:
- brightSurfelIndexedProbe is nonzero, but receiver/surfel hemisphere rejects dominate.
- Sunlit still works, but Mid-Depth remains zero viable.

Likely target/radiance threshold issue:
- distance and hemisphere rejects are modest, but target rejects dominate.
```

- [ ] **Step 3: Record the outcome**

Append this template to `docs/superpowers/plans/2026-05-17-diagnostic-indexed-bright-surfel-lookup.md` and fill in the numbers from the sweep:

```markdown
## Sweep Outcome

| View | Variant | Viable | Accepted | Selected | Indexed Query | Indexed Empty | Indexed Probe | Reject Distance | Reject Receiver Hemi | Reject Surfel Hemi | Reject Invalid Vector | Reject Geometry | Reject Target | Total ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dark Courtyard | Temporal Spatial |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Dark Courtyard | Single Frame |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Sunlit Wall | Temporal Spatial |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Sunlit Wall | Single Frame |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mid-Depth | Temporal Spatial |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mid-Depth | Single Frame |  |  |  |  |  |  |  |  |  |  |  |  |  |

Decision:
- Continue indexed surfel path if Mid-Depth viability becomes nonzero and cost is plausibly controllable.
- Shift to producer/training quality if indexed probes are mostly empty.
- Shift away from local surfels toward many-light virtual reservoirs if indexed probes exist but hemisphere/target rejects still dominate.
```

---

## Success Criteria

- `selectWeightedGlobalBrightReceiverSurfelRecord` is gone.
- Bright surfel store uses `reservoirGiBrightSurfelIndexedStoreIndex`.
- Bright surfel candidate selection uses `selectWeightedIndexedBrightReceiverSurfelRecord`.
- New row summaries include:
  - `brightSurfelIndexedQuery`
  - `brightSurfelIndexedEmpty`
  - `brightSurfelIndexedProbe`
  - `brightSurfelSelectorRejectDistance`
  - `brightSurfelSelectorRejectReceiverHemisphere`
  - `brightSurfelSelectorRejectSurfelHemisphere`
  - `brightSurfelSelectorRejectInvalidVector`
- Existing target-aware combine guardrail remains intact: bright surfel selected source uses `surfelRecord.targetWeight`, not a selector-local PDF.
- Slang build passes.
- CTest passes.
- New sweep provides enough data to decide whether indexed surfels deserve another iteration.

## Self-Review

- Spec coverage: The plan covers indexed store/query, split diagnostics, CPU/UI/logging plumbing, tests, docs, and sweep interpretation.
- Placeholder scan: No placeholder implementation steps remain. The only blank table cells are the explicit sweep-outcome template to fill after running the experiment.
- Type consistency: Counter names use the same root across shader offsets, CPU counters, UI stats, accumulation fields, row summaries, and tests.

## Implementation Notes

- This pass intentionally does not add a new GPU buffer.
- The indexed selector still uses target-aware evaluation before visibility.
- The indexed selector now loads the bright-surfel history header once per attempt and cheaply prechecks slot flags, frame, target weight, and indexed cell before loading full normal/radiance/radius fields.
- The selector-local PDF remains diagnostic and must not replace `targetEvaluation.sourcePdf`.
- If indexed lookup still produces zero Mid-Depth viable surfels, the next likely step is producer-side training quality rather than more selector math.
