# Global Bright Surfel Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current local-cell bright receiver surfel lookup with a bounded global bright-surface candidate sampler that can find useful indirect virtual lights outside the current receiver cell.

**Architecture:** Keep the existing bright surfel storage buffers, proposal mode 8, counters, UI, and sweep row. Change only the shader-side selection/evaluation model: store local accepted receiver records as before, sample a small randomized subset of the whole surfel pool from history, gate attempts to avoid full-screen lookup cost, and use a lower candidate acceptance threshold than the storage threshold. This keeps the path compatible with a future many-light reservoir because the selected surfel is treated as an explicit virtual light candidate with source, target, visibility, and selection counters.

**Tech Stack:** C++17 engine/UI/analysis plumbing, Slang ray generation shader, Vulkan storage buffers, string-contract tests in `tests/PathTracerAnalysisTests.cpp`, CMake/CTest verification, in-app Sponza PT/GI audit sweep.

---

## Context From The Previous Sweep

The local-cell prototype produced useful signal only when the receiver and bright stored surfel were spatially close.

```text
Dark Courtyard Bright Surfel:
  firstHitProbeAvgLuma=0.03574 vs Sun Receiver 0.03499
  brightSurfelHit=13.4
  brightSurfelRejectVisibility=13.4
  brightSurfelAccepted=0.0
  totalMs=68.008 vs Sun Receiver 62.690

Sunlit Courtyard Wall Bright Surfel:
  firstHitProbeAvgLuma=0.06553 vs Sun Receiver 0.05224
  brightSurfelAccepted=17671.8
  reservoirGiSelectedBrightSurfel=16481.4
  totalMs=200.604 vs Sun Receiver 167.139

Mid-Depth Interior Bright Surfel:
  firstHitProbeAvgLuma=0.04374 vs Sun Receiver 0.03674
  brightSurfelHit=65306.2
  brightSurfelRejectTarget=58363.9
  brightSurfelAccepted=0.0
  totalMs=185.044 vs Sun Receiver 162.174
```

The next implementation must address three failures:

```text
Dark Courtyard: near-zero global discovery from local-cell lookup.
Mid-Depth Interior: target threshold rejects every surfel candidate.
All scenarios: brightSurfelAttempt=2073600.0, so every pixel/sample pays the lookup even when the pool is sparse.
```

## File Structure

- Modify `tests/PathTracerAnalysisTests.cpp`
  - Add string-contract checks for the new global sampling constants and helper names.
  - Keep existing bright surfel counter offsets unchanged.
- Modify `src/shaders/Raygen.slang`
  - Add constants for global scan count, sparse attempt stride, and candidate minimum target weight.
  - Add helper functions for sparse attempt gating, global surfel reads, and globally distributed surfel writes.
  - Rename `selectBrightReceiverSurfelRecord(...)` to `selectGlobalBrightReceiverSurfelRecord(...)` and sample the history pool globally instead of querying only `reservoirGiBrightSurfelSpatialIndex(hitPos, slot)`.
  - Keep `storeReservoirGiBrightSurfelRecord(...)` producer-only, write those records into global slots, and keep bright surfel winners excluded from generic persistence.
- Modify `docs/superpowers/plans/2026-05-16-bright-receiver-surfel-reservoir.md`
  - Append a short follow-up note after the next sweep if this plan supersedes the local-cell row.

## Design Constants

Use these exact Slang names:

```slang
RESERVOIR_GI_BRIGHT_SURFEL_GLOBAL_SCAN_COUNT = 8u
RESERVOIR_GI_BRIGHT_SURFEL_ATTEMPT_STRIDE = 4u
RESERVOIR_GI_BRIGHT_SURFEL_CANDIDATE_MIN_TARGET_WEIGHT = 0.005
```

Keep these existing names unchanged:

```slang
RESERVOIR_GI_BRIGHT_SURFEL_MIN_TARGET_WEIGHT = 0.02
RESERVOIR_GI_BRIGHT_SURFEL_MIN_LUMA = 0.02
RESERVOIR_GI_BRIGHT_SURFEL_RADIUS = 0.15
RESERVOIR_GI_BRIGHT_SURFEL_COS_THETA_MAX = 0.8660254
RESERVOIR_GI_SOURCE_BRIGHT_SURFEL = 5u
RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_BRIGHT_SURFEL = 8
```

`RESERVOIR_GI_BRIGHT_SURFEL_MIN_TARGET_WEIGHT` remains the storage threshold. `RESERVOIR_GI_BRIGHT_SURFEL_CANDIDATE_MIN_TARGET_WEIGHT` is the lower candidate-evaluation threshold for selected global surfels.

---

### Task 1: Add Test Contracts For Global Sampling

**Files:**
- Modify: `tests/PathTracerAnalysisTests.cpp`

- [x] **Step 1: Update required shader symbol strings**

In the existing bright surfel shader contract list, add these exact strings:

```cpp
"RESERVOIR_GI_BRIGHT_SURFEL_GLOBAL_SCAN_COUNT",
"RESERVOIR_GI_BRIGHT_SURFEL_ATTEMPT_STRIDE",
"RESERVOIR_GI_BRIGHT_SURFEL_CANDIDATE_MIN_TARGET_WEIGHT",
"shouldAttemptBrightReceiverSurfel",
"reservoirGiBrightSurfelGlobalIndex",
"selectGlobalBrightReceiverSurfelRecord",
```

In the same list, remove this old local-cell selector requirement:

```cpp
"selectBrightReceiverSurfelRecord",
```

- [x] **Step 2: Keep counter offset expectations unchanged**

Confirm the existing offset table still ends with:

```cpp
{"reservoirGiBrightSurfelStore",
 offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelStore), 308u},
{"reservoirGiBrightSurfelAttempt",
 offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelAttempt), 312u},
{"reservoirGiBrightSurfelHit",
 offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelHit), 316u},
{"reservoirGiBrightSurfelMiss",
 offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelMiss), 320u},
{"reservoirGiBrightSurfelRejectVisibility",
 offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelRejectVisibility), 324u},
{"reservoirGiBrightSurfelRejectGeometry",
 offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelRejectGeometry), 328u},
{"reservoirGiBrightSurfelRejectTarget",
 offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelRejectTarget), 332u},
{"reservoirGiBrightSurfelAccepted",
 offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiBrightSurfelAccepted), 336u},
{"reservoirGiSelectedBrightSurfel",
 offsetof(Laphria::PathTracerAnalysisCounters, reservoirGiSelectedBrightSurfel), 340u}
```

- [x] **Step 3: Run tests and verify RED**

Run:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && ctest --test-dir cmake-build-debug --output-on-failure'
```

Expected: `LaphriaEngineUnitTests` fails because the new shader symbols do not exist yet.

---

### Task 2: Add Global Sampling Constants And Gating

**Files:**
- Modify: `src/shaders/Raygen.slang`

- [x] **Step 1: Add constants near the existing bright surfel constants**

Add:

```slang
static const uint RESERVOIR_GI_BRIGHT_SURFEL_GLOBAL_SCAN_COUNT = 8u;
static const uint RESERVOIR_GI_BRIGHT_SURFEL_ATTEMPT_STRIDE = 4u;
static const float RESERVOIR_GI_BRIGHT_SURFEL_CANDIDATE_MIN_TARGET_WEIGHT = 0.005;
```

- [x] **Step 2: Add sparse attempt gating helper**

Place this near the bright surfel helper functions:

```slang
bool shouldAttemptBrightReceiverSurfel(uint2 launchID, uint2 launchSize)
{
    uint sourcePixel = launchID.y * launchSize.x + launchID.x;
    uint hash = pcgHash(sourcePixel ^
                        ubo.frameCount * 747796405u ^
                        uint(launchSize.x) * 2891336453u ^
                        uint(launchSize.y) * 277803737u);
    return (hash % RESERVOIR_GI_BRIGHT_SURFEL_ATTEMPT_STRIDE) == 0u;
}
```

- [x] **Step 3: Gate the surfel candidate in `evaluateBrightReceiverSurfelReservoirGiCandidate`**

At the beginning of `evaluateBrightReceiverSurfelReservoirGiCandidate(...)`, after initializing `candidateRecord`, add:

```slang
    if (!shouldAttemptBrightReceiverSurfel(launchID, launchSize)) {
        return false;
    }
```

`brightSurfelAttempt` should remain inside selection, so skipped pixels do not inflate the attempt counter.

- [x] **Step 4: Run shader build**

Run:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cmake --build cmake-build-debug --target LaphriaEditor'
```

Expected: `Raygen.slang` compiles.

---

### Task 3: Replace Local-Cell Lookup With Global Pool Sampling

**Files:**
- Modify: `src/shaders/Raygen.slang`

- [x] **Step 1: Add global index helper**

Add this near `reservoirGiBrightSurfelSpatialIndex(...)`:

```slang
uint reservoirGiBrightSurfelGlobalIndex(uint2 launchID,
                                        uint2 launchSize,
                                        uint slot)
{
    uint sourcePixel = launchID.y * launchSize.x + launchID.x;
    uint hash = pcgHash(sourcePixel * 1664525u ^
                        ubo.frameCount * 1013904223u ^
                        slot * 2654435761u ^
                        uint(launchSize.x) * 2246822519u ^
                        uint(launchSize.y) * 3266489917u);
    return hash % RESERVOIR_GI_BRIGHT_SURFEL_CAPACITY;
}
```

- [x] **Step 1b: Add global store index helper**

Add this near the other bright surfel index helpers:

```slang
uint reservoirGiBrightSurfelGlobalStoreIndex(uint2 launchID,
                                             uint2 launchSize,
                                             ReservoirGiRecord selectedRecord)
{
    uint sourcePixel = launchID.y * launchSize.x + launchID.x;
    uint hash = pcgHash(sourcePixel * 1664525u ^
                        selectedRecord.sourcePixel * 1013904223u ^
                        ubo.frameCount * 747796405u ^
                        asuint(selectedRecord.candidatePosition.x) * 2246822519u ^
                        asuint(selectedRecord.candidatePosition.y) * 3266489917u ^
                        asuint(selectedRecord.candidatePosition.z) * 668265263u);
    return hash % RESERVOIR_GI_BRIGHT_SURFEL_CAPACITY;
}
```

- [x] **Step 1c: Use globally distributed storage**

In `storeReservoirGiBrightSurfelRecord(...)`, replace the current spatial store index:

```slang
uint surfelIndex = reservoirGiBrightSurfelSpatialIndex(
    selectedRecord.candidatePosition,
    pcgHash(selectedRecord.sourcePixel ^ sourcePixel ^ ubo.frameCount * 2891336453u) & 7u);
```

with:

```slang
uint surfelIndex = reservoirGiBrightSurfelGlobalStoreIndex(
    launchID, launchSize, selectedRecord);
```

Leave `reservoirGiBrightSurfelSpatialIndex(...)` in place for now to avoid unrelated churn.

- [x] **Step 2: Rename the selector helper**

Rename:

```slang
selectBrightReceiverSurfelRecord
```

to:

```slang
selectGlobalBrightReceiverSurfelRecord
```

Update the call site inside `evaluateBrightReceiverSurfelReservoirGiCandidate(...)`.

- [x] **Step 3: Replace the selector loop**

Inside `selectGlobalBrightReceiverSurfelRecord(...)`, replace the loop header and index computation with:

```slang
    [unroll]
    for (uint slot = 0u; slot < RESERVOIR_GI_BRIGHT_SURFEL_GLOBAL_SCAN_COUNT; ++slot) {
        uint surfelIndex = reservoirGiBrightSurfelGlobalIndex(launchID, launchSize, slot);
        ReservoirGiBrightSurfelRecord surfel;
        if (!loadReservoirGiBrightSurfelHistoryRecord(surfelIndex, surfel)) {
            continue;
        }
```

Keep the existing geometry checks and best-score selection.

- [x] **Step 4: Keep scoring receiver-aware**

Confirm the selector still scores with receiver and surfel cosine:

```slang
float score = pathTracerLuminance(max(surfel.radiance, float3(0.0, 0.0, 0.0))) *
              max(receiverCos, 0.0) *
              max(surfelCos, 0.0) /
              max(dist2, 0.0001);
```

- [x] **Step 5: Run shader build**

Run:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cmake --build cmake-build-debug --target LaphriaEditor'
```

Expected: `Raygen.slang` compiles.

---

### Task 4: Relax Candidate Acceptance Without Relaxing Storage

**Files:**
- Modify: `src/shaders/Raygen.slang`

- [x] **Step 1: Use the candidate threshold for selected global surfels**

In `evaluateBrightReceiverSurfelReservoirGiCandidate(...)`, change:

```slang
targetEvaluation.targetWeight <= RESERVOIR_GI_BRIGHT_SURFEL_MIN_TARGET_WEIGHT
```

to:

```slang
targetEvaluation.targetWeight <= RESERVOIR_GI_BRIGHT_SURFEL_CANDIDATE_MIN_TARGET_WEIGHT
```

- [x] **Step 2: Keep storage threshold unchanged**

Confirm `storeReservoirGiBrightSurfelRecord(...)` still rejects stored surfels with:

```slang
selectedRecord.targetWeight < RESERVOIR_GI_BRIGHT_SURFEL_MIN_TARGET_WEIGHT
```

This preserves a clean producer pool while allowing weaker global candidates to be tested by visibility and reservoir weighting.

- [x] **Step 3: Confirm no recursive persistence**

Confirm the generic persistence gate still excludes bright surfel winners:

```slang
bool canPersistSelectedReservoirGi = selectedSource != RESERVOIR_GI_SOURCE_CACHE_RECONNECT &&
    selectedSource != RESERVOIR_GI_SOURCE_BRIGHT_SURFEL &&
    (record.flags & RESERVOIR_GI_CACHE_CONTINUATION_FLAG) == 0u;
```

- [x] **Step 4: Run tests**

Run:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && ctest --test-dir cmake-build-debug --output-on-failure'
```

Expected: all configured tests pass.

---

### Task 5: Verify And Sweep

**Files:**
- Modify: `docs/superpowers/plans/2026-05-16-bright-receiver-surfel-reservoir.md`

- [x] **Step 1: Run full CTest**

Run:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && ctest --test-dir cmake-build-debug --output-on-failure'
```

Expected: all configured tests pass.

- [x] **Step 2: Run editor shader build**

Run:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cmake --build cmake-build-debug --target LaphriaEditor'
```

Expected: `Raygen.slang` compiles and the target is up to date.

- [x] **Step 3: Check whitespace**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors. Existing LF-to-CRLF warnings may appear and should be reported without changing unrelated files.

- [x] **Step 4: Run compact Sponza PT/GI audit sweep**

Compare these rows for each scenario:

```text
Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2
Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Sun Receiver
Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Sun Receiver Bright Surfel
Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Sun Receiver Env First Two
Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Sun Receiver Env First Two Cache Continuation
```

Success criteria:

```text
Dark Courtyard: brightSurfelAccepted > 0 and firstHitProbeAvgLuma >= Sun Receiver luma.
Sunlit Courtyard Wall: Bright Surfel remains within 10% of Env First Two luma and below 1.35x Sun Receiver cost.
Mid-Depth Interior: brightSurfelAccepted > 0 and firstHitProbeAvgLuma >= Sun Receiver luma.
brightSurfelAttempt is at least 2x lower than the previous 2073600.0 value.
reservoirGiSelectedBrightSurfel is nonzero but lower than reservoirGiSelectedLocal + reservoirGiSelectedTemporal + reservoirGiSelectedSpatial.
reservoirGiConfidenceMAvg remains below 10.0 in all three scenarios.
No bright pinpricks, wall leaks, or frame-to-frame flashing are visible.
```

Stop criteria:

```text
Dark Courtyard brightSurfelAccepted remains zero.
Mid-Depth Interior brightSurfelAccepted remains zero.
totalMs exceeds 1.5x Sun Receiver in two or more scenarios.
reservoirGiSelectedBrightSurfel dominates every selected-source category.
reservoirGiConfidenceMAvg exceeds 15.0 in any scenario.
```

- [x] **Step 5: Record outcome**

Append a `## Global Sampling Sweep Result` note to
`docs/superpowers/plans/2026-05-16-bright-receiver-surfel-reservoir.md`.
The note must include the local run date, build identity, the five compared rows, the three scenario outcomes, and one decision selected from:

```text
keep global sampling
tune stride
tune threshold
remove surfel row
```

The reason must explicitly mention `brightSurfelAccepted`, `reservoirGiSelectedBrightSurfel`, `totalMs`, and visible artifacts if any were observed.

## Follow-Up: Sparse Training Result

Local run date: 2026-05-17.
Build identity: local Debug build after sparse bright surfel training, target precheck, weighted global surfel selection, and single-frame control rows.

Key row outcomes from the in-app Sponza PT/GI audit sweep:

| Scenario | Row | Luma | selected bright | accepted bright | hit / miss | precheck reject | training store | total ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dark Courtyard | Temporal Spatial Budget 2 | 0.02832 | 0.0 | 0.0 | 0.0 / 0.0 | 0.0 | 0.0 | 60.146 |
| Dark Courtyard | Sun Receiver | 0.01663 | 0.0 | 0.0 | 0.0 / 0.0 | 0.0 | 0.0 | 57.110 |
| Dark Courtyard | Single Frame Sun Receiver | 0.01131 | 0.0 | 0.0 | 0.0 / 0.0 | 0.0 | 0.0 | 51.477 |
| Dark Courtyard | Bright Surfel | 0.01707 | 0.0 | 0.0 | 6.6 / 518357.4 | 6.6 | 69.0 | 71.958 |
| Dark Courtyard | Single Frame Bright Surfel | 0.01129 | 0.0 | 0.0 | 7.8 / 518462.6 | 7.8 | 67.5 | 65.890 |
| Sunlit Courtyard Wall | Temporal Spatial Budget 2 | 0.02414 | 0.0 | 0.0 | 0.0 / 0.0 | 0.0 | 0.0 | 173.464 |
| Sunlit Courtyard Wall | Sun Receiver | 0.04104 | 0.0 | 0.0 | 0.0 / 0.0 | 0.0 | 0.0 | 162.614 |
| Sunlit Courtyard Wall | Single Frame Sun Receiver | 0.04228 | 0.0 | 0.0 | 0.0 / 0.0 | 0.0 | 0.0 | 90.228 |
| Sunlit Courtyard Wall | Bright Surfel | 0.04122 | 603.4 | 650.4 | 298432.0 / 219984.3 | 297776.3 | 7604.6 | 197.030 |
| Sunlit Courtyard Wall | Single Frame Bright Surfel | 0.04217 | 596.5 | 640.1 | 300114.6 / 218259.5 | 299468.6 | 7592.2 | 126.672 |
| Mid-Depth Interior | Temporal Spatial Budget 2 | 0.02318 | 0.0 | 0.0 | 0.0 / 0.0 | 0.0 | 0.0 | 161.948 |
| Mid-Depth Interior | Sun Receiver | 0.03588 | 0.0 | 0.0 | 0.0 / 0.0 | 0.0 | 0.0 | 162.252 |
| Mid-Depth Interior | Single Frame Sun Receiver | 0.02466 | 0.0 | 0.0 | 0.0 / 0.0 | 0.0 | 0.0 | 75.867 |
| Mid-Depth Interior | Bright Surfel | 0.03577 | 0.0 | 0.0 | 96224.6 / 422132.5 | 96224.6 | 3979.2 | 189.882 |
| Mid-Depth Interior | Single Frame Bright Surfel | 0.02466 | 0.0 | 0.0 | 96324.5 / 422016.8 | 96324.5 | 3943.1 | 109.342 |

Outcome:

- State isolation looks correct: non-bright rows report zero bright surfel counters, and scenario transitions do not appear to carry stale surfel state.
- The sparse path reduced global attempts to roughly `518k`, down from the earlier `2,073,600` attempt shape.
- The precheck is doing real work: Mid-depth rejects about `96k` hit surfels before visibility, avoiding wasted visibility rays.
- The current surfel producer/selector is not useful in Dark Courtyard or Mid-depth: `reservoirGiSelectedBrightSurfel` remains zero there even when Mid-depth has high surfel hit counts.
- Sunlit Courtyard Wall is the only scenario with useful selected bright surfels, but the temporal-spatial bright row costs `197.030 ms` versus `162.614 ms` for Sun Receiver, and luma is essentially unchanged relative to Sun Receiver.

Decision: Continue with surfel producer quality tuning.

## Follow-Up: Target-Aware Selection Result

Local run date: 2026-05-17.
Build identity: local Debug build after target-aware bright surfel selection.

| Scenario | Row | Luma | selector reject geometry | selector reject target | selector viable | accepted bright | selected bright | total ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dark Courtyard | Sun Receiver | 0.01668 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 55.425 |
| Dark Courtyard | Bright Surfel | 0.01631 | 373533.5 | 8.8 | 0.0 | 0.0 | 0.0 | 89.953 |
| Dark Courtyard | Single Frame Sun Receiver | 0.01132 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 49.602 |
| Dark Courtyard | Single Frame Bright Surfel | 0.01127 | 376522.4 | 8.2 | 0.0 | 0.0 | 0.0 | 84.578 |
| Sunlit Courtyard Wall | Sun Receiver | 0.04113 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 158.665 |
| Sunlit Courtyard Wall | Bright Surfel | 0.04141 | 3649069.8 | 486911.0 | 662.9 | 651.8 | 604.2 | 309.831 |
| Sunlit Courtyard Wall | Single Frame Sun Receiver | 0.04231 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 87.072 |
| Sunlit Courtyard Wall | Single Frame Bright Surfel | 0.04219 | 3645879.1 | 491057.6 | 659.8 | 648.8 | 603.2 | 248.916 |
| Mid-Depth Interior | Sun Receiver | 0.03570 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 160.280 |
| Mid-Depth Interior | Bright Surfel | 0.03571 | 3928970.0 | 107097.6 | 0.0 | 0.0 | 0.0 | 282.856 |
| Mid-Depth Interior | Single Frame Sun Receiver | 0.02465 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 74.032 |
| Mid-Depth Interior | Single Frame Bright Surfel | 0.02466 | 3940466.2 | 108774.5 | 0.0 | 0.0 | 0.0 | 213.683 |

Outcome:

- The target-aware selector confirms that Dark Courtyard and Mid-Depth Interior do not have receiver-viable bright surfels in the scanned pool. Both scenarios report `brightSurfelSelectorViable=0.0` in temporal-spatial and single-frame bright rows.
- Mid-Depth still stores training surfels (`brightSurfelTrainingStore=3944.7` temporal-spatial, `3950.4` single-frame), but the selector rejects the scanned pool before visibility: about `3.93M` geometry rejects and `107k-109k` target rejects, with zero accepted bright surfels.
- Sunlit Courtyard Wall remains the only useful case: around `660` viable selector candidates, `649-652` accepted bright surfels, and `603-604` selected bright surfels.
- The cost is not acceptable. Sunlit temporal-spatial bright surfel costs `309.831 ms` versus `158.665 ms` for Sun Receiver, and Mid-Depth temporal-spatial bright surfel costs `282.856 ms` versus `160.280 ms`, without improving luma.
- The old post-selector precheck rejection is now gone as intended (`brightSurfelPrecheckRejectTarget=0.0` in bright rows); rejection has moved into selector geometry/target diagnostics.

Decision: Move to producer/index quality.
