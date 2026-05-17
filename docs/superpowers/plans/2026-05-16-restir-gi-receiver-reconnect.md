# ReSTIR GI Receiver Reconnect Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the failed cache-as-direction sweep row with a conservative receiver-cache reconnection prototype for Sponza ReSTIR GI validation.

**Architecture:** Keep Sun Receiver as the local proposal and add a separate cache reconnect candidate that selects one high-value cached receiver, traces directly to it, and evaluates that reconnected receiver in the current primary-hit domain. The old cache-guided cone proposal remains available as a manual diagnostic, but the validation sweep should use the new reconnect mode.

**Tech Stack:** C++17 engine/UI/analysis plumbing, Slang ray generation shader, string-contract unit tests in `tests/PathTracerAnalysisTests.cpp`, CMake/CTest verification.

---

### File Structure

- Modify `tests/PathTracerAnalysisTests.cpp`: add failing contract checks for proposal mode 7, reconnect counters, shader reconnect source, and the new validation sweep row.
- Modify `src/Core/UISystem.h` and `src/Core/UISystem.cpp`: expose proposal mode 7, UI labels, perf stats, and diagnostics text.
- Modify `src/Core/EngineAuxiliary.h`, `src/Core/EngineCore.h`, and `src/Core/EngineCore.cpp`: add counter fields, normalize mode 7, accumulate/log reconnect metrics, and swap the sweep row from old cone cache to reconnect.
- Modify `src/shaders/Raygen.slang`: add proposal constant 7, counter offsets, source enum 4, local-proposal fallback to Sun Receiver, cache-record selection, direct visibility reconnect evaluation, reservoir combine, selected-source AOV/counters, and config decode clamp.

### Task 1: Failing Contract Tests

**Files:**
- Modify: `tests/PathTracerAnalysisTests.cpp`

- [ ] **Step 1: Add reconnect contract strings**

Add checks that require:

```cpp
"RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_CACHE_RECONNECT"
"Reservoir GI Selected Cache Reconnect"
"receiverReconnectAttempt"
"RESERVOIR_GI_SOURCE_CACHE_RECONNECT"
"RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_CACHE_RECONNECT);"
"Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Receiver Reconnect"
```

- [ ] **Step 2: Run test and verify RED**

Run:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cmake --build cmake-build-debug --target LaphriaEngineUnitTests && ctest --test-dir cmake-build-debug --output-on-failure -R LaphriaEngineUnitTests'
```

Expected: `LaphriaEngineUnitTests` fails because the reconnect symbols and sweep row do not exist yet.

### Task 2: CPU/UI Plumbing

**Files:**
- Modify: `src/Core/UISystem.h`
- Modify: `src/Core/UISystem.cpp`
- Modify: `src/Core/EngineAuxiliary.h`
- Modify: `src/Core/EngineCore.h`
- Modify: `src/Core/EngineCore.cpp`

- [ ] **Step 1: Add proposal mode 7**

Add `MixedCosineSunReceiverCacheReconnect = 7` after the existing cache-guided mode. Update the combo labels and `normalizeReservoirGiProposalMode` to allow mode 7.

- [ ] **Step 2: Add reconnect counters**

Add these fields beside the existing receiver-cache counters:

```cpp
uint32_t reservoirGiReceiverReconnectAttempt = 0;
uint32_t reservoirGiReceiverReconnectHit = 0;
uint32_t reservoirGiReceiverReconnectMiss = 0;
uint32_t reservoirGiReceiverReconnectRejectVisibility = 0;
uint32_t reservoirGiReceiverReconnectRejectTarget = 0;
uint32_t reservoirGiReceiverReconnectAccepted = 0;
uint32_t reservoirGiSelectedCacheReconnect = 0;
```

Mirror them in `PathTracerPerfStats` and `PathTracerExperimentAccum`.

- [ ] **Step 3: Log and accumulate reconnect metrics**

Extend the row-summary format with:

```text
reservoirGiSelectedCacheReconnect=...
receiverReconnectAttempt=...
receiverReconnectHit=...
receiverReconnectMiss=...
receiverReconnectRejectVisibility=...
receiverReconnectRejectTarget=...
receiverReconnectAccepted=...
```

- [ ] **Step 4: Replace validation sweep row**

Change the Sponza row named `Receiver Cache` to `Receiver Reconnect` and use proposal mode 7. Leave the old mode 6 UI option intact for manual comparison.

### Task 3: Shader Reconnect Candidate

**Files:**
- Modify: `src/shaders/Raygen.slang`

- [ ] **Step 1: Add shader constants and offsets**

Add:

```slang
static const int RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_CACHE_RECONNECT = 7;
static const uint RESERVOIR_GI_SOURCE_CACHE_RECONNECT = 4u;
```

Add reconnect counter offsets after the current cache offsets and update the proposal decode clamp to mode 7.

- [ ] **Step 2: Keep local sampling on Sun Receiver**

When proposal mode is mode 7, set the local proposal mode to `RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_GUIDED` so reconnect is additive and does not replace the current best local proposal.

- [ ] **Step 3: Select one reconnect cache record**

Implement a selector that scans a few history cache records, requires finite positive weight, `targetWeight >= 0.05`, current-hemisphere visibility direction, and cached-normal facing compatibility, then picks the best score.

- [ ] **Step 4: Trace direct visibility and evaluate target**

Trace from the current primary hit toward the cached receiver position. Accept only when the hit lands within a small distance tolerance and normal compatibility tolerance. Evaluate with `evaluateReservoirGiTargetAtPrimary`, using cached suffix radiance and a conservative cosine proxy source PDF.

- [ ] **Step 5: Combine as a reuse candidate**

Run the reconnect candidate under a divisor-4 budget, combine accepted reconnect records with `RESERVOIR_GI_SOURCE_CACHE_RECONNECT`, and count selected reconnect separately from mode-6 cache-guided local samples.

### Task 4: Verification

**Files:**
- No additional files.

- [ ] **Step 1: Run unit/editor build and targeted tests**

Run:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cmake --build cmake-build-debug --target LaphriaEngineUnitTests LaphriaEditor && ctest --test-dir cmake-build-debug --output-on-failure -R LaphriaEngineUnitTests'
```

Expected: build succeeds and targeted tests pass.

- [ ] **Step 2: Run full CTest if the targeted pass is clean**

Run:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && ctest --test-dir cmake-build-debug --output-on-failure'
```

Expected: all configured tests pass.

- [ ] **Step 3: Manual sweep criteria**

Ask the user to run the Sponza audit sweep. For the new `Receiver Reconnect` row, compare against `Sun Receiver`:

```text
Dark Courtyard: reconnect should not collapse luma like mode 6 did.
receiverReconnectAccepted should be non-zero.
receiverReconnectRejectVisibility should explain misses instead of receiverCacheRejectNoLight dominating.
totalMs should stay closer to Sun Receiver than old mode 6, unless luma improves materially.
Selected Source should show reconnect color only where direct cache reconnection is plausible.
```

If Dark Courtyard luma remains below Sun Receiver while total time rises materially, kill reconnect or tighten cache thresholds before trying broader cache infrastructure.
