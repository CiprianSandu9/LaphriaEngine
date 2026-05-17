# ReSTIR GI Cache Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prototype receiver-cache radiance continuation at the secondary hit so ReSTIR GI can reuse persistent indirect radiance without treating the cache as a proposal or reconnect winner.

**Architecture:** Add candidate evaluation mode `3` that evaluates the normal shadowed-sun suffix, seeds the receiver cache with that base suffix, queries a spatialized receiver cache near the secondary hit, and evaluates the combined suffix in the current primary-hit domain. Cache-continued final samples are marked non-persistent to prevent feedback loops.

**Tech Stack:** C++17 engine/UI/analysis plumbing, Slang ray generation shader, Vulkan storage buffers already bound for receiver-cache current/history, string-contract tests in `tests/PathTracerAnalysisTests.cpp`, CMake/CTest verification.

---

### Task 1: Pipeline Audit

**Files:**
- Create: `docs/superpowers/specs/2026-05-16-path-tracing-pipeline-audit.md`

- [x] **Step 1: Document current candidate flow**

Record that `sampleFirstHitReservoirGiSingleFrame()` traces local candidates and calls `makeLocalReservoirGiSample()`, and that the latter is the correct suffix-continuation integration point.

- [x] **Step 2: Document cache constraints**

Record that existing receiver-cache records can hold suffix radiance, but cache-continued final samples must not be persisted directly because reconnect already showed self-amplifying cache feedback.

### Task 2: Contract Tests

**Files:**
- Modify: `tests/PathTracerAnalysisTests.cpp`

- [x] **Step 1: Add required shader symbols**

Require these strings:

```cpp
"RESERVOIR_GI_CANDIDATE_SHADOWED_SUN_CACHE_CONTINUATION"
"tryEvaluateReservoirGiReceiverCacheContinuation"
"RESERVOIR_GI_CACHE_CONTINUATION_FLAG"
"reservoirGiReceiverCacheContinuationAttemptOffset"
"reservoirGiReceiverCacheContinuationHitOffset"
"reservoirGiReceiverCacheContinuationMissOffset"
"reservoirGiReceiverCacheContinuationAcceptedOffset"
```

- [x] **Step 2: Add required CPU/UI/sweep symbols**

Require these strings:

```cpp
"reservoirGiReceiverCacheContinuationAttempt"
"reservoirGiReceiverCacheContinuationHit"
"reservoirGiReceiverCacheContinuationMiss"
"reservoirGiReceiverCacheContinuationAccepted"
"Shadowed Sun + Cache Continuation"
"Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Sun Receiver Env First Two Cache Continuation"
```

- [x] **Step 3: Run tests and verify RED**

Run:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cmake --build cmake-build-debug --target LaphriaEngineUnitTests && ctest --test-dir cmake-build-debug --output-on-failure -R LaphriaEngineUnitTests'
```

Expected: unit test fails because the new mode/counters/row do not exist.

### Task 3: CPU and UI Plumbing

**Files:**
- Modify: `src/Core/EngineAuxiliary.h`
- Modify: `src/Core/EngineCore.h`
- Modify: `src/Core/EngineCore.cpp`
- Modify: `src/Core/UISystem.cpp`

- [x] **Step 1: Allow eval mode 3**

Change `packPathTracerMaterialSettings()` to clamp `reservoirGiCandidateEvaluationMode` to `0..3`.

- [x] **Step 2: Add UI label**

Add a candidate-evaluation combo with labels:

```cpp
"Trace Only"
"Unshadowed Sun"
"Shadowed Sun"
"Shadowed Sun + Cache Continuation"
```

- [x] **Step 3: Add continuation counters**

Add `uint32_t` fields to `PathTracerAnalysisCounters` and UI perf stats:

```cpp
reservoirGiReceiverCacheContinuationAttempt
reservoirGiReceiverCacheContinuationHit
reservoirGiReceiverCacheContinuationMiss
reservoirGiReceiverCacheContinuationAccepted
```

Mirror them as `double` fields in `PathTracerExperimentAccumulator`, copy them from mapped counters, print them in row summaries, and accumulate them during sweeps.

- [x] **Step 4: Add sweep row**

Add a Sponza row named:

```text
Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Sun Receiver Env First Two Cache Continuation
```

Use:

```cpp
reservoirGiProposalMode = MixedCosineSunReceiverGuided
environmentNeeBounceMode = 1
reservoirGiCandidateEvaluationMode = 3
```

### Task 4: Shader Prototype

**Files:**
- Modify: `src/shaders/Raygen.slang`

- [x] **Step 1: Add eval mode and counters**

Add:

```slang
static const int RESERVOIR_GI_CANDIDATE_SHADOWED_SUN_CACHE_CONTINUATION = 3;
static const uint RESERVOIR_GI_CACHE_CONTINUATION_FLAG = 32u;
```

Add four analysis counter offsets after `reservoirGiSelectedCacheReconnectOffset`.

- [x] **Step 2: Spatialize receiver-cache storage**

Add a quantized world-position hash for receiver-cache records and use it in `storeReservoirGiReceiverCacheRecord()`. Keep the existing record format.

- [x] **Step 3: Query continuation at secondary hit**

Implement:

```slang
bool tryEvaluateReservoirGiReceiverCacheContinuation(
    float3 hitPos,
    float3 N,
    out float3 cachedSuffixRadiance)
```

The function scans a small fixed number of spatial hash slots around the secondary hit, requires normal compatibility and finite positive radiance, and returns a clamped suffix radiance.

- [x] **Step 4: Integrate in `makeLocalReservoirGiSample()`**

For mode `3`, compute base suffix first, seed the receiver cache with a base record, query cache continuation, add it to the suffix, and mark the final record with `RESERVOIR_GI_CACHE_CONTINUATION_FLAG`.

- [x] **Step 5: Prevent continuation feedback**

At final persistence, require:

```slang
bool canPersistSelectedReservoirGi =
    selectedSource != RESERVOIR_GI_SOURCE_CACHE_RECONNECT &&
    (selectedRecord.flags & RESERVOIR_GI_CACHE_CONTINUATION_FLAG) == 0u;
```

### Task 5: Verification

**Files:**
- No additional files.

- [x] **Step 1: Run targeted build/tests**

Run:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cmake --build cmake-build-debug --target LaphriaEngineUnitTests LaphriaEditor && ctest --test-dir cmake-build-debug --output-on-failure -R LaphriaEngineUnitTests'
```

Expected: shader compiles, editor builds, unit tests pass.

- [x] **Step 2: Run full CTest**

Run:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && ctest --test-dir cmake-build-debug --output-on-failure'
```

Expected: all configured tests pass.

- [ ] **Step 3: Sweep success criteria**

Run the Sponza PT/GI audit sweep. Compare the new cache-continuation row against `Sun Receiver Env First Two`.

Success means:

```text
Dark Courtyard luma improves without reconnect-style cost explosion.
Sunlit Wall does not regress materially.
Mid-Depth improves or remains near the current best row.
receiverCacheContinuationHit and Accepted are nonzero.
receiverCacheStore does not balloon through feedback.
confidenceMAvg remains near the non-cache rows, not reconnect-style 20+.
```
