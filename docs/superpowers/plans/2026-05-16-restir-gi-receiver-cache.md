# ReSTIR GI Receiver Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an experimental sparse bright receiver cache proposal for ReSTIR GI validation in Sponza.

**Architecture:** Add a compact receiver-cache storage buffer pair at raygen bindings 14 and 15, write high-value accepted reservoir candidates, and add a new proposal mode that mixes cosine, Sun Receiver, and cache-guided cone samples. Keep existing modes and the validation preset unchanged until measurements prove the cache should become default.

**Tech Stack:** C++17/Vulkan descriptor and VMA buffer setup, Slang raygen shader, ImGui UI settings, existing C++ contract tests in `tests/PathTracerAnalysisTests.cpp`.

---

### Task 1: Contract Tests

**Files:**
- Modify: `tests/PathTracerAnalysisTests.cpp`

- [ ] **Step 1: Add expected receiver-cache contract strings**

Add required symbols for new shader bindings, proposal mode, counters, UI labels, descriptor layout bindings, frame buffers, and Sponza sweep row. The test should fail before implementation because none of the receiver-cache symbols exist.

- [ ] **Step 2: Run the unit test target and verify failure**

Run:

```powershell
cmake --build build --config Debug --target LaphriaEngineUnitTests
ctest --test-dir build -C Debug --output-on-failure -R LaphriaEngineUnitTests
```

Expected: the test fails with missing receiver-cache contract symbols.

### Task 2: CPU Buffer And Descriptor Plumbing

**Files:**
- Modify: `src/Core/FrameContext.h`
- Modify: `src/Core/FrameContext.cpp`
- Modify: `src/Core/PipelineCollection.cpp`
- Modify: `src/Core/EngineCore.cpp`

- [ ] **Step 1: Add receiver-cache frame resources**

Add a fixed-size cache buffer per frame slot, mapped for host clearing like the reservoir history buffer.

- [ ] **Step 2: Bind current/history cache buffers**

Extend the RT descriptor layout with bindings 14 and 15, then write current and previous frame cache buffers in `createRayTracingDescriptorSets`.

- [ ] **Step 3: Clear cache buffers on experiment/history reset**

Clear cache buffers in the same reset path that clears reservoir GI state.

### Task 3: UI Settings And Stats

**Files:**
- Modify: `src/Core/UISystem.h`
- Modify: `src/Core/UISystem.cpp`
- Modify: `src/Core/EngineAuxiliary.h`
- Modify: `src/Core/EngineCore.cpp`

- [ ] **Step 1: Add proposal enum and label**

Add `MixedCosineSunReceiverCacheGuided` as the next proposal mode and expose it in the Reservoir GI Proposal combo.

- [ ] **Step 2: Add analysis counters and UI text**

Add receiver-cache store/attempt/hit/miss/no-light/accepted/selected counters to the CPU counter struct, collection path, experiment accumulation, and debug UI.

### Task 4: Shader Receiver Cache

**Files:**
- Modify: `src/shaders/Raygen.slang`

- [ ] **Step 1: Add cache record layout and bindings**

Define cache capacity, record size, offsets, and binding declarations.

- [ ] **Step 2: Add cache load/store helpers**

Implement finite checks, previous-cache header validation, receiver selection from hashed slots, and high-value record storage.

- [ ] **Step 3: Add cache-guided proposal sampling**

Implement `trySampleReceiverCacheGuidedReservoirGiProposalDirection` and route the new proposal mode through a cosine/Sun Receiver/cache mixture with a matched approximate PDF.

- [ ] **Step 4: Track cache sample outcomes**

Mark cache-guided local samples, count cache-guided no-light rejects, accepted guided samples, and selected guided samples. Extend selected-source AOV cyan for cache.

### Task 5: Focused Sweep Row

**Files:**
- Modify: `src/Core/EngineCore.cpp`
- Modify: `tests/PathTracerAnalysisTests.cpp`

- [ ] **Step 1: Add a focused cache row**

Add one Sponza sweep row beside Sun Receiver named `Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Receiver Cache`.

- [ ] **Step 2: Keep validation preset unchanged**

Do not change `Load Sponza GI Validation Preset`; users should opt into the cache row or proposal mode for comparison.

### Task 6: Verification

**Files:**
- No source edits expected.

- [ ] **Step 1: Run contract tests**

Run:

```powershell
cmake --build build --config Debug --target LaphriaEngineUnitTests
ctest --test-dir build -C Debug --output-on-failure -R LaphriaEngineUnitTests
```

Expected: all unit tests pass.

- [ ] **Step 2: Build editor shader path**

Run:

```powershell
cmake --build build --config Debug --target LaphriaEditor
```

Expected: build completes successfully.
