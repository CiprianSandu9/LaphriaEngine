# Bright Receiver Surfel Reservoir Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded bright receiver surfel reservoir so ReSTIR GI can sample discovered high-value receiver points as virtual indirect lights.

**Architecture:** Keep this separate from the receiver cache. The receiver cache remains a suffix-continuation experiment, while the new bright receiver surfel pool stores sparse world-space records from bright accepted local samples and contributes one visibility-tested virtual-light candidate into the existing ReSTIR GI combine path. This aligns with a future unified many-light path because analytic sun, environment samples, emissive triangles, and receiver surfels can all become light candidates feeding the same reservoir model.

**Tech Stack:** C++17 engine/UI/analysis plumbing, Slang ray generation shader, Vulkan storage buffers bound in the ray tracing descriptor set, string-contract tests in `tests/PathTracerAnalysisTests.cpp`, CMake/CTest verification.

---

## File Structure

- Modify `tests/PathTracerAnalysisTests.cpp`
  - Add string-contract coverage for new descriptor bindings, buffer names, record layout constants, counters, UI labels, selected-source tracking, and compact sweep rows.
  - Extend `PathTracerAnalysisCounters` offset expectations.
- Modify `src/Core/FrameContext.h`
  - Add bright receiver surfel buffer constants, buffer vectors, mapped pointers, and buffer creation declaration.
- Modify `src/Core/FrameContext.cpp`
  - Allocate, map, clear, destroy, and recreate bright receiver surfel current/history buffers.
- Modify `src/Core/PipelineCollection.cpp`
  - Extend the ray tracing descriptor set layout with storage buffer bindings 16 and 17.
- Modify `src/Core/EngineCore.h`
  - Add experiment accumulator fields for bright receiver surfel counters.
- Modify `src/Core/EngineCore.cpp`
  - Bind the new buffers, clear mapped memory at frame reset, collect counters, log row summaries, and add one compact sweep row per scenario.
- Modify `src/Core/UISystem.h`
  - Add a proposal enum value and perf-stat fields.
- Modify `src/Core/UISystem.cpp`
  - Add a proposal label and live counter readout.
- Modify `src/Core/EngineAuxiliary.h`
  - Allow proposal mode value 8 in the packed material settings.
- Modify `src/shaders/Raygen.slang`
  - Add surfel storage bindings, record layout helpers, store/load/select/evaluate functions, counters, selected-source color, proposal-mode routing, and one surfel virtual-light candidate in the ReSTIR GI combine path.

## Design Constants

Use these exact names for the first implementation slice:

```cpp
kReservoirGiBrightSurfelHeaderSize = 16
kReservoirGiBrightSurfelRecordSize = 128
kReservoirGiBrightSurfelCapacity = 65536
```

```slang
RESERVOIR_GI_BRIGHT_SURFEL_CURRENT_BINDING = 16
RESERVOIR_GI_BRIGHT_SURFEL_HISTORY_BINDING = 17
RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_BRIGHT_SURFEL = 8
RESERVOIR_GI_SOURCE_BRIGHT_SURFEL = 5u
RESERVOIR_GI_BRIGHT_SURFEL_MIN_TARGET_WEIGHT = 0.02
RESERVOIR_GI_BRIGHT_SURFEL_MIN_LUMA = 0.02
RESERVOIR_GI_BRIGHT_SURFEL_RADIUS = 0.15
RESERVOIR_GI_BRIGHT_SURFEL_COS_THETA_MAX = 0.8660254
RESERVOIR_GI_BRIGHT_SURFEL_SCAN_COUNT = 8u
```

The shader record stores:

```slang
struct ReservoirGiBrightSurfelRecord {
    float3 position;
    float3 normal;
    float3 radiance;
    float  targetWeight;
    float  confidence;
    float  radius;
    uint   frameId;
    uint   flags;
    uint   sourcePixel;
};
```

Header layout:

```text
byte 0: capacity
byte 4: frameId
byte 8: reserved
byte 12: reserved
```

Record byte layout:

```text
0   position.xyz
16  normal.xyz
32  radiance.xyz
48  targetWeight
52  confidence
56  radius
60  frameId
64  flags
68  sourcePixel
```

---

### Task 1: Contract Tests for Surfels

**Files:**
- Modify: `tests/PathTracerAnalysisTests.cpp`

- [x] **Step 1: Add required shader symbol strings**

Add these strings to the existing shader contract lists that already cover receiver-cache and ReSTIR GI symbols:

```cpp
"ptReservoirGiBrightSurfelCurrent",
"ptReservoirGiBrightSurfelHistory",
"ReservoirGiBrightSurfelRecord",
"RESERVOIR_GI_BRIGHT_SURFEL_CURRENT_BINDING",
"RESERVOIR_GI_BRIGHT_SURFEL_HISTORY_BINDING",
"RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_BRIGHT_SURFEL",
"RESERVOIR_GI_SOURCE_BRIGHT_SURFEL",
"storeReservoirGiBrightSurfelRecord",
"loadReservoirGiBrightSurfelHistoryRecord",
"selectBrightReceiverSurfelRecord",
"evaluateBrightReceiverSurfelReservoirGiCandidate",
"reservoirGiBrightSurfelStoreOffset",
"reservoirGiBrightSurfelAttemptOffset",
"reservoirGiBrightSurfelHitOffset",
"reservoirGiBrightSurfelMissOffset",
"reservoirGiBrightSurfelRejectVisibilityOffset",
"reservoirGiBrightSurfelRejectGeometryOffset",
"reservoirGiBrightSurfelRejectTargetOffset",
"reservoirGiBrightSurfelAcceptedOffset",
"reservoirGiSelectedBrightSurfelOffset"
```

- [x] **Step 2: Add required C++/UI/sweep strings**

Add these strings to the CPU/UI contract lists:

```cpp
"kReservoirGiBrightSurfelCapacity",
"reservoirGiBrightSurfelBuffers",
"reservoirGiBrightSurfelMapped",
"createReservoirGiBrightSurfelBuffers",
"MixedCosineSunReceiverBrightSurfel",
"reservoirGiBrightSurfelStore",
"reservoirGiBrightSurfelAttempt",
"reservoirGiBrightSurfelHit",
"reservoirGiBrightSurfelMiss",
"reservoirGiBrightSurfelRejectVisibility",
"reservoirGiBrightSurfelRejectGeometry",
"reservoirGiBrightSurfelRejectTarget",
"reservoirGiBrightSurfelAccepted",
"reservoirGiSelectedBrightSurfel",
"Reservoir GI Selected Bright Surfel",
"Reservoir GI Bright Surfel Accepted",
"Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Sun Receiver Bright Surfel"
```

- [x] **Step 3: Extend analysis counter offsets**

Append these entries after `reservoirGiReceiverCacheContinuationAccepted` in the counter layout test:

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

- [x] **Step 4: Run tests and verify RED**

Run:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cmake --build cmake-build-debug --target LaphriaEngineUnitTests && ctest --test-dir cmake-build-debug --output-on-failure -R LaphriaEngineUnitTests'
```

Expected: `LaphriaEngineUnitTests` fails because the surfel buffers, counters, proposal mode, and sweep row do not exist.

---

### Task 2: CPU Buffer and Descriptor Plumbing

**Files:**
- Modify: `src/Core/FrameContext.h`
- Modify: `src/Core/FrameContext.cpp`
- Modify: `src/Core/PipelineCollection.cpp`
- Modify: `src/Core/EngineCore.cpp`

- [x] **Step 1: Add surfel buffer members**

In `src/Core/FrameContext.h`, place these members after the receiver-cache members:

```cpp
static constexpr vk::DeviceSize kReservoirGiBrightSurfelHeaderSize = 16;
static constexpr vk::DeviceSize kReservoirGiBrightSurfelRecordSize = 128;
static constexpr uint32_t       kReservoirGiBrightSurfelCapacity = 65536;
vk::DeviceSize reservoirGiBrightSurfelBufferSize = kReservoirGiBrightSurfelHeaderSize;
std::vector<Laphria::VulkanUtils::VmaBuffer> reservoirGiBrightSurfelBuffers;
std::vector<void *>                          reservoirGiBrightSurfelMapped;
```

Add this private method declaration:

```cpp
void createReservoirGiBrightSurfelBuffers(const VulkanDevice &dev);
```

- [x] **Step 2: Add lifetime calls**

In `src/Core/FrameContext.cpp`, mirror the receiver-cache lifetime code with these calls:

```cpp
destroyBuffersAndReleaseAllocations(reservoirGiBrightSurfelBuffers);
reservoirGiBrightSurfelBuffers.clear();
reservoirGiBrightSurfelMapped.clear();
reservoirGiBrightSurfelBufferSize = kReservoirGiBrightSurfelHeaderSize;
createReservoirGiBrightSurfelBuffers(dev);
```

Use the same placement as the receiver-cache equivalents in destructor, `init()`, `cleanupSwapChainDependents()`, and `recreate()`.

- [x] **Step 3: Add buffer creation function**

Add this function beside `createReservoirGiReceiverCacheBuffers()`:

```cpp
void FrameContext::createReservoirGiBrightSurfelBuffers(const VulkanDevice &dev)
{
    reservoirGiBrightSurfelBuffers.clear();
    reservoirGiBrightSurfelMapped.clear();
    reservoirGiBrightSurfelBuffers.reserve(MAX_FRAMES_IN_FLIGHT);
    reservoirGiBrightSurfelMapped.reserve(MAX_FRAMES_IN_FLIGHT);
    reservoirGiBrightSurfelBufferSize =
        kReservoirGiBrightSurfelHeaderSize +
        static_cast<vk::DeviceSize>(kReservoirGiBrightSurfelCapacity) *
            kReservoirGiBrightSurfelRecordSize;

    for (uint32_t frameIndex = 0; frameIndex < MAX_FRAMES_IN_FLIGHT; ++frameIndex) {
        auto buffer = Laphria::VulkanUtils::createBuffer(
            dev,
            reservoirGiBrightSurfelBufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer,
            VMA_MEMORY_USAGE_CPU_TO_GPU,
            VMA_ALLOCATION_CREATE_MAPPED_BIT);
        reservoirGiBrightSurfelMapped.push_back(
            buffer.memory.mapMemory(0, reservoirGiBrightSurfelBufferSize));
        std::memset(reservoirGiBrightSurfelMapped.back(), 0,
                    static_cast<size_t>(reservoirGiBrightSurfelBufferSize));
        reservoirGiBrightSurfelBuffers.push_back(std::move(buffer));
    }
}
```

- [x] **Step 4: Extend ray tracing descriptor layout**

In `src/Core/PipelineCollection.cpp`, change:

```cpp
std::array<vk::DescriptorSetLayoutBinding, 14> bindings = {
```

to:

```cpp
std::array<vk::DescriptorSetLayoutBinding, 16> bindings = {
```

Append bindings 16 and 17:

```cpp
vk::DescriptorSetLayoutBinding{// 16: PT bright receiver surfel current-frame records
    .binding         = 16,
    .descriptorType  = vk::DescriptorType::eStorageBuffer,
    .descriptorCount = 1,
    .stageFlags      = vk::ShaderStageFlagBits::eRaygenKHR},
vk::DescriptorSetLayoutBinding{// 17: PT bright receiver surfel previous-frame records
    .binding         = 17,
    .descriptorType  = vk::DescriptorType::eStorageBuffer,
    .descriptorCount = 1,
    .stageFlags      = vk::ShaderStageFlagBits::eRaygenKHR}
```

- [x] **Step 5: Bind current/history surfel buffers**

In `EngineCore::createRayTracingDescriptorSets()`, add buffer infos:

```cpp
vk::DescriptorBufferInfo reservoirGiBrightSurfelCurrentInfo{
    .buffer = *frames.reservoirGiBrightSurfelBuffers[i],
    .offset = 0,
    .range  = frames.reservoirGiBrightSurfelBufferSize};
const size_t reservoirGiBrightSurfelHistoryIndex =
    frames.reservoirGiBrightSurfelBuffers.empty()
        ? i
        : (i + frames.reservoirGiBrightSurfelBuffers.size() - 1) %
              frames.reservoirGiBrightSurfelBuffers.size();
vk::DescriptorBufferInfo reservoirGiBrightSurfelHistoryInfo{
    .buffer = *frames.reservoirGiBrightSurfelBuffers[reservoirGiBrightSurfelHistoryIndex],
    .offset = 0,
    .range  = frames.reservoirGiBrightSurfelBufferSize};
```

Add descriptor writes:

```cpp
vk::WriteDescriptorSet reservoirGiBrightSurfelCurrentWrite{
    .dstSet          = *rayTracingDescriptorSets[i],
    .dstBinding      = 16,
    .dstArrayElement = 0,
    .descriptorCount = 1,
    .descriptorType  = vk::DescriptorType::eStorageBuffer,
    .pBufferInfo     = &reservoirGiBrightSurfelCurrentInfo};
vk::WriteDescriptorSet reservoirGiBrightSurfelHistoryWrite{
    .dstSet          = *rayTracingDescriptorSets[i],
    .dstBinding      = 17,
    .dstArrayElement = 0,
    .descriptorCount = 1,
    .descriptorType  = vk::DescriptorType::eStorageBuffer,
    .pBufferInfo     = &reservoirGiBrightSurfelHistoryInfo};
```

Push both writes into `descriptorWrites`.

- [x] **Step 6: Run targeted tests and verify GREEN for plumbing**

Run:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cmake --build cmake-build-debug --target LaphriaEngineUnitTests && ctest --test-dir cmake-build-debug --output-on-failure -R LaphriaEngineUnitTests'
```

Expected: tests still fail only on shader, UI, counters, and sweep strings not completed in later tasks. The project builds through C++ descriptor and buffer changes.

---

### Task 3: Counters, UI, and Sweep Row

**Files:**
- Modify: `src/Core/EngineAuxiliary.h`
- Modify: `src/Core/EngineCore.h`
- Modify: `src/Core/EngineCore.cpp`
- Modify: `src/Core/UISystem.h`
- Modify: `src/Core/UISystem.cpp`

- [x] **Step 1: Add analysis counter fields**

Append these `uint32_t` fields to `PathTracerAnalysisCounters` after `reservoirGiReceiverCacheContinuationAccepted`:

```cpp
uint32_t reservoirGiBrightSurfelStore = 0;
uint32_t reservoirGiBrightSurfelAttempt = 0;
uint32_t reservoirGiBrightSurfelHit = 0;
uint32_t reservoirGiBrightSurfelMiss = 0;
uint32_t reservoirGiBrightSurfelRejectVisibility = 0;
uint32_t reservoirGiBrightSurfelRejectGeometry = 0;
uint32_t reservoirGiBrightSurfelRejectTarget = 0;
uint32_t reservoirGiBrightSurfelAccepted = 0;
uint32_t reservoirGiSelectedBrightSurfel = 0;
```

- [x] **Step 2: Add UI enum and label**

In `UISystem::PathTracerReservoirGiProposalMode`, append:

```cpp
MixedCosineSunReceiverBrightSurfel = 8
```

In the proposal combo label array, append:

```cpp
"Mixed Cosine + Sun Receiver + Bright Surfel"
```

- [x] **Step 3: Allow proposal mode 8 in material packing**

In `packPathTracerMaterialSettings()`, change the max proposal mode to:

```cpp
const int maxMode = static_cast<int>(
    UISystem::PathTracerReservoirGiProposalMode::MixedCosineSunReceiverBrightSurfel);
```

- [x] **Step 4: Add perf-stat and experiment fields**

Add `uint32_t` UI perf fields and `double` experiment accumulator fields with these names:

```cpp
reservoirGiBrightSurfelStore
reservoirGiBrightSurfelAttempt
reservoirGiBrightSurfelHit
reservoirGiBrightSurfelMiss
reservoirGiBrightSurfelRejectVisibility
reservoirGiBrightSurfelRejectGeometry
reservoirGiBrightSurfelRejectTarget
reservoirGiBrightSurfelAccepted
reservoirGiSelectedBrightSurfel
```

In `collectPathTracerAnalysisCounters()`, copy every field from `counters` into `ui.pathTracerPerfStats`.

- [x] **Step 5: Add UI counter readout**

In the path tracer stats section of `src/Core/UISystem.cpp`, add:

```cpp
ImGui::Text("Reservoir GI Selected Bright Surfel: %u", pathTracerPerfStats.reservoirGiSelectedBrightSurfel);
ImGui::Text("Reservoir GI Bright Surfel Store: %u", pathTracerPerfStats.reservoirGiBrightSurfelStore);
ImGui::Text("Reservoir GI Bright Surfel Attempts: %u", pathTracerPerfStats.reservoirGiBrightSurfelAttempt);
ImGui::Text("Reservoir GI Bright Surfel Hits: %u", pathTracerPerfStats.reservoirGiBrightSurfelHit);
ImGui::Text("Reservoir GI Bright Surfel Misses: %u", pathTracerPerfStats.reservoirGiBrightSurfelMiss);
ImGui::Text("Reservoir GI Bright Surfel Reject Visibility: %u", pathTracerPerfStats.reservoirGiBrightSurfelRejectVisibility);
ImGui::Text("Reservoir GI Bright Surfel Reject Geometry: %u", pathTracerPerfStats.reservoirGiBrightSurfelRejectGeometry);
ImGui::Text("Reservoir GI Bright Surfel Reject Target: %u", pathTracerPerfStats.reservoirGiBrightSurfelRejectTarget);
ImGui::Text("Reservoir GI Bright Surfel Accepted: %u", pathTracerPerfStats.reservoirGiBrightSurfelAccepted);
```

- [x] **Step 6: Add experiment logging**

Add every surfel field to `PathTracerExperimentAccumulator`, accumulation, averaging, and `logPathTracerExperimentRow()` output using these exact row keys:

```text
brightSurfelStore
brightSurfelAttempt
brightSurfelHit
brightSurfelMiss
brightSurfelRejectVisibility
brightSurfelRejectGeometry
brightSurfelRejectTarget
brightSurfelAccepted
reservoirGiSelectedBrightSurfel
```

- [x] **Step 7: Add compact sweep row**

In `EngineCore::prepareSponzaPtExperimentRows()`, add one row per Sponza scenario:

```cpp
auto reservoirMixedTemporalSpatialBudget2SunReceiverBrightSurfelRow =
    makeScenarioRow(scenario,
                    "Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Sun Receiver Bright Surfel",
                    UISystem::PathTracerReservoirGiProposalMode::MixedCosineSunReceiverBrightSurfel);
reservoirMixedTemporalSpatialBudget2SunReceiverBrightSurfelRow.reservoirGiMode = 3;
reservoirMixedTemporalSpatialBudget2SunReceiverBrightSurfelRow.reservoirTemporalBudget = 2;
reservoirMixedTemporalSpatialBudget2SunReceiverBrightSurfelRow.reservoirSpatialBudget = 2;
reservoirMixedTemporalSpatialBudget2SunReceiverBrightSurfelRow.reservoirGiCandidateEvaluationMode = 2;
ptExperimentRows.push_back(reservoirMixedTemporalSpatialBudget2SunReceiverBrightSurfelRow);
```

Keep the existing compact rows and append this row after the Sun Receiver row, before cache-continuation rows. This yields five rows per scenario during this experiment.

- [x] **Step 8: Run targeted tests**

Run:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cmake --build cmake-build-debug --target LaphriaEngineUnitTests && ctest --test-dir cmake-build-debug --output-on-failure -R LaphriaEngineUnitTests'
```

Expected: tests still fail only on shader symbols until Task 4 is complete.

---

### Task 4: Shader Storage Helpers

**Files:**
- Modify: `src/shaders/Raygen.slang`

- [x] **Step 1: Add bindings and constants**

Add storage buffers after receiver-cache bindings:

```slang
[[vk::binding(16, 0)]] RWByteAddressBuffer ptReservoirGiBrightSurfelCurrent;
[[vk::binding(17, 0)]] RWByteAddressBuffer ptReservoirGiBrightSurfelHistory;
```

Add constants beside the existing ReSTIR GI constants:

```slang
static const int RESERVOIR_GI_BRIGHT_SURFEL_CURRENT_BINDING = 16;
static const int RESERVOIR_GI_BRIGHT_SURFEL_HISTORY_BINDING = 17;
static const uint RESERVOIR_GI_BRIGHT_SURFEL_CAPACITY = 65536u;
static const uint RESERVOIR_GI_BRIGHT_SURFEL_HEADER_SIZE = 16;
static const uint RESERVOIR_GI_BRIGHT_SURFEL_RECORD_SIZE = 128;
static const uint RESERVOIR_GI_BRIGHT_SURFEL_POSITION_OFFSET = 0;
static const uint RESERVOIR_GI_BRIGHT_SURFEL_NORMAL_OFFSET = 16;
static const uint RESERVOIR_GI_BRIGHT_SURFEL_RADIANCE_OFFSET = 32;
static const uint RESERVOIR_GI_BRIGHT_SURFEL_TARGET_WEIGHT_OFFSET = 48;
static const uint RESERVOIR_GI_BRIGHT_SURFEL_CONFIDENCE_OFFSET = 52;
static const uint RESERVOIR_GI_BRIGHT_SURFEL_RADIUS_OFFSET = 56;
static const uint RESERVOIR_GI_BRIGHT_SURFEL_FRAME_ID_OFFSET = 60;
static const uint RESERVOIR_GI_BRIGHT_SURFEL_FLAGS_OFFSET = 64;
static const uint RESERVOIR_GI_BRIGHT_SURFEL_SOURCE_PIXEL_OFFSET = 68;
static const int RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_BRIGHT_SURFEL = 8;
static const uint RESERVOIR_GI_SOURCE_BRIGHT_SURFEL = 5u;
static const float RESERVOIR_GI_BRIGHT_SURFEL_MIN_TARGET_WEIGHT = 0.02;
static const float RESERVOIR_GI_BRIGHT_SURFEL_MIN_LUMA = 0.02;
static const float RESERVOIR_GI_BRIGHT_SURFEL_RADIUS = 0.15;
static const float RESERVOIR_GI_BRIGHT_SURFEL_COS_THETA_MAX = 0.8660254;
static const uint RESERVOIR_GI_BRIGHT_SURFEL_SCAN_COUNT = 8u;
```

Append counter offsets:

```slang
static const uint reservoirGiBrightSurfelStoreOffset = 308u;
static const uint reservoirGiBrightSurfelAttemptOffset = 312u;
static const uint reservoirGiBrightSurfelHitOffset = 316u;
static const uint reservoirGiBrightSurfelMissOffset = 320u;
static const uint reservoirGiBrightSurfelRejectVisibilityOffset = 324u;
static const uint reservoirGiBrightSurfelRejectGeometryOffset = 328u;
static const uint reservoirGiBrightSurfelRejectTargetOffset = 332u;
static const uint reservoirGiBrightSurfelAcceptedOffset = 336u;
static const uint reservoirGiSelectedBrightSurfelOffset = 340u;
```

Also widen the shader-side proposal decode mask so mode `8` survives unpacking:

```slang
static const uint PT_MATERIAL_RESERVOIR_PROPOSAL_MASK = 0xFu;
```

At material decode, clamp `reservoirGiProposalMode` to the new bright surfel mode instead of mode `7`:

```slang
int reservoirGiProposalMode = clamp(int((packedPathTracerMaterialIndex >> PT_MATERIAL_RESERVOIR_PROPOSAL_SHIFT) &
                                        PT_MATERIAL_RESERVOIR_PROPOSAL_MASK),
                                    0,
                                    RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_BRIGHT_SURFEL);
```

- [x] **Step 2: Add record struct**

Add this struct after `ReservoirGiReceiverCacheRecord`:

```slang
struct ReservoirGiBrightSurfelRecord {
    float3 position;
    float3 normal;
    float3 radiance;
    float targetWeight;
    float confidence;
    float radius;
    uint frameId;
    uint flags;
    uint sourcePixel;
};
```

- [x] **Step 3: Add store/load helpers**

Add these helpers near receiver-cache helpers:

```slang
uint reservoirGiBrightSurfelRecordOffset(uint surfelIndex)
{
    return RESERVOIR_GI_BRIGHT_SURFEL_HEADER_SIZE +
           surfelIndex * RESERVOIR_GI_BRIGHT_SURFEL_RECORD_SIZE;
}

void storeFloat3ToReservoirGiBrightSurfel(uint offset, float3 value)
{
    ptReservoirGiBrightSurfelCurrent.Store(offset + 0, asuint(value.x));
    ptReservoirGiBrightSurfelCurrent.Store(offset + 4, asuint(value.y));
    ptReservoirGiBrightSurfelCurrent.Store(offset + 8, asuint(value.z));
}

float3 loadFloat3FromReservoirGiBrightSurfelHistory(uint offset)
{
    return float3(asfloat(ptReservoirGiBrightSurfelHistory.Load(offset + 0)),
                  asfloat(ptReservoirGiBrightSurfelHistory.Load(offset + 4)),
                  asfloat(ptReservoirGiBrightSurfelHistory.Load(offset + 8)));
}

uint reservoirGiBrightSurfelSpatialIndex(float3 position, uint slotSalt)
{
    int3 cell = int3(floor(position / RESERVOIR_GI_BRIGHT_SURFEL_RADIUS));
    uint h = uint(cell.x) * 73856093u ^ uint(cell.y) * 19349663u ^ uint(cell.z) * 83492791u;
    return (h + slotSalt * 2654435761u) % RESERVOIR_GI_BRIGHT_SURFEL_CAPACITY;
}

void updateReservoirGiBrightSurfelHeader(uint2 launchID, uint frameId)
{
    if (launchID.x == 0u && launchID.y == 0u) {
        ptReservoirGiBrightSurfelCurrent.Store(0, RESERVOIR_GI_BRIGHT_SURFEL_CAPACITY);
        ptReservoirGiBrightSurfelCurrent.Store(4, frameId);
        ptReservoirGiBrightSurfelCurrent.Store(8, 0u);
        ptReservoirGiBrightSurfelCurrent.Store(12, 0u);
    }
}
```

- [x] **Step 4: Add record load**

Add:

```slang
bool loadReservoirGiBrightSurfelHistoryRecord(uint surfelIndex,
                                              out ReservoirGiBrightSurfelRecord record)
{
    uint surfelCapacity = ptReservoirGiBrightSurfelHistory.Load(0);
    if (surfelCapacity == 0u || surfelIndex >= surfelCapacity) {
        return false;
    }

    uint offset = reservoirGiBrightSurfelRecordOffset(surfelIndex);
    record.position = loadFloat3FromReservoirGiBrightSurfelHistory(
        offset + RESERVOIR_GI_BRIGHT_SURFEL_POSITION_OFFSET);
    record.normal = loadFloat3FromReservoirGiBrightSurfelHistory(
        offset + RESERVOIR_GI_BRIGHT_SURFEL_NORMAL_OFFSET);
    record.radiance = loadFloat3FromReservoirGiBrightSurfelHistory(
        offset + RESERVOIR_GI_BRIGHT_SURFEL_RADIANCE_OFFSET);
    record.targetWeight = asfloat(ptReservoirGiBrightSurfelHistory.Load(
        offset + RESERVOIR_GI_BRIGHT_SURFEL_TARGET_WEIGHT_OFFSET));
    record.confidence = asfloat(ptReservoirGiBrightSurfelHistory.Load(
        offset + RESERVOIR_GI_BRIGHT_SURFEL_CONFIDENCE_OFFSET));
    record.radius = asfloat(ptReservoirGiBrightSurfelHistory.Load(
        offset + RESERVOIR_GI_BRIGHT_SURFEL_RADIUS_OFFSET));
    record.frameId = ptReservoirGiBrightSurfelHistory.Load(
        offset + RESERVOIR_GI_BRIGHT_SURFEL_FRAME_ID_OFFSET);
    record.flags = ptReservoirGiBrightSurfelHistory.Load(
        offset + RESERVOIR_GI_BRIGHT_SURFEL_FLAGS_OFFSET);
    record.sourcePixel = ptReservoirGiBrightSurfelHistory.Load(
        offset + RESERVOIR_GI_BRIGHT_SURFEL_SOURCE_PIXEL_OFFSET);

    if (record.frameId == 0u || !all(isfinite(record.position)) ||
        !all(isfinite(record.normal)) || !all(isfinite(record.radiance))) {
        return false;
    }
    return luminance(max(record.radiance, float3(0.0))) >= RESERVOIR_GI_BRIGHT_SURFEL_MIN_LUMA;
}
```

- [x] **Step 5: Add record store**

Add:

```slang
void storeReservoirGiBrightSurfelRecord(uint2 launchID,
                                        uint2 launchSize,
                                        ReservoirGiRecord selectedRecord)
{
    updateReservoirGiBrightSurfelHeader(launchID, ubo.frameCount);

    float radianceLuma = luminance(max(selectedRecord.suffixRadiance, float3(0.0)));
    if (selectedRecord.targetWeight < RESERVOIR_GI_BRIGHT_SURFEL_MIN_TARGET_WEIGHT ||
        radianceLuma < RESERVOIR_GI_BRIGHT_SURFEL_MIN_LUMA ||
        !all(isfinite(selectedRecord.candidatePosition)) ||
        !all(isfinite(selectedRecord.candidateNormal)) ||
        !all(isfinite(selectedRecord.suffixRadiance))) {
        return;
    }

    uint sourcePixel = launchID.y * launchSize.x + launchID.x;
    uint surfelIndex = reservoirGiBrightSurfelSpatialIndex(
        selectedRecord.candidatePosition, sourcePixel & 7u);
    uint offset = reservoirGiBrightSurfelRecordOffset(surfelIndex);
    storeFloat3ToReservoirGiBrightSurfel(
        offset + RESERVOIR_GI_BRIGHT_SURFEL_POSITION_OFFSET,
        selectedRecord.candidatePosition);
    storeFloat3ToReservoirGiBrightSurfel(
        offset + RESERVOIR_GI_BRIGHT_SURFEL_NORMAL_OFFSET,
        normalize(selectedRecord.candidateNormal));
    storeFloat3ToReservoirGiBrightSurfel(
        offset + RESERVOIR_GI_BRIGHT_SURFEL_RADIANCE_OFFSET,
        selectedRecord.suffixRadiance);
    ptReservoirGiBrightSurfelCurrent.Store(
        offset + RESERVOIR_GI_BRIGHT_SURFEL_TARGET_WEIGHT_OFFSET,
        asuint(selectedRecord.targetWeight));
    ptReservoirGiBrightSurfelCurrent.Store(
        offset + RESERVOIR_GI_BRIGHT_SURFEL_CONFIDENCE_OFFSET,
        asuint(max(selectedRecord.confidenceM, 1.0)));
    ptReservoirGiBrightSurfelCurrent.Store(
        offset + RESERVOIR_GI_BRIGHT_SURFEL_RADIUS_OFFSET,
        asuint(RESERVOIR_GI_BRIGHT_SURFEL_RADIUS));
    ptReservoirGiBrightSurfelCurrent.Store(
        offset + RESERVOIR_GI_BRIGHT_SURFEL_FRAME_ID_OFFSET,
        ubo.frameCount);
    ptReservoirGiBrightSurfelCurrent.Store(
        offset + RESERVOIR_GI_BRIGHT_SURFEL_FLAGS_OFFSET,
        selectedRecord.flags);
    ptReservoirGiBrightSurfelCurrent.Store(
        offset + RESERVOIR_GI_BRIGHT_SURFEL_SOURCE_PIXEL_OFFSET,
        sourcePixel);
    ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelStoreOffset, 1u);
}
```

- [x] **Step 6: Run shader compile target**

Run:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cmake --build cmake-build-debug --target LaphriaEditor'
```

Expected: `LaphriaEditor` shader compilation reaches the next missing function or passes if no routing code references the surfel helpers yet.

---

### Task 5: Shader Candidate Evaluation

**Files:**
- Modify: `src/shaders/Raygen.slang`

- [x] **Step 1: Add surfel selection**

Add:

```slang
bool selectBrightReceiverSurfelRecord(float3 hitPos,
                                      float3 N,
                                      uint2 launchID,
                                      uint2 launchSize,
                                      out ReservoirGiBrightSurfelRecord selectedSurfel)
{
    ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelAttemptOffset, 1u);

    uint sourcePixel = launchID.y * launchSize.x + launchID.x;
    float bestScore = 0.0;
    bool found = false;

    [unroll]
    for (uint slot = 0u; slot < RESERVOIR_GI_BRIGHT_SURFEL_SCAN_COUNT; ++slot) {
        uint surfelIndex = reservoirGiBrightSurfelSpatialIndex(hitPos, slot + (sourcePixel & 7u));
        ReservoirGiBrightSurfelRecord surfel;
        if (!loadReservoirGiBrightSurfelHistoryRecord(surfelIndex, surfel)) {
            continue;
        }

        float3 toSurfel = surfel.position - hitPos;
        float dist2 = max(dot(toSurfel, toSurfel), 1.0e-4);
        float3 wi = toSurfel * rsqrt(dist2);
        float receiverCos = dot(N, wi);
        float surfelCos = dot(surfel.normal, -wi);
        if (receiverCos <= 0.0 || surfelCos <= RESERVOIR_GI_BRIGHT_SURFEL_COS_THETA_MAX) {
            continue;
        }

        float score = luminance(max(surfel.radiance, float3(0.0))) *
                      max(receiverCos, 0.0) *
                      max(surfelCos, 0.0) /
                      dist2;
        if (score > bestScore) {
            bestScore = score;
            selectedSurfel = surfel;
            found = true;
        }
    }

    if (!found) {
        ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelMissOffset, 1u);
        return false;
    }

    ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelHitOffset, 1u);
    return true;
}
```

- [x] **Step 2: Add candidate evaluation**

Add:

```slang
bool evaluateBrightReceiverSurfelReservoirGiCandidate(float3 hitPos,
                                                      float3 N,
                                                      float3 V,
                                                      RayPayload primaryPayload,
                                                      uint2 launchID,
                                                      uint2 launchSize,
                                                      out ReservoirGiRecord candidateRecord)
{
    ReservoirGiBrightSurfelRecord surfel;
    if (!selectBrightReceiverSurfelRecord(hitPos, N, launchID, launchSize, surfel)) {
        return false;
    }

    float3 toSurfel = surfel.position - hitPos;
    float dist2 = dot(toSurfel, toSurfel);
    if (dist2 <= 1.0e-4) {
        ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelRejectGeometryOffset, 1u);
        return false;
    }

    float dist = sqrt(dist2);
    float3 wi = toSurfel / dist;
    float receiverCos = dot(N, wi);
    float surfelCos = dot(surfel.normal, -wi);
    if (receiverCos <= 0.0 || surfelCos <= RESERVOIR_GI_BRIGHT_SURFEL_COS_THETA_MAX) {
        ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelRejectGeometryOffset, 1u);
        return false;
    }

    if (!traceShadowVisibility(hitPos, N, wi, max(dist - 0.01, 0.0))) {
        ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelRejectVisibilityOffset, 1u);
        return false;
    }

    float areaProxy = PI * max(surfel.radius * surfel.radius, 1.0e-4);
    float geometry = max(receiverCos, 0.0) * max(surfelCos, 0.0) * areaProxy / max(dist2, 1.0e-4);
    float3 suffixRadiance = max(surfel.radiance, float3(0.0)) * geometry;
    ReservoirGiTargetEvaluation targetEval =
        evaluateReservoirGiTargetAtPrimary(hitPos, N, V, primaryPayload,
                                           surfel.position, surfel.normal,
                                           suffixRadiance, 1.0);
    if (!targetEval.validGeometry || !targetEval.validLight ||
        targetEval.targetWeight <= RESERVOIR_GI_BRIGHT_SURFEL_MIN_TARGET_WEIGHT) {
        ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelRejectTargetOffset, 1u);
        return false;
    }

    candidateRecord = makeInvalidReservoirGiRecord();
    candidateRecord.primaryPosition = hitPos;
    candidateRecord.primaryNormal = N;
    candidateRecord.candidatePosition = surfel.position;
    candidateRecord.candidateNormal = surfel.normal;
    candidateRecord.suffixRadiance = targetEval.suffixRadiance;
    candidateRecord.contribution = targetEval.contribution;
    candidateRecord.sourcePdf = 1.0 / float(RESERVOIR_GI_BRIGHT_SURFEL_SCAN_COUNT);
    candidateRecord.targetWeight = targetEval.targetWeight;
    candidateRecord.weightSum = targetEval.targetWeight / max(candidateRecord.sourcePdf, 1.0e-6);
    candidateRecord.selectedWeight = candidateRecord.weightSum;
    candidateRecord.confidenceM = max(surfel.confidence, 1.0);
    candidateRecord.sourcePixel = surfel.sourcePixel;
    candidateRecord.sourceFrameId = surfel.frameId;
    candidateRecord.frameId = ubo.frameCount;
    candidateRecord.flags = surfel.flags;

    ptAnalysisCounters.InterlockedAdd(reservoirGiBrightSurfelAcceptedOffset, 1u);
    return true;
}
```

- [x] **Step 3: Route proposal mode and combine candidate**

In `sampleFirstHitReservoirGiSingleFrame()`, add:

```slang
bool useBrightReceiverSurfel =
    reservoirGiProposalMode == RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_BRIGHT_SURFEL;
```

Include the new mode in the local proposal routing by treating it as the Sun Receiver mode for local samples:

```slang
int localProposalMode =
    (useReceiverCacheReconnect ?
         RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_GUIDED :
     useBrightReceiverSurfel ?
         RESERVOIR_GI_PROPOSAL_MIXED_COSINE_SUN_RECEIVER_GUIDED :
         reservoirGiProposalMode);
```

After local candidates are combined and before final source counters are recorded, add:

```slang
if (useBrightReceiverSurfel) {
    ReservoirGiRecord surfelRecord;
    if (evaluateBrightReceiverSurfelReservoirGiCandidate(hitPos, N, V, payload,
                                                        launchID, launchSize,
                                                        surfelRecord)) {
        combineReservoirGiCandidate(reservoir,
                                    surfelRecord,
                                    RESERVOIR_GI_SOURCE_BRIGHT_SURFEL,
                                    rngState);
    }
}
```

- [x] **Step 4: Track selected source and debug color**

In selected-source counters, add:

```slang
} else if (selectedSource == RESERVOIR_GI_SOURCE_BRIGHT_SURFEL) {
    ptAnalysisCounters.InterlockedAdd(reservoirGiSelectedBrightSurfelOffset, 1u);
```

In the selected-source debug AOV, color bright receiver surfels yellow:

```slang
if (selectedSource == RESERVOIR_GI_SOURCE_BRIGHT_SURFEL) {
    debugValue = float3(1.0, 0.85, 0.15);
}
```

- [x] **Step 5: Store surfels from accepted local receiver records**

At final persistence, after the receiver-cache persistence guard, add:

```slang
if (selectedSource == RESERVOIR_GI_SOURCE_LOCAL &&
    record.targetWeight >= RESERVOIR_GI_BRIGHT_SURFEL_MIN_TARGET_WEIGHT &&
    luminance(max(record.suffixRadiance, float3(0.0))) >= RESERVOIR_GI_BRIGHT_SURFEL_MIN_LUMA) {
    storeReservoirGiBrightSurfelRecord(launchID, launchSize, record);
}
```

Store only local selected records in this first slice. This prevents temporal, spatial, cache-continuation, and surfel candidates from recursively feeding the surfel pool.

- [x] **Step 6: Run shader build and unit tests**

Run:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cmake --build cmake-build-debug --target LaphriaEngineUnitTests LaphriaEditor && ctest --test-dir cmake-build-debug --output-on-failure -R LaphriaEngineUnitTests'
```

Expected: `LaphriaEditor` builds, Slang compiles `Raygen.slang`, and `LaphriaEngineUnitTests` passes.

---

### Task 6: Full Verification and Sweep Readout

**Files:**
- No additional files.

- [x] **Step 1: Run full CTest**

Run:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && ctest --test-dir cmake-build-debug --output-on-failure'
```

Expected: all configured tests pass.

- [x] **Step 2: Check whitespace**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors. Existing CRLF warnings may appear and should be reported without changing unrelated files.

- [x] **Step 3: Run the compact Sponza PT/GI audit sweep**

Run the existing in-app Sponza PT/GI audit sweep and compare these rows for each scenario:

```text
Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2
Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Sun Receiver
Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Sun Receiver Bright Surfel
Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Sun Receiver Env First Two
Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Sun Receiver Env First Two Cache Continuation
```

Use these success criteria:

```text
Dark Courtyard: Bright Surfel improves firstHitProbeAvgLuma over Sun Receiver by at least 10% with totalMs below 1.35x Sun Receiver.
Sunlit Courtyard Wall: Bright Surfel stays within 10% of Sun Receiver Env First Two luma and below 1.35x Sun Receiver cost.
Mid-Depth Interior: Bright Surfel beats the baseline Budget 2 row and does not exceed 1.35x Sun Receiver cost.
brightSurfelAccepted is nonzero in all three scenarios.
reservoirGiSelectedBrightSurfel is nonzero but less than reservoirGiSelectedLocal + reservoirGiSelectedTemporal + reservoirGiSelectedSpatial.
brightSurfelRejectVisibility is high enough to prove visibility is active.
reservoirGiConfidenceMAvg remains below 10.0 in all three scenarios.
```

Use these stop criteria:

```text
Dark Courtyard brightSurfelAccepted remains zero.
Bright Surfel luma only improves Sunlit Courtyard Wall.
totalMs exceeds 1.5x Sun Receiver in two or more scenarios.
reservoirGiSelectedBrightSurfel dominates every selected-source category.
reservoirGiConfidenceMAvg exceeds 15.0 in any scenario.
The image shows bright pinpricks, wall leaks, or frame-to-frame flashing.
```

- [x] **Step 4: Record outcome**

Append a short result note to this plan under `## Sweep Result` with:

```text
Date:
Build:
Rows compared:
Best row per scenario:
Decision: keep / tune thresholds / remove surfel row
Reason:
```

---

## Notes for the Implementer

- The first slice uses one extra surfel virtual-light candidate per pixel. It is deliberately bounded so cost is easy to compare with the current Sun Receiver and cache-continuation rows.
- The surfel pool stores only selected local receiver records in this plan. That keeps the first version from feeding reused temporal/spatial/cache values back into the virtual-light pool.
- The surfel candidate uses a conservative area proxy and a visibility ray. If it works, the next plan should replace the proxy PDF with a real reservoir-of-surfel sampling model and then add emissive triangle lights into the same abstraction.
- This is compatible with a many-light path because the new candidate is a virtual light with explicit source, target, visibility, and selection counters. The current receiver cache does not have those semantics.

## Sweep Result

Date: 2026-05-16
Build: working tree after bright receiver surfel prototype; `LaphriaEditor` target rebuilt after shader changes.
Rows compared: Budget 2, Sun Receiver, Sun Receiver Bright Surfel, Sun Receiver Env First Two, Sun Receiver Env First Two Cache Continuation.
Best row per scenario:
- Dark Courtyard: Env First Two Cache Continuation by luma; Bright Surfel only reached 0.03574 vs Sun Receiver 0.03499 and accepted no surfel candidates.
- Sunlit Courtyard Wall: Env First Two Cache Continuation by luma; Bright Surfel reached 0.06553, within 10% of Env First Two and 1.20x Sun Receiver cost, with 17671.8 accepted surfel candidates.
- Mid-Depth Interior: Env First Two Cache Continuation by luma; Bright Surfel reached 0.04374 vs Sun Receiver 0.03674 but accepted no surfel candidates.
Decision: tune thresholds / sampling, do not keep the surfel row as-is.
Reason: Bright Surfel passes the sunlit-wall quality/cost shape, but fails the required nonzero acceptance gate in Dark Courtyard and Mid-Depth Interior. The zero-accepted cases still show luma changes, so the next pass should make surfel attempts sparser and diagnose why candidate target/visibility rejection prevents accepted virtual-light reuse in darker receiver regions.

## Global Sampling Sweep Result

Date: 2026-05-17
Build: working tree after global bright surfel sampling pass; full CTest passed, `LaphriaEditor` rebuilt, and `Raygen.slang` compiled.
Rows compared: Budget 2, Sun Receiver, Sun Receiver Bright Surfel, Sun Receiver Env First Two, Sun Receiver Env First Two Cache Continuation.
Dark Courtyard: Bright Surfel reached firstHitProbeAvgLuma 0.03504 vs Sun Receiver 0.03386, but brightSurfelAccepted remained 0.0. Attempts dropped to 518399.2 from the previous full-screen 2073600 range; hits were 9.8 and all hit surfels were visibility rejected. totalMs was 75.128 vs Sun Receiver 62.570.
Sunlit Courtyard Wall: Bright Surfel reached firstHitProbeAvgLuma 0.06534 vs Sun Receiver 0.05290 and Env First Two 0.07329. brightSurfelAccepted was 652.5 and reservoirGiSelectedBrightSurfel was 607.1. totalMs was 230.011 vs Sun Receiver 169.737, slightly above the 1.35x cost target.
Mid-Depth Interior: Bright Surfel reached firstHitProbeAvgLuma 0.04387 vs Sun Receiver 0.03666, but brightSurfelAccepted remained 0.0. It had 95992.9 hits, 5713.1 visibility rejects, and 90279.8 target rejects. totalMs was 198.068 vs Sun Receiver 162.679.
Decision: tune threshold.
Reason: Global sampling reduced brightSurfelAttempt by about 4x and made the sunlit case produce real selected bright surfels, but Dark Courtyard and Mid-Depth Interior still fail the nonzero brightSurfelAccepted criterion. The dominant failure is target rejection in Mid-Depth Interior and visibility rejection or sparse valid hits in Dark Courtyard. reservoirGiSelectedBrightSurfel is controlled where accepted, totalMs is acceptable in Dark/Mid but too high in Sunlit, and no visible artifact report was provided.
