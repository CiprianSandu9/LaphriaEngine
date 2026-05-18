# Compact Surfel Cell Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the diagnostic surfel GI fixed-slot cell grid with a compact count/offset/index-list grid modeled after SurfelGI and SurfelPlus, so generated surfels survive cell population and evaluation can be meaningfully diagnosed.

**Architecture:** Rebuild the surfel cell acceleration structure every surfel frame from persistent surfel records. The new structure uses a count pass, an allocation/offset pass, and a fill pass; each valid surfel contributes to every neighboring cell its radius intersects. The first version is diagnostic-first: bounded allocation, sweep-visible counters, capped evaluation sampling, and no lighting-quality tuning until the grid behaves.

**Tech Stack:** C++17, Vulkan RAII, VMA buffers, Slang compute shaders, source-inspection unit tests in `tests/PathTracerAnalysisTests.cpp`, manual Sponza PT/GI sweep.

---

## Files

- Modify: `C:\Dev\Dizertatie\LaphriaEngine\src\Core\FrameContext.h`
  - Replace fixed cell-slot sizing constants with compact `cellToSurfel` capacity constants.
- Modify: `C:\Dev\Dizertatie\LaphriaEngine\src\Core\FrameContext.cpp`
  - Allocate/reset compact cell metadata, compact cell-to-surfel index list, and surfel counters.
- Modify: `C:\Dev\Dizertatie\LaphriaEngine\src\Core\EngineCore.h`
  - Add record methods for new surfel grid passes.
- Modify: `C:\Dev\Dizertatie\LaphriaEngine\src\Core\EngineCore.cpp`
  - Bind descriptors for renamed/repurposed buffers and record count/allocate/fill/evaluate pass order.
- Modify: `C:\Dev\Dizertatie\LaphriaEngine\src\Core\PipelineCollection.h`
  - Add pipelines for `SurfelCountCells` and `SurfelAllocateCells`; keep `SurfelBuildCells` as the fill pass or rename if the project pattern allows.
- Modify: `C:\Dev\Dizertatie\LaphriaEngine\src\Core\PipelineCollection.cpp`
  - Create/destroy the new compute pipelines.
- Modify: `C:\Dev\Dizertatie\LaphriaEngine\src\shaders\SurfelCommon.slang`
  - Change `SurfelGiCell` fields to `count`, `offset`, `writeCursor`, `overflow`; add compact-list capacity and diagnostics offsets.
- Create: `C:\Dev\Dizertatie\LaphriaEngine\src\shaders\SurfelCountCells.slang`
  - Counts all surfel-cell memberships.
- Create: `C:\Dev\Dizertatie\LaphriaEngine\src\shaders\SurfelAllocateCells.slang`
  - Assigns compact index-list offsets and resets per-cell write cursors.
- Modify: `C:\Dev\Dizertatie\LaphriaEngine\src\shaders\SurfelBuildCells.slang`
  - Converts it into the fill pass that writes surfel indices into the compact list.
- Modify: `C:\Dev\Dizertatie\LaphriaEngine\src\shaders\SurfelEvaluate.slang`
  - Iterates/samples compact cell ranges instead of fixed 16 slots.
- Modify: `C:\Dev\Dizertatie\LaphriaEngine\src\Core\EngineAuxiliary.h`
  - Add new analysis counter fields.
- Modify: `C:\Dev\Dizertatie\LaphriaEngine\src\Core\UISystem.h`
  - Add new perf stat fields.
- Modify: `C:\Dev\Dizertatie\LaphriaEngine\src\Core\UISystem.cpp`
  - Show new surfel grid diagnostics in the path tracer analysis UI.
- Modify: `C:\Dev\Dizertatie\LaphriaEngine\tests\PathTracerAnalysisTests.cpp`
  - Add source-inspection gates for the new pass order, buffer model, diagnostics, and removal of fixed-slot overflow behavior.
- Modify: `C:\Dev\Dizertatie\LaphriaEngine\CMakeLists.txt`
  - Add new shader targets and dependencies.

---

### Task 1: Define Compact Grid Data Model And Tests

**Files:**
- Modify: `tests/PathTracerAnalysisTests.cpp`
- Modify: `src/Core/FrameContext.h`
- Modify: `src/shaders/SurfelCommon.slang`
- Modify: `src/Core/EngineAuxiliary.h`
- Modify: `src/Core/UISystem.h`
- Modify: `src/Core/UISystem.cpp`

- [ ] **Step 1: Write the failing source-inspection test**

Add a block inside `testPathTracerReservoirGiMeasurementContract()` after the existing surfel GI contract checks:

```cpp
const std::array<const char *, 8> compactGridSymbols{
    "SURFEL_GI_CELL_TO_SURFEL_CAPACITY",
    "surfelGiCellTotalMembershipsOffset",
    "surfelGiCellAllocatedMembershipsOffset",
    "surfelGiCellAllocationOverflowOffset",
    "surfelGiCellNonEmptyOffset",
    "surfelGiCellMaxPopulationOffset",
    "uint offset",
    "uint writeCursor",
};
for (const char *symbol : compactGridSymbols)
{
    if (!containsText(surfelCommon, symbol) &&
        !containsText(frameContextHeader, symbol) &&
        !containsText(engineAuxiliaryHeader, symbol) &&
        !containsText(uiHeader, symbol) &&
        !containsText(uiSource, symbol))
    {
        std::cerr << "compact surfel grid missing symbol: " << symbol << "\n";
        return false;
    }
}
if (containsText(surfelCommon, "SURFEL_GI_CELL_SLOT_COUNT") ||
    containsText(frameContextHeader, "kSurfelGiCellSlotCount"))
{
    std::cerr << "compact surfel grid must not use fixed per-cell slot counts\n";
    return false;
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests'
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && ctest --test-dir cmake-build-debug -C Debug --output-on-failure -R LaphriaEngineUnitTests'
```

Expected: `LaphriaEngineUnitTests` fails with `compact surfel grid missing symbol`.

- [ ] **Step 3: Update shared shader constants and cell layout**

In `src/shaders/SurfelCommon.slang`, replace:

```hlsl
static const uint SURFEL_GI_CELL_SLOT_COUNT = 16u;
```

with:

```hlsl
static const uint SURFEL_GI_MAX_CELL_MEMBERSHIPS_PER_SURFEL = 27u;
static const uint SURFEL_GI_CELL_TO_SURFEL_CAPACITY =
    SURFEL_GI_MAX_SURFELS * SURFEL_GI_MAX_CELL_MEMBERSHIPS_PER_SURFEL;
```

Replace `SurfelGiCell` with:

```hlsl
struct SurfelGiCell
{
    uint count;
    uint offset;
    uint writeCursor;
    uint overflow;
};
```

Add diagnostics after the existing surfel GI offsets:

```hlsl
static const uint surfelGiCellTotalMembershipsOffset = 452u;
static const uint surfelGiCellAllocatedMembershipsOffset = 456u;
static const uint surfelGiCellAllocationOverflowOffset = 460u;
static const uint surfelGiCellNonEmptyOffset = 464u;
static const uint surfelGiCellMaxPopulationOffset = 468u;
```

- [ ] **Step 4: Update frame constants**

In `src/Core/FrameContext.h`, replace:

```cpp
static constexpr uint32_t       kSurfelGiCellSlotCount = 16;
static constexpr vk::DeviceSize kSurfelGiCellSlotSize = 4;
```

with:

```cpp
static constexpr uint32_t       kSurfelGiMaxCellMembershipsPerSurfel = 27;
static constexpr uint32_t       kSurfelGiCellToSurfelCapacity =
    kSurfelGiMaxSurfels * kSurfelGiMaxCellMembershipsPerSurfel;
static constexpr vk::DeviceSize kSurfelGiCellToSurfelIndexSize = 4;
```

Keep `kSurfelGiCellSize = 16` because `SurfelGiCell` remains four `uint`s.

- [ ] **Step 5: Add CPU-side diagnostic fields**

In `src/Core/EngineAuxiliary.h`, add these fields to the path tracer analysis counter/perf structures next to existing surfel GI fields:

```cpp
uint32_t surfelGiCellTotalMemberships = 0;
uint32_t surfelGiCellAllocatedMemberships = 0;
uint32_t surfelGiCellAllocationOverflow = 0;
uint32_t surfelGiCellNonEmpty = 0;
uint32_t surfelGiCellMaxPopulation = 0;
```

In `src/Core/UISystem.h`, add matching perf stat fields:

```cpp
uint32_t surfelGiCellTotalMemberships = 0;
uint32_t surfelGiCellAllocatedMemberships = 0;
uint32_t surfelGiCellAllocationOverflow = 0;
uint32_t surfelGiCellNonEmpty = 0;
uint32_t surfelGiCellMaxPopulation = 0;
```

In `src/Core/UISystem.cpp`, add visible labels beside existing surfel GI diagnostic labels:

```cpp
ImGui::Text("Surfel GI cell total memberships: %u", pathTracerPerfStats.surfelGiCellTotalMemberships);
ImGui::Text("Surfel GI cell allocated memberships: %u", pathTracerPerfStats.surfelGiCellAllocatedMemberships);
ImGui::Text("Surfel GI cell allocation overflow: %u", pathTracerPerfStats.surfelGiCellAllocationOverflow);
ImGui::Text("Surfel GI non-empty cells: %u", pathTracerPerfStats.surfelGiCellNonEmpty);
ImGui::Text("Surfel GI max cell population: %u", pathTracerPerfStats.surfelGiCellMaxPopulation);
```

- [ ] **Step 6: Run tests and commit**

Run the two unit-test commands from Step 2. Expected: PASS.

Commit:

```powershell
git add src\Core\FrameContext.h src\shaders\SurfelCommon.slang src\Core\EngineAuxiliary.h src\Core\UISystem.h src\Core\UISystem.cpp tests\PathTracerAnalysisTests.cpp
git commit -m "feat: define compact surfel grid diagnostics"
```

---

### Task 2: Allocate Compact Cell-To-Surfel Buffers

**Files:**
- Modify: `tests/PathTracerAnalysisTests.cpp`
- Modify: `src/Core/FrameContext.cpp`
- Modify: `src/Core/EngineCore.cpp`

- [ ] **Step 1: Write the failing buffer allocation test**

Add to `testPathTracerReservoirGiMeasurementContract()`:

```cpp
const std::string createSurfelBuffers =
    extractFunctionBody(frameContextSource, "void FrameContext::createSurfelGiBuffers(");
if (!containsText(createSurfelBuffers, "kSurfelGiCellToSurfelCapacity") ||
    !containsText(createSurfelBuffers, "kSurfelGiCellToSurfelIndexSize") ||
    containsText(createSurfelBuffers, "kSurfelGiCellSlotCount * kSurfelGiCellSlotSize") ||
    !containsText(engineCore, "FrameContext::kSurfelGiCellToSurfelCapacity * FrameContext::kSurfelGiCellToSurfelIndexSize"))
{
    std::cerr << "compact surfel grid buffers must allocate a global cell-to-surfel index list\n";
    return false;
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests'
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && ctest --test-dir cmake-build-debug -C Debug --output-on-failure -R LaphriaEngineUnitTests'
```

Expected: FAIL with `compact surfel grid buffers must allocate`.

- [ ] **Step 3: Update buffer sizes in `FrameContext.cpp`**

In `createSurfelGiBuffers`, compute:

```cpp
constexpr vk::DeviceSize cellToSurfelBufferSize =
    static_cast<vk::DeviceSize>(kSurfelGiCellToSurfelCapacity) * kSurfelGiCellToSurfelIndexSize;
```

Use `cellToSurfelBufferSize` when creating `surfelGiCellSlotBuffers`. Do not rename the vector in this task unless every descriptor binding is updated in the same commit.

In the clear/fill block, replace the old fixed-slot fill range with:

```cpp
cmd.fillBuffer(*buffer, 0, cellToSurfelBufferSize, 0xffffffffu);
```

- [ ] **Step 4: Update descriptor ranges in `EngineCore.cpp`**

Where the surfel cell slot buffer descriptor range is assigned, use:

```cpp
.range = FrameContext::kSurfelGiCellToSurfelCapacity *
         FrameContext::kSurfelGiCellToSurfelIndexSize
```

Do this for every descriptor write and barrier size that currently uses `kSurfelGiCellSlotCount * kSurfelGiCellSlotSize`.

- [ ] **Step 5: Run tests, editor build, and commit**

Run:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests'
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && ctest --test-dir cmake-build-debug -C Debug --output-on-failure -R LaphriaEngineUnitTests'
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEditor'
```

Expected: all pass.

Commit:

```powershell
git add src\Core\FrameContext.cpp src\Core\EngineCore.cpp tests\PathTracerAnalysisTests.cpp
git commit -m "feat: allocate compact surfel cell index list"
```

---

### Task 3: Add Count And Allocate Passes

**Files:**
- Modify: `tests/PathTracerAnalysisTests.cpp`
- Create: `src/shaders/SurfelCountCells.slang`
- Create: `src/shaders/SurfelAllocateCells.slang`
- Modify: `CMakeLists.txt`
- Modify: `src/Core/PipelineCollection.h`
- Modify: `src/Core/PipelineCollection.cpp`
- Modify: `src/Core/EngineCore.h`
- Modify: `src/Core/EngineCore.cpp`

- [ ] **Step 1: Write the failing pass-order test**

Add to `testPathTracerReservoirGiMeasurementContract()`:

```cpp
const std::string recordRayTracing =
    extractFunctionBody(engineCore, "void EngineCore::recordRayTracingCommandBuffer(");
if (!containsText(cmakeLists, "SurfelCountCells.slang|surfelCountCellsMain") ||
    !containsText(cmakeLists, "SurfelAllocateCells.slang|surfelAllocateCellsMain") ||
    !containsText(engineHeader, "recordSurfelGiCountCellsPass") ||
    !containsText(engineHeader, "recordSurfelGiAllocateCellsPass") ||
    !containsText(recordRayTracing, "recordSurfelGiCountCellsPass(commandBuffer, fi);") ||
    !containsText(recordRayTracing, "recordSurfelGiAllocateCellsPass(commandBuffer, fi);") ||
    recordRayTracing.find("recordSurfelGiCountCellsPass(commandBuffer, fi);") >
        recordRayTracing.find("recordSurfelGiAllocateCellsPass(commandBuffer, fi);") ||
    recordRayTracing.find("recordSurfelGiAllocateCellsPass(commandBuffer, fi);") >
        recordRayTracing.find("recordSurfelGiBuildCellsPass(commandBuffer, fi);"))
{
    std::cerr << "compact surfel grid must record count, allocate, then fill passes\n";
    return false;
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run the unit build and ctest commands from Task 2. Expected: FAIL with `compact surfel grid must record`.

- [ ] **Step 3: Create `SurfelCountCells.slang`**

Create `src/shaders/SurfelCountCells.slang`:

```hlsl
#include "SurfelCommon.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<SurfelGiRecord> surfelGiRecords;
[[vk::binding(1, 0)]] RWStructuredBuffer<SurfelGiCell> surfelGiCells;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> surfelGiCellToSurfel;
[[vk::binding(3, 0)]] RWByteAddressBuffer surfelGiCounters;
[[vk::binding(4, 0)]] RWByteAddressBuffer ptAnalysisCounters;
[[vk::binding(5, 0)]] RWTexture2D<float4> surfelGiDebug;
[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

bool surfelIntersectsCell(SurfelGiRecord record, int3 cell, float3 cameraPos)
{
    if (!isSurfelGiCellValid(cell))
        return false;

    float3 cellMin = (float3(cell) - float3(SURFEL_GI_GRID_DIM, SURFEL_GI_GRID_DIM, SURFEL_GI_GRID_DIM) * 0.5f) *
                     SURFEL_GI_CELL_SIZE + cameraPos;
    float3 cellMax = cellMin + float3(SURFEL_GI_CELL_SIZE, SURFEL_GI_CELL_SIZE, SURFEL_GI_CELL_SIZE);
    float3 closest = clamp(record.positionRadius.xyz, cellMin, cellMax);
    float radius = clamp(record.positionRadius.w, SURFEL_GI_MIN_RADIUS, SURFEL_GI_MAX_RADIUS);
    float3 delta = closest - record.positionRadius.xyz;
    return dot(delta, delta) <= radius * radius;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void surfelCountCellsMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint surfelIndex = dispatchThreadID.x;
    if (surfelIndex >= SURFEL_GI_MAX_SURFELS)
        return;

    SurfelGiRecord record = surfelGiRecords[surfelIndex];
    if ((record.flags & 1u) == 0u)
        return;

    const int3 centerCell = calcSurfelGiCell(record.positionRadius.xyz, ubo.cameraPos.xyz);
    for (int z = -1; z <= 1; ++z)
    for (int y = -1; y <= 1; ++y)
    for (int x = -1; x <= 1; ++x)
    {
        int3 cell = centerCell + int3(x, y, z);
        if (!surfelIntersectsCell(record, cell, ubo.cameraPos.xyz))
            continue;

        uint cellIndex = flattenSurfelGiCell(cell);
        uint previousCount;
        InterlockedAdd(surfelGiCells[cellIndex].count, 1u, previousCount);
        ptAnalysisCounters.InterlockedAdd(surfelGiCellTotalMembershipsOffset, 1u);
    }
}
```

- [ ] **Step 4: Create `SurfelAllocateCells.slang`**

Create `src/shaders/SurfelAllocateCells.slang`:

```hlsl
#include "SurfelCommon.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<SurfelGiRecord> surfelGiRecords;
[[vk::binding(1, 0)]] RWStructuredBuffer<SurfelGiCell> surfelGiCells;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> surfelGiCellToSurfel;
[[vk::binding(3, 0)]] RWByteAddressBuffer surfelGiCounters;
[[vk::binding(4, 0)]] RWByteAddressBuffer ptAnalysisCounters;
[[vk::binding(5, 0)]] RWTexture2D<float4> surfelGiDebug;
[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

[shader("compute")]
[numthreads(128, 1, 1)]
void surfelAllocateCellsMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint cellIndex = dispatchThreadID.x;
    if (cellIndex >= SURFEL_GI_CELL_COUNT)
        return;

    SurfelGiCell cell = surfelGiCells[cellIndex];
    cell.offset = 0xffffffffu;
    cell.writeCursor = 0u;
    cell.overflow = 0u;

    if (cell.count > 0u)
    {
        ptAnalysisCounters.InterlockedAdd(surfelGiCellNonEmptyOffset, 1u);

        uint previousMax;
        ptAnalysisCounters.InterlockedMax(surfelGiCellMaxPopulationOffset, cell.count, previousMax);

        uint allocatedOffset;
        surfelGiCounters.InterlockedAdd(4u, cell.count, allocatedOffset);
        if (allocatedOffset + cell.count <= SURFEL_GI_CELL_TO_SURFEL_CAPACITY)
        {
            cell.offset = allocatedOffset;
            ptAnalysisCounters.InterlockedAdd(surfelGiCellAllocatedMembershipsOffset, cell.count);
        }
        else
        {
            cell.overflow = cell.count;
            ptAnalysisCounters.InterlockedAdd(surfelGiCellAllocationOverflowOffset, cell.count);
        }
    }

    surfelGiCells[cellIndex] = cell;
}
```

- [ ] **Step 5: Add shader targets and pipelines**

In `CMakeLists.txt`, add:

```cmake
Shaders/SurfelCountCells.slang|surfelCountCellsMain
Shaders/SurfelAllocateCells.slang|surfelAllocateCellsMain
```

following the existing shader target list pattern.

In `PipelineCollection.h`, add:

```cpp
vk::raii::Pipeline surfelCountCellsPipeline{nullptr};
vk::raii::Pipeline surfelAllocateCellsPipeline{nullptr};
```

In `PipelineCollection.cpp`, create/destroy those pipelines using the same helper and descriptor layout as the existing surfel compute passes.

- [ ] **Step 6: Record pass methods and ordering**

In `EngineCore.h`, add:

```cpp
void recordSurfelGiCountCellsPass(const vk::raii::CommandBuffer &commandBuffer, uint32_t frameIndex) const;
void recordSurfelGiAllocateCellsPass(const vk::raii::CommandBuffer &commandBuffer, uint32_t frameIndex) const;
```

In `EngineCore.cpp`, implement both methods by mirroring `recordSurfelGiBuildCellsPass`, changing the bound pipeline and dispatch group count:

```cpp
constexpr uint32_t groups = (FrameContext::kSurfelGiMaxSurfels + 127u) / 128u;
commandBuffer.dispatch(groups, 1, 1);
```

for count, and:

```cpp
constexpr uint32_t groups = (FrameContext::kSurfelGiCellCount + 127u) / 128u;
commandBuffer.dispatch(groups, 1, 1);
```

for allocate.

In `recordRayTracingCommandBuffer`, order the passes:

```cpp
recordSurfelGiClearPass(commandBuffer, fi);
recordSurfelGiGeneratePass(commandBuffer, fi);
recordSurfelGiCountCellsPass(commandBuffer, fi);
recordSurfelGiAllocateCellsPass(commandBuffer, fi);
recordSurfelGiBuildCellsPass(commandBuffer, fi);
recordSurfelGiIntegratePass(commandBuffer, fi);
recordSurfelGiEvaluatePass(commandBuffer, fi);
```

- [ ] **Step 7: Run tests, editor build, and commit**

Run Task 2 verification commands. Expected: PASS.

Commit:

```powershell
git add CMakeLists.txt src\shaders\SurfelCountCells.slang src\shaders\SurfelAllocateCells.slang src\Core\PipelineCollection.h src\Core\PipelineCollection.cpp src\Core\EngineCore.h src\Core\EngineCore.cpp tests\PathTracerAnalysisTests.cpp
git commit -m "feat: add compact surfel cell count and allocation passes"
```

---

### Task 4: Fill Compact Cell Lists With Multi-Cell Membership

**Files:**
- Modify: `tests/PathTracerAnalysisTests.cpp`
- Modify: `src/shaders/SurfelBuildCells.slang`

- [ ] **Step 1: Write the failing fill-pass test**

Add:

```cpp
const std::string buildCellsMain =
    stripComments(extractFunctionBody(surfelBuildCells, "void surfelBuildCellsMain("));
if (!containsText(buildCellsMain, "for (int z = -1; z <= 1; ++z)") ||
    !containsText(buildCellsMain, "for (int y = -1; y <= 1; ++y)") ||
    !containsText(buildCellsMain, "for (int x = -1; x <= 1; ++x)") ||
    !containsText(buildCellsMain, "surfelIntersectsCell(record, cell") ||
    !containsText(buildCellsMain, "InterlockedAdd(surfelGiCells[cellIndex].writeCursor") ||
    !containsText(buildCellsMain, "surfelGiCellToSurfel[surfelGiCells[cellIndex].offset + slotIndex] = surfelIndex") ||
    containsText(buildCellsMain, "SURFEL_GI_CELL_SLOT_COUNT"))
{
    std::cerr << "compact surfel fill pass must write multi-cell memberships into offset ranges\n";
    return false;
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run unit build and ctest. Expected: FAIL with `compact surfel fill pass`.

- [ ] **Step 3: Replace `SurfelBuildCells.slang` fill logic**

Use the same `surfelIntersectsCell()` helper from `SurfelCountCells.slang`. In `surfelBuildCellsMain`, replace fixed-slot insertion with:

```hlsl
const int3 centerCell = calcSurfelGiCell(record.positionRadius.xyz, ubo.cameraPos.xyz);
for (int z = -1; z <= 1; ++z)
for (int y = -1; y <= 1; ++y)
for (int x = -1; x <= 1; ++x)
{
    int3 cell = centerCell + int3(x, y, z);
    if (!surfelIntersectsCell(record, cell, ubo.cameraPos.xyz))
        continue;

    const uint cellIndex = flattenSurfelGiCell(cell);
    if (surfelGiCells[cellIndex].offset == 0xffffffffu)
        continue;

    uint slotIndex;
    InterlockedAdd(surfelGiCells[cellIndex].writeCursor, 1u, slotIndex);
    if (slotIndex >= surfelGiCells[cellIndex].count)
    {
        InterlockedAdd(surfelGiCells[cellIndex].overflow, 1u);
        continue;
    }

    surfelGiCellToSurfel[surfelGiCells[cellIndex].offset + slotIndex] = surfelIndex;
}
record.cellIndex = flattenSurfelGiCell(centerCell);
surfelGiRecords[surfelIndex] = record;
```

- [ ] **Step 4: Run tests, editor build, and commit**

Run Task 2 verification commands. Expected: PASS.

Commit:

```powershell
git add src\shaders\SurfelBuildCells.slang tests\PathTracerAnalysisTests.cpp
git commit -m "feat: fill compact surfel cell memberships"
```

---

### Task 5: Evaluate From Compact Cell Ranges With Candidate Cap

**Files:**
- Modify: `tests/PathTracerAnalysisTests.cpp`
- Modify: `src/shaders/SurfelEvaluate.slang`

- [ ] **Step 1: Write the failing evaluation test**

Add:

```cpp
const std::string evaluateMain =
    stripComments(extractFunctionBody(surfelEvaluate, "void surfelEvaluateMain("));
if (!containsText(evaluateMain, "cell.offset == 0xffffffffu") ||
    !containsText(evaluateMain, "min(cell.count, configuredCandidateCount)") ||
    !containsText(evaluateMain, "surfelGiCellToSurfel[cell.offset +") ||
    containsText(evaluateMain, "SURFEL_GI_CELL_SLOT_COUNT"))
{
    std::cerr << "compact surfel evaluation must sample from cell offset ranges\n";
    return false;
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run unit build and ctest. Expected: FAIL with `compact surfel evaluation`.

- [ ] **Step 3: Update evaluation loop**

In `SurfelEvaluate.slang`, replace fixed-slot logic with:

```hlsl
SurfelGiCell cell = surfelGiCells[cellIndex];
if (cell.count == 0u || cell.offset == 0xffffffffu)
{
    ptAnalysisCounters.InterlockedAdd(surfelGiEvalCellEmptyOffset, 1u);
    surfelGiDebug[pixel] = float4(0.0f, 0.0f, 0.0f, 1.0f);
    return;
}

uint configuredCandidateCount = clamp(push.maxEvalCandidates, 1u, SURFEL_GI_MAX_EVAL_CANDIDATES);
uint boundedCandidateCount = min(cell.count, configuredCandidateCount);
uint accepted = 0u;
for (uint i = 0u; i < boundedCandidateCount; ++i)
{
    uint memberIndex = i;
    if (cell.count > boundedCandidateCount)
    {
        memberIndex = (i * 1664525u + pixel.x * 1013904223u + pixel.y * 747796405u + ubo.frameCount) % cell.count;
    }

    uint surfelIndex = surfelGiCellToSurfel[cell.offset + memberIndex];
    if (surfelIndex == 0xffffffffu || surfelIndex >= SURFEL_GI_MAX_SURFELS)
        continue;

    ptAnalysisCounters.InterlockedAdd(surfelGiEvalCandidatesOffset, 1u);
    SurfelGiRecord record = surfelGiRecords[surfelIndex];
    if (acceptsSurfelCandidate(record, receiverPos, receiverNormal))
        ++accepted;
}

float occupancyRatio = saturate(float(boundedCandidateCount) / max(float(configuredCandidateCount), 1.0f));
```

- [ ] **Step 4: Run tests, editor build, and commit**

Run Task 2 verification commands. Expected: PASS.

Commit:

```powershell
git add src\shaders\SurfelEvaluate.slang tests\PathTracerAnalysisTests.cpp
git commit -m "feat: evaluate surfels from compact cell ranges"
```

---

### Task 6: Wire Diagnostics Into Logging And Sweep Gates

**Files:**
- Modify: `tests/PathTracerAnalysisTests.cpp`
- Modify: `src/Core/EngineCore.cpp`
- Modify: `src/Core/EngineAuxiliary.h`
- Modify: `src/Core/UISystem.h`

- [ ] **Step 1: Write the failing logging test**

Add:

```cpp
const std::array<const char *, 5> compactGridLogFields{
    "surfelGiCellTotalMemberships=%.1f",
    "surfelGiCellAllocatedMemberships=%.1f",
    "surfelGiCellAllocationOverflow=%.1f",
    "surfelGiCellNonEmpty=%.1f",
    "surfelGiCellMaxPopulation=%.1f",
};
for (const char *field : compactGridLogFields)
{
    if (!containsText(engineCore, field))
    {
        std::cerr << "Sponza PT/GI sweep log missing compact surfel grid field: " << field << "\n";
        return false;
    }
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run unit build and ctest. Expected: FAIL with missing log field.

- [ ] **Step 3: Copy counter fields into UI stats**

In `collectPathTracerAnalysisCounters`, add:

```cpp
ui.pathTracerPerfStats.surfelGiCellTotalMemberships = counters->surfelGiCellTotalMemberships;
ui.pathTracerPerfStats.surfelGiCellAllocatedMemberships = counters->surfelGiCellAllocatedMemberships;
ui.pathTracerPerfStats.surfelGiCellAllocationOverflow = counters->surfelGiCellAllocationOverflow;
ui.pathTracerPerfStats.surfelGiCellNonEmpty = counters->surfelGiCellNonEmpty;
ui.pathTracerPerfStats.surfelGiCellMaxPopulation = counters->surfelGiCellMaxPopulation;
```

- [ ] **Step 4: Add accumulator fields and row logging**

In the experiment accumulator structure, add `double` fields:

```cpp
double surfelGiCellTotalMemberships = 0.0;
double surfelGiCellAllocatedMemberships = 0.0;
double surfelGiCellAllocationOverflow = 0.0;
double surfelGiCellNonEmpty = 0.0;
double surfelGiCellMaxPopulation = 0.0;
```

In sample accumulation, add:

```cpp
ptExperimentAccum.surfelGiCellTotalMemberships += static_cast<double>(stats.surfelGiCellTotalMemberships);
ptExperimentAccum.surfelGiCellAllocatedMemberships += static_cast<double>(stats.surfelGiCellAllocatedMemberships);
ptExperimentAccum.surfelGiCellAllocationOverflow += static_cast<double>(stats.surfelGiCellAllocationOverflow);
ptExperimentAccum.surfelGiCellNonEmpty += static_cast<double>(stats.surfelGiCellNonEmpty);
ptExperimentAccum.surfelGiCellMaxPopulation += static_cast<double>(stats.surfelGiCellMaxPopulation);
```

In the row summary format and argument list, add:

```cpp
"surfelGiCellTotalMemberships=%.1f, "
"surfelGiCellAllocatedMemberships=%.1f, "
"surfelGiCellAllocationOverflow=%.1f, "
"surfelGiCellNonEmpty=%.1f, "
"surfelGiCellMaxPopulation=%.1f, "
```

with each value multiplied by `invSamples`.

- [ ] **Step 5: Run tests, editor build, and commit**

Run Task 2 verification commands. Expected: PASS.

Commit:

```powershell
git add src\Core\EngineCore.cpp src\Core\EngineAuxiliary.h src\Core\UISystem.h tests\PathTracerAnalysisTests.cpp
git commit -m "feat: log compact surfel grid diagnostics"
```

---

### Task 7: Manual Sweep Acceptance Criteria

**Files:**
- No code changes unless the sweep identifies a compile/runtime issue.

- [ ] **Step 1: Build the editor**

Run:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build cmake-build-debug --config Debug --target LaphriaEditor'
```

Expected: `LaphriaEditor.exe` links successfully.

- [ ] **Step 2: Run the Sponza PT/GI audit sweep manually**

Use the existing in-app control that produced the previous `[INFO] PT Experiment Row Summary` logs.

Expected surfel-debug-row shape:

```text
surfelGiGenerated > 0
surfelGiCellTotalMemberships >= surfelGiGenerated
surfelGiCellAllocatedMemberships > surfelGiCellInserted from the old fixed-slot sweep
surfelGiCellAllocationOverflow == 0 or small enough to explain from capacity
surfelGiCellNonEmpty significantly above the previous 4-24 apparent cells
surfelGiEvalCandidates > 0
```

- [ ] **Step 3: Decide next action from diagnostics**

Use this decision table:

```text
If allocation overflow is high:
  Increase SURFEL_GI_CELL_TO_SURFEL_CAPACITY or reduce radius/membership count.

If non-empty cells are still tiny:
  Investigate calcSurfelGiCell coordinate mapping and reconstructed world positions.

If memberships are healthy but eval accepted is low:
  Investigate candidate radius, normal test, and receiver/surfel hemisphere tests.

If eval accepted is healthy but visual output is weak:
  Move to surfel radiance integration and actual lighting contribution.
```

- [ ] **Step 4: Commit any sweep-only fixes**

If the sweep required a small bug fix, run unit tests and editor build again, then commit:

```powershell
git add <changed-files>
git commit -m "fix: stabilize compact surfel grid diagnostics"
```

---

## Self-Review

- Spec coverage: The plan targets the observed failure mode directly: fixed-slot overflow after successful surfel generation. It mirrors SurfelGI and SurfelPlus by introducing count, offset allocation, and compact cell-to-surfel fill passes.
- Diagnostic coverage: The plan adds counters for total memberships, allocated memberships, allocation overflow, non-empty cells, and max population so the next sweep can distinguish mapping collapse, capacity exhaustion, and evaluation rejection.
- Intentional non-goals: It does not tune surfel lighting, many-light sampling, temporal stability, or stable world-space identity. Those come after the grid is no longer the bottleneck.
- Risk to watch: `SURFEL_GI_CELL_TO_SURFEL_CAPACITY = maxSurfels * 27` is diagnostic-friendly but memory-heavy. If too large for the current buffer budget, reduce to `maxSurfels * 12` and keep allocation overflow as the signal.
