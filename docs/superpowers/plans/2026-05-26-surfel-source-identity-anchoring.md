# Surfel Source Identity Anchoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every persistent surfel a stable source identity and object-space anchor so moved scene nodes update existing surfels instead of leaving stale world-space samples behind.

**Architecture:** Assign stable surfel source node IDs to scene nodes, repurpose ray tracing instance custom indices as per-frame surfel source instance IDs, add a source instance table for model/material lookup, capture per-pixel object-space source data in the GBuffer pass, copy that source data at surfel allocation, and refresh persistent surfels from source transforms during update.

**Tech Stack:** C++17, Vulkan descriptor sets, VMA buffers, Slang ray tracing and compute shaders, existing `LaphriaEngineUnitTests` contract-style tests.

---

## File Structure

Files to modify:

```text
src/Core/SurfelPathTracerResources.h
src/Core/SurfelPathTracerResources.cpp
src/Core/EngineCore.h
src/Core/EngineCore.cpp
src/Core/SurfelPathTracerPasses.h
src/Core/SurfelPathTracerPasses.cpp
src/Core/SurfelPathTracerPipelines.cpp
src/SceneManagement/SceneNode.h
src/shaders/SurfelPathTracerCommon.slang
src/shaders/SurfelPathTracerGBuffer.slang
src/shaders/SurfelPathTracerGBufferClosestHit.slang
src/shaders/SurfelPathTracerEvaluate.slang
src/shaders/SurfelPathTracerUpdate.slang
tests/SurfelPathTracerPipelineTests.cpp
```

Resources introduced:

```text
Storage binding 28: per-frame SurfelPathTracerPixelSource buffer
Storage binding 29: persistent SurfelPathTracerSource buffer, one entry per surfel
Storage binding 30: SurfelPathTracerSourceInstance table, indexed by InstanceID()
Storage binding 31: SurfelPathTracerSourceTransform table, indexed by sourceNodeId
```

The binding numbers are intentionally appended after the current highest storage binding, `27`, to avoid reshuffling existing descriptors.

## Task 1: Add Failing Source-Anchoring Contract Tests

- [ ] Open `tests/SurfelPathTracerPipelineTests.cpp`.

- [ ] Add `src/SceneManagement/SceneNode.h` to the `contractFiles` array and increase the array size by one:

```cpp
const std::array<std::filesystem::path, 33> contractFiles = {
```

```cpp
root / "src" / "SceneManagement" / "SceneNode.h",
```

- [ ] Extend the existing shader/resource contract needles so the test requires the new source data path. Add needles with these exact tokens:

```cpp
"struct SurfelPathTracerSource",
"struct SurfelPathTracerPixelSource",
"struct SurfelPathTracerSourceInstance",
"struct SurfelPathTracerSourceTransform",
"SURFEL_PT_SOURCE_FLAG_VALID",
"SURFEL_PT_SOURCE_FLAG_REFRESH_FAILED",
"sourceNodeId",
"sourceInstanceId",
"sourceInstanceCustomIndex",
"sourcePrimitiveIndex",
"sourceBarycentrics",
"sourceObjectPosition",
"sourceObjectNormal",
"surfelSourceNodeId",
"sourceTransformCount",
"makeSurfelMaterialKey",
"gBufferSourceBuffer",
"surfelSourceBuffer",
"sourceInstanceBuffer",
"sourceTransformBuffer",
"writeSurfelSource",
"refreshAnchoredSurfel",
"[[vk::binding(28, 0)]]",
"[[vk::binding(28, 1)]]",
"[[vk::binding(29, 0)]]",
"[[vk::binding(30, 1)]]",
"[[vk::binding(31, 0)]]",
"std::array<vk::DescriptorSetLayoutBinding, 32>",
```

- [ ] Add C++ source buffer contract needles for:

```cpp
"SurfelPathTracerSource source{}",
"VmaBuffer surfelSourceBuffer",
"VmaBuffer sourceInstanceBuffer",
"VmaBuffer sourceTransformBuffer",
"std::vector<VmaBuffer> gBufferSourceBuffers",
"createBuffer(byteSize(maxSurfels, sizeof(SurfelPathTracerSource))",
"createBuffer(byteSize(maxSourceInstances, sizeof(SurfelPathTracerSourceInstance))",
"createBuffer(byteSize(maxSourceTransforms, sizeof(SurfelPathTracerSourceTransform))",
```

- [ ] Run the tests and confirm they fail because the source anchoring path is not implemented yet:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests"
.\cmake-build-debug\LaphriaEngine\Debug\LaphriaEngineUnitTests.exe
```

Expected result: build succeeds, unit test binary fails on missing surfel source anchoring needles.

- [ ] Commit after the failing tests are in place:

```powershell
git add tests/SurfelPathTracerPipelineTests.cpp
git commit -m "test: require surfel source anchoring contracts"
```

## Task 2: Add Source Data Structs To C++ And Slang

- [ ] In `src/Core/SurfelPathTracerResources.h`, add CPU-side structs near `SurfelPathTracerSurfel`:

```cpp
struct SurfelPathTracerSource {
    uint32_t sourceNodeId = UINT32_MAX;
    uint32_t sourceInstanceId = UINT32_MAX;
    uint32_t sourceInstanceCustomIndex = UINT32_MAX;
    uint32_t sourcePrimitiveIndex = UINT32_MAX;
    glm::vec2 sourceBarycentrics{0.0f};
    uint32_t sourceMaterialKey = 0u;
    uint32_t sourceFlags = 0u;
    glm::vec3 sourceObjectPosition{0.0f};
    uint32_t sourceObjectNormal = 0u;
};

struct SurfelPathTracerPixelSource {
    glm::vec3 sourceObjectPosition{0.0f};
    uint32_t sourceObjectNormal = 0u;
    glm::vec2 sourceBarycentrics{0.0f};
    uint32_t sourcePrimitiveIndex = UINT32_MAX;
    uint32_t sourceInstanceId = UINT32_MAX;
    uint32_t sourceNodeId = UINT32_MAX;
    uint32_t sourceInstanceCustomIndex = UINT32_MAX;
    uint32_t sourceMaterialKey = 0u;
    uint32_t sourceFlags = 0u;
};

struct SurfelPathTracerSourceInstance {
    uint32_t sourceNodeId = UINT32_MAX;
    uint32_t modelId = 0u;
    uint32_t primitiveOffset = 0u;
    uint32_t flags = 0u;
};

struct SurfelPathTracerSourceTransform {
    glm::mat4 objectToWorld{1.0f};
    glm::mat4 worldToObject{1.0f};
    uint32_t flags = 0u;
    uint32_t padding0 = 0u;
    uint32_t padding1 = 0u;
    uint32_t padding2 = 0u;
};
```

- [ ] Add constants in the same header:

```cpp
static constexpr uint32_t SURFEL_PT_SOURCE_FLAG_VALID = 1u << 0u;
static constexpr uint32_t SURFEL_PT_SOURCE_FLAG_REFRESH_FAILED = 1u << 1u;
```

- [ ] In `src/shaders/SurfelPathTracerCommon.slang`, add matching shader structs and flags. Keep field names identical to the C++ structs:

```slang
static const uint SURFEL_PT_SOURCE_FLAG_VALID = 1u << 0u;
static const uint SURFEL_PT_SOURCE_FLAG_REFRESH_FAILED = 1u << 1u;

struct SurfelPathTracerSource {
    uint sourceNodeId;
    uint sourceInstanceId;
    uint sourceInstanceCustomIndex;
    uint sourcePrimitiveIndex;
    float2 sourceBarycentrics;
    uint sourceMaterialKey;
    uint sourceFlags;
    float3 sourceObjectPosition;
    uint sourceObjectNormal;
};

struct SurfelPathTracerPixelSource {
    float3 sourceObjectPosition;
    uint sourceObjectNormal;
    float2 sourceBarycentrics;
    uint sourcePrimitiveIndex;
    uint sourceInstanceId;
    uint sourceNodeId;
    uint sourceInstanceCustomIndex;
    uint sourceMaterialKey;
    uint sourceFlags;
};

struct SurfelPathTracerSourceInstance {
    uint sourceNodeId;
    uint modelId;
    uint primitiveOffset;
    uint flags;
};

struct SurfelPathTracerSourceTransform {
    float4x4 objectToWorld;
    float4x4 worldToObject;
    uint flags;
    uint padding0;
    uint padding1;
    uint padding2;
};
```

- [ ] Add a shared material-key helper in `src/shaders/SurfelPathTracerCommon.slang` so GBuffer and Evaluate use the same packing:

```slang
uint makeSurfelMaterialKey(uint modelId, uint materialIndex)
{
    return ((modelId & 0xFFFFu) << 16u) | (materialIndex & 0xFFFFu);
}
```

- [ ] Add static layout checks in C++ if the repo already has compile-time size checks for GPU structs. If no such helper exists, use local `static_assert` lines next to the structs:

```cpp
static_assert(sizeof(SurfelPathTracerSource) % 16u == 0u);
static_assert(sizeof(SurfelPathTracerPixelSource) % 16u == 0u);
static_assert(sizeof(SurfelPathTracerSourceInstance) == 16u);
static_assert(sizeof(SurfelPathTracerSourceTransform) % 16u == 0u);
```

- [ ] Build the unit test target:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests"
```

Expected result: C++ compiles; contract tests still fail because buffers and shaders are not wired yet.

- [ ] Commit:

```powershell
git add src/Core/SurfelPathTracerResources.h src/shaders/SurfelPathTracerCommon.slang tests/SurfelPathTracerPipelineTests.cpp
git commit -m "feat: define surfel source anchoring data"
```

## Task 3: Allocate Source Buffers

- [ ] In `src/Core/SurfelPathTracerResources.h`, add persistent buffers:

```cpp
VmaBuffer surfelSourceBuffer;
VmaBuffer sourceInstanceBuffer;
VmaBuffer sourceTransformBuffer;
```

- [ ] Add the per-frame GBuffer source buffer wherever the other per-frame GBuffer resources live:

```cpp
std::vector<VmaBuffer> gBufferSourceBuffers;
```

- [ ] Add capacity helpers to the resources class:

```cpp
uint32_t maxSourceInstances = 65536u;
uint32_t maxSourceTransforms = 65536u;
```

These values match the 24-bit Vulkan instance custom index limit with headroom for this project, while keeping the first implementation independent from scene graph container internals.

- [ ] In `src/Core/SurfelPathTracerResources.cpp`, allocate the persistent buffers in `createPersistentBuffers`:

```cpp
createBuffer(byteSize(maxSurfels, sizeof(SurfelPathTracerSource)),
             surfelSourceBuffer,
             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
             VMA_MEMORY_USAGE_GPU_ONLY,
             "SurfelPathTracer.SourceBuffer");

createBuffer(byteSize(maxSourceInstances, sizeof(SurfelPathTracerSourceInstance)),
             sourceInstanceBuffer,
             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
             VMA_MEMORY_USAGE_GPU_ONLY,
             "SurfelPathTracer.SourceInstanceBuffer");

createBuffer(byteSize(maxSourceTransforms, sizeof(SurfelPathTracerSourceTransform)),
             sourceTransformBuffer,
             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
             VMA_MEMORY_USAGE_GPU_ONLY,
             "SurfelPathTracer.SourceTransformBuffer");
```

- [ ] Allocate `gBufferSourceBuffers` alongside the other per-frame extent-dependent resources. Size it as `width * height * sizeof(SurfelPathTracerPixelSource)` and include `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT`.

- [ ] Destroy all four new buffer families in the same cleanup path as the existing surfel path tracer buffers.

- [ ] Clear `surfelSourceBuffer` and the current frame `gBufferSourceBuffer` in the reset/prepare path where dead lists, counters, or GBuffer outputs are cleared.

- [ ] Build:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests"
```

Expected result: C++ compiles; tests still fail on descriptor and shader wiring.

- [ ] Commit:

```powershell
git add src/Core/SurfelPathTracerResources.h src/Core/SurfelPathTracerResources.cpp tests/SurfelPathTracerPipelineTests.cpp
git commit -m "feat: allocate surfel source buffers"
```

## Task 4: Wire Storage Descriptor Bindings 28-31

- [ ] In `src/Core/SurfelPathTracerPipelines.cpp`, extend the storage descriptor set layout array from 28 entries to 32 entries. Bindings 28-31 are storage buffers and must be visible to compute and ray tracing shaders:

```cpp
std::array<vk::DescriptorSetLayoutBinding, 32> storageBindings = {
```

```cpp
vk::DescriptorSetLayoutBinding{.binding = 28, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = storageStages},
vk::DescriptorSetLayoutBinding{.binding = 29, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = storageStages},
vk::DescriptorSetLayoutBinding{.binding = 30, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = storageStages},
vk::DescriptorSetLayoutBinding{.binding = 31, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = storageStages},
```

The descriptor layout create info must keep using the array size:

```cpp
.bindingCount = static_cast<uint32_t>(storageBindings.size())
```

- [ ] In `src/Core/EngineCore.cpp`, increase the storage buffer descriptor pool count:

```cpp
vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 14 * MAX_FRAMES_IN_FLIGHT}
```

- [ ] Extend `createSurfelPathTracerStorageDescriptorSets` with the new buffer infos:

```cpp
VkDescriptorBufferInfo gBufferSourceInfo{
    .buffer = *surfelPathTracerResources.gBufferSourceBuffers[i],
    .offset = 0,
    .range = VK_WHOLE_SIZE,
};

VkDescriptorBufferInfo surfelSourceInfo{
    .buffer = *surfelPathTracerResources.surfelSourceBuffer,
    .offset = 0,
    .range = VK_WHOLE_SIZE,
};

VkDescriptorBufferInfo sourceInstanceInfo{
    .buffer = *surfelPathTracerResources.sourceInstanceBuffer,
    .offset = 0,
    .range = VK_WHOLE_SIZE,
};

VkDescriptorBufferInfo sourceTransformInfo{
    .buffer = *surfelPathTracerResources.sourceTransformBuffer,
    .offset = 0,
    .range = VK_WHOLE_SIZE,
};
```

- [ ] Replace the `bufferInfos` and buffer write loop with an explicit binding table:

```cpp
const std::array<vk::DescriptorBufferInfo, 14> bufferInfos = {
    vk::DescriptorBufferInfo{.buffer = *surfelPathTracerResources.countersBuffer, .offset = 0, .range = VK_WHOLE_SIZE},
    vk::DescriptorBufferInfo{.buffer = *surfelPathTracerResources.surfelBuffer, .offset = 0, .range = VK_WHOLE_SIZE},
    vk::DescriptorBufferInfo{.buffer = *surfelPathTracerResources.aliveBuffer, .offset = 0, .range = VK_WHOLE_SIZE},
    vk::DescriptorBufferInfo{.buffer = *surfelPathTracerResources.deadBuffer, .offset = 0, .range = VK_WHOLE_SIZE},
    vk::DescriptorBufferInfo{.buffer = *surfelPathTracerResources.dirtyBuffer, .offset = 0, .range = VK_WHOLE_SIZE},
    vk::DescriptorBufferInfo{.buffer = *surfelPathTracerResources.recycleBuffer, .offset = 0, .range = VK_WHOLE_SIZE},
    vk::DescriptorBufferInfo{.buffer = *surfelPathTracerResources.rayBuffer, .offset = 0, .range = VK_WHOLE_SIZE},
    vk::DescriptorBufferInfo{.buffer = *surfelPathTracerResources.cellInfoBuffer, .offset = 0, .range = VK_WHOLE_SIZE},
    vk::DescriptorBufferInfo{.buffer = *surfelPathTracerResources.cellCounterBuffer, .offset = 0, .range = VK_WHOLE_SIZE},
    vk::DescriptorBufferInfo{.buffer = *surfelPathTracerResources.cellToSurfelBuffer, .offset = 0, .range = VK_WHOLE_SIZE},
    gBufferSourceInfo,
    surfelSourceInfo,
    sourceInstanceInfo,
    sourceTransformInfo,
};

const std::array<uint32_t, 14> bufferBindings = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 28, 29, 30, 31};

for (size_t bufferIndex = 0; bufferIndex < bufferInfos.size(); ++bufferIndex)
{
    writes.push_back(vk::WriteDescriptorSet{
        .dstSet = *surfelPathTracerStorageDescriptorSets[i],
        .dstBinding = bufferBindings[bufferIndex],
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .pBufferInfo = &bufferInfos[bufferIndex]});
}
```

- [ ] Declare the storage buffers in the shader files that use them. Compute shaders see the storage set as set `0`; the GBuffer ray tracing pipeline sees the same storage descriptor set as set `1`.

In `src/shaders/SurfelPathTracerGBuffer.slang`:

```slang
[[vk::binding(28, 1)]] RWStructuredBuffer<SurfelPathTracerPixelSource> gBufferSourceBuffer;
```

In `src/shaders/SurfelPathTracerGBufferClosestHit.slang`:

```slang
[[vk::binding(30, 1)]] StructuredBuffer<SurfelPathTracerSourceInstance> sourceInstanceBuffer;
```

In `src/shaders/SurfelPathTracerEvaluate.slang`:

```slang
[[vk::binding(28, 0)]] StructuredBuffer<SurfelPathTracerPixelSource> gBufferSourceBuffer;
[[vk::binding(29, 0)]] RWStructuredBuffer<SurfelPathTracerSource> surfelSourceBuffer;
```

In `src/shaders/SurfelPathTracerUpdate.slang`:

```slang
[[vk::binding(29, 0)]] RWStructuredBuffer<SurfelPathTracerSource> surfelSourceBuffer;
[[vk::binding(31, 0)]] StructuredBuffer<SurfelPathTracerSourceTransform> sourceTransformBuffer;
```

- [ ] In `src/Core/SurfelPathTracerPasses.cpp`, extend `recordImageBarrierGBufferToCompute` with a buffer barrier for the per-frame source buffer:

```cpp
vk::BufferMemoryBarrier2 sourceBufferBarrier{
    .srcStageMask = vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
    .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
    .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
    .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead,
    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    .buffer = *resources.gBufferSourceBuffers[frameIndex],
    .offset = 0,
    .size = VK_WHOLE_SIZE,
};
```

- [ ] Attach the buffer barrier to the same dependency info as the image barriers:

```cpp
vk::DependencyInfo dependency{
    .imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
    .pImageMemoryBarriers = barriers.data(),
    .bufferMemoryBarrierCount = 1,
    .pBufferMemoryBarriers = &sourceBufferBarrier,
};
```

- [ ] Build and run the unit test binary:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests"
.\cmake-build-debug\LaphriaEngine\Debug\LaphriaEngineUnitTests.exe
```

Expected result: descriptor contract tests for bindings 28-31 pass; tests still fail on source instance upload and shader behavior.

- [ ] Commit:

```powershell
git add src/Core/SurfelPathTracerPipelines.cpp src/Core/SurfelPathTracerPasses.cpp src/Core/SurfelPathTracerPasses.h src/Core/EngineCore.cpp src/shaders/SurfelPathTracerGBuffer.slang src/shaders/SurfelPathTracerGBufferClosestHit.slang src/shaders/SurfelPathTracerEvaluate.slang src/shaders/SurfelPathTracerUpdate.slang tests/SurfelPathTracerPipelineTests.cpp
git commit -m "feat: bind surfel source storage resources"
```

## Task 5: Build Source Instance And Transform Tables From Scene Nodes

- [ ] In `src/SceneManagement/SceneNode.h`, add a stable surfel source ID field to `SceneNode`:

```cpp
#include <cstdint>
```

```cpp
uint32_t surfelSourceNodeId = UINT32_MAX;
```

- [ ] In `src/Core/EngineCore.h`, add CPU-side upload staging fields. They are `mutable` because the current TLAS build happens inside `recordCommandBuffer`, which is a `const` method:

```cpp
mutable std::vector<Laphria::SurfelPathTracerSourceInstance> surfelSourceInstances;
mutable std::vector<Laphria::SurfelPathTracerSourceTransform> surfelSourceTransforms;
mutable uint32_t currentSurfelSourceInstanceCount = 0u;
mutable uint32_t currentSurfelSourceTransformCount = 0u;
mutable uint32_t nextSurfelSourceNodeId = 0u;
```

- [ ] In `src/Core/EngineCore.cpp`, update the TLAS instance construction path that currently computes:

```cpp
uint32_t customIndex = (node->modelId << 14) | (primitiveOffset & 0x3FFF);
instance.instanceCustomIndex = customIndex;
```

- [ ] At the start of the TLAS build block, clear the per-frame source upload vectors:

```cpp
surfelSourceInstances.clear();
surfelSourceTransforms.clear();
currentSurfelSourceInstanceCount = 0u;
currentSurfelSourceTransformCount = 0u;
```

Before the mesh loop, assign one stable source node ID per scene node and write its current transform into the source transform table:

```cpp
if (node->surfelSourceNodeId == UINT32_MAX) {
    node->surfelSourceNodeId = nextSurfelSourceNodeId++;
}

const uint32_t sourceNodeId = node->surfelSourceNodeId;
if (sourceNodeId >= surfelPathTracerResources.maxSourceTransforms) {
    continue;
}

if (surfelSourceTransforms.size() <= sourceNodeId) {
    surfelSourceTransforms.resize(sourceNodeId + 1u);
}

SurfelPathTracerSourceTransform sourceTransform{};
sourceTransform.objectToWorld = node->getWorldTransform();
sourceTransform.worldToObject = glm::inverse(sourceTransform.objectToWorld);
sourceTransform.flags = SURFEL_PT_SOURCE_FLAG_VALID;
surfelSourceTransforms[sourceNodeId] = sourceTransform;
```

Inside the mesh loop, replace `customIndex` with:

```cpp
const uint32_t sourceInstanceId = static_cast<uint32_t>(surfelSourceInstances.size());

SurfelPathTracerSourceInstance sourceInstance{};
sourceInstance.sourceNodeId = sourceNodeId;
sourceInstance.modelId = node->modelId;
sourceInstance.primitiveOffset = primitiveOffset;
sourceInstance.flags = SURFEL_PT_SOURCE_FLAG_VALID;
surfelSourceInstances.push_back(sourceInstance);

instance.instanceCustomIndex = sourceInstanceId;
```

- [ ] Guard instance capacity before pushing each source instance:

```cpp
if (surfelSourceInstances.size() >= surfelPathTracerResources.maxSourceInstances) {
    continue;
}
```

- [ ] If a node contributes multiple BLAS instances, reuse the stable `sourceNodeId` created before the mesh loop and create one `sourceInstanceId` per BLAS instance. The source transform table represents scene-node transforms; the source instance table represents ray tracing instances.

- [ ] After TLAS construction, upload both vectors into `sourceInstanceBuffer` and `sourceTransformBuffer` using the same staging helper used elsewhere for GPU buffer uploads. Upload only `vector.size() * sizeof(T)`.

- [ ] Store counts:

```cpp
currentSurfelSourceInstanceCount = static_cast<uint32_t>(surfelSourceInstances.size());
currentSurfelSourceTransformCount = static_cast<uint32_t>(surfelSourceTransforms.size());
```

- [ ] Ensure the vectors are rebuilt every frame before GBuffer recording so scene-node animation updates the transform table even when the set of renderable nodes is unchanged.

- [ ] Build:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests"
```

Expected result: compile succeeds; tests still fail on GBuffer/Evaluate/Update shader behavior.

- [ ] Commit:

```powershell
git add src/SceneManagement/SceneNode.h src/Core/EngineCore.h src/Core/EngineCore.cpp tests/SurfelPathTracerPipelineTests.cpp
git commit -m "feat: upload surfel source instance tables"
```

## Task 6: Decode Materials Through SourceInstanceBuffer In Closest Hit

- [ ] In `src/shaders/SurfelPathTracerGBufferClosestHit.slang`, change `decodeGBufferSurface` so `InstanceID()` is interpreted as a source instance ID, not packed model plus primitive offset.

- [ ] Extend `GBufferSurface` with the source fields returned by `decodeGBufferSurface`:

```slang
uint sourceNodeId;
uint sourceInstanceId;
uint sourceInstanceCustomIndex;
uint sourcePrimitiveIndex;
float2 sourceBarycentrics;
float3 sourceObjectPosition;
float3 sourceObjectNormal;
```

- [ ] Add this lookup at the top of `decodeGBufferSurface`:

```slang
uint sourceInstanceId = instanceId;
SurfelPathTracerSourceInstance sourceInstance = sourceInstanceBuffer[sourceInstanceId];
uint modelId = sourceInstance.modelId;
uint primitiveOffset = sourceInstance.primitiveOffset;
uint materialIndex = primitiveOffset + geometryIndex;
```

- [ ] Keep `sourceInstanceCustomIndex` in the payload equal to the raw `InstanceID()` value:

```slang
surface.sourceInstanceId = sourceInstanceId;
surface.sourceNodeId = sourceInstance.sourceNodeId;
surface.sourceInstanceCustomIndex = instanceId;
surface.sourcePrimitiveIndex = primitiveIndex;
surface.sourceBarycentrics = barycentrics;
```

- [ ] Compute object-space anchor data from the mesh vertices before object-to-world transformation:

```slang
float3 objectPosition =
    v0.position * (1.0f - barycentrics.x - barycentrics.y) +
    v1.position * barycentrics.x +
    v2.position * barycentrics.y;

float3 objectNormal = normalize(
    n0 * (1.0f - barycentrics.x - barycentrics.y) +
    n1 * barycentrics.x +
    n2 * barycentrics.y);

surface.sourceObjectPosition = objectPosition;
surface.sourceObjectNormal = objectNormal;
```

- [ ] Copy the source fields from `GBufferSurface` into the closest-hit payload in `main`:

```slang
payload.sourceNodeId = surface.sourceNodeId;
payload.sourceInstanceId = surface.sourceInstanceId;
payload.sourceInstanceCustomIndex = surface.sourceInstanceCustomIndex;
payload.sourcePrimitiveIndex = surface.sourcePrimitiveIndex;
payload.sourceBarycentrics = surface.sourceBarycentrics;
payload.sourceObjectPosition = surface.sourceObjectPosition;
payload.sourceObjectNormal = surface.sourceObjectNormal;
```

- [ ] Keep existing world-space shading behavior unchanged except for replacing the old model/material decode.

- [ ] Build:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests"
```

Expected result: shader compilation succeeds; tests still fail until source data is written and consumed.

- [ ] Commit:

```powershell
git add src/shaders/SurfelPathTracerGBufferClosestHit.slang tests/SurfelPathTracerPipelineTests.cpp
git commit -m "feat: decode surfel source instances in gbuffer"
```

## Task 7: Write Per-Pixel Source Data In GBuffer

- [ ] In both `src/shaders/SurfelPathTracerGBuffer.slang` and `src/shaders/SurfelPathTracerGBufferClosestHit.slang`, extend `SurfelPathTracerGBufferPayload` with:

```slang
uint sourceNodeId;
uint sourceInstanceId;
uint sourceInstanceCustomIndex;
uint sourcePrimitiveIndex;
float2 sourceBarycentrics;
float3 sourceObjectPosition;
float3 sourceObjectNormal;
```

- [ ] Add a helper in `SurfelPathTracerGBuffer.slang`:

```slang
void writeSurfelSource(uint2 pixel, SurfelPathTracerGBufferPayload payload)
{
    uint pixelIndex = pixel.y * uint(gFrame.framebufferSize.x) + pixel.x;

    SurfelPathTracerPixelSource source;
    source.sourceObjectPosition = payload.sourceObjectPosition;
    source.sourceObjectNormal = packNormalOctahedral(payload.sourceObjectNormal);
    source.sourceBarycentrics = payload.sourceBarycentrics;
    source.sourcePrimitiveIndex = payload.sourcePrimitiveIndex;
    source.sourceInstanceId = payload.sourceInstanceId;
    source.sourceNodeId = payload.sourceNodeId;
    source.sourceInstanceCustomIndex = payload.sourceInstanceCustomIndex;
    source.sourceMaterialKey = makeSurfelMaterialKey(payload.modelId, payload.materialIndex);
    source.sourceFlags = SURFEL_PT_SOURCE_FLAG_VALID;

    gBufferSourceBuffer[pixelIndex] = source;
}
```

- [ ] Call `writeSurfelSource(pixel, payload)` on every hit path after the payload is populated.

- [ ] Clear the source entry on miss:

```slang
SurfelPathTracerPixelSource source;
source.sourceFlags = 0u;
gBufferSourceBuffer[pixelIndex] = source;
```

- [ ] Keep the existing motion/material image output intact for denoising and current material lookup behavior.

- [ ] Build and run tests:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests"
.\cmake-build-debug\LaphriaEngine\Debug\LaphriaEngineUnitTests.exe
```

Expected result: source write contracts pass; tests still fail until Evaluate and Update consume the source data.

- [ ] Commit:

```powershell
git add src/shaders/SurfelPathTracerGBuffer.slang tests/SurfelPathTracerPipelineTests.cpp
git commit -m "feat: write gbuffer surfel source data"
```

## Task 8: Copy Pixel Source Data When Allocating Surfels

- [ ] In `src/shaders/SurfelPathTracerEvaluate.slang`, locate the surfel allocation path that writes a new `SurfelPathTracerSurfel`.

- [ ] Add a helper:

```slang
void writeSurfelSource(uint surfelIndex, uint2 pixel, uint materialKey)
{
    uint pixelIndex = pixel.y * uint(gFrame.framebufferSize.x) + pixel.x;
    SurfelPathTracerPixelSource pixelSource = gBufferSourceBuffer[pixelIndex];

    SurfelPathTracerSource source;
    source.sourceNodeId = pixelSource.sourceNodeId;
    source.sourceInstanceId = pixelSource.sourceInstanceId;
    source.sourceInstanceCustomIndex = pixelSource.sourceInstanceCustomIndex;
    source.sourcePrimitiveIndex = pixelSource.sourcePrimitiveIndex;
    source.sourceBarycentrics = pixelSource.sourceBarycentrics;
    source.sourceMaterialKey = materialKey;
    source.sourceFlags = pixelSource.sourceFlags;
    source.sourceObjectPosition = pixelSource.sourceObjectPosition;
    source.sourceObjectNormal = pixelSource.sourceObjectNormal;

    surfelSourceBuffer[surfelIndex] = source;
}
```

- [ ] Call `writeSurfelSource(newSurfelIndex, pixel, surfel.materialKey)` immediately after writing the new surfel record.

- [ ] If `pixelSource.sourceFlags` is invalid, still allocate the surfel using existing behavior but store `sourceFlags = 0u`. This preserves current fallback behavior for non-triangle or invalid-hit paths.

- [ ] Build and run tests:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests"
.\cmake-build-debug\LaphriaEngine\Debug\LaphriaEngineUnitTests.exe
```

Expected result: allocation source-copy contracts pass; tests still fail until Update refreshes anchored surfels.

- [ ] Commit:

```powershell
git add src/shaders/SurfelPathTracerEvaluate.slang tests/SurfelPathTracerPipelineTests.cpp
git commit -m "feat: persist surfel source anchors"
```

## Task 9: Refresh Anchored Surfels In Update

- [ ] In `src/Core/SurfelPathTracerPasses.h`, add `uint32_t sourceTransformCount` to the `recordUpdatePass` signature after `uint32_t cellDimension`.

- [ ] In `src/Core/SurfelPathTracerPasses.cpp`, add the same field to the C++ push constant struct:

```cpp
struct SurfelUpdatePushConstants
{
    uint32_t maxSurfels = 0;
    float cellSize = 1.0f;
    uint32_t cellDimension = 1;
    uint32_t maxRays = 0;
    uint32_t lockSurfels = 0;
    uint32_t minRaysPerSurfel = 1;
    uint32_t maxRaysPerSurfel = 1;
    float varianceSensitivity = 1.0f;
    uint32_t sourceTransformCount = 0u;
    uint32_t pad0 = 0u;
    uint32_t pad1 = 0u;
    uint32_t pad2 = 0u;
};
```

- [ ] Initialize the new push constant field in `recordUpdatePass`:

```cpp
.sourceTransformCount = sourceTransformCount,
```

- [ ] In `src/Core/EngineCore.cpp`, pass `currentSurfelSourceTransformCount` when recording the update pass.

- [ ] In `src/shaders/SurfelPathTracerUpdate.slang`, extend the shader push constant struct to match C++:

```slang
struct UpdatePushConstants {
    uint maxSurfels;
    float cellSize;
    uint cellDimension;
    uint maxRays;
    uint lockSurfels;
    uint minRaysPerSurfel;
    uint maxRaysPerSurfel;
    float varianceSensitivity;
    uint sourceTransformCount;
    uint pad0;
    uint pad1;
    uint pad2;
};
```

- [ ] In `src/shaders/SurfelPathTracerUpdate.slang`, add a helper:

```slang
bool refreshAnchoredSurfel(uint surfelIndex, inout SurfelPathTracerSurfel surfel)
{
    SurfelPathTracerSource source = surfelSourceBuffer[surfelIndex];
    if ((source.sourceFlags & SURFEL_PT_SOURCE_FLAG_VALID) == 0u) {
        return false;
    }

    if (source.sourceNodeId >= push.sourceTransformCount) {
        source.sourceFlags |= SURFEL_PT_SOURCE_FLAG_REFRESH_FAILED;
        surfelSourceBuffer[surfelIndex] = source;
        return false;
    }

    SurfelPathTracerSourceTransform transform = sourceTransformBuffer[source.sourceNodeId];
    if ((transform.flags & SURFEL_PT_SOURCE_FLAG_VALID) == 0u) {
        source.sourceFlags |= SURFEL_PT_SOURCE_FLAG_REFRESH_FAILED;
        surfelSourceBuffer[surfelIndex] = source;
        return false;
    }

    float4 worldPosition = mul(transform.objectToWorld, float4(source.sourceObjectPosition, 1.0f));
    float3 objectNormal = unpackNormalOctahedral(source.sourceObjectNormal);
    float3 worldNormal = normalize(mul(transpose((float3x3)transform.worldToObject), objectNormal));

    surfel.position = worldPosition.xyz;
    surfel.packedNormal = packNormalOctahedral(worldNormal);
    surfel.materialKey = source.sourceMaterialKey;
    return true;
}
```

- [ ] Call `refreshAnchoredSurfel(surfelIndex, surfel)` before visibility, recycle, or sleep decisions that use `surfel.position` and `surfel.packedNormal`.

- [ ] If refresh fails, keep the existing world-space update path active. This maintains compatibility with surfels allocated before the source anchor path existed and any invalid source records.

- [ ] Keep the `source.sourceNodeId >= push.sourceTransformCount` check before every `sourceTransformBuffer[source.sourceNodeId]` read.

- [ ] Build and run tests:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests"
.\cmake-build-debug\LaphriaEngine\Debug\LaphriaEngineUnitTests.exe
```

Expected result: all source anchoring contract tests pass.

- [ ] Commit:

```powershell
git add src/Core/SurfelPathTracerPasses.h src/Core/SurfelPathTracerPasses.cpp src/Core/EngineCore.cpp src/shaders/SurfelPathTracerUpdate.slang tests/SurfelPathTracerPipelineTests.cpp
git commit -m "feat: refresh surfels from source anchors"
```

## Task 10: Add CPU Regression Coverage For Descriptor And Buffer Counts

- [ ] In `tests/SurfelPathTracerPipelineTests.cpp`, add explicit checks that:

```cpp
containsAllNeedles(combined, {
    "maxSourceInstances",
    "maxSourceTransforms",
    "SurfelPathTracer.SourceBuffer",
    "SurfelPathTracer.SourceInstanceBuffer",
    "SurfelPathTracer.SourceTransformBuffer",
    "std::array<vk::DescriptorSetLayoutBinding, 32>",
    "currentSurfelSourceInstanceCount",
    "currentSurfelSourceTransformCount",
    "surfelSourceNodeId",
    "source.sourceNodeId >= push.sourceTransformCount",
});
```

- [ ] Run:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests"
.\cmake-build-debug\LaphriaEngine\Debug\LaphriaEngineUnitTests.exe
```

Expected result: unit tests pass.

- [ ] Commit:

```powershell
git add tests/SurfelPathTracerPipelineTests.cpp
git commit -m "test: cover surfel source anchoring resources"
```

## Task 11: Run Runtime Validation

- [ ] Build the editor target:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build cmake-build-debug --config Debug --target LaphriaEditor"
```

- [ ] Launch the editor through the existing project workflow.

- [ ] Validate these scenes or equivalent local fixtures:

```text
Static mesh scene with surfel GI enabled
Single translated mesh node
Single rotating mesh node
Multiple instances of the same model with different transforms
Camera movement without scene-node movement
Surfel reset/rebuild after toggling the surfel path tracer
```

- [ ] In a rotating or translating scene-node test, confirm that:

```text
Existing surfel positions follow the moving node.
No stale world-space ghosting remains after several frames.
Material appearance still matches the original GBuffer result.
Non-moving scenes do not show visible quality regressions.
```

- [ ] Capture before/after screenshots or notes in the final implementation summary.

- [ ] Commit any final fixes:

```powershell
git add src tests
git commit -m "fix: stabilize surfel source anchoring"
```

## Task 12: Final Verification And Handoff

- [ ] Run the final verification commands:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build cmake-build-debug --config Debug --target LaphriaEngineUnitTests"
.\cmake-build-debug\LaphriaEngine\Debug\LaphriaEngineUnitTests.exe
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build cmake-build-debug --config Debug --target LaphriaEditor"
```

- [ ] Inspect git status:

```powershell
git status --short
```

- [ ] Ensure remaining dirty files are either intentional implementation changes or unrelated pre-existing user changes.

- [ ] Final response should include:

```text
Summary of source anchoring architecture implemented.
Tests and build commands run with pass/fail status.
Any runtime validation completed.
Known residual risks, especially skinned/deformed geometry not covered by scene-node transform anchoring.
Commit hashes produced during implementation.
```

## Risk Notes

- This plan supports true scene-node animation by refreshing surfels through the current scene-node transform table. It does not solve skinned, morphed, or CPU-deformed mesh anchoring.
- Changing `instance.instanceCustomIndex` is central to the architecture. The source instance table must replace every shader use that previously decoded model and primitive offset directly from `InstanceID()`.
- Binding 28 is per-frame because the GBuffer source data is resolution-dependent. Bindings 29-31 are persistent because they are indexed by surfel or source identity.
- Unused source transform entries should be cleared or uploaded with `flags = 0u` so shader refresh failure is deterministic.
- The first runtime quality check should use multiple scene nodes sharing the same model, because that is the scenario the old packed model/primitive identity could not distinguish.
