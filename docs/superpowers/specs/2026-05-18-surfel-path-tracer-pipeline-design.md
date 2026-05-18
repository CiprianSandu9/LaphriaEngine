# Surfel Path Tracer Pipeline Design

## Goal

Create a new native Laphria render backend that adapts the full SurfelGI and SurfelPlus architecture into this repo without replacing the existing `PathTracer` implementation. The backend should support diffuse surfel GI, glossy reflections, temporal/spatial filtering, bilateral cleanup, TAA, tone mapping, debug overlays, and validation controls.

Reference material:

- `https://w298.github.io/SurfelGI/`
- `https://github.com/WANG-Ruipeng/SurfelPlus`
- `.codex_refs/SurfelGI/RenderPasses/Surfel/`
- `.codex_refs/SurfelPlus/src/`
- `.codex_refs/SurfelPlus/shaders/`

## Approved Approach

Add a separate render mode, `RenderMode::SurfelPathTracer`, beside the current `Rasterizer`, `RayTracer`, and `PathTracer` modes.

The new backend should reuse Laphria's existing Vulkan and Slang foundations:

- TLAS and BLAS creation/refit from `ResourceManager` and `FrameContext`.
- Bindless vertex, index, material, and texture descriptor patterns used by the current RT/PT paths.
- Swapchain-size resource lifetime patterns in `FrameContext`.
- Pipeline and descriptor layout creation through the existing `PipelineCollection` style.
- Slang shader compilation through the repo's CMake shader build.
- Swapchain blit and ImGui composition flow from `EngineCore`.

The implementation must not directly port Falcor or nvpro framework structure. SurfelGI and SurfelPlus are architectural references; Laphria owns the final resource, descriptor, shader, UI, and command-buffer shape.

## Pipeline Shape

The backend should adapt the full SurfelPlus-style pass graph:

1. Surfel GBuffer or VBuffer pass.
2. Surfel prepare pass.
3. Surfel update pass.
4. Cell info update pass.
5. Cell-to-surfel update pass.
6. Surfel ray trace pass.
7. Surfel integrate pass.
8. Surfel generation and evaluation pass.
9. Reflection trace pass.
10. Reflection temporal/spatial filtering pass.
11. Bilateral cleanup pass.
12. Light integrate pass.
13. TAA and tone mapping pass.
14. Swapchain blit and editor UI rendering.

The first backend output may be empty or sky-only for skeleton validation, but the design target is the full stack.

## Components

### `SurfelPathTracerResources`

Own all persistent and extent-dependent resources for the new backend. This should be a focused owner rather than another long block of unrelated fields inside `EngineCore`.

Persistent resources include:

- surfel counter buffer,
- surfel buffer,
- alive/free/dirty index buffers,
- recycle metadata buffer,
- surfel ray buffer,
- cell info buffer,
- cell counter buffer,
- cell-to-surfel index buffer,
- debug/readback counters.

Extent-dependent resources include:

- visibility/GBuffer images,
- indirect diffuse lighting image,
- reflection raw and filtered images,
- temporal/spatial history images,
- TAA/tone-map intermediates,
- surfel irradiance and depth atlas images.

Persistent surfel buffers should survive swapchain resize unless capacity, cell layout, atlas size, or another static surfel setting changes. Extent-dependent images and descriptor writes are recreated on resize.

### `SurfelPathTracerPipelines`

Add a dedicated helper class for the new backend's descriptor set layouts, pipeline layouts, compute pipelines, RT pipelines, and Shader Binding Tables. `PipelineCollection` can own this helper, but the surfel backend's creation and member names should remain grouped under `SurfelPathTracer` so the existing path tracer and classic RT code remain distinguishable.

The backend needs compute pipelines for the surfel and filtering passes plus separate RT pipelines for surfel ray tracing and glossy/reflection tracing. Shader Binding Tables should be separate from the existing `rayTracingPipeline` and `classicRTPipeline` tables.

### `SurfelPathTracerPasses`

Provide a thin command-recording layer with one method per pass and explicit synchronization between passes. This layer translates the SurfelPlus pass graph into Laphria command-buffer calls:

- bind the pass pipeline,
- bind descriptor sets,
- push compact pass constants,
- dispatch or trace rays,
- record image/buffer barriers.

This keeps `EngineCore::recordCommandBuffer()` from growing into a second renderer.

### Shaders

Add `SurfelPathTracerCommon.slang` for shared definitions:

- surfel, surfel ray, cell, recycle, and counter structs,
- packed normal helpers,
- spatial cell/hash helpers,
- MSME/running variance helpers,
- surfel irradiance atlas addressing,
- ray offset and finite-value guards,
- debug mode constants.

Add one shader file per pass under `src/shaders/`, following existing names and entry-point conventions. Use Slang, not GLSL. Convert useful SurfelPlus and SurfelGI algorithms into Laphria-compatible Slang rather than importing their shader files verbatim.

### UI

Extend `UISystem` with:

- `SurfelPathTracer` render mode selection,
- reset and lock surfels,
- surfel capacity/cell/atlas quality controls,
- diffuse GI, reflection, filtering, TAA toggles,
- debug overlays for normals/depth/surfel ID/radius/variance/radiance/cell occupancy/reflection,
- basic counters for surfel count, ray budget, filled cells, rejected stores, and history acceptance.

The UI should support validation without reusing the old path tracer's reservoir experiment controls.

## Data Flow

Each frame:

1. Build or refit TLAS through the existing RT/PT path.
2. Produce visibility data: object/primitive or instance ID, depth, normal, material data, and world-space position or reconstructable depth.
3. Clear transient counters and prepare surfel ray/cell state.
4. Update existing surfels by aging/recycling invalid entries, resizing radius, allocating rays from variance/life/visibility, and counting cell occupancy.
5. Accumulate cell offsets and populate the cell-to-surfel buffer.
6. Trace surfel rays against TLAS. Rays gather direct emission, sky/sun contribution, and may terminate through nearby surfel radiance to accelerate convergence.
7. Integrate surfel ray results into surfel radiance using MSME-style running mean/variance and update directional irradiance/depth atlas data.
8. Evaluate screen pixels against nearby surfels to produce diffuse indirect lighting, detect under-covered and over-covered regions, generate new surfels, and remove excessive coverage.
9. Trace glossy/reflection rays at reduced resolution with RIS-style candidate selection and surfel-backed indirect termination.
10. Run reflection temporal/spatial filtering.
11. Run bilateral cleanup using color, normal, depth, material, and variance cues.
12. Integrate direct lighting, diffuse surfel GI, reflection, and optional ambient occlusion into a lighting buffer.
13. Run TAA and tone mapping.
14. Blit to the swapchain and render ImGui.

## Error Handling And Fallbacks

The backend should fail softly:

- If Vulkan ray tracing support is unavailable, disable or hide the mode.
- If surfel resource allocation fails, log the failed resource name and fall back to `Rasterizer` when possible.
- If no models or no TLAS instances are available, clear output and counters instead of dispatching invalid work.
- On resize, recreate extent-dependent images and descriptor writes while preserving persistent surfel buffers.
- When static surfel settings change, wait for device idle, rebuild surfel resources, and reset history.
- Debug/readback counters are optional for rendering; unavailable counters should show as zero or unavailable.

Numerical stability defaults:

- normal-offset ray origin bias,
- bounded cell population,
- finite-value checks before stores,
- luminance clamps for surfel rays and reflections,
- history reset on camera teleport, static setting changes, and render-mode switch.

## Testing Strategy

Use the current `PathTracerAnalysisTests` pattern for structural tests:

- shader contract checks for required pass entry points, bindings, and shared helper functions,
- C++/shader shared-struct size and offset checks for every host-shared surfel struct,
- CPU tests for cell hashing, radius/cell overlap, and fixed-capacity allocation logic,
- render-mode plumbing checks for UI selection and command-buffer routing,
- resize/resource lifetime tests where they can run without a GPU.

Runtime visual validation should use Sponza and small synthetic scenes:

- sky-only backend output for skeleton,
- GBuffer debug AOVs,
- surfel ID/radius/variance/radiance overlays,
- diffuse GI on enclosed and open scenes,
- glossy reflection with and without surfel termination,
- TAA/filtering stability during camera motion.

## Milestones

1. Backend skeleton: render mode, empty/sky output, descriptors, pipelines, resources, resize survival.
2. GBuffer/VBuffer: primary visibility plus normal/depth/material debug AOVs.
3. Persistent surfel resources: counters, alive/free lists, reset/lock UI, readback stats.
4. Surfel placement/evaluation: screen-driven surfel generation and indirect debug output.
5. Cell grid: cell counting, offset accumulation, and cell-to-surfel queries.
6. Surfel ray tracing/integration: radiance, variance, ray budget, directional atlas.
7. Diffuse GI integration: direct lighting plus surfel indirect light.
8. Glossy reflection: reduced-resolution reflection trace with surfel-backed termination.
9. Filtering/TAA/tone map: temporal/spatial reflection filter, bilateral cleanup, final TAA.
10. Quality and validation controls: budgets, resolution scale, debug overlays, reset behavior, Sponza validation preset.

## Non-Goals For This Design

- Replace the existing `PathTracer` backend in the first implementation cycle.
- Port Falcor, nvpro, or GLSL framework code directly.
- Preserve old experimental reservoir-GI implementation details unless a specific idea is useful to the new backend.
- Implement every debug sweep from the previous path tracer.

## Open Implementation Notes

- Prefer a dedicated `SurfelPathTracerResources` owner and command-recording helper to avoid expanding `EngineCore` further.
- The first implementation plan should preserve small, testable milestones even though the approved design target is the full stack.
- Descriptor layout design should reserve room for debug buffers and future atlas variants, but avoid binding unused resources until a pass needs them.
