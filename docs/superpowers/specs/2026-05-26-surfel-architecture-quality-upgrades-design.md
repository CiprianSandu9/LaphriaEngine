# Surfel Architecture Quality Upgrades Design

## Purpose

Build a stronger base for LaphriaEngine's surfel path tracer before adding more visual quality features. The current pipeline already resembles the SurfelPlus and SurfelGI/GIBS family, but it treats surfels primarily as persistent world-space samples in a uniform camera-relative grid. This design upgrades the foundation so later leak rejection, placement, material filtering, and large-scene behavior are easier to implement and tune.

The goal is not to replace the surfel path tracer. The goal is to evolve it in compatible phases, keeping the existing renderer usable after each step.

## Current Pipeline Context

The active surfel path tracer flow is:

1. Ray-traced GBuffer.
2. Persistent buffer preparation.
3. Surfel generation from under-covered visible surfaces.
4. Surfel update, recycling, cell counting, and cell-to-surfel insertion.
5. Surfel ray tracing and integration into persistent radiance/MSME state.
6. Diffuse GI resolve, reflection tracing/filtering, lighting integration, and TAA.

Relevant local files:

- `src/Core/EngineCore.cpp`
- `src/Core/SurfelPathTracerResources.h`
- `src/Core/SurfelPathTracerResources.cpp`
- `src/Core/SurfelPathTracerPasses.cpp`
- `src/Core/SurfelPathTracerPipelines.cpp`
- `src/Core/UISystem.h`
- `src/Core/UISystem.cpp`
- `src/shaders/SurfelPathTracerCommon.slang`
- `src/shaders/SurfelPathTracerGBufferClosestHit.slang`
- `src/shaders/SurfelPathTracerEvaluate.slang`
- `src/shaders/SurfelPathTracerUpdate.slang`
- `src/shaders/SurfelPathTracerCellInfo.slang`
- `src/shaders/SurfelPathTracerCellToSurfel.slang`
- `src/shaders/SurfelPathTracerRaygen.slang`
- `src/shaders/SurfelPathTracerIntegrate.slang`
- `src/shaders/SurfelPathTracerLightIntegrate.slang`
- `tests/SurfelPathTracerPipelineTests.cpp`

## Design Overview

Use an architecture-first sequence:

1. Add geometry-anchored surfels with a world-space fallback.
2. Centralize grid operations behind a grid policy helper layer.
3. Add an experimental non-uniform/frustum-biased grid mode after the helper layer exists.
4. Build depth-aware visibility, improved placement, material-aware filtering, and diagnostics on top of those foundations.

Each phase should be independently testable and should keep the existing uniform world-space path available until the replacement behavior is verified.

## Section 1: Geometry-Anchored Surfels

### Intent

Persist enough source-geometry identity per surfel to refresh position, normal, and material identity from the original hit surface. This follows the W298/SurfelGI direction, where surfels retain geometry information rather than relying only on last known world position.

### Data Model

Extend `SurfelPathTracerSurfel` in both C++ and Slang with compact source fields:

- `sourceInstanceId`
- `sourceNodeId` or equivalent stable scene-node/transform id
- `sourcePrimitiveIndex`
- `sourceBarycentrics`, packed as two floats or a compact equivalent
- `sourceFlags`
- `materialKey`

`materialKey` already exists in the current struct, but it is not yet meaningfully populated. It should become part of the source identity and reuse/removal logic.

`sourceInstanceId` must not be assumed to be a unique scene instance. In the current TLAS build, the instance custom index is packed from model id and primitive offset. True scene-node animation support requires a stable transform/source-node identity and compute-visible transform data, so the implementation should add or expose that identity explicitly.

The source flags should distinguish:

- valid geometry source
- invalid or unknown source
- world-space fallback surfel
- source refresh failed this frame

### Allocation

When a surfel is allocated from a visible GBuffer sample, store the source identity from the ray hit path alongside position and normal.

The GBuffer path currently needs enough payload/image data to preserve or reconstruct:

- source instance id
- source node or transform id
- primitive index
- barycentrics
- material key

If full geometry identity does not fit cleanly into the existing GBuffer images, add a dedicated storage image or structured buffer for source identity rather than overloading unrelated channels.

The source identity must support true scene-node animation. A surfel anchored to an instanced mesh should refresh from the current transform of the node that produced it, not only from the shared model mesh.

### Update

During `SurfelPathTracerUpdate.slang`, attempt to refresh anchored surfels before cell counting:

1. Validate source fields.
2. Reconstruct or fetch source triangle data and the current source transform.
3. Recompute world position and normal from barycentrics.
4. Refresh material key.
5. If refresh succeeds, continue with normal lifetime, cell, and ray allocation logic.
6. If refresh fails, mark source as failed and use current world-space recycling behavior.

The fallback path is important. It lets existing scenes and partially initialized surfels remain valid while the anchored path rolls out.

The implementation plan must choose one of two data-access strategies before coding the refresh path:

1. Extend compute descriptors so `SurfelPathTracerUpdate.slang` can read the same vertex, index, material, and transform data needed to reconstruct the anchored surface.
2. Add a dedicated surfel source metadata path that stores enough resolved source data for update without binding broad scene geometry arrays.

The choice should be made by comparing descriptor churn, memory cost, shader complexity, and compatibility with animated scene-node transforms. Until that choice is made, source refresh should be planned as a separate task after source identity capture.

### Expected Impact

Geometry anchoring should reduce stale lighting, ghost surfels, and delayed convergence when objects move or when source geometry changes. It also gives later material-aware filtering and leak rejection better evidence about which surface a surfel belongs to.

### Risks

This touches shader/C++ struct layout, GBuffer payloads, descriptor resources, and update logic. The implementation plan must add contract tests before changing shader behavior.

## Section 2: Grid Architecture

### Intent

Make grid behavior explicit and swappable. The current camera-relative uniform grid is a good default, but grid math is spread across multiple passes. A grid policy layer prevents counting, insertion, resolve, and path termination from drifting apart.

### Shared Grid Helpers

Centralize these operations in `SurfelPathTracerCommon.slang` or a new included shader file:

- world position to grid coordinate
- coordinate validity
- coordinate to flat cell index
- flat cell index to optional debug coordinate
- cell center and bounds
- surfel-cell intersection
- neighbor iteration range
- maximum cells touched per surfel

The first implementation should preserve current uniform-grid behavior exactly.

### C++ Settings

Add an enum setting for grid mode:

- `Uniform`
- `NonUniformExperimental`

The first committed behavior should keep `Uniform` as the default. Non-uniform mode should be behind UI/debug settings and shader defines until verified.

Grid mode and grid parameters must participate in persistent resource recreation. The resource layer should expose a grid capacity calculation, such as `gridCellCount(settings)`, rather than assuming `cellDimension^3` everywhere.

### Non-Uniform Grid

After the helper layer is stable, add a non-uniform/frustum-biased grid inspired by SurfelPlus. The design goal is better distribution of cell capacity around the camera and viewable region without increasing total cell count aggressively.

The non-uniform grid must provide the same helper interface as the uniform grid. Passes should not know which grid mode is active except through helper calls and push constants/defines.

The non-uniform implementation must define:

- total cell count
- maximum cells touched per surfel
- cell counter buffer size
- cell-to-surfel buffer size
- per-cell storage policy
- debug mapping from flat cell index to a readable occupancy visualization

These values must be shared by C++ resource allocation and shader helper code. A mismatch between resource sizing and shader indexing is a correctness bug, not a tunable artifact.

### Expected Impact

The uniform helper layer reduces implementation risk. The later non-uniform grid should improve large-scene scalability, reduce rejected stores in dense areas, and make the visible GI range more useful.

### Risks

All cell-dependent passes must agree. A mismatch can produce missing GI, invalid reads, or unstable surfel lookup. Tests should verify pass-order contracts and helper names, and runtime debug views should expose filled cells, rejected stores, and occupancy by grid mode.

## Section 3: Quality Features On Top

### Depth-Aware Surfel Visibility

Use the existing surfel depth atlas during:

- diffuse GI resolve
- path termination through surfels
- radiance sharing

The visibility weight should be conservative. Contributions should be reduced when query distance is clearly behind stored surfel depth, but not hard-clipped by default. This reduces light leaks through walls and layered geometry while avoiding flicker.

### Tile-Min Placement

Replace or augment phased placement with group-level minimum coverage placement:

1. Evaluate coverage per valid surface pixel in the compute group.
2. Select the lowest-coverage candidate.
3. Allocate only from that selected pixel, with a probability scaled by coverage deficit and depth.
4. Keep the current phased allocator as a fallback or debug mode during rollout.

### Material-Aware Reuse And Removal

Use `materialKey` and source identity to reduce cross-surface blending:

- lower or reject contribution from incompatible materials
- avoid reusing a surfel for a visibly different source surface
- prefer removing over-covered surfels that match the over-covered surface

This should be tuned after geometry anchoring is stable.

### Diagnostics

Add debug support for:

- anchored vs fallback surfels
- source refresh failures
- grid mode
- cell occupancy and rejected stores by grid mode
- leak rejection weight
- tile placement candidate selection
- material mismatch contribution suppression

Diagnostics should be exposed as counters or debug views only when they help validate behavior.

## Testing Strategy

Use tests in stages:

1. Extend shader contract tests for new struct fields, helper names, pass-order requirements, settings names, and debug counters.
2. Add CPU-side tests for any C++ helper math that mirrors shader grid logic.
3. Add build verification for `LaphriaEngineUnitTests`.
4. Use runtime debug views to validate visual and performance behavior in Sponza or another representative scene.

The plan should not rely only on screenshots. Counters and debug views are part of the validation story.

## Rollout Plan

Recommended implementation order:

1. Add source identity fields and contract tests without changing behavior.
2. Populate source identity from the GBuffer path.
3. Add stable scene-node/transform identity and compute-visible transform data needed for true animation-aware anchoring.
4. Choose and implement the source-data access strategy for update-pass refresh.
5. Refresh anchored surfels in update pass with fallback behavior.
6. Extract uniform grid helper layer and update all passes to use it.
7. Add grid capacity/resource sizing helpers and make grid mode participate in persistent resource recreation.
8. Add grid mode setting and non-uniform experimental implementation.
9. Add depth-aware surfel visibility.
10. Add tile-min placement.
11. Add material-aware contribution filtering.
12. Expand diagnostics and tune defaults.

Each item should be its own task or small group of tasks in the implementation plan.

## Non-Goals

This design does not require:

- replacing the full renderer
- changing the public render mode
- making non-uniform grid the default immediately
- removing existing world-space surfel fallback
- solving all reflection quality issues in the same phase
- adding a new denoiser architecture

## Success Criteria

The upgrade is successful when:

- existing uniform-grid surfel rendering still works
- anchored surfels refresh from source geometry and current scene-node transforms when source data is valid
- fallback surfels continue to render and recycle safely
- all cell-dependent passes use a shared grid helper interface
- non-uniform grid can be toggled experimentally without breaking uniform mode
- depth-aware visibility reduces obvious light leaks in representative scenes
- tile-min placement improves coverage/hole filling without allocation spikes
- tests and debug counters make regressions observable
