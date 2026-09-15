# LaphriaEngine

A Vulkan 1.4 real-time rendering engine in C++20, developed as a master's dissertation project.

The main contribution is a **surfel-based global illumination path tracer**: a persistent, world-space cache of surfels that accumulates multi-bounce diffuse lighting over time, with adaptive per-surfel ray budgeting, cache-guided sampling, and glossy reflections. The engine also ships a rasterizer, a classic ray tracer, and a denoised 1 SPP path tracer so the surfel backend can be compared against reference approaches in the same scene.

---

## Features

### Rendering
- Runtime backend switching from the editor: `Rasterizer`, `RayTracer`, `PathTracer`, `SurfelPathTracer`
- PBR shading (GGX/Smith/Schlick), cascaded shadow maps, bindless resources, dynamic rendering
- Classic RT backend (direct lighting plus shadow rays)
- Path tracing backend with:
  - 1 SPP multi-bounce sampling with environment and sun next-event estimation
  - Temporal reprojection with motion-aware accumulation plus A-Trous denoising
  - Per-stage GPU timing (TLAS, ray trace, reprojection, denoiser) with P50/P95/P99 frame stats
  - Adaptive quality controls (manual, auto balanced, auto aggressive) driven by a target frame time
- Surfel path tracer backend with:
  - Persistent world-space surfel GI cache with GPU-driven placement, recycling, and redundancy removal
  - Uniform-grid cell index with a compact per-frame cell-to-surfel map
  - Adaptive ray scheduling: per-surfel ray counts scale with temporal inconsistency, under a hard per-frame ray cap, with reduced tracing for off-screen surfels
  - Cache-guided path sampling (irradiance guide atlas), surfel path termination, and radiance sharing between neighbouring surfels
  - Traced glossy reflections with a temporal filter, faded into a cache-evaluated specular term at high roughness
  - Bilateral cleanup, TAA, and an ambient-occlusion aware diffuse GI composite
  - A 1 SPP reference probe and difference view for validation, plus a pixel probe that reads back HDR/GBuffer texels
  - 18 debug views (GBuffer channels, surfel ID/radius/radiance/variance/coverage, cell occupancy, reflection raw/filtered, sun visibility, AO, reference color/difference)
  - Per-pass GPU timings with a 300-frame rolling window and copy-to-clipboard stats
- Runtime glTF animation playback
- GPU skinning compute pass; skinned vertex buffers feed both the raster draw and the ray tracing acceleration structures
- Exposure controls (manual and auto exposure), light direction, and environment lighting options (next-event estimation bounce mode, cosine or sky-biased sampling, black environment)

### Physics
- CPU and GPU simulation modes, switchable at runtime
- Broadphase candidate generation via uniform-grid spatial hash
- Narrowphase support for sphere-sphere, AABB-AABB, sphere-AABB
- Static and dynamic bodies, gravity, friction, restitution

### Scene And Editor
- Scene graph with cached world transforms and octree plus frustum culling
- Scene JSON persistence with stable node IDs
- Asset references and animation playback components serialized in scene files
- Editor windows:
  - `Scene Hierarchy` and `Inspector` (transforms, materials, animation preview)
  - `Asset Browser` (project roots, import, import report)
  - `Lighting Control` (light direction, camera speed, culling freeze, scene/project load and save)
  - `Engine Controls` (render backend, exposure, texture color space, path tracer and surfel path tracer settings, advanced lighting, performance stats, physics CPU/GPU mode)

### Asset Pipeline
- glTF 2.0 (`.glb` and `.gltf`) import via `fastgltf`
- Embedded and external image handling with KTX2 and stb fallback
- Animation clip extraction (TRS channels) and runtime clip selection
- Fallback tangent generation for meshes without PBR tangents
- Batched GPU upload path for model import (reduced per-resource queue stalls)
- Import stage timing logs (parse, texture decode/upload, mesh extraction, buffer upload, BLAS build, total)

#### Runtime Assets

Runtime GLB files live in `Assets/` at the repository root. The folder is git-ignored, so it is empty on a fresh clone: place your own models there, or prepare them with the script below. The editor resolves `Assets` by walking up from the working directory, so it is found when running from the build output folder.

Sponza is the scene used throughout the dissertation. A runtime-ready `sponza_runtime.glb` can be produced from the Sponza source model with the preparation script.

#### Runtime Asset Prep Workflow (Heavy Models)

For large assets, keep source and runtime versions separate:
- Source asset: `*_source.glb`
- Runtime asset: `*_runtime.glb`

The script runs three glTF-Transform stages:
1. meshopt geometry compression
2. KTX2 `ETC1S` for every texture slot
3. KTX2 `UASTC` override for color-critical slots (default: base color, normal, emissive)

It requires the [glTF-Transform CLI](https://gltf-transform.dev/cli): install Node.js LTS and run `npm i -g @gltf-transform/cli`, or leave `npx` on `PATH` and the script falls back to `npx @gltf-transform/cli`.

```powershell
python tools/assets/prepare_gltf_assets.py --input Assets/sponza_source.glb --output Assets/sponza_runtime.glb
```

Options: `--uastc-slots baseColorTexture,normalTexture,emissiveTexture` (use `none` to keep ETC1S everywhere), `--dry-run` to print the commands only.

Runtime color-space model (`Engine Controls` > `Texture Color Space`):
- `Hardware SRGB` (default): color textures sampled with hardware SRGB decode
- `Legacy Manual`: UNORM color textures with shader-side `sRGBToLinear`
- Changing this toggle requires reloading/re-importing model assets to fully apply.

### Host API
- `EngineHost` entrypoint around `EngineCore`
- Configurable host options (window title, editor visibility, default camera input, physics simulation)
- Host callbacks (`initialize`, `updateFrame`, `drawUi`, `shutdown`) receive an `EngineServices` handle with the camera, scene, physics, resource manager, UI, and asset/primitive helpers

```cpp
#include "Core/EngineHost.h"

int main() {
    EngineHostOptions options;
    options.windowTitle = "My Application";
    options.showEditorPanels = false;

    EngineHostCallbacks callbacks;
    callbacks.initialize = [](EngineServices &services) {
        services.loadModelAsset("Assets/paladin.glb", nullptr);
    };
    callbacks.updateFrame = [](EngineServices &services, float deltaTimeSeconds) {
        // per-frame logic
    };

    EngineHost host(options, callbacks);
    host.run();
}
```

The bundled `LaphriaEditor` is the default host with editor panels enabled and no callbacks.

---

## Build Targets

- `LaphriaEngine` (static library): core engine and runtime systems
- `LaphriaEditor` (executable): default editor application
- `LaphriaEngine_shaders` (custom target): compiles every Slang shader to SPIR-V; both targets above depend on it

---

## Architecture

| Directory | Contents |
|-----------|----------|
| `src/Core/` | Engine host and core, Vulkan device/frame/swapchain/pipeline systems, surfel path tracer resources, passes, and pipelines, UI/editor, glTF import, editor project files, VMA context |
| `src/Physics/` | Physics runtime plus broadphase grid hashing |
| `src/SceneManagement/` | Scene, scene nodes, octree, frustum helpers |
| `src/shaders/` | Raster, RT/PT, surfel path tracer, denoiser/reprojection, physics, and skinning shaders |
| `Assets/` | Runtime GLB assets used by the editor (git-ignored) |
| `testassets/` | Small sample assets kept outside the runtime asset folder |
| `tools/` | Asset preparation utilities |
| `CMake/` | Local CMake find modules |

### Surfel Path Tracer Frame

Each frame the surfel backend runs, in order: GBuffer ray generation, per-frame prepare, surfel update (placement, recycling, removal), cell info and cell-to-surfel map construction, ray scheduling, surfel ray tracing, radiance integration, per-pixel cache evaluation, reflection trace and filter, bilateral cleanup, light integration (composite and debug views), and TAA. The sky pass provides the output fallback when the backend is disabled.

### Shader Set

| File | Entry Point(s) | Purpose |
|------|----------------|---------|
| `LaphriaEngine.slang` | `vertMain`, `fragMain` | Raster PBR pipeline |
| `Shadow.slang` | `shadowVert`, `shadowFrag` | Cascaded shadow map pass |
| `Compute.slang` | `computeMain` | Compute pass (legacy starfield path) |
| `Skinning.slang` | `skinningMain` | GPU skinning compute stage |
| `Physics.slang` | `physicsMain` | GPU rigid-body integration |
| `RT_Raygen.slang` | `main` | Classic RT ray generation |
| `RT_ClosestHit.slang` | `main` | Classic RT closest hit |
| `RT_AnyHit.slang` | `main` | Classic RT alpha cutout |
| `RT_Miss.slang` | `main` | Classic RT miss |
| `Raygen.slang` | `main` | Path tracer ray generation plus GBuffer writes |
| `ClosestHit.slang` | `main` | Path tracer closest hit and bounce logic |
| `AnyHit.slang` | `main` | Path tracer alpha cutout |
| `Miss.slang` | `main` | Path tracer miss |
| `Reprojection.slang` | `reprojectionMain` | Temporal reprojection |
| `Denoiser.slang` | `atrousMain` | A-Trous denoiser |
| `SurfelPathTracerSky.slang` | `main` | Surfel path tracer sky/output fallback pass |
| `SurfelPathTracerGBuffer.slang` | `main` | Surfel path tracer GBuffer ray generation |
| `SurfelPathTracerGBufferMiss.slang` | `main` | Surfel path tracer GBuffer miss shader |
| `SurfelPathTracerGBufferClosestHit.slang` | `main` | Surfel path tracer GBuffer closest-hit shader |
| `SurfelPathTracerGBufferAnyHit.slang` | `main` | Surfel path tracer GBuffer alpha/visibility shader |
| `SurfelPathTracerPrepare.slang` | `main` | Surfel path tracer per-frame preparation |
| `SurfelPathTracerUpdate.slang` | `main` | Surfel placement, recycling, and redundancy removal |
| `SurfelPathTracerCellInfo.slang` | `main` | Compact surfel cell metadata generation |
| `SurfelPathTracerCellToSurfel.slang` | `main` | Cell-to-surfel lookup construction |
| `SurfelPathTracerRaySchedule.slang` | `main` | Per-surfel ray budgeting (variance-driven, frame cap, off-screen interval) and ray work generation |
| `SurfelPathTracerRaygen.slang` | `main` | Surfel-guided path tracing ray generation |
| `SurfelPathTracerMiss.slang` | `main` | Surfel path tracer miss shader |
| `SurfelPathTracerClosestHit.slang` | `main` | Surfel path tracer closest-hit shader |
| `SurfelPathTracerAnyHit.slang` | `main` | Surfel path tracer alpha/visibility shader |
| `SurfelPathTracerIntegrate.slang` | `main` | Surfel radiance integration |
| `SurfelPathTracerEvaluate.slang` | `main` | Surfel GI lookup and evaluation |
| `SurfelPathTracerReflection.slang` | `main` | Glossy reflection pass |
| `SurfelPathTracerReference.slang` | `main` | Reference ray tracing pass for comparison/debug views |
| `SurfelPathTracerReflectionFilter.slang` | `main` | Reflection history/filter pass |
| `SurfelPathTracerBilateral.slang` | `main` | Bilateral cleanup pass |
| `SurfelPathTracerLightIntegrate.slang` | `main` | Final lighting/debug-view integration |
| `SurfelPathTracerTaa.slang` | `main` | Surfel path tracer TAA pass |
| `ShaderCommon.slang` | - | Shared material, math, and helper utilities |
| `SurfelPathTracerCommon.slang` | - | Shared surfel path tracer structures and helpers |

Compiled shader entries are generated via `slangc` during the CMake build; common shader files are tracked as include dependencies. At runtime the engine looks for `Shaders/*.spv` next to the executable first, then in the working directory.

---

## Build And Run

### Prerequisites

Windows (recommended):
- Visual Studio 2022/2026 with Desktop C++ workload
- CMake 3.29+
- Vulkan SDK 1.4.335+ (`slangc` required)
- vcpkg with `VCPKG_ROOT` set

GPU requirements:
- A Vulkan 1.4 capable driver
- Hardware ray tracing: `VK_KHR_ray_tracing_pipeline` and `VK_KHR_acceleration_structure`
- Buffer device address, descriptor indexing, dynamic rendering, and synchronization2

Device creation fails on GPUs without ray tracing support; the rasterizer alone is not selectable on such hardware.

### Configure

```powershell
cmake --preset vcpkg-vs26
```

or:

```powershell
cmake --preset vcpkg-vs22
```

or:

```powershell
cmake --preset vcpkg-ninja
```

### Build

```powershell
cmake --build build --config Release
```

### Run Editor

```powershell
.\build\LaphriaEngine\Release\LaphriaEditor.exe
```

### Editor Controls

- `W`/`A`/`S`/`D` or arrow keys: move the camera
- `Q`/`E` or `Page Down`/`Page Up`: move the camera down/up
- Right mouse button: mouse look
- Camera speed is adjustable in `Lighting Control`
- Switch the render backend with the radio buttons at the top of `Engine Controls`

## Project File Format

Editor project files are JSON (`*.laphria_project.json`) and include:

- `name`
- `asset_roots`
- `scene_output_path`
- `import_settings` (`import_animations`, `import_materials`, `import_skins`)

Minimal example:

```json
{
  "name": "Laphria Project",
  "asset_roots": ["Assets"],
  "scene_output_path": "scene.json",
  "import_settings": {
    "import_animations": true,
    "import_materials": true,
    "import_skins": true
  }
}
```

---

## Dependencies

| Library | Purpose |
|---------|---------|
| [Vulkan SDK](https://vulkan.lunarg.com/) | Vulkan API plus `slangc` |
| [GLFW](https://www.glfw.org/) | Windowing and input |
| [GLM](https://github.com/g-truc/glm) | Math |
| [fastgltf](https://github.com/spnda/fastgltf) | glTF import |
| [ImGui](https://github.com/ocornut/imgui) | Editor UI |
| [KTX](https://github.com/KhronosGroup/KTX-Software) | KTX2 textures |
| [stb](https://github.com/nothings/stb) | Image decoding fallback |
| [nlohmann/json](https://github.com/nlohmann/json) | Scene and project JSON |
| [VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) | Vulkan memory allocation |

All C++ dependencies are managed through vcpkg. The asset preparation script additionally needs Node.js and the glTF-Transform CLI.

---

## Troubleshooting

`slangc` not found:
- Install the Vulkan SDK and ensure `%VULKAN_SDK%\bin` is in `PATH`.

vcpkg packages not found:
- Ensure `VCPKG_ROOT` is configured.
- Reconfigure with one of the bundled presets.

Device creation fails or no suitable GPU is found:
- The engine requires hardware ray tracing (see GPU requirements above). Update the driver or run on an RT-capable GPU.

Editor starts but the scene is empty:
- `Assets/` is git-ignored. Add runtime GLB files there or point `asset_roots` in the project file at your asset folder.

`Could not find glTF-Transform CLI` when preparing assets:
- Install Node.js LTS, then `npm i -g @gltf-transform/cli`, or make sure `npx` is on `PATH`.

---

## License

LaphriaEngine is released under the GNU General Public License v3.0. See [LICENSE](LICENSE).
