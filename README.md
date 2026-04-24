# LaphriaEngine

A Vulkan 1.4 real-time 3D engine in C++20, built as a dissertation project.

The current codebase includes an editor and validation refactor with expanded build targets, project-file workflows, runtime animation and skinning support, and automated tests.

---

## Features

### Rendering
- Runtime backend switching: `Rasterizer`, `RayTracer`, `PathTracer`
- PBR shading (GGX/Smith/Schlick), cascaded shadow maps, bindless resources, dynamic rendering
- Classic RT backend (direct lighting plus shadow rays)
- Path tracing backend with:
  - 1 SPP multi-bounce sampling
  - Temporal reprojection plus A-Trous denoising
  - Per-stage GPU timing (TLAS, ray trace, reprojection, denoiser)
  - Adaptive quality controls (manual, auto balanced, auto aggressive)
- Runtime glTF animation playback
- GPU skinning compute pass (currently used for rasterization path)
- Gameplay-oriented visual calibration controls (sun, fill, ambient, exposure)

### Physics
- CPU and GPU simulation modes
- Broadphase candidate generation via uniform-grid spatial hash
- Narrowphase support for sphere-sphere, AABB-AABB, sphere-AABB
- Static and dynamic bodies, gravity, friction, restitution

### Scene And Editor
- Scene graph with cached world transforms and octree plus frustum culling
- Scene JSON persistence with stable node IDs
- Asset references and animation playback components serialized in scene files
- Editor panels for:
  - Hierarchy plus inspector
  - Asset browser (project roots, import, import report)
  - Validation panel (project, scene, full validation)
  - Path tracer controls and performance stats

### Asset Pipeline
- glTF 2.0 (`.glb` and `.gltf`) import via `fastgltf`
- Embedded and external image handling with KTX2 and stb fallback
- Animation clip extraction (TRS channels) and runtime clip selection

### Host API
- New `EngineHost` entrypoint around `EngineCore`
- Configurable host options (window title, editor visibility, default input, physics simulation)
- Host callbacks (`initialize`, `updateFrame`, `drawUi`, `shutdown`) through `EngineServices`

---

## Build Targets

The CMake refactor now builds multiple targets:

- `LaphriaEngine` (static library): core engine and runtime systems
- `LaphriaEditor` (executable): default editor application
- `LaphriaEditorValidation` (static library): JSON project and scene validation logic
- `LaphriaValidationRunner` (executable): CLI validator for CI and local checks
- `LaphriaEngineUnitTests` (executable): unit tests for transform, frustum, and broadphase behavior

---

## Architecture

| Directory | Contents |
|-----------|----------|
| `src/Core/` | Engine host and core, Vulkan device/frame/swapchain/pipeline systems, UI/editor, import and validation, VMA context |
| `src/Physics/` | Physics runtime plus broadphase grid hashing |
| `src/SceneManagement/` | Scene, scene nodes, octree, frustum helpers |
| `src/shaders/` | Raster, RT/PT, denoiser/reprojection, physics, and skinning shaders |
| `tests/` | Validation fixtures and unit test entrypoint |

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
| `ShaderCommon.slang` | - | Shared material, math, and helper utilities |

All shaders are compiled via `slangc` during the CMake build.

---

## Build And Run

### Prerequisites

Windows (recommended):
- Visual Studio 2022/2026 with Desktop C++ workload
- CMake 3.29+
- Vulkan SDK 1.4.335+ (`slangc` required)
- vcpkg

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

### Run Validation CLI

```powershell
.\build\LaphriaTools\Release\LaphriaValidationRunner.exe --validate-project --project .\project.laphria_project.json
.\build\LaphriaTools\Release\LaphriaValidationRunner.exe --validate-scene --scene .\scene.json
```

### Run Tests

```powershell
ctest --test-dir build -C Release --output-on-failure
```

---

## Project File Format

Editor project files are JSON (`*.laphria_project.json`) and include:

- `name`
- `asset_roots`
- `scene_output_path`
- `import_settings` (`import_animations`, `import_materials`, `import_skins`, `strict_validation`)

Minimal example:

```json
{
  "name": "Laphria Project",
  "asset_roots": ["Assets"],
  "scene_output_path": "scene.json",
  "import_settings": {
    "import_animations": true,
    "import_materials": true,
    "import_skins": true,
    "strict_validation": false
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
| [nlohmann/json](https://github.com/nlohmann/json) | Scene, project, and validation JSON |
| [VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) | Vulkan memory allocation |

All dependencies are managed through vcpkg.

---

## Troubleshooting

`slangc` not found:
- Install Vulkan SDK and ensure `%VULKAN_SDK%\\bin` is in `PATH`.

vcpkg packages not found:
- Ensure `VCPKG_ROOT` is configured.
- Reconfigure with one of the bundled presets.

Validation CLI exits with failure:
- Check printed `error` entries first (warnings do not fail by default).
