# LaphriaEngine

A Vulkan 1.4 real-time 3D game engine written in C++20, developed as a dissertation project.

It features a physically-based rasterization pipeline, a hardware ray tracing pipeline, cascaded shadow maps, a GPU-accelerated starfield compute pass, GPU-accelerated physics simulation, a hierarchical scene graph with spatial culling, and an integrated ImGui editor.

---

## Features

### Rendering

**Rasterization pipeline**
- PBR shading — GGX distribution, Smith geometry, Schlick–Fresnel, ACES tonemapping
- Cascaded Shadow Maps (CSM) — 4 cascades, 2048×2048 per cascade, PSSM splits, bounding-sphere stabilization, sub-pixel snapping, normal-offset bias
- Starfield background via a Vulkan compute pass (16×16 workgroups, blit to swapchain)
- Bindless material textures (VK_EXT_descriptor_indexing, variable descriptor count)
- Dynamic rendering (no VkRenderPass / VkFramebuffer objects)
- Vulkan synchronization2 barriers throughout

**Ray tracing pipeline** (VK_KHR_ray_tracing_pipeline + VK_KHR_acceleration_structure)
- Hardware TLAS/BLAS acceleration structures, rebuilt each frame
- Raygen → ClosestHit / AnyHit / Miss shader stages
- Alpha-cutout transparency in AnyHit
- RT shadow rays for hard shadows
- Bindless per-model vertex, index, material, and texture arrays

**Common**
- Multi-frame buffering (2 frames in flight)
- Dynamic swapchain resize handling
- Slang shaders compiled to SPIR-V at build time

### Physics
- Dual CPU/GPU simulation via a Vulkan compute pipeline
- Rigid body dynamics: gravity, friction, restitution
- Collision detection: sphere–sphere, AABB–AABB, sphere–AABB
- Static and dynamic bodies with configurable mass
- World-bounds enforcement

### Scene Management
- Hierarchical scene graph with node transforms (position, rotation via quaternions, scale)
- Octree-based spatial culling
- Scene serialization / deserialization (JSON)

### Asset Pipeline
- glTF 2.0 (`.glb` / `.gltf`) loading via fastgltf, including embedded textures
- KTX 2.0 compressed texture loading

### Editor UI
- Integrated ImGui editor
- Object selection and transform manipulation
- Light direction control
- Physics simulation play / pause
- Toggle between rasterization and ray tracing at runtime

---

## Architecture

Source is organized under `src/` into three areas:

| Directory | Contents |
|-----------|----------|
| `src/Core/` | Vulkan subsystems: `VulkanDevice`, `SwapchainManager`, `PipelineCollection`, `FrameContext`, `UISystem`, `InputSystem`, `Camera`, `ResourceManager`, `EngineCore` |
| `src/Physics/` | `PhysicsSystem` (CPU+GPU), `PhysicsDefines` (shared C++/Slang structs) |
| `src/SceneManagement/` | `Scene`, `SceneNode`, `Octree` |
| `src/Shaders/` | See table below |

`EngineCore` acts as a thin coordinator that owns all subsystems. Each subsystem is responsible for its own Vulkan objects.

### Shaders

| File | Entry points | Purpose |
|------|-------------|---------|
| `LaphriaEngine.slang` | `vertMain`, `fragMain` | PBR vertex + fragment (CSM lookup, normal mapping, PBR lighting) |
| `Compute.slang` | `computeMain` | Starfield background — writes to a storage image, blitted to the swapchain |
| `Shadow.slang` | `shadowVert`, `shadowFrag` | Depth-only CSM pass (4 cascades); alpha-cutout discard in fragment |
| `Raygen.slang` | `main` | RT camera ray generation — unprojects each pixel into a world-space ray |
| `ClosestHit.slang` | `main` | RT surface shading — PBR + shadow ray + normal mapping |
| `AnyHit.slang` | `main` | RT alpha-cutout transparency test |
| `Miss.slang` | `main` | RT background sky gradient; shadow ray miss → not occluded |
| `Physics.slang` | `physicsMain` | GPU rigid-body integration compute shader |
| `ShaderCommon.slang` | — | Shared structs (`UniformBuffer`, `ScenePushConstants`, `MaterialData`, `RayPayload`), PBR functions, `mat3Inverse` |

All shaders are compiled from source with `slangc` as part of the CMake build.

---

## Build & Run

### Prerequisites

**Windows (recommended)**
- Visual Studio 2022/2026 with the "Desktop development with C++" workload
- CMake 3.29+
- Vulkan SDK 1.4.335+ (sets `VULKAN_SDK` and provides `slangc`)
- vcpkg

**Other platforms**
- CMake 3.29+
- A C++20 compiler
- Vulkan SDK 1.4.335+ (or equivalent headers + loader)
- `slangc` available in PATH
- vcpkg

### Clone

```bash
git clone <repo-url>
cd LaphriaEngine
```

### vcpkg Setup (Windows)

1. Clone and bootstrap vcpkg:

```powershell
git clone https://github.com/microsoft/vcpkg.git C:\path\to\vcpkg
C:\path\to\vcpkg\bootstrap-vcpkg.bat
```

2. Set environment variables (per-session or via system settings):

```powershell
$env:VCPKG_ROOT = "C:\path\to\vcpkg"
$env:VCPKG_DEFAULT_TRIPLET = "x64-windows"
```

### Configure

The repository includes presets that wire vcpkg automatically:

```powershell
cmake --preset vcpkg-vs26
```

OR

```powershell
cmake --preset vcpkg-vs22
```

OR

```powershell
cmake --preset vcpkg-ninja
```

### Build

```powershell
cmake --build build --config Release
```

### Run

```powershell
.\build\LaphriaEngine\Release\LaphriaEngine.exe
```

A test asset (`testassets/paladin.glb`) is included for quick verification.

---

## Dependencies

| Library | Purpose |
|---------|---------|
| [Vulkan SDK](https://vulkan.lunarg.com/) | Graphics, compute, and ray tracing API |
| [GLFW](https://www.glfw.org/) | Window and input management |
| [GLM](https://github.com/g-truc/glm) | Math (vectors, matrices, quaternions) |
| [fastgltf](https://github.com/spnda/fastgltf) | glTF 2.0 model loading |
| [ImGui](https://github.com/ocornut/imgui) | Immediate-mode UI |
| [KTX](https://github.com/KhronosGroup/KTX-Software) | KTX 2.0 texture loading |
| [nlohmann/json](https://github.com/nlohmann/json) | Scene serialization |
| [stb](https://github.com/nothings/stb) | Embedded image decoding |

All dependencies are managed via vcpkg.

---

## Troubleshooting

**vcpkg packages not found**
- Ensure `VCPKG_ROOT` is set correctly.
- Delete `build/` and re-run `cmake --preset vcpkg` if you previously configured without vcpkg.

**`slangc` not found**
- Install the Vulkan SDK and verify `%VULKAN_SDK%\bin` is in your PATH.
- `slangc` ships with the Vulkan SDK 1.4.335+; ensure that version or newer is installed.
