#include "EngineCore.h"
#include "VulkanUtils.h"
#include "VmaContext.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "../SceneManagement/Scene.h"
#include "EngineAuxiliary.h"
#include "EngineConfig.h"
#include "ResourceManager.h"

using namespace Laphria;

namespace
{
constexpr uint32_t kPtTimestampQueryCountPerFrame = 8;
constexpr double kWindowTitleUpdateIntervalSeconds = 0.5;
enum PtTimestampSlot : uint32_t
{
    kPtTS_TlasStart = 0,
    kPtTS_TlasEnd = 1,
    kPtTS_RayTraceStart = 2,
    kPtTS_RayTraceEnd = 3,
    kPtTS_ReprojectionStart = 4,
    kPtTS_ReprojectionEnd = 5,
    kPtTS_DenoiserStart = 6,
    kPtTS_DenoiserEnd = 7
};
}

EngineCore::EngineCore(EngineHostOptions optionsIn, EngineHostCallbacks callbacksIn)
    : options(std::move(optionsIn)), callbacks(std::move(callbacksIn))
{
}

void EngineCore::run()
{
    try {
        VulkanUtils::resetAllocationCounter();
        initWindow();
        initInput();
        initVulkan();
        initImgui();
        invokeInitializeCallback();
        // Game initialization may load models/maps after Vulkan init. Rebuild RT descriptor
        // sets here so first-time RT/PT switching never uses stale pre-init bindings.
        if (resourceManager) {
            createRayTracingDescriptorSets();
        }
        mainLoop();
        const auto vmaStats = Laphria::VmaContext::getStats();
        LOGI("VMA stats: blocks=%u allocations=%u allocationBytes=%llu",
             vmaStats.blockCount,
             vmaStats.allocationCount,
             static_cast<unsigned long long>(vmaStats.allocationBytes));
        LOGI("Tracked Vulkan allocations: %llu", static_cast<unsigned long long>(VulkanUtils::getAllocationCounter()));
        invokeShutdownCallback();
        cleanup();
    } catch (...) {
        try {
            if (vulkanInitialized) {
                vulkan.logicalDevice.waitIdle();
            }
        } catch (...) {
            // Best-effort cleanup path.
        }

        try {
            cleanup();
        } catch (...) {
            // Suppress cleanup exceptions while propagating the original failure.
        }

        throw;
    }
}

EngineServices EngineCore::buildServices()
{
    return EngineServices{
        .window = window,
        .camera = camera,
        .scene = *scene,
        .physics = *physicsSystem,
        .resourceManager = *resourceManager,
        .ui = ui,
        .loadSceneAsset =
            [this](const std::string &path) {
                scene->loadScene(path, *resourceManager, *pipelines.descriptorSetLayoutMaterial);
            },
        .saveSceneAsset =
            [this](const std::string &path) {
                scene->saveScene(path, *resourceManager);
            },
        .loadModelAsset =
            [this](const std::string &path, const SceneNode::Ptr &parent) {
                scene->loadModel(path, *resourceManager, *pipelines.descriptorSetLayoutMaterial, parent);
            },
        .createCubePrimitive =
            [this](float size) {
                return resourceManager->createCubeModel(size, *pipelines.descriptorSetLayoutMaterial);
            },
        .createCubePrimitiveWithMaterial =
            [this](float size, const Laphria::MaterialData &material) {
                return resourceManager->createCubeModel(size, *pipelines.descriptorSetLayoutMaterial, material);
            },
        .createSpherePrimitive =
            [this](float radius, int slices, int stacks) {
                return resourceManager->createSphereModel(radius, slices, stacks, *pipelines.descriptorSetLayoutMaterial);
            },
        .createCylinderPrimitive =
            [this](float radius, float height, int slices) {
                return resourceManager->createCylinderModel(radius, height, slices, *pipelines.descriptorSetLayoutMaterial);
            }
    };
}

void EngineCore::invokeInitializeCallback()
{
    ui.showEditorPanels = options.showEditorPanels;
    if (callbacks.initialize && scene && physicsSystem && resourceManager) {
        auto services = buildServices();
        callbacks.initialize(services);
    }
}

void EngineCore::invokeShutdownCallback()
{
    if (callbacks.shutdown && scene && physicsSystem && resourceManager) {
        auto services = buildServices();
        callbacks.shutdown(services);
    }
}

void EngineCore::initWindow() {
    if (glfwInit() != GLFW_TRUE) {
        throw std::runtime_error("failed to initialize GLFW");
    }
    windowInitialized = true;

    // GLFW_NO_API: we manage the Vulkan surface ourselves, not via an OpenGL context.
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(WIDTH, HEIGHT, options.windowTitle.c_str(), nullptr, nullptr);
    if (!window) {
        throw std::runtime_error("failed to create GLFW window");
    }
    lastFrameTime = std::chrono::high_resolution_clock::now();
}

void EngineCore::initInput() {
    input.init(window, camera, swapchain.framebufferResized, options.enableDefaultCameraInput);
}

void EngineCore::initVulkan() {
    // Ordering matters here:
    //  1. vulkan → swapchain: surface must exist before swapchain creation.
    //  2. frames → createDescriptorPool: commandPool must exist before ResourceManager.
    //  3. ResourceManager receives a reference to descriptorPool, so the pool must be alive
    //     for the entire lifetime of the ResourceManager.
    //  4. Descriptor set layouts must precede pipeline creation.
    //  5. Descriptor sets must be written after both pool and uniform buffers/images exist.
    vulkan.init(window);
    vulkanInitialized = true;
    swapchain.init(vulkan, window);
    imagesInFlight.assign(swapchain.images.size(), vk::Fence{});
    frames.init(vulkan, swapchain);
    createDescriptorPool();

    resourceManager = std::make_unique<ResourceManager>(vulkan.logicalDevice, vulkan.physicalDevice, frames.commandPool, vulkan.queue,
                                                        descriptorPool);
    scene = std::make_unique<Scene>();
    constexpr float bounds = Laphria::EngineConfig::kDefaultSceneBoundsExtent;
    scene->init({{-bounds, -bounds, -bounds}, {bounds, bounds, bounds}});

    physicsSystem = std::make_unique<PhysicsSystem>();

    pipelines.createDescriptorSetLayouts(vulkan);

    // Pipeline creation order matches dependency on the descriptor set layouts above.
    pipelines.createGraphicsPipeline(vulkan, swapchain.surfaceFormat.format, vulkan.findDepthFormat());
    pipelines.createShadowPipeline(vulkan);
    pipelines.createComputePipeline(vulkan);
    pipelines.createSkinningPipeline(vulkan);
    pipelines.createPhysicsPipeline(vulkan);
    pipelines.createRayTracingPipeline(vulkan);
    pipelines.createShaderBindingTable(vulkan);
    pipelines.createDenoiserPipelines(vulkan);
    pipelines.createClassicRTPipeline(vulkan);
    pipelines.createClassicRTShaderBindingTable(vulkan);

    resourceManager->setSkinningDescriptorSetLayout(*pipelines.skinningDescriptorSetLayout);

    createDescriptorSets();
    createComputeDescriptorSets();
    createPhysicsDescriptorSets();
    createRayTracingDescriptorSets();
    createDenoiserDescriptorSets();
    createTimestampQueryPool();
}

void EngineCore::initImgui() {
    ui.init(vulkan, window, swapchain.surfaceFormat.format, vulkan.findDepthFormat());
    imguiInitialized = true;
}

void EngineCore::mainLoop() {
    size_t prevModelCount = resourceManager->getModelCount();

    while (!glfwWindowShouldClose(window)) {
        // Delta Time calculation
        auto currentTime = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float>(currentTime - lastFrameTime).count();
        lastFrameTime = currentTime;
        updatePerformanceWindowTitle(deltaTime);

        glfwPollEvents();
        camera.update(deltaTime);

        std::optional<EngineServices> services;
        if (scene && physicsSystem && resourceManager) {
            services.emplace(buildServices());
        }

        if (callbacks.updateFrame && services.has_value()) {
            auto &servicesRef = *services;
            callbacks.updateFrame(servicesRef, deltaTime);
        }
        if (scene && resourceManager) {
            scene->update(deltaTime, *resourceManager);
        }

        // Physics Update
        if (options.runPhysicsSimulation && ui.simulationRunning && physicsSystem) {
            auto start = std::chrono::high_resolution_clock::now();

            if (ui.useGPUPhysics) {
                auto cmd = VulkanUtils::beginSingleTimeCommands(vulkan.logicalDevice, frames.commandPool);
                physicsSystem->updateGPU(scene->getAllNodes(), deltaTime, cmd, pipelines.physicsPipelineLayout, pipelines.physicsPipeline, physicsDescriptorSet);
                cmd.end();

                vk::raii::Fence physicsFence(vulkan.logicalDevice, vk::FenceCreateInfo{});
                vk::SubmitInfo submitInfo{};
                submitInfo.commandBufferCount = 1;
                submitInfo.pCommandBuffers = &*cmd;
                vulkan.queue.submit(submitInfo, *physicsFence);

                const vk::Result waitResult = vulkan.logicalDevice.waitForFences(*physicsFence, vk::True, UINT64_MAX);
                if (waitResult != vk::Result::eSuccess) {
                    throw std::runtime_error("failed to wait for GPU physics fence");
                }

                // Readback immediately
                physicsSystem->syncFromGPU(scene->getAllNodes());
            } else {
                physicsSystem->updateCPU(scene->getAllNodes(), deltaTime);
            }

            auto end = std::chrono::high_resolution_clock::now();
            ui.physicsTime = std::chrono::duration<float, std::milli>(end - start).count();
        }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ui.draw(window, *scene, *physicsSystem, *resourceManager, *pipelines.descriptorSetLayoutMaterial, camera);
        if (callbacks.drawUi && services.has_value()) {
            auto &servicesRef = *services;
            callbacks.drawUi(servicesRef);
        }

        if (scene) {
            scene->syncSpatialIndex();
        }

        // If models were loaded during the UI frame, the RT descriptor sets (bindings 5-8:
        // vertex/index/material/texture arrays) must be rebuilt to include the new buffers.
        // buildBLAS already called queue.waitIdle(), so the queue is idle here.
        size_t currentModelCount = resourceManager->getModelCount();
        if (currentModelCount != prevModelCount) {
            prevModelCount = currentModelCount;
            createRayTracingDescriptorSets();
        }

        ImGui::Render();

        drawFrame();
    }

    vulkan.logicalDevice.waitIdle();
}

void EngineCore::updatePerformanceWindowTitle(float deltaTimeSeconds)
{
    if (!window || deltaTimeSeconds <= 0.0f) {
        return;
    }

    titleStatsAccumSeconds += static_cast<double>(deltaTimeSeconds);
    ++titleStatsFrameCount;

    if (titleStatsAccumSeconds < kWindowTitleUpdateIntervalSeconds || titleStatsFrameCount == 0) {
        return;
    }

    const double averageFrameTimeSeconds = titleStatsAccumSeconds / static_cast<double>(titleStatsFrameCount);
    const double fps = 1.0 / averageFrameTimeSeconds;
    const double frameTimeMs = averageFrameTimeSeconds * 1000.0;

    char titleBuffer[256];
    std::snprintf(titleBuffer, sizeof(titleBuffer), "%s | %.1f FPS | %.2f ms",
                  options.windowTitle.c_str(), fps, frameTimeMs);
    glfwSetWindowTitle(window, titleBuffer);

    titleStatsAccumSeconds = 0.0;
    titleStatsFrameCount = 0;
}

void EngineCore::cleanupSwapChain() {
    swapchain.cleanup();
    frames.cleanupSwapChainDependents();
}

void EngineCore::cleanup() {
    if (imguiInitialized) {
        ui.cleanup();
        imguiInitialized = false;
    }

    if (windowInitialized && window) {
        glfwDestroyWindow(window);
        window = nullptr;
    }
    if (windowInitialized) {
        glfwTerminate();
        windowInitialized = false;
    }
}

void EngineCore::recreateSwapChain() {
    // A zero-sized framebuffer means the window is minimized; block here until it is restored.
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }

    vulkan.logicalDevice.waitIdle();

    cleanupSwapChain();
    swapchain.init(vulkan, window);
    imagesInFlight.assign(swapchain.images.size(), vk::Fence{});
    frames.recreate(vulkan, swapchain);
    // Compute, RT, and denoiser descriptor sets reference images that are recreated above
    // (storageImages, rayTracingOutputImages, and G-Buffer images are extent-dependent),
    // so all three must be rewritten after frames.recreate().
    createComputeDescriptorSets();
    createRayTracingDescriptorSets();
    createDenoiserDescriptorSets();
}

void EngineCore::createPhysicsDescriptorSets() {
    vk::DescriptorPoolSize poolSize{vk::DescriptorType::eStorageBuffer, 1};
    vk::DescriptorPoolCreateInfo poolInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize
    };
    physicsDescriptorPool = vk::raii::DescriptorPool(vulkan.logicalDevice, poolInfo);

    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *physicsDescriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &*pipelines.physicsDescriptorSetLayout
    };

    physicsDescriptorSet = std::move(vulkan.logicalDevice.allocateDescriptorSets(allocInfo)[0]);

    // Create SSBO
    constexpr size_t maxObjects = Laphria::EngineConfig::kMaxPhysicsObjects;
    physicsSystem->createSSBO(vulkan.logicalDevice, vulkan.physicalDevice, maxObjects * sizeof(PhysicsObject));

    // Bind SSBO to Set
    vk::DescriptorBufferInfo bufferInfo{
        .buffer = *physicsSystem->getSSBOBuffer(),
        .offset = 0,
        .range = maxObjects * sizeof(PhysicsObject)
    };

    vk::WriteDescriptorSet writeDescriptorSet{
        .dstSet = *physicsDescriptorSet,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .pBufferInfo = &bufferInfo
    };

    vulkan.logicalDevice.updateDescriptorSets(writeDescriptorSet, nullptr);
}

void EngineCore::createComputeDescriptorSets() {
    // One set per Frame In Flight (matching storage images)
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *pipelines.computeDescriptorSetLayout);

    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *descriptorPool, // Use the same global pool
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };

    computeDescriptorSets.clear();
    computeDescriptorSets = vulkan.logicalDevice.allocateDescriptorSets(allocInfo);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::DescriptorImageInfo imageInfo{
            .imageView = *frames.storageImageViews[i],
            .imageLayout = vk::ImageLayout::eGeneral // Compute shader writes to General layout
        };

        vk::WriteDescriptorSet storageImageWrite{
            .dstSet = *computeDescriptorSets[i],
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .pImageInfo = &imageInfo
        };

        vulkan.logicalDevice.updateDescriptorSets(storageImageWrite, {});
    }
}

void EngineCore::createRayTracingDescriptorSets() {
    // One set per frame in flight; bindings shifted to accommodate the new G-Buffer images.
    // RT set bindings: 0 = TLAS, 1 = noisy colour, 2 = normals, 3 = depth, 4 = motion vectors,
    //                  5 = vertex arrays, 6 = index arrays, 7 = material arrays, 8 = texture array.
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *pipelines.rayTracingDescriptorSetLayout);

    std::vector<uint32_t> variableDescCounts(MAX_FRAMES_IN_FLIGHT, Laphria::EngineConfig::kBindlessModelCapacity);
    vk::DescriptorSetVariableDescriptorCountAllocateInfo variableDescCountInfo{
        .descriptorSetCount = MAX_FRAMES_IN_FLIGHT,
        .pDescriptorCounts = variableDescCounts.data()
    };

    vk::DescriptorSetAllocateInfo allocInfo{
        .pNext = &variableDescCountInfo,
        .descriptorPool = *descriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };

    rtDescriptorSets.clear();
    rtDescriptorSets = vulkan.logicalDevice.allocateDescriptorSets(allocInfo);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        // Binding 0 — TLAS.
        // The TLAS write requires a WriteDescriptorSetAccelerationStructureKHR in pNext;
        // it cannot use pBufferInfo or pImageInfo like every other descriptor type.
        vk::WriteDescriptorSetAccelerationStructureKHR tlasInfo{
            .accelerationStructureCount = 1,
            .pAccelerationStructures = &*frames.tlas[i]
        };

        vk::WriteDescriptorSet tlasWrite{
            .pNext = &tlasInfo,
            .dstSet = *rtDescriptorSets[i],
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eAccelerationStructureKHR
        };

        // Binding 1 — noisy colour output (written by the raygen shader in General layout).
        vk::DescriptorImageInfo rtOutputImageInfo{
            .imageView = *frames.rayTracingOutputImageViews[i],
            .imageLayout = vk::ImageLayout::eGeneral
        };
        vk::WriteDescriptorSet rtOutputWrite{
            .dstSet = *rtDescriptorSets[i], .dstBinding = 1, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &rtOutputImageInfo
        };

        // Binding 2 — G-Buffer world normals.
        vk::DescriptorImageInfo normalsInfo{.imageView = *frames.rtGBufferNormalsViews[i], .imageLayout = vk::ImageLayout::eGeneral};
        vk::WriteDescriptorSet normalsWrite{
            .dstSet = *rtDescriptorSets[i], .dstBinding = 2, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &normalsInfo
        };

        // Binding 3 — G-Buffer linear depth.
        vk::DescriptorImageInfo depthInfo{.imageView = *frames.rtGBufferDepthViews[i], .imageLayout = vk::ImageLayout::eGeneral};
        vk::WriteDescriptorSet depthWrite{
            .dstSet = *rtDescriptorSets[i], .dstBinding = 3, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &depthInfo
        };

        // Binding 4 — motion vectors.
        vk::DescriptorImageInfo mvInfo{.imageView = *frames.rtMotionVectorsViews[i], .imageLayout = vk::ImageLayout::eGeneral};
        vk::WriteDescriptorSet mvWrite{
            .dstSet = *rtDescriptorSets[i], .dstBinding = 4, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &mvInfo
        };

        std::vector<vk::WriteDescriptorSet> descriptorWrites;
        descriptorWrites.push_back(tlasWrite);
        descriptorWrites.push_back(rtOutputWrite);
        descriptorWrites.push_back(normalsWrite);
        descriptorWrites.push_back(depthWrite);
        descriptorWrites.push_back(mvWrite);

        // Now we extract ALL global vertices, indices, materials, and textures
        // across all Scene Nodes that have been uploaded into VRAM by ResourceManager
        std::vector<vk::DescriptorBufferInfo> vertexInfos;
        std::vector<vk::DescriptorBufferInfo> indexInfos;
        std::vector<vk::DescriptorBufferInfo> materialInfos;
        std::vector<vk::DescriptorImageInfo> textureInfos;

        // Since our ResourceManager stores ModelResource objects linearly in ID...
        // In a production engine, this would be an iterative flat map or array
        constexpr int totalModels = static_cast<int>(Laphria::EngineConfig::kBindlessModelCapacity);
        for (int modelId = 0; modelId < totalModels; ++modelId) {
            if (ModelResource *model = resourceManager->getModelResource(modelId)) {
                // Writing a null VkBuffer into a descriptor is invalid even with ePartiallyBound.
                if (!*model->vertexBuffer || !*model->indexBuffer || !*model->materialBuffer)
                    throw std::runtime_error("RT descriptor: model " + std::to_string(modelId) + " has null buffer(s)");

                // 1. Accumulate Vertex Buffers (use skinned stream for RT/PT when available)
                const vk::Buffer rtVertexBuffer = (model->hasRuntimeSkinning && *model->skinnedVertexBuffer) ? *model->skinnedVertexBuffer : *model->vertexBuffer;
                vertexInfos.push_back({rtVertexBuffer, 0, VK_WHOLE_SIZE});

                // 2. Accumulate Index Buffers
                indexInfos.push_back({*model->indexBuffer, 0, VK_WHOLE_SIZE});

                // 3. Accumulate Material Buffers
                materialInfos.push_back({*model->materialBuffer, 0, VK_WHOLE_SIZE});

                // 4. Accumulate Textures — pair each view with its own sampler.
                for (size_t texIdx = 0; texIdx < model->textureImageViews.size(); ++texIdx) {
                    textureInfos.push_back({*model->textureSamplers[texIdx], *model->textureImageViews[texIdx], vk::ImageLayout::eShaderReadOnlyOptimal});
                }
            } else {
                break; // Stop at the first empty ID
            }
        }

        if (!vertexInfos.empty()) {
            descriptorWrites.push_back(vk::WriteDescriptorSet{
                .dstSet = *rtDescriptorSets[i],
                .dstBinding = 5,
                .dstArrayElement = 0,
                .descriptorCount = static_cast<uint32_t>(vertexInfos.size()),
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = vertexInfos.data()
            });
        }

        if (!indexInfos.empty()) {
            descriptorWrites.push_back(vk::WriteDescriptorSet{
                .dstSet = *rtDescriptorSets[i],
                .dstBinding = 6,
                .dstArrayElement = 0,
                .descriptorCount = static_cast<uint32_t>(indexInfos.size()),
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = indexInfos.data()
            });
        }

        if (!materialInfos.empty()) {
            descriptorWrites.push_back(vk::WriteDescriptorSet{
                .dstSet = *rtDescriptorSets[i],
                .dstBinding = 7,
                .dstArrayElement = 0,
                .descriptorCount = static_cast<uint32_t>(materialInfos.size()),
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = materialInfos.data()
            });
        }

        if (!textureInfos.empty()) {
            descriptorWrites.push_back(vk::WriteDescriptorSet{
                .dstSet = *rtDescriptorSets[i],
                .dstBinding = 8,
                .dstArrayElement = 0,
                .descriptorCount = static_cast<uint32_t>(textureInfos.size()),
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = textureInfos.data()
            });
        }

        vulkan.logicalDevice.updateDescriptorSets(descriptorWrites, {});
    }
}

void EngineCore::createDenoiserDescriptorSets() {
    // One set per frame in flight. All 13 bindings are storage images.
    std::vector<vk::DescriptorPoolSize> poolSizes = {
        {vk::DescriptorType::eStorageImage, 13 * MAX_FRAMES_IN_FLIGHT}
    };
    vk::DescriptorPoolCreateInfo poolInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = MAX_FRAMES_IN_FLIGHT,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()
    };
    denoiserDescriptorPool = vk::raii::DescriptorPool(vulkan.logicalDevice, poolInfo);

    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *pipelines.denoiserDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *denoiserDescriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };
    denoiserDescriptorSets.clear();
    denoiserDescriptorSets = vulkan.logicalDevice.allocateDescriptorSets(allocInfo);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        size_t prevSlot = (i - 1 + MAX_FRAMES_IN_FLIGHT) % MAX_FRAMES_IN_FLIGHT;
        const size_t atrousBase = i * 2;

        // Build the 13 image info structs in binding order.
        vk::DescriptorImageInfo infos[13] = {
            {.imageView = *frames.rayTracingOutputImageViews[i], .imageLayout = vk::ImageLayout::eGeneral}, // 0: noisy colour
            {.imageView = *frames.rtGBufferNormalsViews[i], .imageLayout = vk::ImageLayout::eGeneral}, // 1: current normals
            {.imageView = *frames.rtGBufferDepthViews[i], .imageLayout = vk::ImageLayout::eGeneral}, // 2: current depth
            {.imageView = *frames.rtMotionVectorsViews[i], .imageLayout = vk::ImageLayout::eGeneral}, // 3: motion vectors
            {.imageView = *frames.historyColorViews[prevSlot], .imageLayout = vk::ImageLayout::eGeneral}, // 4: history colour read
            {.imageView = *frames.historyColorViews[i], .imageLayout = vk::ImageLayout::eGeneral}, // 5: history colour write
            {.imageView = *frames.historyMomentsViews[prevSlot], .imageLayout = vk::ImageLayout::eGeneral}, // 6: history moments read
            {.imageView = *frames.historyMomentsViews[i], .imageLayout = vk::ImageLayout::eGeneral}, // 7: history moments write
            {.imageView = *frames.atrousTempViews[atrousBase + 0], .imageLayout = vk::ImageLayout::eGeneral}, // 8: A-Trous buffer A
            {.imageView = *frames.atrousTempViews[atrousBase + 1], .imageLayout = vk::ImageLayout::eGeneral}, // 9: A-Trous buffer B
            {.imageView = *frames.rayTracingOutputImageViews[i], .imageLayout = vk::ImageLayout::eGeneral}, // 10: final denoised output (reuses slot 0 image)
            {.imageView = *frames.rtGBufferNormalsViews[prevSlot], .imageLayout = vk::ImageLayout::eGeneral}, // 11: previous-frame normals
            {.imageView = *frames.rtGBufferDepthViews[prevSlot], .imageLayout = vk::ImageLayout::eGeneral}, // 12: previous-frame depth
        };

        std::vector<vk::WriteDescriptorSet> writes;
        writes.reserve(13);
        for (uint32_t b = 0; b < 13; ++b) {
            writes.push_back(vk::WriteDescriptorSet{
                .dstSet = *denoiserDescriptorSets[i],
                .dstBinding = b,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageImage,
                .pImageInfo = &infos[b]
            });
        }
        vulkan.logicalDevice.updateDescriptorSets(writes, {});
    }
}

void EngineCore::recordComputeCommandBuffer(const vk::raii::CommandBuffer &commandBuffer, uint32_t imageIndex) const {
    // 1. Execution Barrier — General Layout for Compute Write
    // eGeneral→eGeneral: no content discard; waits for the previous frame's TRANSFER_SRC→eGeneral
    // restore (or the one-time creation pre-transition) before the compute shader writes.
    transition_image_layout(
        *frames.storageImages[frames.frameIndex],
        vk::ImageLayout::eGeneral,
        vk::ImageLayout::eGeneral,
        {},
        vk::AccessFlagBits2::eShaderWrite,
        vk::PipelineStageFlagBits2::eTransfer, // Wait for the previous frame's restore
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::ImageAspectFlagBits::eColor);

    // 2. Compute Dispatch
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelines.computePipeline);

    // Bind Set 0 (storage image) — the simplified layout only exposes this one set.
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelines.computePipelineLayout, 0,
                                     *computeDescriptorSets[frames.frameIndex], nullptr);

    Laphria::ScenePushConstants push{};
    push.skyData = glm::vec4(0.01f, 0.03f, 0.1f, 0.99f);

    commandBuffer.pushConstants<Laphria::ScenePushConstants>(*pipelines.computePipelineLayout,
                                                             vk::ShaderStageFlagBits::eCompute,
                                                             0, push);

    // Dispatch
    // Workgroup size is 16x16.
    uint32_t groupCountX = (swapchain.extent.width + 15) / 16;
    uint32_t groupCountY = (swapchain.extent.height + 15) / 16;
    commandBuffer.dispatch(groupCountX, groupCountY, 1);

    // 3. Blit Storage Image -> SwapChain Image

    // Transition Storage Image: General -> TransferSrc
    transition_image_layout(
        *frames.storageImages[frames.frameIndex],
        vk::ImageLayout::eGeneral,
        vk::ImageLayout::eTransferSrcOptimal,
        vk::AccessFlagBits2::eShaderWrite,
        vk::AccessFlagBits2::eTransferRead,
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::ImageAspectFlagBits::eColor);

    // Transition SwapChain Image: Undefined -> TransferDst
    transition_image_layout(
        swapchain.images[imageIndex],
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eTransferDstOptimal,
        {},
        vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eTopOfPipe,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::ImageAspectFlagBits::eColor);

    // Blit
    vk::ImageBlit blitRegion{
        .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .srcOffsets = {{vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<int32_t>(swapchain.extent.width), static_cast<int32_t>(swapchain.extent.height), 1}}},
        .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .dstOffsets = {{vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<int32_t>(swapchain.extent.width), static_cast<int32_t>(swapchain.extent.height), 1}}}
    };

    commandBuffer.blitImage(*frames.storageImages[frames.frameIndex], vk::ImageLayout::eTransferSrcOptimal,
                            swapchain.images[imageIndex], vk::ImageLayout::eTransferDstOptimal,
                            blitRegion, vk::Filter::eLinear);

    // 3b. Restore storage image to eGeneral so it always matches the layout declared in
    // computeDescriptorSets. This prevents VUID-vkCmdDraw-None-09600 when the rasterizer's
    // draw commands follow in the same command buffer.
    transition_image_layout(
        *frames.storageImages[frames.frameIndex],
        vk::ImageLayout::eTransferSrcOptimal,
        vk::ImageLayout::eGeneral,
        vk::AccessFlagBits2::eTransferRead,
        {},
        vk::PipelineStageFlagBits2::eTransfer,
        vk::PipelineStageFlagBits2::eBottomOfPipe,
        vk::ImageAspectFlagBits::eColor);

    // 4. Transition SwapChain to Color Attachment for Rendering
    transition_image_layout(
        swapchain.images[imageIndex],
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::AccessFlagBits2::eTransferWrite,
        vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentRead,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor);
}

void EngineCore::recordSkinningPass(const vk::raii::CommandBuffer &commandBuffer) const {
    std::unordered_map<int, const SceneNode *> instanceRootsByModel;
    for (const auto &node: scene->getAllNodes()) {
        if (!node || node->modelId < 0) {
            continue;
        }
        ModelResource *modelRes = resourceManager->getModelResource(node->modelId);
        if (!modelRes || !modelRes->hasRuntimeSkinning || !*modelRes->skinningDescriptorSet || !modelRes->skinningJointMatricesMapped) {
            continue;
        }
        const SceneNode *parent = node->getParent();
        const bool isInstanceRoot = (parent == nullptr || parent->modelId != node->modelId);
        if (isInstanceRoot && !instanceRootsByModel.contains(node->modelId)) {
            instanceRootsByModel.emplace(node->modelId, node.get());
        }
    }

    if (instanceRootsByModel.empty()) {
        return;
    }

    vk::MemoryBarrier2 hostToComputeBarrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eHost,
        .srcAccessMask = vk::AccessFlagBits2::eHostWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead};
    vk::DependencyInfo hostToComputeDependency{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &hostToComputeBarrier};
    commandBuffer.pipelineBarrier2(hostToComputeDependency);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelines.skinningPipeline);

    for (const auto &[modelId, rootNode] : instanceRootsByModel) {
        ModelResource *modelRes = resourceManager->getModelResource(modelId);
        if (!modelRes || modelRes->skinningJointMatrixCount == 0 || modelRes->skinningVertexCount == 0) {
            continue;
        }

        std::unordered_map<int, const SceneNode *> nodesBySourceIndex;
        std::vector<const SceneNode *> stack{rootNode};
        while (!stack.empty()) {
            const SceneNode *current = stack.back();
            stack.pop_back();
            if (!current || current->modelId != modelId) {
                continue;
            }
            if (current->sourceNodeIndex >= 0 && !nodesBySourceIndex.contains(current->sourceNodeIndex)) {
                nodesBySourceIndex.emplace(current->sourceNodeIndex, current);
            }
            for (const auto &child : current->getChildren()) {
                if (child) {
                    stack.push_back(child.get());
                }
            }
        }

        std::vector<glm::mat4> jointPalette(modelRes->skinningJointMatrixCount, glm::mat4(1.0f));
        for (const auto &skin : modelRes->skins) {
            for (size_t jointIndex = 0; jointIndex < skin.jointSourceNodeIndices.size(); ++jointIndex) {
                const uint32_t paletteIndex = skin.jointMatrixOffset + static_cast<uint32_t>(jointIndex);
                if (paletteIndex >= jointPalette.size()) {
                    continue;
                }
                const auto nodeIt = nodesBySourceIndex.find(skin.jointSourceNodeIndices[jointIndex]);
                if (nodeIt == nodesBySourceIndex.end() || !nodeIt->second) {
                    continue;
                }
                jointPalette[paletteIndex] = nodeIt->second->getWorldTransform() * skin.inverseBindMatrices[jointIndex];
            }
        }

        memcpy(modelRes->skinningJointMatricesMapped, jointPalette.data(), sizeof(glm::mat4) * jointPalette.size());

        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelines.skinningPipelineLayout, 0, {*modelRes->skinningDescriptorSet}, nullptr);

        Laphria::SkinningPushConstants push{};
        push.vertexCount = modelRes->skinningVertexCount;
        push.jointMatrixOffset = 0;
        push.jointCount = modelRes->skinningJointMatrixCount;
        commandBuffer.pushConstants<Laphria::SkinningPushConstants>(*pipelines.skinningPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, push);

        const uint32_t groupCountX = (modelRes->skinningVertexCount + 63u) / 64u;
        commandBuffer.dispatch(groupCountX, 1, 1);
    }

    vk::MemoryBarrier2 skinningToConsumerBarrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eVertexInput | vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
        .dstAccessMask = vk::AccessFlagBits2::eVertexAttributeRead | vk::AccessFlagBits2::eAccelerationStructureReadKHR};
    vk::DependencyInfo skinningToConsumerDependency{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &skinningToConsumerBarrier};
    commandBuffer.pipelineBarrier2(skinningToConsumerDependency);

    if (ui.renderMode != RenderMode::Rasterizer) {
        resourceManager->recordSkinnedBLASRefit(commandBuffer);
    }
}

void EngineCore::recordClassicRTCommandBuffer(const vk::raii::CommandBuffer &commandBuffer, uint32_t imageIndex) const {
    const uint32_t fi = frames.frameIndex;

    // 1. Transition RT Output Image to General Layout for Writing
    transition_image_layout(
        *frames.rayTracingOutputImages[fi],
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eGeneral,
        {},
        vk::AccessFlagBits2::eShaderWrite,
        vk::PipelineStageFlagBits2::eTopOfPipe,
        vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
        vk::ImageAspectFlagBits::eColor);

    // 2. Bind Classic RT Pipeline and Descriptor Sets
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *pipelines.classicRTPipeline);
    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eRayTracingKHR,
        *pipelines.rayTracingPipelineLayout,
        0,
        {*rtDescriptorSets[fi], *descriptorSets[fi]},
        nullptr);

    ScenePushConstants pushConstants{};
    pushConstants.modelMatrix = glm::mat4(1.0f);
    commandBuffer.pushConstants<ScenePushConstants>(
        *pipelines.rayTracingPipelineLayout,
        vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eMissKHR,
        0,
        pushConstants);

    // 3. Dispatch Rays
    vk::StridedDeviceAddressRegionKHR callableRegion{};
    commandBuffer.traceRaysKHR(
        pipelines.classicRTRaygenRegion,
        pipelines.classicRTMissRegion,
        pipelines.classicRTHitRegion,
        callableRegion,
        swapchain.extent.width,
        swapchain.extent.height,
        1);

    // 4. Transition RT Output Image for Blit (General → TransferSrcOptimal)
    transition_image_layout(
        *frames.rayTracingOutputImages[fi],
        vk::ImageLayout::eGeneral,
        vk::ImageLayout::eTransferSrcOptimal,
        vk::AccessFlagBits2::eShaderWrite,
        vk::AccessFlagBits2::eTransferRead,
        vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::ImageAspectFlagBits::eColor);

    // 5. Transition SwapChain Image for Blit
    transition_image_layout(
        swapchain.images[imageIndex],
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eTransferDstOptimal,
        {},
        vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eTopOfPipe,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::ImageAspectFlagBits::eColor);

    // 6. Blit RT Output to SwapChain Image
    vk::ImageBlit blitRegion{
        .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .srcOffsets = {{vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<int32_t>(swapchain.extent.width), static_cast<int32_t>(swapchain.extent.height), 1}}},
        .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .dstOffsets = {{vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<int32_t>(swapchain.extent.width), static_cast<int32_t>(swapchain.extent.height), 1}}}
    };
    commandBuffer.blitImage(
        *frames.rayTracingOutputImages[fi], vk::ImageLayout::eTransferSrcOptimal,
        swapchain.images[imageIndex], vk::ImageLayout::eTransferDstOptimal,
        blitRegion, vk::Filter::eLinear);

    // 6b. Restore RT output image to eGeneral so it always matches the layout declared in
    // rtDescriptorSets and denoiserDescriptorSets (prevents VUID-vkCmdDraw-None-09600 if
    // the render mode is switched back to Rasterizer in a subsequent frame).
    transition_image_layout(
        *frames.rayTracingOutputImages[fi],
        vk::ImageLayout::eTransferSrcOptimal,
        vk::ImageLayout::eGeneral,
        vk::AccessFlagBits2::eTransferRead,
        {},
        vk::PipelineStageFlagBits2::eTransfer,
        vk::PipelineStageFlagBits2::eBottomOfPipe,
        vk::ImageAspectFlagBits::eColor);

    // 7. Transition SwapChain to Color Attachment for UI Rendering
    transition_image_layout(
        swapchain.images[imageIndex],
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::AccessFlagBits2::eTransferWrite,
        vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentRead,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor);
}

void EngineCore::recordRayTracingCommandBuffer(const vk::raii::CommandBuffer &commandBuffer, uint32_t imageIndex) const {
    const uint32_t fi = frames.frameIndex;
    const uint32_t queryBase = getPathTracerQueryBase(fi);
    const size_t atrousBase = static_cast<size_t>(fi) * 2;
    const size_t atrousA = atrousBase + 0;
    const size_t atrousB = atrousBase + 1;

    const float clampedScale = std::clamp(ui.pathTracerSettings.resolutionScale, 0.5f, 1.0f);
    const float secondaryScale = ui.pathTracerSettings.reduceSecondaryEffects ? 0.90f : 1.0f;
    const float effectiveScale = std::clamp(clampedScale * secondaryScale, 0.5f, 1.0f);
    const uint32_t rtWidth = std::max(1u, static_cast<uint32_t>(static_cast<float>(swapchain.extent.width) * effectiveScale));
    const uint32_t rtHeight = std::max(1u, static_cast<uint32_t>(static_cast<float>(swapchain.extent.height) * effectiveScale));
    const uint32_t gx = (rtWidth + 15) / 16;
    const uint32_t gy = (rtHeight + 15) / 16;

    // 1. Transition all PT images to general layout for writing.
    auto transitionToGeneral = [&](vk::Image img) {
        transition_image_layout(img, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
                                {}, vk::AccessFlagBits2::eShaderWrite,
                                vk::PipelineStageFlagBits2::eTopOfPipe, vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
                                vk::ImageAspectFlagBits::eColor);
    };
    transitionToGeneral(*frames.rayTracingOutputImages[fi]);
    transitionToGeneral(*frames.rtGBufferNormals[fi]);
    transitionToGeneral(*frames.rtGBufferDepth[fi]);
    transitionToGeneral(*frames.rtMotionVectors[fi]);
    transitionToGeneral(*frames.atrousTemp[atrousA]);
    transitionToGeneral(*frames.atrousTemp[atrousB]);

    // 2. Ray tracing dispatch.
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *pipelines.rayTracingPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR,
                                     *pipelines.rayTracingPipelineLayout, 0,
                                     {*rtDescriptorSets[fi], *descriptorSets[fi]}, nullptr);

    ScenePushConstants rtPush{};
    rtPush.modelMatrix = glm::mat4(1.0f);
    commandBuffer.pushConstants<ScenePushConstants>(*pipelines.rayTracingPipelineLayout,
                                                    vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eMissKHR,
                                                    0, rtPush);

    if (*ptTimestampQueryPool) {
        commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eRayTracingShaderKHR, *ptTimestampQueryPool, queryBase + kPtTS_RayTraceStart);
    }
    vk::StridedDeviceAddressRegionKHR callableRegion{};
    commandBuffer.traceRaysKHR(pipelines.raygenRegion, pipelines.missRegion, pipelines.hitRegion,
                               callableRegion, rtWidth, rtHeight, 1);
    if (*ptTimestampQueryPool) {
        commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eRayTracingShaderKHR, *ptTimestampQueryPool, queryBase + kPtTS_RayTraceEnd);
    }

    // 3. Barrier: RT writes -> compute reads.
    auto barrierRTtoCompute = [&](vk::Image img) {
        transition_image_layout(img, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
                                vk::AccessFlagBits2::eShaderWrite, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                                vk::PipelineStageFlagBits2::eRayTracingShaderKHR, vk::PipelineStageFlagBits2::eComputeShader,
                                vk::ImageAspectFlagBits::eColor);
    };
    barrierRTtoCompute(*frames.rayTracingOutputImages[fi]);
    barrierRTtoCompute(*frames.rtGBufferNormals[fi]);
    barrierRTtoCompute(*frames.rtGBufferDepth[fi]);
    barrierRTtoCompute(*frames.rtMotionVectors[fi]);

    // 4. Reprojection pass.
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelines.reprojectionPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                     *pipelines.denoiserPipelineLayout, 0, *denoiserDescriptorSets[fi], nullptr);

    float historyAlpha = ptCameraMoved ? 1.0f : 0.1f;
    DenoisePushConstants reproPush{
        .stepSize = 0,
        .isLastPass = 0,
        .phiColor = historyAlpha,
        .phiNormal = 128.0f,
        .exposureScale = ui.visualsV1.exposure};
    commandBuffer.pushConstants<DenoisePushConstants>(*pipelines.denoiserPipelineLayout,
                                                      vk::ShaderStageFlagBits::eCompute, 0, reproPush);
    if (*ptTimestampQueryPool) {
        commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, *ptTimestampQueryPool, queryBase + kPtTS_ReprojectionStart);
    }
    commandBuffer.dispatch(gx, gy, 1);
    if (*ptTimestampQueryPool) {
        commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, *ptTimestampQueryPool, queryBase + kPtTS_ReprojectionEnd);
    }

    auto barrierCompute = [&](vk::Image img) {
        transition_image_layout(img, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
                                vk::AccessFlagBits2::eShaderWrite, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                                vk::PipelineStageFlagBits2::eComputeShader, vk::PipelineStageFlagBits2::eComputeShader,
                                vk::ImageAspectFlagBits::eColor);
    };
    barrierCompute(*frames.atrousTemp[atrousA]);
    barrierCompute(*frames.historyMoments[fi]);

    // 5. A-Trous denoiser.
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelines.atrousPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                     *pipelines.denoiserPipelineLayout, 0, *denoiserDescriptorSets[fi], nullptr);

    const int atrousIterations = std::clamp(ui.pathTracerSettings.denoiserIterations, 1, 5);
    if (*ptTimestampQueryPool) {
        commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, *ptTimestampQueryPool, queryBase + kPtTS_DenoiserStart);
    }
    for (int iter = 0; iter < atrousIterations; ++iter) {
        const int32_t stepSize = 1 << iter;
        const int32_t isLastPass = (iter == atrousIterations - 1) ? 1 : 0;
        DenoisePushConstants atrousPush{
            .stepSize = stepSize,
            .isLastPass = isLastPass,
            .phiColor = 1.0f,
            .phiNormal = 128.0f,
            .exposureScale = ui.visualsV1.exposure};
        commandBuffer.pushConstants<DenoisePushConstants>(*pipelines.denoiserPipelineLayout,
                                                          vk::ShaderStageFlagBits::eCompute, 0, atrousPush);
        commandBuffer.dispatch(gx, gy, 1);

        if (!isLastPass) {
            const int writeBuf = iter % 2;
            barrierCompute(*frames.atrousTemp[(writeBuf == 0) ? atrousB : atrousA]);
        }
    }
    if (*ptTimestampQueryPool) {
        commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, *ptTimestampQueryPool, queryBase + kPtTS_DenoiserEnd);
    }

    // 6. Blit denoised image to swapchain.
    transition_image_layout(*frames.rayTracingOutputImages[fi],
                            vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal,
                            vk::AccessFlagBits2::eShaderWrite, vk::AccessFlagBits2::eTransferRead,
                            vk::PipelineStageFlagBits2::eComputeShader, vk::PipelineStageFlagBits2::eTransfer,
                            vk::ImageAspectFlagBits::eColor);

    transition_image_layout(swapchain.images[imageIndex],
                            vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
                            {}, vk::AccessFlagBits2::eTransferWrite,
                            vk::PipelineStageFlagBits2::eTopOfPipe, vk::PipelineStageFlagBits2::eTransfer,
                            vk::ImageAspectFlagBits::eColor);

    vk::ImageBlit blitRegion{
        .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .srcOffsets = {{vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<int32_t>(rtWidth), static_cast<int32_t>(rtHeight), 1}}},
        .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .dstOffsets = {{vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<int32_t>(swapchain.extent.width), static_cast<int32_t>(swapchain.extent.height), 1}}}
    };
    commandBuffer.blitImage(*frames.rayTracingOutputImages[fi], vk::ImageLayout::eTransferSrcOptimal,
                            swapchain.images[imageIndex], vk::ImageLayout::eTransferDstOptimal, blitRegion, vk::Filter::eLinear);

    transition_image_layout(*frames.rayTracingOutputImages[fi],
                            vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral,
                            vk::AccessFlagBits2::eTransferRead, {},
                            vk::PipelineStageFlagBits2::eTransfer, vk::PipelineStageFlagBits2::eBottomOfPipe,
                            vk::ImageAspectFlagBits::eColor);

    transition_image_layout(swapchain.images[imageIndex],
                            vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eColorAttachmentOptimal,
                            vk::AccessFlagBits2::eTransferWrite, vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentRead,
                            vk::PipelineStageFlagBits2::eTransfer, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                            vk::ImageAspectFlagBits::eColor);
}

void EngineCore::createDescriptorPool() {
    // Generous pool sizes to accommodate an arbitrary number of loaded models.
    // eSampledImage / eSampler are separate because the shadow map binding uses them
    // as distinct descriptor types (binding 1 and 2 in the global layout).
    constexpr uint32_t poolScale = Laphria::EngineConfig::kDescriptorPoolScale;
    std::array<vk::DescriptorPoolSize, 7> poolSizes = {
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, poolScale},
        // 1000 per loaded model (material textures) + 2×1000 for the two RT descriptor sets.
        vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 5 * poolScale},
        vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, poolScale},
        vk::DescriptorPoolSize{vk::DescriptorType::eSampler, poolScale},
        // 1000 for materials + vertex and index buffers * MAX_FRAMES
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 15 * poolScale},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, poolScale},
        vk::DescriptorPoolSize{vk::DescriptorType::eAccelerationStructureKHR, MAX_FRAMES_IN_FLIGHT}
    };

    vk::DescriptorPoolCreateInfo poolInfo{
        // eFreeDescriptorSet: allows individual sets to be freed (needed by ResourceManager).
        // eUpdateAfterBind: required for bindless descriptor indexing (VK_EXT_descriptor_indexing).
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet |
                 vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind,
        .maxSets = poolScale * MAX_FRAMES_IN_FLIGHT,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()
    };
    descriptorPool = vk::raii::DescriptorPool(vulkan.logicalDevice, poolInfo);
}

void EngineCore::createDescriptorSets() {
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *pipelines.descriptorSetLayoutGlobal);

    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *descriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };

    descriptorSets.clear();
    descriptorSets = vulkan.logicalDevice.allocateDescriptorSets(allocInfo);

    // Global descriptor set layout (Set 0):
    //   binding 0 → UniformBufferObject  (view/proj/light/cascade matrices, camera pos)
    //   binding 1 → shadow depth array   (sampled, ShaderReadOnlyOptimal)
    //   binding 2 → shadow PCF sampler   (comparison sampler)
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::DescriptorBufferInfo bufferInfo{
            .buffer = *frames.uniformBuffers[i],
            .offset = 0,
            .range = sizeof(Laphria::UniformBufferObject)
        };

        vk::WriteDescriptorSet uboWrite{
            .dstSet = *descriptorSets[i],
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo = &bufferInfo
        };

        // The shadow array image starts in eUndefined; we use eShaderReadOnlyOptimal
        // as the declared layout here because the first frame's shadow pass will
        // transition it via eUndefined → eDepthAttachmentOptimal → eShaderReadOnlyOptimal
        // before the main pass samples it.
        vk::DescriptorImageInfo shadowImageInfo{
            .imageView = *frames.shadowArrayViews[i],
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
        };

        vk::WriteDescriptorSet shadowImageWrite{
            .dstSet = *descriptorSets[i],
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eSampledImage,
            .pImageInfo = &shadowImageInfo
        };

        vk::DescriptorImageInfo shadowSamplerInfo{
            .sampler = *frames.shadowSampler
        };

        vk::WriteDescriptorSet shadowSamplerWrite{
            .dstSet = *descriptorSets[i],
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eSampler,
            .pImageInfo = &shadowSamplerInfo
        };

        std::array<vk::WriteDescriptorSet, 3> writes = {uboWrite, shadowImageWrite, shadowSamplerWrite};
        vulkan.logicalDevice.updateDescriptorSets(writes, {});
    }
}

void EngineCore::createTimestampQueryPool() {
    vk::QueryPoolCreateInfo queryPoolInfo{
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * kPtTimestampQueryCountPerFrame};
    ptTimestampQueryPool = vk::raii::QueryPool(vulkan.logicalDevice, queryPoolInfo);
    timestampPeriodNs = vulkan.physicalDevice.getProperties().limits.timestampPeriod;
}

uint32_t EngineCore::getPathTracerQueryBase(uint32_t frameSlot) const {
    return frameSlot * kPtTimestampQueryCountPerFrame;
}

void EngineCore::collectPathTracerTimings(uint32_t frameSlot) {
    if (!*ptTimestampQueryPool || !ptTimestampsValid[frameSlot]) {
        return;
    }

    std::array<uint64_t, kPtTimestampQueryCountPerFrame> timestamps{};
    const VkResult queryResult = vkGetQueryPoolResults(
        static_cast<VkDevice>(*vulkan.logicalDevice),
        static_cast<VkQueryPool>(*ptTimestampQueryPool),
        getPathTracerQueryBase(frameSlot),
        kPtTimestampQueryCountPerFrame,
        sizeof(timestamps),
        timestamps.data(),
        sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT);

    if (queryResult != VK_SUCCESS) {
        return;
    }

    auto toMs = [this](uint64_t start, uint64_t end) -> float {
        if (end <= start) {
            return 0.0f;
        }
        const double deltaTicks = static_cast<double>(end - start);
        const double deltaNs = deltaTicks * static_cast<double>(timestampPeriodNs);
        return static_cast<float>(deltaNs * 1e-6);
    };

    ui.pathTracerPerfStats.tlasBuildMs = toMs(timestamps[kPtTS_TlasStart], timestamps[kPtTS_TlasEnd]);
    ui.pathTracerPerfStats.rayTraceMs = toMs(timestamps[kPtTS_RayTraceStart], timestamps[kPtTS_RayTraceEnd]);
    ui.pathTracerPerfStats.reprojectionMs = toMs(timestamps[kPtTS_ReprojectionStart], timestamps[kPtTS_ReprojectionEnd]);
    ui.pathTracerPerfStats.denoiserMs = toMs(timestamps[kPtTS_DenoiserStart], timestamps[kPtTS_DenoiserEnd]);

    const uint64_t totalStart = (timestamps[kPtTS_TlasStart] != 0) ? timestamps[kPtTS_TlasStart] : timestamps[kPtTS_RayTraceStart];
    const uint64_t totalEnd = (timestamps[kPtTS_DenoiserEnd] != 0) ? timestamps[kPtTS_DenoiserEnd] : timestamps[kPtTS_RayTraceEnd];
    ui.pathTracerPerfStats.totalFrameMs = toMs(totalStart, totalEnd);
}

void EngineCore::updateAdaptivePathTracerSettings() {
    if (ui.pathTracerSettings.qualityMode == UISystem::PathTracerQualityMode::Manual) {
        return;
    }

    const float frameMs = ui.pathTracerPerfStats.totalFrameMs;
    if (frameMs <= 0.0f) {
        return;
    }

    const bool aggressive = (ui.pathTracerSettings.qualityMode == UISystem::PathTracerQualityMode::AutoAggressive);
    const float targetMs = std::max(8.0f, ui.pathTracerSettings.targetFrameMs);
    const float dropMargin = aggressive ? 0.25f : 0.75f;
    const float raiseMargin = aggressive ? 2.5f : 1.5f;

    if (frameMs > targetMs + dropMargin) {
        if (ui.pathTracerSettings.denoiserIterations > 1) {
            ui.pathTracerSettings.denoiserIterations -= 1;
        } else {
            ui.pathTracerSettings.resolutionScale = std::max(0.50f, ui.pathTracerSettings.resolutionScale - 0.05f);
        }
        return;
    }

    if (frameMs < targetMs - raiseMargin) {
        if (ui.pathTracerSettings.resolutionScale < 1.0f) {
            ui.pathTracerSettings.resolutionScale = std::min(1.0f, ui.pathTracerSettings.resolutionScale + 0.05f);
        } else if (ui.pathTracerSettings.denoiserIterations < 5) {
            ui.pathTracerSettings.denoiserIterations += 1;
        }
    }
}

void EngineCore::recordCommandBuffer(uint32_t imageIndex) const {
    auto &commandBuffer = frames.commandBuffers[frames.frameIndex];
    const uint32_t queryBase = getPathTracerQueryBase(frames.frameIndex);
    if (*ptTimestampQueryPool) {
        commandBuffer.resetQueryPool(*ptTimestampQueryPool, queryBase, kPtTimestampQueryCountPerFrame);
    }

    vk::ClearValue clearColor = vk::ClearColorValue(0.02f, 0.02f, 0.02f, 1.0f);
    if (ui.renderMode == RenderMode::Rasterizer) {
        // V1.3: raster path uses direct atmospheric clear color (no compute sky prepass).
        clearColor = vk::ClearColorValue(0.60f, 0.64f, 0.72f, 1.0f);
    }

    recordSkinningPass(commandBuffer);

    // --- Build TLAS ---
    std::vector<vk::AccelerationStructureInstanceKHR> tlasInstances;
    uint32_t nodesWithModelId = 0;
    uint32_t nodesMissingModelResource = 0;
    uint32_t nodesWithNoBlas = 0;
    uint32_t meshRefs = 0;
    uint32_t skippedInvalidMeshRef = 0;

    for (const auto &node: scene->getAllNodes()) {
        if (node->modelId >= 0) {
            ++nodesWithModelId;
            ModelResource *modelRes = resourceManager->getModelResource(node->modelId);
            if (!modelRes) {
                ++nodesMissingModelResource;
                continue;
            }
            if (modelRes->blasElements.empty()) {
                ++nodesWithNoBlas;
                continue;
            }

            glm::mat4 transform = node->getWorldTransform();

            // Convert to vk::TransformMatrixKHR (3x4 row-major array)
            vk::TransformMatrixKHR transformMatrix;
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 4; ++c) {
                    transformMatrix.matrix[r][c] = transform[c][r]; // GLM is column-major
                }
            }

            for (int meshIdx: node->getMeshIndices()) {
                ++meshRefs;
                if (meshIdx < 0 || meshIdx >= modelRes->blasElements.size())
                {
                    ++skippedInvalidMeshRef;
                    continue;
                }

                auto &blas = modelRes->blasElements[meshIdx];

                uint32_t primitiveOffset = 0;
                for (int i = 0; i < meshIdx; ++i) {
                    primitiveOffset += modelRes->meshes[i].primitives.size();
                }

                // Encode modelId in top 10 bits, primitiveOffset in bottom 14 bits
                // InstanceCustomIndex is exactly 24 bit in size.
                assert(node->modelId < 1024 && "modelId exceeds 10-bit limit; customIndex encoding will be corrupted");
                uint32_t customIndex = (node->modelId << 14) | (primitiveOffset & 0x3FFF);

                vk::AccelerationStructureDeviceAddressInfoKHR addressInfo{};
                addressInfo.accelerationStructure = *blas;
                vk::DeviceAddress blasAddress = vulkan.logicalDevice.getAccelerationStructureAddressKHR(addressInfo);

                vk::AccelerationStructureInstanceKHR instance{};
                instance.transform = transformMatrix;
                instance.instanceCustomIndex = customIndex;
                instance.mask = 0xFF; // All rays hit
                instance.instanceShaderBindingTableRecordOffset = 0;
                instance.flags = static_cast<uint32_t>(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);
                instance.accelerationStructureReference = blasAddress;

                tlasInstances.push_back(instance);
            }
        }
    }

    if (ui.renderMode != RenderMode::Rasterizer) {
        if (tlasInstances.size() > frames.MAX_TLAS_INSTANCES) {
            throw std::runtime_error(
                "TLAS instance count (" + std::to_string(tlasInstances.size()) +
                ") exceeds MAX_TLAS_INSTANCES (" + std::to_string(frames.MAX_TLAS_INSTANCES) + ")");
        }

        // Only copy instance data when there is something to copy; building with
        // primitiveCount = 0 is valid and produces a traversable empty TLAS.
        if (!tlasInstances.empty()) {
            size_t dataSize = tlasInstances.size() * sizeof(vk::AccelerationStructureInstanceKHR);
            memcpy(frames.tlasInstanceBuffersMapped[frames.frameIndex], tlasInstances.data(), dataSize);
        }

        // Memory barrier to ensure host writes to the instance buffer are visible to the AS builder
        vk::MemoryBarrier2 hostToDeviceBarrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eHost,
            .srcAccessMask = vk::AccessFlagBits2::eHostWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
            .dstAccessMask = vk::AccessFlagBits2::eAccelerationStructureReadKHR
        };

        vk::DependencyInfo dependencyInfo{
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &hostToDeviceBarrier
        };
        commandBuffer.pipelineBarrier2(dependencyInfo);

        // Build TLAS — always, even when the scene is empty (primitiveCount = 0 is valid).
        vk::AccelerationStructureGeometryInstancesDataKHR instancesData{};
        instancesData.arrayOfPointers = vk::False;
        instancesData.data.deviceAddress = frames.tlasInstanceAddresses[frames.frameIndex];

        vk::AccelerationStructureGeometryKHR tlasGeometry{};
        tlasGeometry.geometryType = vk::GeometryTypeKHR::eInstances;
        tlasGeometry.geometry.instances = instancesData;

        vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{};
        buildInfo.type = vk::AccelerationStructureTypeKHR::eTopLevel;
        buildInfo.flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
        buildInfo.mode = vk::BuildAccelerationStructureModeKHR::eBuild;
        buildInfo.geometryCount = 1;
        buildInfo.pGeometries = &tlasGeometry;
        buildInfo.dstAccelerationStructure = *frames.tlas[frames.frameIndex];
        buildInfo.scratchData.deviceAddress = frames.tlasScratchAddresses[frames.frameIndex];

        vk::AccelerationStructureBuildRangeInfoKHR buildRange{};
        buildRange.primitiveCount = static_cast<uint32_t>(tlasInstances.size());
        buildRange.primitiveOffset = 0;
        buildRange.firstVertex = 0;
        buildRange.transformOffset = 0;

        const vk::AccelerationStructureBuildRangeInfoKHR *pBuildRange = &buildRange;
        if (*ptTimestampQueryPool) {
            commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR, *ptTimestampQueryPool, queryBase + kPtTS_TlasStart);
        }
        commandBuffer.buildAccelerationStructuresKHR(buildInfo, pBuildRange);
        if (*ptTimestampQueryPool) {
            commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR, *ptTimestampQueryPool, queryBase + kPtTS_TlasEnd);
        }

        // Memory barrier to ensure TLAS build finishes before the ray tracing shader reads it
        vk::MemoryBarrier2 asBuildToRayTracingBarrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
            .srcAccessMask = vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
            .dstStageMask = vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
            .dstAccessMask = vk::AccessFlagBits2::eAccelerationStructureReadKHR
        };

        vk::DependencyInfo asDependencyInfo{
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &asBuildToRayTracingBarrier
        };
        commandBuffer.pipelineBarrier2(asDependencyInfo);
    }
    // --- End TLAS Build ---

    // ── Cascaded Shadow Map Pass ──────────────────────────────────────────────
    // Only run for the raster path; both RT pipelines handle their own shadowing.
    if (ui.renderMode == RenderMode::Rasterizer) {
        vk::Image shadowImg = *frames.shadowImages[frames.frameIndex];

        // Transition all 4 cascade layers: eUndefined → eDepthAttachmentOptimal.
        // We always use eUndefined as the old layout so the previous frame's contents
        // are discarded — the depth buffer is cleared at the start of each cascade render.
        vk::ImageMemoryBarrier2 shadowToWrite{
            .srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
            .srcAccessMask = {},
            .dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
            .dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eDepthAttachmentOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = shadowImg,
            .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, NUM_SHADOW_CASCADES}
        };
        vk::DependencyInfo shadowWriteDep{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &shadowToWrite};
        commandBuffer.pipelineBarrier2(shadowWriteDep);

        // Render each cascade into its own layer of the shadow array image.
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines.shadowPipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                         *pipelines.shadowPipelineLayout, 0,
                                         *descriptorSets[frames.frameIndex], nullptr);

        vk::Viewport shadowViewport{
            0.0f, 0.0f,
            static_cast<float>(SHADOW_MAP_DIM), static_cast<float>(SHADOW_MAP_DIM),
            0.0f, 1.0f
        };
        vk::Rect2D shadowScissor{{0, 0}, {SHADOW_MAP_DIM, SHADOW_MAP_DIM}};

        for (uint32_t cascadeIdx = 0; cascadeIdx < NUM_SHADOW_CASCADES; cascadeIdx++) {
            uint32_t viewIdx = frames.frameIndex * NUM_SHADOW_CASCADES + cascadeIdx;

            vk::RenderingAttachmentInfo cascadeDepthAttachment{
                .imageView = *frames.shadowCascadeViews[viewIdx],
                .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue = vk::ClearDepthStencilValue{1.0f, 0}
            };

            vk::RenderingInfo cascadeRenderingInfo{
                .renderArea = {{0, 0}, {SHADOW_MAP_DIM, SHADOW_MAP_DIM}},
                .layerCount = 1,
                .colorAttachmentCount = 0,
                .pDepthAttachment = &cascadeDepthAttachment
            };

            commandBuffer.beginRendering(cascadeRenderingInfo);
            commandBuffer.setViewport(0, shadowViewport);
            commandBuffer.setScissor(0, shadowScissor);

            // Draw all scene nodes into this cascade.
            for (const auto &node: scene->getAllNodes()) {
                if (node->modelId < 0)
                    continue;
                auto *modelRes = resourceManager->getModelResource(node->modelId);
                if (!modelRes)
                    continue;

                resourceManager->bindResources(commandBuffer, node->modelId, modelRes->hasRuntimeSkinning);
                glm::mat4 worldTransform = node->getWorldTransform();

                if (*modelRes->descriptorSet) {
                    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelines.shadowPipelineLayout, 1, {*modelRes->descriptorSet}, nullptr);
                }

                for (int meshIdx: node->getMeshIndices()) {
                    if (meshIdx < 0 || meshIdx >= static_cast<int>(modelRes->meshes.size()))
                        continue;
                    for (const auto &prim: modelRes->meshes[meshIdx].primitives) {
                        Laphria::ScenePushConstants pc{};
                        pc.modelMatrix = worldTransform;
                        pc.cascadeIndex = static_cast<int>(cascadeIdx);
                        pc.materialIndex = prim.flatPrimitiveIndex;
                        commandBuffer.pushConstants<Laphria::ScenePushConstants>(
                            *pipelines.shadowPipelineLayout,
                            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                            0, pc);
                        commandBuffer.drawIndexed(prim.indexCount, 1, prim.firstIndex, prim.vertexOffset, 0);
                    }
                }
            }

            commandBuffer.endRendering();
        }

        // Transition shadow image: eDepthAttachmentOptimal → eShaderReadOnlyOptimal
        // so the main fragment shader can sample it.
        vk::ImageMemoryBarrier2 shadowToRead{
            .srcStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
            .srcAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = vk::ImageLayout::eDepthAttachmentOptimal,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = shadowImg,
            .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, NUM_SHADOW_CASCADES}
        };
        vk::DependencyInfo shadowReadDep{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &shadowToRead};
        commandBuffer.pipelineBarrier2(shadowReadDep);
        // V1.3: remove compute sky from raster path; render directly into a cleared color target.

        transition_image_layout(
            swapchain.images[imageIndex],
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal,
            {},
            vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentRead,
            vk::PipelineStageFlagBits2::eTopOfPipe,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::ImageAspectFlagBits::eColor);
    }

    if (ui.renderMode == RenderMode::PathTracer) {
        recordRayTracingCommandBuffer(commandBuffer, imageIndex);
    } else if (ui.renderMode == RenderMode::RayTracer) {
        recordClassicRTCommandBuffer(commandBuffer, imageIndex);
    }

    transition_image_layout(
        *frames.depthImages[imageIndex],
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eDepthAttachmentOptimal,
        {},
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::ImageAspectFlagBits::eDepth);

    vk::RenderingAttachmentInfo attachmentInfo = {
        .imageView = *swapchain.imageViews[imageIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = (ui.renderMode == RenderMode::Rasterizer) ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = clearColor
    };

    vk::RenderingAttachmentInfo depthAttachmentInfo{
        .imageView = *frames.depthImageViews[imageIndex],
        .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearDepthStencilValue{1.0f, 0}
    };

    vk::RenderingInfo renderingInfo = {
        .renderArea = {.offset = {0, 0}, .extent = swapchain.extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachmentInfo,
        .pDepthAttachment = &depthAttachmentInfo
    };

    commandBuffer.beginRendering(renderingInfo);

    if (ui.renderMode == RenderMode::Rasterizer) {
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines.graphicsPipeline);

        // Y starts at height and height is negative: this flips the Vulkan NDC Y-axis so that
        // +Y points up in clip space, matching GLM's convention (which was designed for OpenGL).
        vk::Viewport viewport{
            0.0f, static_cast<float>(swapchain.extent.height),
            static_cast<float>(swapchain.extent.width),
            -static_cast<float>(swapchain.extent.height), 0.0f, 1.0f
        };
        commandBuffer.setViewport(0, viewport);
        commandBuffer.setScissor(0, vk::Rect2D({0, 0}, swapchain.extent));

        // Global UBO Binding (Set 0)
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelines.graphicsPipelineLayout, 0,
                                         *descriptorSets[frames.frameIndex], nullptr);

        const float aspectRatio = static_cast<float>(swapchain.extent.width) / static_cast<float>(swapchain.extent.height);
        const glm::mat4 view = camera.getViewMatrix();
        const glm::mat4 proj = glm::perspective(
            glm::radians(Laphria::EngineConfig::kMainCameraFovDegrees),
            aspectRatio,
            Laphria::EngineConfig::kMainCameraNearPlane,
            Laphria::EngineConfig::kMainCameraFarPlane);
        const glm::mat4 viewProjection = proj * view;
        const glm::mat4 invViewProjection = glm::inverse(viewProjection);

        const Laphria::Frustum frustum = Laphria::Frustum::fromViewProjection(viewProjection);
        const Laphria::AABB cullBounds = Laphria::Frustum::computeAABB(invViewProjection);
        scene->draw(commandBuffer, pipelines.graphicsPipelineLayout, *resourceManager, cullBounds, frustum);
    }

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *commandBuffer);

    commandBuffer.endRendering();

    // Transition SwapChain to Present Layout
    transition_image_layout(
        swapchain.images[imageIndex],
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::ePresentSrcKHR,
        vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentRead,
        {},
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eBottomOfPipe,
        vk::ImageAspectFlagBits::eColor);
}

// Inline Synchronization2 image barrier recorded into the current frame's command buffer.
// Unlike VulkanUtils::recordImageLayoutTransition (which uses Vulkan 1.0 pipelineBarrier),
// this version accepts explicit stage/access masks for fine-grained GPU dependency control.
void EngineCore::transition_image_layout(
    vk::Image image,
    vk::ImageLayout old_layout,
    vk::ImageLayout new_layout,
    vk::AccessFlags2 src_access_mask,
    vk::AccessFlags2 dst_access_mask,
    vk::PipelineStageFlags2 src_stage_mask,
    vk::PipelineStageFlags2 dst_stage_mask,
    vk::ImageAspectFlags image_aspect_flags) const {
    vk::ImageMemoryBarrier2 barrier = {
        .srcStageMask = src_stage_mask,
        .srcAccessMask = src_access_mask,
        .dstStageMask = dst_stage_mask,
        .dstAccessMask = dst_access_mask,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {
            .aspectMask = image_aspect_flags,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };
    vk::DependencyInfo dependency_info = {
        .dependencyFlags = {},
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier
    };
    frames.commandBuffers[frames.frameIndex].pipelineBarrier2(dependency_info);
}

void EngineCore::drawFrame() {
    if (!renderModeInitialized) {
        lastSubmittedRenderMode = ui.renderMode;
        renderModeInitialized = true;
    } else if (ui.renderMode != lastSubmittedRenderMode) {
        // Renderer switches can otherwise overlap in-flight GPU work that uses different
        // pipeline/resource access patterns (especially PT denoiser scratch buffers).
        vulkan.logicalDevice.waitIdle();
        ptCameraMoved = true; // force history reset on the first PT frame after a mode switch
        lastSubmittedRenderMode = ui.renderMode;
    }

    // Note: inFlightFences, presentCompleteSemaphores, and commandBuffers are indexed by frameIndex,
    //       while renderFinishedSemaphores is indexed by imageIndex
    auto fenceResult = vulkan.logicalDevice.waitForFences(*frames.inFlightFences[frames.frameIndex], vk::True, UINT64_MAX);
    if (fenceResult != vk::Result::eSuccess) {
        throw std::runtime_error("failed to wait for fence!");
    }

    // Runtime skinned BLAS refit currently reuses per-model AS buffers across frames.
    // Serialize in-flight submissions in this mode to avoid cross-frame AS write hazards.
    if (resourceManager && resourceManager->hasRuntimeSkinnedModels()) {
        for (size_t i = 0; i < frames.inFlightFences.size(); ++i) {
            if (i == frames.frameIndex) {
                continue;
            }
            const auto syncResult = vulkan.logicalDevice.waitForFences(*frames.inFlightFences[i], vk::True, UINT64_MAX);
            if (syncResult != vk::Result::eSuccess) {
                throw std::runtime_error("failed to synchronize runtime skinned BLAS updates");
            }
        }
    }

    if (submittedRenderModes[frames.frameIndex] == RenderMode::PathTracer) {
        collectPathTracerTimings(frames.frameIndex);
        updateAdaptivePathTracerSettings();
    }

    auto [result, imageIndex] = swapchain.swapChain.acquireNextImage(
        UINT64_MAX, *frames.presentCompleteSemaphores[frames.frameIndex], nullptr);

    if (result == vk::Result::eErrorOutOfDateKHR) {
        recreateSwapChain();
        return;
    }
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        assert(result == vk::Result::eTimeout || result == vk::Result::eNotReady);
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    // If this swapchain image is still associated with an older in-flight frame, wait for it.
    if (imageIndex < imagesInFlight.size() && imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
        const auto imageFenceResult = vulkan.logicalDevice.waitForFences(
            std::array<vk::Fence, 1>{imagesInFlight[imageIndex]}, vk::True, UINT64_MAX);
        if (imageFenceResult != vk::Result::eSuccess) {
            throw std::runtime_error("failed to wait for in-flight swapchain image fence");
        }
    }

    frames.updateUniformBuffer(frames.frameIndex, camera, swapchain.extent, ui.lightDirection, ui.visualsV1);

    // Detect camera movement for path tracer history reset.
    // Any translation or rotation invalidates the reprojected history.
    ptCameraMoved = (glm::distance(camera.position, ptPrevCameraPos) > 1e-5f ||
                     std::abs(camera.pitch - ptPrevPitch) > 1e-5f ||
                     std::abs(camera.yaw - ptPrevYaw) > 1e-5f);
    ptPrevCameraPos = camera.position;
    ptPrevPitch = camera.pitch;
    ptPrevYaw = camera.yaw;

    // Only reset the fence if we are submitting work
    vulkan.logicalDevice.resetFences(*frames.inFlightFences[frames.frameIndex]);

    frames.commandBuffers[frames.frameIndex].reset();
    vk::raii::CommandBuffer &commandBuffer = frames.commandBuffers[frames.frameIndex];
    commandBuffer.begin(vk::CommandBufferBeginInfo{});

    // 2. Main Pass
    recordCommandBuffer(imageIndex);
    submittedRenderModes[frames.frameIndex] = ui.renderMode;
    ptTimestampsValid[frames.frameIndex] = (ui.renderMode == RenderMode::PathTracer);

    // The swapchain image is accessed at eColorAttachmentOutput (main/ImGui pass) and at
    // eTransfer (blit in compute and RT paths). Both stages must wait for vkAcquireNextImage.
    vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eAllCommands);
    const vk::SubmitInfo submitInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*frames.presentCompleteSemaphores[frames.frameIndex],
        .pWaitDstStageMask = &waitDestinationStageMask,
        .commandBufferCount = 1,
        .pCommandBuffers = &*frames.commandBuffers[frames.frameIndex],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*frames.renderFinishedSemaphores[imageIndex]
    };

    commandBuffer.end();

    vulkan.queue.submit(submitInfo, *frames.inFlightFences[frames.frameIndex]);
    if (imageIndex < imagesInFlight.size()) {
        imagesInFlight[imageIndex] = *frames.inFlightFences[frames.frameIndex];
    }

    const vk::PresentInfoKHR presentInfoKHR{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*frames.renderFinishedSemaphores[imageIndex],
        .swapchainCount = 1,
        .pSwapchains = &*swapchain.swapChain,
        .pImageIndices = &imageIndex
    };

    // VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS is defined so presentKHR should return
    // eErrorOutOfDateKHR as a value rather than throw, but behaviour is inconsistent across
    // loader/driver versions. The try/catch ensures resize detection is never silently lost.
    try {
        result = vulkan.queue.presentKHR(presentInfoKHR);
    } catch (vk::OutOfDateKHRError &) {
        result = vk::Result::eErrorOutOfDateKHR;
    } catch (vk::SurfaceLostKHRError &) {
        result = vk::Result::eErrorOutOfDateKHR;
    }

    if ((result == vk::Result::eSuboptimalKHR) || (result == vk::Result::eErrorOutOfDateKHR) ||
        swapchain.framebufferResized) {
        swapchain.framebufferResized = false;
        recreateSwapChain();
    } else {
        assert(result == vk::Result::eSuccess);
    }
    frames.frameIndex = (frames.frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
}


