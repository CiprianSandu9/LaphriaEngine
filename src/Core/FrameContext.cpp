#include "FrameContext.h"
#include "VulkanUtils.h"
#include "EngineConfig.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>

using namespace Laphria;

namespace
{
void destroyBuffersAndReleaseAllocations(std::vector<VulkanUtils::VmaBuffer> &buffers)
{
	for (auto &buffer : buffers)
	{
		buffer.reset();
	}
}

void destroyImagesAndReleaseAllocations(std::vector<VulkanUtils::VmaImage> &images)
{
	for (auto &image : images)
	{
		image.reset();
	}
}
} // namespace

FrameContext::~FrameContext()
{
	destroyImagesAndReleaseAllocations(shadowImages);
	destroyImagesAndReleaseAllocations(depthImages);
	destroyImagesAndReleaseAllocations(storageImages);
	destroyImagesAndReleaseAllocations(rayTracingOutputImages);
	destroyImagesAndReleaseAllocations(rtGBufferNormals);
	destroyImagesAndReleaseAllocations(rtGBufferDepth);
	destroyImagesAndReleaseAllocations(rtMotionVectors);
	destroyImagesAndReleaseAllocations(historyColor);
	destroyImagesAndReleaseAllocations(historyMoments);
	destroyImagesAndReleaseAllocations(atrousTemp);

	destroyBuffersAndReleaseAllocations(uniformBuffers);
	destroyBuffersAndReleaseAllocations(tlasBuffers);
	destroyBuffersAndReleaseAllocations(tlasScratchBuffers);
	destroyBuffersAndReleaseAllocations(tlasInstanceBuffers);
}

void FrameContext::init(VulkanDevice &dev, SwapchainManager &swapchain) {
    // Command pool must be created first; ResourceManager needs it for staging uploads.
    createCommandPool(dev);
    createUniformBuffers(dev);
    createDepthResources(dev, swapchain);
    createStorageResources(dev, swapchain);
    createRayTracingOutputImages(dev, swapchain);
    createGBufferResources(dev, swapchain);
    createHistoryResources(dev, swapchain);
    createAtrousResources(dev, swapchain);
    // Shadow resources are extent-independent and live for the engine's full lifetime.
    createShadowResources(dev);

    createTLASResources(dev);
    createCommandBuffers(dev);
    createSyncObjects(dev, static_cast<uint32_t>(swapchain.images.size()));
}

void FrameContext::cleanupSwapChainDependents() {
    destroyImagesAndReleaseAllocations(storageImages);
    destroyImagesAndReleaseAllocations(rayTracingOutputImages);
    destroyImagesAndReleaseAllocations(depthImages);
    destroyImagesAndReleaseAllocations(rtGBufferNormals);
    destroyImagesAndReleaseAllocations(rtGBufferDepth);
    destroyImagesAndReleaseAllocations(rtMotionVectors);
    destroyImagesAndReleaseAllocations(historyColor);
    destroyImagesAndReleaseAllocations(historyMoments);
    destroyImagesAndReleaseAllocations(atrousTemp);

    storageImageViews.clear();
    storageImages.clear();
    rayTracingOutputImageViews.clear();
    rayTracingOutputImages.clear();
    depthImageViews.clear();
    depthImages.clear();

    // G-Buffer images are extent-dependent.
    rtGBufferNormalsViews.clear();
    rtGBufferNormals.clear();
    rtGBufferDepthViews.clear();
    rtGBufferDepth.clear();
    rtMotionVectorsViews.clear();
    rtMotionVectors.clear();

    // History and A-Trous images are extent-dependent.
    historyColorViews.clear();
    historyColor.clear();
    historyMomentsViews.clear();
    historyMoments.clear();
    atrousTempViews.clear();
    atrousTemp.clear();
}

void FrameContext::recreate(VulkanDevice &dev, SwapchainManager &swapchain) {
    cleanupSwapChainDependents();
    createStorageResources(dev, swapchain);
    createRayTracingOutputImages(dev, swapchain);
    createDepthResources(dev, swapchain);
    createGBufferResources(dev, swapchain);
    createHistoryResources(dev, swapchain);
    createAtrousResources(dev, swapchain);
}

void FrameContext::createCommandPool(const VulkanDevice &dev) {
    // eResetCommandBuffer lets individual command buffers be reset without resetting the pool.
    vk::CommandPoolCreateInfo poolInfo{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = dev.queueIndex
    };
    commandPool = vk::raii::CommandPool(dev.logicalDevice, poolInfo);
}

void FrameContext::createCommandBuffers(const VulkanDevice &dev) {
    commandBuffers.clear();
    vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = *commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = MAX_FRAMES_IN_FLIGHT
    };
    commandBuffers = vk::raii::CommandBuffers(dev.logicalDevice, allocInfo);
}

void FrameContext::createSyncObjects(VulkanDevice &dev, uint32_t imageCount) {
    assert(presentCompleteSemaphores.empty() && renderFinishedSemaphores.empty() && inFlightFences.empty());

    // renderFinishedSemaphores are indexed by swapchain image index (not frame-in-flight)
    // so that the present operation waits on the correct signal for each image.
    for (uint32_t i = 0; i < imageCount; i++) {
        renderFinishedSemaphores.emplace_back(dev.logicalDevice, vk::SemaphoreCreateInfo());
    }

    // presentCompleteSemaphores and inFlightFences are indexed by frameIndex (0..MAX_FRAMES_IN_FLIGHT-1).
    // Fences are pre-signalled so the first call to waitForFences() in drawFrame() does not stall.
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        presentCompleteSemaphores.emplace_back(dev.logicalDevice, vk::SemaphoreCreateInfo());
        inFlightFences.emplace_back(dev.logicalDevice,
                                    vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
    }
}

void FrameContext::createDepthResources(const VulkanDevice &dev, const SwapchainManager &swapchain) {
    vk::Format depthFormat = dev.findDepthFormat();

    depthImages.clear();
    depthImageViews.clear();

    // One depth image per swapchain image so each in-flight frame has its own depth buffer.
    size_t count = swapchain.images.size();
    depthImages.reserve(count);
    depthImageViews.reserve(count);

    for (size_t i = 0; i < count; i++) {
        VulkanUtils::VmaImage img{};

        VulkanUtils::createImage(dev.logicalDevice, dev.physicalDevice, swapchain.extent.width, swapchain.extent.height,
                                 depthFormat, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eDepthStencilAttachment,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal, img);

        depthImages.push_back(std::move(img));
        depthImageViews.push_back(VulkanUtils::createImageView(dev.logicalDevice, *depthImages.back(), depthFormat, vk::ImageAspectFlagBits::eDepth));
    }
}

void FrameContext::createStorageResources(VulkanDevice &dev, SwapchainManager &swapchain) {
    storageImages.clear();
    storageImageViews.clear();

    storageImages.reserve(MAX_FRAMES_IN_FLIGHT);
    storageImageViews.reserve(MAX_FRAMES_IN_FLIGHT);

    // R16G16B16A16_SFLOAT: 16-bit HDR format so the starfield compute shader can produce
    // high-dynamic-range colors before the blit into the sRGB swapchain image.
    // eStorage: written by the compute shader; eTransferSrc: read during the blit to the swapchain.
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        VulkanUtils::VmaImage img{};

        VulkanUtils::createImage(dev.logicalDevice, dev.physicalDevice, swapchain.extent.width, swapchain.extent.height,
                                 vk::Format::eR16G16B16A16Sfloat, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal, img);

        storageImages.push_back(std::move(img));
        storageImageViews.push_back(VulkanUtils::createImageView(dev.logicalDevice, *storageImages.back(),
                                                                 vk::Format::eR16G16B16A16Sfloat,
                                                                 vk::ImageAspectFlagBits::eColor));
    }

    // Pre-transition all storage images to eGeneral so they always match the layout declared
    // in computeDescriptorSets, even on frames where the compute pass hasn't run yet.
    {
        auto cmd = VulkanUtils::beginSingleTimeCommands(dev.logicalDevice, commandPool);
        for (auto &img: storageImages)
            VulkanUtils::recordImageLayoutTransition(cmd, *img,
                                                     vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
        VulkanUtils::endSingleTimeCommands(dev.logicalDevice, dev.queue, commandPool, cmd);
    }
}

void FrameContext::createRayTracingOutputImages(const VulkanDevice &dev, const SwapchainManager &swapchain) {
    rayTracingOutputImages.clear();
    rayTracingOutputImageViews.clear();

    rayTracingOutputImages.reserve(MAX_FRAMES_IN_FLIGHT);
    rayTracingOutputImageViews.reserve(MAX_FRAMES_IN_FLIGHT);

    // R16G16B16A16_SFLOAT matches the compute storage image format for consistency.
    // eStorage: written by the RT raygen shader in General layout.
    // eTransferSrc: read during the blit to the swapchain image.
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        VulkanUtils::VmaImage img{};

        VulkanUtils::createImage(dev.logicalDevice, dev.physicalDevice, swapchain.extent.width, swapchain.extent.height,
                                 vk::Format::eR16G16B16A16Sfloat, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal, img);

        rayTracingOutputImages.push_back(std::move(img));
        rayTracingOutputImageViews.push_back(VulkanUtils::createImageView(dev.logicalDevice, *rayTracingOutputImages.back(),
                                                                          vk::Format::eR16G16B16A16Sfloat,
                                                                          vk::ImageAspectFlagBits::eColor));
    }

    // Pre-transition to eGeneral so the layout always matches rtDescriptorSets / denoiserDescriptorSets
    // even when the path tracer or classic RT hasn't rendered yet (e.g. on the first rasterizer frame).
    {
        auto cmd = VulkanUtils::beginSingleTimeCommands(dev.logicalDevice, commandPool);
        for (auto &img: rayTracingOutputImages)
            VulkanUtils::recordImageLayoutTransition(cmd, *img,
                                                     vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
        VulkanUtils::endSingleTimeCommands(dev.logicalDevice, dev.queue, commandPool, cmd);
    }
}

void FrameContext::createUniformBuffers(const VulkanDevice &dev) {
    uniformBuffers.clear();
    uniformBuffersMapped.clear();

    // Host-visible + host-coherent so we can memcpy each frame without an explicit flush.
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::DeviceSize bufferSize = sizeof(Laphria::UniformBufferObject);
        VulkanUtils::VmaBuffer buffer{};
        VulkanUtils::createBuffer(dev.logicalDevice, dev.physicalDevice, bufferSize,
                                  vk::BufferUsageFlagBits::eUniformBuffer,
                                  vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                  buffer);
        uniformBuffers.emplace_back(std::move(buffer));
        // Keep the buffer persistently mapped for the engine's lifetime.
        uniformBuffersMapped.emplace_back(uniformBuffers[i].memory.mapMemory(0, bufferSize));
    }
}

void FrameContext::updateUniformBuffer(uint32_t frameIdx, const Camera &camera, vk::Extent2D extent, glm::vec3 lightDirection,
                                       const UISystem::VisualsV1Settings &visualsV1) {
    Laphria::UniformBufferObject ubo{};
    ubo.view = camera.getViewMatrix();

    const float aspectRatio = static_cast<float>(extent.width) / static_cast<float>(extent.height);
    ubo.proj = glm::perspective(glm::radians(Laphria::EngineConfig::kMainCameraFovDegrees),
                                aspectRatio,
                                Laphria::EngineConfig::kMainCameraNearPlane,
                                Laphria::EngineConfig::kMainCameraFarPlane);

    ubo.cameraPos = glm::vec4(camera.position, 1.0f);

    ubo.lightDir = glm::vec4(glm::normalize(lightDirection), 0.0f);

    ubo.viewInverse = glm::inverse(ubo.view);
    ubo.projInverse = glm::inverse(ubo.proj);

    // ── Cascaded Shadow Map matrices ─────────────────────────────────────────
    // Split the camera frustum into NUM_SHADOW_CASCADES sub-frustums using PSSM.
    // Shadow max distance is kept well below the camera far plane for quality.
    constexpr float NEAR_PLANE = 0.01f;
    constexpr float SHADOW_MAX_DIST = 200.0f;
    constexpr float SPLIT_LAMBDA = 0.95f; // 1.0 = pure log, 0.0 = pure linear
    constexpr float Z_PULLBACK = 50.0f; // extend far plane to catch out-of-frustum casters
    constexpr float FOV = glm::radians(Laphria::EngineConfig::kMainCameraFovDegrees);

    float cascadeSplitDepths[NUM_SHADOW_CASCADES];
    for (uint32_t i = 0; i < NUM_SHADOW_CASCADES; i++) {
        float p = (i + 1) / static_cast<float>(NUM_SHADOW_CASCADES);
        float splitLog = NEAR_PLANE * std::pow(SHADOW_MAX_DIST / NEAR_PLANE, p);
        float splitLinear = NEAR_PLANE + (SHADOW_MAX_DIST - NEAR_PLANE) * p;
        cascadeSplitDepths[i] = SPLIT_LAMBDA * splitLog + (1.0f - SPLIT_LAMBDA) * splitLinear;
    }
    ubo.cascadeSplits = glm::vec4(cascadeSplitDepths[0], cascadeSplitDepths[1],
                                  cascadeSplitDepths[2], cascadeSplitDepths[3]);

    // Stable light direction and up vector (avoid gimbal if light is nearly vertical).
    glm::vec3 lightDir = glm::normalize(lightDirection);
    glm::vec3 lightUp = (std::abs(lightDir.y) > 0.99f) ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 1.0f, 0.0f);

    for (uint32_t i = 0; i < NUM_SHADOW_CASCADES; i++) {
        float prevSplit = (i == 0) ? NEAR_PLANE : cascadeSplitDepths[i - 1];
        float currSplit = cascadeSplitDepths[i];

        // Build the inverse view-proj for this sub-frustum to extract world-space corners.
        glm::mat4 subProj = glm::perspective(FOV, aspectRatio, prevSplit, currSplit);
        glm::mat4 invProjView = glm::inverse(subProj * ubo.view);

        // 8 NDC corners of the sub-frustum.
        // Z uses [0, 1] because GLM_FORCE_DEPTH_ZERO_TO_ONE is active.
        constexpr float ndc_x[2] = {-1.0f, 1.0f};
        constexpr float ndc_y[2] = {-1.0f, 1.0f};
        constexpr float ndc_z[2] = {0.0f, 1.0f};

        glm::vec4 worldCorners[8];
        glm::vec3 frustumCenter(0.0f);
        int idx = 0;
        for (int xi = 0; xi < 2; xi++)
            for (int yi = 0; yi < 2; yi++)
                for (int zi = 0; zi < 2; zi++) {
                    glm::vec4 clipPt = invProjView * glm::vec4(ndc_x[xi], ndc_y[yi], ndc_z[zi], 1.0f);
                    clipPt /= clipPt.w;
                    worldCorners[idx++] = clipPt;
                    frustumCenter += glm::vec3(clipPt);
                }
        frustumCenter /= 8.0f;

        // =========================================================================
        // SHADOW STABILIZATION (SHIMMERING FIX)
        // =========================================================================

        // 1. Calculate the bounding sphere of the frustum
        float sphereRadius = 0.0f;
        for (int j = 0; j < 8; j++) {
            float dist = glm::length(glm::vec3(worldCorners[j]) - frustumCenter);
            sphereRadius = std::max(sphereRadius, dist);
        }

        // Round up the radius to fix the projection size (prevents pulsing as camera rotates)
        sphereRadius = std::ceil(sphereRadius * 16.0f) / 16.0f;

        // 2. Position the light camera
        glm::vec3 maxExtents = glm::vec3(sphereRadius);
        glm::vec3 minExtents = -maxExtents;

        // Position the shadow camera far enough back to include casters (pullback)
        // lightDir points from sky to ground. Subtracting it from the center places the camera in the sky.
        glm::vec3 lightCameraPos = frustumCenter - lightDir * (sphereRadius + Z_PULLBACK);
        glm::mat4 lightView = glm::lookAt(lightCameraPos, frustumCenter, lightUp);

        // 3. Create the stable projection matrix
        glm::mat4 lightProj = glm::ortho(minExtents.x, maxExtents.x, minExtents.y, maxExtents.y, 0.001f, sphereRadius * 2.0f + Z_PULLBACK);

        // Vulkan Clip Space Correction for the Shadow Map
        // The main pass uses a negative viewport height to flip Y, but the shadow pass uses a standard viewport.
        // Therefore, we must explicitly flip the projection Y-axis for the shadow map so it matches Vulkan's NDC.
        // Doing this BEFORE the sub-pixel snap ensures the rounding offsets operate in the correct NDC space.
        lightProj[1][1] *= -1.0f;

        // 4. Snap the projection to the texel grid to avoid sub-pixel swimming
        glm::mat4 shadowMatrix = lightProj * lightView;
        glm::vec4 shadowOrigin = shadowMatrix * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        shadowOrigin *= (SHADOW_MAP_DIM / 2.0f); // Map to texture coordinates

        glm::vec4 roundedOrigin = glm::round(shadowOrigin);
        glm::vec4 roundOffset = roundedOrigin - shadowOrigin;
        roundOffset *= (2.0f / SHADOW_MAP_DIM); // Map back to NDC

        lightProj[3][0] += roundOffset.x;
        lightProj[3][1] += roundOffset.y;

        ubo.cascadeViewProj[i] = lightProj * lightView;
    }

    // Path tracer temporal fields — carry the previous frame's VP and advance the frame counter.
    ubo.prevViewProj = prevViewProj;
    ubo.frameCount = frameCount;
    ubo.jitter_x = 0.0f; // Sub-pixel jitter disabled; set to halton values to enable TAA
    ubo.jitter_y = 0.0f;
    ubo._pad0 = 0;
    ubo.gameplayVisuals = glm::vec4(
        std::max(0.0f, visualsV1.sunIntensity),
        std::max(0.0f, visualsV1.fillIntensity),
        std::max(0.0f, visualsV1.ambientBoost),
        std::max(0.0f, visualsV1.exposure));

    // Update persistent state for the next frame.
    prevViewProj = ubo.proj * ubo.view;
    ++frameCount;

    memcpy(uniformBuffersMapped[frameIdx], &ubo, sizeof(ubo));
}

void FrameContext::createShadowResources(const VulkanDevice &dev) {
    // Each frame-in-flight gets one D32_SFLOAT array image with NUM_SHADOW_CASCADES layers.
    // These images are NOT swapchain-extent-dependent, so they are never cleaned on resize.
    constexpr vk::Format SHADOW_FORMAT = vk::Format::eD32Sfloat;

    shadowImages.clear();
    shadowCascadeViews.clear();
    shadowArrayViews.clear();

    shadowImages.reserve(MAX_FRAMES_IN_FLIGHT);
    shadowCascadeViews.reserve(MAX_FRAMES_IN_FLIGHT * NUM_SHADOW_CASCADES);
    shadowArrayViews.reserve(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        VulkanUtils::VmaImage img{};

        VulkanUtils::createImage(
            dev.logicalDevice, dev.physicalDevice,
            SHADOW_MAP_DIM, SHADOW_MAP_DIM,
            SHADOW_FORMAT, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            img,
            NUM_SHADOW_CASCADES);

        shadowImages.push_back(std::move(img));

        // Per-layer 2D views — used as depth attachments when rendering each cascade.
        for (uint32_t c = 0; c < NUM_SHADOW_CASCADES; c++) {
            shadowCascadeViews.push_back(VulkanUtils::createImageViewLayer(
                dev.logicalDevice, *shadowImages.back(),
                SHADOW_FORMAT, vk::ImageAspectFlagBits::eDepth, c));
        }

        // Full 2D_ARRAY view — bound as a sampled image in the main fragment pass.
        shadowArrayViews.push_back(VulkanUtils::createImageViewArray(
            dev.logicalDevice, *shadowImages.back(),
            SHADOW_FORMAT, vk::ImageAspectFlagBits::eDepth,
            NUM_SHADOW_CASCADES));
    }

    // One shared comparison sampler for all frames and cascades.
    // compareOp = eLessOrEqual: SampleCmp returns 1.0 when the fragment is lit (shadowDepth <= shadowMapDepth).
    // Border depth = 1.0 (eFloatOpaqueWhite) so areas outside the shadow map are fully lit.
    vk::SamplerCreateInfo samplerInfo{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToBorder,
        .addressModeV = vk::SamplerAddressMode::eClampToBorder,
        .addressModeW = vk::SamplerAddressMode::eClampToBorder,
        .mipLodBias = 0.0f,
        .anisotropyEnable = vk::False,
        .compareEnable = vk::True,
        .compareOp = vk::CompareOp::eLessOrEqual,
        .minLod = 0.0f,
        .maxLod = 0.0f,
        .borderColor = vk::BorderColor::eFloatOpaqueWhite,
        .unnormalizedCoordinates = vk::False
    };
    shadowSampler = vk::raii::Sampler(dev.logicalDevice, samplerInfo);
}

void FrameContext::createGBufferResources(const VulkanDevice &dev, const SwapchainManager &swapchain) {
    rtGBufferNormals.clear();
    rtGBufferNormalsViews.clear();
    rtGBufferDepth.clear();
    rtGBufferDepthViews.clear();
    rtMotionVectors.clear();
    rtMotionVectorsViews.clear();

    rtGBufferNormals.reserve(MAX_FRAMES_IN_FLIGHT);
    rtGBufferNormalsViews.reserve(MAX_FRAMES_IN_FLIGHT);
    rtGBufferDepth.reserve(MAX_FRAMES_IN_FLIGHT);
    rtGBufferDepthViews.reserve(MAX_FRAMES_IN_FLIGHT);
    rtMotionVectors.reserve(MAX_FRAMES_IN_FLIGHT);
    rtMotionVectorsViews.reserve(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        // Normals — R16G16B16A16_SFLOAT: world-space XYZ packed into RGB, W unused.
        {
            VulkanUtils::VmaImage img{};
            VulkanUtils::createImage(dev.logicalDevice, dev.physicalDevice,
                                     swapchain.extent.width, swapchain.extent.height,
                                     vk::Format::eR16G16B16A16Sfloat, vk::ImageTiling::eOptimal,
                                     vk::ImageUsageFlagBits::eStorage,
                                     vk::MemoryPropertyFlagBits::eDeviceLocal, img);
            rtGBufferNormals.push_back(std::move(img));
            rtGBufferNormalsViews.push_back(VulkanUtils::createImageView(dev.logicalDevice,
                                                                         *rtGBufferNormals.back(), vk::Format::eR16G16B16A16Sfloat, vk::ImageAspectFlagBits::eColor));
        }

        // Depth — R32_SFLOAT: linear ray hit distance (negative = sky miss).
        {
            VulkanUtils::VmaImage img{};
            VulkanUtils::createImage(dev.logicalDevice, dev.physicalDevice,
                                     swapchain.extent.width, swapchain.extent.height,
                                     vk::Format::eR32Sfloat, vk::ImageTiling::eOptimal,
                                     vk::ImageUsageFlagBits::eStorage,
                                     vk::MemoryPropertyFlagBits::eDeviceLocal, img);
            rtGBufferDepth.push_back(std::move(img));
            rtGBufferDepthViews.push_back(VulkanUtils::createImageView(dev.logicalDevice,
                                                                       *rtGBufferDepth.back(), vk::Format::eR32Sfloat, vk::ImageAspectFlagBits::eColor));
        }

        // Motion vectors — R16G16_SFLOAT: screen-space pixel offset in UV space.
        {
            VulkanUtils::VmaImage img{};
            VulkanUtils::createImage(dev.logicalDevice, dev.physicalDevice,
                                     swapchain.extent.width, swapchain.extent.height,
                                     vk::Format::eR16G16Sfloat, vk::ImageTiling::eOptimal,
                                     vk::ImageUsageFlagBits::eStorage,
                                     vk::MemoryPropertyFlagBits::eDeviceLocal, img);
            rtMotionVectors.push_back(std::move(img));
            rtMotionVectorsViews.push_back(VulkanUtils::createImageView(dev.logicalDevice,
                                                                        *rtMotionVectors.back(), vk::Format::eR16G16Sfloat, vk::ImageAspectFlagBits::eColor));
        }
    }

    // Pre-transition all G-Buffer images to eGeneral so they match the declared layout in
    // rtDescriptorSets and denoiserDescriptorSets, even when no RT pass has run yet.
    {
        auto cmd = VulkanUtils::beginSingleTimeCommands(dev.logicalDevice, commandPool);
        for (auto &img: rtGBufferNormals)
            VulkanUtils::recordImageLayoutTransition(cmd, *img,
                                                     vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
        for (auto &img: rtGBufferDepth)
            VulkanUtils::recordImageLayoutTransition(cmd, *img,
                                                     vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
        for (auto &img: rtMotionVectors)
            VulkanUtils::recordImageLayoutTransition(cmd, *img,
                                                     vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
        VulkanUtils::endSingleTimeCommands(dev.logicalDevice, dev.queue, commandPool, cmd);
    }
}

void FrameContext::createHistoryResources(const VulkanDevice &dev, const SwapchainManager &swapchain) {
    historyColor.clear();
    historyColorViews.clear();
    historyMoments.clear();
    historyMomentsViews.clear();

    historyColor.reserve(MAX_FRAMES_IN_FLIGHT);
    historyColorViews.reserve(MAX_FRAMES_IN_FLIGHT);
    historyMoments.reserve(MAX_FRAMES_IN_FLIGHT);
    historyMomentsViews.reserve(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        // Accumulated color history — R16G16B16A16_SFLOAT; written by reprojection, read next frame.
        {
            VulkanUtils::VmaImage img{};
            VulkanUtils::createImage(dev.logicalDevice, dev.physicalDevice,
                                     swapchain.extent.width, swapchain.extent.height,
                                     vk::Format::eR16G16B16A16Sfloat, vk::ImageTiling::eOptimal,
                                     vk::ImageUsageFlagBits::eStorage,
                                     vk::MemoryPropertyFlagBits::eDeviceLocal, img);
            historyColor.push_back(std::move(img));
            historyColorViews.push_back(VulkanUtils::createImageView(dev.logicalDevice,
                                                                     *historyColor.back(), vk::Format::eR16G16B16A16Sfloat, vk::ImageAspectFlagBits::eColor));
        }

        // Moments — R16G16_SFLOAT: R = first moment (mean luminance), G = second moment (mean lum²).
        {
            VulkanUtils::VmaImage img{};
            VulkanUtils::createImage(dev.logicalDevice, dev.physicalDevice,
                                     swapchain.extent.width, swapchain.extent.height,
                                     vk::Format::eR16G16Sfloat, vk::ImageTiling::eOptimal,
                                     vk::ImageUsageFlagBits::eStorage,
                                     vk::MemoryPropertyFlagBits::eDeviceLocal, img);
            historyMoments.push_back(std::move(img));
            historyMomentsViews.push_back(VulkanUtils::createImageView(dev.logicalDevice,
                                                                       *historyMoments.back(), vk::Format::eR16G16Sfloat, vk::ImageAspectFlagBits::eColor));
        }
    }

    // Transition all history images from UNDEFINED to GENERAL once so the layout matches
    // the eGeneral written in the denoiser descriptor set on the first frame (and after any
    // swapchain resize that recreates these images).  atrousTemp gets the same treatment
    // inside recordRayTracingCommandBuffer each frame, but history images must NOT be
    // discarded frame-to-frame, so a one-time init here is the correct approach.
    {
        auto cmd = VulkanUtils::beginSingleTimeCommands(dev.logicalDevice, commandPool);
        for (auto &img: historyColor)
            VulkanUtils::recordImageLayoutTransition(cmd, *img,
                                                     vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
        for (auto &img: historyMoments)
            VulkanUtils::recordImageLayoutTransition(cmd, *img,
                                                     vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
        VulkanUtils::endSingleTimeCommands(dev.logicalDevice, dev.queue, commandPool, cmd);
    }
}

void FrameContext::createAtrousResources(const VulkanDevice &dev, const SwapchainManager &swapchain) {
    atrousTemp.clear();
    atrousTempViews.clear();
    constexpr size_t bufferCount = static_cast<size_t>(MAX_FRAMES_IN_FLIGHT) * 2;
    atrousTemp.reserve(bufferCount);
    atrousTempViews.reserve(bufferCount);

    // Two ping-pong buffers per frame slot. This removes shared scratch hazards between in-flight frames.
    for (size_t i = 0; i < bufferCount; i++) {
        VulkanUtils::VmaImage img{};
        VulkanUtils::createImage(dev.logicalDevice, dev.physicalDevice,
                                 swapchain.extent.width, swapchain.extent.height,
                                 vk::Format::eR16G16B16A16Sfloat, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eStorage,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal, img);
        atrousTemp.push_back(std::move(img));
        atrousTempViews.push_back(VulkanUtils::createImageView(dev.logicalDevice,
                                                               *atrousTemp.back(), vk::Format::eR16G16B16A16Sfloat, vk::ImageAspectFlagBits::eColor));
    }

    // Pre-transition A-Trous ping-pong buffers to eGeneral so they match the declared layout in
    // denoiserDescriptorSets, even when no path tracer denoiser pass has run yet.
    {
        auto cmd = VulkanUtils::beginSingleTimeCommands(dev.logicalDevice, commandPool);
        for (auto &img: atrousTemp)
            VulkanUtils::recordImageLayoutTransition(cmd, *img,
                                                     vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
        VulkanUtils::endSingleTimeCommands(dev.logicalDevice, dev.queue, commandPool, cmd);
    }
}

void FrameContext::createTLASResources(VulkanDevice &dev) {
    vk::AccelerationStructureGeometryInstancesDataKHR instancesData{};
    instancesData.arrayOfPointers = vk::False;

    vk::AccelerationStructureGeometryKHR instancesGeometry{};
    instancesGeometry.geometryType = vk::GeometryTypeKHR::eInstances;
    instancesGeometry.geometry.instances = instancesData;

    vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.type = vk::AccelerationStructureTypeKHR::eTopLevel;
    buildInfo.flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
    buildInfo.mode = vk::BuildAccelerationStructureModeKHR::eBuild;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &instancesGeometry;

    uint32_t primitiveCount = MAX_TLAS_INSTANCES;

    vk::AccelerationStructureBuildSizesInfoKHR sizeInfo = dev.logicalDevice.getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo, primitiveCount);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        // --- TLAS Storage Buffer ---
        VulkanUtils::VmaBuffer storageBuffer{};
        VulkanUtils::createBuffer(dev.logicalDevice, dev.physicalDevice, sizeInfo.accelerationStructureSize,
                                  vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                  vk::MemoryPropertyFlagBits::eDeviceLocal, storageBuffer);
        tlasBuffers.push_back(std::move(storageBuffer));

        vk::AccelerationStructureCreateInfoKHR createInfo{};
        createInfo.buffer = *tlasBuffers.back();
        createInfo.size = sizeInfo.accelerationStructureSize;
        createInfo.type = vk::AccelerationStructureTypeKHR::eTopLevel;
        tlas.emplace_back(dev.logicalDevice, createInfo);

        // --- Scratch Buffer ---
        VulkanUtils::VmaBuffer scratchBuffer{};
        VulkanUtils::createBuffer(dev.logicalDevice, dev.physicalDevice, sizeInfo.buildScratchSize,
                                  vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                  vk::MemoryPropertyFlagBits::eDeviceLocal, scratchBuffer);
        tlasScratchBuffers.push_back(std::move(scratchBuffer));
        tlasScratchAddresses.push_back(VulkanUtils::getBufferDeviceAddress(dev.logicalDevice, tlasScratchBuffers.back()));

        // --- Instance Buffer ---
        VulkanUtils::VmaBuffer instanceBuffer{};
        vk::DeviceSize instanceBufferSize = sizeof(vk::AccelerationStructureInstanceKHR) * MAX_TLAS_INSTANCES;
        VulkanUtils::createBuffer(dev.logicalDevice, dev.physicalDevice, instanceBufferSize,
                                  vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                  vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                  instanceBuffer);

        tlasInstanceBuffersMapped.push_back(instanceBuffer.memory.mapMemory(0, instanceBufferSize));
        tlasInstanceBuffers.push_back(std::move(instanceBuffer));
        tlasInstanceAddresses.push_back(VulkanUtils::getBufferDeviceAddress(dev.logicalDevice, tlasInstanceBuffers.back()));
    }
}
