#include "SwapchainManager.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

void SwapchainManager::init(VulkanDevice &dev, GLFWwindow *window) {
    createSwapChain(dev, window);
    createImageViews(dev);
}

void SwapchainManager::cleanup() {
    // Image views must be destroyed before the swapchain that owns the images.
    imageViews.clear();
    swapChain = nullptr;
}

void SwapchainManager::createSwapChain(VulkanDevice &dev, GLFWwindow *window) {
    auto surfaceCapabilities = dev.physicalDevice.getSurfaceCapabilitiesKHR(*dev.surface);
    extent = chooseSwapExtent(surfaceCapabilities, window);
    surfaceFormat = chooseSwapSurfaceFormat(dev.physicalDevice.getSurfaceFormatsKHR(*dev.surface));

    vk::SwapchainCreateInfoKHR swapChainCreateInfo{
        .surface = *dev.surface,
        .minImageCount = chooseSwapMinImageCount(surfaceCapabilities),
        .imageFormat = surfaceFormat.format,
        .imageColorSpace = surfaceFormat.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        // eColorAttachment: used as a render target in the main pass.
        // eTransferDst: required because the compute starfield blits into the swapchain image.
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst,
        .imageSharingMode = vk::SharingMode::eExclusive,
        .preTransform = surfaceCapabilities.currentTransform,
        .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode = chooseSwapPresentMode(dev.physicalDevice.getSurfacePresentModesKHR(*dev.surface)),
        .clipped = true
    };

    swapChain = vk::raii::SwapchainKHR(dev.logicalDevice, swapChainCreateInfo);
    images = swapChain.getImages();
}

void SwapchainManager::createImageViews(VulkanDevice &dev) {
    assert(imageViews.empty());

    vk::ImageViewCreateInfo imageViewCreateInfo{
        .viewType = vk::ImageViewType::e2D,
        .format = surfaceFormat.format,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
    };
    for (const auto &image: images) {
        imageViewCreateInfo.image = image;
        imageViews.emplace_back(dev.logicalDevice, imageViewCreateInfo);
    }
}

vk::Extent2D SwapchainManager::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities,
                                                 GLFWwindow *window) const {
    // On most platforms currentExtent equals the window size.
    // A value of 0xFFFFFFFF indicates the platform lets us choose our own extent
    // (e.g. Wayland), so we query GLFW for the actual framebuffer size.
    if (capabilities.currentExtent.width != 0xFFFFFFFF) {
        return capabilities.currentExtent;
    }
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    return {
        std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
        std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
    };
}

uint32_t SwapchainManager::chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const &surfaceCapabilities) {
    // Request triple-buffering (3 images) to reduce latency compared to double-buffering.
    // Cap at maxImageCount if the driver enforces an upper limit (0 means no limit).
    auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
    if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount)) {
        minImageCount = surfaceCapabilities.maxImageCount;
    }
    return minImageCount;
}

vk::SurfaceFormatKHR SwapchainManager::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats) {
    if (availableFormats.empty())
        throw std::runtime_error("No swap chain surface formats available");
    // Prefer B8G8R8A8_SRGB + sRGB non-linear: gives correct gamma-corrected output
    // without a manual tonemapping step. Fall back to the first available format.
    const auto formatIt = std::ranges::find_if(
        availableFormats,
        [](const auto &format) {
            return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace ==
                   vk::ColorSpaceKHR::eSrgbNonlinear;
        });
    return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
}

vk::PresentModeKHR SwapchainManager::chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> &availablePresentModes) {
    // FIFO is guaranteed by the spec; throw if a driver omits it (spec violation).
    if (!std::ranges::any_of(availablePresentModes, [](auto presentMode) {
            return presentMode == vk::PresentModeKHR::eFifo;
        }))
        throw std::runtime_error("Vulkan driver missing mandatory FIFO present mode (spec violation)");
    // Prefer Mailbox (triple-buffering equivalent): no tearing, lower latency than FIFO.
    // Fall back to FIFO (vsync) if Mailbox is unavailable.
    return std::ranges::any_of(availablePresentModes, [](const vk::PresentModeKHR value) {
        return vk::PresentModeKHR::eMailbox == value;
    })
               ? vk::PresentModeKHR::eMailbox
               : vk::PresentModeKHR::eFifo;
}
