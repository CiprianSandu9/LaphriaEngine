#ifndef LAPHRIAENGINE_SWAPCHAINMANAGER_H
#define LAPHRIAENGINE_SWAPCHAINMANAGER_H

#include "EngineAuxiliary.h"
#include "VulkanDevice.h"

// Owns swapchain, per-swapchain image views, and format/extent helpers.
// init() creates both swapchain and image views.
// cleanup() destroys them in the correct order.
class SwapchainManager {
public:
    // Creates swapchain + image views.
    void init(VulkanDevice &dev, GLFWwindow *window);

    // Destroys image views then the swapchain.
    void cleanup();

    // Set by the InputSystem framebuffer-resize callback.
    bool framebufferResized = false;

    vk::raii::SwapchainKHR           swapChain{nullptr};
    std::vector<vk::Image>           images;
    vk::SurfaceFormatKHR             surfaceFormat;
    vk::Extent2D                     extent;
    std::vector<vk::raii::ImageView> imageViews;

private:
    void createSwapChain(VulkanDevice &dev, GLFWwindow *window);
    void createImageViews(VulkanDevice &dev);

    [[nodiscard]] vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities,
                                                GLFWwindow *window) const;
    static uint32_t              chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const &surfaceCapabilities);
    static vk::SurfaceFormatKHR  chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats);
    static vk::PresentModeKHR    chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> &availablePresentModes);
};

#endif // LAPHRIAENGINE_SWAPCHAINMANAGER_H
