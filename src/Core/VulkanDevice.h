#ifndef LAPHRIAENGINE_VULKANDEVICE_H
#define LAPHRIAENGINE_VULKANDEVICE_H

#include "EngineAuxiliary.h"
#include <vector>

// Owns Vulkan instance, physical/logical device, surface and queue.
// All downstream subsystems receive a reference to this object.
class VulkanDevice {
public:
    void init(GLFWwindow *window);

    [[nodiscard]] vk::Format findDepthFormat() const;
    [[nodiscard]] vk::Format findSupportedFormat(const std::vector<vk::Format> &candidates,
                                                  vk::ImageTiling tiling,
                                                  vk::FormatFeatureFlags features) const;

    // Public handles – accessed directly by EngineCore and other subsystems.
    vk::raii::Context                context;
    vk::raii::Instance               instance{nullptr};
    vk::raii::DebugUtilsMessengerEXT debugMessenger{nullptr};
    vk::raii::SurfaceKHR             surface{nullptr};
    vk::raii::PhysicalDevice         physicalDevice{nullptr};
    vk::raii::Device                 logicalDevice{nullptr};
    uint32_t                         queueIndex = ~0u; // ~0 == UINT32_MAX, Vulkan convention
    vk::raii::Queue                  queue{nullptr};
	// Ray Tracing hardware properties
	vk::PhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingProperties;

    std::vector<const char *> requiredDeviceExtension = {
        vk::KHRSwapchainExtensionName,
        vk::KHRCreateRenderpass2ExtensionName,
        vk::EXTDescriptorIndexingExtensionName,
    	vk::KHRAccelerationStructureExtensionName,
    	vk::KHRRayTracingPipelineExtensionName,
		vk::KHRDeferredHostOperationsExtensionName
    };

private:
    void createInstance();
    void createSurface(GLFWwindow *window);
    void pickPhysicalDevice();
    void createLogicalDevice();

    static std::vector<const char *> getRequiredExtensions();
};

#endif // LAPHRIAENGINE_VULKANDEVICE_H
