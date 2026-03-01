#include "VulkanDevice.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace Laphria;

void VulkanDevice::init(GLFWwindow *window)
{
	createInstance();
	createSurface(window);
	pickPhysicalDevice();
	createLogicalDevice();
}

void VulkanDevice::createInstance()
{
	constexpr vk::ApplicationInfo appInfo{
	    .pApplicationName   = "Laphria Engine Development App",
	    .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
	    .pEngineName        = "Laphria",
	    .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
	    .apiVersion         = VK_API_VERSION_1_4};

	auto extensions = getRequiredExtensions();

	static constexpr const char *validationLayerName = "VK_LAYER_KHRONOS_validation";

	vk::InstanceCreateInfo createInfo{
	    .pApplicationInfo        = &appInfo,
	    .enabledExtensionCount   = static_cast<uint32_t>(extensions.size()),
	    .ppEnabledExtensionNames = extensions.data()};

	// Validation layers are enabled in debug builds only (NDEBUG not defined).
	if (enableValidationLayers)
	{
		createInfo.enabledLayerCount   = 1;
		createInfo.ppEnabledLayerNames = &validationLayerName;
	}

	instance = vk::raii::Instance(context, createInfo);
	LOGI("Vulkan instance created");
}

void VulkanDevice::createSurface(GLFWwindow *window)
{
	// Use the C API here because GLFW does not expose a vk::raii-compatible surface creator.
	// Wrap the raw handle in a vk::raii::SurfaceKHR immediately so it is destroyed correctly.
	VkSurfaceKHR _surface;
	if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create window surface!");
	}
	surface = vk::raii::SurfaceKHR(instance, _surface);
}

void VulkanDevice::pickPhysicalDevice()
{
	std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();

	struct ScoredDevice
	{
		uint32_t                 score;
		vk::raii::PhysicalDevice device;
	};

	std::vector<ScoredDevice> scoredDevices;

	for (const auto &device : devices)
	{
		uint32_t score = 0;
		auto     props = device.getProperties();

		// ── Hard requirements ─────────────────────────────────────────────
		// The engine relies on Vulkan 1.4 core features (synchronization2, dynamic rendering).
		bool supportsVulkan1_4 = props.apiVersion >= VK_API_VERSION_1_4;

		auto queueFamilies    = device.getQueueFamilyProperties();
		bool supportsGraphics = std::ranges::any_of(queueFamilies, [](const auto &qfp) {
			return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics);
		});

		auto availableExtensions   = device.enumerateDeviceExtensionProperties();
		bool supportsAllExtensions = std::ranges::all_of(requiredDeviceExtension, [&](auto const &req) {
			return std::ranges::any_of(availableExtensions, [&](auto const &avail) {
				return strcmp(avail.extensionName, req) == 0;
			});
		});

		if (!supportsVulkan1_4 || !supportsGraphics || !supportsAllExtensions)
		{
			continue;
		}

		// ── Scoring (higher is better) ────────────────────────────────────
		// Discrete GPUs are strongly preferred over integrated ones.
		if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
		{
			score += 10000;
		}
		else if (props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu)
		{
			score += 1000;
		}

		// Tie-break by device-local memory size (larger VRAM → higher score).
		auto memProps = device.getMemoryProperties();
		for (uint32_t i = 0; i < memProps.memoryHeapCount; i++)
		{
			if (memProps.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal)
			{
				score += static_cast<uint32_t>(memProps.memoryHeaps[i].size / (1024 * 1024));
			}
		}

		scoredDevices.push_back({score, device});
	}

	std::ranges::sort(scoredDevices, [](const auto &a, const auto &b) {
		return a.score > b.score;
	});

	if (!scoredDevices.empty())
	{
		physicalDevice = std::move(scoredDevices[0].device);
		auto props     = physicalDevice.getProperties();
		LOGI("Selected GPU: %s (Score: %u)", props.deviceName.data(), scoredDevices[0].score);

		// --- Extract Ray Tracing Properties ---
		// We use getProperties2 with a StructureChain to append the RT properties struct
		auto propsChain = physicalDevice.getProperties2<
		    vk::PhysicalDeviceProperties2,
		    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();

		rayTracingProperties = propsChain.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
		LOGI("Ray Tracing Properties Loaded:");
		// shaderGroupHandleSize: The exact size (in bytes) of the opaque identifier that represents a shader group (RayGen, Miss, or Hit). This is almost universally 32 bytes on modern hardware.
		LOGI("  - Shader Group Handle Size: %u bytes", rayTracingProperties.shaderGroupHandleSize);
		// shaderGroupBaseAlignment: The memory boundary (in bytes) to which the start of any SBT region must be aligned. This is typically 64 bytes.
		LOGI("  - Shader Group Base Alignment: %u bytes", rayTracingProperties.shaderGroupBaseAlignment);
		// shaderGroupHandleAlignment: The alignment requirement for individual handles within a shader group region.
		LOGI("  - Shader Group Handle Alignment: %u bytes", rayTracingProperties.shaderGroupHandleAlignment);
	}
	else
	{
		throw std::runtime_error("failed to find a suitable GPU!");
	}
}

void VulkanDevice::createLogicalDevice()
{
	std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

	// Find the first queue family that supports both graphics and present on our surface.
	// Using a single combined queue simplifies synchronization (no cross-queue ownership transfers).
	for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++)
	{
		if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
		    physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface))
		{
			queueIndex = qfpIndex;
			break;
		}
	}
	if (queueIndex == ~0u)
	{
		throw std::runtime_error("Could not find a queue for graphics and present -> terminating");
	}

	vk::StructureChain<
	    vk::PhysicalDeviceFeatures2,
	    vk::PhysicalDeviceVulkan13Features,
	    vk::PhysicalDeviceBufferDeviceAddressFeatures,
	    vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
	    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
	    vk::PhysicalDeviceDescriptorIndexingFeatures>
	    featureChain;

	auto &physicalDeviceFeatures             = featureChain.get<vk::PhysicalDeviceFeatures2>().features;
	physicalDeviceFeatures.samplerAnisotropy = vk::True;
	// depthClamp prevents geometry outside the near/far planes from being clipped —
	// required for the shadow pass so casters behind the light frustum are still recorded.
	physicalDeviceFeatures.depthClamp = vk::True;

	// Vulkan 1.3 core features used by the engine:
	//   - synchronization2: VkImageMemoryBarrier2 / pipelineBarrier2.
	//   - dynamicRendering: render passes without VkRenderPass/VkFramebuffer objects.
	auto &vulkan13Features            = featureChain.get<vk::PhysicalDeviceVulkan13Features>();
	vulkan13Features.synchronization2 = vk::True,
	vulkan13Features.dynamicRendering = vk::True;

	// Enable bindless texturing features (required by PipelineCollection descriptor set layout):
	//   - NonUniformIndexing + UpdateAfterBind: textures can be indexed dynamically in shaders.
	//   - PartiallyBound: descriptor slots may remain unbound if not used by a draw call.
	//   - VariableDescriptorCount + RuntimeArray: allows arrays of arbitrary (runtime) size.
	auto &indexingFeatures                                         = featureChain.get<vk::PhysicalDeviceDescriptorIndexingFeatures>();
	indexingFeatures.runtimeDescriptorArray                        = vk::True;
	indexingFeatures.shaderSampledImageArrayNonUniformIndexing     = vk::True;
	indexingFeatures.shaderStorageBufferArrayNonUniformIndexing    = vk::True;
	indexingFeatures.descriptorBindingSampledImageUpdateAfterBind  = vk::True;
	indexingFeatures.descriptorBindingStorageBufferUpdateAfterBind = vk::True;
	indexingFeatures.descriptorBindingPartiallyBound               = vk::True;
	indexingFeatures.descriptorBindingVariableDescriptorCount      = vk::True;

	auto &bdaFeatures               = featureChain.get<vk::PhysicalDeviceBufferDeviceAddressFeatures>();
	bdaFeatures.bufferDeviceAddress = vk::True;

	auto &asFeatures                 = featureChain.get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>();
	asFeatures.accelerationStructure = vk::True;

	auto &rtFeatures              = featureChain.get<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>();
	rtFeatures.rayTracingPipeline = vk::True;

	float                     queuePriority = 0.5f;
	vk::DeviceQueueCreateInfo deviceQueueCreateInfo{
	    .queueFamilyIndex = queueIndex,
	    .queueCount       = 1,
	    .pQueuePriorities = &queuePriority};

	vk::DeviceCreateInfo deviceCreateInfo{
	    .pNext                   = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
	    .queueCreateInfoCount    = 1,
	    .pQueueCreateInfos       = &deviceQueueCreateInfo,
	    .enabledExtensionCount   = static_cast<uint32_t>(requiredDeviceExtension.size()),
	    .ppEnabledExtensionNames = requiredDeviceExtension.data()};

	logicalDevice = vk::raii::Device(physicalDevice, deviceCreateInfo);
	queue         = vk::raii::Queue(logicalDevice, queueIndex, 0);
}

vk::Format VulkanDevice::findDepthFormat() const
{
	// Prefer a pure 32-bit depth format; fall back to combined depth-stencil variants.
	return findSupportedFormat(
	    {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
	    vk::ImageTiling::eOptimal,
	    vk::FormatFeatureFlagBits::eDepthStencilAttachment);
}

vk::Format VulkanDevice::findSupportedFormat(const std::vector<vk::Format> &candidates,
                                             vk::ImageTiling                tiling,
                                             vk::FormatFeatureFlags         features) const
{
	for (const auto format : candidates)
	{
		vk::FormatProperties props = physicalDevice.getFormatProperties(format);

		if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features)
		{
			return format;
		}
		if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features)
		{
			return format;
		}
	}

	throw std::runtime_error("failed to find supported format!");
}

std::vector<const char *> VulkanDevice::getRequiredExtensions()
{
	std::vector<const char *> extensions;

	// GLFW requires a platform-specific surface extension (e.g. VK_KHR_win32_surface).
	uint32_t glfwExtensionCount = 0;
	auto     glfwExtensions     = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
	extensions.assign(glfwExtensions, glfwExtensions + glfwExtensionCount);

	// The debug utils extension is only needed when validation layers are active.
	if (enableValidationLayers)
	{
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}

	return extensions;
}
