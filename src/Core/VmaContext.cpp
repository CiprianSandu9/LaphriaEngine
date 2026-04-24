#define VMA_IMPLEMENTATION
#include "VmaContext.h"

#include <mutex>
#include <stdexcept>

namespace Laphria::VmaContext
{
namespace
{
std::mutex gMutex;
VmaAllocator gAllocator = VK_NULL_HANDLE;
} // namespace

void initialize(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device)
{
	std::scoped_lock lock(gMutex);
	if (gAllocator != VK_NULL_HANDLE)
	{
		return;
	}

	VmaAllocatorCreateInfo allocatorInfo{};
	allocatorInfo.instance = instance;
	allocatorInfo.physicalDevice = physicalDevice;
	allocatorInfo.device = device;
	allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_4;
	allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

	const VkResult result = vmaCreateAllocator(&allocatorInfo, &gAllocator);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create VMA allocator");
	}
}

void shutdown()
{
	std::scoped_lock lock(gMutex);
	if (gAllocator != VK_NULL_HANDLE)
	{
		vmaDestroyAllocator(gAllocator);
		gAllocator = VK_NULL_HANDLE;
	}
}

bool isInitialized()
{
	std::scoped_lock lock(gMutex);
	return gAllocator != VK_NULL_HANDLE;
}

VmaAllocator get()
{
	std::scoped_lock lock(gMutex);
	if (gAllocator == VK_NULL_HANDLE)
	{
		throw std::runtime_error("VMA allocator is not initialized");
	}
	return gAllocator;
}

Stats getStats()
{
	std::scoped_lock lock(gMutex);
	if (gAllocator == VK_NULL_HANDLE)
	{
		return {};
	}

	VmaTotalStatistics totalStats{};
	vmaCalculateStatistics(gAllocator, &totalStats);

	Stats stats{};
	stats.blockCount = totalStats.total.statistics.blockCount;
	stats.allocationCount = totalStats.total.statistics.allocationCount;
	stats.allocationBytes = totalStats.total.statistics.allocationBytes;
	return stats;
}
} // namespace Laphria::VmaContext
