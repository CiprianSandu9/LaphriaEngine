#ifndef LAPHRIAENGINE_VMACONTEXT_H
#define LAPHRIAENGINE_VMACONTEXT_H

#include <vma/vk_mem_alloc.h>

namespace Laphria::VmaContext
{
struct Stats
{
	uint32_t blockCount = 0;
	uint32_t allocationCount = 0;
	VkDeviceSize allocationBytes = 0;
};

void initialize(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device);
void shutdown();

[[nodiscard]] bool isInitialized();
[[nodiscard]] VmaAllocator get();
[[nodiscard]] Stats getStats();
} // namespace Laphria::VmaContext

#endif // LAPHRIAENGINE_VMACONTEXT_H
