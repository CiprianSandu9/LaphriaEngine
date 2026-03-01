#include "VulkanUtils.h"

namespace Laphria
{
namespace VulkanUtils
{
uint32_t findMemoryType(const vk::raii::PhysicalDevice &physicalDevice, uint32_t typeFilter, vk::MemoryPropertyFlags properties)
{
	vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
	{
		if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
		{
			return i;
		}
	}
	throw std::runtime_error("failed to find suitable memory type!");
}

uint32_t alignUp(uint32_t size, uint32_t alignment)
{
	return (size + alignment - 1) & ~(alignment - 1);
}

void createBuffer(const vk::raii::Device &device, const vk::raii::PhysicalDevice &physicalDevice,
                  vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties,
                  vk::raii::Buffer &buffer, vk::raii::DeviceMemory &bufferMemory)
{
	vk::BufferCreateInfo bufferInfo{};
	bufferInfo.size        = size;
	bufferInfo.usage       = usage;
	bufferInfo.sharingMode = vk::SharingMode::eExclusive;

	buffer = vk::raii::Buffer(device, bufferInfo);

	vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
	vk::MemoryAllocateInfo allocInfo{};
	allocInfo.allocationSize  = memRequirements.size;
	allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

	vk::MemoryAllocateFlagsInfo allocFlagsInfo{};
	if (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress)
	{
		allocFlagsInfo.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
		allocInfo.pNext      = &allocFlagsInfo;
	}

	bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
	buffer.bindMemory(*bufferMemory, 0);
}

// Allocates a one-shot command buffer and begins recording.
// Pair with endSingleTimeCommands() which submits and BLOCKS until the queue is idle.
// Avoid in hot paths; prefer recording into the frame's command buffer instead.
vk::raii::CommandBuffer beginSingleTimeCommands(const vk::raii::Device &device, const vk::raii::CommandPool &commandPool)
{
	vk::CommandBufferAllocateInfo allocInfo{};
	allocInfo.level              = vk::CommandBufferLevel::ePrimary;
	allocInfo.commandPool        = *commandPool;
	allocInfo.commandBufferCount = 1;

	vk::raii::CommandBuffer    commandBuffer = std::move(vk::raii::CommandBuffers(device, allocInfo).front());
	vk::CommandBufferBeginInfo beginInfo{};
	beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	commandBuffer.begin(beginInfo);
	return commandBuffer;
}

// Submits the command buffer and stalls the calling thread until the queue drains.
// This is intentionally synchronous (queue.waitIdle) and is only acceptable for
// load-time operations such as texture/buffer uploads.
void endSingleTimeCommands(const vk::raii::Device &device, const vk::raii::Queue &queue, const vk::raii::CommandPool &commandPool, const vk::raii::CommandBuffer &commandBuffer)
{
	commandBuffer.end();
	vk::SubmitInfo submitInfo{};
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers    = &*commandBuffer;
	queue.submit(submitInfo, nullptr);
	queue.waitIdle();
}

void copyBuffer(const vk::raii::Device &device, const vk::raii::CommandPool &commandPool, const vk::raii::Queue &queue,
                const vk::raii::Buffer &srcBuffer, const vk::raii::Buffer &dstBuffer, vk::DeviceSize size)
{
	auto           commandBuffer = beginSingleTimeCommands(device, commandPool);
	vk::BufferCopy copyRegion{};
	copyRegion.size = size;
	commandBuffer.copyBuffer(*srcBuffer, *dstBuffer, copyRegion);
	endSingleTimeCommands(device, queue, commandPool, commandBuffer);
}

void createImage(const vk::raii::Device &device, const vk::raii::PhysicalDevice &physicalDevice,
                 uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling,
                 vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties,
                 vk::raii::Image &image, vk::raii::DeviceMemory &imageMemory,
                 uint32_t arrayLayers)
{
	vk::ImageCreateInfo imageInfo{};
	imageInfo.imageType     = vk::ImageType::e2D;
	imageInfo.extent.width  = width;
	imageInfo.extent.height = height;
	imageInfo.extent.depth  = 1;
	imageInfo.mipLevels     = 1;
	imageInfo.arrayLayers   = arrayLayers;
	imageInfo.format        = format;
	imageInfo.tiling        = tiling;
	imageInfo.initialLayout = vk::ImageLayout::eUndefined;
	imageInfo.usage         = usage;
	imageInfo.samples       = vk::SampleCountFlagBits::e1;
	imageInfo.sharingMode   = vk::SharingMode::eExclusive;
	// A 2D image with arrayLayers > 1 is naturally a 2D Array.
	// The e2DArrayCompatible flag is ONLY for 3D textures that need 2D array views.
	// if (arrayLayers > 1)
	// 	imageInfo.flags = vk::ImageCreateFlagBits::e2DArrayCompatible;

	image = vk::raii::Image(device, imageInfo);

	vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
	vk::MemoryAllocateInfo allocInfo{};
	allocInfo.allocationSize  = memRequirements.size;
	allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

	imageMemory = vk::raii::DeviceMemory(device, allocInfo);
	image.bindMemory(*imageMemory, 0);
}

vk::raii::ImageView createImageView(const vk::raii::Device &device, vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags)
{
	vk::ImageViewCreateInfo viewInfo{};
	viewInfo.image                           = image;
	viewInfo.viewType                        = vk::ImageViewType::e2D;
	viewInfo.format                          = format;
	viewInfo.subresourceRange.aspectMask     = aspectFlags;
	viewInfo.subresourceRange.baseMipLevel   = 0;
	viewInfo.subresourceRange.levelCount     = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount     = 1;

	return vk::raii::ImageView(device, viewInfo);
}

vk::raii::ImageView createImageViewLayer(const vk::raii::Device &device, vk::Image image, vk::Format format,
                                         vk::ImageAspectFlags aspectFlags, uint32_t baseArrayLayer)
{
	vk::ImageViewCreateInfo viewInfo{};
	viewInfo.image                           = image;
	viewInfo.viewType                        = vk::ImageViewType::e2D;
	viewInfo.format                          = format;
	viewInfo.subresourceRange.aspectMask     = aspectFlags;
	viewInfo.subresourceRange.baseMipLevel   = 0;
	viewInfo.subresourceRange.levelCount     = 1;
	viewInfo.subresourceRange.baseArrayLayer = baseArrayLayer;
	viewInfo.subresourceRange.layerCount     = 1;

	return vk::raii::ImageView(device, viewInfo);
}

vk::raii::ImageView createImageViewArray(const vk::raii::Device &device, vk::Image image, vk::Format format,
                                         vk::ImageAspectFlags aspectFlags, uint32_t layerCount)
{
	vk::ImageViewCreateInfo viewInfo{};
	viewInfo.image                           = image;
	viewInfo.viewType                        = vk::ImageViewType::e2DArray;
	viewInfo.format                          = format;
	viewInfo.subresourceRange.aspectMask     = aspectFlags;
	viewInfo.subresourceRange.baseMipLevel   = 0;
	viewInfo.subresourceRange.levelCount     = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount     = layerCount;

	return vk::raii::ImageView(device, viewInfo);
}

// Records a Vulkan 1.0-style image memory barrier (VkImageMemoryBarrier) for a
// predefined set of common layout transitions. Uses the older pipelineBarrier() API
// rather than pipelineBarrier2(), which is reserved for the inline barriers in EngineCore
// that need fine-grained Synchronization2 control.
void recordImageLayoutTransition(const vk::raii::CommandBuffer &commandBuffer, vk::Image image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                                 vk::ImageAspectFlags aspectMask)
{
	vk::ImageMemoryBarrier barrier{};
	barrier.oldLayout                       = oldLayout;
	barrier.newLayout                       = newLayout;
	barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	barrier.image                           = image;
	barrier.subresourceRange.aspectMask     = aspectMask;
	barrier.subresourceRange.baseMipLevel   = 0;
	barrier.subresourceRange.levelCount     = 1;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount     = 1;

	vk::PipelineStageFlags sourceStage;
	vk::PipelineStageFlags destinationStage;

	if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal)
	{
		barrier.srcAccessMask = {};
		barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
		sourceStage           = vk::PipelineStageFlagBits::eTopOfPipe;
		destinationStage      = vk::PipelineStageFlagBits::eTransfer;
	}
	else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
	{
		barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
		sourceStage           = vk::PipelineStageFlagBits::eTransfer;
		destinationStage      = vk::PipelineStageFlagBits::eFragmentShader;
	}
	else if (oldLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
	{
		barrier.srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
		sourceStage           = vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests;
		destinationStage      = vk::PipelineStageFlagBits::eFragmentShader;
	}
	else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal)
	{
		barrier.srcAccessMask = {};
		barrier.dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
		sourceStage           = vk::PipelineStageFlagBits::eTopOfPipe;
		destinationStage      = vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests;
	}
	else
	{
		throw std::invalid_argument("unsupported layout transition!");
	}

	commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, {}, barrier);
}

void recordCopyBufferToImage(const vk::raii::CommandBuffer &commandBuffer, vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height)
{
	vk::BufferImageCopy region{};
	region.bufferOffset                    = 0;
	region.bufferRowLength                 = 0;
	region.bufferImageHeight               = 0;
	region.imageSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
	region.imageSubresource.mipLevel       = 0;
	region.imageSubresource.baseArrayLayer = 0;
	region.imageSubresource.layerCount     = 1;
	region.imageOffset                     = {0, 0, 0};
	region.imageExtent                     = {width, height, 1};
	commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
}

// Uploads CPU data to a device-local buffer via a host-visible staging buffer.
// Pattern: allocate host-coherent staging buffer → memcpy → copy to device-local buffer.
// The staging buffer is destroyed at the end of this call (RAII).
void createDeviceLocalBufferFromData(const vk::raii::Device &device, const vk::raii::PhysicalDevice &physicalDevice,
                                     const vk::raii::CommandPool &commandPool, const vk::raii::Queue &queue,
                                     const void *data, vk::DeviceSize size, vk::BufferUsageFlags usage,
                                     vk::raii::Buffer &buffer, vk::raii::DeviceMemory &bufferMemory)
{
	vk::raii::Buffer       stagingBuffer{nullptr};
	vk::raii::DeviceMemory stagingMemory{nullptr};

	createBuffer(device, physicalDevice, size, vk::BufferUsageFlagBits::eTransferSrc,
	             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
	             stagingBuffer, stagingMemory);

	void *mapped = stagingMemory.mapMemory(0, size);
	memcpy(mapped, data, size);
	stagingMemory.unmapMemory();

	createBuffer(device, physicalDevice, size, vk::BufferUsageFlagBits::eTransferDst | usage,
	             vk::MemoryPropertyFlagBits::eDeviceLocal, buffer, bufferMemory);

	copyBuffer(device, commandPool, queue, stagingBuffer, buffer, size);
}

// Uploads raw pixel data to a device-local sampled image via a staging buffer.
// Transitions: Undefined → TransferDst → ShaderReadOnly.
void createTextureImageFromData(const vk::raii::Device &device, const vk::raii::PhysicalDevice &physicalDevice,
                                const vk::raii::CommandPool &commandPool, const vk::raii::Queue &queue,
                                const void *data, vk::DeviceSize size, uint32_t width, uint32_t height, vk::Format format,
                                vk::raii::Image &image, vk::raii::DeviceMemory &imageMemory)
{
	vk::raii::Buffer       stagingBuffer{nullptr};
	vk::raii::DeviceMemory stagingMemory{nullptr};

	createBuffer(device, physicalDevice, size, vk::BufferUsageFlagBits::eTransferSrc,
	             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
	             stagingBuffer, stagingMemory);

	void *mapped = stagingMemory.mapMemory(0, size);
	memcpy(mapped, data, size);
	stagingMemory.unmapMemory();

	createImage(device, physicalDevice, width, height, format, vk::ImageTiling::eOptimal,
	            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
	            vk::MemoryPropertyFlagBits::eDeviceLocal, image, imageMemory);

	transitionImageLayout(device, commandPool, queue, *image, format, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
	copyBufferToImage(device, commandPool, queue, *stagingBuffer, *image, width, height);
	transitionImageLayout(device, commandPool, queue, *image, format, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
}

void transitionImageLayout(const vk::raii::Device &device, const vk::raii::CommandPool &commandPool, const vk::raii::Queue &queue,
                           vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout)
{
	auto                 commandBuffer = beginSingleTimeCommands(device, commandPool);
	vk::ImageAspectFlags aspectMask    = vk::ImageAspectFlagBits::eColor;

	recordImageLayoutTransition(commandBuffer, image, oldLayout, newLayout, aspectMask);

	endSingleTimeCommands(device, queue, commandPool, commandBuffer);
}

void copyBufferToImage(const vk::raii::Device &device, const vk::raii::CommandPool &commandPool, const vk::raii::Queue &queue,
                       vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height)
{
	auto commandBuffer = beginSingleTimeCommands(device, commandPool);
	recordCopyBufferToImage(commandBuffer, buffer, image, width, height);
	endSingleTimeCommands(device, queue, commandPool, commandBuffer);
}

vk::DeviceAddress getBufferDeviceAddress(const vk::raii::Device &device, const vk::raii::Buffer &buffer)
{
	vk::BufferDeviceAddressInfo info{};
	info.buffer = *buffer;
	return device.getBufferAddress(info);
}
}        // namespace VulkanUtils
}        // namespace Laphria
