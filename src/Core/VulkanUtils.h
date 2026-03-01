#ifndef LAPHRIAENGINE_VULKANUTILS_H
#define LAPHRIAENGINE_VULKANUTILS_H

#include "EngineAuxiliary.h"

namespace Laphria
{
namespace VulkanUtils
{
// Memory Helper
uint32_t findMemoryType(const vk::raii::PhysicalDevice &physicalDevice, uint32_t typeFilter, vk::MemoryPropertyFlags properties);
uint32_t alignUp(uint32_t size, uint32_t alignment);

// Buffer Helpers
void createBuffer(const vk::raii::Device &device, const vk::raii::PhysicalDevice &physicalDevice,
                  vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties,
                  vk::raii::Buffer &buffer, vk::raii::DeviceMemory &bufferMemory);

void copyBuffer(const vk::raii::Device &device, const vk::raii::CommandPool &commandPool, const vk::raii::Queue &queue,
                const vk::raii::Buffer &srcBuffer, const vk::raii::Buffer &dstBuffer, vk::DeviceSize size);

// Image Helpers
void createImage(const vk::raii::Device &device, const vk::raii::PhysicalDevice &physicalDevice,
                 uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling,
                 vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties,
                 vk::raii::Image &image, vk::raii::DeviceMemory &imageMemory,
                 uint32_t arrayLayers = 1);

vk::raii::ImageView createImageView(const vk::raii::Device &device, vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags);

// Creates a 2D image view for a single layer of an array image.
vk::raii::ImageView createImageViewLayer(const vk::raii::Device &device, vk::Image image, vk::Format format,
                                         vk::ImageAspectFlags aspectFlags, uint32_t baseArrayLayer);

// Creates a 2D_ARRAY image view spanning all layers of an array image.
vk::raii::ImageView createImageViewArray(const vk::raii::Device &device, vk::Image image, vk::Format format,
                                         vk::ImageAspectFlags aspectFlags, uint32_t layerCount);

void transitionImageLayout(const vk::raii::Device &device, const vk::raii::CommandPool &commandPool, const vk::raii::Queue &queue,
                           vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);

void copyBufferToImage(const vk::raii::Device &device, const vk::raii::CommandPool &commandPool, const vk::raii::Queue &queue,
                       vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height);

// Recording Helpers (Non-blocking, for existing command buffers)
void recordImageLayoutTransition(const vk::raii::CommandBuffer &commandBuffer, vk::Image image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                                 vk::ImageAspectFlags aspectMask = vk::ImageAspectFlagBits::eColor);

void recordCopyBufferToImage(const vk::raii::CommandBuffer &commandBuffer, vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height);

// Composite Helpers (Staging -> Device)
void createDeviceLocalBufferFromData(const vk::raii::Device &device, const vk::raii::PhysicalDevice &physicalDevice,
                                     const vk::raii::CommandPool &commandPool, const vk::raii::Queue &queue,
                                     const void *data, vk::DeviceSize size, vk::BufferUsageFlags usage,
                                     vk::raii::Buffer &buffer, vk::raii::DeviceMemory &bufferMemory);

void createTextureImageFromData(const vk::raii::Device &device, const vk::raii::PhysicalDevice &physicalDevice,
                                const vk::raii::CommandPool &commandPool, const vk::raii::Queue &queue,
                                const void *data, vk::DeviceSize size, uint32_t width, uint32_t height, vk::Format format,
                                vk::raii::Image &image, vk::raii::DeviceMemory &imageMemory);

// Command Helpers
vk::raii::CommandBuffer beginSingleTimeCommands(const vk::raii::Device &device, const vk::raii::CommandPool &commandPool);

void endSingleTimeCommands(const vk::raii::Device &device, const vk::raii::Queue &queue, const vk::raii::CommandPool &commandPool, const vk::raii::CommandBuffer &commandBuffer);

vk::DeviceAddress getBufferDeviceAddress(const vk::raii::Device &device, const vk::raii::Buffer &buffer);
}        // namespace VulkanUtils
}        // namespace Laphria

#endif        // LAPHRIAENGINE_VULKANUTILS_H
