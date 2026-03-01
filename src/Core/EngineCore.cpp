#include "EngineCore.h"
#include "VulkanUtils.h"

#include <array>
#include <cassert>
#include <chrono>
#include <glm/glm.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../SceneManagement/Scene.h"
#include "EngineAuxiliary.h"
#include "ResourceManager.h"

void EngineCore::initWindow()
{
	glfwInit();

	// GLFW_NO_API: we manage the Vulkan surface ourselves, not via an OpenGL context.
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

	window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
	if (!window)
	{
		throw std::runtime_error("failed to create GLFW window");
	}
}

void EngineCore::initInput()
{
	input.init(window, camera, swapchain.framebufferResized);
}

void EngineCore::initVulkan()
{
	// Ordering matters here:
	//  1. vulkan → swapchain: surface must exist before swapchain creation.
	//  2. frames → createDescriptorPool: commandPool must exist before ResourceManager.
	//  3. ResourceManager receives a reference to descriptorPool, so the pool must be alive
	//     for the entire lifetime of the ResourceManager.
	//  4. Descriptor set layouts must precede pipeline creation.
	//  5. Descriptor sets must be written after both pool and uniform buffers/images exist.
	vulkan.init(window);
	swapchain.init(vulkan, window);
	frames.init(vulkan, swapchain);
	createDescriptorPool();

	resourceManager = std::make_unique<ResourceManager>(vulkan.logicalDevice, vulkan.physicalDevice, frames.commandPool, vulkan.queue,
	                                                    descriptorPool);
	scene           = std::make_unique<Scene>();
	scene->init({{-1000, -1000, -1000}, {1000, 1000, 1000}});        // Big bounds to ensure the model fits

	physicsSystem = std::make_unique<PhysicsSystem>();

	pipelines.createDescriptorSetLayouts(vulkan);

	// Pipeline creation order matches dependency on the descriptor set layouts above.
	pipelines.createGraphicsPipeline(vulkan, swapchain.surfaceFormat.format, vulkan.findDepthFormat());
	pipelines.createShadowPipeline(vulkan);
	pipelines.createComputePipeline(vulkan);
	pipelines.createPhysicsPipeline(vulkan);
	pipelines.createRayTracingPipeline(vulkan);
	pipelines.createShaderBindingTable(vulkan);

	createDescriptorSets();
	createComputeDescriptorSets();
	createPhysicsDescriptorSets();
	createRayTracingDescriptorSets();
}

void EngineCore::initImgui()
{
	ui.init(vulkan, window, swapchain.surfaceFormat.format, vulkan.findDepthFormat());
}

void EngineCore::mainLoop()
{
	size_t prevModelCount = resourceManager->getModelCount();

	while (!glfwWindowShouldClose(window))
	{
		// Delta Time calculation
		static auto lastTime    = std::chrono::high_resolution_clock::now();
		auto        currentTime = std::chrono::high_resolution_clock::now();
		float       deltaTime   = std::chrono::duration<float>(currentTime - lastTime).count();
		lastTime                = currentTime;

		glfwPollEvents();
		camera.update();

		// Physics Update
		if (ui.simulationRunning && physicsSystem)
		{
			auto start = std::chrono::high_resolution_clock::now();

			if (ui.useGPUPhysics)
			{
				// For simplicity in this prototype:
				// Using `beginSingleTimeCommands` to run physics immediately here.
				// This stalls the CPU (via queue.waitIdle inside endSingleTimeCommands) but simplifies
				// synchronization since we need the readback result on the same frame anyway.
				// Known performance limitation: a production implementation would use async transfers
				// with double-buffered SSBOs and a dedicated transfer queue to avoid the stall.

				auto cmd = VulkanUtils::beginSingleTimeCommands(vulkan.logicalDevice, frames.commandPool);
				physicsSystem->updateGPU(scene->getAllNodes(), deltaTime, cmd, pipelines.physicsPipelineLayout, pipelines.physicsPipeline, physicsDescriptorSet);
				// Barrier is inside updateGPU.
				VulkanUtils::endSingleTimeCommands(vulkan.logicalDevice, vulkan.queue, frames.commandPool, cmd);

				// Readback immediately
				physicsSystem->syncFromGPU(scene->getAllNodes());
			}
			else
			{
				physicsSystem->updateCPU(scene->getAllNodes(), deltaTime);
			}

			auto end       = std::chrono::high_resolution_clock::now();
			ui.physicsTime = std::chrono::duration<float, std::milli>(end - start).count();
		}

		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ui.draw(window, *scene, *physicsSystem, *resourceManager, *pipelines.descriptorSetLayoutMaterial);

		// If models were loaded during the UI frame, the RT descriptor sets (bindings 2-5:
		// vertex/index/material/texture arrays) must be rebuilt to include the new buffers.
		// buildBLAS already called queue.waitIdle(), so the queue is idle here.
		size_t currentModelCount = resourceManager->getModelCount();
		if (currentModelCount != prevModelCount)
		{
			prevModelCount = currentModelCount;
			createRayTracingDescriptorSets();
		}

		ImGui::Render();

		drawFrame();
	}

	vulkan.logicalDevice.waitIdle();
}

void EngineCore::cleanupSwapChain()
{
	swapchain.cleanup();
	frames.cleanupSwapChainDependents();
}

void EngineCore::cleanup()
{
	ui.cleanup();

	glfwDestroyWindow(window);
	glfwTerminate();
}

void EngineCore::recreateSwapChain()
{
	// A zero-sized framebuffer means the window is minimized; block here until it is restored.
	int width = 0, height = 0;
	glfwGetFramebufferSize(window, &width, &height);
	while (width == 0 || height == 0)
	{
		glfwGetFramebufferSize(window, &width, &height);
		glfwWaitEvents();
	}

	vulkan.logicalDevice.waitIdle();

	cleanupSwapChain();
	swapchain.init(vulkan, window);
	frames.recreate(vulkan, swapchain);
	// Compute and RT descriptor sets reference images that are recreated above
	// (storageImages and rayTracingOutputImages are extent-dependent), so both
	// must be rewritten after frames.recreate().
	createComputeDescriptorSets();
	createRayTracingDescriptorSets();
}

void EngineCore::createPhysicsDescriptorSets()
{
	vk::DescriptorPoolSize       poolSize{vk::DescriptorType::eStorageBuffer, 1};
	vk::DescriptorPoolCreateInfo poolInfo{
	    .flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
	    .maxSets       = 1,
	    .poolSizeCount = 1,
	    .pPoolSizes    = &poolSize};
	physicsDescriptorPool = vk::raii::DescriptorPool(vulkan.logicalDevice, poolInfo);

	vk::DescriptorSetAllocateInfo allocInfo{
	    .descriptorPool     = *physicsDescriptorPool,
	    .descriptorSetCount = 1,
	    .pSetLayouts        = &*pipelines.physicsDescriptorSetLayout};

	physicsDescriptorSet = std::move(vulkan.logicalDevice.allocateDescriptorSets(allocInfo)[0]);

	// Create SSBO (Max 10000 objects)
	size_t maxObjects = 10000;
	physicsSystem->createSSBO(vulkan.logicalDevice, vulkan.physicalDevice, maxObjects * sizeof(PhysicsObject));

	// Bind SSBO to Set
	vk::DescriptorBufferInfo bufferInfo{
	    .buffer = *physicsSystem->getSSBOBuffer(),
	    .offset = 0,
	    .range  = maxObjects * sizeof(PhysicsObject)};

	vk::WriteDescriptorSet writeDescriptorSet{
	    .dstSet          = *physicsDescriptorSet,
	    .dstBinding      = 0,
	    .dstArrayElement = 0,
	    .descriptorCount = 1,
	    .descriptorType  = vk::DescriptorType::eStorageBuffer,
	    .pBufferInfo     = &bufferInfo};

	vulkan.logicalDevice.updateDescriptorSets(writeDescriptorSet, nullptr);
}

void EngineCore::createComputeDescriptorSets()
{
	// One set per Frame In Flight (matching storage images)
	std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *pipelines.computeDescriptorSetLayout);

	vk::DescriptorSetAllocateInfo allocInfo{
	    .descriptorPool     = *descriptorPool,        // Use the same global pool
	    .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
	    .pSetLayouts        = layouts.data()};

	computeDescriptorSets.clear();
	computeDescriptorSets = vulkan.logicalDevice.allocateDescriptorSets(allocInfo);

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		vk::DescriptorImageInfo imageInfo{
		    .imageView   = *frames.storageImageViews[i],
		    .imageLayout = vk::ImageLayout::eGeneral        // Compute shader writes to General layout
		};

		vk::WriteDescriptorSet storageImageWrite{
		    .dstSet          = *computeDescriptorSets[i],
		    .dstBinding      = 0,
		    .dstArrayElement = 0,
		    .descriptorCount = 1,
		    .descriptorType  = vk::DescriptorType::eStorageImage,
		    .pImageInfo      = &imageInfo};

		vulkan.logicalDevice.updateDescriptorSets(storageImageWrite, {});
	}
}

void EngineCore::createRayTracingDescriptorSets()
{
	// One set per frame in flight: each set references that frame's TLAS and RT output image.
	std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *pipelines.rayTracingDescriptorSetLayout);

	// RT set bindings: 0 = TLAS, 1 = output image, 2 = vertex arrays, 3 = index arrays,
	// 4 = material arrays, 5 = texture array (variably-sized; up to 1000 descriptors per set).
	std::vector<uint32_t>                                variableDescCounts(MAX_FRAMES_IN_FLIGHT, 1000);
	vk::DescriptorSetVariableDescriptorCountAllocateInfo variableDescCountInfo{
	    .descriptorSetCount = MAX_FRAMES_IN_FLIGHT,
	    .pDescriptorCounts  = variableDescCounts.data()};

	vk::DescriptorSetAllocateInfo allocInfo{
	    .pNext              = &variableDescCountInfo,
	    .descriptorPool     = *descriptorPool,
	    .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
	    .pSetLayouts        = layouts.data()};

	rtDescriptorSets.clear();
	rtDescriptorSets = vulkan.logicalDevice.allocateDescriptorSets(allocInfo);

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		// Binding 0 — TLAS.
		// The TLAS write requires a WriteDescriptorSetAccelerationStructureKHR in pNext;
		// it cannot use pBufferInfo or pImageInfo like every other descriptor type.
		vk::WriteDescriptorSetAccelerationStructureKHR tlasInfo{
		    .accelerationStructureCount = 1,
		    .pAccelerationStructures    = &*frames.tlas[i]};

		vk::WriteDescriptorSet tlasWrite{
		    .pNext           = &tlasInfo,
		    .dstSet          = *rtDescriptorSets[i],
		    .dstBinding      = 0,
		    .dstArrayElement = 0,
		    .descriptorCount = 1,
		    .descriptorType  = vk::DescriptorType::eAccelerationStructureKHR};

		// Binding 1 — RT output image (written by the raygen shader in General layout).
		vk::DescriptorImageInfo rtOutputImageInfo{
		    .imageView   = *frames.rayTracingOutputImageViews[i],
		    .imageLayout = vk::ImageLayout::eGeneral};

		vk::WriteDescriptorSet rtOutputWrite{
		    .dstSet          = *rtDescriptorSets[i],
		    .dstBinding      = 1,
		    .dstArrayElement = 0,
		    .descriptorCount = 1,
		    .descriptorType  = vk::DescriptorType::eStorageImage,
		    .pImageInfo      = &rtOutputImageInfo};

		std::vector<vk::WriteDescriptorSet> descriptorWrites;
		descriptorWrites.push_back(tlasWrite);
		descriptorWrites.push_back(rtOutputWrite);

		// Now we extract ALL global vertices, indices, materials, and textures
		// across all Scene Nodes that have been uploaded into VRAM by ResourceManager
		std::vector<vk::DescriptorBufferInfo> vertexInfos;
		std::vector<vk::DescriptorBufferInfo> indexInfos;
		std::vector<vk::DescriptorBufferInfo> materialInfos;
		std::vector<vk::DescriptorImageInfo>  textureInfos;

		// Since our ResourceManager stores ModelResource objects linearly in ID...
		// In a production engine, this would be an iterative flat map or array
		int totalModels = 1000;        // We capped our bindless array at 1000 slots
		for (int modelId = 0; modelId < totalModels; ++modelId)
		{
			if (ModelResource *model = resourceManager->getModelResource(modelId))
			{
				// Writing a null VkBuffer into a descriptor is invalid even with ePartiallyBound.
				// A loaded model must always have all three buffers; assert to catch regressions.
				assert(*model->vertexBuffer   && "RT descriptor: model has no vertex buffer");
				assert(*model->indexBuffer    && "RT descriptor: model has no index buffer");
				assert(*model->materialBuffer && "RT descriptor: model has no material buffer");

				// 1. Accumulate Vertex Buffers
				vertexInfos.push_back({*model->vertexBuffer, 0, VK_WHOLE_SIZE});

				// 2. Accumulate Index Buffers
				indexInfos.push_back({*model->indexBuffer, 0, VK_WHOLE_SIZE});

				// 3. Accumulate Material Buffers
				materialInfos.push_back({*model->materialBuffer, 0, VK_WHOLE_SIZE});

				// 4. Accumulate Textures — pair each view with its own sampler.
				for (size_t texIdx = 0; texIdx < model->textureImageViews.size(); ++texIdx)
				{
					textureInfos.push_back({*model->textureSamplers[texIdx], *model->textureImageViews[texIdx], vk::ImageLayout::eShaderReadOnlyOptimal});
				}
			}
			else
			{
				break;        // Stop at the first empty ID
			}
		}

		if (!vertexInfos.empty())
		{
			descriptorWrites.push_back(vk::WriteDescriptorSet{
			    .dstSet          = *rtDescriptorSets[i],
			    .dstBinding      = 2,
			    .dstArrayElement = 0,
			    .descriptorCount = static_cast<uint32_t>(vertexInfos.size()),
			    .descriptorType  = vk::DescriptorType::eStorageBuffer,
			    .pBufferInfo     = vertexInfos.data()});
		}

		if (!indexInfos.empty())
		{
			descriptorWrites.push_back(vk::WriteDescriptorSet{
			    .dstSet          = *rtDescriptorSets[i],
			    .dstBinding      = 3,
			    .dstArrayElement = 0,
			    .descriptorCount = static_cast<uint32_t>(indexInfos.size()),
			    .descriptorType  = vk::DescriptorType::eStorageBuffer,
			    .pBufferInfo     = indexInfos.data()});
		}

		if (!materialInfos.empty())
		{
			descriptorWrites.push_back(vk::WriteDescriptorSet{
			    .dstSet          = *rtDescriptorSets[i],
			    .dstBinding      = 4,
			    .dstArrayElement = 0,
			    .descriptorCount = static_cast<uint32_t>(materialInfos.size()),
			    .descriptorType  = vk::DescriptorType::eStorageBuffer,
			    .pBufferInfo     = materialInfos.data()});
		}

		if (!textureInfos.empty())
		{
			descriptorWrites.push_back(vk::WriteDescriptorSet{
			    .dstSet          = *rtDescriptorSets[i],
			    .dstBinding      = 5,
			    .dstArrayElement = 0,
			    .descriptorCount = static_cast<uint32_t>(textureInfos.size()),
			    .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
			    .pImageInfo      = textureInfos.data()});
		}

		vulkan.logicalDevice.updateDescriptorSets(descriptorWrites, {});
	}
}

void EngineCore::recordComputeCommandBuffer(const vk::raii::CommandBuffer &commandBuffer, uint32_t imageIndex) const
{
	// 1. Transition Storage Image (frameIndex) to General layout for writing
	// Barrier waiting for the previous frame's Blit (Transfer Read) to finish before we start Compute Write
	transition_image_layout(
	    *frames.storageImages[frames.frameIndex],
	    vk::ImageLayout::eUndefined,        // Discard content but wait for execution
	    vk::ImageLayout::eGeneral,
	    {},        // srcAccess ignored for Undefined, but we rely on stage dependency
	    vk::AccessFlagBits2::eShaderWrite,
	    vk::PipelineStageFlagBits2::eTransfer,        // Wait for the previous frame's blit
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
	    .srcOffsets     = {{vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<int32_t>(swapchain.extent.width), static_cast<int32_t>(swapchain.extent.height), 1}}},
	    .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
	    .dstOffsets     = {{vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<int32_t>(swapchain.extent.width), static_cast<int32_t>(swapchain.extent.height), 1}}}};

	commandBuffer.blitImage(*frames.storageImages[frames.frameIndex], vk::ImageLayout::eTransferSrcOptimal,
	                        swapchain.images[imageIndex], vk::ImageLayout::eTransferDstOptimal,
	                        blitRegion, vk::Filter::eLinear);

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

void EngineCore::recordRayTracingCommandBuffer(const vk::raii::CommandBuffer &commandBuffer, uint32_t imageIndex) const
{
	// 1. Transition RT Output Image to General layout for writing
	transition_image_layout(
	    *frames.rayTracingOutputImages[frames.frameIndex],
	    vk::ImageLayout::eUndefined,
	    vk::ImageLayout::eGeneral,
	    {},
	    vk::AccessFlagBits2::eShaderWrite,
	    vk::PipelineStageFlagBits2::eTopOfPipe,
	    vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
	    vk::ImageAspectFlagBits::eColor);

	// 4. Bind Pipeline and Descriptor Sets
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *pipelines.rayTracingPipeline);

	// Set 0 = RT layout (TLAS, output image, vertex/index/material/texture arrays)
	// Set 1 = global layout (UBO, CSM shadow maps)
	commandBuffer.bindDescriptorSets(
	    vk::PipelineBindPoint::eRayTracingKHR,
	    *pipelines.rayTracingPipelineLayout,
	    0,
	    {*rtDescriptorSets[frames.frameIndex], *descriptorSets[frames.frameIndex]},
	    nullptr);

	// 5. Push Constants (ScenePushConstants)
	ScenePushConstants pushConstants{};
	pushConstants.modelMatrix = glm::mat4(1.0f);        // Default identity

	commandBuffer.pushConstants<ScenePushConstants>(
	    *pipelines.rayTracingPipelineLayout,
	    vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eMissKHR,
	    0,
	    pushConstants);

	// 5. Trace Rays Dispatch
	vk::StridedDeviceAddressRegionKHR callableRegion{};
	commandBuffer.traceRaysKHR(
	    pipelines.raygenRegion,
	    pipelines.missRegion,
	    pipelines.hitRegion,
	    callableRegion,
	    swapchain.extent.width,
	    swapchain.extent.height,
	    1);

	// 6. Transition RT Image for Transfer
	transition_image_layout(
	    *frames.rayTracingOutputImages[frames.frameIndex],
	    vk::ImageLayout::eGeneral,
	    vk::ImageLayout::eTransferSrcOptimal,
	    vk::AccessFlagBits2::eShaderWrite,
	    vk::AccessFlagBits2::eTransferRead,
	    vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
	    vk::PipelineStageFlagBits2::eTransfer,
	    vk::ImageAspectFlagBits::eColor);

	// 7. Transition SwapChain image for Transfer
	transition_image_layout(
	    swapchain.images[imageIndex],
	    vk::ImageLayout::eUndefined,
	    vk::ImageLayout::eTransferDstOptimal,
	    {},
	    vk::AccessFlagBits2::eTransferWrite,
	    vk::PipelineStageFlagBits2::eTopOfPipe,
	    vk::PipelineStageFlagBits2::eTransfer,
	    vk::ImageAspectFlagBits::eColor);

	// 8. Blit RT Output to Swapchain Image
	vk::ImageBlit blitRegion{
	    .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
	    .srcOffsets     = {{vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<int32_t>(swapchain.extent.width), static_cast<int32_t>(swapchain.extent.height), 1}}},
	    .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
	    .dstOffsets     = {{vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<int32_t>(swapchain.extent.width), static_cast<int32_t>(swapchain.extent.height), 1}}}};

	commandBuffer.blitImage(*frames.rayTracingOutputImages[frames.frameIndex], vk::ImageLayout::eTransferSrcOptimal,
	                        swapchain.images[imageIndex], vk::ImageLayout::eTransferDstOptimal,
	                        blitRegion, vk::Filter::eLinear);

	// 9. Transition SwapChain to Color Attachment for UI Rendering
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

void EngineCore::createDescriptorPool()
{
	// Generous pool sizes to accommodate an arbitrary number of loaded models.
	// eSampledImage / eSampler are separate because the shadow map binding uses them
	// as distinct descriptor types (binding 1 and 2 in the global layout).
	std::array<vk::DescriptorPoolSize, 7> poolSizes = {
	    vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 1000},
	    // 1000 per loaded model (material textures) + 2×1000 for the two RT descriptor sets.
	    vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 5000},
	    vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 1000},
	    vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1000},
	    // 1000 for materials + vertex and index buffers * MAX_FRAMES
	    vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 15000},
	    vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1000},
	    vk::DescriptorPoolSize{vk::DescriptorType::eAccelerationStructureKHR, MAX_FRAMES_IN_FLIGHT}};

	vk::DescriptorPoolCreateInfo poolInfo{
	    // eFreeDescriptorSet: allows individual sets to be freed (needed by ResourceManager).
	    // eUpdateAfterBind: required for bindless descriptor indexing (VK_EXT_descriptor_indexing).
	    .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet |
	             vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind,
	    .maxSets       = 1000 * MAX_FRAMES_IN_FLIGHT,
	    .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
	    .pPoolSizes    = poolSizes.data()};
	descriptorPool = vk::raii::DescriptorPool(vulkan.logicalDevice, poolInfo);
}

void EngineCore::createDescriptorSets()
{
	std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *pipelines.descriptorSetLayoutGlobal);

	vk::DescriptorSetAllocateInfo allocInfo{
	    .descriptorPool     = *descriptorPool,
	    .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
	    .pSetLayouts        = layouts.data()};

	descriptorSets.clear();
	descriptorSets = vulkan.logicalDevice.allocateDescriptorSets(allocInfo);

	// Global descriptor set layout (Set 0):
	//   binding 0 → UniformBufferObject  (view/proj/light/cascade matrices, camera pos)
	//   binding 1 → shadow depth array   (sampled, ShaderReadOnlyOptimal)
	//   binding 2 → shadow PCF sampler   (comparison sampler)
	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		vk::DescriptorBufferInfo bufferInfo{
		    .buffer = *frames.uniformBuffers[i],
		    .offset = 0,
		    .range  = sizeof(Laphria::UniformBufferObject)};

		vk::WriteDescriptorSet uboWrite{
		    .dstSet          = *descriptorSets[i],
		    .dstBinding      = 0,
		    .dstArrayElement = 0,
		    .descriptorCount = 1,
		    .descriptorType  = vk::DescriptorType::eUniformBuffer,
		    .pBufferInfo     = &bufferInfo};

		// The shadow array image starts in eUndefined; we use eShaderReadOnlyOptimal
		// as the declared layout here because the first frame's shadow pass will
		// transition it via eUndefined → eDepthAttachmentOptimal → eShaderReadOnlyOptimal
		// before the main pass samples it.
		vk::DescriptorImageInfo shadowImageInfo{
		    .imageView   = *frames.shadowArrayViews[i],
		    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};

		vk::WriteDescriptorSet shadowImageWrite{
		    .dstSet          = *descriptorSets[i],
		    .dstBinding      = 1,
		    .dstArrayElement = 0,
		    .descriptorCount = 1,
		    .descriptorType  = vk::DescriptorType::eSampledImage,
		    .pImageInfo      = &shadowImageInfo};

		vk::DescriptorImageInfo shadowSamplerInfo{
		    .sampler = *frames.shadowSampler};

		vk::WriteDescriptorSet shadowSamplerWrite{
		    .dstSet          = *descriptorSets[i],
		    .dstBinding      = 2,
		    .dstArrayElement = 0,
		    .descriptorCount = 1,
		    .descriptorType  = vk::DescriptorType::eSampler,
		    .pImageInfo      = &shadowSamplerInfo};

		std::array<vk::WriteDescriptorSet, 3> writes = {uboWrite, shadowImageWrite, shadowSamplerWrite};
		vulkan.logicalDevice.updateDescriptorSets(writes, {});
	}
}

void EngineCore::recordCommandBuffer(uint32_t imageIndex) const
{
	auto &commandBuffer = frames.commandBuffers[frames.frameIndex];

	vk::ClearValue clearColor = vk::ClearColorValue(0.02f, 0.02f, 0.02f, 1.0f);

	// --- Build TLAS ---
	std::vector<vk::AccelerationStructureInstanceKHR> tlasInstances;

	for (const auto &node : scene->getAllNodes())
	{
		if (node->modelId >= 0)
		{
			ModelResource *modelRes = resourceManager->getModelResource(node->modelId);
			if (!modelRes || modelRes->blasElements.empty())
				continue;

			glm::mat4 transform = node->getWorldTransform();

			// Convert to vk::TransformMatrixKHR (3x4 row-major array)
			vk::TransformMatrixKHR transformMatrix;
			for (int r = 0; r < 3; ++r)
			{
				for (int c = 0; c < 4; ++c)
				{
					transformMatrix.matrix[r][c] = transform[c][r];        // GLM is column-major
				}
			}

			for (int meshIdx : node->getMeshIndices())
			{
				if (meshIdx < 0 || meshIdx >= modelRes->blasElements.size())
					continue;

				auto &blas = modelRes->blasElements[meshIdx];

				uint32_t primitiveOffset = 0;
				for (int i = 0; i < meshIdx; ++i)
				{
					primitiveOffset += modelRes->meshes[i].primitives.size();
				}

				// Encode modelId in top 10 bits, primitiveOffset in bottom 14 bits
				// InstanceCustomIndex is exactly 24 bit in size.
				assert(node->modelId < 1024 && "modelId exceeds 10-bit limit; customIndex encoding will be corrupted");
				uint32_t customIndex = (node->modelId << 14) | (primitiveOffset & 0x3FFF);

				vk::AccelerationStructureDeviceAddressInfoKHR addressInfo{};
				addressInfo.accelerationStructure = *blas;
				vk::DeviceAddress blasAddress     = vulkan.logicalDevice.getAccelerationStructureAddressKHR(addressInfo);

				vk::AccelerationStructureInstanceKHR instance{};
				instance.transform                              = transformMatrix;
				instance.instanceCustomIndex                    = customIndex;
				instance.mask                                   = 0xFF;        // All rays hit
				instance.instanceShaderBindingTableRecordOffset = 0;
				instance.flags                                  = static_cast<uint32_t>(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);
				instance.accelerationStructureReference         = blasAddress;

				tlasInstances.push_back(instance);
			}
		}
	}

	if (ui.useRayTracing)
	{
		// Only copy instance data when there is something to copy; building with
		// primitiveCount = 0 is valid and produces a traversable empty TLAS.
		if (!tlasInstances.empty())
		{
			size_t dataSize = tlasInstances.size() * sizeof(vk::AccelerationStructureInstanceKHR);
			memcpy(frames.tlasInstanceBuffersMapped[frames.frameIndex], tlasInstances.data(), dataSize);
		}

		// Memory barrier to ensure host writes to the instance buffer are visible to the AS builder
		vk::MemoryBarrier2 hostToDeviceBarrier{
		    .srcStageMask  = vk::PipelineStageFlagBits2::eHost,
		    .srcAccessMask = vk::AccessFlagBits2::eHostWrite,
		    .dstStageMask  = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
		    .dstAccessMask = vk::AccessFlagBits2::eAccelerationStructureReadKHR};

		vk::DependencyInfo dependencyInfo{
		    .memoryBarrierCount = 1,
		    .pMemoryBarriers    = &hostToDeviceBarrier};
		commandBuffer.pipelineBarrier2(dependencyInfo);

		// Build TLAS — always, even when the scene is empty (primitiveCount = 0 is valid).
		vk::AccelerationStructureGeometryInstancesDataKHR instancesData{};
		instancesData.arrayOfPointers    = vk::False;
		instancesData.data.deviceAddress = frames.tlasInstanceAddresses[frames.frameIndex];

		vk::AccelerationStructureGeometryKHR tlasGeometry{};
		tlasGeometry.geometryType       = vk::GeometryTypeKHR::eInstances;
		tlasGeometry.geometry.instances = instancesData;

		vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{};
		buildInfo.type                      = vk::AccelerationStructureTypeKHR::eTopLevel;
		buildInfo.flags                     = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
		buildInfo.mode                      = vk::BuildAccelerationStructureModeKHR::eBuild;
		buildInfo.geometryCount             = 1;
		buildInfo.pGeometries               = &tlasGeometry;
		buildInfo.dstAccelerationStructure  = *frames.tlas[frames.frameIndex];
		buildInfo.scratchData.deviceAddress = frames.tlasScratchAddresses[frames.frameIndex];

		vk::AccelerationStructureBuildRangeInfoKHR buildRange{};
		buildRange.primitiveCount  = static_cast<uint32_t>(tlasInstances.size());
		buildRange.primitiveOffset = 0;
		buildRange.firstVertex     = 0;
		buildRange.transformOffset = 0;

		const vk::AccelerationStructureBuildRangeInfoKHR *pBuildRange = &buildRange;
		commandBuffer.buildAccelerationStructuresKHR(buildInfo, pBuildRange);

		// Memory barrier to ensure TLAS build finishes before the ray tracing shader reads it
		vk::MemoryBarrier2 asBuildToRayTracingBarrier{
		    .srcStageMask  = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
		    .srcAccessMask = vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
		    .dstStageMask  = vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
		    .dstAccessMask = vk::AccessFlagBits2::eAccelerationStructureReadKHR};

		vk::DependencyInfo asDependencyInfo{
		    .memoryBarrierCount = 1,
		    .pMemoryBarriers    = &asBuildToRayTracingBarrier};
		commandBuffer.pipelineBarrier2(asDependencyInfo);
	}
	// --- End TLAS Build ---

	// ── Cascaded Shadow Map Pass ──────────────────────────────────────────────
	// Only run for the raster path; the RT pipeline handles its own shadowing.
	if (!ui.useRayTracing)
	{
		vk::Image shadowImg = *frames.shadowImages[frames.frameIndex];

		// Transition all 4 cascade layers: eUndefined → eDepthAttachmentOptimal.
		// We always use eUndefined as the old layout so the previous frame's contents
		// are discarded — the depth buffer is cleared at the start of each cascade render.
		vk::ImageMemoryBarrier2 shadowToWrite{
		    .srcStageMask        = vk::PipelineStageFlagBits2::eTopOfPipe,
		    .srcAccessMask       = {},
		    .dstStageMask        = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
		    .dstAccessMask       = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		    .oldLayout           = vk::ImageLayout::eUndefined,
		    .newLayout           = vk::ImageLayout::eDepthAttachmentOptimal,
		    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .image               = shadowImg,
		    .subresourceRange    = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, NUM_SHADOW_CASCADES}};
		vk::DependencyInfo shadowWriteDep{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &shadowToWrite};
		commandBuffer.pipelineBarrier2(shadowWriteDep);

		// Render each cascade into its own layer of the shadow array image.
		commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines.shadowPipeline);
		commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
		                                 *pipelines.shadowPipelineLayout, 0,
		                                 *descriptorSets[frames.frameIndex], nullptr);

		vk::Viewport shadowViewport{0.0f, 0.0f,
		                            static_cast<float>(SHADOW_MAP_DIM), static_cast<float>(SHADOW_MAP_DIM),
		                            0.0f, 1.0f};
		vk::Rect2D   shadowScissor{{0, 0}, {SHADOW_MAP_DIM, SHADOW_MAP_DIM}};

		for (uint32_t cascadeIdx = 0; cascadeIdx < NUM_SHADOW_CASCADES; cascadeIdx++)
		{
			uint32_t viewIdx = frames.frameIndex * NUM_SHADOW_CASCADES + cascadeIdx;

			vk::RenderingAttachmentInfo cascadeDepthAttachment{
			    .imageView   = *frames.shadowCascadeViews[viewIdx],
			    .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
			    .loadOp      = vk::AttachmentLoadOp::eClear,
			    .storeOp     = vk::AttachmentStoreOp::eStore,
			    .clearValue  = vk::ClearDepthStencilValue{1.0f, 0}};

			vk::RenderingInfo cascadeRenderingInfo{
			    .renderArea           = {{0, 0}, {SHADOW_MAP_DIM, SHADOW_MAP_DIM}},
			    .layerCount           = 1,
			    .colorAttachmentCount = 0,
			    .pDepthAttachment     = &cascadeDepthAttachment};

			commandBuffer.beginRendering(cascadeRenderingInfo);
			commandBuffer.setViewport(0, shadowViewport);
			commandBuffer.setScissor(0, shadowScissor);

			// Draw all scene nodes into this cascade.
			for (const auto &node : scene->getAllNodes())
			{
				if (node->modelId < 0)
					continue;
				auto *modelRes = resourceManager->getModelResource(node->modelId);
				if (!modelRes)
					continue;

				resourceManager->bindResources(commandBuffer, node->modelId);
				glm::mat4 worldTransform = node->getWorldTransform();

				if (*modelRes->descriptorSet)
				{
					commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelines.shadowPipelineLayout, 1, {*modelRes->descriptorSet}, nullptr);
				}

				for (int meshIdx : node->getMeshIndices())
				{
					if (meshIdx < 0 || meshIdx >= static_cast<int>(modelRes->meshes.size()))
						continue;
					for (const auto &prim : modelRes->meshes[meshIdx].primitives)
					{
						Laphria::ScenePushConstants pc{};
						pc.modelMatrix   = worldTransform;
						pc.cascadeIndex  = static_cast<int>(cascadeIdx);
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
		    .srcStageMask        = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
		    .srcAccessMask       = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		    .dstStageMask        = vk::PipelineStageFlagBits2::eFragmentShader,
		    .dstAccessMask       = vk::AccessFlagBits2::eShaderRead,
		    .oldLayout           = vk::ImageLayout::eDepthAttachmentOptimal,
		    .newLayout           = vk::ImageLayout::eShaderReadOnlyOptimal,
		    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .image               = shadowImg,
		    .subresourceRange    = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, NUM_SHADOW_CASCADES}};
		vk::DependencyInfo shadowReadDep{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &shadowToRead};
		commandBuffer.pipelineBarrier2(shadowReadDep);

		// Starfield compute pass — dispatches the compute shader which writes the starfield
		// to the storage image, then blits it to the swapchain image (transitioning it to
		// eColorAttachmentOptimal) so the main pass can render geometry on top.
		recordComputeCommandBuffer(commandBuffer, imageIndex);
	}

	if (ui.useRayTracing)
	{
		// Run Ray Tracing Pipeline
		recordRayTracingCommandBuffer(commandBuffer, imageIndex);
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
	    .imageView   = *swapchain.imageViews[imageIndex],
	    .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
	    .loadOp      = vk::AttachmentLoadOp::eLoad,
	    .storeOp     = vk::AttachmentStoreOp::eStore,
	    .clearValue  = clearColor};

	vk::RenderingAttachmentInfo depthAttachmentInfo{
	    .imageView   = *frames.depthImageViews[imageIndex],
	    .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
	    .loadOp      = vk::AttachmentLoadOp::eClear,
	    .storeOp     = vk::AttachmentStoreOp::eStore,
	    .clearValue  = vk::ClearDepthStencilValue{1.0f, 0}};

	vk::RenderingInfo renderingInfo = {
	    .renderArea           = {.offset = {0, 0}, .extent = swapchain.extent},
	    .layerCount           = 1,
	    .colorAttachmentCount = 1,
	    .pColorAttachments    = &attachmentInfo,
	    .pDepthAttachment     = &depthAttachmentInfo};

	commandBuffer.beginRendering(renderingInfo);

	if (!ui.useRayTracing)
	{
		commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines.graphicsPipeline);

		// Y starts at height and height is negative: this flips the Vulkan NDC Y-axis so that
		// +Y points up in clip space, matching GLM's convention (which was designed for OpenGL).
		vk::Viewport viewport{
		    0.0f, static_cast<float>(swapchain.extent.height),
		    static_cast<float>(swapchain.extent.width),
		    -static_cast<float>(swapchain.extent.height), 0.0f, 1.0f};
		commandBuffer.setViewport(0, viewport);
		commandBuffer.setScissor(0, vk::Rect2D({0, 0}, swapchain.extent));

		// Global UBO Binding (Set 0)
		commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelines.graphicsPipelineLayout, 0,
		                                 *descriptorSets[frames.frameIndex], nullptr);

		// Define culling bounds based on camera position and view distance
		// This is a simple "box cull" around the camera for now
		glm::vec3 camPos       = camera.position;
		float     viewDistance = 2000.0f;
		AABB      cullBounds   = {
            camPos - glm::vec3(viewDistance),
            camPos + glm::vec3(viewDistance)};
		scene->draw(commandBuffer, pipelines.graphicsPipelineLayout, *resourceManager, cullBounds);
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
    vk::Image               image,
    vk::ImageLayout         old_layout,
    vk::ImageLayout         new_layout,
    vk::AccessFlags2        src_access_mask,
    vk::AccessFlags2        dst_access_mask,
    vk::PipelineStageFlags2 src_stage_mask,
    vk::PipelineStageFlags2 dst_stage_mask,
    vk::ImageAspectFlags    image_aspect_flags) const
{
	vk::ImageMemoryBarrier2 barrier = {
	    .srcStageMask        = src_stage_mask,
	    .srcAccessMask       = src_access_mask,
	    .dstStageMask        = dst_stage_mask,
	    .dstAccessMask       = dst_access_mask,
	    .oldLayout           = old_layout,
	    .newLayout           = new_layout,
	    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .image               = image,
	    .subresourceRange    = {
	           .aspectMask     = image_aspect_flags,
	           .baseMipLevel   = 0,
	           .levelCount     = 1,
	           .baseArrayLayer = 0,
	           .layerCount     = 1}};
	vk::DependencyInfo dependency_info = {
	    .dependencyFlags         = {},
	    .imageMemoryBarrierCount = 1,
	    .pImageMemoryBarriers    = &barrier};
	frames.commandBuffers[frames.frameIndex].pipelineBarrier2(dependency_info);
}

void EngineCore::drawFrame()
{
	// Note: inFlightFences, presentCompleteSemaphores, and commandBuffers are indexed by frameIndex,
	//       while renderFinishedSemaphores is indexed by imageIndex
	auto fenceResult = vulkan.logicalDevice.waitForFences(*frames.inFlightFences[frames.frameIndex], vk::True, UINT64_MAX);
	if (fenceResult != vk::Result::eSuccess)
	{
		throw std::runtime_error("failed to wait for fence!");
	}

	auto [result, imageIndex] = swapchain.swapChain.acquireNextImage(
	    UINT64_MAX, *frames.presentCompleteSemaphores[frames.frameIndex], nullptr);

	if (result == vk::Result::eErrorOutOfDateKHR)
	{
		recreateSwapChain();
		return;
	}
	if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
	{
		assert(result == vk::Result::eTimeout || result == vk::Result::eNotReady);
		throw std::runtime_error("failed to acquire swap chain image!");
	}
	frames.updateUniformBuffer(frames.frameIndex, camera, swapchain.extent, ui.lightDirection);

	// Only reset the fence if we are submitting work
	vulkan.logicalDevice.resetFences(*frames.inFlightFences[frames.frameIndex]);

	frames.commandBuffers[frames.frameIndex].reset();
	vk::raii::CommandBuffer &commandBuffer = frames.commandBuffers[frames.frameIndex];
	commandBuffer.begin(vk::CommandBufferBeginInfo{});

	// 2. Main Pass
	recordCommandBuffer(imageIndex);

	// The swapchain image is accessed at eColorAttachmentOutput (main/ImGui pass) and at
	// eTransfer (blit in compute and RT paths). Both stages must wait for vkAcquireNextImage.
	vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput |
	                                                vk::PipelineStageFlagBits::eTransfer);
	const vk::SubmitInfo   submitInfo{
	      .waitSemaphoreCount   = 1,
	      .pWaitSemaphores      = &*frames.presentCompleteSemaphores[frames.frameIndex],
	      .pWaitDstStageMask    = &waitDestinationStageMask,
	      .commandBufferCount   = 1,
	      .pCommandBuffers      = &*frames.commandBuffers[frames.frameIndex],
	      .signalSemaphoreCount = 1,
	      .pSignalSemaphores    = &*frames.renderFinishedSemaphores[imageIndex]};

	commandBuffer.end();

	vulkan.queue.submit(submitInfo, *frames.inFlightFences[frames.frameIndex]);

	const vk::PresentInfoKHR presentInfoKHR{
	    .waitSemaphoreCount = 1,
	    .pWaitSemaphores    = &*frames.renderFinishedSemaphores[imageIndex],
	    .swapchainCount     = 1,
	    .pSwapchains        = &*swapchain.swapChain,
	    .pImageIndices      = &imageIndex};

	// VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS is defined so presentKHR should return
	// eErrorOutOfDateKHR as a value rather than throw, but behaviour is inconsistent across
	// loader/driver versions. The try/catch ensures resize detection is never silently lost.
	try
	{
		result = vulkan.queue.presentKHR(presentInfoKHR);
	}
	catch (vk::OutOfDateKHRError &)
	{
		result = vk::Result::eErrorOutOfDateKHR;
	}
	catch (vk::SurfaceLostKHRError &)
	{
		result = vk::Result::eErrorOutOfDateKHR;
	}

	if ((result == vk::Result::eSuboptimalKHR) || (result == vk::Result::eErrorOutOfDateKHR) ||
	    swapchain.framebufferResized)
	{
		swapchain.framebufferResized = false;
		recreateSwapChain();
	}
	else
	{
		assert(result == vk::Result::eSuccess);
	}
	frames.frameIndex = (frames.frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
}