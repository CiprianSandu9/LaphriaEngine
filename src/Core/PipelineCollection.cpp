#include "PipelineCollection.h"

#include <array>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "EngineAuxiliary.h"
#include "VulkanUtils.h"

using namespace Laphria;

// ── Top-level init ─────────────────────────────────────────────────────────

void PipelineCollection::createDescriptorSetLayouts(VulkanDevice &dev)
{
	createGlobalDescriptorSetLayout(dev);
	createMaterialDescriptorSetLayout(dev);
	createComputeDescriptorSetLayout(dev);
	createRayTracingDescriptorSetLayout(dev);
	createPhysicsDescriptorSetLayout(dev);
}

// ── Descriptor Set Layout Implementations ──────────────────────────────────

void PipelineCollection::createGlobalDescriptorSetLayout(VulkanDevice &dev)
{
	// Binding 0 — UBO (view/proj/light/cascade matrices), used by all pipeline stages.
	// Binding 1 — CSM shadow depth array (sampled image). ePartiallyBound so RT/compute
	//             pipelines that bind this set without providing binding 1 are still valid.
	// Binding 2 — CSM comparison sampler. Same ePartiallyBound rationale.
	std::array<vk::DescriptorSetLayoutBinding, 3> globalBindings = {
	    vk::DescriptorSetLayoutBinding{
	        .binding         = 0,
	        .descriptorType  = vk::DescriptorType::eUniformBuffer,
	        .descriptorCount = 1,
	        .stageFlags      = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment |
	                      vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR |
	                      vk::ShaderStageFlagBits::eMissKHR},
	    vk::DescriptorSetLayoutBinding{
	        .binding         = 1,
	        .descriptorType  = vk::DescriptorType::eSampledImage,
	        .descriptorCount = 1,
	        .stageFlags      = vk::ShaderStageFlagBits::eFragment},
	    vk::DescriptorSetLayoutBinding{
	        .binding         = 2,
	        .descriptorType  = vk::DescriptorType::eSampler,
	        .descriptorCount = 1,
	        .stageFlags      = vk::ShaderStageFlagBits::eFragment}};

	std::array<vk::DescriptorBindingFlags, 3> bindFlags = {
	    vk::DescriptorBindingFlags{},                           // binding 0 — always provided
	    vk::DescriptorBindingFlagBits::ePartiallyBound,         // binding 1 — optional for RT/compute
	    vk::DescriptorBindingFlagBits::ePartiallyBound};        // binding 2 — optional for RT/compute

	vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{
	    .bindingCount  = static_cast<uint32_t>(bindFlags.size()),
	    .pBindingFlags = bindFlags.data()};

	vk::DescriptorSetLayoutCreateInfo layoutInfoGlobal{
	    .pNext        = &bindingFlagsInfo,
	    .bindingCount = static_cast<uint32_t>(globalBindings.size()),
	    .pBindings    = globalBindings.data()};
	descriptorSetLayoutGlobal = vk::raii::DescriptorSetLayout(dev.logicalDevice, layoutInfoGlobal);
}

void PipelineCollection::createMaterialDescriptorSetLayout(VulkanDevice &dev)
{
	std::array<vk::DescriptorSetLayoutBinding, 2> matBindings = {
	    vk::DescriptorSetLayoutBinding{
	        .binding         = 0,
	        .descriptorType  = vk::DescriptorType::eStorageBuffer,
	        .descriptorCount = 1,
	        .stageFlags      = vk::ShaderStageFlagBits::eFragment},
	    vk::DescriptorSetLayoutBinding{
	        .binding         = 1,
	        .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
	        .descriptorCount = 1000,
	        .stageFlags      = vk::ShaderStageFlagBits::eFragment}};
	std::array<vk::DescriptorBindingFlags, 2> flags = {
	    vk::DescriptorBindingFlags{},
	    vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eVariableDescriptorCount |
	        vk::DescriptorBindingFlagBits::eUpdateAfterBind};
	vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlags{
	    .bindingCount  = static_cast<uint32_t>(flags.size()),
	    .pBindingFlags = flags.data()};
	vk::DescriptorSetLayoutCreateInfo layoutInfoMat{
	    .pNext        = &bindingFlags,
	    .flags        = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool,
	    .bindingCount = static_cast<uint32_t>(matBindings.size()),
	    .pBindings    = matBindings.data()};
	descriptorSetLayoutMaterial = vk::raii::DescriptorSetLayout(dev.logicalDevice, layoutInfoMat);
}

void PipelineCollection::createComputeDescriptorSetLayout(VulkanDevice &dev)
{
	vk::DescriptorSetLayoutBinding storageImageBinding{
	    .binding         = 0,
	    .descriptorType  = vk::DescriptorType::eStorageImage,
	    .descriptorCount = 1,
	    .stageFlags      = vk::ShaderStageFlagBits::eCompute};
	vk::DescriptorSetLayoutCreateInfo layoutInfoCompute{
	    .bindingCount = 1,
	    .pBindings    = &storageImageBinding};
	computeDescriptorSetLayout = vk::raii::DescriptorSetLayout(dev.logicalDevice, layoutInfoCompute);
}

void PipelineCollection::createPhysicsDescriptorSetLayout(VulkanDevice &dev)
{
	vk::DescriptorSetLayoutBinding ssboBinding{
	    .binding         = 0,
	    .descriptorType  = vk::DescriptorType::eStorageBuffer,
	    .descriptorCount = 1,
	    .stageFlags      = vk::ShaderStageFlagBits::eCompute};
	vk::DescriptorSetLayoutCreateInfo layoutInfo{
	    .bindingCount = 1,
	    .pBindings    = &ssboBinding};
	physicsDescriptorSetLayout = vk::raii::DescriptorSetLayout(dev.logicalDevice, layoutInfo);
}

void PipelineCollection::createRayTracingDescriptorSetLayout(VulkanDevice &dev)
{
	std::array<vk::DescriptorSetLayoutBinding, 6> bindings = {
	    vk::DescriptorSetLayoutBinding{// TLAS
	                                   .binding         = 0,
	                                   .descriptorType  = vk::DescriptorType::eAccelerationStructureKHR,
	                                   .descriptorCount = 1,
	                                   .stageFlags      = vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR},
	    vk::DescriptorSetLayoutBinding{// Output Image
	                                   .binding         = 1,
	                                   .descriptorType  = vk::DescriptorType::eStorageImage,
	                                   .descriptorCount = 1,
	                                   .stageFlags      = vk::ShaderStageFlagBits::eRaygenKHR},
	    vk::DescriptorSetLayoutBinding{// Vertex Buffers Array
	                                   .binding         = 2,
	                                   .descriptorType  = vk::DescriptorType::eStorageBuffer,
	                                   .descriptorCount = 1000,
	                                   .stageFlags      = vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eAnyHitKHR},
	    vk::DescriptorSetLayoutBinding{// Index Buffers Array
	                                   .binding         = 3,
	                                   .descriptorType  = vk::DescriptorType::eStorageBuffer,
	                                   .descriptorCount = 1000,
	                                   .stageFlags      = vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eAnyHitKHR},
	    vk::DescriptorSetLayoutBinding{// Material Buffers Array
	                                   .binding         = 4,
	                                   .descriptorType  = vk::DescriptorType::eStorageBuffer,
	                                   .descriptorCount = 1000,
	                                   .stageFlags      = vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eAnyHitKHR},
	    vk::DescriptorSetLayoutBinding{// Textures Array
	                                   .binding         = 5,
	                                   .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
	                                   .descriptorCount = 1000,
	                                   .stageFlags      = vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eAnyHitKHR}};
	std::array<vk::DescriptorBindingFlags, 6> flags = {
	    vk::DescriptorBindingFlags{},
	    vk::DescriptorBindingFlags{},
	    vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind,
	    vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind,
	    vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind,
	    vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eVariableDescriptorCount | vk::DescriptorBindingFlagBits::eUpdateAfterBind};
	vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlags{
	    .bindingCount  = static_cast<uint32_t>(flags.size()),
	    .pBindingFlags = flags.data()};
	vk::DescriptorSetLayoutCreateInfo layoutInfo{
	    .pNext        = &bindingFlags,
	    .flags        = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool,
	    .bindingCount = static_cast<uint32_t>(bindings.size()),
	    .pBindings    = bindings.data()};
	rayTracingDescriptorSetLayout = vk::raii::DescriptorSetLayout(dev.logicalDevice, layoutInfo);
}

// ── Pipeline Layout Implementations ────────────────────────────────────────

void PipelineCollection::createShadowPipelineLayout(VulkanDevice &dev)
{
	// The shadow pass only needs the global UBO (set 0) for cascade view-proj matrices.
	// Push constants carry modelMatrix (offset 0) and cascadeIndex (offset 68).
	vk::PushConstantRange pushConstantRange{
	    .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
	    .offset     = 0,
	    .size       = sizeof(ScenePushConstants)};
	std::array                   layouts = {*descriptorSetLayoutGlobal, *descriptorSetLayoutMaterial};
	vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
	    .setLayoutCount         = static_cast<uint32_t>(layouts.size()),
	    .pSetLayouts            = layouts.data(),
	    .pushConstantRangeCount = 1,
	    .pPushConstantRanges    = &pushConstantRange};
	shadowPipelineLayout = vk::raii::PipelineLayout(dev.logicalDevice, pipelineLayoutInfo);
}

void PipelineCollection::createGraphicsPipelineLayout(VulkanDevice &dev)
{
	vk::PushConstantRange pushConstantRange{
	    .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
	    .offset     = 0,
	    .size       = sizeof(ScenePushConstants)};
	std::array                   layouts = {*descriptorSetLayoutGlobal, *descriptorSetLayoutMaterial};
	vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
	    .setLayoutCount         = static_cast<uint32_t>(layouts.size()),
	    .pSetLayouts            = layouts.data(),
	    .pushConstantRangeCount = 1,
	    .pPushConstantRanges    = &pushConstantRange};
	graphicsPipelineLayout = vk::raii::PipelineLayout(dev.logicalDevice, pipelineLayoutInfo);
}

void PipelineCollection::createComputePipelineLayout(VulkanDevice &dev)
{
	// The starfield compute shader only uses Set 0 (storage image) and push constants.
	// Keeping the layout minimal avoids requiring the global UBO and material sets to be
	// bound during the compute dispatch.
	vk::PushConstantRange pushConstantRange{
	    .stageFlags = vk::ShaderStageFlagBits::eCompute,
	    .offset     = 0,
	    .size       = sizeof(ScenePushConstants)};
	vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
	    .setLayoutCount         = 1,
	    .pSetLayouts            = &*computeDescriptorSetLayout,
	    .pushConstantRangeCount = 1,
	    .pPushConstantRanges    = &pushConstantRange};
	computePipelineLayout = vk::raii::PipelineLayout(dev.logicalDevice, pipelineLayoutInfo);
}

void PipelineCollection::createPhysicsPipelineLayout(VulkanDevice &dev)
{
	vk::PushConstantRange pushConstantRange{
	    .stageFlags = vk::ShaderStageFlagBits::eCompute,
	    .offset     = 0,
	    .size       = 128};
	vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
	    .setLayoutCount         = 1,
	    .pSetLayouts            = &*physicsDescriptorSetLayout,
	    .pushConstantRangeCount = 1,
	    .pPushConstantRanges    = &pushConstantRange};
	physicsPipelineLayout = vk::raii::PipelineLayout(dev.logicalDevice, pipelineLayoutInfo);
}

void PipelineCollection::createRayTracingPipelineLayout(VulkanDevice &dev)
{
	std::array            layouts = {*rayTracingDescriptorSetLayout, *descriptorSetLayoutGlobal};
	vk::PushConstantRange pushConstantRange{
	    .stageFlags = vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eMissKHR,
	    .offset     = 0,
	    .size       = sizeof(ScenePushConstants)};
	vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
	    .setLayoutCount         = static_cast<uint32_t>(layouts.size()),
	    .pSetLayouts            = layouts.data(),
	    .pushConstantRangeCount = 1,
	    .pPushConstantRanges    = &pushConstantRange};
	rayTracingPipelineLayout = vk::raii::PipelineLayout(dev.logicalDevice, pipelineLayoutInfo);
}

// ── Pipeline Implementations ───────────────────────────────────────────────

void PipelineCollection::createGraphicsPipeline(VulkanDevice &dev, vk::Format colorFormat, vk::Format depthFormat)
{
	vk::raii::ShaderModule shaderModule = createShaderModule(dev, readFile("Shaders/LaphriaEngine.slang.spv"));

	vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
	    .stage = vk::ShaderStageFlagBits::eVertex, .module = *shaderModule, .pName = "vertMain"};
	vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
	    .stage = vk::ShaderStageFlagBits::eFragment, .module = *shaderModule, .pName = "fragMain"};
	vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

	auto                                   bindingDescription    = Vertex::getBindingDescription();
	auto                                   attributeDescriptions = Vertex::getAttributeDescriptions();
	vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
	    .vertexBindingDescriptionCount   = 1,
	    .pVertexBindingDescriptions      = &bindingDescription,
	    .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
	    .pVertexAttributeDescriptions    = attributeDescriptions.data()};
	vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
	    .topology               = vk::PrimitiveTopology::eTriangleList,
	    .primitiveRestartEnable = vk::False};
	vk::PipelineViewportStateCreateInfo viewportState{
	    .viewportCount = 1,
	    .scissorCount  = 1};
	vk::PipelineRasterizationStateCreateInfo rasterizer{
	    .depthClampEnable        = vk::False,
	    .rasterizerDiscardEnable = vk::False,
	    .cullMode                = vk::CullModeFlagBits::eBack,
	    .frontFace               = vk::FrontFace::eCounterClockwise,
	    .depthBiasEnable         = vk::False,
	    .lineWidth               = 1.0f};
	vk::PipelineMultisampleStateCreateInfo multisampling{
	    .rasterizationSamples = vk::SampleCountFlagBits::e1,
	    .sampleShadingEnable  = vk::False};
	vk::PipelineDepthStencilStateCreateInfo depthStencil{
	    .depthTestEnable       = vk::True,
	    .depthWriteEnable      = vk::True,
	    .depthCompareOp        = vk::CompareOp::eLess,
	    .depthBoundsTestEnable = vk::False,
	    .stencilTestEnable     = vk::False};
	vk::PipelineColorBlendAttachmentState colorBlendAttachment{
	    .blendEnable    = vk::False,
	    .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
	                      vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
	vk::PipelineColorBlendStateCreateInfo colorBlending{
	    .logicOpEnable   = vk::False,
	    .logicOp         = vk::LogicOp::eCopy,
	    .attachmentCount = 1,
	    .pAttachments    = &colorBlendAttachment};
	std::vector                        dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
	vk::PipelineDynamicStateCreateInfo dynamicState{
	    .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
	    .pDynamicStates    = dynamicStates.data()};

	createGraphicsPipelineLayout(dev);

	vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{
	    .colorAttachmentCount    = 1,
	    .pColorAttachmentFormats = &colorFormat,
	    .depthAttachmentFormat   = depthFormat};
	vk::GraphicsPipelineCreateInfo pipelineInfo{
	    .pNext               = &pipelineRenderingCreateInfo,
	    .stageCount          = 2,
	    .pStages             = shaderStages,
	    .pVertexInputState   = &vertexInputInfo,
	    .pInputAssemblyState = &inputAssembly,
	    .pViewportState      = &viewportState,
	    .pRasterizationState = &rasterizer,
	    .pMultisampleState   = &multisampling,
	    .pDepthStencilState  = &depthStencil,
	    .pColorBlendState    = &colorBlending,
	    .pDynamicState       = &dynamicState,
	    .layout              = *graphicsPipelineLayout,
	    .renderPass          = nullptr};
	graphicsPipeline = vk::raii::Pipeline(dev.logicalDevice, nullptr, pipelineInfo);
}

void PipelineCollection::createShadowPipeline(VulkanDevice &dev)
{
	createShadowPipelineLayout(dev);

	vk::raii::ShaderModule shadowModule = createShaderModule(dev, readFile("Shaders/Shadow.slang.spv"));

	vk::PipelineShaderStageCreateInfo vertStage{
	    .stage  = vk::ShaderStageFlagBits::eVertex,
	    .module = *shadowModule,
	    .pName  = "shadowVert"};
	vk::PipelineShaderStageCreateInfo fragStage{
	    .stage  = vk::ShaderStageFlagBits::eFragment,
	    .module = *shadowModule,
	    .pName  = "shadowFrag"};
	std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = {vertStage, fragStage};

	// Use the same vertex input layout as the main pipeline so the same VBOs are compatible.
	auto                                   bindingDesc = Vertex::getBindingDescription();
	auto                                   attribDescs = Vertex::getAttributeDescriptions();
	vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
	    .vertexBindingDescriptionCount   = 1,
	    .pVertexBindingDescriptions      = &bindingDesc,
	    .vertexAttributeDescriptionCount = static_cast<uint32_t>(attribDescs.size()),
	    .pVertexAttributeDescriptions    = attribDescs.data()};

	vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
	    .topology               = vk::PrimitiveTopology::eTriangleList,
	    .primitiveRestartEnable = vk::False};

	vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};

	// Hardware depth bias is disabled; shadow acne is handled instead by a normal-offset bias
	// applied in the fragment shader (biasedPos = worldPos + N * bias * cascadeScale).
	// depthClampEnable prevents shadow casters outside the frustum from being clipped.
	vk::PipelineRasterizationStateCreateInfo rasterizer{
	    .depthClampEnable        = vk::True,        // clamp rather than clip geometry at the frustum planes
	    .rasterizerDiscardEnable = vk::False,
	    .cullMode                = vk::CullModeFlagBits::eNone,
	    .frontFace               = vk::FrontFace::eClockwise,
	    .depthBiasEnable         = vk::False,
	    .depthBiasConstantFactor = 0.0f,
	    .depthBiasSlopeFactor    = 0.0f,
	    .lineWidth               = 1.0f};

	vk::PipelineMultisampleStateCreateInfo multisampling{
	    .rasterizationSamples = vk::SampleCountFlagBits::e1,
	    .sampleShadingEnable  = vk::False};

	vk::PipelineDepthStencilStateCreateInfo depthStencil{
	    .depthTestEnable       = vk::True,
	    .depthWriteEnable      = vk::True,
	    .depthCompareOp        = vk::CompareOp::eLess,
	    .depthBoundsTestEnable = vk::False,
	    .stencilTestEnable     = vk::False};

	// No color attachments — depth-only pass.
	vk::PipelineColorBlendStateCreateInfo colorBlending{
	    .logicOpEnable   = vk::False,
	    .attachmentCount = 0,
	    .pAttachments    = nullptr};

	std::vector                        dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
	vk::PipelineDynamicStateCreateInfo dynamicState{
	    .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
	    .pDynamicStates    = dynamicStates.data()};

	// Depth-only dynamic rendering — no color format, depth = D32_SFLOAT.
	vk::PipelineRenderingCreateInfo renderingInfo{
	    .colorAttachmentCount  = 0,
	    .depthAttachmentFormat = vk::Format::eD32Sfloat};

	vk::GraphicsPipelineCreateInfo pipelineInfo{
	    .pNext               = &renderingInfo,
	    .stageCount          = static_cast<uint32_t>(shaderStages.size()),
	    .pStages             = shaderStages.data(),
	    .pVertexInputState   = &vertexInputInfo,
	    .pInputAssemblyState = &inputAssembly,
	    .pViewportState      = &viewportState,
	    .pRasterizationState = &rasterizer,
	    .pMultisampleState   = &multisampling,
	    .pDepthStencilState  = &depthStencil,
	    .pColorBlendState    = &colorBlending,
	    .pDynamicState       = &dynamicState,
	    .layout              = *shadowPipelineLayout,
	    .renderPass          = nullptr};
	shadowPipeline = vk::raii::Pipeline(dev.logicalDevice, nullptr, pipelineInfo);
}

void PipelineCollection::createComputePipeline(VulkanDevice &dev)
{
	vk::raii::ShaderModule shaderModule = createShaderModule(dev, readFile("Shaders/Compute.slang.spv"));

	vk::PipelineShaderStageCreateInfo computeShaderStageInfo{
	    .stage  = vk::ShaderStageFlagBits::eCompute,
	    .module = *shaderModule,
	    .pName  = "computeMain"};

	createComputePipelineLayout(dev);

	vk::ComputePipelineCreateInfo pipelineInfo{
	    .stage  = computeShaderStageInfo,
	    .layout = *computePipelineLayout};
	computePipeline = vk::raii::Pipeline(dev.logicalDevice, nullptr, pipelineInfo);
}

void PipelineCollection::createPhysicsPipeline(VulkanDevice &dev)
{
	createPhysicsPipelineLayout(dev);

	vk::raii::ShaderModule            shaderModule = createShaderModule(dev, readFile("Shaders/Physics.slang.spv"));
	vk::PipelineShaderStageCreateInfo computeShaderStageInfo{
	    .stage  = vk::ShaderStageFlagBits::eCompute,
	    .module = *shaderModule,
	    .pName  = "physicsMain"};
	vk::ComputePipelineCreateInfo pipelineInfo{
	    .stage  = computeShaderStageInfo,
	    .layout = *physicsPipelineLayout};
	physicsPipeline = vk::raii::Pipeline(dev.logicalDevice, nullptr, pipelineInfo);
}

void PipelineCollection::createRayTracingPipeline(VulkanDevice &dev)
{
	vk::raii::ShaderModule rgenModule  = createShaderModule(dev, readFile("Shaders/Raygen.slang.spv"));
	vk::raii::ShaderModule rmissModule = createShaderModule(dev, readFile("Shaders/Miss.slang.spv"));
	vk::raii::ShaderModule rchitModule = createShaderModule(dev, readFile("Shaders/ClosestHit.slang.spv"));
	vk::raii::ShaderModule ranyModule  = createShaderModule(dev, readFile("Shaders/AnyHit.slang.spv"));

	std::array<vk::PipelineShaderStageCreateInfo, 4> stages = {
	    vk::PipelineShaderStageCreateInfo{
	        .stage  = vk::ShaderStageFlagBits::eRaygenKHR,
	        .module = *rgenModule,
	        .pName  = "main"},
	    vk::PipelineShaderStageCreateInfo{
	        .stage  = vk::ShaderStageFlagBits::eMissKHR,
	        .module = *rmissModule,
	        .pName  = "main"},
	    vk::PipelineShaderStageCreateInfo{
	        .stage  = vk::ShaderStageFlagBits::eClosestHitKHR,
	        .module = *rchitModule,
	        .pName  = "main"},
	    vk::PipelineShaderStageCreateInfo{
	        .stage  = vk::ShaderStageFlagBits::eAnyHitKHR,
	        .module = *ranyModule,
	        .pName  = "main"}};

	std::array<vk::RayTracingShaderGroupCreateInfoKHR, 3> groups = {
	    vk::RayTracingShaderGroupCreateInfoKHR{// Group 0 - RayGen
	                                           .type               = vk::RayTracingShaderGroupTypeKHR::eGeneral,
	                                           .generalShader      = 0,
	                                           .closestHitShader   = VK_SHADER_UNUSED_KHR,
	                                           .anyHitShader       = VK_SHADER_UNUSED_KHR,
	                                           .intersectionShader = VK_SHADER_UNUSED_KHR},
	    vk::RayTracingShaderGroupCreateInfoKHR{// Group 1 - Miss
	                                           .type               = vk::RayTracingShaderGroupTypeKHR::eGeneral,
	                                           .generalShader      = 1,
	                                           .closestHitShader   = VK_SHADER_UNUSED_KHR,
	                                           .anyHitShader       = VK_SHADER_UNUSED_KHR,
	                                           .intersectionShader = VK_SHADER_UNUSED_KHR},
	    vk::RayTracingShaderGroupCreateInfoKHR{// Group 2 - Closest Hit + Any Hit
	                                           .type               = vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
	                                           .generalShader      = VK_SHADER_UNUSED_KHR,
	                                           .closestHitShader   = 2,
	                                           .anyHitShader       = 3,
	                                           .intersectionShader = VK_SHADER_UNUSED_KHR}};

	createRayTracingPipelineLayout(dev);

	vk::RayTracingPipelineCreateInfoKHR pipelineInfo{
	    .stageCount                   = static_cast<uint32_t>(stages.size()),
	    .pStages                      = stages.data(),
	    .groupCount                   = static_cast<uint32_t>(groups.size()),
	    .pGroups                      = groups.data(),
	    .maxPipelineRayRecursionDepth = 1,
	    .layout                       = *rayTracingPipelineLayout};

	rayTracingPipeline = dev.logicalDevice.createRayTracingPipelineKHR(nullptr, nullptr, pipelineInfo);
}

void PipelineCollection::createShaderBindingTable(VulkanDevice &dev)
{
	const uint32_t handleSize      = dev.rayTracingProperties.shaderGroupHandleSize;
	const uint32_t handleAlignment = dev.rayTracingProperties.shaderGroupHandleAlignment;
	const uint32_t baseAlignment   = dev.rayTracingProperties.shaderGroupBaseAlignment;

	const uint32_t handleSizeAligned = VulkanUtils::alignUp(handleSize, handleAlignment);

	const uint32_t raygenSBTSize = VulkanUtils::alignUp(handleSizeAligned, baseAlignment);
	const uint32_t missSBTSize   = VulkanUtils::alignUp(handleSizeAligned, baseAlignment);
	const uint32_t hitSBTSize    = VulkanUtils::alignUp(handleSizeAligned, baseAlignment);

	const uint32_t       groupCount = 3;
	const uint32_t       sbtSize    = groupCount * handleSize;
	std::vector<uint8_t> handles    = rayTracingPipeline.getRayTracingShaderGroupHandlesKHR<uint8_t>(0, groupCount, sbtSize);

	auto createSBTBuffer = [&](vk::raii::Buffer &buffer, vk::raii::DeviceMemory &memory, uint32_t size, void *data, uint32_t handleOffset) {
		VulkanUtils::createBuffer(
		    dev.logicalDevice, dev.physicalDevice, size,
		    vk::BufferUsageFlagBits::eShaderBindingTableKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
		    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		    buffer, memory);

		void *mapped = memory.mapMemory(0, size);
		memcpy(mapped, (uint8_t *) data + handleOffset, handleSize);
		memory.unmapMemory();
	};

	createSBTBuffer(raygenSBTBuffer, raygenSBTMemory, raygenSBTSize, handles.data(), 0);
	createSBTBuffer(missSBTBuffer, missSBTMemory, missSBTSize, handles.data(), handleSize);
	createSBTBuffer(hitSBTBuffer, hitSBTMemory, hitSBTSize, handles.data(), handleSize * 2);

	vk::BufferDeviceAddressInfo raygenInfo{.buffer = *raygenSBTBuffer};
	raygenRegion.deviceAddress = dev.logicalDevice.getBufferAddress(raygenInfo);
	raygenRegion.stride        = raygenSBTSize;
	raygenRegion.size          = raygenSBTSize;

	vk::BufferDeviceAddressInfo missInfo{.buffer = *missSBTBuffer};
	missRegion.deviceAddress = dev.logicalDevice.getBufferAddress(missInfo);
	missRegion.stride        = handleSizeAligned;
	missRegion.size          = missSBTSize;

	vk::BufferDeviceAddressInfo hitInfo{.buffer = *hitSBTBuffer};
	hitRegion.deviceAddress = dev.logicalDevice.getBufferAddress(hitInfo);
	hitRegion.stride        = handleSizeAligned;
	hitRegion.size          = hitSBTSize;
}

// ── Helpers ────────────────────────────────────────────────────────────────

vk::raii::ShaderModule PipelineCollection::createShaderModule(VulkanDevice            &dev,
                                                              const std::vector<char> &code) const
{
	if (code.empty() || code.size() % 4 != 0)
	{
		throw std::runtime_error("invalid SPIR-V shader: code must be non-empty and a multiple of 4 bytes");
	}
	vk::ShaderModuleCreateInfo createInfo{
	    .codeSize = code.size(), .pCode = reinterpret_cast<const uint32_t *>(code.data())};
	return vk::raii::ShaderModule{dev.logicalDevice, createInfo};
}

std::vector<char> PipelineCollection::readFile(const std::string &filename)
{
	std::ifstream file(filename, std::ios::ate | std::ios::binary);
	if (!file.is_open())
	{
		throw std::runtime_error("failed to open file: " + filename);
	}
	size_t            fileSize = static_cast<size_t>(file.tellg());
	std::vector<char> buffer(fileSize);
	file.seekg(0);
	file.read(buffer.data(), fileSize);
	if (file.fail())
	{
		throw std::runtime_error("failed to read file: " + filename);
	}
	file.close();
	return buffer;
}
