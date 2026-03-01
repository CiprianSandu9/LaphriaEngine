#include "UISystem.h"

#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <stdexcept>


#include "EngineAuxiliary.h"

using namespace Laphria;

void UISystem::init(VulkanDevice &dev, GLFWwindow *window,
                    vk::Format colorFormat, vk::Format depthFormat)
{
	std::vector<vk::DescriptorPoolSize> poolSizes = {
	    {vk::DescriptorType::eSampler, 1000},
	    {vk::DescriptorType::eCombinedImageSampler, 1000},
	    {vk::DescriptorType::eSampledImage, 1000},
	    {vk::DescriptorType::eStorageImage, 1000},
	    {vk::DescriptorType::eUniformTexelBuffer, 1000},
	    {vk::DescriptorType::eStorageTexelBuffer, 1000},
	    {vk::DescriptorType::eUniformBuffer, 1000},
	    {vk::DescriptorType::eStorageBuffer, 1000},
	    {vk::DescriptorType::eUniformBufferDynamic, 1000},
	    {vk::DescriptorType::eStorageBufferDynamic, 1000},
	    {vk::DescriptorType::eInputAttachment, 1000}};

	vk::DescriptorPoolCreateInfo poolInfo = {};
	poolInfo.flags                        = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
	poolInfo.maxSets                      = 1000;
	poolInfo.poolSizeCount                = static_cast<uint32_t>(poolSizes.size());
	poolInfo.pPoolSizes                   = poolSizes.data();

	imguiDescriptorPool = vk::raii::DescriptorPool(dev.logicalDevice, poolInfo);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO &io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

	ImGui::StyleColorsDark();

	ImGuiStyle &style = ImGui::GetStyle();
	style.ScaleAllSizes(static_cast<float>(WIDTH) / HEIGHT);

	ImGui_ImplGlfw_InitForVulkan(window, true);
	ImGui_ImplVulkan_InitInfo init_info = {};
	init_info.ApiVersion                = VK_API_VERSION_1_4;
	init_info.Instance                  = *dev.instance;
	init_info.PhysicalDevice            = *dev.physicalDevice;
	init_info.Device                    = *dev.logicalDevice;
	init_info.Queue                     = *dev.queue;
	init_info.DescriptorPool            = *imguiDescriptorPool;
	init_info.MinImageCount             = 3;
	init_info.ImageCount                = 3;
	init_info.MSAASamples               = VK_SAMPLE_COUNT_1_BIT;
	init_info.UseDynamicRendering       = true;

	static const auto color_format = static_cast<VkFormat>(colorFormat);

	VkPipelineRenderingCreateInfoKHR pipeline_info = {};
	pipeline_info.sType                            = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
	pipeline_info.colorAttachmentCount             = 1;
	pipeline_info.pColorAttachmentFormats          = &color_format;
	pipeline_info.depthAttachmentFormat            = static_cast<VkFormat>(depthFormat);
	pipeline_info.stencilAttachmentFormat          = VK_FORMAT_UNDEFINED;

	init_info.PipelineRenderingCreateInfo = pipeline_info;

	ImGui_ImplVulkan_Init(&init_info);
	ImGui_ImplVulkan_CreateFontsTexture();
}

void UISystem::draw(GLFWwindow *window, Scene &scene, PhysicsSystem &physics,
                    ResourceManager &rm, vk::DescriptorSetLayout matLayout)
{
	drawMainMenuBar(window);
	drawSceneHierarchy(scene, rm, matLayout);
	drawInspector();
	drawPhysicsUI(scene, physics, rm, matLayout);

	ImGui::Begin("Lighting Control");
	ImGui::DragFloat3("Light Direction", glm::value_ptr(lightDirection), 0.01f, 0.0f, 0.0f);
	ImGui::Text("Dir: %.2f, %.2f, %.2f", lightDirection.x, lightDirection.y, lightDirection.z);
	ImGui::Separator();
	static bool freezeCulling = false;
	if (ImGui::Checkbox("Freeze Culling", &freezeCulling))
	{
		scene.setFreezeCulling(freezeCulling);
	}
	if (freezeCulling)
	{
		ImGui::TextColored(ImVec4(1, 1, 0, 1), "Culling frustum is frozen");
	}
	ImGui::End();

	if (showModelLoadDialog)
	{
		ImGui::OpenPopup("Load Model");
		showModelLoadDialog = false;
	}
	if (ImGui::BeginPopupModal("Load Model", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
	{
		ImGui::InputText("Path", modelLoadPath, IM_ARRAYSIZE(modelLoadPath));
		if (ImGui::Button("Load", ImVec2(120, 0)))
		{
			try
			{
				scene.loadModel(modelLoadPath, rm, matLayout, selectedNode);
				LOGI("Loaded model: %s", modelLoadPath);
			}
			catch (const std::exception &e)
			{
				LOGI("Failed to load model: %s", e.what());
			}
			ImGui::CloseCurrentPopup();
		}
		ImGui::SameLine();
		if (ImGui::Button("Cancel", ImVec2(120, 0)))
		{
			ImGui::CloseCurrentPopup();
		}
		ImGui::EndPopup();
	}

	if (showSceneSaveDialog)
	{
		ImGui::OpenPopup("Save Scene");
		showSceneSaveDialog = false;
	}
	if (ImGui::BeginPopupModal("Save Scene", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
	{
		ImGui::InputText("Path", scenePath, IM_ARRAYSIZE(scenePath));
		if (ImGui::Button("Save", ImVec2(120, 0)))
		{
			scene.saveScene(scenePath, rm);
			ImGui::CloseCurrentPopup();
		}
		ImGui::SameLine();
		if (ImGui::Button("Cancel", ImVec2(120, 0)))
		{
			ImGui::CloseCurrentPopup();
		}
		ImGui::EndPopup();
	}

	if (showSceneLoadDialog)
	{
		ImGui::OpenPopup("Load Scene");
		showSceneLoadDialog = false;
	}
	if (ImGui::BeginPopupModal("Load Scene", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
	{
		ImGui::InputText("Path", scenePath, IM_ARRAYSIZE(scenePath));
		if (ImGui::Button("Load", ImVec2(120, 0)))
		{
			try
			{
				scene.loadScene(scenePath, rm, matLayout);
				LOGI("Loaded scene: %s", scenePath);
			}
			catch (const std::exception &e)
			{
				LOGI("Failed to load scene: %s", e.what());
			}
			ImGui::CloseCurrentPopup();
		}
		ImGui::SameLine();
		if (ImGui::Button("Cancel", ImVec2(120, 0)))
		{
			ImGui::CloseCurrentPopup();
		}
		ImGui::EndPopup();
	}
}

void UISystem::cleanup()
{
	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

void UISystem::drawMainMenuBar(GLFWwindow *window)
{
	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("Load Model..."))
			{
				showModelLoadDialog = true;
			}
			if (ImGui::MenuItem("Save Scene..."))
			{
				showSceneSaveDialog = true;
			}
			if (ImGui::MenuItem("Load Scene..."))
			{
				showSceneLoadDialog = true;
			}
			if (ImGui::MenuItem("Exit"))
			{
				glfwSetWindowShouldClose(window, true);
			}
			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();
	}
}

void UISystem::drawSceneHierarchy(Scene &scene, ResourceManager &rm, vk::DescriptorSetLayout matLayout)
{
	ImGui::Begin("Scene Hierarchy");

	if (scene.getRoot())
	{
		drawSceneNode(scene.getRoot(), scene);
	}

	if (!nodesPendingDeletion.empty())
	{
		for (const auto &node : nodesPendingDeletion)
		{
			scene.deleteNode(node);
		}
		nodesPendingDeletion.clear();
	}

	ImGui::End();
}

void UISystem::drawSceneNode(const SceneNode::Ptr &node, Scene &scene)
{
	ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;
	if (selectedNode == node)
	{
		flags |= ImGuiTreeNodeFlags_Selected;
	}
	if (node->getChildren().empty())
	{
		flags |= ImGuiTreeNodeFlags_Leaf;
	}

	bool opened = ImGui::TreeNodeEx(reinterpret_cast<void *>(reinterpret_cast<intptr_t>(node.get())), flags, "%s", node->name.c_str());

	if (ImGui::IsItemClicked())
	{
		selectedNode = node;
	}

	if (ImGui::BeginPopupContextItem())
	{
		if (ImGui::MenuItem("Delete"))
		{
			if (node != scene.getRoot())
			{
				nodesPendingDeletion.push_back(node);
				if (selectedNode == node)
					selectedNode = nullptr;
			}
		}
		if (ImGui::MenuItem("Add Child"))
		{
			auto child = std::make_shared<SceneNode>("New Node");
			node->addChild(child);
			scene.rebuildOctree();
		}
		ImGui::EndPopup();
	}

	if (opened)
	{
		for (auto &child : node->getChildren())
		{
			drawSceneNode(child, scene);
		}
		ImGui::TreePop();
	}
}

void UISystem::drawInspector()
{
	ImGui::Begin("Inspector");

	if (selectedNode)
	{
		char nameBuf[128];
		strncpy_s(nameBuf, selectedNode->name.c_str(), sizeof(nameBuf));
		if (ImGui::InputText("Name", nameBuf, sizeof(nameBuf)))
		{
			selectedNode->name = nameBuf;
		}

		ImGui::Separator();
		ImGui::Text("Transform");

		glm::vec3 pos = selectedNode->getPosition();
		if (ImGui::DragFloat3("Position", glm::value_ptr(pos), 0.1f))
		{
			selectedNode->setPosition(pos);
		}

		glm::vec3 euler = selectedNode->getEulerRotation();
		if (ImGui::DragFloat3("Rotation", glm::value_ptr(euler), 0.5f))
		{
			selectedNode->setEulerRotation(euler);
		}

		glm::vec3 scale = selectedNode->getScale();
		if (ImGui::DragFloat3("Scale", glm::value_ptr(scale), 0.1f))
		{
			selectedNode->setScale(scale);
		}
	}
	else
	{
		ImGui::Text("No object selected.");
	}

	ImGui::End();
}

void UISystem::drawPhysicsUI(Scene &scene, PhysicsSystem &physics,
                             ResourceManager &rm, vk::DescriptorSetLayout matLayout)
{
	ImGui::Begin("Engine Controls");

	ImGui::Text("Rendering Backend:");
	if (ImGui::RadioButton("Rasterizer", !useRayTracing))
		useRayTracing = false;
	ImGui::SameLine();
	if (ImGui::RadioButton("Ray Tracer (RTX)", useRayTracing))
		useRayTracing = true;

	ImGui::Separator();

	ImGui::Text("Physics Backend:");
	if (ImGui::RadioButton("CPU", !useGPUPhysics))
		useGPUPhysics = false;
	ImGui::SameLine();
	if (ImGui::RadioButton("GPU", useGPUPhysics))
		useGPUPhysics = true;

	ImGui::Separator();

	if (ImGui::Button(simulationRunning ? "Pause" : "Play"))
	{
		simulationRunning = !simulationRunning;
	}
	ImGui::SameLine();
	if (ImGui::Button("Random Impulse"))
	{
		for (auto &node : scene.getAllNodes())
		{
			if (node->physics.enabled && !node->physics.isStatic)
			{
				std::uniform_real_distribution<float> dist(0.0f, 1.0f);
				float                                 rx        = dist(rng);
				float                                 ry        = dist(rng);
				float                                 rz        = dist(rng);
				glm::vec3                             randomDir = glm::normalize(glm::vec3(rx, ry, rz) * 2.0f - 1.0f);
				node->physics.velocity += randomDir * 15.0f;
			}
		}
	}
	ImGui::SameLine();
	if (ImGui::Button("Reset"))
	{
		simulationRunning = false;
		for (auto &node : scene.getAllNodes())
		{
			node->resetToInitialState();
		}
	}

	ImGui::Separator();
	ImGui::Text("Scenarios (Predefined):");
	if (ImGui::Button("100S-250C-500CY"))
	{
		scene.createPhysicsScenario(1, rm, matLayout);
		for (auto &node : scene.getAllNodes())
			node->storeInitialState();
		simulationRunning = false;
	}
	if (ImGui::Button("250S-500C-1000CY"))
	{
		scene.createPhysicsScenario(2, rm, matLayout);
		for (auto &node : scene.getAllNodes())
			node->storeInitialState();
		simulationRunning = false;
	}
	if (ImGui::Button("500S-1000C-2500CY"))
	{
		scene.createPhysicsScenario(3, rm, matLayout);
		for (auto &node : scene.getAllNodes())
			node->storeInitialState();
		simulationRunning = false;
	}

	ImGui::Separator();
	ImGui::Text("Global Physics Parameters:");
	{
		glm::vec3 gravity = glm::vec3(0.f, -9.81f, 0.f);
		if (ImGui::DragFloat3("Gravity", glm::value_ptr(gravity), 0.1f, -50.f, 50.f))
		{
			physics.setGravity(gravity);
		}
		static float globalFriction = 0.5f;
		if (ImGui::SliderFloat("Global Friction (GPU)", &globalFriction, 0.0f, 1.0f))
		{
			physics.setGlobalFriction(globalFriction);
		}
	}

	ImGui::Separator();
	ImGui::Text("Metrics:");
	ImGui::Text("Compute Time: %.3f ms", physicsTime);
	ImGui::Text("Object Count: %d", static_cast<int>(scene.getAllNodes().size()));

	ImGui::End();
}
