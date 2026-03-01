#ifndef LAPHRIAENGINE_UISYSTEM_H
#define LAPHRIAENGINE_UISYSTEM_H

#include <random>

#include "../Physics/PhysicsSystem.h"
#include "../SceneManagement/Scene.h"
#include "EngineAuxiliary.h"
#include "VulkanDevice.h"


// Owns ImGui lifecycle, all editor draw calls, and UI-driven simulation state.
class UISystem
{
  public:
	// Call after the swapchain has been created (needs colorFormat / depthFormat).
	void init(VulkanDevice &dev, GLFWwindow *window,
	          vk::Format colorFormat, vk::Format depthFormat);

	// Record one ImGui frame worth of widgets.
	// Must be called between ImGui::NewFrame() and ImGui::Render() in EngineCore.
	void draw(GLFWwindow *window, Scene &scene, PhysicsSystem &physics,
	          ResourceManager &rm, vk::DescriptorSetLayout matLayout);

	void cleanup();

	// ── State shared with EngineCore's main loop ──────────────────────────────
	bool      useGPUPhysics     = false;
	bool      useRayTracing     = false;
	bool      simulationRunning = false;
	float     physicsTime       = 0.0f;        // updated by EngineCore after each tick
	glm::vec3 lightDirection    = glm::vec3(-0.25f, -1.0f, 0.0f);

  private:
	vk::raii::DescriptorPool imguiDescriptorPool{nullptr};

	// Editor state
	SceneNode::Ptr              selectedNode{nullptr};
	std::vector<SceneNode::Ptr> nodesPendingDeletion;
	bool                        showModelLoadDialog = false;
	char                        modelLoadPath[512]  = "assets/paladin.glb";
	bool                        showSceneSaveDialog = false;
	bool                        showSceneLoadDialog = false;
	char                        scenePath[512]      = "scene.json";
	std::mt19937                rng{std::random_device{}()};

	void drawMainMenuBar(GLFWwindow *window);
	void drawSceneHierarchy(Scene &scene, ResourceManager &rm, vk::DescriptorSetLayout matLayout);
	void drawSceneNode(const SceneNode::Ptr &node, Scene &scene);
	void drawInspector();
	void drawPhysicsUI(Scene &scene, PhysicsSystem &physics,
	                   ResourceManager &rm, vk::DescriptorSetLayout matLayout);
};

#endif        // LAPHRIAENGINE_UISYSTEM_H
