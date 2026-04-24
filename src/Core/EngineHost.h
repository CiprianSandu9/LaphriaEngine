#ifndef LAPHRIAENGINE_ENGINEHOST_H
#define LAPHRIAENGINE_ENGINEHOST_H

#include <functional>
#include <memory>
#include <string>

struct GLFWwindow;
class Camera;
class Scene;
class SceneNode;
class PhysicsSystem;
class UISystem;

class ResourceManager;
namespace Laphria
{
struct MaterialData;
}

struct EngineServices
{
	GLFWwindow      *window;
	Camera          &camera;
	Scene           &scene;
	PhysicsSystem   &physics;
	ResourceManager &resourceManager;
	UISystem        &ui;
	std::function<void(const std::string &path)> loadSceneAsset;
	std::function<void(const std::string &path)> saveSceneAsset;
	std::function<void(const std::string &path, const std::shared_ptr<SceneNode> &parent)> loadModelAsset;
	std::function<std::shared_ptr<SceneNode>(float size)> createCubePrimitive;
	std::function<std::shared_ptr<SceneNode>(float size, const Laphria::MaterialData &material)> createCubePrimitiveWithMaterial;
	std::function<std::shared_ptr<SceneNode>(float radius, int slices, int stacks)> createSpherePrimitive;
	std::function<std::shared_ptr<SceneNode>(float radius, float height, int slices)> createCylinderPrimitive;
};

struct EngineHostCallbacks
{
	std::function<void(EngineServices &services)> initialize;
	std::function<void(EngineServices &services, float deltaTimeSeconds)> updateFrame;
	std::function<void(EngineServices &services)> drawUi;
	std::function<void(EngineServices &services)> shutdown;
};

struct EngineHostOptions
{
	std::string windowTitle = "LaphriaEngine";
	bool        showEditorPanels = true;
	bool        runPhysicsSimulation = true;
	bool        enableDefaultCameraInput = true;
};

class EngineHost
{
  public:
	EngineHost();
	EngineHost(EngineHostOptions options, EngineHostCallbacks callbacks = {});

	void run() const;

  private:
	EngineHostOptions   options;
	EngineHostCallbacks callbacks;
};

#endif        // LAPHRIAENGINE_ENGINEHOST_H
