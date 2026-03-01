#include "Scene.h"
#include "../Core/ResourceManager.h"
#include "SceneNode.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <random>


Scene::Scene()
{
	root = std::make_shared<SceneNode>("Root");
}

void Scene::init(AABB worldBounds)
{
	octree = std::make_unique<Octree>(worldBounds);
}

void Scene::addNode(const SceneNode::Ptr &node, const SceneNode::Ptr &parent)
{
	if (parent)
	{
		parent->addChild(node);
	}
	else if (root)
	{
		root->addChild(node);
	}

	// Add to flat list (and children recursively if any)
	std::vector<SceneNode::Ptr> stack;
	stack.push_back(node);
	while (!stack.empty())
	{
		auto n = stack.back();
		stack.pop_back();
		allNodes.push_back(n);
		for (const auto &c : n->getChildren())
			stack.push_back(c);
	}

	if (octree)
	{
		// Recursively add node and its children to octree
		stack.clear();
		stack.push_back(node);
		while (!stack.empty())
		{
			auto current = stack.back();
			stack.pop_back();

			octree->insert(current);

			for (const auto &child : current->getChildren())
			{
				stack.push_back(child);
			}
		}
	}
}

void Scene::deleteNode(const SceneNode::Ptr &node)
{
	if (!node || node == root)
		return;

	// Collect the node and all its descendants
	std::vector<SceneNode::Ptr> toRemove;
	std::vector<SceneNode::Ptr> stack{node};
	while (!stack.empty())
	{
		auto current = stack.back();
		stack.pop_back();
		toRemove.push_back(current);
		for (const auto &child : current->getChildren())
			stack.push_back(child);
	}

	// Remove all collected nodes from allNodes
	allNodes.erase(
	    std::remove_if(allNodes.begin(), allNodes.end(), [&](const SceneNode::Ptr &n) {
		    return std::find(toRemove.begin(), toRemove.end(), n) != toRemove.end();
	    }),
	    allNodes.end());

	if (node->getParent())
	{
		node->getParent()->removeChild(node);
		rebuildOctree();
	}
}

void Scene::rebuildOctree() const
{
	if (!octree || !root)
		return;

	octree->clear();

	std::vector<SceneNode::Ptr> stack;
	stack.push_back(root);
	while (!stack.empty())
	{
		auto current = stack.back();
		stack.pop_back();

		octree->insert(current);

		for (const auto &child : current->getChildren())
		{
			stack.push_back(child);
		}
	}
}

void Scene::loadModel(const std::string &path, ResourceManager &resourceManager, vk::DescriptorSetLayout layout,
                      const SceneNode::Ptr &parent)
{
	auto node = resourceManager.loadGltfModel(path, layout);
	addNode(node, parent);
}

void serializeNode(const SceneNode::Ptr &node, nlohmann::json &j, ResourceManager &resourceManager)
{
	j["name"] = node->name;

	// Transform
	glm::vec3 pos = node->getPosition();
	glm::quat rot = node->getRotation();
	glm::vec3 scl = node->getScale();

	j["position"] = {pos.x, pos.y, pos.z};
	j["rotation"] = {rot.w, rot.x, rot.y, rot.z};        // w, x, y, z
	j["scale"]    = {scl.x, scl.y, scl.z};

	// Model Ref
	if (node->modelId != -1)
	{
		if (auto *res = resourceManager.getModelResource(node->modelId))
		{
			j["modelPath"] = res->path;
		}
	}

	// Mesh Indices
	j["meshIndices"] = node->getMeshIndices();

	// Children
	j["children"] = nlohmann::json::array();
	for (const auto &child : node->getChildren())
	{
		nlohmann::json childJ;
		serializeNode(child, childJ, resourceManager);
		j["children"].push_back(childJ);
	}
}

void Scene::saveScene(const std::string &path, ResourceManager &resourceManager) const
{
	if (!root)
		return;

	nlohmann::json rootJ;
	serializeNode(root, rootJ, resourceManager);

	std::ofstream o(path);
	o << std::setw(4) << rootJ << std::endl;

	std::cout << "Saved scene to " << path << std::endl;
}

SceneNode::Ptr deserializeNode(const nlohmann::json &j, ResourceManager &resourceManager,
                               std::map<std::string, int> &pathCache,
                               vk::DescriptorSetLayout     layout)
{
	auto node = std::make_shared<SceneNode>(j.value("name", "Node"));

	// Transform
	if (j.contains("position"))
	{
		auto p = j["position"];
		node->setPosition(glm::vec3(p[0], p[1], p[2]));
	}
	if (j.contains("rotation"))
	{
		auto r = j["rotation"];
		node->setRotation(glm::quat(r[0], r[1], r[2], r[3]));
	}
	if (j.contains("scale"))
	{
		auto s = j["scale"];
		node->setScale(glm::vec3(s[0], s[1], s[2]));
	}

	// Model
	if (j.contains("modelPath"))
	{
		std::string modelPath = j["modelPath"];

		int modelId = -1;

		// Cache Check
		auto it = pathCache.find(modelPath);
		if (it != pathCache.end())
		{
			modelId = it->second;
		}
		else
		{
			// Load if not in cache
			try
			{
				// loadGltfModel returns a root node. We only want the Resource ID.
				// The returned node structure is discarded because we reconstruct from JSON.
				auto modelRoot       = resourceManager.loadGltfModel(modelPath, layout);
				modelId              = modelRoot->modelId;        // Extract ID from the loaded node
				pathCache[modelPath] = modelId;
			}
			catch (const std::exception &e)
			{
				std::cerr << "Failed to load model during deserialization: " << modelPath << " (" << e.what() << ")" << std::endl;
			}
		}

		if (modelId != -1)
		{
			node->modelId = modelId;
		}
	}

	// Mesh Indices
	if (j.contains("meshIndices"))
	{
		node->meshIndices = j["meshIndices"].get<std::vector<int>>();
	}

	// Children
	if (j.contains("children"))
	{
		for (const auto &childJ : j["children"])
		{
			node->addChild(deserializeNode(childJ, resourceManager, pathCache, layout));
		}
	}

	return node;
}

void Scene::loadScene(const std::string &path, ResourceManager &resourceManager, vk::DescriptorSetLayout layout)
{
	std::ifstream i(path);
	if (!i.is_open())
	{
		std::cerr << "Failed to open scene file: " << path << std::endl;
		return;
	}

	nlohmann::json j;
	i >> j;

	// Clear current scene
	root = nullptr;
	if (octree)
		octree->clear();

	// Temp path cache for this load session
	std::map<std::string, int> pathCache;

	std::cout << "Loading scene from " << path << std::endl;

	root = deserializeNode(j, resourceManager, pathCache, layout);

	rebuildOctree();
}

void Scene::update(float deltaTime)
{
	// Traverse and update logic if needed (animations etc)
}

void Scene::setFreezeCulling(bool freeze)
{
	freezeCulling = freeze;
}

void Scene::draw(const vk::raii::CommandBuffer &cmd, const vk::raii::PipelineLayout &pipelineLayout,
                 ResourceManager &resourceManager, const AABB &cullBounds) const
{
	if (!root || !octree)
		return;

	// 1. Cull against octree — freeze culling snapshots the bounds for debugging
	std::vector<SceneNode::Ptr> visibleNodes;
	if (freezeCulling)
	{
		// Keep using the bounds that were active when freeze was first applied
		octree->query(frozenCullBounds, visibleNodes);
	}
	else
	{
		frozenCullBounds = cullBounds;        // Keep snapshot up-to-date for when freeze is toggled
		octree->query(cullBounds, visibleNodes);
	}

	for (const auto &node : visibleNodes)
	{
		drawNode(node, cmd, pipelineLayout, resourceManager);
	}
}

void Scene::drawNode(const SceneNode::Ptr &node, const vk::raii::CommandBuffer &cmd, const vk::raii::PipelineLayout &graphicsPipelineLayout,
                     const ResourceManager &resourceManager)
{
	// Compute global transform efficiently
	glm::mat4 globalTransform = node->getWorldTransform();

	// Draw if it has a model
	if (node->modelId != -1)
	{
		if (auto *modelRes = resourceManager.getModelResource(node->modelId))
		{
			// Bind Mesh Buffers
			resourceManager.bindResources(cmd, node->modelId);

			// Bind Material/Texture Descriptor Set (Set 1)
			if (*modelRes->descriptorSet)
			{
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *graphicsPipelineLayout, 1, {*modelRes->descriptorSet},
				                       nullptr);
			}

			for (int meshIdx : node->getMeshIndices())
			{
				if (meshIdx >= 0 && meshIdx < modelRes->meshes.size())
				{
					const auto &mesh = modelRes->meshes[meshIdx];
					for (const auto &primitive : mesh.primitives)
					{
						ScenePushConstants pc{};
						pc.modelMatrix   = globalTransform;
						pc.materialIndex = primitive.flatPrimitiveIndex;
						cmd.pushConstants<Laphria::ScenePushConstants>(*graphicsPipelineLayout,
						                                               vk::ShaderStageFlagBits::eVertex |
						                                                   vk::ShaderStageFlagBits::eFragment,
						                                               0, pc);

						cmd.drawIndexed(primitive.indexCount, 1, primitive.firstIndex, primitive.vertexOffset, 0);
					}
				}
			}
		}
	}
}

// ----------------------------------------------------------------------------
// Physics ScenariosImplementation
// ----------------------------------------------------------------------------

void Scene::clearScene()
{
	allNodes.clear();
	sphereModelId   = -1;
	cubeModelId     = -1;
	cylinderModelId = -1;

	if (root)
	{
		root = std::make_shared<SceneNode>("Root");
		allNodes.clear();

		// Re-init octree
		if (octree)
			octree = std::make_unique<Octree>(octree->getBounds());
	}
}

void Scene::createPhysicsScenario(int type, ResourceManager &rm, vk::DescriptorSetLayout layout)
{
	clearScene();

	if (sphereModelId == -1)
	{
		auto node     = rm.createSphereModel(1.0f, 32, 16, layout);
		sphereModelId = node->modelId;
	}
	if (cubeModelId == -1)
	{
		auto node   = rm.createCubeModel(1.0f, layout);
		cubeModelId = node->modelId;
	}
	if (cylinderModelId == -1)
	{
		auto node       = rm.createCylinderModel(0.5f, 1.0f, 32, layout);
		cylinderModelId = node->modelId;
	}

	std::mt19937 rng(std::random_device{}());
	// Bounds for spawn (keep inside box -50 to 50)
	std::uniform_real_distribution<float> distPos(-40.0f, 40.0f);
	std::uniform_real_distribution<float> distVel(-5.0f, 5.0f);
	std::uniform_real_distribution<float> distScale(0.5f, 2.0f);

	int spheres   = 0;
	int cubes     = 0;
	int cylinders = 0;

	if (type == 1)
	{
		// 100 Spheres, 250 Cubes, 500 Cylinders
		spheres   = 100;
		cubes     = 250;
		cylinders = 500;
	}
	else if (type == 2)
	{
		// 250 Spheres, 500 Cubes, 1000 Cylinders
		spheres   = 250;
		cubes     = 500;
		cylinders = 1000;
	}
	else if (type == 3)
	{
		// 500 Spheres, 1000 Cubes, 2500 Cylinders
		spheres   = 500;
		cubes     = 1000;
		cylinders = 2500;
	}
	else
	{
		spheres   = 10;
		cubes     = 10;
		cylinders = 10;
	}

	int total = spheres + cubes + cylinders;

	for (int i = 0; i < total; ++i)
	{
		int objType = 0;        // 0=Sphere, 1=Box, 2=Cylinder
		if (i < spheres)
			objType = 0;
		else if (i < spheres + cubes)
			objType = 1;
		else
			objType = 2;

		std::string name = (objType == 0) ? "Sphere" : (objType == 1 ? "Cube" : "Cylinder");
		auto        node = std::make_shared<SceneNode>(name);

		if (objType == 0)
			node->modelId = sphereModelId;
		else if (objType == 1)
			node->modelId = cubeModelId;
		else
			node->modelId = cylinderModelId;

		node->addMeshIndex(0);

		float s = distScale(rng);
		node->setScale(glm::vec3(s));
		node->setPosition(glm::vec3(distPos(rng), distPos(rng), distPos(rng)));

		node->physics.enabled  = true;
		node->physics.isStatic = false;
		node->physics.mass     = 1.0f * s * s * s;
		node->physics.velocity = glm::vec3(distVel(rng), distVel(rng), distVel(rng));

		if (objType == 0)
		{
			node->physics.colliderType = SceneNode::ColliderType::Sphere;
			node->physics.radius       = 1.0f * s;
		}
		else if (objType == 1)
		{
			node->physics.colliderType = SceneNode::ColliderType::Box;
			node->physics.halfExtents  = glm::vec3(0.5f) * s;
		}
		else
		{
			// Cylinder: treated as Box for physics now
			node->physics.colliderType = SceneNode::ColliderType::Cylinder;
			// A cylinder with radius 0.5 and height 1.0 (from 0.5 to -0.5)
			// Bounding box half extents = (radius, height/2, radius)
			// radius = 0.5 * s, height/2 = 0.5 * s
			node->physics.halfExtents = glm::vec3(0.5f, 0.5f, 0.5f) * s;
		}

		node->physics.restitution = 0.8f;
		node->physics.friction    = 0.5f;

		addNode(node);
	}
}
