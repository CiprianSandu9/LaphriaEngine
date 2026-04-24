#ifndef LAPHRIAENGINE_SCENE_H
#define LAPHRIAENGINE_SCENE_H

#include "Frustum.h"
#include "SceneNode.h"
#include "Octree.h"
#include <vulkan/vulkan_raii.hpp>
#include <string>

// Forward declaration
class ResourceManager;

// Manages the scene graph (hierarchy of SceneNodes), an octree for spatial culling,
// and convenience methods for model loading, serialization, and physics scenarios.
// The root node acts as the invisible world origin; all loaded models are attached below it.
class Scene {
public:
    Scene();

    ~Scene() = default;

    // Must be called before any nodes are added. worldBounds defines the octree's spatial extent.
    void init(Laphria::AABB worldBounds);

    // Node Management
    SceneNode::Ptr getRoot() { return root; }

    void addNode(const SceneNode::Ptr &node, const SceneNode::Ptr &parent = nullptr);

    const std::vector<SceneNode::Ptr> &getAllNodes() const { return allNodes; }
    std::vector<SceneNode::Ptr> &getAllNodes() { return allNodes; }

    void deleteNode(const SceneNode::Ptr &node);

    // Scenarios
    void createPhysicsScenario(int type, ResourceManager &rm, vk::DescriptorSetLayout layout);

    void clearScene();

    void rebuildOctree() const;
    void updateWorldTransforms() const;
    void syncSpatialIndex() const;

    // Resource Loading
    void loadModel(const std::string &path, ResourceManager &resourceManager, vk::DescriptorSetLayout layout, const SceneNode::Ptr &parent = nullptr);

    // Serialization
    void saveScene(const std::string &path, ResourceManager &resourceManager) const;

    void loadScene(const std::string &path, ResourceManager &resourceManager, vk::DescriptorSetLayout layout);

    // Runtime
    void update(float deltaTime, const ResourceManager &resourceManager) const;

    // Draws all nodes whose world position falls within cullBounds (octree-accelerated query).
    void draw(const vk::raii::CommandBuffer &cmd, const vk::raii::PipelineLayout &pipelineLayout, const ResourceManager &resourceManager,
              const Laphria::AABB &cullBounds, const Laphria::Frustum &frustum) const;

    // When freeze is true, the culling AABB is locked to its current value for debugging.
    void setFreezeCulling(bool freeze);

private:
    SceneNode::Ptr root;
    std::vector<SceneNode::Ptr> allNodes;
    std::unique_ptr<Laphria::Octree> octree;
    bool freezeCulling = false;
    mutable Laphria::AABB frozenCullBounds{{0,0,0},{0,0,0}};

    // Cached Model IDs for physics primitives
    int sphereModelId = -1;
    int cubeModelId = -1;
    int cylinderModelId = -1;

    // Temporary helper to draw a node and its children (without culling for now)
    // Draw a single node (non-recursive)
    static void drawNode(const SceneNode::Ptr &node, const vk::raii::CommandBuffer &cmd, const vk::raii::PipelineLayout &graphicsPipelineLayout, const ResourceManager &resourceManager);
};

#endif //LAPHRIAENGINE_SCENE_H
