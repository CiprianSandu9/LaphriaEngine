#ifndef LAPHRIAENGINE_PHYSICSSYSTEM_H
#define LAPHRIAENGINE_PHYSICSSYSTEM_H

#include "../SceneManagement/SceneNode.h"
#include <vector>
#include <vulkan/vulkan_raii.hpp>

#include "PhysicsDefines.h"

class PhysicsSystem {
public:
    PhysicsSystem();

    ~PhysicsSystem() = default;

    // CPU Logic
    void updateCPU(std::vector<SceneNode::Ptr> &nodes, float deltaTime);

    // GPU Logic
    void updateGPU(std::vector<SceneNode::Ptr> &nodes, float deltaTime,
                   const vk::raii::CommandBuffer &cmd,
                   const vk::raii::PipelineLayout &layout,
                   const vk::raii::Pipeline &pipeline,
                   const vk::raii::DescriptorSet &descriptorSet);

    void syncFromGPU(std::vector<SceneNode::Ptr> &nodes);

    // Configuration
    void setGravity(const glm::vec3 &g) { gravity = g; }

    void setWorldBounds(const glm::vec3 &min, const glm::vec3 &max) {
        worldMin = min;
        worldMax = max;
    }

    void setGlobalFriction(float f) { globalFriction = f; }

    void createSSBO(const vk::raii::Device &device, const vk::raii::PhysicalDevice &physDevice, size_t size);

    [[nodiscard]] const vk::raii::Buffer &getSSBOBuffer() const { return physicsSSBO; }

private:
    glm::vec3 gravity{0.0f, -9.81f, 0.0f};
    glm::vec3 worldMin{-50.0f};
    glm::vec3 worldMax{50.0f};
    float globalFriction{0.5f};

    // CPU Helpers
    void integrate(const SceneNode::Ptr &node, float dt) const;

    void checkBoundaries(const SceneNode::Ptr &node);

    void resolveCollisions(const std::vector<SceneNode::Ptr> &nodes);

    // Collision Detection Primitives
    bool checkSphereSphere(SceneNode::Ptr &a, SceneNode::Ptr &b);

    bool checkAABBAABB(SceneNode::Ptr &a, SceneNode::Ptr &b);

    bool checkSphereAABB(SceneNode::Ptr &sphere, SceneNode::Ptr &box);

    static void solveContact(SceneNode::Ptr &a, SceneNode::Ptr &b, const glm::vec3 &normal, float penetration);

    // GPU Members
    std::vector<PhysicsObject> hostPhysicsObjects;
    vk::raii::Buffer physicsSSBO{nullptr};
    vk::raii::DeviceMemory physicsSSBOMemory{nullptr};
    void *physicsSSBOMapped{nullptr};
    size_t currentSSBOSize = 0;

    void updateSSBO(std::vector<SceneNode::Ptr> &nodes);
};

#endif // LAPHRIAENGINE_PHYSICSSYSTEM_H
