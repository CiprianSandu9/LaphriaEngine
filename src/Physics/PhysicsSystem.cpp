#include "PhysicsSystem.h"
#include "PhysicsDefines.h"
#include "../Core/VulkanUtils.h"
#include "../Core/EngineAuxiliary.h"
#include <cassert>

using namespace Laphria;
#include <glm/gtx/norm.hpp>

PhysicsSystem::PhysicsSystem() {
}

void PhysicsSystem::updateCPU(std::vector<SceneNode::Ptr> &nodes, float deltaTime) {
    // 1. Integration (Move objects)
    for (auto &node: nodes) {
        if (!node->physics.enabled || node->physics.isStatic) continue;
        integrate(node, deltaTime);
        checkBoundaries(node);
    }

    // 2. Collision Detection & Resolution
    resolveCollisions(nodes);
}

void PhysicsSystem::integrate(const SceneNode::Ptr &node, float dt) const {
    auto &phys = node->physics;

    // Symplectic Euler
    // v += a * dt
    phys.velocity += (gravity + phys.acceleration) * dt;

    // Apply Friction (Simple Damping)
    phys.velocity *= (1.0f - phys.friction * dt);

    // x += v * dt
    glm::vec3 pos = node->getPosition();
    pos += phys.velocity * dt;
    node->setPosition(pos);

    // Reset acceleration
    phys.acceleration = glm::vec3(0.0f);
}

void PhysicsSystem::checkBoundaries(const SceneNode::Ptr &node) {
    glm::vec3 pos = node->getPosition();
    auto &phys = node->physics;

    glm::vec3 offset(0.0f);
    if (phys.colliderType == SceneNode::ColliderType::Sphere) {
        offset = glm::vec3(phys.radius);
    } else {
        // Box or Cylinder
        offset = phys.halfExtents;
    }

    // Bounce off each axis of the world AABB.
    // The 0.5 m/s threshold stops very slow objects from jittering at rest against a wall.
    for (int i = 0; i < 3; i++) {
        if (pos[i] - offset[i] < worldMin[i]) {
            pos[i] = worldMin[i] + offset[i];
            if (std::abs(phys.velocity[i]) < 0.5f) {
                phys.velocity[i] = 0.0f;
            } else {
                phys.velocity[i] = -phys.velocity[i] * phys.restitution;
            }
        } else if (pos[i] + offset[i] > worldMax[i]) {
            pos[i] = worldMax[i] - offset[i];
            if (std::abs(phys.velocity[i]) < 0.5f) {
                phys.velocity[i] = 0.0f;
            } else {
                phys.velocity[i] = -phys.velocity[i] * phys.restitution;
            }
        }
    }

    node->setPosition(pos);
}

void PhysicsSystem::resolveCollisions(const std::vector<SceneNode::Ptr> &nodes) {
    // Naive O(N^2)
    for (size_t i = 0; i < nodes.size(); i++) {
        for (size_t j = i + 1; j < nodes.size(); j++) {
            SceneNode::Ptr a = nodes[i];
            SceneNode::Ptr b = nodes[j];

            if (!a->physics.enabled || !b->physics.enabled) continue;
            if (a->physics.isStatic && b->physics.isStatic) continue;

            auto typeA = a->physics.colliderType;
            auto typeB = b->physics.colliderType;

            // Treat Cylinder as Box
            if (typeA == SceneNode::ColliderType::Cylinder) typeA = SceneNode::ColliderType::Box;
            if (typeB == SceneNode::ColliderType::Cylinder) typeB = SceneNode::ColliderType::Box;

            // Dispatch based on types
            if (typeA == SceneNode::ColliderType::Sphere && typeB == SceneNode::ColliderType::Sphere) {
                checkSphereSphere(a, b);
            } else if (typeA == SceneNode::ColliderType::Box && typeB == SceneNode::ColliderType::Box) {
                checkAABBAABB(a, b);
            } else if (typeA == SceneNode::ColliderType::Sphere && typeB == SceneNode::ColliderType::Box) {
                checkSphereAABB(a, b);
            } else if (typeA == SceneNode::ColliderType::Box && typeB == SceneNode::ColliderType::Sphere) {
                checkSphereAABB(b, a); // Swap
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Collision Primitives
// ----------------------------------------------------------------------------

bool PhysicsSystem::checkSphereSphere(SceneNode::Ptr &a, SceneNode::Ptr &b) {
    glm::vec3 delta = a->getPosition() - b->getPosition();
    float distSq = glm::length2(delta);
    float radiusSum = a->physics.radius + b->physics.radius;

    if (distSq < radiusSum * radiusSum) {
        float dist = std::sqrt(distSq);
        glm::vec3 normal = dist > 0.0001f ? delta / dist : glm::vec3(0, 1, 0);
        float penetration = radiusSum - dist;
        solveContact(a, b, normal, penetration);
        return true;
    }
    return false;
}

bool PhysicsSystem::checkAABBAABB(SceneNode::Ptr &a, SceneNode::Ptr &b) {
    glm::vec3 posA = a->getPosition();
    glm::vec3 posB = b->getPosition();

    glm::vec3 extA = a->physics.halfExtents;
    glm::vec3 extB = b->physics.halfExtents;

    glm::vec3 minA = posA - extA;
    glm::vec3 maxA = posA + extA;
    glm::vec3 minB = posB - extB;
    glm::vec3 maxB = posB + extB;

    bool overlapX = maxA.x >= minB.x && minA.x <= maxB.x;
    bool overlapY = maxA.y >= minB.y && minA.y <= maxB.y;
    bool overlapZ = maxA.z >= minB.z && minA.z <= maxB.z;

    if (overlapX && overlapY && overlapZ) {
        // Find Minimum Translation Vector (MTV)
        float depths[6] = {
            maxB.x - minA.x,
            maxA.x - minB.x,
            maxB.y - minA.y,
            maxA.y - minB.y,
            maxB.z - minA.z,
            maxA.z - minB.z
        };

        int minAxis = 0;
        float minDepth = depths[0];

        for (int i = 1; i < 6; i++) {
            if (depths[i] < minDepth) {
                minDepth = depths[i];
                minAxis = i;
            }
        }

        glm::vec3 normal(0.0f);
        if (minAxis == 0) normal = glm::vec3(1, 0, 0); // Push A Right
        else if (minAxis == 1) normal = glm::vec3(-1, 0, 0); // Push A Left
        else if (minAxis == 2) normal = glm::vec3(0, 1, 0);
        else if (minAxis == 3) normal = glm::vec3(0, -1, 0);
        else if (minAxis == 4) normal = glm::vec3(0, 0, 1);
        else if (minAxis == 5) normal = glm::vec3(0, 0, -1);

        solveContact(a, b, normal, minDepth);
        return true;
    }
    return false;
}

bool PhysicsSystem::checkSphereAABB(SceneNode::Ptr &sphere, SceneNode::Ptr &box) {
    glm::vec3 spherePos = sphere->getPosition();
    glm::vec3 boxPos = box->getPosition();
    glm::vec3 boxHalf = box->physics.halfExtents;

    // Get Closest Point on AABB to Sphere Center
    glm::vec3 delta = spherePos - boxPos;
    glm::vec3 closest = boxPos + glm::clamp(delta, -boxHalf, boxHalf);

    glm::vec3 distanceVec = spherePos - closest;
    float distSq = glm::length2(distanceVec);
    float radius = sphere->physics.radius;

    if (distSq < radius * radius) {
        float dist = std::sqrt(distSq);
        glm::vec3 normal = dist > 0.0001f ? distanceVec / dist : glm::vec3(0, 1, 0);
        float penetration = radius - dist;
        solveContact(sphere, box, normal, penetration);
        return true;
    }

    return false;
}

// Resolves a contact between two colliders using impulse-based dynamics.
// normal: collision normal pointing from B to A (A's separating direction).
// penetration: overlap depth along the normal.
void PhysicsSystem::solveContact(SceneNode::Ptr &a, SceneNode::Ptr &b, const glm::vec3 &normal, float penetration) {
    auto &physA = a->physics;
    auto &physB = b->physics;

    // Static bodies have infinite effective mass (invMass = 0), so impulses do not move them.
    float invMassA = physA.isStatic ? 0.0f : 1.0f / physA.mass;
    float invMassB = physB.isStatic ? 0.0f : 1.0f / physB.mass;
    float totalInvMass = invMassA + invMassB;

    if (totalInvMass <= 0.0f) return;

    // 1. Positional Correction (prevent sinking)
    // We allow a small 'slop' before correcting to avoid micro-jitter at rest.
    // 80% correction per frame leaves a margin to avoid overshooting.
    constexpr float percent = 0.8f;
    constexpr float slop = 0.01f;
    glm::vec3 correction = std::max(penetration - slop, 0.0f) / totalInvMass * percent * normal;

    if (!physA.isStatic) a->setPosition(a->getPosition() + correction * invMassA);
    if (!physB.isStatic) b->setPosition(b->getPosition() - correction * invMassB);

    // 2. Velocity Impulse (coefficient-of-restitution impulse formula)
    // j = -(1 + e) * (v_rel · n) / (1/mA + 1/mB)
    glm::vec3 relVel = physA.velocity - physB.velocity;
    float velAlongNormal = glm::dot(relVel, normal);

    // Do not resolve if velocities are already separating (objects moving apart).
    if (velAlongNormal > 0) return;

    float e = std::min(physA.restitution, physB.restitution);

    // Suppress bounce when the relative approach speed is small to prevent low-energy jitter.
    if (std::abs(velAlongNormal) < 0.5f) {
        e = 0.0f;
    }

    float j = -(1.0f + e) * velAlongNormal;
    j /= totalInvMass;

    glm::vec3 impulse = j * normal;

    if (!physA.isStatic) physA.velocity += impulse * invMassA;
    if (!physB.isStatic) physB.velocity -= impulse * invMassB;
}

// ----------------------------------------------------------------------------
// GPU Physics Implementation
// ----------------------------------------------------------------------------

void PhysicsSystem::createSSBO(const vk::raii::Device &device, const vk::raii::PhysicalDevice &physDevice, size_t size) {
    if (size == 0) return;

    // Destroy old if exists and too small(check handle via *)
    if (*physicsSSBO && currentSSBOSize >= size) return;

    physicsSSBO = nullptr;
    physicsSSBOMemory = nullptr;

    VulkanUtils::createBuffer(device, physDevice, size,
                              vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                              vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                              physicsSSBO, physicsSSBOMemory);

    physicsSSBOMapped = physicsSSBOMemory.mapMemory(0, size);
    currentSSBOSize = size;
}

void PhysicsSystem::updateSSBO(std::vector<SceneNode::Ptr> &nodes) {
    hostPhysicsObjects.clear();

    for (auto &node: nodes) {
        PhysicsObject obj{};
        obj.position = node->getPosition();
        obj.velocity = node->physics.velocity;
        if (node->physics.isStatic) {
            obj.mass = 0.0f; // Infinite mass
            obj.active = node->physics.enabled ? 1 : 0;
        } else {
            obj.mass = node->physics.mass;
            obj.active = node->physics.enabled ? 1 : 0;
        }

        obj.radius = node->physics.radius;
        obj.halfExtents = node->physics.halfExtents;
        obj.type = static_cast<int>(node->physics.colliderType);
        obj.restitution = node->physics.restitution;
        obj.friction = node->physics.friction;

        hostPhysicsObjects.push_back(obj);
    }

    if (hostPhysicsObjects.empty()) return;

    size_t dataSize = hostPhysicsObjects.size() * sizeof(PhysicsObject);
    if (dataSize > currentSSBOSize) {
        LOGE("Physics SSBO overflow! Capacity: %zu bytes, Requested: %zu bytes. Truncating to capacity — some objects will be dropped from simulation.",
             currentSSBOSize, dataSize);
        assert(false && "Physics SSBO overflow: increase SSBO allocation size");
        dataSize = currentSSBOSize;
    }

    // Copy to Mapped Memory
    memcpy(physicsSSBOMapped, hostPhysicsObjects.data(), dataSize);
}


// GPU physics runs two compute dispatches per frame:
//   Stage 0 — Integration: apply gravity + damping, advance positions.
//   Stage 1 — Collision resolution: naïve O(N²) broadphase + impulse response.
// A memory barrier separates the two stages so stage 1 sees the updated positions from stage 0.
// A final Compute→Host barrier ensures the host-coherent SSBO is readable after the queue drains.
void PhysicsSystem::updateGPU(std::vector<SceneNode::Ptr> &nodes, float deltaTime,
                              const vk::raii::CommandBuffer &cmd,
                              const vk::raii::PipelineLayout &layout,
                              const vk::raii::Pipeline &pipeline,
                              const vk::raii::DescriptorSet &descriptorSet) {
    if (!physicsSSBOMapped) return; // Must be initialized externally or via better design

    updateSSBO(nodes); // Serialize SceneNode state → host-coherent SSBO

    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *layout, 0, {*descriptorSet}, nullptr);

    struct Constants {
        float deltaTime;
        uint32_t objectCount;
        float gravity;
        float friction;

        glm::vec4 worldMin;
        glm::vec4 worldMax;
        uint32_t stage;
    } push;

    push.deltaTime = deltaTime;
    push.objectCount = static_cast<uint32_t>(hostPhysicsObjects.size());
    push.gravity = gravity.y;
    push.friction = globalFriction;
    push.worldMin = glm::vec4(worldMin, 0.0f);
    push.worldMax = glm::vec4(worldMax, 0.0f);
    push.stage = 0; // Stage 0: Integration

    cmd.pushConstants<Constants>(*layout, vk::ShaderStageFlagBits::eCompute, 0, push);

    // Group size 64
    uint32_t groupCount = (push.objectCount + 63) / 64;
    cmd.dispatch(groupCount, 1, 1);

    // Intermediate Barrier: Compute Write -> Compute Read (Global)
    vk::MemoryBarrier2 interBarrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite
    };

    vk::DependencyInfo interDepInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &interBarrier
    };

    cmd.pipelineBarrier2(interDepInfo);

    // Stage 1: Collision Resolution
    push.stage = 1;
    cmd.pushConstants<Constants>(*layout, vk::ShaderStageFlagBits::eCompute, 0, push);
    cmd.dispatch(groupCount, 1, 1);

    // 3. Barrier (Memory Barrier to ensure write visibility to Host)
    vk::MemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead
    };

    vk::DependencyInfo depInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier
    };

    cmd.pipelineBarrier2(depInfo);
}

// Reads physics results back from the host-coherent SSBO (after the queue has drained)
// and updates SceneNode positions/velocities to match.
// The SSBO memory is host-coherent, so no explicit cache invalidation is needed.
void PhysicsSystem::syncFromGPU(std::vector<SceneNode::Ptr> &nodes) {
    if (!physicsSSBOMapped || nodes.empty()) return;

    PhysicsObject *gpuObjs = static_cast<PhysicsObject *>(physicsSSBOMapped);

    for (size_t i = 0; i < nodes.size() && i < hostPhysicsObjects.size(); i++) {
        PhysicsObject &obj = gpuObjs[i];
        SceneNode::Ptr &node = nodes[i];

        if (obj.active) {
            node->setPosition(obj.position);
            node->physics.velocity = obj.velocity;
            // Update other props if needed
        }
    }
}
