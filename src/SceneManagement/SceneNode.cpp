#include "SceneNode.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <algorithm>
#include <atomic>
#include <sstream>

namespace
{
std::string makeNodeId()
{
    static std::atomic<uint64_t> nextId{1};
    const uint64_t id = nextId.fetch_add(1, std::memory_order_relaxed);
    std::ostringstream oss;
    oss << "node_" << id;
    return oss.str();
}
} // namespace

SceneNode::SceneNode(const std::string &name) : name(name) {
    stableId = makeNodeId();
    updateLocalTransform();
}

void SceneNode::addChild(const Ptr &child) {
    if (child) {
        child->parent = this;
        child->markWorldTransformDirtyRecursive();
        children.push_back(child);
    }
}

void SceneNode::removeChild(const Ptr &child) {
    auto it = std::ranges::find(children, child);
    if (it != children.end()) {
        (*it)->parent = nullptr;
        (*it)->markWorldTransformDirtyRecursive();
        children.erase(it);
    }
}

void SceneNode::updateLocalTransform() {
    glm::mat4 T = glm::translate(glm::mat4(1.0f), position);
    glm::mat4 R = glm::toMat4(rotation);
    glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);
    localTransform = T * R * S;
    markWorldTransformDirtyRecursive();
}

void SceneNode::markWorldTransformDirtyRecursive() const {
    worldTransformDirty = true;
    for (const auto &child : children) {
        if (child) {
            child->markWorldTransformDirtyRecursive();
        }
    }
}

const glm::mat4 &SceneNode::getWorldTransform() const {
    if (worldTransformDirty) {
        if (parent) {
            worldTransform = parent->getWorldTransform() * localTransform;
        } else {
            worldTransform = localTransform;
        }
        worldTransformDirty = false;
    }
    return worldTransform;
}

void SceneNode::updateWorldTransformRecursive(const glm::mat4 &parentWorld, bool parentDirty) const {
    const bool mustRecompute = parentDirty || worldTransformDirty;
    if (mustRecompute) {
        worldTransform = parentWorld * localTransform;
        worldTransformDirty = false;
    }

    for (const auto &child : children) {
        if (child) {
            child->updateWorldTransformRecursive(worldTransform, mustRecompute);
        }
    }
}

SceneNode::Ptr SceneNode::clone() const {
    auto newNode = std::make_shared<SceneNode>(name);
    newNode->position = position;
    newNode->rotation = rotation;
    newNode->eulerRotation = eulerRotation;
    newNode->scale = scale;
    newNode->meshIndices = meshIndices;
    newNode->modelId = modelId;
    newNode->sourceNodeIndex = sourceNodeIndex;
    newNode->physics = physics;
    newNode->assetRef = assetRef;
    newNode->animation = animation;
    newNode->initialPosition = initialPosition;
    newNode->initialRotation = initialRotation;
    newNode->updateLocalTransform();

    for (const auto &child: children) {
        newNode->addChild(child->clone());
    }
    return newNode;
}
