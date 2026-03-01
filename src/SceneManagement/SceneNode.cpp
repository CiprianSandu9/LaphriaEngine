#include "SceneNode.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

SceneNode::SceneNode(const std::string &name) : name(name) {
    updateLocalTransform();
}

void SceneNode::addChild(const Ptr &child) {
    if (child) {
        child->parent = this;
        children.push_back(child);
    }
}

void SceneNode::removeChild(const Ptr &child) {
    auto it = std::ranges::find(children, child);
    if (it != children.end()) {
        (*it)->parent = nullptr;
        children.erase(it);
    }
}

void SceneNode::updateLocalTransform() {
    glm::mat4 T = glm::translate(glm::mat4(1.0f), position);
    glm::mat4 R = glm::toMat4(rotation);
    glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);
    localTransform = T * R * S;
}

glm::mat4 SceneNode::getWorldTransform() const {
    if (parent) {
        return parent->getWorldTransform() * localTransform;
    }
    return localTransform;
}

SceneNode::Ptr SceneNode::clone() const {
    auto newNode = std::make_shared<SceneNode>(name);
    newNode->position = position;
    newNode->rotation = rotation;
    newNode->eulerRotation = eulerRotation;
    newNode->scale = scale;
    newNode->meshIndices = meshIndices;
    newNode->modelId = modelId;
    newNode->physics = physics;
    newNode->initialPosition = initialPosition;
    newNode->initialRotation = initialRotation;
    newNode->updateLocalTransform();

    for (const auto &child: children) {
        newNode->addChild(child->clone());
    }
    return newNode;
}
