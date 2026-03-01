#include "Camera.h"
#ifndef GLM_ENABLE_EXPERIMENTAL
#	define GLM_ENABLE_EXPERIMENTAL
#endif
#include "glm/gtx/quaternion.hpp"
#include "glm/gtx/transform.hpp"

glm::mat4 Camera::getViewMatrix() const {
    // Build the camera's world transform (T * R), then invert it to get view space.
    // Inverting is cheaper than computing the inverse analytically here because
    // the rotation matrix is orthogonal (inverse == transpose), but GLM handles that.
    glm::mat4 cameraTranslation = glm::translate(glm::mat4(1.f), position);
    glm::mat4 cameraRotation = getRotationMatrix();
    return glm::inverse(cameraTranslation * cameraRotation);
}

glm::mat4 Camera::getRotationMatrix() const {
    glm::quat pitchRotation = glm::angleAxis(pitch, glm::vec3{1.f, 0.f, 0.f});
    glm::quat yawRotation   = glm::angleAxis(yaw,   glm::vec3{0.f, 1.f, 0.f});

    // Yaw is applied first (world-space Y) then pitch (local X).
    // Multiplying in this order keeps the horizon level when the camera pans.
    return glm::toMat4(yawRotation) * glm::toMat4(pitchRotation);
}

void Camera::processInput(float x, float y, float z) {
    velocity.x = x;
    velocity.y = y;
    velocity.z = z;
}

void Camera::update() {
    glm::mat4 cameraRotation = getRotationMatrix();
    // Transform the local-space velocity into world space and advance the position.
    // The 0.5f factor is a fixed speed scalar; velocity components are set to ±0.1
    // by the InputSystem, giving a final step of ±0.05 units per frame.
    position += glm::vec3(cameraRotation * glm::vec4(velocity * 0.5f, 0.f));
}
