#include "InputSystem.h"

#include <glm/gtc/constants.hpp>

void InputSystem::init(GLFWwindow *window, Camera &cam, bool &resized) {
    camera              = &cam;
    framebufferResizedPtr = &resized;

    // Initialise camera to a sensible default position
    cam.velocity = glm::vec3(0.f);
    cam.position = glm::vec3(0.0f, 1.f, 3.0f);
    cam.pitch    = 0;
    cam.yaw      = 0;

    // This instance becomes the GLFW user pointer so static callbacks can reach it
    glfwSetWindowUserPointer(window, this);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

void InputSystem::mouseButtonCallback(GLFWwindow *window, int button, int action, int /*mods*/) {
    auto *input = static_cast<InputSystem *>(glfwGetWindowUserPointer(window));

    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            input->rightMouseDown = true;
            glfwGetCursorPos(window, &input->lastMouseX, &input->lastMouseY);
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        } else if (action == GLFW_RELEASE) {
            input->rightMouseDown = false;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
    }
}

void InputSystem::mousePositionCallback(GLFWwindow *window, double xpos, double ypos) {
    auto *input = static_cast<InputSystem *>(glfwGetWindowUserPointer(window));

    if (input->rightMouseDown) {
        double deltaX = xpos - input->lastMouseX;
        double deltaY = ypos - input->lastMouseY;

        constexpr float sensitivity = 0.005f;
        input->camera->yaw   -= static_cast<float>(deltaX) * sensitivity;
        input->camera->pitch -= static_cast<float>(deltaY) * sensitivity;

        constexpr float maxPitch = glm::radians(89.0f);
        input->camera->pitch = glm::clamp(input->camera->pitch, -maxPitch, maxPitch);
    }

    input->lastMouseX = xpos;
    input->lastMouseY = ypos;
}

void InputSystem::keyCallback(GLFWwindow *window, int key, int /*scancode*/, int action, int /*mods*/) {
    auto *input = static_cast<InputSystem *>(glfwGetWindowUserPointer(window));
    if (!input) return;

    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_W || key == GLFW_KEY_UP)         input->camera->velocity.z = -.1f;
        if (key == GLFW_KEY_S || key == GLFW_KEY_DOWN)       input->camera->velocity.z =  .1f;
        if (key == GLFW_KEY_A || key == GLFW_KEY_LEFT)       input->camera->velocity.x = -.1f;
        if (key == GLFW_KEY_D || key == GLFW_KEY_RIGHT)      input->camera->velocity.x =  .1f;
        if (key == GLFW_KEY_Q || key == GLFW_KEY_PAGE_DOWN)  input->camera->velocity.y = -.1f;
        if (key == GLFW_KEY_E || key == GLFW_KEY_PAGE_UP)    input->camera->velocity.y =  .1f;
    }

    if (action == GLFW_RELEASE) {
        if (key == GLFW_KEY_W || key == GLFW_KEY_UP   || key == GLFW_KEY_S || key == GLFW_KEY_DOWN)
            input->camera->velocity.z = 0.f;
        if (key == GLFW_KEY_A || key == GLFW_KEY_LEFT || key == GLFW_KEY_D || key == GLFW_KEY_RIGHT)
            input->camera->velocity.x = 0.f;
        if (key == GLFW_KEY_Q || key == GLFW_KEY_PAGE_DOWN || key == GLFW_KEY_E || key == GLFW_KEY_PAGE_UP)
            input->camera->velocity.y = 0.f;
    }
}

void InputSystem::framebufferResizeCallback(GLFWwindow *window, int /*width*/, int /*height*/) {
    auto *input = static_cast<InputSystem *>(glfwGetWindowUserPointer(window));
    *input->framebufferResizedPtr = true;
}
