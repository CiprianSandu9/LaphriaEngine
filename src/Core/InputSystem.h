#ifndef LAPHRIAENGINE_INPUTSYSTEM_H
#define LAPHRIAENGINE_INPUTSYSTEM_H

#include "EngineAuxiliary.h"
#include "Camera.h"

// Owns GLFW input callbacks and raw mouse/keyboard state.
// Becomes the GLFW window user-pointer so static callbacks resolve to this instance.
class InputSystem {
public:
    // Call after the GLFW window has been created.
    // Sets the window user pointer to this instance and registers all callbacks.
    void init(GLFWwindow *window, Camera &camera, bool &framebufferResized);

    double lastMouseX{0.0};
    double lastMouseY{0.0};
    bool   rightMouseDown{false};

private:
    Camera *camera{nullptr};             // non-owning
    bool   *framebufferResizedPtr{nullptr}; // non-owning

    static void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
    static void mousePositionCallback(GLFWwindow *window, double xpos, double ypos);
    static void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
    static void framebufferResizeCallback(GLFWwindow *window, int width, int height);
};

#endif // LAPHRIAENGINE_INPUTSYSTEM_H
