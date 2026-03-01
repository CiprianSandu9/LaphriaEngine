#ifndef LAPHRIAENGINE_CAMERA_H
#define LAPHRIAENGINE_CAMERA_H
#ifndef GLM_FORCE_RADIANS
#	define GLM_FORCE_RADIANS
#endif
#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE
#	define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif
#ifndef GLM_ENABLE_EXPERIMENTAL
#	define GLM_ENABLE_EXPERIMENTAL
#endif
#ifndef GLM_FORCE_CXX11
#	define GLM_FORCE_CXX11
#endif

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

// First-person free-flight camera using yaw/pitch Euler angles.
// Rotation convention: yaw rotates around world Y, pitch around local X (clamped to ±89°).
class Camera
{
  public:
	glm::vec3 velocity;        // Local-space movement direction set by InputSystem each frame
	glm::vec3 position;
	float     pitch{0.f};        // Vertical rotation (radians); negative looks up
	float     yaw{0.f};          // Horizontal rotation (radians)

	// Returns the view matrix (world → camera space).
	glm::mat4 getViewMatrix() const;

	// Returns just the rotation component (no translation), useful for skybox rendering.
	glm::mat4 getRotationMatrix() const;

	// Sets the camera's local-space velocity (called by InputSystem).
	void processInput(float x, float y, float z);

	// Advances the camera position by velocity * fixedScale along the current facing direction.
	void update();
};

#endif        // LAPHRIAENGINE_CAMERA_H
