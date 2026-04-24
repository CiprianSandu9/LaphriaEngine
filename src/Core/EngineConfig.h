#ifndef LAPHRIAENGINE_ENGINECONFIG_H
#define LAPHRIAENGINE_ENGINECONFIG_H

#include <cstdint>

namespace Laphria::EngineConfig
{
constexpr float kDefaultSceneBoundsExtent = 1000.0f;

constexpr uint32_t kMaxPhysicsObjects = 10000;
constexpr uint32_t kMaxTLASInstances = 10000;

constexpr uint32_t kBindlessModelCapacity = 1000;
constexpr uint32_t kDescriptorPoolScale = 1000;

constexpr float kMainCameraFovDegrees = 45.0f;
constexpr float kMainCameraNearPlane = 0.1f;
constexpr float kMainCameraFarPlane = 1000.0f;

constexpr float kPhysicsBroadphaseCellSize = 4.0f;
} // namespace Laphria::EngineConfig

#endif // LAPHRIAENGINE_ENGINECONFIG_H
