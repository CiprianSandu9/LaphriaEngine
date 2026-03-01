#ifndef LAPHRIAENGINE_PHYSICSDEFINES_H
#define LAPHRIAENGINE_PHYSICSDEFINES_H

#ifdef __cplusplus
#include <glm/glm.hpp>
using vec3 = glm::vec3;
#else
// Slang/GLSL
typedef float3 vec3;
#endif

// Shared physics object layout used by both the C++ CPU simulation and the Slang compute shader.
// Field layout follows std430 packing (vec3 = 12 bytes, padded to 16 by the trailing float):
//   [  0] vec3 position  + float radius      = 16 bytes
//   [ 16] vec3 velocity  + float mass        = 16 bytes
//   [ 32] vec3 halfExtents + int type        = 16 bytes
//   [ 48] int active + float restitution + float friction + float padding = 16 bytes
// Total: 64 bytes per object.
//
// Collision convention:
//   type 0 (Sphere): uses position + radius.
//   type 1 (AABB):   uses position ± halfExtents.
//   mass = 0 means infinite mass (static body).
struct PhysicsObject {
    vec3 position;
    float radius;       // Sphere collider radius (unused for AABB)

    vec3 velocity;
    float mass;         // 0 = static / infinite mass

    vec3 halfExtents;   // AABB half-extents (unused for Sphere)
    int type;           // 0=Sphere, 1=AABB

    int active;         // 1=participates in simulation, 0=skipped
    float restitution;  // Coefficient of restitution (bounciness) in [0, 1]
    float friction;     // Velocity damping coefficient per second
    float padding;      // Explicit padding to reach 64-byte alignment
};

#endif // LAPHRIAENGINE_PHYSICSDEFINES_H
