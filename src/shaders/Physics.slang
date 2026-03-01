#include "../Physics/PhysicsDefines.h"

// Define the structured buffer
[[vk::binding(0, 0)]]
RWStructuredBuffer<PhysicsObject> objects;

struct Constants {
    float deltaTime;
    uint objectCount;
    float gravity; // Just Y component or simplified
    float friction;
    float4 worldMin;
    float4 worldMax;
    uint stage; // 0 = Integration, 1 = Collision
};

[[vk::push_constant]]
Constants push;

[shader("compute")]
[numthreads(64, 1, 1)]
void physicsMain(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint idx = dispatchThreadID.x;
    if (idx >= push.objectCount) return;

    PhysicsObject obj = objects[idx];
    
    // Skip inactive
    if (obj.active == 0) return;
    
    // ------------------------------------------------------------------------
    // Stage 0: Integration & Bounds
    // ------------------------------------------------------------------------
    if (push.stage == 0) {
        if (obj.mass > 0.0001) {
            float3 gravityVec = float3(0, push.gravity, 0);
            obj.velocity += gravityVec * push.deltaTime;
            obj.velocity *= (1.0 - push.friction * push.deltaTime);
            
            float3 oldPos = obj.position;
            obj.position += obj.velocity * push.deltaTime;
        }

        // Container Bounds
        float3 minBound = push.worldMin.xyz;
        float3 maxBound = push.worldMax.xyz;

        float3 offset = float3(0);
        if (obj.type == 0) { // Sphere
             offset = float3(obj.radius);
        } else { // AABB (1) or Cylinder (2)
             offset = obj.halfExtents;
        }
        
        [unroll]
        for (int i = 0; i < 3; i++) {
            if (obj.position[i] - offset[i] < minBound[i]) {
                obj.position[i] = minBound[i] + offset[i];
                if (abs(obj.velocity[i]) < 0.5) {
                    obj.velocity[i] = 0.0;
                } else {
                    obj.velocity[i] = -obj.velocity[i] * obj.restitution;
                }
            } else if (obj.position[i] + offset[i] > maxBound[i]) {
                obj.position[i] = maxBound[i] - offset[i];
                if (abs(obj.velocity[i]) < 0.5) {
                    obj.velocity[i] = 0.0;
                } else {
                    obj.velocity[i] = -obj.velocity[i] * obj.restitution;
                }
            }
        }
    }
    
    // ------------------------------------------------------------------------
    // Stage 1: Collision Resolution
    // ------------------------------------------------------------------------
    else if (push.stage == 1) {
        // Naive Collision (O(N) per thread -> O(N^2) total)
        for (uint j = 0; j < push.objectCount; j++) {
            if (idx == j) continue;
            
            PhysicsObject other = objects[j];
            if (other.active == 0) continue;
            
            bool collision = false;
            float3 normal = float3(0, 1, 0);
            float penetration = 0;
            
            // Mapping Cylinder(2) to AABB(1) for collisions
            uint typeA = obj.type == 2 ? 1 : obj.type;
            uint typeB = other.type == 2 ? 1 : other.type;

            // Sphere-Sphere
            if (typeA == 0 && typeB == 0) {
                float3 delta = obj.position - other.position;
                float distSq = dot(delta, delta);
                float rSum = obj.radius + other.radius;
                if (distSq < rSum * rSum) {
                    float dist = sqrt(distSq);
                    normal = dist > 0.0001 ? delta / dist : float3(0,1,0);
                    penetration = rSum - dist;
                    collision = true;
                }
            }
            // AABB-AABB
            else if (typeA == 1 && typeB == 1) {
                 float3 dist = abs(obj.position - other.position);
                 float3 sumExt = obj.halfExtents + other.halfExtents;
                 float3 pen = sumExt - dist;

                 // Check for overlap in all axes
                 bool overlap = (pen.x > 0) && (pen.y > 0) && (pen.z > 0);
                                
                 if (overlap) {
                     // Find Minimum Translation Vector (MTV)
                     if (pen.x < pen.y && pen.x < pen.z) {
                         penetration = pen.x;
                         normal = float3(obj.position.x > other.position.x ? 1.0 : -1.0, 0.0, 0.0);
                     } else if (pen.y < pen.z) {
                         penetration = pen.y;
                         normal = float3(0.0, obj.position.y > other.position.y ? 1.0 : -1.0, 0.0);
                     } else {
                         penetration = pen.z;
                         normal = float3(0.0, 0.0, obj.position.z > other.position.z ? 1.0 : -1.0);
                     }
                     collision = true;
                 }
            }
            // Sphere-AABB
            else if (typeA == 0 && typeB == 1) {
                float3 spherePos = obj.position;
                float3 boxPos = other.position;
                float3 boxHalf = other.halfExtents;
                
                float3 delta = spherePos - boxPos;
                float3 closest = boxPos + clamp(delta, -boxHalf, boxHalf);
                
                float3 distVec = spherePos - closest;
                float distSq = dot(distVec, distVec);
                
                if (distSq < obj.radius * obj.radius) {
                    float dist = sqrt(distSq);
                    normal = dist > 0.0001 ? distVec / dist : float3(0,1,0);
                    penetration = obj.radius - dist;
                    collision = true;
                }
            }
            // AABB-Sphere
            else if (typeA == 1 && typeB == 0) {
                float3 boxPos = obj.position;
                float3 spherePos = other.position;
                float3 boxHalf = obj.halfExtents;
                
                float3 delta = spherePos - boxPos;
                float3 closest = boxPos + clamp(delta, -boxHalf, boxHalf);
                
                float3 distVec = closest - spherePos; // Normal points OUT from sphere towards box (push box away)
                
                float3 d = spherePos - boxPos; // Center to Center
                // Closest point on Box to Sphere Center
                float3 c = boxPos + clamp(d, -boxHalf, boxHalf);
                
                float3 diff = spherePos - c; // Vector from Surface to Center of Sphere
                float distSq = dot(diff, diff);
                
                if (distSq < other.radius * other.radius) {
                    float dist = sqrt(distSq);
                    // Normal for SPHERE would be diff/dist (push sphere away from box).
                    // So Normal for BOX should be -(diff/dist).
                    normal = dist > 0.0001 ? -(diff / dist) : float3(0,1,0); 
                    penetration = other.radius - dist;
                    collision = true;
                }
            }
            
            if (collision) {
                 // Resolve
                 float invMassA = (obj.mass > 0.0001) ? 1.0 / obj.mass : 0.0;
                 float invMassB = (other.mass > 0.0001) ? 1.0 / other.mass : 0.0;
                 float totalInvMass = invMassA + invMassB;
                 
                 if (totalInvMass > 0.00001) {
                     // Positional correction
                     obj.position += normal * (penetration * (invMassA / totalInvMass));
                     
                     // Impulse
                     float3 relVel = obj.velocity - other.velocity;
                     float vn = dot(relVel, normal);
                     
                     // If moving towards each other
                     if (vn < 0) {
                         float e = min(obj.restitution, other.restitution);
                         
                         // Fix jitter: if impact velocity is low, no bounce
                         if (abs(vn) < 0.5) {
                             e = 0.0;
                         }
                         
                         float jImp = -(1.0 + e) * vn / totalInvMass;
                         
                         // Apply impulse
                         obj.velocity += normal * jImp * invMassA;
                     }
                 }
            }
        }
    }

    // Write back
    objects[idx] = obj;
}
