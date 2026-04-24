#include "ShaderCommon.slang"

// Path tracer RayPayload - surface properties returned by ClosestHit to Raygen.
struct RayPayload {
    float3 albedo;
    float3 emission;
    float3 worldNormal;
    float3 hitPos;
    float  roughness;
    float  metallic;
    float3 F0;
    float  ao;
    float  hitT;
    uint   instanceID;
};

// Set 1 - global UBO (needed for lightDir -> sun direction).
[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

[shader("miss")]
void main(inout RayPayload payload) {
    float3 rayDir = normalize(WorldRayDirection());
    float3 sunDir = normalize(-ubo.lightDir.xyz); // FROM scene TOWARD sun

    payload.emission    = evalSkyColor(rayDir, sunDir);
    payload.hitT        = -1.0;   // Negative sentinel: ray escaped to sky, no surface hit.
    payload.albedo      = float3(0.0, 0.0, 0.0);
    payload.worldNormal = -rayDir;
    payload.hitPos      = WorldRayOrigin() + WorldRayDirection() * 10000.0;
    payload.roughness   = 1.0;
    payload.metallic    = 0.0;
    payload.F0          = float3(0.0, 0.0, 0.0);
    payload.ao          = 1.0;
    payload.instanceID  = 0;
}
