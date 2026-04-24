#include "ShaderCommon.slang"

// Classic ray tracer RayPayload.
struct RayPayload {
    float3 color;
    uint   isShadowed;
};

// Set 1 - global UBO (needed for lightDir -> sun direction).
[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

[shader("miss")]
void main(inout RayPayload payload) {
    if (payload.isShadowed == 1) {
        payload.isShadowed = 0;  // Not occluded - ray reached sky without hitting anything.
        return;
    }

    float3 rayDir = normalize(WorldRayDirection());
    float3 sunDir = normalize(-ubo.lightDir.xyz); // FROM scene TOWARD sun

    payload.color = evalSkyColor(rayDir, sunDir);
}
