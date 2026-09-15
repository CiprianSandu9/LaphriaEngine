#include "SurfelPathTracerCommon.slang"

[[vk::binding(0, 2)]] ConstantBuffer<UniformBuffer> ubo;

[shader("miss")]
void main(inout SurfelPathTracerPayload payload) {
    payload.radiance = clampLuminance(evalSkyColor(normalize(WorldRayDirection()), normalize(-ubo.lightDir.xyz)),
                                      SURFEL_PT_MAX_RADIANCE_LUMINANCE);
    payload.emission = float3(0.0);
    payload.hitKind = 0u;
    payload.hitT = -1.0;
}
