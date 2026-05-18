#include "SurfelPathTracerCommon.slang"

struct SurfelPathTracerPayload {
    float3 radiance;
    float hitT;
};

[[vk::binding(0, 2)]] ConstantBuffer<UniformBuffer> ubo;

[shader("miss")]
void main(inout SurfelPathTracerPayload payload) {
    payload.radiance = clampLuminance(evalSkyColor(normalize(WorldRayDirection()), normalize(-ubo.lightDir.xyz)),
                                      SURFEL_PT_MAX_RADIANCE_LUMINANCE);
    payload.hitT = -1.0;
}
