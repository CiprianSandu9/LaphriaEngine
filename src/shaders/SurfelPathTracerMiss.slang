#include "SurfelPathTracerCommon.slang"

struct SurfelPathTracerPayload {
    float3 radiance;
    float hitT;
};

[[vk::binding(0, 2)]] ConstantBuffer<UniformBuffer> ubo;

[shader("miss")]
void main(inout SurfelPathTracerPayload payload) {
    payload.radiance = evalSkyColor(normalize(WorldRayDirection()), normalize(-ubo.lightDir.xyz));
    payload.hitT = -1.0;
}
