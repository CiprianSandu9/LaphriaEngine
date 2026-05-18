#include "SurfelPathTracerCommon.slang"

struct SurfelPathTracerPayload {
    float3 radiance;
    float hitT;
};

[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;

[shader("raygeneration")]
void main() {}
