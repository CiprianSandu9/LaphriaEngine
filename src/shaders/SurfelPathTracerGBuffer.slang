#include "SurfelPathTracerCommon.slang"

struct SurfelPathTracerGBufferPayload {
    float hitT;
    uint materialIndex;
    float3 worldNormal;
};

[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;

[shader("raygeneration")]
void main() {}
