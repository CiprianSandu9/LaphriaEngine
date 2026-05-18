#include "SurfelPathTracerCommon.slang"

struct SurfelPathTracerGBufferPayload {
    float hitT;
    uint materialIndex;
    float3 worldNormal;
};

struct BuiltInTriangleIntersectionAttributes {
    float2 barycentrics;
};

[shader("closesthit")]
void main(inout SurfelPathTracerGBufferPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {
    payload.hitT = RayTCurrent();
    payload.materialIndex = 0u;
    payload.worldNormal = normalize(WorldRayDirection() * -1.0);
}
