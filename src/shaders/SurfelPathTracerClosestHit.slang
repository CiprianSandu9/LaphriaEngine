#include "SurfelPathTracerCommon.slang"

struct SurfelPathTracerPayload {
    float3 radiance;
    float hitT;
};

struct BuiltInTriangleIntersectionAttributes {
    float2 barycentrics;
};

[shader("closesthit")]
void main(inout SurfelPathTracerPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {
    payload.radiance = float3(0.0, 0.0, 0.0);
    payload.hitT = RayTCurrent();
}
