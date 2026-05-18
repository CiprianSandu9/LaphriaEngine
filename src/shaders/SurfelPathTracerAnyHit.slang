#include "SurfelPathTracerCommon.slang"

struct SurfelPathTracerPayload {
    float3 radiance;
    float hitT;
};

struct BuiltInTriangleIntersectionAttributes {
    float2 barycentrics;
};

[shader("anyhit")]
void main(inout SurfelPathTracerPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {}
