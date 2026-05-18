#include "SurfelPathTracerCommon.slang"

struct SurfelPathTracerGBufferPayload {
    float hitT;
    uint materialIndex;
    float3 worldNormal;
};

struct BuiltInTriangleIntersectionAttributes {
    float2 barycentrics;
};

[shader("anyhit")]
void main(inout SurfelPathTracerGBufferPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {}
