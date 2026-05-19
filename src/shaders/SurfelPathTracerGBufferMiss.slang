#include "SurfelPathTracerCommon.slang"

struct SurfelPathTracerGBufferPayload {
    float hitT;
    uint modelId;
    uint materialIndex;
    float3 worldNormal;
    float3 baseColor;
};

[shader("miss")]
void main(inout SurfelPathTracerGBufferPayload payload) {
    payload.hitT = -1.0;
    payload.modelId = 0u;
    payload.materialIndex = 0u;
    payload.worldNormal = float3(0.0, 0.0, 0.0);
    payload.baseColor = float3(0.0, 0.0, 0.0);
}
