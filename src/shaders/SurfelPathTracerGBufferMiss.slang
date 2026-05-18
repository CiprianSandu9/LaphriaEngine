#include "SurfelPathTracerCommon.slang"

struct SurfelPathTracerGBufferPayload {
    float hitT;
    uint materialIndex;
    float3 worldNormal;
};

[shader("miss")]
void main(inout SurfelPathTracerGBufferPayload payload) {
    payload.hitT = -1.0;
    payload.materialIndex = SURFEL_PT_INVALID_INDEX;
    payload.worldNormal = float3(0.0, 1.0, 0.0);
}
