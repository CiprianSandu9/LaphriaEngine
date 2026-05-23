#include "SurfelPathTracerCommon.slang"

struct SurfelPathTracerGBufferPayload {
    float hitT;
    uint modelId;
    uint materialIndex;
    float3 worldNormal;
    float3 baseColor;
    float roughness;
    float metallic;
    float dielectricSpec;
    float ao;
    float3 emissive;
};

struct BuiltInTriangleIntersectionAttributes {
    float2 barycentrics;
};

[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(5, 0)]] ByteAddressBuffer globalVertices[];
[[vk::binding(6, 0)]] ByteAddressBuffer globalIndices[];
[[vk::binding(7, 0)]] StructuredBuffer<MaterialData> globalMaterials[];
[[vk::binding(8, 0)]] Sampler2D globalTextures[];

static const uint kVertexStride = 60;

float2 loadGBufferTexCoord(ByteAddressBuffer buf, uint idx)
{
    uint base = idx * kVertexStride;
    return asfloat(buf.Load2(base + 40));
}

uint loadGBufferIndex(ByteAddressBuffer buf, uint idx)
{
    return buf.Load(idx * 4);
}

[shader("anyhit")]
void main(inout SurfelPathTracerGBufferPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {
    uint instanceId = InstanceID();
    uint modelId = instanceId >> 14;
    uint primitiveOffset = instanceId & 0x3FFF;
    uint geometryIndex = GeometryIndex();

    MaterialData mat = globalMaterials[NonUniformResourceIndex(modelId)][primitiveOffset + geometryIndex];
    if (mat.alphaCutoff <= 0.0) {
        return;
    }

    uint primitiveIndex = PrimitiveIndex();
    ByteAddressBuffer vertBuf = globalVertices[NonUniformResourceIndex(modelId)];
    ByteAddressBuffer idxBuf = globalIndices[NonUniformResourceIndex(modelId)];

    uint i0 = loadGBufferIndex(idxBuf, mat.firstIndex + primitiveIndex * 3 + 0);
    uint i1 = loadGBufferIndex(idxBuf, mat.firstIndex + primitiveIndex * 3 + 1);
    uint i2 = loadGBufferIndex(idxBuf, mat.firstIndex + primitiveIndex * 3 + 2);

    float2 uv0 = loadGBufferTexCoord(vertBuf, mat.vertexOffset + i0);
    float2 uv1 = loadGBufferTexCoord(vertBuf, mat.vertexOffset + i1);
    float2 uv2 = loadGBufferTexCoord(vertBuf, mat.vertexOffset + i2);

    float3 bary = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y,
                         attribs.barycentrics.x,
                         attribs.barycentrics.y);
    float2 uv = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;

    float alpha = mat.baseColorFactor.a;
    if (mat.baseColorIndex >= 0) {
        alpha *= globalTextures[NonUniformResourceIndex(mat.baseColorIndex + mat.globalTextureOffset)].SampleLevel(uv, 0.0).a;
    }

    if (alpha < mat.alphaCutoff) {
        IgnoreHit();
    }
}
