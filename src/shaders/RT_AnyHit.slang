#include "ShaderCommon.slang"

// Classic ray tracer RayPayload.
struct RayPayload {
    float3 color;
    uint   isShadowed;
};

struct BuiltInTriangleIntersectionAttributes {
    float2 barycentrics;
};

// Set 0 Bindings — binding numbers match the shared RT/PT descriptor set layout.
[[vk::binding(0, 0)]] RaytracingAccelerationStructure topLevelAS;
[[vk::binding(5, 0)]] ByteAddressBuffer globalVertices[];
[[vk::binding(6, 0)]] ByteAddressBuffer globalIndices[];
[[vk::binding(7, 0)]] StructuredBuffer<MaterialData> globalMaterials[];
[[vk::binding(8, 0)]] Sampler2D globalTextures[];

[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

static const uint kVertexStride = 60;

float2 loadTexCoord(ByteAddressBuffer buf, uint idx)
{
    // texCoord is offset 40
    uint base = idx * kVertexStride;
    return asfloat(buf.Load2(base + 40));
}

uint loadIndex(ByteAddressBuffer buf, uint idx)
{
    return buf.Load(idx * 4);
}

[shader("anyhit")]
void main(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {

    uint customIndex     = InstanceID();
    uint modelId         = customIndex >> 14;
    uint primitiveOffset = customIndex & 0x3FFF;
    uint geometryIndex   = GeometryIndex();

    MaterialData mat = globalMaterials[NonUniformResourceIndex(modelId)][primitiveOffset + geometryIndex];

    if (mat.alphaCutoff <= 0.0) {
        return;  // alphaCutoff disabled — accept all hits without a texture lookup.
    }

    uint firstIndex   = mat.firstIndex;
    uint vertexOffset = mat.vertexOffset;
    uint primitiveIndex = PrimitiveIndex();

    ByteAddressBuffer vertBuf = globalVertices[NonUniformResourceIndex(modelId)];
    ByteAddressBuffer idxBuf  = globalIndices[NonUniformResourceIndex(modelId)];

    uint i0 = loadIndex(idxBuf, firstIndex + primitiveIndex * 3 + 0);
    uint i1 = loadIndex(idxBuf, firstIndex + primitiveIndex * 3 + 1);
    uint i2 = loadIndex(idxBuf, firstIndex + primitiveIndex * 3 + 2);

    float2 uv0 = loadTexCoord(vertBuf, vertexOffset + i0);
    float2 uv1 = loadTexCoord(vertBuf, vertexOffset + i1);
    float2 uv2 = loadTexCoord(vertBuf, vertexOffset + i2);

    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
    float2 uv = uv0 * barycentrics.x + uv1 * barycentrics.y + uv2 * barycentrics.z;

    float alpha = mat.baseColorFactor.a;
    if (mat.baseColorIndex >= 0) {
        float sampledAlpha = globalTextures[NonUniformResourceIndex(mat.baseColorIndex + mat.globalTextureOffset)].SampleLevel(uv, 0.0).a;
        alpha *= sampledAlpha;
    }

    if (alpha < mat.alphaCutoff) {
        IgnoreHit();
    }
}
