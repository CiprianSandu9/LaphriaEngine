#include "SurfelPathTracerCommon.slang"

struct SurfelPathTracerGBufferPayload {
    float hitT;
    uint modelId;
    uint materialIndex;
    float3 worldNormal;
    float3 baseColor;
};

struct BuiltInTriangleIntersectionAttributes {
    float2 barycentrics;
};

[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(5, 0)]] ByteAddressBuffer globalVertices[];
[[vk::binding(6, 0)]] ByteAddressBuffer globalIndices[];
[[vk::binding(7, 0)]] StructuredBuffer<MaterialData> globalMaterials[];
[[vk::binding(8, 0)]] Sampler2D globalTextures[];

[[vk::binding(0, 2)]] ConstantBuffer<UniformBuffer> ubo;

static const uint kVertexStride = 60;

struct GBufferSurface {
    float3 worldNormal;
    uint modelId;
    uint materialIndex;
    MaterialData material;
    float3 baseColor;
};

Vertex loadGBufferVertex(ByteAddressBuffer buf, uint idx)
{
    uint base = idx * kVertexStride;
    Vertex v;
    v.pos = asfloat(buf.Load3(base));
    v.normal = asfloat(buf.Load3(base + 12));
    v.tangent = asfloat(buf.Load4(base + 24));
    v.texCoord = asfloat(buf.Load2(base + 40));
    v.color = asfloat(buf.Load3(base + 48));
    return v;
}

uint loadGBufferIndex(ByteAddressBuffer buf, uint idx)
{
    return buf.Load(idx * 4);
}

GBufferSurface decodeGBufferSurface(uint instanceId, uint geometryIndex, uint primitiveIndex, float2 barycentrics)
{
    uint modelId = instanceId >> 14;
    uint primitiveOffset = instanceId & 0x3FFF;
    uint materialIndex = primitiveOffset + geometryIndex;

    MaterialData mat = globalMaterials[NonUniformResourceIndex(modelId)][materialIndex];

    ByteAddressBuffer vertBuf = globalVertices[NonUniformResourceIndex(modelId)];
    ByteAddressBuffer idxBuf = globalIndices[NonUniformResourceIndex(modelId)];

    uint i0 = loadGBufferIndex(idxBuf, mat.firstIndex + primitiveIndex * 3 + 0);
    uint i1 = loadGBufferIndex(idxBuf, mat.firstIndex + primitiveIndex * 3 + 1);
    uint i2 = loadGBufferIndex(idxBuf, mat.firstIndex + primitiveIndex * 3 + 2);

    Vertex v0 = loadGBufferVertex(vertBuf, mat.vertexOffset + i0);
    Vertex v1 = loadGBufferVertex(vertBuf, mat.vertexOffset + i1);
    Vertex v2 = loadGBufferVertex(vertBuf, mat.vertexOffset + i2);

    float3 bary = float3(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);
    float3 objectNormal = v0.normal * bary.x + v1.normal * bary.y + v2.normal * bary.z;
    float2 uv = v0.texCoord * bary.x + v1.texCoord * bary.y + v2.texCoord * bary.z;
    float3 vertexColor = v0.color * bary.x + v1.color * bary.y + v2.color * bary.z;
    float3 worldNormal = normalize(mul(objectNormal, (float3x3)WorldToObject3x4()));
    if (dot(worldNormal, WorldRayDirection()) > 0.0) {
        worldNormal = -worldNormal;
    }

    float4 baseColor = mat.baseColorFactor;
    if (mat.baseColorIndex >= 0) {
        float4 sampled = globalTextures[NonUniformResourceIndex(mat.baseColorIndex + mat.globalTextureOffset)].SampleLevel(uv, 0.0);
        baseColor.rgb *= decodeColorSample(sampled.rgb, ubo.textureColorSpaceModel);
        baseColor.a *= sampled.a;
    }
    baseColor.rgb *= vertexColor;

    GBufferSurface surface;
    surface.worldNormal = worldNormal;
    surface.modelId = modelId;
    surface.materialIndex = materialIndex;
    surface.material = mat;
    surface.baseColor = max(baseColor.rgb, float3(0.0));
    return surface;
}

[shader("closesthit")]
void main(inout SurfelPathTracerGBufferPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {
    GBufferSurface surface = decodeGBufferSurface(InstanceID(), GeometryIndex(), PrimitiveIndex(), attribs.barycentrics);
    payload.hitT = RayTCurrent();
    payload.modelId = surface.modelId;
    payload.materialIndex = surface.materialIndex;
    payload.worldNormal = surface.worldNormal;
    payload.baseColor = surface.baseColor;
}
