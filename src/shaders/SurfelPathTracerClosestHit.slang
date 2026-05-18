#include "SurfelPathTracerCommon.slang"

struct SurfelPathTracerPayload {
    float3 radiance;
    float hitT;
};

struct BuiltInTriangleIntersectionAttributes {
    float2 barycentrics;
};

[[vk::binding(5, 0)]] ByteAddressBuffer globalVertices[];
[[vk::binding(6, 0)]] ByteAddressBuffer globalIndices[];
[[vk::binding(7, 0)]] StructuredBuffer<MaterialData> globalMaterials[];

[[vk::binding(0, 2)]] ConstantBuffer<UniformBuffer> ubo;

static const uint kVertexStride = 60;

Vertex loadSurfelVertex(ByteAddressBuffer buf, uint idx)
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

uint loadSurfelIndex(ByteAddressBuffer buf, uint idx)
{
    return buf.Load(idx * 4);
}

[shader("closesthit")]
void main(inout SurfelPathTracerPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {
    uint instanceId = InstanceID();
    uint modelId = instanceId >> 14;
    uint primitiveOffset = instanceId & 0x3FFF;
    uint materialIndex = primitiveOffset + GeometryIndex();

    MaterialData mat = globalMaterials[NonUniformResourceIndex(modelId)][materialIndex];
    ByteAddressBuffer vertBuf = globalVertices[NonUniformResourceIndex(modelId)];
    ByteAddressBuffer idxBuf = globalIndices[NonUniformResourceIndex(modelId)];

    uint primitiveIndex = PrimitiveIndex();
    uint i0 = loadSurfelIndex(idxBuf, mat.firstIndex + primitiveIndex * 3 + 0);
    uint i1 = loadSurfelIndex(idxBuf, mat.firstIndex + primitiveIndex * 3 + 1);
    uint i2 = loadSurfelIndex(idxBuf, mat.firstIndex + primitiveIndex * 3 + 2);

    Vertex v0 = loadSurfelVertex(vertBuf, mat.vertexOffset + i0);
    Vertex v1 = loadSurfelVertex(vertBuf, mat.vertexOffset + i1);
    Vertex v2 = loadSurfelVertex(vertBuf, mat.vertexOffset + i2);

    float3 bary = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y,
                         attribs.barycentrics.x,
                         attribs.barycentrics.y);
    float3 objectNormal = v0.normal * bary.x + v1.normal * bary.y + v2.normal * bary.z;
    float3 worldNormal = normalize(mul(objectNormal, (float3x3)WorldToObject3x4()));
    if (dot(worldNormal, WorldRayDirection()) > 0.0) {
        worldNormal = -worldNormal;
    }

    float3 baseColor = max(mat.baseColorFactor.rgb, float3(0.0, 0.0, 0.0));
    float3 sunDir = normalize(-ubo.lightDir.xyz);
    float direct = max(dot(worldNormal, sunDir), 0.0);
    float3 sky = evalSkyColor(worldNormal, sunDir) * 0.35 + evalGroundColor(sunDir) * 0.15;

    payload.radiance = clampLuminance(baseColor * (sky + SUN_RADIANCE * direct), SURFEL_PT_MAX_RADIANCE_LUMINANCE);
    payload.hitT = RayTCurrent();
}
