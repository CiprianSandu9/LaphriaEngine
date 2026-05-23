#include "SurfelPathTracerCommon.slang"

struct BuiltInTriangleIntersectionAttributes {
    float2 barycentrics;
};

[[vk::binding(5, 0)]] ByteAddressBuffer globalVertices[];
[[vk::binding(6, 0)]] ByteAddressBuffer globalIndices[];
[[vk::binding(7, 0)]] StructuredBuffer<MaterialData> globalMaterials[];
[[vk::binding(8, 0)]] Sampler2D globalTextures[];

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

    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y,
                                 attribs.barycentrics.x,
                                 attribs.barycentrics.y);
    float2 uv = v0.texCoord * barycentrics.x + v1.texCoord * barycentrics.y + v2.texCoord * barycentrics.z;
    float3 vertexColor = v0.color * barycentrics.x + v1.color * barycentrics.y + v2.color * barycentrics.z;
    float3 geomNormal = v0.normal * barycentrics.x + v1.normal * barycentrics.y + v2.normal * barycentrics.z;
    float3 N = normalize(mul(geomNormal, (float3x3)WorldToObject3x4()));
    if (dot(N, WorldRayDirection()) > 0.0) {
        N = -N;
    }

    if (mat.normalIndex >= 0) {
        float4 interpolatedTangent = v0.tangent * barycentrics.x + v1.tangent * barycentrics.y + v2.tangent * barycentrics.z;
        float3 T = mul((float3x3)ObjectToWorld3x4(), interpolatedTangent.xyz);
        float tangentW = sign(interpolatedTangent.w);
        float tangentLengthSq = dot(T, T);
        if (tangentLengthSq > 0.0001) {
            T = T * rsqrt(tangentLengthSq);
            T = T - N * dot(N, T);
            tangentLengthSq = dot(T, T);
            if (tangentLengthSq > 0.0001) {
                T = T * rsqrt(tangentLengthSq);
                float3 B = cross(N, T) * tangentW;
                float3 sampledNormal = globalTextures[NonUniformResourceIndex(mat.normalIndex + mat.globalTextureOffset)].SampleLevel(uv, 0.0).rgb;
                float3 tangentNormal = sampledNormal * 2.0 - 1.0;
                tangentNormal.xy *= mat.normalScale;
                N = normalize(mul(normalize(tangentNormal), float3x3(T, B, N)));
            }
        }
    }

    float4 baseColor = mat.baseColorFactor;
    if (mat.baseColorIndex >= 0) {
        float4 sampled = globalTextures[NonUniformResourceIndex(mat.baseColorIndex + mat.globalTextureOffset)].SampleLevel(uv, 0.0);
        baseColor.rgb *= decodeColorSample(sampled.rgb, ubo.textureColorSpaceModel);
        baseColor.a *= sampled.a;
    }
    baseColor.rgb *= vertexColor;

    float metallic = mat.metallicFactor;
    float roughness = mat.roughnessFactor;
    if (mat.metallicRoughnessIndex >= 0) {
        float4 mr = globalTextures[NonUniformResourceIndex(mat.metallicRoughnessIndex + mat.globalTextureOffset)].SampleLevel(uv, 0.0);
        roughness *= mr.g;
        metallic *= mr.b;
    }
    roughness = clamp(roughness, MIN_ROUGHNESS, 1.0);

    float3 emissive = mat.emissiveFactor;
    if (mat.emissiveIndex >= 0) {
        emissive *= decodeColorSample(globalTextures[NonUniformResourceIndex(mat.emissiveIndex + mat.globalTextureOffset)].SampleLevel(uv, 0.0).rgb,
                                      ubo.textureColorSpaceModel);
    }

    float ao = 1.0;
    if (mat.occlusionIndex >= 0) {
        float aoSample = globalTextures[NonUniformResourceIndex(mat.occlusionIndex + mat.globalTextureOffset)].SampleLevel(uv, 0.0).r;
        ao = 1.0 + mat.occlusionStrength * (aoSample - 1.0);
    }
    ao = saturate(ao);

    float dielectricSpec = mat.specularFactor;
    if (mat.specularTextureIndex >= 0) {
        dielectricSpec *= globalTextures[NonUniformResourceIndex(mat.specularTextureIndex + mat.globalTextureOffset)].SampleLevel(uv, 0.0).a;
    }

    payload.radiance = float3(0.0);
    payload.hitT = RayTCurrent();
    payload.hitPosition = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    payload.hitNormal = N;
    payload.baseColor = max(baseColor.rgb, float3(0.0));
    payload.emission = max(emissive, float3(0.0));
    payload.ao = ao;
    payload.metallic = saturate(metallic);
    payload.roughness = roughness;
    payload.f0 = lerp(float3(0.08 * dielectricSpec), payload.baseColor, payload.metallic);
    payload.hitKind = 1u;
    payload.instanceID = InstanceID();
}
