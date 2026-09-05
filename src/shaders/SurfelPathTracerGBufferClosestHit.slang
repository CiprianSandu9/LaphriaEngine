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
    // 1 for alpha-tested materials (foliage, curtains): the surfel cache treats
    // those as unoriented thin geometry, see Evaluate.
    float alphaTested;
    uint sourceNodeId;
    uint sourceInstanceId;
    uint sourceInstanceCustomIndex;
    uint sourcePrimitiveIndex;
    float2 sourceBarycentrics;
    float3 sourceObjectPosition;
    float3 sourceObjectNormal;
};

struct BuiltInTriangleIntersectionAttributes {
    float2 barycentrics;
};

[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(5, 0)]] ByteAddressBuffer globalVertices[];
[[vk::binding(6, 0)]] ByteAddressBuffer globalIndices[];
[[vk::binding(7, 0)]] StructuredBuffer<MaterialData> globalMaterials[];
[[vk::binding(8, 0)]] Sampler2D globalTextures[];
[[vk::binding(30, 1)]] StructuredBuffer<SurfelPathTracerSourceInstance> sourceInstanceBuffer;

[[vk::binding(0, 2)]] ConstantBuffer<UniformBuffer> ubo;

static const uint kVertexStride = 60;

struct GBufferSurface {
    float3 worldNormal;
    uint modelId;
    uint materialIndex;
    MaterialData material;
    float3 baseColor;
    float roughness;
    float metallic;
    float dielectricSpec;
    float ao;
    float3 emissive;
    float alphaTested;
    uint sourceNodeId;
    uint sourceInstanceId;
    uint sourceInstanceCustomIndex;
    uint sourcePrimitiveIndex;
    float2 sourceBarycentrics;
    float3 sourceObjectPosition;
    float3 sourceObjectNormal;
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
    uint sourceInstanceId = InstanceIndex();
    uint sourceInstanceCustomIndex = instanceId;
    SurfelPathTracerSourceInstance sourceInstance = sourceInstanceBuffer[sourceInstanceId];
    uint modelId = sourceInstance.modelId;
    uint primitiveOffset = sourceInstance.primitiveOffset;
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
    float3 objectPosition = v0.pos * bary.x + v1.pos * bary.y + v2.pos * bary.z;
    float3 objectNormal = normalize(v0.normal * bary.x + v1.normal * bary.y + v2.normal * bary.z);
    float2 uv = v0.texCoord * bary.x + v1.texCoord * bary.y + v2.texCoord * bary.z;
    float3 vertexColor = v0.color * bary.x + v1.color * bary.y + v2.color * bary.z;
    float3 geomNormal = v0.normal * bary.x + v1.normal * bary.y + v2.normal * bary.z;
    float3 worldNormal = normalize(mul(geomNormal, (float3x3)WorldToObject3x4()));
    if (dot(worldNormal, WorldRayDirection()) > 0.0) {
        worldNormal = -worldNormal;
    }

    if (mat.normalIndex >= 0) {
        float4 interpolatedTangent = v0.tangent * bary.x + v1.tangent * bary.y + v2.tangent * bary.z;
        float3 T = mul((float3x3)ObjectToWorld3x4(), interpolatedTangent.xyz);
        float tangentW = sign(interpolatedTangent.w);
        float tangentLengthSq = dot(T, T);
        if (tangentLengthSq > 0.0001) {
            T = T * rsqrt(tangentLengthSq);
            T = T - worldNormal * dot(worldNormal, T);
            tangentLengthSq = dot(T, T);
            if (tangentLengthSq > 0.0001) {
                T = T * rsqrt(tangentLengthSq);
                float3 B = cross(worldNormal, T) * tangentW;
                float3 sampledNormal = globalTextures[NonUniformResourceIndex(mat.normalIndex + mat.globalTextureOffset)].SampleLevel(uv, 0.0).rgb;
                float3 tangentNormal = sampledNormal * 2.0 - 1.0;
                tangentNormal.xy *= mat.normalScale;
                worldNormal = normalize(mul(normalize(tangentNormal), float3x3(T, B, worldNormal)));
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
    float3 f0 = lerp(float3(0.08 * dielectricSpec), baseColor.rgb, metallic);

    GBufferSurface surface;
    surface.worldNormal = worldNormal;
    surface.modelId = modelId;
    surface.materialIndex = materialIndex;
    surface.material = mat;
    surface.baseColor = max(baseColor.rgb, float3(0.0));
    surface.roughness = roughness;
    surface.metallic = saturate(metallic);
    surface.dielectricSpec = max(dielectricSpec + f0.x * 0.0, 0.0);
    surface.ao = ao;
    surface.emissive = max(emissive, float3(0.0));
    surface.alphaTested = mat.alphaCutoff > 0.0 ? 1.0 : 0.0;
    surface.sourceInstanceId = sourceInstanceId;
    surface.sourceNodeId = sourceInstance.sourceNodeId;
    surface.sourceInstanceCustomIndex = sourceInstanceCustomIndex;
    surface.sourcePrimitiveIndex = primitiveIndex;
    surface.sourceBarycentrics = barycentrics;
    surface.sourceObjectPosition = objectPosition;
    surface.sourceObjectNormal = objectNormal;
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
    payload.roughness = surface.roughness;
    payload.metallic = surface.metallic;
    payload.dielectricSpec = surface.dielectricSpec;
    payload.ao = surface.ao;
    payload.emissive = surface.emissive;
    payload.alphaTested = surface.alphaTested;
    payload.sourceNodeId = surface.sourceNodeId;
    payload.sourceInstanceId = surface.sourceInstanceId;
    payload.sourceInstanceCustomIndex = surface.sourceInstanceCustomIndex;
    payload.sourcePrimitiveIndex = surface.sourcePrimitiveIndex;
    payload.sourceBarycentrics = surface.sourceBarycentrics;
    payload.sourceObjectPosition = surface.sourceObjectPosition;
    payload.sourceObjectNormal = surface.sourceObjectNormal;
}
