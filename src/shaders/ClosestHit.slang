#include "ShaderCommon.slang"

// Path tracer RayPayload — surface properties returned by ClosestHit to Raygen.
struct RayPayload {
    float3 albedo;
    float3 emission;
    float3 worldNormal;
    float3 hitPos;
    float  roughness;
    float  metallic;
    float3 F0;
    float  ao;
    float  hitT;
    uint   instanceID;
};

struct BuiltInTriangleIntersectionAttributes {
    float2 barycentrics;
};

// Set 0 Bindings (bindings 2-4 are now G-Buffer images written by Raygen, not ClosestHit)
[[vk::binding(0, 0)]] RaytracingAccelerationStructure topLevelAS;
[[vk::binding(5, 0)]] ByteAddressBuffer globalVertices[];
[[vk::binding(6, 0)]] ByteAddressBuffer globalIndices[];
[[vk::binding(7, 0)]] StructuredBuffer<MaterialData> globalMaterials[];
[[vk::binding(8, 0)]] Sampler2D globalTextures[];

// Set 1 Bindings
[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

// C++ Vertex layout (EngineAuxiliary.h):
//   float3 pos:      offset  0 (12 bytes)
//   float3 normal:   offset 12 (12 bytes)
//   float4 tangent:  offset 24 (16 bytes)
//   float2 texCoord: offset 40 ( 8 bytes)
//   float3 color:    offset 48 (12 bytes)
//   stride: 60 bytes
static const uint kVertexStride = 60;

Vertex loadVertex(ByteAddressBuffer buf, uint idx)
{
    uint base  = idx * kVertexStride;
    Vertex v;
    v.pos      = asfloat(buf.Load3(base));
    v.normal   = asfloat(buf.Load3(base + 12));
    v.tangent  = asfloat(buf.Load4(base + 24));
    v.texCoord = asfloat(buf.Load2(base + 40));
    v.color    = asfloat(buf.Load3(base + 48));
    return v;
}

uint loadIndex(ByteAddressBuffer buf, uint idx)
{
    return buf.Load(idx * 4);
}

[shader("closesthit")]
void main(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {

    uint customIndex     = InstanceID();
    uint modelId         = customIndex >> 14;
    uint primitiveOffset = customIndex & 0x3FFF;
    uint geometryIndex   = GeometryIndex();

    MaterialData mat = globalMaterials[NonUniformResourceIndex(modelId)][primitiveOffset + geometryIndex];

    uint firstIndex   = mat.firstIndex;
    uint vertexOffset = mat.vertexOffset;
    uint primitiveIndex = PrimitiveIndex();

    ByteAddressBuffer vertBuf = globalVertices[NonUniformResourceIndex(modelId)];
    ByteAddressBuffer idxBuf  = globalIndices[NonUniformResourceIndex(modelId)];

    uint i0 = loadIndex(idxBuf, firstIndex + primitiveIndex * 3 + 0);
    uint i1 = loadIndex(idxBuf, firstIndex + primitiveIndex * 3 + 1);
    uint i2 = loadIndex(idxBuf, firstIndex + primitiveIndex * 3 + 2);

    Vertex v0 = loadVertex(vertBuf, vertexOffset + i0);
    Vertex v1 = loadVertex(vertBuf, vertexOffset + i1);
    Vertex v2 = loadVertex(vertBuf, vertexOffset + i2);

    float3 barycentrics = float3(
        1.0 - attribs.barycentrics.x - attribs.barycentrics.y,
        attribs.barycentrics.x,
        attribs.barycentrics.y);

    float2 uv = v0.texCoord * barycentrics.x + v1.texCoord * barycentrics.y + v2.texCoord * barycentrics.z;

    // World-space shading normal: transpose(inverse(ObjectToWorld)) = transpose(WorldToObject).
    float3 geomNormal = v0.normal * barycentrics.x + v1.normal * barycentrics.y + v2.normal * barycentrics.z;
    float3 N = normalize(mul(geomNormal, (float3x3)WorldToObject3x4()));
    // Flip toward ray origin so backface hits (eTriangleFacingCullDisable) shade correctly.
    // Must happen before TBN construction so the tangent frame inherits the right orientation.
    if (dot(N, WorldRayDirection()) > 0.0)
        N = -N;

    // Normal mapping
    if (mat.normalIndex >= 0) {
        float3 T = mul((float3x3)ObjectToWorld3x4(),
            v0.tangent.xyz * barycentrics.x + v1.tangent.xyz * barycentrics.y + v2.tangent.xyz * barycentrics.z);
        float tangentW = sign(v0.tangent.w * barycentrics.x + v1.tangent.w * barycentrics.y + v2.tangent.w * barycentrics.z);

        float tls = dot(T, T);
        if (tls > 0.0001) {
            T = T * rsqrt(tls);
            T = T - N * dot(N, T);
            tls = dot(T, T);
            if (tls > 0.0001) {
                T = T * rsqrt(tls);
                float3 B = cross(N, T) * tangentW;

                float3 sampledNormal = globalTextures[NonUniformResourceIndex(mat.normalIndex + mat.globalTextureOffset)].SampleLevel(uv, 0.0).rgb;
                float3 tangentNormal = sampledNormal * 2.0 - 1.0;
                tangentNormal.xy *= mat.normalScale;
                tangentNormal = normalize(tangentNormal);
                N = normalize(mul(tangentNormal, float3x3(T, B, N)));
            }
        }
    }

    // Base colour
    float4 baseColor = mat.baseColorFactor;
    if (mat.baseColorIndex >= 0) {
        float4 sampled = globalTextures[NonUniformResourceIndex(mat.baseColorIndex + mat.globalTextureOffset)].SampleLevel(uv, 0.0);
        baseColor.rgb *= sRGBToLinear(sampled.rgb);
        baseColor.a   *= sampled.a;
    }

    // Metallic-roughness
    float metallic  = mat.metallicFactor;
    float roughness = mat.roughnessFactor;
    if (mat.metallicRoughnessIndex >= 0) {
        float4 mr = globalTextures[NonUniformResourceIndex(mat.metallicRoughnessIndex + mat.globalTextureOffset)].SampleLevel(uv, 0.0);
        roughness *= mr.g;
        metallic  *= mr.b;
    }
    roughness = clamp(roughness, MIN_ROUGHNESS, 1.0);

    // Emissive
    float3 emissive = mat.emissiveFactor;
    if (mat.emissiveIndex >= 0) {
        emissive *= sRGBToLinear(
            globalTextures[NonUniformResourceIndex(mat.emissiveIndex + mat.globalTextureOffset)].SampleLevel(uv, 0.0).rgb);
    }

    // Ambient Occlusion
    float ao = 1.0;
    if (mat.occlusionIndex >= 0) {
        float aoSample = globalTextures[NonUniformResourceIndex(mat.occlusionIndex + mat.globalTextureOffset)].SampleLevel(uv, 0.0).r;
        ao = 1.0 + mat.occlusionStrength * (aoSample - 1.0);
    }

    // Fresnel base reflectance
    float dielectricSpec = mat.specularFactor;
    if (mat.specularTextureIndex >= 0)
        dielectricSpec *= globalTextures[NonUniformResourceIndex(mat.specularTextureIndex + mat.globalTextureOffset)].SampleLevel(uv, 0.0).a;
    float3 F0 = lerp(float3(0.08 * dielectricSpec), baseColor.rgb, metallic);

    // Return surface properties — Raygen does all lighting via the path tracing loop.
    payload.albedo      = baseColor.rgb;
    payload.emission    = emissive;
    payload.worldNormal = N;
    payload.hitPos      = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    payload.roughness   = roughness;
    payload.metallic    = metallic;
    payload.F0          = F0;
    payload.ao          = ao;
    payload.hitT        = RayTCurrent();
    payload.instanceID  = InstanceID();
}
