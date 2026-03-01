#include "ShaderCommon.slang"

struct BuiltInTriangleIntersectionAttributes {
    float2 barycentrics;
};

// Set 0 Bindings
[[vk::binding(0, 0)]] RaytracingAccelerationStructure topLevelAS;
[[vk::binding(1, 0)]] RWTexture2D<float4> image;
[[vk::binding(2, 0)]] ByteAddressBuffer globalVertices[];
[[vk::binding(3, 0)]] ByteAddressBuffer globalIndices[];
[[vk::binding(4, 0)]] StructuredBuffer<MaterialData> globalMaterials[];
[[vk::binding(5, 0)]] Sampler2D globalTextures[];

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

    uint customIndex    = InstanceID();
    uint modelId        = customIndex >> 14;
    uint primitiveOffset = customIndex & 0x3FFF;
    uint geometryIndex  = GeometryIndex();

    MaterialData mat = globalMaterials[NonUniformResourceIndex(modelId)][primitiveOffset + geometryIndex];

    uint firstIndex  = mat.firstIndex;
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

    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);

    float3 normal = v0.normal * barycentrics.x + v1.normal * barycentrics.y + v2.normal * barycentrics.z;
    float2 uv     = v0.texCoord * barycentrics.x + v1.texCoord * barycentrics.y + v2.texCoord * barycentrics.z;

    // Correct normal transform: transpose(inverse(ObjectToWorld)) = transpose(WorldToObject).
    // mul(row_vec, M) = M^T * col_vec, so mul(normal, WorldToObject_33) gives the right result.
    float3 worldNormal = normalize(mul(normal, (float3x3)WorldToObject3x4()));

    float3 hitPos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    float3 V = normalize(ubo.cameraPos.xyz - hitPos);

    // ========================================================================
    // Base Color
    // ========================================================================
    float4 baseColor = mat.baseColorFactor;
    if (mat.baseColorIndex >= 0) {
        float4 sampled = globalTextures[NonUniformResourceIndex(mat.baseColorIndex + mat.globalTextureOffset)].SampleLevel(uv, 0.0);
        baseColor.rgb *= sRGBToLinear(sampled.rgb);
        baseColor.a *= sampled.a;
    }

    // ========================================================================
    // Metallic-Roughness
    // ========================================================================
    float metallic = mat.metallicFactor;
    float roughness = mat.roughnessFactor;

    if (mat.metallicRoughnessIndex >= 0) {
        float4 mrSample = globalTextures[NonUniformResourceIndex(mat.metallicRoughnessIndex + mat.globalTextureOffset)].SampleLevel(uv, 0.0);
        roughness *= mrSample.g;
        metallic *= mrSample.b;
    }
    roughness = clamp(roughness, MIN_ROUGHNESS, 1.0);

    // ========================================================================
    // Normal Mapping
    // ========================================================================
    float3 N = normalize(worldNormal);

    if (mat.normalIndex >= 0) {
        float3 T = normalize(mul((float3x3)ObjectToWorld3x4(), v0.tangent.xyz * barycentrics.x + v1.tangent.xyz * barycentrics.y + v2.tangent.xyz * barycentrics.z));
        float tangentW = v0.tangent.w * barycentrics.x + v1.tangent.w * barycentrics.y + v2.tangent.w * barycentrics.z;

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
                tangentNormal = normalize(tangentNormal);

                float3x3 TBN = float3x3(T, B, N);
                N = normalize(mul(tangentNormal, TBN));
            }
        }
    }

    // ========================================================================
    // Ambient Occlusion
    // ========================================================================
    float ao = 1.0;
    if (mat.occlusionIndex >= 0) {
        float aoSample = globalTextures[NonUniformResourceIndex(mat.occlusionIndex + mat.globalTextureOffset)].SampleLevel(uv, 0.0).r;
        ao = 1.0 + mat.occlusionStrength * (aoSample - 1.0);
    }

    // ========================================================================
    // Emissive
    // ========================================================================
    float3 emissive = mat.emissiveFactor;
    if (mat.emissiveIndex >= 0) {
        float3 emissiveSample = globalTextures[NonUniformResourceIndex(mat.emissiveIndex + mat.globalTextureOffset)].SampleLevel(uv, 0.0).rgb;
        emissive *= sRGBToLinear(emissiveSample);
    }

    // Dielectric Specular
    float dielectricSpecular = mat.specularFactor;
    if (mat.specularTextureIndex >= 0) {
        dielectricSpecular *= globalTextures[NonUniformResourceIndex(mat.specularTextureIndex + mat.globalTextureOffset)].SampleLevel(uv, 0.0).a;
    }
    float3 F0 = float3(0.08 * dielectricSpecular, 0.08 * dielectricSpecular, 0.08 * dielectricSpecular);
    F0 = lerp(F0, baseColor.rgb, metallic);

    float3 Lo = float3(0.0);

    // ========================================================================
    // RT Shadow
    // ========================================================================
    float3 L = normalize(-ubo.lightDir.xyz);
    
    // Slight offset to prevent shadow acne
    float3 shadowRayOrigin = hitPos + N * 0.005;
    
    RayDesc shadowRay;
    shadowRay.Origin = shadowRayOrigin;
    shadowRay.Direction = L;
    shadowRay.TMin = 0.005;
    shadowRay.TMax = 10000.0;

    RayPayload shadowPayload;
    shadowPayload.color = float3(0.0);
    shadowPayload.isShadowed = 1;

    // Call TraceRay for shadow (skip Closest Hit)
    TraceRay(topLevelAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 0xFF, 0, 0, 0, shadowRay, shadowPayload);

    // If shadowPayload.isShadowed == 1, point is occluded. 
    float shadow = float(shadowPayload.isShadowed);

    // Sun light
    {
        float3 H = normalize(V + L);
        float3 radiance = float3(1.0, 0.98, 0.95) * 3.0;

        float NdotL = max(dot(N, L), 0.0);
        float NdotV = max(dot(N, V), 0.0);

        float D = distributionGGX(N, H, roughness);
        float G = geometrySmith(N, V, L, roughness);
        float3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        float3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 0.0001);
        float3 kD = (1.0 - F) * (1.0 - metallic);
        float3 diffuse = kD * baseColor.rgb / PI;

        Lo += (diffuse + specular) * radiance * NdotL * (1.0 - shadow);
    }

    // Fill light
    {
        float3 fillL = normalize(float3(-0.3, 0.5, -0.5));
        float3 H = normalize(V + fillL);
        float3 radiance = float3(0.6, 0.7, 0.9) * 0.5;

        float NdotL = max(dot(N, fillL), 0.0);
        float NdotV = max(dot(N, V), 0.0);

        float D = distributionGGX(N, H, roughness);
        float G = geometrySmith(N, V, fillL, roughness);
        float3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        float3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 0.0001);
        float3 kD = (1.0 - F) * (1.0 - metallic);
        float3 diffuse = kD * baseColor.rgb / PI;

        Lo += (diffuse + specular) * radiance * NdotL; // No shadows for fill light
    }

    // Ambient
    float3 skyColor = float3(0.3, 0.4, 0.6);
    float3 groundColor = float3(0.1, 0.08, 0.06);
    float3 ambientL = lerp(groundColor, skyColor, N.y * 0.5 + 0.5);

    float3 kS = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
    float3 kD_amb = (1.0 - kS) * (1.0 - metallic);

    float3 ambientDiffuse = kD_amb * baseColor.rgb * ambientL;
    float3 ambientSpecular = kS * ambientL * (1.0 - roughness * 0.5);
    float3 ambientContribution = (ambientDiffuse + ambientSpecular) * ao * 0.3;

    float3 finalColor = ambientContribution + Lo + emissive;
    finalColor = acesTonemap(finalColor);

    payload.color = finalColor;
}
