#ifndef SHADER_COMMON_H
#define SHADER_COMMON_H

static const float PI = 3.14159265359;
static const float MIN_ROUGHNESS = 0.04;

struct RayPayload {
    float3 color;
    uint isShadowed;
};

// ============================================================================
// PBR Functions
// ============================================================================

float distributionGGX(float3 N, float3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return a2 / max(denom, 0.0001);
}

float geometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float geometrySmith(float3 N, float3 V, float3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return geometrySchlickGGX(NdotV, roughness) * geometrySchlickGGX(NdotL, roughness);
}

float3 fresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (1.0 - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

float3 fresnelSchlickRoughness(float cosTheta, float3 F0, float roughness) {
    float3 maxRefl = max(float3(1.0 - roughness, 1.0 - roughness, 1.0 - roughness), F0);
    return F0 + (maxRefl - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

float3 sRGBToLinear(float3 srgb) {
    return pow(srgb, float3(2.2, 2.2, 2.2));
}

float3 linearToSRGB(float3 linear) {
    return pow(linear, float3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2));
}

float3 acesTonemap(float3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// 3x3 matrix inverse via cofactor expansion (SPIRV does not expose a built-in inverse).
float3x3 mat3Inverse(float3x3 m) {
    float a = m[0][0], b = m[0][1], c = m[0][2];
    float d = m[1][0], e = m[1][1], f = m[1][2];
    float g = m[2][0], h = m[2][1], i = m[2][2];

    float det = a * (e * i - f * h)
              - b * (d * i - f * g)
              + c * (d * h - e * g);

    float invDet = 1.0 / det;

    // adjugate transposed (= inverse * det)
    float3x3 inv;
    inv[0][0] =  (e * i - f * h) * invDet;
    inv[0][1] = -(b * i - c * h) * invDet;
    inv[0][2] =  (b * f - c * e) * invDet;
    inv[1][0] = -(d * i - f * g) * invDet;
    inv[1][1] =  (a * i - c * g) * invDet;
    inv[1][2] = -(a * f - c * d) * invDet;
    inv[2][0] =  (d * h - e * g) * invDet;
    inv[2][1] = -(a * h - b * g) * invDet;
    inv[2][2] =  (a * e - b * d) * invDet;
    return inv;
}


struct UniformBuffer {
    float4x4 view;
    float4x4 proj;
    float4 cameraPos;
    float4 lightDir;

    float4x4 viewInverse;
    float4x4 projInverse;

    // Cascaded Shadow Map — split depths (positive view-space distances) and per-cascade light VP
    float4   cascadeSplits;
    float4x4 cascadeViewProj[4];
};

struct ScenePushConstants {
    float4x4 modelMatrix;
    int materialIndex;
    int cascadeIndex;   // which CSM cascade is rendered (shadow pass); unused padding for main pass
    int padding2;
    int padding3;
    float4 skyData; // xyz = color, w = threshold
};

struct VSInput {
    [[vk::location(0)]] float3 inPosition;
    [[vk::location(1)]] float3 inNormal;
    [[vk::location(2)]] float4 inTangent;
    [[vk::location(3)]] float2 inTexCoord;
    [[vk::location(4)]] float3 inColor;
};

struct MaterialData {
    int baseColorIndex;
    int metallicRoughnessIndex;
    int normalIndex;
    int occlusionIndex;
    int emissiveIndex;
    int specularTextureIndex;
    uint firstIndex;
    uint vertexOffset;
    int globalTextureOffset;

    float4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    float normalScale;
    float occlusionStrength;
    float3 emissiveFactor;
    float specularFactor;
    float alphaCutoff;
};

// Represents the C++ Vertex exactly (pos, normal, tangent, texCoord, color)
struct Vertex {
    float3 pos;
    float3 normal;
    float4 tangent;
    float2 texCoord;
    float3 color;
};

#endif // SHADER_COMMON_H
