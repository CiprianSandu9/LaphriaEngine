#ifndef SHADER_COMMON_H
#define SHADER_COMMON_H

static const float PI          = 3.14159265359;
static const float MIN_ROUGHNESS = 0.04;

// ============================================================================
// NOTE: RayPayload is defined per-pipeline in each shader set:
//   - Classic RT (RT_Raygen.slang etc.): struct RayPayload { float3 color; uint isShadowed; }
//   - Path Tracer (Raygen.slang etc.):   struct RayPayload { float3 albedo; ...; float hitT; uint instanceID; }
// ============================================================================
// Uniform Buffer — must mirror UniformBufferObject in EngineAuxiliary.h exactly.
// ============================================================================
struct UniformBuffer {
    float4x4 view;
    float4x4 proj;
    float4   cameraPos;
    float4   lightDir;

    float4x4 viewInverse;
    float4x4 projInverse;

    // Cascaded Shadow Map (used by rasterizer path only; still present for struct compatibility)
    float4   cascadeSplits;
    float4x4 cascadeViewProj[4];

    // Path tracer temporal fields
    float4x4 prevViewProj;   // Previous frame combined VP for motion vector reprojection
    uint     frameCount;     // Monotonically increasing RNG seed
    float    jitter_x;       // Sub-pixel x jitter in NDC (zero when disabled)
    float    jitter_y;       // Sub-pixel y jitter in NDC
    uint     _pad0;
};

// ============================================================================
// Rasterizer structs (unchanged)
// ============================================================================
struct ScenePushConstants {
    float4x4 modelMatrix;
    int materialIndex;
    int cascadeIndex;   // which CSM cascade is rendered (shadow pass); unused in main pass
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

// ============================================================================
// PBR Math — kept intact; Raygen uses these for GGX importance sampling.
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
    return pow(max(linear, float3(0.0, 0.0, 0.0)), float3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2));
}

// Procedural sky radiance for a given view direction and sun direction.
// sunDir = normalize(-lightDir) — the vector pointing FROM the scene TOWARD the sun.
// Used by Miss shaders (background), and evaluated at zenith/nadir for ambient terms.
float3 evalSkyColor(float3 rayDir, float3 sunDir) {
    // Vertical gradient: 0 at/below horizon, 1 at zenith.
    float skyT         = clamp(rayDir.y * 0.5 + 0.5, 0.0, 1.0);
    float sunElevation = sunDir.y; // -1 = fully below horizon, +1 = directly overhead

    // Colour palettes (linear HDR).
    float3 dayZenith     = float3(0.15, 0.35, 0.75);
    float3 dayHorizon    = float3(0.50, 0.65, 0.88);
    float3 sunsetZenith  = float3(0.12, 0.18, 0.45);
    float3 sunsetHorizon = float3(0.90, 0.42, 0.08);
    float3 nightZenith   = float3(0.005, 0.006, 0.020);
    float3 nightHorizon  = float3(0.008, 0.009, 0.015);

    // dayFactor:    0 when sun is at/below horizon, ramps to 1 at ~24° elevation.
    // sunsetFactor: peaks at 1 when sun is on the horizon, falls off on both sides.
    float dayFactor    = clamp(sunElevation * 2.5,            0.0, 1.0);
    float sunsetFactor = clamp(1.0 - abs(sunElevation) * 5.0, 0.0, 1.0);

    float3 zenithColor  = lerp(lerp(nightZenith,  sunsetZenith,  sunsetFactor), dayZenith,  dayFactor);
    float3 horizonColor = lerp(lerp(nightHorizon, sunsetHorizon, sunsetFactor), dayHorizon, dayFactor);

    // Quadratic vertical gradient for a natural sky-dome falloff.
    return lerp(horizonColor, zenithColor, skyT * skyT);
}

// Ground-bounce colour for the ambient hemisphere lower lobe.
// Represents diffuse irradiance reflected up from the terrain — always earth-toned,
// never sky-coloured. sunDir = normalize(-lightDir).
float3 evalGroundColor(float3 sunDir) {
    float sunElevation = sunDir.y;
    float dayFactor    = clamp(sunElevation * 2.5,            0.0, 1.0);
    float sunsetFactor = clamp(1.0 - abs(sunElevation) * 5.0, 0.0, 1.0);

    float3 dayGround    = float3(0.10, 0.08, 0.06);   // sun-lit warm earth
    float3 sunsetGround = float3(0.14, 0.08, 0.04);   // slightly warm-tinted at dusk/dawn
    float3 nightGround  = float3(0.005, 0.004, 0.003); // near-black at night

    return lerp(lerp(nightGround, sunsetGround, sunsetFactor), dayGround, dayFactor);
}

float3 acesTonemap(float3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// 3x3 matrix inverse via cofactor expansion (SPIRV has no built-in inverse).
float3x3 mat3Inverse(float3x3 m) {
    float a = m[0][0], b = m[0][1], c = m[0][2];
    float d = m[1][0], e = m[1][1], f = m[1][2];
    float g = m[2][0], h = m[2][1], i = m[2][2];

    float det = a * (e * i - f * h)
              - b * (d * i - f * g)
              + c * (d * h - e * g);

    float invDet = 1.0 / det;

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

// ============================================================================
// Random Number Generation — PCG hash, seeded by pixel index and frame count.
// ============================================================================

uint pcgHash(uint seed) {
    uint state = seed * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Advance RNG state and return a uniform float in [0, 1).
float randomFloat(inout uint seed) {
    seed = pcgHash(seed);
    return float(seed) / 4294967296.0;
}

// ============================================================================
// Sampling — cosine-weighted hemisphere and GGX reflection.
// ============================================================================

// Cosine-weighted sample on the hemisphere oriented around N.
// xi: two uniform random numbers in [0, 1).
float3 cosineSampleHemisphere(float2 xi, float3 N) {
    float r   = sqrt(xi.x);
    float phi = 2.0 * PI * xi.y;

    // Local tangent frame
    float3 up      = abs(N.y) < 0.999 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 tangent  = normalize(cross(up, N));
    float3 bitangent = cross(N, tangent);

    float3 localDir = float3(r * cos(phi), sqrt(max(0.0, 1.0 - xi.x)), r * sin(phi));
    return normalize(tangent * localDir.x + N * localDir.y + bitangent * localDir.z);
}

// GGX-importance-sampled microfacet normal (half-vector), then reflect incident ray.
// Returns the sampled outgoing direction. xi: two uniform random numbers.
float3 ggxSampleDirection(float2 xi, float3 N, float3 V, float roughness) {
    float a = roughness * roughness;

    float phi      = 2.0 * PI * xi.x;
    float cosTheta = sqrt((1.0 - xi.y) / max(1.0 + (a * a - 1.0) * xi.y, 0.0001));
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

    float3 up       = abs(N.y) < 0.999 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 tangent  = normalize(cross(up, N));
    float3 bitangent = cross(N, tangent);

    float3 H = normalize(tangent * (sinTheta * cos(phi)) +
                         N       *  cosTheta              +
                         bitangent * (sinTheta * sin(phi)));
    return reflect(-V, H);
}

#endif // SHADER_COMMON_H
