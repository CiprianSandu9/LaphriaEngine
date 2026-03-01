#include "ShaderCommon.slang"



struct VSOutput {
    float4 pos : SV_Position;
    float3 worldPos;
    float3 normal;
    float4 tangent;
    float2 texCoord;
    float3 color;
};

[[vk::binding(0, 0)]]
ConstantBuffer<UniformBuffer> ubo;

// CSM shadow map array (sampled image) and its hardware PCF comparison sampler.
[[vk::binding(1, 0)]]
Texture2DArray<float> shadowMap;

[[vk::binding(2, 0)]]
SamplerComparisonState shadowSampler;

[[vk::binding(0, 1)]]
StructuredBuffer<MaterialData> materialBuffer;

[[vk::binding(1, 1)]]
Sampler2D textures[];


[[vk::push_constant]]
ScenePushConstants push;



// ============================================================================
// Vertex Shader
// ============================================================================

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;

    float4 worldPos = mul(push.modelMatrix, float4(input.inPosition, 1.0));
    output.worldPos = worldPos.xyz;
    output.pos = mul(ubo.proj, mul(ubo.view, worldPos));

    float3x3 modelMat3   = (float3x3)push.modelMatrix;
    float3x3 normalMatrix = transpose(mat3Inverse(modelMat3));
    output.normal  = normalize(mul(normalMatrix, input.inNormal));
    // Tangents transform with the model matrix (not the inverse-transpose).
    output.tangent = float4(normalize(mul(modelMat3, input.inTangent.xyz)), input.inTangent.w);

    output.texCoord = input.inTexCoord;
    output.color = input.inColor;

    return output;
}

// ============================================================================
// Fragment Shader
// ============================================================================

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_TARGET {
    MaterialData material = materialBuffer[push.materialIndex];
    // ========================================================================
    // Base Color
    // ========================================================================
    float4 baseColor = material.baseColorFactor;

    if (material.baseColorIndex >= 0) {
        float4 sampled = textures[NonUniformResourceIndex(material.baseColorIndex)].Sample(input.texCoord);
        baseColor.rgb *= sRGBToLinear(sampled.rgb);
        baseColor.a *= sampled.a;
    }
    
    baseColor.rgb *= input.color;

    if (baseColor.a < material.alphaCutoff) {
        discard;
    }

    // ========================================================================
    // Metallic-Roughness
    // ========================================================================
    float metallic = material.metallicFactor;
    float roughness = material.roughnessFactor;

    if (material.metallicRoughnessIndex >= 0) {
        float4 mrSample = textures[NonUniformResourceIndex(material.metallicRoughnessIndex)].Sample(input.texCoord);
        roughness *= mrSample.g;
        metallic *= mrSample.b;
    }

    roughness = clamp(roughness, MIN_ROUGHNESS, 1.0);

    // ========================================================================
    // Normal Mapping
    // ========================================================================
    float3 N = normalize(input.normal);

    if (material.normalIndex >= 0) {
        float3 T = input.tangent.xyz;
        float tangentLengthSq = dot(T, T);

        if (tangentLengthSq > 0.0001) {
            T = T * rsqrt(tangentLengthSq);
            T = T - N * dot(N, T);
            tangentLengthSq = dot(T, T);

            if (tangentLengthSq > 0.0001) {
                T = T * rsqrt(tangentLengthSq);
                float3 B = cross(N, T) * input.tangent.w;

                float3 sampledNormal = textures[NonUniformResourceIndex(material.normalIndex)].Sample(input.texCoord).rgb;
                float3 tangentNormal = sampledNormal * 2.0 - 1.0;
                tangentNormal.xy *= material.normalScale;
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

    if (material.occlusionIndex >= 0) {
        float aoSample = textures[NonUniformResourceIndex(material.occlusionIndex)].Sample(input.texCoord).r;
        ao = 1.0 + material.occlusionStrength * (aoSample - 1.0);
    }

    // ========================================================================
    // Emissive
    // ========================================================================
    float3 emissive = material.emissiveFactor;

    if (material.emissiveIndex >= 0) {
        float3 emissiveSample = textures[NonUniformResourceIndex(material.emissiveIndex)].Sample(input.texCoord).rgb;
        emissive *= sRGBToLinear(emissiveSample);
    }

    // ========================================================================
    // PBR Lighting
    // ========================================================================
    float3 V = normalize(ubo.cameraPos.xyz - input.worldPos);

    // Dielectric Specular (KHR_materials_specular)
    float dielectricSpecular = material.specularFactor;
    if (material.specularTextureIndex >= 0) {
        dielectricSpecular *= textures[NonUniformResourceIndex(material.specularTextureIndex)].Sample(input.texCoord).a;
    }
    
    float3 F0 = float3(0.08 * dielectricSpecular, 0.08 * dielectricSpecular, 0.08 * dielectricSpecular);

    F0 = lerp(F0, baseColor.rgb, metallic);

    float3 Lo = float3(0.0, 0.0, 0.0);


    // ========================================================================
    // Cascaded Shadow Map — select cascade and compute shadow factor
    // ========================================================================
    float shadowFactor = 1.0;
    {
        // View-space Z of the fragment (positive = distance in front of camera).
        float fragViewZ = -(mul(ubo.view, float4(input.worldPos, 1.0)).z);

        // Only apply shadow within the shadow max distance (cascadeSplits.w).
        if (fragViewZ < ubo.cascadeSplits.w) {
            int cascadeIndex = 3;
            if      (fragViewZ < ubo.cascadeSplits.x) cascadeIndex = 0;
            else if (fragViewZ < ubo.cascadeSplits.y) cascadeIndex = 1;
            else if (fragViewZ < ubo.cascadeSplits.z) cascadeIndex = 2;

            // Apply a normal bias to offset the position along the normal depending on angle to light
            // This mitigates Peter Panning without needing massive constant pipeline depth bias
            float3 sunL = normalize(-ubo.lightDir.xyz);
            float NdotL = saturate(dot(N, sunL));
            
            // Bias scales down as the normal points directly at the light
            float biasAmount = (1.0 - NdotL) * 0.05 + 0.01; 
            
            // Different cascades need different bias scales due to different orthographic volumes
            float cascadeScale = (cascadeIndex == 0) ? 0.2 : (cascadeIndex == 1) ? 0.5 : (cascadeIndex == 2) ? 1.0 : 2.0;
            float3 biasedPos = input.worldPos + N * biasAmount * cascadeScale;

            float4 shadowCoord = mul(ubo.cascadeViewProj[cascadeIndex], float4(biasedPos, 1.0));
            float2 shadowUV    = shadowCoord.xy * 0.5 + 0.5;
            float  shadowDepth = shadowCoord.z;  

            // 3x3 PCF filtering
            float sum = 0.0;
            float2 texelSize = 1.0 / 2048.0; // SHADOW_MAP_DIM from C++

            for (int y = -1; y <= 1; ++y) {
                for (int x = -1; x <= 1; ++x) {
                    float2 offset = float2(x, y) * texelSize;
                    sum += shadowMap.SampleCmp(shadowSampler, float3(shadowUV + offset, cascadeIndex), shadowDepth);
                }
            }
            shadowFactor = sum / 9.0;
        }
    }

    // Sun light
    {
        float3 L = normalize(-ubo.lightDir.xyz); // Vector TO the light source
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

        Lo += (diffuse + specular) * radiance * NdotL * shadowFactor;
    }

    // Fill light
    {
        float3 L = normalize(float3(-0.3, 0.5, -0.5));
        float3 H = normalize(V + L);
        float3 radiance = float3(0.6, 0.7, 0.9) * 0.5;

        float NdotL = max(dot(N, L), 0.0);
        float NdotV = max(dot(N, V), 0.0);

        float D = distributionGGX(N, H, roughness);
        float G = geometrySmith(N, V, L, roughness);
        float3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        float3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 0.0001);
        float3 kD = (1.0 - F) * (1.0 - metallic);
        float3 diffuse = kD * baseColor.rgb / PI;

        Lo += (diffuse + specular) * radiance * NdotL;
    }

    // Ambient
    float3 skyColor = float3(0.3, 0.4, 0.6);
    float3 groundColor = float3(0.1, 0.08, 0.06);
    float3 ambient = lerp(groundColor, skyColor, N.y * 0.5 + 0.5);

    float3 kS = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
    float3 kD = (1.0 - kS) * (1.0 - metallic);

    float3 ambientDiffuse = kD * baseColor.rgb * ambient;
    float3 ambientSpecular = kS * ambient * (1.0 - roughness * 0.5);
    float3 ambientContribution = (ambientDiffuse + ambientSpecular) * ao * 0.3;

    // ========================================================================
    // Final Output
    // ========================================================================
    float3 color = ambientContribution + Lo + emissive;
    color = acesTonemap(color);

    return float4(color, baseColor.a);
}
