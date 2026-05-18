#include "SurfelPathTracerCommon.slang"

[[vk::binding(0, 0)]] RWTexture2D<float4> outputImage;
[[vk::binding(1, 0)]] RWTexture2D<float4> gBufferNormal;
[[vk::binding(2, 0)]] RWTexture2D<float> gBufferDepth;
[[vk::binding(16, 0)]] RWTexture2D<float4> reflectionImages;
[[vk::binding(17, 0)]] RWTexture2D<float4> filteredReflectionImages;
[[vk::binding(18, 0)]] RWTexture2D<float4> lightingImages;
[[vk::binding(20, 0)]] RWTexture2D<float4> gBufferAlbedo;

[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

struct LightIntegratePushConstants {
    uint width;
    uint height;
    uint enableDiffuseGi;
    uint enableReflections;
    uint useBilateralReflection;
    uint debugView;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] LightIntegratePushConstants push;

static const uint SURFEL_DEBUG_FINAL_COLOR = 0u;
static const uint SURFEL_DEBUG_GBUFFER_NORMAL = 1u;
static const uint SURFEL_DEBUG_GBUFFER_DEPTH = 2u;
static const uint SURFEL_DEBUG_SURFEL_RADIANCE = 5u;
static const uint SURFEL_DEBUG_REFLECTION_RAW = 8u;
static const uint SURFEL_DEBUG_REFLECTION_FILTERED = 9u;

float3 primaryRayDirection(uint2 pixel) {
    float2 uv = (float2(pixel) + float2(0.5, 0.5)) / float2(max(push.width, 1u), max(push.height, 1u));
    float2 d = uv * 2.0 - 1.0;
    float4 target = mul(ubo.projInverse, float4(d.x, -d.y, 1.0, 1.0));
    return normalize(mul(ubo.viewInverse, float4(normalize(target.xyz / target.w), 0.0)).xyz);
}

uint2 halfResolutionPixel(uint2 pixel) {
    uint2 halfPixel = min(pixel / 2u, uint2(max((push.width + 1u) / 2u, 1u) - 1u,
                                           max((push.height + 1u) / 2u, 1u) - 1u));
    return halfPixel;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint2 pixel = tid.xy;
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    float depth = gBufferDepth[pixel];
    float3 normalRaw = gBufferNormal[pixel].xyz;
    float3 sunDir = normalize(-ubo.lightDir.xyz);
    float3 viewSky = evalSkyColor(primaryRayDirection(pixel), sunDir);

    if (depth <= 0.0 || !isFinite3(normalRaw) || length(normalRaw) <= 0.001) {
        lightingImages[pixel] = float4(viewSky, 1.0);
        return;
    }

    float3 normal = normalize(normalRaw);
    float3 albedo = max(gBufferAlbedo[pixel].rgb, float3(0.0));
    float3 diffuseGi = push.enableDiffuseGi != 0u ? max(outputImage[pixel].rgb, float3(0.0)) : float3(0.0);
    float directSun = max(dot(normal, sunDir), 0.0);
    float3 skyAmbient = evalSkyColor(normal, sunDir) * 0.25 + evalGroundColor(sunDir) * 0.10;
    float3 directLighting = SUN_RADIANCE * directSun + skyAmbient;
    float3 diffuseLighting = albedo * directLighting;
    uint2 reflectionPixel = halfResolutionPixel(pixel);
    float3 reflectionRaw = reflectionImages[reflectionPixel].rgb;
    float3 reflectionFiltered = push.useBilateralReflection != 0u
                                ? reflectionImages[reflectionPixel].rgb
                                : filteredReflectionImages[reflectionPixel].rgb;
    float3 reflection = push.enableReflections != 0u ? reflectionFiltered : float3(0.0);

    float3 lighting = clampLuminance(diffuseLighting + diffuseGi + reflection, SURFEL_PT_MAX_RADIANCE_LUMINANCE);

    if (push.debugView == SURFEL_DEBUG_GBUFFER_NORMAL) {
        lightingImages[pixel] = float4(normal * 0.5 + 0.5, 1.0);
    } else if (push.debugView == SURFEL_DEBUG_GBUFFER_DEPTH) {
        float depthVis = saturate(depth / 50.0);
        lightingImages[pixel] = float4(depthVis, depthVis, depthVis, 1.0);
    } else if (push.debugView == SURFEL_DEBUG_SURFEL_RADIANCE) {
        lightingImages[pixel] = float4(diffuseGi, 1.0);
    } else if (push.debugView == SURFEL_DEBUG_REFLECTION_RAW) {
        lightingImages[pixel] = float4(reflectionRaw, 1.0);
    } else if (push.debugView == SURFEL_DEBUG_REFLECTION_FILTERED) {
        lightingImages[pixel] = float4(reflectionFiltered, 1.0);
    } else {
        lightingImages[pixel] = float4(lighting, 1.0);
    }
}
