#include "SurfelPathTracerCommon.slang"

[[vk::binding(0, 0)]] RWTexture2D<float4> outputImage;
[[vk::binding(1, 0)]] RWTexture2D<float4> gBufferNormal;
[[vk::binding(2, 0)]] RWTexture2D<float> gBufferDepth;
[[vk::binding(16, 0)]] RWTexture2D<float4> reflectionImages;
[[vk::binding(17, 0)]] RWTexture2D<float4> filteredReflectionImages;
[[vk::binding(18, 0)]] RWTexture2D<float4> lightingImages;
[[vk::binding(20, 0)]] RWTexture2D<float4> gBufferAlbedo;
[[vk::binding(11, 0)]] RWStructuredBuffer<SurfelPathTracerCellInfo> cellInfoBuffer;
[[vk::binding(12, 0)]] RWByteAddressBuffer cellCounterBuffer;

[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

struct LightIntegratePushConstants {
    uint width;
    uint height;
    uint enableDiffuseGi;
    uint enableReflections;
    uint useBilateralReflection;
    uint debugView;
    float cellSize;
    uint cellDimension;
    uint perCellSurfelLimit;
    uint pad1;
};

[[vk::push_constant]] LightIntegratePushConstants push;

static const uint SURFEL_DEBUG_FINAL_COLOR = 0u;
static const uint SURFEL_DEBUG_GBUFFER_NORMAL = 1u;
static const uint SURFEL_DEBUG_GBUFFER_DEPTH = 2u;
static const uint SURFEL_DEBUG_CELL_OCCUPANCY = 7u;
static const uint SURFEL_DEBUG_SURFEL_RADIANCE = 5u;
static const uint SURFEL_DEBUG_REFLECTION_RAW = 8u;
static const uint SURFEL_DEBUG_REFLECTION_FILTERED = 9u;
static const uint SURFEL_DEBUG_GBUFFER_ALBEDO = 13u;
static const uint SURFEL_DEBUG_DIFFUSE_GI = 14u;
static const uint SURFEL_DEBUG_SUN_VISIBILITY = 15u;
static const float SURFEL_PT_DIFFUSE_GI_SCALE = 0.35;

float3 primaryRayDirection(uint2 pixel) {
    float2 uv = (float2(pixel) + float2(0.5, 0.5)) / float2(max(push.width, 1u), max(push.height, 1u));
    float2 d = uv * 2.0 - 1.0;
    float4 target = mul(ubo.projInverse, float4(d.x, -d.y, 1.0, 1.0));
    return normalize(mul(ubo.viewInverse, float4(normalize(target.xyz / target.w), 0.0)).xyz);
}

float3 reconstructWorldPosition(uint2 pixel, float depth) {
    return ubo.cameraPos.xyz + primaryRayDirection(pixel) * depth;
}

uint2 halfResolutionPixel(uint2 pixel) {
    uint2 halfPixel = min(pixel / 2u, uint2(max((push.width + 1u) / 2u, 1u) - 1u,
                                           max((push.height + 1u) / 2u, 1u) - 1u));
    return halfPixel;
}

float3 cellOccupancyDebugColor(float3 worldPosition) {
    uint cellIndex = cameraRelativeCellIndexForPosition(worldPosition,
                                                        ubo.cameraPos.xyz,
                                                        push.cellSize,
                                                        push.cellDimension);
    if (cellIndex == SURFEL_PT_INVALID_INDEX) {
        return float3(0.15, 0.0, 0.3);
    }

    SurfelPathTracerCellInfo cellInfo = cellInfoBuffer[cellIndex];
    uint storedCount = cellInfo.surfelCount;
    uint rawCount = cellCounterBuffer.Load((1u + cellIndex) * 4u);
    rawCount = max(rawCount, storedCount);
    uint perCellLimit = max(push.perCellSurfelLimit, 1u);
    float occupancy = saturate(float(storedCount) / float(perCellLimit));
    float overflow = saturate(float(rawCount - storedCount) / float(perCellLimit));
    float3 occupancyColor = float3(overflow, occupancy, 1.0 - occupancy);
    return occupancyColor;
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
    float3 rawSurfelRadiance = max(outputImage[pixel].rgb, float3(0.0));
    float3 diffuseGi = push.enableDiffuseGi != 0u ? albedo * rawSurfelRadiance * SURFEL_PT_DIFFUSE_GI_SCALE : float3(0.0);
    float directSun = max(dot(normal, sunDir), 0.0);
    float sunVisibility = saturate(gBufferNormal[pixel].w);
    float3 directLighting = SUN_RADIANCE * directSun * sunVisibility;
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
    } else if (push.debugView == SURFEL_DEBUG_GBUFFER_ALBEDO) {
        lightingImages[pixel] = float4(albedo, 1.0);
    } else if (push.debugView == SURFEL_DEBUG_SURFEL_RADIANCE) {
        lightingImages[pixel] = float4(rawSurfelRadiance, 1.0);
    } else if (push.debugView == SURFEL_DEBUG_REFLECTION_RAW) {
        lightingImages[pixel] = float4(reflectionRaw, 1.0);
    } else if (push.debugView == SURFEL_DEBUG_REFLECTION_FILTERED) {
        lightingImages[pixel] = float4(reflectionFiltered, 1.0);
    } else if (push.debugView == SURFEL_DEBUG_DIFFUSE_GI) {
        lightingImages[pixel] = float4(diffuseGi, 1.0);
    } else if (push.debugView == SURFEL_DEBUG_SUN_VISIBILITY) {
        lightingImages[pixel] = float4(sunVisibility.xxx, 1.0);
    } else if (push.debugView == SURFEL_DEBUG_CELL_OCCUPANCY) {
        float3 worldPosition = reconstructWorldPosition(pixel, depth);
        float3 occupancyColor = cellOccupancyDebugColor(worldPosition);
        lightingImages[pixel] = float4(occupancyColor, 1.0);
    } else {
        lightingImages[pixel] = float4(lighting, 1.0);
    }
}
