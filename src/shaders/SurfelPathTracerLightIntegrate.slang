#include "SurfelPathTracerCommon.slang"

[[vk::binding(0, 0)]] RWTexture2D<float4> outputImage;
[[vk::binding(1, 0)]] RWTexture2D<float4> gBufferNormal;
[[vk::binding(2, 0)]] RWTexture2D<float> gBufferDepth;
[[vk::binding(16, 0)]] RWTexture2D<float4> reflectionImages;
[[vk::binding(17, 0)]] RWTexture2D<float4> filteredReflectionImages;
[[vk::binding(18, 0)]] RWTexture2D<float4> lightingImages;
[[vk::binding(20, 0)]] RWTexture2D<float4> gBufferAlbedo;
[[vk::binding(21, 0)]] RWTexture2D<float4> gBufferMaterial;
[[vk::binding(22, 0)]] RWTexture2D<float4> gBufferEmissive;
[[vk::binding(27, 0)]] RWTexture2D<float4> referenceImage;
[[vk::binding(5, 0)]] RWStructuredBuffer<SurfelPathTracerSurfel> surfelBuffer;
[[vk::binding(11, 0)]] RWStructuredBuffer<SurfelPathTracerCellInfo> cellInfoBuffer;
[[vk::binding(12, 0)]] RWByteAddressBuffer cellCounterBuffer;
[[vk::binding(13, 0)]] RWStructuredBuffer<uint> cellToSurfelBuffer;

[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

struct LightIntegratePushConstants {
    uint width;
    uint height;
    uint enableDiffuseGi;
    uint enableReflections;
    uint useBilateralReflection;
    uint debugView;
    float cellSize;
    float surfelMaxRadiusScale;
    uint cellDimension;
    uint perCellSurfelLimit;
    uint useOriginalStyleGiNormalization;
};

[[vk::push_constant]] LightIntegratePushConstants push;

static const uint SURFEL_DEBUG_FINAL_COLOR = 0u;
static const uint SURFEL_DEBUG_GBUFFER_NORMAL = 1u;
static const uint SURFEL_DEBUG_GBUFFER_DEPTH = 2u;
static const uint SURFEL_DEBUG_SURFEL_ID = 3u;
static const uint SURFEL_DEBUG_SURFEL_RADIUS = 4u;
static const uint SURFEL_DEBUG_SURFEL_RADIANCE = 5u;
static const uint SURFEL_DEBUG_SURFEL_VARIANCE = 6u;
static const uint SURFEL_DEBUG_CELL_OCCUPANCY = 7u;
static const uint SURFEL_DEBUG_REFLECTION_RAW = 8u;
static const uint SURFEL_DEBUG_REFLECTION_FILTERED = 9u;
static const uint SURFEL_DEBUG_SURFEL_COVERAGE = 10u;
static const uint SURFEL_DEBUG_REFERENCE_COLOR = 11u;
static const uint SURFEL_DEBUG_REFERENCE_DIFFERENCE = 12u;
static const uint SURFEL_DEBUG_GBUFFER_ALBEDO = 13u;
static const uint SURFEL_DEBUG_DIFFUSE_GI = 14u;
static const uint SURFEL_DEBUG_SUN_VISIBILITY = 15u;
static const uint SURFEL_DEBUG_AMBIENT_OCCLUSION = 16u;
static const uint SURFEL_DEBUG_DIFFUSE_GI_BEFORE_AO = 17u;
static const float SURFEL_PT_DIFFUSE_GI_SCALE = 1.00;

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

bool loadSelectedContributingSurfel(float3 worldPosition,
                                    float3 normal,
                                    out SurfelPathTracerSurfel selectedSurfel,
                                    out uint selectedSurfelIndex) {
    selectedSurfel = (SurfelPathTracerSurfel)0;
    selectedSurfelIndex = SURFEL_PT_INVALID_INDEX;
    uint dim = max(push.cellDimension, 1u);
    int3 centerCoord = cameraRelativeCellCoord(worldPosition, ubo.cameraPos.xyz, push.cellSize);
    if (!isCellCoordValid(centerCoord, dim)) {
        return false;
    }

    float bestWeight = 0.0;
    bool foundSurfel = false;
    for (int shell = 0; shell <= 1; ++shell) {
        for (int z = -shell; z <= shell; ++z) {
            for (int y = -shell; y <= shell; ++y) {
                for (int x = -shell; x <= shell; ++x) {
                    if (max(abs(x), max(abs(y), abs(z))) != shell) {
                        continue;
                    }

                    int3 cellCoord = centerCoord + int3(x, y, z);
                    if (!isCellCoordValid(cellCoord, dim)) {
                        continue;
                    }

                    uint cellIndex = flattenCameraRelativeCellCoord(cellCoord, dim);
                    SurfelPathTracerCellInfo cellInfo = cellInfoBuffer[cellIndex];
                    for (uint slot = 0u; slot < cellInfo.surfelCount; ++slot) {
                        uint surfelIndex = cellToSurfelBuffer[cellInfo.surfelOffset + slot];
                        if (surfelIndex == SURFEL_PT_INVALID_INDEX) {
                            continue;
                        }

                        SurfelPathTracerSurfel surfel = surfelBuffer[surfelIndex];
                        if (!isActiveSurfel(surfel)) {
                            continue;
                        }

                        float3 surfelNormal = unpackNormalOctahedral(surfel.packedNormal);
                        float normalWeight = saturate(dot(normal, surfelNormal));
                        float distanceWeight = saturate(1.0 - length(worldPosition - surfel.position) / max(surfel.radius, 0.0001));
                        float weight = normalWeight * distanceWeight;
                        if (weight > bestWeight) {
                            bestWeight = weight;
                            selectedSurfel = surfel;
                            selectedSurfelIndex = surfelIndex;
                            foundSurfel = true;
                        }
                    }
                }
            }
        }
    }

    return foundSurfel;
}

float3 surfelIdColor(uint surfelIndex) {
    if (surfelIndex == SURFEL_PT_INVALID_INDEX) {
        return float3(0.0);
    }
    uint hash = pcgHash(surfelIndex + 1u);
    float3 color = float3(float(hash & 255u),
                          float((hash >> 8u) & 255u),
                          float((hash >> 16u) & 255u)) / 255.0;
    return lerp(float3(0.15), float3(1.0), color);
}

float3 evaluatePrimaryDirectLighting(float3 viewDir,
                                     float3 normal,
                                     float3 baseColor,
                                     float metallic,
                                     float roughness,
                                     float3 f0,
                                     float3 sunDir,
                                     float sunVisibility)
{
    float directSun = max(dot(normal, sunDir), 0.0);
    if (directSun <= 0.0 || sunVisibility <= 0.0) {
        return float3(0.0);
    }

    float3 V = normalize(viewDir);
    float3 L = normalize(sunDir);
    float3 H = normalize(L + V);
    float3 F = fresnelSchlick(max(dot(H, V), 0.0), f0);
    float D = distributionGGX(normal, H, roughness);
    float G = geometrySmith(normal, V, L, roughness);
    float3 kD = (float3(1.0) - F) * (1.0 - metallic);
    float3 diffuse = kD * baseColor / PI;
    float3 specular = (D * G * F) / max(4.0 * max(dot(normal, V), 0.0001) * directSun, 0.0001);
    return (diffuse + specular) * SUN_RADIANCE * directSun * sunVisibility;
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
    float3 material = gBufferMaterial[pixel].rgb;
    float roughness = clamp(material.x, MIN_ROUGHNESS, 1.0);
    float metallic = saturate(material.y);
    float dielectricSpec = max(material.z, 0.0);
    float ao = saturate(gBufferMaterial[pixel].w);
    float3 f0 = lerp(float3(0.08 * dielectricSpec), albedo, metallic);
    float3 emissive = max(gBufferEmissive[pixel].rgb, float3(0.0));
    float3 rawSurfelRadiance = max(outputImage[pixel].rgb, float3(0.0));
    float surfelCoverage = max(outputImage[pixel].a, 0.0);
    float diffuseNormalization = push.useOriginalStyleGiNormalization != 0u ? 1.0 : (1.0 / PI);
    float3 diffuseBsdf = albedo * diffuseNormalization;
    float3 diffuseGiBeforeAo = push.enableDiffuseGi != 0u
                               ? diffuseBsdf * rawSurfelRadiance * SURFEL_PT_DIFFUSE_GI_SCALE
                               : float3(0.0);
    float3 diffuseGi = diffuseGiBeforeAo * ao;
    float sunVisibility = saturate(gBufferNormal[pixel].w);
    float3 worldPosition = reconstructWorldPosition(pixel, depth);
    float3 viewDir = normalize(ubo.cameraPos.xyz - worldPosition);
    float3 directLighting = evaluatePrimaryDirectLighting(viewDir,
                                                          normal,
                                                          albedo,
                                                          metallic,
                                                          roughness,
                                                          f0,
                                                          sunDir,
                                                          sunVisibility);
    uint2 reflectionPixel = halfResolutionPixel(pixel);
    float3 reflectionRaw = reflectionImages[reflectionPixel].rgb;
    float3 reflectionFiltered = push.useBilateralReflection != 0u
                                ? reflectionImages[reflectionPixel].rgb
                                : filteredReflectionImages[reflectionPixel].rgb;
    float3 reflection = push.enableReflections != 0u ? reflectionFiltered : float3(0.0);

    float3 lighting = clampLuminance(directLighting + diffuseGi + reflection + emissive, SURFEL_PT_MAX_RADIANCE_LUMINANCE);
    float4 referenceSample = referenceImage[pixel];
    float3 referenceColor = referenceImage[pixel].rgb;
    if (referenceSample.a <= 0.0) {
        referenceColor = lighting;
    }

    if (push.debugView == SURFEL_DEBUG_GBUFFER_NORMAL) {
        lightingImages[pixel] = float4(normal * 0.5 + 0.5, 1.0);
    } else if (push.debugView == SURFEL_DEBUG_GBUFFER_DEPTH) {
        float depthVis = saturate(depth / 50.0);
        lightingImages[pixel] = float4(depthVis, depthVis, depthVis, 1.0);
    } else if (push.debugView == SURFEL_DEBUG_GBUFFER_ALBEDO) {
        lightingImages[pixel] = float4(albedo, 1.0);
    } else if (push.debugView == SURFEL_DEBUG_SURFEL_ID) {
        SurfelPathTracerSurfel surfel;
        uint surfelIndex;
        loadSelectedContributingSurfel(worldPosition, normal, surfel, surfelIndex);
        lightingImages[pixel] = float4(surfelIdColor(surfelIndex), 1.0);
    } else if (push.debugView == SURFEL_DEBUG_SURFEL_RADIUS) {
        SurfelPathTracerSurfel surfel;
        uint surfelIndex;
        float radius = loadSelectedContributingSurfel(worldPosition, normal, surfel, surfelIndex)
            ? surfel.radius / max(push.cellSize * push.surfelMaxRadiusScale, 1e-4)
            : 0.05;
        lightingImages[pixel] = float4(radius.xxx, 1.0);
    } else if (push.debugView == SURFEL_DEBUG_SURFEL_RADIANCE) {
        lightingImages[pixel] = float4(rawSurfelRadiance, 1.0);
    } else if (push.debugView == SURFEL_DEBUG_SURFEL_VARIANCE) {
        SurfelPathTracerSurfel surfel;
        uint surfelIndex;
        float3 variance = loadSelectedContributingSurfel(worldPosition, normal, surfel, surfelIndex) ? surfel.varianceAndInconsistency.xyz : float3(0.05, 0.0, 0.05);
        lightingImages[pixel] = float4(variance, 1.0);
    } else if (push.debugView == SURFEL_DEBUG_REFLECTION_RAW) {
        lightingImages[pixel] = float4(reflectionRaw, 1.0);
    } else if (push.debugView == SURFEL_DEBUG_REFLECTION_FILTERED) {
        lightingImages[pixel] = float4(reflectionFiltered, 1.0);
    } else if (push.debugView == SURFEL_DEBUG_SURFEL_COVERAGE) {
        lightingImages[pixel] = float4(surfelCoverage.xxx, 1.0);
    } else if (push.debugView == SURFEL_DEBUG_REFERENCE_COLOR) {
        lightingImages[pixel] = float4(referenceColor, 1.0);
    } else if (push.debugView == SURFEL_DEBUG_REFERENCE_DIFFERENCE) {
        float3 referenceDifference = referenceSample.a > 0.0 ? abs(referenceColor - lighting) : float3(0.0);
        lightingImages[pixel] = float4(referenceDifference, 1.0);
    } else if (push.debugView == SURFEL_DEBUG_DIFFUSE_GI) {
        lightingImages[pixel] = float4(diffuseGi, 1.0);
    } else if (push.debugView == SURFEL_DEBUG_SUN_VISIBILITY) {
        lightingImages[pixel] = float4(sunVisibility.xxx, 1.0);
    } else if (push.debugView == SURFEL_DEBUG_AMBIENT_OCCLUSION) {
        lightingImages[pixel] = float4(ao.xxx, 1.0);
    } else if (push.debugView == SURFEL_DEBUG_DIFFUSE_GI_BEFORE_AO) {
        lightingImages[pixel] = float4(diffuseGiBeforeAo, 1.0);
    } else if (push.debugView == SURFEL_DEBUG_CELL_OCCUPANCY) {
        float3 occupancyColor = cellOccupancyDebugColor(worldPosition);
        lightingImages[pixel] = float4(occupancyColor, 1.0);
    } else {
        lightingImages[pixel] = float4(lighting, 1.0);
    }
}
