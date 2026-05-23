#include "SurfelPathTracerCommon.slang"

struct ReflectionPushConstants {
    uint width;
    uint height;
    uint fullWidth;
    uint fullHeight;
    uint maxSurfels;
    uint cellDimension;
    float cellSize;
    uint frameIndex;
    uint enabled;
    uint enableSurfelTermination;
    uint maxSurfelSamplesPerQuery;
    uint pad2;
};

[[vk::push_constant]] ReflectionPushConstants push;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(7, 0)]] StructuredBuffer<MaterialData> globalMaterials[];

[[vk::binding(1, 1)]] RWTexture2D<float4> gBufferNormal;
[[vk::binding(2, 1)]] RWTexture2D<float> gBufferDepth;
[[vk::binding(3, 1)]] RWTexture2D<float4> gBufferMotionMaterial;
[[vk::binding(5, 1)]] RWStructuredBuffer<SurfelPathTracerSurfel> surfelBuffer;
[[vk::binding(11, 1)]] RWStructuredBuffer<SurfelPathTracerCellInfo> cellInfoBuffer;
[[vk::binding(13, 1)]] RWStructuredBuffer<uint> cellToSurfelBuffer;
[[vk::binding(16, 1)]] RWTexture2D<float4> reflectionImages;
[[vk::binding(20, 1)]] RWTexture2D<float4> gBufferAlbedo;
[[vk::binding(21, 1)]] RWTexture2D<float4> gBufferMaterial;
[[vk::binding(22, 1)]] RWTexture2D<float4> gBufferEmissive;

[[vk::binding(0, 2)]] ConstantBuffer<UniformBuffer> ubo;

float3 primaryRayDirection(uint2 pixel) {
    float2 uv = (float2(pixel) + float2(0.5, 0.5)) / float2(max(push.fullWidth, 1u), max(push.fullHeight, 1u));
    float2 d = uv * 2.0 - 1.0;
    float4 target = mul(ubo.projInverse, float4(d.x, -d.y, 1.0, 1.0));
    return normalize(mul(ubo.viewInverse, float4(normalize(target.xyz / target.w), 0.0)).xyz);
}

float3 reconstructWorldPosition(uint2 pixel, float depth) {
    return ubo.cameraPos.xyz + primaryRayDirection(pixel) * depth;
}

float3 terminatePathWithSurfels(float3 position,
                                float3 normal,
                                uint maxSamples,
                                uint maxSurfels,
                                float cellSize,
                                uint cellDimension,
                                float3 cameraPosition,
                                RWStructuredBuffer<SurfelPathTracerSurfel> surfels,
                                RWStructuredBuffer<SurfelPathTracerCellInfo> cells,
                                RWStructuredBuffer<uint> cellToSurfels)
{
    uint cellIndex = cameraRelativeCellIndexForPosition(position, cameraPosition, cellSize, cellDimension);
    if (cellIndex == SURFEL_PT_INVALID_INDEX) {
        return float3(0.0);
    }

    SurfelPathTracerCellInfo cellInfo = cells[cellIndex];
    uint count = min(cellInfo.surfelCount, maxSamples);
    float3 sum = float3(0.0);
    float weightSum = 0.0;
    for (uint i = 0u; i < count; ++i) {
        uint surfelIndex = cellToSurfels[cellInfo.surfelOffset + i];
        if (surfelIndex == SURFEL_PT_INVALID_INDEX || surfelIndex >= maxSurfels) {
            continue;
        }

        SurfelPathTracerSurfel surfel = surfels[surfelIndex];
        if (!isActiveSurfel(surfel)) {
            continue;
        }

        float3 surfelNormal = unpackNormalOctahedral(surfel.packedNormal);
        float distanceWeight = saturate(1.0 - length(position - surfel.position) / max(surfel.radius, 1e-4));
        float normalWeight = saturate(dot(normal, surfelNormal));
        float weight = distanceWeight * normalWeight;
        sum += surfel.radiance * weight;
        weightSum += weight;
    }
    return weightSum > 1e-5 ? sum / weightSum : float3(0.0);
}

float traceSunVisibility(float3 position, float3 normal, float3 sunDir)
{
    RayDesc shadowRay;
    float bias = max(SURFEL_PT_RAY_BIAS * 4.0, 0.005);
    shadowRay.Origin = position + normal * bias;
    shadowRay.Direction = normalize(sunDir);
    shadowRay.TMin = bias;
    shadowRay.TMax = 10000.0;

    SurfelPathTracerPayload shadowPayload = makeVisibilitySurfelPayload();
    TraceRay(tlas,
             RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
             0xFF, 0, 0, 0, shadowRay, shadowPayload);
    return shadowPayload.hitT < 0.0 ? 1.0 : 0.0;
}

float3 evaluateShadowedDirectLighting(float3 viewDir,
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
    float nDotV = max(dot(normal, V), 0.0001);
    float vDotH = max(dot(V, H), 0.0001);
    float3 F = fresnelSchlick(vDotH, f0);
    float D = distributionGGX(normal, H, roughness);
    float G = geometrySmith(normal, V, L, roughness);
    float3 kD = (float3(1.0) - F) * (1.0 - metallic);
    float3 diffuse = kD * baseColor / PI;
    float3 specular = (D * F * G) / max(4.0 * nDotV * directSun, 0.0001);
    return (diffuse + specular) * directSun * SUN_RADIANCE * sunVisibility;
}

[shader("raygeneration")]
void main() {
    uint2 halfPixel = DispatchRaysIndex().xy;
    if (halfPixel.x >= push.width || halfPixel.y >= push.height) {
        return;
    }

    if (push.enabled == 0u) {
        reflectionImages[halfPixel] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    uint2 fullPixel = min(halfPixel * 2u + uint2(1u, 1u), uint2(push.fullWidth - 1u, push.fullHeight - 1u));
    float depth = gBufferDepth[fullPixel];
    float3 rawNormal = gBufferNormal[fullPixel].xyz;
    if (depth <= 0.0 || !isFinite3(rawNormal) || length(rawNormal) <= 0.001) {
        reflectionImages[halfPixel] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    float3 normal = normalize(rawNormal);
    float3 viewDir = normalize(ubo.cameraPos.xyz - reconstructWorldPosition(fullPixel, depth));
    float3 baseColor = max(gBufferAlbedo[fullPixel].rgb, float3(0.0));
    float4 material = gBufferMaterial[fullPixel];
    float roughness = clamp(material.x, MIN_ROUGHNESS, 1.0);
    float metallic = saturate(material.y);
    float dielectricSpec = max(material.z, 0.0);
    float3 f0 = lerp(float3(0.08 * dielectricSpec), baseColor, metallic);

    uint seed = pcgHash(halfPixel.x * 1973u ^ halfPixel.y * 9277u ^ push.frameIndex * 26699u ^ ubo.frameCount);
    SurfelPtReservoir reservoir;
    reservoir.direction = normalize(reflect(-viewDir, normal));
    reservoir.brdfWeight = float3(1.0);
    reservoir.pdf = 1.0;
    reservoir.weightSum = 0.0;
    reservoir.selectedWeight = 0.0;

    const uint candidateCount = 16u;
    for (uint candidateIndex = 0u; candidateIndex < candidateCount; ++candidateIndex) {
        float2 candidateXi = float2(randomFloat(seed), randomFloat(seed));
        float3 candidateDir = normalize(ggxSampleDirection(candidateXi, normal, viewDir, roughness));
        if (dot(candidateDir, normal) <= 0.0) {
            continue;
        }

        float3 brdf = ggxSpecularBrdf(candidateDir, normal, viewDir, f0, roughness);
        float pdf = max(ggxPdf(candidateDir, normal, viewDir, roughness), 1e-5);
        float candidateWeight = (1.0 / float(candidateCount)) * luminance(brdf) / pdf;
        addReflectionCandidate(reservoir, candidateDir, brdf, pdf, candidateWeight, randomFloat(seed));
    }

    bool reservoirFinite = isFiniteScalar(reservoir.weightSum) &&
                           isFiniteScalar(reservoir.pdf) &&
                           isFinite3(reservoir.brdfWeight);
    if (!reservoirFinite || reservoir.weightSum <= 1e-5 || reservoir.pdf <= 1e-5 ||
        luminance(reservoir.brdfWeight) <= 1e-5) {
        reflectionImages[halfPixel] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    float3 position = reconstructWorldPosition(fullPixel, depth);
    RayDesc ray;
    ray.Origin = position + normal * SURFEL_PT_RAY_BIAS;
    ray.Direction = normalize(reservoir.direction);
    ray.TMin = SURFEL_PT_RAY_BIAS;
    ray.TMax = 10000.0;

    SurfelPathTracerPayload payload = makeEmptySurfelPayload();
    TraceRay(tlas, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);

    float3 reflected = payload.radiance;
    if (payload.hitKind != 0u) {
        float3 secondaryViewDir = normalize(-ray.Direction);
        float3 sunDir = normalize(-ubo.lightDir.xyz);
        float sunVisibility = traceSunVisibility(payload.hitPosition, payload.hitNormal, sunDir);
        reflected = evaluateShadowedDirectLighting(secondaryViewDir,
                                                   payload.hitNormal,
                                                   payload.baseColor,
                                                   payload.metallic,
                                                   payload.roughness,
                                                   payload.f0,
                                                   sunDir,
                                                   sunVisibility) + payload.emission;
        if (push.enableSurfelTermination != 0u) {
            float3 cachedIncident = terminatePathWithSurfels(payload.hitPosition,
                                                             payload.hitNormal,
                                                             push.maxSurfelSamplesPerQuery,
                                                             push.maxSurfels,
                                                             push.cellSize,
                                                             push.cellDimension,
                                                             ubo.cameraPos.xyz,
                                                             surfelBuffer,
                                                             cellInfoBuffer,
                                                             cellToSurfelBuffer);
            reflected += (1.0 - payload.metallic) * payload.baseColor / PI * cachedIncident;
        }
    }
    float3 radiance = reflected;
    float risWeight = reservoir.weightSum / max(luminance(reservoir.brdfWeight), 1e-5);
    float nDotSelected = max(dot(reservoir.direction, normal), 0.0);
    radiance *= risWeight * reservoir.brdfWeight * nDotSelected;
    radiance = clampLuminance(radiance, SURFEL_PT_MAX_RADIANCE_LUMINANCE);
    float luma = luminance(radiance);
    float variance = luma * luma * saturate(roughness);
    reflectionImages[halfPixel] = float4(radiance, variance);
}
