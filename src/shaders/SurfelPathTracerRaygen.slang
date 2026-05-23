#include "SurfelPathTracerCommon.slang"

struct SurfelRayTracePushConstants {
    uint rayCount;
    uint maxSurfels;
    uint enableGuidedSampling;
    uint irradianceAtlasWidth;
    uint activeMaxDepth;
    uint sleepingMaxDepth;
    uint enableSurfelTermination;
    uint maxSurfelSamplesPerQuery;
    uint cellDimension;
    float cellSize;
};

[[vk::push_constant]] SurfelRayTracePushConstants push;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;

[[vk::binding(4, 1)]] RWByteAddressBuffer counters;
[[vk::binding(5, 1)]] RWStructuredBuffer<SurfelPathTracerSurfel> surfelBuffer;
[[vk::binding(10, 1)]] RWStructuredBuffer<uint> rayBuffer;
[[vk::binding(11, 1)]] RWStructuredBuffer<SurfelPathTracerCellInfo> cellInfoBuffer;
[[vk::binding(13, 1)]] RWStructuredBuffer<uint> cellToSurfelBuffer;
[[vk::binding(14, 1)]] RWTexture2D<float4> irradianceAtlas;

[[vk::binding(0, 2)]] ConstantBuffer<UniformBuffer> ubo;

float luminanceAtAtlasCoord(uint2 coord, RWTexture2D<float4> atlas)
{
    return max(atlas[coord].a, 0.0);
}

bool hasGuidedSurfelHistory(SurfelPathTracerSurfel surfel)
{
    return surfel.shortMeanAndLife.w > 2.0;
}

float3 cosineSampleHemisphereLocal(float2 xi)
{
    float phi = 2.0 * PI * xi.x;
    float cosTheta = sqrt(max(1.0 - xi.y, 0.0));
    float sinTheta = sqrt(max(xi.y, 0.0));
    return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

bool sampleGuidedSurfelDirection(uint surfelIndex,
                                 float2 cdfXi,
                                 float2 jitterXi,
                                 RWTexture2D<float4> atlas,
                                 uint atlasWidth,
                                 out float3 localDirection,
                                 out float pdf)
{
    uint2 base = atlasTileBase(surfelIndex, atlasWidth);
    float2 centerJitter = float2(0.0, 0.0);
    float total = 0.0;
    for (uint y = 0u; y < SURFEL_PT_ATLAS_TILE_SIZE; ++y)
    for (uint x = 0u; x < SURFEL_PT_ATLAS_TILE_SIZE; ++x)
    {
        float2 texelCenter = float2(x, y) + 0.5 + centerJitter;
        float2 oct = clamp(texelCenter / float(SURFEL_PT_ATLAS_TILE_SIZE) * 2.0 - 1.0,
                           float2(-1.0),
                           float2(1.0));
        float3 texelDirection = octUvToDir(oct);
        if (texelDirection.z > 0.0)
        {
            total += luminanceAtAtlasCoord(base + uint2(x, y), atlas);
        }
    }

    if (total <= 1e-5)
    {
        localDirection = float3(0.0, 0.0, 1.0);
        pdf = 0.0;
        return false;
    }

    float threshold = cdfXi.x * total;
    float cumulative = 0.0;
    for (uint y = 0u; y < SURFEL_PT_ATLAS_TILE_SIZE; ++y)
    for (uint x = 0u; x < SURFEL_PT_ATLAS_TILE_SIZE; ++x)
    {
        float2 texelCenter = float2(x, y) + 0.5 + centerJitter;
        float2 centeredOct = clamp(texelCenter / float(SURFEL_PT_ATLAS_TILE_SIZE) * 2.0 - 1.0,
                                   float2(-1.0),
                                   float2(1.0));
        float3 centeredDirection = octUvToDir(centeredOct);
        if (centeredDirection.z <= 0.0)
        {
            continue;
        }

        float weight = luminanceAtAtlasCoord(base + uint2(x, y), atlas);
        cumulative += weight;
        if (cumulative >= threshold)
        {
            float2 texelJitter = jitterXi - 0.5;
            texelCenter = float2(x, y) + 0.5 + texelJitter;
            float2 oct = clamp(texelCenter / float(SURFEL_PT_ATLAS_TILE_SIZE) * 2.0 - 1.0,
                               float2(-1.0),
                               float2(1.0));
            localDirection = octUvToDir(oct);
            if (localDirection.z <= 0.0)
            {
                pdf = 0.0;
                return false;
            }
            pdf = max(weight / total, 1e-5);
            return true;
        }
    }

    localDirection = float3(0.0, 0.0, 1.0);
    pdf = 0.0;
    return false;
}

void storeRayRecord(uint rayIndex, uint surfelIndex, float3 radiance, float pdf, float hitT, float3 localRayDir) {
    uint base = rayIndex * SURFEL_PT_RAY_RECORD_UINTS;
    rayBuffer[base + SURFEL_PT_RAY_SURFEL_INDEX] = surfelIndex;
    rayBuffer[base + SURFEL_PT_RAY_RADIANCE_X] = asuint(radiance.x);
    rayBuffer[base + SURFEL_PT_RAY_RADIANCE_Y] = asuint(radiance.y);
    rayBuffer[base + SURFEL_PT_RAY_RADIANCE_Z] = asuint(radiance.z);
    rayBuffer[base + SURFEL_PT_RAY_PDF] = asuint(pdf);
    rayBuffer[base + SURFEL_PT_RAY_HIT_T] = asuint(hitT);
    rayBuffer[base + SURFEL_PT_RAY_FLAGS] = 1u;
    rayBuffer[base + SURFEL_PT_RAY_LOCAL_DIRECTION] = packRayLocalDirection(localRayDir);
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
    if (directSun <= 0.0 || sunVisibility <= 0.0)
    {
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
    if (cellIndex == SURFEL_PT_INVALID_INDEX)
    {
        return float3(0.0);
    }

    SurfelPathTracerCellInfo cellInfo = cells[cellIndex];
    uint count = min(cellInfo.surfelCount, maxSamples);
    float3 sum = float3(0.0);
    float weightSum = 0.0;
    for (uint i = 0u; i < count; ++i)
    {
        uint surfelIndex = cellToSurfels[cellInfo.surfelOffset + i];
        if (surfelIndex == SURFEL_PT_INVALID_INDEX || surfelIndex >= maxSurfels)
        {
            continue;
        }

        SurfelPathTracerSurfel surfel = surfels[surfelIndex];
        if (!isActiveSurfel(surfel))
        {
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

float3 traceSurfelPath(RayDesc initialRay, uint maxDepth, uint surfelIndex, uint seed, out float firstHitT)
{
    float3 throughput = float3(1.0);
    float3 radiance = float3(0.0);
    RayDesc ray = initialRay;
    firstHitT = -1.0;
    for (uint depth = 0u; depth < max(maxDepth, 1u); ++depth)
    {
        SurfelPathTracerPayload payload = makeEmptySurfelPayload();
        TraceRay(tlas, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);
        if (depth == 0u)
        {
            firstHitT = payload.hitT;
        }
        if (payload.hitKind == 0u)
        {
            radiance += throughput * payload.radiance;
            counters.InterlockedAdd(SURFEL_PT_COUNTER_PATH_MISSES_OFFSET, 1u);
            break;
        }

        float3 viewDir = normalize(-ray.Direction);
        float3 sunDir = normalize(-ubo.lightDir.xyz);
        float sunVisibility = traceSunVisibility(payload.hitPosition, payload.hitNormal, sunDir);
        radiance += throughput * evaluateShadowedDirectLighting(viewDir,
                                                                payload.hitNormal,
                                                                payload.baseColor,
                                                                payload.metallic,
                                                                payload.roughness,
                                                                payload.f0,
                                                                sunDir,
                                                                sunVisibility);
        radiance += throughput * payload.emission;

        if (push.enableSurfelTermination != 0u && depth + 1u >= maxDepth)
        {
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
            float3 diffuseBrdf = (1.0 - payload.metallic) * payload.baseColor / PI;
            radiance += throughput * diffuseBrdf * cachedIncident;
            counters.InterlockedAdd(SURFEL_PT_COUNTER_SURFEL_TERMINATED_PATHS_OFFSET, 1u);
            break;
        }

        float2 xi = float2(randomFloat(seed), randomFloat(seed));
        float3 normal = payload.hitNormal;
        float3 viewDirForBsdf = normalize(-ray.Direction);
        float3 F = fresnelSchlick(max(dot(normal, viewDirForBsdf), 0.0), payload.f0);
        float specProb = clamp((F.r + F.g + F.b) / 3.0 + payload.metallic * 0.5, 0.1, 0.9);
        float diffuseProb = 1.0 - specProb;
        float3 nextDir;
        float3 bsdfWeight;
        float pdf;
        if (randomFloat(seed) < specProb)
        {
            nextDir = normalize(ggxSampleDirection(xi, normal, viewDirForBsdf, payload.roughness));
            float nDotL = max(dot(normal, nextDir), 0.0);
            if (nDotL <= 0.0)
            {
                break;
            }
            float3 H = normalize(viewDirForBsdf + nextDir);
            float G = geometrySmith(normal, viewDirForBsdf, nextDir, payload.roughness);
            float3 Fs = fresnelSchlick(max(dot(H, viewDirForBsdf), 0.0), payload.f0);
            float nDotV = max(dot(normal, viewDirForBsdf), 0.0001);
            float nDotH = max(dot(normal, H), 0.0001);
            float vDotH = max(dot(viewDirForBsdf, H), 0.0001);
            bsdfWeight = (G * Fs * vDotH) / max(nDotV * nDotH, 0.0001);
            pdf = specProb;
        }
        else
        {
            nextDir = cosineSampleHemisphere(xi, normal);
            if (max(dot(normal, nextDir), 0.0) <= 0.0)
            {
                break;
            }
            float3 kD = (float3(1.0) - F) * (1.0 - payload.metallic);
            bsdfWeight = kD * payload.baseColor;
            pdf = diffuseProb;
        }
        throughput *= bsdfWeight / max(pdf, 0.0001);
        ray.Origin = payload.hitPosition + payload.hitNormal * SURFEL_PT_RAY_BIAS;
        ray.Direction = normalize(nextDir);
    }
    return clampLuminance(radiance, SURFEL_PT_MAX_RADIANCE_LUMINANCE);
}

[shader("raygeneration")]
void main() {
    uint rayIndex = DispatchRaysIndex().x;
    uint requestedRays = counters.Load(SURFEL_PT_COUNTER_REQUESTED_RAYS_OFFSET);
    uint runnableRays = min(push.rayCount, requestedRays);
    if (rayIndex >= runnableRays) {
        return;
    }

    uint surfelIndex = rayBuffer[rayIndex * SURFEL_PT_RAY_RECORD_UINTS + SURFEL_PT_RAY_SURFEL_INDEX];
    if (surfelIndex == SURFEL_PT_INVALID_INDEX || surfelIndex >= push.maxSurfels) {
        return;
    }

    SurfelPathTracerSurfel surfel = surfelBuffer[surfelIndex];
    if (!isActiveSurfel(surfel)) {
        return;
    }

    float3 normal = unpackNormalOctahedral(surfel.packedNormal);
    uint seed = pcgHash(rayIndex ^ (counters.Load(SURFEL_PT_COUNTER_FRAME_INDEX_OFFSET) * 1664525u));
    float2 xi = float2(randomFloat(seed), randomFloat(seed));
    float2 guideXi = float2(randomFloat(seed), randomFloat(seed));
    float2 guideJitter = float2(randomFloat(seed), randomFloat(seed));
    float3 localDir;
    float pdf;
    bool guided = push.enableGuidedSampling != 0u &&
                  hasGuidedSurfelHistory(surfel) &&
                  sampleGuidedSurfelDirection(surfelIndex, guideXi, guideJitter, irradianceAtlas, push.irradianceAtlasWidth, localDir, pdf);
    if (!guided)
    {
        localDir = cosineSampleHemisphereLocal(xi);
        pdf = max(localDir.z / PI, 1e-5);
        counters.InterlockedAdd(SURFEL_PT_COUNTER_COSINE_RAYS_OFFSET, 1u);
    }
    else
    {
        counters.InterlockedAdd(SURFEL_PT_COUNTER_GUIDED_RAYS_OFFSET, 1u);
    }

    float3 rayDir = localToWorldDirection(localDir, normal);
    float3 localRayDir = worldToLocalDirection(rayDir, normal);

    RayDesc ray;
    ray.Origin = surfel.position + normal * SURFEL_PT_RAY_BIAS;
    ray.Direction = normalize(rayDir);
    ray.TMin = SURFEL_PT_RAY_BIAS;
    ray.TMax = 10000.0;

    uint maxDepth = surfel.sleepState == SURFEL_PT_SLEEP_SLEEPING ? push.sleepingMaxDepth : push.activeMaxDepth;
    float firstHitT;
    float3 clampedRadiance = traceSurfelPath(ray, max(maxDepth, 1u), surfelIndex, seed, firstHitT);
    storeRayRecord(rayIndex, surfelIndex, clampedRadiance, max(pdf, 1e-5), firstHitT, localRayDir);
}
