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
    float surfelSupportRadius;
    uint useOriginalStyleGiNormalization;
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

float luminanceAtAtlasCoord(uint2 coord,
                            uint guideToken,
                            RWTexture2D<float4> atlas,
                            out bool isCurrentAllocation)
{
    float4 guideSample = atlas[coord];
    float decodedToken = -guideSample.a;
    isCurrentAllocation = isfinite(decodedToken) &&
                          decodedToken >= 1.0 &&
                          uint(round(decodedToken)) == guideToken;
    return isCurrentAllocation ? max(luminance(max(guideSample.rgb, float3(0.0))), 0.0) : 0.0;
}

// The guide uses six uniform octahedral UV bins per axis. Corner bins do not
// intersect the upper hemisphere; edge bins are triangles with half a texel of
// UV area. Solid angles are exact spherical polygon areas for those bins.
static const float kGuidedBinUvArea[36] = {
    0.000000000, 0.000000000, 0.055555556, 0.055555556, 0.000000000, 0.000000000,
    0.000000000, 0.055555556, 0.111111111, 0.111111111, 0.055555556, 0.000000000,
    0.055555556, 0.111111111, 0.111111111, 0.111111111, 0.111111111, 0.055555556,
    0.055555556, 0.111111111, 0.111111111, 0.111111111, 0.111111111, 0.055555556,
    0.000000000, 0.055555556, 0.111111111, 0.111111111, 0.055555556, 0.000000000,
    0.000000000, 0.000000000, 0.055555556, 0.055555556, 0.000000000, 0.000000000
};
static const float kGuidedBinSolidAngle[36] = {
    0.000000000, 0.000000000, 0.111341014, 0.111341014, 0.000000000, 0.000000000,
    0.000000000, 0.206128881, 0.412257761, 0.412257761, 0.206128881, 0.000000000,
    0.111341014, 0.412257761, 0.317469895, 0.317469895, 0.412257761, 0.111341014,
    0.111341014, 0.412257761, 0.317469895, 0.317469895, 0.412257761, 0.111341014,
    0.000000000, 0.206128881, 0.412257761, 0.412257761, 0.206128881, 0.000000000,
    0.000000000, 0.000000000, 0.111341014, 0.111341014, 0.000000000, 0.000000000
};
static const float kGuidedProposalProbability = 0.75;
static const uint kGuidedMinimumLife = 16u;
static const uint kGuidedMinimumValidBins = 8u;

uint guidedBinIndex(uint2 coord)
{
    return coord.y * SURFEL_PT_ATLAS_TILE_SIZE + coord.x;
}

float upperOctahedralJacobian(float2 uv)
{
    float3 unnormalizedDirection = float3(uv.x, uv.y, 1.0 - abs(uv.x) - abs(uv.y));
    float directionLength = max(length(unnormalizedDirection), 1e-5);
    // For n = normalize(u, v, 1 - |u| - |v|), the solid-angle
    // differential is du dv / |(u, v, 1 - |u| - |v|)|^3.
    return 1.0 / (directionLength * directionLength * directionLength);
}

float guidedBinWeight(uint2 localCoord,
                      uint2 atlasBase,
                      uint guideToken,
                      RWTexture2D<float4> atlas,
                      out bool isCurrentAllocation)
{
    uint binIndex = guidedBinIndex(localCoord);
    if (kGuidedBinSolidAngle[binIndex] <= 0.0)
    {
        isCurrentAllocation = false;
        return 0.0;
    }
    return luminanceAtAtlasCoord(atlasBase + localCoord,
                                 guideToken,
                                 atlas,
                                 isCurrentAllocation) * kGuidedBinSolidAngle[binIndex];
}

void guidedDistributionStats(uint surfelIndex,
                             uint guideToken,
                             RWTexture2D<float4> atlas,
                             uint atlasWidth,
                             out float total,
                             out uint validBinCount)
{
    uint2 base = atlasTileBase(surfelIndex, atlasWidth);
    total = 0.0;
    validBinCount = 0u;
    for (uint y = 0u; y < SURFEL_PT_ATLAS_TILE_SIZE; ++y)
    for (uint x = 0u; x < SURFEL_PT_ATLAS_TILE_SIZE; ++x)
    {
        uint binIndex = guidedBinIndex(uint2(x, y));
        if (kGuidedBinSolidAngle[binIndex] <= 0.0)
        {
            continue;
        }
        bool isCurrentAllocation;
        total += guidedBinWeight(uint2(x, y), base, guideToken, atlas, isCurrentAllocation);
        validBinCount += isCurrentAllocation ? 1u : 0u;
    }
}

float guidedPdfForDirection(uint surfelIndex,
                            uint guideToken,
                            float3 localDirection,
                            RWTexture2D<float4> atlas,
                            uint atlasWidth,
                            float distributionTotal)
{
    if (localDirection.z <= 0.0 || distributionTotal <= 1e-5)
    {
        return 0.0;
    }

    uint2 localCoord = atlasDirectionCoord(localDirection);
    uint binIndex = guidedBinIndex(localCoord);
    float uvArea = kGuidedBinUvArea[binIndex];
    if (uvArea <= 0.0)
    {
        return 0.0;
    }

    uint2 base = atlasTileBase(surfelIndex, atlasWidth);
    bool isCurrentAllocation;
    float probabilityMass = guidedBinWeight(localCoord,
                                            base,
                                            guideToken,
                                            atlas,
                                            isCurrentAllocation) / distributionTotal;
    float jacobian = upperOctahedralJacobian(dirToOctUv(localDirection));
    return probabilityMass / max(uvArea * jacobian, 1e-5);
}

bool hasGuidedSurfelHistory(SurfelPathTracerSurfel surfel)
{
    return surfel.irradianceAtlasOffset != 0u &&
           surfel.shortMeanAndLife.w > float(kGuidedMinimumLife);
}

float3 cosineSampleHemisphereLocal(float2 xi)
{
    float phi = 2.0 * PI * xi.x;
    float cosTheta = sqrt(max(1.0 - xi.y, 0.0));
    float sinTheta = sqrt(max(xi.y, 0.0));
    return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

bool sampleGuidedSurfelDirection(uint surfelIndex,
                                 uint guideToken,
                                 float cdfXi,
                                 float2 jitterXi,
                                 RWTexture2D<float4> atlas,
                                 uint atlasWidth,
                                 float distributionTotal,
                                 out float3 localDirection,
                                 out float pdf)
{
    uint2 base = atlasTileBase(surfelIndex, atlasWidth);
    if (distributionTotal <= 1e-5)
    {
        localDirection = float3(0.0, 0.0, 1.0);
        pdf = 0.0;
        return false;
    }

    float threshold = cdfXi * distributionTotal;
    float cumulative = 0.0;
    for (uint y = 0u; y < SURFEL_PT_ATLAS_TILE_SIZE; ++y)
    for (uint x = 0u; x < SURFEL_PT_ATLAS_TILE_SIZE; ++x)
    {
        uint2 localCoord = uint2(x, y);
        uint binIndex = guidedBinIndex(localCoord);
        if (kGuidedBinSolidAngle[binIndex] <= 0.0)
        {
            continue;
        }
        bool isCurrentAllocation;
        float weight = guidedBinWeight(localCoord,
                                       base,
                                       guideToken,
                                       atlas,
                                       isCurrentAllocation);
        if (weight <= 0.0)
        {
            continue;
        }

        cumulative += weight;
        if (threshold < cumulative)
        {
            float2 oct = ((float2(x, y) + saturate(jitterXi)) /
                          float(SURFEL_PT_ATLAS_TILE_SIZE)) * 2.0 - 1.0;
            float outsideUpperDiamond = abs(oct.x) + abs(oct.y) - 1.0;
            if (outsideUpperDiamond > 0.0)
            {
                float2 octSign = float2(oct.x >= 0.0 ? 1.0 : -1.0,
                                        oct.y >= 0.0 ? 1.0 : -1.0);
                oct -= outsideUpperDiamond * octSign;
            }
            localDirection = octUvToDir(oct);
            float jacobian = upperOctahedralJacobian(oct);
            float probabilityMass = weight / distributionTotal;
            pdf = probabilityMass / max(kGuidedBinUvArea[binIndex] * jacobian, 1e-5);
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

float3 traceSurfelPath(RayDesc initialRay,
                       uint maxDepth,
                       uint seed,
                       out float firstHitT)
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
            counters.InterlockedAdd(SURFEL_PT_COUNTER_SURFEL_TERMINATION_ATTEMPTS_OFFSET, 1u);
            float3 cachedIncident;
            bool cacheHit = sampleSurfelIrradianceCell(payload.hitPosition,
                                                       payload.hitNormal,
                                                       push.maxSurfelSamplesPerQuery,
                                                       seed ^ (depth * 747796405u),
                                                       SURFEL_PT_INVALID_INDEX,
                                                       1u /* full 3x3x3 neighbourhood */,
                                                       push.maxSurfels,
                                                       push.cellSize,
                                                       push.surfelSupportRadius,
                                                       push.cellDimension,
                                                       ubo.cameraPos.xyz,
                                                       surfelBuffer,
                                                       cellInfoBuffer,
                                                       cellToSurfelBuffer,
                                                       cachedIncident);
            if (cacheHit)
            {
                float diffuseNormalization = push.useOriginalStyleGiNormalization != 0u ? 1.0 : (1.0 / PI);
                float3 diffuseBrdf = (1.0 - payload.metallic) * payload.baseColor * diffuseNormalization;
                radiance += throughput * diffuseBrdf * cachedIncident;
                counters.InterlockedAdd(SURFEL_PT_COUNTER_SURFEL_TERMINATION_HITS_OFFSET, 1u);
                break;
            }
        }

        if (depth + 1u >= maxDepth)
        {
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
    bool guideAvailable = push.enableGuidedSampling != 0u && hasGuidedSurfelHistory(surfel);
    float guideTotal = 0.0;
    uint validGuideBins = 0u;
    if (guideAvailable)
    {
        guidedDistributionStats(surfelIndex,
                                surfel.irradianceAtlasOffset,
                                irradianceAtlas,
                                push.irradianceAtlasWidth,
                                guideTotal,
                                validGuideBins);
    }
    guideAvailable = guideAvailable &&
                     guideTotal > 1e-5 &&
                     validGuideBins >= kGuidedMinimumValidBins;
    float guidedComponentPdf = 0.0;
    bool guided = guideAvailable &&
                  guideXi.y < kGuidedProposalProbability &&
                  sampleGuidedSurfelDirection(surfelIndex,
                                              surfel.irradianceAtlasOffset,
                                              guideXi.x,
                                              guideJitter,
                                              irradianceAtlas,
                                              push.irradianceAtlasWidth,
                                              guideTotal,
                                              localDir,
                                              guidedComponentPdf);
    if (!guided)
    {
        localDir = cosineSampleHemisphereLocal(xi);
        float cosinePdf = max(localDir.z / PI, 0.0);
        guidedComponentPdf = guideAvailable
                             ? guidedPdfForDirection(surfelIndex,
                                                     surfel.irradianceAtlasOffset,
                                                     localDir,
                                                     irradianceAtlas,
                                                     push.irradianceAtlasWidth,
                                                     guideTotal)
                             : 0.0;
        pdf = guideAvailable
              ? kGuidedProposalProbability * guidedComponentPdf +
                (1.0 - kGuidedProposalProbability) * cosinePdf
              : cosinePdf;
        counters.InterlockedAdd(SURFEL_PT_COUNTER_COSINE_RAYS_OFFSET, 1u);
    }
    else
    {
        float cosinePdf = max(localDir.z / PI, 0.0);
        pdf = kGuidedProposalProbability * guidedComponentPdf +
              (1.0 - kGuidedProposalProbability) * cosinePdf;
        counters.InterlockedAdd(SURFEL_PT_COUNTER_GUIDED_RAYS_OFFSET, 1u);
    }

    float3 rayDir = localToWorldDirection(localDir, normal);
    float3 localRayDir = worldToLocalDirection(rayDir, normal);

    RayDesc ray;
    ray.Origin = surfel.position + normal * SURFEL_PT_RAY_BIAS;
    ray.Direction = normalize(rayDir);
    ray.TMin = SURFEL_PT_RAY_BIAS;
    ray.TMax = 10000.0;

    uint maxDepth = max(surfel.sleepState == SURFEL_PT_SLEEP_SLEEPING ?
                            push.sleepingMaxDepth : push.activeMaxDepth,
                        1u);
    float firstHitT;
    float3 clampedRadiance = traceSurfelPath(ray,
                                             maxDepth,
                                             seed,
                                             firstHitT);
    storeRayRecord(rayIndex, surfelIndex, clampedRadiance, max(pdf, 1e-5), firstHitT, localRayDir);
}
