#include "SurfelPathTracerCommon.slang"

struct SurfelReferencePushConstants {
    uint width;
    uint height;
    uint maxDepth;
    uint enabled;
};

[[vk::push_constant]] SurfelReferencePushConstants push;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;

[[vk::binding(27, 1)]] RWTexture2D<float4> referenceImage;

[[vk::binding(0, 2)]] ConstantBuffer<UniformBuffer> ubo;

float3 primaryRayDirection(uint2 pixel)
{
    float2 uv = (float2(pixel) + float2(0.5, 0.5)) / float2(max(push.width, 1u), max(push.height, 1u));
    float2 d = uv * 2.0 - 1.0;
    float4 target = mul(ubo.projInverse, float4(d.x, -d.y, 1.0, 1.0));
    return normalize(mul(ubo.viewInverse, float4(normalize(target.xyz / target.w), 0.0)).xyz);
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

bool sampleReferenceContinuation(inout uint seed,
                                 float3 normal,
                                 float3 viewDir,
                                 float3 baseColor,
                                 float metallic,
                                 float roughness,
                                 float3 f0,
                                 out float3 nextDir,
                                 out float3 bsdfWeight,
                                 out float pdf)
{
    nextDir = normal;
    bsdfWeight = float3(0.0);
    pdf = 0.0;

    float2 xi = float2(randomFloat(seed), randomFloat(seed));
    float3 F = fresnelSchlick(max(dot(normal, viewDir), 0.0), f0);
    float specProb = clamp((F.r + F.g + F.b) / 3.0 + metallic * 0.5, 0.1, 0.9);
    float diffuseProb = 1.0 - specProb;

    if (randomFloat(seed) < specProb) {
        nextDir = normalize(ggxSampleDirection(xi, normal, viewDir, roughness));
        float nDotL = max(dot(normal, nextDir), 0.0);
        if (nDotL <= 0.0) {
            return false;
        }

        float3 H = normalize(viewDir + nextDir);
        float G = geometrySmith(normal, viewDir, nextDir, roughness);
        float3 Fs = fresnelSchlick(max(dot(H, viewDir), 0.0), f0);
        float nDotV = max(dot(normal, viewDir), 0.0001);
        float nDotH = max(dot(normal, H), 0.0001);
        float vDotH = max(dot(viewDir, H), 0.0001);
        bsdfWeight = (G * Fs * vDotH) / max(nDotV * nDotH, 0.0001);
        pdf = specProb;
        return true;
    }

    nextDir = cosineSampleHemisphere(xi, normal);
    if (max(dot(normal, nextDir), 0.0) <= 0.0) {
        return false;
    }

    float3 kD = (float3(1.0) - F) * (1.0 - metallic);
    bsdfWeight = kD * baseColor;
    pdf = diffuseProb;
    return true;
}

float3 traceReferencePath(RayDesc initialRay, uint maxDepth, uint seed)
{
    float3 throughput = float3(1.0);
    float3 radiance = float3(0.0);
    RayDesc ray = initialRay;
    uint depthCount = max(maxDepth, 1u);

    for (uint depth = 0u; depth < depthCount; ++depth) {
        SurfelPathTracerPayload payload = makeEmptySurfelPayload();
        TraceRay(tlas, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);
        if (payload.hitKind == 0u) {
            radiance += throughput * payload.radiance;
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

        float3 nextDir;
        float3 bsdfWeight;
        float pdf;
        if (!sampleReferenceContinuation(seed,
                                         payload.hitNormal,
                                         viewDir,
                                         payload.baseColor,
                                         payload.metallic,
                                         payload.roughness,
                                         payload.f0,
                                         nextDir,
                                         bsdfWeight,
                                         pdf)) {
            break;
        }

        throughput *= bsdfWeight / max(pdf, 0.0001);
        ray.Origin = payload.hitPosition + payload.hitNormal * SURFEL_PT_RAY_BIAS;
        ray.Direction = normalize(nextDir);
        ray.TMin = SURFEL_PT_RAY_BIAS;
        ray.TMax = 10000.0;
    }

    return clampLuminance(radiance, SURFEL_PT_MAX_RADIANCE_LUMINANCE);
}

[shader("raygeneration")]
void main()
{
    uint2 pixel = DispatchRaysIndex().xy;
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    if (push.enabled == 0u) {
        referenceImage[pixel] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    RayDesc ray;
    ray.Origin = ubo.cameraPos.xyz;
    ray.Direction = primaryRayDirection(pixel);
    ray.TMin = SURFEL_PT_RAY_BIAS;
    ray.TMax = 10000.0;

    uint seed = pcgHash(pixel.x * 1973u ^ pixel.y * 9277u ^ ubo.frameCount * 26699u);
    float3 referenceRadiance = traceReferencePath(ray, push.maxDepth, seed);
    referenceImage[pixel] = float4(referenceRadiance, 1.0);
}
