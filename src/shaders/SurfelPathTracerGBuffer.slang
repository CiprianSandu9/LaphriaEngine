#include "SurfelPathTracerCommon.slang"

struct SurfelPathTracerGBufferPayload {
    float hitT;
    uint modelId;
    uint materialIndex;
    float3 worldNormal;
    float3 baseColor;
    float roughness;
    float metallic;
    float dielectricSpec;
    float ao;
    float3 emissive;
    uint sourceNodeId;
    uint sourceInstanceId;
    uint sourceInstanceCustomIndex;
    uint sourcePrimitiveIndex;
    float2 sourceBarycentrics;
    float3 sourceObjectPosition;
    float3 sourceObjectNormal;
};

[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;

[[vk::binding(0, 1)]] RWTexture2D<float4> outputImage;
[[vk::binding(1, 1)]] RWTexture2D<float4> gBufferNormal;
[[vk::binding(2, 1)]] RWTexture2D<float> gBufferDepth;
[[vk::binding(3, 1)]] RWTexture2D<float4> gBufferMotionMaterial;
[[vk::binding(20, 1)]] RWTexture2D<float4> gBufferAlbedo;
[[vk::binding(21, 1)]] RWTexture2D<float4> gBufferMaterial;
[[vk::binding(22, 1)]] RWTexture2D<float4> gBufferEmissive;
[[vk::binding(28, 1)]] RWStructuredBuffer<SurfelPathTracerPixelSource> gBufferSourceBuffer;

[[vk::binding(0, 2)]] ConstantBuffer<UniformBuffer> ubo;

SurfelPathTracerGBufferPayload makeEmptyGBufferPayload()
{
    SurfelPathTracerGBufferPayload payload;
    payload.hitT = -1.0;
    payload.modelId = 0u;
    payload.materialIndex = 0u;
    payload.worldNormal = float3(0.0, 0.0, 0.0);
    payload.baseColor = float3(0.0, 0.0, 0.0);
    payload.roughness = 1.0;
    payload.metallic = 0.0;
    payload.dielectricSpec = 1.0;
    payload.ao = 1.0;
    payload.emissive = float3(0.0);
    payload.sourceNodeId = SURFEL_PT_INVALID_INDEX;
    payload.sourceInstanceId = SURFEL_PT_INVALID_INDEX;
    payload.sourceInstanceCustomIndex = SURFEL_PT_INVALID_INDEX;
    payload.sourcePrimitiveIndex = SURFEL_PT_INVALID_INDEX;
    payload.sourceBarycentrics = float2(0.0);
    payload.sourceObjectPosition = float3(0.0);
    payload.sourceObjectNormal = float3(0.0);
    return payload;
}

SurfelPathTracerGBufferPayload makeVisibilityGBufferPayload()
{
    SurfelPathTracerGBufferPayload payload = makeEmptyGBufferPayload();
    payload.hitT = 1.0;
    return payload;
}

float traceSunVisibility(float3 position, float3 normal, float3 sunDir)
{
    RayDesc shadowRay;
    float bias = max(SURFEL_PT_RAY_BIAS * 4.0, 0.005);
    shadowRay.Origin = position + normal * bias;
    shadowRay.Direction = normalize(sunDir);
    shadowRay.TMin = bias;
    shadowRay.TMax = 10000.0;

    SurfelPathTracerGBufferPayload shadowPayload = makeVisibilityGBufferPayload();
    TraceRay(tlas,
             RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
             0xFF, 0, 0, 0, shadowRay, shadowPayload);
    return shadowPayload.hitT < 0.0 ? 1.0 : 0.0;
}

[shader("raygeneration")]
void main() {
    uint2 launchID = DispatchRaysIndex().xy;
    uint2 launchSize = DispatchRaysDimensions().xy;

    float2 pixelCenter = float2(launchID) + float2(0.5, 0.5);
    float2 inUV = pixelCenter / float2(launchSize);
    float2 d = inUV * 2.0 - 1.0;

    float4 target = mul(ubo.projInverse, float4(d.x, -d.y, 1.0, 1.0));
    float3 rayDir = mul(ubo.viewInverse, float4(normalize(target.xyz / target.w), 0.0)).xyz;

    RayDesc ray;
    ray.Origin = ubo.cameraPos.xyz;
    ray.Direction = normalize(rayDir);
    ray.TMin = SURFEL_PT_RAY_BIAS;
    ray.TMax = 10000.0;

    SurfelPathTracerGBufferPayload payload = makeEmptyGBufferPayload();

    TraceRay(tlas, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);

    if (payload.hitT < 0.0) {
        float3 sunDir = normalize(-ubo.lightDir.xyz);
        outputImage[launchID] = float4(evalSkyColor(normalize(ray.Direction), sunDir), 1.0);
        gBufferNormal[launchID] = float4(0.0, 0.0, 0.0, 0.0);
        gBufferDepth[launchID] = -1.0;
        gBufferMotionMaterial[launchID] = float4(0.0, 0.0, 0.0, 0.0);
        gBufferAlbedo[launchID] = float4(0.0, 0.0, 0.0, 0.0);
        gBufferMaterial[launchID] = float4(1.0, 0.0, 1.0, 1.0);
        gBufferEmissive[launchID] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    float3 hitPos = ray.Origin + ray.Direction * payload.hitT;
    float4 prevClip = mul(ubo.prevViewProj, float4(hitPos, 1.0));
    float2 prevNDC = (prevClip.xy / prevClip.w) * float2(0.5, -0.5) + 0.5;
    float2 unjitteredUV = (float2(launchID) + 0.5) / float2(launchSize);
    float2 motion = unjitteredUV - prevNDC;

    float3 normal = normalize(payload.worldNormal);
    float3 sunDir = normalize(-ubo.lightDir.xyz);
    float directSun = max(dot(normal, sunDir), 0.0);
    float sunVisibility = directSun > 0.0 ? traceSunVisibility(hitPos, normal, sunDir) : 0.0;
    outputImage[launchID] = float4(payload.baseColor, 1.0);
    gBufferNormal[launchID] = float4(normal, sunVisibility);
    gBufferDepth[launchID] = payload.hitT;
    gBufferMotionMaterial[launchID] = float4(motion, float(payload.modelId), float(payload.materialIndex));
    gBufferAlbedo[launchID] = float4(payload.baseColor, 1.0);
    gBufferMaterial[launchID] = float4(payload.roughness, payload.metallic, payload.dielectricSpec, payload.ao);
    gBufferEmissive[launchID] = float4(payload.emissive, 0.0);
}
