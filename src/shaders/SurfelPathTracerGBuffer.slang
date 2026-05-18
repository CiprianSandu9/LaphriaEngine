#include "SurfelPathTracerCommon.slang"

struct SurfelPathTracerGBufferPayload {
    float hitT;
    uint modelId;
    uint materialIndex;
    float3 worldNormal;
    float3 baseColor;
};

[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;

[[vk::binding(0, 1)]] RWTexture2D<float4> outputImage;
[[vk::binding(1, 1)]] RWTexture2D<float4> gBufferNormal;
[[vk::binding(2, 1)]] RWTexture2D<float> gBufferDepth;
[[vk::binding(3, 1)]] RWTexture2D<float4> gBufferMotionMaterial;
[[vk::binding(20, 1)]] RWTexture2D<float4> gBufferAlbedo;

[[vk::binding(0, 2)]] ConstantBuffer<UniformBuffer> ubo;

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

    SurfelPathTracerGBufferPayload payload;
    payload.hitT = -1.0;
    payload.modelId = 0u;
    payload.materialIndex = 0u;
    payload.worldNormal = float3(0.0, 0.0, 0.0);
    payload.baseColor = float3(0.0, 0.0, 0.0);

    TraceRay(tlas, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);

    if (payload.hitT < 0.0) {
        float3 sunDir = normalize(-ubo.lightDir.xyz);
        outputImage[launchID] = float4(evalSkyColor(normalize(ray.Direction), sunDir), 1.0);
        gBufferNormal[launchID] = float4(0.0, 0.0, 0.0, 0.0);
        gBufferDepth[launchID] = -1.0;
        gBufferMotionMaterial[launchID] = float4(0.0, 0.0, 0.0, 0.0);
        gBufferAlbedo[launchID] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    float3 hitPos = ray.Origin + ray.Direction * payload.hitT;
    float4 prevClip = mul(ubo.prevViewProj, float4(hitPos, 1.0));
    float2 prevNDC = (prevClip.xy / prevClip.w) * float2(0.5, -0.5) + 0.5;
    float2 unjitteredUV = (float2(launchID) + 0.5) / float2(launchSize);
    float2 motion = unjitteredUV - prevNDC;

    float3 normal = normalize(payload.worldNormal);
    outputImage[launchID] = float4(payload.baseColor, 1.0);
    gBufferNormal[launchID] = float4(normal, 0.0);
    gBufferDepth[launchID] = payload.hitT;
    gBufferMotionMaterial[launchID] = float4(motion, float(payload.modelId), float(payload.materialIndex));
    gBufferAlbedo[launchID] = float4(payload.baseColor, 1.0);
}
