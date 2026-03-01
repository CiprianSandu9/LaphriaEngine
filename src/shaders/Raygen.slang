#include "ShaderCommon.slang"

[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;
[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(1, 0)]] RWTexture2D<float4> outputImage;



[shader("raygeneration")]
void main() {
    uint2 launchID = DispatchRaysIndex().xy;
    uint2 launchSize = DispatchRaysDimensions().xy;

    float2 pixelCenter = float2(launchID) + float2(0.5, 0.5);
    float2 inUV = pixelCenter / float2(launchSize);
    float2 d = inUV * 2.0 - 1.0;
    // Use precomputed inverse matrices from the UBO; inverse() is unavailable in SPIRV targets.
    float4x4 invView = ubo.viewInverse;
    float4x4 invProj = ubo.projInverse;

    // Rasterizer uses a negative viewport height to flip Y (OpenGL style).
    // We must mirror this mathematically in the Ray Tracer's NDC ray unprojection. 
    float4 target = mul(invProj, float4(d.x, -d.y, 1.0, 1.0));
    float3 rayDir = mul(invView, float4(normalize(target.xyz / target.w), 0.0)).xyz;
    float3 origin = ubo.cameraPos.xyz;

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = normalize(rayDir);
    ray.TMin = 0.001;
    ray.TMax = 10000.0;

    RayPayload payload;
    payload.color = float3(0.0, 0.0, 0.0);
    payload.isShadowed = 0;

    // Call TraceRay
    TraceRay(tlas, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);

    outputImage[launchID] = float4(payload.color, 1.0);
}
