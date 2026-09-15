#include "SurfelPathTracerCommon.slang"

[[vk::binding(0, 0)]] RWTexture2D<float4> outputImage;
[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint width;
    uint height;
    outputImage.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    float2 pixelCenter = float2(tid.xy) + float2(0.5, 0.5);
    float2 inUV = pixelCenter / float2(width, height);
    float2 d = inUV * 2.0 - 1.0;

    float4 target = mul(ubo.projInverse, float4(d.x, -d.y, 1.0, 1.0));
    float3 rayDir = mul(ubo.viewInverse, float4(normalize(target.xyz / target.w), 0.0)).xyz;
    float3 sunDir = normalize(-ubo.lightDir.xyz);
    outputImage[tid.xy] = float4(evalSkyColor(normalize(rayDir), sunDir), 1.0);
}
