#include "ShaderCommon.slang"

[[vk::binding(0, 0)]]
ConstantBuffer<UniformBuffer> ubo;

[[vk::push_constant]]
ScenePushConstants push;

// Depth-only pass output: clip-space position written to the depth buffer; UV forwarded for alpha cutout.
struct VSOutput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

[shader("vertex")]
VSOutput shadowVert(
    [[vk::location(0)]] float3 inPosition,
    [[vk::location(1)]] float3 inNormal,
    [[vk::location(2)]] float4 inTangent,
    [[vk::location(3)]] float2 inTexCoord,
    [[vk::location(4)]] float3 inColor)
{
    VSOutput output;
    float4 worldPos = mul(push.modelMatrix, float4(inPosition, 1.0));
    output.position = mul(ubo.cascadeViewProj[push.cascadeIndex], worldPos);
    output.texCoord = inTexCoord;
    return output;
}

[[vk::binding(0, 1)]] StructuredBuffer<MaterialData> materialBuffer;
[[vk::binding(1, 1)]] Sampler2D textures[];   // combined image-sampler — matches the material DSL

[shader("fragment")]
void shadowFrag(VSOutput input)
{
    MaterialData material = materialBuffer[push.materialIndex];
    float alpha = material.baseColorFactor.a;

    if (material.baseColorIndex >= 0) {
        float4 sampled = textures[NonUniformResourceIndex(material.baseColorIndex)].Sample(input.texCoord);
        alpha *= sampled.a;
    }

    if (alpha < material.alphaCutoff) {
        discard;
    }
}
