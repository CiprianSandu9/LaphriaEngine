#include "ShaderCommon.slang"

struct SkinningPushConstantsCS {
    uint vertexCount;
    uint jointMatrixOffset;
    uint jointCount;
    uint _pad;
};

[[vk::binding(0, 0)]] ByteAddressBuffer inputVertices;
[[vk::binding(1, 0)]] RWByteAddressBuffer outputVertices;
[[vk::binding(2, 0)]] ByteAddressBuffer influences;
[[vk::binding(3, 0)]] StructuredBuffer<float4x4> jointMatrices;

[[vk::push_constant]] SkinningPushConstantsCS push;

static const uint kVertexStride     = 60; // Must match C++ sizeof(Vertex)
static const uint kInfluenceStride  = 48; // Must match C++ sizeof(SkinningInfluence)

static const uint kPosOffset        = 0;
static const uint kNormalOffset     = 12;
static const uint kTangentOffset    = 24;
static const uint kTangentWOffset   = 36;

static const uint kInfluenceJointsOffset  = 0;
static const uint kInfluenceWeightsOffset = 16;

float loadFloat(ByteAddressBuffer buffer, uint offset) {
    return asfloat(buffer.Load(offset));
}

float3 loadFloat3(ByteAddressBuffer buffer, uint offset) {
    return float3(
        asfloat(buffer.Load(offset + 0)),
        asfloat(buffer.Load(offset + 4)),
        asfloat(buffer.Load(offset + 8)));
}

uint4 loadUInt4(ByteAddressBuffer buffer, uint offset) {
    return uint4(
        buffer.Load(offset + 0),
        buffer.Load(offset + 4),
        buffer.Load(offset + 8),
        buffer.Load(offset + 12));
}

float4 loadFloat4(ByteAddressBuffer buffer, uint offset) {
    return float4(
        asfloat(buffer.Load(offset + 0)),
        asfloat(buffer.Load(offset + 4)),
        asfloat(buffer.Load(offset + 8)),
        asfloat(buffer.Load(offset + 12)));
}

void storeFloat3(RWByteAddressBuffer buffer, uint offset, float3 value) {
    buffer.Store(offset + 0, asuint(value.x));
    buffer.Store(offset + 4, asuint(value.y));
    buffer.Store(offset + 8, asuint(value.z));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void skinningMain(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint vertexIndex = dispatchThreadID.x;
    if (vertexIndex >= push.vertexCount) {
        return;
    }

    uint vertexBaseOffset = vertexIndex * kVertexStride;
    uint influenceBaseOffset = vertexIndex * kInfluenceStride;

    [unroll]
    for (uint word = 0; word < (kVertexStride / 4); ++word) {
        uint byteOffset = vertexBaseOffset + word * 4;
        outputVertices.Store(byteOffset, inputVertices.Load(byteOffset));
    }

    float3 sourcePos = loadFloat3(inputVertices, vertexBaseOffset + kPosOffset);
    float3 sourceNormal = loadFloat3(inputVertices, vertexBaseOffset + kNormalOffset);
    float3 sourceTangent = loadFloat3(inputVertices, vertexBaseOffset + kTangentOffset);
    float sourceTangentW = loadFloat(inputVertices, vertexBaseOffset + kTangentWOffset);

    uint4 joints = loadUInt4(influences, influenceBaseOffset + kInfluenceJointsOffset);
    float4 weights = loadFloat4(influences, influenceBaseOffset + kInfluenceWeightsOffset);

    float totalWeight = weights.x + weights.y + weights.z + weights.w;
    if (totalWeight <= 1e-6) {
        return;
    }

    float4 normalizedWeights = weights / totalWeight;
    float4x4 skinMatrix = float4x4(0.0);

    [unroll]
    for (uint i = 0; i < 4; ++i) {
        uint jointIndex = joints[i];
        if (jointIndex < push.jointCount) {
            uint paletteIndex = push.jointMatrixOffset + jointIndex;
            skinMatrix += jointMatrices[paletteIndex] * normalizedWeights[i];
        }
    }

    float4 skinnedPos = mul(skinMatrix, float4(sourcePos, 1.0));
    float3x3 skinMat3 = (float3x3)skinMatrix;
    float3x3 normalMatrix = transpose(mat3Inverse(skinMat3));
    float3 skinnedNormal = normalize(mul(normalMatrix, sourceNormal));
    float3 skinnedTangent = normalize(mul(skinMat3, sourceTangent));

    storeFloat3(outputVertices, vertexBaseOffset + kPosOffset, skinnedPos.xyz);
    storeFloat3(outputVertices, vertexBaseOffset + kNormalOffset, skinnedNormal);
    storeFloat3(outputVertices, vertexBaseOffset + kTangentOffset, skinnedTangent);
    outputVertices.Store(vertexBaseOffset + kTangentWOffset, asuint(sourceTangentW));
}
