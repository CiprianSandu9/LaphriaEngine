#include "SurfelPathTracerCommon.slang"

struct SurfelPathTracerPayload {
    float3 radiance;
    float hitT;
};

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
    uint pad0;
    uint pad1;
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

float3 sampleSurfelRadiance(float3 position, float3 normal) {
    uint dim = max(push.cellDimension, 1u);
    int3 centerCoord = cameraRelativeCellCoord(position, ubo.cameraPos.xyz, push.cellSize);
    if (!isCellCoordValid(centerCoord, dim)) {
        return float3(0.0);
    }

    float3 radiance = float3(0.0);
    float weightSum = 0.0;
    for (int z = -1; z <= 1; ++z) {
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                int3 coord = centerCoord + int3(x, y, z);
                if (!isCellCoordValid(coord, dim)) {
                    continue;
                }

                uint cellIndex = flattenCameraRelativeCellCoord(coord, dim);
                SurfelPathTracerCellInfo cellInfo = cellInfoBuffer[cellIndex];
                for (uint slot = 0u; slot < cellInfo.surfelCount; ++slot) {
                    uint surfelIndex = cellToSurfelBuffer[cellInfo.surfelOffset + slot];
                    if (surfelIndex == SURFEL_PT_INVALID_INDEX || surfelIndex >= push.maxSurfels) {
                        continue;
                    }

                    SurfelPathTracerSurfel surfel = surfelBuffer[surfelIndex];
                    if (surfel.flags == 0u || surfel.radius <= 0.0 || !isFinite3(surfel.position)) {
                        continue;
                    }

                    float3 surfelNormal = unpackNormalOctahedral(surfel.packedNormal);
                    float normalWeight = saturate(dot(normal, surfelNormal));
                    float distanceWeight = saturate(1.0 - length(position - surfel.position) / max(surfel.radius * 2.0, 0.0001));
                    float weight = normalWeight * distanceWeight;
                    radiance += clampLuminance(surfel.radiance, SURFEL_PT_MAX_RADIANCE_LUMINANCE) * weight;
                    weightSum += weight;
                }
            }
        }
    }
    return weightSum > 0.0001 ? radiance / weightSum : float3(0.0);
}

MaterialData loadMaterial(float4 motionMaterial) {
    uint modelId = uint(max(motionMaterial.z, 0.0) + 0.5);
    uint materialIndex = uint(max(motionMaterial.w, 0.0) + 0.5);
    return globalMaterials[NonUniformResourceIndex(modelId)][materialIndex];
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
    MaterialData material = loadMaterial(gBufferMotionMaterial[fullPixel]);
    float roughness = clamp(material.roughnessFactor, MIN_ROUGHNESS, 1.0);
    float metallic = saturate(material.metallicFactor);
    float3 baseColor = max(material.baseColorFactor.rgb, float3(0.0));
    float3 f0 = lerp(float3(0.04), baseColor, metallic);

    uint seed = pcgHash(halfPixel.x * 1973u ^ halfPixel.y * 9277u ^ push.frameIndex * 26699u ^ ubo.frameCount);
    float2 xi = float2(randomFloat(seed), randomFloat(seed));
    float3 rayDir = normalize(ggxSampleDirection(xi, normal, viewDir, roughness));
    if (dot(rayDir, normal) <= 0.0) {
        rayDir = reflect(-viewDir, normal);
    }

    float3 position = reconstructWorldPosition(fullPixel, depth);
    RayDesc ray;
    ray.Origin = position + normal * SURFEL_PT_RAY_BIAS;
    ray.Direction = normalize(rayDir);
    ray.TMin = SURFEL_PT_RAY_BIAS;
    ray.TMax = 10000.0;

    SurfelPathTracerPayload payload;
    payload.radiance = float3(0.0);
    payload.hitT = -1.0;
    TraceRay(tlas, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);

    float3 indirectTermination = sampleSurfelRadiance(position, normal);
    float3 reflected = payload.hitT >= 0.0 ? payload.radiance + indirectTermination * 0.35 : payload.radiance;
    float3 fresnel = fresnelSchlickRoughness(saturate(dot(normal, viewDir)), f0, roughness);
    float3 radiance = clampLuminance(reflected * fresnel, SURFEL_PT_MAX_RADIANCE_LUMINANCE);
    float luma = luminance(radiance);
    float variance = luma * luma * saturate(roughness);
    reflectionImages[halfPixel] = float4(radiance, variance);
}
