#include "SurfelPathTracerCommon.slang"

struct SurfelPathTracerPayload {
    float3 radiance;
    float hitT;
};

struct SurfelRayTracePushConstants {
    uint rayCount;
    uint maxSurfels;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] SurfelRayTracePushConstants push;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;

[[vk::binding(4, 1)]] RWByteAddressBuffer counters;
[[vk::binding(5, 1)]] RWStructuredBuffer<SurfelPathTracerSurfel> surfelBuffer;
[[vk::binding(10, 1)]] RWStructuredBuffer<uint> rayBuffer;

[[vk::binding(0, 2)]] ConstantBuffer<UniformBuffer> ubo;

void storeRayRecord(uint rayIndex, uint surfelIndex, float3 radiance, float pdf, float hitT) {
    uint base = rayIndex * SURFEL_PT_RAY_RECORD_UINTS;
    rayBuffer[base + SURFEL_PT_RAY_SURFEL_INDEX] = surfelIndex;
    rayBuffer[base + SURFEL_PT_RAY_RADIANCE_X] = asuint(radiance.x);
    rayBuffer[base + SURFEL_PT_RAY_RADIANCE_Y] = asuint(radiance.y);
    rayBuffer[base + SURFEL_PT_RAY_RADIANCE_Z] = asuint(radiance.z);
    rayBuffer[base + SURFEL_PT_RAY_PDF] = asuint(pdf);
    rayBuffer[base + SURFEL_PT_RAY_HIT_T] = asuint(hitT);
    rayBuffer[base + SURFEL_PT_RAY_FLAGS] = 1u;
    rayBuffer[base + 7u] = 0u;
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
    float3 rayDir = cosineSampleHemisphere(xi, normal);
    float pdf = max(dot(normal, rayDir), 0.0) / PI;

    RayDesc ray;
    ray.Origin = surfel.position + normal * SURFEL_PT_RAY_BIAS;
    ray.Direction = normalize(rayDir);
    ray.TMin = SURFEL_PT_RAY_BIAS;
    ray.TMax = 10000.0;

    SurfelPathTracerPayload payload;
    payload.radiance = float3(0.0, 0.0, 0.0);
    payload.hitT = -1.0;

    TraceRay(tlas, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);

    float3 clampedRadiance = clampLuminance(payload.radiance, SURFEL_PT_MAX_RADIANCE_LUMINANCE);
    storeRayRecord(rayIndex, surfelIndex, clampedRadiance, max(pdf, 0.0001), payload.hitT);
}
