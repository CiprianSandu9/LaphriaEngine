#include "SurfelPathTracerCommon.slang"

[[vk::binding(4, 0)]] RWByteAddressBuffer counters;
[[vk::binding(5, 0)]] RWStructuredBuffer<SurfelPathTracerSurfel> surfelBuffer;
[[vk::binding(10, 0)]] RWStructuredBuffer<uint> rayBuffer;
[[vk::binding(12, 0)]] RWByteAddressBuffer cellCounterBuffer;

struct UpdatePushConstants {
    uint maxSurfels;
    float cellSize;
    uint cellDimension;
    uint maxRays;
    float4 cameraPosition;
};

[[vk::push_constant]] UpdatePushConstants push;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint surfelIndex = tid.x;
    if (surfelIndex >= push.maxSurfels) {
        return;
    }

    SurfelPathTracerSurfel surfel = surfelBuffer[surfelIndex];
    if (surfel.flags == 0u || surfel.radius <= 0.0f || !isFinite3(surfel.position)) {
        return;
    }

    surfel.shortMeanAndLife.w = min(surfel.shortMeanAndLife.w + 1.0f, float(SURFEL_PT_MAX_LIFE));

    uint cellIndex = cameraRelativeCellIndexForPosition(surfel.position,
                                                        push.cameraPosition.xyz,
                                                        push.cellSize,
                                                        push.cellDimension);
    if (cellIndex == SURFEL_PT_INVALID_INDEX) {
        surfel.rayOffset = 0u;
        surfel.rayCount = 0u;
        surfelBuffer[surfelIndex] = surfel;
        return;
    }
    uint cellCountOffset = (1u + cellIndex) * 4u;
    cellCounterBuffer.InterlockedAdd(cellCountOffset, 1u);

    if (surfel.rayCount > 0u && push.maxRays > 0u) {
        uint rayOffset = 0u;
        counters.InterlockedAdd(SURFEL_PT_COUNTER_REQUESTED_RAYS_OFFSET, surfel.rayCount, rayOffset);
        uint writableRayCount = rayOffset < push.maxRays ? min(surfel.rayCount, push.maxRays - rayOffset) : 0u;
        if (writableRayCount < surfel.rayCount) {
            counters.InterlockedMin(SURFEL_PT_COUNTER_REQUESTED_RAYS_OFFSET, push.maxRays);
        }

        surfel.rayOffset = rayOffset;
        surfel.rayCount = writableRayCount;
        for (uint ray = 0u; ray < writableRayCount; ++ray) {
            uint rayIndex = rayOffset + ray;
            uint base = rayIndex * SURFEL_PT_RAY_RECORD_UINTS;
            rayBuffer[base + SURFEL_PT_RAY_SURFEL_INDEX] = surfelIndex;
            rayBuffer[base + SURFEL_PT_RAY_FLAGS] = 0u;
        }
    } else {
        surfel.rayOffset = 0u;
        surfel.rayCount = 0u;
    }

    surfelBuffer[surfelIndex] = surfel;
}
