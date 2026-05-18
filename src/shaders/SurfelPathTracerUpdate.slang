#include "SurfelPathTracerCommon.slang"

[[vk::binding(4, 0)]] RWByteAddressBuffer counters;
[[vk::binding(5, 0)]] RWStructuredBuffer<SurfelPathTracerSurfel> surfelBuffer;
[[vk::binding(12, 0)]] RWByteAddressBuffer cellCounterBuffer;

struct UpdatePushConstants {
    uint maxSurfels;
    float cellSize;
    uint cellDimension;
    uint pad0;
};

[[vk::push_constant]] UpdatePushConstants push;

static const uint COUNTER_REQUESTED_RAYS_OFFSET = 12u;

uint cellIndexForPosition(float3 position, float cellSize, uint cellDimension) {
    float safeCellSize = max(cellSize, 0.0001f);
    uint dim = max(cellDimension, 1u);
    float halfDim = float(dim) * 0.5f;
    float maxCoord = float(dim - 1u);
    float3 centeredCell = floor(position / safeCellSize) + float3(halfDim, halfDim, halfDim);
    float3 clampedCell = clamp(centeredCell, float3(0.0f, 0.0f, 0.0f), float3(maxCoord, maxCoord, maxCoord));
    uint3 coord = uint3(clampedCell);
    return coord.x + coord.y * dim + coord.z * dim * dim;
}

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

    uint cellIndex = cellIndexForPosition(surfel.position, push.cellSize, push.cellDimension);
    uint cellCountOffset = (1u + cellIndex) * 4u;
    cellCounterBuffer.InterlockedAdd(cellCountOffset, 1u);

    if (surfel.rayCount > 0u) {
        uint rayOffset = 0u;
        counters.InterlockedAdd(COUNTER_REQUESTED_RAYS_OFFSET, surfel.rayCount, rayOffset);
        surfel.rayOffset = rayOffset;
    } else {
        surfel.rayOffset = 0u;
    }

    surfelBuffer[surfelIndex] = surfel;
}
