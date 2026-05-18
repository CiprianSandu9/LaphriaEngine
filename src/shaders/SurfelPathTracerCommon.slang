#ifndef SURFEL_PATH_TRACER_COMMON_H
#define SURFEL_PATH_TRACER_COMMON_H

#include "ShaderCommon.slang"

static const uint SURFEL_PT_MAX_LIFE = 255u;
static const uint SURFEL_PT_INVALID_INDEX = 0xFFFFFFFFu;
static const float SURFEL_PT_RAY_BIAS = 0.001;

struct SurfelPathTracerSurfel {
    float3 position;
    float radius;
    float3 radiance;
    uint packedNormal;
    uint rayOffset;
    uint rayCount;
    uint irradianceAtlasOffset;
    uint flags;
    float4 meanAndVariance;
    float4 shortMeanAndLife;
};

struct SurfelPathTracerCellInfo {
    uint surfelOffset;
    uint surfelCount;
};

struct SurfelPathTracerCounters {
    uint aliveSurfels;
    uint deadSurfels;
    uint dirtySurfels;
    uint requestedRays;
    uint filledCells;
    uint rejectedStores;
    uint frameIndex;
    uint pad0;
};

uint packNormalOctahedral(float3 normal) {
    float3 n = normalize(normal);
    float invL1 = 1.0 / max(abs(n.x) + abs(n.y) + abs(n.z), 0.000001);
    float2 oct = n.xy * invL1;
    if (n.z < 0.0) {
        float2 signNotZero = float2(oct.x >= 0.0 ? 1.0 : -1.0, oct.y >= 0.0 ? 1.0 : -1.0);
        oct = (1.0 - abs(oct.yx)) * signNotZero;
    }
    uint2 encoded = uint2(round(saturate(oct * 0.5 + 0.5) * 65535.0));
    return encoded.x | (encoded.y << 16u);
}

float3 unpackNormalOctahedral(uint packedNormal) {
    uint2 encoded = uint2(packedNormal & 0xFFFFu, packedNormal >> 16u);
    float2 oct = float2(encoded) / 65535.0 * 2.0 - 1.0;
    float3 n = float3(oct.x, oct.y, 1.0 - abs(oct.x) - abs(oct.y));
    if (n.z < 0.0) {
        float2 signNotZero = float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
        n.xy = (1.0 - abs(n.yx)) * signNotZero;
    }
    return normalize(n);
}

bool isFinite3(float3 value) {
    return abs(value.x) < 3.402823e38 &&
           abs(value.y) < 3.402823e38 &&
           abs(value.z) < 3.402823e38;
}

#endif // SURFEL_PATH_TRACER_COMMON_H
