#ifndef LAPHRIA_SURFEL_COMMON_SLANG
#define LAPHRIA_SURFEL_COMMON_SLANG

#include "ShaderCommon.slang"

static const uint SURFEL_GI_MAX_SURFELS = 32768u;
static const uint SURFEL_GI_GRID_DIM = 64u;
static const uint SURFEL_GI_CELL_COUNT = SURFEL_GI_GRID_DIM * SURFEL_GI_GRID_DIM * SURFEL_GI_GRID_DIM;
static const uint SURFEL_GI_MAX_CELL_MEMBERSHIPS_PER_SURFEL = 27u;
static const uint SURFEL_GI_CELL_TO_SURFEL_CAPACITY =
    SURFEL_GI_MAX_SURFELS * SURFEL_GI_MAX_CELL_MEMBERSHIPS_PER_SURFEL;
static const uint SURFEL_GI_MAX_EVAL_CANDIDATES = 16u;
static const uint SURFEL_GI_DENSE_CELL_LIMIT = 64u;
static const float SURFEL_GI_CELL_SIZE = 0.75f;
static const float SURFEL_GI_MIN_RADIUS = 0.08f;
static const float SURFEL_GI_MAX_RADIUS = 0.75f;

static const uint surfelGiClearDispatchesOffset = 404u;
static const uint surfelGiGenerateAttemptsOffset = 408u;
static const uint surfelGiGeneratedOffset = 412u;
static const uint surfelGiGenerateRejectInvalidOffset = 416u;
static const uint surfelGiGenerateRejectCoverageOffset = 420u;
static const uint surfelGiCellInsertAttemptsOffset = 424u;
static const uint surfelGiCellInsertedOffset = 428u;
static const uint surfelGiCellOverflowOffset = 432u;
static const uint surfelGiEvalAttemptsOffset = 436u;
static const uint surfelGiEvalCellEmptyOffset = 440u;
static const uint surfelGiEvalCandidatesOffset = 444u;
static const uint surfelGiEvalAcceptedOffset = 448u;
static const uint surfelGiCellTotalMembershipsOffset = 452u;
static const uint surfelGiCellAllocatedMembershipsOffset = 456u;
static const uint surfelGiCellAllocationOverflowOffset = 460u;
static const uint surfelGiCellNonEmptyOffset = 464u;
static const uint surfelGiCellMaxPopulationOffset = 468u;
static const uint surfelGiEvalDenseCellOffset = 472u;
static const uint surfelGiEvalDenseCellSkippedOffset = 476u;

struct SurfelGiRecord
{
    float4 positionRadius;
    float4 normalConfidence;
    float4 radianceAge;
    uint flags;
    uint lastSeenFrame;
    uint cellIndex;
    uint pad0;
};

struct SurfelGiCell
{
    uint count;
    uint offset;
    uint writeCursor;
    uint overflow;
};

uint flattenSurfelGiCell(int3 cell)
{
    return uint(cell.x) +
           uint(cell.y) * SURFEL_GI_GRID_DIM +
           uint(cell.z) * SURFEL_GI_GRID_DIM * SURFEL_GI_GRID_DIM;
}

bool isSurfelGiCellValid(int3 cell)
{
    return all(cell >= int3(0, 0, 0)) &&
           all(cell < int3(int(SURFEL_GI_GRID_DIM), int(SURFEL_GI_GRID_DIM), int(SURFEL_GI_GRID_DIM)));
}

int3 calcSurfelGiCell(float3 worldPos, float3 cameraPos)
{
    float3 local = worldPos - cameraPos;
    float3 grid = local / SURFEL_GI_CELL_SIZE + float3(SURFEL_GI_GRID_DIM, SURFEL_GI_GRID_DIM, SURFEL_GI_GRID_DIM) * 0.5f;
    return int3(floor(grid));
}

float calcSurfelGiRadius(float hitT, uint2 extent)
{
    const float shortestAxis = max(float(min(extent.x, extent.y)), 1.0f);
    const float projectedRadius = (hitT / shortestAxis) * 4.0f;
    return clamp(projectedRadius,
                 SURFEL_GI_MIN_RADIUS,
                 min(SURFEL_GI_MAX_RADIUS, SURFEL_GI_CELL_SIZE * 0.5f));
}

float3 calcSurfelGiPrimaryRayDirection(uint2 pixel,
                                       uint2 extent,
                                       float4x4 projInverse,
                                       float4x4 viewInverse)
{
    float2 pixelCenter = float2(pixel) + float2(0.5, 0.5);
    float2 inUV = pixelCenter / float2(extent);
    float2 d = inUV * 2.0f - 1.0f;
    float4 target = mul(projInverse, float4(d.x, -d.y, 1.0f, 1.0f));
    return mul(viewInverse, float4(normalize(target.xyz / target.w), 0.0f)).xyz;
}

float3 reconstructSurfelGiFirstHitWorldPosition(uint2 pixel,
                                                uint2 extent,
                                                float hitT,
                                                float3 cameraPos,
                                                float4x4 projInverse,
                                                float4x4 viewInverse)
{
    return cameraPos + calcSurfelGiPrimaryRayDirection(pixel, extent, projInverse, viewInverse) * hitT;
}

#endif
