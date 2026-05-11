#ifndef LAPHRIAENGINE_PATHTRACERANALYSIS_H
#define LAPHRIAENGINE_PATHTRACERANALYSIS_H

#include <string>
#include <vector>

namespace Laphria
{
struct PathTracerSweepConfig
{
    float resolutionScale = 1.0f;
    int   denoiserIterations = 1;
    bool  enableReprojection = true;
    bool  enableMotionAwareAccumulation = true;
    bool  reduceSecondaryEffects = false;
};

struct PercentileTriplet
{
    float p50 = 0.0f;
    float p95 = 0.0f;
    float p99 = 0.0f;
};

struct PathTracerScoreInput
{
    float totalFrameMsP95 = 0.0f;
    float targetBudgetMs = 16.67f;
    float historyRejectionRatio = 0.0f;
    float skyHitRatio = 0.0f;
    float fireflyClampRatio = 0.0f;
    float visualFidelityScore = 0.5f; // [0,1] subjective or offline-eval-fed score
};

struct PathTracerRunScore
{
    bool        budgetPass = false;
    float       performanceScore = 0.0f;
    float       stabilityScore = 0.0f;
    float       fidelityScore = 0.0f;
    float       compositeScore = 0.0f;
    std::string budgetResult;
};

struct PathTracerHistoryClampInput
{
    float neighborhoodMinLum = 0.0f;
    float neighborhoodMaxLum = 0.0f;
    float historyLum = 0.0f;
    float previousMeanLum = 0.0f;
    float previousVariance = 0.0f;
};

struct PathTracerHistoryClampResult
{
    float clampedHistoryLum = 0.0f;
    float halfRange = 0.0f;
};

enum class PathTracerBacklogPriority
{
    High = 0,
    Medium = 1,
    Low = 2
};

struct PathTracerBacklogItem
{
    std::string             name;
    std::string             expectedArtifactImpact;
    float                   estimatedMsCost = 0.0f;
    float                   measuredMsCost = 0.0f;
    bool                    budgetPass = true;
    PathTracerBacklogPriority priority = PathTracerBacklogPriority::Medium;
};

std::vector<PathTracerSweepConfig> buildPathTracerBaselineSweepMatrix();
PercentileTriplet                  computePercentiles(const std::vector<float> &samples);
PathTracerRunScore                 scorePathTracerRun(const PathTracerScoreInput &input);
PathTracerHistoryClampResult       computePathTracerHistoryClamp(const PathTracerHistoryClampInput &input);
std::vector<PathTracerBacklogItem> buildDefaultFidelityBacklog(float rayTraceP95Ms,
                                                               float reprojectionP95Ms,
                                                               float denoiserP95Ms,
                                                               float totalP95Ms,
                                                               float budgetMs);
}

#endif
