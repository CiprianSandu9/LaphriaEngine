#include "PathTracerAnalysisTests.h"

#include "../src/Core/PathTracerAnalysis.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <unordered_set>

namespace
{
uint64_t packConfigKey(const Laphria::PathTracerSweepConfig &cfg)
{
    const int scaled = static_cast<int>(std::lround(cfg.resolutionScale * 100.0f));
    return (static_cast<uint64_t>(scaled & 0xFF) << 24u) |
           (static_cast<uint64_t>(cfg.denoiserIterations & 0xFF) << 16u) |
           (static_cast<uint64_t>(cfg.enableReprojection ? 1 : 0) << 8u) |
           (static_cast<uint64_t>(cfg.enableMotionAwareAccumulation ? 1 : 0));
}
}

bool testPathTracerBaselineSweepMatrix()
{
    const auto matrix = Laphria::buildPathTracerBaselineSweepMatrix();
    if (matrix.size() != 90) {
        std::cerr << "baseline sweep size mismatch: expected 90 got " << matrix.size() << "\n";
        return false;
    }

    std::unordered_set<uint64_t> uniqueKeys;
    for (const auto &cfg : matrix) {
        uniqueKeys.insert(packConfigKey(cfg));
    }
    if (uniqueKeys.size() != matrix.size()) {
        std::cerr << "baseline sweep has duplicate configurations\n";
        return false;
    }

    for (const auto &cfg : matrix) {
        if (!cfg.enableReprojection && cfg.enableMotionAwareAccumulation) {
            std::cerr << "invalid sweep config: motion-aware enabled while reprojection disabled\n";
            return false;
        }
    }
    return true;
}

bool testPathTracerPercentiles()
{
    const std::vector<float> values = {10.0f, 40.0f, 20.0f, 30.0f, 50.0f};
    const auto pct = Laphria::computePercentiles(values);
    if (std::abs(pct.p50 - 30.0f) > 0.0001f) {
        std::cerr << "p50 mismatch\n";
        return false;
    }
    if (std::abs(pct.p95 - 50.0f) > 0.0001f) {
        std::cerr << "p95 mismatch\n";
        return false;
    }
    if (std::abs(pct.p99 - 50.0f) > 0.0001f) {
        std::cerr << "p99 mismatch\n";
        return false;
    }
    return true;
}

bool testPathTracerScoreBudgetGate()
{
    Laphria::PathTracerScoreInput input{};
    input.totalFrameMsP95 = 18.0f;
    input.targetBudgetMs = 16.67f;
    input.historyRejectionRatio = 0.30f;
    input.skyHitRatio = 0.05f;
    input.fireflyClampRatio = 0.02f;
    input.visualFidelityScore = 0.85f;

    const auto score = Laphria::scorePathTracerRun(input);
    if (score.budgetPass) {
        std::cerr << "budget gate should fail when p95 exceeds target\n";
        return false;
    }
    if (score.compositeScore >= 0.80f) {
        std::cerr << "composite score unexpectedly high for budget fail case\n";
        return false;
    }
    return true;
}
