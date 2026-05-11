#include "PathTracerAnalysisTests.h"

#include "../src/Core/PathTracerAnalysis.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>

namespace
{
std::string readTextFile(const std::filesystem::path &path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return {};
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

bool containsText(const std::string &haystack, const char *needle)
{
    return haystack.find(needle) != std::string::npos;
}

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
    if (matrix.size() != 9) {
        std::cerr << "baseline sweep size mismatch: expected 9 got " << matrix.size() << "\n";
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

bool testPathTracerHistoryClampPreservesDimIndirectHistory()
{
    Laphria::PathTracerHistoryClampInput input{};
    input.neighborhoodMinLum = 0.0f;
    input.neighborhoodMaxLum = 0.002f;
    input.historyLum = 0.035f;
    input.previousMeanLum = 0.032f;
    input.previousVariance = 0.000004f;

    const auto result = Laphria::computePathTracerHistoryClamp(input);
    if (result.clampedHistoryLum < 0.030f) {
        std::cerr << "history clamp crushed dim indirect history: "
                  << result.clampedHistoryLum << "\n";
        return false;
    }

    Laphria::PathTracerHistoryClampInput darkSpike{};
    darkSpike.neighborhoodMinLum = 0.0f;
    darkSpike.neighborhoodMaxLum = 0.002f;
    darkSpike.historyLum = 1.0f;
    darkSpike.previousMeanLum = 0.032f;
    darkSpike.previousVariance = 0.000004f;

    const auto spikeResult = Laphria::computePathTracerHistoryClamp(darkSpike);
    if (spikeResult.clampedHistoryLum > 0.050f) {
        std::cerr << "history clamp preserved an excessive dark-region spike: "
                  << spikeResult.clampedHistoryLum << "\n";
        return false;
    }

    return true;
}

bool testPathTracerDebugAovContract()
{
    const std::filesystem::path sourceRoot =
#ifdef LAPHRIA_SOURCE_DIR
        std::filesystem::path(LAPHRIA_SOURCE_DIR);
#else
        std::filesystem::current_path();
#endif

    const std::string uiHeader = readTextFile(sourceRoot / "src" / "Core" / "UISystem.h");
    const std::string uiSource = readTextFile(sourceRoot / "src" / "Core" / "UISystem.cpp");
    const std::string raygen = readTextFile(sourceRoot / "src" / "shaders" / "Raygen.slang");
    const std::string denoiser = readTextFile(sourceRoot / "src" / "shaders" / "Denoiser.slang");

    if (uiHeader.empty() || uiSource.empty() || raygen.empty() || denoiser.empty()) {
        std::cerr << "failed to load path tracer debug AOV contract files\n";
        return false;
    }

    const char *requiredLabels[] = {
        "Direct Lighting",
        "Indirect Lighting",
        "Sky Contribution",
        "Throughput",
        "Bounce Count",
        "Shadow Visibility"};
    for (const char *label : requiredLabels) {
        if (!containsText(uiSource, label)) {
            std::cerr << "missing path tracer debug AOV UI label: " << label << "\n";
            return false;
        }
    }

    const char *requiredEnumValues[] = {
        "PathDirectLighting",
        "PathIndirectLighting",
        "PathSkyContribution",
        "PathThroughput",
        "PathBounceCount",
        "PathShadowVisibility"};
    for (const char *enumValue : requiredEnumValues) {
        if (!containsText(uiHeader, enumValue)) {
            std::cerr << "missing path tracer debug AOV enum value: " << enumValue << "\n";
            return false;
        }
    }

    const char *requiredShaderSymbols[] = {
        "debugDirectLighting",
        "debugIndirectLighting",
        "debugSkyContribution",
        "debugThroughput",
        "debugBounceCount",
        "debugShadowVisibility"};
    for (const char *symbol : requiredShaderSymbols) {
        if (!containsText(raygen, symbol) && !containsText(denoiser, symbol)) {
            std::cerr << "missing path tracer debug AOV shader symbol: " << symbol << "\n";
            return false;
        }
    }

    return true;
}
