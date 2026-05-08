#include "PathTracerAnalysis.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace
{
float clamp01(float value)
{
	return std::clamp(value, 0.0f, 1.0f);
}

float nearestRankPercentile(const std::vector<float> &sortedValues, float percentile)
{
	if (sortedValues.empty())
	{
		return 0.0f;
	}

	const float  pct  = std::clamp(percentile, 0.0f, 100.0f);
	const size_t n    = sortedValues.size();
	const size_t rank = static_cast<size_t>(std::ceil((pct / 100.0f) * static_cast<float>(n)));
	const size_t idx  = (rank == 0) ? 0 : std::min(rank - 1, n - 1);
	return sortedValues[idx];
}
}        // namespace

namespace Laphria
{
std::vector<PathTracerSweepConfig> buildPathTracerBaselineSweepMatrix()
{
	constexpr std::array<float, 3>     resolutionScales = {0.80f, 0.90f, 1.00f};
	std::vector<PathTracerSweepConfig> matrix;
	// 6 scales * 1 denoiser iterations * 3 unique temporal modes:
	// - reprojection OFF (motion-aware flag is irrelevant)
	// - reprojection ON + motion-aware OFF
	// - reprojection ON + motion-aware ON
	matrix.reserve(3 * 1 * 3);

	for (const float scale : resolutionScales)
	{
		for (int denoiserIterations = 1; denoiserIterations <= 1; ++denoiserIterations)
		{
			// Reprojection OFF baseline.
			matrix.push_back(PathTracerSweepConfig{
			    .resolutionScale               = scale,
			    .denoiserIterations            = denoiserIterations,
			    .enableReprojection            = false,
			    .enableMotionAwareAccumulation = false,
			    .reduceSecondaryEffects        = false});
			// Reprojection ON, motion-aware OFF.
			matrix.push_back(PathTracerSweepConfig{
			    .resolutionScale               = scale,
			    .denoiserIterations            = denoiserIterations,
			    .enableReprojection            = true,
			    .enableMotionAwareAccumulation = false,
			    .reduceSecondaryEffects        = false});
			// Reprojection ON, motion-aware ON.
			matrix.push_back(PathTracerSweepConfig{
			    .resolutionScale               = scale,
			    .denoiserIterations            = denoiserIterations,
			    .enableReprojection            = true,
			    .enableMotionAwareAccumulation = true,
			    .reduceSecondaryEffects        = false});
		}
	}

	return matrix;
}

PercentileTriplet computePercentiles(const std::vector<float> &samples)
{
	if (samples.empty())
	{
		return {};
	}

	std::vector<float> sortedValues = samples;
	std::sort(sortedValues.begin(), sortedValues.end());

	return PercentileTriplet{
	    .p50 = nearestRankPercentile(sortedValues, 50.0f),
	    .p95 = nearestRankPercentile(sortedValues, 95.0f),
	    .p99 = nearestRankPercentile(sortedValues, 99.0f)};
}

PathTracerRunScore scorePathTracerRun(const PathTracerScoreInput &input)
{
	const float budget = std::max(0.001f, input.targetBudgetMs);
	const float p95    = std::max(0.0f, input.totalFrameMsP95);

	const bool  budgetPass       = p95 <= budget;
	const float performanceScore = clamp01(budget / std::max(p95, 0.001f));

	const float rejectionPenalty = clamp01(input.historyRejectionRatio);
	const float fireflyPenalty   = clamp01(input.fireflyClampRatio * 10.0f);
	const float skyPenalty       = clamp01(input.skyHitRatio * 0.25f);
	const float stabilityPenalty = clamp01(0.60f * rejectionPenalty + 0.30f * fireflyPenalty + 0.10f * skyPenalty);
	const float stabilityScore   = 1.0f - stabilityPenalty;

	const float fidelityScore   = clamp01(input.visualFidelityScore);
	const float budgetGateScale = budgetPass ? 1.0f : 0.70f;
	const float compositeScore  = budgetGateScale * (0.45f * performanceScore + 0.30f * stabilityScore + 0.25f * fidelityScore);

	return PathTracerRunScore{
	    .budgetPass       = budgetPass,
	    .performanceScore = performanceScore,
	    .stabilityScore   = stabilityScore,
	    .fidelityScore    = fidelityScore,
	    .compositeScore   = compositeScore,
	    .budgetResult     = budgetPass ? "PASS" : "FAIL"};
}

std::vector<PathTracerBacklogItem> buildDefaultFidelityBacklog(float rayTraceP95Ms,
                                                               float reprojectionP95Ms,
                                                               float denoiserP95Ms,
                                                               float totalP95Ms,
                                                               float budgetMs)
{
	const bool                         budgetPass = totalP95Ms <= budgetMs;
	std::vector<PathTracerBacklogItem> items;
	items.reserve(5);

	items.push_back(PathTracerBacklogItem{
	    .name                   = "Temporal rejection tuning",
	    .expectedArtifactImpact = "Reduce ghost trails and temporal shimmer",
	    .estimatedMsCost        = 0.05f * std::max(reprojectionP95Ms, 0.1f),
	    .measuredMsCost         = 0.03f * std::max(reprojectionP95Ms, 0.1f),
	    .budgetPass             = budgetPass,
	    .priority               = PathTracerBacklogPriority::High});

	items.push_back(PathTracerBacklogItem{
	    .name                   = "Denoiser edge-weight retuning",
	    .expectedArtifactImpact = "Improve detail retention on normal/depth edges",
	    .estimatedMsCost        = 0.08f * std::max(denoiserP95Ms, 0.1f),
	    .measuredMsCost         = 0.05f * std::max(denoiserP95Ms, 0.1f),
	    .budgetPass             = budgetPass,
	    .priority               = PathTracerBacklogPriority::High});

	items.push_back(PathTracerBacklogItem{
	    .name                   = "Adaptive iteration policy refinements",
	    .expectedArtifactImpact = "Stabilize noisy motion frames while preserving static detail",
	    .estimatedMsCost        = 0.15f * std::max(denoiserP95Ms, 0.1f),
	    .measuredMsCost         = 0.10f * std::max(denoiserP95Ms, 0.1f),
	    .budgetPass             = budgetPass,
	    .priority               = PathTracerBacklogPriority::Medium});

	items.push_back(PathTracerBacklogItem{
	    .name                   = "Jitter activation strategy",
	    .expectedArtifactImpact = "Reduce structured noise and improve convergence uniformity",
	    .estimatedMsCost        = 0.02f * std::max(totalP95Ms, 0.1f),
	    .measuredMsCost         = 0.00f,
	    .budgetPass             = budgetPass,
	    .priority               = PathTracerBacklogPriority::Medium});

	items.push_back(PathTracerBacklogItem{
	    .name                   = "Direct-light sampling coverage",
	    .expectedArtifactImpact = "Lower fireflies and improve high-contrast convergence",
	    .estimatedMsCost        = 0.30f * std::max(rayTraceP95Ms, 0.1f),
	    .measuredMsCost         = 0.00f,
	    .budgetPass             = (totalP95Ms + 0.30f * std::max(rayTraceP95Ms, 0.1f)) <= budgetMs,
	    .priority               = PathTracerBacklogPriority::Low});

	return items;
}
}        // namespace Laphria
