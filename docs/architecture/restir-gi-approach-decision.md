# ReSTIR GI Approach Decision Plan

Date: 2026-05-16

This note defines how to choose the next indirect-lighting architecture after the Sponza sweep showed that bounce-1 sun NEE and per-candidate environment suffix sampling are not viable fixes.

## Current Evidence

- Local candidate rejection is dominated by `localRejectNoLight`.
- Sun Receiver is the best current practical preset, especially for Sunlit Wall and Mid-Depth Interior.
- Dark Courtyard remains hard.
- Bounce-1 direct sun NEE costs more without meaningful indirect-lighting gain.
- Per-candidate environment suffix sampling found more valid candidates but made average selected contribution worse and frame time much worse.

Conclusion: the next problem is useful receiver discovery, not more suffix lighting.

## Candidate Approaches

### ReSTIR-PG-Lite

Use accepted high-weight reservoir samples from previous frames to guide future candidate directions.

Reference:

- ReSTIR PG: Path Guiding using Spatiotemporal Reservoir Resampling, NVIDIA Research, 2025.
  https://research.nvidia.com/labs/rtr/publication/zeng2025restirpg/

Why it fits:

- It attacks bad initial candidate distribution directly.
- It can reuse existing reservoir history records.
- It has a smaller implementation surface than a full world-space cache.

Main risk:

- The old history-guided proposal was sparse. A PG-lite version must guide from stronger/high-weight records and probably aggregate locally/tile-wise rather than relying on a single reprojected pixel.

### Sparse Receiver/Radiance Cache

Store useful world-space receiver samples or radiance estimates, then sample/query them when generating indirect candidates.

References:

- RTXGI / SHaRC / NRC repository.
  https://github.com/NVIDIA-RTX/RTXGI
- Dynamic Diffuse Global Illumination with Ray-Traced Irradiance Fields, NVIDIA Research, 2019.
  https://research.nvidia.com/publication/2019-05_dynamic-diffuse-global-illumination-ray-traced-irradiance-fields
- Dynamic Diffuse Global Illumination Resampling, Roblox.
  https://about.roblox.com/publications/dynamic-diffuse-global-illumination-resampling

Why it fits:

- Dark Courtyard may need useful receivers that are not visible in nearby screen history.
- A cache can persist useful receivers across screen-space changes.

Main risk:

- Larger system: allocation, invalidation, spatial lookup, bias control, and memory budget all become real problems.

### ReSTIR PT / Reconnection Improvements

Improve the reuse machinery: better path reconnection, duplicate control, neighbor selection, and robustness.

References:

- ReSTIR GI: Path Resampling for Real-Time Path Tracing, NVIDIA Research, 2021.
  https://research.nvidia.com/publication/2021-06_restir-gi-path-resampling-real-time-path-tracing
- ReSTIR PT Enhanced, NVIDIA Research, 2026.
  https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced/

Why it fits:

- The current temporal/spatial reuse improves luma but costs a lot.
- Better reuse may reduce cost and artifacts once discovery improves.

Main risk:

- It does not solve discovery by itself. It should follow a successful guide/cache signal, not precede it.

## Decision Criteria

Use the three Sponza views:

- Dark Courtyard
- Sunlit Courtyard Wall
- Mid-Depth Interior

Compare against current Sun Receiver preset.

Primary metrics:

- Visual indirect quality in raw final and reservoir contribution AOVs.
- `firstHitProbeAvgLuma`
- `reservoirGiAcceptedLumaSum`
- `localRejectNoLight`
- `localValid`
- `reservoirGiSelectedLocal`
- `reservoirGiSelectedTemporal`
- `reservoirGiSelectedSpatial`
- `rayTraceMs`
- `totalMs`

Hard fail criteria:

- More than 25% total frame cost increase without clear image improvement.
- Dark Courtyard remains unchanged.
- More accepted samples but lower average useful contribution, like the rejected Env Suffix experiment.
- Obvious flicker, ghosting, or blotchy reuse.

## New Diagnostics

Two AOVs were added to support the decision:

- `Reservoir GI Local No Light`: grayscale local no-light rejection ratio per first-hit reservoir sample.
- `Reservoir GI Selected Source`: red = local, green = temporal, blue = spatial, black = none.

Use these with existing AOVs:

- `Reservoir GI Contribution`
- `Reservoir GI Accepted Luma`
- `Reservoir GI Candidate Surface Hit`
- `Reservoir GI Candidate Sun Visible`
- `Reservoir GI Candidate Positive Weight`
- `Reservoir GI Selected Weight`

## Recommended Next Prototype

Start with ReSTIR-PG-lite.

Prototype requirements:

- Guide from accepted high-weight records, not every accepted record.
- Prefer a small tile or neighborhood guide over a single reprojected pixel.
- Mix guided samples with the current Sun Receiver proposal instead of replacing it.
- Track guide attempts, guide hits, accepted guided samples, and no-light rejection for guided samples.

Kill if:

- Guide usage is sparse like the old history guide.
- It does not reduce `localRejectNoLight`.
- It improves only Sunlit Wall while Dark Courtyard remains unchanged.
- It causes visible temporal lag or instability.

If killed, move to sparse receiver/radiance cache.

## 2026-05-18 Surfel Cleanup Update

The standalone compute `Surfel*.slang` GI prototype is no longer the basis for the sparse receiver/radiance cache direction. It was removed from the active plan to avoid confusing a parallel surfel-lighting experiment with a reservoir-owned proposal/cache system.

The remaining bright-surfel reservoir proposal in `Raygen.slang` will be evaluated separately. It should be kept only if it behaves like useful receiver evidence for reservoir candidate generation and preserves the reservoir estimator audit contract.
