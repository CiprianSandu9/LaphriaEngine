# ReSTIR GI Sponza Investigation Handoff

Date: 2026-05-16

This note keeps the Sponza path tracing / ReSTIR GI investigation state in the repository. The original handoff was provided from `C:\Users\Diaxxa\Downloads\restir-gi-sponza-handoff.md`; this copy adds the newer RTX 5080 sweep timings.

## Goal

Improve indirect lighting in difficult Sponza views:

- Dark courtyard areas.
- Sunlit courtyard wall regions.
- Mid-depth interior regions.
- Occluded or semi-enclosed spaces where bounced sunlight and environment light should still matter.

The main symptom is that baseline path-traced and reservoir-assisted results are too dark or too sparse in regions that should receive indirect light. The ReSTIR GI implementation also spends many candidate rays on samples that later have no usable light contribution.

## Current Best Validation Preset

The current Sponza validation preset uses:

- ReSTIR GI mode: `TemporalSpatial`
- Proposal mode: `MixedCosineSunReceiverGuided`
- Reservoir candidates: `1`
- Candidate RIS: off
- Spatial neighbors: `2`
- Temporal budget divisor: `2`
- Spatial budget divisor: `2`
- Environment NEE bounce mode: `First Only` (`environmentNeeBounceMode = 0`)
- Direct sun bounce mode: `First Only` (`directSunBounceMode = 1`)
- Max bounces: `8`
- Candidate evaluation mode: `2`

The focused Sponza sweep keeps these rows:

- `Reservoir 1C Shadowed Sun First Mixed`
- `Reservoir 1C Shadowed Sun First Mixed Temporal Budget 2`
- `Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N`
- `Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2`
- `Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Sun Receiver`
- `Reservoir 1C Shadowed Sun First Mixed Temporal Spatial 2N Budget 2 Env First Two`

The automated sweep no longer includes `Dual Sun`, but the mode remains available manually in the UI.

## Important Findings So Far

### Local Miss Acceptance Removed

Local miss candidates can still be counted, but they are not accepted as local GI samples. Accepted local samples are surface samples with valid target weight and usable suffix radiance. This keeps the reservoir path focused on real bounced-light surface candidates.

### Invalid Local Candidates Are Mostly No-Light Rejections

`localSurfaceInvalid` was split into:

- `localRejectGeometry`
- `localRejectNoLight`
- `localRejectZeroTarget`
- `localRejectBadPdf`

Most invalid local samples are rejected as `NoLight`, not bad PDFs. This points to proposal quality rather than a primary PDF/math failure: many candidate rays hit surfaces whose suffix radiance is near black.

### Temporal/Spatial Budget Result

`TemporalSpatial` with 2 spatial neighbors and temporal/spatial budget divisor 2 remains the working comparison baseline. Budget 3 reduced reuse too aggressively, 1-neighbor variants were weaker, and uncontrolled reuse often increased cost without enough quality gain.

### Environment NEE First Two

Environment NEE is configurable:

- `First Only` (`environmentNeeMode = 0`)
- `First Two` (`environmentNeeMode = 1`)
- `All Bounces` (`environmentNeeMode = 2`)

`First Two` often improves luminance, especially in occluded areas, but it remains a comparison knob rather than the default because the cost/quality tradeoff needs image-side validation.

### Sun Receiver Proposal

`MixedCosineSunReceiverGuided` is the most promising proposal variant so far. It targets likely sun-receiver surfaces, using an axis derived from `-ubo.lightDir.xyz` projected into the current hemisphere. The proposal PDF is matched to the mixture, and proposal mode packing was widened to support modes >= 4.

Sun Receiver is now the Sponza validation preset default.

### Persistent Surfel Cache Diagnostic Gates

The surfel cache remains diagnostic-only unless all of these are true:

- `surfelGiGenerated` is non-zero in all three Sponza validation views.
- `surfelGiEvalCandidates / surfelGiEvalAttempts` is bounded below 16 candidates per valid pixel.
- `surfelGiCellOverflow` is below 10% of `surfelGiCellInsertAttempts`.
- Enabling surfel diagnostics does not increase `totalMs` by more than 25% over Sun Receiver.
- Debug AOVs show coherent local coverage rather than sparse isolated points.

### History Guide

History-guided proposal instrumentation remains in code/UI, but the focused sweep no longer includes it. It was safe and technically functional, but it did not pull its weight because history loads and neighbor hits were too sparse or too weak in the hardest views.

## RTX 5080 Sweep Update

New sweep basis:

- GPU/system: RTX 5080 system.
- Samples: `32` per row.
- Max bounces: `8`.
- Direct sun: first only.
- Reservoir candidate evaluation mode: `2`.
- Focused Sponza rows only.

The table below uses `firstHitProbeAvgLuma` as the current luma proxy and `totalMs` as the frame timing proxy.

| View | Row | Luma | Total ms | Delta vs 2N Budget 2 |
| --- | --- | ---: | ---: | ---: |
| Dark Courtyard | Single-frame Mixed | 0.01306 | 68.474 | -68.1% luma, -20.765 ms |
| Dark Courtyard | Temporal Budget 2 | 0.01333 | 79.784 | -67.4% luma, -9.455 ms |
| Dark Courtyard | Temporal Spatial 2N | 0.03413 | 80.757 | -16.6% luma, -8.482 ms |
| Dark Courtyard | Temporal Spatial 2N Budget 2 | 0.04090 | 89.239 | baseline |
| Dark Courtyard | Sun Receiver | 0.03919 | 88.926 | -4.2% luma, -0.313 ms |
| Dark Courtyard | Env First Two | 0.04383 | 86.422 | +7.2% luma, -2.817 ms |
| Sunlit Courtyard Wall | Single-frame Mixed | 0.01959 | 136.106 | -50.9% luma, -69.669 ms |
| Sunlit Courtyard Wall | Temporal Budget 2 | 0.01943 | 174.616 | -51.3% luma, -31.159 ms |
| Sunlit Courtyard Wall | Temporal Spatial 2N | 0.03413 | 202.533 | -14.5% luma, -3.242 ms |
| Sunlit Courtyard Wall | Temporal Spatial 2N Budget 2 | 0.03991 | 205.775 | baseline |
| Sunlit Courtyard Wall | Sun Receiver | 0.06830 | 186.793 | +71.1% luma, -18.982 ms |
| Sunlit Courtyard Wall | Env First Two | 0.04792 | 209.491 | +20.1% luma, +3.716 ms |
| Mid-Depth Interior | Single-frame Mixed | 0.01845 | 106.664 | -38.2% luma, -83.804 ms |
| Mid-Depth Interior | Temporal Budget 2 | 0.01828 | 144.883 | -38.8% luma, -45.585 ms |
| Mid-Depth Interior | Temporal Spatial 2N | 0.02823 | 183.498 | -5.5% luma, -6.970 ms |
| Mid-Depth Interior | Temporal Spatial 2N Budget 2 | 0.02986 | 190.468 | baseline |
| Mid-Depth Interior | Sun Receiver | 0.03813 | 186.134 | +27.7% luma, -4.334 ms |
| Mid-Depth Interior | Env First Two | 0.03524 | 191.708 | +18.0% luma, +1.240 ms |

### RTX 5080 Read

- Sun Receiver still looks like the best practical default. It is a major win on Sunlit Courtyard Wall, a clear win on Mid-Depth Interior, and roughly neutral in Dark Courtyard.
- Dark Courtyard remains the outlier. In this run, `Env First Two` beats Sun Receiver on the luma proxy and is also slightly faster than the 2N Budget 2 comparison row, but that timing delta is small enough that it should be treated as run noise until repeated.
- The Sunlit Wall result is especially strong: Sun Receiver gives much higher luma and lower total time than the 2N Budget 2 baseline.
- Mid-Depth Interior also supports keeping Sun Receiver: it improves luma while being slightly faster than the baseline row.
- Single-frame and temporal-only rows remain useful diagnostics, but they are far below the temporal/spatial rows in luma for all three views.
- The no-light rejection pattern still dominates. Example scale from this run: Dark Courtyard and Mid-Depth Interior are around 2.7-2.9M `localRejectNoLight` per sampled row, while valid local samples are only thousands to low hundreds of thousands depending on view.

## Current Interpretation

### Candidate Rejection Is Still the Core Problem

Most local candidate rays still hit surfaces with no useful suffix radiance. Better proposal distributions are more promising than blindly increasing candidate count.

### Dark Courtyard Needs Separate Treatment

Sun Receiver helps the views with reachable sunlit receiver surfaces, but Dark Courtyard remains hard. It may need a different strategy, such as environment-aware proposal work, selective secondary NEE, or a more explicit bright-region proposal.

### Environment NEE Is a Quality Knob

`Env First Two` improves the luma proxy in this RTX 5080 run for all three views. It is not clearly too expensive on this hardware, but it still needs repeated runs and visual validation before changing the default.

### Reservoir Env Suffix Experiment Rejected

Bounce-1 direct sun NEE was reported to cost more without a meaningful indirect-lighting gain. That is consistent with the current reservoir implementation: local reservoir candidates already evaluate a shadowed secondary sun suffix when `reservoirEvalMode = 2`, so the normal path's bounce-1 direct sun setting does not solve local candidate discovery.

An experiment added `reservoirEvalMode = 3`, named `Shadowed Sun + Env Suffix`, which kept the existing shadowed sun suffix and added one sky-biased environment NEE sample to the candidate suffix. The RTX 5080 sweep rejected it:

- Dark Courtyard: luma fell from Sun Receiver `0.03892` to `0.00886`, while total time rose from `87.161ms` to `175.693ms`.
- Sunlit Courtyard Wall: luma fell from Sun Receiver `0.06931` to `0.04549`, while total time rose from `194.511ms` to `305.870ms`.
- Mid-Depth Interior: luma was slightly below Sun Receiver (`0.03539` vs `0.03727`), while total time rose from `188.757ms` to `269.026ms`.

The row increased `localValid` and `reservoirGiAccepted`, but mostly by accepting many low-value environment-lit candidates. It increased accepted luma sum while lowering average selected/target weight and badly hurting frame time. This is not a viable path for the focused sweep.

Conclusion: do not add environment NEE blindly inside every reservoir candidate. If environment helps later, it needs a sparse, confidence-gated, or separate light-cache strategy, not per-candidate suffix sampling.

### Direct Sun Beyond Bounce 0 Is Still Unresolved

`directSunBounceMode = 1` keeps direct sun NEE first-hit only. Secondary surfaces do not explicitly sample the sun unless a reservoir/proposal path discovers useful receivers. This remains a separate experiment.

### Image-Side Validation Is Required

The sweep uses average luminance proxies. They do not prove that a proposal is visually clean. Compare screenshots/AOVs for:

- Dark Courtyard
- Sunlit Courtyard Wall
- Mid-Depth Interior

Focus comparisons:

- `Temporal Spatial 2N Budget 2`
- `Sun Receiver`
- `Env First Two`

## Files Touched During The Investigation

Main files changed in the preceding investigation:

- `src/Core/EngineCore.cpp`
- `src/Core/EngineCore.h`
- `src/Core/EngineAuxiliary.h`
- `src/Core/UISystem.cpp`
- `src/Core/UISystem.h`
- `src/shaders/Raygen.slang`
- `tests/PathTracerAnalysisTests.cpp`

Unrelated dirty files observed and intentionally not touched:

- `.idea/editor.xml`
- `.idea/vcs.xml`
- `.idea/ctestState.xml`

## Verification State From Prior Handoff

These passed after promoting Sun Receiver and removing Dual Sun from the automated sweep:

```powershell
cmake --build build --config Debug --target LaphriaEngineUnitTests
ctest --test-dir build -C Debug --output-on-failure -R LaphriaEngineUnitTests
cmake --build build --config Debug --target LaphriaEditor
ctest --test-dir build -C Debug --output-on-failure
```

`git diff --check` reported only LF-to-CRLF warnings.

## Recommended Next Steps

1. Capture visual comparisons for the three Sponza validation views.
2. Compare `Temporal Spatial 2N Budget 2`, `Sun Receiver`, and `Env First Two` at the same sample count.
3. Repeat the RTX 5080 focused sweep once more before treating the small Dark Courtyard timing differences as stable.
4. If Sun Receiver is visually clean, keep it as the validation preset and likely practical default for Sponza-like scenes.
5. Investigate Dark Courtyard separately, starting with why `Env First Two` helps it more than Sun Receiver in the luma proxy.
6. Revisit direct sun NEE beyond bounce 0 as a separate experiment.
7. Consider a proposal that targets known bright receiver regions more explicitly, but only after visual validation confirms the current Sun Receiver behavior.

## Reservoir Estimator Audit Gate

The current ReSTIR GI reservoir should be treated as plausible but not yet proven. Before surfels become a lighting contributor or a trusted proposal source, the estimator must pass the audit gates in `docs/architecture/reservoir-gi-estimator-contract.md`.

In particular, the current `candidateCount / (candidateCount + 1)` reservoir output scale and the local non-RIS `candidateCount` division are audit targets, not assumed-correct normalization.
