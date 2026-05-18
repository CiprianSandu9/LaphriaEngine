# Reservoir GI Estimator Contract

Date: 2026-05-18

## Reference Model

The audit baseline is NVIDIA RTXDI ReSTIR GI:

- `RTXDI_MakeGIReservoir(samplePos, sampleNormal, sampleRadiance, samplePdf)` stores radiance and starts `weightSum` as inverse sample PDF.
- `RTXDI_CombineGIReservoirs` streams normalized reservoirs using `targetPdf * newReservoir.weightSum * newReservoir.M`.
- `RTXDI_FinalizeGIResampling` converts the streamed RIS weight into final reservoir weight.
- Final shading uses the selected secondary sample as an indirect sample:
  `primaryBRDF * reservoir.radiance * reservoir.weightSum`.

Reference sources:

- https://github.com/NVIDIA-RTX/RTXDI/blob/main/Doc/RestirGI.md
- https://raw.githubusercontent.com/NVIDIA-RTX/RTXDI-Library/main/Include/Rtxdi/GI/Reservoir.hlsli

## Current Implementation Questions

- `Raygen.slang` stores pre-shaded `ReservoirGiRecord::contribution`, not only secondary radiance plus final reservoir weight.
- Non-RIS local candidates are divided by `candidateCount`.
- All returned reservoir contribution is additionally multiplied by `candidateCount / (candidateCount + 1)`.
- Temporal/spatial reuse reconnects samples, but stored `targetWeight`, `selectedWeight`, `weightSum`, and `confidenceM` need a single written semantic contract.

## Audit Acceptance Gates

- Single-frame one-candidate audit output must match the equivalent plain first-hit diffuse probe within 2 percent average luminance on a static debug scene.
- Increasing local candidate count from 1 to 2 to 4 must not systematically darken or brighten the mean by more than 5 percent in audit rows.
- Toggling candidate RIS must preserve mean within 5 percent while changing variance/selection diagnostics.
- Temporal reuse on a static camera must reduce noise or increase accepted reuse without shifting mean more than 5 percent after warmup.
- Spatial reuse on a static camera must not shift mean more than 5 percent relative to temporal-only after warmup.

## Decision Gate

- If the estimator passes, surfels may be designed as a proposal/source for ReSTIR GI.
- If the estimator fails only because of explicit damping heuristics, isolate those heuristics behind audit toggles and decide whether to keep them as artistic/stability bias.
- If the estimator fails core normalization, fix it before using surfels for lighting.
