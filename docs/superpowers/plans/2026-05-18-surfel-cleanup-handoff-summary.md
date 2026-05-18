# Surfel Cleanup Handoff Summary

Date: 2026-05-18

## Why We Are Executing This Plan

We are moving forward with the surfel cleanup plan because the latest reservoir audit changed the diagnosis.

The earlier concern was that our reservoir GI path might be fundamentally underestimating indirect light or that surfel/cache work was needed to explain the darker output. The audit rows showed something more specific: the reservoir estimator had an artificial final damping factor, `candidateCount / (candidateCount + 1)`, applied after reservoir contribution was already computed.

After removing that damping:

- Single-frame 1-candidate audit rows reported `reservoirGiAuditCurrentLuma == reservoirGiAuditReferenceLuma`.
- 2-candidate audit rows also stopped showing the previous systematic under-estimation.
- Temporal and temporal+spatial static audit rows reported `reservoirGiAuditRelativeErrorPct=0.00`.
- The old diagnostic `reservoirGiAuditProbeScale` still showed values like `0.50000` and `0.67188`, but the estimator output no longer used that scale.
- Visual brightness changed, but not in an obviously unstable or unusable way.

That makes the reservoir estimator a credible baseline again. The next architecture step should therefore preserve reservoir ownership of final indirect lighting and avoid adding a parallel surfel lighting path that bypasses the estimator.

## What We Learned About The Current Surfel Path

The existing standalone `Surfel*.slang` compute implementation is not a good foundation for the receiver-cache direction.

The reference surfel repos we inspected are architected around persistent world-space surfel records:

- explicit surfel lifecycle,
- persistent records across frames,
- per-frame cell indexing over persistent records,
- bounded lookup in dense cells,
- radiance update/integration,
- coverage, age, radius, and rejection diagnostics.

Our current compute surfel path is different. It is a separate debug GI subsystem with its own shader pass graph:

- `SurfelClear.slang`
- `SurfelGenerate.slang`
- `SurfelCountCells.slang`
- `SurfelAllocateCells.slang`
- `SurfelBuildCells.slang`
- `SurfelIntegrate.slang`
- `SurfelEvaluate.slang`
- `SurfelCommon.slang`

It also brings along separate buffers, descriptor sets, UI toggles, debug AOVs, counters, sweep rows, and dense-cell behavior. Recent sweep rows showed surfels being generated and inserted, but evaluation was dominated by dense-cell skipping or otherwise failed to produce useful accepted candidates. More importantly, even if fixed, this path would still be a parallel surfel GI experiment rather than a reservoir-owned proposal/cache system.

Keeping it around is now more harmful than helpful: it gives future work misleading names, stale failure modes, and a tempting but wrong integration surface.

## Architectural Direction

The guiding rule is:

> Surfels or cache records may help the reservoir discover useful indirect candidates, but the reservoir remains the final estimator.

That means:

- no direct surfel lighting contribution to final color,
- no separate surfel GI path competing with ReSTIR GI,
- no reuse of the existing compute surfel debug output as a production signal,
- no proposal/cache work until the reservoir estimator stays audit-clean.

The desirable future shape is closer to a reservoir receiver cache:

- train from useful receiver/candidate events observed by the path tracer,
- index persistent receiver/radiance evidence in world space,
- query it only as an explicit reservoir proposal source,
- reconnect/validate through reservoir-compatible target and visibility logic,
- measure whether it reduces `localRejectNoLight` or increases useful accepted candidates without biasing audit means.

## Why Bright-Surfel Code Is Treated Separately

There is still bright-surfel reservoir proposal/history code in `Raygen.slang` and related engine plumbing.

We are not removing it in the first cleanup step because it is already inside the reservoir candidate path, not the standalone compute surfel path. It may be closer to what we eventually want: a reservoir-owned source of receiver evidence.

The name is probably misleading. The concept to evaluate is not "bright surfels" as a new architecture; it is whether stored receiver/cache evidence can improve reservoir candidate discovery. Brightness should be treated as one possible training signal, not the identity of the future cache.

However, it is not trusted yet. The plan evaluates it only after the standalone compute surfel subsystem is gone, and it should be evaluated in stages: first as shadow-only diagnostics that train, query, probe, and measure candidate viability without entering reservoir selection; then as an enabled proposal only if the pre-selection funnel is demonstrably alive.

The bright-surfel proposal should be kept only if the explicit shadow and proposal sweeps show:

- training stores are positive,
- indexed queries and probes are positive,
- geometry, receiver-hemisphere, surfel-hemisphere, target, distance, invalid-vector, and visibility rejects do not dominate probed candidates,
- viable and accepted candidates are positive,
- selected bright-surfel candidates are positive in at least one scenario,
- audit relative error stays within 5 percent,
- runtime increase stays within the 25 percent budget unless Dark Courtyard clearly improves.

If it fails those criteria, it should be removed or replaced by a cleaner receiver-cache plan depending on which stage failed. If it passes, it should likely be renamed/refactored as a reservoir receiver cache rather than kept under the misleading bright-surfel framing. If the only positive signal is that high-luma records are easy to rediscover, that is not enough to justify promoting the abstraction.

## Execution Plan

The current detailed plan is:

`docs/superpowers/plans/2026-05-18-surfel-architecture-and-sweep-cleanup.md`

Execute it in this order:

1. Freeze and commit the reservoir estimator fix.
2. Replace tests that currently require the standalone surfel compute path with tests that require its removal.
3. Remove the standalone `Surfel*.slang` compute runtime completely:
   - shader compilation,
   - shader files,
   - compute pipelines,
   - descriptor layouts and writes,
   - frame resources,
   - UI toggles,
   - debug AOVs,
   - denoiser surfel debug binding,
   - counters and ratios,
   - row summary fields,
   - Sponza surfel-debug sweep row.
4. Trim the Sponza sweep to reservoir audit/regression rows only.
5. Update architecture docs to record that standalone compute surfels are abandoned.
6. Add two explicit bright-surfel evaluation sweeps: a shadow diagnostic sweep with baseline/static audit rows, and a separate proposal-enabled sweep that repeats the baseline static audit before proposal rows for same-run audit comparison.
7. Remove the shader hard-disable for bright surfels, but keep the feature disabled by default through proposal-mode selection and keep a shadow-only switch available for diagnostics.
8. Run the shadow diagnostic sweep first; run the proposal-enabled sweep only if the train/query/probe/viability funnel is alive; decide keep/refactor/delete/not-proven.
9. Record the decision in architecture docs.

## Important Guardrails

- Do not integrate surfel lighting directly into final color.
- Do not add a new surfel validation sweep for the deleted compute path.
- Do not let default Sponza sweeps include bright-surfel proposal rows.
- Do not run or trust proposal-enabled bright-surfel rows until the separate shadow diagnostic sweep shows a live pre-selection funnel.
- Do not promote "bright surfel" naming into the future architecture; use receiver-cache language if the concept survives.
- Do not keep `SurfelGiOccupancy`, `SurfelGiGather`, or `surfelGiDebugView` after removing the compute path.
- Do not leave tests reading deleted `Surfel*.slang` files.
- Do not claim the final tree passes unless tests and editor build are rerun after implementation.
- Do not include unrelated dirty files such as `.idea/editor.xml` or `.codex_refs/` in commits.

## Current State Before Execution

At the time this handoff was written:

- The reservoir estimator fix is present in the working tree but should be verified and committed first.
- The detailed cleanup/evaluation plan exists as an untracked plan document.
- The standalone compute surfel implementation still exists in code.
- The bright-surfel reservoir proposal still exists and is hard-disabled in shader code.
- No implementation code has been changed as part of the plan rewrite/handoff summary.

## Expected End State

After executing the plan:

- The reservoir estimator fix is committed and audit-clean.
- The standalone compute surfel subsystem is gone.
- Sponza sweeps are shorter and focused on reservoir regression.
- Architecture docs explain why the compute surfel path was removed.
- Bright-surfel reservoir proposal has explicit shadow/proposal evaluation sweeps.
- A measured keep/refactor/delete/not-proven decision exists for the bright-surfel code.
