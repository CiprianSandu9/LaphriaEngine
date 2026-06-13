# GI Research Report Rewrite Design

## Goal

Rewrite the Laphria GI research report so it reads as a focused research argument, not a project-history dump. The report should explain the rendering problem, the hypotheses tested, the decisions that followed from evidence, and the current SurfelPathTracer architecture in the repository.

## Inputs

- `docs/presentation/laphria-gi-research-report-source-table.md`
- `docs/presentation/laphria-gi-research-report-draft.md`
- `Laphria-GI-Research-Report.docx`
- Current source state under `src/Core` and `src/shaders`
- Git history for major GI milestones
- Presentation manifest and screenshots under `docs/presentation`

## Rewrite Principles

The report should prioritize research logic:

- State the original GI problem as a sampling and estimator question.
- Explain each major experiment by the hypothesis it tested.
- Keep implementation history only when it changes the research conclusion.
- Avoid support noise such as test-target status, build-system churn, deleted fixtures, or repository hygiene details.
- Treat screenshots and presentation files as supporting evidence, not as standalone proof.
- Distinguish historical truth from current repository state when behavior has changed.

## Source Table Design

Replace the current detailed chronology emphasis with a claim-to-evidence map. The table can remain chronological, but rows should be merged into research phases rather than one row per implementation burst.

Recommended rows:

1. Path tracer and diagnostic foundation
2. Indirect-lighting diagnosis in Sponza
3. Candidate multiplication and first-bounce probes
4. Cache/reuse and receiver-record experiments
5. Reservoir estimator audit
6. Standalone surfel compute-path removal
7. SurfelPathTracer architecture pivot
8. Current SurfelPathTracer backend state
9. Remaining validation and research questions

Each row should include:

- Research question or claim
- Evidence or implementation state
- Consequence for the report narrative
- Primary source documents and commits

The source table should no longer foreground tests unless a test directly expresses a rendering or estimator contract that matters to the research claim.

## Report Structure

Use this report structure:

1. Executive Summary
2. Problem And Research Question
3. Related Work And Technology Background
4. Diagnostic Method
5. First Finding: The Light Was There, But Rare
6. Experiments In Candidate Discovery And Reuse
7. Reservoir Estimator Audit
8. Architectural Decision: Separate Prototype Paths From The Surfel Backend
9. Current Repository State: SurfelPathTracer
10. Evidence And Validation Limits
11. Remaining Work
12. Conclusion

Sections 6 through 8 should be condensed compared with the current draft. They should explain why the project moved from brute-force probes to caches, from caches to reservoir-audited candidates, and from the old compute surfel path to a dedicated SurfelPathTracer backend.

## Expanded Current-State Section

The current-state section should be the largest structural change. It should describe how the repository currently works at a research-relevant level:

- Classic path tracer remains useful as a semantic reference and diagnostic baseline.
- SurfelPathTracer is a separate render backend with its own pass flow.
- GBuffer and source metadata capture scene/material/source information.
- Persistent surfels live in a bounded cache with camera-relative cells and cell-to-surfel lookup.
- Source anchoring lets surfels refresh from source geometry and transforms rather than only screen-space discovery.
- Surfel rays can use guided sampling from an irradiance atlas.
- Radiance integration uses MSME-style history and optional radiance sharing.
- Final lighting combines direct lighting, diffuse GI, reflections, emissive terms, debug views, and TAA/reference-validation paths.
- The current diffuse-GI integration differs from the older `albedo / PI` contract: direct lighting still uses a diffuse BRDF in `evaluatePrimaryDirectLighting`, but diffuse surfel GI currently uses `diffuseBsdf = albedo` and should be described as current behavior needing evaluation.

This section should avoid API-level detail unless it clarifies the research state. It should not become a README.

## Remaining Work Section

Revise remaining work to avoid saying implemented systems are unimplemented. Use these categories:

- Measurement and tuning for guided surfel ray sampling
- Validation of radiance sharing and MSME stability
- Evaluation of source anchoring quality and performance
- Resolution of current diffuse-GI unit semantics
- Performance, memory, and quality trade-off measurements
- Clear comparison against the classic path tracer/reference views

## DOCX Requirements

Regenerate `Laphria-GI-Research-Report.docx` from the updated markdown draft using the existing script unless the implementation plan finds a better local pattern.

The DOCX should:

- Preserve working hyperlinks for external related-work references.
- Avoid leftover markdown syntax or code fences.
- Keep a simple professional layout.
- Stay mostly prose-first; do not embed screenshots unless explicitly requested in a separate change.

## Acceptance Criteria

- The source table supports the revised report emphasis and contains no obsolete instruction to over-describe implementation history.
- The draft report focuses on research logic and current architecture.
- Current repository state is described accurately, including guided sampling, radiance sharing, temporal history, source anchoring, and the current diffuse-GI semantic caveat.
- Support details like missing/deleted tests are not discussed in the report unless directly relevant to a rendering claim.
- The regenerated DOCX passes the existing structural QA script.
- If visual DOCX rendering is unavailable because LibreOffice is missing, the final handoff says so plainly.
