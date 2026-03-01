# Specification Surface Review: 113577-V1

## Summary of Baseline Groups

**G1_math and G1_reading**: Two baseline groups for the same conceptual claim (peer teacher effects on student achievement) applied to two outcome domains. This is appropriate since math and reading are both headline outcomes in the paper and use separate peer VA measures.

- No missing baseline groups: the paper's main tables report both subjects.
- The two groups share the same design structure but differ in outcome and treatment variable.

## Design Selection

- `panel_fixed_effects` is correct. The identification relies on within-teacher variation in peer composition across years, using two-way FE (teacher + school-year).
- `design_audit` is adequate: records the two-way FE structure, estimator (felsdvreg), and clustering.
- Design variants correctly enumerate the paper's progressive FE columns (no FE, school-year only, student FE, teacher FE, teacher + school-year).

## RC Axes Assessment

- **FE structure**: This is the paper's primary axis of variation and is well-captured.
- **Controls**: Leave-one-out by block (lagged score, demographics, class, teacher experience) is appropriate.
- **Treatment definition**: Including peer observable VA and peer raw characteristics as alternatives is good -- this is a substantive choice the paper discusses.
- **Missing axes**: No functional form variations (e.g., nonlinear peer effects). No weighting. These are minor for this design.

## Controls Multiverse Policy

- `controls_count_min=5, controls_count_max=14` is reasonable given the paper's progressive control addition.
- `linked_adjustment=false` is correct -- no bundled estimator.
- The lagged score is always included (key identification element), which should probably be noted as a mandatory control in the surface.

## Inference Plan

- Canonical `infer/se/cluster/teacher` matches the paper.
- School-year clustering alternative is valuable.
- Two-way clustering (teacher and school) could be added but is not blocking.

## Budget Assessment

- 55 per group is adequate given the moderate specification space.
- 110 total (both groups) is reasonable.

## What's Missing

- Lagged score should be explicitly listed as a mandatory control (identification-critical).
- Two-way clustering (teacher x school-year) could strengthen inference robustness.
- No balanced panel restriction is listed (could matter for felsdvreg).

## Verdict

**Approved to run.** The surface correctly identifies the two-subject baseline structure and the FE-variation axis that is central to the paper's argument.
