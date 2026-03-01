# Specification Surface Review: 113566-V1

## Summary of Baseline Groups

**G1 (Grade Retention on Dropout)**: Correctly identified as the paper's main claim. The three baselines (grade 6, grade 8, older grade 8) represent the same claim object applied to three subpopulations that differ in the grade of retention. These are appropriately grouped under one baseline group since the estimand (LATE of retention on dropout) is the same.

- The grade-specific baselines are reasonable because the paper presents all three as main results in Table 2.
- No missing baseline groups for the core dropout outcome.

## Design Selection

- `regression_discontinuity` is correct. This is a fuzzy RD design using test score cutoffs as instruments.
- `design_audit` is thorough: records the running variable, cutoff, RD type (fuzzy), bandwidth, instrument structure, and clustering.
- The design variants cover the paper's actual robustness checks (Table 3): bandwidth variation, polynomial order, alternative instrument sets, and control richness levels. This is well-aligned with the manuscript's revealed search space.

## RC Axes Assessment

- **Design variants**: The paper's Table 3 provides an unusually rich set of within-design robustness checks. These are correctly enumerated: bandwidth, polynomial, instrument set, knot specification, pass-dummy alternatives.
- **Sample restrictions**: Grade, cohort, and failure-type subsamples are appropriate.
- **Controls**: Correctly treated as bundled blocks (experiment-interacted) rather than individual covariates. The 3 discrete richness levels match the paper's approach.
- **Missing axes**: No donut-hole RD specification (excluding observations very close to the cutoff). This is a standard RD robustness check that could be added. Also, no local polynomial RD (rdrobust-style) with data-driven bandwidth -- the paper uses parametric splines rather than modern nonparametric RD methods.

## Controls Multiverse Policy

- `linked_adjustment=true` is correct: the IV structure bundles instruments and controls in experiment-specific interactions. Individual covariate leave-one-out is not meaningful here.
- The 3 discrete control richness levels (group only, group+index, full) are the right approach for this design.

## Inference Plan

- Canonical `infer/se/cluster/gp` matches the paper exactly.
- The paper itself reports non-clustered and robust-only variants, which are included as inference variants.

## Budget Assessment

- 70 max core specs is adequate. The cross-product of design variants (~10) x grades (3) x sample restrictions could exceed this, but the constraint that many combinations are infeasible (e.g., grade-specific with grade restriction is redundant) keeps the realized count manageable.

## What's Missing

- Donut-hole specifications (excluding observations at index==0) could be added.
- Modern nonparametric RD (rdrobust/rdrobust-style) with data-driven bandwidth selection.
- The OLS estimates from Table 2 could be included as `design/*` comparisons to assess the IV premium.

## Verdict

**Approved to run.** The surface is well-aligned with the paper's extensive robustness check table and correctly handles the complex bundled instrument/covariate structure.
