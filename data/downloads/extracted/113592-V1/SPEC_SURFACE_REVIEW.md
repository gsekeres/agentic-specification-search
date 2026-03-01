# Specification Surface Review: 113592-V1

## Summary of Baseline Groups

**G1 (Grant Eligibility on College Enrollment)**: Correctly identified as the paper's main claim. The three cutoff samples (0_X, 1_0, 6_1) represent the same claim object at different points on the income distribution, which is appropriate as a single baseline group with multiple baseline spec IDs.

- No missing baseline groups for the enrollment outcome.
- Persistence/completion outcomes (Table 5) are correctly excluded as `explore/*`.

## Design Selection

- `regression_discontinuity` is correct. This is a sharp RD at income eligibility cutoffs.
- `design_audit` is thorough: records running variable, cutoff, RD type (sharp), bandwidth (optimal IK), kernel (triangular), polynomial order (1), and estimator (LLR).
- Design variants appropriately cover bandwidth, polynomial, kernel, and procedure alternatives.

## RC Axes Assessment

- **Design variants**: Good coverage of standard RD robustness: bandwidth sensitivity, polynomial order, kernel choice, and bias correction procedure.
- **Sample restrictions**: Year-by-year, gender, study level, and academic performance quartile subgroups from Table 4 are well-captured.
- **Donut hole**: Correctly included as an RD-specific robustness check.
- **Missing axes**: Covariates could be added as precision controls (standard in modern RD practice), but the paper does not do this, so excluding them is faithful to the manuscript. Parametric polynomial RD (Table 6) could be added as a `design/*` variant.

## Controls Multiverse Policy

- `controls_count_min=0, controls_count_max=0` is correct: the paper uses pure nonparametric LLR with no covariates.
- `linked_adjustment=false` is correct (no bundled estimator).

## Inference Plan

- Conventional LLR SEs from rdob as canonical is correct.
- Robust bias-corrected (CCFT) is an important variant for modern RD practice.
- Clustering at discrete income levels is a reasonable concern if income has mass points.

## Budget Assessment

- 70 max core specs is adequate for the cross-product of 3 cutoffs x ~7 design variants + ~13 sample restrictions + donut.

## What's Missing

- Parametric polynomial RD from Table 6 / appendix tables could be added as design variants.
- Adding covariates as precision controls is standard modern RD practice but absent from the paper.
- Placebo cutoff tests (standard RD diagnostic) are not included in the diagnostics plan.

## Verdict

**Approved to run.** The surface is well-aligned with the paper's nonparametric RD framework and correctly handles the multiple-cutoff structure.
