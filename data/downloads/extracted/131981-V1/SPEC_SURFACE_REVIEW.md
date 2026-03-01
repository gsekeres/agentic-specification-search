# Specification Surface Review: 131981-V1

**Reviewer**: Automated verification agent
**Date**: 2026-02-24
**Status**: APPROVED TO RUN

---

## A) Baseline Groups

- **G1** (Mental Health Effect of Curfew): Well-defined single claim object. The outcome concept (mental distress index), treatment concept (age-based curfew exposure), estimand (sharp RD at age-65 cutoff), and target population (Turkish adults near age-65 threshold) are all correctly specified.
- The paper does report effects on multiple mental health indices (z_depression, z_somatic, z_nonsomatic, sum_srq). Rather than creating separate baseline groups for each outcome, the surface correctly treats z_depression as the primary outcome and includes the others as `rc/form/outcome/*` robustness checks. This is appropriate because the paper frames z_depression as the headline measure.
- No missing baseline groups: mobility, channels, political, and religiosity outcomes are correctly excluded as mechanisms/secondary analyses.

## B) Design Selection

- `design_code = regression_discontinuity` is correct for this sharp RD at an age cutoff.
- `design_audit` block includes all required RD fields: running_var, cutoff, rd_type, bandwidth, kernel, poly_order, bias_correction. This is complete.
- Design variants are appropriate:
  - Bandwidth variations (17, 24, 36, 45, 48, 60, 72) match the paper's revealed search space (Tables 4, A4)
  - Local quadratic polynomial matches Table A8
- No over-expansion: the design variants stick to what the paper itself explores.

## C) RC Axes

- **Controls LOO by block**: Appropriate for this design where controls are FE blocks. Six block-level LOO specs.
- **Control sets** (none, minimal, full): Appropriate span. The "no controls" spec matches the balance-test approach in Table A2.
- **Donut holes** (1, 2, 3 months): Standard RD robustness check. Appropriate.
- **Alternative outcomes**: z_somatic, z_nonsomatic, sum_srq all preserve the mental health claim object while varying the outcome measure. Appropriate.
- **Joint specs**: Bandwidth x polynomial, bandwidth x controls, outcome x bandwidth provide useful cross-cutting variation.
- No missing high-leverage axes. Data construction is not an issue (single dataset, straightforward construction).

### Minor Revisions Made

1. Confirmed that the `rc/form/outcome/*` specs for alternative mental health indices preserve the claim object (all measure mental distress at the same cutoff).
2. Verified that functional-form interpretation is preserved: coefficient on before1955 always measures the RD jump in mental health at cutoff.

## D) Controls Multiverse Policy

- Controls-count envelope (min=2, max=~92) is correctly derived from the baseline specs.
- Mandatory controls: the RD polynomial terms (dif, dif*before1955) are always included -- they are the design-defining terms, not optional controls.
- Block structure is explicitly documented: FE groups (month, province, ethnicity, education, survey_taker_id, female) move as atomic blocks.
- No linked adjustment needed (single-equation estimator).

## E) Inference Plan

- Canonical inference = clustering at modate (survey month-year). This matches the paper's `vce(cluster modate)`.
- Variants (HC1, province clustering) are clearly described as inference-only recomputations.
- Province clustering is a reasonable stress test (coarser clustering).

## F) Budgets and Sampling

- Budget of ~80 specs is feasible and informative.
- Full enumeration approach is correct: the control blocks create a small, tractable LOO set (~6 specs), and bandwidth/polynomial/donut variations are finite.
- No random sampling needed.
- Seed (131981) is documented.

## G) Diagnostics Plan

- Covariate balance test is listed. This is the standard RD diagnostic (Table A2 in the paper).
- Scope = baseline_group is correct (balance does not depend on control sets).
- McCrary density test is not included. This is a minor omission; the paper does not discuss density manipulation, likely because birth month is effectively exogenous. Acceptable to omit.

## Summary of Changes

No substantive changes to the surface. The surface as built is well-structured, faithful to the paper's revealed search space, and ready for execution.

## Final Assessment

**APPROVED TO RUN**. The specification surface is conceptually coherent, statistically principled, faithful to the manuscript's revealed surface, and auditable. Proceed to execution.
