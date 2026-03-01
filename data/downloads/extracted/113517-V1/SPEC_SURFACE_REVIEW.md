# Specification Surface Review: 113517-V1

## Summary

The surface defines a single baseline group (G1) for the predictive association between EE reallocation rates and wage growth. This is appropriate: the paper has one main claim object (Table 1) varied across 4 outcome measures and 9 second-stage specifications.

## Baseline Group Assessment

**G1: Predictive power of EE for wage growth** -- APPROVED

- The claim object is well-defined: predictive association (not causal) between market-level EE rates and wage growth.
- Four baseline spec IDs (one per outcome variable) is correct since the paper treats all four outcomes as equally important headline results.
- The `design_code` of `cross_sectional_ols` is appropriate despite the two-stage structure, because the second stage is OLS with absorbed FE and the identification is cross-sectional association (not IV or panel identification).

## Design Audit

The `design_audit` block adequately captures the two-stage structure, FE specifications, and weighting scheme. No changes needed.

## RC Axes Review

### Controls (flow inclusion patterns)
- The progression and LOO specs correctly map to the paper's revealed Table 1 specifications.
- LOO drops are well-chosen: each of the 5 non-EE flow variables can be individually dropped from the full specification.
- `rc/controls/sets/none` maps to a bivariate (xee + ym_num only) specification, which is Spec 1 in the paper.
- `rc/controls/sets/minimal` maps to the minimal EE + UE specification (Spec 4).

### Sample restrictions
- Job stayers (paper-revealed Spec 8) is correctly included.
- Time splits (early/late half) and outlier trims are standard stress tests.

### Functional form
- The 4 outcome variable variants are correctly typed as `rc/form/outcome/*` since the paper treats them as alternative measurement of the same concept (wage/earnings growth).
- The EE interaction term (Spec 9) is correctly included as `rc/form/model/xee_interaction`.

### Weights
- Unweighted alternative is a standard stress test.

### Fixed effects
- Dropping year_month trend and replacing with year_month FE are both reasonable.

## Constraints and Linkage

- `linked_adjustment: true` is correct. The first-stage regressions produce the predicted flow variables; they should not be independently varied when changing second-stage controls.
- `controls_count_min: 1, controls_count_max: 6` correctly reflects the range from the paper's simplest (EE only + ym_num) to most complete (all 6 flows + ym_num) specifications.

## Inference Plan

- The canonical inference (iid) matches the paper's reported areg default standard errors.
- HC1 and market-level clustering are sensible variants for the inference_results.csv output.

## Budget Assessment

- Target of 80 specs is generous but feasible given the data size (~10M rows) and model simplicity (OLS with absorbed FE).
- Full enumeration is indeed feasible: the control dimension has at most 2^5 = 32 subsets of the 5 non-EE flow variables, and there are 4 outcome variables.
- Actual expected count: ~55-65 specs (4 baselines + ~12-15 control variations x 1 outcome + 4 outcome variations + 5 sample variations + a few FE/weight/interaction specs).

## What's Missing (minor)

1. No exploration of alternative market definitions (e.g., dropping education from the group variables). This is appropriate as it would change the first-stage estimation and is not revealed by the paper.
2. No alternative first-stage control sets. This is appropriate per the linkage constraint.

## Changes Made

No changes to the surface JSON. The surface is well-constructed.

## Verdict

**APPROVED TO RUN.** The surface is conceptually coherent, statistically principled, faithful to the revealed search space, and auditable. The budget is adequate for the target of 50+ specifications.
