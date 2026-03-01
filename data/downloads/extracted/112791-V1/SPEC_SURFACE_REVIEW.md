# Specification Surface Review: 112791-V1

**Paper**: Baicker et al. (2014), "The Impact of Medicaid on Labor Market Activity and Program Participation: Evidence from the Oregon Health Insurance Experiment," AER P&P
**Design**: randomized_experiment
**Date reviewed**: 2026-02-25

---

## Summary of Baseline Groups

| Group | Claim Object | Baseline Specs | Status |
|-------|-------------|----------------|--------|
| G1 | ITT of lottery selection on employment/earnings (Table 1) | 3 outcomes (any_earn, earn, earn_ab_fpl_adj) | Verified |
| G2 | ITT of lottery selection on government benefit receipt (Table 2) | 8 outcomes (4 binary + 4 dollar amounts) | Verified |

**Changes made**: Minor corrections to the JSON (see below).

---

## A) Baseline Groups

- G1 and G2 are correctly separated by claim object (labor supply vs. program participation). Each corresponds to a distinct table in the paper.
- The baseline specs correctly reflect the do-file: `reg {outcome} treatment nnn* {lagged_outcome} [pw=weight_ssa_admin], cluster(reservation_id)`.
- Variable names verified against `ssa_analysis_replication.do`: `any_earn2009`, `earn2009`, `earn_ab_fpl_adj_2009` for G1; `any_snapamt2009`, `any_tanfamt2009`, `any_ssiben2009`, `any_diben2009`, `snapamt2009`, `tanfamt2009`, `ssiben2009`, `diben2009` for G2.
- No missing baseline groups. The paper also reports disability application outcomes (Table A8) and summary indices (Table A9), but these are correctly excluded as secondary analyses.

## B) Design Selection

- `design_code: randomized_experiment` is correct. The OHIE lottery is the canonical RCT.
- Design audit is well-populated: randomization_unit, strata_or_blocks, estimand, weights, cluster_var, sample_condition are all recorded.
- The IV/LATE estimates (2SLS with `ohp_all_ever_admin` instrumented by `treatment`) are correctly excluded from the core surface. They target a different estimand (LATE for compliers).
- Design variants (`diff_in_means`, `strata_fe`) are appropriate.

## C) RC Axes

- **Controls**: Drop lagged outcome and add lottery_list demographics are the two main control axes, matching the paper's robustness tables (A10-A11). Verified against the `lottery_list` local: `birthyear_list female_list english_list self_list first_day_list have_phone_list pobox_list zip_msa zip_hh_inc_list` (9 variables). Correct.
- **Time periods**: Year 2008 and years 2008-2009 pooled are explicitly in the paper's robustness tables. Variable naming convention confirmed (e.g., `earn2008`, `earn0809`).
- **Weights**: Unweighted variant is a standard robustness check for the sampling weights.
- **Alternative outcomes (G1)**: `wage2009`, `se2009`, `any_wage2009`, `any_se2009` are confirmed in the code (generated from `wage{year}` and `se{year}` variables).
- **Missing axis**: The `source` variable in the do-file toggles between `ohp_all_ever_ssa` and `ohp_all_ever_ad2008` for the first-stage variable across time periods, but this only matters for IV estimates (excluded from core surface). No issue.

## D) Controls Multiverse Policy

- `controls_count_min: 8` (nnn* dummies always required). This is correct -- the paper states there are 8 lottery draw categories.
- `controls_count_max: 19` (nnn* + lagged outcome + 9 lottery_list demographics + 1 constant-absorbed). The maximum should actually be ~18 (8 nnn* + 1 lagged + 9 demographics = 18). **Minor correction**: Updated to 18 in JSON.
- Lottery draw FE (`nnn*`) are mandatory for design integrity. This constraint is correctly documented.
- `linked_adjustment: false` is correct -- no bundled estimator issue in the ITT analysis.

## E) Inference Plan

- **Canonical**: Cluster at `reservation_id` (household/randomization unit). Matches the paper. Correct.
- **Variants**: HC1 and HC3 are reasonable stress tests. No issue.
- Note: The paper uses `cluster(reservation_id)` consistently. No concerns about inference choices.

## F) Budgets and Sampling

- G1: 60 specs is reasonable for 3 baseline outcomes x ~20 variants. Feasible.
- G2: 70 specs for 8 outcomes x ~8 variants. Feasible.
- Controls subset sampler: exhaustive. Correct -- the control pool is small and structured (mandatory nnn*, optional lagged outcome, optional demographics block).
- Seed: 112791. Documented.

## G) Diagnostics Plan

- Balance of covariates (Table A3): Appropriate.
- Attrition differential: Appropriate (match rate difference).
- First-stage (noncompliance): Appropriate for the IV supplementary analysis.

---

## Key Constraints and Linkage Rules

1. **Lottery draw FE (`nnn*`)**: Always included. Removing them would violate the randomization design.
2. **Weights (`weight_ssa_admin`)**: Only used for 2009 and 2008-2009 outcomes (confirmed in code: `local weight09="weight_ssa_admin"`, `local weight08="weight_ssa_admin"`, but 2007 and pre-2008 use `noweight`). This means the unweighted variant for 2008 outcomes is actually the paper's default for earlier periods.
3. **Lagged outcome**: Optional precision control; dropping it is a valid robustness check.
4. **Data availability**: SSA administrative data is not publicly available. Code is provided but data must be obtained through restricted access. This is flagged in the surface.

---

## What's Missing

- Nothing significant is missing. The surface is comprehensive for the paper's primary claims.
- The pre-specified earnings variables (`wage`, `se`, `any_wage`, `any_se`) are included in G1 alternative outcomes.
- Economic self-sufficiency indices (Table A9) are correctly excluded as secondary.

---

## Changes Made to JSON

1. Corrected `controls_count_max` from 19 to 18 in both G1 and G2 constraints (8 nnn* + 1 lagged outcome + 9 lottery_list = 18).
2. No other structural changes needed.

---

## Final Assessment

**APPROVED TO RUN.** The surface is well-structured, faithful to the paper's revealed specification search, and comprehensive. The two baseline groups correctly separate the labor supply and program participation claims. Data unavailability is the main practical constraint but is clearly flagged.
