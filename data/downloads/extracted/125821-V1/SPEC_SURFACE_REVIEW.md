# Specification Surface Review: 125821-V1

## Paper: School Spending and Student Outcomes: Evidence from Revenue Limit Elections in Wisconsin (Baron, AEJ: Economic Policy 2022)

## Summary of Baseline Groups

**G1**: Dynamic RD effect of operational referendum passage on student outcomes, using the Cellini et al. (2010, QJE) one-step estimator. Running variable is re-centered vote share (perc - 50), cutoff at 0. The estimator uses `areg outcome [treatment lags + polynomial controls in vote share + year dummies], absorb(district_code) cluster(district_code)` with the focal parameter being the 10-year post-election average: `lincom .10*(op_win_prev1 + ... + op_win_prev10)`.

Single baseline group is appropriate despite multiple outcomes. All outcomes share the same treatment concept (operational referendum passage), the same identification strategy (dynamic RD at 50% vote share), and the same population (Wisconsin school districts). The different outcomes represent distinct empirical manifestations of the same causal question ("does winning a spending referendum improve student outcomes?").

### Multiple outcome baselines

The surface correctly includes 5 additional baseline spec IDs:
- `baseline__dropout_rate` (Table 5 Panel A Col 1, aw=student_count)
- `baseline__wkce_math10` (Table 5 Panel A Col 3, aw=num_takers_math10)
- `baseline__log_instate_enr` (Table 5 Panel A Col 4, unweighted + grade9lagged control)
- `baseline__rev_lim_mem` (Table 4 Col 1, unweighted; first-stage spending effect)
- `baseline__tot_exp_mem` (Table 4 Col 2, unweighted)

These are all retained. The headline claim is `advprof_math10` (main baseline spec).

## Changes Made

### A) Removed redundant design spec ID

**Removed `design/regression_discontinuity/poly/local_cubic`**: The baseline already uses cubic polynomial (Table 5 Panel A). This would duplicate the baseline.

### B) Removed 8 redundant or problematic RC spec IDs

1. **Removed `rc/controls/sets/linear_polynomial`, `rc/controls/sets/quadratic_polynomial`, `rc/controls/sets/cubic_polynomial`**: These duplicate the design polynomial variation axis. For the Cellini one-step estimator, polynomial order IS the only controls variation dimension. The design spec IDs (`design/regression_discontinuity/poly/local_linear`, `local_quadratic`) already capture this. Having both `design/*` and `rc/controls/*` for the same variation would produce duplicate rows.

2. **Removed `rc/data/focal_parameter/ten_year_avg`**: The baseline focal parameter is already the 10-year average. Redundant.

3. **Removed `rc/weights/aw_num_takers`**: This is the baseline weight for test score outcomes (advprof_math10, wkce_math10). Redundant.

4. **Removed `rc/sample/restrict/cross_section_post` and `rc/sample/restrict/cross_section_pre`**: These were already captured by `rc/joint/cross_section_rd/rdrobust_post` and `rc/joint/cross_section_rd/rdrobust_pre_placebo`. The joint specs are more appropriate since they involve a fundamentally different estimator (rdrobust instead of Cellini one-step), not merely a sample restriction.

5. **Removed `rc/form/outcome/log_outcome`**: Problematic because (a) `log_instate_enr` is already in log form, (b) `advprof_math10` is a proficiency rate (0-100%) where log transformation is not standard and potentially misleading, (c) `dropout_rate` is also a rate. If outcome transformations are needed, they should be defined per-outcome, not as a blanket spec.

### C) Added outcome-specific notes

1. **Added `outcome_specific_weights` to constraints**: Documents that weighting varies by outcome (student_count for dropout, num_takers for test scores, unweighted for spending). This is important for the runner to implement correctly.

2. **Added `outcome_specific_controls_note`**: The `log_instate_enr` outcome uniquely adds `grade9lagged` (lagged grade 9 enrollment) as an additional control beyond the standard Cellini polynomial set. Verified in `onestep_tables.do` line 110: `areg log_instate_enr $cubic grade9lagged, absorb(district_code) cluster(district_code)`.

### D) Budget adjustment

Reduced `max_specs_core_total` from 70 to 65. Revised count: 2 design specs x 6 outcomes = 12, plus 8 RC specs (some apply per-outcome: tried_both, passed_both, trim, unweighted, five_year_avg, rdrobust_post, rdrobust_pre_placebo), plus 6 baselines = approximately 50-60 specs depending on which RC specs are crossed with which outcomes. Budget of 65 is appropriate.

## Key Constraints and Linkage Rules

- **Linked adjustment**: Yes. Operational and bond referendum polynomial controls are always varied jointly (both op_percent_prev and bond_percent_prev at the same polynomial degree). The code confirms this: `$cubic`, `$quadratic`, `$linear` globals all specify both sets at the same order.
- **Controls variation**: Not applicable in the traditional sense. The Cellini estimator has a fixed structure; variation is through polynomial order only.
- **Focal parameter**: The scalar summary is `lincom .10*(op_win_prev1 + ... + op_win_prev10)` (10-year average) or `.20*(op_win_prev1 + ... + op_win_prev5)` (5-year average). The full lag coefficients should be stored in `coefficient_vector_json` with the focal parameter computed via linear combination.
- **Cross-sectional RD variants**: The `rc/joint/cross_section_rd/*` specs use a fundamentally different estimator (`rdrobust`) on cross-sectional data rather than the panel Cellini estimator. These change both the estimator and the data structure, making them "joint" variations. They are correctly classified under `rc/joint/` rather than as separate baseline groups since they estimate the same local effect at the same cutoff.

## Budget and Sampling Assessment

- Total enumerated specs: ~50-60 depending on outcome x RC crossing rules. Within budget of 65.
- No control-subset sampling needed: polynomial order is a discrete 3-level variable, not a combinatorial control space.
- The universe is fully tractable.

## What's Missing

1. **Reading test scores** (Table B1): The paper reports results for reading proficiency in addition to math. These are excluded from the core surface but could be added as `explore/outcome` variants. Not blocking.

2. **Bond referendum effects** (Table 7 Panel B): Different treatment concept (capital spending). Correctly excluded from G1, could be a separate explore/treatment group. Not blocking.

3. **Grade 8 and Grade 4 test scores** (Tables B1 B-C): Additional outcomes at different grade levels. Could expand the baseline outcome set. Not blocking.

4. **No `rc/form/outcome` specs remain after cleanup**: If an outcome transformation is desired (e.g., standardized effect sizes), it should be defined as a per-outcome spec with explicit interpretation notes. Not blocking.

## Verification Against Code

- **Global definitions**: Verified in `onestep_tables.do` lines 29-45. The `$cubic` global includes `op_win_prev* bond_win_prev* yrdums* op_ismeas_prev* bond_ismeas_prev* op_month_prev* bond_month_prev* op_percent_prev* op_percent2_prev* op_percent3_prev* bond_percent_prev* bond_percent2_prev* bond_percent3_prev* recurring_prev* op_numelec_prev* bond_numelec_prev*`.
- **Focal parameter**: `$tenyr_op` = `.10*(op_win_prev1 + ... + op_win_prev10)` (line 48-49). `$fiveyr_op` = `.20*(op_win_prev1 + ... + op_win_prev5)` (line 56).
- **Estimation**: `areg outcome $cubic [aw=weight], absorb(district_code) cluster(district_code)` then `lincom $tenyr_op`. Confirmed for all outcomes.
- **Sample restrictions**: `if tried_both==1` (line 164-174), `if passed_both==1` (line 178-188) -- confirmed in Table 7 code.
- **Weights**: dropout uses `[aw=student_count]` (line 101), math10 uses `[aw=num_takers_math10]` (line 104), wkce uses `[aw=num_takers_math10]` (line 107), log_instate_enr is unweighted + grade9lagged control (line 110), spending outcomes unweighted (lines 82-86).
- **Dataset**: `onestep_panel_tables.dta` is in `Data/Final/`. Confirmed present.
- **Cross-sectional RD**: `itt_cross_section_results.do` uses rdrobust on `itt_cross_section.dta` with `perc` re-centered by `replace perc= perc-50`.
- **DCdensity**: `DCdensity.ado` is present in `Do Files/` directory.

## Final Assessment

**Approved to run.** The surface is conceptually coherent and faithful to the Cellini dynamic RD estimator framework. Key changes: (1) removed 9 redundant or problematic specs, (2) added outcome-specific weighting and control documentation, (3) adjusted budget. The linked adjustment constraint is correctly specified, and the multiple-outcome baseline structure is appropriate for this paper. No blocking issues.
