# Verification Report: 125821-V1

## Paper
Baron (2022), "School Spending and Student Outcomes: Evidence from Revenue Limit Elections in Wisconsin", AEJ: Economic Policy

## Design
Cellini et al. (2010) one-step dynamic regression discontinuity. Sharp RD at the 50% vote share cutoff for operational revenue limit referenda in Wisconsin school districts, 1996-2014. The focal parameter is the 10-year average post-election effect of referendum passage on student/fiscal outcomes.

## Baseline Groups Found

### G1: Effect of operational referendum passage on student/fiscal outcomes (Table 5)

Six baseline outcomes spanning three domains:

| Baseline spec_run_id | spec_id | Outcome | Coefficient | SE | p-value | N | R2 |
|---|---|---|---|---|---|---|---|
| 125821-V1_run0001 | baseline | advprof_math10 | 5.886 | 1.685 | 0.0005 | 3253 | 0.843 |
| 125821-V1_run0002 | baseline__dropout_rate | dropout_rate | -0.105 | 0.072 | 0.146 | 7614 | 0.673 |
| 125821-V1_run0003 | baseline__wkce_math10 | wkce_math10 | 4.529 | 1.701 | 0.008 | 3256 | 0.879 |
| 125821-V1_run0004 | baseline__log_instate_enr | log_instate_enr | 0.071 | 0.044 | 0.112 | 3580 | 0.965 |
| 125821-V1_run0005 | baseline__rev_lim_mem | rev_lim_mem | 315.541 | 111.876 | 0.005 | 7792 | 0.831 |
| 125821-V1_run0006 | baseline__tot_exp_mem | tot_exp_mem | 298.268 | 155.147 | 0.056 | 7792 | 0.797 |

- **Expected sign**: Positive for test scores, spending, and enrollment; negative for dropout rate.
- **Primary headline**: advprof_math10 (math proficiency, grade 10) -- Table 5 Panel A Column 2.

## Counts

| Metric | Count |
|--------|-------|
| Total rows in specification_results.csv | 63 |
| Valid (run_success=1) | 61 |
| Invalid (run_success=0) | 2 |
| Core tests (is_core_test=1) | 55 |
| Non-core (valid but not core) | 6 |
| Baseline rows | 6 |
| Inference variants (inference_results.csv) | 6 |

## Category Counts

| Category | Count |
|----------|-------|
| core_method (baselines + design poly order + rdrobust_post) | 24 |
| core_sample (restrict/outlier trim) | 24 |
| core_weights (unweighted) | 3 |
| core_data (5-year avg focal parameter) | 6 |
| noncore_heterogeneity (pre-placebo falsification) | 6 |

## Failed Rows

| spec_run_id | spec_id | Outcome | Error |
|---|---|---|---|
| 125821-V1_run0032 | rc/sample/outliers/trim_y_1_99 | dropout_rate | Matrix is singular |
| 125821-V1_run0039 | rc/sample/outliers/trim_y_5_95 | wkce_math10 | Matrix is singular |

Both failures occur when aggressive outlier trimming removes enough observations to make the Cellini one-step design matrix singular (too many parameters relative to remaining observations). The error fields are correctly populated: empty numeric scalars, CVJ contains `error` and `error_details` keys.

## Robustness Assessment

### Sign Consistency (core specs only)

| Outcome | Core specs | Expected sign | % expected sign |
|---------|-----------|---------------|-----------------|
| advprof_math10 | 10 | 10/10 | 100% |
| dropout_rate | 9 | 9/9 | 100% |
| wkce_math10 | 9 | 9/9 | 100% |
| log_instate_enr | 9 | 9/9 | 100% |
| rev_lim_mem | 9 | 9/9 | 100% |
| tot_exp_mem | 9 | 8/9 | 89% |
| **Overall** | **55** | **54/55** | **98.2%** |

The single sign reversal is in tot_exp_mem under the `passed_both` sample restriction (run0030, coef=-185.05, p=0.307), which drastically reduces the sample and yields an imprecise, insignificant estimate.

### Statistical Significance (core specs only)

| Outcome | sig at 5% | sig at 1% |
|---------|-----------|-----------|
| advprof_math10 | 10/10 (100%) | 6/10 (60%) |
| dropout_rate | 2/9 (22%) | 1/9 (11%) |
| wkce_math10 | 6/9 (67%) | 3/9 (33%) |
| log_instate_enr | 3/9 (33%) | 3/9 (33%) |
| rev_lim_mem | 7/9 (78%) | 6/9 (67%) |
| tot_exp_mem | 2/9 (22%) | 1/9 (11%) |
| **Overall** | **30/55 (54.5%)** | **20/55 (36.4%)** |

The headline outcome (advprof_math10) is robustly significant across all 10 core specifications. Fiscal outcomes (rev_lim_mem) are also robust. Dropout rate and total expenditures are less precisely estimated in the baseline and remain mostly insignificant across robustness checks.

### Placebo Pre-election Tests (non-core)

All 6 pre-election placebo RD estimates are statistically insignificant (p > 0.27), which supports the validity of the RD design. None show evidence of pre-existing trends at the cutoff.

### Inference Variants

6 HC1 (heteroskedasticity-robust without clustering) variants for the 6 baseline outcomes. Compared to the clustered baseline:
- advprof_math10: p drops from 0.0005 to 0.000004 (more significant without clustering)
- dropout_rate: p drops from 0.146 to 0.061 (still insignificant but closer)
- rev_lim_mem: p drops from 0.005 to ~0 (more significant)
- tot_exp_mem: p drops from 0.056 to 0.000005 (much more significant without clustering)

Clustering at the district level is the conservative choice; the clustered SEs are substantially larger, as expected with panel data.

## Top Issues

1. **rc/weights/unweighted only covers 3 of 6 outcomes** (advprof_math10, dropout_rate, wkce_math10). This is correct behavior: rev_lim_mem, tot_exp_mem, and log_instate_enr are already estimated without analytic weights in the baseline, so the unweighted variant would be a duplicate.

2. **No log_instate_enr outcome for rdrobust_post** was missing from the original plan but is present (run0057). All 6 outcomes are covered by the rdrobust cross-section approach.

3. **Matrix singularity in outlier trimming**: 2 of 12 outlier-trimming specifications fail. This is a structural limitation of the Cellini one-step estimator which has a large number of parameters (~100+ regressors from polynomial lag expansions). Aggressive trimming reduces the effective sample below the rank threshold.

4. **Treatment variable labels vary**: The baseline uses `op_win_prev1_through_10`, the 5-year avg specs use `op_win_prev1_through_5`, and the rdrobust specs use `op_win (vote share > 50%)`. These are all representations of the same treatment concept (operational referendum passage) but at different aggregation levels or estimation approaches. This is appropriate variation, not drift.

## Recommendations

1. **Consider adding bandwidth variation for rdrobust specs**: The cross-section rdrobust approach uses default bandwidth selection. Adding halved/doubled bandwidth variants would strengthen the design robustness dimension.

2. **Consider donut-hole RD specifications**: Excluding observations very close to the 50% cutoff would test sensitivity to strategic manipulation near the threshold.

3. **The 55 core specifications provide adequate coverage** of the specification surface for this paper. The Cellini one-step design has limited axes of variation (polynomial order, sample restrictions, weights, focal parameter aggregation window), and the run covers all planned axes.
