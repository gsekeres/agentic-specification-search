# Specification Search Log: 125821-V1

**Paper**: "School Spending and Student Outcomes: Evidence from Revenue Limit Elections in Wisconsin" - Jason Baron, AEJ: Economic Policy

**Paper ID**: 125821-V1
**Design**: Regression Discontinuity (Cellini et al. 2010 one-step dynamic RD estimator)
**Date executed**: 2026-02-25

---

## Surface Summary

- **Baseline groups**: 1 (G1)
- **Design code**: `regression_discontinuity`
- **Baseline estimator**: Cellini one-step dynamic RD: `areg outcome <controls> | district_code, cluster(district_code)`
- **Focal parameter**: 10-year average of operational referendum win coefficients: `0.10*(op_win_prev1 + ... + op_win_prev10)`
- **Running variable**: `perc` (re-centered vote share, percent in favor minus 50)
- **Cutoff**: 0 (50% vote share)
- **Polynomial order (baseline)**: Cubic (3rd order in vote share)
- **Fixed effects**: District (`district_code`)
- **Clustering**: District (`district_code`)
- **Budget**: max 65 core specs, 0 control subsets
- **Seed**: 125821
- **Surface hash**: computed from SPECIFICATION_SURFACE.json

### Baseline outcomes (6 total):
1. `advprof_math10` - Math proficiency grade 10 (headline, Table 5 Panel A Col 2), weighted by `num_takers_math10`
2. `dropout_rate` - Dropout rate, weighted by `student_count`
3. `wkce_math10` - WKCE math score grade 10, weighted by `num_takers_math10`
4. `log_instate_enr` - Log in-state postsecondary enrollment, unweighted (adds `grade9lagged` control)
5. `rev_lim_mem` - Revenue limits per member, unweighted
6. `tot_exp_mem` - Total expenditures per member, unweighted

---

## Specification Counts

| Category | Planned | Executed | Success | Failed |
|----------|---------|----------|---------|--------|
| `baseline` | 6 | 6 | 6 | 0 |
| `design/regression_discontinuity/poly/*` | 12 | 12 | 12 | 0 |
| `rc/sample/restrict/*` | 12 | 12 | 12 | 0 |
| `rc/sample/outliers/*` | 12 | 12 | 10 | 2 |
| `rc/weights/unweighted` | 3 | 3 | 3 | 0 |
| `rc/data/focal_parameter/five_year_avg` | 6 | 6 | 6 | 0 |
| `rc/joint/cross_section_rd/rdrobust_post` | 6 | 6 | 6 | 0 |
| `rc/joint/cross_section_rd/rdrobust_pre_placebo` | 6 | 6 | 6 | 0 |
| **Total estimate specs** | **63** | **63** | **61** | **2** |
| `infer/se/hc/hc1` (inference only) | 6 | 6 | 6 | 0 |
| Diagnostics | 2 | 2 | 2 | 0 |

---

## Failures

1. **`rc/sample/outliers/trim_y_1_99` (dropout_rate)**: "Matrix is singular." -- Trimming the 1st/99th percentile of dropout_rate creates too many zero-valued observations, leading to multicollinearity in the large control set (~289 regressors with absorbed district FE).

2. **`rc/sample/outliers/trim_y_5_95` (wkce_math10)**: "Matrix is singular." -- Same issue: aggressive trimming on WKCE math scores combined with the large number of election-lag control variables causes rank deficiency.

---

## Deviations from Surface

- **`rc/weights/unweighted`**: Only run for the 3 outcomes that have non-trivial baseline weights (`advprof_math10`, `dropout_rate`, `wkce_math10`). The remaining 3 outcomes (`log_instate_enr`, `rev_lim_mem`, `tot_exp_mem`) are already unweighted at baseline, so `rc/weights/unweighted` would be redundant.

- **Cross-section rdrobust specs**: The surface specifies `rc/joint/cross_section_rd/*` which switches from the Cellini one-step panel estimator to a standard local polynomial RD (rdrobust) on the cross-section data. For `log_instate_enr`, the cross-section equivalent is `perc_instate` (proportion in-state postsecondary enrollment, not log-transformed). These 12 specs use the rdrobust robust bias-corrected estimator with CCFT bandwidth.

---

## Design Variant Details

### Polynomial order (design axis)
- **Baseline (cubic)**: `op_percent_prev*, op_percent2_prev*, op_percent3_prev*` (and bond equivalents) -- 3rd-order polynomial in vote share
- **Quadratic**: Drops `op_percent3_prev*` and `bond_percent3_prev*`
- **Linear**: Drops both quadratic and cubic terms, keeps only `op_percent_prev*` and `bond_percent_prev*`

### Focal parameter variation
- **Baseline**: 10-year average: `0.10 * sum(op_win_prev1..10)`
- **5-year average**: `0.20 * sum(op_win_prev1..5)` -- shorter horizon average

---

## Software Stack

- **Python**: 3.12
- **pyfixest**: 0.40+ (for Cellini one-step panel estimation with absorbed district FE and clustered SE)
- **rdrobust**: 1.3+ (for cross-section local polynomial RD)
- **rddensity**: (for McCrary-type density test)
- **pandas**: 2.x
- **numpy**: 2.x
- **scipy**: 1.x (for t-distribution critical values in lincom)

---

## Key Results Summary

### Baseline (headline: advprof_math10, cubic, 10-year average)
- Coefficient: 5.886, SE: 1.685, p=0.0005, N=3253
- Interpretation: Passing an operational spending referendum raises math proficiency by ~5.9 percentage points (average over 10 post-election years)

### Robustness pattern
- **Spending outcomes** (rev_lim_mem, tot_exp_mem): Positive, significant effects across most specifications
- **Student outcomes** (advprof_math10, wkce_math10): Positive, significant effects that are robust to polynomial order, sample restrictions, and cross-section RD
- **Dropout rate**: Negative direction (as expected) but often marginally significant
- **Postsecondary enrollment** (log_instate_enr): Positive but often insignificant in the panel; significant in cross-section RD
- **Cross-section rdrobust post-election**: Confirms panel results; all 6 outcomes show significant effects
- **Cross-section rdrobust pre-election placebo**: All outcomes show null effects (p>0.27), supporting the validity of the RD design

### Diagnostics
- **McCrary density test**: T_jk = 1.867, p = 0.062 (marginally passes at 5% level -- no strong evidence of manipulation)
- **Covariate balance**: Pre-election balance tests on `econ_disadv_percent` and `enrollment` using rdrobust
