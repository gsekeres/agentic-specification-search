# Verification Report: 125321-V1

## Paper
**Title**: Can Technology Solve the Principal-Agent Problem? Evidence from China's War on Air Pollution
**Authors**: Michael Greenstone, Guojun He, Ruixue Jia, Tong Liu
**Journal**: AER
**Method**: Regression Discontinuity Design (staggered)

## Baseline Groups

### G1: Effect of automated monitoring on reported PM10
- **Baseline spec_ids**: baseline
- **Claim**: Automation of air pollution monitoring stations increases reported PM10 levels by approximately 30 ug/m3, indicating prior manipulation/under-reporting of pollution.
- **Expected sign**: Positive (automation reveals previously hidden pollution)
- **Outcome**: pm10
- **Treatment**: after (indicator for post-automation date)
- **Baseline coefficient**: 32.71 (SE = 4.10, p < 0.001)
- **Specification**: OLS with linear polynomial in running variable, station FE (pm10_n) + month FE, weather controls (wind_speed, rain, temp, rh), clustered at city level

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **70** |
| **Baseline** | **1** |
| **Core tests (including baseline)** | **57** |
| **Non-core tests** | **13** |
| **Invalid** | **0** |
| **Unclear** | **0** |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_controls | 14 | Control variable variations (drop one, single covariate, progressive addition, no controls) |
| core_sample | 24 | Sample restrictions (wave, deadline, year, winsorize, trim, window, donut hole) |
| core_method | 11 | Method variations (polynomial order, bandwidth, kernel) |
| core_fe | 5 | Fixed effects structure variations |
| core_funcform | 2 | Functional form (log PM10, IHS PM10) |
| core_inference | 1 | Clustering variation (station vs city) |
| noncore_placebo | 9 | Placebo tests (fake cutoffs and unaffected outcomes) |
| noncore_alt_outcome | 2 | Alternative pollutants (SO2, NO2) |
| noncore_heterogeneity | 2 | Subgroup heterogeneity (wave 1 vs wave 2 local RD) |

## Top 5 Most Suspicious Rows

### 1. rd/placebo/cutoff_-180 (coef=29.80, p<0.001)
**Issue**: This placebo cutoff test shows a large, highly significant positive coefficient (29.80) comparable to the baseline (32.71). For a valid placebo test, the coefficient should be close to zero and insignificant. The likely explanation is that the "after" variable still uses the actual automation date rather than being redefined relative to the shifted cutoff at T=-180. This means the specification is still capturing the true treatment effect, not testing for a spurious pre-existing discontinuity. The same concern applies to cutoff_-90 (coef=30.20) and cutoff_-60 (coef=31.18).

### 2. rd/placebo/cutoff_-90 (coef=30.20, p<0.001)
**Issue**: Same as above. The "after" indicator appears not to have been redefined for the shifted placebo cutoff. The coefficient is nearly identical to the baseline, strongly suggesting this is not actually a placebo test but a re-estimation of the original effect.

### 3. rd/placebo/cutoff_-60 (coef=31.18, p<0.001)
**Issue**: Same problem as the other placebo cutoff tests. The coefficient (31.18) is essentially the same as the baseline (32.71).

### 4. rd/placebo/outcome_wind_speed (coef=-0.13, p=0.0008)
**Issue**: This weather placebo outcome shows a statistically significant coefficient, which should ideally be zero. While the effect size is small (-0.13 m/s change in wind speed), significance at the 0.1% level suggests either a very large sample size inflating precision or a genuine (though small) confound. This is worth noting but does not undermine the main result given the trivial magnitude.

### 5. rd/placebo/outcome_rh (coef=-2.13, p<0.001)
**Issue**: Relative humidity shows a significant negative discontinuity at the automation date. This could indicate a seasonal/temporal confound coinciding with the automation date, though the station FE and month FE should absorb most seasonal variation. The effect is not large enough to explain the PM10 result through weather channels (the rh control coefficient in the baseline is only -0.13).

## Key Observations

### Duplicated Specifications
- **rd/kernel/triangular** is identical to **rd/bandwidth/fixed_90d** (same coefficients, same N=74,958). Both use local polynomial RD with triangular kernel and bw=90.
- **robust/control/add_1_wind_speed** is identical to **robust/control/only_wind_speed** (same coefficients, same N). Both have only wind_speed as control.
- **robust/control/add_4_rh** is identical to **baseline** (same coefficients). The final step of the control progression reproduces the baseline exactly.

### Robustness Pattern
The main finding is highly robust across parametric specifications. The baseline coefficient (~32.7) is stable across polynomial orders (32.7-32.8), control variations (25.6-34.5), year exclusions (26.4-36.8), and FE structures (22.6-33.3). All parametric specifications with the full sample yield significant positive effects.

However, local polynomial RD estimates at narrow bandwidths (60-90 days) are much smaller and insignificant (~0.9-1.8), which diverges substantially from the parametric estimates. This is a known tension in RD analysis: local estimates may differ from global polynomial fits.

### Classification Rationale for Non-Core Specs
- **SO2 and NO2 outcomes**: Classified as noncore_alt_outcome because the paper's core claim is specifically about PM10 manipulation. While effects on other pollutants are mentioned, they test a different (though related) hypothesis.
- **Placebo tests**: All 9 placebo specs (5 cutoff shifts + 4 weather outcomes) are diagnostic tests of the research design validity, not alternative implementations of the core claim.
- **Heterogeneity specs**: The wave 1/wave 2 local RD specs are subgroup analyses using a different estimation method (local polynomial with bw=90 vs parametric), making them heterogeneity tests rather than pure robustness checks.

## Recommendations for Spec-Search Script

1. **Fix placebo cutoff implementation**: The placebo cutoff tests (rd/placebo/cutoff_*) appear to shift the cutoff but not redefine the treatment indicator. The "after" variable should be 1{T >= shifted_cutoff} for each placebo, not 1{T >= 0}. Currently these specs are not valid placebos.

2. **Remove exact duplicates**: rd/kernel/triangular duplicates rd/bandwidth/fixed_90d; robust/control/add_1_wind_speed duplicates robust/control/only_wind_speed; robust/control/add_4_rh duplicates baseline. These inflate the spec count without adding information.

3. **Consider adding rdrobust estimates**: The paper's original analysis uses rdrobust for optimal bandwidth selection. Adding rdrobust-based estimates (if feasible) would provide a more direct comparison to the published results.

4. **Clarify window vs bandwidth specs**: The window restrictions (rd/sample/window_*) use parametric OLS within a restricted time window, while bandwidth specs (rd/bandwidth/*) use kernel-weighted local polynomial RD. Both are sample/method variations but they differ fundamentally in estimation approach. The spec_tree_path for window specs is listed under sample_restrictions but they could also be considered method variations.
