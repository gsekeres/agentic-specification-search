# Verification Report: 116540-V2

## Paper Information
- **Paper ID**: 116540-V2
- **Title**: Individual Development Accounts and Homeownership among Low-Income Renters
- **Journal**: AEJ: Policy
- **Method**: RCT with cross-sectional OLS (Linear Probability Model)

## Baseline Groups

### G1: ITT effect of IDA program on homeownership
- **Baseline spec_ids**: baseline
- **Outcome**: own_home_u42 (homeownership at Wave 4)
- **Treatment**: treat (random assignment to IDA program)
- **Expected sign**: Positive
- **Baseline coefficient**: 0.011 (SE = 0.039, p = 0.77)
- **Notes**: Single baseline with full demographic and socioeconomic controls. The result is a clear null finding.

## Summary Counts

| Category | Count |
|----------|-------|
| Total specifications | 80 |
| Core tests | 61 |
| Non-core tests | 17 |
| Invalid | 2 |
| Unclear | 0 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 32 |
| core_sample | 21 |
| core_inference | 4 |
| core_funcform | 4 |
| core_method | 0 |
| core_fe | 0 |
| noncore_placebo | 5 |
| noncore_heterogeneity | 12 |
| noncore_alt_outcome | 0 |
| noncore_alt_treatment | 0 |
| noncore_diagnostic | 0 |
| invalid | 2 |
| unclear | 0 |

### Core controls (32 specs)
Includes the baseline, 4 control-progression builds (bivariate, demographics, socioeconomic, full), 18 leave-one-out specifications, and 9 single-covariate specifications. All use own_home_u42 as outcome and treat as treatment on the full baseline sample.

### Core sample (21 specs)
Sample restriction specifications including splits by gender, marital status, age, education, cohort timing, welfare receipt, insurance status, housing subsidy status, bank account ownership, and children. All preserve the same outcome and treatment. Note: high_income and low_income subsamples were classified as noncore_heterogeneity because the paper explicitly discusses income-based heterogeneity as a separate finding.

### Core inference (4 specs)
Three HC-variant standard errors (HC0, HC2, HC3) and one cohort-clustered specification. All produce the same point estimate (0.0112) but different standard errors.

### Core functional form (4 specs)
Age as quadratic, log income, continuous income, and income quartiles instead of binary indicators. All preserve the ITT estimand.

### Non-core heterogeneity (12 specs)
Ten interaction-term specifications (treat x covariate) that estimate differential treatment effects rather than the main ITT. Also includes the high_income and low_income subsample splits which the paper treats as heterogeneity analysis.

### Non-core placebo (5 specs)
Five placebo outcome tests using own_bus, own_car, own_ira, sat_heal_good, and sat_fin_good as outcomes instead of own_home_u42.

### Invalid (2 specs)
1. **robust/method/lpm_no_controls**: Exact duplicate of robust/build/bivariate (identical coefficient 0.0284, same N=652, same controls="No controls"). The spec_id label suggests a different method (LPM vs OLS) but for a binary outcome they are identical.
2. **robust/het/interaction_race_cau**: Coefficient is exactly 0.0 with an empty p-value. The race_cau variable has no variation in the sample (all respondents appear to be non-white), making the interaction term undefined.

## Top 5 Most Suspicious Rows

1. **robust/het/interaction_race_cau** (row 42): Coefficient = 0.0, p-value is empty/missing. The race_cau variable has no variation (the coefficient on race_cau in the baseline model is ~5.7e-18, effectively zero). This interaction term is meaningless.

2. **robust/method/lpm_no_controls** (row 39): Exact duplicate of robust/build/bivariate (row 2). Same coefficient (0.02843), same SE, same p-value, same N. The model_type is labeled "LPM" vs "OLS" but these are identical for a binary outcome. Should be removed or flagged.

3. **robust/sample/nonwhite_only** (row 33): Claims to restrict to "Non-white respondents only" but has the same N=652 as the full sample and the same coefficient as robust/loo/drop_race_cau. This suggests all respondents are non-white (consistent with race_cau having no variation), so this is not a genuine subsample restriction.

4. **robust/sample/less_than_hs** (row 37): Very small subsample (N=41) which raises serious power concerns. The coefficient (-0.116) is large and negative, opposite in sign to most other specifications, but likely driven by noise.

5. **robust/sample/college_grad** (row 64): Small subsample (N=74). Point estimate (0.088) is positive but insignificant (p=0.50). Small sample makes inference unreliable.

## Recommendations for Spec-Search Script

1. **Duplicate detection**: The script should check for exact coefficient duplicates before recording a new specification. Row 39 (LPM no controls) is identical to row 2 (bivariate) and should not be counted separately.

2. **Variable validation**: Before running interaction specifications, the script should verify that the interacted variable has variation in the sample. The race_cau interaction (row 42) is meaningless because race_cau is constant.

3. **Subsample validation**: The nonwhite_only subsample (row 33) does not actually restrict the sample (N=652 = full sample). The script should verify that sample restrictions actually reduce N before recording them.

4. **Heterogeneity classification**: The script correctly labels interaction terms in the spec_tree_path but records the interaction coefficient (treat x covariate) as the primary coefficient. This is appropriate for heterogeneity analysis but these should be clearly distinguished from main-effect specifications in downstream analysis.

5. **Logit/probit**: The SPECIFICATION_SEARCH.md notes that logit and probit models failed due to perfect prediction. Given that the outcome is binary, it would be valuable to troubleshoot these models (e.g., by reducing the control set) to provide non-LPM estimates as a robustness check.
