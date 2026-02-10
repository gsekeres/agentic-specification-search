# Verification Report: 112870-V2

## Paper Information
- **Title**: Optimal Life Cycle Unemployment Insurance
- **Authors**: Michelacci and Ruffo
- **Journal**: AER
- **Total Specifications**: 52

## Baseline Groups

### G1: UI Benefits-Unemployment Elasticity
- **Claim**: Higher UI benefits are associated with higher unemployment, with the elasticity varying by age group.
- **Baseline specs**: `baseline` (age-structured, coef +0.508), `baseline_pooled` (two-way FE, coef -0.476)
- **Expected sign**: Negative (in pooled specification: higher benefits -> more unemployment, so positive log-log elasticity means higher benefits = higher unemployment)
- **Baseline coefficients**: 0.508 (SE 0.071) and -0.476 (SE 0.074)
- **Outcome**: `lnun` (log unemployment rate)
- **Treatment**: `lnwba` (log weekly benefit amount)

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baselines)** | **32** | |
| core_controls | 10 | 2 baselines + no controls, demographics, education, 5 leave-one-out drops |
| core_fe | 5 | No FE, state FE only, year FE only, two-way FE, first difference |
| core_sample | 12 | Age groups (5), young/old (2), early/late period (2), drop years (5), trim (2), prime age, part splits (2) |
| core_inference | 2 | No clustering, year clustering |
| core_funcform | 5 | Levels, log-level, level-log, linear trend, quadratic trend |
| **Non-core tests** | **20** | |
| noncore_heterogeneity | 6 | Married/white interactions, unmarried/married/nonwhite/white subsamples |
| noncore_placebo | 3 | Lag treatment, lead treatment, randomized treatment |
| noncore_alt_outcome | 0 | |
| **Total** | **52** | |

Note: robust/sample/age_group_2 through age_group_6 are classified as core_sample since they test the main relationship on age subsamples central to the paper's thesis. The heterogeneity subsamples by marital status and race are classified as noncore since they test a different dimension of heterogeneity.

## Robustness Assessment

**Moderate** support. The pooled negative elasticity is robust across controls and time periods. However, results are highly sensitive to FE structure (no FE yields near-zero coefficient) and show strong age heterogeneity (young positive, old negative), which is the paper's key finding.
