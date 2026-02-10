# Verification Report: 138401-V1

## Paper Information
- **Title**: The Long-Term Effects of Measles Vaccination on Earnings and Employment
- **Method**: Panel fixed effects with state x cohort variation
- **Data**: ACS 2000-2017 (16 million observations)
- **Total Specifications**: 38

## Baseline Groups

### G1: Employment
- **Claim**: Measles vaccination exposure affects long-run employment.
- **Baseline spec**: `baseline`
- **Expected sign**: Positive (though observed sign is negative)
- **Baseline coefficient**: -0.00074 (p < 0.001)
- **Outcome**: `employed`
- **Treatment**: `treatment` (high measles state x post-vaccine cohort)

### G2: Log Wages
- **Claim**: Measles vaccination exposure affects long-run wages.
- **Baseline spec**: `baseline_lnwage`
- **Expected sign**: Positive (though observed sign is negative)
- **Baseline coefficient**: -0.0015 (p < 0.001)
- **Outcome**: `lnwage`
- **Treatment**: `treatment`

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests** | **31** | |
| core_controls | 5 | 2 baselines + no controls + black only + female only |
| core_fe | 4 | bpl only, year only, birthyr only, bpl+year |
| core_sample | 17 | Gender, race, cohort, year, region, measles intensity |
| core_inference | 3 | bpl cluster, birthyr cluster, robust SE |
| core_funcform | 1 | Winsorized wages |
| **Non-core tests** | **7** | |
| noncore_alt_outcome | 2 | lnwage (in robust/outcome), lnwage males |
| noncore_heterogeneity | 2 | Treatment x black, treatment x female |
| noncore_placebo | 1 | Random treatment |
| **Total** | **38** | |

## Robustness Assessment

**MODERATE** support (with caveats). 84.2% of specs significant, but the coefficient sign is negative, which is unexpected. The simplified treatment construction (median split) differs from the original paper's continuous measure. The massive sample size (16M) means even tiny effects are statistically significant. Placebo test with random treatment shows no effect (as expected). Some subsamples (females only) had insufficient variation.
