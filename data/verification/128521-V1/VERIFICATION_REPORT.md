# Verification Report: 128521-V1

## Paper Information
- **Title**: Recessions, Mortality, and Migration Bias: Evidence from the Lancashire Cotton Famine
- **Method**: Difference-in-Differences
- **Total Specifications**: 58

## Baseline Groups

### G1: Aggregate Mortality Rate
- **Claim**: Aggregate mortality rates appear to decrease in cotton districts during the famine due to migration bias.
- **Baseline specs**: `baseline`, `baseline_controls`, `baseline_nearby`, `baseline_continuous`
- **Expected sign**: Negative (demonstrating migration bias)
- **Baseline coefficient**: -5.51 (p=0.19, unweighted); -4.92 (p=0.002, with controls)
- **Outcome**: `agg_mr_tot`
- **Treatment**: `cotton_dist_post`

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests** | **31** | |
| core_controls | 13 | 4 baselines + leave-one-out + incremental controls |
| core_fe | 5 | No FE, unit only, time only, region x time, county x time |
| core_sample | 12 | Drop regions, trim, winsorize, population subgroups, weights |
| core_inference | 3 | No cluster, county cluster, region cluster |
| core_funcform | 2 | IHS outcome, density squared |
| **Non-core tests** | **27** | |
| noncore_alt_outcome | 7 | Age-specific mortality rates |
| noncore_alt_treatment | 4 | Continuous treatment, nearby district spillovers |
| noncore_heterogeneity | 5 | Density, elderly share, region, cotton share interactions |
| noncore_placebo | 2 | Nearby effect, population growth |
| **Total** | **58** | |

## Robustness Assessment

**STRONG** support for the migration bias finding. 84.5% negative, 62.1% significant at 5%. Robust to controls, clustering, and sample restrictions. Age-specific outcomes show largest effects for under-15s.
