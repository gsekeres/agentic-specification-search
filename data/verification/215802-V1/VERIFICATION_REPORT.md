# Verification Report: 215802-V1

## Paper
- **Title**: Long-Run Impacts of Childhood Access to the Safety Net
- **Authors**: Hoynes, Schanzenbach & Almond
- **Journal**: American Economic Review
- **Paper ID**: 215802-V1

## Critical Context

This paper **cannot be properly replicated** from the public replication package. The original identification strategy requires restricted PSID data with county-of-birth identifiers and county-level Food Stamp Program (FSP) rollout timing. The specification search instead uses **current food stamp receipt** as the treatment variable in cross-sectional OLS (for health outcomes) and panel fixed effects (for income, employment, work limitation, and hospitalization). This represents a fundamentally different research question from the original paper.

## Baseline Groups

| Group | Spec IDs | Outcome | Treatment | Expected Sign |
|-------|----------|---------|-----------|---------------|
| G1 | baseline, baseline_year_fe, baseline_cohort_fe | good_health | fs_receipt | - |
| G2 | panel/income/baseline | log_income | fs_receipt | - |
| G3 | panel/employed/baseline | employed | fs_receipt | - |
| G4 | panel/worklimit/baseline | work_limited | fs_receipt | + |
| G5 | panel/hospital/any | any_hospital | fs_receipt | + |

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **80** |
| Baseline specs | 8 |
| Core test specs | 47 |
| Non-core specs (excl baselines) | 25 |
| Invalid | 0 |
| Unclear | 0 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 11 |
| core_fe | 17 |
| core_funcform | 5 |
| core_inference | 6 |
| core_sample | 21 |
| noncore_alt_outcome | 8 |
| noncore_alt_treatment | 5 |
| noncore_heterogeneity | 10 |
| noncore_placebo | 3 |

## Detailed Category Counts

- **core_controls**: 11 (control progression, leave-one-out)
- **core_sample**: 21 (gender, race, cohort, education, household role subsamples across all groups)
- **core_inference**: 6 (clustering and SE variations)
- **core_fe**: 17 (FE structure variations across all baseline groups)
- **core_funcform**: 5 (age polynomial, log age, education dummies, poor_health flip)
- **core_method**: 0 (no estimation method changes)
- **noncore_placebo**: 3 (yob outcome, random treatment, pre-reform)
- **noncore_alt_outcome**: 8 (health_scale, excellent_health, log_hospdays, CS versions of panel outcomes, ordinal)
- **noncore_alt_treatment**: 5 (AFDC, any_welfare, both_programs, fs_only, baseline_any_welfare)
- **noncore_heterogeneity**: 10 (interaction terms for gender, race, age, education, marital, head, femhead, cohort, famsize, kids)
- **noncore_diagnostic**: 0
- **invalid**: 0
- **unclear**: 0

## Top 5 Most Suspicious Rows

1. **robust/cluster/race**: Clustering by race produces only 2 clusters, yielding t-stat of -30.1 and p=0.001. Artifact of too-few clusters. SE (0.0006) implausibly small. Confidence: 0.70.

2. **robust/cluster/sex**: Clustering by sex also produces only 2 clusters. SE (0.027) unreliable. Confidence: 0.70.

3. **robust/outcome/poor_health**: Coefficient = +0.0180, exactly negative of baseline (-0.0180). poor_health = 1 - good_health, algebraic flip. Classified as core_funcform. Confidence: 0.90.

4. **robust/funcform/ordinal** and **baseline_health_scale**: Identical coefficients (0.0292) and SEs. Appear to be duplicate regressions using health_scale. Both classified as noncore_alt_outcome. Confidence: 0.80.

5. **robust/placebo/yob_outcome**: Year of birth as outcome. Coefficient (0.875) significant (p=0.008). Concerning for placebo -- suggests FS receipt correlated with birth year, reflecting cohort confounding. Confidence: 0.95.

## Recommendations

1. **Fundamental limitation**: Cannot test the paper actual claim (childhood FSP exposure effects) without restricted data. All results reflect correlates of current program participation.

2. **Remove race/sex clustering specs or flag them**: Only 2 clusters produces unreliable inference. Should be flagged as potentially invalid.

3. **Consider merging poor_health with baseline**: Algebraically redundant (1 - good_health). Adds no independent information.

4. **Deduplicate ordinal/health_scale**: robust/funcform/ordinal and baseline_health_scale are the same regression.

5. **Panel CS counterparts**: CS versions of panel outcomes (work_limited_cs, employed_cs, log_income_cs) classified as noncore_alt_outcome due to fundamentally different estimation strategy (no individual FE).

6. **Heterogeneity specs**: All 10 report main effect of fs_receipt (not interaction coefficient). Main effect meaning changes with interaction present -- classified as noncore_heterogeneity.
