# Verification Report: 193625-V1

## Paper Information

- **Paper ID**: 193625-V1
- **Title**: Differential Effects of Recessions on Graduate Income by College Tier
- **Journal**: AEJ: Policy 2025
- **Method**: Difference-in-Differences
- **Verification Date**: 2026-02-04

## Baseline Groups

### G1: Main DiD effect of recession severity on graduate income

- **Claim**: Graduates from universities in severely recession-affected commuting zones experience lower median income post-recession (the DiD interaction `treat = post x badreccz` is negative).
- **Baseline spec_id**: `baseline`
- **Outcome**: `lnk_medpos` (log median positive income)
- **Treatment**: `treat` (post x badreccz)
- **Fixed Effects**: university (super_opeid) + cohort
- **Controls**: Full controls (parental income quintiles, lncount, parental top percentiles, female)
- **Cluster**: super_opeid
- **Baseline coefficient**: -0.035 (p < 1e-15)
- **Expected sign**: Negative

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **86** |
| Core tests (is_core_test=1) | 60 |
| Non-core tests (is_core_test=0) | 26 |
| Invalid | 3 |
| Baselines (is_baseline=1) | 1 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_controls | 23 | Control set variations (LOO drops, no controls, progressive additions) |
| core_sample | 23 | Sample restrictions (tier subsamples, LOO cohort drops, winsorize, trim) |
| core_fe | 5 | Fixed effects structure variations (unit-only, time-only, twoway, none, tier x cohort) |
| core_funcform | 4 | Functional form changes (log median, log mean, levels, IHS) |
| core_method | 3 | Treatment definition changes (continuous shock, median threshold, 75th percentile threshold) |
| core_inference | 2 | Clustering variations (CZ-level, heteroskedasticity-robust) |
| noncore_heterogeneity | 10 | Interaction terms with tier dummies, female share, parental income, univ size, elite status |
| noncore_alt_outcome | 9 | Income quintile fractions (k_q1-k_q5), top percentile fractions (k_top10pc/5pc/1pc), zero income |
| noncore_alt_treatment | 2 | Single-arm CZ subsamples with treatment = `post` instead of `treat` |
| noncore_placebo | 2 | Fake treatment years (1981, 1982) in pre-period |
| invalid | 3 | Missing coefficients due to collinearity or ill-defined samples |

## Top 5 Most Suspicious / Noteworthy Rows

1. **`did/fe/unit_only`** (row 2): Coefficient = -0.317, which is an order of magnitude larger than the baseline (-0.035). Without cohort FE, the `treat` variable absorbs general time trends across the post-recession period, making the coefficient uninterpretable as a pure recession effect. Classified as core_fe since it tests FE robustness, but the estimate is not comparable to the baseline in magnitude.

2. **`did/fe/cz_x_cohort`** (row 5): Empty coefficient. CZ x cohort FE likely absorbs all variation in the `treat` variable (which is defined as post x badreccz, i.e., it is a CZ-cohort-level variable). This is a fundamental identification issue, not a data problem. Correctly marked invalid.

3. **`robust/sample/severe_cz_only`** and **`robust/sample/mild_cz_only`** (rows 55-56): These specs change the treatment variable from `treat` to `post`. Within a single arm of the CZ classification, `treat = post * badreccz` reduces to either `post` or 0. The resulting coefficient is not testing the same estimand. Both have empty coefficients. Classified as noncore_alt_treatment.

4. **`robust/sample/pre_recession_cohorts`** and **`robust/sample/post_recession_cohorts`** (rows 53-54): Empty coefficients. Within a single time period, `treat` is either always 0 (pre) or equals `badreccz` (post). The pre-only sample makes `treat` perfectly collinear (all zeros), so estimation fails. These are structurally invalid decompositions of the DiD.

5. **`did/fe/twoway`** (row 4): Numerically identical to `baseline` (same coefficient to machine precision). This is redundant because the baseline already uses two-way FE (super_opeid + cohort). Not suspicious per se, but inflates the spec count without adding information. Classified as core_fe.

## Robustness of Core Tests

Among the 60 core test specifications:
- 55 have valid (non-missing) coefficients
- 53 of 55 valid core tests (96.4%) show negative coefficients, consistent with the baseline
- 2 positive coefficients appear in tier-specific subsamples: tier1_only (+0.023, p=0.54) and elite_only (+0.001, p=0.97), both statistically insignificant. This is consistent with the paper's finding that elite colleges may protect graduates.
- The coefficient range among core tests with the same outcome (lnk_medpos) is [-0.317, +0.023], but excluding the extreme unit_only FE spec, the range narrows to [-0.060, +0.023].

## Recommendations for the Spec-Search Script

1. **Remove structurally invalid specs**: The pre-recession-only, post-recession-only, severe-CZ-only, and mild-CZ-only specifications are fundamentally ill-defined for a DiD design. These should be excluded from future runs or flagged automatically.

2. **Flag redundant specs**: `did/fe/twoway` is numerically identical to `baseline` and should be deduplicated.

3. **CZ x cohort FE identification**: The `did/fe/cz_x_cohort` specification fails because the treatment varies at the CZ-cohort level. The script should detect when FE absorb the treatment variable and skip such specs.

4. **Separate heterogeneity from core tests**: The 10 heterogeneity specs (interaction terms) are correctly tagged with `robust/heterogeneity/` path prefixes, but the SPECIFICATION_SEARCH.md counts them alongside main outcome specs. These should be clearly separated in summary statistics.

5. **Alternative outcomes as distinct baseline groups**: If the paper also makes claims about income distribution (quintile fractions), these could be set up as additional baseline groups (G2, G3, etc.) rather than treated as alternative outcomes of G1. This would depend on whether the paper presents these as core claims or supplementary evidence.
