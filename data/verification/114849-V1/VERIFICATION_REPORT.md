# Verification Report: 114849-V1

## Paper
**Title**: Does Prison Make People More Criminal? Evidence from Italian Mass Pardons  
**Journal**: AEJ-Applied  
**Method**: Instrumental Variables (IV/2SLS)

## Baseline Groups

### G1: Incapacitation effect of incarceration on crime
- **Claim**: Releasing prisoners via collective pardons increases total crime (negative coefficient on log weighted change in jail population).
- **Baseline spec_id**: baseline
- **Outcome**: lchange_all (log change in total crime)
- **Treatment**: lwchange_jail (log weighted change in jail population)
- **Expected sign**: Negative
- **Estimation**: IV-2SLS with year dummies and special dummies, clustered by region
- **Baseline coefficient**: -0.223 (p = 0.071)

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **52** |
| Baseline | 1 |
| Core tests (incl. baseline) | 43 |
| Non-core | 5 |
| Invalid | 4 |
| Unclear | 0 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_sample | 29 | Drop-region, drop-pardon-year, trimming, winsorizing, subperiod, geographic subsample |
| core_inference | 4 | Clustering variations (year, none, south, decade) |
| core_funcform | 4 | Levels, delta-log, standardized, unadjusted treatment timing |
| core_method | 3 | Baseline IV-2SLS, LIML, OLS |
| core_controls | 2 | Year dummies only, no special dummies |
| core_fe | 1 | Region + year FE |
| noncore_diagnostic | 2 | First stage and reduced form regressions |
| noncore_heterogeneity | 2 | Small population regions, 1960s decade only |
| noncore_placebo | 1 | Exogeneity test (lagged crime predicting pardons) |
| invalid | 4 | Computational failures (p-values exceeding 1.0) |

## Top 5 Most Suspicious Rows

1. **iv/controls/none** (p=2.89): No-controls IV specification produces a p-value of 2.887, far exceeding 1.0. This indicates a computational error in the standard error or test statistic calculation, likely from a weak/imprecise IV with no controls. Marked invalid.

2. **iv/controls/minimal** (p=8.50): Minimal controls (only special dummies year90, year91) produces p=8.504. Same computational issue as above. Marked invalid.

3. **robust/control/year_dummies_5yr** (p=5.20): Five-year interval year dummies produces p=5.205. Likely too few controls for the IV to be well-identified. Marked invalid.

4. **robust/sample/pardon_years_only** (p=3.15): Restricting to pardon years only with no controls produces p=3.149. Very small effective sample with no controls leads to computational failure. Marked invalid.

5. **robust/treatment/unadjusted** (spec_id: robust/treatment/unadjusted): Uses lchange_jail (unadjusted) instead of lwchange_jail (timing-adjusted). The treatment variable name differs from baseline, which could be interpreted as an alternative treatment definition. Classified as core_funcform with lower confidence (0.8) since the paper explicitly discusses timing adjustments as part of the same estimand.

## Notes on Classification Decisions

### OLS specification (iv/method/ols)
Classified as core_method (confidence 0.85). The OLS estimate (coef=0.0005, p=0.98) is a direct comparison to the IV estimate, showing the endogeneity bias. While it is not an IV specification, it tests the same claim under different identification assumptions and is a standard robustness comparison in IV papers.

### Heterogeneity subsample specs
Two specs (robust/heterogeneity/small_pop and robust/heterogeneity/decade_60s) are classified as non-core heterogeneity rather than core sample restrictions. The distinction is that these represent subpopulations where the effect might differ (heterogeneity), rather than tests of whether the overall result is robust to sample composition. The 1960s-only spec (coef=0.013, essentially zero) shows a qualitatively different result from the baseline, reinforcing that this is a heterogeneity finding rather than a robustness check.

### Clustering variations
Four clustering specs (year, none, south, decade) all have the same point estimate (-0.223) as the baseline, confirming they only change inference (standard errors). This is expected behavior for clustering variations.

### Identical coefficients
Several specs have identical coefficients to the baseline (-0.2226): balanced_panel and dlog_outcome. The balanced panel spec likely indicates the panel was already balanced. The dlog_outcome spec confirms that lchange_all is already defined as delta-log, so dlog_all is an algebraically identical transformation.

## Recommendations for Spec-Search Script

1. **Fix p-value computation**: Four specs have p-values exceeding 1.0. The estimation script should check for this and either flag or re-estimate using a more numerically stable method. These likely arise from weak-instrument issues in underspecified models.

2. **First stage F-statistic**: All first_stage_F values are null. For an IV paper, the first-stage F-statistic is critical for assessing instrument strength. The script should compute and report this.

3. **Deduplicate redundant specs**: robust/control/year_dummies_only and robust/control/no_special_dummies appear to be identical (same coefficient -0.113, same p-value 0.524). The script should check for and avoid generating duplicate specifications.

4. **Consider adding South-only subsample**: The script includes north_only but not a south_only counterpart, which would be a natural complement.

5. **Consider crime-type outcomes**: The SPECIFICATION_SEARCH.md notes the effect varies by crime type (property vs violent), but only lchange_all (total crime) is used as outcome. Adding property crime and violent crime outcomes would capture distinct baseline claims from the paper.
