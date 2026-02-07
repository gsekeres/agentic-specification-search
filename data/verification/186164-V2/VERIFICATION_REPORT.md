# Verification Report: 186164-V2

## Paper Information
- **Title**: Reducing Inequality Through Dynamic Complementarity: Evidence from Head Start and Public School Spending
- **Authors**: Rucker C. Johnson and C. Kirabo Jackson
- **Journal**: AEJ: Economic Policy

## Baseline Groups

### G1: SFR -> Per-Pupil Expenditure (First Stage)
- **Baseline spec_ids**: baseline
- **Claim**: School finance reforms (SFR) increase per-pupil school expenditure
- **Expected sign**: Positive
- **Baseline coefficient**: 0.021 (p=0.51, not significant with state-level clustering)
- **Outcome**: outcome1 (log per-pupil expenditure)
- **Treatment**: post_sfr (post-school-finance-reform indicator)
- **Note**: This is the first-stage relationship for the paper's IV strategy. The paper's main results on educational attainment, wages, and poverty use individual-level PSID data that is not available in the replication package.

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **67** |
| Baseline | 1 |
| Core tests (is_core_test=1) | 37 |
| Non-core tests | 28 |
| Invalid | 2 |
| Unclear | 0 |

Note: The baseline is included in the core test count (is_core_test=1 and is_baseline=1). Total = 37 core (incl baseline) + 28 non-core + 2 invalid = 67.

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_controls | 13 | Control set variations (add/drop controls, control progressions) |
| core_sample | 11 | Sample restrictions (time period splits, trimming, winsorizing, weights, year drops, reform-state-only) |
| core_fe | 4 | Fixed effects variations (year, region, division, none) |
| core_funcform | 4 | Functional form changes (levels, IHS, standardized, real dollars) |
| core_inference | 3 | Clustering variations (district, division, region) |
| core_method | 2 | Estimation method changes (first difference, long difference) |
| noncore_heterogeneity | 18 | Heterogeneity subsample splits and interaction terms |
| noncore_alt_treatment | 5 | Alternative treatment definitions (different reform types) |
| noncore_placebo | 5 | Placebo and permutation tests |
| invalid | 2 | Exact duplicates of baseline |

## Top 5 Most Suspicious Rows

1. **did/fe/state (row 6)**: This is an exact numerical duplicate of the baseline specification. Same coefficient (0.020589), same SE (0.031292), same p-value (0.5106), same N (324321), same R-squared (0.5158). The FE field says "State FE" while baseline says "FIPSTATE" but these are the same thing. Marked invalid.

2. **robust/cluster/state (row 24)**: Also an exact duplicate of baseline. The baseline already clusters at the state level, so "state clustering" is not a variation. Same coefficient and p-value. Marked invalid.

3. **robust/placebo/lead (row 58)**: The lead placebo shows a coefficient of 0.084 (p=0.003), which is larger and more significant than the baseline (0.021, p=0.51). This is concerning for identification because it suggests pre-trends or anticipation effects, but it is correctly classified as a placebo.

4. **robust/placebo/pre_treatment (row 54)**: The pre-treatment placebo shows a positive coefficient (0.062, p=0.055) that is marginally significant. Combined with the lead test, this raises concerns about pre-existing trends in spending that may confound the SFR effect.

5. **did/controls/none (row 11) and did/controls/minimal (row 12)**: The no-controls specification shows a coefficient of 0.365 (p<0.001) and the minimal-controls spec shows 0.245 (p<0.001), both dramatically larger than the baseline (0.021). This huge sensitivity to controls inclusion suggests substantial omitted variable bias in the uncontrolled estimates and raises questions about the identifying variation.

## Key Classification Decisions

### Treatment variations (noncore_alt_treatment)
The five treatment variations (intensity, court_order, eq_spend, tax_limit, spend_limit) change the treatment variable from the baseline's post_sfr to different reform type indicators. While these could be considered robustness checks if the paper treats all reform types as equivalent instruments, they substantively change the causal object being estimated (e.g., the effect of court-ordered reforms vs. legislative reforms). Conservatively classified as noncore_alt_treatment.

### Regional and income subsamples (noncore_heterogeneity)
Regional subsamples (Northeast, Midwest, South, West) and income quartile subsamples are classified as noncore_heterogeneity because they partition the sample to examine effect heterogeneity rather than providing alternative estimates of the average treatment effect. The results show dramatic variation: South (+12.5%) and Northeast (+10.6%) vs. Midwest (-0.7%) and West (-5.3%), suggesting these are not robustness checks but rather heterogeneity analysis.

### Poverty/urban subsamples from heterogeneity section (noncore_heterogeneity)
The high_poverty, low_poverty, urban, and rural subsamples from the robustness/heterogeneity path are classified as noncore_heterogeneity for the same reason as regional subsamples.

### Time period splits (core_sample)
The early_period and late_period splits are classified as core_sample (rather than heterogeneity) because time period robustness is a standard check for the stability of a treatment effect, even though the early period shows a negative coefficient.

## Recommendations

1. **Remove duplicate specifications**: did/fe/state and robust/cluster/state are exact duplicates of the baseline and should be de-duplicated in the analysis script.

2. **Re-examine treatment variations**: If the paper treats all SFR types as equivalent instruments, these could be reclassified as core_method. The current classification is conservative.

3. **Note data limitation**: The fundamental limitation is that this specification search tests the first-stage relationship (SFR -> spending) using district-level data, not the paper's main claim about dynamic complementarity of Head Start and K12 spending on individual outcomes. Any meta-analysis using these results must note this distinction.

4. **Pre-trends concern**: The positive and significant lead/placebo coefficients (rows 54, 58) are a methodological red flag that warrants discussion. The analysis script should prominently flag these results.

5. **Controls sensitivity**: The dramatic change in coefficient from 0.365 (no controls) to 0.021 (full controls) suggests the identifying variation is sensitive to time-varying confounders interacted with baseline characteristics. This merits scrutiny.
