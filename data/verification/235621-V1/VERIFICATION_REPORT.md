# Verification Report: 235621-V1

## Paper: The Price of Experience
- **Journal**: AER (American Economic Review)
- **Topic**: Returns to labor market experience and wage dynamics
- **Method**: Panel Fixed Effects with Mincerian Wage Equations
- **Data**: Panel Study of Income Dynamics (PSID), 1968-2007

---

## Baseline Groups

### G1: Returns to Experience
- **Claim**: Labor market experience has a positive and significant effect on log wages, with diminishing returns (concave wage-experience profile).
- **Baseline spec_id**: baseline
- **Outcome**: lw (log wages)
- **Treatment**: e (years of labor market experience)
- **Expected sign**: Positive (+)
- **Baseline coefficient**: 0.0490 (SE = 0.000475, p < 0.001)
- **Interpretation**: Approximately 4.9% wage increase per additional year of experience at zero experience, controlling for experience-squared and education.

---

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **70** |
| Baseline (also counted in core) | 1 |
| Core tests (is_core_test=1) | 63 |
| Non-core tests (is_core_test=0) | 7 |
| Invalid | 0 |
| Unclear | 0 |

### Core Test Breakdown (includes baseline in core_controls)

| Category | Count |
|----------|-------|
| core_controls | 15 |
| core_sample | 30 |
| core_inference | 4 |
| core_fe | 5 |
| core_funcform | 7 |
| core_method | 2 |

### Non-Core Breakdown

| Category | Count |
|----------|-------|
| noncore_heterogeneity | 6 |
| noncore_alt_treatment | 1 |
| noncore_alt_outcome | 0 |
| noncore_placebo | 0 |
| noncore_diagnostic | 0 |

---

## Top 5 Most Suspicious Rows

1. **robust/outcome/returns_education** (spec_id: robust/outcome/returns_education)
   - **Issue**: Treatment variable is education (s), not experience (e). This is a completely different estimand (returns to education ~9.9% vs returns to experience ~4.9%). Classified as noncore_alt_treatment.
   - **Confidence**: 0.95

2. **robust/het/by_region_nonsouth** (spec_id: robust/het/by_region_nonsouth)
   - **Issue**: Labeled as heterogeneity but has identical coefficient (0.0496) to robust/sample/exclude_south. This is a subsample split, not an interaction heterogeneity analysis. The spec_tree_path says heterogeneity.md but it is functionally a sample restriction. Classified as core_sample despite label.
   - **Confidence**: 0.85

3. **robust/form/e_log** (spec_id: robust/form/e_log)
   - **Issue**: Treatment variable is log_e rather than e. The coefficient (0.219) is an elasticity, not a semi-elasticity, so it is not directly magnitude-comparable to baseline (0.049). Still tests same directional claim. Classified as core_funcform with lower confidence.
   - **Confidence**: 0.80

4. **robust/form/y_level** (spec_id: robust/form/y_level)
   - **Issue**: Outcome is wages in levels (w) not logs (lw). Coefficient of 0.354 is in dollar units, not percentage terms. Not directly comparable in magnitude to baseline. Classified as core_funcform with lower confidence.
   - **Confidence**: 0.80

5. **robust/weights/unweighted** (spec_id: robust/weights/unweighted)
   - **Issue**: Has identical coefficient (0.04642) and SE to robust/control/drop_college. May be a duplicate specification or may differ only in the weighting scheme (both unweighted). The controls listed differ ("e2, s, male, black" for unweighted vs "baseline minus college" for drop_college), so they appear to be the same underlying model but labeled differently.
   - **Confidence**: 0.90

---

## Notes on Classification Decisions

### Heterogeneity interaction specs (noncore)
Six specifications add interaction terms (male*e, black*e, college*e, late_cohort*e, late_period*e, male*college*e) to the regression. When an interaction is included, the main coefficient on e becomes the conditional effect for the omitted group (e.g., female return when male*e is added). This changes the estimand from the average population return to a subgroup-specific return. These are classified as noncore_heterogeneity because the coefficient on e no longer estimates the same quantity as the baseline.

### Subsample splits labeled as heterogeneity (core)
Two specs (by_region_south, by_region_nonsouth) are labeled under robustness/heterogeneity.md but actually run the same regression on geographic subsamples without interaction terms. These are functionally identical to sample restrictions and are classified as core_sample.

### Demographic subsample splits (core)
Specs restricting to male_only, female_only, college_only, noncollege_only, black_only, nonblack_only, young, old, etc. run the same Mincer equation on subsamples. While these can be viewed as heterogeneity analyses, they test the same directional claim (experience raises wages) in each subgroup using the identical regression specification. They are classified as core_sample.

### Functional form variations
Several specs change the functional form in ways that alter the coefficient scale:
- e_log: Coefficient is an elasticity (~0.22) not semi-elasticity (~0.05)
- y_level: Coefficient is in dollar terms (~0.35) not log points
- y_standardized: Coefficient is in SD units (~0.066)
These still test the same directional hypothesis but with lower confidence in direct comparability.

---

## Recommendations for Spec-Search Script

1. **Separate heterogeneity interactions from subsample splits**: The by_region_south and by_region_nonsouth specs are labeled under heterogeneity.md but are subsample splits. The spec_tree_path should distinguish between interaction-based heterogeneity and subsample-based heterogeneity.

2. **Flag duplicate specifications**: robust/weights/unweighted appears to duplicate robust/control/drop_college (same coefficient, SE, and R-squared). The script should detect and flag such duplicates.

3. **Note scale incomparability for functional form changes**: When the treatment or outcome is transformed (log_e, w, lw_std), the coefficient changes scale. The script could flag these as "directional only" comparisons.

4. **Clarify baseline definition**: The baseline uses no FE and minimal controls (e, e2, s), while many robustness specs use year FE and demographic controls. It may be useful to define an "expanded baseline" for comparison with the FE/controls-augmented specifications.
