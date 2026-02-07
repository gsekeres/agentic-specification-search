# Verification Report: 206781-V1

## Paper
"Who Should Get Social Insurance? A Machine Learning Approach to Cash Transfer Targeting" (AER)

## Baseline Groups

### G1: Cash transfers increase household consumption
- **Baseline spec_id**: baseline/el_cons_T_hh
- **Outcome**: el_cons_T_hh (time-demeaned household consumption)
- **Treatment**: treat (randomized cash transfer)
- **Coefficient**: 285.88, p < 0.001
- **Expected sign**: positive

### G2: Cash transfers increase household assets
- **Baseline spec_id**: baseline/el_assets_T_hh
- **Outcome**: el_assets_T_hh (time-demeaned household assets)
- **Treatment**: treat
- **Coefficient**: 189.44, p < 1e-11
- **Expected sign**: positive

### G3: Cash transfers increase household income
- **Baseline spec_id**: baseline/el_income_T_hh
- **Outcome**: el_income_T_hh (time-demeaned household income)
- **Treatment**: treat
- **Coefficient**: 75.27, p = 0.086
- **Expected sign**: positive
- **Note**: Not significant at 5%; weakest baseline claim.

## Counts

| Category | Count |
|----------|-------|
| Total specifications | 71 |
| Baseline specs | 3 |
| Core test specs (incl. baselines) | 50 |
| Non-core specs | 21 |
| Invalid specs | 0 |
| Unclear specs | 0 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 28 |
| core_inference | 6 |
| core_sample | 5 |
| core_funcform | 11 |
| noncore_heterogeneity | 18 |
| noncore_placebo | 3 |
| noncore_alt_outcome | 0 |
| noncore_alt_treatment | 0 |
| noncore_diagnostic | 0 |
| invalid | 0 |
| unclear | 0 |

Note: The 3 baselines are counted within core_controls. 50 core tests total = 28 core_controls + 6 core_inference + 5 core_sample + 11 core_funcform.

## Top 5 Most Suspicious Rows

1. **ols/controls/full** (spec_id): This is identical to the baseline/el_cons_T_hh specification (same controls, same outcome, same clustering). It produces the exact same coefficient (285.88). Redundant but not invalid.

2. **robust/cluster/village_code**: Identical to baseline (same clustering at village_code). Same coefficient (285.88). Purely redundant.

3. **robust/het/interaction_* specs**: All six interaction specs report the main treatment coefficient (coefficient on "treat"), which is the treatment effect for the *omitted category* (e.g., small households when interacting with large_hh). This is not the average treatment effect and is not directly comparable to the baseline. Classified as noncore_heterogeneity.

4. **robust/sample/small_hh and robust/sample/large_hh**: These are labeled as "sample restrictions" in the spec search but are actually subgroup heterogeneity analyses. A sample restriction would be trimming outliers; splitting by household size is heterogeneity. Same applies to owns_land/no_land and widow/not_widow splits. Classified as noncore_heterogeneity.

5. **robust/placebo/balance_bl_hhsize**: Shows p=0.029 for treatment predicting baseline household size, which could indicate a balance concern in the RCT. However, this is a diagnostic/balance check, not a test of the main claim.

## Assessment of Baseline Claims

- **G1 (Consumption)**: Well-identified. The baseline is clear and most robustness checks target this outcome. 34 total core tests (including baseline) with highly stable results across control variations (coefficients range 266-317). Functional form variations (log, IHS) confirm positive effect. Trimming reduces magnitude as expected.

- **G2 (Assets)**: Adequate coverage with 8 core tests. No-controls, trimmed sample, unweighted, non-demeaned, per-capita, and log versions all confirm strong positive effect.

- **G3 (Income)**: Adequate coverage with 8 core tests, but the baseline itself is not significant at 5% (p=0.086). Several robustness checks also fail significance (no-controls p=0.054, trimmed p=0.154, unweighted p=0.202, log p=0.152). This claim is genuinely fragile.

## Recommendations for the Spec-Search Script

1. **Remove redundant specs**: ols/controls/full duplicates the baseline exactly. robust/cluster/village_code duplicates the baseline exactly. These inflate the spec count without adding information.

2. **Separate heterogeneity from sample restrictions**: The sample splits by household characteristics (small_hh/large_hh, owns_land/no_land, widow/not_widow) should be labeled as heterogeneity, not sample restrictions. True sample restrictions are trimming, dropping outliers, or restricting to specific survey waves.

3. **Report interaction coefficients properly**: The interaction specs report the main "treat" coefficient, which is the treatment effect for the omitted subgroup. To be useful, they should also report the interaction term coefficient and/or the marginal effect for the included subgroup.

4. **Add more robustness for G2 and G3**: The spec search is heavily tilted toward G1 (consumption). Consider adding control progression and LOO analyses for assets and income outcomes too.

5. **Consider adding FE specifications**: The paper does not use fixed effects in the baseline, but village or sublocation FE could be informative robustness checks (instead of just clustering at those levels).
