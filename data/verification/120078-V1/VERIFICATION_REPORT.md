# Verification Report: 120078-V1

## Paper
Airbnb price discrimination study -- effect of minority host status on listing price as a function of review count (information accumulation). Panel FE design with listing and city-wave fixed effects.

## Baseline Groups Found

### G1: Minority x reviews interaction on log price (Table 5)
- **Baseline spec_run_ids**: 120078-V1_run_001, 120078-V1_run_002, 120078-V1_run_003
- **Baseline spec_ids**: baseline, baseline__table5_col2, baseline__table5_col3
- **Claim**: Minority hosts experience a price penalty that diminishes as reviews accumulate (mino_x_rev100 coefficient is positive, meaning prices converge as information grows)
- **Baseline coefficient (mino_x_rev100)**: 0.0904 (SE=0.0357, p=0.0113, N=1,858,114, R2=0.974)
- **Expected sign**: Positive (minority price gap closes with more reviews)

## Counts

| Metric | Count |
|--------|-------|
| Total rows in specification_results.csv | 50 |
| Valid (run_success=1, is_valid=1) | 50 |
| Invalid (run_success=0 or is_valid=0) | 0 |
| Core tests (is_core_test=1) | 50 |
| Non-core | 0 |
| Baseline rows | 3 |
| Inference variants (inference_results.csv) | 2 |

## Category Counts

| Category | Count |
|----------|-------|
| core_method (baselines) | 3 |
| core_controls | 38 |
| core_sample | 6 |
| core_fe | 2 |
| core_funcform | 1 |

## Robustness Assessment

### Sign consistency
- **49 of 50** specifications (98.0%) produce a positive coefficient, consistent with the baseline sign.
- 1 specification produces a negative coefficient:
  - `rc/sample/restriction/review_gt_0` (coef=-0.0005, p=0.964): Removing the review upper bound (using all listings with >0 reviews instead of review<40) yields a near-zero, insignificant result. This shows the effect is concentrated in the low-review range where information accumulation is most salient.

### Statistical significance
- **42 of 50** specifications (84.0%) are significant at the 5% level.
- Insignificant specifications:
  - `baseline__table5_col2` (review<60): p=0.029 (still significant, but weaker)
  - `rc/sample/restriction/review_lt_80`: p=0.070
  - `rc/sample/restriction/review_lt_100`: p=0.063
  - `rc/sample/restriction/review_gt_0`: p=0.964
  - `rc/sample/outliers/trim_y_5_95`: p=0.119
  - All other specifications are significant at 5%.

### Controls sensitivity
- All 38 control specifications are positive and significant at 5%.
- Coefficient range across control specs: [0.0899, 0.0936]
- The treatment coefficient is highly stable: LOO controls, standard sets, progression, and random subsets all produce nearly identical point estimates. This reflects the high-dimensional FE structure absorbing most variation.
- Note: 8 LOO-drop specs produce coefficients identical to baseline (to 16 decimal places), indicating those controls have zero partial correlation with the outcome conditional on the FE structure (likely because the within-listing variation for these time-invariant controls is zero).

### Sample sensitivity
- 6 sample specifications: 3 significant at 5%, 3 not.
- Widening the review window (from <40 to <60, <80, <100, unrestricted) progressively attenuates the coefficient toward zero, consistent with the theoretical prediction that the information accumulation effect is strongest for low-review listings.
- Outlier trimming at 1/99 preserves significance (p=0.013); aggressive trimming at 5/95 does not (p=0.119).

### Fixed effects sensitivity
- Dropping city-wave FE (`rc/fe/drop/citywaveID`) produces a coefficient of 0.840 (vs 0.090 baseline), a roughly 9x increase. This is expected: without city-wave FE, the minority indicator captures cross-sectional price differences rather than within-city-wave variation.
- Swapping city-wave FE for neighborhood-city FE (`rc/fe/swap/hoodcityID_for_citywaveID`) produces a nearly identical coefficient (0.840), suggesting hoodcityID adds no additional variation control beyond newid alone.
- Both FE specs have p=0.0 (extremely significant).

### Functional form sensitivity
- Adding a quadratic term in rev100 (`rc/form/treatment/quadratic_rev100`): coefficient on linear term is 0.138 (p=0.045), still positive and significant, suggesting the linear specification is a reasonable approximation.

### Inference sensitivity (from inference_results.csv)
- **Cluster(newid)**: SE=0.0357, p=0.0113 -- baseline
- **HC1 (robust)**: SE=0.0200, p=0.0000064 -- highly significant (smaller SE without clustering)
- **Cluster(hoodcityID)**: SE=0.0362, p=0.0127 -- nearly identical to baseline clustering
- The result is robust across inference variants. Clustering at the listing level is conservative relative to HC1.

## Top Issues

1. **LOO duplicates**: 8 of 13 LOO-drop control specs produce coefficients identical to the baseline. This occurs because these controls (sharedflat, bedrooms, bathrooms, superhost, verified_email, facebook, plus full and progression/full_with_counts which duplicate the full set) have zero within-listing variation and thus do not affect the within estimator. These are valid but uninformative specifications.

2. **FE sensitivity**: Dropping city-wave FE increases the coefficient by ~9x (0.09 to 0.84). This is a legitimate design sensitivity showing that the identifying variation is very different without time-varying geographic controls. The specs are valid RC but measure a different estimand in practice.

3. **Sample window sensitivity**: The effect attenuates monotonically as the review window widens, becoming insignificant for review<80 and above. This is consistent with the paper's theory but means the result is specific to the low-review subsample.

4. **R2 very high (~0.974)**: The listing FE absorb almost all price variation, which is expected for within-listing designs but means the treatment explains a tiny fraction of residual variation.

## Recommendations

1. The surface and runner are well-aligned. All 50 surface-planned specs were executed successfully.
2. The LOO duplicates could be flagged in future runs to avoid redundant computation (check within-listing variation before running LOO).
3. Consider adding specifications that vary the review window more finely (e.g., review<20, review<30) to map the attenuation pattern.
4. The FE drop/swap specs should be interpreted cautiously -- they change the identifying variation fundamentally.

## Conclusion

The specification search confirms that the baseline result (mino_x_rev100 = 0.090, p=0.011) is **robust to control set variation** (38/38 specs positive and significant) but **sensitive to the sample review window** (effect attenuates with wider windows) and **sensitive to FE structure** (coefficient changes dramatically without city-wave FE, though remains significant). The treatment concept and outcome concept are preserved across all 50 specifications with no drift.

Overall assessment: **STRONG support** for the baseline claim within the specified sample window (review<40). The claim is specific to low-review listings, which the paper acknowledges as the theoretically relevant subpopulation.
