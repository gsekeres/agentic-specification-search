# Verification Report: 181166-V1

## Baseline Groups
- **G1**: Technological change and displaced worker earnings
  - Baseline spec_run_ids: ['181166-V1_run_001', '181166-V1_run_002', '181166-V1_run_003']
  - Baseline spec_ids: ['baseline', 'baseline__table3_col1_soc4', 'baseline__table3_col3_ft']
  - Expected sign: positive (more tech change in occupation -> better post-displacement earnings recovery)

## Counts
- **Total rows**: 43
- **Core**: 43
- **Non-core**: 0
- **Invalid**: 0
- **Baselines**: 3

## Category Breakdown
| Category | Count |
|----------|-------|
| core_controls | 27 |
| core_sample | 8 |
| core_method | 3 |
| core_fe | 2 |
| core_data | 2 |
| core_weights | 1 |

## Sign and Significance
- Positive coefficient: 39/43 (91%)
- Negative coefficient: 4/43 (9%)
- Significant at 5%: 0/43 (0%)
- Coefficient range: [-0.0738, 0.1748]
- Baseline: coef=0.015, p=0.756, N=220

## Assessment
- **NOT ROBUST / NON-SIGNIFICANT**: The baseline result is not statistically significant (p=0.756), and no specification in the entire 43-spec run achieves significance at the 5% level.
- The direction is mostly positive (91% of specs), but the effect size is small and standard errors are large relative to the point estimate across all specifications.
- LOO analysis: no single control changes the conclusion. All LOO specs remain non-significant with p-values ranging from 0.48 to 0.88.
- Control set variations: none significant (p=0.29 to 0.76).
- Random control subsets: none significant (p=0.22 to 0.88).
- Sample restrictions show mixed signs: age_45_65 (coef=-0.074, p=0.18), college (coef=-0.005, p=0.95), male_only (coef=0.087, p=0.56). Small subsamples reduce N substantially (N=56-139).
- Winsorization at 1/99: flips sign to negative (coef=-0.006, p=0.92).
- Unweighted: flips sign to negative (coef=-0.004, p=0.91).
- This pattern is consistent with a null result where the point estimate is noise around zero.

## Issues
1. **Missing AD occupation code specs**: The surface planned baseline__table3_col4_AD, baseline__table3_col5_AD_ft, and rc/data/treatment/AD_occ_codes, but these were not implemented. This is likely due to data construction complexity (Autor-Dorn crosswalks).
2. **Small sample**: N=220 for the main sample is quite small, which explains the wide standard errors and lack of power.
3. **Sign instability**: 4 of 43 specs show negative coefficients (age 45-65 subsample, college subsample, winsor 1/99, unweighted), indicating the direction of the effect is not stable.
4. **Non-significant baseline**: The paper's own result may have been significant with different standard errors or in Stata (the original uses reghdfe with SOC-4 clustering). The pyfixest replication produces p=0.756, suggesting either a data processing difference or the original result relied on different clustering/weighting choices.

## Recommendations
- Verify the data construction matches the original Stata code exactly (sample restrictions, winsorization, normalization).
- Check whether the original paper's Table 3 Col 2 result is statistically significant. If so, the discrepancy between the original and the replication needs investigation.
- Implement the Autor-Dorn occupation code variants to complete the surface plan.
- Consider whether the SOC-4 clustering is correctly implemented (number of clusters matters for inference with few clusters).
