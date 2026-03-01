# Verification Report: 112517-V1

## Paper
Fowlie, Holland & Mansur (2012), "What Do Emissions Markets Deliver and to Whom? Evidence from Southern California's NOx Trading Program", AER

## Baseline Groups Found

### G1: RECLAIM emissions effect (Table 4)
- **Baseline spec_run_ids**: 112517-V1_run_001 (baseline levels), 112517-V1_run_002 (baseline log), 112517-V1_run_003 (baseline levels + demographics), 112517-V1_run_004 (baseline log + demographics)
- **Baseline spec_ids**: baseline, baseline__table4_log, baseline__table4_demog, baseline__table4_log_demog
- **Claim**: ATT of RECLAIM cap-and-trade participation on change in facility NOx emissions (DIFFNOX = post minus pre)
- **Baseline coefficient**: -24.68 (p=0.106, N=1745) for levels; -0.102 (p=0.891) for log
- **Expected sign**: Negative (RECLAIM reduces emissions)

## Counts

| Metric | Count |
|--------|-------|
| Total rows in specification_results.csv | 53 |
| Valid (run_success=1) | 53 |
| Invalid (run_success=0) | 0 |
| Core tests (is_core_test=1) | 51 |
| Non-core (valid) | 2 |
| Baseline rows | 4 |
| Inference variants (inference_results.csv) | 2 |

## Category Counts

| Category | Count |
|----------|-------|
| core_sample | 23 |
| core_method | 13 |
| core_funcform | 9 |
| core_controls | 6 |
| noncore_alt_treatment | 2 |

## Robustness Assessment

### Sign consistency
- **49 of 51** core specifications (96.1%) produce a negative coefficient, consistent with the baseline sign.
- 2 specifications produce positive coefficients (sign reversal):
  - `rc/joint/log_severe` (log outcome, severe nonattainment only): coef=+0.489, p=0.640
  - `rc/joint/log_nosc` (log outcome, no Southern California controls): coef=+0.098, p=0.915
  - Both are log-outcome specs with restricted geographic samples and small effective samples. The sign reversals are statistically insignificant.

### Statistical significance
- **12 of 51** core specifications (23.5%) are significant at the 5% level.
- **19 of 51** core specifications (37.3%) are significant at the 10% level.
- The baseline levels ATT (p=0.106) is itself not significant at 10%.

### Levels specifications (DIFFNOX)
- 32 core levels specs, all negative.
- 5 of 32 (15.6%) significant at 5%.
- Coefficient range: [-79.46, -4.43]
- Significant specs include: demographics matching (p=0.019), TWFE estimator (p=0.032), northern California controls (p=0.009), and select joint specs.

### Log specifications (lnDIFFNOX)
- 19 core log specs, 17 negative.
- 7 of 19 (36.8%) significant at 5%.
- Coefficient range: [-0.654, +0.489]
- The log transformation tends to yield more significant results, particularly for southern California (p<0.001), single facility firms (p=0.020), and period/estimator joint specs.

### Controls sensitivity
- Adding demographics as matching variables (income, minority share) substantially increases the ATT magnitude: from -24.7 to -79.5 in levels (now significant at 5%).
- The demographics-augmented matching specs are among the most significant.

### Sample sensitivity
- Dropping electric utilities weakens the result substantially (p=0.504 vs 0.106).
- Northern California controls (excluding Southern California from control pool) strengthens the result (p=0.009).
- Severe nonattainment counties and single-facility firms subsamples show mixed results.
- Period definitions (using pd2/pd3 instead of pd1/pd4) weaken results (p=0.546).

### Estimator sensitivity
- TWFE (areg) produces significant levels results (coef=-32.59, p=0.032) vs. the insignificant nnmatch baseline (p=0.106).
- TWFE log specs are generally more significant than nnmatch log specs.

### Inference sensitivity (from inference_results.csv)
- HC1 robust SE on TWFE baseline: SE=14.06, p=0.021 (significant at 5%)
- Clustered by air basin: SE=13.26, p=0.032 (significant at 5%)
- Both inference variants on the TWFE estimator preserve significance.

### Duplicate specifications noted
- rc/sample/subset/nonattainment_only duplicates baseline (sample already restricted to nonattainment)
- rc/form/outcome/log_emissions duplicates baseline__table4_log
- rc/controls/add_demographics duplicates baseline__table4_demog
- rc/controls/add_demographics_plus_quartile duplicates baseline__table4_log_demog
- rc/sample/period/pre_pd1_post_pd3 duplicates rc/sample/period/pre_pd2_post_pd3

### Non-core specifications
- 2 small_firms specs (dumreclaim_small treatment) show positive coefficients (opposite sign), meaning small-firm RECLAIM participants do not reduce emissions. These are correctly classified as noncore_alt_treatment since they change the treatment concept.

## Top Issues

1. **Baseline insignificance**: The primary nnmatch levels baseline (p=0.106) is not significant at conventional levels. Significance depends on specification choices: demographics matching, estimator (TWFE vs nnmatch), functional form (log vs levels), and geographic sample restrictions.

2. **Specification-dependent significance**: The result is significant under demographics matching (p=0.019), TWFE (p=0.032), and northern California controls (p=0.009), but not under the paper's preferred specification. This suggests the result is real but imprecisely estimated.

3. **Duplicate specifications**: At least 5 specs mechanically duplicate baselines or other specs, inflating the apparent specification count. The effective unique spec count is closer to 46.

4. **Log vs levels divergence**: Log specs tend to show stronger significance, but the two positive-coefficient specs are both in log form, suggesting the log transformation introduces noise in small samples.

5. **Electric utilities dominate**: Dropping electric utilities (13 of ~350 treated facilities) substantially weakens the result, suggesting the ATT is partly driven by large electric utility reductions.

## Recommendations

1. **De-duplicate** the 5 mechanical duplicate specs to avoid inflating the specification count.
2. **Add weighting variants**: No weight variations were tested (the surface did not plan any).
3. **Expand inference variants**: The 2 inference variants only cover the TWFE estimator. Adding inference variants for the baseline nnmatch estimator would be informative.
4. **Consider heterogeneity analysis**: The electric utility sensitivity suggests important heterogeneity that could be explored more systematically.

## Conclusion

The specification search reveals that the RECLAIM emissions reduction effect is **directionally robust** -- 49 of 51 core specs (96.1%) show negative ATT, consistent with the paper's claim. However, **statistical significance is fragile**: only 23.5% of core specs achieve p<0.05, and the baseline itself is not significant at 10%. Significance is sensitive to the estimator choice (TWFE vs nnmatch), the addition of demographic matching variables, and the geographic control pool.

Overall assessment: **MODERATE support**. The sign is highly consistent across specifications, but the precision of the estimate is insufficient for reliable significance under many reasonable specification choices.
