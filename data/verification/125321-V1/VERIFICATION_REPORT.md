# Verification Report: 125321-V1

## Paper
Chinese air pollution monitoring automation study -- sharp RD design estimating the effect of switching from manual to automated pollution monitoring on reported PM10 concentrations.

## Baseline Groups Found

### G1: RD effect of automation on reported PM10 (Table 1A)
- **Baseline spec_run_ids**: 125321-V1_run_001, 125321-V1_run_002
- **Baseline spec_ids**: baseline, baseline__raw_pm10
- **Claim**: Automation of monitoring stations leads to a discrete increase in reported PM10 levels at the automation date, implying pre-automation manual reporting underreported pollution.
- **Baseline coefficient (RD estimate)**: 34.92 (SE=6.53, p=1.81e-08, N=231,511)
- **Raw PM10 baseline**: 34.69 (SE=11.33, p=0.008, N=90,578)
- **Expected sign**: Positive (automation reveals previously underreported pollution)

## Counts

| Metric | Count |
|--------|-------|
| Total rows in specification_results.csv | 23 |
| Valid (run_success=1, is_valid=1) | 23 |
| Invalid (run_success=0 or is_valid=0) | 0 |
| Core tests (is_core_test=1) | 23 |
| Non-core | 0 |
| Baseline rows | 2 |
| Inference variants (inference_results.csv) | 2 |
| Diagnostic rows (diagnostics_results.csv) | 6 |

## Category Counts

| Category | Count |
|----------|-------|
| core_method (baselines + design) | 9 |
| core_controls | 2 |
| core_sample | 9 |
| core_funcform | 1 |
| core_fe | 1 |
| core_data | 1 |

## Robustness Assessment

### Sign consistency
- **23 of 23** specifications (100%) produce a positive coefficient, consistent with the baseline sign.
- No negative coefficients observed. The automation effect is uniformly positive across all specifications.

### Statistical significance
- **22 of 23** specifications (95.7%) are significant at the 5% level.
- **21 of 23** specifications (91.3%) are significant at the 1% level.
- 1 specification is not significant at 5%:
  - `rc/fe/alt/station_yearmonth_fe` (coef=6.34, p=0.441): Residualizing with station + year-month FE (instead of station + month FE) absorbs within-year variation that drives the RD effect, dramatically attenuating the coefficient. This is an important sensitivity finding.

### Design sensitivity (RD-specific)
- **Bandwidth**: Half baseline BW (coef=32.25, p=0.0008) and double BW (coef=27.71, p<0.001) both preserve significance. Point estimates decrease with wider bandwidths, typical of RD.
- **Polynomial order**: Local quadratic (coef=36.16, p<0.001) is similar to local linear baseline.
- **Kernel**: Uniform (coef=35.84, p<0.001) and Epanechnikov (coef=35.99, p<0.001) kernels produce nearly identical results to the triangular baseline.
- **Procedure**: Conventional (coef=34.92, SE=5.81, p<0.001) has smaller SE than robust bias-corrected (SE=6.53); both significant. The baseline uses robust bias-corrected, which is conservative.

### Controls sensitivity
- No residualization (raw PM10, coef=34.69, p=0.008): Still significant but with much larger SE (11.33 vs 6.53), reflecting the value of residualizing out station/month/weather effects.
- Weather-only residualization (coef=32.22, p=0.0009): Intermediate SE, still significant.

### Sample sensitivity
- **Wave subsamples**: Wave 1 only (coef=27.51, p=0.020) and Wave 2 only (coef=64.51, p<0.001) are both significant. Wave 2 has a much larger effect, consistent with heterogeneity across automation phases.
- **Deadline stations**: coef=57.13, p<0.001. Stations automated at regulatory deadlines show an even larger effect.
- **76 cities subset**: coef=33.84, p=0.003. Robust.
- **No missing PM10**: Identical to baseline (coef=34.92), confirming the sample already excludes missing PM10 obs.
- **Outlier trimming**: 1-99 percentile (coef=20.18, p<0.001) and 5-95 percentile (coef=9.96, p=0.0003) both significant but attenuated, showing some sensitivity to extreme PM10 values.
- **Donut holes**: Excluding 1 day (coef=33.49, p<0.001) and 3 days (coef=38.91, p<0.001) around cutoff both preserve the result, ruling out manipulation-day transients.

### Functional form sensitivity
- Log PM10 outcome (coef=0.272, p<0.001): Significant. On the log scale, automation increases PM10 by ~27%, which is consistent with the level estimate relative to mean PM10.

### Fixed effects sensitivity
- Station + year-month FE (coef=6.34, p=0.441): NOT significant. This is the most important vulnerability. Year-month FE absorb temporal variation that overlaps with the automation timing, severely attenuating the effect. This suggests the RD effect may partly reflect broader temporal trends coinciding with automation dates.

### Inference sensitivity (from inference_results.csv)
- **Cluster(city)**: SE=6.53, p=1.81e-08 -- baseline (robust bias-corrected)
- **Cluster(station)**: SE=3.16, p<0.001 -- much smaller SE with finer clustering
- **HC1 (no clustering)**: SE=1.18, p<0.001 -- smallest SE
- The result is highly robust to inference variants. City-level clustering is the most conservative.

### Data construction sensitivity
- Monthly aggregation (coef=38.10, p<0.001): Aggregating to monthly level preserves significance with slightly larger point estimate but much fewer observations (N=8,389).

## Top Issues

1. **Year-month FE vulnerability**: The `rc/fe/alt/station_yearmonth_fe` specification reduces the coefficient from 34.9 to 6.3 (p=0.441). This is the only specification that is not significant and suggests potential confounding from temporal trends coinciding with the automation rollout.

2. **Duplicate specifications**: `rc/sample/restrict/no_missing_pm10` produces results identical to baseline (same coefficient, SE, p, N), indicating the baseline sample already excludes missing PM10. Similarly, `design/regression_discontinuity/procedure/robust_bias_corrected` is identical to baseline because the baseline already uses robust bias-corrected inference. These are valid but redundant.

3. **Wave heterogeneity**: Wave 2 stations show a much larger effect (64.5 vs 27.5 for Wave 1), suggesting substantial heterogeneity. The pooled baseline estimate (34.9) may mask important variation.

4. **Outlier sensitivity**: Aggressive trimming (5-95 percentile) reduces the coefficient by ~71% (from 34.9 to 10.0). High-pollution days drive a substantial portion of the estimated effect.

5. **R2 not reported**: R2 is blank for all rows, which is standard for RD designs using rdrobust (the package does not report a global R2). This is not an error.

## Recommendations

1. The year-month FE result warrants discussion -- if temporal trends confound the RD, the identifying assumption is weakened.
2. Remove the two duplicate specifications (no_missing_pm10, robust_bias_corrected) or flag them as mechanical duplicates.
3. Consider adding placebo cutoff tests at non-automation dates to strengthen the design validity.
4. The diagnostics (manipulation test, balance checks, AOD placebo) are properly separated in diagnostics_results.csv.

## Conclusion

The specification search confirms that the baseline RD result (automation increases reported PM10 by ~35 units, p<0.001) is **robust to most perturbations**: bandwidth, polynomial, kernel, controls, most sample restrictions, and inference variants all preserve a large, significant positive effect. The main vulnerability is the year-month FE specification, which absorbs the effect entirely, raising questions about temporal confounding. Outlier trimming also attenuates the effect substantially.

Overall assessment: **STRONG support** for the baseline claim, with one notable caveat regarding year-month FE absorption. 22 of 23 specifications are significant at 5%, and all 23 are positive.
