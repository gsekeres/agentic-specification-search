# Verification Report: 112474-V1

**Paper**: Dinkelman (2011), "The Effects of Rural Electrification on Employment: New Evidence from South Africa"
**Design**: Instrumental Variables (2SLS)
**Date verified**: 2026-02-24

## Baseline Groups

### G1: Female Employment (Headline Result)
- **Claim**: Electrification increases female employment rate (LATE via land gradient IV)
- **Expected sign**: positive
- **Baseline spec_run_ids**: 112474-V1_run001, 112474-V1_run002, 112474-V1_run003
- **Baseline spec_ids**: baseline, baseline__table4_col8_female, baseline__table4_col7_female
- **Baseline coefficient**: 0.095 (p = 0.087)

### G2: Male Employment (Null Result)
- **Claim**: Electrification has no significant effect on male employment rate
- **Expected sign**: null (no directional prediction)
- **Baseline spec_run_ids**: 112474-V1_run043, 112474-V1_run044, 112474-V1_run045
- **Baseline spec_ids**: baseline, baseline__table4_col8_male, baseline__table4_col7_male
- **Baseline coefficient**: 0.035 (p = 0.591)

## Counts

| Metric | Value |
|--------|-------|
| Total rows | 80 |
| Core | 80 |
| Non-core | 0 |
| Invalid | 0 |
| Baselines | 6 (3 per group) |
| Run success rate | 100% (80/80) |

## Category Breakdown

| Category | G1 | G2 | Total |
|----------|----|----|-------|
| core_method | 5 | 5 | 10 |
| core_controls | 26 | 26 | 52 |
| core_sample | 6 | 6 | 12 |
| core_fe | 2 | 2 | 4 |
| core_funcform | 1 | 1 | 2 |
| **Total** | **40** | **40** | **80** |

## Sign and Significance

### G1: Female Employment (40 specs)
- Positive coefficient: 39/40 (97.5%)
- Only negative: rc/estimation/ols_reduced_form (coef = -0.0007, reduced form on instrument, expected to be negative since higher gradient = less electrification)
- Significant at p < 0.10: 15/40 (37.5%)
- Significant at p < 0.05: 3/40 (7.5%)
- Coefficient range (excluding reduced form): [0.025, 0.130]
- Median coefficient: ~0.087

### G2: Male Employment (40 specs)
- Positive coefficient: 32/40 (80%)
- Significant at p < 0.10: 0/40 (0%)
- Coefficient range: [-0.072, 0.079]
- Median coefficient: ~0.030

## Inference Variants (from inference_results.csv)

| Variant | G1 SE | G1 p | G2 SE | G2 p |
|---------|-------|------|-------|------|
| CRV1(placecode0) [canonical] | 0.0553 | 0.087 | 0.0659 | 0.591 |
| HC1 (robust) | 0.0510 | 0.063 | 0.0625 | 0.571 |
| CRV1(dccode0) | 0.0532 | 0.107 | 0.0852 | 0.687 |

## Issues Identified

### Minor Issues

1. **Nonstandard `estimation` top-level key in CVJ** (run040, run082): The `rc/estimation/ols_reduced_form` specs include an `estimation` block as a top-level key in `coefficient_vector_json`. Per schema conventions, this should live under `extra` or `design`. This is a minor schema deviation with no impact on results.

2. **`treatment_var` metadata for reduced form** (run040, run082): These specs report `treatment_var=T` in the CSV, but the focal coefficient is on `mean_grad_new` (the instrument). The coefficient in the scalar field correctly reflects the instrument coefficient. This is a metadata labeling inconsistency, not a validity issue.

3. **Missing run numbers 41-42**: The spec_run_id sequence skips from run040 to run043. Likely reserved for diagnostics or inference computations handled separately. Not a data integrity issue.

4. **R-squared missing for IV specs**: Only the two OLS reduced-form specs report R-squared. This is expected behavior for IV models.

### No Major Issues

- No duplicate spec_run_ids
- No infer/* rows in specification_results.csv (properly in inference_results.csv)
- All CVJ audit keys present (coefficients, inference, software, surface_hash)
- All inference spec_ids match canonical (infer/se/cluster/placecode0)
- All numeric fields finite for run_success=1 rows
- No outcome/treatment drift within baseline groups (except asinh transform and reduced form, both correctly handled)
- No population changes within groups (all largeareas==1 except the full_sample restriction, which is a valid core_sample variant)

## Assessment

### G1 (Female Employment): MODERATE Robustness
The positive effect of electrification on female employment is consistent in sign across 39/40 specifications (97.5%), but statistical significance is sensitive to specification choices. The baseline p-value of 0.087 is marginal, and only 37.5% of specs achieve p < 0.10. The result is strongest with the full sample (no large-areas restriction, p = 0.026) and with heavy outlier trimming (p = 0.039). Control variation (LOO, progression, random subsets) preserves the positive sign but often pushes the p-value above 0.10. The weak first-stage F-statistic (~8.3) compounds uncertainty. Overall, the sign is robust but significance is fragile.

### G2 (Male Employment): STRONG Null Robustness
The null result for male employment is very robust. No specification out of 40 produces a significant effect at any conventional level. Coefficients are centered near zero with moderate sign variation (80% positive), consistent with a true null.

## Recommendations

1. **Schema fix**: Move the `estimation` top-level key to `extra` in the CVJ for reduced-form specs.
2. **Metadata fix**: For reduced-form specs, consider reporting `treatment_var=mean_grad_new` to match the actual focal coefficient, or add a field distinguishing the focal variable from the conceptual treatment.
3. **Diagnostics**: The first-stage F-statistic (~8.3) is below the Staiger-Stock threshold of 10. Consider adding Anderson-Rubin confidence intervals or tF diagnostics for weak-instrument-robust inference.
4. **Additional specs**: Consider adding province-level clustering or Conley spatial SEs as inference variants, given the geographic nature of the treatment.
