# Verification Report: 140921-V1

## Paper
- **Title**: Assortative Matching at the Top of the Distribution: Evidence from the World Most Exclusive Marriage Market
- **Author**: Marc Goni
- **Journal**: AER

## Baseline Groups

### G1: Married a commoner (cOut)
- **Baseline spec_ids**: baseline
- **Claim**: The interruption of the London Season (1861-63) increased the probability that aristocratic daughters married a commoner.
- **Expected sign**: +
- **Baseline coefficient**: 0.0043, p=0.052

### G2: Married an heir (mheir)
- **Baseline spec_ids**: baseline_mheir
- **Claim**: The interruption decreased the probability of marrying an heir.
- **Expected sign**: -
- **Baseline coefficient**: -0.0034, p=0.057

### G3: Landholding mismatch (fmissmatch, fmissmatch2, fdown)
- **Baseline spec_ids**: baseline_fmissmatch, baseline_fmissmatch2, baseline_fdown
- **Claim**: The interruption increased mismatch in landholdings between spouses.
- **Expected signs**: fmissmatch +, fmissmatch2 -, fdown +
- **Baseline coefficients**: fmissmatch=0.524 (p=0.014), fmissmatch2=-0.516 (p=0.032), fdown=0.009 (p=0.004)

## Summary Counts

| Category | Count |
|----------|-------|
| Total specifications | 66 |
| Baselines | 5 |
| Core test specs | 50 |
| Non-core specs | 16 |
| Invalid | 0 |
| Unclear | 0 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 24 |
| core_sample | 18 |
| core_inference | 3 |
| core_funcform | 4 |
| core_method | 1 |
| noncore_placebo | 4 |
| noncore_alt_outcome | 4 |
| noncore_alt_treatment | 1 |
| noncore_heterogeneity | 6 |
| noncore_diagnostic | 1 |
| invalid | 0 |
| unclear | 0 |

## Top 5 Most Suspicious Rows

1. **robust/control/add_hengpee**: Identical coefficient and SE to baseline. The baseline already includes hengpee, so this is a redundant duplicate. Confidence lowered to 0.7.

2. **robust/sample/trimmed_1pct**: Identical coefficient and SE to baseline. Since syntheticT is constructed from marriage rates by age, 1 percent trimming has no effect. Confidence lowered to 0.7.

3. **iv/first_stage/reduced_form**: Identical to baseline OLS (same cOut on syntheticT with same controls). Labeled as reduced form in IV context, but this IS the baseline OLS since syntheticT is the instrument.

4. **robust/placebo/predetermined_biorder**: The placebo test on birth order is significant (coef=0.031, p=0.030). This is concerning for the identifying assumption since syntheticT should not predict predetermined characteristics.

5. **iv/method/ols**: Uses mourn (actual marriage during interruption) as treatment instead of syntheticT. The OLS coefficient is near zero (0.002, p=0.97), likely reflecting endogeneity. Classified as noncore_alt_treatment.

## Duplicate Detection

Several specifications are exact duplicates of others:
- robust/outcome/mheir = baseline_mheir (identical coefficients)
- robust/outcome/fmissmatch = baseline_fmissmatch (identical coefficients)
- robust/outcome/fmissmatch2 = baseline_fmissmatch2 (identical coefficients)
- robust/outcome/fdown = baseline_fdown (identical coefficients)
- robust/extended/cOut = robust/control/extended (identical coefficients)
- robust/control/add_hengpee = baseline (identical coefficients)
- robust/sample/trimmed_1pct = baseline (identical coefficients)
- iv/first_stage/reduced_form = baseline (identical coefficients)

These duplicates inflate the apparent number of distinct specifications from 66 to approximately 58 unique specifications.

## Recommendations

1. **De-duplicate specifications**: The spec search script should avoid recording the same regression twice under different labels.

2. **Flag identical results**: When a robustness check produces identical coefficient and SE to the baseline, the script should note this.

3. **Heterogeneity classification**: The heterogeneity specs report the main effect from a model with interactions. These are not clean tests of the baseline claim because the main effect changes interpretation when interactions are present.

4. **IV specification gap**: The spec search did not include a proper 2SLS/IV estimate (instrumenting mourn with syntheticT). Only the first stage and a naive OLS on mourn were included.

5. **Celibacy and distance outcomes**: These are distinct outcomes not part of the core claim about marital sorting. Correctly classified as noncore_alt_outcome.
