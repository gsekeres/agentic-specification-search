# Verification Report: 140921-V1

## Paper Information
- **Title**: Assortative Matching at the Top of the Distribution: Evidence from the World's Most Exclusive Marriage Market
- **Authors**: Marc Goni
- **Journal**: American Economic Review
- **Total Specifications**: 66

## Baseline Groups

### G1: Season interruption (syntheticT) increases probability of marrying commoner (cOu...
- **Expected sign**: +
- **Baseline spec(s)**: baseline
- **Outcome**: cOut
- **Treatment**: syntheticT
- **Notes**: Coef=0.0043, p=0.052. Borderline significant. N varies by specification.

### G2: Season interruption reduces probability of marrying an heir
- **Expected sign**: -
- **Baseline spec(s)**: baseline_mheir
- **Outcome**: mheir
- **Treatment**: syntheticT
- **Notes**: Coef=-0.0034, p=0.057. Borderline significant.

### G3: Season interruption increases absolute mismatch in landholdings
- **Expected sign**: +
- **Baseline spec(s)**: baseline_fmissmatch
- **Outcome**: fmissmatch
- **Treatment**: syntheticT
- **Notes**: Coef=0.524, p=0.014. Significant.

### G4: Season interruption reduces signed mismatch (husband lower landholdings)
- **Expected sign**: -
- **Baseline spec(s)**: baseline_fmissmatch2
- **Outcome**: fmissmatch2
- **Treatment**: syntheticT
- **Notes**: Coef=-0.516, p=0.032. Significant.

### G5: Season interruption increases probability of marrying down
- **Expected sign**: +
- **Baseline spec(s)**: baseline_fdown
- **Outcome**: fdown
- **Treatment**: syntheticT
- **Notes**: Coef=0.009, p=0.004. Significant.

## Classification Summary

| Category | Count |
|----------|-------|
| Baselines | 5 |
| Core tests (non-baseline) | 43 |
| Non-core tests | 18 |
| **Total** | **66** |

### Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 20 |
| core_funcform | 4 |
| core_inference | 3 |
| core_method | 3 |
| core_sample | 18 |
| noncore_alt_outcome | 8 |
| noncore_heterogeneity | 6 |
| noncore_placebo | 4 |

## Global Notes

Historical study of the London Season (elite marriage market) disruption 1861-63. Primary outcome (married commoner) is borderline significant at 5%. Sorting outcomes (landholding mismatch, marrying down) are more clearly significant. 66 specs total. IV first-stage F=8.4 (weak instrument concern).
