# Verification Report: 126722-V1

## Paper Information
- **Title**: Allocating Health Care Resources Efficiently (Lopez, Sautmann, Schaner 2020)
- **Journal**: AEJ-Applied
- **Method**: Cross-sectional OLS (Randomized Experiment)
- **Total Specifications**: 92

## Baseline Groups

### G1: Patient Voucher Effect on Malaria Treatment Purchase
- **Claim**: Patient vouchers increase the probability of purchasing malaria treatment.
- **Baseline spec**: `baseline`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.153 (p < 0.001)
- **Outcome**: `treat_sev_simple_mal`
- **Treatment**: `patient_voucher`

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baseline)** | **55** | |
| core_controls | 28 | Baseline + no controls + date FE + basic + full + leave-one-out (11) + incremental addition (11) + include doc info |
| core_sample | 24 | Age, gender, malaria risk, symptoms, language, education, ethnicity, illness duration, etc. |
| core_inference | 3 | Robust, clinic-day cluster, date cluster |
| **Non-core tests** | **37** | |
| noncore_alt_outcome | 14 | Prescribed treatment, severe malaria, voucher usage, expected match, match components |
| noncore_alt_treatment | 3 | Doctor voucher, any voucher, patient voucher only |
| noncore_placebo | 4 | Predicted malaria, age, days ill, symptoms |
| noncore_heterogeneity | 13 | Interactions with risk, age, gender, symptoms, info, education, days ill, ethnicity, French, literacy |
| **Total** | **92** | |

## Robustness Assessment

**STRONG** support. 100% of primary outcome specs show positive effects; 94.5% significant at 5%. Effect magnitude is stable (median 0.145). Patient vouchers consistently outperform doctor vouchers. Placebo tests pass. Subgroup effects are consistently positive.
