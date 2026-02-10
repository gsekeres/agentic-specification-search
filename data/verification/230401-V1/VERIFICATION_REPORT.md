# Verification Report: 230401-V1

## Paper Information
- **Paper ID**: 230401-V1
- **Total Specifications**: 60
- **Baseline Specifications**: 4
- **Core Test Specifications**: 36
- **Non-Core Specifications**: 24

## Baseline Groups

### G1: Black-owned businesses pay higher interest rates than White-owned businesses on 
- **Expected sign**: +
- **Baseline spec(s)**: `baseline__black_50`
- **Outcome**: `loanrate_w2`
- **Treatment**: `black_50`
- **Notes**: Coefficient: 2.90 pp (SE~0.94, p=0.002). State and time FE, state-clustered SE. N=1,366.

### G2: Hispanic-owned businesses pay higher interest rates than White-owned businesses.
- **Expected sign**: +
- **Baseline spec(s)**: `baseline__hisp_50`
- **Outcome**: `loanrate_w2`
- **Treatment**: `hisp_50`
- **Notes**: Coefficient: 2.93 pp (p=0.008). N=1,550.

### G3: Asian-owned businesses pay higher interest rates than White-owned businesses.
- **Expected sign**: +
- **Baseline spec(s)**: `baseline__asian_50`
- **Outcome**: `loanrate_w2`
- **Treatment**: `asian_50`
- **Notes**: Coefficient: 2.56 pp (p=0.009). N=1,253.

### G4: Native American-owned businesses pay higher interest rates (not significant).
- **Expected sign**: +
- **Baseline spec(s)**: `baseline__native_50`
- **Outcome**: `loanrate_w2`
- **Treatment**: `native_50`
- **Notes**: Coefficient: 1.29 pp (p=0.26). Not significant. N=1,324.

**Global Notes**: Paper: 'Racial Discrimination in Small Business Lending' (AER). Cross-sectional OLS with extensive controls, state+time FE, state-clustered SE. Black discrimination is the primary focus (most robustness specs). Placebo tests are concerning: race predicts some predetermined characteristics (business age, CEO experience). High credit score subsample (N=46) shows extreme coefficient (14.98) - likely unreliable. 60 total specs.

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baselines)** | **36** | |
| core_controls | 11 | |
| core_fe | 4 | |
| core_funcform | 3 | |
| core_inference | 3 | |
| core_method | 4 | |
| core_sample | 11 | |
| **Non-core tests** | **24** | |
| noncore_alt_outcome | 9 | |
| noncore_alt_treatment | 3 | |
| noncore_heterogeneity | 9 | |
| noncore_placebo | 3 | |
| **Total** | **60** | |

## Verification Checks

- Total specs in CSV: 60
- Unique spec_ids: 60
- All spec_ids unique: Yes
- Every spec has a category: Yes
- Every spec has is_baseline + is_core_test: Yes
