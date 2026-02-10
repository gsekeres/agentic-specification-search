# Verification Report: 214341-V1

## Paper Information
- **Paper ID**: 214341-V1
- **Total Specifications**: 164
- **Baseline Specifications**: 8
- **Core Test Specifications**: 87
- **Non-Core Specifications**: 77

## Baseline Groups

### G1: Workers earn positive markups (surplus > 1) relative to their minimum willingnes
- **Expected sign**: +
- **Baseline spec(s)**: `baseline__hrly_surp_rel_min`
- **Outcome**: `hrly_surp_rel_min`
- **Treatment**: `mean_surplus`
- **Notes**: Mean surplus ratio: 1.24 (p<0.001). Unweighted. N=84 survey respondents with hourly contracts.

### G2: Workers earn positive markups relative to ex-post WTA (adjusted after job comple
- **Expected sign**: +
- **Baseline spec(s)**: `baseline__hrly_surp_rel_expost`
- **Outcome**: `hrly_surp_rel_expost`
- **Treatment**: `mean_surplus`
- **Notes**: Mean surplus ratio: 1.16 (p<0.001). Smaller than ex-ante WTA markup.

### G3: Workers earn positive markups relative to their outside option wage.
- **Expected sign**: +
- **Baseline spec(s)**: `baseline__hrly_surp_rel_outside`
- **Outcome**: `hrly_surp_rel_outside`
- **Treatment**: `mean_surplus`
- **Notes**: Mean surplus ratio: 1.50 (p<0.001). N=46 (fewer workers report outside wages). Largest markup.

### G4: Workers earn positive markups on fixed-price contracts relative to ex-post WTA.
- **Expected sign**: +
- **Baseline spec(s)**: `baseline__fixed_surp_rel_expost`
- **Outcome**: `fixed_surp_rel_expost`
- **Treatment**: `mean_surplus`
- **Notes**: Mean surplus ratio: 1.87 (p<0.001). N=99 fixed-price contracts. Largest markup of all.

**Global Notes**: Paper: 'Who Benefits from the Online Gig Economy?' by Stanton & Thomas (AER). Survey of 113 workers on a gig platform. Four surplus measures (hourly WTA, hourly ex-post, hourly outside wage, fixed-price ex-post). Many specs apply across all 4 outcomes. Balance tests compare survey participants vs non-participants. Quantile analysis shows surplus distribution. OLS predictor specs test what predicts surplus levels. IPW weights adjust for selection into survey.

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baselines)** | **87** | |
| core_controls | 3 | |
| core_funcform | 8 | |
| core_inference | 4 | |
| core_method | 8 | |
| core_sample | 64 | |
| **Non-core tests** | **77** | |
| noncore_alt_treatment | 12 | |
| noncore_diagnostic | 3 | |
| noncore_heterogeneity | 60 | |
| noncore_placebo | 2 | |
| **Total** | **164** | |

## Verification Checks

- Total specs in CSV: 164
- Unique spec_ids: 164
- All spec_ids unique: Yes
- Every spec has a category: Yes
- Every spec has is_baseline + is_core_test: Yes
