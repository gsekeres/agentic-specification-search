# Verification Report: 223321-V1

## Paper Information
- **Paper ID**: 223321-V1
- **Total Specifications**: 62
- **Baseline Specifications**: 3
- **Core Test Specifications**: 37
- **Non-Core Specifications**: 25

## Baseline Groups

### G1: Children of Mexican immigrant parents who arrived at ages 0-8 are more likely to
- **Expected sign**: +
- **Baseline spec(s)**: `baseline`
- **Outcome**: `not_hisp`
- **Treatment**: `par_Arrived0_8`
- **Notes**: Coefficient: 0.0014 (SE~0.0008, p=0.10). NOT significant at 5% in the full specification. The no-controls (0.011, p<0.001) and basic-controls (0.012, p<0.001) versions are significant, but the effect is largely absorbed by intermarriage controls.

**Global Notes**: Paper: Effect of Parent's Age at Arrival on Child's Hispanic Identity (AER P&P 2025). WLS with Census/ACS data (N=281,076). The key finding is that the raw effect (1.1pp) drops to 0.14pp and becomes insignificant when intermarriage is controlled. This suggests the mechanism is through marriage patterns rather than direct cultural transmission. The paper presents 3 baselines: no controls, basic controls, and full controls. The full controls spec is the preferred one. Placebo test on parent's own identity is significant (0.18pp), which is a concern. Heterogeneity by parent gender is stark (strong for fathers, null for mothers). 62 total specs.

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baselines)** | **37** | |
| core_controls | 18 | |
| core_inference | 4 | |
| core_method | 5 | |
| core_sample | 10 | |
| **Non-core tests** | **25** | |
| noncore_alt_treatment | 6 | |
| noncore_heterogeneity | 18 | |
| noncore_placebo | 1 | |
| **Total** | **62** | |

## Verification Checks

- Total specs in CSV: 62
- Unique spec_ids: 62
- All spec_ids unique: Yes
- Every spec has a category: Yes
- Every spec has is_baseline + is_core_test: Yes
