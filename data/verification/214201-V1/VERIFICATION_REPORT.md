# Verification Report: 214201-V1

## Paper Information
- **Paper ID**: 214201-V1
- **Total Specifications**: 55
- **Baseline Specifications**: 1
- **Core Test Specifications**: 32
- **Non-Core Specifications**: 23

## Baseline Groups

### G1: Mission-based incentives (video treatment without bonus) increase household visi
- **Expected sign**: +
- **Baseline spec(s)**: `baseline`
- **Outcome**: `lhw_visit`
- **Treatment**: `treat_mission_nobonus`
- **Notes**: Coefficient: 0.051 (SE=0.012, p<0.001). Block and wave FE, household weights, LHW-clustered SE. 5.1pp increase on a ~36% base = ~14% increase.

**Global Notes**: Paper: Mission vs Financial Incentives for Community Health Workers. Block-randomized RCT in Pakistan with multiple treatment arms. The primary claim is about the mission treatment (video showing health impact). Financial incentives (treat_bonus_pr) show ~2x the effect. Social recognition treatment is a placebo (null). Pre-treatment placebo confirms no pre-existing differences. 55 total specs.

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baselines)** | **32** | |
| core_controls | 4 | |
| core_fe | 3 | |
| core_inference | 3 | |
| core_method | 2 | |
| core_sample | 20 | |
| **Non-core tests** | **23** | |
| noncore_alt_outcome | 8 | |
| noncore_alt_treatment | 4 | |
| noncore_heterogeneity | 9 | |
| noncore_placebo | 2 | |
| **Total** | **55** | |

## Verification Checks

- Total specs in CSV: 55
- Unique spec_ids: 55
- All spec_ids unique: Yes
- Every spec has a category: Yes
- Every spec has is_baseline + is_core_test: Yes
