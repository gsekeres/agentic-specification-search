# Verification Report: 215802-V1

## Paper Information
- **Paper ID**: 215802-V1
- **Total Specifications**: 80
- **Baseline Specifications**: 1
- **Core Test Specifications**: 34
- **Non-Core Specifications**: 46

## Baseline Groups

### G1: Current food stamp receipt is associated with health status (cross-sectional). N
- **Expected sign**: -
- **Baseline spec(s)**: `baseline`
- **Outcome**: `good_health`
- **Treatment**: `fs_receipt`
- **Notes**: Coefficient: -0.018 (SE~0.013, p=0.18). Negative association reflects selection, not causation. The original paper's childhood exposure design cannot be replicated with public data.

**Global Notes**: Paper: 'Long-Run Impacts of Childhood Access to the Safety Net' by Hoynes, Schanzenbach & Almond (AER). CRITICAL: The public replication package lacks county identifiers needed for the original identification strategy. This spec search uses current FS receipt (not childhood exposure) as treatment, yielding fundamentally different estimates driven by selection. Panel FE specs for income/employment show strong negative associations (FS recipients are poorer), confirming selection. Cross-sectional health results are not significant. 80 total specs.

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baselines)** | **34** | |
| core_controls | 10 | |
| core_fe | 2 | |
| core_funcform | 5 | |
| core_inference | 6 | |
| core_method | 1 | |
| core_sample | 10 | |
| **Non-core tests** | **46** | |
| noncore_alt_outcome | 28 | |
| noncore_alt_treatment | 5 | |
| noncore_heterogeneity | 10 | |
| noncore_placebo | 3 | |
| **Total** | **80** | |

## Verification Checks

- Total specs in CSV: 80
- Unique spec_ids: 80
- All spec_ids unique: Yes
- Every spec has a category: Yes
- Every spec has is_baseline + is_core_test: Yes
