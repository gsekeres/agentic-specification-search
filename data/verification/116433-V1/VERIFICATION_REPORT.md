# Verification Report: 116433-V1

## Paper
**Title**: Referrals: Peer Screening and Enforcement in a Consumer Credit Field Experiment
**Authors**: Bryan, Karlan, and Zinman
**Journal**: American Economic Review

## Baseline Groups

2x2 factorial RCT testing enforcement and selection across repaid, charged_off, interest, portion. 8 baseline groups (G1-G8).

| Group | Treatment | Outcome | Baseline spec_id | Coef | p-value |
|-------|-----------|---------|-------------------|------|---------|
| G1 | enforcement | charged_off | baseline_charged_off_enforcement | -0.102 | 0.016 |
| G2 | enforcement | repaid | baseline_repaid_enforcement | -0.094 | 0.056 |
| G3 | enforcement | interest | baseline_interest_enforcement | -0.188 | 0.002 |
| G4 | enforcement | portion | baseline_portion_enforcement | -0.129 | 0.017 |
| G5 | selection | charged_off | baseline_charged_off_selection | 0.040 | 0.339 |
| G6 | selection | repaid | baseline_repaid_selection | 0.039 | 0.413 |
| G7 | selection | interest | baseline_interest_selection | 0.067 | 0.262 |
| G8 | selection | portion | baseline_portion_selection | 0.050 | 0.335 |

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **236** |
| Baseline | 8 |
| Core tests (incl. baseline) | 170 |
| Non-core | 66 |
| Invalid | 0 |
| Unclear | 0 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 108 |
| core_funcform | 6 |
| core_inference | 20 |
| core_method | 4 |
| core_sample | 32 |
| noncore_alt_treatment | 6 |
| noncore_heterogeneity | 54 |
| noncore_placebo | 6 |

## Top 5 Most Suspicious Rows

### 1. robust/sample/low_income_repaid_enforcement and low_income_repaid_selection
- **Issue**: Both report identical coefficients (-0.042) and p-values (0.559)
- **Cause**: Coding error; wrong coefficient extracted for one treatment
- **Classification**: noncore_heterogeneity (demographic subgroup)

### 2. Sample restriction specs on binary outcomes (repaid, charged_off)
- **Issue**: All trim/winsor specs for binary outcomes identical to baseline
- **Cause**: Trimming/winsorizing binary variables has no effect
- **Note**: Technically valid but uninformative; classified as core_sample

### 3. robust/treatment/interaction_repaid_enforcement
- **Issue**: Main effects from interaction model differ from additive baseline
- **Classification**: noncore_heterogeneity

### 4. robust/het/relationship_work_repaid_enforcement
- **Issue**: Coefficient near zero (0.0001, p=0.998) vs full-sample -0.094
- **Cause**: Small work-colleague subsample; no treatment effect
- **Classification**: noncore_heterogeneity

### 5. robust/cluster/branch_repaid_enforcement
- **Issue**: p-value = 4.6e-16, which is suspiciously small
- **Cause**: Branch-level clustering with few clusters may produce anti-conservative inference
- **Classification**: core_inference (kept as valid but flagged)

## Recommendations

1. **Fix low_income subgroup specs**: Identical coefficients for enforcement and selection in low_income_repaid suggest a bug in coefficient extraction.
2. **Remove/flag binary-outcome sample restrictions**: Trimming/winsorizing binary outcomes produces no variation. These 16 specs are duplicates of baselines.
3. **Consider consolidating baseline groups**: 8 groups from 4 outcomes x 2 treatments may be excessive. The primary claim centers on enforcement effects (G1-G4).
4. **Heterogeneity specs**: 54 non-core specs are from demographic subgroups, relationship-type subgroups, and interaction models. These should not be pooled with core robustness specs.
5. **Branch clustering**: The branch-level clustering produces extremely small p-values for some specs (4.6e-16), likely due to too few clusters. Consider flagging these in downstream analysis.
