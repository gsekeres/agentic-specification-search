# Verification Report: 114707-V1

## Paper Overview
- **Paper**: Do Hospital Mergers Reduce Costs? Evidence on Prices and Hospital Market Consolidation (Schmitt, 2017)
- **Paper ID**: 114707-V1
- **Method**: Difference-in-Differences with hospital and year fixed effects
- **Verified**: 2026-02-03
- **Verifier**: verification_agent

## Baseline Groups

### G1: Merger effect on non-medical prices
- **Claim**: Hospital mergers increase non-medical prices by approximately 7%, estimated via DiD with hospital and year FE.
- **Baseline spec_ids**: `baseline`
- **Outcome**: `lnprnonmed` (log non-medical prices)
- **Treatment**: `post` (post-merger indicator)
- **Expected sign**: Positive (+)
- **Baseline coefficient**: 0.070, SE = 0.017, p < 0.001
- **Controls**: lncmi, pctmcaid, lnbeds, fp, hhi, sysoth
- **FE**: Hospital + Year
- **Cluster**: Hospital
- **Weights**: Total discharges

Only one baseline group is warranted. The paper's core claim is about non-medical prices. Medical prices are explicitly treated as a placebo/falsification outcome in the paper.

## Counts

| Category | Count |
|----------|-------|
| Total specifications | 85 |
| Baseline | 1 |
| Core tests (incl. baseline) | 53 |
| Non-core | 31 |
| Invalid | 1 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_controls | 14 | Leave-one-out drops, control progression, no-controls |
| core_sample | 28 | Period splits, year drops, winsorization, trimming, balanced panel, matched/same-system |
| core_fe | 3 | Hospital-only, year-only, no FE |
| core_inference | 2 | Robust SE, cluster by year |
| core_funcform | 5 | Standardized, level, IHS, quadratic HHI, quadratic beds |
| core_method | 1 | Unweighted regression |
| noncore_heterogeneity | 15 | Subgroup splits (FP/NP, size, HHI, system) and interaction terms |
| noncore_alt_treatment | 7 | Active/passive, in-state, indirect, combinations with matched/same-system samples |
| noncore_placebo | 5 | Medical prices placebo (x2), pre-trend, fake treatment lead, pre-period effect |
| noncore_alt_outcome | 2 | Medical prices on matched and same-system samples |
| noncore_diagnostic | 2 | Event study contemporaneous (t=0) and long-run (t=+4) coefficients |
| invalid | 1 | Failed estimation (non-system hospitals) |

## Top 5 Most Suspicious Rows

1. **robust/het/by_system_nonmember** (INVALID): Estimation failed with missing coefficient and p-value. The SPECIFICATION_SEARCH.md notes this is likely due to collinearity between the sysoth control variable and the sample restriction to non-system hospitals. Correctly marked as invalid.

2. **robust/placebo/outcome_lnprmed**: This is a duplicate of `robust/outcome/lnprmed` -- both have identical coefficients (-0.00041) and p-values (0.944). The same specification appears twice under different tree paths (robustness/placebo_tests.md vs robustness/measurement.md). Not technically wrong, but inflates the spec count. Both are correctly classified as non-core.

3. **robust/control/add_sysoth**: This is numerically identical to baseline (coefficient = 0.07017, p = 4.41e-05). It is the endpoint of the control progression where all 6 controls have been added, reproducing the baseline. It is classified as core_controls but is effectively a duplicate of baseline.

4. **did/dynamic/leads_lags** (treatment = `_p0`): The reported coefficient (0.0423, p = 0.072) is the event study effect at t=0, not the cumulative post-merger effect. This is a different estimand from the baseline `post` variable. Correctly classified as noncore_diagnostic, but users should be aware this is not directly comparable to baseline.

5. **did/treatment/hhi_terciles** (treatment = `post_hhi1`): The coefficient (0.028, p = 0.24) represents mergers in the lowest HHI tercile only. This is really a heterogeneity test that interacts treatment with market structure, not an alternative test of the main claim. Correctly classified as noncore_heterogeneity.

## Recommendations

1. **Duplicate removal**: The placebo medical-prices specification appears twice (`robust/outcome/lnprmed` and `robust/placebo/outcome_lnprmed`). The spec-search script should check for duplicate coefficient/SE/p-value combinations and avoid recording the same regression twice under different tree paths.

2. **Baseline duplication**: `robust/control/add_sysoth` is mechanically identical to baseline since it adds all controls sequentially and the final step produces the full control set. This is expected behavior for a control progression but could be flagged in the script to avoid double-counting.

3. **Event study coefficients**: The event study specs report individual lead/lag coefficients rather than the overall post-merger effect. This is appropriate for diagnostics but should not be mixed with the main specification pool. The spec-search script correctly tags these under `did/dynamic` but could benefit from a clearer "diagnostic" flag.

4. **Heterogeneity treatment definitions**: Several specs labeled as `did/treatment/` (hhi_terciles, bedshr_terciles, sizediff) are really heterogeneity analyses that interact the treatment with market/hospital characteristics, rather than alternative treatment definitions. The spec-search script could classify these more clearly under heterogeneity.

5. **The baseline claim is correctly identified**: The spec-search script correctly identifies Table 2 Column 2 as the main result, with outcome = lnprnonmed, treatment = post, and the full set of controls. No corrections needed to the baseline identification.
