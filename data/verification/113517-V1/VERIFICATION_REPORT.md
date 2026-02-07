# Verification Report: 113517-V1

## Paper
**Title**: The Relative Power of Employment-to-Employment Reallocation and Unemployment Exits in Predicting Wage Growth
**Journal**: AEJ Policy
**Paper ID**: 113517-V1

## Baseline Groups

### G1: EE reallocation predicts wage growth (primary claim)
- **Baseline spec_ids**: baseline, baseline_ee_ue, baseline_ee_ue_ur
- **Outcome**: xdlogern_nom (change in log nominal earnings)
- **Treatment**: xee (employment-to-employment transition rate)
- **Expected sign**: positive
- **Notes**: The horse race specification (baseline_ee_ue) where EE is regressed alongside UE is the most canonical. baseline alone (EE without controls) and baseline_ee_ue_ur (EE controlling for both UE and UR) are also baselines for the same claim.

### G2: UE transitions predict wage growth (secondary claim)
- **Baseline spec_ids**: baseline_ue, baseline_ee_ue_ue
- **Outcome**: xdlogern_nom
- **Treatment**: xue (unemployment-to-employment transition rate)
- **Expected sign**: positive
- **Notes**: baseline_ue is UE alone; baseline_ee_ue_ue is the UE coefficient from the horse race with EE. The much smaller coefficient in the horse race supports the paper's relative-power claim.

### G3: Unemployment rate predicts wage growth (tertiary claim)
- **Baseline spec_ids**: baseline_ur
- **Outcome**: xdlogern_nom
- **Treatment**: xur (unemployment rate)
- **Expected sign**: positive
- **Notes**: Single baseline. UR alone with market FE. This is a secondary comparison variable.

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **64** |
| Baselines | 6 |
| Core tests (non-baseline) | 51 |
| Non-core tests | 7 |
| Invalid | 0 |
| Unclear | 0 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_method | 11 (baselines + weights + first diff) |
| core_controls | 7 |
| core_sample | 21 |
| core_inference | 3 |
| core_fe | 4 |
| core_funcform | 11 |
| noncore_heterogeneity | 4 |
| noncore_placebo | 2 |
| noncore_alt_treatment | 1 |
| noncore_alt_outcome | 0 |
| noncore_diagnostic | 0 |
| invalid | 0 |
| unclear | 0 |

## Classification Decisions

### Demographic subgroup splits (by_gender, by_race, by_education)
These 6 specs run the identical EE regression on demographic subsamples (male markets, female markets, white markets, black markets, low-education markets, high-education markets). Because they preserve the same estimand (EE coefficient on wage growth) and only restrict the sample, they are classified as **core_sample** rather than noncore_heterogeneity. This is a judgment call; one could argue they are heterogeneity analyses. The key distinction is that the reported coefficient is the *main* EE effect within the subsample, not an interaction term.

### Interaction heterogeneity specs
The 4 interaction specs (interact_xee_xue, interaction_gender, interaction_race, interaction_education) add interaction terms to the model. The reported EE coefficient becomes a *conditional* main effect (e.g., EE effect for females when male interaction is included). These are classified as **noncore_heterogeneity** because the reported coefficient's interpretation changes due to the interaction.

### Placebo/dynamics specs
- robust/placebo/lag_treatment and robust/placebo/lead_treatment use shifted (lagged/lead) EE as the treatment. These are **noncore_placebo** as they test a different causal timing.
- robust/dynamics/contemp_and_lag includes both contemporaneous and lagged EE. The *contemporaneous* EE coefficient is reported, so this is classified as **core_controls** (adding lagged terms as controls).

### First differences
panel/method/first_diff uses first-differenced variables (d_outcome, dxee). This is a valid alternative estimation method for the same underlying relationship, classified as **core_method**.

### Duplicate specifications
Several specs produce identical results to baseline_ee_ue:
- robust/control/add_ue (same as baseline_ee_ue)
- robust/loo/drop_ur (same as baseline_ee_ue)
- panel/fe/unit (same as baseline_ee_ue)
- robust/cluster/none (same coefficient, robust SEs which are the default)
- robust/weights/weighted (same as baseline_ee_ue)
- robust/sample/min_obs_10 (filter did not bind)
- robust/sample/min_obs_20 (filter did not bind)

These are still valid specifications (the duplication indicates the baseline uses those settings), but they do not add independent information.

## Top 5 Most Suspicious Rows

1. **robust/sample/min_obs_10 and min_obs_20**: Identical results to baseline_ee_ue (N=16288 unchanged). The minimum-observation filters did not exclude any markets, making these specifications uninformative. They should ideally use stricter thresholds.

2. **robust/treatment/ur_main**: Reports UR coefficient (0.0067, p=0.069) from the full model with EE and UE as controls. Assigned to G3 but baseline_ur is UR *alone* (coeff=0.075); the controlled version is a very different object. Classified as noncore_alt_treatment because the comparison is not clean.

3. **robust/form/x_log**: The log-transformed EE coefficient is negative (-0.0013). This sign reversal may reflect the log transformation's interaction with the within-market variation rather than a genuine reversal of the relationship. The negative sign warrants attention but is plausibly a functional form artifact.

4. **robust/control/none**: Identical to the baseline spec (same coefficient 0.309, same N=16288). This is a pure duplicate labeled as a robustness check.

5. **robust/form/x_standardized**: Very small coefficient (0.00197) due to standardization. The R-squared drops to 0.004, much lower than other specs, which is suspicious and may indicate a coding issue with the standardization.

## Recommendations

1. **Remove duplicate specifications**: At least 7 specs are exact duplicates of baseline_ee_ue. Consider removing or flagging them to avoid inflating the specification count.

2. **Tighten minimum-observation filters**: min_obs_10 and min_obs_20 do not bind. Use more restrictive thresholds (e.g., 50+ or 100+ observations per market) to create meaningful variation.

3. **Investigate x_standardized R-squared**: The very low R-squared (0.004) for the standardized treatment spec is suspicious and should be verified against the original code.

4. **Consider separating G2 robustness specs**: Currently only 2 specs directly test the UE claim (baseline_ue and baseline_ee_ue_ue, plus robust/treatment/ue_main). The spec search was heavily focused on EE. Adding more UE-focused robustness specs would better test the relative-power claim.

5. **Clarify the baseline_ee_ue vs baseline distinction**: The horse race (controlling for UE) is the most relevant baseline for the paper's *relative power* claim, but most robustness specs compare to the EE-alone coefficient. The spec search should ensure the primary comparison is to the horse race specification.
