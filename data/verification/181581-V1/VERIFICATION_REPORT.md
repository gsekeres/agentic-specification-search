# Verification Report: 181581-V1

**Paper**: "When a Doctor Falls from the Sky: The Impact of Easing Doctor Supply Constraints on Mortality"
**Author**: Edward Okeke
**Journal**: AER: Insights (2023)
**Verified**: 2026-02-04
**Verifier**: verification_agent

---

## Baseline Groups

### G1: Doctor treatment effect on 7-day infant mortality
- **Claim**: Adding doctors to primary health centers reduces 7-day infant mortality in Nigeria (RCT).
- **Expected sign**: Negative
- **Baseline spec_ids**: `baseline`, `baseline_basic_controls`, `baseline_extended_controls`
- **Notes**: Three columns from Table 4. Col 1 has strata FE only, no controls. Col 2 adds basic individual/demographic controls and quarter FE. Col 3 (preferred) adds health center controls. All use `mort7` as outcome and `doctor` as treatment, clustered at facility level.

### G2: MLP treatment effect on 7-day infant mortality
- **Claim**: Adding mid-level providers (MLPs) to health centers has a smaller and insignificant effect on 7-day infant mortality.
- **Expected sign**: Negative (but expected to be smaller/insignificant relative to G1)
- **Baseline spec_ids**: `baseline_mlp`
- **Notes**: Secondary finding. The MLP arm is included in the same regression as the doctor arm in `baseline`. Only 2 specs in the entire file test this treatment: `baseline_mlp` and `robust/treatment/mlp_vs_control`.

---

## Summary Counts

| Category | Count |
|----------|-------|
| Total specifications | 93 |
| Baseline specs | 4 |
| Core test specs (is_core_test=1) | 69 |
| Non-core specs | 19 |
| Invalid specs | 5 |
| Unclear specs | 0 |

### Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_controls | 36 | Baselines + leave-one-out + control progression variations |
| core_sample | 24 | Drop-quarter, drop-state, demographic subsamples, dose subsamples, doctor-only/mlp-only |
| core_fe | 5 | Fixed effects structure variations (none, strata-only, time-only, state, state+time) |
| core_inference | 3 | Clustering variations (robust SE, strata-level, state-level) |
| core_funcform | 1 | 30-day mortality (related mortality outcome, preserves the claim) |
| noncore_heterogeneity | 9 | Interaction terms (doctor x male, first, hausa, cct, hc equipment, hc deliveries, hc workers, autonomy, car) |
| noncore_alt_outcome | 5 | Non-mortality outcomes (birthweight, low birthweight, log weight, log length, postpartum fever) |
| noncore_placebo | 4 | Pre-determined characteristics (past death, mother age, mother education, ethnicity) |
| noncore_alt_treatment | 1 | Pooled any_treat variable (changes the causal object) |
| invalid | 5 | Specs identical to baseline_extended_controls (ghost variables/quarters) |

---

## Top 5 Most Suspicious Rows

1. **`robust/loo/drop_magedum_5`**: Coefficient and SE are identical to `baseline_extended_controls` to full floating-point precision. This strongly suggests `magedum_5` does not exist in the extended controls model. The baseline coefficient_vector_json shows only `magedum_1` through `magedum_4`, confirming this is a ghost variable. Marked as **invalid**.

2. **`robust/loo/drop_mschool_4`**: Same issue as above. The coefficient_vector_json for the extended baseline shows `mschool_1`, `mschool_2`, and `mschool_3` only. `mschool_4` does not exist. Marked as **invalid**.

3. **`robust/loo/drop_hc_clean_4`**: Same issue. The coefficient_vector_json shows `hc_clean_1`, `hc_clean_2`, and `hc_clean_3`. `hc_clean_4` does not exist. Marked as **invalid**.

4. **`robust/sample/drop_qtr_-1187`**: Quarter -1187 is an implausible quarter number, likely resulting from a coding error in generating quarter values. Coefficient identical to baseline_extended_controls. Marked as **invalid**.

5. **`robust/sample/drop_qtr_190`**: Quarter 190 produces identical results to baseline_extended_controls, meaning no observations were actually dropped. This quarter may not exist in the data or has zero observations. Marked as **invalid**.

---

## Classification Rationale

### Heterogeneity specs (noncore)
The 9 heterogeneity specs (`robust/heterogeneity/*`) report the main effect of `doctor` from a model that includes an interaction term (e.g., `doctor x male`). While the reported coefficient is the main effect of doctor, the model is structurally different because the main effect's interpretation changes conditional on the interaction variable's value. The interaction term makes the main effect represent the treatment effect for the omitted category only. These are classified as **noncore_heterogeneity** rather than core tests.

### Alternative outcomes (noncore)
Five outcome variables (birthweight, low birthweight, log weight, log length, postpartum fever) are substantively different from the mortality claim. The paper itself notes these are expected to show no effect given the intervention focuses on emergency care. These are not tests of the core mortality hypothesis.

### 30-day mortality (core)
`robust/outcome/mort30` uses 30-day mortality instead of 7-day mortality. This is classified as **core_funcform** because it tests the same fundamental claim (doctor assignment reduces infant mortality) with a slightly different measurement window. The claim is about mortality reduction, and extending the window from 7 to 30 days is a measurement variation, not a fundamentally different outcome.

### Treatment variations
- `robust/treatment/doctor_vs_control`: Core sample restriction (drops MLP arm, keeping only doctor vs control comparison). Preserves the G1 claim.
- `robust/treatment/mlp_vs_control`: Core sample restriction for G2 (drops doctor arm, MLP vs control only).
- `robust/treatment/any_intervention`: Noncore because `any_treat` pools doctor and MLP arms, changing the causal object from "doctor" to "any health worker."

### Dose subsamples
`robust/sample/dose_1` and `robust/sample/dose_2` split the doctor arm by implementation intensity. These are classified as core_sample because they test whether the doctor treatment effect holds across dose levels, preserving the same treatment variable and outcome.

---

## Recommendations for Spec-Search Script

1. **Remove ghost LOO specs**: The script should verify that a variable actually exists in the model before attempting to drop it. Variables like `magedum_5`, `mschool_4`, and `hc_clean_4` are not in the extended controls model. A check comparing the variable name against the actual coefficient vector would prevent these invalid specs.

2. **Validate quarter/time drops**: The script should verify that dropping a quarter actually changes the sample size before recording the result. Quarters -1187 and 190 produced no change, indicating they don't exist in the data.

3. **Control progression ordering**: The incremental control addition uses an arbitrary ordering of controls. The add_5_magedum_4 and add_6_magedum_5 specs produce identical results (since magedum_5 doesn't exist), suggesting the variable list used for control progression includes non-existent variables.

4. **Heterogeneity reporting**: The heterogeneity specs report the main effect of `doctor`, but the more informative coefficient would be the interaction term itself. Consider reporting both the main effect and the interaction coefficient, or at minimum flagging in the output that the model includes an interaction.

5. **Baseline claim is correct**: The baseline specifications correctly identify the paper's main claim (Table 4). The treatment variable (`doctor`), outcome (`mort7`), fixed effects (strata, strata+quarter), and clustering (facility) all match the paper. No changes needed to the baseline identification.
