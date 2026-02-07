# Verification Report: 128521-V1

**Paper**: "Recessions, Mortality, and Migration Bias: Evidence from the Lancashire Cotton Famine"
**Journal**: AER
**Paper ID**: 128521-V1
**Verified**: 2026-02-03
**Verifier**: verification_agent

---

## Baseline Groups

### G1: Aggregate mortality DiD (cotton vs non-cotton districts)

**Claim**: Cotton districts experienced an apparent decrease in aggregate mortality rates relative to non-cotton districts during the Lancashire Cotton Famine (1861-1865). The negative coefficient in aggregate data reflects migration bias -- healthy individuals migrated out of cotton districts, making the remaining population appear healthier.

**Expected sign**: Negative (-)

**Baseline spec_ids**: baseline, baseline_controls, baseline_nearby, baseline_continuous

- baseline: No controls, district + period FE, population-weighted. Coef = -5.513, p = 0.191.
- baseline_controls: Full demographic and region controls. Coef = -4.924, p = 0.002.
- baseline_nearby: Full controls + nearby district indicators. Coef = -4.762, p = 0.005.
- baseline_continuous: Continuous treatment (cotton_eshr_post). Coef = -6.624, p = 0.003.

All four baselines use TWFE with district + period FE on agg_mr_tot. The primary difference is the control set and treatment definition. baseline_continuous uses a continuous cotton employment share rather than a binary cotton district indicator.

---

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **58** |
| Baselines | 4 |
| Core tests (non-baseline) | 35 |
| Non-core tests | 19 |
| Invalid | 0 |
| Unclear | 0 |

### Core test breakdown

| Category | Count |
|----------|-------|
| core_controls | 14 |
| core_sample | 11 |
| core_inference | 3 |
| core_fe | 3 |
| core_funcform | 1 |
| core_method | 7 |
| **Total core (incl. baselines)** | **39** |

Note: The 4 baselines are also tagged is_core_test=1 and included in the 39 total.

### Non-core breakdown

| Category | Count |
|----------|-------|
| noncore_alt_outcome | 7 |
| noncore_alt_treatment | 3 |
| noncore_heterogeneity | 5 |
| noncore_placebo | 2 |
| noncore_diagnostic | 2 |
| **Total non-core** | **19** |

---

## Classification Decisions

### Core tests (35 non-baseline specs)

1. **Control variations (10 specs)**: Leave-one-out control drops (robust/control/drop_*), control progression additions (robust/control/add_*), minimal controls (did/controls/minimal), no controls duplicate (did/controls/none), and density-squared addition (robust/funcform/density_squared). All preserve the same outcome (agg_mr_tot) and treatment (cotton_dist_post) with district + period FE.

2. **Sample restrictions (11 specs)**: Region drops (robust/sample/drop_region_*), nearby district exclusion, trimming/winsorizing, population-based subsamples. All preserve the core estimand.

3. **Inference variations (3 specs)**: Robust SE, county clustering, region clustering. Same point estimates as baseline_nearby, only standard errors change.

4. **FE variations (3 specs)**: Unit-only FE, region x time FE, county x time FE. These preserve the within-unit comparison that is essential to the DiD design.

5. **Functional form (1 spec)**: IHS/log outcome transformation. Preserves treatment and direction of effect.

6. **Method/weights (7 specs)**: Unweighted, log population weights, age-group population weights, continuous treatment duplicate. All preserve the core estimand while varying weighting or treatment measurement.

### Non-core specifications (19 specs)

1. **Alternative outcomes (7 specs)**: Age-specific mortality rates (under-15, 15-24, 25-34, 35-44, 45-54, 55-64, over-64). These are decompositions of total mortality by age group. While related to the main claim, they test a different outcome variable and represent heterogeneity across age groups rather than the total mortality claim.

2. **Alternative treatments (3 specs)**: nearby_post_25, nearby_post_50, nearby_post_75. These report the coefficient on nearby district indicators rather than the cotton district treatment. They test a fundamentally different causal object (spillover to nearby areas rather than direct effect on cotton districts).

3. **Heterogeneity interactions (5 specs)**: Treatment interacted with high density, elderly share, regional indicators, and high cotton share. These test whether the effect differs by subgroup, not the average treatment effect.

4. **Placebos (2 specs)**: Nearby effect on non-cotton districts only; population growth as outcome (cross-sectional, not DiD). These are by definition not tests of the main claim.

5. **Diagnostics (2 specs)**: No FE and time-only FE specifications. Both omit unit fixed effects, producing positive coefficients (+2.67 and +2.62) that reflect cross-sectional differences between cotton and non-cotton districts rather than within-unit changes. These fundamentally change the estimand and are not meaningful robustness checks of the DiD claim.

---

## Top 5 Most Suspicious Rows

1. **did/fe/none (spec row 6)**: Reports coefficient of +2.67 with no fixed effects. This is a pure cross-sectional comparison, not a DiD. Classified as noncore_diagnostic because it fundamentally changes the estimand.

2. **did/fe/time_only (spec row 8)**: Reports coefficient of +2.62 with only period FE. Without unit FE, this is not a proper DiD. Classified as noncore_diagnostic.

3. **did/controls/none (spec row 11)**: This is an exact duplicate of baseline -- same coefficient (-5.513), same SE, same everything. The spec_tree_path differs but the results are identical. Classified as core but noted as redundant.

4. **robust/treatment/continuous (spec row 39)**: This appears to be an exact duplicate of baseline_continuous -- same coefficient (-6.624), same SE. Classified as core_method but is redundant with the baseline.

5. **robust/placebo/pop_growth (spec row 53)**: Uses pop_growth as outcome and cotton_dist (not cotton_dist_post) as treatment, with no FE and only 538 obs (cross-section). This is a fundamentally different specification -- a cross-sectional regression of pre-famine population growth on cotton district status. Correctly classified as placebo/non-core, but the spec_id labeling could be clearer.

---

## Recommendations for Spec-Search Script

1. **Deduplicate identical specifications**: did/controls/none is identical to baseline, and robust/treatment/continuous is identical to baseline_continuous. The script should detect and flag duplicates.

2. **Flag FE-omission specs more clearly**: did/fe/none and did/fe/time_only produce sign-reversed coefficients because they omit the core identifying variation. The search script should tag these as diagnostic/sensitivity rather than as proper robustness checks.

3. **Clarify age-specific outcome classification**: The age-specific mortality specs could be classified as either alternative outcomes or as heterogeneity. If the paper claim is specifically about total mortality, these should be classified as alternative outcomes (as done here). If the claim encompasses age-specific effects, they could be core tests.

4. **Treatment variable consistency in nearby specs**: The robust/treatment/nearby_post_* specs report the coefficient on nearby indicators while including cotton_dist_post as a control. This effectively tests spillover effects, not the main treatment effect. The script should better distinguish between using a different treatment definition for the same causal effect vs. testing a different causal effect entirely.

5. **Cross-sectional placebo specs**: The robust/placebo/pop_growth spec uses only 538 obs (one cross-section) with no FE. The search script should flag specs that fundamentally depart from the panel DiD structure.
