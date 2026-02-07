# Verification Report: 205581-V2

## Paper: Universalism - Global Evidence (AER)

## Baseline Groups

Three baseline groups were identified, all using the same specification structure (OLS with country FE, strata-clustered SEs, standard demographic controls) but differing in outcome variable:

| Group | Baseline spec_id | Outcome | Coefficient | p-value |
|-------|------------------|---------|-------------|---------|
| G1 | baseline/univ_overall | univ_overall (composite) | 0.554 | <0.001 |
| G2 | baseline/univ_domestic | univ_domestic | 0.392 | 0.006 |
| G3 | baseline/univ_foreign | univ_foreign | 0.678 | <0.001 |

All three test the same treatment (moral_treatment) on the same sample (Full sample) with country FE and controls (age, agesq, male_num, college, city, income_code), SEs clustered at strata level.

## Summary Counts

| Category | Count |
|----------|-------|
| Total specifications | 66 |
| Baselines | 3 |
| Core tests (non-baseline) | 53 |
| Non-core specifications | 10 |
| Invalid | 0 |
| Unclear | 0 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 24 |
| core_sample | 18 |
| core_funcform | 7 |
| core_inference | 3 |
| core_fe | 2 |
| core_method | 2 |
| noncore_heterogeneity | 6 |
| noncore_alt_treatment | 3 |
| noncore_alt_outcome | 1 |
| invalid | 0 |
| unclear | 0 |

## Classification Rationale

### Core Tests (53 specs)
The majority of specifications are genuine robustness checks of the primary claim (G1: moral framing increases composite universalism). These include:
- **Control variations (24)**: Leave-one-out drops, stepwise additions, no-controls, demographics-only, full controls, plus control variations for G2/G3 outcomes
- **Sample restrictions (18)**: Income splits, gender splits, age splits, education splits, urban/rural, outcome trimming, country drops, complete cases
- **Functional form (7)**: Scaled outcome, log, IHS, quadratic age, raw data versions for all three outcomes
- **Inference (3)**: No clustering, strata clustering, country clustering
- **FE structure (2)**: No FE (pooled OLS), country FE only
- **Method (2)**: Survey weighting variation, 3-arm model (moral effect)

### Non-core Tests (10 specs)
- **Heterogeneity interactions (6)**: These specs add interaction terms (treat x male, treat x age, etc.) but report the main effect of moral_treatment, which is no longer the average treatment effect in an interaction model. The coefficient represents the effect for the reference category only. These are not direct tests of the baseline claim.
- **Alternative treatments (3)**: deserving_vs_baseline, moral_vs_deserving, and three_arm_deserving change the treatment variable, which changes the causal object from the baseline claim. These are informative but not comparable to the baseline.
- **Alternative outcome (1)**: univ_diff (foreign minus domestic) is a different estimand not represented in any baseline.

## Top 5 Most Suspicious Rows

1. **robust/cluster/strata** (row 43): Exact duplicate of baseline/univ_overall -- identical coefficient (0.5542407) and SE (0.1266493). The cluster_var field says "strata" which is the same as baseline. This is a no-op specification that adds no information.

2. **robust/estimation/country_fe_only** (row 46): Exact duplicate of baseline/univ_overall -- identical coefficient and SE. The baseline already uses country FE, so "country FE only" is not a variation.

3. **robust/funcform/quadratic_age** (row 50): Exact duplicate of baseline/univ_overall. The baseline already includes agesq (age squared), so adding "quadratic age" is redundant.

4. **robust/weights/survey_weighted** (row 64): Described as "Unweighted (weights available but not applied)" but has identical coefficient/SE to baseline. If baseline was also unweighted, this is another no-op. The label is misleading.

5. **robust/sample/complete_cases** (row 65): Exact duplicate of baseline/univ_overall. The baseline already uses complete-case analysis within the OLS, so this adds no information.

6. **robust/control/add_income_code** (row 23): Last step of control progression, identical to baseline. This is expected but still a duplicate.

**Note**: At least 6 specifications produce identical coefficients/SEs to the baseline, suggesting the specification search generated some redundant runs. These are classified as core but flagged with lower confidence (0.80-0.90).

## Recommendations for Spec-Search Script

1. **Deduplicate before saving**: Several specifications are exact copies of the baseline (robust/cluster/strata, robust/estimation/country_fe_only, robust/funcform/quadratic_age, robust/weights/survey_weighted, robust/sample/complete_cases). The script should detect and either skip or flag exact duplicates.

2. **Heterogeneity coefficient extraction**: The heterogeneity specs report the main effect (moral_treatment coefficient) from an interaction model rather than the interaction term. For these to be useful as heterogeneity tests, the script should extract the interaction coefficient (e.g., treat_x_male) instead. As-is, the reported coefficient is the treatment effect for the omitted category, which is not directly comparable to the average treatment effect in the baseline.

3. **Treatment variation labeling**: The three_arm_moral spec uses treat_moral as the treatment variable, while the baseline uses moral_treatment. These are different parameterizations of the same treatment (binary in baseline vs. two dummies in three-arm). The script should note this distinction more clearly.

4. **Raw data outcomes**: The raw data outcome specifications (univ_overall_rawdata, etc.) use a different measurement scale. While valid as robustness checks, the script should document that these coefficients are not directly comparable in magnitude to the baseline.

5. **Weights specification**: The "survey_weighted" spec appears to actually be unweighted (same as baseline). If the intent was to test with survey weights, the implementation may be incorrect.
