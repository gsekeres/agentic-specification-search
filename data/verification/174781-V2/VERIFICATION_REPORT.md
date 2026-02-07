# Verification Report: 174781-V2

## Paper
**Title**: Work and Mental Health among Rohingya Refugees  
**Journal**: AER  
**Method**: Cross-sectional OLS (RCT)  
**Core hypothesis**: Cash-for-work programs improve mental health among Rohingya refugees more than equivalent unconditional cash transfers, due to non-pecuniary benefits of employment.

---

## Baseline Groups

### G1: Composite Mental Health Index (Primary)
- **Baseline spec_id**: `baseline`
- **Outcome**: `mental_health_index` (composite of PHQ depression, stress, life satisfaction, sociability, self-worth, perceived control, stability)
- **Treatment**: `b_treat_work` (cash-for-work assignment)
- **Expected sign**: Positive
- **Baseline result**: coef = 0.211, SE = 0.042, p < 0.001

### G2: Individual Mental Health Sub-Indices (Secondary)
- **Baseline spec_ids**: `baseline/phq_sd_scale`, `baseline/stress_index`, `baseline/life_satisfaction`, `baseline/sociability_a`, `baseline/selfworth_index`, `baseline/control_index`, `baseline/stability`
- **Outcomes**: Seven sub-components of the mental health index
- **Treatment**: `b_treat_work`
- **Expected sign**: Positive (higher = better mental health)
- **Note**: Not all sub-indices are individually significant (selfworth_index p=0.108)

---

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **78** |
| Baseline specs | 8 |
| Core test specs (incl. baselines) | 49 |
| Non-core specs | 29 |
| Invalid specs | 0 |
| Unclear specs | 0 |

### Core Test Breakdown

| Core Category | Count |
|---------------|-------|
| core_controls | 30 |
| core_sample | 12 |
| core_inference | 3 |
| core_fe | 3 |
| core_funcform | 1 |
| core_method | 0 |

### Non-Core Breakdown

| Non-Core Category | Count |
|-------------------|-------|
| noncore_alt_outcome | 10 |
| noncore_placebo | 8 |
| noncore_heterogeneity | 8 |
| noncore_alt_treatment | 3 |
| noncore_diagnostic | 0 |

---

## Classification Rationale

### Core tests (49 specs, including 8 baselines)
These are specifications that test the same causal claim (effect of work treatment on mental health) with variations in:
- **Controls** (30, incl. 8 baselines): Build-up progressions, leave-one-out, full control sets for each outcome, and the 8 baselines themselves. These preserve the estimand while varying conditioning.
- **Sample** (12): Camp-specific, gender, age, education, and marital status subsamples, plus trimming and one treatment-arm sample restriction (work_vs_control). These test the same claim on subpopulations or with outlier handling.
- **Inference** (3): Alternative clustering (none, camp, enumerator). Same point estimates with different standard errors.
- **Fixed effects** (3): No FE, camp-only FE, enumerator-only FE. These vary the conditioning set while preserving the estimand.
- **Functional form** (1): Winsorized outcome (mental_health_index_w1). Same construct, different tail treatment.

### Non-core: Alternative outcomes (10 specs)
Individual PHQ depression items (phq_1 through phq_9) and a binary depression indicator. These use individual questionnaire items rather than the composite index, measure a narrower construct, and have reversed sign direction (negative = improvement). They are informative about mechanisms but do not test the same claim as G1 or G2.

### Non-core: Placebo tests (8 specs)
These regress treatment on **baseline** (pre-treatment) outcomes to check randomization balance. They are diagnostic tests, not tests of the core claim.

### Non-core: Heterogeneity (8 specs)
Interaction terms (work x gender, age, education, marriage, violence, baseline MH) and baseline-outcome splits (high/low). These test whether the treatment effect varies by subgroup, not whether it exists. The main effects in interaction models are not directly comparable to baseline because the coefficient represents the effect at the reference level of the interacted variable.

### Non-core: Alternative treatments (3 specs)
- `cash_vs_control`: Uses `b_treat_largecash` as treatment -- different causal object.
- `work_vs_cash`: Excludes control group, comparing work to cash arm -- different estimand.
- `any_vs_control`: Uses `any_treatment` pooling work and cash -- different causal object.

### Borderline: work_vs_control (classified as core_sample)
This spec excludes the large cash arm from the sample but keeps the same treatment variable (`b_treat_work`) and outcome. It is effectively a sample restriction, so it is classified as core_sample with slightly lower confidence (0.85).

---

## Top 5 Most Suspicious Rows

1. **robust/het/interaction_age** (spec_id): The main effect coefficient is 0.059 (p=0.75), far from the baseline 0.211. This is because the main effect now represents the treatment effect at age=0, which is extrapolation. The spec is correctly classified as non-core heterogeneity, but a naive comparison would be misleading.

2. **robust/het/interaction_married** (spec_id): Main effect coefficient is 0.092 (p=0.28), representing effect for unmarried. Same extrapolation issue as age interaction.

3. **robust/placebo/outcome_predetermined** (spec_id): Shows a significant effect (coef=0.094, p=0.043) of treatment on baseline mental health, suggesting possible randomization imbalance. While marginally significant, this is expected in multiple testing contexts.

4. **robust/placebo/baseline_stability** (spec_id): Shows a significant effect (coef=0.195, p=0.012) on baseline stability. More concerning balance issue than the predetermined outcome placebo.

5. **robust/placebo/baseline_control_index** (spec_id): Shows a marginally significant effect (coef=0.169, p=0.039) on baseline perceived control. Another balance concern.

---

## Recommendations

1. **Placebo significance**: Two of eight placebo tests are significant at 5% (stability p=0.012, control_index p=0.039) and a third is marginal (predetermined outcome p=0.043). The spec search script should flag this pattern more prominently, as 3/8 significant placebos exceeds the expected rate under randomization.

2. **Heterogeneity interaction main effects**: The interaction specs report the main effect of treatment (at zero of the interacted variable), which is not comparable to the baseline. The spec search script could be improved to either (a) report the average marginal effect or (b) more clearly label that these coefficients are conditional on the interaction variable.

3. **Treatment variation clarity**: The treatment variation specs (cash_vs_control, work_vs_cash, any_vs_control) change either the treatment variable or the comparison group. The script correctly tags these under robust/treatment/ but they could be more clearly flagged as testing a different estimand.

4. **Individual PHQ items**: These are placed under robust/outcome/ which is appropriate, but the sign reversal (negative coefficients = improvement) could cause confusion in automated aggregation. A note or separate sign convention field would help.

5. **No issues with baseline specification**: The baseline correctly implements the paper's main specification with camp and enumerator FE, baseline outcome and gender controls, block-level clustering. No corrections needed.
