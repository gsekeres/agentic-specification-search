# Verification Report: 184341-V2

**Paper**: "Emotional and Behavioral Impacts of Homeschooling Support on Children"
**Journal**: AER: Papers & Proceedings
**Paper ID**: 184341-V2
**Verified**: 2026-02-04
**Verifier**: verification_agent

---

## 1. Baseline Groups Found

This paper is a randomized controlled trial evaluating a telementoring intervention on children's behavioral/emotional outcomes measured via the Strengths and Difficulties Questionnaire (SDQ). The specification search identified 10 baseline specifications corresponding to 10 distinct baseline groups:

| Group | Baseline spec_id | Outcome | Description |
|-------|------------------|---------|-------------|
| G1 | baseline/sdq_totdiff_e1 | sdq_totdiff_e1 | Total difficulties, Endline 1 (primary) |
| G2 | baseline/sdq_totdiff_e2 | sdq_totdiff_e2 | Total difficulties, Endline 2 |
| G3 | baseline/sdq_emotion_e1 | sdq_emotion_e1 | Emotional symptoms, Endline 1 |
| G4 | baseline/sdq_conduct_e1 | sdq_conduct_e1 | Conduct problems, Endline 1 |
| G5 | baseline/sdq_hyper_e1 | sdq_hyper_e1 | Hyperactivity, Endline 1 |
| G6 | baseline/sdq_peer_e1 | sdq_peer_e1 | Peer problems, Endline 1 |
| G7 | baseline/sdq_emotion_e2 | sdq_emotion_e2 | Emotional symptoms, Endline 2 |
| G8 | baseline/sdq_conduct_e2 | sdq_conduct_e2 | Conduct problems, Endline 2 |
| G9 | baseline/sdq_hyper_e2 | sdq_hyper_e2 | Hyperactivity, Endline 2 |
| G10 | baseline/sdq_peer_e2 | sdq_peer_e2 | Peer problems, Endline 2 |

**Notes on baseline structure**:
- G1 is the paper's primary claim and receives the vast majority of robustness checks (42 core specs).
- G2 receives 6 robustness specs (E2 variants of total difficulties).
- G3-G6 each receive 3 robustness specs (no-controls + gender splits).
- G7-G10 are standalone baselines with no corresponding robustness specs.
- The 10 baselines represent genuinely distinct claims (different outcome variables and/or different time periods), so they should not be merged.

---

## 2. Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | 78 |
| **Baselines** | 10 |
| **Core test specifications** | 61 |
| **Non-core specifications** | 7 |
| **Invalid specifications** | 0 |
| **Unclear specifications** | 0 |

---

## 3. Category Counts

| Category | Count | Description |
|----------|-------|-------------|
| core_controls | 34 | Leave-one-out, control progression, no-controls variants |
| core_sample | 25 | Gender splits, age splits, income splits, religion splits, winsorizing, trimming |
| core_inference | 4 | Classical SE, HC2, HC3, clustered SE |
| core_fe | 4 | No FE, grade-only FE, union-only FE, E2 no FE |
| core_funcform | 4 | Standardized, log, arcsinh, binary |
| noncore_heterogeneity | 5 | Treatment x covariate interactions |
| noncore_placebo | 2 | Treatment on pre-treatment outcomes |

---

## 4. Classification Rationale

### Core tests (61 specs)
All core tests preserve the fundamental estimand: the average treatment effect of telementoring on an SDQ outcome in a given time period. They vary the specification along permissible dimensions:
- **Controls**: Leave-one-out drops, incremental addition, no controls
- **Sample**: Gender, age, income, religion subgroups; winsorizing and trimming
- **Inference**: Alternative standard error estimators (classical, HC2, HC3, clustered)
- **Fixed effects**: Dropping or using subsets of FE (grade only, union only, none)
- **Functional form**: Log/asinh transformations, standardization, binarization

### Non-core: Heterogeneity (5 specs)
The 5 interaction specifications (robust/het/interaction_*) add treatment-by-covariate interaction terms. These change the interpretation of the treatment coefficient from an average treatment effect to a conditional ATE (the effect at the interacted covariate = 0). This is a different estimand and is primarily a test of heterogeneity rather than a robustness check of the baseline claim. Classified as non-core.

### Non-core: Placebo (2 specs)
The 2 placebo specs test treatment effects on pre-treatment outcomes (baseline literacy and numeracy scores). These are validation checks of the randomization, not tests of the treatment effect claim. Correctly identified as placebos.

---

## 5. Top 5 Most Suspicious Rows

1. **robust/het/interaction_age** (spec_id): The treatment coefficient is -8.81 (vs baseline -1.27). This is because the coefficient represents the treatment effect when child_age = 0, which is extrapolation outside the data range (children are 4-6 years old). The coefficient is not directly comparable to the baseline. This is correctly classified as non-core heterogeneity.

2. **robust/het/interaction_baseline_literacy** (spec_id): Similarly, the treatment coefficient is -3.88 because it represents the effect when baseline literacy = 0. Not comparable to baseline ATE.

3. **robust/het/interaction_baseline_numeracy** (spec_id): Treatment coefficient is -3.74 for same reason as above (effect at numeracy = 0).

4. **robust/control/add_reli_dummy** (spec_id): This is the final step of the control progression and is exactly identical to the baseline specification (same coefficient -1.267, same SE). It is a correct but redundant core test.

5. **robust/sample/female_only** (spec_id): The effect is -0.78 with p = 0.094, which is not significant at 5%. This could indicate heterogeneity rather than pure robustness, but since it uses the same outcome/treatment/model with only a sample restriction, it is correctly classified as core_sample.

---

## 6. Recommendations for the Spec-Search Script

1. **Reduce redundancy in control progression**: The `robust/control/add_reli_dummy` specification is identical to the baseline (all controls present). This is a minor issue but could be flagged or de-duplicated.

2. **Heterogeneity coefficient interpretation**: The heterogeneity specs report the main treatment coefficient, not the interaction term. When an interaction is present, the treatment coefficient becomes the conditional effect at the interacted variable = 0, which can produce misleading coefficient magnitudes (e.g., -8.81 for treatment x age). The script could instead report the average marginal effect or note that this coefficient is not directly comparable.

3. **E2 subscale robustness**: The script includes E2 subscale baselines (G7-G10) but does not run any robustness checks on them. If these baselines are important, robustness specs should be added. If they are not important, they could be excluded to avoid cluttering the baseline count.

4. **Missing E1 subscale robustness depth**: The subscale robustness checks only include no-controls and gender splits (3 specs each). Consider adding control progression, inference variation, and functional form checks for the subscales if they are considered important claims.

5. **Placebo tests are well-constructed**: The two placebo tests correctly use pre-treatment outcomes and confirm randomization balance. No changes needed.

---

## 7. Data Quality

- All 78 rows have valid, finite coefficients and standard errors.
- No missing p-values detected.
- Treatment variable is consistently `treat` across all specifications.
- Outcome variables are consistently named and clearly identifiable.
- The specification search is well-structured and covers the standard robustness dimensions for an RCT.
