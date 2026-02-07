# Verification Report: 180741-V1

## Paper
**Title**: Enabling or Limiting Cognitive Flexibility? Evidence of Demand for Moral Commitment  
**Journal**: AER: Insights (2023)  
**Method**: Cross-sectional OLS (Linear Probability Models)

---

## Baseline Groups

### G1: Main Moral Commitment Effect (Choice Experiment)
- **Claim**: Advisors who prefer to see the incentive first (choicebefore=1) are more likely to recommend the incentivized product (recommendincentive=1) when assigned their preferred information order.
- **Baseline spec_ids**: baseline (N=4448, assigned preferences), baseline_combined (N=5908, all with interactions)
- **Expected sign**: Positive
- **Baseline coefficient**: 0.195 (baseline), 0.181 (baseline_combined)

### G2: NoChoice Experiment (Random Assignment)
- **Claim**: In the NoChoice experiment, seeing the incentive first (seeincentivefirst=1) increases the probability of recommending the incentivized product.
- **Baseline spec_ids**: nochoice/baseline_conflict (N=213), nochoice/baseline_combined (N=299)
- **Expected sign**: Positive
- **Baseline coefficient**: 0.142 (conflict only), 0.148 (combined)

### G3: Demand for Moral Commitment
- **Claim**: The costly treatment (seeincentivecostly) negatively predicts demand for moral commitment (choicebefore).
- **Baseline spec_ids**: demand/baseline (N=5908)
- **Expected sign**: Negative
- **Baseline coefficient**: -0.139

---

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **58** |
| Core test specs | 41 |
| Non-core specs | 17 |
| Invalid specs | 0 |
| Unclear specs | 0 |

### Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 18 |
| core_sample | 18 |
| core_inference | 3 |
| core_funcform | 2 |
| core_fe | 0 |
| core_method | 0 |
| noncore_placebo | 2 |
| noncore_alt_outcome | 2 |
| noncore_alt_treatment | 4 |
| noncore_heterogeneity | 5 |
| noncore_diagnostic | 4 |
| invalid | 0 |
| unclear | 0 |

### Baseline specs: 7
- G1: baseline, baseline_combined
- G2: nochoice/baseline_conflict, nochoice/baseline_combined
- G3: demand/baseline

---

## Top 5 Most Suspicious Rows

1. **robust/treatment/getbefore** (confidence=0.7): This spec reports treatment_var=getbefore but has the **exact same coefficient** (0.1955) as the baseline spec. In the assigned-preferences subsample, choicebefore and getbefore are identical by construction (participants who chose "before" were assigned "before"). This is tautological, not a meaningful robustness check. Classified as noncore_alt_treatment.

2. **robust/sample/mturk_only and robust/sample/choicefree_only** (confidence=0.95): These two specs have **identical coefficients** (0.2339, N=1931). They appear to be duplicates -- the MTurk sample IS the ChoiceFree treatment sample. This is not an error per se, but inflates the spec count. Both kept as core_sample since each represents a legitimate subsample restriction.

3. **robust/placebo/noconflict_effect** (confidence=0.9): This spec has the **exact same coefficient** (0.060) as robust/sample/noconflict_only but is labeled as a placebo test. The spec_tree_path says "placebo_tests.md" while noconflict_only is under "sample_restrictions.md". Classified as noncore_placebo since the no-conflict condition is where the paper does NOT expect a strong effect (no conflict of interest to resolve).

4. **demand/with_selfishness** (confidence=0.85): This spec changes the treatment (from seeincentivecostly to stdalpha/selfishness) and has spec_tree_path suggesting it is a baseline. However, it tests a fundamentally different causal object -- whether selfishness predicts demand for moral commitment. Classified as noncore_alt_treatment for G3.

5. **robust/treatment/interaction_pref_assign** (confidence=0.8): The treatment variable is choicebefore_X_getbefore, an interaction term. The coefficient (0.051) and interpretation differ from the baseline main effect. This tests whether the combination of preference and assignment matters, which is a different estimand from the main effect of preference alone.

---

## Recommendations for Spec-Search Script

1. **De-duplicate identical specs**: robust/sample/mturk_only and robust/sample/choicefree_only produce identical results. The script should detect and flag when two sample restrictions yield the same N and coefficient.

2. **Flag tautological treatment swaps**: robust/treatment/getbefore is tautologically identical to baseline in the assigned-preferences subsample. The script should check whether a "treatment alternative" produces the exact same coefficient as the baseline before including it.

3. **Separate experiments should have separate baseline groups**: The NoChoice experiment (G2) and demand analysis (G3) are distinct from the main Choice experiment (G1). The spec-search script correctly identifies them with different spec_id prefixes, but the SPECIFICATION_SEARCH.md summary mixes all 58 specs in a single robustness assessment. The summary statistics should be computed per baseline group.

4. **Placebo vs sample restriction overlap**: robust/placebo/noconflict_effect duplicates robust/sample/noconflict_only. The script should not produce the same regression under two different categories.

5. **Heterogeneity specs report interaction coefficients, not main effects**: All heterogeneity specs correctly report the interaction term coefficient, but these should be explicitly excluded from the main robustness summary since they test a different quantity (differential effect, not the average effect).
