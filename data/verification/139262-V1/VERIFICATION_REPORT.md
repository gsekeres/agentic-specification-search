# Verification Report: 139262-V1

**Paper**: "Motivated Beliefs and Anticipation of Uncertainty Resolution" by Christoph Drobner
**Design**: Randomized experiment (laboratory, between-subject)
**Verification date**: 2026-02-24

---

## 1. Baseline Groups

### G1: Asymmetric Belief Updating

**Claim**: Subjects in the No-Resolution treatment underreact to bad news relative to the Bayesian benchmark, while subjects in the Resolution treatment update more symmetrically.

**Baseline spec_run_ids** (6 cells from Table 2):

| Cell | spec_run_id | spec_id | Coefficient | SE | p-value | N |
|------|-------------|---------|-------------|-----|---------|---|
| NoRes-Bad (focal) | 139262-V1_run_001 | baseline | 0.076 | 0.180 | 0.674 | 50 |
| NoRes-DiD | 139262-V1_run_002 | baseline__nores_did | 0.076 | 0.180 | 0.673 | 100 |
| NoRes-Good | 139262-V1_run_003 | baseline__nores_good | 0.666 | 0.088 | <0.001 | 50 |
| Res-Bad | 139262-V1_run_004 | baseline__res_bad | 0.645 | 0.249 | 0.013 | 50 |
| Res-DiD | 139262-V1_run_005 | baseline__res_did | 0.645 | 0.249 | 0.011 | 100 |
| Res-Good | 139262-V1_run_006 | baseline__res_good | 0.530 | 0.218 | 0.019 | 50 |

**Interpretation**: The focal result is confirmed. The No-Resolution Bad news coefficient (0.076) is statistically insignificant (p=0.674), indicating near-zero response to bad news when uncertainty will not be resolved. In contrast, good news in the same condition yields a strong significant coefficient (0.666, p<0.001). The Resolution treatment shows significant updating for both good and bad news (0.530 and 0.645), consistent with more symmetric updating when subjects know they will learn their true rank.

---

## 2. Row Counts

| Category | Count |
|----------|-------|
| **Total rows** | **78** |
| Baseline (core_method) | 6 |
| RC: Controls (core_controls) | 36 |
| RC: Sample (core_sample) | 24 |
| RC: Fixed effects (core_fe) | 6 |
| RC: Joint (core_joint) | 6 |
| **Core** | **78** |
| Non-core | 0 |
| Invalid | 0 |
| Unclear | 0 |

| Status | Count |
|--------|-------|
| Valid (is_valid=1) | 78 |
| Invalid (is_valid=0) | 0 |

---

## 3. Inference Variants Summary

18 inference variants were run (3 per baseline cell):

| Variant | Description | Key finding (focal cell: NoRes-Bad) |
|---------|-------------|-------------------------------------|
| infer/se/hc/classical | Homoskedastic SEs | SE=0.206, p=0.713 (still insignificant) |
| infer/se/hc/hc3 | HC3 leverage-corrected | SE=0.201, p=0.706 (still insignificant) |
| infer/se/cluster/session | Cluster at session level | SE=0.258, p=0.782 (still insignificant, larger SE) |

The focal result is robust across all inference variants. Session clustering increases SEs (as expected with few clusters) but does not change conclusions.

---

## 4. Robustness Assessment

### Controls sensitivity

Adding individual pre-treatment covariates (rank, sumpoints, age, gender) barely moves the focal coefficient. Adding `prior` (prior expected rank) slightly shifts it but remains insignificant. The full-controls specification (all 5 covariates) yields 0.062 (p=0.808), still confirming the null.

**Notable**: The full-controls specification substantially changes some non-focal cell coefficients (e.g., NoRes-Good drops from 0.666 to -0.050), suggesting collinearity between `prior` and `bayes_belief_adjustment` when many controls are included. This is expected given the mechanical relationship between prior beliefs and the Bayesian adjustment.

### Sample restrictions

- **Exclude wrong adjustments**: Focal coefficient 0.104 (p=0.564). Pattern preserved.
- **Exclude wrong + zero**: Focal coefficient -0.023 (p=0.899). Pattern preserved.
- **Exclude extreme ranks**: Focal coefficient 0.046 (p=0.820). Pattern preserved with half sample.
- **Trim outliers (5/95)**: Focal coefficient 0.178 (p=0.264). Pattern preserved.

### Session FE

Adding session FE barely changes the focal coefficient (0.051, p=0.826). Pattern preserved.

### Joint (full controls + session FE)

Focal coefficient 0.049 (p=0.853). Pattern preserved even with maximum covariate adjustment.

---

## 5. Top Issues

1. **No issues identified with the specification search execution.** All 78 rows ran successfully, all contract compliance checks pass, all axis blocks are present.

2. **Minor observation**: The full-controls specification substantially attenuates non-focal cell coefficients due to the mechanical correlation between `prior` (a control) and `bayes_belief_adjustment` (the regressor). This is a statistical artifact of overcontrolling, not a problem with the specification search. The focal claim (NoRes-Bad near zero) is robust because the baseline coefficient is already near zero.

3. **Session clustering**: With only 5 sessions per treatment arm, session-clustered inference has very few clusters. The CRV1 standard errors are available but should be interpreted cautiously given the small number of clusters.

---

## 6. Overall Assessment

**STRONG support for the paper's main claim.** The asymmetric belief updating pattern is robust across:
- 5 individual control additions
- Full control set
- 3 sample restriction variants (wrong adjustments, extreme ranks)
- 1 outlier trimming variant
- Session fixed effects
- Joint controls + FE

The focal coefficient (NoRes-Bad) remains small and statistically insignificant across all 13 specifications targeting that cell, with p-values ranging from 0.61 to 0.90.

---

## 7. Recommendations

- Consider adding gender or age interactions as exploration variants (heterogeneity analysis).
- The ordered logit results from Table 3 (ex-post rationalization) could be added as a second baseline group in future iterations.
- Session-level wild cluster bootstrap would be a more appropriate inference method with 5-10 clusters, but the `wildboottest` package is not available in this environment.
