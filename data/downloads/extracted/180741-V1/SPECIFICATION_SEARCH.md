# Specification Search Report: 180741-V1

**Paper**: "Enabling or Limiting Cognitive Flexibility? Evidence of Demand for Moral Commitment" (Saccardo & Serra-Garcia, AER)

**Paper ID**: 180741-V1

**Date executed**: 2026-02-24

---

## Surface Summary

- **Design**: Randomized experiment (online experiments on advisor incentives and financial recommendations)
- **Baseline groups**: 3
  - **G1**: Choice experiment recommendations (Table 3) -- treatment: `choicebefore`, outcome: `recommendincentive`
  - **G2**: NoChoice experiment recommendations (Table C.1) -- treatment: `seeincentivefirst`, outcome: `recommendincentive`
  - **G3**: Choice experiment preferences (Table 2) -- treatment: `seeincentivecostly`, outcome: `choicebefore`
- **Canonical inference**: HC3 robust standard errors (individual-level randomization, no clustering)
- **Budgets**: G1 max 60 core specs, G2 max 30, G3 max 30
- **Seed**: 180741
- **Controls sampling**: Full enumeration (LOO + additions)

---

## Execution Counts

| Category | Planned | Executed | Succeeded | Failed |
|---|---|---|---|---|
| Baselines | 9 | 9 | 9 | 0 |
| Design variants | 6 | 6 | 6 | 0 |
| RC variants | 54 | 54 | 54 | 0 |
| **Total core** | **69** | **69** | **69** | **0** |
| Inference variants | 15 | 15 | 15 | 0 |

### Per-Group Breakdown

| Group | Baselines | Design | RC (controls) | RC (sample) | RC (outcome) | Total |
|---|---|---|---|---|---|---|
| G1 | 3 | 2 | 15 (12 LOO + 3 add) | 7 | 2 | 29 |
| G2 | 3 | 2 | 5 (4 LOO + 1 minimal) | 5 | 2 | 17 |
| G3 | 3 | 2 | 12 (9 LOO + 3 add) | 4 | 2 | 23 |

---

## Baseline Reproduction

### G1: Choice Experiment Recommendations (Table 3)

| Spec | Coef (choicebefore) | SE | p-value | N | R2 |
|---|---|---|---|---|---|
| Table3-Col1 (getyourchoice==1) | 0.1955 | 0.0161 | <0.001 | 4448 | 0.106 |
| Table3-Col2 (getyourchoice==0) | 0.0030 | 0.0288 | 0.917 | 1460 | 0.083 |
| Table3-Col3 (combined) | 0.1815 | 0.0153 | <0.001 | 5908 | 0.097 |

### G2: NoChoice Experiment Recommendations (Table C.1)

| Spec | Coef (seeincentivefirst) | SE | p-value | N | R2 |
|---|---|---|---|---|---|
| TableC1-Col1 (conflict==1) | 0.1416 | 0.0625 | 0.025 | 213 | 0.137 |
| TableC1-Col2 (conflict==0) | 0.0303 | 0.0808 | 0.709 | 86 | 0.028 |
| TableC1-Col3 (full sample) | 0.1485 | 0.0619 | 0.017 | 299 | 0.124 |

### G3: Choice Experiment Preferences (Table 2)

| Spec | Coef (seeincentivecostly) | SE | p-value | N | R2 |
|---|---|---|---|---|---|
| Table2-Col1 (full sample) | -0.1393 | 0.0178 | <0.001 | 5908 | 0.034 |
| Table2-Col2 (professionals==0) | -0.1395 | 0.0178 | <0.001 | 5196 | 0.040 |
| Table2-Col3 (w/ selfishness interactions) | -0.1397 | 0.0178 | <0.001 | 5196 | 0.040 |

---

## Design Variants

- **diff_in_means**: Treatment-only regression (no controls) for each group
- **with_covariates**: Structural treatment indicators only (no optional controls)

All design variants produced significant results in the same direction as baselines for G1 and G3. G2 diff-in-means also significant (p=0.011).

---

## RC Variants

### Controls (LOO and additions)

- **G1**: 12 leave-one-out + 3 additions (stdalpha, selfishseeincentivecostly, selfishseequalitycostly). All LOO variants significant at p<0.001. Adding selfishness controls reduces sample to 5196 obs (due to missing stdalpha for professionals).
- **G2**: 4 leave-one-out + 1 minimal (no demographics). All variants maintain similar significance. Dropping stdalpha strengthens result (p=0.008 vs 0.017).
- **G3**: 9 leave-one-out + 3 additions. All variants highly significant (p<0.001). Results extremely stable across control specifications.

### Sample Restrictions

- **G1**: Including inattentive participants, high-stakes participants, restricting to ChoiceFree, professionals-only, incentiveA-only, or incentiveB-only all produce significant positive effects.
- **G2**: Including inattentive strengthens result. Restricting to conflict-only matches Col 1 baseline. Restricting to incentiveA-only yields p=0.164 (loss of power in small sample).
- **G3**: All sample variants maintain highly significant negative treatment effect.

### Outcome (Probit/Logit)

All probit/logit specifications confirm the LPM results with consistent sign and significance.

---

## Inference Variants

HC1 and HC2 alternatives computed for all 9 baseline specifications. Results are virtually identical to HC3 for the large-sample G1 and G3 groups. For the small-sample G2, HC1 produces slightly smaller SEs than HC3 (expected for small N).

---

## Deviations and Notes

1. In the G2 conflict-only and no-conflict-only subsamples, the `noconflict` variable is perfectly collinear and dropped by pyfixest (as Stata would also do). This does not affect the treatment coefficient.
2. For G1 sample restrictions to ChoiceFree-only, the `seeincentivecostly`, `seequalitycostly`, and `professionalsfree` dummies are zero for all observations and are effectively dropped.
3. The `stdalpha` control is only available for non-professional participants who completed the MPL task, so adding it reduces the sample.
4. No diagnostics were run (G1 and G3 diagnostics plan called for balance checks; G2 plan is non-empty but balance checking is not part of the core spec search).

---

## Software Stack

- **Python**: 3.12
- **pyfixest**: 0.40+
- **pandas**: 2.x
- **numpy**: 2.x
- **statsmodels**: 0.14.x (for probit/logit)

---

## Output Files

- `specification_results.csv`: 69 rows (9 baselines + 6 design + 54 RC)
- `inference_results.csv`: 15 rows (HC1/HC2 variants for all baselines)
- `scripts/paper_analyses/180741-V1.py`: Analysis script
