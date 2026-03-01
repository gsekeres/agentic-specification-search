# Specification Search Report: 139262-V1

**Paper**: "Motivated Beliefs and Anticipation of Uncertainty Resolution" by Christoph Drobner
**Design**: Randomized experiment (laboratory, between-subject)
**Run date**: 2026-02-24

---

## Surface Summary

- **Baseline groups**: 1 (G1: belief updating asymmetry)
- **Baseline cells**: 6 (NoRes-Bad, NoRes-DiD, NoRes-Good, Res-Bad, Res-DiD, Res-Good)
- **Budget**: 80 specs (core total)
- **Seed**: 139262 (full enumeration, no sampling needed)
- **Canonical inference**: HC1 (robust) standard errors

---

## Execution Summary

### Counts

| Category | Planned | Executed | Succeeded | Failed |
|----------|---------|----------|-----------|--------|
| Baseline | 6 | 6 | 6 | 0 |
| RC (controls single-add) | 30 | 30 | 30 | 0 |
| RC (controls full) | 6 | 6 | 6 | 0 |
| RC (sample quality) | 18 | 18 | 18 | 0 |
| RC (sample outliers) | 6 | 6 | 6 | 0 |
| RC (session FE) | 6 | 6 | 6 | 0 |
| RC (joint) | 6 | 6 | 6 | 0 |
| **Total core** | **78** | **78** | **78** | **0** |
| Inference variants | 18 | 18 | 18 | 0 |

### Key Results

**Focal cell (No-Resolution, Bad news)**:
- Baseline coefficient: 0.0762 (SE=0.1798, p=0.6738, N=50)

**Summary across all cells**:
- nores_bad: coef=0.0762, SE=0.1798, p=0.6738, N=50
- nores_did: coef=0.0762, SE=0.1798, p=0.6728, N=100
- nores_good: coef=0.6655, SE=0.0881, p=0.0000, N=50
- res_bad: coef=0.6448, SE=0.2492, p=0.0128, N=50
- res_did: coef=0.6448, SE=0.2492, p=0.0112, N=100
- res_good: coef=0.5296, SE=0.2181, p=0.0190, N=50

---

## Software Stack

- Python 3.12.7
- pyfixest: 0.40.1
- pandas: 2.2.3
- numpy: 2.1.3
- statsmodels: 0.14.6

---

## Deviations from Surface

- The `design/randomized_experiment/estimator/diff_in_means` and `design/randomized_experiment/estimator/with_covariates` design variants were not run as separate `design/*` rows because: (a) diff_in_means for belief adjustments is equivalent to the OLS baseline with only `bayes_belief_adjustment` as the regressor (the baseline already IS the simple regression), and (b) with_covariates is covered by the RC controls variants (single-add and full controls).
- Added `rc/joint/controls_fe/full_plus_session` as a joint axis combining full controls with session FE.

---

## Notes

- Data was constructed from raw Excel files following `data_creation.do` exactly.
- Bayesian posterior beliefs were computed using the signal structure described in the paper.
- The "wrong direction" belief adjustments flag matches the paper's definition.
- Session FE uses string-encoded session variable for categorical absorption.
