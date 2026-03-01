# Specification Search Log: 114542-V1

**Paper**: Cattaneo, Galiani, Gertler, Martinez & Titiunik (2009). "Housing, Health and Happiness."
**Date**: 2026-02-24

---

## Surface Summary

- **Paper ID**: 114542-V1
- **Design**: Randomized Experiment (Piso Firme program)
- **Baseline groups**: 3 (G1: cement floors, G2: satisfaction/mental health, G3: child health)
- **Surface hash**: sha256:2106c0229bcc5cb674b0d5bdbc34b383d36a7d1ad79a75b7d087a730579d3968
- **Seed**: 114542

## Execution Summary

### Total Counts
- **Planned specifications**: 95
- **Executed successfully**: 95
- **Failed**: 0
- **Inference variants**: 4

### By Baseline Group

#### G1: Cement Floor Coverage (Household-Level)
- Primary outcome: S_shcementfloor
- Baselines: 4
- Design variants: 1
- RC specs: 30
- Inference variants: 2

#### G2: Satisfaction and Mental Health (Household-Level)
- Primary outcome: S_satisfloor
- Additional outcomes: S_satishouse, S_satislife, S_cesds, S_pss
- Baselines: 5
- RC specs: 25
- Inference variants: 1

#### G3: Child Health (Individual-Level)
- Primary outcome: S_parcount
- Additional outcomes: S_diarrhea, S_anemia, S_haz, S_whz
- Baselines: 5
- RC specs: 25
- Inference variants: 1

## Deviations and Notes

1. **Missing value imputation**: Exactly replicates Stata code -- missing control values replaced with 0 plus indicator dummies for missingness.
2. **Clustering**: All specifications use CRV1 at `idcluster` (census block) level, matching the paper's `cl(idcluster)`.
3. **dtriage dummies**: Individual-level regressions (G3) include trimester-age-gender dummy variables as in the original Stata code.
4. **Cognitive outcomes excluded**: S_mccdts (N=601) and S_pbdypct (N=1,589) have very high missingness and were excluded from the specification surface.
5. **idmun clustering**: Only computed for G1 baseline as an inference variant. Note that with only ~2 municipalities, inference is unreliable.

## Software Stack

- Python 3.12.7
- pyfixest 0.40.1
- pandas 2.2.3
- numpy 2.1.3
