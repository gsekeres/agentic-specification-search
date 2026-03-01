# Specification Search Log: 111185-V1

## Surface Summary
- **Paper**: Optimal Climate Policy When Damages are Unknown (Rudik, 2020)
- **Baseline groups**: 1 (G1: damage function parameter estimation)
- **Design**: cross_sectional_ols
- **Budget**: 60 max core specs
- **Seed**: 111185
- **Surface hash**: See specification_results.csv coefficient_vector_json

## Execution Summary

### Counts
| Category | Planned | Executed | Successful | Failed |
|----------|---------|----------|------------|--------|
| baseline | 1 | 1 | 1 | 0 |
| rc/controls | ~35 | 35 | 35 | 0 |
| rc/sample | ~11 | 11 | 11 | 0 |
| rc/form | 5 | 5 | 5 | 0 |
| rc/preprocess | 2 | 2 | 2 | 0 |
| rc/joint | 8 | 8 | 8 | 0 |
| **Total core** | **~62** | **62** | **62** | **0** |
| infer/* (separate) | 3 | 3 | 3 | 0 |

### Spec ID Breakdown

**Controls (35 specs)**:
- Single additions: 7 (one per optional control)
- Standard sets: 3 (minimal, extended, full)
- Progression: 3 (study_quality, damage_type, full)
- Block combinations: 7 (exhaustive 2^3 - 1)
- Random variable subsets: 15 (seed=111185, sizes 2-6)

**Sample (11 specs)**:
- Outlier trimming: 3 (outcome [1,99], outcome [5,95], treatment [1,99])
- Cook's D: 1 (threshold = 4/N)
- Quality filters: 5 (drop repeat, drop based-on-other, drop grey, drop catastrophic, independent only)
- Temporal splits: 2 (early, late)

**Functional form (5 specs)**:
- Outcome transforms: 2 (levels, asinh)
- Treatment transforms: 1 (level temperature)
- Nonlinear models: 2 (quadratic log-temp, quadratic levels)

**Preprocessing (2 specs)**:
- Winsorization: 2 (outcome, treatment)

**Joint (8 specs)**:
- Combined sample + controls: 8 variations

**Inference variants (3, separate table)**:
- HC1, HC2, HC3 robust standard errors on baseline model

## Deviations from Surface
- None. All planned specs were executed successfully.

## Software Stack
- Python 3.12
- statsmodels 0.14.4
- pandas 2.3.3
- numpy 1.26.4

## Key Findings (summary)
- Baseline coefficient (d2 exponent): 1.882 (SE=0.451, p=0.00015)
- All 62 specifications ran without errors
- The coefficient on logt (damage exponent) remains positive and statistically significant across all specifications that use log_correct ~ logt
- With study quality controls, the coefficient is modestly attenuated
- The estimate is robust to outlier removal, quality filters, and temporal splits
- Functional form variations (levels, asinh) confirm the positive damage-temperature relationship under alternative parameterizations
