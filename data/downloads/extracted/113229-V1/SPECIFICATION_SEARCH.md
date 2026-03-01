# Specification Search: 113229-V1

## Paper
- **Title**: Betrayal Aversion: Evidence from Brazil, China, Oman, Switzerland, Turkey, and the United States
- **Authors**: Bohnet, Greig, Herrmann & Zeckhauser (AER 2008)
- **Design**: Randomized experiment (lab)
- **Paper ID**: 113229-V1

## Surface Summary
- **Baseline groups**: 1 (G1: Betrayal Aversion)
- **Budget**: max 60 core specs, 15 control subsets
- **Seed**: 113229
- **Canonical inference**: Cluster SE at session level (CRV1)

## Baseline Group G1
- **Claim**: TG MAP > RDG MAP (betrayal aversion)
- **Outcome**: map (minimum acceptable probability)
- **Treatment**: tg (Trust Game dummy), with dp (Dictator-Principal dummy); RDG is omitted
- **Sample**: movers only (mover==1), N=494
- **Baseline specs**: Table 2 Cols 1-3

## Execution Summary

### Counts
- **Planned core specs**: 65
- **Executed successfully**: 65
- **Failed**: 0
- **Inference variants**: 6 (6 success, 0 failed)

### Spec Breakdown
| Category | Count |
|----------|-------|
| Baselines (Table 2 Cols 1-3) | 3 |
| Design variants | 2 |
| Controls: single additions | 9 |
| Controls: standard sets | 3 |
| Controls: progression | 9 |
| Controls: random subsets | 15 |
| Sample: drop country (with controls) | 6 |
| Sample: drop country (no controls) | 6 |
| Sample: gender subsamples | 2 |
| Sample: outlier trimming | 2 |
| Preprocessing: complete cases | 1 |
| Functional form: logit MAP | 1 |
| Functional form: standardized MAP | 1 |
| Functional form: rank MAP | 1 |
| Treatment isolation: TG vs RDG | 2 |
| Treatment isolation: DP vs RDG | 2 |
| **Total core specs** | **65** |

### Baseline Results
- **baseline**: coef=0.1473, se=0.0365, p=0.0003, N=494, R2=0.0599
- **baseline__table2_col2**: coef=0.1449, se=0.0276, p=0.0000, N=443, R2=0.1422
- **baseline__table2_col3**: coef=0.1433, se=0.0369, p=0.0005, N=443, R2=0.1505

### Inference Variants
- HC1 (heteroskedasticity-robust, no clustering) for all 3 baselines
- HC3/CRV3 (jackknife) for all 3 baselines
- Total: 6 inference recomputations

### Deviations from Surface
- None. All planned axes executed.
- Wild cluster bootstrap not implemented (package not available; noted in surface review as optional).

### Software Stack
- Python 3.12.7
- pyfixest 0.40.1
- pandas 2.2.3
- numpy 2.1.3
- statsmodels 0.14.6
