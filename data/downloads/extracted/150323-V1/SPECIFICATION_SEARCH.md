# Specification Search: 150323-V1

## Paper
Akhtari, Moreira & Trucco (2022), "Political Turnover, Bureaucratic Turnover, and the Quality of Public Services", AER.

## Surface Summary
- **Baseline groups**: 3 (G1: test scores, G2: municipal personnel, G3: headmaster/teacher)
- **Design**: Sharp regression discontinuity (piecewise-linear within bandwidth)
- **Running variable**: pX (incumbent party vote margin), cutoff at 0
- **Budgets**: G1=60, G2=30, G3=25 (115 target)
- **Seed**: 150323

## Memory Optimization
- **Student-level data (G1)**: Full dataset (11.5M rows), no subsampling.
  Bandwidth selection via rdrobust uses a 100,000-obs subsample for tractability.
- **School-level and RAIS data (G2, G3)**: Full sample, selective column loading.
- **Non-municipal student data**: Skipped; school-level aggregates used for placebo tests.

## Execution Summary
- **Total specifications planned**: 74
- **Successfully executed**: 72
- **Failed**: 2
- **Inference variants**: 2

### By Group
| Group | Total | Success | Failed |
|-------|-------|---------|--------|
| G1 | 34 | 32 | 2 |
| G2 | 21 | 21 | 0 |
| G3 | 19 | 19 | 0 |

### Coefficient Summaries
- **G1 (test scores)**: Baseline coef: -0.0755, range: [-0.0995, 0.0383], sig(5%): 21/32
- **G2 (personnel)**: Baseline coef: 0.0524, range: [-0.0452, 0.1758], sig(5%): 3/21
- **G3 (headmaster)**: Baseline coef: 0.2785, range: [0.0034, 0.2927], sig(5%): 17/19

## Optimal Bandwidths
- G1 (4th grade combined): 0.07978
- G1 (8th grade combined): 0.05401
- G2 (SHhired_Mun_lead): 0.13322
- G3 (headmaster): 0.08480

## Deviations and Notes
- **Bandwidth subsampling**: rdrobust bandwidth selection uses a 100K-obs subsample (masspoints=off) for computational tractability. Regressions run on full data within the selected bandwidth.
- **Math/Portuguese scores separately (G1)**: Not available in the student-level data. Only combined scores (both_score_indiv_4_stdComb). Specs rc/form/outcome/math_score_only and port_score_only marked as failed.
- **Unstandardized scores (G1)**: Used year-specific standardized scores (std08/std12) as alternative to combined standardization (stdComb).
- **Triangular kernel (G1)**: Implemented via weighted OLS with triangular kernel weights and re-optimized bandwidth.
- **G1 nonmunic placebo**: Used school-level aggregated scores (both_score_4_std) instead of student-level to avoid loading 2.3GB non-municipal student file.
- **G2 after outcomes**: Created from RAIS _10/_14 year suffixes for t+2 personnel outcomes.
- **Teacher outcomes (G3)**: Restricted to schools matched to DOCENTES census in all 4 years (count==4).
- **G2 bandwidth/rc overlap**: Some bandwidth specs appear in both design/ and rc/ prefixes; these are kept as the surface specifies both.

## Software Stack
- Python 3.12.7
- pyfixest 0.40.1
- pandas 2.2.3
- numpy 2.1.3
- rdrobust 1.3.0

Surface hash: `sha256:7f554eb66334d58d7a8b04d8a58d4028cddb072b524ed3102bf8013828e84ac5`
