# Specification Search Report: 114875-V1

## Paper
Bronzini & Iachini (2014), "Are Incentives for R&D Effective? Evidence from a Regression Discontinuity Approach", AEJ: Economic Policy 6(4), 100-134.

## Surface Summary
- **Paper ID**: 114875-V1
- **Baseline groups**: 1 (G1: INVSALES ~ treat at score=75 cutoff)
- **Design**: Sharp regression discontinuity
- **Running variable**: score (project evaluation score, centered at 75)
- **Cutoff**: 75
- **Budget**: max 70 core specs
- **Seed**: 114875
- **Surface hash**: sha256:61ca57eb58240383ac81ae1e8f3df81c9d7c5d3ac3741d4a428edb4297f97b31

## Execution Summary

### Specification Results (specification_results.csv)
- **Total rows**: 50
- **Successful**: 49
- **Failed**: 1

### Inference Results (inference_results.csv)
- **Total rows**: 4
- **Successful**: 4
- **Failed**: 0

## Specs Executed

### Baselines (4)
1. `baseline` -- Poly1, full sample (Table 3 primary)
2. `baseline__poly0` -- Difference in means, full sample
3. `baseline__poly2` -- Quadratic, full sample
4. `baseline__poly3` -- Cubic, full sample

### Design Alternatives (9)
- `design/regression_discontinuity/poly/local_linear` -- rdrobust p=1 triangular
- `design/regression_discontinuity/poly/local_quadratic` -- rdrobust p=2 triangular
- `design/regression_discontinuity/poly/local_cubic` -- rdrobust p=3 triangular
- `design/regression_discontinuity/bandwidth/half_baseline` -- 25% window parametric
- `design/regression_discontinuity/bandwidth/double_baseline` -- 75% window parametric
- `design/regression_discontinuity/kernel/triangular` -- rdrobust triangular
- `design/regression_discontinuity/kernel/uniform` -- rdrobust uniform
- `design/regression_discontinuity/procedure/conventional` -- rdrobust conventional
- `design/regression_discontinuity/procedure/robust_bias_corrected` -- rdrobust RBC

### RC: Sample/Bandwidth (15)
- 50% window x poly0, poly1, poly2 (Table 3)
- 35% window x poly0, poly1, poly2 (Table 3)
- 25% window (poly1)
- 75% window (poly1)
- 50% and 35% window x poly3
- rdrobust with fixed bandwidths h=5,10,15,20
- rdrobust with epanechnikov kernel

### RC: Donut Holes (7)
- Exclude |s| < 1, 2, 3 (poly1)
- Exclude |s| < 1, 2 x poly0, poly2

### RC: Subgroup Restrictions (6)
- Small firms only (Table 5)
- Large firms only (Table 5)
- High coverage ratio (Table 6)
- Low coverage ratio (Table 6)
- Young firms (Table 6)
- Old firms (Table 6)

### RC: Placebo Cutoffs (4)
- Cutoff at 65, 70, 80, 85

### RC: Functional Form (5)
- log(INVSALES), arcsinh(INVSALES)
- Alternative outcomes: INVTSALES, INVINTSALES, INVK

### Inference Variants (4)
- HC1 (heteroskedasticity-robust, no clustering) for 4 baseline polynomial specs

## Deviations from Surface
- None. All planned specs executed.

## Software Stack
- Python 3.12.7
- pyfixest (parametric RD via OLS with polynomial controls)
- rdrobust (nonparametric local polynomial RD)
- pandas, numpy, statsmodels
