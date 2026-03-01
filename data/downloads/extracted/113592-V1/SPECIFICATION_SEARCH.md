# Specification Search: 113592-V1

## Paper
**Fack & Grenet (2015)**: "Improving College Access and Success for Low-Income Students:
Evidence from a Large French Financial Aid Program"
*American Economic Journal: Applied Economics*, 7(2), 1-34.

## Surface Summary
- **Paper ID**: 113592-V1
- **Baseline groups**: 1 (G1: Grant eligibility on college enrollment)
- **Design**: Sharp regression discontinuity
- **Three cutoff samples**: 0_X (no grant vs any grant), 1_0 (L1 vs L0), 6_1 (higher levels)
- **Running variable**: inc_distance (negated income distance from cutoff)
- **Estimator**: Local linear regression (rdrobust, Python)
- **Canonical inference**: Conventional LLR standard errors (nn vce)
- **Budget**: 70 max core specs
- **Seed**: 113592
- **Surface hash**: sha256:51ed790a35126c89e9212486f6cec13b2aeea45f0cca9715ea2120f79139bf97

## Important Note: Simulated Data
The original sample data files (sample_0_X.dta, sample_1_0.dta, sample_6_1.dta) contain
**confidential French administrative data** and are not included in the replication package.
All results are produced using simulated data that matches the paper's variable structure,
approximate sample sizes, and approximate treatment effect magnitudes.

## Execution Summary

### Counts
| Category | Planned | Executed | Success | Failed |
|----------|---------|----------|---------|--------|
| Baseline | 3 | 3 | 3 | 0 |
| Design | 21 | 21 | 21 | 0 |
| RC | 39 | 39 | 39 | 0 |
| **Total** | **63** | **63** | **63** | **0** |
| Inference variants | - | 57 | 57 | 0 |

### Design Variants Executed (per cutoff)
1. **bandwidth/half_baseline** - Half the optimal bandwidth
2. **bandwidth/double_baseline** - Double the optimal bandwidth
3. **bandwidth/fixed_full_bw** - Fixed bandwidth from paper
4. **poly/local_quadratic** - Local quadratic (p=2)
5. **kernel/uniform** - Uniform (rectangular) kernel
6. **kernel/epanechnikov** - Epanechnikov kernel
7. **procedure/robust_bias_corrected** - CCFT robust bias-corrected

### RC Sample Restrictions Executed (per cutoff)
1. Year 2008, 2009, 2010 separately
2. Females only, males only
3. Level 1, 2, 3 students
4. Bac quartile 1, 2, 3, 4
5. Donut hole: exclude |inc_distance| < 0.02

### Inference Variants
- **Canonical**: Conventional LLR standard errors (nn vce from rdrobust)
- **Variant 1**: Robust bias-corrected (CCFT) - run for all baseline and design specs

### What Was Skipped
- Table 5 persistence outcomes (different estimand, would be `explore/*`)
- Table 6 parametric polynomial RD (global polynomial, different design)
- Clustering at discrete income levels (simulated data has continuous income)
- McCrary density test and balance tests (diagnostics plan not executed)

## Software Stack
- Python 3.12.7
- rdrobust: 1.3.0
- pandas: 2.2.3
- numpy: 2.1.3

## Deviations from Surface
1. **Simulated data**: All results use simulated data rather than the confidential administrative
   data from the paper. Treatment effects and sample sizes are calibrated to approximately match
   the paper's reported values.
2. **rdrobust vs rdob_m**: The paper uses a modified version of Imbens's rdob.ado (IK bandwidth).
   We use the Python rdrobust package which implements Calonico-Cattaneo-Titiunik (CCT) bandwidth
   selection by default. This is a more modern implementation that includes bias-correction.
3. **Clustering inference variant skipped**: The paper does not cluster; the surface suggested
   clustering at discrete income levels as a variant, but simulated data has continuous income.
