# Specification Search: 112517-V1

## Paper
Fowlie, Holland & Mansur (2012), "What Do Emissions Markets Deliver and to Whom?
Evidence from Southern California's NOx Trading Program", AER 102(2).

## Surface Summary
- **Paper ID**: 112517-V1
- **Baseline groups**: 1 (G1: Emissions Reduction Effect)
- **Design**: Difference-in-differences via nearest-neighbor matching
- **Budget**: 80 max core specs
- **Seed**: 112517

## Execution Summary

### Specifications Executed
- **Total planned**: 53
- **Successful**: 53
- **Failed**: 0

### Breakdown by type
- Baseline: 4 specs
- Design variants: 1 specs
- RC variants: 48 specs
- Inference variants: 2 specs (in inference_results.csv)

### Method
The paper uses nearest-neighbor matching (nnmatch) with exact matching on 4-digit SIC code,
bias adjustment, and Abadie-Imbens robust variance. Since no standard Python package implements
the exact Stata nnmatch command, we implemented the matching estimator manually following
Abadie & Imbens (2006, 2011):
- Match each treated unit to m nearest controls on matching variables within exact match groups
- Apply linear regression bias adjustment (OLS on matched controls, predict at treated covariate values)
- Compute Abadie-Imbens robust variance estimator using local variance estimates

The paper also reports OLS with industry FE (areg) as a comparison estimator, which we implement
using pyfixest.

**Important note on NN matching approximation**: The Python implementation of NN matching with
bias adjustment may differ from Stata's nnmatch in edge cases (tie-breaking, Mahalanobis metric
vs Euclidean, exact variance computation). The areg/OLS specifications are exact replications.
The NN matching results should be interpreted as close approximations. The sign and significance
patterns across specifications are the primary objects of interest for the specification search.

### RC Axes
1. **Controls**: Demographics (income1, pctminor1) as matching variables (0 or 2 vars)
2. **Sample**: Drop electric utilities, Southern Cal, Northern Cal, No SCAQMD, Severe, Single-facility, Small firms
3. **Period**: pd1-pd4 (baseline), pd2-pd3 (short-term), pd1-pd3 (intermediate)
4. **Functional form**: Levels (DIFFNOX) vs. Logs (lnDIFFNOX)
5. **Matching parameters**: m = 1, 2, 3, 4, 5
6. **Estimator**: nnmatch vs. areg/OLS with industry FE
7. **Joint combinations**: Multiple axes varied simultaneously

### Deviations
- pd1-pd3 period specification uses the pd2-pd3 panel with pd1 PRENOX for matching (since no
  pre-built pd1-pd3 panel exists in the replication data)
- Small firms specifications redefine treatment (SCAQMD location among non-RECLAIM firms)
  following Table 5 of the paper

## Software Stack
- Python 3.12.7
- pandas 2.2.3
- numpy 2.1.3
- pyfixest 0.40.1
- statsmodels 0.14.6
- scipy 1.15.1
- sklearn (for propensity score cleaning)
