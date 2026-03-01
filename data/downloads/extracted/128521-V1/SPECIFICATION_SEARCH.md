# Specification Search Report: 128521-V1

## Surface Summary

- **Paper ID**: 128521-V1
- **Design**: Difference-in-differences (TWFE)
- **Baseline groups**: 2
  - G1: Total mortality (census_mr_tot ~ cotton_dist_post | master_name + period), aweight=pop_census_tot, cluster(master_name)
  - G2: Age-specific mortality (census_mr_{age} ~ cotton_dist_post), 7 age groups
- **Budgets**: G1=80, G2=30
- **Seed**: 128521
- **Surface hash**: sha256:3815919d7b77fddbd6073e18d06207f70762f78198560b672b3e3990b826ad53

## Execution Summary

| Group | Planned | Executed | Success | Failed |
|-------|---------|----------|---------|--------|
| G1    | 30  | 30   | 30 | 0 |
| G2    | 21  | 21   | 21 | 0 |
| **Total** | **51** | **51** | **51** | **0** |

### Inference variants: 4 rows written to inference_results.csv

## Specification Details

### G1: Total Mortality
- **Baseline**: Table 2 Col 1 (no controls), Col 2 (demographics + region FE), Col 3 (+ nearby rings)
- **RC/controls/loo**: Drop ln_popdensity, linkable_shr, age_shares, region_x_period, nearby_rings (5 LOO specs)
- **RC/sample**: Exclude Manchester; Exclude Manchester+Liverpool+Leeds
- **RC/weights**: Unweighted
- **RC/data**: No foreign-born links; 6 linking restriction variants (unique_nlast, unique_nlast_nfirst, dist<200, dist<100, dist<50, dist==0)
- **Inference**: County-level clustering, HC1 robust

### G2: Age-Specific Mortality
- **Baseline**: 7 age groups (under15, 15-24, 25-34, 35-44, 45-54, 55-64, over64), Table 3 specification
- **RC/sample**: Exclude Manchester (7 specs)
- **RC/weights**: Unweighted (7 specs)

## Data Preparation

The regression dataset was reconstructed from raw data following the Stata do-files:
1. Generated age weights from Registrar mortality data and linked GRO-census records
2. Collapsed linked deaths to district x period panel
3. Merged with aggregate registrar deaths and census population counts
4. Computed lambda scaling factors (linked/aggregate ratio)
5. Calculated standardized mortality rates per 1000 persons per year
6. Merged cotton district indicators and population controls
7. Generated region x period interaction dummies

For linking restriction variants, the entire pipeline was re-run with different linked data subsets, including re-computed age weights and lambdas.

## Deviations and Notes

1. **Permutation test**: Not implemented (requires 538 iterations of spatial reassignment). Reported as inference variant in surface but skipped.
2. **Region x period FE**: Generated as interaction dummies (region x post) rather than absorbed, since the original code uses `xi i.region*post` which produces the same set of indicators.
3. **Data reconstruction**: The _Temp directory files were not pre-built; the entire data preparation was replicated in Python from raw data files. Minor numerical differences may arise from floating-point precision.

## Software Stack

- Python 3.12.7
- pyfixest: 0.40.1
- pandas: 2.2.3
- numpy: 2.1.3
