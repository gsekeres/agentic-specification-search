# Specification Search: 112451-V1

## Paper
Furman & Stern (2011), 'Biological Resource Centers and the Advancement of Technological Knowledge'
RAND Journal of Economics, 42(1), 1-32.

## Surface Summary
- **Baseline groups**: 1 (G1: Effect of BRC deposit on citations)
- **Design**: Difference-in-differences with article fixed effects
- **Estimator**: Poisson FE (proxy for paper's conditional FE negative binomial, xtnbreg)
- **Budget**: 60 max specs
- **Seed**: 112451

## Execution Summary
- **Total specs planned**: 51
- **Specs executed successfully**: 48
- **Specs failed**: 3
- **Inference variants**: 4

## Deviations from Surface
- The paper uses `xtnbreg` (conditional fixed-effects negative binomial), which is not
  available in Python. Poisson FE (`pyfixest.fepois`) is used as the primary estimator.
  This is a standard robustness check for count data and gives consistent estimates
  under correct conditional mean specification.
- Bootstrap SE (the paper's canonical inference) cannot be directly replicated in the
  Poisson FE framework. HC1 (heteroskedasticity-robust) SE is used as the canonical
  inference choice, with cluster-robust SE at article and pair levels as variants.

## RC Axes Executed

### Controls
- Add post_brc_yrs (time-since-deposit trend)
- LOO: drop window, drop age_brc1, drop both
- Grouped age/year dummies (5-year groups)
- Polynomial age/year (quadratic)

### Sample
- BRC-only sample (sample0==1 & window==0) with grouped and polynomial controls
- Trim top 1% of citation counts
- Winsorize at 99th percentile

### Functional Form
- Poisson FE (baseline proxy)
- OLS on log(cites+1)
- OLS on asinh(cites)
- OLS on raw cites (linear FE)
- Winsorized outcome transforms

### Fixed Effects
- Article FE (baseline)
- Pair FE (instead of or alongside article FE)
- Article + pair FE

### Joint Variations
- Sample x functional form (BRC-only x OLS/Poisson, trim x OLS/Poisson)
- FE x functional form (pair FE x OLS log/asinh)
- Controls x functional form (post_brc_yrs x OLS variants)
- Cluster SE at article and pair levels across specifications

## Inference Plan
- **Canonical**: Bootstrap SE (approximated by HC1 robust SE in Python)
- **Variant 1**: Cluster-robust SE at article level (rart_num)
- **Variant 2**: Cluster-robust SE at pair level (pair_num)
- **Variant 3**: HC1 for OLS variants

## Software Stack
- Python 3.12.7
- numpy 2.1.3
- pandas 2.2.3
- statsmodels 0.14.6
- linearmodels 6.1
- pyfixest 0.40.1
- scipy 1.15.1
- rdrobust 1.3.0
- openpyxl 3.1.5
- pyreadstat 1.3.3

## Failed Specifications
- `rc/controls/sets/polynomial_age_year`: Matrix is singular.
- `rc/sample/subset/sample0_brc_only_polynomial`: Matrix is singular.
- `rc/joint/sample_form/sample0_poisson_poly`: Matrix is singular.
