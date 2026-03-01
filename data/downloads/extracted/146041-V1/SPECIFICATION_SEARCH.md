# Specification Search: 146041-V1

## Surface Summary

- **Paper**: Hendricks & Schoellman (2018), "Human Capital and Development Accounting: New Evidence from Immigrant Earnings"
- **Baseline groups**: 1 (G1)
- **Design**: cross_sectional_ols (bivariate regression, no controls)
- **Baseline**: Table 2 Row 1 Col 3 -- log(irAQ53_dum_skti_hrs_secall) ~ l_y, micro sample (N~12 countries)
- **Focal parameter**: slope coefficient on l_y (log income per worker) = elasticity of AQ w.r.t. income
- **Budget**: max 80 core specs
- **Seed**: 146041

## Execution Summary

### Specification Results
- **Planned**: 67
- **Executed successfully**: 66
- **Failed**: 1

### Inference Results
- **Inference variants executed**: 65
- **Canonical inference**: HC1 (heteroskedasticity-robust)
- **Variant**: HC3 (small-sample correction)

## Baseline Result
- **Coefficient**: 1.4081
- **SE (HC1)**: 0.4967
- **p-value**: 0.0177
- **N**: 12
- **R-squared**: 0.5613

## Specification Universe Description

The main axis of variation is **outcome variable construction** rather than controls (all regressions are bivariate). Specifications vary along:

1. **Elasticity of substitution (sigma)**: 1.3, 1.5 (baseline), 2.0
2. **Wage premium estimation**: education dummies (baseline) vs experience+gender adjusted
3. **Labor supply measure**: hours (baseline), body count, working-age population
4. **Mincerian return**: country-specific (baseline) vs common
5. **Skill threshold**: upper-secondary (baseline), some-college, tertiary-only
6. **Country sample**: micro (N~12, baseline), US Pooled (N~101), Barro-Lee (N~92)
7. **Sector subsamples**: agriculture, manufacturing, low-skill services, high-skill services
8. **Self-employment**: wage-employed only (baseline) vs including self-employed
9. **Sample restrictions**: 10+ years in US, good English, no downgrading, no mismatch, sorting controls
10. **Outlier trimming**: 1/99 and 5/95 percentiles of income

## Deviations and Notes

1. **rc/form/outcome/aq_sigma_4p0**: Failed. Sigma=4.0 AQ is not pre-computed in the data. The devacc_main.do only uses sigma=4 for development accounting calculations, not for the AQ regression variables.

2. **Table 3 AQ column regressions (rc/sample/restriction/*)**: For Table 3 rows that change the Q estimation method (10+ years, good English, no downgrading, sorting), the AQ regression outcome variable (irAQ53_dum_skti_hrs_secall) is unchanged. These rows produce the same point estimate for the AQ column. They are included for completeness and to faithfully represent the paper's revealed specification space.

3. **Continent dummies (rc/controls/add/continent_dummies)**: Implemented as an Americas vs. non-Americas dummy given the small sample size (12 countries). Full continent dummies would consume too many degrees of freedom.

4. **Treatment alternatives**: Used log(y_relUS_2005) and log(y_relUS2000) as alternative treatments since the baseline l_y is already PWT-based.

## Software Stack

- Python 3.12.7
- numpy: 2.1.3
- pandas: 2.2.3
- scipy: 1.15.1
- Manual OLS implementation (no pyfixest needed for bivariate regression)
