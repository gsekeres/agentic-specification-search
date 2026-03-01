# Specification Surface: 146041-V1

## Paper Overview

This is a development accounting paper that estimates cross-country differences in "aggregate skill quality" (AQ) and relates them to income per worker. The key empirical object is the elasticity of AQ with respect to income per worker, estimated via bivariate cross-country OLS. The paper constructs AQ by combining country-specific Mincerian wage regressions (estimated from census micro-data) with labor supply shares and a CES production function framework.

## Baseline Groups

### G1: Cross-country elasticity of aggregate skill quality

**Claim object**: The paper's main quantitative claim (Table 2, Table 3) is that quality-adjusted human capital (AQ) varies more strongly with income per worker than quantity-only measures (H/L), with an elasticity of approximately 0.5-0.7 for AQ vs ~0.3 for the wage ratio alone and ~0.1 for quantity.

**Baseline spec**: Table 2, Row 1, Column 3 -- bivariate OLS regression of log(irAQ53_dum_skti_hrs_secall) on log(income per worker). This is the baseline AQ measure using sigma=1.5, hours-weighted labor supply, education dummies for wages, and the micro-data sample.

**Design code**: `cross_sectional_ols` -- bivariate cross-country regression with no controls.

## What Is Included and Why

### Outcome construction variants (the main axis of variation)

The paper's revealed search space is dominated by **outcome construction** rather than control sets. All regressions in Tables 2-3 are bivariate (no covariates beyond l_y). The specification universe therefore varies along:

1. **Elasticity of substitution (sigma)**: 1.3, 1.5 (baseline), 2.0
2. **Wage premium estimation**: education dummies only vs. experience + gender adjusted
3. **Labor supply measure**: hours-weighted (baseline), body count, working-age population
4. **Mincerian return assumption**: country-specific (baseline) vs. common Mincerian return
5. **Skill threshold**: upper-secondary cutoff (baseline), some-college, tertiary-only
6. **Sample for Q estimation**: baseline migrants, 10+ years in US, good English, no skill downgrading, no mismatch, sector sorting controls, region sorting controls, selection-adjusted
7. **Country sample for elasticity**: US immigrants only, pooled countries, pooled with bilateral controls, micro-data sample, broad Barro-Lee sample
8. **Sector subsamples**: agriculture, manufacturing, low-skill services, high-skill services
9. **Self-employment**: wage-employed only vs. including self-employed
10. **Outcome alternatives**: AQ, wage ratio, quantity ratio (H5/L3), Barro-Lee AQ

### Outlier and sample trimming

- Trimming at 1/99 and 5/95 percentiles of income per worker
- Excluding individual outlier countries

### Functional form

- The regressions are log-log by construction (both outcome and regressor in logs), so no additional functional form variation is needed.

## What Is Excluded and Why

- **Controls**: Not applicable. The paper's regressions are all bivariate; adding controls would change the estimand because with only ~11 countries there is no degrees of freedom for meaningful multivariate regression.
- **Fixed effects**: Not applicable for the same reason.
- **Alternative estimators (IPW, AIPW, matching)**: Not applicable -- no causal identification; this is a descriptive correlation.
- **Diagnostics**: Not included in core. With N~11-90, regression diagnostics have limited power.
- **Sensitivity/exploration**: Not in core universe.

## Inference Plan

- **Canonical**: HC1 robust SEs (matching Stata `robust` option used throughout)
- **Variant**: HC3 SEs for small-sample correction (important given N~11-90)

## Budgets and Sampling

- **Target**: ~50-80 specifications
- Full enumeration is feasible because the universe is defined by discrete choices (sigma x wage measure x labor supply x sample x sector) rather than combinatorial control subsets.
- No control-subset sampling needed.

## Key Notes

- This paper is unusual for our pipeline because the "regression" is a simple bivariate cross-country correlation, and the main specification search dimension is *how the outcome variable is constructed* rather than which controls to include.
- The unit of observation is country (~11 in the micro sample, ~90 in the Barro-Lee sample).
- The focal parameter is the slope coefficient on l_y (log income per worker), interpretable as an elasticity.
