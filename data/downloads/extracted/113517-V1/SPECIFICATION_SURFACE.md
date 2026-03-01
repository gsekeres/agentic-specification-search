# Specification Surface: 113517-V1

## Paper Overview
"The Relative Power of Employment-to-Employment Reallocation and Unemployment Exits in Predicting Wage Growth" (Moscarini & Postel-Vinay, AER P&P 2017)

## Baseline Groups

### G1: Predictive power of EE reallocation for wage growth

**Claim object**: The EE (employment-to-employment) reallocation rate has significant predictive power for aggregate wage growth, above and beyond unemployment exits and other labor market flows.

**Design**: Cross-sectional OLS (two-stage procedure)
- First stage: Regress individual-level transition rates on covariates + market x time FE using `areg` with analytic weights. Extract market-time FE as predicted group-level rates.
- Second stage: Regress predicted wage growth (market-time FE from first-stage wage regression) on predicted flow rates + year_month trend + market FE.

**Baseline spec**: Table 1, Specification 6 (all flows), with `logern_nom` (nominal log earnings) as the outcome. This is the most complete specification including all labor market flows (EE, UE, UR, NE, EN, EU).

**Additional baselines**: The same specification (Spec 6) is run for 4 different dependent variables:
- `logern_nom` (nominal log earnings)
- `logern` (real log earnings)
- `loghwr_nom` (nominal log hourly wage)
- `loghwr` (real log hourly wage)

## Revealed Search Space

The paper reveals the following dimensions of variation in Table 1:
1. **Which predicted flows enter the second stage**: 9 combinations (EE only; UE only; UR only; EE+UE; EE+UE+UR; all six flows; grouped flows; all flows for job stayers; all flows + EE interaction)
2. **Four outcome variables**: nominal/real x earnings/hourly wages
3. **Job stayers subsample**: restricting to individuals who did not change jobs (eetrans_i==0)

The paper does NOT reveal:
- Alternative first-stage specifications
- Alternative fixed effects structures
- Alternative weighting schemes
- Clustering of standard errors

## RC Axes (Core Universe)

### 1. Controls progression (which flows enter)
Maps directly to the paper's revealed search space:
- `rc/controls/progression/ee_only`: xee + ym_num | mkt
- `rc/controls/progression/ee_ue`: xee + xue + ym_num | mkt
- `rc/controls/progression/ee_ue_ur`: xee + xue + xur + ym_num | mkt
- `rc/controls/progression/all_flows`: xee + xue + xur + xne + xen + xeu + ym_num | mkt (baseline)
- `rc/controls/progression/grouped_flows`: xee + xur + xnue + xenu + ym_num | mkt

### 2. Leave-one-out controls
Drop one predicted flow at a time from the full specification:
- `rc/controls/loo/drop_xue`
- `rc/controls/loo/drop_xur`
- `rc/controls/loo/drop_xne`
- `rc/controls/loo/drop_xen`
- `rc/controls/loo/drop_xeu`

### 3. Sample restrictions
- `rc/sample/subpop/job_stayers`: eetrans_i==0 & lagemp>0 (paper-revealed)
- `rc/sample/time/early_half`: first half of sample periods
- `rc/sample/time/late_half`: second half of sample periods
- `rc/sample/outliers/trim_y_1_99`: trim outcome outliers at 1/99 pctiles
- `rc/sample/outliers/trim_y_5_95`: trim outcome outliers at 5/95 pctiles

### 4. Functional form (outcome variable)
The paper treats all 4 outcome variables as equally important:
- `rc/form/outcome/logern_nom` (baseline)
- `rc/form/outcome/logern`
- `rc/form/outcome/loghwr_nom`
- `rc/form/outcome/loghwr`

### 5. Model specification
- `rc/form/model/xee_interaction`: add xee * eetrans_i interaction term (paper-revealed Spec 9)

### 6. Weights
- `rc/weights/main/unweighted`: run without analytic weights

### 7. Fixed effects / time trend
- `rc/fe/drop/year_month_trend`: drop year_month from second stage
- `rc/fe/add/year_month_fe`: replace continuous year_month with year_month FE

## Inference Plan

**Canonical**: IID standard errors (matching the paper's areg default)

**Variants** (for inference_results.csv):
- HC1 (heteroskedasticity-robust)
- Cluster at market level

## Budget

Target: 50-80 specifications total. Full enumeration is feasible given the small number of second-stage controls (6 predicted flow variables). The main combinatorial dimension is the cross-product of outcome variables x flow inclusion patterns x sample restrictions, which is manageable.

## Excluded from Core

- First-stage variations (these are intermediate steps; varying them would change the predicted flow variables and is not revealed by the paper)
- Alternative market definitions (sex x race x agegroup x education is the only definition used)
- Exploration of additional SIPP panels or time periods not in the data
