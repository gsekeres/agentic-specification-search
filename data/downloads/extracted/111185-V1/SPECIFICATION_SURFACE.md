# Specification Surface: 111185-V1 (Rudik, "Optimal Climate Policy When Damages are Unknown")

## Paper Summary

This paper develops a dynamic programming model of optimal climate policy under uncertainty about climate damage function parameters. The sole standard econometric regression is a single bivariate OLS regression in Table 1 that estimates damage function parameters by regressing log damages on log temperature using the Howard & Sterner (2017) meta-analysis data (49 observations, 43 used after dropping non-positive damage estimates).

The estimated damage exponent (d2 = 1.88) and intercept (d1 = -5.31) feed into the structural model as calibrated parameters. All other tables and figures come from solving/simulating the dynamic programming model.

## Baseline Groups

### G1: Damage exponent estimation (Table 1)

- **Outcome concept**: Log climate damages (log of corrected damage fraction)
- **Treatment concept**: Log temperature increase (log degrees Celsius)
- **Estimand concept**: Elasticity of damages with respect to temperature (the damage exponent d2 in a power-law damage function D = exp(d1) * T^d2)
- **Target population**: Howard & Sterner (2017) meta-analysis of global climate damage estimates
- **Baseline spec**: `reg log_correct logt` (OLS, classical SE, N=43, coef=1.882, SE=0.451, p<0.001)

This is the only baseline group because there is only one regression in the replication package.

## What is Included (Core Universe)

### Controls axis (rc/controls/*)

The baseline is a bivariate regression with zero controls. The Howard & Sterner dataset contains several meta-analytic study characteristics that could serve as controls:

- **Study quality block**: Grey literature flag, Repeat observation flag, Based-on-other-study flag
- **Damage type block**: Market-only damages flag, Catastrophic damages flag
- **Study design block**: Preindustrial temperature baseline flag, Publication year

With N=43 and a bivariate baseline, we constrain the control count to [0, 6] to avoid overfitting.

Specifications:
1. **Standard sets**: none (= baseline), minimal, extended, full
2. **Single-control additions**: Add each of the 7 available controls one at a time
3. **Control progression**: bivariate -> study_quality -> damage_type -> full
4. **Block-exhaustive combinations**: 2^3 = 8 block combinations (study_quality, damage_type, study_design)

### Sample axis (rc/sample/*)

- **Outlier trimming**: Trim outcome at 1/99 and 5/95 percentiles; trim treatment at 1/99
- **Cook's D influence**: Drop observations with Cook's D > 4/N
- **Study quality filters**: Drop repeat observations, drop based-on-other, drop grey literature, drop catastrophic damage estimates
- **Temporal splits**: Early studies (pre-2006) vs late studies (2006+)

### Functional form axis (rc/form/*)

The baseline is log-log (elasticity interpretation). Alternatives:
- **Levels outcome**: correct_d ~ logt (semi-log)
- **Asinh outcome**: asinh(correct_d) ~ logt (handles zeros/negatives)
- **Levels treatment**: log_correct ~ t (semi-elasticity)
- **Quadratic treatment**: log_correct ~ logt + logt^2 (nonlinear in logs)
- **Levels-quadratic**: correct_d ~ t + t^2 (polynomial damage function in levels, quadratic)

### Preprocessing axis (rc/preprocess/*)

- Winsorize outcome at 1/99 percentiles
- Winsorize treatment at 1/99 percentiles

## What is Excluded (and Why)

- **Fixed effects**: No panel structure; cross-sectional meta-analysis data. No FE axis.
- **Clustering**: No natural clustering structure. Study-level clustering could be considered but with only 41 unique studies across 43 observations, clustering is essentially observation-level.
- **Weights**: No weighting in baseline; weighting by precision/study quality would change the estimand.
- **IPW/AIPW/Matching**: Treatment (temperature) is continuous and not binary; these methods are inapplicable.
- **DML**: Not applicable for a bivariate meta-regression with continuous treatment.
- **Subpopulation heterogeneity**: Explore-level, not core RC.

## Inference Plan

- **Canonical**: Classical (non-robust) OLS standard errors (matching baseline)
- **Variants** (written to inference_results.csv):
  - HC1 (Stata robust)
  - HC2 (better small-sample properties)
  - HC3 (most conservative for small samples)

## Budget and Sampling

- **Total budget**: ~80 core specifications
- **Controls subset budget**: 32 (8 exhaustive block combinations + 24 variable-level subsets)
- **Seed**: 111185
- **Sampler**: Exhaustive block combinations first, then stratified-by-size variable-level draws

## Key Constraints

- N=43 severely limits the number of controls (max 6 per spec)
- All controls are binary meta-analytic characteristics; no continuous controls except Year
- The paper's claim is about the damage exponent d2, not about the intercept d1
- Log-log functional form is integral to the power-law damage function interpretation
