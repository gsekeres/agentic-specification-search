# Specification Surface: 111185-V1

**Paper**: "Optimal Climate Policy When Damages are Unknown" (Ivan Rudik, AEJ: Economic Policy 2020)
**Classification**: `structural_calibration` (primary), `cross_sectional_ols` (secondary)
**Created**: 2026-02-13

---

## 1. Paper Overview and Regression Scope

This is primarily a structural/calibration paper in environmental economics. The paper develops an integrated assessment model (DICE-type) extended with Bayesian learning about unknown damage function parameters and robust control (ambiguity aversion). The structural model is solved via value function iteration requiring >20,000 core-hours of computation.

**The only reduced-form regression** is Table 1: an OLS regression of log damages on log temperature (N=43) using the Howard & Sterner (2017) meta-analysis dataset. This bivariate regression estimates the power-law damage function parameters (coefficient d1 and exponent d2) that feed as calibration inputs into the structural model. All other results (Tables 2-3, Figures 5-8) are structural model outputs, not statistical estimations.

**Implication for specification search**: The specification surface applies exclusively to this single calibration regression. The "claim" being tested is the damage elasticity estimate (d2 = 1.88, the exponent in D(T) = d1 * T^d2), which Rudik uses as an input to his structural model.

---

## 2. Baseline Group

### G1: Damage Function Elasticity (Table 1)

**Claim object**:
- **Outcome concept**: Log of climate damage (GDP loss transformed to damage function form: log((D/100)/(1-D/100)))
- **Treatment concept**: Log temperature change (log degrees Celsius warming)
- **Estimand concept**: Elasticity of damages with respect to temperature -- the power-law exponent d2
- **Target population**: Howard & Sterner (2017) meta-analysis: 49 damage estimates from published climate damage studies (1994-2015), restricted to 43 observations with positive damage values (6 dropped due to zero or negative damages producing undefined log values)

**Baseline specification (Table 1)**:
- Regression: `reg log_correct logt` (Stata)
- Coefficient on `logt`: 1.882 (SE = 0.451, p = 0.00015)
- R-squared: 0.299
- N = 43
- No controls, no robust standard errors, no fixed effects
- Classical (homoskedastic) standard errors

---

## 3. Revealed Search Space Analysis

The paper's revealed search space is **extremely narrow**:

1. **Only one regression** is reported (Table 1).
2. **No alternative specifications** are shown -- no robustness tables, no alternative control sets, no alternative samples.
3. The regression is a **calibration input**, not the paper's main contribution. The paper's novelty is in the structural model framework.
4. The Howard & Sterner (2017) dataset contains 146 variables of study-level characteristics, but none are used as controls.

This means our specification surface must be constructed almost entirely from **standardized batteries** rather than manuscript-revealed variations. The paper opens essentially zero forking paths in the regression itself.

---

## 4. Data Description

**Dataset**: Howard & Sterner (2017) replication data (`10640_2017_166_MOESM10_ESM.dta`)
- 49 observations (damage estimates from published studies)
- 146 variables (study characteristics, temperature measures, damage measures)
- Regression sample: 43 observations (6 dropped: 3 with D_new=0, 3 with D_new<0)

**Key variables**:
- `D_new`: GDP loss as percentage
- `t`: Temperature change (degrees Celsius), equals `T_new`
- `correct_d`: Damage transformed to structural form: (D_new/100)/(1 - D_new/100)
- `log_correct`: log(correct_d) -- the dependent variable
- `logt`: log(t) -- the independent variable

**Available meta-regression controls** (all non-missing in regression sample):
- `Year`: Publication year (1994-2015)
- `Market`: Market damages only (dummy, 17/43)
- `Nonmarket`: Non-market damages only (dummy, 3/43)
- `Grey`: Grey literature (dummy, 9/43)
- `Preindustrial`: Pre-industrial temperature baseline (dummy, 20/43)
- `Based_On_Other`: Based on another study (dummy, 10/43)
- `Method_1` through `Method_5`: Method dummies (enumerative, survey, experimental, statistical, science)
- `Groups_1`, `Groups_2`, `Groups_3`: Study grouping dummies

**Alternative temperature measures**:
- `Temp_adj_FUND_curr`: FUND model temperature adjustment
- `Temp_adj_NASA`: NASA temperature adjustment
- `Temp_adj_AVG`: Average temperature adjustment

**Notable data features**:
- The Weitzman (2010) observation at T=12C, D=99% is an extreme outlier (Cook's D = 2.41, next highest = 0.42)
- 4 observations exceed the Cook's D > 4/N threshold
- Residuals are non-normal (Jarque-Bera p < 0.001, skewness = 1.24, kurtosis = 5.10)
- Very small sample severely limits the number of feasible controls

---

## 5. Core Universe Specification

### 5.1 Design

The design is `cross_sectional_ols`. The baseline estimator is simple OLS. No alternative design-level estimators are applicable -- this is not a causal-inference setting but a meta-analytic calibration regression.

### 5.2 Controls Axis (`rc/controls/*`)

**Rationale**: The baseline has zero controls. Howard & Sterner (2017) is a meta-analysis dataset with study-level characteristics that are standard meta-regression controls. Adding these tests whether the damage elasticity is robust to study-design confounds.

**Control pool** (10 variables):
- `Year`, `Market`, `Nonmarket`, `Grey`, `Preindustrial`, `Based_On_Other`
- `Method_1`, `Method_2`, `Method_3`, `Method_5`
- (Note: `Method_4` is all zeros in regression sample, excluded)

**Control blocks**:
1. **study_type**: Market, Grey, Preindustrial (basic study design characteristics)
2. **method**: Method_1, Method_2, Method_3, Method_5 (estimation methodology)
3. **quality**: Based_On_Other, Nonmarket (data quality/type indicators)
4. **temporal**: Year (publication year trend)

**Hard cap**: Maximum 4 controls. With N=43 and 41 baseline degrees of freedom, adding more than 4 controls risks severe overfitting. The paper uses 0 controls, so the control-count envelope is [0, 4].

**Planned control specs**:
- `rc/controls/sets/none`: Baseline (bivariate) -- identical to baseline
- Single-control additions (10 specs): Add each control one at a time
- Control sets (4 specs): study_type block, method block, quality block, all-blocks combined
- Control progression (4 specs): bivariate -> +study_type -> +study_type+method -> full

**Block-combination subsets**: With 4 blocks, there are 2^4 - 1 = 15 non-empty combinations. After removing the baseline (empty set) and already-covered single-block and full combinations, this adds approximately 10 additional unique block combinations.

### 5.3 Sample Axis (`rc/sample/*`)

**Rationale**: The regression sample is driven entirely by whether log(correct_d) is finite (i.e., whether correct_d > 0). Alternative sample restrictions test robustness to influential observations and study selection criteria.

**Planned sample specs** (9 specs):
- **Outlier handling**:
  - Trim log_correct at [1%, 99%] and [5%, 95%]
  - Drop Weitzman 12C observation (most influential by far, Cook's D = 2.41)
  - Drop all observations with Cook's D > 4/N (drops 4 obs)
- **Temporal stability**:
  - Pre-2006 studies only (early literature)
  - Post-2006 studies only (recent literature)
- **Quality filters**:
  - Drop derivative studies (Based_On_Other=1)
  - Drop grey literature (Grey=1)

### 5.4 Functional Form Axis (`rc/form/*`)

**Rationale**: The log-log specification imposes a power-law relationship. Alternatives test whether this functional form drives the result.

**Planned functional-form specs** (3 specs):
- `rc/form/outcome/level`: Levels regression (correct_d on t)
- `rc/form/model/quadratic_treatment`: log_correct on logt + logt^2 (tests curvature)
- `rc/form/model/levels_quadratic`: correct_d on t + t^2 (standard DICE-type quadratic)

**Interpretation note**: The levels and quadratic specifications change how the coefficient maps to d2, so marginal effects at a reference temperature must be computed for comparability.

### 5.5 Preprocessing Axis (`rc/preprocess/*`)

**Rationale**: Temperature measurement and damage transformation are key preprocessing choices.

**Planned preprocessing specs** (5 specs):
- Winsorize log_correct at [1%, 99%]
- Three alternative temperature adjustments (FUND, NASA, AVG baselines)
- Include zero-damage observations using asinh(correct_d) transformation (recovers 3 obs)

### 5.6 Inference Axis (`infer/se/*`)

**Rationale**: The baseline uses classical (homoskedastic) standard errors. With N=43 and evidence of non-normality, robust SE variants are important.

**Planned inference specs** (4 specs):
- HC1 (standard robust)
- HC2 (less biased for small samples)
- HC3 (best small-sample properties)
- Cluster by study/primary author (accounts for within-author dependence; 37 unique studies)

---

## 6. Constraints

| Constraint | Value | Rationale |
|---|---|---|
| `controls_count_min` | 0 | Baseline is bivariate |
| `controls_count_max` | 4 | N=43 hard-limits feasible controls |
| `linked_adjustment` | false | Not applicable (single-equation OLS) |
| `small_sample_flag` | true | N=43 requires caution on all axes |

---

## 7. Budget and Sampling Plan

**Budget**: 80 specifications maximum (core total).

**Enumeration strategy**: Full enumeration is feasible. No random sampling is needed.

| Axis | Planned specs |
|---|---|
| Baseline | 1 |
| Controls (single-add) | 10 |
| Controls (sets) | 4 |
| Controls (progression) | 4 |
| Controls (block subsets) | ~10 |
| Sample | 9 |
| Functional form | 3 |
| Preprocessing | 5 |
| Inference | 4 |
| **Total (one-axis-at-a-time)** | **~50** |
| High-value interactions | ~5-10 |
| **Estimated total** | **~55-60** |

The total is well within the 80-spec budget. No subset sampling is required.

**Interactions** (limited, high-value only):
- Drop Weitzman outlier + HC3 SEs (outlier + inference interaction)
- Drop Weitzman outlier + add study_type controls (outlier + controls interaction)
- HC3 SEs + study_type controls (inference + controls interaction)
- Levels quadratic + drop Weitzman (form + sample interaction)

---

## 8. Diagnostics Plan (Not Part of Core Universe)

| Diagnostic | Scope | Description |
|---|---|---|
| `diag/regression/influence/cooks_d` | baseline_group | Cook's D for all observations. Critical given the Weitzman extreme outlier. |
| `diag/regression/normality/jarque_bera` | baseline_group | Normality test on residuals. Baseline already shows significant non-normality. |
| `diag/regression/heteroskedasticity/breusch_pagan` | baseline_group | Test whether classical SEs are appropriate. |

---

## 9. What Is Excluded and Why

### Excluded from core (by design):
- **Structural model variations**: The paper's main contribution is the dynamic programming model. Varying structural parameters (discount rate, climate sensitivity, robust control penalty theta) would require solving the full structural model (>20,000 core-hours). These are not feasible and are not regression specifications.
- **Alternative estimand explorations**: CATE, heterogeneity by study characteristics, alternative damage function forms (Weitzman, Dietz) are mentioned in the structural model but not as regression-level explorations.
- **Multiple imputation**: With only 6 dropped observations and a clear structural reason for dropping them (log of non-positive values), MI is not well-motivated.
- **DML/IPW/matching**: Not applicable -- this is not a causal treatment effect regression. It is a meta-analytic parameter estimation.
- **Spatial/temporal clustering**: The dataset is cross-sectional study-level data, not panel or spatially linked.

### Excluded controls:
- `Method_4`: All zeros in regression sample (no variation).
- Model dummies (`model_DICE`, `model_PAGE`, etc.): 13 model dummies with very few observations each. Including these would consume too many degrees of freedom given N=43.
- `primary_author_*` dummies: 12 author dummies; too many relative to N=43.
- `Groups_1`, `Groups_2`, `Groups_3`: Collinear with study characteristics already in the pool (Groups_1+Groups_2+Groups_3 partitions the sample). One could be included, but the study_type block already captures similar variation.
- `Preferred`, `Eco_market`, `CrossNational`, `CorrectBias`: These have substantial missing data (only 12-27 non-missing out of 49), which would further reduce the already-small sample.

---

## 10. Key Interpretive Notes

1. **This is a calibration input, not the paper's main claim.** The damage elasticity d2=1.88 feeds into the structural model, but the paper's contribution is the model framework and its policy implications. The specification surface tests the robustness of this calibration input.

2. **Small N severely constrains the surface.** With only 43 observations, many standard specification-search strategies (high-dimensional controls, extensive subsampling) are infeasible. The surface is deliberately conservative.

3. **The baseline is bivariate.** The paper uses zero controls, which means any added control is a departure from the author's choice. Meta-regression controls are standard in the meta-analysis literature (Howard & Sterner themselves use them), so this is a natural robustness axis.

4. **Outlier sensitivity is the most consequential axis.** The Weitzman (2010) observation at 12C warming with 99% GDP loss is an extreme outlier with Cook's D = 2.41. Whether the elasticity estimate is robust to this observation is the single most important robustness question for this regression.
