# Specification Surface: 113517-V1

**Paper**: "The Relative Power of Employment-to-Employment Reallocation and Unemployment Exits in Predicting Wage Growth"
**Authors**: Giuseppe Moscarini & Fabien Postel-Vinay (AER Papers & Proceedings, 2017)
**Design**: Panel fixed effects (two-stage composition-adjustment procedure)
**Date created**: 2026-02-13

---

## 1. Paper Overview and Main Claims

This paper asks which labor market flow -- employment-to-employment (EE) reallocation or unemployment-to-employment (UE) exits -- is a stronger predictor of wage growth. The authors use SIPP panel data (1996--2013) with a two-stage procedure:

1. **First stage**: Individual-level regressions absorbing market-by-time fixed effects to extract composition-adjusted, market-level transition rate FEs and wage growth FEs.
2. **Second stage**: Market-level regressions of the predicted wage growth FE on predicted transition rate FEs, absorbing market FE with a year_month continuous control.

The main finding is that EE reallocation is a much stronger predictor of (nominal) earnings growth than UE exits, supporting the job-ladder theory. This relationship weakens or reverses for hourly wages, suggesting EE transitions affect earnings primarily through hours/job characteristics rather than hourly rates.

Table 1 is the only regression table in the paper. It presents 9 second-stage specifications (varying which labor market flow FEs enter the regression) for each of 4 dependent variables (nominal earnings, real earnings, nominal hourly wage, real hourly wage), totaling 36 regressions.

---

## 2. Baseline Groups

The paper presents 4 outcome variables as headline results in a single table. The qualitative conclusions differ across outcome variables (strong EE effect for earnings, weak/reversed for hourly wages), so we treat each outcome as a separate baseline group.

| Group | Outcome | Key finding | Baseline coefficient (Spec 1 / Spec 6) |
|-------|---------|-------------|----------------------------------------|
| G1 | Nominal earnings growth (xdlogern_nom) | Strong positive EE effect | 0.045 / 0.045 |
| G2 | Real earnings growth (xdlogern) | Strong positive EE effect | 0.037 / 0.039 |
| G3 | Nominal hourly wage growth (xdloghwr_nom) | Weak/insignificant EE effect with full controls | 0.0096 / 0.0006 (p=0.10) |
| G4 | Real hourly wage growth (xdloghwr) | Negative EE effect with full controls | 0.0039 / -0.0037 |

**Why 4 groups**: The paper frames the contrast across outcome variables as a key result. The EE coefficient is robustly positive for earnings but not for hourly wages. These are genuinely different outcome concepts (earnings vs. hourly wages; nominal vs. real), and the sign/significance patterns differ qualitatively.

---

## 3. Baseline Specifications (Canonical Implementations)

For each baseline group, the paper presents 9 second-stage specifications that vary which labor market flow FEs are included:

| Spec | Second-stage regressors | Description |
|------|------------------------|-------------|
| 1 | xee | EE only |
| 2 | xue | UE only |
| 3 | xur | UR only |
| 4 | xee, xue | EE + UE |
| 5 | xee, xue, xur | EE + UE + UR |
| 6 | xee, xue, xur, xne, xen, xeu | All flows |
| 7 | xee, xur, xnue, xenu | Grouped flows |
| 8 | (same as 6, job stayers only) | All flows, job stayers |
| 9 | xee, xee_i, xue, xur, xne, xen, xeu | All flows + EE interaction |

All second-stage specs also include year_month_num (continuous) and absorb market FE.

We designate Specs 1, 5, and 6 as the three **anchor specs** for one-axis-at-a-time RC: Spec 1 is the simplest (bivariate), Spec 5 adds the two main comparison flows, and Spec 6 is the fullest.

---

## 4. Two-Stage Procedure: Linked Adjustment

This paper's two-stage structure is central to the specification surface design. The stages are **linked**: changing first-stage controls alters the generated regressors used in the second stage. Key details:

**First-stage controls**:
- `e_controls` (for EE/EU/EN/wage regressions): lagstate, laguni, lagsiz, lagocc, lagind, lagpub + lag(depvar) for EE and wage
- `u_controls` (for UE/NE/UR regressions): lagstate only
- Absorbed FE: mkt_t (market x year_month, ~17,500 groups)
- Weights: wgt (SIPP person weights)

**Second-stage structure**:
- Absorbed FE: mkt (100 demographic market groups)
- Continuous control: year_month_num
- Weights: wgt

**Linkage constraint**: We treat first-stage controls as linked/shared. When varying first-stage controls, the same change applies to all first-stage regressions of the same eligibility type (all employed-eligible regressions share e_controls; all unemployed-eligible share u_controls). This preserves the paper's intended composition-adjustment procedure.

---

## 5. Core-Eligible Universe

### 5.1 Second-stage flow composition (rc/controls/sets/*)

All 9 specs from Table 1 are directly enumerated. This is the paper's **revealed** specification search space for second-stage regressor composition.

### 5.2 First-stage control variants (rc/controls/first_stage/*)

These vary the individual-level controls used to extract the composition-adjusted FE:

| Variant | Description | Rationale |
|---------|-------------|-----------|
| e_controls_minimal | Use only lagstate for all first-stage regressions | Tests whether the rich employed-side controls matter for the FE extraction |
| e_controls_extended | Use full e_controls for UE/NE/UR regressions too | Tests symmetry of control sets across labor market status |
| no_lag_depvar | Drop lag(depvar) from EE and wage first-stage regressions | Lagged dependent variable in FE models can introduce Nickell bias concerns (though this is a cross-sectional FE, not panel FE, in the first stage) |
| add_lag_hours | Add lagged log(hours) to e_controls | Tests whether hours variation changes the composition adjustment |

**Why these are core-eligible**: Varying first-stage controls changes the *implementation* of the composition adjustment but preserves the estimand concept (within-market association between EE and wage growth, purged of composition effects). The paper's choice of first-stage controls is a researcher degree of freedom.

### 5.3 Sample restrictions (rc/sample/*)

| Variant | Description | Rationale |
|---------|-------------|-----------|
| drop_first_panel | Drop 1996 SIPP panel | Earliest panel may have different measurement properties |
| drop_last_panel | Drop 2008 SIPP panel | Latest panel has partial coverage |
| pre_crisis | year_month < 2008-01 | Pre-Great Recession only |
| post_crisis | year_month >= 2008-01 | Great Recession and recovery |
| drop_short_spells | Require >= 3 consecutive observations | Quality filter on individual-level data entering first stage |

**Why these are core-eligible**: The SIPP panels span 1996--2013, covering major macroeconomic events. The EE-wage relationship may be structurally different across periods. These are standard time-stability checks that do not change the target population concept (they restrict to subperiods while targeting the same demographic market cells).

### 5.4 Fixed effects / time control (rc/fe/*)

| Variant | Description | Rationale |
|---------|-------------|-----------|
| time_dummies | Replace year_month continuous control with year_month FE | Tests whether the linear time trend assumption drives results. With ~176 time periods, absorbing time FE is feasible but greatly reduces variation. |
| time_quadratic | Add year_month^2 to continuous controls | Allows nonlinear time trends |
| drop_market_fe | Drop market FE from second stage | Pooled second-stage regression. Changes identification from within-market to pooled. Borderline explore/ but included because it tests whether between-market variation contributes. |

### 5.5 Weights (rc/weights/*)

| Variant | Description | Rationale |
|---------|-------------|-----------|
| unweighted | No weights | Tests sensitivity to SIPP sampling weights |
| trim_p99 | Trim weights at 99th percentile | Guards against extreme weight influence |

### 5.6 Market definition (rc/data/aggregation/*)

| Variant | Description | Rationale |
|---------|-------------|-----------|
| coarser_markets | mkt = sex x agegroup x education (50 groups, drop race) | Tests sensitivity to market granularity; reduces noise in small cells |
| finer_markets | mkt = sex x race x agegroup x education x state (~5,000 groups) | Tests whether finer geographic disaggregation changes results. May be computationally infeasible. |

**Why market definition is core-eligible**: The definition of a "demographic market" (the mkt variable) is a key researcher degree of freedom that determines both the first-stage FE groups and the second-stage unit of analysis. The paper's choice (sex x race x agegroup x education = 100 groups) is one reasonable option but not the only one.

### 5.7 Inference (infer/se/*)

| Variant | Description | Rationale |
|---------|-------------|-----------|
| hc1 | HC1 robust SE | Stata areg default |
| cluster_market | Cluster at market level (100 clusters) | Natural clustering given market-level generated regressors |
| cluster_time | Cluster at year_month level (~176 clusters) | Addresses temporal serial correlation |
| cluster_market_time | Two-way market x time | Addresses both sources of dependence |

**Why clustering matters especially here**: The second stage uses generated regressors (predicted FE from first stage), which are constant within mkt_t cells. This creates strong within-cell correlation. Clustering at the market or time level is important for valid inference. The baseline Stata areg SE treat each individual observation as independent, which likely understates uncertainty.

---

## 6. What is Excluded (and Why)

### 6.1 Not in core: Alternative outcomes / treatments (explore/*)

The paper presents 4 outcome variables, which we treat as separate baseline groups. We do not add further alternative outcomes (e.g., level changes instead of log changes, or median wage growth) since these would change the outcome concept.

### 6.2 Not in core: Heterogeneity (explore/heterogeneity/*)

The paper does not present subgroup analyses as headline results. We could explore the EE-wage relationship by sex, race, age group, or education, but these are not part of the paper's revealed search space.

### 6.3 Not in core: Alternative estimators (explore/estimand/*)

- IV estimation (instrumenting actual EE transitions with predicted EE) would change the estimand.
- The paper explicitly takes a predictive/correlational approach, not a causal identification approach. We respect this framing.

### 6.4 Not in core: Sensitivity analysis (sens/*)

Formal sensitivity to unobserved confounding (Oster bounds, etc.) is not part of the core surface but could be added to the diagnostics plan.

---

## 7. Constraints and Guardrails

### 7.1 Control-count envelope

- **Second-stage regressors**: min = 1 (EE only), max = 7 (all flows + EE interaction). Always includes year_month_num.
- **First-stage controls**: e_controls has 6 variables + lag(depvar); u_controls has 1 variable (lagstate). Variants range from 1 to 7+ variables.

### 7.2 Linkage constraints

- First-stage and second-stage are **linked**: changing first-stage controls changes the generated regressors.
- e_controls are shared across all employed-eligible first-stage regressions (EE, EU, EN, wage growth).
- u_controls are shared across all unemployed-eligible first-stage regressions (UE, NE, UR).
- When varying first-stage controls, changes apply jointly across all regressions of the same eligibility type.

### 7.3 Feasibility constraints

- Market definition changes require re-running the entire first-stage FE extraction, which is computationally expensive (~10M observations).
- Finer market definitions (adding state) may create very thin mkt_t cells (few or zero observations per cell), making FE extraction unreliable.
- Time FE absorption in the second stage is feasible (~176 time periods, 100 markets) but substantially reduces variation.

---

## 8. Budgets and Sampling

### 8.1 Budget per baseline group

**Target**: ~100--120 specifications per baseline group.

**Breakdown (approximate)**:

| Component | Count | Notes |
|-----------|-------|-------|
| Baseline flow specs | 9 | All 9 Table 1 specs enumerated |
| First-stage control variants | 12 | 4 variants x 3 anchor specs |
| Sample variants | 15 | 5 variants x 3 anchor specs |
| FE variants | 9 | 3 variants x 3 anchor specs |
| Weight variants | 6 | 2 variants x 3 anchor specs |
| Market definition variants | 6 | 2 variants x 3 anchor specs |
| Inference variants | 12 | 4 variants x 3 anchor specs |
| Cross-axis high-value | ~20 | Selected two-axis combos |
| **Total** | **~89 + 20 = ~109** | |

### 8.2 Sampling plan

**Full enumeration is feasible.** The specification universe is small enough to enumerate completely. No random sampling is needed.

The key simplification: we run one-axis-at-a-time RC from 3 anchor specs (Spec 1, Spec 5, Spec 6), plus all 9 flow specs as the baseline battery, plus a small number of cross-axis combinations.

### 8.3 Total across all groups

4 baseline groups x ~100 specs = ~400 total. However, many specifications share first-stage FE extractions (the first stage only needs to be re-run for first-stage control variants and market definition variants, not for second-stage-only changes). This makes the computational cost much lower than 400 independent regressions.

---

## 9. Diagnostics Plan (Not Part of Core)

| Diagnostic | Scope | Rationale |
|-----------|-------|-----------|
| Generated regressors SE correction (Murphy-Topel) | baseline_group | The two-stage procedure uses predicted FE from first stage. Standard SE in the second stage do not account for first-stage estimation uncertainty. This is a well-known problem that the paper does not address. |
| Serial correlation in second-stage residuals (Wooldridge test) | baseline_group | Markets are observed over ~176 time periods. Serial correlation in the second-stage residuals would require HAC-type corrections. |

---

## 10. Key Implementation Notes for Spec Search Agent

1. **First-stage extraction must be cached**: The first-stage FE extraction is the most expensive computation. Cache the extracted FE for the baseline first-stage controls and only re-extract when first-stage controls or market definitions change.

2. **Generated regressor problem**: The second-stage SE are inconsistent due to generated regressors. Report standard SE as the baseline (matching the paper) but flag this as a known issue in the diagnostics.

3. **Eligibility overlap issue**: As noted in the replication report, the do-file applies individual eligibility restrictions after extracting mkt_t-level FE, but EZeligible and UZeligible are mutually exclusive. The replication resolved this by mapping FE to all observations within each mkt_t cell. Maintain this resolution consistently across all specs.

4. **Hourly wage sample restriction**: For G3 and G4 (hourly wage outcomes), the sample is further restricted to lagphr==1 (paid hourly). This changes the EZeligible and DWeligible definitions. Ensure this is applied consistently.

5. **Job stayers restriction (Spec 8)**: This is a sample restriction within a specific flow specification, not a standalone sample variant. It applies only when all flows are included in the second stage.

6. **Interaction term (Spec 9)**: xee_i = xee * eetrans_i interacts the market-level EE rate with the individual's own EE transition indicator. This tests whether the EE-wage relationship differs for job movers vs. stayers.
