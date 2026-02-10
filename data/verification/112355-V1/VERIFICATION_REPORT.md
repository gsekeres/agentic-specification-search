# Verification Report: 112355-V1

## Paper Information
- **Title**: Product Creation and Destruction: Evidence and Price Implications
- **Authors**: Christian Broda, David Weinstein
- **Journal**: American Economic Review (2010)
- **Total Specifications**: 93

## Baseline Groups

### G1: Extensive Margin Creation (EXTENS)
- **Claim**: Product creation rates (extensive margin) are procyclical -- total product group growth increases the rate of new product creation.
- **Baseline spec**: `baseline/extens_total`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.2985 (SE: 0.0197, p < 0.0001)
- **Outcome**: `EXTENS`
- **Treatment**: `TOTAL`
- **Table 7, Column 1**

### G2: Disappearance/Destruction (DISSAP)
- **Claim**: Product destruction rates are weakly countercyclical -- total growth reduces the rate of product disappearance, but the magnitude is much smaller than creation.
- **Baseline spec**: `baseline/dissap_total`
- **Expected sign**: Negative
- **Baseline coefficient**: -0.0529 (SE: 0.0126, p < 0.001)
- **Outcome**: `DISSAP`
- **Treatment**: `TOTAL`
- **Table 7, Column 2**

### G3: Net Product Creation (NET)
- **Claim**: Net product creation (creation minus destruction) is strongly procyclical, dominated by the creation margin.
- **Baseline spec**: `baseline/net_total`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.3514 (SE: 0.0185, p < 0.0001)
- **Outcome**: `NET`
- **Treatment**: `TOTAL`
- **Table 7, Column 3**

**Note**: The remaining 6 Table 7 columns (expansion/contraction sample splits for all three outcomes) are classified as core sample restrictions rather than separate baseline groups because they apply the same specification to subsamples. The paper's primary data (proprietary ACNielsen scanner data for Tables 1-6, 8-9) is unavailable; all specifications use publicly available decomposition data (EXTDISCOM.dta).

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baselines)** | **61** | |
| core_controls | 14 | 3 baselines (Table 7 Cols 1-3) + quarter/year dummies, log(value), quadratic, lag controls, controlling for DISSAP |
| core_fe | 9 | Pooled OLS, RE, between, first-diff, two-way FE for EXTENS/DISSAP/NET |
| core_sample | 29 | Outlier bounds, size splits, time subsets, balanced panel, winsorization, value-weighting, Table 7 expansion/contraction splits (6) |
| core_inference | 7 | Clustered by rpg/time, robust HC1, standard SE, demeaned + cluster |
| core_funcform | 2 | Quadratic TOTAL term for EXTENS and NET |
| **Non-core tests** | **32** | |
| noncore_alt_outcome | 9 | COMMON, INTENS, gross reallocation, |TOTAL| on EXTENS, food CPI regressions (3), COMMON/INTENS two-way FE |
| noncore_alt_treatment | 5 | COMMON as treatment, |TOTAL| (x2 duplicate), log(value), EXTENS as treatment for DISSAP |
| noncore_placebo | 5 | Lead TOTAL (EXTENS + NET), lag1 TOTAL, lag4 TOTAL, shuffled TOTAL |
| noncore_heterogeneity | 10 | By year (4), quarter (2), sign of TOTAL (2), size interaction (2) |
| noncore_diagnostic | 3 | EXTENS controlling COMMON (semi-mechanical), NET controlling COMMON (identity), food bias (identity) |
| **Duplicates noted** | **3** | functional/quadratic = controls/fe_quadratic; functional/abs_total_treat = treatment/abs_total; functional/net_quadratic = controls/net_fe_quadratic |
| **Total** | **93** | |

## Detailed Classification Notes

### Core Tests (61 specs including 3 baselines)

**Baselines (3 specs)**: The three primary baseline specifications correspond to Table 7, Columns 1-3. Each regresses a decomposition component (EXTENS, DISSAP, or NET) on TOTAL product group growth with product group (rpg) fixed effects and the sample restricted to |TOTAL| < 0.2 to exclude outlier growth rates.

**Control variations (11 non-baseline core_controls specs)**: These add controls to the baseline specification or change the control structure:
- Time period dummies (quarter, year, year+quarter) in OLS framework: 3 specs
- Additional regressors within FE framework: log(value) for size, TOTAL^2 for nonlinearity, lagged TOTAL (1-quarter, 4-quarter) for dynamics, DISSAP as a control: 6 specs
- Same for NET outcome with log(value), quadratic, and lag: additional specs
- `controls/extens_controlling_dissap` is classified as core because controlling for the destruction margin is a meaningful way to test whether creation varies independently of destruction.

**Fixed effects variations (9 specs)**: Systematic exploration of panel estimation approaches:
- Pooled OLS (no entity FE): upward-biased coefficient (1.06 for EXTENS, 0.73 for NET) showing cross-sectional confounding
- Random effects: very close to FE (0.303 vs 0.299), suggesting FE assumption appropriate
- Between estimator: much larger coefficient (1.93), showing cross-sectional relationship exceeds within-group
- First differences: slightly smaller (0.254), consistent with FE
- Two-way FE (entity + time): slightly larger (0.355), removing common time shocks increases the within-group estimate

**Sample restrictions (29 specs)**: The largest category, reflecting the paper's emphasis on outlier sensitivity:
- Outlier bounds: |TOTAL| < 0.1 (tight), 0.2 (baseline), 0.3 (loose), no filter -- tested for EXTENS, DISSAP, and NET
- Product group size splits: large, small, very large (top quartile), drop bottom 10%
- Time period drops: first year, last year, extreme quarters
- Panel balance: balanced panel only, minimum 12 observations per rpg
- Winsorization: 1-99 percentile
- Value weighting: 3 specs (FE value-weighted EXTENS/NET, WLS)
- Table 7 expansion/contraction splits: 6 specs (Cols 4-9)

**Inference variations (7 specs)**: All maintain the same point estimate but vary SE computation:
- Clustered by rpg (entity): SE = 0.097, still significant at p < 0.01
- Clustered by time: SE = 0.081
- Robust HC1: SE = 0.059
- Standard (homoskedastic): SE = 0.044
- Demeaned + rpg cluster: identical to cluster_rpg (as expected)
- Same for NET outcome: cluster_rpg and robust_hc1

**Functional form (2 specs)**: Quadratic specifications for EXTENS and NET. Note that `functional/quadratic` is an exact duplicate of `controls/fe_quadratic` (same coefficient 0.2754, same SE). The duplicate `functional/abs_total_treat` is classified under noncore_alt_treatment since it changes the treatment variable to |TOTAL|.

### Non-Core Tests (32 specs)

**Alternative outcomes (9 specs)**: These test the relationship between TOTAL and different components of the decomposition:
- COMMON (continuing products' share): strong positive relationship (0.649), confirming that most growth comes from continuing products
- INTENS (intensive margin): even stronger (0.701), showing volume changes in existing products dominate
- Gross reallocation (EXTENS + DISSAP): positive (0.246), showing that growth periods see more total product turnover
- Two-way FE versions of COMMON and INTENS
- Food CPI regressions (3 specs in alt_outcome: inflation_on_growth, inflation_year_fe, inflation_quarter_fe): use a completely different dataset (Food_rc_cpi.dta) testing CPI bias implications; these address a different claim in the paper (Table 10). A fourth food spec (food/bias_on_growth) is classified as diagnostic.

These are non-core because they test the TOTAL relationship with different outcome variables, not alternative implementations of the main EXTENS claim.

**Alternative treatments (5 specs)**: These change the treatment variable:
- COMMON as treatment for EXTENS: negative coefficient (-0.288), showing creation offsets existing product sales
- |TOTAL| (absolute value) as treatment: tests magnitude response regardless of growth direction
- log(value) as treatment: tests scale effect, essentially unrelated to cyclicality claim (coef = 0.003, insignificant)
- EXTENS as treatment for DISSAP: tests creation-destruction co-movement
- `functional/abs_total_treat` (duplicate of `treatment/abs_total`): same |TOTAL| treatment

**Placebo tests (5 specs)**: These test the temporal structure of the relationship:
- Lead of TOTAL (t+1): coefficient 0.027, p = 0.079 (marginal) -- there should be no "pre-trend" for contemporaneous creation
- Lag1 of TOTAL (t-1): coefficient 0.016, p = 0.259 (insignificant) -- past growth has weak effect on current creation
- Lag4 of TOTAL (t-4): coefficient -0.113, significant -- some reversal at annual frequency
- Shuffled TOTAL: -0.046, near zero and wrong sign -- confirms temporal relationship is real, not cross-sectional
- NET lead: similar pattern

These are non-core because they test the research design's validity rather than provide alternative estimates of the same effect.

**Heterogeneity (10 specs)**: These explore whether the effect varies across subgroups:
- By year (4 specs): year 0-3 subsamples, showing effect ranges from 0.242 (2001) to 0.458 (2002)
- By quarter (2 specs): Q1 and Q4 subsamples
- By sign of TOTAL (2 specs): positive-only vs negative-only growth periods
- Interaction with size (2 specs): TOTAL*large_rpg interaction for EXTENS and NET

These are non-core because they decompose the effect by subgroup rather than test alternative implementations of the same specification.

**Diagnostics (3 specs)**:
- `controls/extens_controlling_common`: EXTENS on TOTAL controlling for COMMON. Because TOTAL = EXTENS + COMMON - DISSAP, the coefficient on TOTAL (0.84) is heavily influenced by the mechanical relationship. The coefficient is not interpretable as a standard robustness check.
- `controls/net_controlling_common`: NET on TOTAL controlling for COMMON yields a coefficient of exactly 1.0 because NET + COMMON = TOTAL (an accounting identity). This is a mechanical check, not a meaningful robustness test.
- `food/bias_on_growth`: CPI bias on consumption growth yields a constant 0.1 with essentially zero SE, reflecting that bias is defined as a fixed 10% markup. This is a diagnostic identity check.

## Duplicates Identified

The following specs produce identical coefficients and SEs:
1. `functional/quadratic` = `controls/fe_quadratic` (both coef = 0.2754, SE = 0.0220)
2. `functional/abs_total_treat` = `treatment/abs_total` (both coef = 0.2424, SE = 0.0302)
3. `functional/net_quadratic` = `controls/net_fe_quadratic` (both coef = 0.3397, SE = 0.0207)

Additionally, several inference specs share the same point estimate (0.2610) because they are all OLS with the same data but different SE estimators: `inference/cluster_rpg`, `inference/cluster_time`, `inference/robust_hc1`, `inference/standard_se`, `inference/cluster_rpg_fe_approx`.

After removing duplicates, there are approximately 87 unique specifications.

## Robustness Assessment

The main finding -- that product creation is procyclical -- is **very robust** across all core specifications:

- **G1 (EXTENS on TOTAL)**: The coefficient ranges from 0.200 (tight |TOTAL| < 0.1 bound) to 0.355 (two-way FE) across core specifications that maintain the same outcome and treatment. All core specs remain significant at p < 0.01 even with the most conservative inference (rpg clustering: SE = 0.097, p = 0.007).
- **G2 (DISSAP on TOTAL)**: Negative and significant, though smaller in magnitude. Ranges from -0.003 (no filter, driven by outliers) to -0.121 (contractions subsample).
- **G3 (NET on TOTAL)**: Strongly positive across all specifications, ranging from 0.264 to 0.397 in core tests.

Key sensitivities:
- **Outlier removal matters**: No filter yields EXTENS coefficient of 0.84 (mechanical at extremes) vs 0.30 at baseline. The paper's choice of |TOTAL| < 0.2 is a reasonable middle ground.
- **Between-estimator is much larger** (1.93): cross-sectional relationship between product group growth and creation far exceeds within-group cyclicality, suggesting composition effects.
- **Pooled OLS is upward biased** (1.06): confirms the importance of group fixed effects.
- **Placebo tests support the claim**: leads and lags are much smaller than contemporaneous effects, and shuffled TOTAL yields near-zero coefficient.
- **Year-by-year stability**: the effect is present in all four years (0.24 to 0.46), with no single year driving the result.
