# Verification Report: 112785-V1

## Paper Information
- **Title**: Demand and Defective Growth Patterns
- **Authors**: Sandile Hlatshwayo, Michael Spence
- **Journal**: AER: Papers & Proceedings (2014)
- **Total Specifications**: 111

## Baseline Groups

### G1: VA Percent Growth, Recovery 2009-2012 (Primary)
- **Claim**: The tradable sector had higher value-added percent growth than the nontradable sector during the post-crisis recovery (2009-2012), accounting for more than half of gross VA gains despite being approximately one-third of the economy.
- **Baseline spec**: spec 1 (OLS) and spec 4 (Welch t-test)
- **Expected sign**: Positive
- **Baseline coefficient**: 16.34 (SE: 15.38, p = 0.288, N = 34, R2 = 0.027)
- **Outcome**: `va_pct_change`
- **Treatment**: `tradable_indicator`

### G2: VA Absolute Change, Recovery 2009-2012
- **Claim**: The tradable sector had higher absolute VA change during the recovery.
- **Baseline spec**: spec 2
- **Expected sign**: Positive
- **Baseline coefficient**: -7.22 (SE: 11.76, p = 0.539, N = 34)
- **Outcome**: `va_abs_change`
- **Treatment**: `tradable_indicator`
- **Note**: The negative sign reflects that large nontradable sectors (e.g., real estate, government) dominate in absolute levels even though tradable sectors grew faster in percentage terms.

### G3: VA Log Change, Recovery 2009-2012
- **Claim**: The tradable sector had higher log VA change during the recovery.
- **Baseline spec**: spec 3
- **Expected sign**: Positive
- **Baseline coefficient**: 7.45 (SE: 7.55, p = 0.323, N = 34)
- **Outcome**: `va_log_change`
- **Treatment**: `tradable_indicator`

**Note**: The original paper presents no formal regressions or statistical tests. All results are descriptive accounting decompositions. The specification search agent constructed formal cross-sectional tests of the paper's claims. All baseline groups are statistically insignificant, reflecting the fundamental challenge of testing aggregate claims (about sector totals) via cross-industry regressions with only 34 observations.

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baselines)** | **73** | |
| baseline | 4 | Specs 1-4: OLS with VA pct/abs/log change, t-test |
| core_alt_period | 18 | Specs 5-24 (excl. 15, 16): VA growth in alternative time periods (1990-2000, 2000-2007, 2007-2009, 1990-2007, 1990-2012, etc.) |
| core_estimation | 5 | Specs 35-39: Panel regressions with year FE for VA year-over-year growth |
| core_sample | 7 | Specs 53-55, 108-109: Pure sector classification, excluding mfg sub-sectors |
| core_treatment | 4 | Specs 56-59: Conservative and liberal sector classification alternatives |
| core_loo | 24 | Specs 60-83: Leave-one-out dropping each industry |
| core_weights | 3 | Specs 84-86: WLS weighted by initial VA |
| core_funcform | 4 | Specs 16, 87-89: Annualized growth, relative growth |
| core_inference | 5 | Specs 98-100, 106-107: Mann-Whitney U tests, bootstrap, permutation |
| **Non-core tests** | **38** | |
| noncore_alt_outcome | 21 | Specs 25-34, 40-42, 90-94, 101-105: Employment outcomes, share of gains, log-level panel, productivity |
| noncore_did | 10 | Specs 43-52: DiD interaction tests for structural break |
| noncore_diagnostic | 3 | Specs 95-97: Aggregate productivity comparisons with no p-values |
| noncore_alt_period | 1 | Spec 110: Crisis decline test (different hypothesis) |
| **Total** | **111** | |

## Detailed Classification Notes

### Core Tests (73 specs including 4 baselines)

**Baselines (4 specs)**: Specs 1-4 represent the primary formalization of the paper's headline claim. Spec 1 is OLS of VA percent change (2009-2012) on a tradable indicator (coef = 16.34, p = 0.288). Spec 2 uses absolute VA change (negative coefficient, opposite to expected). Spec 3 uses log VA change. Spec 4 is a Welch t-test equivalent of spec 1. All are insignificant due to the small cross-section of 34 industry observations.

**Alternative period tests (18 specs)**: Specs 5-24 run the same cross-sectional regression across 10 different time periods (pre-boom, boom, crisis, full period, etc.), each in both percent change and annualized growth form. These directly test the paper's narrative that nontradable dominated pre-crisis (negative coefficients expected) and tradable dominated post-crisis (positive expected). Results confirm: pre-crisis coefficients are negative, recovery-period coefficients are positive. None are individually significant. Spec 15 is an exact duplicate of spec 1 (identical coefficient, SE, p-value).

**Panel regressions (5 specs)**: Specs 35-39 use annual panel data with year fixed effects, providing a different estimation approach from the cross-sectional baseline. VA year-over-year growth is regressed on the tradable indicator with year FE. Post-crisis panels (specs 37, 39) show positive tradable coefficients (3.98 and 2.85) but remain insignificant. These are core because they test the same VA growth hypothesis using an alternative econometric framework.

**Sample restrictions (7 specs)**: Specs 53-55 restrict to "pure" tradable/nontradable industries (dropping mixed classification), and specs 108-109 exclude manufacturing sub-sectors to test aggregation sensitivity. The pure-sector recovery specification (spec 53) shows a larger coefficient (35.0) but with wider confidence intervals due to smaller N (14).

**Alternative treatment definitions (4 specs)**: Specs 56-59 reclassify mixed industries as either all-nontradable (conservative) or all-tradable (liberal). Conservative classification yields negative coefficients (nontradable includes more industries); liberal classification yields positive and marginally significant coefficients (p = 0.098 for recovery). This demonstrates that sector classification is a key degree of freedom.

**Leave-one-out (24 specs)**: Specs 60-83 drop each of the 24 industries one at a time. All 24 maintain a positive coefficient, confirming the direction is not driven by any single industry. However, dropping the Auto industry (spec 76) reduces the coefficient from 16.3 to 1.5, revealing that the auto industry's strong post-crisis recovery drives most of the tradable growth advantage. No leave-one-out spec is individually significant.

**Weighted regressions (3 specs)**: Specs 84-86 use WLS weighted by initial-year VA. The recovery WLS coefficient (5.1) is smaller than the unweighted baseline (16.3), suggesting that smaller tradable industries grew faster but larger ones did not show as strong an advantage.

**Functional form (4 specs)**: Spec 16 uses annualized growth, and specs 87-89 use growth relative to the economy average. Spec 87 is numerically identical to spec 1, confirming that subtracting a constant (aggregate growth) from both tradable and nontradable groups does not change the OLS coefficient.

**Inference robustness (5 specs)**: Specs 98-100 are Mann-Whitney U nonparametric tests for VA growth in recovery, full period, and boom. Specs 106-107 are bootstrap (500 reps) and permutation (1000 permutations) tests on the baseline. All confirm insignificance: bootstrap p = 0.284, permutation p = 0.546.

### Non-Core Tests (38 specs)

**Employment outcomes (15 specs)**: Specs 25-34 (cross-sectional), 40-42 (panel), and 101-102 (Mann-Whitney) test employment growth instead of VA growth. These address a fundamentally different claim -- job creation rather than value creation. The paper emphasizes that tradable productivity (VA per worker) grew while tradable employment fell, so employment specs test a distinct hypothesis. Notably, tradable employment is significantly lower in absolute terms (spec 28: p < 0.001 for 1990-2000, spec 30: p = 0.004 for 2000-2007, spec 32: p = 0.003 for full period) and in panel form (spec 41: p < 0.001 for pre-crisis). This represents one of the paper's key findings: the tradable sector shed jobs while gaining value.

**DiD structural break tests (10 specs)**: Specs 43-52 test whether the tradable sector's relative growth advantage changed (increased) post-crisis, using a tradable x post interaction term. This tests a different claim from the baseline: not whether tradable grew faster post-crisis, but whether the tradable-nontradable differential was larger post-crisis than pre-crisis. All DiD interactions are insignificant, meaning there is no statistically detectable structural break.

**Share of growth outcomes (5 specs)**: Specs 90-94 regress each industry's share of total gross VA gains on the tradable indicator. This is a different outcome variable that captures composition rather than growth rate. The pre-boom share spec (spec 92, p = 0.014) is significant, showing that nontradable industries took a larger share of total gains.

**Log-level panel (3 specs)**: Specs 103-105 use log VA levels (not growth rates) with trend and post-crisis interactions. These test different hypotheses about trend growth and level shifts rather than the growth differential the paper focuses on. All three show large negative significant coefficients for the tradable sector, reflecting that tradable industries are smaller in absolute VA levels.

**Productivity comparisons (3 specs)**: Specs 95-97 are aggregate productivity (VA/worker) comparisons between tradable and nontradable sectors. These are purely descriptive with no p-values and serve as diagnostic checks.

**Crisis decline (1 spec)**: Spec 110 tests whether tradable declined more during the 2007-2009 crisis -- a different hypothesis from the recovery claim.

## Duplicates Identified

The following specifications produce identical coefficients and SEs:

1. **Spec 1 = Spec 15 = Spec 87**: All three produce coefficient = 16.344, SE = 15.376 for VA pct change in the recovery period. Spec 15 is labeled "alternative_periods" and spec 87 is labeled "relative_growth" but the regression is numerically identical (subtracting a constant from the dependent variable does not change OLS coefficients).

2. **Spec 1 and Spec 4**: Near-identical (same coefficient 16.344, slightly different SE due to Welch vs OLS) -- the t-test is equivalent to a two-sample mean comparison.

After removing duplicates, there are approximately 108 unique specifications.

## Robustness Assessment

The main finding -- that the tradable sector grew faster in VA terms during the 2009-2012 recovery -- receives **WEAK** support from formal statistical testing:

- **G1 (VA pct growth)**: The recovery-period coefficient is positive in 100% of leave-one-out specs (range: 1.5 to 18.8) and in 89% of all recovery-period specs across categories. However, zero core recovery-period specifications are significant at 5%. The positive direction is robust; the magnitude and significance are not.

- **G2 (VA absolute change)**: The coefficient is negative, contradicting the expected sign. This reflects the paper's claim being about percentage growth (or shares), not absolute levels.

- **G3 (VA log change)**: Positive and directionally consistent with G1 but insignificant.

Key sensitivities:

- **Auto industry drives the result**: Dropping the auto industry (spec 76) reduces the tradable coefficient from 16.3pp to 1.5pp, revealing heavy dependence on the auto sector's strong post-crisis recovery. All other leave-one-out drops have minimal impact (range 14.7 to 18.8).

- **Sector classification matters**: Conservative classification (mixed industries classified as nontradable) flips the sign negative (-322pp in recovery, spec 56). Liberal classification (mixed as tradable) yields the largest positive effect (335pp, spec 58, p = 0.098). The baseline classification is between these extremes.

- **Employment tells a different story**: Tradable employment growth is significantly lower than nontradable employment growth, particularly pre-crisis. The paper's narrative about tradable dominance applies to value-added, not employment.

- **No structural break detected**: All 10 DiD interaction specifications are insignificant, suggesting the post-crisis shift toward tradable growth was not a statistically detectable break from historical patterns.

- **Inference methods agree**: Bootstrap (p = 0.284), permutation (p = 0.546), and Mann-Whitney (p = 0.742) all confirm insignificance, ruling out that the parametric p-value is misleadingly large.

- **Fundamental limitation**: The paper makes claims about aggregate sector totals using 34 industry-level observations. Cross-sectional regressions have inherently low power for testing aggregate composition claims with small N.
