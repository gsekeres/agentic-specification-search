# Verification Report: 112533-V1

## Paper
**Title**: The Finnish Great Depression: From Russia with Love
**Authors**: Gorodnichenko, Mendoza, and Tesar
**Journal**: American Economic Review
**Method**: Cross-sectional OLS

## Baseline Groups Found

| Group | Claim | Expected Sign | Baseline Spec IDs |
|-------|-------|---------------|-------------------|
| G1 | Industries more exposed to Soviet trade experienced larger employment declines in Finland's 1990s depression | - | baseline |

The paper has a single empirical cross-sectional result (Figure 1, Panel C). The main paper contribution is a calibrated general equilibrium model; the cross-sectional regression is supporting evidence.

- **Baseline coefficient**: -14.55 (paper: -14.54)
- **Baseline SE (HC1)**: 6.44 (paper: 6.04)
- **Baseline p-value**: 0.024
- **N**: 31 Finnish manufacturing industries

## Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **87** |
| Baselines | 1 |
| Core tests (non-baseline) | 42 |
| Non-core tests | 44 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_controls | 11 | 1 baseline + 10 control addition specs |
| core_inference | 6 | Classical, HC0-HC3, bootstrap SEs |
| core_sample | 15 | 13 sample restrictions + 2 panel pooling specs |
| core_funcform | 11 | 4 functional form + 7 estimation method variants |
| noncore_alt_outcome | 20 | Different outcome years, production, VA, investment, exports, alt detrending |
| noncore_alt_treatment | 13 | 7 alternative treatment measures + 5 time-series DiD + 1 cross-country DiD |
| noncore_heterogeneity | 6 | Interaction terms with sector/size/export intensity |
| noncore_placebo | 5 | Pre-period outcomes (1989, 1990) and non-Soviet export share |

## Classification Decisions

### Core Test Classifications

**Inference variations (6 specs)**: The ols/se/* specs change only the standard error computation while keeping the same point estimate. These are straightforward core inference tests. Note that ols/se/HC1 duplicates the baseline method.

**Sample restrictions (13 specs)**: The ols/sample/* specs restrict or modify the sample while keeping the same outcome (empl_1993) and treatment (row_export_1988). These include dropping aggregate industries, dropping high-exposure outliers, restricting by sector type, dropping by size quartile, winsorizing, and trimming. All are meaningful robustness checks of the baseline claim. The consumer_goods spec (n=5) has very low power and a coefficient (-57) driven by a tiny sample.

**Panel pooling (2 specs)**: The ols/panel/* specs pool the cross-section across years 1991-1996. These keep the same treatment variable (row_export_1988) and estimate the same cross-sectional relationship, but expand the sample by using multiple outcome years. Classified as core/sample because they test the same claim with a more efficient sample design. Confidence is lower (0.8) because the change in outcome variable (empl_dev rather than empl_1993) is substantive.

**Controls (10 specs)**: The ols/controls/* specs add covariates to the baseline regression. These test whether the baseline relationship is robust to controlling for industry size, export intensity, sector composition, pre-trends, and combinations thereof. The pre_trend_1990 control is particularly informative: adding 1990 employment deviation nearly halves the coefficient (-6.72 vs -14.55), suggesting some cross-sectional variation reflects pre-existing differences.

**Functional form and estimation method (11 specs)**:
- ols/form/* (4 specs): log treatment, quadratic, rank-transformed, and standardized treatment. These change how the treatment variable enters the regression. Classified as core/funcform.
- ols/method/* (7 specs): WLS (3 variants, but wls_empl and wls_empl_share are duplicates), quantile regression (25th, 50th, 75th), and Huber robust regression. These change the estimation method while keeping the same outcome and treatment. Classified as core/funcform.
- Note: ols/form/quadratic is identical to ols/controls/su_share_sq (same coefficients and R-squared), representing a duplicate.

### Non-Core Classifications

**Alternative treatments (7 specs)**: The ols/treatment/* specs change the treatment variable itself to SU exports (rather than ROW), different base years (1989, 1990), a binary indicator, or an export-sales ratio. These test whether the result holds with different measures of Soviet trade exposure, which constitutes testing a different treatment rather than a robustness check of the baseline specification.

**Alternative outcomes (20 specs)**: The ols/outcome/* specs change the dependent variable to employment in other years (1991-1996), production, value added, investment, exports, or employment measured with different detrending methods. These test related but distinct hypotheses about the effect of Soviet trade exposure on various economic outcomes. The alternative detrending specs (uniform 1980-89, 1975-89, 1985-89) are particularly notable because they change how the outcome is constructed; the uniform and 1985-89 specs reverse the sign of the coefficient, revealing sensitivity to the detrending procedure.

**Placebo tests (5 specs)**: The ols/placebo/* specs test pre-period outcomes (employment 1989 and 1990, production 1989, investment 1989) and an alternative treatment (non-Soviet export share). These are validity checks rather than robustness tests. Notably, all pre-period placebos are statistically significant, which is problematic for the causal interpretation.

**Heterogeneity (6 specs)**: The ols/interact/* specs add interaction terms between the treatment and industry characteristics (heavy industry, high-tech, size, export intensity, consumer goods, aggregate). These test whether the effect differs across subgroups rather than testing the baseline claim itself.

**Time-series DiD (5 specs)**: The ts/did/* specs use a fundamentally different identification strategy: comparing Soviet-oriented vs non-Soviet sectors over time in a difference-in-differences framework. The unit of observation, treatment variable (DiD interaction), and outcome variables are all different. Classified as noncore_alt_treatment.

**Cross-country DiD (1 spec)**: The cross_country/did/finland_vs_sweden spec compares Finland to Sweden in a country-level DiD. This tests an entirely different claim (aggregate GDP rather than industry employment) with a different identification strategy. Classified as noncore_alt_treatment.

## Notable Issues

### 1. Significant pre-period placebos
All three pre-period employment/production/investment placebos (1989) are statistically significant, as is the 1990 employment placebo. This raises concerns about pre-existing trends: industries with high Soviet export share were already declining before the USSR collapsed. This is consistent with the finding that controlling for 1990 employment deviation halves the coefficient.

### 2. Sensitivity to detrending method
The result reverses sign when using uniform 1980-89 detrending (coef = +5.90) or 1985-89 detrending (coef = +25.84, p < 0.001). The paper's result depends on industry-specific trend periods, and the choice of detrending window is consequential.

### 3. Duplicate specifications
- ols/method/wls_empl and ols/method/wls_empl_share produce identical results (same coefficients, SE, and R-squared).
- ols/form/quadratic and ols/controls/su_share_sq produce identical results.

### 4. Small sample concerns
The baseline uses only 31 observations. Some sample restrictions reduce this further (consumer_goods: n=5, heavy_industry: n=11, empl_1996: n=17), making inference unreliable in those subsamples.

### 5. Permutation test missing
The SPECIFICATION_SEARCH.md mentions a permutation inference spec but no corresponding row appears in the CSV. This may be a gap in the specification search output.

## Recommendations

1. **Flag the detrending sensitivity**: The sign reversal under alternative detrending methods is the most substantive finding from this specification search and should be highlighted in any robustness summary.

2. **Address pre-trend concerns**: The significant placebos and the attenuation when controlling for 1990 deviations suggest the cross-sectional result may partly reflect pre-existing trends rather than the causal impact of the Soviet collapse.

3. **Remove duplicates**: Two pairs of duplicate specifications should be consolidated (wls_empl/wls_empl_share and quadratic/su_share_sq).

4. **Consider the panel specs carefully**: The pooled panel specs are borderline between core and non-core. They use a different outcome variable (empl_dev pooled across years) but test the same fundamental claim with greater statistical power.
