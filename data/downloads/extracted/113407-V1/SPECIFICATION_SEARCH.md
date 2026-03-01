# Specification Search: 113407-V1

## Paper
- **Title**: Living Arrangements, Doubling Up, and the Great Recession: Was This Time Different?
- **Authors**: Hoynes (2014)
- **Journal**: AER Papers & Proceedings, 104(5), 107-112
- **Design**: Panel fixed effects (state-year panel, 1980-2013)

## Surface Summary
- **Baseline groups**: 2 (G1: young adult living alone 18-30, G2: household size non-elderly)
- **G1 budget**: 55 max core specs
- **G2 budget**: 25 max core specs
- **Seed**: 113407
- **Controls pool**: None (no covariates beyond state and year FE)

## Baseline Results

### G1: Fraction Young Adults (18-30) Living Alone
- **Outcome**: `myadult_aloneall_1830`
- **Treatment**: `urate` (state unemployment rate)
- **Coefficient**: -0.403090
- **SE**: 0.207292 (clustered at state)
- **p-value**: 0.057465
- **N**: 1734
- **Interpretation**: A 1 pp increase in state unemployment rate is associated with a 0.4031 decrease in the fraction of young adults living alone.

### G2: Average Household Size (Non-Elderly)
- **Outcome**: `h_numpers_noneld`
- **Treatment**: `urate` (state unemployment rate)
- **Coefficient**: 2.127034
- **SE**: 0.958667 (clustered at state)
- **p-value**: 0.031073
- **N**: 1734
- **Interpretation**: A 1 pp increase in state unemployment rate is associated with a 2.1270 increase in average household size.

## Execution Summary

| Metric | G1 | G2 | Total |
|--------|----|----|-------|
| Planned specs | 37 | 16 | 53 |
| Successful | 37 | 16 | 53 |
| Failed | 0 | 0 | 0 |
| Same sign as baseline | 33/37 | 15/16 | 48/53 |
| Significant (p<0.05) | 9/37 | 9/16 | 18/53 |
| Significant (p<0.10) | 24/37 | 9/16 | 33/53 |

## Inference Variants

| Variant | Group | Coefficient | SE | p-value | N |
|---------|-------|------------|-----|---------|---|
| infer/se/hc/hc1 | G1 | -0.403090 | 0.097685 | 0.000039 | 1734 |
| infer/se/cluster/region | G1 | -0.403090 | 0.227908 | 0.175103 | 1734 |
| infer/se/hac/driscoll_kraay | G1 | -0.446430 | 0.107834 | 0.000035 | 1734 |
| infer/se/hc/hc1 | G2 | 2.127034 | 0.345163 | 0.000000 | 1734 |
| infer/se/hac/driscoll_kraay | G2 | 2.186392 | 0.385606 | 0.000000 | 1734 |

## Spec IDs Executed

### G1 Core Specs
- `baseline`: OK (coef=-0.403090, p=0.057465, N=1734)
- `baseline__table1_panA_col4`: OK (coef=-0.369182, p=0.060233, N=1734)
- `baseline__table1_panA_col5`: OK (coef=-0.426822, p=0.051324, N=1734)
- `design/panel_fixed_effects/estimator/first_difference`: OK (coef=0.140364, p=0.346555, N=1683)
- `rc/sample/age_group/18_24`: OK (coef=-0.369182, p=0.060233, N=1734)
- `rc/sample/age_group/25_30`: OK (coef=-0.426822, p=0.051324, N=1734)
- `rc/form/outcome/level`: OK (coef=-0.403090, p=0.057465, N=1734)
- `rc/form/outcome/log`: OK (coef=-0.832440, p=0.052755, N=1734)
- `rc/form/treatment/log_urate`: OK (coef=-0.026765, p=0.040598, N=1734)
- `rc/form/treatment/recession_specific`: OK (coef=-0.267631, p=0.099768, N=1734)
- `rc/form/treatment/urate_80`: OK (coef=-0.267631, p=0.099768, N=1734)
- `rc/form/treatment/urate_rest`: OK (coef=-0.365414, p=0.144958, N=1734)
- `rc/form/treatment/urate_07`: OK (coef=-0.728045, p=0.067350, N=1734)
- `rc/fe/drop_state_fe`: OK (coef=-1.027150, p=0.013823, N=1734)
- `rc/fe/add_state_trend`: OK (coef=-0.175635, p=0.137054, N=1734)
- `rc/fe/add_region_year`: OK (coef=-0.304183, p=0.038611, N=1734)
- `rc/fe/add_division_year`: OK (coef=-0.295846, p=0.139308, N=1734)
- `rc/weights/unweighted`: OK (coef=-0.286722, p=0.062506, N=1734)
- `rc/sample/time/pre_2007`: OK (coef=-0.299562, p=0.132060, N=1377)
- `rc/sample/time/post_1990`: OK (coef=-0.092123, p=0.568558, N=1224)
- `rc/sample/time/1990_2013`: OK (coef=-0.092123, p=0.568558, N=1224)
- `rc/sample/time/drop_recession_years`: OK (coef=-0.412096, p=0.048337, N=1275)
- `rc/sample/outliers/trim_urate_1_99`: OK (coef=-0.450662, p=0.034486, N=1700)
- `rc/sample/outliers/drop_high_urate_states`: OK (coef=-0.332580, p=0.026755, N=1292)
- `rc/sample/outliers/drop_small_states`: OK (coef=-0.401815, p=0.071245, N=1292)
- `rc/joint/outcome_age/18_24_log_urate`: OK (coef=-0.022039, p=0.073004, N=1734)
- `rc/joint/outcome_age/25_30_log_urate`: OK (coef=-0.028013, p=0.049377, N=1734)
- `rc/joint/outcome_age/18_24_unweighted`: OK (coef=-0.276574, p=0.200057, N=1734)
- `rc/joint/outcome_age/25_30_unweighted`: OK (coef=-0.181600, p=0.091038, N=1734)
- `rc/joint/form_time/recession_pre2007`: OK (coef=-0.222446, p=0.204925, N=1377)
- `rc/joint/form_time/recession_post1990`: OK (coef=0.283249, p=0.189287, N=1224)
- `rc/joint/form_time/log_urate_pre2007`: OK (coef=-0.022028, p=0.096114, N=1377)
- `rc/joint/form_time/log_urate_post1990`: OK (coef=0.001063, p=0.917369, N=1224)
- `rc/joint/outcome_age/18_24_recession`: OK (coef=-0.264713, p=0.127140, N=1734)
- `rc/joint/outcome_age/25_30_recession`: OK (coef=-0.312535, p=0.046972, N=1734)
- `rc/joint/form_time/log_urate_no_recession`: OK (coef=-0.024297, p=0.045387, N=1275)
- `rc/joint/form_time/log_urate_1990_2013`: OK (coef=0.001063, p=0.917369, N=1224)

### G2 Core Specs
- `baseline`: OK (coef=2.127034, p=0.031073, N=1734)
- `baseline__table1_panA_col2`: OK (coef=0.425183, p=0.109214, N=1734)
- `design/panel_fixed_effects/estimator/first_difference`: OK (coef=-0.154637, p=0.707787, N=1683)
- `rc/form/outcome/h_numfams_noneld`: OK (coef=0.425183, p=0.109214, N=1734)
- `rc/form/treatment/recession_specific`: OK (coef=2.414533, p=0.000782, N=1734)
- `rc/form/treatment/log_urate`: OK (coef=0.130348, p=0.040459, N=1734)
- `rc/form/outcome/log`: OK (coef=0.575596, p=0.029265, N=1734)
- `rc/fe/drop_state_fe`: OK (coef=3.452865, p=0.144857, N=1734)
- `rc/fe/add_state_trend`: OK (coef=0.703761, p=0.152001, N=1734)
- `rc/fe/add_region_year`: OK (coef=1.811944, p=0.001768, N=1734)
- `rc/weights/unweighted`: OK (coef=2.212656, p=0.000048, N=1734)
- `rc/sample/time/pre_2007`: OK (coef=2.189682, p=0.015591, N=1377)
- `rc/sample/time/post_1990`: OK (coef=0.010002, p=0.986402, N=1224)
- `rc/sample/time/drop_recession_years`: OK (coef=2.009603, p=0.025331, N=1275)
- `rc/sample/outliers/trim_urate_1_99`: OK (coef=2.031660, p=0.032487, N=1700)
- `rc/sample/outliers/drop_high_urate_states`: OK (coef=0.860130, p=0.127838, N=1292)

## Deviations from Surface

- None. All surface specs executed as planned.

## Software Stack
- Python 3.12.7
- pyfixest 0.40.1
- pandas 2.2.3
- numpy 2.1.3
- statsmodels 0.14.6
