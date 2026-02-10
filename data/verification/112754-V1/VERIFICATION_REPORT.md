# Verification Report: 112754-V1

## Paper Information
- **Title**: Search, Liquidity, and the Dynamics of House Prices and Construction
- **Authors**: Allen Head, Huw Lloyd-Ellis, and Hongfei Sun
- **Journal**: American Economic Review (2014), Vol. 104, No. 4
- **Total Specifications**: 65

## Baseline Groups

### G1: Price Equation -- Income Effect (Primary Result)
- **Claim**: House prices respond positively to city-specific income shocks in a Panel VAR system.
- **Baseline spec**: `baseline/price_eq`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.2909 (SE: 0.0398, t=7.32, p < 0.0001)
- **Outcome**: `price`
- **Treatment**: `income_L1`
- **N**: 2544, **R2**: 0.943

### G2: Wage Equation -- Income Persistence
- **Claim**: Income exhibits strong persistence (near unit root), driving the dynamic responses in the model.
- **Baseline spec**: `baseline/wage_eq`
- **Expected sign**: Positive
- **Baseline coefficient**: 1.1081 (SE: 0.0209, t=53.01, p < 0.0001)
- **Outcome**: `wage`
- **Treatment**: `income_L1`
- **N**: 2544, **R2**: 0.832

### G3: Sales Growth Equation -- Income Effect
- **Claim**: Sales growth responds positively to income shocks, reflecting search-friction-driven market activity.
- **Baseline spec**: `baseline/salesgrow_eq`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.3870 (SE: 0.1281, t=3.02, p=0.003)
- **Outcome**: `salesgrow`
- **Treatment**: `income_L1`
- **N**: 2544, **R2**: 0.179

### G4: Construction Rate Equation -- Income Effect
- **Claim**: Construction rates respond positively to income shocks, capturing housing supply dynamics.
- **Baseline spec**: `baseline/conrate_eq`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.0365 (SE: 0.0048, t=7.65, p < 0.0001)
- **Outcome**: `conrate`
- **Treatment**: `income_L1`
- **N**: 2544, **R2**: 0.653

### G5: Population Growth Equation -- Income Effect
- **Claim**: Population growth responds positively to income shocks, capturing migration responses.
- **Baseline spec**: `baseline/popgrow_eq`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.0736 (SE: 0.0224, t=3.29, p=0.001)
- **Outcome**: `popgrow`
- **Treatment**: `income_L1`
- **N**: 2544, **R2**: 0.688

**Note**: All 5 baseline equations are jointly estimated in the Panel VAR(2) system, but the price equation (G1) is the primary result of interest. The specification search predominantly varies the price equation. G2-G5 baselines are included because they are integral components of the full VAR system used to calibrate the structural model.

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baselines)** | **54** | |
| core_baseline | 5 | The 5 equations of the baseline VAR(2) system |
| core_lag | 3 | Alternative lag lengths: VAR(1), VAR(3), VAR(4) |
| core_model | 4 | Pooled OLS, two-way FE, random effects, first differences |
| core_inference | 3 | City-clustered SE, robust HC SE, HAC kernel SE |
| core_controls | 8 | Drop variables (3), bivariate, trivariate (2), add variables (2) |
| core_funcform | 1 | Growth rates transformation |
| core_sample | 25 | Geographic subsamples (7), time periods (6), leave-one-out (6), outliers (4), construction subsample, balanced panel |
| core_measurement | 4 | Alternative depreciation factors (0.90, 0.93, 0.98, 0.99) |
| core_data | 1 | Original 2014 data version |
| **Non-core tests** | **11** | |
| noncore_alt_outcome | 1 | Rent replacing price |
| noncore_alt_treatment | 10 | Price effect on other vars (4), L2 income coefficients (5), construction wage as treatment (1) |

## Detailed Classification Notes

### Core Tests (54 specs including 5 baselines)

**Baselines (5 specs)**: The five equations of the Panel VAR(2) system, each regressing one endogenous variable on two lags of all five variables, with city FE (LSDV) and time FE (demeaned via year dummies). All five perfectly replicate the paper's reported estimates.

**Lag structure (3 specs)**: VAR(1), VAR(3), and VAR(4) for the price equation. The VAR(1) result is strikingly different -- the income coefficient drops to essentially zero (0.0005, p=0.99), suggesting the second lag is crucial for capturing the income-to-price transmission. VAR(3) and VAR(4) give similar results to VAR(2) (0.315 and 0.244 respectively).

**Estimation method (4 specs)**: Pooled OLS (0.321), two-way FE (0.291, identical to baseline because time FE are already removed by demeaning), random effects (0.321), and first differences (0.282). All produce very similar results, suggesting the income-price relationship is not driven by the estimation approach.

**Inference (3 specs)**: City-clustered, robust HC, and HAC kernel standard errors. All maintain the same point estimate (0.291). HAC SEs are substantially larger (0.104 vs 0.040 baseline) but the coefficient remains significant at the 1% level.

**Variable set (8 specs)**: Dropping variables (sales, construction, popgrow), bivariate income-price VAR, two trivariate variants, and adding construction wage or construction labor. Coefficients range from 0.266 to 0.354. The result is robust to the set of included endogenous variables.

**Functional form (1 spec)**: Growth rates transformation yields coefficient 0.333, similar to the level specification.

**Sample restrictions (25 specs)**: The largest category:
- *Geographic subsamples (7)*: Coastal cities give a notably smaller coefficient (0.110, p=0.109), while inland/non-coastal cities give larger coefficients (0.320-0.378). Sunbelt (0.304) and non-sunbelt (0.235) show moderate variation. Large cities (0.209) vs small cities (0.318) also differ.
- *Time periods (6)*: Pre-2000 (0.161), post-1990 (0.186), pre-crisis (0.228), shorter 1985-2005 (0.150), drop last 2 years (0.228), and early-only 1982-1994 (0.065, p=0.378 -- insignificant). The income-price relationship clearly strengthened in the later period.
- *Leave-one-out (6)*: Dropping each of the 6 largest MSAs (NYC, LA, Chicago, Philly, Dallas, Miami). Coefficients range from 0.288 to 0.301 -- highly stable.
- *Outlier treatment (4)*: Winsorizing at 1% (0.145), 5% (0.133); trimming at 1% (0.115, p=0.026), 5% (0.043, p=0.639). Extreme observations substantially drive the baseline magnitude.
- *Construction subsample (1)*: 98 cities with construction data (0.322).
- *Balanced panel (1)*: Confirms the baseline panel is already balanced (0.291, identical).

**Measurement (4 specs)**: Alternative housing stock depreciation factors (0.90, 0.93, 0.98, 0.99 vs baseline 0.96). All give virtually identical results (0.290-0.291), confirming the depreciation rate assumption is inconsequential.

**Data version (1 spec)**: Original 2014 data gives identical results to the 2016 update (0.291).

### Non-Core Tests (11 specs)

**Alternative outcome (1 spec)**: `robust/outcome/rent` replaces house prices with rents. The coefficient is 0.112 (significant at p=0.019). This tests a fundamentally different relationship (income effect on rents rather than house prices) and is classified as non-core because it changes the outcome variable.

**Alternative treatment -- price effects (4 specs)**: These extract the L1 price coefficient from each non-price equation of the baseline VAR (wage, sales growth, construction rate, population growth). These are different parameters from different equations testing how prices affect other variables, not how income affects prices. The coefficients are: wage (0.042), salesgrow (-0.703), conrate (-0.002, insig), popgrow (-0.031).

**Alternative treatment -- L2 income (5 specs)**: These extract the second-lag income coefficient from each of the five baseline equations. These are different parameters that capture the delayed/reversal component of the income effect. All are negative (ranging from -0.045 to -0.354), consistent with partial mean-reversion. They test the dynamic structure rather than providing alternative estimates of the primary L1 effect.

**Alternative treatment -- construction wage (1 spec)**: `robust/vars/cwage_replaces_income` uses construction wages instead of general income as the treatment variable. Coefficient is 0.030 (p=0.071). This tests a different treatment variable entirely.

## Duplicates Identified

- `robust/sample/pre_crisis` and `robust/sample/drop_last2` produce identical coefficients (0.228) and sample sizes (2332), as expected since the full sample ends in 2007 and dropping 2 years yields 1982-2005.

After accounting for this duplication, there are 64 effectively unique specifications.

## Robustness Assessment

The primary finding -- that city-specific income shocks increase house prices -- is **moderately robust** across specifications:

**Strong robustness:**
- **Estimation method**: LSDV, pooled OLS, RE, FD, two-way FE all give coefficients in the range 0.28-0.32.
- **Variable selection**: Dropping variables, adding variables, bivariate and trivariate specifications give 0.27-0.35.
- **Leave-one-out cities**: Dropping any single large metro gives 0.29-0.30.
- **Inference**: Clustered, robust, and HAC standard errors all maintain significance (p < 0.01).
- **Depreciation rate**: Negligible sensitivity (0.290-0.291 across all values).
- **Data version**: Identical results with original and updated data.

**Notable sensitivities:**
- **Lag structure**: VAR(1) produces a near-zero coefficient (0.0005). The second lag is essential.
- **Time period**: The early period (1982-1994) gives a small, insignificant coefficient (0.065, p=0.38). The income-price relationship appears to have strengthened over time, possibly reflecting increasing housing market financialization.
- **Outlier treatment**: Winsorizing at 1-5% reduces the coefficient by 50-85% (to 0.13-0.15). Trimming at 5% renders it insignificant (0.043, p=0.64). This is the most important sensitivity -- extreme observations substantially drive the baseline magnitude.
- **Geography**: Coastal cities show a much smaller effect (0.11, p=0.11) compared to inland/non-coastal cities (0.32-0.38). This may reflect supply elasticity differences.
- **City size**: Large cities show a smaller effect (0.21) than small cities (0.32).

**Overall**: The direction of the effect is robust (positive income-to-price relationship in 53 of 54 core specs; the one exception is VAR(1) at effectively zero). The magnitude is sensitive to sample composition (time period, geography, outlier treatment), with the baseline estimate of 0.29 representing an upper range driven partly by extreme observations and the 2000s housing boom period.
