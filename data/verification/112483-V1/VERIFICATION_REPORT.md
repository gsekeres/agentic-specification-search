# Verification Report: 112483-V1

## Paper
**Title**: Automobiles on Steroids: Product Attribute Trade-Offs and Technological Progress in the Automobile Sector
**Authors**: Christopher R. Knittel
**Journal**: American Economic Review, December 2011
**Method**: Panel fixed effects (OLS with year and manufacturer FE)

## Baseline Groups

| Group | Claim | Baseline spec_id | Outcome | Coef | SE | p-value | N |
|-------|-------|-------------------|---------|------|----|---------|---|
| G1 | Vehicle weight reduces fuel economy (weight-MPG elasticity ~ -0.42) | baseline | lmpg | -0.4186 | 0.029 | <0.001 | 14,423 |

The paper's central empirical contribution is estimating the trade-off between vehicle attributes and fuel economy. The focal coefficient is the elasticity of MPG with respect to curb weight in a Cobb-Douglas specification with year + manufacturer FE, clustered SE by manufacturer, for passenger cars excluding outliers. This matches Table 2, Column 3 of the paper.

## Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **73** |
| Baselines | 1 |
| Core tests (non-baseline) | 58 |
| Non-core tests | 14 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| baseline | 1 | Exact paper replication (Table 2 Col 3) |
| core_controls | 10 | Control set variations (add/drop torque, HP, manual, diesel, turbo/super, interactions) |
| core_fe | 4 | FE structure variations (none, year only, mfr only, pooled full sample) |
| core_inference | 5 | SE/clustering variations (robust, classical, two-way, year, nameplate) |
| core_sample | 31 | Sample restrictions (trucks, time periods, windows, trimming, winsorizing, manufacturer drops) |
| core_funcform | 6 | Functional form variations (level-level, semi-log, level MPG DV, log GPM DV, HP^2, HP/weight ratio) |
| core_method | 2 | WLS estimation variants |
| noncore_funcform | 10 | Translog/polynomial specs where linear coefficient is not interpretable; kitchen_sink duplicate |
| noncore_alt_coef | 3 | Specs reporting lhp coefficient instead of lcurbwt (different estimand) |
| noncore_method | 1 | Random effects (wrong sign, confirms FE necessary) |

## Classification Decisions

### Core Test Classifications

**Control set variations (10 specs)**: These add or drop controls from the baseline Cobb-Douglas specification. All maintain the same functional form and report the same lcurbwt coefficient, making them directly comparable robustness checks.

- **Dropping individual controls**: no_torque (-0.444), no_hp (-0.428), no_manual (-0.487), no_diesel_dummy (-0.330) -- all remain negative and significant. The diesel dummy has the largest impact: dropping it attenuates the coefficient by ~0.09, likely because diesel vehicles are heavier and more fuel-efficient, creating omitted variable bias.
- **Minimal controls**: weight_only (-0.892) and none (-0.894) show severe upward bias in magnitude when HP and torque are omitted, as weight absorbs their correlated effects.
- **Adding turbo/super**: cd_with_turbo_super (-0.383) slightly attenuates relative to baseline (-0.419).
- **Interactions**: truck_interaction (-0.441), diesel_interaction (-0.415), manual_interaction (-0.359) add interaction terms but the main lcurbwt coefficient remains the cars-specific elasticity.

**Fixed effects variations (4 specs)**: All are directly comparable and show the coefficient is moderately sensitive to FE structure.
- No FE: -0.413 (cross-sectional estimate)
- Year only: -0.398
- Mfr only: -0.455
- No FE full sample: -0.370
The coefficient ranges from -0.37 to -0.46, confirming the negative sign is robust to FE choice.

**Standard errors/inference (5 specs)**: All produce the identical coefficient (-0.4186) by construction, differing only in SE. SEs range from 0.008 (classical) to 0.034 (two-way cluster), a factor of 4x. The coefficient remains highly significant under all clustering schemes.

**Sample restrictions (31 specs)**: The largest category, with substantial variation across time periods and vehicle subsamples.
- **Cars vs trucks**: The truck elasticity (-0.356) is somewhat smaller than cars (-0.419), consistent with different engineering constraints.
- **Time period variation**: The elasticity declines monotonically over time: 1980s (-0.518), 1990s (-0.403), 2000s (-0.257). The rolling windows confirm this: 1980-1986 (-0.494), 1985-1991 (-0.486), 1990-1996 (-0.429), 1995-2001 (-0.331), 2000-2006 (-0.257). This temporal attenuation is the most interesting robustness finding -- it suggests the weight-fuel economy trade-off weakened as technology improved.
- **Transmission**: Manual cars have a larger elasticity (-0.499) than automatic (-0.382).
- **Outlier treatment**: Including outliers (-0.395), trimming 1% (-0.407), trimming 5% (-0.379), winsorizing 1% (-0.399) -- all very similar to baseline.
- **Manufacturer dropouts**: Dropping GMC (-0.394), Chrysler (-0.417), Ford (-0.436) -- minimal sensitivity.
- **Decade_2000s and window_2000_2006 are identical**: Both produce lcurbwt=-0.257, same N=3774. This is a duplicate.

**Functional form (6 core specs)**: These change the specification but the reported coefficient remains interpretable as a weight-fuel economy relationship.
- Level MPG DV (-14.04): Different scale, same qualitative finding.
- Log GPM DV (+0.4186): Mechanically sign-flipped baseline (identical in magnitude).
- Level-level (curbwt=-0.0064): Different units, same negative relationship.
- Semi-log (curbwt=-0.000199): Different units, same negative relationship.
- CD + HP^2 (-0.390): Adding squared HP barely changes weight coefficient.
- HP/weight ratio (-0.681): Constraining HP proportional to weight substantially changes the elasticity.

**Estimation method (2 core specs)**: WLS with inverse manufacturer-year counts (-0.415) and inverse year counts (-0.417) are both very close to baseline OLS, confirming robustness to weighting.

### Non-Core Classifications

**Translog/polynomial specifications (10 specs, including the kitchen_sink duplicate)**: These are classified as non-core because the reported linear coefficient on lcurbwt is not comparable to the baseline. In a translog model, the total weight elasticity is: d(lmpg)/d(lcurbwt) = beta_1 + 2*beta_11*lcurbwt + beta_12*lhp + beta_13*ltorque, which must be evaluated at specific sample means. The reported linear term alone (which is near zero or even positive) does not represent the elasticity. Including these in a specification curve would be misleading.

Affected specs: translog (-0.043 p=0.97), translog_with_turbo_super (0.197 p=0.88), translog_year_only (0.462 p=0.72), trucks_translog (1.391 p=0.27), trucks_translog_turbo_super (1.431 p=0.23), full_sample_translog (0.723 p=0.33), cd_plus_weight_sq (-4.467), cd_plus_weight_hp_interact (-1.135), cubic_weight (-76.53).

Also: kitchen_sink (0.197 p=0.88) is an exact duplicate of translog_with_turbo_super (identical coefficient, SE, N, R-squared), so it adds no information.

**Alternative coefficient specs (3 specs)**: These report the lhp (log horsepower) coefficient instead of lcurbwt. Since they track a different estimand (HP-MPG elasticity rather than weight-MPG elasticity), they are not robustness checks of the baseline claim. The lhp coefficient is: -0.262 (cars baseline), -2.937 (cars translog, not interpretable), -0.071 (trucks, p=0.097).

**Random effects (1 spec)**: The RE estimate of lcurbwt is +0.739, which has the wrong sign. This is a classic Hausman test-like result: unobserved manufacturer heterogeneity (e.g., brand-level technology differences) is correlated with vehicle attributes, making the RE estimator inconsistent. This spec is classified as non-core because it is a diagnostic rather than a robustness check -- its purpose is to validate the choice of FE over RE, not to test the robustness of the weight elasticity.

## Notable Issues

### 1. Temporal attenuation of the weight elasticity
The most striking finding is that the weight-fuel economy elasticity approximately halves from the 1980s (-0.52) to the 2000s (-0.26). This is consistent with the paper's narrative that technological progress improved the engineering frontier, making it possible to increase weight with less fuel economy penalty. However, it also means the baseline estimate (-0.42) represents an average over a period of substantial change, and may not reflect current-period trade-offs.

### 2. Translog coefficients are not comparable
Ten specifications (including the kitchen_sink duplicate) use translog or polynomial functional forms where the linear lcurbwt coefficient is not interpretable in isolation. These specs report coefficients ranging from -76.5 to +1.4, which would appear extreme in a specification curve but are artifacts of the functional form. Any specification curve analysis must exclude these or compute the total elasticity at sample means.

### 3. Kitchen sink is a duplicate
The kitchen_sink specification produces results identical to translog_with_turbo_super (lcurbwt=0.197, SE=1.260, p=0.876, N=14423, R2=0.890). This is likely because the "all interactions and turbo/super" specification is effectively the full translog. This duplicate should be consolidated.

### 4. Decade_2000s duplicates window_2000_2006
Both specs produce identical results (lcurbwt=-0.257, N=3774, same SE and R2). This is because the 2000s decade (2000-2006) and the 2000-2006 window are identical samples.

### 5. Random effects diagnostic
The positive RE coefficient (+0.74) is strong evidence that the FE specification is necessary. This confirms the paper's identification strategy is appropriate but the RE spec should not be included in the specification curve.

### 6. Omitted variable sensitivity
Dropping controls reveals meaningful sensitivity in specific cases:
- Dropping diesel dummy: coefficient attenuates from -0.42 to -0.33 (21% reduction in magnitude)
- Dropping manual controls: coefficient increases from -0.42 to -0.49 (17% increase)
- Dropping all controls: coefficient inflates to -0.89 (113% increase)

However, the core result (negative, significant weight elasticity) is robust across all control set variations.

### 7. SE clustering matters for magnitude but not significance
The five SE variants all produce the identical coefficient but SEs range from 0.008 (classical) to 0.034 (two-way cluster). The coefficient is significant at 1% under all clustering schemes, so the choice of clustering does not affect the qualitative conclusion.

## Summary Statistics for Core Specifications

Considering only the 58 core (non-baseline) specifications that report the lcurbwt coefficient in a Cobb-Douglas framework:

- **Comparable Cobb-Douglas core specs**: 47 (excluding 5 that report different coefficient names or level-unit coefficients)
- **Range**: -0.894 to -0.257
- **Excluding extreme omitted-variable specs** (weight_only, none): -0.499 to -0.257
- **Median** (well-specified core specs): approximately -0.40
- **Fraction negative and significant at 5%**: 100% of comparable core specs
- **Fraction within 25% of baseline**: majority of well-specified core specs

The weight-fuel economy trade-off is among the most robust empirical findings in the specification search.

## Recommendations

1. **Exclude translog/polynomial specs from specification curves**: The 9 translog/polynomial specs (plus the kitchen_sink duplicate) report non-comparable linear coefficients. If these functional forms are desired, compute the total elasticity at sample means.

2. **Remove duplicates**: kitchen_sink duplicates translog_with_turbo_super; decade_2000s duplicates window_2000_2006. After deduplication, there are 71 unique specifications.

3. **Separate the alternative coefficient specs**: The 3 lhp specs track a different estimand and should not be mixed with lcurbwt robustness checks.

4. **Treat RE as diagnostic only**: The random effects spec has the wrong sign and serves as a Hausman diagnostic, not a robustness check.

5. **Highlight temporal heterogeneity**: The monotonic decline in the weight elasticity from -0.52 (1980s) to -0.26 (2000s) is substantively important and may be worth flagging separately in any meta-analysis.
