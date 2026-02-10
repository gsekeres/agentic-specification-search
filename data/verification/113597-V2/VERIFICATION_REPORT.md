# Verification Report: 113597-V2

## Paper Information
- **Title**: The Impacts of Microfinance: Evidence from Joint-Liability Lending in Mongolia
- **Journal**: AEJ-Applied
- **Method**: RCT with cross-sectional ANCOVA analysis
- **Total Specifications**: 66 (65 valid, 1 failed)

## Baseline Groups

### G1: Enterprise Ownership
- **Claim**: Joint-liability microfinance group loans increase enterprise ownership among eligible households in rural Mongolia.
- **Baseline spec**: `baseline`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.0775 (SE: 0.0326, p = 0.026)
- **Outcome**: `enterprise`
- **Treatment**: `group`
- **N**: 611

### G2: Sole Entrepreneurship
- **Claim**: Group loans increase sole entrepreneurship.
- **Baseline spec**: `baseline/soleent`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.0851 (SE: 0.0380, p = 0.035)
- **Outcome**: `soleent`
- **Treatment**: `group`
- **N**: 611

### G3: Log Total Consumption
- **Claim**: Group loans increase total household consumption (in logs).
- **Baseline spec**: `baseline/ln_totalc`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.1142 (SE: 0.0617, p = 0.077)
- **Outcome**: `ln_totalc`
- **Treatment**: `group`
- **N**: 611
- **Note**: Marginally insignificant at the 5% level.

### G4: Log Food Consumption
- **Claim**: Group loans increase food consumption (in logs).
- **Baseline spec**: `baseline/ln_foodc`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.1698 (SE: 0.0725, p = 0.028)
- **Outcome**: `ln_foodc`
- **Treatment**: `group`
- **N**: 611

### G5: Scaled Profit
- **Claim**: Group loans affect enterprise profits (scaled by baseline mean).
- **Baseline spec**: `baseline/scaled_profit_r`
- **Expected sign**: Positive
- **Baseline coefficient**: -7.85 (SE: 4.27, p = 0.076)
- **Outcome**: `scaled_profit_r`
- **Treatment**: `group`
- **N**: 610
- **Note**: The baseline coefficient is negative, contradicting the expected positive sign. This result is marginally insignificant at 5%.

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baselines)** | **49** | |
| core_controls | 25 | 5 baselines + 5 duplicates of baselines in robust/outcome + 3 control set variations + 11 leave-one-out control drops + 1 duplicate |
| core_fe | 2 | No aimag dummies, explicit aimag FE |
| core_funcform | 3 | Consumption in levels, IHS profit, per capita consumption |
| core_inference | 6 | Robust SE, aimag clustering, household clustering (for enterprise, soleent, ln_totalc, scaled_profit_r) |
| core_sample | 13 | Drop aimag (5), winsorize profit (3), trim, education split (2), baseline loan split (2) |
| **Non-core tests** | **17** | |
| noncore_alt_outcome | 4 | scaled_assets, hours_total, ln_nondurc, ln_durc |
| noncore_alt_treatment | 2 | Individual loan vs control (enterprise, soleent) |
| noncore_heterogeneity | 7 | Education, baseline loan, age, hh_size, buddhist interactions (enterprise, soleent, ln_totalc) |
| noncore_placebo | 3 | Predetermined outcomes: buddhist, hahl, under16 |
| noncore_diagnostic | 1 | Failed soum FE spec (n=0) |
| **Total** | **66** | |

## Detailed Classification Notes

### Core Tests (49 specs including 5 baselines)

**Baselines (5 specs)**: Each baseline tests the effect of group loan treatment assignment on a different outcome variable using the same ANCOVA specification: OLS with baseline outcome control, demographic controls (age, education, household composition, religion, ethnicity), survey timing controls, and aimag dummies, with standard errors clustered at the soum level (where randomization occurred).

**Control variations (20 non-baseline core_controls specs)**: These modify the control set while keeping the enterprise outcome and group treatment:
- No controls (bivariate): coefficient = 0.085, p = 0.28. Without controls, the effect is slightly larger but no longer significant -- the controls improve precision.
- Baseline outcome only: coefficient = 0.085, p = 0.21. ANCOVA with only the lagged dependent variable.
- Demographics only: coefficient = 0.084, p = 0.16.
- Leave-one-out (11 specs): Dropping each individual control one at a time. The coefficient remains very stable (range: 0.077 to 0.082) and significant in all 11 cases. This demonstrates that no single control variable is driving the result.
- Duplicates (5 specs): `robust/outcome/enterprise`, `robust/outcome/soleent`, `robust/outcome/scaled_profit_r`, `robust/outcome/ln_totalc`, and `robust/outcome/ln_foodc` are exact replicas of the corresponding baselines. They produce identical coefficients and p-values.

**Fixed effects variations (2 specs)**:
- No aimag dummies: coefficient = 0.080, p = 0.15. Dropping the aimag dummies from the control set makes the result insignificant, suggesting that controlling for province-level differences matters for precision.
- Explicit aimag FE: coefficient = 0.0775, p = 0.026. Numerically identical to baseline (the aimag dummies in baseline are equivalent to aimag FE).

**Sample restrictions (13 specs)**: These test the sensitivity of the enterprise result (and profit result) to sample composition:
- Drop aimag (5 specs): Dropping each of the 5 aimag (provinces) one at a time. The coefficient ranges from 0.030 (drop aimag 3) to 0.113 (drop aimag 4). Dropping aimag 3 substantially reduces the effect and makes it insignificant (p = 0.41), suggesting that one province drives a meaningful share of the overall result.
- Trim 1% tails: coefficient = 0.084, p = 0.014. Trimming consumption outliers slightly increases the coefficient.
- Education split: Low education subsample shows larger effect (0.109, p = 0.012) than high education (0.042, p = 0.32). This is a core sample restriction because it applies the same specification to subsamples, rather than adding an interaction term.
- Baseline loan split: Had loan (0.089, p = 0.029) vs no loan (0.080, p = 0.073). Effect is similar in magnitude but loses significance in the smaller subsample.
- Winsorize profit (3 specs): Applying 1%, 5%, or 10% winsorization to profits. These are core for G5. All three show negative coefficients (-7517 to -9051), contrary to the expected positive effect. The winsorized profits are significantly negative, suggesting the negative profit result is not driven by outliers.

**Inference variations (6 specs)**: These keep the same point estimate but vary the standard error computation:
- Robust SE (no clustering): p = 0.037 (significant).
- Aimag clustering: p = 0.187 (insignificant). With only 5 aimag, clustering at the province level inflates SEs substantially.
- Household clustering: p = 0.037 (significant). Finer clustering than soum.
- Aimag clustering for soleent: p = 0.208 (insignificant).
- Aimag clustering for ln_totalc: p = 0.124 (insignificant).
- Aimag clustering for scaled_profit_r: p = 0.159 (insignificant).

The aimag clustering results are important: with only 5 clusters, the standard errors are severely inflated and none of the results remain significant. This is a well-known problem with few clusters.

**Functional form variations (3 specs)**:
- Total consumption in levels (not logs): coefficient = 132,828, p = 0.17. Large in magnitude but insignificant -- levels are sensitive to outliers.
- IHS profit: coefficient = -0.632, p = 0.19. Inverse hyperbolic sine transformation of profit. Negative and insignificant, consistent with the negative profit baseline.
- Log per capita consumption: coefficient = 0.109, p = 0.089. Very similar to G3 baseline (0.114), marginally insignificant.

### Non-Core Tests (17 specs)

**Alternative outcomes (4 specs)**: These test outcomes not covered by any of the 5 baseline claims:
- Scaled assets: coefficient = -29.3, p = 0.91. No effect on household assets.
- Total hours worked: coefficient = -2.41, p = 0.53. No effect on labor supply.
- Log non-durable consumption: coefficient = -0.130, p = 0.51. Negative and insignificant.
- Log durable consumption: coefficient = -0.067, p = 0.31. Negative and insignificant.

These are non-core because they measure outcomes distinct from the baseline claims. Notably, the consumption sub-components (non-durable and durable) show negative effects despite the total consumption baseline being positive, which is noteworthy.

**Alternative treatments (2 specs)**: These use individual loan assignment (not group loan) as treatment:
- Individual loan on enterprise: coefficient = 0.003, p = 0.94. No effect.
- Individual loan on soleent: coefficient = -0.004, p = 0.93. No effect.

These are non-core because they test a fundamentally different intervention arm (individual vs. group lending), not alternative estimates of the group loan effect.

**Heterogeneity (7 specs)**: These add interaction terms to test whether the treatment effect varies by subgroup:
- Education interaction (enterprise): coefficient on group = 0.148, p = 0.007.
- Baseline loan interaction: coefficient on group = 0.059, p = 0.21.
- Age interaction: coefficient on group = 0.117, p = 0.026.
- Household size interaction: coefficient on group = 0.070, p = 0.17.
- Buddhist interaction: coefficient on group = 0.192, p = 0.032.
- Education interaction (soleent): coefficient on group = 0.242, p = 0.0003.
- Education interaction (ln_totalc): coefficient on group = 0.116, p = 0.15.

These are non-core because they modify the estimating equation by adding an interaction term, thereby testing effect heterogeneity rather than the average treatment effect.

**Placebo tests (3 specs)**: These use predetermined characteristics as outcomes:
- Buddhist: coefficient = 0.016, p = 0.69. As expected, no effect on predetermined religious affiliation.
- Hahl ethnicity: coefficient = 0.089, p = 0.22. No significant effect, though the coefficient is sizeable.
- Children under 16: coefficient = -0.069, p = 0.51. No significant effect.

All three placebos show insignificant effects, supporting the validity of the randomization.

**Diagnostics (1 spec)**:
- Soum FE: This specification failed with n = 0 because soum fixed effects perfectly absorb the treatment variable (treatment was randomized at the soum level, so within-soum there is no treatment variation). This is correctly identified as a failed specification.

## Duplicates Identified

The following specs produce identical coefficients and standard errors:
1. `robust/outcome/enterprise` = `baseline` (coef = 0.0775, SE = 0.0326)
2. `robust/outcome/soleent` = `baseline/soleent` (coef = 0.0851, SE = 0.0380)
3. `robust/outcome/ln_totalc` = `baseline/ln_totalc` (coef = 0.1142, SE = 0.0617)
4. `robust/outcome/ln_foodc` = `baseline/ln_foodc` (coef = 0.1698, SE = 0.0725)
5. `robust/outcome/scaled_profit_r` = `baseline/scaled_profit_r` (coef = -7.852, SE = 4.27)

Additionally, `robust/fe/aimag_fe` produces numerically identical results to `baseline` because the baseline specification already includes aimag dummies, which are equivalent to aimag fixed effects.

All inference variation specs (robust_only, aimag, household, aimag_soleent, aimag_ln_totalc, aimag_scaled_profit_r) share the same point estimates as their respective baselines but differ in standard errors.

After removing the 5 exact duplicates and the 1 invalid spec, there are 60 unique valid specifications.

## Robustness Assessment

### G1: Enterprise Ownership -- MODERATE support

The primary enterprise result (coefficient = 0.0775, p = 0.026) is **moderately robust**:

- **Stable across controls**: Leave-one-out analysis shows the coefficient ranges from 0.077 to 0.082 -- very stable. No single control drives the result.
- **Sensitive to inference**: Aimag-level clustering (only 5 clusters) renders the result insignificant (p = 0.19). The few-clusters problem is a genuine concern.
- **Sensitive to geography**: Dropping aimag 3 reduces the coefficient to 0.030 (p = 0.41), suggesting geographic heterogeneity in the treatment effect.
- **Sensitive to control set**: Without controls or with minimal controls, the raw treatment-control difference (0.085) is not significant, indicating that the controls improve precision but the result is not overwhelmingly strong.
- **Fixed effects matter**: Dropping aimag dummies makes the result insignificant (p = 0.15).

### G2: Sole Entrepreneurship -- MODERATE support

Similar pattern to G1: significant at 5% with soum clustering (p = 0.035) but insignificant with aimag clustering (p = 0.21).

### G3: Log Total Consumption -- WEAK support

The baseline is already marginally insignificant (p = 0.077). Aimag clustering makes it clearly insignificant (p = 0.12). The levels specification and per capita specification are also insignificant. The claim has weak empirical support.

### G4: Log Food Consumption -- MODERATE support

Significant at 5% (p = 0.028) in the baseline. No additional robustness checks were run specifically for this outcome beyond the baseline.

### G5: Scaled Profit -- CONTRADICTED

The baseline shows a negative coefficient (-7.85, p = 0.076), contradicting the expected positive effect. The winsorized profit specifications are significantly negative (p = 0.007 to 0.038), strengthening the conclusion that the treatment may have had a negative effect on profits. The IHS transformation also shows a negative coefficient. Aimag clustering makes the result insignificant (p = 0.16), but the consistent negative sign across all profit specifications is notable.

### Cross-cutting concerns
- **Few-cluster problem**: With only 5 aimag, aimag-level clustering is problematic and renders all results insignificant. The paper's choice to cluster at the soum level (where randomization occurred) is defensible but more conservative clustering is a concern.
- **Geographic heterogeneity**: The result is sensitive to dropping specific provinces, particularly aimag 3.
- **Individual loan arm shows null effects**: The individual loan treatment (custom/indiv_vs_control) shows no effect on either enterprise or sole entrepreneurship, suggesting the group liability mechanism specifically drives the result.
- **Placebo tests pass**: All three predetermined-outcome placebos are insignificant, supporting randomization balance.
