# Verification Report: 114708-V1

## Paper Information
- **Title**: Screening in Contract Design: Evidence from the ACA Health Insurance Exchanges
- **Authors**: Michael Geruso, Timothy Layton, Daniel Prinz
- **Journal**: American Economic Review
- **Total Specifications**: 90

## Baseline Groups

### G1: CSR Generosity Effect on Copays (Silver Plans)
- **Claim**: Silver94 (most generous CSR variant) plans have systematically lower drug copays than standard Silver plans, and the effect increases with drug tier.
- **Baseline spec**: `baseline`
- **Expected sign**: Negative (more generous = lower copays)
- **Baseline coefficient**: -19.64 (SE: 0.66, p ~ 0)
- **Outcome**: `copay_amount`
- **Treatment**: `Silver94_vs_Silver`
- **N**: 18,389, R2 = 0.36

### G2: Generosity x Tier Interaction (All Metals)
- **Claim**: The relationship between plan generosity and drug copays is dramatically heterogeneous by drug tier -- more generous plans amplify the tier-based cost-sharing gradient, consistent with screening.
- **Baseline spec**: `ols/interact/generosity_x_hightier`
- **Expected sign**: Positive (generosity amplifies tier differences)
- **Baseline coefficient**: 10.44 (SE: 0.29, p ~ 0)
- **Outcome**: `copay_amount`
- **Treatment**: `generosity_x_high_tier`
- **N**: 33,885, R2 = 0.29

**Note**: The paper's main regressions (Table 3) use proprietary Marketscan claims + MMIT formulary data to construct therapeutic-class-level selection incentive measures. The publicly available CCIIO data used here captures the observable cost-sharing patterns that are the consequence of the screening documented in the paper.

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baselines)** | **46** | |
| core_controls | 8 | 2 baselines + CSR variant, interaction variant, control progressions (bivariate, state, issuer, deductible, full) |
| core_fe | 8 | No FE, Tier, Tier+Issuer, Tier+State, Tier+Issuer+State, interaction+Issuer, interaction+Specialty+Issuer, interaction+State |
| core_sample | 18 | Full sample, Silver-only, drop metal levels (leave-one-out x7), drop states (x5), trimming/winsorizing, Bronze+Silver, Gold+Platinum |
| core_inference | 5 | Clustered by state, issuer, plan, tier; HC1 robust |
| core_funcform | 3 | Log dependent, quadratic generosity, binary high copay |
| **Non-core tests** | **44** | |
| noncore_alt_outcome | 15 | Subject-to-deductible (overall + by tier x4), coinsurance pct, combined cost-sharing, copay spreads (x3), plan-level copays by tier (x4) |
| noncore_alt_treatment | 1 | Tier as continuous treatment with metal FE |
| noncore_heterogeneity | 28 | By metal level (x5), by tier (x4), by state (x5), pairwise metal comparisons (x6), Silver-vs-Bronze by tier (x4), CSR94 by tier (x4) |
| **Duplicates noted** | **5** | panel/fe/none = robust/controls/none; panel/fe/issuer = robust/controls/tier_issuer; robust/sample/drop_platinum = robust/leave_one_out/drop_platinum; robust/form/binary_subj_ded = robust/outcome/subj_ded; robust/sample/drop_catastrophic = robust/leave_one_out/drop_catastrophic |
| **Total** | **90** | |

## Detailed Classification Notes

### Core Tests (46 specs including 2 baselines)

**Baselines (2 specs)**: The two primary baselines capture different facets of the same screening phenomenon: (1) the Silver94 vs Standard Silver copay gap within the Silver tier, and (2) the generosity x high-tier interaction across all metal levels.

**Control progressions (6 non-baseline core specs)**: These systematically build up the control structure from bivariate to fully saturated (Tier + Issuer + State FE), and also test the Silver87 CSR variant and generosity x specialty interaction.

**FE structure (8 specs)**: Systematic exploration of fixed effects from none to Tier+Issuer+State. The generosity coefficient is remarkably stable across FE structures (3.43 to 4.26), showing the result is not driven by any particular source of confounding.

**Sample restrictions (18 specs)**: Leave-one-out analysis for all metal levels and largest states ensures no single group drives the result. Sample stability confirmed: trimming and winsorizing have minimal impact. The negative coefficient in Silver-only and Gold+Platinum subsamples reflects within-group variation (more generous CSR variants and Platinum have lower copays).

**Inference (5 specs)**: Same point estimate (4.26) with different SE estimators. Significant at 5% with all clustering choices except tier (only 4 clusters, p = 0.21). State clustering (SE = 0.49) and issuer clustering (SE = 0.59) are the most conservative meaningful alternatives.

**Functional form (3 specs)**: Log copay, quadratic generosity, and binary high-copay threshold all confirm the relationship.

### Non-Core Tests (44 specs)

**Alternative outcomes (15 specs)**: These examine different dimensions of cost-sharing beyond copay amounts: subject-to-deductible indicators (overall and by tier), coinsurance percentages, combined cost-sharing, copay spreads between tiers, and plan-level copays by tier. These explore the breadth of the screening mechanism rather than providing alternative estimates of the same effect.

**Alternative treatment (1 spec)**: Tier as continuous treatment with metal FE reverses the regression structure.

**Heterogeneity (28 specs)**: The largest non-core category. Includes: (1) individual metal-level effects vs Bronze (5 specs), (2) tier-specific generosity effects (4 specs), showing the gradient from -$0.33/level for generics to +$21.68/level for specialty, (3) state-specific effects (5 specs), (4) pairwise metal comparisons (6 specs), (5) Silver-vs-Bronze within each tier (4 specs), and (6) CSR94 effect by tier (4 specs). These decompose the aggregate effect rather than test alternative implementations.

## Duplicates Identified

1. `panel/fe/none` = `robust/controls/none` (both coef = 3.432, SE = 0.139)
2. `panel/fe/issuer` = `robust/controls/tier_issuer` (both coef = 3.855, SE = 0.119)
3. `robust/sample/drop_platinum` = `robust/leave_one_out/drop_platinum` (both coef = 4.990, SE = 0.141)
4. `robust/form/binary_subj_ded` = `robust/outcome/subj_ded` (both coef = -0.109, SE = 0.001)
5. `robust/sample/drop_catastrophic` = `robust/leave_one_out/drop_catastrophic` (both coef = 1.891, SE = 0.141)

After removing duplicates, there are approximately 85 unique specifications.

## Robustness Assessment

The paper's finding that plan generosity systematically predicts drug cost-sharing structure is **very robust** in the publicly available data:

- **G1 (CSR effect)**: The Silver94 copay reduction (-$19.64) is highly significant and monotonically increasing by tier (from -$5.23 for generics to -$51.03 for specialty drugs). This pattern is exactly what the screening theory predicts.

- **G2 (Generosity x tier interaction)**: The interaction coefficient is positive and highly significant (10.44, p ~ 0) and stable across FE structures (9.43 to 10.44). Adding issuer FE reduces the interaction only slightly (from 10.44 to 9.43), suggesting within-issuer variation also supports the pattern.

- **98% of specifications significant at 5%**: The only insignificant spec uses tier-level clustering with only 4 clusters.

- **Key sensitivity**: The generosity-copay relationship is non-monotonic across metal levels. Gold and Platinum plans have *lower* copays than Silver (negative coefficient in Gold+Platinum subsample), while Bronze-to-Silver shows a large positive gap. This reflects the composition effect noted in the paper -- more generous plans cover more expensive drugs.

- **Caveat**: This analysis uses only publicly available CCIIO data, not the proprietary Marketscan + MMIT data used for the paper's main therapeutic-class-level regressions (Table 3). The CCIIO analysis confirms the *observable* cost-sharing patterns but cannot directly test the selection incentive mechanism.
