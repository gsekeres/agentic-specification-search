# Verification Report: 116329-V1

## Paper
**Title**: Do Opposites Detract? Intrahousehold Preference Heterogeneity and Inefficient Strategic Savings
**Authors**: Simone Schaner
**Journal**: American Economic Journal: Applied Economics, 2015
**Method**: Cross-sectional OLS with experimental variation in interest rates (field experiment in Kenya)

## Baseline Groups Found

| Group | Claim | Expected Sign | Baseline Spec IDs |
|-------|-------|---------------|-------------------|
| G1 | Poorly matched couples are more likely to save individually in dominated bank accounts | + | 1 |
| G2 | Poorly matched couples experience greater interest rate losses from using dominated accounts | + | 79, 80 |

### G1: Individual Savings (Primary Finding -- Table 4)
- **Baseline estimate**: 0.1064 (10.6 percentage point increase in probability of saving individually)
- **Baseline SE**: 0.036 (HC1 robust)
- **Baseline p-value**: 0.003
- **Baseline N**: 331 couples
- **Baseline R-squared**: 0.121
- **Dependent variable mean**: 12.4% (7.9% for well-matched couples)
- **Parameters**: savedI outcome, bad treatment (del_PFlog_p50), C1 basic controls (interest rate dummies, cash prize, extra statement, synthetic ATM), non-polygamous couples with max joint interest rate

### G2: Interest Rate Losses (Secondary Finding -- Table 6)
- **Baseline estimate (C1 no controls)**: 0.350, p = 0.050, N = 544
- **Baseline estimate (C2 interest rate controls)**: 0.421, p = 0.012, N = 544
- **Parameters**: loss_i0 outcome (interest rate loss from using individual instead of joint accounts), bad treatment, full sample (no interest rate restriction)
- The C2 specification (spec 80) with interest rate controls is the primary baseline because interest rate controls are essential to the experimental design

## Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **108** |
| Baselines | 3 |
| Core tests (non-baseline) | 101 |
| Non-core tests | 0 |
| Specifications with mixed sign | 4 |

Note: All 108 specifications are classified as either baseline or core. Every specification tests a direct implication of the same fundamental hypothesis (preference heterogeneity drives strategic savings behavior). There are no non-core specifications because all outcomes, samples, and estimation methods are directly relevant to the paper's main claim.

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| baseline | 3 | Table 4 C1 replication (spec 1) and Table 6 C1-C2 replication (specs 79-80) |
| core_controls | 4 | Progressive controls C2-C5 on savedI, all couples (specs 3, 5, 7, 9) |
| core_controls_sample | 4 | Progressive controls C2-C5 on savedI, savers only (specs 4, 6, 8, 10) |
| core_sample | 1 | Savers-only subsample with C1 basic (spec 2) |
| core_alt_outcome | 20 | Alternative outcomes: saved_ncI, openI, ln_avg_balI, depI across C1-C5, all couples (specs 11-49 odd) |
| core_alt_outcome_sample | 20 | Alternative outcomes across C1-C5, savers only (specs 12-50 even) |
| core_alt_het | 12 | Alternative heterogeneity measures: p33, p67 splits; probit-based discount factors (specs 51-62) |
| core_continuous_het | 5 | Continuous heterogeneity measure across C1-C5 (specs 63-67) |
| core_se_type | 5 | Clustered standard errors across C1-C5 (specs 68-72) |
| core_sample_restriction | 6 | No interest rate restriction (specs 73-76) and include polygamous (specs 77-78) |
| core_loss | 4 | Table 6 C3-C6 progressive controls on interest rate loss (specs 81-84) |
| core_acct_level | 5 | Account-level regressions with cluster SEs (specs 85-89) |
| core_estimation | 2 | Probit and Logit marginal effects (specs 90-91) |
| core_drop_controls | 5 | Drop-one-group from full controls (specs 92-96) |
| core_couple_outcome | 8 | Couple-level outcomes: saved, savedJ, savedIH, savedIW (specs 97-104) |
| core_winsorized | 4 | Winsorized continuous outcomes at 95th and 99th percentiles (specs 105-108) |

## Classification Decisions

### Core Test Classifications

**Progressive controls (specs 2-10, 8 core + 1 baseline)**: These replicate Table 4 Columns 1-5, progressively adding time preference controls (C2), demographics and village FE (C3), economic variables (C4), and decision-making variables (C5). Both the full sample and savers-only subsample are tested. These are core because they directly test whether the baseline finding survives controlling for potential confounders. The savers-only subsample tests the intensive margin of account choice conditional on saving.

**Alternative outcomes (specs 11-50, 40 core)**: These replace the primary outcome (savedI) with four alternatives: saved_ncI (non-cash individual savings), openI (opened individual account), ln_avg_balI (log average individual balance), and depI (number of deposits to individual accounts). Each is tested across all five control sets and both samples (all, savers only). These are core because they measure different facets of the same underlying behavior -- using individual rather than joint accounts. The log balance and deposit outcomes test whether the effect extends to the intensive margin of savings amounts.

**Alternative heterogeneity measures (specs 51-62, 12 core)**: These replace the baseline heterogeneity measure (median split of log discount factor differences) with: (1) 33rd percentile split (more restrictive "poorly matched" definition), (2) 67th percentile split (more inclusive definition), and (3) probit-based discount factor estimates. Each is tested with savedI and saved_ncI outcomes at C1 and C4 control levels. These are core because the specific method of measuring preference heterogeneity is a key researcher degree of freedom. The 33rd percentile split produces larger effects (0.145-0.167) while the 67th percentile produces smaller effects (0.110-0.118), as expected.

**Continuous heterogeneity (specs 63-67, 5 core)**: These use the continuous absolute difference in discount factors rather than a binary split. The coefficient represents the effect of a one-unit increase in |delta_H - delta_W| on savedI. Coefficients (0.19-0.23) are significant at 0.1% level across all control sets. These are core because they test whether the binary split is driving the result or whether the underlying continuous relationship is robust.

**Clustered standard errors (specs 68-72, 5 core)**: These replicate the C1-C5 specifications using clustered rather than HC1 standard errors. Point estimates are identical; only SEs change slightly. All remain significant at 0.3% level. These are core because the choice of standard error type is a standard robustness concern.

**Sample restrictions (specs 73-78, 6 core)**: These relax two key sample restrictions: (1) removing the requirement that the joint interest rate equals the maximum offered rate (N=544 vs 331), and (2) including polygamous couples (N=438 vs 331). Both maintain significance for savedI. The no-restriction sample produces smaller coefficients (0.067-0.114) as expected, since the interest rate restriction creates the strongest incentive to use joint accounts. These are core because sample selection is a key researcher degree of freedom.

**Interest rate loss specifications (specs 79-84, 2 baseline + 4 core)**: These test Table 6's finding that poorly matched couples experience greater interest rate losses. The baseline C1 (no controls) and C2 (interest rate controls) are baselines for G2. The remaining C3-C6 are core. The coefficient is stable (0.43-0.47) but marginal significance weakens with added controls (p = 0.05-0.07), likely due to reduced power with many controls.

**Account-level regressions (specs 85-89, 5 core)**: These analyze at the account level (N=1632, approximately 5 accounts per couple) rather than couple level. The saved outcome is significant (p=0.025) but the open outcome is null (p=0.624) and avg_bal is null (p=0.730). These are core because they test whether the couple-level finding translates to the account level, though the pooling of multiple accounts per couple dilutes the effect.

**Probit and Logit (specs 90-91, 2 core)**: These replicate the baseline using probit (ME=0.101) and logit (ME=0.092) instead of OLS (0.106). The marginal effects are very close to the OLS estimate, confirming the linear probability model is adequate. Both are significant at 0.1%.

**Drop control groups (specs 92-96, 5 core)**: These start from the full model (C5) and drop one control group at a time: time preferences, demographics, village FE, economic variables, and decision-making variables. The most notable change is that dropping demographics increases the coefficient from 0.146 to 0.171 (p<0.001), suggesting some demographic variables correlate with both heterogeneity and savings. All remain significant at 1.2% or better.

**Couple-level outcomes (specs 97-104, 8 core)**: These test related couple-level outcomes: saved (any account, including joint), savedJ (joint only), savedIH (husband's individual), and savedIW (wife's individual). The savedJ coefficient is negative (-0.045 to -0.056) and insignificant, consistent with the theory that poorly matched couples substitute from joint to individual accounts rather than increasing total savings. savedIH (0.072-0.092) and savedIW (0.060-0.096) are both positive and significant, confirming both spouses contribute to the individual savings effect.

**Winsorized outcomes (specs 105-108, 4 core)**: These winsorize continuous outcomes (avg_balI, depI) at the 95th and 99th percentiles to address outlier concerns. The 95th percentile winsorization produces significant results for both outcomes. The 99th percentile is significant for depI (p=0.008) but marginal for avg_balI (p=0.071), suggesting some outlier sensitivity in the level (but not count) measure.

### Non-Core Classifications

There are no non-core specifications. All 108 specifications test direct implications of the paper's central hypothesis. Even the account-level and interest rate loss specifications are testing the same mechanism (preference heterogeneity driving inefficient savings behavior) through different measurement approaches.

## Robustness Assessment

### G1: Individual Savings Robustness

The primary finding is **highly robust**:

- **32 specifications** use the primary outcome (savedI) with binary treatment: **31 of 32 are significant at 5%** (96.9%). The one exception is spec 10 (savers only, full controls, p=0.097), which loses power due to the small sample (N=147) and large number of controls (101).

- **Coefficient range for savedI**: [0.067, 0.321], always positive. The lower bound (0.067) comes from relaxing the interest rate restriction; the upper bound (0.321) comes from savers-only subsample with demographic controls.

- **Progressive controls increase the estimate**: Adding controls raises the coefficient from 0.106 (C1) to 0.146 (C5), suggesting that the baseline result is conservative and omitted variable bias attenuates the estimated effect.

- **All heterogeneity measures agree**: Log discount factors (p50: 0.106, p33: 0.145, p67: 0.110), probit-based (0.103-0.157), and continuous (0.191-0.230) all yield positive significant effects on savedI.

- **Estimation method does not matter**: OLS (0.106), probit ME (0.101), and logit ME (0.092) produce nearly identical estimates.

- **SE type does not matter**: HC1 and clustered SEs are nearly identical, reflecting the couple-level unit of analysis.

**Weaker results for alternative outcomes**:
- saved_ncI (non-cash savings): Positive in all 10 specs, significant at 5% in 5 of 10 (50%).
- openI (account opening): Positive in all 10 specs, significant at 5% in 3 of 10 (30%). The full-sample results are generally insignificant (p=0.16-0.41).
- ln_avg_balI (log balance): Positive and significant at 5% in 8 of 10 specs (80%).
- depI (deposits): Positive in all 10 specs, significant at 5% in 4 of 10 (40%).

### G2: Interest Rate Loss Robustness

The interest rate loss finding is **moderately robust** but sensitive to controls:

- **6 specifications**: Coefficient ranges from 0.35 to 0.47, always positive.
- The baseline with no controls (spec 79) is borderline significant (p=0.050).
- Adding interest rate controls improves significance (spec 80, p=0.012), but further controls weaken it (p=0.052-0.066).
- The qualitative direction is robust, but the finding is sensitive to the specific p < 0.05 threshold.

### Mixed-Sign Specifications

4 of 108 specifications have negative coefficients:
- Spec 85 (account-level open): -0.008, p=0.624 -- null effect on account opening at the account level.
- Spec 88 (account-level avg_bal): 14.30, p=0.730 -- actually positive but economically small relative to the mean (150.7).
- Spec 99 (savedJ, C1): -0.045, p=0.344 -- negative effect on joint savings, consistent with substitution.
- Spec 100 (savedJ, C4): -0.056, p=0.369 -- negative effect on joint savings, consistent with substitution.

The negative savedJ coefficients are theoretically expected (substitution from joint to individual accounts) and are not evidence against the main claim. Only the account-level open specification (spec 85) is a genuine null result, and it is easily explained by the dilution from pooling multiple accounts per couple.

## Notable Issues

### 1. The treatment variable is not randomly assigned
The "badly matched" indicator is based on observationally estimated time preferences, not random assignment. While the paper exploits random variation in interest rates from the field experiment, the key heterogeneity measure (|delta_H - delta_W|) is derived from household survey responses. This means the specification search tests robustness to analytical choices but cannot fully address selection concerns.

### 2. Small baseline sample
The primary sample is only 331 couples (from 778 originally randomized), restricted to non-polygamous couples eligible for follow-up who received the highest joint interest rate. This restriction is necessary for the economic mechanism (dominated accounts only exist when the joint rate is maximal) but reduces power. The savers-only subsample (N=147) is particularly small.

### 3. Median split is somewhat arbitrary
The baseline uses a median split of |delta_H - delta_W| to define "poorly matched." The p33 split produces larger effects (0.145-0.167) and the p67 split produces smaller effects (0.110-0.118), suggesting the effect exists throughout the distribution but is concentrated among the most heterogeneous couples. The continuous measure (0.191-0.230) confirms a linear dose-response relationship.

### 4. Effect increases with controls
The coefficient consistently increases as controls are added (0.106 -> 0.146), which is unusual. This suggests negative omitted variable bias in the unconditional regression -- observable characteristics that predict both preference heterogeneity and lower individual savings are omitted from the basic specification. While this means the baseline is conservative, it also raises the question of what unobservable characteristics might matter.

### 5. Savers-only subsample shows larger effects but lower power
The savers-only subsample (N=147) consistently shows 2-3x larger coefficients than the full sample, reflecting the intensive margin of account choice among those who save. However, several savers-only specifications lose significance with full controls (specs 10, 20, 30, 40, 50), likely due to insufficient degrees of freedom (101 controls with 147 observations).

## Recommendations

1. **Focus on savedI specifications (32 specs) for the specification curve**: These directly test the paper's primary claim with consistent outcome and treatment definitions, varying only controls, sample, heterogeneity measure, and estimation method.

2. **The G2 loss result should be presented separately**: The interest rate loss specifications (6 specs) test a different outcome and are marginally significant. They support the mechanism but should not be pooled with the savedI specifications.

3. **Flag the savers-only small-sample issue**: Specifications with N=147 and 85-101 controls are likely underpowered. Consider flagging these in the specification curve.

4. **The continuous heterogeneity measure is especially informative**: Specs 63-67 avoid the arbitrary median split entirely and show the strongest results (all p < 0.002), suggesting the binary split is conservative.

5. **The negative savedJ coefficients support rather than undermine the claim**: These should be interpreted as evidence of substitution from joint to individual accounts, consistent with the strategic savings hypothesis.
