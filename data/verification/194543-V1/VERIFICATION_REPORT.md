# Verification Report: 194543-V1

## Paper Information
- **Title**: The Burden of Debt
- **Authors**: Martinez-Marquina and Shi
- **Journal**: American Economic Review (2024)
- **Total Specifications**: 82

## Baseline Groups

### G1: Debt Treatment Effect on Optimal Allocation
- **Claim**: Subjects randomized into the Low Debt treatment are significantly less likely to maximize returns across all initial allocation decisions (days 1-4) compared to the No Debt control, demonstrating that the mere presence of debt impairs financial decision-making.
- **Baseline spec**: `baseline`
- **Expected sign**: Negative
- **Baseline coefficient**: -0.2558 (SE: 0.0640, p < 0.001)
- **Outcome**: `ind_optimal_ia_all` (binary: 1 if optimal across all days)
- **Treatment**: `treat` (Low Debt = 1, No Debt = 0)
- **N**: 172 (86 per group)
- **R-squared**: 0.086
- **Matches stat1403**: `reg ind_optimal_ia_all i.treatment if day == 4 & treatment <= 1, r`

**Note**: This is a single baseline group because the paper has one primary treatment comparison. The experimental design is a randomized online experiment (MTurk) where the treatment is exogenous. The outcome variable `ind_optimal_ia_all` is a cumulative binary indicator measured at Day 4 that captures whether the subject was optimal on ALL initial allocation decisions across days 1-4.

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baseline)** | **53** | |
| core_controls | 13 | 1 baseline + 12 control variations (demographics, errors, financial literacy, batch FE, age functional forms) |
| core_sample | 18 | Batch splits (3), error restrictions (3), demographic splits (9: gender, education, race, debt, age, student loan), all-three-treatments |
| core_inference | 7 | HC1/HC2/HC3, classical, clustered by batch, permutation, full controls with HC3 (includes controls/none_classical reclassified) |
| core_treatment | 5 | High Debt vs No Debt, High vs Low Debt, Any Debt vs None, dose-response, both dummies |
| core_method | 4 | Logit, probit (with and without controls), average marginal effects |
| core_panel | 7 | Pooled with day FE (3 variants), day-by-day analysis (Days 1-4) |
| **Non-core tests** | **29** | |
| noncore_alt_outcome | 14 | Share optimal, share to specific accounts, avg return rate, log return rate, number of accounts, Herfindahl, total share, additional allocations, any-to-a1, log with controls |
| noncore_placebo | 6 | One-shot scenarios (2 in outcomes + 1 in placebo), balance tests for age/gender/education |
| noncore_external | 6 | Redistribution treatment (3), borrowing treatment (3) -- different experimental conditions |
| noncore_diagnostic | 3 | Quantile regressions (median/25th), no_points_debt1 (mechanical duplicate of baseline) |
| **Total** | **82** | |

## Detailed Classification Notes

### Core Tests (53 specs including baseline)

**Baseline (1 spec)**: The single baseline specification regresses `ind_optimal_ia_all` (binary indicator for optimal allocation across all 4 days) on the Low Debt treatment indicator using OLS with HC1 robust standard errors, restricted to Day 4 observations for subjects in the No Debt or Low Debt conditions. The coefficient of -0.2558 means Low Debt subjects are 25.6 percentage points less likely to be optimal.

**Control variations (12 non-baseline core_controls specs)**: These systematically vary the control set while maintaining the same outcome, treatment, and sample:
- No controls / classical SE: coefficient -0.256 (identical point estimate, different SE)
- Basic demographics (age + male): -0.253
- Full paper controls (8 covariates): -0.235
- Full controls + batch FE: -0.226
- Comprehension errors only: -0.233
- Financial literacy proxies: -0.254
- Demographics without errors: -0.257
- Education only, age + education: -0.252
- Continuous age (instead of median split) + full controls: -0.233
- Quadratic age + basic demographics: -0.257
- Full controls + continuous age + age squared: -0.228

The coefficient is remarkably stable across all control sets, ranging from -0.226 to -0.257, all significant at p < 0.001. This is expected in a randomized experiment where controls should not affect the treatment coefficient substantially (randomization ensures orthogonality). The stability confirms successful randomization.

**Sample restrictions (18 specs)**: The largest category tests whether the effect holds across subpopulations:
- By recruitment batch (3 specs): Batch 1 (-0.298, p=0.024, N=47), Batch 2 (-0.163, p=0.060, N=78), Batch 3 (-0.353, p=0.005, N=47). Batch 2 is marginally significant, likely due to smaller effect in that wave.
- By comprehension errors (3 specs): Low errors (-0.333, p<0.001, N=90), very low errors (-0.311, p<0.001, N=133), understood instructions (-0.268, p<0.001, N=160). Effect is larger among participants who understood the task better.
- By demographics (9 specs): Males (-0.258, N=100), females (-0.248, N=72), college (-0.242, N=121), non-college (-0.276, N=51), white (-0.305, N=136), non-white (-0.107, N=36, p=0.44), real debt holders (-0.243, N=106), no real debt (-0.269, N=66), young (-0.280, N=82), old (-0.233, N=90), student loan holders (-0.252, N=85).
- All three treatments (1 spec): N=258, coefficient -0.256 (identical to baseline because Low Debt vs No Debt contrast is the same).

Notable: The non-white subsample (N=36) is the only demographic split that loses significance, attributable to the very small sample size rather than a true null effect.

**Inference variations (7 specs)**: All maintain identical point estimates (-0.256) but vary SE computation:
- HC1 (baseline): SE = 0.064, p < 0.001
- HC2: SE = 0.064, p < 0.001 (nearly identical to HC1)
- HC3: SE = 0.064, p < 0.001 (slightly larger)
- Classical: SE = 0.064, p < 0.001
- Clustered by batch: SE = 0.061, p < 0.001
- Permutation (1000 draws): p = 0.000
- Full controls + HC3: SE = 0.065, p < 0.001

The result is completely robust to inference method. With only 3 batches, clustering by batch yields similar SEs because few clusters actually reduce degrees of freedom.

**Alternative treatment definitions (5 specs)**: These change the treatment comparison while keeping the same outcome:
- High Debt vs No Debt: -0.128, p = 0.071 (weaker and marginally significant -- interesting because more debt does not mean more distortion)
- High vs Low Debt: +0.128, p = 0.032 (High Debt subjects do BETTER than Low Debt, consistent with non-monotonic dose-response)
- Any Debt vs No Debt: -0.192, p = 0.002 (pooling both debt arms dilutes the effect)
- Dose-response (continuous): -0.000015, p = 0.179 (no linear relationship between debt amount and suboptimality)
- Both dummies: -0.256 for Low Debt (identical to baseline, as expected)

These are core because they test the same claim (debt causes suboptimal behavior) using alternative operationalizations of the treatment variable.

**Estimation methods (4 specs)**: Logit and probit marginal effects:
- Logit: AME = -0.252, p < 0.001
- Probit: AME = -0.248, p < 0.001
- Logit + full controls: AME = -0.224, p < 0.001
- Probit + full controls: AME = -0.227, p < 0.001

All very close to the OLS coefficient, confirming that for a binary outcome with moderate prevalence, the linear probability model provides an excellent approximation.

**Panel/repeated measures (7 specs)**: These exploit the multi-day structure:
- Pooled days 1-4 with day FE: -0.160, p < 0.001 (N=688)
- Pooled + demographics: -0.171, p < 0.001
- Pooled + clustered by individual: -0.160, p = 0.011 (wider SE from within-individual correlation)
- Day 1: -0.279, p < 0.001 (strongest)
- Day 2: -0.233, p = 0.001
- Day 3: -0.116, p = 0.125 (not significant)
- Day 4: -0.012, p = 0.879 (no effect)

This reveals an important dynamic: the debt treatment effect is strongest on Day 1 and converges to zero by Day 4 as subjects learn. The pooled specification averages across days. The day-by-day decomposition is core because it directly tests the persistence of the same treatment effect over the experiment's time horizon.

### Non-Core Tests (29 specs)

**Alternative outcomes (14 specs)**: These measure different aspects of allocation behavior:
- `share_optimal_day1` / `share_a1_day1`: Continuous share of optimal allocation on Day 1 (-0.261). These are identical (same coefficient/SE), suggesting perco_orig = share_a1_initial in this sample.
- `share_a3_day1`: Share to the 15%-return account (+0.188), opposite sign as expected (debt holders put more into Debt 1 account).
- `avg_rate_day1` / `ln_avg_rate_day1`: Weighted return rate (-0.017 level, -0.099 log). Continuous efficiency loss.
- `optimal_ia_pooled`: Per-day binary optimal indicator pooled (distinct from ind_optimal_ia_all which is ALL-days cumulative).
- `optimal_ap_all`: Optimal in additional (rebalancing) allocations (-0.233).
- `num_accounts_day1`: Number of accounts used (+0.337, p=0.10). Insignificant but positive, suggesting debt holders diversify more (suboptimally).
- `no_points_debt1`: Mechanically identical to baseline (same coef/SE), likely a recoding of the same variable.
- `share_tot_a1`: Total share to Account 1 across days (-0.111).
- `share_a2_day1`: Share to Account 2, 10% return (+0.060, p=0.003). Debt holders allocate more to second-best account.
- `herfindahl_day1`: Portfolio concentration index (-0.109, p=0.018). Less concentrated = more diversified = suboptimal.
- `form/log_dep_avgrate`: Duplicate of `outcome/ln_avg_rate_day1` (identical coefficient -0.099).
- `form/log_dep_full`: Log avg rate with full controls (-0.092).
- `form/any_to_a1`: Any allocation to Account 1 (-0.116, p=0.002). Extensive margin of allocating anything to best account.

These are non-core because they substitute different dependent variables. While they support the overall narrative (debt impairs allocation quality), they test different operationalizations of the outcome rather than being robustness checks of the same specification.

**Placebo/balance tests (6 specs)**: These validate the research design:
- One-shot no-debt scenario (2 specs: `oneshot_control_optimal` and `placebo/oneshot_control_share`): Treatment coefficient is effectively zero (coef ~ 0, p > 0.37) when debt is absent from the scenario. This is the key placebo: the treatment groups perform identically when the scenario removes debt.
- One-shot debt scenario: Near-zero insignificant effect (-0.023, p = 0.76).
- Balance tests (3 specs): Age (p=0.51), gender (p=0.54), education (p=0.41). All confirm successful random assignment.

These are non-core because they test design validity rather than estimate the treatment effect.

**External datasets (6 specs)**: These use different experimental conditions from the same study:
- Redistribution treatment (3 specs): A separate experimental arm where accounts are redistributed. Effect is strong (-0.289, -0.286 with controls) and a consolidation measure (-0.202) is also significant.
- Borrowing treatment (3 specs): A separate experimental arm testing borrowing behavior. Borrow max from both accounts (-0.284, -0.313 with controls) and log returns (-0.055).

These are non-core because they test different experimental conditions (redistribution/borrowing) from the same paper. They support the broader claim but involve different datasets and different treatment mechanisms.

**Diagnostics (3 specs)**:
- `method/quantile_50`: Median regression shows zero effect because the median outcome is 1.0 (optimal) in both groups. This confirms the effect is on the extensive margin -- shifting some subjects from optimal to suboptimal rather than making everyone slightly worse. Not a robustness check of the same claim but a characterization of the effect's nature.
- `method/quantile_25`: 25th percentile regression (-0.143, p=0.145). The effect is concentrated above the 25th percentile.
- `outcome/no_points_debt1`: Coefficient and SE are identical to baseline (-0.256, SE=0.064). This is a mechanical duplicate -- the variable `no_points_a3` appears to be equivalent to `ind_optimal_ia_all` in this sample.

## Duplicates Identified

The following specs produce identical coefficients and SEs:
1. `outcome/share_optimal_day1` = `outcome/share_a1_day1` (both coef = -0.261, SE = 0.046) -- `perco_orig` and `share_a1_initial` appear to be the same variable
2. `outcome/no_points_debt1` = `baseline` (both coef = -0.256, SE = 0.064) -- `no_points_a3` appears identical to `ind_optimal_ia_all`
3. `form/log_dep_avgrate` = `outcome/ln_avg_rate_day1` (both coef = -0.099, SE = 0.029) -- same specification listed in two categories
4. `controls/none_classical` = `inference/classical` (both coef = -0.256, SE = 0.064, identical p-values) -- same specification (OLS with classical SE) appears in both controls and inference categories

After removing duplicates, there are approximately 78 unique specifications.

## Robustness Assessment

The main finding -- that holding debt causes suboptimal allocation -- is **very robust** across core specifications:

- **Same-outcome core tests** (48 specs with `ind_optimal_ia_all`): Coefficients range from -0.353 (Batch 3 only) to +0.128 (High vs Low Debt, which tests a different contrast). Excluding different-contrast treatment specs, the range is -0.353 to -0.163, with a median of approximately -0.254. Of 48 same-outcome specs, 91.7% are significant at the 5% level.

- **Control robustness**: The coefficient varies by less than 0.03 across 12 different control sets (-0.226 to -0.257), consistent with successful randomization. Including comprehension errors as controls slightly reduces the coefficient (from -0.256 to -0.233), suggesting that some of the treatment effect operates through confusion/comprehension difficulty.

- **Inference robustness**: The result is significant under every inference method tested (HC1/HC2/HC3, classical, clustered, permutation). The permutation test yields p = 0.000.

- **Demographic subgroups**: The effect is present and significant across gender, education, age, real debt status, and student loan status splits. The only insignificant subgroup is non-white participants (N=36, p=0.44), attributable to very small sample size.

- **Estimation method**: Logit and probit marginal effects (-0.248 to -0.252) are nearly identical to the OLS coefficient (-0.256).

Key sensitivities and qualifications:

1. **The effect fades over time**: Day-by-day analysis shows the treatment effect is -0.279 on Day 1 but -0.012 (null) on Day 4. The cumulative `ind_optimal_ia_all` measure reflects early-day differences. This is not a threat to the main finding but is important context: debt impairs initial decision-making, and subjects learn to overcome it over 4 days.

2. **Non-monotonic dose-response**: High Debt produces a WEAKER effect (-0.128, p=0.07) than Low Debt (-0.256, p<0.001). The continuous dose-response is insignificant (p=0.18). This is consistent with the paper's interpretation (the presence of debt, not its amount, is what matters) but means the effect should not be extrapolated linearly.

3. **Batch heterogeneity**: Batch 2 shows a weaker effect (-0.163, p=0.06) compared to Batches 1 and 3 (-0.298 and -0.353). This is not uncommon in MTurk experiments where participant composition varies across recruitment waves.

4. **Small subgroup fragility**: The non-white subsample (N=36) loses significance entirely. While the point estimate (-0.107) is in the expected direction, the study lacks power to detect effects in small demographic subgroups.

5. **Extensive margin interpretation**: The quantile regression at the median confirms that most subjects in both groups achieve optimal allocation (median = 1.0). The treatment effect is about shifting a fraction of subjects from optimal to suboptimal, not about continuous degradation of everyone's performance.
