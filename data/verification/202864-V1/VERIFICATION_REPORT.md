# Verification Report: 202864-V1

## Paper
- **Title**: Eliciting ambiguity with mixing bets
- **Authors**: Patrick Schmidt
- **Journal**: AER: Insights
- **Paper ID**: 202864-V1

## Baseline Groups

### G1: Individual-level mixing attitude prediction
- **Claim**: Mixing attitude (att_std) positively predicts multiple mixing behavior in the ambiguous domain.
- **Baseline spec_ids**: baseline/table1_ambiguous, baseline/table1/ambiguous (identical regressions)
- **Key result**: Coefficient = 0.066, p = 0.27, N = 88
- **Note**: This is the paper individual-level claim. It is consistently positive but never statistically significant across specifications.

### G2: Domain-level mixing differences
- **Claim**: There is more multiple mixing in the ambiguous domain than in the risk domain.
- **Baseline spec_ids**: baseline/multiple_mixing
- **Key result**: Coefficient = -0.25, p < 0.001, N = 341
- **Note**: This is the paper strongest result and primary contribution.

### Not a meaningful baseline: baseline/topic
- This spec regresses mixing_intensity_std (not multiple_mixing) on topic. The coefficient is effectively zero (~2.8e-16). It does not correspond to either core claim and is classified as invalid.

## Summary Counts

| Metric | Count |
|--------|-------|
| Total specifications | 64 |
| Core test specifications | 40 |
| Non-core specifications | 22 |
| Invalid specifications | 2 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 9 |
| core_fe | 1 |
| core_funcform | 3 |
| core_inference | 5 |
| core_method | 8 |
| core_sample | 14 |
| invalid | 2 |
| noncore_alt_outcome | 7 |
| noncore_diagnostic | 5 |
| noncore_heterogeneity | 9 |
| unclear | 0 |

## Core vs Non-core Classification Logic

Most robustness specifications in this search test the G1 claim (att_std -> multiple_mixing in ambiguous domain). The key classification decisions:

1. **Control variations** (drop/add controls) that preserve att_std as treatment and multiple_mixing as outcome in the ambiguous domain are **core tests** of G1.
2. **Sample restrictions** that change the domain (risk_only, stock_only, etc.) are classified as **core_sample** because they test the same regression in a different subsample, though the ambiguous domain is the primary domain of interest.
3. **Alternative outcomes** (probability, interval_length, cooperation) are **non-core** because they change the estimand.
4. **Heterogeneity interactions** are **non-core** because they test effect modification, not the main effect.
5. **Inference variations** (HC1/HC2/HC3, clustering) are **core** because they only change standard errors.
6. **Domain comparison t-tests** that compare risk vs ambiguous directly test G2 and are core. Other pairwise domain comparisons are non-core diagnostics.
7. **Table 2 cooperation analyses** represent an entirely separate claim and are non-core.

## Top 5 Most Suspicious Rows

1. **baseline/topic** (spec_id row 1): The coefficient is ~0 (2.8e-16) with p~1.0. This uses mixing_intensity_std as outcome, which is different from the paper primary outcome multiple_mixing. It appears to be a mis-specified baseline that does not correspond to any paper claim.

2. **robust/control/drop_att_std**: When att_std is dropped, the treatment variable changes to risk_std. This is no longer testing G1 (att_std -> multiple_mixing). The spec search script appears to use remaining[0] as the treatment after dropping, which is incorrect when att_std itself is the treatment being dropped.

3. **robust/het/high_mixing_attitude** and **robust/het/low_mixing_attitude**: These split the sample by median attitude and then report the coefficient on risk_std (not att_std). Since the treatment of interest is removed from variation by construction, these do not test G1.

4. **robust/control/add_1**: This spec has identical results to robust/control/none (same coefficient, SE, p-value), which suggests the control progression may have a bug where add_1 only includes att_std as a control rather than adding a new control on top of att_std.

5. **robust/funcform/logit_transform**: Uses p_logit (log-odds of probability belief) as outcome, which is a different concept from multiple_mixing. This is more properly an alternative outcome than a functional form variation.

## Recommendations for Spec-Search Script

1. **Fix baseline/topic**: Either remove it or change the outcome to multiple_mixing to match the paper main claim.
2. **Fix drop_att_std**: When dropping the treatment variable itself, the spec should be excluded or flagged rather than using the first remaining control as the treatment.
3. **Fix control progression**: robust/control/add_1 appears identical to robust/control/none. The control progression should incrementally add controls beyond att_std, not start from att_std alone.
4. **Fix high/low mixing attitude**: These should either (a) still include att_std as treatment or (b) be clearly labeled as heterogeneity analyses where the treatment is risk_std.
5. **Separate domain t-tests from placebo tests**: The pairwise domain t-tests are better classified as diagnostic analyses rather than placebo tests. The risk_vs_ambiguous comparison directly supports G2 and should be distinguished from supplementary pairwise comparisons.
6. **Remove duplicate baseline**: baseline/table1_ambiguous and baseline/table1/ambiguous are identical. One should be removed.
