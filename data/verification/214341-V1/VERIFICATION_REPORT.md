# Verification Report: 214341-V1

**Paper**: "Who Benefits from the Online Gig Economy?" (Stanton and Thomas, AER)
**Paper ID**: 214341-V1
**Verified**: 2026-02-03
**Verifier**: verification_agent

---

## 1. Baseline Groups

Four baseline groups were identified, corresponding to four distinct surplus measures. Each group has two baseline specifications (unweighted and IPW-weighted).

| Group | Outcome Variable | Unweighted Coef | Weighted Coef | N | Claim |
|-------|-----------------|-----------------|---------------|---|-------|
| G1 | hrly_surp_rel_min | 1.238 | 1.377 | 84 | Workers earn ~24-38% markup relative to min WTA |
| G2 | hrly_surp_rel_expost | 1.159 | 1.154 | 84 | Workers earn ~15-16% markup relative to ex-post WTA |
| G3 | hrly_surp_rel_outside | 1.498 | 1.228 | 46 | Workers earn ~23-50% markup relative to outside wage |
| G4 | fixed_surp_rel_expost | 1.869 | 1.617 | 99 | Workers earn ~62-87% markup on fixed-price contracts |

**Baseline spec_ids**: baseline (unweighted) and baseline_weighted (IPW-weighted), each appearing 4 times for the 4 outcomes. Note: the spec_id alone is not unique; the combination of spec_id + outcome_var is unique.

---

## 2. Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **164** |
| Baseline specs | 8 |
| Core test specs (incl. baselines) | 128 |
| Non-core specs | 36 |
| Invalid | 0 |
| Unclear | 0 |

### Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_method | 36 | Baselines (8), quantile analyses (20), weight variants (8) |
| core_sample | 80 | Subgroup restrictions (48), outlier treatment (24), bid framing (8) |
| core_funcform | 8 | Log and IHS transformations of surplus measures |
| core_inference | 4 | Bootstrap inference (1000 reps) for each surplus measure |
| noncore_diagnostic | 18 | Balance tests (3), control regressions (3), OLS predictors (12) |
| noncore_heterogeneity | 16 | Interaction/subgroup regressions testing differential effects |
| noncore_placebo | 2 | Balance/placebo tests on non-surplus outcomes |

---

## 3. Classification Rationale

### Core tests (128 specs)

The core claim is that workers earn positive surplus (markup > 1). A specification is classified as **core** if it computes or tests the mean/quantile/transformed version of one of the four surplus measures. This includes:

- **Sample restrictions** (80 specs): US-only, non-US, high/low experience, high/low rate, tercile splits, bid framing splits, winsorization, and trimming. All compute the same mean surplus in subsamples.
- **Quantile analysis** (20 specs): 10th, 25th, 50th, 75th, 90th percentiles of each surplus measure. These characterize the distribution of surplus and test whether it is positive throughout.
- **Functional form** (8 specs): Log and IHS transformations of each surplus measure. These preserve the directionality of the claim.
- **Bootstrap inference** (4 specs): Same point estimates with bootstrap standard errors.
- **Weight variants** (8 specs): Trimmed and normalized IPW weights.
- **Baselines** (8 specs): Unweighted and weighted means of each surplus measure.

### Non-core tests (36 specs)

- **Heterogeneity** (16 specs): These regress surplus on c_US, high_experience, high_rate, or interaction terms. The treatment variable is no longer mean_surplus but a demographic characteristic. These test differential surplus, not whether surplus is positive. Classified as noncore_heterogeneity.

- **OLS predictors** (12 specs): These are multivariate OLS regressions of surplus on log_numjobs, log_profile_rate, and c_US simultaneously. The coefficient reported is for one predictor conditional on others. These test what predicts surplus levels, not whether surplus is positive. Classified as noncore_diagnostic.

- **Control variation regressions** (3 specs): These regress hrly_surp_rel_min on log_numjobs (with progressive addition of controls). The treatment variable is log_numjobs, not mean_surplus. Different estimand. Classified as noncore_diagnostic.

- **Balance tests** (3 specs): These test whether survey respondents differ from non-respondents on numjobs, scraped_profile_rate, and c_US. Completely different outcomes and treatment. Classified as noncore_diagnostic.

- **Placebo tests** (2 specs): These regress scraped_profile_rate and numjobs on InSurvey. Different outcomes and treatment. Classified as noncore_placebo.

---

## 4. Top 5 Most Suspicious/Noteworthy Rows

1. **Rows 12-14 (robust/control/add_*)**: Treatment variable is log_numjobs, not mean_surplus. These are OLS regressions predicting surplus from worker characteristics, not testing whether surplus > 1. They appear to duplicate the OLS predictor specs (rows 143-155) but are classified under control progression. The reported coefficient is for log_numjobs, which is a different estimand from the baseline claim.

2. **Row 136 (robust/weights/normalized_hrly_surp_rel_min)**: Coefficient (1.377) and SE (0.103) are identical to baseline_weighted for G1. The normalized weights specification appears to produce exactly the same result as the IPW-weighted baseline, suggesting it may be a duplicate.

3. **Row 139 (robust/weights/normalized_hrly_surp_rel_expost)**: Same issue as above -- coefficient and SE match baseline_weighted for G2 exactly. Likely duplicate.

4. **Rows 155-156 (placebo tests)**: These are identical to balance_test/scraped_profile_rate and balance_test/numjobs (rows 10-11). Same coefficients, same SEs, same outcomes. They appear twice under different spec_ids (balance_test/ vs robust/placebo/). Data duplication issue.

5. **Rows 143-146 vs 12-14**: The OLS predictor spec for log_numjobs on hrly_surp_rel_min (row 143) reports the same coefficient as robust/control/add_c_US (row 14) because they are the same regression. The control progression and OLS predictors categories overlap, creating potential double-counting.

---

## 5. Recommendations for Spec-Search Script

1. **De-duplicate placebo/balance tests**: Rows 10-11 and 155-156 are identical specifications with different spec_ids. The script should avoid recording the same regression under multiple categories.

2. **De-duplicate normalized weights**: The normalized weights specifications appear to produce identical results to baseline_weighted. If the normalization does not change the estimates, these should either be dropped or the weight normalization methodology should be revisited.

3. **Clarify treatment variable for non-mean specs**: The control progression specs (rows 12-14) and OLS predictor specs (rows 143-154) report coefficients on log_numjobs, c_US, etc., while the baseline claim is about mean surplus > 1. These are fundamentally different estimands and should be clearly flagged as auxiliary/diagnostic in the spec search rather than mixed in with robustness checks.

4. **Non-unique spec_ids**: The spec_ids baseline and baseline_weighted each appear 4 times (once per outcome). A unique identifier combining spec_id + outcome_var would be more appropriate for downstream use.

5. **Consider tercile/subsample splits more carefully**: The subgroup splits (US/non-US, high/low experience, terciles) compute mean surplus in subsamples. While these are classified as core_sample here because they preserve the estimand (mean surplus), they could also be viewed as heterogeneity analyses. The current classification treats subsample means as core because they directly test whether surplus > 1 in each subgroup.

---

## 6. Data Quality Notes

- All 164 rows have valid, finite coefficients and standard errors. No invalid rows detected.
- All baseline coefficients are positive and greater than 1, consistent with the paper claim of positive worker surplus.
- The treatment_var column is somewhat misleading: for baselines and sample restrictions, it is mean_surplus (a descriptive statistic label), while for heterogeneity and OLS specs, it contains the actual RHS variable name. This inconsistency should be noted in downstream analyses.
- P-values of 0.0 appear for many specifications, likely due to floating-point underflow for very large t-statistics. These should be interpreted as p < machine epsilon rather than literally zero.
