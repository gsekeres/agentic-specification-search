# Verification Report: 116136-V2

## Paper
- **Title**: Yours, Mine and Ours: Do Divorce Laws Affect the Intertemporal Behavior of Married Couples?
- **Author**: Alessandra Voena (2015)
- **Journal**: American Economic Review
- **Paper ID**: 116136-V2

## Baseline Groups

### G1: Effect of unilateral divorce + community property on household assets
- **Claim**: Unilateral divorce laws combined with community property regimes increase household asset accumulation relative to other property division regimes.
- **Expected sign**: Positive (+)
- **Baseline spec_ids**: baseline, baseline/children, baseline/state_controls, baseline/full
- **Preferred baseline**: baseline/full (Individual FE, full controls, state clustering; coef=16,957, p=0.003)
- **Notes**: The four baselines represent progressive additions of controls. The simplest baseline (Age + Year only) is marginally insignificant (p=0.063), while the full-controls baseline is significant at 1%.

## Summary Counts

| Category | Count |
|----------|-------|
| Total specifications | 66 |
| Baselines | 4 |
| Core tests (incl. baselines) | 45 |
| Non-core tests | 15 |
| Invalid (null/dropped coefficients) | 6 |
| **Total core (is_core_test=1)** | **45** |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 15 |
| core_sample | 20 |
| core_inference | 3 |
| core_fe | 3 |
| core_funcform | 3 |
| core_method | 1 |
| noncore_placebo | 2 |
| noncore_alt_outcome | 2 |
| noncore_alt_treatment | 3 |
| noncore_heterogeneity | 7 |
| noncore_diagnostic | 1 |
| invalid | 6 |

Note: The 4 baselines are counted under core_controls above (they are both is_baseline=1 and is_core_test=1).

## Top 5 Most Suspicious Rows

### 1. panel/method/first_diff (spec 60)
- **Issue**: The first-difference estimator yields a **large negative and highly significant** coefficient (-7,887, p=0.0). This directly contradicts the positive baseline finding (coef=+16,957). First differences and fixed effects should yield similar estimates under standard assumptions. The R-squared is essentially zero (0.0008), and the standard error is implausibly small (430) compared to the FE specification (5,337). This strongly suggests a coding error in the first-difference implementation -- likely the treatment variable was incorrectly differenced (computing changes in the interaction term rather than the interaction of the change in unilateral status with property regime).
- **Classification**: Kept as core_method but flagged as highly suspicious.

### 2. robust/funcform/log_outcome (spec 32)
- **Issue**: Log transformation of assets yields a **negative** coefficient (-0.031, p=0.55), contradicting the positive level effect in the baseline. This is run on positive-assets-only subsample. The sign reversal when moving to logs is concerning because it suggests the level result may be driven by extreme values rather than a broad-based effect. However, the negative coefficient is not significant.
- **Classification**: core_funcform (preserves interpretation as percent change).

### 3. robust/funcform/ihs_outcome (spec 33)
- **Issue**: IHS transformation also yields a **negative** coefficient (-0.35, p=0.21). Like the log specification, this contradicts the positive level effect. IHS is defined for all values (including zeros), so this is on the full sample. The sign reversal from level to IHS is suspicious.
- **Classification**: core_funcform (preserves interpretation).

### 4. robust/placebo/fake_timing_minus5 (spec 47)
- **Issue**: The placebo test with fake treatment 5 years earlier yields a **positive** coefficient of 26,765 (p=0.11) -- larger than the actual baseline effect (16,957). While not significant, this suggests potential pre-trends that could confound the main result. A well-identified treatment should show near-zero placebo effects.
- **Classification**: noncore_placebo (correctly classified as non-core).

### 5. robust/sample/late_period, eqdistr_only, title_only, age_older, 1980s, 1990s (specs 21, 25, 26, 42, 65, 66)
- **Issue**: Six specifications have **null/missing coefficients** because the treatment variable uni_comprop was dropped due to perfect collinearity within the restricted subsample. This is expected for geographic subsamples (equitable distribution states and title states have no community property variation by definition) but unexpected for temporal subsamples (late period, 1980s, 1990s), suggesting limited temporal variation in later years.
- **Classification**: All marked as invalid.

## Recommendations for Spec-Search Script

1. **Fix first-difference implementation**: The first-difference specification (panel/method/first_diff) likely has a coding error. The treatment variable should be the interaction of the *change* in unilateral divorce status with the (time-invariant) property regime, not the first difference of the interaction. Verify the implementation and re-run.

2. **Handle collinear subsamples gracefully**: The script should detect when the treatment variable is dropped (null coefficient) and either skip the specification or report it as failed with a clear reason, rather than reporting null values.

3. **Consider duplicate detection**: spec_id robust/sample/drop_first_year and robust/sample/drop_year_1967 appear to be identical (both exclude year 1967, same coefficient 12,145 and p-value 0.299). The spec search script may have generated duplicate specifications under different names.

4. **Functional form sign reversal deserves attention**: The log and IHS specifications showing negative signs (opposite to baseline) is a substantive finding that should be highlighted more prominently in the search summary.

5. **Add pre-trend tests**: The current placebo tests only shift timing by 5 years. Adding event-study-style pre-trend coefficients would better assess parallel trends.
