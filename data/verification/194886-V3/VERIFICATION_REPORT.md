# Verification Report: 194886-V3

## Paper Information
- **Paper ID**: 194886-V3
- **Title**: Resisting Social Pressure in the Household Using Mobile Money: Experimental Evidence on Microenterprise Investment in Uganda
- **Journal**: AEJ-Applied
- **Method**: 3-arm RCT (Cash control, Mobile Account [MA], Mobile Disburse [MD])

## Baseline Groups

### G1: Business Profits (earn_business)
- **Baseline spec_ids**: baseline_earn_business_MA, baseline_earn_business_MD
- **Claim**: MD and MA treatments increase monthly business profits relative to cash control
- **Expected sign**: Positive
- **Key result**: MD coefficient = 17.61 (p < 0.001); MA coefficient = 2.88 (p = 0.42)

### G2: Total Savings (much_saved)
- **Baseline spec_ids**: baseline_much_saved_MA, baseline_much_saved_MD
- **Claim**: MD and MA treatments increase total savings relative to cash control
- **Expected sign**: Positive
- **Key result**: MD coefficient = 8.46 (p = 0.41); MA coefficient = 0.93 (p = 0.92)

### G3: Business Capital (capital)
- **Baseline spec_ids**: baseline_capital_MA, baseline_capital_MD
- **Claim**: MD and MA treatments increase business capital relative to cash control
- **Expected sign**: Positive
- **Key result**: MD coefficient = 69.21 (p = 0.004); MA coefficient = 13.48 (p = 0.58)

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **116** |
| Baselines (is_baseline=1) | 6 |
| Core tests (is_core_test=1) | 68 |
| Non-core tests (is_core_test=0) | 48 |
| Invalid | 0 |
| Unclear | 0 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 22 |
| core_sample | 32 |
| core_inference | 6 |
| core_fe | 4 |
| core_funcform | 4 |
| noncore_alt_outcome | 10 |
| noncore_heterogeneity | 36 |
| noncore_placebo | 2 |
| noncore_alt_treatment | 0 |
| noncore_diagnostic | 0 |

Note: The 6 baselines are included in the core_controls count (is_baseline=1 AND is_core_test=1).
Total core = 22 + 32 + 6 + 4 + 4 = 68 (including 6 baselines).
Total non-core = 10 + 36 + 2 = 48.
Grand total = 68 + 48 = 116.

## Top 5 Most Suspicious Rows

1. **robust/se/hc1_MA and robust/se/hc1_MD**: These produce identical coefficients and standard errors as the baseline specifications. The baseline already uses HC1/robust standard errors, so these are exact duplicates. Classified as core_inference but flagged as redundant.

2. **robust/control/baseline_only_MA and robust/control/baseline_only_MD**: Also identical to baselines. The "baseline only" control variation is the same specification as the baseline. Classified as core_controls but flagged as duplicates.

3. **primary/much_saved_MA, primary/much_saved_MD, primary/capital_MA, primary/capital_MD**: These 4 specs duplicate the baseline specs for much_saved and capital exactly (same coefficients, same SE). They appear to be re-runs or re-extractions of the same regressions. Classified as core_controls.

4. **robust/sample/high_profit_base_* and robust/sample/low_profit_base_***: These split the sample by baseline profit level. This is borderline between a sample restriction (core) and a heterogeneity analysis (non-core). Classified as core_sample with lower confidence (0.80) since the split is by a pre-treatment characteristic and the full-sample effect is still the primary claim.

5. **robust/sample/married_only_* and robust/sample/unmarried_only_***: Similarly, these split by marital status. Borderline between sample restriction and heterogeneity. Classified as core_sample with confidence 0.85.

## Key Design Notes

- This is a 3-arm RCT where both MA (treatment2) and MD (treatment3) are included in every regression. The spec search records each coefficient as a separate row, doubling the apparent spec count.
- The paper's strongest and most emphasized result is the MD effect on earn_business (G1). Most robustness checks are concentrated on this outcome.
- Heterogeneity analyses are extensive (36 rows = 9 moderator variables x 2 coefficients [main + interact] x 2 treatment arms). All are classified as non-core because the interaction model changes the estimand of the main treatment effect.
- The heterogeneity "main" effect rows report the treatment coefficient when the moderator equals zero, which is not the same estimand as the unconditional ATE in the baseline.

## Recommendations for Spec-Search Script

1. **Deduplicate exact copies**: The script should detect and skip specs that produce identical coefficient/SE as an existing spec (e.g., robust/se/hc1 duplicating baseline, robust/control/baseline_only duplicating baseline, primary/* duplicating baselines).
2. **Separate heterogeneity interaction terms from main effects**: The script records both the "main" treatment coefficient and the interaction coefficient from heterogeneity regressions. These should be flagged distinctly in spec_tree_path.
3. **Consider consolidating MA/MD rows**: Since both coefficients come from the same regression, the spec search could record them as a single specification with a vector of coefficients, rather than two separate rows.
