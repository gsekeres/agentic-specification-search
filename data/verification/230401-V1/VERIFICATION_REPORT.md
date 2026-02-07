# Verification Report: 230401-V1

## Paper: Racial Discrimination in Small Business Lending
**Journal**: AER  
**Verified**: 2026-02-03  
**Verifier**: verification_agent

---

## Baseline Groups Found

### G1: Black-owned business interest rate discrimination (PRIMARY)
- **Baseline spec_ids**: baseline (treatment_var=black_50)
- **Outcome**: loanrate_w2 (winsorized interest rate)
- **Treatment**: black_50 (Black ownership >= 50%)
- **Coefficient**: 2.90 pp, p=0.002, N=1366
- **Expected sign**: positive
- **Notes**: This is the paper's primary claim. 54 of 60 specifications use black_50 as the treatment variable.

### G2: Hispanic-owned business interest rate discrimination
- **Baseline spec_ids**: baseline (treatment_var=hisp_50)
- **Outcome**: loanrate_w2
- **Treatment**: hisp_50 (Hispanic ownership >= 50%)
- **Coefficient**: 2.93 pp, p=0.008, N=1550

### G3: Asian-owned business interest rate discrimination
- **Baseline spec_ids**: baseline (treatment_var=asian_50)
- **Outcome**: loanrate_w2
- **Treatment**: asian_50 (Asian ownership >= 50%)
- **Coefficient**: 2.56 pp, p=0.009, N=1253

### G4: Native American-owned business interest rate discrimination
- **Baseline spec_ids**: baseline (treatment_var=native_50)
- **Outcome**: loanrate_w2
- **Treatment**: native_50 (Native American ownership >= 50%)
- **Coefficient**: 1.30 pp, p=0.26, N=1324
- **Notes**: Not statistically significant.

---

## Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **60** |
| Baselines | 4 |
| Core tests (incl. baselines) | 40 |
| Non-core tests | 20 |
| Invalid | 1 |
| Unclear | 0 |

### Core test breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_controls | 11 | Control set variations and leave-one-out |
| core_sample | 11 | Sample restrictions (loans/lines, size, trimming) |
| core_method | 7 | Baselines (4) + 3 near-duplicate treatment variants |
| core_funcform | 4 | Log, IHS, quadratic controls, loan spread |
| core_fe | 4 | FE structure variations (no FE, state only, time only, +industry) |
| core_inference | 3 | Clustering variations (HC1, time, industry) |

### Non-core test breakdown

| Category | Count | Description |
|----------|-------|-------------|
| noncore_heterogeneity | 9 | Interaction terms (6) + credit score subsamples (3) |
| noncore_alt_outcome | 7 | Collateral outcomes (coll, collmore, blien, bcoll, pcoll, bpcoll, signature) |
| noncore_placebo | 3 | Predetermined outcomes (term, busage, ceoexp) |
| invalid | 1 | collval_num: missing p-value, near-zero coefficient |

---

## Top 5 Most Suspicious Rows

### 1. robust/outcome/collval_num (INVALID)
- **Issue**: Missing p-value, coefficient essentially zero (0.0000), N=376 (much smaller than baseline)
- **Likely cause**: Convergence or extraction failure. The collval_num variable may have too few non-missing values.
- **Action**: Marked as invalid.

### 2. robust/heterogeneity/creditscore_high_credit (SUSPICIOUS)
- **Issue**: Extreme coefficient (14.98 pp) with N=46. This is roughly 5x the baseline estimate.
- **Likely cause**: Very small subsample of high-credit-score businesses. Point estimate is unreliable.
- **Action**: Classified as noncore_heterogeneity. Would be unreliable even if classified as core.

### 3. robust/sample/nonbank_lenders (SUSPICIOUS DUPLICATE)
- **Issue**: Coefficient (2.9036) and p-value (0.002318) are identical to the baseline with treatment=black_50, and N=1426 (larger than baseline N=1366). The label says "nonbank_lenders" but the results are identical to multiple other specs (e.g., clustering variants also show N=1426 with identical coefficient).
- **Likely cause**: This may be the full sample (including nonbank lenders) rather than a restriction to nonbank lenders only. The baseline may already restrict to the same sample. Not an error per se, but the sample_desc is potentially misleading.
- **Action**: Kept as core_sample. The coefficient identity with baseline suggests the baseline already includes these observations or the filter is not binding.

### 4. robust/treatment/hisp_50, robust/treatment/asian_50, robust/treatment/native_50 (NEAR-DUPLICATES)
- **Issue**: These have slightly different N from their corresponding baselines (e.g., hisp_50 robustness has N=1641 vs baseline N=1550; asian_50 has N=1312 vs N=1253; native_50 has N=1381 vs N=1324) but identical coefficients and p-values.
- **Likely cause**: Identical coefficients suggest same model but the N difference implies a different sample construction (perhaps including observations with missing data on some controls). Coefficients are identical, which is suspicious if samples truly differ.
- **Action**: Classified as core_method under their respective baseline groups. The N discrepancy warrants investigation.

### 5. robust/placebo/term (CONCERNING PLACEBO FAILURE)
- **Issue**: Race (black_50) significantly predicts loan term (coef=-4.26, p=0.006). This is a placebo test failure suggesting potential confounding or selection.
- **Likely cause**: As the SPECIFICATION_SEARCH.md notes, this raises concerns about unobserved confounding. However, this is correctly classified as a placebo test and is appropriately non-core.
- **Action**: Classified as noncore_placebo. The failure does not invalidate the main finding but is noteworthy.

---

## Recommendations for Spec-Search Script

1. **Baseline deduplication**: The four baseline rows all share spec_id="baseline", which makes them harder to distinguish. Consider using distinct spec_ids like "baseline_black", "baseline_hisp", "baseline_asian", "baseline_native" to avoid ambiguity.

2. **Treatment variant specs are near-duplicates**: The robust/treatment/* specs for hisp_50, asian_50, and native_50 are essentially identical to the baselines for those groups. Consider either (a) not generating these if baselines already exist, or (b) making them genuinely different (e.g., pooled regression with all minority indicators simultaneously).

3. **N discrepancies in treatment variants**: The N values differ between baseline and robust/treatment specs for the same treatment variable (e.g., baseline hisp_50 has N=1550 but robust/treatment/hisp_50 has N=1641). This may indicate different sample handling. The coefficients and p-values are identical despite different N, which is impossible unless the extra observations are dropped due to missing data. This should be investigated.

4. **collval_num extraction failure**: The collval_num spec has a missing p-value and near-zero coefficient. The script should handle this outcome variable differently (it may be ordinal or have convergence issues).

5. **Heterogeneity specs report main effects**: The interaction-based heterogeneity specs (woman_owned, family, conditiongood, loss21, loan, newcredit) report the main effect of black_50 rather than the interaction coefficient. While the main effect is still informative, these specs conflate two different purposes. Consider extracting the interaction term separately or flagging these more explicitly.

6. **Sample restriction naming**: The "nonbank_lenders" spec appears to produce identical results to the unrestricted sample, suggesting the restriction is either not binding or the label is wrong. Review the sample construction.
