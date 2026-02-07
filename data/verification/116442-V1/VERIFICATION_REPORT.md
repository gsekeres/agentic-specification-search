# Verification Report: 116442-V1

## Paper Information
- **Paper ID**: 116442-V1
- **Title**: Competition and the Use of Foggy Pricing
- **Journal**: AEJ-Microeconomics
- **Authors**: Miravete et al.

## Baseline Groups

### G1: Effect of duopoly entry on foggy pricing
- **Claim**: Duopoly entry in US cellular markets increases the number of foggy (dominated) tariff plans offered by wireline carriers.
- **Baseline spec_id**: baseline
- **Outcome**: log_FOGGYi (log count of foggy/dominated tariff plans)
- **Treatment**: DUOPOLY (0/1 indicator for duopoly market structure)
- **Fixed effects**: Market + Time two-way FE
- **Controls**: AP_PEAK + AP_OFFP (Arrow-Pratt risk aversion indices)
- **Clustering**: Market level
- **Baseline coefficient**: 0.314, SE=0.203, p=0.125 (not significant at conventional levels)

## Summary Counts

| Category | Count |
|----------|-------|
| Total specifications | 56 |
| Baseline | 1 |
| Core tests (incl. baseline) | 39 |
| Non-core tests | 17 |
| Invalid | 0 |
| Unclear | 0 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_controls | 12 | Control variable additions/removals |
| core_sample | 13 | Sample restriction variations |
| core_fe | 4 | Fixed effects structure variations |
| core_funcform | 6 | Functional form/outcome measure variations |
| core_inference | 2 | Clustering/SE variations |
| core_method | 2 | Estimation method variations (baseline + first diff) |
| noncore_heterogeneity | 8 | Interaction terms testing effect heterogeneity |
| noncore_placebo | 1 | Fake treatment timing test |
| noncore_alt_outcome | 2 | Different outcome concepts (total plans, effective plans) |
| noncore_diagnostic | 6 | Event-study dynamic effects with different treatment vars |
| **TOTAL** | **56** | |

## Key Issues and Suspicious Rows

### 1. Exact duplicate: did/fe/twoway = baseline
The spec did/fe/twoway produces exactly the same coefficient (0.3144), SE (0.2033), and p-value (0.1252) as the baseline. These are the same specification with different spec_ids. This inflates the effective spec count by 1.

### 2. Collinear controls produce identical results
The specs robust/control/add_MULTIMKT, robust/control/add_MKT_AGE, robust/control/add_step3, robust/control/add_step4, and robust/funcform/quadratic_treatment all produce coefficients identical to baseline (0.3144). This occurs because the added regressors (MULTIMKT, MKT_AGE, DUOPOLY_sq) are perfectly collinear with the market fixed effects and are dropped by the estimator. The DUOPOLY variable is binary, so DUOPOLY_sq = DUOPOLY. These specs do not provide genuinely new information.

### 3. No-observation time periods: drop_time_20 and drop_time_25
These two sample restriction specs produce results identical to baseline, indicating there are no observations at TIME=20 or TIME=25 in the data (the data spans TIME periods that do not include these values). These specs provide no additional information.

### 4. Duplicate outcome specs: robust/outcome/FOGGYi_levels and robust/funcform/levels
These two specs produce identical results (coef=0.124, SE=0.184, p=0.502). They run the same regression with FOGGYi_count as the outcome. This is a duplication across two robustness categories.

### 5. robust/control/add_step5 = did/controls/full and robust/control/add_step6 = robust/control/add_step5
Step 5 and did/controls/full are the same specification (full control set). Step 6 adds BELL, which is collinear with market FE and dropped, producing the same result as step 5.

### Top 5 Most Suspicious Rows

1. **did/fe/twoway** (confidence: 0.95) -- Exact duplicate of baseline; should be deduplicated or flagged.
2. **robust/control/add_MULTIMKT** (confidence: 0.85) -- Controls field says MULTIMKT added but coefficient unchanged; collinear with FE.
3. **robust/funcform/quadratic_treatment** (confidence: 0.80) -- DUOPOLY_sq added but DUOPOLY is binary, so DUOPOLY_sq = DUOPOLY; coefficient unchanged.
4. **robust/sample/drop_time_20** (confidence: 0.70) -- No observations dropped; identical to baseline.
5. **robust/sample/drop_time_25** (confidence: 0.70) -- No observations dropped; identical to baseline.

## Recommendations for Fixing the Spec-Search Script

1. **Detect and flag collinear controls**: Before reporting, check whether adding a control changes the coefficient. If the coefficient is identical (within floating-point tolerance), flag the spec as collinear control dropped rather than a genuine robustness variation.

2. **Detect empty sample restrictions**: When dropping observations by a filter value, check whether any observations were actually dropped (compare N before and after). Flag specs where N is unchanged.

3. **Avoid binary-variable squared terms**: The script adds DUOPOLY_sq for a binary variable, which is identical to DUOPOLY itself. This should be skipped when the treatment is binary.

4. **Deduplicate identical specs**: The did/fe/twoway spec is programmatically generated alongside the baseline but uses the same specification. Consider checking for duplicates before outputting.

5. **Separate the robust/outcome/FOGGYi_levels and robust/funcform/levels** entries to avoid double-counting the same regression.

6. **Event study coefficients**: The dynamic lag specs use different treatment variables (post_1 through post_6) rather than DUOPOLY. These are event-time indicators from a single regression, not independent robustness checks of the main claim. Consider classifying them as diagnostic/event-study rather than as independent specs.

7. **Consider Poisson PML**: The original GAUSS code appears to use Poisson pseudo-maximum likelihood (PPML) for count outcomes. The current spec search uses OLS with log outcomes. Adding a Poisson specification would better match the paper actual methodology.
