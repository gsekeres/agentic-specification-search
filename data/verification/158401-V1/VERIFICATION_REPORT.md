# Verification Report: 158401-V1

## Paper: Market Access and Quality Upgrading: Evidence from Four Field Experiments

**Paper ID**: 158401-V1
**Journal**: AER
**Verified**: 2026-02-03
**Verifier**: verification_agent

---

## 1. Baseline Groups

Five baseline groups were identified, corresponding to five distinct outcome variables in the paper's main results:

| Group | Claim | Baseline spec_id | Outcome | Significant? |
|-------|-------|-------------------|---------|--------------|
| G1 | Market access increases yield | baseline_yield | yield_ha_ton | Yes (p=0.025) |
| G2 | Market access increases price | baseline_price_ugx | price_ugx | Yes (p=0.002) |
| G3 | Market access increases harvest | baseline_harvest_ton | harvest_ton | No (p=0.504) |
| G4 | Market access increases harvest value | baseline_harvest_value_ugx | harvest_value_ugx | No (p=0.252) |
| G5 | Market access affects share sold | baseline_share_sold | share_sold | No (p=0.965) |

The primary claim and focus of most robustness checks is **G1 (yield)**.

---

## 2. Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | 54 |
| **Baselines** | 5 |
| **Core test specs (incl. baselines)** | 39 |
| **Non-core specs** | 10 |
| **Invalid specs** | 5 |
| **Unclear specs** | 0 |

### Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_controls | 12 | Control set variations (5 baselines + 7 robustness) |
| core_sample | 16 | Sample restriction variations (season drops, winsorization, trimming, balanced, jackknife) |
| core_fe | 3 | Fixed effects variations (none, village, two-way) |
| core_inference | 2 | Clustering variations (robust HC, household cluster) |
| core_funcform | 4 | Functional form (log, IHS, standardized) |
| core_method | 2 | Estimation method (first differences, pooled OLS) |
| noncore_placebo | 1 | Pre-treatment placebo test |
| noncore_alt_outcome | 1 | Acreage (not a baseline outcome) |
| noncore_heterogeneity | 8 | Interaction terms and subgroup splits (HH size, distance, time, baseline yield) |
| invalid | 5 | Exact duplicates of baseline specs (panel/fe/season + 4 robust/outcome/*) |

---

## 3. Core Tests by Baseline Group

| Baseline Group | Core Tests (incl. baseline) | Non-core | Invalid |
|----------------|----------------------------|----------|---------|
| G1 (yield) | 29 | 0 | 1 (panel/fe/season duplicate) |
| G2 (price) | 2 | 0 | 1 (robust/outcome/price_ugx duplicate) |
| G3 (harvest) | 2 | 0 | 1 (robust/outcome/harvest_ton duplicate) |
| G4 (harvest value) | 1 | 0 | 1 (robust/outcome/harvest_value_ugx duplicate) |
| G5 (share sold) | 1 | 0 | 1 (robust/outcome/share_sold duplicate) |
| No group | 0 | 10 | 0 |

---

## 4. Top 5 Most Suspicious Rows

1. **panel/fe/season** (Row 6): This is an exact duplicate of baseline_yield. Same coefficient (0.2939), SE (0.1208), p-value (0.025), and N (640). Marked as invalid. The spec search apparently re-ran the baseline under a different spec_id.

2. **robust/outcome/harvest_ton** (Row 27): Exact duplicate of baseline_harvest_ton. Same coefficient (0.2528), SE (0.3711), p-value (0.504), N (658). Marked invalid.

3. **robust/outcome/price_ugx** (Row 28): Exact duplicate of baseline_price_ugx. Same coefficient (59.76), SE (16.85), p-value (0.002), N (623). Marked invalid.

4. **robust/outcome/harvest_value_ugx** (Row 29): Exact duplicate of baseline_harvest_value_ugx. Same coefficient (287146.59), SE (243017.55), p-value (0.252), N (625). Marked invalid.

5. **robust/outcome/share_sold** (Row 30): Exact duplicate of baseline_share_sold. Same coefficient (-0.0016), SE (0.0367), p-value (0.965), N (625). Marked invalid.

All five suspicious rows are exact coefficient/SE/p-value duplicates of their corresponding baselines, meaning the "robustness/measurement" category simply re-reports baseline results without any actual variation.

---

## 5. Notes on Heterogeneity Classifications

Several specs were classified as **noncore_heterogeneity** rather than core tests:

- **robust/heterogeneity/hh_size** (Row 38): The coefficient (0.403) is for treatment interacted with HH size. The main treatment effect in this interaction model does not have the same interpretation as the baseline average treatment effect.
- **robust/sample/small_hh** and **robust/sample/large_hh** (Rows 39-40): These are subsample splits by HH size. While they use the same outcome and treatment, they estimate a conditional ATE for a subgroup, not the overall ATE.
- **robust/heterogeneity/distance** (Row 41) and **robust/sample/close_to_road** / **robust/sample/far_from_road** (Rows 42-43): Same reasoning as above for distance-based splits.
- **robust/heterogeneity/baseline_yield** (Row 44): Interaction with baseline yield changes the estimand.
- **custom/treatment_x_time** (Row 52): Treatment-time interaction estimates a dynamic treatment effect, not the average.

---

## 6. Notes on Functional Form Variations

- **robust/funcform/ihs_yield** and **robust/funcform/standardized** are classified as core for G1 because IHS and standardization preserve the direction and approximate interpretation of the yield effect.
- **robust/funcform/log_harvest** is classified as core for G3 and **robust/funcform/log_price** as core for G2, as log transforms preserve the direction of the effect and are standard functional form alternatives.

---

## 7. Notes on First Differences

- **panel/method/first_diff** uses outcome "yield_fd" (first-differenced yield). While this changes the dependent variable, first differencing is a standard alternative panel estimator that targets the same treatment effect. Classified as core_method for G1 with lower confidence (0.80) due to the outcome transformation.

---

## 8. Recommendations for Spec Search Script

1. **Eliminate duplicate specs**: The "robustness/measurement" category (robust/outcome/*) re-runs exact copies of the baseline specs for alternative outcomes. These should either be removed or should actually vary something (e.g., add controls, change FE). The spec search script should check for coefficient equality before recording a new spec.

2. **Separate heterogeneity from robustness**: Interaction terms and subsample splits by covariates should be clearly tagged as heterogeneity, not mixed into sample_restrictions. The script currently places subsample splits (small_hh, large_hh, close_to_road, far_from_road) under robustness/sample_restrictions, but these are heterogeneity analyses.

3. **Add more robustness for non-yield outcomes**: G2-G5 have almost no robustness checks beyond the duplicated baselines. Consider adding FE variations, control variations, and sample restrictions for price, harvest, harvest value, and share sold.

4. **Clarify ANCOVA vs. no-control distinction**: did/controls/none and robust/loo/drop_ancova appear to produce the same regression (same coef 0.311, SE 0.131). The script should deduplicate these or note the equivalence.
