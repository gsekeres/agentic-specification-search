# Verification Report: 120483-V1

## Paper
**Malaria Ecology and the Spread of Slavery in the Early United States**
- Journal: AEJ Applied Economics
- Core hypothesis: Higher malaria ecology is associated with higher slave shares in US counties

---

## Baseline Groups

### G1: Cross-sectional malaria-slavery relationship
- **Claim**: Higher malaria ecology (MAL) is associated with a higher share of slaves (slaveratio), estimated via county-level cross-sectional OLS with state fixed effects.
- **Baseline spec_ids**: baseline, baseline_crop, baseline_full, baseline_1860, baseline_1860_full, baseline_slave_states_1790, baseline_slave_states_1860
- **Expected sign**: Positive
- **Notes**: Seven baselines span 1790 and 1860 cross-sections with varying controls (none, crop, full) and samples (all counties, slave states). The primary fully-controlled specifications are baseline_full (1790) and baseline_1860_full (1860). Notably, baseline_full (1790 with full controls) has p=0.156, failing conventional significance.

### G2: Panel malaria-slavery relationship
- **Claim**: Malaria ecology affected black population shares in a panel of US states from 1630-1780, using state and year fixed effects.
- **Baseline spec_ids**: panel/baseline
- **Expected sign**: Positive
- **Notes**: Uses different treatment variable (mal1690_x_ME1790_std) and outcome (black_totalpop) from G1. Distinct identification strategy.

---

## Summary Counts

| Metric | Count |
|--------|-------|
| Total specifications | 76 |
| Core test specs | 62 |
| Non-core specs | 14 |
| Invalid specs | 0 |
| Unclear specs | 0 |
| Baseline specs | 8 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 28 |
| core_sample | 22 |
| core_inference | 2 |
| core_fe | 2 |
| core_funcform | 4 |
| core_method | 4 |
| noncore_heterogeneity | 6 |
| noncore_placebo | 1 |
| noncore_alt_outcome | 7 |
| invalid | 0 |
| unclear | 0 |

---

## Top 5 Most Suspicious Rows

1. **robust/control/none** (spec_id: robust/control/none): This is an exact duplicate of the baseline spec (no controls, 1790). The coefficient and SE match baseline exactly (coef=0.1636, SE=0.0502). Classified as core but flagged as redundant.

2. **robust/control/crop_only** (spec_id: robust/control/crop_only): Coefficient (0.1544) and SE (0.0318) match baseline_crop exactly. This is a duplicate specification. Classified as core but flagged as redundant.

3. **robust/heterogeneity/slave_state** (spec_id: robust/heterogeneity/slave_state): The main MAL coefficient here is -0.024 (p=0.68), which is negative and insignificant. However, the interaction term MAL_x_slavestate is 0.224 (p=0.012). The reported coefficient represents the effect in non-slave states, not the overall effect. This is heterogeneity analysis, not a core test of the main claim.

4. **cross_country/americas_minimal** and **cross_country/americas_full**: These use a completely different sample (US, Brazil, Cuba states) and outcome variable (coloredratio). While testing a related hypothesis, the sample and outcome are sufficiently different from the county-level US analysis to classify as non-core alternative outcomes.

5. **robust/outcome/blackratio**: Uses blackratio_1860 instead of slaveratio. These are closely related measures (free black + slave vs slave only), so this is classified as a core functional form variation. However, there is ambiguity about whether this tests the same estimand.

---

## Recommendations

1. **Deduplicate**: robust/control/none is identical to baseline, and robust/control/crop_only is identical to baseline_crop. Future spec searches should detect and flag duplicates.

2. **Heterogeneity specs need careful interpretation**: The heterogeneity interaction specs (cotton, slave_state, elevation, coastal, temperature, precipitation) report the base MAL coefficient, which is the effect conditional on the interaction variable being zero/low. This is not the same as the average effect. These should not be included as core tests.

3. **Panel specs form a distinct group**: The panel specifications (G2) use fundamentally different data, time period, treatment variable, and outcome from the cross-sectional analysis (G1). They should be analyzed separately.

4. **Political outcomes are downstream mechanisms**: The vote share outcomes (Lincoln, Breckinridge, Grant, Seymour, proslavery) test whether malaria affects political outcomes via slavery, which is a downstream mechanism test, not a test of the core malaria-slavery claim.

5. **Cross-country specs differ fundamentally**: The Americas cross-country analysis (US, Brazil, Cuba) uses different data and a different outcome (coloredratio). While conceptually related, it is not a robustness check of the county-level US analysis.
