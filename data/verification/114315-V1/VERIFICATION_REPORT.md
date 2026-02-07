# Verification Report: 114315-V1

## Paper
**The Geography of Trade in Online Transactions: Evidence from eBay and MercadoLibre**
- Journal: AEJ: Microeconomics
- Method: Gravity equation with panel fixed effects
- Data: Simulated (original eBay data is confidential)

---

## Baseline Groups

### G1: Home Bias (Same-State Effect on Transaction Counts)
- **Baseline spec_ids**: baseline, baseline_gravity_samestate
- **Claim**: Being in the same state significantly increases online trade, even controlling for distance and state FE.
- **Expected sign**: Positive
- **Primary baseline**: baseline (two-way state FE, coeff = 3.307, SE = 0.088)
- **Secondary baseline**: baseline_gravity_samestate (no FE, OLS with population controls, coeff = 3.294, SE = 0.082)

### G2: Distance Elasticity
- **Baseline spec_ids**: baseline_gravity_basic
- **Claim**: Distance negatively affects online trade volume (gravity model prediction).
- **Expected sign**: Negative
- **Baseline**: baseline_gravity_basic (lndist on lntcount, coeff = -1.266, SE = 0.032)

### G3: Home Bias on Trade Volume
- **Baseline spec_ids**: baseline_volume
- **Claim**: Home bias holds for dollar volume, not just counts.
- **Expected sign**: Positive
- **Baseline**: baseline_volume (samestate on lntvol, coeff = 3.301, SE = 0.103)

---

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **57** |
| Baselines | 4 |
| Core tests (including baselines) | 43 |
| Non-core tests | 14 |
| Invalid | 0 |
| Unclear | 0 |

### Category Breakdown

| Category | Count |
|----------|-------|
| core_method (baselines) | 4 |
| core_controls | 11 |
| core_sample | 13 |
| core_inference | 4 |
| core_fe | 4 |
| core_funcform | 7 |
| noncore_heterogeneity | 10 |
| noncore_placebo | 2 |
| noncore_alt_outcome | 1 |
| noncore_alt_treatment | 1 |
| noncore_diagnostic | 0 |
| invalid | 0 |
| unclear | 0 |

Verification: 4 + 11 + 13 + 4 + 4 + 7 = 43 core; 10 + 2 + 1 + 1 = 14 non-core; 43 + 14 = 57 total.

---

## Top 5 Most Suspicious Rows

1. **robust/sample/winsor_5pct** (confidence: 0.6): The samestate coefficient flips to -0.48 (negative, significant). This is the only specification where the home bias effect reverses sign. With simulated data, 5% winsorization likely truncates the same-state pairs (which have high trade counts) too aggressively. This should be investigated further; the spec is technically a core sample test but the result is anomalous and may reflect a data processing artifact rather than genuine fragility.

2. **robust/outcome/binary_above_median** (confidence: 0.8): The samestate coefficient is -0.58 (negative). A binary above-median outcome fundamentally changes the estimand from how-much-more-trade to probability-of-high-trade. The negative sign may reflect that same-state pairs are disproportionately high-trade in levels, making the binary coding behave counter-intuitively with FE. Classified as non-core alternative outcome.

3. **robust/sample/cross_state_only** (confidence: 0.7): This specification drops same-state pairs entirely and uses lndist as the treatment variable. It cannot test the home bias claim (G1) because samestate is undefined for this subsample. Assigned to G2 as a sample restriction, but the mapping is imperfect since it fundamentally changes the estimand.

4. **robust/treatment/same_region** (confidence: 0.85): Uses sameregion instead of samestate as treatment. The coefficient is -0.18, negative and significant. This changes the causal object from state-level to region-level home bias, which is a fundamentally different claim. The negative sign makes sense if controlling for distance already absorbs region-level proximity. Classified as non-core alternative treatment.

5. **panel/fe/twoway** (confidence: 0.95): This specification produces identical coefficients and standard errors as baseline. It appears to be a duplicate rather than a meaningfully different specification. Not suspicious per se, but inflates the specification count without adding information.

---

## Recommendations

1. **Winsorization robustness**: The 5% winsorization sign flip warrants investigation. Consider adding intermediate winsorization levels (2%, 3%) to understand the tipping point.

2. **Duplicate detection**: panel/fe/twoway and baseline are identical (same coefficient, SE, p-value, n_obs, R-squared). The spec search should detect and flag exact duplicates to avoid inflating the count.

3. **Treatment variable consistency**: Specs like robust/sample/cross_state_only and robust/treatment/distance_only switch the treatment variable from samestate to lndist. These should be more clearly separated from G1 core tests in the spec search output, perhaps tagged with a different treatment label in the tree path.

4. **Heterogeneity classification**: The 10 heterogeneity specifications all report the main samestate coefficient (not the interaction term). While the reported coefficient is samestate, the model includes an interaction, so the coefficient is conditional on the interaction being zero. This should be noted in the spec search script to avoid conflating the conditional main effect with the unconditional baseline.

5. **Simulated data caveat**: All results are from simulated data calibrated to reported coefficients. The robustness of specifications to sample manipulations (winsorization, trimming, subsetting) may not reflect the true data properties. This is a fundamental limitation that should be disclosed prominently in any downstream analysis.
