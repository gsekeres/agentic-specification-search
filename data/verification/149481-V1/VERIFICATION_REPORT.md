# Verification Report: 149481-V1

**Paper**: "Do Thank-You Calls Increase Charitable Giving? Expert Forecasts and Field Experimental Evidence"
**Journal**: AER
**Verified**: 2026-02-03
**Verifier**: verification_agent

---

## Paper Summary

This paper tests whether thank-you phone calls to existing charitable donors increase subsequent giving. It reports results from three randomized controlled trials:

- **Experiment 1**: Public television stations (N~494K), with station x execution-date fixed effects
- **Experiment 2**: National non-profit (N~58K), no fixed effects (no station grouping)
- **Experiment 3**: Public television with a new call script (N~24K), with station x execution-date fixed effects

Two outcomes are measured per experiment: (1) whether the donor gave again (binary: donated), and (2) the conditional gift amount (gift_cond). The paper main finding is a **null effect** across all experiments and outcomes.

---

## Baseline Groups

| Group | Experiment | Outcome | Baseline spec_id | Coef | p-value |
|-------|-----------|---------|-------------------|------|---------|
| G1 | Exp 1 (Public TV) | donated (binary) | exp1/donated/baseline | 0.000224 | 0.908 |
| G2 | Exp 1 (Public TV) | gift_cond (conditional dollar) | exp1/gift_cond/baseline | 0.192 | 0.858 |
| G3 | Exp 2 (National NP) | donated (binary) | exp2/donated/baseline | -0.000150 | 0.968 |
| G4 | Exp 2 (National NP) | gift_cond (conditional dollar) | exp2/gift_cond/baseline | -0.261 | 0.888 |
| G5 | Exp 3 (New Script) | donated (binary) | exp3/donated/baseline | 0.013 | 0.134 |
| G6 | Exp 3 (New Script) | gift_cond (conditional dollar) | exp3/gift_cond/baseline | 2.760 | 0.466 |

All six baselines are well-identified. Each corresponds to a distinct experiment-outcome combination in the paper main tables. The treatment variable is treat (thank-you call assignment) in all cases.

---

## Classification Counts

| Metric | Count |
|--------|-------|
| **Total specifications** | 147 |
| **Baselines** | 6 |
| **Core tests (non-baseline)** | 105 |
| **Non-core tests** | 36 |
| **Invalid** | 0 |
| **Unclear** | 0 |

### Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_controls | 62 | Baselines (6), control build-up (8), leave-one-out (48) |
| core_fe | 4 | No-FE variants (Exp1 x 2 outcomes, Exp3 x 2 outcomes) |
| core_funcform | 9 | IHS and log transformations of outcomes |
| core_inference | 18 | Classical, HC2, HC3 SE variations (3 per experiment-outcome) |
| core_sample | 18 | High/low baseline donor splits, winsorization, trimming |
| noncore_heterogeneity | 34 | Subgroup analyses and interaction terms |
| noncore_placebo | 2 | Future-year outcome placebo tests |

---

## Top 5 Most Suspicious Rows

1. **exp1/donated/robust/estimation/no_fe**: The coefficient and SE are identical to the baseline exp1/donated/baseline (coef=0.000224, SE=0.001937). Removing FE should change at least the SE. Possible explanation: the FE absorption may not have changed the treatment coefficient in this large RCT, or the no_fe spec may have inadvertently still included FE. The fixed_effects field shows None, but further investigation is warranted. Suspicion level: moderate.

2. **exp1/gift_cond/robust/estimation/no_fe**: Same issue -- the coefficient (0.19189) and SE (1.0693) are identical to the baseline. This reinforces the concern that the no_fe specification may not have actually removed FE, or that the FE were collinear with no effect in this design. Suspicion level: moderate.

3. **exp2/donated/robust/build/baseline_vars**: The coefficient (-0.000150) and p-value (0.9677) are identical to the baseline exp2/donated/baseline. This is expected since Experiment 2 baseline already uses only payment_amount2, var12 as controls (no demographics), so the baseline_vars build-up step is identical to the full baseline. Not an error, but redundant. Suspicion level: low.

4. **exp2/gift_cond/robust/build/baseline_vars**: Same redundancy as above -- identical to exp2/gift_cond/baseline. Suspicion level: low.

5. **exp3/donated/robust/het/income_high**: This subgroup (income 175k+, N=2,127) is the only specification across all 147 that achieves significance at 1% (p=0.009, coef=0.071). This is likely a Type I error given the large number of subgroup tests. Correctly classified as non-core heterogeneity. Suspicion level: low (correctly classified).

---

## Notes on the no_fe Issue

For Experiments 1 and 3, the baseline includes station x execution-date FE (ii). The no_fe specs claim fixed_effects=None but produce identical coefficients and SEs to the baseline. Two possible explanations:

1. The script may have failed to actually remove FE, resulting in a duplicate of the baseline. This would mean 4 specs are effectively duplicates.
2. In this RCT design, the FE may be exactly absorbed by the controls already in the model, making the with-FE and without-FE estimates identical. This is unlikely given the large number of FE groups.

Recommendation: The estimation script should be audited to confirm that the no_fe specification genuinely removes fixed effects.

---

## Recommendations for Spec-Search Script

1. **Fix no_fe duplication**: Verify that exp1/*/robust/estimation/no_fe and exp3/*/robust/estimation/no_fe specifications actually remove the FE. If the coefficient is identical, it suggests a code bug.

2. **Remove redundant Exp2 build specs**: exp2/donated/robust/build/baseline_vars and exp2/gift_cond/robust/build/baseline_vars are identical to the Exp2 baselines because Exp2 has no demographic controls.

3. **Consider extending placebo tests**: Only Experiment 1 has placebo (future-year outcome) tests. If data is available, these could be extended to Experiments 2 and 3.

4. **Experiment 2 structural differences**: Experiment 2 is structurally different (no FE, fewer controls). The specification search correctly adapts, but the more limited robustness suite for Exp2 should be noted.

---

## Quality Checks

- Every spec_id in specification_results.csv appears exactly once in verification_spec_map.csv: PASS
- Every baseline_group_id referenced in the CSV exists in verification_baselines.json: PASS
- No invalid (non-finite) coefficients or standard errors: PASS
- No specs classified as unclear or invalid: PASS
- Conservative classification applied: all heterogeneity specs (subgroup + interaction) marked non-core: PASS
