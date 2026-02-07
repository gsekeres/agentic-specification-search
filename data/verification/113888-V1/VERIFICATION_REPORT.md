# Verification Report: 113888-V1

**Paper**: Keep it Simple: A Field Experiment on Information and Credit Application Behavior
**Journal**: AER
**Verified**: 2026-02-03
**Verifier**: verification_agent

---

## Baseline Groups

### G1: Business Practices (Primary)
- **Claim**: ROT financial training improves standardized business practices (e_zBusPrac)
- **Baseline spec_id**: baseline (outcome=e_zBusPrac, treatment=treat_rot)
- **Expected sign**: +
- **Baseline coefficient**: 0.139 (p < 0.001)

### G2: Reporting Mistakes (Secondary)
- **Claim**: ROT training reduces reporting mistakes (e_repAnyMistake)
- **Baseline spec_id**: baseline (outcome=e_repAnyMistake, treatment=treat_rot)
- **Expected sign**: -
- **Baseline coefficient**: -0.081 (p = 0.015)

### G3: Sales (Secondary)
- **Claim**: ROT training increases sales (e_zSales, e_salesWkBad_w01)
- **Baseline spec_id**: baseline (outcome=e_zSales or e_salesWkBad_w01, treatment=treat_rot)
- **Expected sign**: +
- **Baseline coefficient**: 0.093 / 974.7 (p = 0.051 / 0.062)

---
## Summary Counts

| Metric | Count |
|--------|-------|
| Total specifications | 75 |
| Baseline specs | 4 |
| Core test specs (non-baseline) | 36 |
| Non-core specs | 35 |
| Invalid | 0 |
| Unclear | 0 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 17 |
| core_funcform | 4 |
| core_inference | 4 |
| core_method | 7 |
| core_sample | 8 |
| noncore_alt_outcome | 5 |
| noncore_alt_treatment | 7 |
| noncore_heterogeneity | 20 |
| noncore_placebo | 3 |

---
## Top 5 Most Suspicious Rows

1. baseline_pooled rows (4 rows): Use pooled treat instead of treat_rot. Changes causal object. Classified noncore_alt_treatment.

2. robust/treatment/both_arms_acc (treat_acc, coef=0.076, p=0.167): Extracts accounting arm from two-arm model. Unusual extraction.

3. robust/treatment/accounting_only (treat_acc, coef=0.006, p=0.914): Accounting-only model. Near-zero effect. Correctly noncore_alt_treatment.

4. robust/sample/full_sample (N=797): Says full sample but N matches baseline. Coefficient differs (0.101 vs 0.139). Needs clarification.

5. outcome/e_zSales vs baseline e_zSales: Coefficients differ (0.100 vs 0.093, p=0.020 vs 0.051). Different samples or controls.

---

## Recommendations

1. Disambiguate baseline spec_ids: Use unique IDs per outcome.
2. Separate baseline_pooled: Mark with distinct tree path.
3. Move demographic subgroups from robust/sample/ to robust/het/.
4. Clarify full_sample spec: Note if identical to baseline.
5. Resolve outcome/* vs baseline overlap: Clarify differences or remove duplicates.
