# Verification Report: 112431-V1

## Paper
Ferraz & Finan (2011), "Electoral Accountability and Corruption: Evidence from the Audits of Local Governments," AER 101(4).

## Baseline Groups

### G1: pcorrupt ~ first
- **Claim**: First-term mayors (reelection-eligible) have less corruption than term-limited second-term mayors
- **Expected sign**: Negative
- **Baseline spec**: 112431-V1_run0001 (Table 4 Col 6)
  - Coefficient: -0.0275
  - SE: 0.0113 (HC1 robust)
  - p-value: 0.015
  - N: 476

## Row Counts

| Metric | Count |
|--------|-------|
| Total rows | 53 |
| Core (is_core_test=1) | 53 |
| Non-core | 0 |
| Invalid (is_valid=0) | 0 |
| Baseline | 1 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_method | 1 |
| core_controls | 46 |
| core_fe | 2 |
| core_sample | 2 |
| core_funcform | 2 |

## Verification Checks

### 1. spec_run_id Uniqueness: PASS
All 53 spec_run_ids are unique.

### 2. baseline_group_id Consistency: PASS
All rows belong to G1, matching the pre-run surface.

### 3. spec_id Typing: PASS
- 1 baseline row
- 52 rc/* rows (controls: 46, fe: 2, sample: 2, form: 2)
- 0 design/* rows (baseline IS the OLS design, so no separate design variant needed)
- 0 infer/* rows in specification_results.csv (correct -- inference variants in separate file)

### 4. Run Success: PASS
All 53 rows have run_success=1. No failures.

### 5. Numeric Fields: PASS
All coefficient, std_error, p_value, n_obs fields are finite for all rows.

### 6. JSON Validity: PASS
All coefficient_vector_json fields parse as valid JSON.

### 7. Outcome/Treatment Consistency: PASS
- Treatment: `first` throughout (no drift)
- Outcome: `pcorrupt` for 51 specs, `log1p_pcorrupt` for 1, `asinh_pcorrupt` for 1
- The functional form variants change the outcome transform but preserve the underlying concept (corruption share)

### 8. No Infer/* in Spec Results: PASS
Inference variants are correctly segregated to inference_results.csv (2 rows).

## Inference Results Audit

| Variant | Run Success | Coefficient | SE | p-value |
|---------|------------|------------|------|---------|
| infer/se/cluster/uf | 1 | -0.0275 | 0.0101 | 0.012 |
| infer/se/hc/hc3 | 0 | NaN | NaN | NaN |

The HC3 variant failed because pyfixest does not support HC3 as a vcov dict key with absorbed FE. The cluster-SE variant succeeded and produces even tighter inference (SE=0.0101 vs 0.0113, p=0.012 vs 0.015), strengthening the result.

## Robustness Summary

**All 53 specifications yield negative coefficients**, ranging from -0.0312 to -0.0171.

- **44 of 53 (83%)** are significant at p < 0.05 with HC1 SE
- **53 of 53 (100%)** are significant at p < 0.10
- **3 of 53 (6%)** are significant at p < 0.01

The specifications that lose significance at the 5% level are all minimal-control variants (bivariate + state FE, or with only mayor demographics). As controls are added, the effect strengthens and becomes more precisely estimated.

**Assessment: STRONG robustness.** The paper's main finding is extremely well-supported across all RC dimensions examined.

## Recommendations

1. The HC3 inference variant should be re-attempted using explicit state dummies instead of absorbed FE, so pyfixest's HC3 constraint is bypassed. This is a minor implementation detail.
2. Future work could add the Table 5 outcomes (ncorrupt, ncorrupt_os) as additional baseline groups to test robustness of alternative corruption measures.
3. Table 6 RDD-polynomial specifications could be added as joint RC specs (sample restriction + polynomial controls), but these are secondary to the main claim.
