# Verification Report: 128521-V1

**Paper:** Lancashire Cotton Famine and mortality (historical DiD)
**Verification date:** 2026-02-25
**Verifier:** Post-run audit (prompt 06)

---

## Baseline Groups

### G1: Total mortality rate (DiD)
- **Claim:** ATT of Cotton Famine exposure on total mortality rate in cotton-dependent districts vs non-cotton districts. TWFE DiD with district and period FE, population weights, clustered at district level.
- **Baseline specs:**
  - `128521-V1_run_001` (`baseline`): Table 2 Col 1, no controls. coef=2.194, p<0.001, N=1076.
  - `128521-V1_run_002` (`baseline__table2_col2`): Col 2, adds density/pop shares/region x period. coef=2.024, p<0.001, N=1076.
  - `128521-V1_run_003` (`baseline__table2_col3`): Col 3, adds nearby district spillover rings. coef=2.534, p<0.001, N=1076.
- **Expected sign:** Positive (famine increases mortality).

### G2: Age-specific mortality rates (DiD)
- **Claim:** Age-decomposed ATT of Cotton Famine on mortality by 7 age groups. Same TWFE DiD design with age-group-specific controls and weights.
- **Baseline specs:** 7 baselines (one per age group):
  - `run_035` (under15): coef=0.224, p=0.836 (NS)
  - `run_038` (15-24): coef=0.171, p=0.757 (NS)
  - `run_041` (25-34): coef=0.894, p=0.188 (NS)
  - `run_044` (35-44): coef=1.512, p=0.108 (NS)
  - `run_047` (45-54): coef=3.066, p=0.005 (sig)
  - `run_050` (55-64): coef=6.740, p<0.001 (sig)
  - `run_053` (over64): coef=13.477, p<0.001 (sig)
- **Expected sign:** Positive. Effects are monotonically increasing with age.

---

## Counts

| Metric | Count |
|--------|-------|
| Total rows | 51 |
| Valid | 51 |
| Invalid | 0 |
| Baseline | 10 |
| Core | 51 |
| Non-core | 0 |

### Category Breakdown

| Category | Count |
|----------|-------|
| core_sample | 14 |
| core_weights | 11 |
| core_method | 10 |
| core_controls | 9 |
| core_data | 7 |

### By Baseline Group

| Group | Total | Core | Baselines |
|-------|-------|------|-----------|
| G1 | 30 | 30 | 3 |
| G2 | 21 | 21 | 7 |

---

## Sanity Checks

- **spec_run_id uniqueness:** PASS. All 51 spec_run_ids are unique.
- **baseline_group_id present:** PASS. All rows have G1 or G2.
- **run_success:** PASS. All 51 rows have run_success=1.
- **Numeric fields finite:** PASS. All numeric fields are finite.
- **No infer/* rows in specification_results.csv:** PASS.
- **coefficient_vector_json structure:** PASS. All rows have `coefficients`, `inference`, `software`, `surface_hash`. rc/controls rows have `controls`. rc/data rows have `data_construction`. rc/sample rows have `sample`.
- **Inference canonical match:** PASS. All spec_results rows use `infer/se/cluster/district` matching surface canonical.
- **Inference variants in inference_results.csv:** PASS. 4 inference variant rows (county cluster, robust HC) appear only in inference_results.csv.

---

## Issues

### Minor Issues

1. **Near-duplicate rows (run_026 and run_030):** Both have spec_id `rc/sample/exclude_manchester_liverpool_leeds` and produce identical coefficients (1.884, p=0.001, N=1070). The only difference is the `base_col` label in coefficient_vector_json (`col2` vs `col2_extra`) and sample_desc text. Flagged with lower confidence (0.75) for run_030.

2. **G2 has no LOO control variants:** G2 specifications only include sample exclusion (exclude_manchester) and weight variants (unweighted) for each age group, with no leave-one-out control variations. The surface budgeted 0 for controls_subset, which is consistent.

3. **Some G1 LOO specs appear twice (with and without nearby rings):** For example, rc/controls/loo/ln_popdensity appears as both run_004 (with nearby rings) and run_019 (without). This is valid since the base model differs (Col 3 vs Col 2 controls), providing additional robustness.

4. **Gaps in run numbering:** Run IDs skip 031-034, corresponding to inference variants in inference_results.csv.

---

## Recommendations

1. **Add LOO controls for G2:** Consider adding LOO specs for at least the older age groups (45-54, 55-64, over64) where effects are significant.

2. **Consider functional form variants:** No rc/form/* specs appear. Log mortality or Poisson count models could test sensitivity to functional form.

3. **Address the near-duplicate:** run_030 appears to be a redundant execution of the same specification as run_026. Consider removing or documenting the distinction.
