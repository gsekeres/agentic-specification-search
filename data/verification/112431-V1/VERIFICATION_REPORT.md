# Verification Report: 112431-V1

**Paper**: Ferraz & Finan (2011), "Electoral Accountability and Corruption: Evidence from the Audits of Local Governments", AER 101(4), 1274-1311.

**Verified**: 2026-02-13

---

## 1. Baseline Groups Found

| Group | Baseline spec_run_id | spec_id | Outcome | Treatment | Coef | SE | p-value | N | Expected Sign |
|-------|---------------------|---------|---------|-----------|------|-----|---------|---|---------------|
| G1 | 112431-V1_run0001 | baseline | pcorrupt | first | -0.02748 | 0.01126 | 0.0151 | 476 | Negative |
| G2 | 112431-V1_run0109 | baseline | ncorrupt | first | -0.47095 | 0.14779 | 0.0016 | 476 | Negative |
| G3 | 112431-V1_run0125 | baseline | ncorrupt_os | first | -0.01051 | 0.00437 | 0.0166 | 476 | Negative |

All three baseline groups match the SPECIFICATION_SURFACE.json definition exactly. No spurious or missing baseline groups.

---

## 2. Row Counts

| Metric | Count |
|--------|-------|
| **Total rows** | 139 |
| **Valid rows** | 139 |
| **Invalid rows** | 0 |
| **Core rows** | 139 |
| **Non-core rows** | 0 |
| **Unclear rows** | 0 |
| **Baseline rows** | 3 |

### By Baseline Group

| Group | Total | Core | Baseline |
|-------|-------|------|----------|
| G1 | 108 | 108 | 1 |
| G2 | 16 | 16 | 1 |
| G3 | 15 | 15 | 1 |

---

## 3. Category Counts

| Category | Count | Description |
|----------|-------|-------------|
| core_controls | 106 | Control-set variants (progressive, LOO block, LOO var, add, block-combo) |
| core_method | 9 | Baselines (3) + design specs (3) + estimator variants (3: Tobit x2, NegBin x1) |
| core_sample | 9 | Trimming, Cook's D, sample restrictions |
| core_inference | 8 | Classical, HC2, HC3, cluster SE variants |
| core_funcform | 5 | asinh outcome transforms (2) + RDD polynomial specs (3) |
| core_fe | 2 | Drop state FE, region FE |

---

## 4. Significance Summary

| Group | Specs | All Negative | Sig at 5% | Sig at 10% |
|-------|-------|-------------|-----------|------------|
| G1 (pcorrupt) | 108 | 108 (100%) | 92 (85.2%) | 108 (100%) |
| G2 (ncorrupt) | 16 | 16 (100%) | 15 (93.8%) | 15 (93.8%) |
| G3 (ncorrupt_os) | 15 | 15 (100%) | 14 (93.3%) | 14 (93.3%) |
| **All** | **139** | **139 (100%)** | **121 (87.1%)** | **137 (98.6%)** |

All 139 specifications produce negative coefficients, consistent with the expected sign across all three baseline groups.

---

## 5. Top Issues

### 5.1 Exact Numerical Duplicates (11 clusters, 26 rows total)

Multiple rows produce identical numerical results because they represent the same underlying regression reached via different spec-tree paths. This is expected behavior from exhaustive block enumeration overlapping with named spec_ids. Key examples:

- **Baseline = Design = Full block combo**: run0001, run0002, run0007 (G1) are the same regression. Similarly for G2 (run0109, run0110) and G3 (run0125, run0126, run0128).
- **LOO block = block combo complement**: e.g., run0009 (drop_prefchar2_continuous) = run0092 (block combo without prefchar2_continuous). 6 such pairs in G1.
- **LOO block = LOO var for single-var blocks**: run0012 (drop_fiscal block) = run0024 (drop_lrec_trans), because the "fiscal" block contains only lrec_trans.
- **Progressive control sets = block combos**: e.g., run0004 (prefchar2 set) = run0033 (block combo of prefchar2_continuous + prefchar2_party). 3 such pairs.

**Impact**: These duplicates are valid spec-tree nodes but do not add independent information. The SPECIFICATION_SEARCH.md notes that the empty set and full set of block combos were already excluded to avoid counting the bivariate and baseline twice. However, intermediate overlaps remain. For robustness counting purposes, deduplication would reduce G1's effective unique specifications from 108 to approximately 95.

### 5.2 R-squared NaN for MLE-based Models (2 rows)

- run0104 (Tobit, G1): R-squared is NaN. Expected for MLE-based Tobit.
- run0137 (Tobit, G3): R-squared is NaN. Same reason.
- run0122 (NegBin, G2): R-squared reported as 0.150, which may be a pseudo-R-squared. The SPECIFICATION_SEARCH.md notes known SE instability for the NegBin model with many FE dummies (p=0.724, the only non-significant-at-10% spec for G2).

### 5.3 Outcome Variable Naming for asinh Transforms (2 rows)

- run0100 (G1): outcome_var = "asinh_pcorrupt" (baseline is "pcorrupt")
- run0121 (G2): outcome_var = "asinh_ncorrupt" (baseline is "ncorrupt")

These are classified as **core_funcform** because they are monotonic transforms of the baseline outcome, preserving the same conceptual claim. The sign expectation is unchanged. This is not outcome drift.

### 5.4 Sample Size Variation in RDD and Sample-Restriction Specs

Several G1 specs operate on restricted samples:
- RDD polynomial specs (run0098, run0101-run0103): N=328 (running variable nonmissing)
- Cook's D trimming (run0097): N=452
- Percentile trimming: N=452-471
- pmismanagement nonmissing (run0099): N=366

These are all documented sample restrictions, not data errors. They do not change the conceptual claim (still pcorrupt ~ first for audited municipalities), so they remain core.

### 5.5 No Cluster SE for G2 or G3

G1 includes a state-level cluster SE variant (run0108), but G2 and G3 do not. The surface did not plan cluster SE for G2/G3 (only HC2 and HC3 were planned). This is a minor coverage gap but not an error.

### 5.6 No Issues with Treatment Variable

The treatment variable "first" is consistent across all 139 rows with no drift.

---

## 6. Surface-to-Results Concordance

### G1 (pcorrupt): Surface planned ~109, executed 108

The surface estimated 109 specs. The executed count of 108 is 1 fewer, likely because the surface's "1 baseline + 1 design + 6 control sets + 6 LOO blocks + 14 LOO vars + 2 add + 62 block combos + 2 FE + 5 sample + 4 form + 1 estimator + 4 infer = 108" already accounts for deduplication between baseline and design (they are listed separately but run as separate rows). The count matches cleanly.

### G2 (ncorrupt): Surface planned 15, executed 16

The surface estimated 15 total: "1 baseline + 2 control sets + 6 LOO blocks + 2 sample + 1 form + 1 estimator + 2 infer = 15". The actual count is 16, which includes 1 baseline + 1 design + 2 control sets + 6 LOO blocks + 2 sample + 1 form + 1 estimator + 2 infer = 16. The extra row is the design spec (run0110), which was not separately counted in the surface budget estimate but was executed.

### G3 (ncorrupt_os): Surface planned 14, executed 15

The surface estimated 14 total: "1 baseline + 2 control sets + 6 LOO blocks + 2 sample + 1 estimator + 2 infer = 14". The actual count is 15, including 1 design spec (run0126) not separately counted in the surface budget.

---

## 7. Recommendations

1. **Deduplicate or tag overlapping block-combo specs**: The exhaustive block enumeration creates known overlaps with named control-set specs and LOO-block specs. Consider adding a `is_duplicate_of` column in future runs to facilitate deduplication when computing summary statistics. The SPECIFICATION_SEARCH.md partially addresses this (by excluding the empty-set and full-set combos) but intermediate overlaps remain.

2. **Include design specs in surface budget counts**: The surface budget estimates for G2 and G3 did not count the design row, causing a minor 1-row discrepancy. Future surfaces should explicitly count design rows.

3. **Add cluster SE for G2/G3**: For completeness, state-level cluster SE variants could be added to G2 and G3 (as was done for G1), though the surface explicitly chose not to include them.

4. **Document pseudo-R-squared for MLE models**: The NegBin spec (run0122) reports an R-squared of 0.150 without clarification that this is a pseudo-R-squared. Consider using a separate column or explicit labeling.

5. **Consider wild cluster bootstrap**: The SPECIFICATION_SURFACE.json notes that state-level clustering has only 26 clusters and that wild cluster bootstrap would be more reliable. If the environment supports it in future runs, this would strengthen inference for the cluster SE specs.

---

## 8. Verification Quality Checks

- Every `spec_run_id` in `specification_results.csv` appears exactly once in `verification_spec_map.csv`: **PASS** (139/139)
- Every `baseline_group_id` referenced in the CSV exists in `verification_baselines.json`: **PASS** (G1, G2, G3)
- All explanations in the `why` column are concrete and anchored in observable row fields: **PASS**
- No non-core or invalid rows: All 139 rows are valid core specifications.
