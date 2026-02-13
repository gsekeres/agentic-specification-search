# Verification Report: 113517-V1

**Paper**: Moscarini & Postel-Vinay (2017), "The Relative Power of Employment-to-Employment Reallocation and Unemployment Exits in Predicting Wage Growth," *AER Papers & Proceedings*, 107(5), 203-07.

**Verified**: 2026-02-13

---

## 1. Baseline Groups Found

The paper has 4 baseline groups, each with 5 flow specifications from Table 1.

### G1: Nominal Earnings Growth (xdlogern_nom)

| Spec | spec_run_id | Coef (xee) | SE | p-value | N | Paper Value |
|------|-------------|------------|-----|---------|---|-------------|
| ee_only (Spec 1) | run0058 | 0.0450 | 0.0008 | <1e-10 | 6,158,358 | 0.045 |
| ee_ue (Spec 4) | run0059 | 0.0474 | 0.0008 | <1e-10 | 6,059,753 | -- |
| ee_ue_ur (Spec 5) | run0060 | 0.0414 | 0.0008 | <1e-10 | 6,059,753 | -- |
| all_flows (Spec 6) | run0061 | 0.0450 | 0.0008 | <1e-10 | 6,057,561 | 0.045 |
| grouped_flows (Spec 7) | run0062 | 0.0516 | 0.0008 | <1e-10 | 6,057,561 | -- |

Spec 1 and Spec 6 match Table 1 exactly.

### G2: Real Earnings Growth (xdlogern)

| Spec | spec_run_id | Coef (xee) | SE | p-value | N | Paper Value |
|------|-------------|------------|-----|---------|---|-------------|
| ee_only (Spec 1) | run0007 | 0.0373 | 0.0008 | <1e-10 | 6,158,358 | 0.037 |
| ee_ue (Spec 4) | run0008 | 0.0386 | 0.0008 | <1e-10 | 6,059,753 | -- |
| ee_ue_ur (Spec 5) | run0009 | 0.0354 | 0.0008 | <1e-10 | 6,059,753 | -- |
| all_flows (Spec 6) | run0010 | 0.0387 | 0.0008 | <1e-10 | 6,057,561 | 0.039 |
| grouped_flows (Spec 7) | run0011 | 0.0448 | 0.0008 | <1e-10 | 6,057,561 | -- |

Spec 1 matches exactly. Spec 6 matches at reported precision (0.039 rounded vs. 0.0387).

### G3: Nominal Hourly Wage Growth (xdloghwr_nom)

| Spec | spec_run_id | Coef (xee) | SE | p-value | N | Paper Value |
|------|-------------|------------|-----|---------|---|-------------|
| ee_only (Spec 1) | run0200 | 0.0096 | 0.0003 | <1e-10 | 3,176,722 | 0.0096 |
| ee_ue (Spec 4) | run0201 | 0.0040 | 0.0004 | <1e-10 | 3,148,524 | -- |
| ee_ue_ur (Spec 5) | run0202 | 0.0016 | 0.0004 | 1.3e-5 | 3,148,524 | -- |
| all_flows (Spec 6) | run0203 | 0.0006 | 0.0003 | 0.104 | 3,147,108 | 0.0006 (insig) |
| grouped_flows (Spec 7) | run0204 | 0.0018 | 0.0004 | 1.3e-5 | 3,147,108 | -- |

Spec 1 and Spec 6 match Table 1 exactly. The insignificance of Spec 6 (p=0.104) is correctly reproduced.

### G4: Real Hourly Wage Growth (xdloghwr)

| Spec | spec_run_id | Coef (xee) | SE | p-value | N | Paper Value |
|------|-------------|------------|-----|---------|---|-------------|
| ee_only (Spec 1) | run0149 | 0.0039 | 0.0004 | <1e-10 | 3,176,722 | 0.0039 |
| ee_ue (Spec 4) | run0150 | -0.0018 | 0.0004 | 2.2e-6 | 3,148,524 | -- |
| ee_ue_ur (Spec 5) | run0151 | -0.0024 | 0.0004 | <1e-10 | 3,148,524 | -- |
| all_flows (Spec 6) | run0152 | -0.0037 | 0.0004 | <1e-10 | 3,147,108 | -0.0037 |
| grouped_flows (Spec 7) | run0153 | -0.0026 | 0.0004 | <1e-10 | 3,147,108 | -- |

Spec 1 and Spec 6 match Table 1 exactly. The sign reversal in Spec 6 is correctly reproduced.

All 4 baseline groups from SPECIFICATION_SURFACE.json are present and verified. No spurious or missing groups.

---

## 2. Row Counts

| Metric | Count |
|--------|-------|
| **Total rows** | 284 |
| **Valid rows** | 284 |
| **Invalid rows** | 0 |
| **Core rows** | 284 |
| **Non-core rows** | 0 |
| **Unclear rows** | 0 |
| **Baseline rows** | 20 |
| **Succeeded** | 260 |
| **Failed** | 24 |

### By Baseline Group

| Group | Total | Succeeded | Failed | Baseline |
|-------|-------|-----------|--------|----------|
| G1 | 71 | 65 | 6 | 5 |
| G2 | 71 | 65 | 6 | 5 |
| G3 | 71 | 65 | 6 | 5 |
| G4 | 71 | 65 | 6 | 5 |

---

## 3. Category Counts

| Category | Count | Failed | Description |
|----------|-------|--------|-------------|
| core_method | 20 | 0 | Baseline flow specifications (5 per group) |
| core_controls | 68 | 24 | First-stage control variants (5 variants x 3 anchors + 2 cross-axis, per group) |
| core_sample | 72 | 0 | Sample restriction variants (6 variants x 3 anchors, per group) |
| core_fe | 24 | 0 | Fixed-effects/time-control variants (2 variants x 3 anchors, per group) |
| core_weights | 32 | 0 | Weighting variants (2 variants x 3 anchors + 2 cross-axis, per group) |
| core_data | 12 | 0 | Market definition variants (1 variant x 3 anchors, per group) |
| core_inference | 56 | 0 | Inference/SE variants (4 variants x 3 anchors + 2 cross-axis, per group) |

### Cross-axis specs (24 total, 6 per group)

The one-axis-at-a-time design uses 3 anchor flow specs (ee_only, ee_ue_ur, all_flows). Cross-axis specs extend selected RC axes to non-anchor flows (ee_ue and grouped_flows):

| Cross-axis combination | Count |
|------------------------|-------|
| no_lag_depvar x {ee_ue, grouped_flows} | 8 (2 per group) |
| cluster_market x {ee_ue, grouped_flows} | 8 (2 per group) |
| unweighted x {ee_ue, grouped_flows} | 8 (2 per group) |

The SPECIFICATION_SEARCH.md budget table listed cross-axis as 12 total, but execution produced 24 (6 per group rather than 3). This appears to reflect expanded cross-axis coverage during execution. The total of 284 specs matches the search report exactly.

---

## 4. Significance Summary

| Group | Specs | Same Sign as Spec 1 | Sig at 5% | Sig at 10% | Coef Range | Median |
|-------|-------|---------------------|-----------|------------|------------|--------|
| G1 (nom earn) | 65 | 65 (100%) positive | 57 (87.7%) | 61 (93.8%) | [0.009, 0.773] | 0.045 |
| G2 (real earn) | 65 | 65 (100%) positive | 54 (83.1%) | 57 (87.7%) | [0.008, 0.842] | 0.038 |
| G3 (nom hr wage) | 65 | 55 (84.6%) positive | 45 (69.2%) | 45 (69.2%) | [-0.100, 0.045] | 0.003 |
| G4 (real hr wage) | 65 | 26 (40.0%) positive | 50 (76.9%) | 50 (76.9%) | [-0.027, 0.039] | -0.002 |
| **All** | **260** | -- | **206 (79.2%)** | **213 (81.9%)** | -- | -- |

Note on G4 sign consistency: The "expected sign" for G4 is null_or_negative per the surface. Spec 1 is positive (0.0039) while Spec 6 is negative (-0.0037). The 40% positive rate reflects genuine sign instability across specifications, which is the paper's point: the EE-hourly wage relationship is not robust.

---

## 5. Top Issues

### 5.1 All 24 Failures from Two First-Stage Control Variants

All 24 failed specifications come from exactly two first-stage control variants:

| Variant | Failures | Cause |
|---------|----------|-------|
| e_controls_extended | 12 (3 anchors x 4 groups) | Applies full employed-side controls (laguni, lagsiz, lagocc, lagind, lagpub) to unemployed-eligible regressions (UE/NE/UR); these variables are mostly NaN for unemployed individuals, producing empty subsamples |
| u_controls_extended | 12 (3 anchors x 4 groups) | Adds C(laguni) to unemployed controls; insufficient variation in laguni within mkt_t cells for unemployed observations, causing zero-size arrays in singleton detection |

These failures are legitimate data constraints. The extended control variants push variables designed for one labor force status onto a different population where those variables are structurally missing or have no variation. The failures are evenly distributed across all 4 groups (6 per group = 3 anchors x 2 failing variants), confirming the data constraint is systematic, not group-specific.

### 5.2 Inference Sensitivity: Clustering Dramatically Inflates Standard Errors

The baseline uses HC1 SE at the individual level (Stata areg default), which produces very small SE (~0.0008 for earnings groups) because N > 6 million. Clustering at the market or time level dramatically inflates SE:

**G1 ee_only example**:
- HC1: SE = 0.0008, p < 1e-10
- Market cluster (100 clusters): SE = 0.0219, p = 0.043
- Time cluster (~176 clusters): SE = 0.0265, p = 0.092
- Two-way (market x time): SE = 0.0281, p = 0.113

This is a ~27-35x SE inflation. The EE-earnings effect survives market clustering (p < 0.05) but becomes borderline or insignificant with time or two-way clustering. Given the generated-regressor structure (market-level FEs are constant within mkt_t cells), clustering is arguably more appropriate than individual-level HC1.

**G4 all_flows example** (where baseline coefficient is negative):
- HC1: SE = 0.0004, p < 1e-10
- Market cluster: SE = 0.0065, p = 0.571
- The negative hourly wage effect becomes wholly insignificant with any clustering.

### 5.3 First-Stage Controls Matter for Coefficient Magnitude

The `e_controls_minimal` variant (dropping employed-side controls except lagstate) substantially increases EE coefficients. For G1, this pushes coefficients from ~0.045 to ~0.08-0.77 (the large values come from the coarser composition adjustment absorbing less variation). This suggests the composition adjustment is substantively important for the EE coefficient magnitude.

The `no_lag_depvar` variant (dropping lagged dependent variable from first-stage) modestly increases coefficients, consistent with the lag absorbing EE-correlated variation.

### 5.4 No Treatment or Outcome Drift

The treatment variable is consistently `xee` across all 284 rows. Each baseline group uses exactly one outcome variable with no exceptions:
- G1: xdlogern_nom (all 71 rows)
- G2: xdlogern (all 71 rows)
- G3: xdloghwr_nom (all 71 rows)
- G4: xdloghwr (all 71 rows)

### 5.5 G3/G4 Sign Instability Is Substantive, Not a Bug

For G3 and G4, sign reversals occur across specifications. In G3, 10 of 65 specs produce negative coefficients. In G4, 39 of 65 specs are negative (median is -0.002). This instability is the paper's key finding: the EE-wage relationship operates through earnings (hours and job characteristics), not through hourly wage rates. The specification search confirms this interpretation is robust.

---

## 6. Surface-to-Results Concordance

### Per-group breakdown (all 4 groups are symmetric)

| Component | Surface Plan | Actual | Match |
|-----------|-------------|--------|-------|
| Baseline flow specs | 5 | 5 | Exact |
| First-stage control variants (one-axis) | 15 | 15 | Exact |
| First-stage control variants (cross-axis) | -- | 2 | Cross-axis |
| Sample variants | 18 | 18 | Exact |
| FE variants | 6 | 6 | Exact |
| Weight variants (one-axis) | 6 | 6 | Exact |
| Weight variants (cross-axis) | -- | 2 | Cross-axis |
| Market definition variants | 3 | 3 | Exact |
| Inference variants (one-axis) | 12 | 12 | Exact |
| Inference variants (cross-axis) | -- | 2 | Cross-axis |
| **Total per group** | **65 + 6 cross** | **71** | **Match** |

### Total budget reconciliation

| Component | Surface Budget Table | Actual | Notes |
|-----------|---------------------|--------|-------|
| baseline | 20 | 20 | Match |
| rc/controls/first_stage | 72 | 68 | Surface included 4 per group from cross-axis; actual embeds 2 per group |
| rc/sample | 72 | 72 | Match |
| rc/fe | 24 | 24 | Match |
| rc/weights | 24 | 32 | Surface excluded cross-axis weights; actual includes 2 per group |
| rc/data/aggregation | 12 | 12 | Match |
| infer/se | 48 | 56 | Surface excluded cross-axis inference; actual includes 2 per group |
| cross-axis (listed separately) | 12 | 0 | Cross-axis specs absorbed into primary categories above |
| **Total** | **284** | **284** | **Match** |

The total of 284 matches exactly. The discrepancy is purely an accounting difference: the SPECIFICATION_SEARCH.md budget table lists 12 cross-axis specs separately, while the actual results embed the 24 cross-axis specs within their primary categories (controls, weights, inference). The reallocation of cross-axis counts across categories is: +8 to controls (net -4 from surface's 72), +8 to weights (net +8 from surface's 24), +8 to inference (net +8 from surface's 48).

---

## 7. Recommendations

1. **Flag clustering as the preferred inference method in robustness summaries**: The baseline HC1 SE produce artificially small p-values (p < 1e-10 for most specs) because N > 6 million but the effective degrees of freedom are ~100 markets x ~176 time periods. Market-clustered SE are more appropriate and produce materially different significance conclusions. Future analyses should consider clustering as the default rather than a robustness check.

2. **Document the generated-regressor problem more prominently**: The two-stage procedure uses predicted FE as generated regressors. Neither the baseline nor any specification variant implements Murphy-Topel SE corrections. While this is acknowledged in the implementation notes, the practical implication is that all reported SE (even clustered) likely understate uncertainty. This was flagged in the surface's diagnostics plan but not executed.

3. **Consider dropping the two systematically-failing first-stage variants from the spec count**: The `e_controls_extended` and `u_controls_extended` variants are infeasible due to structural data missingness (employment-side controls do not exist for unemployed individuals). These 24 rows inflate the failure count without providing information. They could be reclassified as "infeasible by design" rather than counted as failures.

4. **Reconcile cross-axis budget table**: The SPECIFICATION_SEARCH.md budget table lists 12 cross-axis specs separately, but 24 were actually executed (6 per group). The discrepancy is minor but could confuse future readers. The per-group budget allocation table correctly shows 71 specs including 6 cross-axis, which matches execution.

5. **Explore/diagnostic specs were not executed**: The surface defined 5 explore/ specs (UE-only, UR-only, EE interaction, pooled identification, finer markets) and 2 diagnostic specs (Murphy-Topel SE, Wooldridge serial correlation test). None were executed. This is appropriate for the core specification search but means the diagnostics plan remains open.

---

## 8. Verification Quality Checks

- Every `spec_run_id` in `specification_results.csv` appears exactly once in `verification_spec_map.csv`: **PASS** (284/284)
- Every `baseline_group_id` referenced in the CSV exists in `verification_baselines.json`: **PASS** (G1, G2, G3, G4)
- All classifications in the `why` column are concrete and anchored in observable row fields: **PASS**
- No non-core or invalid rows: All 284 rows are valid core specifications: **PASS**
- Treatment variable consistency: `xee` in all 284 rows: **PASS**
- Outcome variable consistency within groups: No drift detected: **PASS**
- Baseline coefficient verification against Table 1: All reported values match at published precision: **PASS**
- Failed specs are correctly documented with error messages in the `notes` field: **PASS** (24/24 failures have FAILED prefix in notes)
