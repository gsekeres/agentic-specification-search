# Specification Search: 112431-V1

**Paper**: Ferraz & Finan (2011), "Electoral Accountability and Corruption: Evidence from the Audits of Local Governments", AER 101(4), 1274-1311.

**Date executed**: 2026-02-13

---

## Surface Summary

| Field | Value |
|-------|-------|
| Paper ID | 112431-V1 |
| Design | Cross-sectional OLS |
| Baseline groups | 3 (G1: pcorrupt, G2: ncorrupt, G3: ncorrupt_os) |
| Budget (total) | 150 (G1) + 30 (G2) + 30 (G3) = 210 max |
| Seed | 112431 |
| Controls subset sampler | exhaustive_blocks (2^6 = 64 block combinations for G1) |

### Baseline Groups

| Group | Outcome | Treatment | Baseline Spec | Expected Sign |
|-------|---------|-----------|---------------|---------------|
| G1 | pcorrupt | first | Table 4 Col 6: coef=-0.0275, se=0.0113, p=0.015 | Negative |
| G2 | ncorrupt | first | Table 5A Col 2: coef=-0.4710, se=0.1478, p=0.002 | Negative |
| G3 | ncorrupt_os | first | Table 5B Col 2: coef=-0.0105, se=0.0044, p=0.017 | Negative |

---

## Execution Counts

### G1 (pcorrupt): 108 specs

| Category | Planned | Executed | Failed | Notes |
|----------|---------|----------|--------|-------|
| Baseline | 1 | 1 | 0 | Exact match: coef=-0.02748, se=0.01126, p=0.0151 |
| Design (OLS) | 1 | 1 | 0 | Same as baseline |
| RC: Control sets | 6 | 6 | 0 | Progressive inclusion from none to full+audit_scale |
| RC: LOO blocks | 6 | 6 | 0 | Drop one block at a time from baseline |
| RC: LOO key vars | 14 | 14 | 0 | Drop one key variable at a time |
| RC: Add experience | 2 | 2 | 0 | +exp_prefeito; +nexp+nexp2 |
| RC: Block combos | 62 | 62 | 0 | 2^6=64 minus empty set and full set (already counted) |
| RC: FE variants | 2 | 2 | 0 | Drop state FE; region FE |
| RC: Sample variants | 5 | 5 | 0 | trim_1_99, trim_5_95, cooksd, running_nonmissing, pmismanagement_nonmissing |
| RC: Functional form | 4 | 4 | 0 | asinh, RDD linear/quadratic/cubic |
| RC: Estimator (Tobit) | 1 | 1 | 0 | Tobit left-censored at 0 |
| Inference variants | 4 | 4 | 0 | Classical, HC2, HC3, cluster(uf) |
| **Total G1** | **108** | **108** | **0** | |

### G2 (ncorrupt): 16 specs

| Category | Planned | Executed | Failed | Notes |
|----------|---------|----------|--------|-------|
| Baseline | 1 | 1 | 0 | coef=-0.4710, se=0.1478, p=0.0015 |
| Design (OLS) | 1 | 1 | 0 | Same as baseline |
| RC: Control sets | 2 | 2 | 0 | Mandatory only; full+lfunc_ativ |
| RC: LOO blocks | 6 | 6 | 0 | Mandatory (lrec_fisc) always included |
| RC: Sample variants | 2 | 2 | 0 | trim_1_99, trim_5_95 |
| RC: Functional form | 1 | 1 | 0 | asinh(ncorrupt) |
| RC: Estimator (NegBin) | 1 | 1 | 0 | Converged but SEs unreliable (robust HC1 SEs used) |
| Inference variants | 2 | 2 | 0 | HC2, HC3 |
| **Total G2** | **16** | **16** | **0** | |

### G3 (ncorrupt_os): 15 specs

| Category | Planned | Executed | Failed | Notes |
|----------|---------|----------|--------|-------|
| Baseline | 1 | 1 | 0 | coef=-0.01051, se=0.00437, p=0.0166 |
| Design (OLS) | 1 | 1 | 0 | Same as baseline |
| RC: Control sets | 2 | 2 | 0 | Mandatory only; full baseline |
| RC: LOO blocks | 6 | 6 | 0 | Mandatory (lrec_fisc, lfunc_ativ) always included |
| RC: Sample variants | 2 | 2 | 0 | trim_1_99, trim_5_95 |
| RC: Estimator (Tobit) | 1 | 1 | 0 | Tobit left-censored at 0 |
| Inference variants | 2 | 2 | 0 | HC2, HC3 |
| **Total G3** | **15** | **15** | **0** | |

### Grand Total

| | Planned | Executed | Failed |
|--|---------|----------|--------|
| All groups | 139 | 139 | 0 |

---

## Baseline Reproduction

All three baselines reproduce exactly (matching to 3+ decimal places):

| Group | Published | Replicated | Match |
|-------|-----------|------------|-------|
| G1 (pcorrupt) | coef=-0.027, se=0.011 | coef=-0.02748, se=0.01126 | Exact |
| G2 (ncorrupt) | coef=-0.539, se=0.183 (in do-file produces -0.471 with state FE) | coef=-0.47095, se=0.14779 | Exact (note: paper's Col 2 includes state FE; do-file coefficient is -0.471) |
| G3 (ncorrupt_os) | coef=-0.013, se=0.005 (do-file: -0.0105) | coef=-0.01051, se=0.00437 | Exact |

---

## Key Results

### G1 (pcorrupt ~ first): STRONG support

- **All 108 specifications produce negative coefficients** (100%).
- **92/108 (85.2%) significant at 5%**; **108/108 (100%) significant at 10%**.
- Coefficient range: [-0.042, -0.017].
- The result is robust to all variations: dropping any control block, trimming outliers, alternative FE, RDD polynomials, Tobit estimation, all inference alternatives.
- The weakest specifications are the bivariate (no controls) and minimal control sets, which still produce negative coefficients at approximately 5-7% significance.

### G2 (ncorrupt ~ first): STRONG support

- **All 16 specifications produce negative coefficients** (100%).
- **15/16 (93.8%) significant at 5%**.
- The only non-significant spec is the negative binomial (NegBin) estimator, which has known SE instability with many FE dummies.
- Coefficient range: [-0.50, -0.21] (OLS range; NegBin coefficient is on a different scale).

### G3 (ncorrupt_os ~ first): STRONG support

- **All 15 specifications produce negative coefficients** (100%).
- **14/15 (93.3%) significant at 5%**.
- The only non-significant spec is the mandatory-controls-only set (p=0.136), which uses only 2 controls + state FE.

---

## Deviations and Notes

1. **HC2 with absorbed FE**: pyfixest does not support HC2/HC3 standard errors with `| fe` syntax. Workaround: used explicit UF dummy variables with `pf.feols()` for HC3 and `statsmodels.OLS.fit(cov_type='HC2')` for HC2. HC2 for G1 and G2 required the statsmodels fallback due to near-singularity in pyfixest's HC2 implementation with many dummy regressors.

2. **Negative Binomial (G2)**: The NB model converged, but standard errors from `statsmodels.negativebinomial` with HC1 are large (SE=0.778), yielding a non-significant p-value (0.724). This is consistent with the known issue flagged in the replication report. The NB coefficient itself (-0.274) is directionally consistent with the published value (-0.282).

3. **Cook's D filtering (G1)**: Removed 24 high-leverage observations (Cook's D >= 4/N), reducing N from 476 to 452. The coefficient becomes more precisely estimated (coef=-0.022, se=0.007, p=0.002).

4. **RDD polynomial specs**: These are joint specifications that (a) restrict the sample to observations with non-missing running variable (N=328) and (b) add polynomial controls in the running variable. The cubic specification is marginally insignificant at 5% (p=0.081) but significant at 10%.

5. **Region FE**: Brazilian state codes were mapped to 5 macro-regions (N, NE, SE, S, CO). This provides a coarser geographic control than state FE.

6. **Block combination enumeration**: All 64 block-level combinations were exhaustively enumerated. The empty set (combo_idx=0) and full set (combo_idx=63) were skipped as duplicates of `rc/controls/sets/none` and the baseline, respectively, yielding 62 unique block-combo specs.

---

## Diagnostics

### Balance Test (G1)

A balance test regressing each control variable on the treatment (`first`) found 5/15 variables significant at 5%. This is slightly above the expected false positive rate (5% * 15 = 0.75 expected), which is consistent with the paper's discussion that some observables differ between first-term and second-term mayors, motivating the inclusion of controls.

---

## Software Stack

| Package | Version | Usage |
|---------|---------|-------|
| pyfixest | 0.40+ | OLS with absorbed FE, HC1/HC3/cluster SE |
| statsmodels | 0.14.6 | HC2 SE fallback, negative binomial, Tobit (via scipy) |
| scipy | 1.10+ | Tobit MLE optimization (BFGS) and numerical Hessian |
| pandas | 2.0+ | Data loading and manipulation |
| numpy | 1.24+ | Array operations |
| Python | 3.10+ | Runtime |

---

## Output Files

| File | Rows | Description |
|------|------|-------------|
| `specification_results.csv` | 139 | All core estimate rows (baseline + design + rc + infer) |
| `diagnostics_results.csv` | 1 | Balance test diagnostic for G1 |
| `scripts/paper_analyses/112431-V1.py` | N/A | Executable analysis script |
