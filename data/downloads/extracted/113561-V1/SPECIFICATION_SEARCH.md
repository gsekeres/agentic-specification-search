# Specification Search: 113561-V1

**Paper**: Fong & Luttmer (2009), "What Determines Giving to Hurricane Katrina Victims? Experimental Evidence on Racial Group Loyalty," *American Economic Journal: Applied Economics*, 1(2), 64-87.

**Design**: Randomized experiment (online survey). Respondents randomly shown pictures of Hurricane Katrina victims varying by race. Focal treatment: `picshowblack` (shown black victims vs white/race-obscured).

**Date executed**: 2026-02-13

---

## Surface Summary

| Field | Value |
|-------|-------|
| Baseline groups | 4 (G1: giving, G2: hypgiv_tc500, G3: subjsupchar, G4: subjsupgov) |
| Design classification | randomized_experiment |
| Baseline estimator | WLS with HC1 robust SE, survey weights (tweight) |
| Baseline sample | White respondents (N ~ 907-915) |
| Focal treatment | picshowblack (with picraceb, picobscur as other treatment arms) |
| Seed | 113561 |
| Total planned specs | 135 (37 + 34 + 32 + 32) |
| Sampling | Full enumeration (no random sampling needed) |

### Budget allocation per group

| Spec type | G1 | G2 | G3 | G4 |
|-----------|----|----|----|----|
| baseline | 1 | 1 | 1 | 1 |
| design/diff_in_means | 1 | 1 | 1 | 1 |
| rc/controls/sets | 2 | 2 | 2 | 2 |
| rc/controls/progression | 4 | 2 | 2 | 2 |
| rc/controls/manipulation_coding | 1 | 1 | 1 | 1 |
| rc/controls/loo | 17 | 17 | 17 | 17 |
| rc/sample/subpopulation | 5 | 5 | 5 | 5 |
| rc/weights/unweighted | 1 | 1 | 1 | 1 |
| rc/preprocess/outcome | 2 | 2 | 0 | 0 |
| infer/se/hc | 3 | 2 | 2 | 2 |
| **Total** | **37** | **34** | **32** | **32** |

---

## Execution Results

### Counts

| Category | Planned | Executed | Failed | Skipped |
|----------|---------|----------|--------|---------|
| baseline | 4 | 4 | 0 | 0 |
| design/* | 4 | 4 | 0 | 0 |
| rc/controls/* | 90 | 90 | 0 | 0 |
| rc/sample/* | 20 | 20 | 0 | 0 |
| rc/weights/* | 4 | 4 | 0 | 0 |
| rc/preprocess/* | 4 | 4 | 0 | 0 |
| infer/* | 9 | 9 | 0 | 0 |
| **Total** | **135** | **135** | **0** | **0** |

### Baseline verification

| Group | Outcome | Replicated coef | Executed coef | Diff | Replicated N | Executed N |
|-------|---------|-----------------|---------------|------|--------------|------------|
| G1 | giving | -4.198 | -4.1983 | 0.0003 | 915 | 915 |
| G2 | hypgiv_tc500 | -2.181 | -2.1813 | 0.0003 | 913 | 913 |
| G3 | subjsupchar | -0.221 | -0.2205 | 0.0005 | 907 | 907 |
| G4 | subjsupgov | -0.435 | -0.4351 | 0.0001 | 913 | 913 |

All baselines match the replication script to within rounding tolerance (< 0.001).

### Key findings across specification surface

**G1 (giving, dictator game)**:
- Baseline: coef = -4.198, p = 0.370 (not significant)
- Coefficient range: [-8.868, 1.440]
- P-value range: [0.185, 0.828]
- Significant at p < 0.05: 0/37 (0%)
- Significant at p < 0.10: 0/37 (0%)
- Result is consistently insignificant across all 37 specifications.

**G2 (hypothetical giving, topcoded at $500)**:
- Baseline: coef = -2.181, p = 0.591 (not significant)
- Coefficient range: [-3.872, 1.201]
- P-value range: [0.271, 0.908]
- Significant at p < 0.05: 0/34 (0%)
- Significant at p < 0.10: 0/34 (0%)
- Result is consistently insignificant across all 34 specifications.

**G3 (charity support, 1-7 scale)**:
- Baseline: coef = -0.221, p = 0.167 (not significant)
- Coefficient range: [-0.282, 0.092]
- P-value range: [0.095, 0.384]
- Significant at p < 0.05: 0/32 (0%)
- Significant at p < 0.10: 2/32 (6.3%)
- Marginally close to significance in a few specs (extended controls, full sample), but generally insignificant.

**G4 (government support, 1-7 scale)**:
- Baseline: coef = -0.435, p = 0.026 (significant at 5%)
- Coefficient range: [-0.503, -0.138]
- P-value range: [0.009, 0.234]
- Significant at p < 0.05: 28/32 (87.5%)
- Significant at p < 0.10: 29/32 (90.6%)
- This is the paper's strongest result for white respondents. Significance is robust across most specs. The 4 insignificant specs are: full sample (pooling races dilutes the effect), Slidell-only (marginal at p=0.090), Biloxi-only (p=0.234), and race-shown-only (p=0.232). The subpopulation splits reduce power, and the full-sample result includes black respondents who show the opposite effect.

---

## Deviations and Notes

1. **design/diff_in_means and rc/controls/sets/none are numerically identical**: Both are regressions of outcome on treatment dummies only (no controls). The surface lists both (one as a design variant, one as an RC). The estimates are identical by construction.

2. **rc/controls/progression/manipulation_plus_demographics_plus_charity is numerically identical to baseline for G1**: On the white subsample, the baseline formula includes `black` and `other` as controls, but these are all zeros (dropped as collinear). The progression without race controls produces identical results.

3. **rc/controls/progression/full is numerically identical to rc/controls/sets/extended**: Both use the full control set (nraud + race + demographics + charitable + extra controls).

4. **Outcome preprocessing for G1 (giving)**:
   - `topcode_giving_at_99`: Topcode giving at value 99 (changes 437 observations from 100 to 99). This produces a slightly different coefficient (-4.171 vs -4.198).
   - `winsor_1_99`: Winsorize by flooring at 1 and capping at 99. Standard percentile-based 1/99 winsorization is a no-op on this bounded [0, 100] variable, so value-based bounds were used instead.

5. **Outcome preprocessing for G2 (hypothetical giving)**:
   - `topcode_hypgiv_at_250`: More aggressive topcode ($250 vs baseline $500). Coefficient strengthens to -3.872 (from -2.181).
   - `no_topcode`: Raw hypothetical giving without topcoding. Coefficient flips sign to +1.201 (p=0.836), illustrating sensitivity to extreme values.

6. **race_shown_only subsample (picobscur==0)**: Creates perfect collinearity between `picraceb` and `picshowblack` (they are identical when the obscured condition is excluded). The collinearity detection drops `picraceb`, and the coefficient on `picshowblack` is interpretable but has a different estimand (black vs white, excluding obscured).

7. **Balance check diagnostic (G1)**: 1/18 covariates significant at p < 0.05 (married, p=0.030); 2/18 at p < 0.10 (retired, p=0.052). This is consistent with expected false positive rates in a randomized experiment.

---

## Software Stack

| Package | Version | Role |
|---------|---------|------|
| Python | 3.x | Runtime |
| pandas | 2.x | Data loading/manipulation |
| numpy | 1.x | Numerical operations |
| statsmodels | 0.14+ | WLS estimation with HC1/HC2/HC3 robust SEs |

**Estimator**: `statsmodels.api.WLS` with `cov_type='HC1'` (baseline), `'HC2'`, `'HC3'`, `'nonrobust'` (classical).

**Data**: `katrina.dta` loaded via `pd.read_stata()`.

---

## Output Files

| File | Rows | Description |
|------|------|-------------|
| `specification_results.csv` | 135 | All core specifications (baseline + design + rc + infer) |
| `diagnostics_results.csv` | 1 | Balance check for G1 |
| `spec_diagnostics_map.csv` | 37 | Links G1 specs to balance diagnostic |
| `scripts/paper_analyses/113561-V1.py` | - | Executable analysis script |
