# Specification Search Report: 112474-V1

**Paper**: Dinkelman (2011), "The Effects of Rural Electrification on Employment: New Evidence from South Africa"
**Design**: Instrumental Variables (2SLS)
**Date**: 2026-02-24

---

## Surface Summary

- **Paper ID**: 112474-V1
- **Baseline groups**: 2
  - G1: Female employment change (d_prop_emp_f)
  - G2: Male employment change (d_prop_emp_m)
- **Design**: IV with land gradient (mean_grad_new) as instrument for electrification (T)
- **Budget per group**: 40 specs (80 total)
- **Seed**: 112474 (for random control subsets)
- **Surface hash**: sha256 computed at runtime

---

## Execution Summary

| Metric | Value |
|--------|-------|
| Total specs planned | 80 |
| Total specs executed | 80 |
| Successful | 80 |
| Failed | 0 |
| Inference variants | 4 (2 per group) |

### Breakdown by Group

| Group | Outcome | Planned | Executed | Successful | Failed |
|-------|---------|---------|----------|------------|--------|
| G1 | d_prop_emp_f | 40 | 40 | 40 | 0 |
| G2 | d_prop_emp_m | 40 | 40 | 40 | 0 |

### Breakdown by Spec Type

| Spec Type | Count per Group | Total |
|-----------|----------------|-------|
| baseline | 1 | 2 |
| additional baselines | 2 | 4 |
| design/iv/estimator/liml | 1 | 2 |
| rc/controls/loo/* | 12 | 24 |
| rc/controls/progression/* | 4 | 8 |
| rc/controls/subset/* | 10 | 20 |
| rc/sample/restriction/* | 4 | 8 |
| rc/sample/outliers/* | 2 | 4 |
| rc/fe/* | 2 | 4 |
| rc/form/outcome/asinh | 1 | 2 |
| rc/estimation/ols_reduced_form | 1 | 2 |
| **Total** | **40** | **80** |

---

## Key Results

### G1: Female Employment (Headline Result)

- **Baseline (Table 4, Col 9)**: coef = 0.0951, SE = 0.0553, p = 0.087
  - Interpretation: Electrification increases female employment rate by 9.5 percentage points (LATE)
  - First-stage F ~ 8.3 (weak-instrument concern)

- **Coefficient range across 40 specs**: [-0.0007, 0.1297]
- **Median coefficient**: 0.087
- **Sign consistency**: 39/40 specs have positive coefficients (97.5%)
- **Significance (p < 0.10)**: 15/40 specs (37.5%)
- **Significance (p < 0.05)**: 3/40 specs (7.5%)

The female employment result is **moderately robust** to specification changes. The coefficient is positive across nearly all specifications but statistical significance varies considerably. Key patterns:
- LOO controls: all positive, most marginally significant (p = 0.07-0.11)
- Fewer controls (bivariate, geographic only): smaller and insignificant coefficients
- Full sample (no size restriction): larger and more significant (coef = 0.130, p = 0.026)
- Trimmed samples: significant at 5-95 percentiles (p = 0.039)
- Spillover-restricted samples: smaller and less significant

### G2: Male Employment (Null Result)

- **Baseline (Table 4, Col 9)**: coef = 0.0355, SE = 0.0659, p = 0.591
  - Interpretation: No significant effect of electrification on male employment

- **Coefficient range across 40 specs**: [-0.0723, 0.0791]
- **Median coefficient**: 0.030
- **Sign consistency**: 32/40 specs have positive coefficients (80%)
- **Significance (p < 0.10)**: 0/40 specs (0%)

The male employment null result is **very robust**. No specification produces a statistically significant effect. Coefficients are centered near zero with some variation in sign, consistent with a true null.

### Inference Variants (Baseline Spec)

| Variant | G1 SE | G1 p-value | G2 SE | G2 p-value |
|---------|-------|-----------|-------|-----------|
| CRV1(placecode0) [canonical] | 0.0553 | 0.087 | 0.0659 | 0.591 |
| HC1 (robust) | 0.0510 | 0.063 | 0.0625 | 0.571 |
| CRV1(dccode0) | 0.0532 | 0.107 | 0.0852 | 0.687 |

HC1 standard errors are slightly smaller (more significant). District-level clustering (10 clusters) yields slightly larger SEs for G1 and substantially larger SEs for G2, consistent with within-district correlation.

---

## Deviations and Notes

1. **Variable naming**: The Stata code uses `sexratio0` but the data file has `sexratio0_a`. We use `sexratio0_a` directly.
2. **LIML vs 2SLS**: Results are identical as expected for a just-identified model (1 instrument, 1 endogenous variable).
3. **Reduced form**: The OLS reduced form for G1 (outcome regressed on instrument) shows a significant negative relationship between land gradient and female employment change (p = 0.031), confirming the causal chain.
4. **No specs failed**: All 80 specifications executed successfully.

---

## Software Stack

- Python 3.12
- pyfixest 0.40+
- linearmodels 6.1
- pandas, numpy, scipy (standard versions)
- All IV specs use pyfixest except LIML (linearmodels)
