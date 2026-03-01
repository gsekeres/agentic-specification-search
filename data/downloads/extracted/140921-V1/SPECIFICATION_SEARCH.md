# Specification Search: 140921-V1

**Paper**: "Assortative Matching at the Top of the Distribution: Evidence from the World's Most Exclusive Marriage Market" by Marc Goni (AER, 2023)

**Executed**: 2026-02-25

---

## Surface Summary

- **Paper ID**: 140921-V1
- **Design**: cross_sectional_ols (probit marginal effects for G1; OLS for G2)
- **Baseline groups**: 2 (G1: married a commoner, G2: wealth mismatch)
- **Budgets**: G1 max 60, G2 max 50 (total ~110 budget, 50 specs used)
- **Seed**: G1=140921, G2=140922
- **Sampling**: Full enumeration (very small control pool: 3-4 controls)

### Baseline Group G1: Married a commoner (cOut)

- **Estimator**: Probit marginal effects (paper baseline)
- **Treatment**: syntheticT (synthetic probability of marrying during 1861-63 Season interruption)
- **Outcome**: cOut (binary: married a commoner)
- **Controls**: pr4, biorder, hengpee (Panel A); + distlondon (Panel B)
- **Clustering**: birth year (byear), 22 clusters
- **Canonical inference**: cluster(byear)
- **Design alternative**: OLS/LPM

### Baseline Group G2: Wealth sorting (fmissmatch)

- **Estimator**: OLS (paper baseline)
- **Treatment**: syntheticT
- **Outcome**: fmissmatch (absolute mismatch in landholding percentile ranks)
- **Controls**: same as G1
- **Clustering**: birth year (byear), 22 clusters
- **Canonical inference**: cluster(byear)

---

## Execution Summary

### Counts

| Category | G1 | G2 | Total |
|----------|----|----|-------|
| baseline | 2 | 2 | 4 |
| design/* | 2 | 0 | 2 |
| rc/controls/* | 7 | 7 | 14 |
| rc/sample/* | 7 | 9 | 16 |
| rc/form/* | 6 | 8 | 14 |
| rc/fe/* | 1 | 1 | 2 |
| **Total specs** | **25** | **25** | **50** |
| infer/* (separate) | 3 | 2 | 5 |

### Success rate

- **Planned**: 50 specification results + 5 inference results
- **Executed**: 50 + 5
- **Successful**: 50 + 5 (100%)
- **Failed**: 0

---

## Baseline Replication

### G1 Baselines

| Spec | Coefficient | SE | p-value | N |
|------|------------|-----|---------|---|
| baseline (Probit ME, Panel A) | 0.004534 | 0.002082 | 0.029 | 644 |
| baseline__table2_panelb_col1 (Probit ME, Panel B) | 0.006195 | 0.002495 | 0.013 | 484 |
| design/ols (OLS/LPM, Panel B) | 0.005952 | 0.002519 | 0.028 | 484 |
| design/ols__panela (OLS/LPM, Panel A) | 0.004328 | 0.002101 | 0.052 | 644 |

The probit marginal effects match the paper's reported values (Table 2, Col 1). OLS coefficients are very close to the probit marginal effects, as expected for a linear probability model analog.

### G2 Baselines

| Spec | Coefficient | SE | p-value | N |
|------|------------|-----|---------|---|
| baseline (OLS, Panel A) | 0.524 | 0.196 | 0.014 | 324 |
| baseline__table2_panelb_col3 (OLS, Panel B) | 0.512 | 0.213 | 0.025 | 260 |

G2 baselines match the paper's Table 2, Col 3 values.

---

## Key Findings from Specification Search

### G1: Effect on out-marriage

- The result is robust across control-set variations (LOO, bivariate, Panel A vs Panel B).
- Binary treatment (top quintile) yields a marginally significant positive coefficient.
- Restricting the age window to 18-30 or 18-33 weakens significance (p > 0.10), consistent with the treatment being strongest for the 20-26 age range.
- Adding birth-year FE absorbs most of syntheticT variation (which is a function of age/birth year), making the within-byear coefficient very noisy (p = 0.65).
- Trimming extreme syntheticT values (5th-95th) strengthens the effect slightly.

### G2: Effect on wealth sorting

- The result is robust across control variations and most sample restrictions.
- Signed mismatch (fmissmatch2) shows a significant negative coefficient, consistent with the absolute mismatch finding.
- Log transformation preserves significance.
- Heavy trimming of fmissmatch (5th-95th) weakens significance (p = 0.14), suggesting some sensitivity to extreme mismatch values.
- Birth-year FE again make the result insignificant, as expected.

### Inference Variants

- HC1 (no clustering) produces slightly larger standard errors for the G1 OLS spec compared to cluster(byear), and slightly smaller for some G2 specs.
- Note: G1 inference variants compare OLS estimates (not probit ME) due to the limitation of recomputing probit marginal effect SEs with different covariance types.

---

## Deviations from Surface

- **rc/fe/add/byear (G1)**: Used OLS/LPM instead of probit with birth-year FE, because probit with 22 birth-year FE levels suffers from the incidental parameters problem. This is standard practice for FE estimation with binary outcomes.
- **rc/form/treatment/quadratic_syntheticT (G1)**: Used OLS/LPM instead of probit to avoid convergence issues with the quadratic term.
- **Inference variants for G1 probit baselines**: The OLS HC1 inference variant uses OLS estimates (not probit ME), so the coefficient differs from the base probit specification. This is noted as a WARN in validation and is expected.

---

## Software Stack

- **Python**: 3.x
- **pyfixest**: for OLS estimation with cluster-robust SEs
- **statsmodels**: for probit estimation with clustered SEs and marginal effects
- **pandas**: data loading and manipulation
- **numpy**: numerical operations
