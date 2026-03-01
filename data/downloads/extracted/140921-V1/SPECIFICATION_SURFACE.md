# Specification Surface: 140921-V1

**Paper**: "Assortative Matching at the Top of the Distribution: Evidence from the World's Most Exclusive Marriage Market" by Marc Goni

**Design**: cross_sectional_ols (probit and OLS)

**Created**: 2026-02-24

---

## Paper Summary

This paper studies how access to a centralized marriage market (the London Season) affected assortative matching among the British peerage. The quasi-experimental variation comes from the 3-year interruption of the Season during 1861-63 (Queen Victoria's mourning period after Prince Albert's death). The paper constructs a "synthetic probability" of marrying during the interruption (syntheticT) based on each woman's age in 1861, using this continuous treatment intensity measure to estimate the effect on out-marriage and sorting.

The main results are in Table 2 (probit and OLS), with IV results in Table 3 and non-parametric evidence in Tables 4-5 and Figure 8. Tables 6-7 examine downstream political economy consequences using IV estimation (a different claim object outside our scope).

---

## Baseline Groups

### G1: Effect on probability of marrying a commoner

**Claim object**:
- **Outcome**: Whether a peer's daughter married a commoner (binary: cOut)
- **Treatment**: Synthetic probability of marrying during the Season's interruption (syntheticT, continuous 0-100)
- **Estimand**: Cross-sectional effect of reduced market access on out-marriage probability
- **Population**: 644 peers' daughters aged 15-35 in 1861, first marriage only, excluding foreigners and royals

**Baseline specs**:
- Table 2, Panel A, Col 1: probit marginal effects of syntheticT on cOut with controls (pr4, biorder, hengpee), clustered by birth year
- Table 2, Panel B, Col 1: same with distlondon added as control

The paper's primary estimator is probit with marginal effects. The OLS analog (which we can run directly) is the design alternative `design/cross_sectional_ols/estimator/ols`.

**Why this is the main claim**: Table 2 is the headline results table. Column 1 (married a commoner) is the paper's signature finding and the outcome most discussed in the text.

### G2: Effect on wealth sorting (mismatch)

**Claim object**:
- **Outcome**: Absolute mismatch in landholding percentile ranks between spouses (fmissmatch, continuous)
- **Treatment**: syntheticT
- **Estimand**: Effect of reduced market access on assortative matching by wealth
- **Population**: Subset of baseline sample with non-missing landholding data for both spouses (~324 couples)

**Baseline specs**:
- Table 2, Panel A, Col 3: OLS of fmissmatch on syntheticT + controls, clustered by birth year
- Table 2, Panel B, Col 3: same with distlondon added

**Why a separate group**: This is a different outcome concept (continuous wealth sorting vs. binary out-marriage) and uses a different subsample (restricted to couples with landholding data).

---

## What Is Included and Why

### Design alternatives
- **OLS linear probability model**: The paper's primary estimator for binary outcomes is probit (marginal effects), but the OLS/LPM analog is the standard design alternative for this cross-sectional setting. For the continuous outcomes in G2, OLS is already the baseline estimator.

### Robustness checks (rc/*)

**Controls (leave-one-out)**:
- Drop each of the 3-4 controls one at a time (pr4, biorder, hengpee, distlondon). The control set is very small (3-4 variables), making full LOO enumeration trivial.
- No controls (bivariate)
- Full with distlondon vs. without distlondon

**Sample restrictions**:
- Trim syntheticT distribution (5th-95th percentile) to check sensitivity to extreme treatment intensities
- Restrict age window (18-30 or 18-33 instead of 15-35) to check sensitivity to age-range definition
- Drop mourning cohort observations as a robustness check

**Functional form**:
- For G1: OLS/LPM instead of probit (already listed as design alternative)
- For G1: Binary treatment (top quintile of syntheticT, as the paper does for non-parametric Tables 4-5)
- For G2: Signed mismatch (fmissmatch2 = husband - wife) instead of absolute mismatch
- For G2: Log transform of absolute mismatch

**Fixed effects**:
- Add birth-year FE (byear) to check if results survive absorbing cohort-level variation. The paper clusters by byear but does not include byear FE in the probit/OLS specs.

### Inference variants
- **Canonical**: Cluster at birth year (byear), matching the paper
- **Variant 1**: Heteroskedasticity-robust HC1 (no clustering)
- **Variant 2**: Bootstrap-t for small-cluster correction (Cameron, Gelbach, Miller 2008), as reported in the paper's Table 2 small-cluster supplement

---

## What Is Excluded and Why

### Tables 3-5 (IV and non-parametric)
Table 3 uses IV estimation (cmp / ivregress liml) with syntheticT as instrument for mourning-period exposure. This is a different identification strategy (IV, not selection-on-observables) and would require a separate design code. Tables 4-5 and Figure 8 use non-parametric methods (contingency tables, Kendall rank correlations, KS tests). These are excluded from the core specification surface as they represent different estimand/method families.

### Tables 6-7 (political economy outcomes)
Tables 6-7 examine downstream consequences (political power, education provision) using IV estimation with cOut as the endogenous regressor. This is a distinct claim object (political outcomes, not marriage outcomes) using a different estimand (2SLS/LIML effect of out-marriage on politics), and uses a different dataset (final-data-sec4.dta, collapsed to family level). Excluded as exploration.

### Other outcomes in Table 2
Table 2 also reports results for mheir (married an heir), fmissmatch2 (signed mismatch), fdown (married down), and celibacy. These are included as baseline_spec_ids (fmissmatch2) or could be exploration, but the primary claims are cOut and fmissmatch.

### Appendix analyses
The online appendix (master-appendix.do) contains additional robustness checks run by the authors. These were not reviewed in detail but could inform future expansion.

---

## Constraints

- **Control-count envelope**: [3, 4] -- the paper uses exactly 3 controls in Panel A and 4 in Panel B. The control pool is inherently very small (historical peerage data with limited covariates).
- **Cluster structure**: Birth year (byear) is the natural clustering unit but yields a small number of clusters (~20 birth years). The paper supplements with bootstrap-t correction.
- **No linked adjustment**: Simple OLS/probit, no bundled components.

---

## Budget and Sampling

- **G1**: ~60 specs total. Full enumeration feasible given only 3-4 controls.
- **G2**: ~50 specs total. Full enumeration feasible.
- No random subset sampling needed -- the control pool is too small for combinatorial explosion.

---

## Key Implementation Notes

1. **Data files**: `data/final-data.dta` is the main analysis dataset. `data/final-data-sec4.dta` is for Tables 6-7 only.
2. **Sample filter**: `base_sample==1` for G1; additionally `fmissmatch!=.` for G2.
3. **Treatment variable**: `syntheticT` is a continuous variable (synthetic probability, 0-100 scale) constructed from age-specific marriage hazard rates.
4. **Probit marginal effects**: The paper reports average marginal effects from probit. For specification search, OLS/LPM coefficients are the natural analog and are directly interpretable as partial effects.
5. **Small number of clusters**: Birth-year clustering yields ~20 clusters. The paper acknowledges this and reports bootstrap-t corrections. Inference variants should include both standard cluster-robust and HC1 as alternatives.
