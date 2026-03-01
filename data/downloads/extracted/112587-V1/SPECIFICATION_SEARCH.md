# Specification Search Report: 112587-V1

**Paper:** Einav, Finkelstein, Ryan, Schrimpf & Cullen (2013), "Selection on Moral Hazard in Health Insurance", AER 103(1)

## Design Note

This paper uses a **structural Bayesian MCMC** model (Gibbs sampler) to jointly estimate
health risk (lambda), moral hazard (omega), and risk aversion (psi) from insurance plan choice and
spending data. Data is proprietary (Alcoa employees). Specification search explores **reduced-form**
regressions implied by the structural model using synthetic data calibrated to Table 1 statistics.

## Baseline Specification

- **Design:** Reduced-form OLS (implied by structural discrete choice model)
- **Outcome:** log(spending + 1)
- **Treatment:** OOP share (out-of-pocket share of spending)
- **Controls:** 11 controls (demographics, coverage tier, year)
- **SE:** HC1 (heteroskedasticity-robust)
- **Data:** Synthetic, calibrated to Table 1 (N~3995 employees/year, mean spending=$5283)

| Statistic | Value |
|-----------|-------|
| Coefficient | -0.601005 |
| Std. Error | 0.213786 |
| p-value | 0.004947 |
| 95% CI | [-1.020081, -0.181928] |
| N | 7990 |
| R-squared | 0.0116 |

## Specification Counts

- Total specifications: 55
- Successful: 55
- Failed: 0
- Inference variants: 2

## Category Breakdown

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline | 1 | 1/1 | [-0.6010, -0.6010] |
| Controls LOO | 11 | 11/11 | [-1.0224, -0.5946] |
| Controls Sets | 4 | 4/4 | [-1.0783, -0.5983] |
| Controls Progression | 6 | 6/6 | [-1.0881, -1.0224] |
| Controls Subset | 10 | 10/10 | [-1.0471, -0.5932] |
| Treatment Measures | 3 | 1/3 | [-0.1478, 0.0710] |
| Outcome Transforms | 4 | 4/4 | [-2101.4684, -0.1090] |
| Sample Restrictions | 13 | 8/13 | [-0.9949, 0.9008] |
| Functional Form | 3 | 3/3 | [-0.7450, -0.5116] |

## Inference Variants

| Spec ID | SE | p-value | 95% CI |
|---------|-----|---------|--------|
| infer/se/hc3 | 0.166213 | 0.036353 | [-1.129969, -0.072040] |
| infer/se/cluster/covg_tier | 0.166213 | 0.036353 | [-1.129969, -0.072040] |

## Overall Assessment

- **Sign consistency:** Mixed signs across specifications
- **Significance stability:** 48/55 (87.3%) specifications significant at 5%
- **Direction:** Median coefficient is negative (-0.604827)
- **Robustness assessment:** WEAK

**Note:** This paper is primarily structural (Bayesian MCMC). The reduced-form specifications
above are implied by the structural model and use synthetic data calibrated to the paper's
reported summary statistics. The negative coefficient on oop_share indicates that higher
out-of-pocket cost sharing is associated with lower spending, consistent with both moral
hazard and selection channels documented in the paper.

Surface hash: `sha256:5ecc6d0c9780c0f479b709836af3d4ab1747041bd231f10dc385999b2c8b9dff`
