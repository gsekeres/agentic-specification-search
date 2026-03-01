# Specification Search Report: 112815-V1

**Paper:** Hoxby (2014), "The Economics of Online Postsecondary Education: MOOCs, Nonselective Education, and Highly Selective Education", AER P&P 104(5)

## Important Note

This paper is primarily descriptive with no causal claims. The original regressions require
restricted-access BPS 2004/2009 data (NCES license) which is not included in the package.
The specifications below use the available IPEDS institution-level data to characterize how
NSPE (nonselective postsecondary education) institutions differ in financial structure from
other institutions, matching the paper's second-part analysis.

## Baseline Specification

- **Design:** Cross-sectional OLS (descriptive, no causal interpretation)
- **Outcome:** tuitionrevshare (tuition revenue share)
- **Treatment:** nspe (NSPE institution indicator)
- **Controls:** public, forprofit (sector dummies)
- **Fixed effects:** none
- **Standard errors:** HC1 (heteroskedasticity-robust)

| Statistic | Value |
|-----------|-------|
| Coefficient | -0.046689 |
| Std. Error | 0.006501 |
| p-value | 0.000000 |
| 95% CI | [-0.059433, -0.033945] |
| N | 5519 |
| R-squared | 0.5875 |

## Specification Counts

- Total specifications: 71
- Successful: 71
- Failed: 0
- Inference variants: 5

## Category Breakdown

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baselines | 7 | 7/7 | [-0.3470, 0.6955] |
| Controls Sets | 7 | 6/7 | [-0.2659, 0.0018] |
| Controls LOO | 4 | 4/4 | [-0.0853, -0.0325] |
| Controls Progression | 6 | 6/6 | [-0.2659, -0.0181] |
| Controls Subset | 10 | 6/10 | [-0.2993, 0.0019] |
| Sample Variants | 11 | 6/11 | [-0.4973, 0.0102] |
| Outcome Variants | 18 | 16/18 | [-0.4080, 1.3070] |
| Fixed Effects | 3 | 3/3 | [-0.1209, -0.0247] |
| Functional Form (WLS) | 5 | 3/5 | [-0.3470, -0.0196] |

## Inference Variants

| Spec ID | SE | p-value | 95% CI |
|---------|-----|---------|--------|
| infer/se/hc/hc1 | 0.006501 | 0.000000 | [-0.059433, -0.033945] |
| infer/se/hc/hc3 | 0.059190 | 0.452967 | [-0.183181, 0.089803] |
| infer/se/cluster/sector | 0.019030 | 0.039726 | [-0.090573, -0.002805] |
| infer/se/cluster/carnegie | 0.015266 | 0.006467 | [-0.078641, -0.014737] |
| infer/se/iid | 0.008185 | 0.000000 | [-0.062734, -0.030644] |

## Overall Assessment

### Primary outcome (tuitionrevshare)
- **Sign consistency:** Mixed signs across specifications
- **Significance stability:** 33/44 (75.0%) specifications significant at 5%
- **Direction:** Median coefficient is negative (-0.052370)
- **Robustness assessment:** WEAK

### Caveats
- This paper is explicitly descriptive with no causal claims
- The original regressions (incres09 ~ cert* + instid*) require restricted BPS data
- These specifications use available IPEDS data as an adapted specification search
- The NSPE indicator is partly collinear with Carnegie classification by construction

Surface hash: `sha256:def1dc4a61c849332a21c129f767664121dbcb974faa5a9190ca7fd5010fecb0`
