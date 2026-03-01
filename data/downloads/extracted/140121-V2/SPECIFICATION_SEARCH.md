# Specification Search Report: 140121-V2

**Paper:** Jones & Marinescu (2022), "The Labor Market Impacts of Universal and Permanent Cash Transfers: Evidence from the Alaska Permanent Fund", AER 112(7)

## Baseline Specification

- **Design:** Panel DiD (TWFE analog of paper's synthetic control method)
- **Outcome:** employed (employment-to-population ratio)
- **Treatment:** alaska_post (Alaska x Post-1982 interaction)
- **Controls:** None in baseline (pure DiD); demographic controls in robustness
- **Fixed effects:** statefip + year
- **Clustering:** statefip (state-level)
- **Note:** Paper's primary method is synthetic control; DiD is a robustness check (Appendix C). Industry composition controls (ind1-ind5) are excluded because they are mechanically collinear with the employment outcome.

| Statistic | Value |
|-----------|-------|
| Coefficient | -0.000378 |
| Std. Error | 0.002511 |
| p-value | 0.880996 |
| 95% CI | [-0.005422, 0.004666] |
| N | 1938 |
| R-squared | 0.8816 |

## Specification Counts

- Total specifications: 57
- Successful: 57
- Failed: 0
- Inference variants: 3

## Category Breakdown

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline (all outcomes) | 4 | 3/4 | [-1.5049, 0.0140] |
| Controls (add/LOO/subset) | 17 | 13/17 | [0.0002, 0.0370] |
| Sample Subgroups | 10 | 8/10 | [-0.0516, 0.0372] |
| Time Windows | 8 | 5/8 | [-1.5518, 0.0155] |
| Fixed Effects | 3 | 0/3 | [-0.0024, -0.0004] |
| Alt Outcomes + Controls | 13 | 11/13 | [-1.5049, 0.0235] |
| Treatment Timing | 2 | 2/2 | [-0.0096, 0.0107] |

## Inference Variants

| Spec ID | SE | p-value | 95% CI |
|---------|-----|---------|--------|
| infer/se/hc/hc1 | 0.004642 | 0.935132 | [-0.009482, 0.008727] |
| infer/se/hc/hc3 | FAILED | - | - |
| infer/se/iid | 0.007909 | 0.961898 | [-0.015890, 0.015134] |

## Overall Assessment

- **Employment specifications:** 40 total
- **Sign consistency (employment):** Mixed signs across specifications
- **Significance stability (employment):** 27/40 (67.5%) significant at 5%
- **Direction:** Median coefficient is positive (0.009548)
- **Overall significance (all outcomes):** 42/57 (73.7%)
- **Robustness assessment:** WEAK
- **Note:** The paper's main finding is a null effect on employment and a positive effect on part-time work. The paper uses synthetic control with permutation inference as the primary method; DiD provides qualitatively similar results.

Surface hash: `sha256:15b6981762df337c41781c3f1232dbdcd31517b0b5fd9ee848c80c9daab9361a`
