# Specification Search Report: 116167-V1

**Paper:** Durante, Pinotti & Tesei (2019), "The Political Legacy of Entertainment TV", AER 109(7)

## Baseline Specification

- **Design:** Cross-sectional OLS
- **Outcome:** berl (Forza Italia vote share in 1994)
- **Treatment:** signal (early exposure to Mediaset commercial TV, standardized)
- **Controls:** signalfree + area, area2, altitude, altitude2, ruggedness, electorate, lnincome, highschool_college81
- **Fixed effects:** district + sll2001 (local labor system)
- **Clustering:** district
- **Weights:** pop81 (population weights)
- **Sample:** Trimmed top/bottom 2.5% of signal distribution

| Statistic | Value |
|-----------|-------|
| Coefficient | 0.851497 |
| Std. Error | 0.226982 |
| p-value | 0.000204 |
| 95% CI | [0.405179, 1.297814] |
| N | 7482 |
| R-squared | 0.9032 |

## Specification Counts

- Total specifications: 52
- Successful: 52
- Failed: 0
- Inference variants: 3

## Category Breakdown

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline | 4 | 4/4 | [0.8515, 1.3649] |
| Controls LOO | 7 | 7/7 | [0.4901, 0.9636] |
| Controls Sets | 4 | 4/4 | [0.8052, 1.3649] |
| Controls Progression | 6 | 6/6 | [0.8515, 3.6826] |
| Controls Add | 6 | 6/6 | [0.8126, 0.8610] |
| Fixed Effects | 3 | 3/3 | [0.5516, 0.8873] |
| Sample Restriction | 14 | 12/14 | [0.1978, 1.0559] |
| Sample Outliers | 2 | 2/2 | [0.6924, 0.7751] |
| Weights | 2 | 2/2 | [0.8515, 0.9760] |
| Functional Form | 1 | 1/1 | [0.8515, 0.8515] |
| Matched Neighbors | 3 | 3/3 | [0.5844, 0.8336] |

## Inference Variants

| Spec ID | SE | p-value | 95% CI |
|---------|-----|---------|--------|
| infer/se/hc/hc1 | 0.187343 | 0.000006 | [0.484242, 1.218751] |
| infer/se/cluster/sll2001 | 0.212559 | 0.000069 | [0.434128, 1.268865] |
| infer/se/cluster/two_way_district_sll | 0.211101 | 0.000067 | [0.436406, 1.266588] |

## Overall Assessment

- **Sign consistency:** All specifications have the same sign
- **Significance stability:** 50/52 (96.2%) specifications significant at 5%
- **Direction:** Median coefficient is positive (0.851497)
- **Robustness assessment:** STRONG

Surface hash: `sha256:f2453230f24900eff8a8f52ba17e4069d7c3660b9d1f72f81812803d9969abd5`
