# Specification Search Report: 116248-V1

**Paper:** Costa-Gomes & Crawford (2006), "Cognition and Behavior in Two-Person Guessing Games: An Experimental Study", AER 96(5)

## Baseline Specification

- **Design:** Panel OLS (subject x game)
- **Outcome:** abs_dev_eq (absolute deviation of guess from equilibrium prediction)
- **Treatment:** comply_l1 (binary: guess within 0.5 of L1-type prediction)
- **Controls:** 2 controls (game range width, own target)
- **Fixed effects:** subject + game
- **Clustering:** subject

### Interpretation

The paper classifies experimental subjects into behavioral types (Eq, L1, L2, L3, D1, D2)
using MLE on guesses and information search patterns in 16 two-person guessing games.
The specification search tests whether L1-type compliance (the modal type) robustly
predicts lower deviations from equilibrium predictions across specification choices.

| Statistic | Value |
|-----------|-------|
| Coefficient | -25.344567 |
| Std. Error | 10.323088 |
| p-value | 0.016146 |
| 95% CI | [-45.873159, -4.815974] |
| N | 1360 |
| R-squared | 0.3838 |

## Specification Counts

- Total specifications: 55
- Successful: 55
- Failed: 0
- Inference variants: 3

## Category Breakdown

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline | 1 | 1/1 | [-25.3446, -25.3446] |
| Alt. Type Indicators | 5 | 5/5 | [-133.0547, -39.6303] |
| Alt. Outcomes | 7 | 4/7 | [-12489.0763, 29.9337] |
| Controls LOO | 6 | 5/6 | [-54.7923, -14.7223] |
| Controls Sets | 3 | 3/3 | [-34.4139, -23.4120] |
| Controls Progression | 6 | 6/6 | [-55.0626, -19.5770] |
| Sample Restrictions | 14 | 5/14 | [-48.9505, -3.2191] |
| Fixed Effects | 5 | 5/5 | [-46.0174, -16.9025] |
| Functional Form | 2 | 1/2 | [-0.0466, -0.0035] |
| Cross-Type | 6 | 6/6 | [-136.6700, -72.6964] |

## Inference Variants

| Spec ID | SE | p-value | 95% CI |
|---------|-----|---------|--------|
| infer/se/hc/hc1 | 7.854420 | 0.001284 | [-40.753761, -9.935372] |
| infer/se/cluster/session | 2.696850 | 0.000714 | [-32.832224, -17.856910] |
| infer/se/cluster/game | 25.842132 | 0.342273 | [-80.425768, 29.736635] |

## Overall Assessment

- **Sign consistency:** Mixed signs across specifications
- **Significance stability:** 41/55 (74.5%) specifications significant at 5%
- **Direction:** Median coefficient is negative (-30.735447)
- **Robustness assessment:** WEAK

Surface hash: `sha256:4cd9c18cd348f52b0d80897da609add46213d6ad22c0af719507df7bb198bb95`
