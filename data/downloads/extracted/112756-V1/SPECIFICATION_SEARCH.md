# Specification Search Report: 112756-V1

**Paper:** Martinez-Bravo (2014), "The Role of Local Officials in New Democracies: Evidence from Indonesia", AER 104(4)

## Baseline Specification

- **Design:** Cross-sectional OLS
- **Outcome:** GolkarFirst (binary: Golkar wins plurality in 1999 election)
- **Treatment:** kelurDum (kelurahan vs desa administrative status)
- **Controls:** 23 controls (geography, religion, facilities)
- **Fixed effects:** kab (district)
- **Clustering:** kab (district)

| Statistic | Value |
|-----------|-------|
| Coefficient | 0.053182 |
| Std. Error | 0.012318 |
| p-value | 0.000025 |
| 95% CI | [0.028890, 0.077473] |
| N | 43394 |
| R-squared | 0.3815 |

## Specification Counts

- Total specifications: 51
- Successful: 51
- Failed: 0
- Inference variants: 2

## Category Breakdown

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline | 2 | 2/2 | [0.0532, 0.2592] |
| Controls LOO | 20 | 20/20 | [0.0404, 0.0561] |
| Controls Sets | 4 | 3/4 | [0.0065, 0.0560] |
| Controls Progression | 5 | 4/5 | [0.0065, 0.0739] |
| Controls Subset | 15 | 12/15 | [0.0132, 0.0601] |
| Sample Trimming | 2 | 2/2 | [0.0532, 0.0532] |
| Fixed Effects | 2 | 2/2 | [0.0501, 0.2488] |
| Functional Form | 1 | 1/1 | [0.2592, 0.2592] |

## Inference Variants

| Spec ID | SE | p-value | 95% CI |
|---------|-----|---------|--------|
| infer/se/hc/hc1 | 0.008165 | 0.000000 | [0.037178, 0.069185] |
| infer/se/cluster/kec | 0.010967 | 0.000001 | [0.031678, 0.074686] |

## Overall Assessment

- **Sign consistency:** All specifications have the same sign
- **Significance stability:** 46/51 (90.2%) specifications significant at 5%
- **Direction:** Median coefficient is positive (0.053160)
- **Robustness assessment:** STRONG

Surface hash: `sha256:28b522798b8f1602a78ddcdbe86bd1304a08e60fc2877bea2bc293b06cef26b4`
