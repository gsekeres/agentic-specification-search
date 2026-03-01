# Specification Search Report: 113630-V1

**Paper:** Bjorkman Martina & Svensson (2014), "Experimental Evidence on the Long-Run Impact of Community-Based Monitoring", AEJ: Applied Economics

## Study Design

- **Design:** Cluster-randomized controlled trial (2 experiments)
- **Randomization unit:** Health facility (hfcode)
- **Strata:** District (dcode)
- **Experiment 1 (I&P):** Information & Participation, sample1==1, 2004-2009
- **Experiment 2 (P):** Participation only, sample1==0, 2006-2009

## G1: Child Mortality

### Baseline (Table 3, Panel A, Col I)

- **Outcome:** Crude under-5 death rate (annualized)
- **Treatment:** Community monitoring (I&P)
- **FE:** District (dcode)
- **SEs:** Robust (HC1) — data at cluster level

| Statistic | Value |
|-----------|-------|
| Coefficient | -0.384141 |
| Std. Error | 0.198816 |
| p-value | 0.060445 |
| 95% CI | [-0.785963, 0.017680] |
| N | 50 |
| R-squared | 0.3735 |

## G2: Weight-for-Age Z-Score

### Baseline (Table 5, Col I)

- **Outcome:** Weight-for-age z-score (zw1)
- **Treatment:** Community monitoring (I&P)
- **Sample:** Infants 0-12 months, z in (-4.5, 4.5), weight < cort97
- **FE:** District (dcode)
- **SEs:** Clustered at health facility

| Statistic | Value |
|-----------|-------|
| Coefficient | 0.008525 |
| Std. Error | 0.017701 |
| p-value | 0.632219 |
| 95% CI | [-0.027047, 0.044097] |
| N | 687 |
| R-squared | 0.0203 |

## Specification Counts

- Total specifications: 48
- Successful: 48
- Failed: 0
- Inference variants: 4

## Category Breakdown

| Category | Group | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|-------|---------------|------------|
| Baseline | G1 | 5 | 0/5 | [-4.4123, 0.0928] |
| Baseline | G2 | 4 | 1/4 | [0.0085, 0.0692] |
| Design variants | G1 | 2 | 0/2 | [-0.3911, 0.0440] |
| Design variants | G2 | 1 | 0/1 | [0.0003, 0.0003] |
| Controls | G1 | 15 | 2/15 | [-4.7103, -0.3688] |
| Controls | G2 | 4 | 0/4 | [0.0063, 0.0555] |
| Sample | G1 | 3 | 0/3 | [-0.5902, 0.0464] |
| Sample | G2 | 5 | 2/5 | [0.0085, 0.0683] |
| Outcome form | G1 | 5 | 1/5 | [-10.3176, 0.0928] |
| Outcome form | G2 | 4 | 1/4 | [-0.0184, 0.0048] |

## Inference Variants

| Spec ID | Group | SE | p-value | 95% CI |
|---------|-------|-----|---------|--------|
| infer/se/cluster/hfcode | G1_mortality | 0.198816 | 0.059134 | [-0.783676, 0.015394] |
| infer/se/cluster/dcode | G1_mortality | 0.202815 | 0.094833 | [-0.851833, 0.083551] |
| infer/se/hc/hc1 | G2_weight | 0.021294 | 0.689021 | [-0.033286, 0.050336] |
| infer/se/cluster/dcode_weight | G2_weight | 0.023236 | 0.723205 | [-0.045056, 0.062107] |

## Overall Assessment

### G1 (Mortality)
- **Sign consistency:** Mixed signs
- **Significance stability:** 3/30 (10.0%) significant at 5%
- **Direction:** Median coefficient is negative (-0.392923)
- **Robustness assessment:** FRAGILE

### G2 (Weight)
- **Sign consistency:** Mixed signs
- **Significance stability:** 4/18 (22.2%) significant at 5%
- **Direction:** Median coefficient is positive (0.011976)
- **Robustness assessment:** FRAGILE

Surface hash: `sha256:bb9fa2c1a76bb9c5abc3516a8a332bc3f1b30612ae78094dd27f9dc887aa76bf`
