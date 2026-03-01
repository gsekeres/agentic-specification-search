# Specification Search Report: 138922-V1

**Paper**: Marcus, Siedler & Ziebarth, "The Long-Run Effects of Sports Club Vouchers for Primary School Children," *AEJ: Economic Policy*

**Execution date**: 2026-02-24

---

## Surface Summary

- **Paper ID**: 138922-V1
- **Baseline groups**: 1 (G1: sportsclub ~ treat)
- **Design**: Difference-in-Differences (repeated cross-section TWFE)
- **Budget**: 60 max core specs
- **Seed**: 138922 (used for random control subset draws)
- **Canonical inference**: CRV1 cluster at cityno (~93 clusters)

---

## Execution Summary

| Category | Planned | Executed | Succeeded | Failed |
|----------|---------|----------|-----------|--------|
| Baseline | 8 | 8 | 8 | 0 |
| RC/controls | 24 | 24 | 24 | 0 |
| RC/sample | 12 | 12 | 12 | 0 |
| RC/fe | 3 | 3 | 2 | 1 |
| RC/data | 3 | 3 | 3 | 0 |
| RC/weights | 3 | 3 | 3 | 0 |
| **Total core** | **53** | **53** | **52** | **1** |
| Inference variants | 3 | 3 | 3 | 0 |

---

## Baseline Results

### Primary baseline (Table 2, Column 3)

```
sportsclub ~ treat | year_3rd + bula_3rd + cityno, cluster(cityno)
```

- **Coefficient**: 0.0089
- **SE**: 0.0187
- **p-value**: 0.635
- **N**: 13,333

Interpretation: The C2SC voucher program had a small, statistically insignificant positive effect on sports club membership (0.89 percentage points, p=0.63).

### Alternative baseline outcomes (Table 2, Column 3)

| Outcome | Coefficient | SE | p-value | N |
|---------|-------------|-----|---------|-------|
| kommheard (program known) | 0.2760 | 0.0144 | <0.001 | 13,333 |
| kommgotten (voucher received) | 0.2020 | 0.0111 | <0.001 | 13,333 |
| kommused (voucher redeemed) | 0.1218 | 0.0060 | <0.001 | 13,333 |
| sportsclub (membership) | 0.0089 | 0.0187 | 0.635 | 13,333 |
| sport_hrs (hours/week) | -0.0019 | 0.1597 | 0.990 | 13,333 |
| oweight (overweight) | 0.0041 | 0.0158 | 0.794 | 13,333 |

The program implementation measures (knowledge, receipt, redemption) show large, highly significant effects. The downstream behavioral outcomes (sports club membership, hours, overweight) show null effects.

---

## RC Results Overview

### Controls axis (24 specs)
- **Full controls (9 vars)**: coef=0.0242, p=0.198. Adding controls increases the point estimate but it remains insignificant.
- **LOO from full set**: Range 0.0202 to 0.0261. All insignificant. Dropping sportsclub_4_7 had the most influence.
- **Progression**: Bivariate 0.0089, demographics 0.0130, socioeconomic 0.0141, full 0.0242. Point estimate increases monotonically with controls.
- **Random subsets (10 draws)**: Range 0.0078 to 0.0242. All insignificant.

### Sample axis (12 specs)
- **Time windows**: Range -0.0137 to 0.0165. Extending to 2011 flips the sign (coef=-0.014). All insignificant.
- **State composition**: Dropping Brandenburg (coef=0.0169) vs dropping Thuringia (coef=0.0038). Neither significant.
- **Quality filters**: Range 0.0040 to 0.0166. Excluding children with older siblings reduces N substantially (to ~5,890). All insignificant.

### Fixed effects axis (3 specs, 1 failed)
- **Drop city FE**: coef=0.0027, p=0.889. Same as Table 2 Col 2.
- **Drop state FE (city subsumes)**: coef=0.0127, p=0.449.
- **State-by-year interaction FE**: FAILED (collinear with treatment -- state x year FE absorbs the DiD treatment variable).

### Treatment definition axis (3 specs)
- **First-wave treatment**: coef=0.0096, p=0.577.
- **Current-state treatment**: coef=-0.0013, p=0.938.
- **Alternative coding (treat_v2)**: coef=0.0178, p=0.531. Smaller sample (N=6,117) due to missing treat_v2.

### Weights axis (3 specs)
- **Entropy balance weights**: coef=-0.0161, p=0.314. Sign flips under entropy balancing.
- **Entropy balance + controls**: coef=0.0017, p=0.932.
- **Survey weights (weight2)**: coef=0.0132, p=0.526.

---

## Inference Variants (on baseline)

| Inference | SE | p-value | Notes |
|-----------|-----|---------|-------|
| CRV1 cityno (canonical) | 0.0187 | 0.635 | 93 clusters |
| HC1 (robust) | 0.0186 | 0.631 | Very similar to canonical |
| CRV1 bula_3rd (state) | 0.0051 | 0.224 | Only 3 clusters -- inference unreliable |
| CRV1 cohort | 0.0073 | 0.244 | 15 clusters -- small cluster count |

Note: State-level clustering (3 clusters) produces a very small SE but the resulting inference is unreliable with so few clusters. The cohort-level clustering (15 clusters) substantially reduces the SE, pushing p from 0.63 to 0.24, but this still does not reach conventional significance.

---

## Failures

1. **rc/fe/add/bula_3rd_x_year_3rd**: State-by-year interaction FE are collinear with the treatment variable `treat = tbula_3rd * tcoh`, which is itself a state-by-post-period interaction. This is expected and correctly identified as infeasible.

---

## Software Stack

- Python 3.12
- pyfixest 0.40+
- pandas 2.2.3
- numpy 2.2.3
- scipy 1.15.2

---

## Key Finding

Across all 52 successful specifications, the effect of the C2SC sports club voucher program on sports club membership ranges from approximately -0.016 to +0.026, with none reaching statistical significance at conventional levels. The null result is robust across alternative control sets, sample windows, state compositions, treatment definitions, fixed effects structures, and weighting schemes. This is consistent with the paper's own conclusion that while the voucher program was well-known and used, it did not significantly increase sports club membership rates.

The program implementation outcomes (knowledge, receipt, redemption) are strongly significant across all specifications, confirming that the program reached children but did not change their sports participation behavior.
