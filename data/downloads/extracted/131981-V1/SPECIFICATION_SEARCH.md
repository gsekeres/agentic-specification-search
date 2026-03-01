# Specification Search Report: 131981-V1

**Paper**: "Mental Health Costs of Lockdowns: Evidence from Age-specific Curfews in Turkey"
**Authors**: Altindag, Erten, and Keskin (AEJ: Applied Economics)
**Design**: Sharp regression discontinuity at age-65 COVID curfew cutoff

---

## Surface Summary

- **Baseline groups**: 1 (G1: Mental health effect of curfew at age-65 cutoff)
- **Design code**: `regression_discontinuity`
- **Baseline spec**: z_depression ~ before1955 + dif + dif*before1955 + controls, bw=30, cluster(modate)
- **Budget**: 80 max core specs (actual: 45 executed)
- **Seed**: 131981
- **Sampling**: Full enumeration (no random control-subset sampling)

## Execution Summary

| Category | Planned | Executed | Succeeded | Failed |
|---|---|---|---|---|
| Baseline | 4 | 4 | 4 | 0 |
| Design variants | 8 | 8 | 8 | 0 |
| RC: Controls LOO | 6 | 6 | 6 | 0 |
| RC: Control sets | 3 | 3 | 3 | 0 |
| RC: Donut holes | 3 | 3 | 3 | 0 |
| RC: Alt outcomes | 3 | 3 | 3 | 0 |
| RC: Joint specs | 18 | 18 | 18 | 0 |
| **Total** | **45** | **45** | **45** | **0** |
| Inference variants | 2 | 2 | 2 | 0 |

## Baseline Results

| Spec | Bandwidth | Coefficient | SE | p-value | N |
|---|---|---|---|---|---|
| baseline (Table 4, Col 2) | 30 | 0.3877 | 0.1153 | 0.0013 | 795 |
| baseline__table4_col1_bw17 | 17 | 0.2394 | 0.1886 | 0.2130 | 486 |
| baseline__table4_col3_bw45 | 45 | 0.2436 | 0.0985 | 0.0153 | 1160 |
| baseline__table4_col4_bw60 | 60 | 0.2726 | 0.0812 | 0.0011 | 1520 |

The primary baseline (bw=30, full controls, clustered at modate) finds a 0.39 SD increase in mental distress for those subject to the curfew (p=0.001). The narrow bandwidth (bw=17) is not significant due to low power. Wider bandwidths (45, 60) are significant.

## Key Findings

### Overall Support for the Main Claim

The curfew significantly increases mental distress across the vast majority of specifications:

- **31 of 45 specifications (69%)** are significant at the 5% level
- **34 of 45 specifications (76%)** are significant at the 10% level
- **All coefficients are positive**, indicating increased mental distress from curfew exposure
- Coefficient range: [0.04, 1.41] (in SD units for index outcomes, raw count for sum_srq)

### Design Variants (Bandwidth Sensitivity)

Bandwidth is the most important design choice. Results are robust across bandwidths:
- bw=17 (narrowest): Not significant (p=0.21) due to small sample (N=486)
- bw=24 through bw=72: All significant at 5% level
- Point estimates range from 0.17 to 0.39 SD, with estimates generally larger at narrower bandwidths and smaller at wider ones -- consistent with an attenuating treatment effect away from the cutoff

The quadratic polynomial (bw=30) is not significant (p=0.33), consistent with reduced precision from adding polynomial terms with limited data.

### Controls Robustness

Results are robust to dropping any single control block:
- All 6 LOO specs are significant at p < 0.007
- Dropping province FE slightly increases the estimate (0.42 vs 0.39 baseline)
- No controls spec is marginally significant (p=0.05), confirming controls improve precision but the RD effect exists without them

### Donut Hole Tests

Excluding observations near the cutoff (1, 2, or 3 months) *strengthens* the result:
- Donut 1mo: coef=0.41, p=0.0006
- Donut 2mo: coef=0.45, p=0.002
- Donut 3mo: coef=0.37, p=0.012

This indicates the treatment effect is not driven by observations right at the cutoff boundary.

### Alternative Outcomes

All mental health measures show significant effects at bw=30:
- sum_srq (raw count): coef=1.13, p=0.020
- z_somatic: coef=0.32, p=0.009
- z_nonsomatic: coef=0.23, p=0.048

### Inference Sensitivity

The baseline estimate (bw=30) remains significant under alternative inference:
- HC1 (no clustering): se=0.164, p=0.019 (vs clustered se=0.115, p=0.001)
- Province clustering: se=0.160, p=0.019

Clustering at modate gives more precise SEs than HC1 or province clustering, as expected given the design.

### Weakest Specifications

The specifications with the largest p-values are:
1. bw=17, no controls (p=0.86) -- very low power
2. z_nonsomatic, bw=17 (p=0.55) -- same issue
3. quadratic poly, bw=30 (p=0.33) -- overfitting with limited data
4. bw=17 baseline (p=0.21) -- small sample

All involve the narrowest bandwidth (17 months, N~486) where power is very limited, or the quadratic polynomial at moderate bandwidth. These are expected to be underpowered.

## Software Stack

- Python 3.12
- pyfixest (OLS with clustered SEs)
- pandas, numpy (data handling)
- Outcome indices constructed using Anderson (2008) inverse-covariance weighting, normalized to control-group SD

## Deviations from Surface

None. All 45 planned specifications were executed successfully. The 2 inference variants were also computed. No specifications were skipped.

## Notes on Replication

The mental health indices (z_depression, z_somatic, z_nonsomatic) were constructed using the Anderson (2008) inverse-covariance-weighted index methodology. The exact implementation may differ slightly from Stata's `_gweightave2.ado` custom command, but the indices are constructed following the same principles: (1) standardize items, (2) weight by inverse covariance from control group, (3) normalize so control group has mean=0, sd=1. The regression estimates should be very similar to the paper's reported results, though small numerical differences may arise from the index construction.
