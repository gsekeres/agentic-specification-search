# Specification Search Report: 113109-V1

**Paper:** Charles, Hurst, & Notowidigdo (2018), "Housing Booms and Busts, Labor Market Outcomes, and College Attendance", AER

## Design

- **Method:** Instrumental Variables (2SLS)
- **Instrument:** Saiz housing supply elasticity
- **Treatment (G1):** hp_growth_real_00_06 (deltaP + units_growth)
- **Treatment (G2):** housing_demand_shock (same construction)
- **Controls:** college_share_2000, female_employed_share_2000, pop_prev, share_foreign_18_55_2000
- **Clustering:** statefip

## G1: Employment (Table 1)

### Baseline

| Statistic | Value |
|-----------|-------|
| Outcome | d_emp_18_25_le |
| Coefficient | 0.048465 |
| Std. Error | 0.016969 |
| p-value | 0.006366 |
| 95% CI | [0.014329, 0.082602] |
| N | 275 |
| R-squared | nan |
| Weight | msa_total_emp_all_2000 |

## G2: College Enrollment (Table 3)

### baseline__table3_iv_any_college_18_25

| Statistic | Value |
|-----------|-------|
| Outcome | d_any_18_25_a1 |
| Coefficient | -0.020007 |
| Std. Error | 0.006486 |
| p-value | 0.003408 |
| 95% CI | [-0.033054, -0.006959] |
| N | 275 |
| Weight | pop (exp(pop_prev)) |

### baseline__table3_bachelor_18_25

| Statistic | Value |
|-----------|-------|
| Outcome | d_bachelor_18_25_a1 |
| Coefficient | 0.003201 |
| Std. Error | 0.003532 |
| p-value | 0.369485 |
| 95% CI | [-0.003905, 0.010307] |
| N | 275 |
| Weight | pop (exp(pop_prev)) |

## Specification Counts

- Total specifications: 86
- Successful: 86
- Failed: 0
- Inference variants: 3

### G1 Category Breakdown

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline | 1 | 1/1 | [0.0485, 0.0485] |
| Design (LIML) | 1 | 1/1 | [0.0485, 0.0485] |
| Controls LOO | 4 | 4/4 | [0.0434, 0.0522] |
| Controls Sets | 8 | 8/8 | [0.0388, 0.0493] |
| Controls Subset | 20 | 20/20 | [0.0400, 0.0553] |
| Sample | 5 | 5/5 | [0.0365, 0.0494] |
| Treatment/Outcome | 2 | 2/2 | [0.0200, 0.0588] |
| Instrument | 4 | 4/4 | [0.0463, 0.0645] |
| Weights | 2 | 2/2 | [0.0290, 0.0485] |
| Joint | 5 | 5/5 | [0.0446, 0.0500] |

### G2 Category Breakdown

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline | 2 | 1/2 | [-0.0200, 0.0032] |
| Design (LIML) | 1 | 1/1 | [-0.0200, -0.0200] |
| Controls LOO | 4 | 4/4 | [-0.0201, -0.0120] |
| Controls Sets | 1 | 1/1 | [-0.0114, -0.0114] |
| Controls Subset | 15 | 15/15 | [-0.0218, -0.0130] |
| Sample | 2 | 2/2 | [-0.0212, -0.0189] |
| Treatment/Outcome | 3 | 1/3 | [-0.0232, 0.0124] |
| Instrument | 4 | 4/4 | [-0.0300, -0.0189] |
| Weights | 1 | 0/1 | [-0.0129, -0.0129] |
| Joint | 1 | 1/1 | [-0.0131, -0.0131] |

## Inference Variants

| Group | Spec ID | SE | p-value | 95% CI |
|-------|---------|-----|---------|--------|
| G1 | infer/se/hc/hc1 | 0.013089 | 0.000259 | [0.022695, 0.074235] |
| G1 | infer/se/hc/hc2 | FAILED | - | - |
| G2 | infer/se/hc/hc1 | 0.007494 | 0.008054 | [-0.034761, -0.005252] |

## Overall Assessment

### G1 (Employment)
- **Sign consistency:** All specifications have the same sign
- **Significance stability:** 52/52 (100.0%) significant at 5%
- **Direction:** Median coefficient is positive (0.046839)
- **Robustness assessment:** STRONG

### G2 (Education)
- **Sign consistency:** Mixed signs across specifications
- **Significance stability:** 30/34 (88.2%) significant at 5%
- **Direction:** Median coefficient is negative (-0.016902)
- **Robustness assessment:** WEAK

Surface hash: `sha256:1c7d2d191aa3ba70c2cf8a145c9cd29620bee46b399a4d9870ff717759705e78`
