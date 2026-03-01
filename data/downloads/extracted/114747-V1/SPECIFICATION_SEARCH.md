# Specification Search Report: 114747-V1

**Paper:** Dranove, Hughes, and Meltzer (2003/2012), "Incentives and Promotion for Adverse Drug Reactions", AEJ: Economic Policy

## Data Limitations

The provided .dta file is missing key treatment variables (promotion/advertising expenditures: q1totalexp, q2q4totalexp, etc.) and many controls (generic, sh_count* demographics, age-gender shares, permonths exposure). This means **G1 (Poisson ADR ~ promotion) cannot be executed**. Analysis focuses on **G2 (FDA labeling changes ~ ADR counts)** using available variables.

## Baseline Specification (G2)

- **Design:** Logit / LPM (Linear Probability Model)
- **Outcome:** any_fda_reaction (binary: any FDA labeling change)
- **Treatment:** v1-v4 (veryserious ADR count interacted with condition dummies)
- **Focal coefficient:** v3 (arthritis condition, matching Table 3 focus)
- **Controls:** c1-c3 (condition dummies) + Dappr_cats_1-4 (drug approval age)
- **Fixed effects:** Year-month dummies
- **Note:** Missing controls (generic, $char, $age) reduce precision vs. paper's specification

| Statistic | Value |
|-----------|-------|
| Coefficient (v3) | 0.004220 |
| Std. Error | 0.001124 |
| p-value | 0.000175 |
| 95% CI | [0.002016, 0.006424] |
| N | 1274 |
| Pseudo R-squared | 0.1383 |

## Specification Counts

- Total specifications: 66
- Successful: 66
- Failed: 0
- Inference variants: 4

## Category Breakdown

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline (logit) | 4 | 2/4 | [-0.008565, 0.004220] |
| Design/Estimator | 17 | 5/17 | [-0.001120, 0.020824] |
| Controls LOO | 4 | 0/4 | [0.000326, 0.000396] |
| Controls Sets | 4 | 0/4 | [0.000299, 0.000398] |
| Controls Block | 2 | 0/2 | [0.000299, 0.000398] |
| Controls Subset | 8 | 0/8 | [0.000313, 0.000381] |
| Controls Logit | 2 | 2/2 | [0.004907, 0.005552] |
| Sample Restrictions | 8 | 1/8 | [0.000008, 0.014246] |
| Functional Form | 12 | 3/12 | [0.000008, 0.767129] |
| Fixed Effects | 5 | 0/5 | [-0.000080, 0.000330] |

## Inference Variants

| Spec ID | SE | p-value | 95% CI |
|---------|-----|---------|--------|
| infer/se/robust/sandwich | 0.000219 | 0.136690 | [-0.000103, 0.000754] |
| infer/se/cluster/drug | 0.000054 | 0.000000 | [0.000217, 0.000434] |
| infer/se/cluster/condition | 0.000066 | 0.015863 | [0.000116, 0.000535] |
| infer/se/iid | 0.000126 | 0.010003 | [0.000078, 0.000573] |

## Overall Assessment

- **Sign consistency:** Mixed signs across specifications
- **Significance stability:** 13/66 (19.7%) specifications significant at 5%
- **Direction:** Median coefficient is positive (0.000326)
- **Robustness assessment:** FRAGILE

**Note:** This assessment is limited by missing data. The paper's G1 (promotion -> ADR) specifications cannot be run because promotion expenditure variables are not in the provided dataset. G2 results (ADR -> FDA labeling) are estimated with fewer controls than the paper, which may affect inference.

Surface hash: `sha256:4269f414b754908305ca8055844c183564fc8748c22f7b4f59071f5d4df4780a`
