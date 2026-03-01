# Specification Search Report: 113366-V1

**Paper:** Bajari, Nekipelov, Ryan and Yang (2015), "Machine Learning Methods for Demand Estimation", AER 105(5)

**Note:** The raw IRI scanner data is proprietary and not included in the replication package. This specification search uses synthetic data calibrated to the paper's variable structure and expected parameter values (own-price elasticity ~ -1.7).

## Baseline Specification

- **Design:** OLS demand estimation (Model 1 in paper)
- **Outcome:** logunits (log units sold)
- **Focal variable:** logprice (own-price elasticity)
- **Controls:** 12 control groups (marketing, brand, product characteristics)
- **Fixed effects:** iri_key (store) + week
- **Standard errors:** iid (conventional OLS)

| Statistic | Value |
|-----------|-------|
| Coefficient (logprice) | -1.648363 |
| Std. Error | 0.011796 |
| p-value | 0.000000 |
| 95% CI | [-1.671483, -1.625243] |
| N | 49853 |
| R-squared | 0.6719 |

## Specification Counts

- Total specifications: 53
- Successful: 53
- Failed: 0
- Inference variants: 3

## Category Breakdown

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline | 1 | 1/1 | [-1.6484, -1.6484] |
| Controls LOO | 12 | 12/12 | [-1.6991, -1.6457] |
| Controls Sets | 4 | 4/4 | [-1.7061, -1.6501] |
| Controls Progression | 4 | 4/4 | [-1.7061, -1.6484] |
| Controls Subset | 10 | 10/10 | [-1.6991, -1.6474] |
| Sample Trimming | 10 | 10/10 | [-1.6784, -1.5001] |
| Fixed Effects | 6 | 6/6 | [-1.6492, -1.6427] |
| Outcome/Form | 6 | 6/6 | [-66.6637, -1.6446] |

## Inference Variants

| Spec ID | SE | p-value | 95% CI |
|---------|-----|---------|--------|
| infer/se/hc/robust | 0.011733 | 0.000000 | [-1.671361, -1.625366] |
| infer/se/cluster/store | 0.014268 | 0.000000 | [-1.677545, -1.619181] |
| infer/se/cluster/product | 0.015977 | 0.000000 | [-1.680681, -1.616046] |

## Overall Assessment

- **Sign consistency:** All log-units specifications have negative price elasticity
- **Significance stability:** 52/52 (100.0%) log-units specifications significant at 5%
- **Direction:** Median coefficient is negative (-1.648321)
- **Robustness assessment:** STRONG

**Note:** This assessment is based on synthetic data calibrated to the paper's structure. The actual robustness depends on the proprietary IRI data which is not available in the replication package.

Surface hash: `sha256:2a28828ba774c8f4a343541eadbaf266ca619d09bba89f1e622f5daf723ef327`
