# Verification Report: 194841-V1

## Paper Information
- **Paper**: Rational Inattention and the Business Cycle Effects of Productivity and News Shocks
- **Authors**: Bartosz Mackowiak and Mirko Wiederholt
- **Journal**: American Economic Review
- **Paper ID**: 194841-V1

## Baseline Groups

### G1: Coibion-Gorodnichenko Under-Reaction Test
- **Claim**: Forecast revisions positively predict forecast errors (beta > 0)
- **Baseline spec_ids**: baseline
- **Outcome**: log_forecast_error
- **Treatment**: forecast_revision
- **Expected sign**: Positive (+)
- **Baseline result**: Beta = 0.757, SE = 0.299, p = 0.012, n = 157

---

## Summary Counts

| Category | Count |
|----------|-------|
| Total specifications | 70 |
| Baseline | 1 |
| Core tests (incl. baseline) | 55 |
| Non-core tests | 15 |
| Invalid | 0 |
| Unclear | 0 |

### Core Test Breakdown
| Category | Count |
|----------|-------|
| core_sample | 30 |
| core_method | 8 |
| core_controls | 7 |
| core_inference | 6 |
| core_funcform | 4 |

### Non-Core Test Breakdown
| Category | Count |
|----------|-------|
| noncore_heterogeneity | 6 |
| noncore_alt_outcome | 5 |
| noncore_placebo | 4 |

---

## Classification Rationale

### Core Tests (55 specs)

**Sample restrictions (30)**: Subsample analyses, outlier treatments, recession splits. All use same outcome/treatment.

Note: Two duplicate pairs exist:
- early_period = subsample1_early (coef 0.480)
- late_period = subsample2_late (coef 1.210)

**Estimation methods (8)**: WLS, M-estimation, quantile, first differences, GLS AR(1). FD and GLS reverse sign.

**Controls (7)**: Time trends, lagged FE/FR, recession indicator, full model. Lagged FE and full model reverse sign.

**Inference (6)**: HC1-3, HAC(4,8), bootstrap. Identical coefficient (0.757), all significant at 5%.

**Functional form (4)**: Levels, growth rates, quadratic, standardized. Growth rate reverses sign.

### Non-Core Tests (15 specs)

**Heterogeneity (6)**: Interaction terms change coefficient interpretation. Tests moderation not claim.

**Placebo (4)**: Different treatment variables (random, lagged, lead FR). Diagnostics by definition.

**Alt outcomes (5)**: Absolute, squared, sign, binary error. Fundamentally change estimand.

---

## Top 5 Most Suspicious Rows

1. early_period/subsample1_early: Exact duplicates (coef=0.480)
2. late_period/subsample2_late: Exact duplicates (coef=1.210)
3. growth_rates: Sign reversal (beta=-0.394). Core with confidence 0.75.
4. lead_1q_forecast_revision: Mechanically expected strong effect (beta=1.57)
5. first_differences: Sign reversal (beta=-0.308) suggesting persistent trends

---

## Recommendations

1. Deduplicate subsample analyses
2. Report interaction terms for heterogeneity specs
3. Flag lead placebos as mechanical
4. Tag sign-reversing specs

---

## Quality Checks

- All 70 spec_ids covered exactly once in verification_spec_map.csv
- baseline_group_id G1 in CSV matches verification_baselines.json
- No invalid or unclear specs
- Conservative: heterogeneity classified as non-core
