# Verification Report: 125821-V1

## Paper Information
- **Topic**: Effects of school operating referendum elections on district expenditures in Wisconsin
- **Method**: Regression Discontinuity Design
- **Total Specifications**: 62

## Baseline Groups

### G1: Total Expenditures per Member
- **Claim**: Passing an operating referendum increases school district expenditures per member.
- **Baseline spec**: `baseline`
- **Expected sign**: Positive
- **Baseline coefficient**: 641.24 (SE: 344.34, p = 0.063)
- **Outcome**: `tot_exp_mem`
- **Treatment**: `treatment` (referendum passage indicator)
- **N**: 4,240

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baseline)** | **40** | |
| core_controls | 9 | Baseline + control set variations + leave-one-out controls |
| core_sample | 17 | Bandwidth variations, donut holes, sample restrictions |
| core_inference | 2 | Clustering variations |
| core_funcform | 5 | Polynomial order + functional form (log, IHS) |
| core_method | 3 | Kernel function variations |
| **Non-core tests** | **22** | |
| noncore_alt_outcome | 8 | Alternative expenditure outcomes (instructional, compensation, dropout, salary, support services, debt, student-teacher ratio) |
| noncore_placebo | 4 | Placebo cutoffs at 40%, 45%, 55%, 60% |
| noncore_heterogeneity | 3 | Urban, large district, high poverty interactions |
| **Total** | **62** | |

## Robustness Assessment

The main finding has **MODERATE** support. The baseline result is only marginally significant (p=0.063). Key observations:
- 79% of specs show positive coefficients
- Only 29% achieve significance at 5%
- Donut hole specs strengthen results (3/4 significant)
- Placebo cutoff tests generally pass (no significant discontinuities at false cutoffs)
- Bandwidth sensitivity: smaller bandwidths yield insignificant results; medium bandwidths (7-15pp) often significant
- Alternative outcomes show mixed results

## Duplicates
- `rd/bandwidth/bw10` is identical to `baseline`
- Kernel specs (triangular, uniform, epanechnikov) produce identical results at same bandwidth
- Some polynomial specs duplicate bandwidth specs
