# Verification Report: 202864-V1

## Paper Information
- **Title**: Eliciting Ambiguity with Mixing Bets
- **Authors**: Schmidt
- **Journal**: AER: Insights
- **Total Specifications**: 64

## Classification Summary

| Category | Count |
|----------|-------|
| **Core tests (incl. baselines)** | **42** |
| core_sample | 16 |
| core_controls | 14 |
| core_inference | 5 |
| core_funcform | 3 |
| core_method | 2 |
| **Non-core tests** | **24** |
| noncore_heterogeneity | 9 |
| noncore_alt_outcome | 8 |
| noncore_placebo | 7 |
| **Total** | **64** |

## Robustness Assessment

**MODERATE** support for the main hypothesis.

The main finding -- more mixing in ambiguous than risky domains -- is robust and highly significant (p < 0.001). Individual-level predictors (att_std) are directionally consistent (78% positive) but rarely significant due to small sample (N=88). Results are consistent across OLS, logit, and probit models. Domain comparisons survive multiple testing corrections.
