# Verification Report: 113561-V1

## Paper
"What Determines Giving to Hurricane Katrina Victims? Experimental Evidence on Racial Group Loyalty"
Fong & Luttmer, AEJ: Applied Economics, 2009

## Baseline Groups

### G1: Effect of picshowblack on giving (all respondents)
- **Baseline spec_run_ids**: S001 (Table 3 Col 2), S002 (Table 4 Panel 1 with nraudworthy)
- **Baseline coefficient**: -2.30 (SE=3.85, p=0.55, N=1343)
- **Expected sign**: negative (weakly)
- **Statistical significance**: Not significant at any conventional level

## Summary Counts

| Category | Count |
|----------|-------|
| Total rows | 52 |
| Core-eligible | 52 |
| Non-core | 0 |
| Invalid | 0 |
| Unclear | 0 |

### By category
| Category | Count |
|----------|-------|
| core_method | 3 |
| core_controls | 31 |
| core_sample | 8 |
| core_funcform | 3 |
| core_weights | 2 |
| Baselines | 2 |
| Additional baselines | 1 |

## Classification Details

All 52 specifications preserve the baseline claim object:
- Same outcome concept (giving, or monotone transforms thereof)
- Same treatment (picshowblack)
- Same estimand (ITT under random assignment)
- Same population (all respondents, with sample-restriction variants)

### Functional form variants
- S040 (log1p), S041 (asinh), S042 (binary): These transform the outcome but preserve the concept of "charitable giving to Katrina victims." The coefficient interpretation changes (semi-elasticity vs level vs extensive margin), but the claim object is preserved.

### Sample restriction variants
- S034 (main survey), S035-S036 (city subsamples), S037 (race-shown): These restrict the sample but the target population concept (all respondents) is unchanged since these are data quality/design robustness checks, not population changes.
- S038 (trimming), S039 (drop extreme): Standard outlier handling.
- S051-S052 (engaged respondents, not-fast completers): Quality filters.

## Issues Found

1. **No issues with outcome/treatment drift**: All 52 rows use the correct outcome (giving or transforms) and treatment (picshowblack).
2. **No issues with estimand drift**: All rows estimate the ITT effect.
3. **No invalid extractions**: All 52 runs succeeded.
4. **Trimming spec (S038) ineffective**: The 1/99 percentile trimming of giving had no effect because giving is bounded [0,100] by design.

## Key Findings Across Specifications

The baseline result (picshowblack on giving, all respondents) is:
- **Consistently negative** across all 52 specifications (range: -7.37 to -0.22)
- **Consistently insignificant** at the 5% level in all specifications
- **Very stable across control variations**: LOO estimates range from -2.42 to -1.87
- **More variable across sample restrictions**: City subsamples show -0.64 (Slidell) to -3.61 (Biloxi)
- **Strongest when restricting to interior choices**: -5.04 when dropping giving=0 and giving=100

## Recommendations

1. The specification search confirms the paper's finding: picshowblack effect on giving is not significant for the full sample.
2. The paper's main contribution (Table 6 heterogeneity by racial attitudes) was intentionally excluded from the core specification search.
3. No changes to the surface are recommended.
