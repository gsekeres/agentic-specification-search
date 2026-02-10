# Verification Report: 135401-V1

## Paper Information
- **Title**: Breastfeeding and Child Development
- **Authors**: Del Bono, Giardili
- **Journal**: American Economic Journal: Applied Economics, 14(3): 329-66, 2022
- **DOI**: 10.1257/app.20180385

## Baseline Groups

### G1: Cognitive development (headline result)
- **Claim**: Breastfeeding >= 90 days causally increases child cognitive development (summary index), estimated via Wooldridge NTSLS with weekend exposure instrument.
- **Baseline spec_id**: 7
- **Coefficient**: 0.464 (SE 0.179, p=0.010)
- **Expected sign**: Positive

### G2: Non-cognitive development (secondary result)
- **Claim**: Breastfeeding >= 90 days has a positive effect on child non-cognitive development (summary index).
- **Baseline spec_id**: 8
- **Coefficient**: 0.319 (SE 0.224, p=0.154)
- **Expected sign**: Positive

### G3: Health outcomes (null result)
- **Claim**: Breastfeeding >= 90 days has no significant effect on the child health summary index.
- **Baseline spec_id**: 9
- **Coefficient**: 0.009 (SE 0.082, p=0.913)
- **Expected sign**: Zero / null

## Counts Summary

| Category | Count |
|----------|-------|
| **Total specifications** | **194** |
| Baselines | 3 |
| Core tests (including baselines) | 44 |
| Non-core tests | 149 |
| Invalid | 1 |
| Unclear | 0 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_method | 19 | Alternative estimation methods (OLS, 2SLS, NTSLS) and instruments (exposure vs polynomial) on summary indices |
| core_funcform | 18 | Alternative breastfeeding duration thresholds (bfd30, bfd60, bfd120) on summary indices |
| core_sample | 4 | Robustness sample restrictions from Table F7 (exclude induced, include emergency C-sections, imputed scores, benchmark duplicate) |
| core_controls | 2 | Additional controls from Table F7 (cubic polynomial in hour, hour-of-birth dummies) |
| core_fe | 1 | No hospital fixed effects (Table F7) |
| noncore_alt_outcome | 149 | Individual cognitive, non-cognitive, and health components (not summary indices) |
| invalid | 1 | Spec 105 (w2_obesity OLS) has missing coefficient |

## Core Tests by Baseline Group

| Baseline Group | Core specs (incl. baseline) |
|---------------|---------------------------|
| G1 (cog_index) | 22 |
| G2 (non_cog_index) | 12 |
| G3 (health_index) | 10 |

## Top 5 Suspicious / Noteworthy Rows

1. **Spec 105 (w2_obesity, OLS)**: Missing coefficient -- the CSV has an empty coefficient field and p_value=1.0. Marked invalid. Likely a Stata output extraction issue where the coefficient was suppressed or missing.

2. **Specs 10-11 (cog_index, OLS, polynomial source)**: These are apparent duplicates of spec 1. Spec 10 and spec 11 have identical values (OLS, cog_index, bfd90, coefficient=0.057, SE=0.019). The duplication likely arises from extracting the same OLS result from the polynomial source file when it was already captured from the exposure source file.

3. **Specs 162-185 (cognitive appendix duplicates)**: Specs 162-185 from `components_cognitive_appendix.xls` contain many exact duplicates of specs 45-59 from `components_cognitive_toppanel.xls`. These appear to be the same regressions reported in both the main table and appendix of the paper, extracted twice from different output files.

4. **Spec 38 (Benchmark NTSLS robustness)**: This is an exact duplicate of spec 7 (the baseline). The robustness table (Table F7) begins by reporting the benchmark result, which is identical to the baseline. It has the same coefficient (0.464), SE (0.179), and sample size (5015).

5. **Spec 12 (non_cog_index, OLS, polynomial source)**: Initially assigned to G1 due to the source file grouping, but outcome is non_cog_index which belongs to G2. Corrected to G2 in the final verification_spec_map.csv.

## Assessment of Specification Search Quality

The specification search successfully extracted 194 specifications from the Stata output files. The key observations are:

**Strengths**:
- The baseline specification (spec 7) is correctly identified and matches Table 5 of the paper.
- The search covers all five dimensions of variation described in the paper: outcomes, methods, treatment thresholds, sample restrictions, and instrument choices.
- The robustness checks from Table F7 are all captured.

**Weaknesses**:
- The search is heavily weighted toward individual components (149 of 194 specs) rather than the summary indices that represent the paper's actual claims. This inflates the total count without adding many core tests.
- There are numerous duplicates (specs 10-11 duplicate spec 1; specs 162-185 duplicate specs 45-59) from extracting the same results from multiple output files.
- Only 7 robustness checks from Table F7 are captured for the cognitive index (G1). Non-cognitive and health robustness checks from Table F7 are not included, likely because the paper only reports Table F7 for the cognitive index.

## Recommendations

1. **Deduplication**: A post-processing step should detect and flag duplicate specifications (identical outcome, treatment, method, coefficient, SE, and sample size). This would reduce the effective count by approximately 20 specs.

2. **Focus on summary indices**: Future specification searches for this paper type should prioritize summary index outcomes over individual components, or at minimum tag components as secondary from the outset.

3. **Robustness for G2/G3**: The paper reports fewer robustness checks for non-cognitive and health outcomes. The search correctly reflects this limitation of the original paper.

4. **No baseline flag for G2/G3**: Only spec 7 is marked `is_baseline=True` in the original CSV. Specs 8 and 9 (the G2 and G3 baselines) should arguably also be flagged as baselines in the source data, since they represent the paper's canonical results for those outcome domains.
