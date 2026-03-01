# Specification Search: 113561-V1

## Paper
"What Determines Giving to Hurricane Katrina Victims? Experimental Evidence on Racial Group Loyalty"
Fong & Luttmer, AEJ: Applied Economics, 2009

## Surface Summary
- **Baseline groups**: 1 (G1: effect of picshowblack on giving)
- **Design**: randomized_experiment
- **Budget**: 60 specs max
- **Sampling**: full enumeration (no random sampling needed)
- **Seed**: 113561

## Execution Summary

| Category | Planned | Executed | Succeeded | Failed |
|----------|---------|----------|-----------|--------|
| Baseline | 2 | 2 | 2 | 0 |
| Design | 1 | 1 | 1 | 0 |
| RC: Controls (sets) | 5 | 5 | 5 | 0 |
| RC: Controls (LOO) | 20 | 20 | 20 | 0 |
| RC: Controls (progression) | 5 | 5 | 5 | 0 |
| RC: Controls (subsets) | 6 | 6 | 6 | 0 |
| RC: Sample | 8 | 8 | 8 | 0 |
| RC: Functional form | 3 | 3 | 3 | 0 |
| RC: Weights | 2 | 2 | 2 | 0 |
| **Total** | **52** | **52** | **52** | **0** |

## Inference Variants (separate table)
- 3 inference variants computed for baseline spec (classical, HC2, HC3)
- Written to `inference_results.csv`

## Key Findings

### Baseline result
- **Coefficient**: -2.30 (SE = 3.85, p = 0.550, N = 1343)
- The baseline effect of showing black victims on giving is negative but statistically insignificant

### Robustness across specifications
- All 52 specifications produce negative point estimates (range: -7.37 to -0.22)
- No specification achieves conventional significance (p < 0.05) except:
  - None of the 52 specs is significant at the 5% level
- The result is remarkably stable across control sets: LOO coefficients range from -2.42 to -1.87
- Sample restrictions produce more variation: Slidell-only gives -0.64, Biloxi-only gives -3.61
- Dropping extreme choices (0 and 100) strengthens the effect to -5.04 (p = 0.164)
- Excluding fast completers gives -7.37 (p = 0.196)
- Functional form changes (log, asinh, binary) all show negligible effects near zero

### Interpretation
The main finding (all respondents) shows a weak, insignificant negative effect of showing black victims. This is consistent with the paper's Table 3 Col 2 result. The paper's main contribution is demonstrating that this effect is moderated by racial identification (Table 6), which we do not test in the core specification search (those are heterogeneity/interaction analyses that change the estimand).

## Software Stack
- Python 3.12
- pyfixest 0.40+
- pandas 2.x
- numpy 1.x
- scipy 1.10+

## Deviations from Surface
- None. All planned specs were executed successfully.
- `rc/sample/outliers/trim_y_1_99` had no effect because giving is bounded [0,100] with no values outside the 1-99 percentile range (min=0, max=100).
