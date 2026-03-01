# Specification Search: 171681-V1

## Surface Summary
- **Paper**: Ambuehl, Bernheim & Lusardi, "Evaluating Deliberative Competence"
- **Design**: Randomized Experiment (online, MTurk)
- **Baseline Groups**: 2 (G1: compounding knowledge, G2: financial competence)
- **Surface hash**: sha256:041c4646c5a78637a0e4e376de7e0cc1d8b9ee062acf774a10779689e5e352bd
- **Seeds**: 171681 (G1), 171682 (G2)

## Execution Summary

| Metric | G1 | G2 | Total |
|--------|----|----|-------|
| Planned | 29 | 40 | 69 |
| Succeeded | 29 | 40 | 69 |
| Failed | 0 | 0 | 0 |

Inference variants: 8 rows (HC1, HC3 on baselines)

## Software Stack
- Python 3.12.7
- pyfixest, statsmodels, scipy, pandas, numpy

## Notes
1. Data management replicated from raw Qualtrics CSV (8 Exp A + 2 Exp B batches). Includes MPL switch-point extraction (coarse + fine), multi-switcher flagging, treatment renaming/reshaping, midpoint adjustment, discount rate construction, and all demographic variable construction.
2. G2 baselines and many G2 variants use no-constant OLS with all treatment arm dummies. The focal coefficient is the Wald test Full == Control.
3. `rc/sample/attrition/include_multi_switchers__requires_remanagement` SKIPPED: requires re-running the full pipeline without multi==0 exclusion.
4. Constructed outcomes finCompCorr and finCompCorrSq built inline following the paper's formulas.
5. Minor numerical differences from published tables are expected due to floating-point differences between Python and Stata in the complex MPL extraction pipeline.
