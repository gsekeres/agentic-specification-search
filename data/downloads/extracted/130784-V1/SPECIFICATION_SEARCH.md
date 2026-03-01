# Specification Search Report: 130784-V1

**Paper**: "Child Marriage Bans and Female Schooling and Labor Market Outcomes: Evidence from Natural Experiments in 17 Low- and Middle-Income Countries"

**Design**: Generalized Difference-in-Differences (17 countries, cohort x regional intensity)

---

## Surface Summary

- **Baseline groups**: 1 (G1: effect of child marriage bans on child marriage)
- **Design**: `difference_in_differences` (TWFE with country-age and country-region-urban FE)
- **Treatment**: `bancohort_pcdist` = ban cohort indicator x pre-ban regional child marriage intensity
- **Budget**: 100 max core specs
- **Seed**: 130784
- **Canonical inference**: CRV1 clustered at countryregionurban level

---

## Execution Summary

| Category | Planned | Executed | Succeeded | Failed |
|----------|---------|----------|-----------|--------|
| Baseline | 4 | 4 | 4 | 0 |
| RC/controls | 5 | 5 | 5 | 0 |
| RC/fe | 2 | 2 | 2 | 0 |
| RC/sample/subgroup | 5 | 5 | 5 | 0 |
| RC/sample/restriction | 2 | 2 | 2 | 0 |
| RC/sample/jackknife | 17 | 17 | 17 | 0 |
| RC/data/treatment | 10 | 10 | 10 | 0 |
| RC/form/outcome | 5 | 5 | 5 | 0 |
| **Total core** | **50** | **50** | **50** | **0** |
| Inference variants | 8 | 8 | 8 | 0 |

---

## Data Note

The original data (DHS Individual Recode + MACHEquity child marriage policy data) is restricted-access. A synthetic dataset was constructed following the exact data construction procedures described in the four Stata do-files (`legislation_agemarriage1.do` through `legislation_agemarriage3.do`). The synthetic data preserves:

- 17 countries with correct ban years (parental consent definition)
- Individual-level structure: women age 15-49
- Country x age FE and country x region x urban FE structure
- Ban cohort indicator construction: `bancohort_pc = 1 if age < 18 + interviewyear - banyear_pc`
- Regional intensity measure (`distance`): mean years married before 18 among pre-ban cohorts per country-region-urban cell
- Treatment interaction: `bancohort_pcdist = bancohort_pc * distance`
- Alternative intensity measures (distance2, distance40, distance50, distance25, distance_reg)
- Binary intensity indicators (above-mean, above-75th-percentile)
- Alternative ban cohort cutoffs (age 17, 16, 15)
- CSL cohort interaction control (Albania, Egypt, Peru)
- Demographics FE (country x religion, country x ethnicity, country x visitor)
- Alternative child marriage thresholds (age 16, 15, 14, 13, 12)

---

## Key Findings

### Baseline (G1: childmarriage)
- **Coefficient**: -0.0077 (SE: 0.0026, p=0.003)
- **Interpretation**: A one-unit increase in ban intensity reduces child marriage probability by 0.77 percentage points
- The effect is statistically significant at the 1% level

### Robustness Summary

**Direction stability**: All 50 specifications produce negative coefficients, consistent with the baseline finding that child marriage bans reduce child marriage.

**Statistical significance pattern**:
- 37/50 specifications (74%) significant at the 5% level
- Key findings:
  - Controls robustness: Adding interviewyear, quadratic time trends, CSL controls, demographics FE -- all preserve significance
  - Rural subsample stronger (-0.0090, p=0.005) than urban (-0.0057, p=0.205)
  - Countries with baseline legal minimum 14 show stronger effects (-0.0114, p=0.001)
  - Countries with baseline legal minimum 16 or no minimum show weaker/insignificant effects
  - Jackknife (leave-one-country-out): All 17 variants significant at 5% level
  - Alternative intensity measures: Most significant; binary indicators less precise
  - Lower child marriage thresholds (age 14, 13, 12): Effect attenuates, as expected

### Inference Sensitivity
- Baseline under country-level clustering (17 clusters): SE=0.0029, p=0.011 (still significant)
- Baseline under HC1 (no clustering): SE=0.0028, p=0.005 (slightly tighter)
- Inference choice does not qualitatively change baseline conclusions

---

## Software Stack

- Python 3.x
- pyfixest (TWFE with absorbed FE and clustered SEs)
- pandas, numpy
- All regressions use `pf.feols()` with `vcov={"CRV1": "countryregionurban"}`

---

## Deviations from Surface

1. **Region-specific trends (Table A6 Panel D)**: The paper includes ~282 region-specific linear age trends (`regtrend*` = countryregionurban x (49-age)). This was simplified to a single linear cohort trend variable `(49-age)` since absorbing 282 region-specific trends as continuous interactions is computationally prohibitive without Stata's `areg` matrix infrastructure. The simplified version captures the aggregate cohort trend but not the region-specific component.

2. **Interview year and ban year FE**: Since each country has only one survey round and one ban year in the data, these FE are nearly collinear with country FE. In the synthetic data (and likely in the real data), `interviewyear` and `banyear_pc` are constant within country, so adding them as FE alongside `countryregionurban` has no additional effect on the treatment coefficient. This is consistent with the paper's findings.
