# Specification Search: 138401-V1

## Surface Summary

- **Paper**: Measles, the MMR Vaccine, and Adult Labor Market Outcomes
- **Design**: Continuous difference-in-differences (TWFE)
- **Baseline groups**: 1 (G1)
- **Focal outcome**: ln_cpi_income (log CPI-adjusted wage income)
- **Treatment**: M12_exp_rate (12-year pre-vaccine measles rate x exposure / 100,000)
- **Fixed effects**: bpl, birthyr, year, ageblackfemale, bpl_black, bpl_female, bpl_black_female
- **Clustering**: bplcohort (birth state x birth year)
- **Budget**: 80 max core specs
- **Seed**: 138401

## Execution Summary

### Specification Results (specification_results.csv)
- **Total specs planned**: 49
- **Successful**: 46
- **Failed**: 3

| Category | Count |
|---|---|
| Baseline specs (6 outcomes) | 6 |
| Design (TWFE) | 1 |
| RC/controls/loo | 2 |
| RC/controls/sets | 1 |
| RC/sample/restriction | 12 |
| RC/fe/add | 4 |
| RC/fe/drop | 4 |
| RC/fe/simplify | 1 |
| RC/data/treatment_construction (M2-M11) | 10 |
| RC/form/outcome | 2 |
| RC/joint/sample_and_fe | 6 |
| **Total** | **49** |

### Inference Results (inference_results.csv)
- **Total inference variants**: 10
- **Successful**: 10
- **Failed**: 0

Inference variants run on the focal baseline (ln_cpi_income):
- bpl (birth state)
- bplexposure (birth state x exposure)
- bpl_region4 (Census region, 4 clusters)
- bpl_region9 (Census division, 9 clusters)
- stateexposure (state of residence x exposure)
- statecohort (state of residence x birth year)
- statefip (state of residence)
- birthyr (birth year)
- exposure (exposure level)
- HC1 (heteroskedasticity-robust, no clustering)

## Deviations and Notes

- Data cleaning replicates acs_cleaning.do: age 26-59 (age>25 & age<60), native-born (bpl<57), black/white only
- All ACS years 2000-2017 pooled
- The Stata code uses `reg ... , robust cluster()` which is equivalent to CRV1 clustering in pyfixest
- Birth-state-specific linear cohort trends implemented via pyfixest `i(bpl, cohort)` interaction syntax
- Mean reversion control: pre-cohort average of ln_cpi_income x precohort indicator, following Table 4 Stata code
- Demographic subsample specs (men_only, women_only, white_only, black_only, joint white_men/black_men/white_women) correctly drop collinear demographic FE interactions
- Treatment construction variants M2-M11 use different pre-vaccine averaging windows (Appendix Table 2)
- OPTIMIZATION: selective column loading (12 of 30 columns), chunked row filtering during load
- Full dataset used (no subsampling)
- Per-specification timeout of 600s to avoid runaway computations

## Software Stack
- Python 3.12.7
- pyfixest: 0.40.1
- pandas: 2.2.3
- numpy: 2.1.3

## Failed Specifications

- **baseline__cpi_incwage** (138401-V1_run_002): CPUDispatcher(<function demean at 0x10eaaa980>) returned a result with an exception set
- **rc/fe/add/bpl_cohort_trend** (138401-V1_run_025): CPUDispatcher(<function demean at 0x10eaaa980>) returned a result with an exception set
- **rc/form/outcome/level_income** (138401-V1_run_042): CPUDispatcher(<function demean at 0x10eaaa980>) returned a result with an exception set
