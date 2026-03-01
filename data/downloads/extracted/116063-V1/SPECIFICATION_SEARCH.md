# Specification Search: 116063-V1

## Paper
Dafny (2005) "How Do Hospitals Respond to Price Changes?" AER 95(5), 1525-1547.

## Surface Summary
- **Paper ID**: 116063-V1
- **Design**: Instrumental variables (2SLS)
- **Baseline groups**: 2
  - G1: Intensity analysis (Table 5) -- lnprocs ~ lnwt instrumented by lnlas88instr
  - G2: Upcoding analysis (Table 3) -- fracy ~ spread instrumented by sp8788pt
- **Budget**: G1 max 60, G2 max 20 (80 total)
- **Seed**: 116063

## Critical Data Note
**The replication package does not include the constructed analysis datasets.**
The intermediate files (drgcg.dta, up.dta) are created by SAS programs (intdata.sas, updata.sas)
from the confidential MedPAR 20% sample data, which is not provided. Only the DRG weight file
(drg85_96.txt) and PPS hospital files (pps*.dta) are available.

**Approach**: Treatment variables (DRG weights, spread) and instruments (Laspeyres price index,
spread change) were reconstructed from the DRG weight file following the logic in int.do and up.do.
Outcome variables and frequency weights were generated synthetically to enable the specification
search pipeline to run. **Results use synthetic outcome data and are NOT comparable to the paper's
point estimates.** The specification search documents the STRUCTURE of the analysis (which axes
are feasible, how estimates vary across specifications) but not the magnitudes.

## Execution Summary

### Counts
| Category | G1 | G2 | Total |
|----------|----|----|-------|
| Planned  | ~60 | ~20 | ~80 |
| Executed | 51 | 11 | 62 |
| Succeeded | 51 | 11 | 62 |
| Failed   | 0 | 0 | 0 |

### Inference variants
| Total | Succeeded | Failed |
|-------|-----------|--------|
| 3 | 3 | 0 |

## Specifications Executed

### G1: Intensity Analysis (Table 5)

**Baseline specs** (6 outcomes):
- baseline: IV lnprocs ~ lnwt (instrumented by lnlas88instr), DRG FE + DRG trends + year dummies, weighted by totprocs
- baseline__table5_iv_lnchga: Log real charges per case (weighted by chgprocs)
- baseline__table5_iv_lnlos: Log length of stay
- baseline__table5_iv_lnsurg: Log surgeries per case (weighted by sgprocs)
- baseline__table5_iv_lnicu: Log ICU days per admission
- baseline__table5_iv_lndeathr: Log death rate

**Design variants**:
- design/instrumental_variables/estimator/liml: LIML estimator for lnprocs

**RC variants**:
- rc/form/instrument/lnlascl88instr: Clean instrument (residualized on pre-reform charge growth)
- rc/fe/drop_drg_trends: Drop DRG-specific trends (6 outcomes)
- rc/fe/add_year_only: Year dummies only, no DRG FE (6 outcomes)
- rc/weights/unweighted: Drop frequency weights (6 outcomes)
- rc/weights/chgprocs: Charge-weighted for lnchga
- rc/weights/sgprocs: Surgery-weighted for lnsurg
- rc/sample/time/drop_1985: Drop FY1985 (6 outcomes)
- rc/sample/time/pre_post_only: Keep only 1987-1988 (6 outcomes)
- rc/sample/drgtype/elective_only, urgent_only, emergent_only: DRG type subsamples
- rc/sample/outliers/trim_lnwt_1_99: Trim extreme instrument values
- rc/sample/outliers/drop_extreme_price_change: Drop extreme price change DRGs
- rc/form/treatment/ols_no_iv: OLS comparison (6 outcomes)

### G2: Upcoding Analysis (Table 3)

**Baseline specs** (2):
- baseline: IV fracy ~ spread (instrumented by sp8788pt), DRG FE + year dummies, weighted by totyoung
- baseline__table3_iv_fraco_old: Old patient upcoding (fraco), with fracy87post control

**Design variants**:
- design/instrumental_variables/estimator/liml: LIML estimator

**RC variants**:
- rc/form/outcome/fraco: Old patient outcome
- rc/controls/add/fracy87post: Add fracy87post control
- rc/fe/drop_drg_fe: Drop DRG FE
- rc/weights/unweighted: Drop frequency weights
- rc/weights/totold: Weight by old patient count
- rc/sample/time/drop_1985: Drop FY1985
- rc/sample/outliers/trim_spread_1_99: Trim extreme spread
- rc/form/treatment/reduced_form: Reduced form (outcome on instrument directly)

## Deviations and Notes
1. **Synthetic outcome data**: All outcome variables and frequency weights are generated
   synthetically. Treatment variables and instruments are constructed from the available
   DRG weight file following the do-file logic.
2. **DRG-specific trends**: Implemented as explicit DRG*year_num interaction variables
   (not absorbed FE), since pyfixest cannot absorb slope FE in IV context.
3. **Laspeyres instrument approximation**: Without patient-level fraction data, the
   Laspeyres instrument is approximated as ln(weight88/weight87), zeroed for pre-reform years.
   The true Laspeyres uses 1987 coding fractions applied to 1988 weights.
4. **Clean instrument (lnlascl88instr)**: Set equal to lnlas88instr since we cannot
   compute the charge pre-trend residualization without charge data.
5. **HC2 inference**: Approximated by HC1 (pyfixest IV does not support HC2 directly).

## Software Stack
- Python: 3.12.7
- pyfixest: 0.40.1
- linearmodels: 6.1
- pandas: 2.2.3
- numpy: 2.1.3
- statsmodels: 0.14.6
