# Specification Surface: 116063-V1

## Paper Overview
- **Title**: How Do Hospitals Respond to Price Changes? (Dafny, 2005 AER)
- **Design**: Instrumental variables
- **Data**: DRG-year panel from Medicare MedPAR 20% sample (FY1985-1991), collapsed at the DRG-pair level. DRG weights from the Federal Register.
- **Key finding**: Hospitals respond to exogenous price increases by increasing the volume and intensity of services provided. Additionally, hospitals upcode patients into higher-paying DRG categories when the financial incentive (price spread) increases.

## Baseline Groups

### G1: Intensity Analysis -- Effect of DRG Price on Treatment Volume (Table 5)

**Claim object**: The elasticity of hospital treatment volume (log procedures) with respect to DRG price (log weight), identified via the 1988 Medicare PPS reform that mechanically changed DRG weights through a Laspeyres price index.

**Baseline specification** (Table 5, IV for lnprocs):
- Second stage: `lnprocs ~ lnwt + year_dummies + drg_dummies + drg_trends [w=totprocs], robust`
- First stage: `lnwt ~ lnlas88instr + year_dummies + drg_dummies + drg_trends [w=totprocs], robust`
- Instrument: `lnlas88instr` = log Laspeyres price index (mechanical price based on 1987 coding patterns and new 1988 weights), zeroed out for pre-reform years
- Panel unit: DRG (pair-level, ~90 pairs)
- Time: FY1985-1991 (7 years)
- FE: DRG dummies (`drg_*`) and DRG-specific linear year trends (`drgXyea_*`)
- Weights: Frequency weights (total procedures per DRG-year)
- SE: Heteroskedasticity-robust

**Additional baseline specs** (same IV structure, different outcomes from Table 5):
- `baseline__table5_iv_lnchga`: Log real charges per case (weighted by chgprocs)
- `baseline__table5_iv_lnlos`: Log length of stay
- `baseline__table5_iv_lnsurg`: Log surgeries per case (weighted by sgprocs)
- `baseline__table5_iv_lnicu`: Log ICU days per admission
- `baseline__table5_iv_lndeathr`: Log death rate

### G2: Upcoding Analysis -- Effect of Price Spread on Complication Coding (Table 3)

**Claim object**: The causal effect of the DRG price spread (price difference between complication and non-complication codes) on the fraction of young patients coded into the higher-paying complication DRG.

**Baseline specification** (Table 3, Column 1):
- Second stage: `fracy ~ spread + year_dummies + drg_dummies [w=totyoung], robust`
- First stage: `spread ~ sp8788pt + year_dummies + drg_dummies [w=totyoung], robust`
- Instrument: `sp8788pt` = 1987-88 DRG weight spread change interacted with post-1987 dummy
- Young patients are used because they should not have mechanically different comorbidity profiles across years

**Additional baseline spec**:
- `baseline__table3_iv_fraco_old`: Old patient upcoding (adds `fracy87post` as control)

## RC Axes Included

### Instrument choice (G1)
- **Clean instrument**: Use `lnlascl88instr` (Laspeyres instrument cleaned of pre-trend charge correlation) instead of the baseline `lnlas88instr`
- **Reduced form**: Report reduced-form (intention-to-treat) estimates directly

### Outcome definitions (G1)
- Six outcomes spanning the full intensity response: volume (lnprocs), charges (lnchga), length of stay (lnlos), surgeries (lnsurg), ICU days (lnicu), death rate (lndeathr)
- Each outcome is a separate spec with potentially different weights

### Fixed effects structure
- **Drop DRG-specific trends**: Retain DRG FE and year FE only (removes trend controls)
- **Year dummies only**: No DRG FE (tests the across-DRG variation)

### Weighting
- **Unweighted**: Drop frequency weights
- **Alternative weights**: Match weight to outcome (chgprocs for charges, sgprocs for surgeries)

### Sample restrictions
- **Drop FY1985**: Start from 1986 (addresses possible pre-reform trending)
- **Pre-post only**: Keep only 1987 (last pre) and 1988 (first post)
- **DRG type subsamples**: Elective-only, urgent-only, emergent-only (Table 5 extension by DRG admission type)
- **Outlier trimming**: Trim extreme log weight changes; drop DRG-pairs with extreme instrument values

### Design alternatives
- **OLS (no IV)**: Run OLS version for comparison (paper reports these in text and Table 5 bottom panel)
- **LIML**: Limited information maximum likelihood (standard IV robustness)

### Joint variations
- **Outcome x weight**: Match frequency weight to outcome measure
- **Sample x DRG type x outcome**: Cross DRG-type subsamples with outcomes

### Upcoding (G2) RC axes
- **Old patient outcome**: `fraco` instead of `fracy`
- **Add control**: `fracy87post` for old-patient specification
- **Reduced form**: Directly regress outcome on instrument
- **Unweighted**: Drop frequency weights
- **Alternative weights**: `totold` for old-patient outcome
- **Drop FY1985**: Start from 1986
- **Outlier trimming**: Trim extreme spread values

## What Is Excluded and Why

- **Hospital-level analysis (Tables 6-7, inthosp.do)**: The hospital-level analysis uses a different instrument (share of young patients with complications in 1987, `shycc87`) and a different panel structure (hospital-year). This is a distinct estimation strategy and would require a separate baseline group. Excluded because the primary data (MedPAR 20% sample at hospital level) is confidential and not included in the replication package. The .dta files included are at the DRG or DRG-pair level.
- **Hospital-level upcoding (Table 4, up.do hospital section)**: Similarly requires confidential hospital-level data not included.
- **Exploration / heterogeneity by DRG type**: The paper's analysis by elective/urgent/emergent DRG type is treated as a sample restriction RC rather than separate baseline groups, since the paper frames the full-sample result as the headline claim.
- **Controls pool**: There are no additional covariates beyond the FE structure in either analysis. The controls axis is empty.
- **Sensitivity analysis**: No formal sensitivity analysis (e.g., Conley IV bounds) is conducted.

## Budgets and Sampling

- **G1 (Intensity)**: Max 60 core specs. Full enumeration of the discrete axes. Primary variation is across 6 outcomes x instrument variants x FE variants x sample restrictions x design alternatives.
- **G2 (Upcoding)**: Max 20 core specs. Fewer axes of variation.
- **Combined target**: ~80 total core specs.
- **Seed**: 116063

## Inference Plan

- **Canonical**: Heteroskedasticity-robust SE (Eicker-Huber-White), matching the paper's `ivreg ... , robust`
- **Variants**: Cluster at DRG level (panel unit); HC2 (small-sample corrected)
- Inference variants are recorded in `inference_results.csv`

## Key Linkage Constraints

- IV is a bundled estimator: first stage and second stage share the same FE structure (DRG dummies + DRG-specific trends + year dummies) and the same weight. Changes to FE structure or weights apply to both stages jointly.
- The instrument is fixed for each baseline group (lnlas88instr for G1; sp8788pt for G2). The "clean" instrument (lnlascl88instr) is an RC variant for G1.
- Outcome-specific weights: charges regressions use chgprocs as weight, surgery regressions use sgprocs. This linkage between outcome and weight is recorded in joint specs.
