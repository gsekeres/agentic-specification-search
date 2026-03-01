# Specification Search: 133501-V1

## Paper
Huh & Reif (2021), "Teenage Driving, Mortality, and Risky Behaviors", AER 111(12).

## Surface Summary
- **Paper ID**: 133501-V1
- **Baseline groups**: 1 (G1: Sharp RD on mortality at MDA cutoff)
- **Design code**: regression_discontinuity
- **Running variable**: agemo_mda (age in months relative to MDA)
- **Cutoff**: 0
- **Primary outcome**: cod_MVA (motor vehicle accident mortality per 100,000)
- **Canonical inference**: Robust bias-corrected (HC) from rdrobust
- **Max specs budget**: 80
- **Seed**: 133501

## Execution Summary
- **Total specs planned**: 48
- **Specs executed successfully**: 48
- **Specs failed**: 0

### Breakdown by type:
- **Baseline**: 3 (cod_MVA, cod_any, cod_sa_poisoning)
- **Design/bandwidth**: 6 (mserd, msetwo, cerrd, certwo, half, double)
- **Design/polynomial**: 3 (p=1, p=2, p=3)
- **Design/kernel**: 3 (triangular, uniform, epanechnikov)
- **Design/procedure**: 2 (conventional, robust bias-corrected)
- **RC/controls/loo**: 1 (drop firstmonth)
- **RC/sample**: 6 (male, female, mda192, mda_not192, early_period, late_period)
- **RC/form**: 1 (log1p transform)
- **RC/data/outcome_alt**: 5 (cod_any, cod_external, cod_sa_poisoning, cod_sa_drowning, cod_extother)
- **Cross-product bw x outcome**: 6 (half/double/cerrd x cod_any/cod_sa_poisoning)
- **Cross-product poly x outcome**: 4 (p=2/p=3 x cod_any/cod_sa_poisoning)
- **Cross-product sample x outcome**: 8 (male/female/mda subgroups x alternative outcomes)

## Deviations from Surface
- `rc/sample/restriction/early_period` and `rc/sample/restriction/late_period`: Constructed by aggregating year-bin datasets (4-year bins) from the derived mortality data. Year-bin files only contain cod_MVA and cod_sa_poisoning, so only cod_MVA was used.
- No inference variants were requested; inference_results.csv is empty.

## Data
- Unit of observation: age-in-months cell (96 cells spanning -48 to +47 months relative to MDA)
- Death rates computed as: 100000 * deaths / (population / 12)
- Population data from SEER; deaths from CDC mortality files
- Pooled dataset: mortality_none.dta (both sexes, all MDA types, 1983-2014)
- Subgroup datasets: mortality_male.dta, mortality_female.dta, mortality_mda192.dta, mortality_mda_not192.dta
- Year-bin datasets: mortality_{sex}_{year}.dta (4-year bins for time trends)

## Software Stack
- Python 3.12.7
- rdrobust (Python): 1.3.0
- pandas: 2.2.3
- numpy: 2.1.3

## Key Findings
The primary baseline (Table 1 MVA pooled) estimates a sharp RD effect of approximately 4.9 deaths per 100,000 person-years at the MDA cutoff, with robust bias-corrected p-value ~ 0.01.
Results are robust across bandwidth choices, polynomial orders, and kernels. The effect is substantially larger for males than females. Alternative outcomes (all-cause, external) also show positive discontinuities, while non-driving-related causes (SA poisoning, drowning) show smaller or null effects, consistent with the driving mechanism.
