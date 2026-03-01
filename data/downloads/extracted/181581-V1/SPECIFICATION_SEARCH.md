# Specification Search Report: 181581-V1

## Paper
"When a Doctor Falls from the Sky: The Impact of Easing Doctor Supply Constraints on Mortality" (Okeke)

## Surface Summary
- **Paper ID**: 181581-V1
- **Baseline groups**: 1 (G1)
- **Design code**: randomized_experiment
- **Baseline outcome**: mort7 (7-day neonatal mortality)
- **Treatment**: doctor (Doctor Arm assignment), with mlp (MLP Arm) included
- **Canonical inference**: cluster SEs at fid (health facility level)
- **Budget**: max 80 core specs, 10 control subset draws
- **Seed**: 181581

## Execution Summary
- **Total specification rows**: 45
- **Successful**: 45
- **Failed**: 0
- **Inference variant rows**: 2
- **Inference successful**: 2
- **Inference failed**: 0

## Specs Executed

### Baselines (2)
- `baseline`: Table 4 Col 1 (mort7 ~ mlp + doctor | strata, cl(fid))
- `baseline__basic_controls`: Table 4 Col 2 (+ cont_base + qtr FE)

### Design variants (1)
- `design/randomized_experiment/estimator/diff_in_means`: No FE, simple regression

### RC: Control sets (3)
- `rc/controls/sets/none`: No controls
- `rc/controls/sets/basic`: cont_base (10 vars)
- `rc/controls/sets/extended`: cont_base + cont_hc (14+ vars)

### RC: Control progression (4)
- `rc/controls/progression/strata_only`
- `rc/controls/progression/strata_qtr`
- `rc/controls/progression/strata_qtr_individual`
- `rc/controls/progression/strata_qtr_individual_hc`

### RC: LOO controls (12)
- Drop each of: male, first, hausa, gest, car, last, auton, cct, hc_deliveries, hc_cesarean, hc_transfusion, hc_clean

### RC: Subset controls (10)
- 10 random draws from control pool (seed=181581)

### RC: Fixed effects (2)
- `rc/fe/add/qtr`: Add quarter FE to strata-only baseline
- `rc/fe/drop/qtr`: Drop quarter FE from basic-controls spec

### RC: Sample restrictions (4)
- `rc/sample/restriction/exclude_multiple_births`: Singleton births only
- Cross-combinations with extended controls, basic controls, mort30

### RC: Outcome definition (4)
- `rc/data/outcome/mort30`: 30-day mortality (no controls, basic, extended, singleton)

### RC: Treatment definition (4)
- `rc/data/treatment/doctor_only`: Drop mlp from regression (various control combos)

### Inference variants (2)
- `infer/se/hc/hc1`: HC1 robust SEs
- `infer/se/cluster/strata`: Cluster at strata level

## Software Stack
- Python 3.12.7
- pyfixest 0.40.1
- pandas 2.2.3
- numpy 2.1.3

## Data Notes
- Used pre-constructed `data/analysis/child.dta` (N=9126)
- Variable name fixes: autonomy->auton, hc_drugs->hc_drug, hc_nopower->hc_nopow
- Factor variables (magedum, mschool, hc_clean, hc_cond) expanded to dummies
- qtr (datetime) converted to integer for FE absorption

## Deviations from Surface
- Double-lasso (dsregress, Table 4 Col 4) is not replicated because it requires Stata 17.
  The core surface explicitly excludes this.
- Exploration specs (high_dose, low_dose, doctor_arm_only) were not executed per surface instructions.
