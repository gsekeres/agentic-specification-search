# Specification Search Report: 120483-V1

## Paper
- **Title**: Malaria Stability and Slavery
- **Paper ID**: 120483-V1

## Surface Summary
- **Baseline groups**: 1 (G1)
- **Design code**: cross_sectional_ols
- **Baseline spec**: Table 1 Col 5 (1860, all states, full controls)
- **Budget**: max 75 core specs, 15 control subsets
- **Seed**: 120483

## Canonical Inference
- Paper uses Conley spatial SEs (acreg) as primary inference
- **Canonical inference used here**: State-clustered SEs (CRV1) -- Conley spatial SEs not available in pyfixest
- State-clustered SEs are reported as alternative inference in the paper (curly braces)

## Execution Summary
- **Total specifications planned**: 52
- **Successful**: 52
- **Failed**: 0
- **Inference variants**: 1

## Specifications Executed

### Baseline (1 spec)
- `baseline__table1_col5`: 1860 data, full controls, state-clustered SEs

### LOO Control Drops (15 specs)
- Dropped each of the 15 controls in the 1860 specification one at a time

### Control Sets (3 specs)
- `rc/controls/sets/none`: No controls (MAL + state FE only)
- `rc/controls/sets/crop_only`: Crop suitability controls only
- `rc/controls/sets/geo_only`: Geography controls only

### Control Progression (2 specs)
- `rc/controls/progression/crop_suitability`: Add crop controls
- `rc/controls/progression/crop_and_geo`: Add crop + geography (=full)

### Random Control Subsets (15 specs)
- 15 random draws from the control pool, seed=120483

### Sample Restrictions (4 specs)
- `rc/sample/restrict/slave_states_only`: 1860, slave states only
- `rc/sample/restrict/1790_data`: 1790 dataset with 1790 controls
- `rc/sample/outliers/trim_y_1_99`: Trim outcome 1-99 pct
- `rc/sample/outliers/trim_y_5_95`: Trim outcome 5-95 pct

### FE Variants (1 spec)
- `rc/fe/drop/state_g`: Drop state FE, use HC1

### Functional Form (2 specs)
- `rc/form/outcome/asinh`: asinh(slaveratio)
- `rc/form/outcome/log1p`: log1p(slaveratio)

## Inference Results
- `infer/se/hc/hc1`: HC1 for baseline

## Deviations from Surface
- Conley spatial SEs (100km, 250km, 500km) not available in Python/pyfixest. The surface notes that state-clustered SEs are an acceptable canonical fallback.
- Conley inference variants (`infer/se/spatial/conley_100km`, `conley_250km`, `conley_500km`) not implemented.

## Software Stack
- Python 3.12.7
- pyfixest: 0.40.1
- pandas: 2.2.3
- numpy: 2.1.3
