# Specification Search: 113744-V1

## Paper
- **Title**: Promoting Healthy Choices: Information vs. Convenience
- **Authors**: Wisdom, Downs, and Loewenstein (2010)
- **Journal**: American Economic Journal: Applied Economics
- **Design**: Randomized field experiment at Subway restaurants

## Surface Summary
- **Baseline groups**: 1 (G1: Total Meal Calories)
- **Design code**: randomized_experiment
- **Surface hash**: sha256:3c4ace064fcc0f07c864efacbe735c6cf587e1140cfb40441a647e201ef7fa97
- **Seed**: 113744
- **Budget**: max_specs_core_total=65, max_specs_controls_subset=20

## Baseline Specifications
- `baseline`: TotalCal ~ CalInfo + CalRef + HealthyMenu + UnhealthyMenu + Age + Female + AfrAmer, HC1 SE
- `baseline__table3_sandwich_cal`: SandwichCal ~ same
- `baseline__table3_non_sandwich_cal`: NonSandwichCal ~ same
- **Focal coefficient**: HealthyMenu (paper's main finding: menu ordering matters more than info)

## Executed Specifications

### Total: 51 specification rows + 2 inference rows

| Category | Count |
|----------|-------|
| Baseline | 3 |
| Design variants | 5 |
| RC: Controls LOO | 3 |
| RC: Controls Add | 5 |
| RC: Controls Sets | 8 |
| RC: Controls Subsets | 17 |
| RC: Sample | 6 |
| RC: Functional Form | 3 |
| RC: Preprocess | 1 |
| Inference variants | 2 |

### Successes: 51 / 51 spec rows, 2 / 2 inference rows

## Deviations from Surface
- None. All surface-specified specs were executed.

## Software Stack
- Python 3.12.7
- pyfixest: 0.40.1
- statsmodels: 0.14.6
- pandas: 2.2.3
- numpy: 2.1.3
- pyreadstat: 1.3.3

## Notes
- Data loaded from SPSS (.sav) file via pyreadstat.
- The paper uses multiple treatment indicators (CalInfo, CalRef, HealthyMenu, UnhealthyMenu) in a single regression.
- Focal coefficient throughout is HealthyMenu (paper's primary finding: menu ordering > calorie information).
- Missing data handled by listwise deletion consistent with the paper's SPSS syntax.
- OpenedSeal subsample is very small (N~111) as it is only Study 1 customers who opened the seal.
- Control subset sampling uses seed=113744 with stratified_size draws across control pool sizes.
