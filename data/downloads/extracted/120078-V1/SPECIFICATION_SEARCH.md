# Specification Search: 120078-V1

## Paper
- **Title**: Can Information Reduce Ethnic Discrimination? Evidence from Airbnb
- **Authors**: Combes, Decreuse, Laounan & Trannoy
- **Journal**: AEJ: Applied Economics

## Surface Summary
- **Baseline group**: G1 (Table 5 Col 1)
- **Design**: Panel fixed effects (within estimator)
- **Outcome**: log_price (daily log-price of Airbnb listing)
- **Treatment**: minodummy x rev100 (minority host x review count/100)
- **FE**: newid (listing) + citywaveID (city x wave)
- **Cluster**: newid (listing)
- **Sample**: Drev100 > 0 & review > 0 & review < 40
- **Budget**: max 70 core specs, 15 control subsets
- **Seed**: 120078

## Execution Summary
- **Planned specs**: 50 estimate rows + 2 inference rows
- **Successful**: 50 estimate + 2 inference
- **Failed**: 0 estimate + 0 inference

### Spec breakdown
| Category | Count |
|----------|-------|
| baseline | 1 |
| baseline (additional) | 2 |
| rc/controls/loo | 13 |
| rc/controls/sets | 5 |
| rc/controls/progression | 5 |
| rc/controls/subset | 15 |
| rc/sample/restriction | 4 |
| rc/sample/outliers | 2 |
| rc/fe/drop | 1 |
| rc/fe/swap | 1 |
| rc/form/treatment | 1 |
| **Total estimate rows** | **50** |
| infer/se/hc | 1 |
| infer/se/cluster | 1 |
| **Total inference rows** | **2** |

## Notes
- In the listing FE model, many lesX controls are time-invariant and get absorbed.
  pyfixest silently drops collinear variables, matching Stata xtreg behavior.
- Wave-minority interactions (c.minodummy#ib10.wave) created manually (wave 10 = reference).
- Focal coefficient: mino_x_rev100 = minodummy * rev100.
- Table 5 Col 3 adds quadratic terms.

## Software
- Python 3.12.7
- pyfixest 0.40.1
- pandas 2.2.3
- numpy 2.1.3
