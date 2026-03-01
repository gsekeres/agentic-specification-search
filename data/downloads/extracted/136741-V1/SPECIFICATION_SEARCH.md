# Specification Search: 136741-V1
**Paper**: Williams (2022) 'Historical Lynchings and the Contemporary Voting Behavior of Blacks'
**Design**: Cross-sectional OLS with state FE

## Surface Summary
- Baseline groups: 1 (G1)
- Budget: max 80 core specs, 20 control subsets
- Seed: 136741
- Surface hash: sha256:049f4b26c68793cf74a231b87fb447a4e240f15588921c34d2755fca403f7c88

## Baseline
- Outcome: Blackrate_regvoters (% black registered voters)
- Treatment: lynchcapitamob (black lynchings per 10k black pop, 1882-1930)
- Controls: 7 historical (illiteracy, county age, newspapers, farm value, small farms, land inequality, free blacks)
- FE: State_FIPS (6 states)
- Inference: IID (default OLS SEs, matching paper)
- Baseline coefficient: -0.4692 (SE=0.1437, p=0.0012, N=267)

## Execution Summary
- Specification results: 52 rows (52 success, 0 failed)
- Inference results: 2 rows (2 success, 0 failed)

### Breakdown by type:
- baseline: 1
- rc/controls/loo: 7 (drop each historical control)
- rc/controls/sets: 4 (none, historical+slaves, historical+contemporary, kitchen sink)
- rc/controls/progression: 9 (bivariate through all contemporary)
- rc/controls/subset: 20 (random draws, seed=136741)
- rc/sample/outliers: 2 (trim 1/99, trim 5/95)
- rc/sample/restriction: 2 (cap at 100, drop above 100)
- rc/fe/drop: 1 (no state FE)
- rc/data/treatment: 4 (1910/1920/1930 denominators, Stevenson data)
- rc/form/outcome: 2 (asinh, log1p)
- TOTAL: 52 specification rows

### Inference variants (baseline only):
- infer/se/hc/hc1: HC1 robust SEs
- infer/se/cluster/State_FIPS: Cluster at state level (6 clusters, caution)

## Software
- Python 3.12.7
- pyfixest 0.40.1
- pandas 2.2.3
- numpy 2.1.3

## Deviations
- None. All surface specs executed successfully.
