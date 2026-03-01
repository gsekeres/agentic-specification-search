# Verification Report: 120483-V1

## Baseline Groups
- **G1**: Malaria ecology (MAL) and slave share in US counties
  - Baseline spec_run_ids: ['120483-V1_run_001']
  - Baseline spec_ids: ['baseline__table1_col5']
  - Expected sign: positive (higher malaria ecology -> higher slave share)

## Counts
- **Total rows**: 52
- **Core**: 52
- **Non-core**: 0
- **Invalid**: 0
- **Baselines**: 1

## Category Breakdown
| Category | Count |
|----------|-------|
| core_controls | 35 |
| core_sample | 11 |
| core_funcform | 3 |
| core_fe | 2 |
| core_method | 1 |

## Sign and Significance
- All 52 specifications show a positive coefficient (100%)
- Significant at 5%: 49/52 (94%)
- Coefficient range: [0.0574, 0.2355]
- The 3 non-significant specs are all 1790-era subsamples (N=244-285 vs N=1955 for 1860):
  - rc/sample/restrict/1790_data (p=0.156)
  - rc/sample/restrict/1790_slave_states (p=0.224)
  - rc/form/outcome/asinh_1790 (p=0.118)

## Assessment
- **STRONG robustness**: The positive association between malaria ecology and slavery is highly robust across 52 specifications.
- All 52 specifications produce positive coefficients; 94% are significant at the 5% level.
- LOO analysis (15 specs): no single control drives the result. Coefficients remain stable at 0.17-0.20.
- Random control subsets (15 specs): all positive and significant.
- Control set variations (none, crop-only, geo-only): all significant, with larger effects when fewer controls are included.
- Sample restrictions (slave states, 1790 data): remain positive but 1790 subsample loses significance due to small N (285 vs 1955).
- Dropping state FE: effect becomes larger (0.23), consistent with state-level confounders attenuating the within-state estimate.
- Functional form (asinh, log1p): results hold with transformed outcomes.
- Outlier trimming (1/99, 5/95): results stable.

## Issues
1. **Surface-to-runner deviation on baseline**: The surface lists two baseline specs (Table1-Col3 for 1790 and Table1-Col5 for 1860), but the runner only coded Table1-Col5 as the baseline row. The 1790 data is captured as rc/sample/restrict/1790_data instead. This is functionally reasonable since 1860 is the primary specification.
2. **Canonical inference fallback**: The surface specifies Conley 100km spatial SEs as canonical, but the runner used state-clustered SEs (infer/se/cluster/state) because Conley SEs are unavailable in pyfixest. The inference_results.csv contains one HC1 variant. This is a known limitation.
3. **Extra specs beyond surface plan**: Runs 044-052 are joint sample+controls combinations (e.g., slave_states_no_controls, slave_states_crop_only, 1790_slave_states, etc.) and FE/sample combos not in the original rc_spec_ids list. These are coherent extensions of the planned axes and are classified as core.

## Recommendations
- Consider adding Table1-Col3 (1790 data) as a second explicit baseline row to match the surface plan.
- If Conley spatial SEs become available, re-run inference to match the paper's canonical standard errors.
- The joint sample+controls specs are informative but should be documented in the surface if retained.
