# Verification Report: 113517-V1

## Baseline Groups Found

### G1: EE reallocation rate predicting wage growth
- **Baseline spec_run_ids**: 113517-V1_R001, 113517-V1_R038, 113517-V1_R039, 113517-V1_R040
- **Baseline spec_ids**: baseline, baseline__logern_spec6, baseline__loghwr_nom_spec6, baseline__loghwr_spec6
- **Expected sign**: positive (EE rate positively predicts wage growth)
- **Notes**: Primary baseline is `baseline` (logern_nom). Three additional baselines for logern, loghwr_nom, loghwr.

## Summary Counts

| Metric | Count |
|--------|-------|
| Total rows | 53 |
| Valid | 53 |
| Invalid | 0 |
| Core | 53 |
| Non-core | 0 |
| Baseline | 4 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 24 |
| core_fe | 2 |
| core_funcform | 1 |
| core_joint | 16 |
| core_method | 4 |
| core_sample | 5 |
| core_weights | 1 |

## Issues

- No invalid rows detected
- All 53 specifications executed successfully (run_success=1)
- All spec_ids are properly whitelisted in the surface core universe
- Surface hash matches across all rows

## Recommendations

1. The 16 WARNs from the validator are about missing #section-anchors in spec_tree_path for joint.md -- these are non-blocking
2. The specification search covers the paper's main claim robustly across controls, samples, weights, FE, and joint combinations
3. All 4 outcome variants (earnings/wages x nominal/real) are represented
