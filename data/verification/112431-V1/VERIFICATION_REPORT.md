# Verification Report: 112431-V1

## Baseline Groups
- **G1**: Effect of reelection incentives on corruption
  - Baseline spec_run_ids: ['112431-V1_run_001', '112431-V1_run_002', '112431-V1_run_003']
  - Baseline spec_ids: ['baseline', 'baseline__ncorrupt', 'baseline__ncorrupt_os']
  - Expected sign: negative (first-term mayors less corrupt)

## Counts
- **Total rows**: 51
- **Core**: 51
- **Non-core**: 0
- **Invalid**: 0
- **Baselines**: 3

## Category Breakdown
category
core_controls    42
core_method       3
core_sample       2
core_fe           2
core_funcform     2

## Sign and Significance
- pcorrupt-family specs: 49 total
- Negative coefficient: 49/49 (100%)
- Significant at 5%: 44/49 (90%)
- Coefficient range: [-0.0312, -0.0175]

## Assessment
- **STRONG robustness**: The negative effect of reelection incentives on corruption is highly robust.
- All 49 pcorrupt-family specifications show a negative coefficient.
- 44/49 (90%) are significant at the 5% level.
- The 5 non-significant specs (at 5%) are all bivariate or minimal-control specifications, 
  which is expected given omitted variable concerns.
- LOO analysis shows no single control drives the result.
- Random control subsets confirm robustness across the covariate space.
- Trimmed samples remain significant, ruling out outlier-driven results.
- State FE removal weakens the result slightly but it remains significant.
- Functional form transforms (asinh, log1p) confirm the result.

## Recommendations
- Consider adding exploration specs for heterogeneity (e.g., by region, population size)
- The clustering variant (state-level) gives slightly tighter SEs than HC1
- Future work could add matching or IPW estimators if software supports it
