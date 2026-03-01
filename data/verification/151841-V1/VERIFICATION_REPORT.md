# Verification Report: 151841-V1

## Paper
Targeting High Ability Entrepreneurs Using Community Information: Mechanism Design in the Field (Beaman, Magruder, Robinson)

## Design
Randomized experiment (lottery-assigned capital grants). Panel FE interaction model. Focal parameter: Winner*Rank interaction (heterogeneous ITT).

## Baseline Groups

- **G1**: Community ranking predicts heterogeneous returns to capital grants
  - Baseline spec_run_ids: ['151841-V1_run_001', '151841-V1_run_002', '151841-V1_run_003', '151841-V1_run_004']
  - Baseline spec_ids: all labeled 'baseline'
  - Expected sign: positive (higher rank -> larger treatment effect)
  - Outcomes: Trim_Income (cols 1-2), Trim_Profits_30Days (cols 5-6)
  - With/without 26 baseline controls interacted with Winner

## Counts
- **Total rows**: 83
- **Core**: 83
- **Non-core**: 0
- **Invalid**: 0
- **Baselines**: 4

## Category Breakdown
category
core_data        20
core_controls    19
core_funcform    12
core_sample      12
core_method      10
core_fe           6
core_weights      4

## Surface Coverage
- 54 unique spec_ids planned in surface
- 46 unique spec_ids executed (83 rows, since most specs run on both income and profits)
- 5 planned specs not executed: adjusted_income, adjusted_profits, adjusted_profits_with_controls, client_level, client_level_with_controls
- 1 extra spec_id in results: 'baseline' (maps to 4 surface baseline__xxx IDs)

## Inference
- All specification_results.csv rows use canonical inference: cluster at GroupNumber (infer/se/cluster/group)
- inference_results.csv contains 8 inference variants (HC1 and cluster-at-HH) for the 4 baseline specs
- No infer/* rows in specification_results.csv (correct)

## Duplicate Specifications
13 pairs of duplicate regressions found (same coefficient, SE, N):
1. design/randomized_experiment/estimator/ancova == rc/joint/ancova_no_controls (income: run_005==run_055; profits: run_006==run_057)
2. rc/form/outcome/log_income == rc/joint/log_outcome_no_controls (run_034==run_060)
3. rc/form/outcome/log_profits == rc/joint/log_outcome_no_controls (run_035==run_061)
4. rc/data/rank_construction/include_self_rank == rc/joint/include_self_rank_no_controls (income: run_026==run_072; profits: run_027==run_073)
5. rc/data/rank_construction/relative_rank == rc/joint/relative_rank_no_controls (income: run_028==run_064; profits: run_029==run_065)
6. rc/data/rank_construction/median_rank == rc/joint/median_rank_no_controls (income: run_030==run_068; profits: run_031==run_069)
7. rc/form/treatment/tercile_rank == rc/joint/tercile_rank_no_controls (income: run_036==run_080; profits: run_037==run_081)
8. rc/controls/loo/paneld1_business_chars (profits) == rc/controls/subset/panela1_panelb1_panelc1 (profits) (run_016==run_025, coef=-1220.39)

These duplicates arise because the "no controls" joint variants are mechanically identical to the single-axis specs. All are kept as valid but represent 13 unique distinct regressions fewer than the 83 rows suggest. Effective unique regressions: ~70.

## Sign and Significance Analysis

### Trim_Income (N=33 specs)
- All 33 coefficients are positive
- 25/33 significant at 5%
- Coefficient range: [122.58, 6022.72]
- Baseline (no controls): coef=2303.35, p=0.017
- Baseline (with controls): coef=2147.65, p=0.034

### Trim_Profits_30Days (N=42 specs)
- 26/42 positive, 16/42 negative
- 1/42 significant at 5%
- Coefficient range: [-5201.91, 3712.07]
- Baseline (no controls): coef=466.06, p=0.724
- Baseline (with controls): coef=-91.29, p=0.939

### log_Income (N=3 specs)
- All positive, all significant at 5%
- Range: [0.2083, 0.2247]

### log_Profits (N=3 specs)
- All positive, all significant at 5%
- Range: [0.3575, 0.3644]

## Assessment

**MIXED robustness**: The income result is highly robust; the profits result is fragile.

**Income (Trim_Income)**: Strong evidence. All 33 income-family specifications show a positive Winner*Rank interaction, with 25/33 (76%) significant at 5%. The result is stable across control sets, rank constructions, sample restrictions, FE choices, weighting, and functional form. Only relative rank (a fundamentally different scaling) shows a near-zero effect, and the tercile parameterization strengthens the result.

**Profits (Trim_Profits_30Days)**: Weak evidence. Only 1/42 profits specs is significant at 5% (and that is the psychometric controls spec with a *negative* sign). 16/42 specs have negative point estimates. The baseline profits specifications are not significant (p=0.72, p=0.94). The ANCOVA design yields a larger, marginally significant effect (p=0.08), suggesting the panel FE approach may attenuate the profits signal. The log(profits+1) transform produces significant results, but this changes the estimand substantially.

**Key moderators**:
- Adding psychometric controls substantially attenuates income (coef drops from 2148 to 745, becomes insignificant) and flips profits sign (coef=-5202, p=0.053)
- ANCOVA estimator strengthens the profits result (coef=2702, p=0.08) relative to panel FE baseline
- Strata FE instead of HH FE gives larger income coefficient but insignificant profits
- Relative rank construction eliminates the income signal (coef=123, p=0.88)

## Top Issues
1. **13 duplicate regressions**: Joint no-controls specs duplicate single-axis specs
2. **5 missing surface specs**: adjusted_income/profits and client_level specs not executed
3. **Baseline spec_id naming**: All 4 baselines use generic 'baseline' rather than surface's detailed names
4. **Controls subset specs only run on profits**: Rows 17-25 (subset specs) are profits-only; no income counterparts

## Recommendations
1. Deduplicate or flag the joint no-controls specs to avoid inflating the specification count
2. Execute the 5 missing surface specs (adjusted_income, adjusted_profits, client_level variants)
3. Run controls subset specs on income as well as profits for symmetry
4. Use more specific baseline spec_ids matching the surface (e.g., baseline__table2_panelA_col1_income)
5. The large gap between income and profits robustness is substantively important and should be highlighted in reporting
