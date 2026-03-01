# Verification Report: 130784-V1

**Paper**: "Child Marriage Bans and Female Schooling and Labor Market Outcomes: Evidence from Natural Experiments in 17 Low- and Middle-Income Countries"

## Baseline Groups

- **G1**: Effect of child marriage bans (cohort exposure x regional pre-ban intensity) on child marriage
  - Baseline spec_run_ids: `130784-V1__baseline__001`, `130784-V1__baseline__marriage_age__002`, `130784-V1__baseline__educ__003`, `130784-V1__baseline__employed__004`
  - Baseline spec_ids: `baseline`, `baseline__marriage_age`, `baseline__educ`, `baseline__employed`
  - Expected sign: negative (bans reduce child marriage)
  - Primary baseline: childmarriage ~ bancohort_pcdist, coef=-0.0077, p=0.003 (Table4-PanelA-Col1)
  - Three additional baselines use alternative outcomes from the same table (marriage_age, educ, employed)

## Counts

- **Total rows**: 50
- **Core**: 47
- **Non-core**: 3 (alternative-outcome baselines: marriage_age, educ, employed)
- **Invalid**: 0
- **Baselines**: 4 (1 primary + 3 alternative-outcome)

## Category Breakdown

| Category | Count |
|----------|-------|
| core_sample | 24 |
| core_data | 10 |
| core_controls | 5 |
| core_funcform | 5 |
| noncore_alt_outcome | 3 |
| core_fe | 2 |
| core_method | 1 |

## Sign and Significance

### Primary claim (childmarriage ~ bancohort_pcdist, N=32 specs)
- Negative coefficient: 32/32 (100%)
- Significant at 5%: 29/32 (91%)
- Coefficient range: [-0.0114, -0.0018]

### All childmarriage-family specs (including threshold variants + alt treatment, N=47)
- Negative coefficient: 47/47 (100%)
- Significant at 5%: 37/47 (79%)

### Non-significant childmarriage specs (6 total)
1. `rc/sample/subgroup/urban_only`: coef=-0.0057, p=0.205 (urban subsample, plausibly weaker effect)
2. `rc/sample/subgroup/baseline_16_countries`: coef=-0.0018, p=0.715 (countries with baseline minimum 16+, weaker treatment contrast)
3. `rc/sample/subgroup/no_minimum_countries`: coef=-0.0076, p=0.174 (countries with no minimum in 1995, only 3 countries, small N=17016)
4. `rc/data/treatment/binary_mean_above`: coef=-0.0102, p=0.118 (binary intensity indicator, less precise than continuous)
5. `rc/data/treatment/bancohort_age16`: coef=-0.0047, p=0.119 (alt ban cohort cutoff age 16)
6. `rc/data/treatment/bancohort_age15`: coef=-0.0044, p=0.177 (alt ban cohort cutoff age 15)

All non-significant specs retain the expected negative sign. Insignificance is interpretable: weaker treatment contrast (countries already at 16+), small subsamples (3-country group), binary vs continuous intensity, or cohort cutoffs (16, 15) far from the legal threshold (18).

## Structural Audit

- **spec_run_id uniqueness**: All 50 spec_run_ids are unique. Pass.
- **baseline_group_id**: All rows assigned to G1. Matches the surface. Pass.
- **spec_tree_path**: All paths reference valid spec-tree nodes with anchors where appropriate. Pass.
- **coefficient_vector_json**: All rows have required keys (coefficients, inference, software, surface_hash). Pass.
- **Axis blocks**: All rc/* rows contain the correct axis-appropriate block (controls, sample, fixed_effects, functional_form, data_construction) with matching spec_ids. Pass.
- **functional_form blocks**: All rc/form/* rows have non-empty functional_form objects. Pass.
- **Inference**: All 50 rows use canonical inference (infer/se/cluster/countryregionurban). Pass.
- **Numeric fields**: All finite, no NaN or Inf values for run_success=1 rows. Pass.
- **No infer/* rows**: specification_results.csv contains 0 infer/* rows. Pass.
- **Inference file**: inference_results.csv contains 8 inference variants (country-level clustering + HC1 for each of the 4 baselines), all run_success=1. Pass.

## Issues

1. **One missing surface-planned spec**: `rc/controls/add/interviewyear_banyear_fe` appears in the surface core_universe but was not executed. This is a minor gap (46/47 planned RC specs executed). The SPECIFICATION_SEARCH.md notes that interviewyear and banyear_pc are constant within country and thus collinear with country FE, which may explain why this spec was dropped.

2. **Alternative-outcome baselines classified as non-core**: The surface lists baseline__marriage_age, baseline__educ, and baseline__employed as baseline specs under G1. However, these change the outcome concept from the primary claim (childmarriage). They are classified as `noncore_alt_outcome` because they test a different estimand (effect on marriage age / education / employment rather than child marriage probability). This is conservative; if the paper's claim object is meant to encompass all four outcomes, they could be reclassified as core.

3. **Identical coefficients for some controls specs**: Rows 005 (interviewyear), 006 (interviewyear_quad), 007 (cslcohortdist), 009 (interviewyear_fe), 010 (banyear_pc_fe), and 050 (regtrend) all produce coef=-0.007686, p=0.003 -- identical to baseline. The SPECIFICATION_SEARCH.md explains this: interviewyear and banyear_pc are constant within country (one survey round per country), so adding them alongside countryregionurban FE has no additional effect. Similarly, cslcohortdist may be absorbed. This is not invalid but indicates these controls add no information beyond what FE already capture.

## Assessment

**STRONG robustness**: The negative effect of child marriage bans on child marriage rates is highly robust across the specification surface.

- All 47 childmarriage-family specifications produce negative coefficients (100% sign stability).
- 37/47 (79%) are significant at 5%, and 29/32 (91%) of the primary-claim specs are significant at 5%.
- The 17-country jackknife shows no single country drives the result (all 17 variants significant at 5%).
- Controls robustness is strong (though several controls are absorbed by FE structure).
- Alternative intensity measures mostly preserve significance; binary indicators are less precise as expected.
- Lower child marriage thresholds (age 14, 13, 12) show attenuating effects, consistent with theory.
- Inference sensitivity is mild: country-level clustering (17 clusters) and HC1 both preserve baseline significance.
- The only substantive sources of insignificance are (a) small or weakly-treated subsamples and (b) less informative treatment measures.

## Recommendations

1. **Consider reclassifying alt-outcome baselines**: If the paper's claim object is a joint statement about marriage, education, and employment, the three alt-outcome baselines could be moved to core. Currently classified conservatively as noncore_alt_outcome.
2. **Add the missing spec**: `rc/controls/add/interviewyear_banyear_fe` from the surface was not executed. Could be added for completeness, though it is expected to be identical to baseline given FE structure.
3. **Region-specific trends**: The simplified regtrend control produces an identical coefficient to baseline (as expected given the simplification noted in SPECIFICATION_SEARCH.md). The paper's full specification with ~282 region-specific linear age trends could not be replicated computationally.
4. **Synthetic data caveat**: All results are based on synthetic data constructed to match the paper's data construction procedures. Point estimates may differ from original results, but the specification surface structure and robustness patterns are informative for understanding sensitivity.
