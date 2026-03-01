# Verification Report: 131981-V1

**Paper**: "Mental Health Costs of Lockdowns: Evidence from Age-specific Curfews in Turkey"
**Authors**: Altindag, Erten, and Keskin (AEJ: Applied Economics)
**Design**: Sharp regression discontinuity at age-65 COVID curfew cutoff
**Verified**: 2026-02-24

---

## Baseline Groups

### G1: Mental health effect of curfew at age-65 cutoff (single group)

- **Claim**: Sharp RD effect of being born before December 1955 (age >= 65, subject to curfew) on mental distress (z_depression, ICW-weighted SRQ index), at the age-65 threshold
- **Expected sign**: Positive (curfew restricts movement and social interaction, increasing mental distress)
- **Baseline spec_run_ids**: 131981-V1_run001, 131981-V1_run002, 131981-V1_run003, 131981-V1_run004
- **Baseline spec_ids**: baseline, baseline__table4_col1_bw17, baseline__table4_col3_bw45, baseline__table4_col4_bw60
- **Baseline results**:

| spec_id | Bandwidth | Coefficient | SE | p-value | N |
|---|---|---|---|---|---|
| baseline (Table 4 Col 2) | 30 | 0.3877 | 0.1153 | 0.0013 | 795 |
| baseline__table4_col1_bw17 | 17 | 0.2394 | 0.1886 | 0.2130 | 486 |
| baseline__table4_col3_bw45 | 45 | 0.2436 | 0.0985 | 0.0153 | 1160 |
| baseline__table4_col4_bw60 | 60 | 0.2726 | 0.0812 | 0.0011 | 1520 |

The primary baseline (bw=30, run001) finds a 0.39 SD increase in mental distress for those subject to the curfew (p=0.001). The narrow bandwidth (bw=17) is not significant due to low power.

---

## Summary Counts

| Category | Count |
|---|---|
| Total rows in specification_results.csv | 45 |
| is_valid = 1 | 45 |
| is_valid = 0 | 0 |
| is_core_test = 1 | 45 |
| is_core_test = 0 | 0 |
| is_baseline = 1 | 4 |
| is_baseline = 0 | 41 |
| Inference rows (inference_results.csv) | 2 |

### By category

| Category | Count |
|---|---|
| core_method | 30 |
| core_controls | 9 |
| core_sample | 3 |
| core_funcform | 3 |
| noncore_* | 0 |
| invalid | 0 |

### core_method breakdown (30 rows)

- 4 baseline specs (baseline, baseline__table4_col1_bw17, baseline__table4_col3_bw45, baseline__table4_col4_bw60)
- 8 design/bandwidth specs (bw=17, 24, 36, 45, 48, 60, 72 + duplicate of bw17 as Table 4 Col 1)
- 1 design/poly spec (local_quadratic)
- 3 rc/joint/bw_poly specs (bw30, bw45, bw60 with quadratic polynomial)
- 3 rc/joint/bw_controls specs (bw17, bw45, bw60 without controls)
- 9 rc/joint/outcome_bw specs (3 outcomes x 3 bandwidths)
- 3 rc/joint/outcome_donut specs (3 outcomes x donut=2mo)

### core_controls (9 rows)

- 6 rc/controls/loo specs (drop_month_fe, drop_province_fe, drop_ethnicity_fe, drop_education_fe, drop_female, drop_survey_taker_fe)
- 3 rc/controls/sets specs (no_controls, minimal_demo, full_baseline)

### core_sample (3 rows)

- rc/sample/donut/exclude_1month, exclude_2month, exclude_3month

### core_funcform (3 rows)

- rc/form/outcome/sum_srq, z_somatic, z_nonsomatic

---

## Step 0: Sanity Checks

| Check | Result | Details |
|---|---|---|
| spec_run_id unique | PASS | 45 unique IDs, 45 rows |
| baseline_group_id exists | PASS | All rows G1 |
| spec_id consistent with spec_tree_path | PASS | 8 unique paths, all reference valid spec-tree .md nodes |
| spec_tree_path has #anchor | PARTIAL | All paths except modules/robustness/joint.md lack a specific anchor; acceptable since joint.md is a single-section module |
| run_success exists and is 0/1 | PASS | All 45 rows run_success=1 |
| No run_success=0 rows | PASS | 0 failures |
| coefficient_vector_json has required audit keys | PASS | All 45 rows have coefficients, inference, software, surface_hash |
| design block present in all rows | PASS | All rows have design.regression_discontinuity block |
| rc/form/* rows have functional_form | PASS | 3 rc/form rows + all 12 rc/joint/outcome* rows have functional_form |
| rc/controls/* rows have controls block with matching spec_id | PASS | All 9 verified |
| rc/sample/* rows have sample block with matching spec_id | PASS | All 3 verified |
| rc/joint/* rows have joint block with matching spec_id | PASS | All 18 verified |
| All inference.spec_id = canonical (infer/se/cluster/modate) | PASS | All 45 rows use canonical inference |
| Numeric fields finite for run_success=1 | PASS | All coefficient, SE, p-value, N present and finite |
| p-values in [0,1] | PASS | All p-values valid |
| No infer/* rows in specification_results.csv | PASS | Confirmed; 2 infer/* rows correctly appear only in inference_results.csv |
| No arbitrary top-level CVJ keys | PASS | Extra keys only: controls, sample, joint, functional_form (all allowed) |

---

## Step 1: Surface Alignment

- Surface defines 1 baseline group (G1) with 45 total spec_ids (4 baseline + 8 design + 33 rc/*)
- Results contain exactly 45 rows matching the surface spec_ids 1:1
- No missing baseline groups; no spurious groups
- No linked_adjustment (surface sets linked_adjustment=false)
- All rc/joint/* wildcard patterns in the surface (rc/joint/bw_poly/*, rc/joint/bw_controls/*, rc/joint/outcome_bw/*, rc/joint/outcome_donut/*) are fully covered by 18 executed rows
- Surface coverage is complete

---

## Step 2: Baseline Identification

Four specs are baseline (spec_id == "baseline" or starts with "baseline__"):
- run001 (baseline): Primary, bw=30, Table 4 Col 2
- run002 (baseline__table4_col1_bw17): Table 4 Col 1, bw=17
- run003 (baseline__table4_col3_bw45): Table 4 Col 3, bw=45
- run004 (baseline__table4_col4_bw60): Table 4 Col 4, bw=60

Note: rc/controls/sets/full_baseline (run021) is NOT a baseline spec; it is a robustness check that happens to produce identical results to run001 (same controls, same sample, same inference). This is expected and correct.

---

## Step 3: Classification Notes

**Alternative outcome specs (rc/form/* and rc/joint/outcome_*/*)**: z_somatic, z_nonsomatic, and sum_srq are subscales or raw counts from the same SRQ-20 instrument. The surface's claim_object defines the outcome concept broadly as "Mental distress index" and explicitly lists these specs in core_universe rc_spec_ids. They are classified as core_funcform (for rc/form/* rows) and core_method (for rc/joint/outcome_*/* rows), consistent with the surface's intent. No noncore_alt_outcome override applied.

**Design block metadata inconsistency in bw_poly joint specs**: For rc/joint/bw_poly/* rows (run028, run029, run030), the `coefficient_vector_json.design.regression_discontinuity.poly_order` field shows 1 instead of 2. However, `joint.details.poly_order` correctly records 2 and the executed coefficients exactly match the local_quadratic design spec (run012 at bw=30, N=795, coef=0.2007). This is a metadata propagation artifact from the design-block template; execution was correct. Rows remain valid.

**No drift detected**: Treatment variable (before1955), running variable (dif), cutoff (0), and RD design are identical across all 45 rows. No population changes or heterogeneity-only specs present.

---

## Inference Variants (inference_results.csv)

- 2 inference variant rows, both linked to run001 (primary baseline, bw=30)
- Correctly separated from specification_results.csv
- Both rows run_success=1
- HC1 (no clustering): SE=0.1644, p=0.0187 -- still significant
- Province clustering: SE=0.1600, p=0.0186 -- still significant
- Baseline estimate robust to alternative SE choices; modate clustering gives the tightest SEs as expected

---

## Issues Found

No substantive issues. This is a clean specification run:

1. No outcome or treatment drift
2. No invalid extractions
3. No missing or spurious specifications relative to the surface
4. Perfect spec_run_id uniqueness (45 unique)
5. Correct separation of inference variants into inference_results.csv
6. All coefficient_vector_json structures well-formed with all required audit keys
7. Design block poly_order metadata inconsistency in bw_poly joint specs is cosmetic; execution verified correct

Minor flag (cosmetic only):
- **design.poly_order stale in bw_poly rows**: rc/joint/bw_poly/{bw30,bw45,bw60}_poly2 show poly_order=1 in the design block but 2 in the joint.details block. The joint block is authoritative; coefficients confirm correct execution. Recommend fixing the runner to update design block from joint.details when writing CVJ.

---

## Key Findings Across Specifications

- **31 of 45 specs (69%)** significant at 5%; **34 of 45 (76%)** at 10%
- **All 45 coefficients are positive** (consistent with the causal claim direction)
- Coefficient range: [0.038, 1.41] (SD units for index outcomes; raw count for sum_srq)
- Weakest specs involve the narrowest bandwidth (bw=17, N~486) or quadratic polynomial at bw=30
- Controls robustness strong: all 6 LOO specs significant at p < 0.007
- Donut hole specs strengthen the result (coefficients 0.37-0.45, all p < 0.013)
- Alternative outcomes at bw=30: sum_srq p=0.020, z_somatic p=0.009, z_nonsomatic p=0.048 (all significant)

**Overall verdict**: STRONG support for the main claim. The curfew increases mental distress across the vast majority of specifications, with positive sign maintained universally.

---

## Recommendations

1. No changes to the surface are needed; it accurately describes the executed specifications.
2. The planned diagnostics (covariate balance test, McCrary density test) from the surface diagnostics_plan were not executed; a future diagnostics pass should run these.
3. Fix runner: update design.regression_discontinuity.poly_order in CVJ for bw_poly joint specs.
4. Consider extending LOO control analysis to alternative outcomes (z_somatic, z_nonsomatic) in future runs.
5. Add #anchor to spec_tree_path for joint.md when the module supports sub-section anchors.
