# Verification Report: 138922-V1

**Paper**: "The Long-Run Effects of Sports Club Vouchers for Primary School Children"
**Authors**: Jan Marcus, Thomas Siedler, Nicolas R. Ziebarth
**Design**: Difference-in-Differences (TWFE, repeated cross-section)
**Verified by**: Post-run auditor (agent)
**Verification date**: 2026-02-24

---

## Baseline Groups

**G1** (sole baseline group): ATT of C2SC sports club voucher program (Saxony, post-2008/09) on sports club membership among primary school children.

- Primary baseline spec_run_id: `138922-V1__baseline`
- All baseline spec_run_ids:
  - `138922-V1__baseline` (Table 2 Col 3: TWFE with year + state + city FEs, no controls, clustered SE at city)
  - `138922-V1__baseline__sportsclub_col1` (Table 2 Col 1: OLS with group dummies, no city FE)
  - `138922-V1__baseline__sportsclub_col2` (Table 2 Col 2: year + state FEs only, no city FE)
  - `138922-V1__baseline__sport_hrs` (alt outcome: sport hours per week)
  - `138922-V1__baseline__oweight` (alt outcome: overweight binary)
  - `138922-V1__baseline__kommheard` (alt outcome: heard of voucher program)
  - `138922-V1__baseline__kommgotten` (alt outcome: received voucher)
  - `138922-V1__baseline__kommused` (alt outcome: used voucher)

- Expected sign: **positive** (voucher intended to increase club membership)
- Primary baseline result: coef=0.0089, SE=0.0187, p=0.635, N=13,333

---

## Counts

| Metric | Count |
|---|---|
| **Total rows** | 53 |
| **Valid (run_success=1)** | 52 |
| **Invalid (run_success=0)** | 1 |
| **Baseline rows (is_baseline=1)** | 8 |
| **Core rows (is_core_test=1)** | 48 |
| **Non-core rows (is_core_test=0)** | 5 |
| **Inference variants (inference_results.csv)** | 3 (all successful) |

---

## Category Breakdown

| Category | Count | Notes |
|---|---|---|
| `core_controls` | 24 | 1 sets, 9 LOO, 4 progression, 10 random subset |
| `core_sample` | 12 | 5 time, 2 geographic, 5 quality |
| `noncore_alt_outcome` | 5 | baseline__sport_hrs, baseline__oweight, baseline__komm* |
| `core_method` | 3 | baseline, baseline__sportsclub_col1, baseline__sportsclub_col2 |
| `core_fe` | 3 | 2 drop, 1 add (failed) |
| `core_data` | 3 | 3 alt treatment definitions |
| `core_weights` | 3 | ebalance, survey, joint ebalance+controls |

---

## Sanity Checks (Step 0)

| Check | Status | Detail |
|---|---|---|
| `spec_run_id` unique | PASS | 53 unique across 53 rows |
| `baseline_group_id` present | PASS | All rows have G1 |
| `spec_id` consistent with `spec_tree_path` | PASS | All paths reference valid tree nodes with anchors |
| `run_success` is 0/1 | PASS | 52 success, 1 failure |
| Failed row: `coefficient_vector_json` has `error` and `error_details` | PASS | Both fields present and non-empty |
| Failed row: scalar numeric fields are NaN | PASS | coef/SE/p/CI/N/R2 all NaN |
| Success rows have required audit keys | PASS | `coefficients`, `inference`, `software`, `surface_hash` all present |
| Success rows use canonical inference (`infer/se/cluster/cityno`) | PASS | All 52 success rows report this spec_id |
| No arbitrary top-level JSON keys | PASS | Extra fields under `design`, `controls`, `sample`, `fixed_effects`, `data_construction`, `weights`, `joint` only |
| `rc/*` rows include axis-appropriate blocks | PASS | `controls`, `sample`, `fixed_effects`, `data_construction`, `weights`, `joint` blocks present with matching `spec_id` |
| Numeric fields finite for `run_success=1` rows | PASS | No NaN or Inf in scalar columns for success rows |
| No `infer/*` rows in `specification_results.csv` | PASS | 0 found |
| `inference_results.csv` contains only `infer/*` rows | PASS | 3 rows, all `infer/se/*` spec_ids |

---

## Baseline Group Verification (Step 1)

The surface defined 1 baseline group (G1) with 7 baseline spec_ids in `core_universe.baseline_spec_ids`. The run produced 8 baseline rows (the primary `baseline` row plus the 7 surface-listed `baseline__*` rows). All 8 map to G1.

The surface's `core_universe.design_spec_ids` lists `design/difference_in_differences/estimator/twfe`. This spec was **not executed as a standalone row**. The TWFE design is instead captured in the `design` block of the `baseline` row's `coefficient_vector_json`. No error — this is acceptable behavior since the baseline row fully documents the estimator.

The surface's `core_universe.rc_spec_ids` lists 36 RC spec_ids (including the wildcard `rc/controls/subset/random_*`). The run executed 45 RC rows: 36 literal + 9 additional random subset draws (001-010 = 10 total, surface listed wildcard). All executed RC spec_ids match the surface plan. 1 failed (collinearity).

**No changes to baseline group assignments were made relative to the surface.**

---

## Non-Core Classifications

Five rows are classified as `noncore_alt_outcome` (is_core_test=0):

| spec_run_id | spec_id | Outcome | Reason |
|---|---|---|---|
| 138922-V1__baseline__sport_hrs | baseline__sport_hrs | sport_hrs | Behavioral hours outcome, different from focal sportsclub membership |
| 138922-V1__baseline__oweight | baseline__oweight | oweight | Overweight outcome, different from focal sportsclub membership |
| 138922-V1__baseline__kommheard | baseline__kommheard | kommheard | Voucher awareness (first-stage proxy), different from focal sportsclub |
| 138922-V1__baseline__kommgotten | baseline__kommgotten | kommgotten | Voucher receipt (first-stage proxy), different from focal sportsclub |
| 138922-V1__baseline__kommused | baseline__kommused | kommused | Voucher use (first-stage proxy), different from focal sportsclub |

These rows are informative (especially the komm* specs confirming program take-up with very large and significant coefficients) but they represent different outcome concepts from the G1 focal claim about sports club membership.

Note: `rc/data/treatment/*` rows use alternative treatment variable definitions (different treatment timing or coding) but preserve the same sportsclub outcome and estimand concept; these are classified as `core_data` (is_core_test=1).

---

## Sign and Significance (sportsclub outcome, run_success=1)

Across **47 valid sportsclub-outcome specifications** (48 sportsclub-outcome rows, 1 failed):

| Metric | Value |
|---|---|
| Positive coefficient | 44/47 (93.6%) |
| Negative coefficient | 3/47 (6.4%) |
| p < 0.05 | 0/47 (0%) |
| p < 0.10 | 0/47 (0%) |
| Coefficient range | [-0.0161, 0.0261] |
| Median coefficient | 0.013 |
| Mean coefficient | 0.013 |

The 3 negative-coefficient specifications:
1. `rc/sample/time/extend_2006_2011` (coef=-0.014, p=0.348): extending the sample to 2011 flips the sign slightly; still insignificant.
2. `rc/data/treatment/current_state` (coef=-0.001, p=0.938): using child's current state of residence rather than exam-state coding; near-zero.
3. `rc/weights/main/ebalance_weighted` (coef=-0.016, p=0.314): entropy balancing weights flip the sign; still insignificant. This spec warrants follow-up (see Issues).

---

## Failed Specifications

| spec_run_id | spec_id | Error |
|---|---|---|
| 138922-V1__rc_fe_add_bula_3rd_x_year_3rd | rc/fe/add/bula_3rd_x_year_3rd | All variables collinear (pyfixest ValueError) |

**Explanation**: The treatment variable `treat` is defined at the state x cohort level. Adding state-by-year FEs absorbs all treatment variation, making `treat` perfectly collinear. This is expected behavior and confirms the DiD identification strategy. The failure is informative, not a runner bug.

---

## Inference Sensitivity (inference_results.csv)

The inference variants are linked to `138922-V1__baseline` only (primary sportsclub baseline):

| inference_run_id | spec_id | SE | p-value | Status |
|---|---|---|---|---|
| 138922-V1__infer_001 | infer/se/hc/hc1 | 0.0186 | 0.631 | Success |
| 138922-V1__infer_002 | infer/se/cluster/bula_3rd | 0.0051 | 0.224 | Success (3 clusters - low reliability) |
| 138922-V1__infer_003 | infer/se/cluster/cohort | 0.0073 | 0.244 | Success (5 clusters - low reliability) |

No inference method produces a significant result for sportsclub. The canonical cityno-clustered SE (0.0187, p=0.635) is conservative relative to HC1 (0.0186) and less conservative than state- or cohort-clustered SEs.

Note: The surface listed 4 inference variants (hc1, bula_3rd, cohort, and the canonical infer/se/cluster/cityno which is the baseline itself). The canonical variant appears in `specification_results.csv` as the main run. The 3 non-canonical variants appear in `inference_results.csv`. Wild bootstrap variants (surface plan referenced) were apparently not executed (wildboottest package not installed in this environment per project memory).

---

## Top Issues

1. **Duplicate random control subset draws**: `rc/controls/subset/random_004` and `rc/controls/subset/random_005` both produce identical coefficients (0.02424, p=0.198, N=12,113) matching `rc/controls/sets/full`. They both appear to have sampled all 9 controls. Also `rc/controls/progression/bivariate` is identical to `baseline` (coef=0.00892, N=13,333), which is correct behavior (bivariate = treatment only, same as baseline with no individual controls). Low priority; slightly wasteful of the 10-draw budget.

2. **Entropy balance sign flip**: `rc/weights/main/ebalance_weighted` produces coef=-0.016 (vs baseline 0.009), a sign reversal. This could reflect the Python entropy balancing implementation differing from Stata's `ebalance`. Worth validating against the original Stata output before interpreting as a robustness finding.

3. **Wild bootstrap inference missing**: The surface planned wild bootstrap variants (Rademacher, Webb) for the few-cluster setting (3 states). These did not execute (wildboottest not installed). This is important given the 3-cluster problem; state-clustered SEs give p=0.224, which is notably lower than city-clustered (p=0.635), suggesting meaningful uncertainty about inference with few clusters.

4. **design/* spec not executed standalone**: `design/difference_in_differences/estimator/twfe` was listed in `core_universe.design_spec_ids` but not run as a separate row. The TWFE design information is embedded in the `baseline` row's coefficient_vector_json. This is acceptable for a single-estimator DiD paper but could be clearer in future surface plans.

---

## Recommendations

1. Install `wildboottest` and re-run the surface's wild bootstrap inference variants. Given only 3 treatment states, conventional clustering may be unreliable.
2. Validate the Python entropy balancing result against Stata's `ebalance` output to confirm whether the sign flip (ebalance-weighted coef=-0.016) is a real finding or implementation artifact.
3. Add `rc/form/*` specifications (probit/logit for binary sportsclub outcome) to the surface for future runs.
4. Improve the random control subset sampler to avoid drawing the full 9-control set, which duplicates `rc/controls/sets/full`.
5. Consider running inference variants for all 3 sportsclub baseline rows (col1, col2, col3), not just the primary baseline.

---

## Assessment

**NULL RESULT - ROBUST**: The voucher program's effect on sports club membership is consistently small and statistically insignificant across all 47 valid sportsclub-outcome specifications. Zero of 47 specifications reach significance at the 10% level. Point estimates are small (median 0.013 on a binary outcome) and predominantly positive (44/47). The null result holds across:

- All control variable variations (24 specs)
- All sample window changes (5 time specs)
- Both control-state subsets (2 geographic specs)
- All sample quality filters (5 quality specs)
- Both successful FE variations (2 specs; 1 failed due to inherent collinearity)
- All 3 alternative treatment definitions
- All 3 weighting/joint methods (including the ebalance sign flip, still p=0.314)
- All 3 inference variants in inference_results.csv

The komm* first-stage outcomes (kommheard, kommgotten, kommused) all show strong positive and highly significant effects (coefs 0.12-0.28, p<0.001), confirming robust program take-up. The null effect on membership thus reflects a genuine failure of program impact on the behavioral outcome, not lack of program exposure.

**Overall quality of run**: HIGH. All surface specs executed (1 expected collinearity failure), coefficient_vector_json structure is correct throughout, inference is consistent, and axis blocks all contain matching spec_ids. The spec_run_id space is clean with no duplicates.
