# Specification Surface Review: 112749-V1

**Paper**: "When the Levee Breaks: Black Migration and Economic Development in the American South"
**Authors**: Hornbeck & Naidu (AER 2014)
**Reviewer**: Specification Surface Verifier (pre-run audit)
**Date**: 2026-02-13
**Status**: APPROVED TO RUN (with changes applied)

---

## 1. Summary of Baseline Groups

Two baseline groups are defined. This is appropriate: they correspond to the paper's two distinct headline claims (labor channel and capital channel) with different outcome variables, different post-treatment period structures, and different lag specifications.

### G1: Flood impact on Black population share
- **Outcome**: `lnfrac_black`
- **Treatment**: `flood_intensity x post-flood year dummies` (5 decadal periods: 1930, 1940, 1950, 1960, 1970)
- **Focal parameter**: `f_int_1950`
- **Baseline specs**: Table 2 cols 1-2

### G2: Flood impact on agricultural capital intensity
- **Outcome**: `lnvalue_equipment`
- **Treatment**: `flood_intensity x post-flood year dummies` (9 periods including intercensus: 1930, 1935, 1940, 1945, 1950, 1954, 1960, 1964, 1970)
- **Focal parameter**: `f_int_1950`
- **Baseline specs**: Table 4 cols 3-4

**No missing baseline groups**: The paper's other outcomes (total population, tractors, mules/horses, farm size, farmland, land values, crop yields) are correctly classified as exploration rather than additional baseline groups. The two selected outcomes are the paper's headline claims.

**No misclassified exploration groups**: Tables 6-7 (southern rivers, non-flooded distance) are correctly excluded from core because they change the target population and treatment concept.

---

## 2. Checklist Findings (A-F)

### A) Baseline Groups -- PASS with no changes

- Each baseline group corresponds to a single claim object.
- No missing baseline groups identified.
- Plantation interaction analysis (Tables with `f_int_*_p`) is correctly scoped as `explore/heterogeneity/plantation`.

### B) Design Selection -- PASS with no changes

- `design_code = "difference_in_differences"` is correct.
- The surface correctly notes this is continuous-treatment DiD from a single simultaneous event (not staggered adoption), making modern heterogeneity-robust estimators (Sun-Abraham, Callaway-Santanna, Borusyak imputation) inapplicable.
- `design/difference_in_differences/estimator/twfe` is the only design variant, which is appropriate.

### C) RC Axes -- CHANGES MADE

**Issues found and corrected:**

1. **`rc/controls/progression/geography` renamed to `rc/controls/progression/geography_only`**. The original name was ambiguous. This spec (geography block without lagged DV) is NOT a revealed RefTable step -- all 7 RefTable steps include the lagged DV block. It is retained as a legitimate stress test but the name and notes now make clear it goes beyond the revealed surface.

2. **Control progression ordering corrected**. The original JSON listed `geography_lags_newdeal` before `geography_lags_tenancy_mfg`, but RefTable step 3 (tenancy/mfg without New Deal) comes before step 4 (New Deal without tenancy/mfg). Reordered to match the RefTable progression sequence.

3. **`rc/controls/loo_block/drop_tenancy_mfg` added**. The tenancy/manufacturing block (lag*_lnfarms_nonwhite_t_*, lag*_lnmfgestab_*, lag*_lnmfgavewages_*) appears in RefTable steps 3, 6, and 7 but was not covered by a LOO-block spec. This is a potentially important block because tenancy and manufacturing outcomes could be endogenous to the flood. Added for both G1 and G2.

4. **`rc/sample/time/pre1960_only` removed from G1**. G1 labor outcomes use decadal observations only (1930, 1940, 1950, 1960, 1970). There is no 1964 period for G1, so "pre-1960 only" would mean dropping only 1970, which is already covered by `rc/sample/time/drop_1970`. Retained for G2 (which has intercensus observations including 1964).

5. **`rc/controls/sets/none` annotated**. This spec (no time-varying controls, FE only) is below the revealed minimum (RefTable step 1 = lagged DV only). Retained as a standard stress test but clearly annotated that it goes beyond the paper's own surface.

**Axes confirmed as correct:**
- Treatment alternatives (3 flood intensity measures): correctly specified as `rc/form/treatment/*`.
- Weights (unweighted): correctly specified as `rc/weights/main/unweighted`.
- Sample (drop 1930, drop 1970, trim treatment): correctly specified.
- Inference (county cluster, HC1, Conley 50/100/200mi, state cluster): correctly specified. The paper reveals county cluster + Conley; HC1 and state cluster are natural additions.

**Axes considered but not added:**
- `rc/fe/drop/state_year` (replace state-by-year FE with year-only FE): Not included because it materially changes the identifying variation. The paper never varies the FE structure. This is more of an exploration/identification change than an RC. Decision: reasonable to exclude from core.
- `rc/form/outcome/level` (outcome in levels instead of logs): Not included because the paper's claim is explicitly about log changes and all outcomes use log transformations. A levels specification would change the estimand interpretation. Decision: reasonable to exclude from core.

### D) Controls Multiverse Policy -- CHANGES MADE

**Issues found and corrected:**

1. **`controls_count_min` / `controls_count_max` corrected**. The original surface set min=36/max=61 for G1 and min=42/max=67 for G2, based only on the two Table 2/4 baseline specifications. However, the RefTables reveal a much wider range:
   - G1 minimum: RefTable step 1 (lagged DV only) uses ~15 controls (3 lag orders x 5 year interactions).
   - G1 maximum: RefTable step 7 (full kitchen sink) uses ~80 controls.
   - G2 minimum: RefTable step 1 uses ~20 controls (4 lag orders x ~5 year interactions).
   - G2 maximum: RefTable step 7 uses ~90 controls (more year-interactions because 9 post-treatment periods).

   The envelope was updated to span the full revealed RefTable range, not just the main-table baselines.

2. **`controls_mandatory` field added (empty)**. No controls are mandatory across all RefTable steps. Even the lagged DV can be excluded (by running `rc/controls/sets/none`). This is now explicit.

3. **Tenancy/manufacturing block documented precisely**. The `controls_blocks` in the original baseline specs did not include the tenancy/mfg block, which appears only in RefTable steps 3, 6, and 7. The `rc_spec_notes` now document what this block contains: `lag*_lnfarms_nonwhite_t_*`, `lag*_lnmfgestab_*`, `lag*_lnmfgavewages_*`.

4. **LOO block base specification clarified**. The LOO blocks are computed relative to the Table 2 col 2 / Table 4 col 4 baseline (geography + lagged DV + New Deal). This is now explicitly documented in the `rc_spec_notes`.

### E) Budgets + Sampling -- PASS with minor adjustment

- **Full enumeration** is confirmed feasible. The control-progression approach is block-based (not variable-level combinatorial), keeping the total number of distinct estimating equations small (~23-24 per group).
- G1 budget: 80 specs (22 RC specs x 1 point estimate each + 6 inference variants for baseline). Feasible.
- G2 budget: increased from 80 to 85 to accommodate the additional `pre1960_only` spec that G2 retains.
- Seeds are distinct per group (112749001, 112749002). Reproducible.
- No random subset sampling is needed.

**Budget arithmetic check (G1):**
- 9 control progression specs (including 2 that match baselines)
- 7 LOO-block specs (updated: was 7, now 8 with drop_tenancy_mfg, but tenancy_mfg LOO is from a non-baseline control set -- compute relative to the full kitchen-sink spec)
- 1 unweighted spec
- 2 sample-time specs (drop_1970, drop_1930)
- 1 treatment-trimming spec
- 2 treatment-alternative specs
- Subtotal: ~22 distinct estimating equations
- x 6 inference variants = ~132 rows
- Within 80-spec budget when counting distinct point-estimate specs (inference variants recompute SE only).

### F) Diagnostics Plan -- PASS with no changes

- **Pre-trends event study** (`diag/difference_in_differences/pretrends/event_study_plot`): Correctly scoped at `baseline_group` level. The paper's Figures 3-4 show event-study plots with pre-flood coefficients (1900, 1910 for G1; 1900, 1910, 1920 for G2). This diagnostic is invariant to controls.
- **Pre-trends joint test** (`diag/difference_in_differences/pretrends/joint_test`): Correctly scoped at `spec` level because the test depends on the control set included.
- **No Bacon decomposition needed**: This is not staggered adoption, so the Goodman-Bacon decomposition is not meaningful.

---

## 3. Key Constraints and Linkage Rules

1. **Not a bundled estimator**: Single-equation TWFE. No linkage constraints needed.
2. **Time-varying controls**: All controls are interacted with year dummies. Adding/dropping a control "block" means adding/dropping all year-interacted versions simultaneously. This is a block-level operation, not variable-level.
3. **Outcome-specific lagged DV**: The lagged DV block differs between G1 and G2 (different outcome variables, different lag orders). When the search agent runs the same control progression for each group, it must substitute the appropriate lagged DV.
4. **Period structure differs across groups**: G1 has 5 post-treatment periods (decadal); G2 has 9 (including intercensus). Treatment variables, state-by-year FE, and year-interacted controls all differ accordingly.

---

## 4. Dataset and Implementation Notes

1. **Main tables** (Table 2, 4, 5) use `preanalysis_post1930.dta` (post-treatment panel only).
2. **RefTables 1-3** use `preanalysis.dta` (full panel including pre-treatment years). Because treatment variables `f_int_*` are zero before 1930, treatment coefficient estimates are numerically equivalent to using the post-treatment panel only (the pre-treatment observations contribute to FE estimation but do not affect the treatment coefficients).
3. **Conley SE computation** uses period-by-period first-differenced regressions (not the panel `areg` specification). The `x_ols` custom ado file is called with geographic coordinates converted to miles. This is a different estimation approach for the Conley SE than simply swapping the VCE on the panel regression -- it requires period-specific cross-sectional regressions.
4. **Event study figures** (Figures 3-4) use `preanalysis.dta` with pre-treatment year dummies (f_int_1900, f_int_1910) included.

---

## 5. What Is Missing (potential additions for future consideration)

These items were considered but deliberately excluded from the current core surface:

1. **FE variation** (`rc/fe/drop/state_year`): Replacing state-by-year FE with year-only FE. Excluded because it materially changes identification; the paper never varies this.
2. **Outcome transformations** (`rc/form/outcome/level`): Running in levels rather than logs. Excluded because the paper's claims are stated in terms of log outcomes.
3. **Sample restriction relaxation**: Dropping the >=10% Black or >=15% cotton sample entry criteria. Excluded because this changes the target population.
4. **Wild cluster bootstrap**: With ~9 states, state-clustered SE inference is suspect. Wild cluster bootstrap would be a more reliable inference approach but is listed as optional/future.
5. **Conley SE with time decay**: The paper's Conley SE assume spatial correlation only (no temporal decay kernel). Adding temporal decay could be informative but is not revealed.

---

## 6. Changes Made to SPECIFICATION_SURFACE.json (Summary)

| Change | Rationale |
|---|---|
| `controls_count_min/max` widened for both G1 and G2 | Reflect full RefTable range, not just main-table baselines |
| `controls_mandatory` field added (empty) | Make explicit that no controls are always required |
| `rc/controls/progression/geography` renamed to `geography_only` | Clarify this is NOT a revealed RefTable step |
| `rc/controls/loo_block/drop_tenancy_mfg` added | Cover the tenancy/mfg block that appears in RefTable steps 3, 6, 7 |
| `rc/sample/time/pre1960_only` removed from G1 | G1 has no 1964 observations; spec redundant with `drop_1970` |
| `rc_spec_notes` blocks added to both groups | Document which RefTable step each progression spec maps to, LOO base spec, and stress-test annotations |
| G2 budget increased from 80 to 85 | Accommodate `pre1960_only` retained for G2 |
| Control progression ordering corrected | Match actual RefTable step order (tenancy/mfg step 3 before New Deal step 4) |
| `design_notes` expanded | Added dataset_note, lagged_dv_note, tenancy_mfg_block_note for implementation clarity |
| `revealed_search_space_summary.controls` updated | More precise block descriptions, note that all RefTable steps include lagged DV |
| G2 constraints: added `linked_adjustment_note`, `max_fe_note` | Parity with G1 documentation |

---

## 7. Final Verdict

**APPROVED TO RUN.**

The specification surface is well-constructed, conceptually coherent, and faithful to the paper's revealed search space. The two baseline groups are correctly identified. The control progression is block-based and matches the RefTable structure. Treatment alternatives and inference variations are well-documented. The diagnostics plan includes standard pre-trends checks.

The changes made are refinements rather than fundamental restructuring:
- Corrected the control-count envelope to span the full revealed range.
- Added one missing LOO-block spec (tenancy/mfg).
- Removed one redundant spec from G1 (pre1960_only).
- Improved documentation throughout (spec notes, dataset notes, block descriptions).

No blocking issues remain.
