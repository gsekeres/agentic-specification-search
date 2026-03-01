# Specification Surface Review: 158401-V1

**Paper**: "Market Access and Quality Upgrading: Evidence from Four Field Experiments"
**Authors**: Tessa Bold, Selene Ghisolfi, Frances Nsonzi, and Jakob Svensson
**Design**: Cluster-randomized experiment (village-level, ITT, ANCOVA)
**Reviewer**: Specification Surface Verifier Agent
**Date**: 2026-02-24

---

## A) Baseline Groups

### Summary

Two baseline groups, both for the Market Access experiment (sample frame 1):

| Group | Label | Table | Outcomes |
|-------|-------|-------|----------|
| G1 | Market Access -> Investment | Table 5, Panel A | 8 investment outcomes (5 continuous monetary, 3 binary) |
| G2 | Market Access -> Productivity/Income | Table 6, Panel A | 8 productivity/income outcomes (all continuous) |

### Assessment

**Well-defined.** Each baseline group corresponds to a single claim object (same treatment, same population, same estimand concept, distinct outcome families). The split into investment vs. productivity/income is well-motivated: Table 5 targets the behavioral mechanism channel (quality upgrading investment), while Table 6 targets the welfare/reduced-form effect (productivity and income).

**No changes needed** to baseline group definitions.

### Exclusions (verified as correct)

- **Extension Service experiment (Panel B)**: Correctly excluded -- serves as comparison/falsification, not a primary claim.
- **Table 4 (maize quality)**: Correctly excluded -- different dataset (`quality_for_analysis.dta`), different unit (bags of maize), bounds estimators.
- **Table 7 (traders)**: Correctly excluded -- seller-level data, different claim object.
- **Appendix Table 12 (production function)**: Correctly excluded -- mediation analysis, not a direct ITT.
- **Appendix Table 14 (selection-adjusted)**: Correctly excluded -- changes the estimand.

---

## B) Design Selection

### Design code: `randomized_experiment`

**Correct.** Confirmed from CLASSIFICATION.json (high confidence) and the code. Treatment (`buy_treatment`) is randomly assigned at the village level (12 of 20 villages). ANCOVA specification with season FE and village-clustered SEs.

### Design audit

**Changes made:**
1. **Added `n_clusters: 20`** -- critical for interpreting clustered SE reliability. With only 20 clusters (12T, 8C), standard cluster-robust SEs may be imprecise.
2. **Expanded `ancova_control` description** -- now explicitly states the construction rule: "mean at last baseline season, fallback to pooled baseline" matching the code logic.
3. **Added `post_treatment_seasons` and `post_treatment_labels`** -- seasons 4-7 (fall2018 through spring2020) are post-treatment for group 1. This is needed for the time-drop RC specs.

### Design variants

| spec_id | Assessment |
|---------|-----------|
| `design/randomized_experiment/estimator/diff_in_means` | **Keep.** Pure randomization-based estimate without ANCOVA or season FE. Aggressive but standard in RCT robustness. |
| `design/randomized_experiment/estimator/with_covariates` | **Keep.** ANCOVA + 5 household characteristics. Standard precision-enhancing approach. |

No missing design variants. The `design/randomized_experiment/estimator/ancova` variant was correctly omitted since ANCOVA is the baseline estimator itself.

---

## C) RC Axes (Core Robustness)

### Changes made

1. **Removed `rc/controls/loo/drop_ancova`**: This was redundant with `rc/controls/sets/none`. For outcomes with ANCOVA controls, dropping the ANCOVA control is functionally identical to "treatment + season FE only" (which is `rc/controls/sets/none`). For `sort_d` and `winnow_d` (no ANCOVA), it is a no-op. Removing this avoids running duplicate specifications.

2. **Added explicit `rc_notes` block**: Documents applicability restrictions for each RC spec:
   - Functional form transforms (asinh/log1p): only for continuous monetary outcomes, not binary
   - Trimming: for G1 this is analyst-added (paper only trims Table 6 outcomes in Appendix Table 11)
   - For G2 trimming: replicates Appendix Table 11 rule exactly (non-negative continuous at top 1% per season; profit variables at top and bottom 1% per season)

### Assessment of included axes

| RC axis | spec_ids | Assessment |
|---------|----------|-----------|
| Controls: none | `rc/controls/sets/none` | **Keep.** Tests sensitivity to ANCOVA adjustment. |
| Controls: extended | `rc/controls/sets/extended_hh_chars` | **Keep.** Adds 5 pre-treatment HH characteristics from Table 3 balance tests. |
| Sample: trim | `rc/sample/outliers/trim_y_1_99` | **Keep.** For G2, replicates Appendix Table 11. For G1, analyst-added but defensible for continuous monetary outcomes. |
| Sample: drop first post-season | `rc/sample/time/drop_first_post_season` | **Keep.** Drop season 4 (fall 2018); tests whether effects driven by initial-season novelty. |
| Sample: drop last post-season | `rc/sample/time/drop_last_post_season` | **Keep.** Drop season 7 (spring 2020, collected during COVID); tests data quality concerns. |
| Sample: balanced panel | `rc/sample/panel/balanced_only` | **Keep.** Tests attrition sensitivity. |
| Form: asinh | `rc/form/outcome/asinh` | **Keep with restriction.** Apply only to continuous monetary outcomes. Preserves sign; approximate semi-elasticity for large values. |
| Form: log1p | `rc/form/outcome/log1p` | **Keep with restriction.** Same restriction as asinh. Alternative transformation. |

### Missing axes considered

- **Winsorization** (`rc/preprocess/outliers/winsor_y_1_99`): Not included. Trimming is sufficient and matches the paper's approach. Could be added if budget allows.
- **Alternative ANCOVA control construction** (e.g., pooled baseline instead of last-season): Not included. The code's fallback logic already handles this, so there is limited variation to exploit.
- **Data construction choices**: The variable construction in `FINAL_create_vars_main_g1.do` involves many researcher choices (season-specific exchange rates, imputation rules for missing costs, variable component aggregation). These are high-leverage but extremely paper-specific and difficult to vary systematically. Noted as a limitation but not added to the surface.

---

## D) Controls Multiverse Policy

### Assessment

**Correct.** The controls count envelope [0, 6] is derived correctly:
- Baseline: 0-1 controls (ANCOVA only, or no ANCOVA for sort_d/winnow_d)
- Extended: ANCOVA + 5 household characteristics = 6 controls max

**No linked adjustment**: Correct. Each outcome's ANCOVA control is outcome-specific (e.g., `expenses_fert_seeds_p3` for `expenses_fert_seeds`), so there is no shared covariate set forcing linkage across outcomes.

**Mandatory vs optional controls**: The surface implicitly treats the ANCOVA control as "part of the baseline" and the 5 household characteristics as optional additions. This is correct -- the paper's canonical spec uses ANCOVA, and the extended set adds household chars.

---

## E) Inference Plan

### Canonical inference

**Correct.** Village-clustered SE (`ea_code`), matching the paper's baseline. Village is the randomization unit.

**Warning noted**: With only 20 clusters (12T, 8C), cluster-robust SEs may understate true uncertainty. The HC1 variant provides a useful stress test.

### Inference variants

| Variant | Assessment |
|---------|-----------|
| `infer/se/hc/hc1` | **Keep.** Important given small cluster count. |
| `infer/ri/fisher/permutation` | **Keep.** Matches paper's supplementary inference exactly. |

**Correction made**: The permutation seed was listed as 760130 for both G1 and G2. Verified from `permutations.do`: seed 760130 is for group 1 (Market Access experiment), and seed 880805 is for group 2 (Extension Service experiment). Since both G1 and G2 use the Market Access experiment (group 1), seed 760130 is correct for both. No change needed -- the original was correct.

---

## F) Budgets + Sampling

### Changes made

1. **Updated budget from 60 to 64 per group**: The previous budget (60 per group) was loosely estimated. The precise enumeration yields 35 (G1) + 36 (G2) = 71 total specs. Budget set to 64 per group to provide headroom.

2. **Updated spec_enumeration to match rc_spec_ids**: The original enumeration plan omitted 3 RC variants that were listed in `rc_spec_ids`: `rc/sample/time/drop_first_post_season`, `rc/sample/time/drop_last_post_season`, and `rc/form/outcome/log1p`. These are now included in the enumeration plan with correct counts accounting for binary/non-monetary exclusions.

### Updated enumeration

| Category | G1 | G2 | Total |
|----------|----|----|-------|
| Baseline specs | 8 | 8 | 16 |
| design/diff_in_means (3 focal) | 3 | 3 | 6 |
| design/with_covariates (3 focal) | 3 | 3 | 6 |
| rc/controls/none (3 focal) | 3 | 3 | 6 |
| rc/controls/extended (3 focal) | 3 | 3 | 6 |
| rc/sample/trim (2 monetary G1 / 3 focal G2) | 2 | 3 | 5 |
| rc/sample/drop_first_season (3 focal) | 3 | 3 | 6 |
| rc/sample/drop_last_season (3 focal) | 3 | 3 | 6 |
| rc/sample/balanced_only (3 focal) | 3 | 3 | 6 |
| rc/form/asinh (2 monetary focal) | 2 | 2 | 4 |
| rc/form/log1p (2 monetary focal) | 2 | 2 | 4 |
| **Total** | **35** | **36** | **71** |

Full enumeration is feasible. No sampling needed.

### Focal outcome selection (unchanged)

- **G1**: expenses_fert_seeds (continuous monetary), tarpaulin_d (binary), expenses_postharvest (continuous monetary)
- **G2**: surplus (continuous monetary), harvest_value (continuous monetary), yield (continuous non-monetary)

**Note**: For functional form transforms (asinh/log1p) and trimming in G1, binary/non-monetary focal outcomes are correctly excluded, reducing counts from 3 to 2 for those RC categories.

---

## G) Diagnostics Plan

### Assessment

**Appropriate.** Two diagnostics for G1:
1. Balance test (replicates Table 3) -- scope: baseline_group
2. Attrition differential (replicates Appendix Table 6) -- scope: baseline_group

G2 includes only the balance test. **Recommendation**: Also add the attrition diagnostic for G2, since the same paper reports attrition results for both experiments. However, since G1 and G2 share the same experiment and sample, the attrition test from G1 technically covers G2 as well. The current setup is acceptable.

Both diagnostics have `scope: baseline_group`, which is correct -- they depend on the experimental design/sample, not on specific control sets.

---

## Summary of All Changes Made

1. **Removed `rc/controls/loo/drop_ancova`** from both G1 and G2 `rc_spec_ids` (redundant with `rc/controls/sets/none`).
2. **Added `n_clusters: 20`** to both design_audit blocks.
3. **Expanded `ancova_control` description** in both design_audit blocks.
4. **Added `post_treatment_seasons` and `post_treatment_labels`** to both design_audit blocks.
5. **Added `rc_notes` blocks** documenting applicability restrictions for transforms, trimming, and the drop_ancova removal rationale.
6. **Added Fisher RI seed provenance note** (seed 760130 confirmed from permutations.do for group 1).
7. **Updated cluster count in inference notes** (20 villages: 12T, 8C).
8. **Updated budgets** from 60 to 64 per group, with corrected arithmetic.
9. **Updated spec_enumeration** to include previously omitted time-drop and log1p specs, with correct per-category counts reflecting binary/non-monetary exclusions. Total target: 71 (was 56).

---

## What's Missing (non-blocking)

1. **Data construction robustness**: The variable construction in `FINAL_create_vars_main_g1.do` involves many researcher choices (exchange rate conversion, component aggregation, imputation of missing costs, phone survey handling). These are high-leverage preprocessing decisions that could be varied, but doing so requires deep paper-specific knowledge and is out of scope for the standard RC axes.

2. **COVID sensitivity**: Season 7 (spring 2020) was partially collected by phone survey during COVID (`form_used=="Phone_Market_Survey_spring2020"`), with a reduced set of variables. The `rc/sample/time/drop_last_post_season` spec partially addresses this. A more targeted sensitivity analysis could restrict to non-phone-survey observations, but this is complex to implement.

3. **Extension Service experiment as falsification**: The Panel B results (ext_treatment) could serve as a useful placebo/falsification check. Not included in the core surface but could be added as `explore/*` or `diag/*`.

4. **Multiple hypothesis testing adjustment**: With 8 outcomes per group, MHT correction (e.g., Bonferroni, Holm, FDR) would be appropriate as a `post/*` object. Not included in the core surface.

---

## Verdict

**APPROVED TO RUN.** The specification surface is conceptually coherent, statistically principled, faithful to the paper's design, and auditable. The 71-spec budget across 2 baseline groups (16 baseline + 55 RC/design) provides adequate coverage of the key robustness dimensions for this cluster-RCT. All changes made are conservative and improve precision/auditability without changing the fundamental structure.
