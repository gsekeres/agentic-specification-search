# Specification Surface Review: 111185-V1

**Paper**: "Optimal Climate Policy When Damages are Unknown" (Ivan Rudik, AEJ: Economic Policy 2020)
**Reviewed**: 2026-02-13
**Status**: APPROVED TO RUN (with changes applied)

---

## 1. Summary of Baseline Groups

The surface defines a single baseline group (G1), which is correct. The paper contains exactly one reduced-form regression: Table 1, OLS of log damages on log temperature (N=43) from the Howard & Sterner (2017) meta-analysis dataset. All other results are structural model outputs requiring >20,000 core-hours and are out of scope.

**Claim object**: Well-defined. The outcome, treatment, estimand (log-log elasticity = power-law exponent d2), and target population are all clearly specified.

**No missing baseline groups**: There is genuinely only one regression in this paper.

---

## 2. Changes Made to the Surface

### A) Removed duplicate spec_ids (3 duplicates found)

1. **`rc/controls/sets/none`**: Removed. The baseline is already bivariate (0 controls), so this spec would produce an identical regression. It is the baseline itself.

2. **`rc/controls/progression/bivariate`**: Removed. Same issue -- the first step of the control progression is the bivariate baseline.

3. **`rc/sample/outliers/drop_high_influence`**: Removed. This was explicitly described as "Same as drop_high_influence" in the `cooksd_4_over_n` entry. Both specs drop observations with Cook's D > 4/N. Retained `cooksd_4_over_n` as the canonical spec_id since it matches the tree's standard naming convention (`specification_tree/modules/robustness/sample.md`).

### B) Reclassified 3 specs from core (`rc/*`) to exploration (`explore/*`)

1. **`rc/form/outcome/level` -> `explore/form/outcome/level`**: A levels regression (correct_d on t) estimates a fundamentally different parameter from the baseline. The baseline estimand is the log-log elasticity d2, the power-law exponent. A levels regression coefficient is NOT d2. It is the slope d(correct_d)/dt, which equals d1*d2*t^(d2-1) and depends on the evaluation point t. This is an estimand change per ARCHITECTURE.md Section 2.H and CLAIM_GROUPING.md Section 5: "If outcome_var changes in a way that changes the outcome concept -> explore/*."

2. **`rc/form/model/levels_quadratic` -> `explore/form/model/levels_quadratic`**: Similarly, a quadratic-in-levels specification (correct_d on t + t^2) estimates D(T) = a*T + b*T^2 -- the DICE/Nordhaus functional form. This is a completely different damage function parameterization from the power-law D(T) = d1*T^d2. The coefficients are not comparable to d2 without substantial reinterpretation.

3. **`rc/preprocess/treatment/include_zero_damage_obs` -> `explore/preprocess/treatment/include_zero_damage_obs`**: This spec simultaneously (a) changes the outcome transformation from log to asinh, and (b) changes the sample by adding 3 previously excluded zero-damage observations. Both changes affect the estimand: asinh-elasticity differs from log-elasticity, and the target population changes. This is a joint estimand+population change.

**Rationale**: The ARCHITECTURE.md is clear that "core = estimand-preserving." The functional form module (`specification_tree/modules/robustness/functional_form.md`) notes: "If the paper's claim is inherently about percent changes/elasticities (or is written in logs), [log/asinh variants] are RC." But here the claim is specifically about the log-log elasticity d2, and the levels specs produce coefficients with completely different interpretations. The conservative default per CLAIM_GROUPING.md is to classify as exploration when unsure.

### C) Added weighting axis (new: `rc/weights/*`)

Added two specs:
- `rc/weights/main/unweighted`: Explicit reference (identical to baseline).
- `rc/weights/main/wls_inverse_variance_proxy`: WLS using precision weights if available in the Howard & Sterner dataset.

**Rationale**: Weighting is a standard and high-leverage axis in meta-regression. The baseline is unweighted OLS, which gives equal weight to all 43 study estimates regardless of their precision. In the meta-analysis literature, WLS (weighted by inverse variance of each study's estimate or a precision proxy) is the standard approach. This was a notable omission from the original surface.

**Feasibility note**: Whether this is feasible depends on whether the H&S dataset contains study-level standard errors, confidence interval widths, or sample sizes that could serve as precision proxies. The spec search agent should check the dataset variables (there are 146 variables) and implement this if feasible.

### D) Added Ramsey RESET to diagnostics plan

The original diagnostics plan included Cook's D, Jarque-Bera, and Breusch-Pagan but omitted Ramsey RESET. Given that (a) the regression is bivariate, (b) functional form is a key axis, and (c) the quadratic treatment spec explicitly tests for nonlinearity, RESET is a natural diagnostic to include. Added `diag/regression/specification/ramsey_reset`.

### E) Resolved controls_count_max inconsistency

The original surface had `controls_count_max: 4` as a hard cap, but then included specs like `rc/controls/progression/full` and `rc/controls/progression/study_type_plus_method` that would include 7+ controls. This was an internal inconsistency.

**Resolution**: Introduced a two-tier constraint:
- `controls_count_max: 4` applies to combinatorial subset sampling (the `exhaustive_blocks` sampler).
- `controls_count_max_named_exception: 7` applies to explicitly named multi-block specs (study_type+method progression, full progression). These are individually enumerated and accepted despite exceeding the subset-sampling cap because they represent substantively meaningful meta-regression configurations. With N=43 and 7 controls, 35 residual degrees of freedom remain -- marginal but acceptable for a named spec.

### F) Added collinearity note for method dummies

Added explicit `collinearity_note` to the controls axis definition. Method_1 through Method_5 are study-method classification dummies (with Method_4 being all zeros in the regression sample). Including the full method block (4 dummies) simultaneously may cause near-perfect collinearity if these dummies nearly partition the sample. The spec search agent should check VIF when running the full method block.

### G) Corrected universe size arithmetic

The original surface claimed 40 one-axis-at-a-time specs, but the count was off due to including the 3 duplicate specs and not accounting for the reclassifications. After corrections:

| Component | Original count | Revised count | Notes |
|---|---|---|---|
| Baseline | 1 | 1 | Unchanged |
| Controls (single-add) | 10 | 10 | Unchanged |
| Controls (sets) | 4 | 2 | Removed `none` (= baseline); kept `study_characteristics_basic`, `study_characteristics_extended` |
| Controls (progression) | 4 | 3 | Removed `bivariate` (= baseline); kept 3 progression steps |
| Sample variants | 9 | 8 | Removed `drop_high_influence` (= `cooksd_4_over_n`) |
| Functional form (core) | 3 | 1 | 2 reclassified to exploration |
| Preprocessing (core) | 5 | 4 | 1 reclassified to exploration |
| Weights (new) | 0 | 1 | Added WLS inverse-variance proxy |
| Inference | 4 | 4 | Unchanged |
| **One-axis total (core)** | **40** | **34** | |
| Block-combo subsets | 15 | ~8 | Respecting 4-control subset cap |
| High-value interactions | ~5 | 6 | Explicitly defined |
| **Total core** | **~55** | **~49** | |
| Exploration specs | 0 | 3 | Newly scoped |
| **Grand total** | **~55** | **~52** | |

### H) Made high-value interactions explicit

The original surface mentioned "high-value interactions" informally. The revised surface enumerates them with spec_ids, descriptions, and axes_changed metadata, following the contract for `rc/joint/*` specs.

---

## 3. Key Constraints and Linkage Rules

| Constraint | Value | Notes |
|---|---|---|
| `controls_count_min` | 0 | Baseline is bivariate |
| `controls_count_max` (subset sampling) | 4 | N=43 hard-limits feasible controls |
| `controls_count_max` (named exceptions) | 7 | For explicitly enumerated multi-block specs |
| `linked_adjustment` | false | Single-equation OLS; not applicable |
| `small_sample_flag` | true | N=43 requires caution on all axes |

No bundled estimators. No linkage constraints needed.

---

## 4. Budget and Sampling Assessment

**Budget**: 65 core specs + 5 exploration specs. This is adequate for full enumeration of the planned surface. No random sampling is needed.

**Sampling**: The `exhaustive_blocks` sampler over 4 control blocks with a 4-control cap produces a manageable number of combinations. Blocks with sizes [3, 4, 2, 1] yield the following within-cap combinations:
- Single blocks: study_type(3), quality(2), temporal(1) are within cap; method(4) is at the cap
- Two-block combos within cap: quality+temporal(3), study_type+temporal(4) -- both within cap; study_type+quality(5) exceeds cap
- Only a few multi-block combos stay within 4 controls

This gives roughly 5-8 additional block-combination specs beyond the explicitly named sets, well within budget.

**Reproducibility**: Seed 111185 is specified. Deterministic full enumeration ensures reproducibility without concern about sampling variation.

---

## 5. "What's Missing" List

### Considered and deliberately excluded:
- **Robust regression (median regression, M-estimators)**: Could be a useful robustness check given the outlier sensitivity of this regression. However, with N=43, asymptotic properties of quantile regression are unreliable. Deliberate exclusion is acceptable.
- **Bootstrap inference**: With N=43, pairs bootstrap could be informative (especially given non-normal residuals). The resampling module (`specification_tree/modules/inference/resampling.md`) covers this. Could be added as `infer/resampling/pairs_bootstrap` but is not strictly necessary given HC2/HC3 are already included. Low priority.
- **Oster sensitivity analysis**: The surface includes control progressions that would provide inputs for an Oster delta calculation. However, since the baseline is bivariate (0 controls) and the treatment is not a causal intervention (this is a meta-regression, not a treatment effect), Oster-style selection-on-unobservables framing is not well-motivated. Correctly excluded.
- **Multiple-testing adjustment**: With only one baseline group and one estimand, MHT correction is not applicable at the claim level. Post-processing spec-curve summaries could be useful but are a `post/*` object, not part of the core surface.

### Items the spec search agent should verify at runtime:
1. **WLS feasibility**: Check whether the H&S dataset contains variables usable as precision weights (e.g., study-reported standard errors, sample sizes, confidence interval widths). If no suitable variable exists, drop `rc/weights/main/wls_inverse_variance_proxy` and the associated interaction.
2. **Method dummy collinearity**: Check VIF when running specs that include the full method block (Method_1 through Method_5 minus Method_4). If severe collinearity arises, the agent should note this and potentially drop one method dummy.
3. **Temperature adjustment variable validity**: Verify that `Temp_adj_FUND_curr`, `Temp_adj_NASA`, and `Temp_adj_AVG` produce positive adjusted temperatures for all observations (negative or zero temperatures would make `logt` undefined).

---

## 6. Diagnostics Assessment

The diagnostics plan is appropriate for this regression:
- Cook's D (critical given the Weitzman outlier)
- Jarque-Bera normality test (residuals are known to be non-normal)
- Breusch-Pagan heteroskedasticity test (baseline uses classical SEs)
- Ramsey RESET (added: functional form adequacy)

All diagnostics have scope `baseline_group`, which is correct since they do not depend on controls (the baseline is bivariate).

---

## 7. Final Assessment

**APPROVED TO RUN.** The surface is now:
- **Conceptually coherent**: Single baseline group with a well-defined claim object.
- **Statistically principled**: Core RC is cleanly separated from exploration. Estimand-changing functional-form specs are correctly scoped as exploration.
- **Faithful to the revealed manuscript surface**: The paper reveals essentially zero forking paths (only one bivariate regression with no alternatives shown). The surface appropriately constructs the specification battery from standardized modules. The control-count envelope and small-sample warnings correctly reflect the N=43 constraint.
- **Auditable**: Budgets, sampling, duplicates resolved, universe size corrected, constraints made internally consistent.

No blocking issues remain.
