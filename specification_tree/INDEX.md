# Specification Tree Index (Typed, Orthogonal)

This directory defines the **specification-tree contract** used to make replication, robustness, diagnostics, sensitivity analysis, and exploration **auditable and mechanically filterable**.

Start with:

- `specification_tree/ARCHITECTURE.md` (conceptual contract + typing)
- `specification_tree/CONTRACT.md` (practical output/schema rules)
- `specification_tree/DESIGN_AUDIT_FIELDS.md` (where to record design-defining parameters in JSON)
- `specification_tree/CLAIM_GROUPING.md` (baseline groups; core vs exploration)
- `specification_tree/REVEALED_SEARCH_SPACE.md` (revealed vs potential search; linkage constraints)
- `specification_tree/SPECIFICATION_SURFACE.md` (paper-specific universe + constraints + budgets)
- `specification_tree/SPEC_UNIVERSE_AND_SAMPLING.md` (define the universe; budgeted sampling)
- `specification_tree/COVERAGE.md` (coverage checklist + gaps)

## 1) Designs (method / identification families)

Pick the relevant design file(s) and run the **design-specific** estimator implementations and diagnostics it lists.

| Design family | File |
|---|---|
| Difference-in-differences | `specification_tree/designs/difference_in_differences.md` |
| Event study | `specification_tree/designs/event_study.md` |
| Regression discontinuity | `specification_tree/designs/regression_discontinuity.md` |
| Instrumental variables | `specification_tree/designs/instrumental_variables.md` |
| Shift-share / Bartik | `specification_tree/designs/shift_share.md` |
| Randomized experiment | `specification_tree/designs/randomized_experiment.md` |
| Synthetic control | `specification_tree/designs/synthetic_control.md` |
| Panel fixed effects | `specification_tree/designs/panel_fixed_effects.md` |
| Cross-sectional / OLS | `specification_tree/designs/cross_sectional_ols.md` |
| Dynamic panel | `specification_tree/designs/dynamic_panel.md` |
| Discrete choice | `specification_tree/designs/discrete_choice.md` |
| Local projection | `specification_tree/designs/local_projection.md` |
| Structural VAR | `specification_tree/designs/structural_var.md` |
| Structural calibration / moments | `specification_tree/designs/structural_calibration.md` |
| Bunching | `specification_tree/designs/bunching_estimation.md` |
| Duration / survival | `specification_tree/designs/duration_survival.md` |
| DSGE Bayesian | `specification_tree/designs/dsge_bayesian_estimation.md` |

## 2) Universal modules (apply across designs)

These are **typed** and should be referenced (not duplicated) from design files and agent prompts.

### Robustness checks (estimand-preserving; `rc/*`)

- `specification_tree/modules/robustness/controls.md`
- `specification_tree/modules/robustness/sample.md`
- `specification_tree/modules/robustness/fixed_effects.md`
- `specification_tree/modules/robustness/preprocessing.md`
- `specification_tree/modules/robustness/data_construction.md`
- `specification_tree/modules/robustness/functional_form.md`
- `specification_tree/modules/robustness/joint.md`
- `specification_tree/modules/robustness/weights.md`

### Inference (`infer/*`)

- `specification_tree/modules/inference/standard_errors.md`
- `specification_tree/modules/inference/resampling.md`

### Diagnostics (`diag/*`)

- `specification_tree/modules/diagnostics/placebos.md`
- `specification_tree/modules/diagnostics/design_diagnostics.md`
- `specification_tree/modules/diagnostics/regression_diagnostics.md`

### Sensitivity analysis / partial identification (`sens/*`)

- `specification_tree/modules/sensitivity/unobserved_confounding.md`
- `specification_tree/modules/sensitivity/assumptions/instrumental_variables.md`
- `specification_tree/modules/sensitivity/assumptions/difference_in_differences.md`
- `specification_tree/modules/sensitivity/assumptions/regression_discontinuity.md`
- `specification_tree/modules/sensitivity/assumptions/randomized_experiment.md`
- `specification_tree/modules/sensitivity/assumptions/synthetic_control.md`

### Post-processing (set-level; `post/*`)

- `specification_tree/modules/postprocess/multiple_testing.md`
- `specification_tree/modules/postprocess/specification_curve.md`

### Exploration (concept/estimand changes; `explore/*`)

- `specification_tree/modules/exploration/variable_definitions.md`
- `specification_tree/modules/exploration/heterogeneity.md`
- `specification_tree/modules/exploration/cate_estimation.md`
- `specification_tree/modules/exploration/policy_learning.md`
- `specification_tree/modules/exploration/alternative_estimands.md`

### Estimation wrappers (implementation alternatives)

- `specification_tree/modules/estimation/dml.md` (DML as nuisance-learning layer)

## 3) Typed `spec_id` namespaces (required)

Every recorded row must use a typed namespace so its statistical object is mechanically recoverable:

- `baseline` (reserved; paper’s canonical estimate(s) for a claim object)
- `design/{design_code}/...` (within-design estimator/implementation)
- `rc/{axis}/{variant}` (estimand-preserving robustness checks)
- `infer/{axis}/{variant}` (inference variations)
- `diag/{family}/{axis}/{variant}` (diagnostics/falsification)
- `sens/{family}/{variant}` (sensitivity/partial-ID)
- `post/{axis}/{variant}` (set-level transforms)
- `explore/{axis}/{variant}` (concept/estimand changes)

Examples:

- `design/difference_in_differences/estimator/twfe`
- `rc/controls/loo/drop_age`
- `infer/se/cluster/unit_time`
- `diag/difference_in_differences/pretrends/joint_test`
- `sens/assumption/instrumental_variables/exclusion/conley_bound_small`
- `post/mht/family_core_rc/bh`
- `explore/heterogeneity/interaction/gender`

See `specification_tree/CONTRACT.md` for the required output fields and the rules for scalar summaries of vector estimates (event studies, local projections, SVARs).
