# Design-Specific Audit Fields (What to Record in JSON)

Baseline claim objects (baseline groups) pin down concepts (outcome/treatment/estimand/population). But many designs also have **design-defining parameters** that must be recorded so results remain interpretable when detached from code (and so downstream tooling can sanity-check “same design vs different design”).

## Where to store design metadata

For successful estimate-like rows (`baseline`, `design/*`, `rc/*`), store design metadata under:

- `coefficient_vector_json.design.{design_code}`

Use stable, plain keys. Do not add new top-level keys to `coefficient_vector_json`; put any design-specific objects under `design` (and anything else under `extra`).

## Surface source of truth (recommended)

For each baseline group, record these fields once in `SPECIFICATION_SURFACE.json` under:

- `baseline_groups[].design_audit`

Then, for outputs:

- for `baseline` and `rc/*` rows, copy `design_audit` verbatim into `coefficient_vector_json.design.{design_code}`;
- for `design/*` rows, start from the surface `design_audit` and override any design-defining parameters changed by the variant.

Avoid estimator-only `design_audit` blocks: include at least one additional design-defining field so results remain interpretable without code.

This document provides a **minimal recommended** set of fields per design family. Record more when helpful.

## Recommended minimal fields by design

### `cross_sectional_ols`

- `estimator`: `ols` / `ipw_ate` / `aipw_ate` / `matching_nn` / …
- `model_formula` (or structured `outcome`, `treatment`, `controls`)
- `selection_story`: short note like `"unconfoundedness (selection-on-observables)"`
- If IPW/AIPW: `propensity_model`, `trimming_rule` (if used)

### `panel_fixed_effects`

- `panel_unit`, `panel_time`
- `fe_structure`: list (e.g., `["unit","time"]`, `["unit","time","region:time"]`)
- `differencing`: `none` / `first_difference` / `long_difference` (+ horizon)

### `difference_in_differences`

- `treatment_timing`: cohort/adoption definition (or event date)
- `estimand`: `ATT` / `ATE` / `event_time_path` (short string; match the claim object)
- `comparison_groups`: how controls are formed (never-treated, not-yet-treated, etc.)
- If staggered: `cohorts`, `event_window`, `reference_period`

### `event_study`

- `event_time_var` (or definition)
- `event_window`: `{min,max}` or explicit list
- `reference_period`
- `bin_endpoints`: true/false (+ rule if true)

### `regression_discontinuity`

- `running_var`
- `cutoff`
- `rd_type`: `sharp` / `fuzzy` / `kink`
- `bandwidth`: numeric value or rule label
- `kernel`
- `poly_order`
- `bias_correction`: `none` / `rbc` (and any key tuning knobs)

### `instrumental_variables`

- `endog_vars`, `instrument_vars`
- `first_stage`: include a compact strength summary (e.g., F/KP) when available
- `n_instruments`
- `overid_df` (if overidentified)
- Also record a `bundle` block in `coefficient_vector_json` when adjustment is linked across stages (see `specification_tree/REVEALED_SEARCH_SPACE.md`)

### `shift_share`

- `share_unit`, `share_base_year`, `share_vars`
- `shock_series`, `shock_window`
- `leave_one_out` rule
- `normalization` rule

### `randomized_experiment`

- `randomization_unit`
- `strata_or_blocks` (if used)
- `estimand`: `ITT` / `TOT` (match the claim object)
- If clustered assignment: `n_clusters` (if known)

### `synthetic_control`

- `treated_unit(s)`
- `donor_pool_rule` (how donors are defined)
- `predictor_set` (or a pointer to the set)
- `pre_period`
- `estimator`: `scm` / `sdid` / `gsynth`

### `dynamic_panel`

- `lags`: `{y_lags: k, x_lags: …}` (as applicable)
- `estimator`: `diff_gmm` / `sys_gmm` / …
- `instruments`: strategy label (`collapsed`, lag window, etc.)
- If long-run effects are reported: record them under `design.dynamic_panel.long_run` (see the design file)

### `discrete_choice`

- `model_family`: `logit` / `probit` / `lpm` / …
- `reported_object`: `index_coef` / `odds_ratio` / `ame`
- If AME: `ame_at` rule (sample mean, average over sample, etc.)

### `local_projection`

- `shock_definition` (series/instrument/proxy; short description)
- `horizons`: `{h_min,h_max,grid}` or explicit list
- `dynamics`: how lags are included (y lags, shock lags, both)
- If HAC is used: bandwidth rule (also record under `inference`)

### `structural_var`

- `var_vars` (ordered list)
- `lags`
- `identification_scheme`
- `shock_normalization`
- `irf_horizons`

### `structural_calibration`

- `targets` (moment list or compact description)
- `parameters` (key calibrated/estimated parameters)
- `fit_summary` (e.g., loss/objective value)
- `counterfactual` (policy/scenario label)

### `bunching_estimation`

- `threshold` (kink/notch value)
- `bin_width` (if binned)
- `window`: `{left,right}` around threshold
- `counterfactual_spec`: polynomial order / excluded region (donut) rule

### `duration_survival`

- `event_definition`
- `time_scale`
- `censoring_rule`
- `model_family`: `cox_ph` / `weibull_aft` / …
- If Cox: `ties_method`

### `dsge_bayesian_estimation`

DSGE objects are JSON-heavy; store full details under `design.dsge_bayesian_estimation` as described in `specification_tree/designs/dsge_bayesian_estimation.md`.
