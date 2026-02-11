# Design Diagnostics (Diagnostics)

## Spec ID format

Use:

- `diag/{design}/{axis}/{variant}`

Examples:

- `diag/difference_in_differences/pretrends/joint_test`
- `diag/regression_discontinuity/manipulation/mccrary_density`
- `diag/instrumental_variables/strength/first_stage_f`
- `diag/randomized_experiment/balance/covariates`

## Purpose

Diagnostics are **assumption checks**, **credibility audits**, or **falsification exercises**.
They are not alternative estimates of the baseline estimand.

They can be essential for replication quality, but should remain separate from:

- `rc/*` (estimand-preserving robustness checks),
- `sens/*` (assumption relaxations / partial identification).

## Scope + linkage (important)

Diagnostics do not have a clean one-to-one mapping to “specs”. Some depend only on the baseline claim object; others depend on the full estimating equation (including the control set).

When recording diagnostics, explicitly specify a `diagnostic_scope` and link each diagnostic run to the spec run(s) it is intended to validate using `spec_diagnostics_map.csv` (see `specification_tree/CONTRACT.md`).

Recommended scopes:

- `paper`: invariant across baseline groups (rare)
- `baseline_group`: depends on the claim object but not on the exact control set (often)
- `spec`: depends on the exact specification (common for bundled estimators like IV/AIPW)
- `design_variant`: depends on a particular design implementation (e.g., TWFE vs Sun–Abraham)

Rule of thumb examples:

- IV first-stage strength is usually `spec`-scoped (changes with controls/FE/sample).
- RD McCrary density is usually `baseline_group`-scoped (running variable + cutoff), unless you intentionally vary the bandwidth.
- DiD/event-study pretrend tests are usually `spec`-scoped (change with window/controls/weights).

## A) DiD / event study diagnostics

| spec_id | Description |
|---|---|
| `diag/difference_in_differences/pretrends/joint_test` | Joint test of pre-treatment coefficients |
| `diag/difference_in_differences/pretrends/linear_trend_test` | Test for linear differential pre-trend |
| `diag/difference_in_differences/weights/bacon_decomposition` | Goodman–Bacon weight decomposition (TWFE) |
| `diag/difference_in_differences/overlap/cohort_counts` | Cohort/event-time cell counts (support diagnostics) |

## B) RD diagnostics

| spec_id | Description |
|---|---|
| `diag/regression_discontinuity/manipulation/mccrary_density` | McCrary density test at cutoff |
| `diag/regression_discontinuity/balance/covariate_continuity` | Covariate continuity/balance around cutoff |
| `diag/regression_discontinuity/functional_form/bin_sensitivity_plot` | Binned scatter/visual RD plot summaries |

## C) IV diagnostics

| spec_id | Description |
|---|---|
| `diag/instrumental_variables/strength/first_stage_f` | First-stage strength summary (first-stage F / Kleibergen–Paap where relevant) and partial R² |
| `diag/instrumental_variables/validity/overid_test` | Overidentification test (Hansen/Sargan) |
| `diag/instrumental_variables/endogeneity/hausman` | Endogeneity test (Durbin–Wu–Hausman) |
| `diag/instrumental_variables/monotonicity/compliance_check` | Compliance monotonicity plausibility (where implementable) |

Note: exclusion restriction sensitivity belongs in `sens/assumption/instrumental_variables/*`.

## D) Randomized-experiment diagnostics

| spec_id | Description |
|---|---|
| `diag/randomized_experiment/balance/covariates` | Balance table/test for baseline covariates |
| `diag/randomized_experiment/attrition/attrition_diff` | Differential attrition diagnostic |
| `diag/randomized_experiment/noncompliance/first_stage` | Noncompliance rate / first-stage strength (ITT→TOT) |

## E) Synthetic control diagnostics

| spec_id | Description |
|---|---|
| `diag/synthetic_control/fit/pre_rmspe` | Pre-treatment fit (RMSPE) |
| `diag/synthetic_control/weights/donor_weight_concentration` | Donor weight concentration / effective donors |
| `diag/synthetic_control/placebo/in_space` | In-space placebo distribution summary |

## F) Panel diagnostics

| spec_id | Description |
|---|---|
| `diag/panel_fixed_effects/fe_vs_re/hausman` | Hausman test comparing FE vs RE (when RE is meaningful) |
| `diag/panel_fixed_effects/serial_corr/wooldridge` | Wooldridge test for serial correlation in panel residuals |
| `diag/panel_fixed_effects/cross_sectional_dep/pesaran_cd` | Pesaran CD test for cross-sectional dependence |

## G) Discrete-choice diagnostics

| spec_id | Description |
|---|---|
| `diag/discrete_choice/fit/pseudo_r2` | Pseudo-\(R^2\) summary (paper-specific; e.g., McFadden) |
| `diag/discrete_choice/fit/ll_aic_bic` | Log-likelihood + AIC/BIC summary |
| `diag/discrete_choice/iia/test` | IIA diagnostic (multinomial logit; if implementable) |

## H) Dynamic-panel diagnostics

| spec_id | Description |
|---|---|
| `diag/dynamic_panel/ar/ar1` | AR(1) test in differenced residuals (Arellano–Bond) |
| `diag/dynamic_panel/ar/ar2` | AR(2) test in differenced residuals |
| `diag/dynamic_panel/overid/hansen` | Hansen J (or Sargan) overidentification test |
| `diag/dynamic_panel/instruments/count` | Instrument count + proliferation guardrails |

## I) Local-projection diagnostics

| spec_id | Description |
|---|---|
| `diag/local_projection/irf/path_recorded` | Sanity check: full IRF path recorded with intended horizon grid |
| `diag/local_projection/inference/hac_bandwidth_rule` | HAC bandwidth/lag-selection rule recorded when HAC is used |

## J) SVAR diagnostics

| spec_id | Description |
|---|---|
| `diag/structural_var/stability/roots_inside_unit_circle` | VAR stability (roots inside unit circle) |
| `diag/structural_var/fit/residual_autocorr` | Residual autocorrelation/lag adequacy summary |
| `diag/structural_var/normalization/shock_scale_recorded` | Shock normalization/scale recorded |

## K) Structural calibration diagnostics

| spec_id | Description |
|---|---|
| `diag/structural_calibration/fit/moment_table` | Moment-fit table (data vs model) |
| `diag/structural_calibration/fit/overid_test` | Overidentification/J test (when meaningful/available) |
| `diag/structural_calibration/uncertainty/bootstrap_or_posterior` | Uncertainty source recorded (bootstrap vs posterior) |

## L) Bunching diagnostics

| spec_id | Description |
|---|---|
| `diag/bunching_estimation/placebo/thresholds` | Placebo threshold(s) |
| `diag/bunching_estimation/density/continuity` | Density continuity around threshold |
| `diag/bunching_estimation/donut/exclude_threshold_bin` | Donut hole: exclude threshold bin |

## M) Duration/survival diagnostics

| spec_id | Description |
|---|---|
| `diag/duration_survival/ph_assumption/schoenfeld` | Proportional-hazards test (Schoenfeld residuals; Cox PH only) |
| `diag/duration_survival/fit/concordance_index` | Concordance index (C-index) |
| `diag/duration_survival/censoring/counts` | Events vs censored counts |

## N) DSGE diagnostics

| spec_id | Description |
|---|---|
| `diag/dsge_bayesian_estimation/mcmc/convergence` | MCMC convergence summaries (trace/R-hat-like; acceptance rates) |
| `diag/dsge_bayesian_estimation/fit/marginal_likelihood` | Marginal likelihood / model fit summary |
| `diag/dsge_bayesian_estimation/filter/kalman_ok` | Kalman filter likelihood computed without numerical failure |

## Output contract (`coefficient_vector_json`)

Diagnostics are heterogeneous; store outputs in a `diagnostic` block.
If a scalar test statistic and p-value exist, include them there (and optionally in scalar fields if your pipeline supports it).

Example:

```json
{
  "diagnostic": {
    "spec_id": "diag/regression_discontinuity/manipulation/mccrary_density",
    "statistic": 1.72,
    "p_value": 0.085,
    "bandwidth": 0.5,
    "notes": "Suggestive but not definitive manipulation evidence."
  }
}
```
