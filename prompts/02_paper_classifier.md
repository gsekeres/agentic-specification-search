# Paper Design Classifier Prompt (Pre-Surface)

Use this prompt to classify an empirical paper’s primary identification/design family **before** building a specification surface.

---

## Inputs

- **Package directory**: `{EXTRACTED_PACKAGE_PATH}`

---

## Task

1) Read the README/documentation.
2) Inspect the main analysis scripts (do-files, R scripts, Python).
3) Identify the primary empirical design family used for the *main claim(s)*.

---

## Classification categories (must match `specification_tree/designs/*.md`)

Return one primary `design_code` from:

- `difference_in_differences`
- `event_study`
- `regression_discontinuity`
- `instrumental_variables`
- `randomized_experiment`
- `synthetic_control`
- `panel_fixed_effects`
- `cross_sectional_ols`
- `discrete_choice`
- `dynamic_panel`
- `local_projection`
- `structural_var`
- `structural_calibration`
- `bunching_estimation`
- `duration_survival`
- `dsge_bayesian_estimation`

Also return `secondary_design_codes` (possibly empty) for meaningful secondary methods used in the paper (e.g., DiD main + event study robustness).

---

## Output format (JSON only)

```json
{
  "paper_id": "{PAPER_ID}",
  "design_code": "difference_in_differences",
  "confidence": "high|medium|low",
  "evidence": [
    "Uses reghdfe with unit and time fixed effects and treat×post",
    "Main tables are DiD estimates"
  ],
  "secondary_design_codes": ["event_study"],
  "notes": "Event study appears in appendix as robustness."
}
```

---

## Decision rules (high-level)

- `difference_in_differences` vs `panel_fixed_effects`:
  - DiD: treatment changes over time for some units and main parameter is a treatment×post-type contrast.
  - Panel FE: repeated observations with FE but no clear treated-vs-control adoption structure.
- `event_study`:
  - explicit relative-time leads/lags and a dynamic path is a main object.
- `instrumental_variables`:
  - 2SLS/IV estimator with explicit instrument(s) and first-stage discussion.
- `regression_discontinuity`:
  - running variable + cutoff/bandwidth logic; rdrobust-like code.
- `synthetic_control`:
  - treated unit + donor pool + synthetic weights/placebos.
- `randomized_experiment`:
  - random assignment / strata / treatment arms; ITT/TOT framing.

If truly ambiguous, choose the design family most aligned with the paper’s **main** interpreted claim and lower the confidence.

