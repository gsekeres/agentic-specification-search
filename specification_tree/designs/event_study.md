# Design: Event Study / Dynamic Treatment Effects

This design file enumerates **within-design implementation choices** for dynamic treatment-effect paths (leads/lags relative to an event).

Event studies typically produce a **vector** of coefficients; the output contract requires a declared scalar focal parameter. See `specification_tree/CONTRACT.md`.

## Baseline (REQUIRED)

- **spec_id**: `baseline` (or `baseline__{slug}` for additional baseline claim objects)
- Exact replication of the paper’s canonical event-study estimate(s), including:
  - the event-time window,
  - the reference period,
  - all lead/lag coefficients (stored in JSON).
- Record design-defining metadata under `coefficient_vector_json.design.event_study` (see `specification_tree/DESIGN_AUDIT_FIELDS.md`).

## Design implementation variants (`design/event_study/*`)

Spec ID format:

- `design/event_study/{axis}/{variant}`

### A) Estimator family (staggered adoption)

| spec_id | Description |
|---|---|
| `design/event_study/estimator/twfe` | Standard TWFE event study |
| `design/event_study/estimator/sun_abraham` | Sun & Abraham interaction-weighted event-time estimator |
| `design/event_study/estimator/callaway_santanna` | Callaway & Sant’Anna group-time ATT aggregation |
| `design/event_study/estimator/borusyak_imputation` | Borusyak et al. imputation event-study style |

### B) Window and endpoint handling

| spec_id | Description |
|---|---|
| `design/event_study/window/symmetric` | Symmetric window around event |
| `design/event_study/window/short` | Shorter window (paper-justified) |
| `design/event_study/window/long` | Longer window (paper-justified) |
| `design/event_study/window/bin_endpoints` | Bin endpoints to avoid sparse tails |
| `design/event_study/window/asymmetric` | Asymmetric (more lags than leads, or vice versa) |

### C) Reference period choice

| spec_id | Description |
|---|---|
| `design/event_study/reference/t_minus_1` | \(t=-1\) as reference (common default) |
| `design/event_study/reference/t_minus_2` | \(t=-2\) as reference |
| `design/event_study/reference/first_lead` | First available lead as reference |
| `design/event_study/reference/average_pre` | Average of all pre-periods as reference |

## Standard event-study diagnostics (must record for each baseline group)

These are **diagnostics**, not estimates of the dynamic estimand. They are not part of the default core surface, but they should be computed and recorded because they are standard for credible event-study reporting.

- `diag/difference_in_differences/pretrends/joint_test` (joint test of pre-period coefficients)
- `diag/difference_in_differences/overlap/cohort_counts` (support diagnostics in staggered adoption / sparse tails)

## Typed references to universal modules (do not duplicate here)

### Robustness checks (`rc/*`)

- Controls: `specification_tree/modules/robustness/controls.md`
- Sample rules: `specification_tree/modules/robustness/sample.md`
- Fixed effects: `specification_tree/modules/robustness/fixed_effects.md`
- Pre-processing/coding: `specification_tree/modules/robustness/preprocessing.md`
- Functional form: `specification_tree/modules/robustness/functional_form.md`

### Inference (`infer/*`)

- SE and clustering: `specification_tree/modules/inference/standard_errors.md`
- Resampling: `specification_tree/modules/inference/resampling.md`

### Exploration (`explore/*`)

- Alternative treatment timing/exposure mappings: `specification_tree/modules/exploration/variable_definitions.md`
