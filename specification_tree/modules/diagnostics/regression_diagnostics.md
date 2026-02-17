# General Regression Diagnostics (Diagnostics)

## Spec ID format

Use:

- `diag/regression/{axis}/{variant}`

Examples:

- `diag/regression/heteroskedasticity/breusch_pagan`
- `diag/regression/specification/ramsey_reset`
- `diag/regression/multicollinearity/vif_max`

## Purpose

These diagnostics assess model adequacy and common failure modes in regression-based analyses.
They are **not** alternative estimates of the baseline estimand.

They can be useful for verification reports and for interpreting fragile results, but should not be counted as core RC.

## A) Heteroskedasticity diagnostics

| spec_id | Description |
|---|---|
| `diag/regression/heteroskedasticity/breusch_pagan` | Breusch–Pagan test |
| `diag/regression/heteroskedasticity/white` | White test |

## B) Functional-form / omitted-variable diagnostics

| spec_id | Description |
|---|---|
| `diag/regression/specification/ramsey_reset` | Ramsey RESET test |

## C) Multicollinearity diagnostics

| spec_id | Description |
|---|---|
| `diag/regression/multicollinearity/vif_max` | Maximum VIF among regressors |
| `diag/regression/multicollinearity/condition_number` | Condition number of design matrix |

## D) Influence / leverage diagnostics

| spec_id | Description |
|---|---|
| `diag/regression/influence/cooks_distance` | Cook’s distance summary (max + tail share) |
| `diag/regression/influence/leverage` | High-leverage summary (max + tail share) |

## Output contract (`diagnostic_json`)

Write regression diagnostics to `diagnostics_results.csv` and store outputs in `diagnostic_json` with a `diagnostic` block. Example:

```json
{
  "diagnostic": {
    "spec_id": "diag/regression/specification/ramsey_reset",
    "statistic": 1.83,
    "p_value": 0.14,
    "notes": "No strong evidence of misspecification at 5%."
  }
}
```
