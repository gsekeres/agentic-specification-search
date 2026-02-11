# Sample / Population Restrictions (Robustness Checks)

## Spec ID format

Use:

- `rc/sample/{axis}/{variant}`

Examples:

- `rc/sample/outliers/trim_y_1_99`
- `rc/sample/panel/balanced_only`
- `rc/sample/time/drop_first_period`

## Purpose

Sample rules are among the highest-leverage degrees of freedom in real replication packages:

- trimming/winsorization vs dropping,
- balanced vs unbalanced panels,
- eligibility filters and quality flags,
- handling influential observations,
- common-support trimming.

This module is for **estimand-preserving** (core-eligible) sample robustness checks.

## The key distinction: RC vs population change (exploration)

Some “sample restrictions” are genuine RC; others change the target population and hence the estimand.

### A) RC: within-population robustness (core-eligible)

These are intended to preserve the baseline claim’s *target population concept* and only change implementation details:

- outlier rules that the paper treats as a cleaning choice,
- dropping low-quality / flagged observations,
- balanced vs unbalanced panel when the intended population is “the panel units”,
- mild common-support trimming used as a robustness check (not a new target).

### B) Exploration: subpopulation estimands (not core by default)

These change the **target population** (or are framed as heterogeneity):

- “male only”, “urban only”, “firms > 50 employees”, “high-income tercile”,
- cohort-only claims, region-only effects, etc.

These belong in:

- `specification_tree/modules/exploration/heterogeneity.md` (`explore/heterogeneity/*`), or
- a separate baseline group if the paper’s headline claim is explicitly about that subgroup.

See `specification_tree/CLAIM_GROUPING.md`.

## A) Time / period restrictions

| spec_id | Description |
|---|---|
| `rc/sample/time/drop_first_period` | Drop first period/year (guardrail against early measurement error) |
| `rc/sample/time/drop_last_period` | Drop last period/year (guardrail against end-of-sample anomalies) |
| `rc/sample/time/early_half` | First half of the study window (if framed as stability check) |
| `rc/sample/time/late_half` | Second half of the study window |
| `rc/sample/time/predefined_breaks` | Paper-stated breakpoints (e.g., pre/post crisis) as robustness, if population concept unchanged |

## B) Outliers and influential observations

Prefer **coding-based** outlier handling (winsor/top-code) in `rc/preprocess/*` when it preserves the sample.
Use sample trimming when the paper’s intent is explicitly “drop outliers”.

| spec_id | Description |
|---|---|
| `rc/sample/outliers/trim_y_1_99` | Drop outcome outside [1%,99%] |
| `rc/sample/outliers/trim_y_5_95` | Drop outcome outside [5%,95%] |
| `rc/sample/outliers/trim_x_1_99` | Drop treatment outside [1%,99%] |
| `rc/sample/outliers/cooksd_4_over_n` | Drop Cook’s D > 4/n (linear models) |
| `rc/sample/outliers/leverage_high` | Drop top leverage tail (paper-defined) |

## C) Panel composition and missingness-related sample rules

Missingness *handling* (MI, missing indicators) lives in `rc/preprocess/missing/*`.
This section is for sample rules that change which units/periods are included.

| spec_id | Description |
|---|---|
| `rc/sample/panel/balanced_only` | Balanced panel only |
| `rc/sample/panel/unbalanced` | Unbalanced panel (baseline) |
| `rc/sample/panel/drop_singletons` | Drop singleton unit observations (FE stability) |
| `rc/sample/panel/min_obs_ge_2` | Keep units with ≥2 observations |
| `rc/sample/panel/min_obs_ge_5` | Keep units with ≥5 observations |

## D) Data quality and eligibility filters

| spec_id | Description |
|---|---|
| `rc/sample/quality/complete_cases` | Complete cases for baseline variables (if within the intended data definition) |
| `rc/sample/quality/high_quality_flag` | Restrict to observations passing a paper-defined quality flag |
| `rc/sample/quality/drop_imputed_values` | Drop imputed/allocated values when such flags exist |

## E) Common-support and trimming (when used as RC)

| spec_id | Description |
|---|---|
| `rc/sample/support/trim_ps_01_99` | Trim propensity score to [0.01,0.99] (binary treatment; if baseline is unconfoundedness-style) |
| `rc/sample/support/trim_ps_05_95` | Trim propensity score to [0.05,0.95] |

If the paper’s baseline estimand is explicitly “on-support” (overlap-restricted), then these may be baseline rather than RC.

## Required audit fields (`coefficient_vector_json`)

Every `rc/sample/*` row should include a `sample` block:

```json
{
  "sample": {
    "spec_id": "rc/sample/outliers/trim_y_1_99",
    "axis": "outliers",
    "rule": "trim",
    "params": {"var": "y", "lower_q": 0.01, "upper_q": 0.99},
    "n_obs_before": 10000,
    "n_obs_after": 9800,
    "notes": "Trimming outcome outliers only; target population unchanged."
  }
}
```
