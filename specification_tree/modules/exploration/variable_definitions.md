# Alternative Variable Definitions (Exploration)

## Spec ID format

This module defines **concept / estimand exploration** via alternative outcome or treatment definitions and coding choices.

Use:

- `explore/definition/{target}/{variant}`

Examples:

- `explore/definition/treatment/binary_threshold_paper`
- `explore/definition/treatment/binary_threshold_median`
- `explore/definition/outcome/components_instead_of_index`

## Purpose

Many “reasonable” robustness checks in applied work are actually **changes to what is being measured**:

- different cutoffs/thresholds,
- different binning/dichotomization rules,
- switching between components and indices,
- alternative coding of exposures (any vs intensity),
- alternative timing definitions (first exposure vs cumulative exposure).

These are often scientifically valuable, but they typically change the estimand and should be **explicitly separated from core replication robustness**.

Important overlap:

- Some recodings (especially continuous exposure → binary “any exposure”) can also be used as an `rc/form/treatment/*` functional-form stress test when the treatment concept is unchanged and the surface explicitly records the intended coefficient interpretation.
- Use `explore/definition/*` when the goal is to **redefine** the concept/estimand (new cutoff definitions, new timing rules, materially different outcome constructs), rather than to stress-test the baseline functional form.

## Treatment definition exploration

### Dichotomization / threshold rules (often changes the estimand)

| spec_id | Description |
|--------|-------------|
| `explore/definition/treatment/binary_threshold_paper` | Use the paper’s stated cutoff rule (if baseline is continuous) |
| `explore/definition/treatment/binary_threshold_median` | Median split of continuous treatment |
| `explore/definition/treatment/binary_threshold_q25` | Treatment = 1 if above 25th percentile |
| `explore/definition/treatment/binary_threshold_q75` | Treatment = 1 if above 75th percentile |
| `explore/definition/treatment/binary_threshold_grid` | Sweep a small grid of plausible cutoffs and summarize |

### Binning / discrete intensity levels

| spec_id | Description |
|--------|-------------|
| `explore/definition/treatment/terciles` | Treatment intensity terciles |
| `explore/definition/treatment/quartiles` | Treatment intensity quartiles |
| `explore/definition/treatment/quintiles` | Treatment intensity quintiles |

### Timing / exposure-window definitions (panel/time series)

| spec_id | Description |
|--------|-------------|
| `explore/definition/treatment/first_exposure` | Treatment = indicator for first exposure event |
| `explore/definition/treatment/ever_treated` | Ever-treated indicator |
| `explore/definition/treatment/time_since_exposure` | Redefine treatment by time-since-event bins |
| `explore/definition/treatment/event_time_rebin` | Alternative event-time binning for dynamic designs |

## Outcome definition exploration

### Components vs index

| spec_id | Description |
|--------|-------------|
| `explore/definition/outcome/components_instead_of_index` | Replace an index outcome with its component outcomes |
| `explore/definition/outcome/index_instead_of_components` | Replace components with a constructed index |

### Alternative codings

| spec_id | Description |
|--------|-------------|
| `explore/definition/outcome/binary_threshold_paper` | Convert a continuous outcome to a binary indicator using paper’s cutoff |
| `explore/definition/outcome/binary_threshold_median` | Median-split outcome |
| `explore/definition/outcome/rank_transform` | Rank outcome (ordinalization) |

## Output contract (`exploration_results.csv`)

Write `explore/*` objects to `exploration_results.csv` (see `specification_tree/CONTRACT.md`) and store outputs in `exploration_json` with an `exploration` block.

## Required labeling and audit fields

Every `explore/*` row must clearly record **what concept changed**. Include:

- what definition changed (threshold, components, timing, coding),
- how it differs from baseline,
- a short reason why it is plausible.

Suggested `exploration_json` block:

```json
{
  "exploration": {
    "spec_id": "explore/definition/treatment/binary_threshold_median",
    "changed": ["treatment_definition"],
    "baseline_treatment": "continuous intensity",
    "new_treatment": "binary above median",
    "reason": "Explores nonlinearity/threshold effects; not part of baseline estimand."
  }
}
```
