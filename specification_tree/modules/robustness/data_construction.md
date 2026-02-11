# Data Construction (Robustness Checks)

## Spec ID format

This module defines **estimand-preserving** (core-eligible) robustness checks that vary *how the analysis dataset is constructed* from raw tables, without changing the baseline claim concept.

Use:

- `rc/data/{axis}/{variant}`

Examples:

- `rc/data/merge/inner_vs_left`
- `rc/data/unit_of_analysis/household_vs_individual`
- `rc/data/aggregation/mean_vs_sum`

## Why this matters

In real replication packages, many fragile “spec choices” occur **upstream of regression**:

- merges and linkage rules,
- duplicate handling,
- collapsing/reshaping into the analysis panel,
- deflators and real/nominal conversions,
- construction of rates and denominators,
- top-coding rules in administrative microdata.

These are frequently under-documented and can materially change estimates even when the estimand is “supposed” to be unchanged.

## Core principle

These checks are core-eligible when they preserve:

- outcome concept,
- treatment concept,
- estimand concept,
- target population.

If a construction change **redefines** the outcome/treatment/population, it belongs in `explore/*` instead.

## Merge + linkage robustness

| spec_id | Description |
|--------|-------------|
| `rc/data/merge/key_audit` | Audit key uniqueness and many-to-one vs many-to-many risks |
| `rc/data/merge/drop_duplicates_first` | Resolve duplicates by “first” rule (documented) |
| `rc/data/merge/drop_duplicates_last` | Resolve duplicates by “last” rule |
| `rc/data/merge/aggregate_duplicates_mean` | Aggregate duplicate matches by mean (when duplicates represent multiple records) |
| `rc/data/merge/aggregate_duplicates_sum` | Aggregate duplicate matches by sum (flows) |
| `rc/data/merge/inner_vs_left` | Compare inner join vs left join when plausible |
| `rc/data/merge/missing_match_indicator` | Include indicator for “unmatched” and keep unmatched rows where sensible |

## Unit of analysis and aggregation

| spec_id | Description |
|--------|-------------|
| `rc/data/unit_of_analysis/micro` | Use microdata unit (person/firm) as baseline |
| `rc/data/unit_of_analysis/aggregate` | Use aggregated unit (county-year, school-grade, etc.) where baseline is aggregate |
| `rc/data/aggregation/mean_vs_sum` | Swap mean vs sum when both are plausible summaries |
| `rc/data/aggregation/median_vs_mean` | Swap median vs mean aggregation |
| `rc/data/aggregation/weight_by_pop` | Population-weight aggregation |
| `rc/data/aggregation/unweighted` | Unweighted aggregation |

## Panel construction

| spec_id | Description |
|--------|-------------|
| `rc/data/panel/balanced_only` | Restrict to balanced panel when missingness is plausibly ignorable |
| `rc/data/panel/unbalanced` | Use unbalanced panel (baseline) |
| `rc/data/panel/trim_singletons` | Drop singleton unit observations (panel FE stability) |
| `rc/data/panel/event_alignment_alt` | Alternative alignment of event time when multiple plausible event dates exist |

## Deflators, denominators, and “rates”

| spec_id | Description |
|--------|-------------|
| `rc/data/deflator/nominal` | Use nominal values (if baseline is real) as a contrast |
| `rc/data/deflator/real_cpi` | CPI-deflated real values (baseline or alternative CPI) |
| `rc/data/deflator/ppp` | PPP-adjusted conversion (cross-country) |
| `rc/data/rates/per_capita` | Convert totals to per-capita rates |
| `rc/data/rates/per_worker` | Convert totals to per-worker rates |
| `rc/data/rates/log_rate_vs_rate` | Compare log(rate) vs rate when concept is “growth/elasticity” |

## Required audit fields (`coefficient_vector_json`)

Every `rc/data/*` spec must include:

```json
{
  "data_construction": {
    "spec_id": "rc/data/merge/inner_vs_left",
    "axis": "merge",
    "description": "Left join main panel to auxiliary file; keep unmatched with missing indicators.",
    "keys": ["unit_id", "year"],
    "dedup_rule": "aggregate_duplicates_mean",
    "n_rows_before": 123456,
    "n_rows_after": 120001,
    "notes": "Preserves target population; changes only linkage."
  }
}
```
