# Controls / Adjustment Set (Robustness Checks)

## Spec ID format

Use:

- `rc/controls/{family}/{variant}`

Examples:

- `rc/controls/sets/none`
- `rc/controls/loo/drop_age`
- `rc/controls/single/add_education`
- `rc/controls/progression/baseline`

## Purpose

Adjustment-set choices are among the highest-leverage degrees of freedom in observational work. This module standardizes **estimand-preserving** control robustness:

- which controls are included,
- how controls are grouped into sets,
- how sensitive the treatment effect is to any one control.

This module is distinct from:

- `specification_tree/modules/robustness/preprocessing.md` (control *coding* choices),
- `specification_tree/modules/sensitivity/unobserved_confounding.md` (formal sensitivity/bounds).

## A) Standard control sets

| spec_id | Description |
|--------|-------------|
| `rc/controls/sets/none` | Bivariate (treatment only, plus baseline FE if applicable) |
| `rc/controls/sets/minimal` | Minimal/essential controls only (paper-defined or closest analogue) |
| `rc/controls/sets/baseline` | Paper’s baseline control set |
| `rc/controls/sets/extended` | Baseline + additional defensible controls (paper-defined) |
| `rc/controls/sets/full` | “All reasonable” controls available in package |

## B) Leave-one-out controls (LOO)

For each baseline control variable `X_j`:

| spec_id | Description |
|--------|-------------|
| `rc/controls/loo/drop_{var}` | Drop one control variable from baseline |

Example: `rc/controls/loo/drop_age`.

## C) Single-control (treatment + one control)

For each baseline control variable `X_j`:

| spec_id | Description |
|--------|-------------|
| `rc/controls/single/add_{var}` | Treatment + {var} (+ baseline FE) |

Example: `rc/controls/single/add_income`.

## D) Control progression (build-up)

Define a small number of semantically meaningful blocks (paper-dependent), e.g.:

- demographics
- baseline socioeconomic
- geography
- time-varying shocks
- baseline outcomes / lags

Then run a progression:

| spec_id | Description |
|--------|-------------|
| `rc/controls/progression/bivariate` | Treatment only |
| `rc/controls/progression/demographics` | + demographics block |
| `rc/controls/progression/socioeconomic` | + socioeconomic block |
| `rc/controls/progression/geography` | + geography block |
| `rc/controls/progression/temporal` | + temporal block (if applicable) |
| `rc/controls/progression/baseline` | Paper baseline |
| `rc/controls/progression/full` | “All reasonable” |

This progression provides inputs for Oster/AET-style sensitivity in `sens/unobs/*`.

## E) High-dimensional control-set search (combinatorial, budgeted)

When the plausible adjustment set is large, enumerating all subsets is intractable. But the **adjustment-set inclusion vector** is often the most important degree of freedom in observational work.

Policy:

- Treat “control-set search” as an explicit, auditable RC family.
- Use **budgeted, reproducible subset generation** to approximate the combinatorial space.
- When the baseline estimator is bundled (IV, AIPW/DML), enforce any **linkage constraints** (shared controls across components) rather than inventing independent mix-and-match combinations unless the manuscript clearly reveals unlinked components.

See `specification_tree/REVEALED_SEARCH_SPACE.md`.

### E.1) Subset-generation spec_ids

Use:

- `rc/controls/subset/{variant}`

Recommended variants:

| spec_id | Description |
|---|---|
| `rc/controls/subset/exhaustive_blocks` | Exhaust all combinations of a small number of control *blocks* (if ≤ ~7 blocks) |
| `rc/controls/subset/exhaustive_key_k` | Exhaust all subsets of the top-k “key controls” (k chosen so \(2^k\) is feasible) |
| `rc/controls/subset/random_001` | Random subset draw #1 (seeded) |
| `rc/controls/subset/random_002` | Random subset draw #2 (seeded) |
| `rc/controls/subset/stratified_size_001` | Stratified random draw controlling subset size (seeded) |

### E.2) Budget and reproducibility

For each baseline group, define:

- `controls_pool`: the candidate controls eligible for inclusion (exclude obvious post-treatment variables by default),
- `controls_mandatory`: controls always included (often empty; can include “must-have” pre-specified controls),
- `max_subset_specs`: maximum number of subset specs to run (e.g., 50–200 depending on compute budget).

Recommended additional constraints (surface-derived):

- `subset_size_min`, `subset_size_max`: allowed number of controls in a sampled subset (excluding mandatory controls).
  - Default: set these using the paper’s **main-spec control-count envelope** (min/max number of controls across main-table canonical specs for the baseline group). See `specification_tree/SPECIFICATION_SURFACE.md`.

Subset generation must be reproducible:

- set an explicit seed (e.g., hash of `{paper_id, baseline_group_id}` + a global seed),
- record the seed and draw index in JSON.

### E.3) Recommended subset-generation algorithm (practical default)

1) Define semantically meaningful **blocks** (as in Section D).
2) If the number of blocks is small (e.g., ≤7), run all `2^B` block combinations as `rc/controls/subset/exhaustive_blocks`.
3) Then draw additional variable-level subsets (random or stratified-by-size) until reaching `max_subset_specs`, subject to:
   - each control is included in at least one draw and excluded in at least one draw (coverage),
   - avoid degenerate sets (empty set already covered by `rc/controls/sets/none`),
   - enforce the subset-size constraints (`subset_size_min/max`) when defined.

## Required audit fields (`coefficient_vector_json`)

Every `rc/controls/*` row should include:

```json
{
  "controls": {
    "spec_id": "rc/controls/loo/drop_age",
    "family": "loo",
    "dropped": ["age"],
    "added": [],
    "set_name": "baseline_minus_one",
    "n_controls": 12
  }
}
```

For subset-search rows, include:

```json
{
  "controls": {
    "spec_id": "rc/controls/subset/random_017",
    "family": "subset",
    "method": "random",
    "seed": 12345,
    "draw_index": 17,
    "pool": ["x1", "x2", "x3", "x4"],
    "mandatory": ["x1"],
    "subset_size_min": 6,
    "subset_size_max": 12,
    "included": ["x1", "x3"],
    "excluded": ["x2", "x4"],
    "n_controls": 2,
    "notes": "Linked across AIPW nuisance models."
  }
}
```
