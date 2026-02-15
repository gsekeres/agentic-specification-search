# Spec Tree Contract (What Agents Must Emit)

This document defines the **practical contract** between:

- the spec-tree markdown files in `specification_tree/`,
- the spec-search agent (which *runs* things), and
- the verification + estimation pipelines (which *filter, group, and model* things).

The goal is to make the output **typed, auditable, and mechanically usable** even when papers differ wildly in designs and data pipelines.

For the conceptual architecture and typing rationale, see `specification_tree/ARCHITECTURE.md`.

## 1) Every output row must have a typed `spec_id`

The first path segment of `spec_id` defines the **node kind** and default **core eligibility**:

| Namespace | Node kind | Default core-eligible? | Meaning |
|---|---:|---:|---|
| `baseline` | estimate | ✅ | Paper’s canonical estimate(s) for a baseline claim object |
| `design/*` | estimate | ✅ | Within-design estimator/implementation alternatives |
| `rc/*` | estimate | ✅ | Estimand-preserving robustness checks (ceteris paribus) |
| `infer/*` | inference | ❌ | Inference-only recomputations (SE/resampling), recorded separately from the estimate table |
| `diag/*` | diagnostic | ❌ | Diagnostics/falsification; not a new estimate of the estimand |
| `sens/*` | sensitivity | ❌ | Assumption relaxations / partial-ID / breakdown points |
| `post/*` | postprocess | ❌ | Set-level transforms (MHT, spec-curve summaries) |
| `explore/*` | explore | ❌ | Concept/estimand changes (alt outcomes/treatments, heterogeneity, policy) |

If a paper’s *baseline claim itself* is heterogeneous (e.g., the headline estimand is subgroup ATT), verification may explicitly mark some `explore/*` nodes as core for that baseline group (and record why).

## 2) Minimal required columns for estimate-like rows

For `baseline`, `design/*`, `rc/*` rows, the CSV must contain:

- `spec_run_id` (unique within paper; stable ID for this executed row)
- `spec_id`
- `spec_tree_path` (relative path + section anchor)
- `baseline_group_id` (the baseline claim object this row belongs to; from the pre-run surface)
- `outcome_var`, `treatment_var`
- `coefficient`, `std_error`, `p_value`
- `ci_lower`, `ci_upper` (if available; else empty)
- `n_obs`, `r_squared` (if meaningful; else empty)
- `coefficient_vector_json` (required; may be `{}` if truly unavailable)
- `sample_desc`, `fixed_effects`, `controls_desc`, `cluster_var` (empty allowed when not applicable)
- `run_success` (0/1)
- `run_error` (empty string allowed; required when `run_success=0`)

These rows should be computed under the baseline group’s **canonical inference choice** recorded in the surface. Alternative uncertainty recomputations belong in `inference_results.csv`.

### Run-success rule (required)

Every planned spec should appear as a row in `specification_results.csv`, even if it fails, so the run is auditable against the surface budget.

- If the run produced a valid focal estimate and uncertainty, set `run_success=1` and leave `run_error=""`.
- If the run failed (or produced unusable focal outputs), set `run_success=0`, set numeric fields to `NaN` where applicable, and provide a short `run_error` string.

## 3) `spec_tree_path` rules (auditability)

`spec_tree_path` must point to the *defining node* in the markdown tree:

- Prefer paths under `specification_tree/` (not absolute file system paths).
- Include a `#section-anchor` when possible.
- Do not use “custom” unless the spec truly cannot be placed in the tree.

Examples:

- `designs/difference_in_differences.md#estimators`
- `modules/robustness/controls.md#leave-one-out-controls-loo`
- `modules/inference/standard_errors.md#two-way-clustering`

## 4) Scalar summaries for vector estimates (required)

Many designs naturally produce **vectors** (event studies, local projections, SVAR IRFs, distributed lags).
The pipeline still needs a **scalar evidence index** per row.

Contract:

1) `coefficient`, `std_error`, `p_value` must correspond to a **declared focal parameter**, and
2) the full vector must be stored in `coefficient_vector_json`.

Include a `focal` block in `coefficient_vector_json`:

```json
{
  "focal": {
    "parameter": "event_time",
    "label": "tau=0",
    "selector": {"rel_time": 0},
    "summary_rule": "single"
  }
}
```

Recommended focal-parameter conventions:

- **Event study**: focal = `rel_time = 0` (impact) or average of `rel_time in {0,1,2}` (short-run mean). Record which.
- **Local projections**: focal = horizon `h = 1` (or paper’s headline horizon). If multiple horizons are baseline claims, use multiple baselines.
- **SVAR IRFs**: focal = impulse response at a designated horizon (e.g., `h=4` quarters) and variable. Record the shock normalization.

If the paper’s *claim* is explicitly “the path”, then verification should treat multiple horizons as multiple baseline groups (or store a path-level postprocess object in `post/*` and keep scalar summaries for estimation).

## 5) One-axis-at-a-time by default; explicit joint specs when needed

Default: RC should vary **one axis at a time** to keep interpretation clean (controls OR sample OR coding OR inference).

When a paper’s workflow is inherently multi-dimensional (e.g., macro “stylized facts” that vary detrending × HP λ × country set × sample window), use:

- `rc/joint/{family}/{variant}`

and include explicit metadata:

```json
{
  "joint": {
    "axes_changed": ["sample", "preprocess.series", "data.deflator"],
    "details": {
      "sample_window": "1980-2000",
      "hp_lambda": 1600,
      "deflator": "CPI"
    }
  }
}
```

### Bundled estimators and linkage constraints

For multi-component estimators (IV, AIPW/DML, synth), the manuscript may reveal that some axes are **linked** (e.g., the same covariate set used across components). In that case, search should enforce joint variation rather than inventing independent mix-and-match combinations.

Record this using a `bundle` block in `coefficient_vector_json` as described in `specification_tree/REVEALED_SEARCH_SPACE.md`.

### Budgeted enumeration for intractable axes

Some axes are combinatorial (especially covariate inclusion). When full enumeration is infeasible, the spec-search agent may generate a **budgeted, reproducible subsample** of specs.

Contract:

- the generation rule must be documented in the relevant spec-tree module (e.g., `rc/controls/subset/*` in `specification_tree/modules/robustness/controls.md`),
- the output must record the random seed, draw index, and the realized choice set in `coefficient_vector_json`,
- the total budget used should be recorded in the paper’s `SPECIFICATION_SEARCH.md`.

## 6) DML contract (important)

Treat DML as an **estimation wrapper** inside maintained assumptions, not as a design family:

- DML ATE/ATT (IRM/PLR) can be `rc/estimation/dml/*` when it preserves the baseline claim object.
- CATE/policy learning belongs in `explore/*` unless the baseline claim is heterogeneous.

See `specification_tree/modules/estimation/dml.md`.

## 7) Non-estimate rows (diag/sens/post/explore)

Diagnostics, sensitivity objects, post-processing, and exploration are **not** part of the default “core estimate” table.
To avoid schema conflicts (e.g., missing p-values) and to make linkage explicit, prefer **separate output tables**:

- `specification_results.csv`: estimate-like rows only (`baseline`, `design/*`, `rc/*`)
- `inference_results.csv`: inference-only recomputations (`infer/*`), linked to the base estimate via `spec_run_id`
- `diagnostics_results.csv`: diagnostics only (`diag/*`)
- `spec_diagnostics_map.csv`: join table linking specs ↔ diagnostics

### A) `inference_results.csv` (recommended schema)

One row per inference recomputation of an existing estimate row.

- `paper_id`
- `inference_run_id` (unique within paper)
- `spec_run_id` (the base estimate row being recomputed)
- `spec_id` (typed `infer/*`)
- `spec_tree_path`
- `baseline_group_id`
- `coefficient`, `std_error`, `p_value`, `ci_lower`, `ci_upper`
- `n_obs`, `r_squared` (blank allowed)
- `cluster_var` (blank allowed; record the clustering variable(s) if applicable)
- `coefficient_vector_json` (required; include inference metadata and any warnings)
- `run_success` (0/1)
- `run_error` (empty string allowed; required when `run_success=0`)

### B) `diagnostics_results.csv` (recommended schema)

- `paper_id`
- `diagnostic_run_id` (unique within paper)
- `diag_spec_id` (typed `diag/*` id)
- `spec_tree_path`
- `diagnostic_scope` (one of: `paper`, `baseline_group`, `spec`, `design_variant`)
- `diagnostic_context_id` (stable hash/key of inputs used)
- `diagnostic_json` (required; full output payload)
- `run_success` (0/1)
- `run_error` (empty string allowed; required when `run_success=0`)

### C) `spec_diagnostics_map.csv` (foreign-key linking)

Each row links one spec run to one diagnostic run:

- `paper_id`
- `spec_run_id`
- `diagnostic_run_id`
- `relationship` (e.g., `computed_for_spec`, `shared_invariant_check`)

This supports the common situation where:

- some diagnostics are invariant and shared across many specs (e.g., RD manipulation of the running variable), while
- others are spec-dependent (e.g., IV first-stage strength under a particular control set).

### D) When you must keep a single CSV

If tooling constraints require storing diagnostics in the same CSV as estimates:

- include a typed `spec_id` (`diag/*`),
- allow empty numeric fields, and
- store outputs in `coefficient_vector_json.diagnostic`.

But the **separate-table + join-map** approach is preferred.
