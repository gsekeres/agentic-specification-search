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
- `coefficient_vector_json` (required; for failures must include `error` + `error_details`; for successes must use the reserved-key schema with the full coefficient vector stored under `coefficients`)
- `sample_desc`, `fixed_effects`, `controls_desc`, `cluster_var` (empty allowed when not applicable)
- `run_success` (0/1)
- `run_error` (empty string allowed; required when `run_success=0`)

These rows should be computed under the baseline group’s **canonical inference choice** recorded in the surface. Alternative uncertainty recomputations belong in `inference_results.csv`.

For `rc/form/*` rows, `coefficient_vector_json` must include a `functional_form` object with enough detail to interpret the coefficient (see `specification_tree/modules/robustness/functional_form.md`).

### `coefficient_vector_json` schema (required)

`coefficient_vector_json` must be a JSON object with reserved top-level keys for audit metadata.
To avoid collisions between coefficient names and metadata keys, store the full coefficient vector under a dedicated `coefficients` dictionary.

Top-level schema rule (required): do not invent arbitrary new top-level keys. Put design-specific objects under `design`, wrapper configuration under `estimation` (when applicable), and any remaining paper-specific fields under `extra` so the payload stays mechanically stable across designs.

Minimum required blocks for **successful** estimate-like rows (`run_success=1`):

- `coefficients`: object mapping parameter name → estimate (or an empty object only if truly unavailable)
- `inference`: object describing the inference choice used for the scalar `std_error`/`p_value` in this row
- `software`: object describing the runner language/version and key package versions
- `surface_hash`: deterministic hash of the `SPECIFICATION_SURFACE.json` used for this run
- `design`: must include a non-empty `design.{design_code}` object with design-defining parameters (see `specification_tree/DESIGN_AUDIT_FIELDS.md`). For `baseline` and `rc/*` rows, this should match the baseline group’s `design_audit` in `SPECIFICATION_SURFACE.json` (verbatim), so outputs remain interpretable when detached from code.

Minimum required blocks for **failed** rows (`run_success=0`):

- `error`: non-empty string (mirrors `run_error`)
- `error_details`: non-empty object with at least `stage`, `exception_type`, `exception_message`

Example failure payload:

```json
{
  "error": "Singular matrix",
  "error_details": {
    "stage": "estimation",
    "exception_type": "LinAlgError",
    "exception_message": "Singular matrix",
    "traceback_tail": "..."
  }
}
```

Recommended for failures:

- keep *all* scalar numeric columns missing (`NaN`) and store any partial outputs under `partial` in JSON
- still include `inference` / `software` / `surface_hash` when available (so failures remain auditable)

Inference block rule (required):

- for estimate rows (`baseline`, `design/*`, `rc/*`), set `inference.spec_id` to the baseline group’s **canonical** inference choice from the surface (`inference_plan.canonical.spec_id`)
- for inference-only rows in `inference_results.csv` (`infer/*`), set `inference.spec_id` equal to the row’s `spec_id`

RC axis blocks (required when the corresponding `rc/*` prefix is present):

- `rc/controls/*` → `controls` (see `specification_tree/modules/robustness/controls.md`)
- `rc/sample/*` → `sample` (see `specification_tree/modules/robustness/sample.md`)
- `rc/fe/*` → `fixed_effects` (see `specification_tree/modules/robustness/fixed_effects.md`)
- `rc/preprocess/*` → `preprocess` (see `specification_tree/modules/robustness/preprocessing.md`)
- `rc/estimation/*` → `estimation` (see `specification_tree/modules/estimation/dml.md`)
- `rc/weights/*` → `weights` (see `specification_tree/modules/robustness/weights.md`)
- `rc/data/*` → `data_construction` (see `specification_tree/modules/robustness/data_construction.md`)
- `rc/form/*` → `functional_form` (see `specification_tree/modules/robustness/functional_form.md`)
- `rc/joint/*` → `joint` (see `specification_tree/modules/robustness/joint.md`)

Axis-block self-identification (required): each required RC axis block must include a `spec_id` field equal to the row’s `spec_id` (e.g., `coefficient_vector_json.controls.spec_id == spec_id`). This makes the audit metadata self-describing even when detached from the CSV row.

### Run-success rule (required)

Every planned spec should appear as a row in `specification_results.csv`, even if it fails, so the run is auditable against the surface budget.

- If the run produced a valid focal estimate and uncertainty, set `run_success=1` and leave `run_error=""`.
- If the run failed (or produced unusable focal outputs), set `run_success=0`, set **all scalar numeric fields** (`coefficient`, `std_error`, `p_value`, `ci_lower`, `ci_upper`, `n_obs`, `r_squared`) to `NaN`, provide a short single-line `run_error` string, and set `coefficient_vector_json` to a JSON object with at least `{"error": run_error, "error_details": {...}}` (use `error_details` for structured context like exception type, stage, traceback tail, and key inputs).

## 3) `spec_tree_path` rules (auditability)

`spec_tree_path` must point to the *defining node* in the markdown tree:

- Prefer paths under `specification_tree/` (not absolute file system paths).
- Use a spec-tree markdown file path (must include `.md`); use `custom` only when unavoidable.
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

## 7) Non-core outputs (diag/sens/post/explore)

Diagnostics, sensitivity objects, post-processing, and exploration are **not** part of the default “core estimate” table.
To keep the estimate table clean and to make linkage explicit, prefer **separate output tables**:

- `specification_results.csv`: estimate-like rows only (`baseline`, `design/*`, `rc/*`)
- `inference_results.csv`: inference-only recomputations (`infer/*`), linked to the base estimate via `spec_run_id`
- `diagnostics_results.csv`: diagnostics only (`diag/*`)
- `spec_diagnostics_map.csv`: join table linking specs ↔ diagnostics
- `exploration_results.csv`: exploration only (`explore/*`)
- `sensitivity_results.csv`: sensitivity only (`sens/*`)
- `postprocess_results.csv`: post-processing only (`post/*`)

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
- `coefficient_vector_json` (required; use the same reserved-key schema as estimate rows: include `coefficients`, `inference`, `software`, `surface_hash`; for failures include at least `{"error": run_error}`)
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

### D) `exploration_results.csv` (recommended schema)

Exploration outputs are heterogeneous: some are re-estimated regressions with alternative definitions, others are function-valued objects (CATE/policy rules).
Write them to `exploration_results.csv` and store the full output in `exploration_json`.

- `paper_id`
- `exploration_run_id` (unique within paper)
- `spec_run_id` (optional link to a base estimate row; empty allowed)
- `spec_id` (typed `explore/*`)
- `spec_tree_path`
- `baseline_group_id` (empty allowed if not tied to a baseline group)
- `exploration_json` (required; full output payload)
- `run_success` (0/1)
- `run_error` (empty string allowed; required when `run_success=0`)

`exploration_json` requirements:

- for `run_success=1`: include `software`, `surface_hash`, and an `exploration` block with `exploration.spec_id == spec_id`
- for `run_success=0`: include `error` and `error_details` with at least `stage`, `exception_type`, `exception_message`

### E) `sensitivity_results.csv` (recommended schema)

- `paper_id`
- `sensitivity_run_id` (unique within paper)
- `spec_run_id` (optional link to a base estimate row; empty allowed)
- `spec_id` (typed `sens/*`)
- `spec_tree_path`
- `baseline_group_id` (empty allowed)
- `sensitivity_scope` (one of: `paper`, `baseline_group`, `spec`)
- `sensitivity_context_id` (stable hash/key of inputs used; empty allowed)
- `sensitivity_json` (required; full output payload)
- `run_success` (0/1)
- `run_error` (empty string allowed; required when `run_success=0`)

`sensitivity_json` requirements:

- for `run_success=1`: include `software`, `surface_hash`, and a `sensitivity` block with `sensitivity.spec_id == spec_id`
- for `run_success=0`: include `error` and `error_details` with at least `stage`, `exception_type`, `exception_message`

### F) `postprocess_results.csv` (recommended schema)

Post-processing objects are defined on a *family* of specs (not a single regression). Store them as set-level JSON outputs:

- `paper_id`
- `postprocess_run_id` (unique within paper)
- `spec_id` (typed `post/*`)
- `spec_tree_path`
- `baseline_group_id` (empty allowed)
- `postprocess_json` (required; full output payload)
- `run_success` (0/1)
- `run_error` (empty string allowed; required when `run_success=0`)

`postprocess_json` requirements:

- for `run_success=1`: include `software`, `surface_hash`, and a `postprocess` block with `postprocess.spec_id == spec_id`
- for `run_success=0`: include `error` and `error_details` with at least `stage`, `exception_type`, `exception_message`
