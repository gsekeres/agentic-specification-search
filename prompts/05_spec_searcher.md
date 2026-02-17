# Specification Runner Agent (Surface-Driven) Instructions

Use this prompt to **run** a specification search for paper **{PAPER_ID}** using an **approved specification surface**.

This is the *execution* stage. Do **not** redefine the universe of specs here: if the surface is missing something important, stop and request a surface update.

---

## Inputs

- **Package directory**: `{EXTRACTED_PACKAGE_PATH}`
- **Surface file (required)**: `{EXTRACTED_PACKAGE_PATH}/SPECIFICATION_SURFACE.json`
- **Spec tree**: `specification_tree/` (typed designs + modules)
- **Method reference**: `prompts/reference/estimation_methods.md` (Stata→Python translation guide)

Important environment constraint:
- Stata is not available. Implement using **Python** and/or **R**.
- **Consult `prompts/reference/estimation_methods.md`** for translation recipes before implementing.

---

## Outputs (REQUIRED)

Write the following to `{EXTRACTED_PACKAGE_PATH}` (the **top-level** extracted package directory, NOT a subfolder like `Codes-and-data/`):

1) `specification_results.csv` (estimates only: `baseline`, `design/*`, `rc/*`)
2) `SPECIFICATION_SEARCH.md` (run log + what was executed vs skipped)

If the surface includes an inference plan (canonical inference + optional variants), also write:

3) `inference_results.csv` (inference-only recomputations: `infer/*`, linked to a base `spec_run_id`)

If the surface includes a diagnostics plan, also write:

4) `diagnostics_results.csv` (diagnostics only: `diag/*`)
5) `spec_diagnostics_map.csv` (spec-run ↔ diagnostic-run links)

If you intentionally execute any non-core objects (not part of the default core surface), write them to separate tables (do not mix them into `specification_results.csv`):

6) `exploration_results.csv` (exploration only: `explore/*`)
7) `sensitivity_results.csv` (sensitivity only: `sens/*`)
8) `postprocess_results.csv` (post-processing only: `post/*`)

**Important**: Input data files may be in subfolders (e.g., `Codes-and-data/`, `data/`), but all output files listed above must be written directly to `{EXTRACTED_PACKAGE_PATH}/`, not to any subfolder. The analysis script should use separate paths for reading input data vs writing outputs.

Also save your executable script to:

- `scripts/paper_analyses/{PAPER_ID}.py` (repo-relative path)

---

## Core contract (must obey)

Use the typed spec-tree contract in:

- `specification_tree/CONTRACT.md`
- `specification_tree/SPECIFICATION_SURFACE.md`

Hard rules:

- Every estimate-like row must have a typed `spec_id` and a unique `spec_run_id`.
- Every row must include `baseline_group_id` from the surface (do not invent new groups during execution).
- Do not write `infer/*` rows into `specification_results.csv`.
- Do **not** run `explore/*`, `sens/*`, `diag/*`, or `post/*` into `specification_results.csv`.
  - Diagnostics go to `diagnostics_results.csv` only.

---

## Step 0: Load + validate the surface

Open `SPECIFICATION_SURFACE.json` and validate:

- `paper_id` matches `{PAPER_ID}`
- baseline groups exist and are non-empty
- `inference_plan.canonical` exists for each baseline group (canonical inference for estimate rows)
- budgets + sampling seed exist (when combinatorial axes are included)
- linkage constraints for bundled estimators are explicit when relevant (`linked_adjustment`)

Record a brief summary in `SPECIFICATION_SEARCH.md` (designs, #baseline groups, budgets, seed).

---

## Step 1: Reproduce the baseline spec(s) exactly

For each `baseline_group_id` in the surface:

1) run the baseline spec(s) exactly as in the paper’s code,
   - emit a `spec_id=baseline` row, and
   - emit any additional baseline rows only if they are explicitly listed in `core_universe.baseline_spec_ids`,
2) record the scalar focal estimate in `coefficient/std_error/p_value`,
3) store the full coefficient/vector output in `coefficient_vector_json`,
4) include the *exact* outcome/treatment variable names used.

---

## Step 2: Execute the approved core surface

For each baseline group, run the surface’s core universe:

- `design/*` variants (within-design implementations)
- `rc/*` variants (estimand-preserving robustness)

All baseline/design/rc rows in `specification_results.csv` must be estimated under the baseline group’s **canonical inference choice** (as specified in the surface).

If the surface requests additional inference variants, recompute uncertainty (SEs/p-values/intervals) and write those rows to `inference_results.csv` instead of `specification_results.csv`.

### Budgeted combinatorics (controls)

If the surface includes control-subset sampling:

- sample only within the control-count envelope (`controls_count_min/max`) after mandatory controls,
- generate a reproducible set of draws using the surface seed,
- record the draw index + realized included/excluded controls in `coefficient_vector_json` for each draw.

### Bundled estimators (IV, AIPW/DML, synth)

If the surface says `linked_adjustment=true` for a baseline group:

- vary controls jointly across bundle components (do not mix-and-match component control sets).

Always record a `bundle` block in `coefficient_vector_json` for bundled estimators (see `specification_tree/REVEALED_SEARCH_SPACE.md`).

---

## Step 3 (optional): Execute surface diagnostics into a separate table

If `diagnostics_plan` is present in the surface:

- compute each `diag/*` object at the specified `scope` (`baseline_group`, `spec`, etc.)
- write one row per diagnostic run in `diagnostics_results.csv`
- link diagnostic runs to spec runs via `spec_diagnostics_map.csv`

Linkage must be explicit so it is clear whether a diagnostic was computed:

- under the **same** covariate set as a given estimate (scope `spec`), or
- once as an invariant check (scope `baseline_group`).

---

## Step 4: Write outputs

### 4.1 `specification_results.csv` (estimates only)

Must include at least:

- `paper_id`
- `spec_run_id`
- `spec_id`
- `spec_tree_path` (must reference a spec-tree `.md` node; include `#anchor` when possible; use `custom` only when unavoidable)
- `baseline_group_id`
- `outcome_var`, `treatment_var`
- `coefficient`, `std_error`, `p_value`
- `ci_lower`, `ci_upper` (blank allowed)
- `n_obs`, `r_squared` (blank allowed)
- `coefficient_vector_json`
- `sample_desc`, `fixed_effects`, `controls_desc`, `cluster_var` (blank allowed)
- `run_success` (0/1)
- `run_error` (empty string allowed; required when `run_success=0`)

Every planned spec should appear as a row, even if it fails. For failures:

- set `run_success=0`
- set **all scalar numeric fields** (`coefficient`, `std_error`, `p_value`, `ci_lower`, `ci_upper`, `n_obs`, `r_squared`) to `NaN`
- record a short `run_error` string (exception message or concrete failure reason)
- set `coefficient_vector_json` to a JSON object with at least `{"error": run_error, "error_details": {...}}`
  - `error_details` must be a JSON object with at least: `stage`, `exception_type`, `exception_message` (optionally include a `traceback_tail`)
  - keep `run_error` single-line; put rich structured context under `error_details`; store any partial outputs under `partial`

#### Required `coefficient_vector_json` schema (all successful rows)

For `run_success=1` rows, `coefficient_vector_json` must be a JSON object with reserved audit keys:

- `coefficients`: full coefficient vector (dict)
- `inference`: the inference choice used for the scalar `std_error`/`p_value` in this row (matches the baseline group’s **canonical** inference choice for estimate rows; equals the variant for `infer/*` rows)
- `software`: runner language/version + key package versions (exact)
- `surface_hash`: deterministic hash of `SPECIFICATION_SURFACE.json` used for this run
- `design`: must include a non-empty `design.{design_code}` object. For `baseline` and `rc/*` rows, copy the baseline group’s `design_audit` from the surface verbatim into `design.{design_code}` (so results remain interpretable when detached from code). For `design/*` rows, start from the surface `design_audit` and override any design-defining parameters changed by the variant.

Top-level schema rule (required): do not invent arbitrary new top-level keys. Put design-specific objects under `design`, wrapper configuration under `estimation` (when applicable), and any remaining paper-specific fields under `extra` so downstream parsing stays stable across designs.

Use additional typed blocks when relevant (required for the corresponding `rc/*` prefixes): `controls`, `sample`, `fixed_effects`, `preprocess`, `estimation`, `weights`, `data_construction`, `functional_form`, `joint`.

Optional helper (recommended): you may import `scripts/agent_output_utils.py` to construct contract-compliant payloads (`make_success_payload`, `make_failure_payload`, `error_details_from_exception`, `surface_hash`, `software_block`).

Example (estimate row):

```json
{
  "coefficients": {"treat": 0.12, "x1": 0.03},
  "inference": {"spec_id": "infer/se/cluster/unit", "params": {"cluster_var": "unit_id"}},
  "software": {"runner_language": "python", "runner_version": "3.10.15", "packages": {"pandas": "2.3.3"}},
  "surface_hash": "sha256:...hex...",
  "design": {"difference_in_differences": {"estimator": "twfe", "panel_unit": "unit_id", "panel_time": "year"}},
  "controls": {"spec_id": "rc/controls/loo/drop_x1", "family": "loo", "dropped": ["x1"], "n_controls": 5}
}
```

For `rc/form/*` rows, `coefficient_vector_json` must include a `functional_form` object with a non-empty `interpretation` (and for binarize/threshold operations include `threshold`, `direction`, `units`). See `specification_tree/modules/robustness/functional_form.md`.

### 4.2 `inference_results.csv` (inference-only; if run)

One row per `(spec_run_id, infer spec_id)` recomputation. Must include at least:

- `paper_id`
- `inference_run_id` (unique within paper)
- `spec_run_id` (the base estimate row being recomputed)
- `spec_id` (typed `infer/*`)
- `spec_tree_path` (must reference a spec-tree `.md` node; include `#anchor` when possible; use `custom` only when unavoidable)
- `baseline_group_id`
- `coefficient`, `std_error`, `p_value` (computed under this inference choice)
- `ci_lower`, `ci_upper` (blank allowed)
- `n_obs`, `r_squared` (blank allowed)
- `coefficient_vector_json` (required; include inference metadata and any warnings)
- `run_success` (0/1)
- `run_error` (empty string allowed; required when `run_success=0`)

### 4.3 `SPECIFICATION_SEARCH.md`

Include:

- surface summary (baseline groups, budgets, seed)
- counts: planned vs executed vs failed (with reasons)
- any deviations (e.g., a spec variant skipped because infeasible in this package)
- exact software stack used (Python/R packages, versions if available)

### 4.4 Diagnostics tables (if run)

Follow the schemas in `specification_tree/CONTRACT.md`.

---

## Quality checks before finishing

- `spec_run_id` is unique within `{PAPER_ID}`
- every executed row’s `spec_tree_path` points to an existing design/module node
- every executed row’s `spec_tree_path` includes a `.md` path (or is exactly `custom`)
- no `diag/*` rows appear in `specification_results.csv`
- no `infer/*` rows appear in `specification_results.csv`
- `coefficient_vector_json` is valid JSON for every row (for failures include at least `{"error": run_error, "error_details": {...}}`)
- all `rc/form/*` rows include `coefficient_vector_json.functional_form`
- all `run_success=1` rows include `coefficient_vector_json.coefficients`, `.inference`, `.software`, `.surface_hash`, and a non-empty `coefficient_vector_json.design.{design_code}`
- run `python scripts/validate_agent_outputs.py --paper-id {PAPER_ID}` and ensure it reports 0 `ERROR` issues
