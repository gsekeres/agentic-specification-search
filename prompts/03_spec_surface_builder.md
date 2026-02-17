# Specification Surface Builder Agent Instructions (Pre-Run)

Use this prompt to construct a **paper-specific specification surface** *before any models are run*.

This is the “define the universe” stage. Do **not** run regressions here.

---

## Inputs

- **Paper/package**: `{PAPER_ID}`
- **Package directory**: `{EXTRACTED_PACKAGE_PATH}`
- **Spec tree**: `specification_tree/` (typed designs + modules)

---

## Outputs (REQUIRED)

Write both files to `{EXTRACTED_PACKAGE_PATH}` (the **top-level** extracted package directory, NOT a subfolder):

1) `SPECIFICATION_SURFACE.json` (machine-readable surface)
2) `SPECIFICATION_SURFACE.md` (human-readable summary + rationale)

Do not run regressions and do not write `specification_results.csv` in this stage.

---

## Conceptual contract (must follow)

Use:

- `specification_tree/ARCHITECTURE.md`
- `specification_tree/SPECIFICATION_SURFACE.md`
- `specification_tree/REVEALED_SEARCH_SPACE.md`
- `specification_tree/SPEC_UNIVERSE_AND_SAMPLING.md`
- `specification_tree/CONTRACT.md`

Key goal: produce an auditable, reviewable surface keyed to the **baseline claim object(s)** and the manuscript’s **revealed search space**.

---

## Step 1: Identify baseline claim object(s) (baseline groups)

From the manuscript outputs and replication code, identify the paper’s interpreted “main” claim(s). Create one `baseline_group_id` per claim object.

For each baseline group, record a `claim_object`:

- `outcome_concept`
- `treatment_concept`
- `estimand_concept`
- `target_population`

Important:

- If the paper makes multiple main claims (multiple outcomes or multiple target populations as headline results), create **multiple baseline groups**.
- Do not treat heterogeneity tables or alternative outcomes as baseline groups unless the paper clearly frames them as main claims.

---

## Step 2: Identify baseline specs (canonical implementations)

For each baseline group:

- identify the canonical baseline spec(s) (often main table columns/rows),
- record the exact outcome and treatment variable names,
- list the control set(s) and count them (for the control-count envelope).

If you plan to emit multiple baseline-like rows in `specification_results.csv` for the same baseline group (beyond the single `baseline` row), list those additional baseline IDs in `core_universe.baseline_spec_ids` so the runner + validator treat them as explicit, surface-approved baselines.

If the baseline estimator is bundled (IV, AIPW/DML, synth):

- record components and whether adjustment is **linked/shared** across components (`linked_adjustment`).

Also record a small `design_audit` block for each baseline group (required):

- This should include the **design-defining parameters** that make the estimate interpretable out of context (see `specification_tree/DESIGN_AUDIT_FIELDS.md`).
- Keep it concise (typically ~3–10 keys). Include more only when it materially affects interpretation (e.g., RD bandwidth/kernel/poly order; event-study window/reference; IV instrument set).

---

## Step 3: Define the core-eligible universe (what will be run)

The default core universe contains only:

- `baseline`
- `design/*` (within-design implementations)
- `rc/*` (robustness checks)

For each baseline group, specify:

1) the baseline group’s `design_code` (file stem from `specification_tree/designs/*.md`)
2) which `rc/*` axes are feasible and high-value for this package
3) the baseline group’s **canonical inference choice** (the SE/uncertainty method used for all baseline/design/rc rows)

Do not include `explore/*`, `sens/*`, `post/*`, or `diag/*` in the core universe.

### Inference plan (separate from core universe)

Inference variants (`infer/*`) are not treated as additional estimating-equation variants. The surface should:

- pick one **canonical** inference choice per baseline group (used for all estimate rows), and
- optionally list additional `infer/*` variants to compute as separate outputs (written to `inference_results.csv`).

All inference spec_ids in the surface must be typed `infer/*` (including the canonical).

---

## Step 4: Constraints and revealed-surface guardrails

For each baseline group, define constraints that prevent incoherent expansion:

- `controls_count_min/max` (control-count envelope from main specs)
- `linked_adjustment` (for bundled estimators)
- any feasibility caps (e.g., max FE dimensions, max polynomial degree)
- functional-form policy when relevant (whether outcome/treatment transforms are treated as preserving the claim object, and the intended coefficient interpretation)

If the paper reveals multiple control sets or components that are intended to move together, enforce those linkage constraints explicitly.

---

## Step 5: Budget + sampling plan (when the universe is too large)

If an axis is combinatorial (especially controls):

- define the control pool, mandatory controls, and admissible subset-size range,
- set budgets like `max_specs_controls_subset`,
- specify a reproducible sampler (and a `seed`),
- describe how draws are stratified (e.g., by subset size).

If full enumeration is feasible, state that explicitly.

---

## Step 6: Optional diagnostics plan (separate table; not part of core universe)

You may include a `diagnostics_plan` per baseline group for standard design diagnostics (e.g., IV first-stage strength, DiD pretrends), but:

- diagnostics are written to `diagnostics_results.csv`, not to `specification_results.csv`,
- each diagnostic must specify a `scope` (`baseline_group`, `spec`, etc.) so linkage is auditable.

See `specification_tree/CONTRACT.md`.

---

## Output format: `SPECIFICATION_SURFACE.json` (minimum)

Use this shape (extend as needed):

```json
{
  "paper_id": "{PAPER_ID}",
  "created_at": "YYYY-MM-DD",
  "baseline_groups": [
    {
      "baseline_group_id": "G1",
      "design_code": "difference_in_differences",
      "design_audit": {
        "estimator": "twfe",
        "panel_unit": "unit_id",
        "panel_time": "year",
        "fe_structure": ["unit", "time"],
        "cluster_vars": ["unit"]
      },
      "claim_object": {
        "outcome_concept": "…",
        "treatment_concept": "…",
        "estimand_concept": "…",
        "target_population": "…"
      },
      "baseline_specs": [
        {
          "label": "Table2-Col1",
          "outcome_var": "y",
          "treatment_var": "d",
          "controls": ["x1", "x2"],
          "n_controls": 2
        }
      ],
      "constraints": {
        "controls_count_min": 2,
        "controls_count_max": 10,
        "linked_adjustment": true
      },
      "core_universe": {
        "baseline_spec_ids": [
          "baseline__table2_col2"
        ],
        "design_spec_ids": [
          "design/difference_in_differences/estimator/twfe"
        ],
        "rc_spec_ids": [
          "rc/controls/loo/*",
          "rc/sample/outliers/trim_y_1_99"
        ]
      },
      "inference_plan": {
        "canonical": {
          "spec_id": "infer/se/cluster/unit",
          "params": {"cluster_var": "firm_id"},
          "notes": "Matches the paper's baseline inference."
        },
        "variants": [
          {"spec_id": "infer/se/hc/hc1", "params": {}, "notes": "Robust-only (no clustering)."},
          {"spec_id": "infer/se/cluster/state", "params": {"cluster_var": "state"}, "notes": "Coarser clustering as a stress test."}
        ]
      },
      "budgets": {
        "max_specs_core_total": 150,
        "max_specs_controls_subset": 80
      },
      "sampling": {
        "seed": 12345,
        "controls_subset_sampler": "stratified_size"
      },
      "diagnostics_plan": [
        {"diag_spec_id": "diag/difference_in_differences/pretrends/joint_test", "scope": "spec"}
      ]
    }
  ]
}
```

Write a clear `SPECIFICATION_SURFACE.md` that explains:

- baseline groups and why
- what is included/excluded (and why)
- how budgets/sampling were chosen
- key linkage constraints (especially controls across bundles)
