# Specification Surface (Per-Paper Operational Object)

This document defines the **specification surface**: the paper-specific object that determines *what will actually be run*, how large the search space is, and how we sample from it when it is intractable.

It sits between:

- the global menus in `specification_tree/` (designs + modules), and
- the per-paper execution script produced by the spec-search agent.

The surface is designed to be **auditable** and **reviewable before running models**.

## 1) Definition

For a paper \(i\) and a baseline claim object (baseline group) \(g\), the **specification surface** \(S_{ig}\) consists of:

1) a **core-eligible universe** \(\mathcal{U}^{core}_{ig}\) (typed namespaces: `baseline`, `design/*`, `rc/*`),
2) an optional **exploration universe** \(\mathcal{U}^{explore}_{ig}\) (`explore/*`),
3) a **diagnostics plan** \(\mathcal{D}_{ig}\) (`diag/*`) with an explicit scope/linkage rule,
4) an **inference plan** (one canonical inference choice for estimate rows + optional `infer/*` variants recorded separately),
5) **constraints** that rule out invalid/incoherent combinations (linkage constraints, size caps, feasibility constraints),
6) a **budget** (max specs to run) and a **sampling plan** when \(|\mathcal{U}|\) is too large.

Diagnostics (`diag/*`) and sensitivity analysis (`sens/*`) are kept in the repo, but **excluded from the default surface** until we decide how they enter the theory pipeline.

## 2) Workflow (two-stage verification around the surface)

1) **Identify baseline claim objects** (baseline groups): outcome/treatment/estimand/population concepts.
2) **Identify baseline specs**: the paper’s canonical implementations for each baseline group.
3) **Define a candidate specification surface** \(S_{ig}\):
   - pick relevant design file(s),
   - pick relevant universal RC axes and define an inference plan,
   - define the admissible universe and budgets,
   - define constraints from the manuscript surface.
4) **Surface review (verifier intervention)**:
   - verifier critiques/edits the surface *before any runs* (especially constraints and linkage rules).
5) **Execute specs**: run the approved surface (deterministic core battery + sampled draws as needed).
6) **Post-run verification**: verifier audits outputs (core vs non-core classification, validity checks) as in the current verification protocol.

This makes the most consequential “what is the universe?” choices explicit and reviewable.

## 2.1) Diagnostics: explicit scope + linkage (recommended)

Diagnostics are often “standard” for a design, but they do **not** have a clean one-to-one relationship with specs:

- some depend only on the data-generating object (e.g., RD running-variable manipulation),
- some depend on the full estimating equation (e.g., IV first-stage strength with a particular control set),
- some are meaningful only for one estimator family (e.g., Goodman–Bacon for TWFE).

To keep this auditable, the surface should specify for each baseline group:

1) **which diagnostics to compute** (`diag/*` ids),
2) **the diagnostic scope** (paper / baseline_group / spec / design_variant), and
3) **the linkage rule** (which specs are “covered” by which diagnostic runs).

Operationally, prefer separate outputs:

- `specification_results.csv` for estimates (`baseline`, `design/*`, `rc/*`),
- `inference_results.csv` for inference-only recomputations (`infer/*`),
- `diagnostics_results.csv` for diagnostics (`diag/*`),
- `spec_diagnostics_map.csv` linking spec runs ↔ diagnostic runs.

See `specification_tree/CONTRACT.md`.

## 3) Manuscript-derived constraints (the “surface” part)

The manuscript reveals not only *which axes exist*, but often **implicit caps** on complexity. These are useful constraints on \(\mathcal{U}\).

### A) Control-count envelope (recommended default)

Let the paper’s main specifications for baseline group \(g\) include one or more “main table” columns/rows with different control sets.
Define:

- \(k_{min}\) = minimum number of controls used in the paper’s main specs for \(g\),
- \(k_{max}\) = maximum number of controls used in the paper’s main specs for \(g\).

Default constraint for control-subset sampling:

- sample only control sets with size in \([k_{min}, k_{max}]\) (after adding any mandatory controls).

Interpretation:

- this treats the paper’s own reported control complexity as the revealed “reasonable” envelope,
- it prevents the agent from generating ultra-high-dimensional “kitchen sink” sets the paper never considers,
- it makes the sampled control universe comparable across papers.

If the paper has only one main spec, set \(k_{min}=k_{max}=k_{baseline}\) by default.

### B) Linkage constraints for bundled estimators

For IV, AIPW/DML, synth, etc., define whether adjustment is **linked** across components (see `specification_tree/REVEALED_SEARCH_SPACE.md`).
If linked, the control-count envelope applies jointly across components.

### C) Other feasible caps (optional)

Depending on the design, the surface may also include caps like:

- max number of fixed-effect dimensions,
- max polynomial degree / spline knots,
- max number of subgroup splits (exploration),
- max number of alternative outcome/treatment definitions.

## 4) A minimal surface schema (recommended)

Store the pre-run surface as a machine-readable file (e.g., `SPECIFICATION_SURFACE.json`) in the package directory.

Sketch:

```json
{
  "paper_id": "112233-V1",
  "baseline_groups": [
    {
      "baseline_group_id": "G1",
      "design_code": "difference_in_differences",
      "claim_object": {
        "outcome_concept": "earnings",
        "treatment_concept": "policy adoption",
        "estimand_concept": "ATT",
        "target_population": "eligible workers"
      },
      "baseline_specs": [
        {"label": "Table2-Col1", "n_controls": 6, "controls": ["x1","x2", "..."]},
        {"label": "Table2-Col2", "n_controls": 12, "controls": ["x1","x2", "..."]}
      ],
      "constraints": {
        "controls_count_min": 6,
        "controls_count_max": 12,
        "linked_adjustment": true
      },
      "core_universe": {
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
          "params": {"cluster_var": "unit_id"},
          "notes": "Used for all baseline/design/rc estimate rows."
        },
        "variants": [
          {"spec_id": "infer/se/hc/hc1", "params": {}, "notes": "Robust-only (no clustering)."},
          {"spec_id": "infer/se/cluster/state", "params": {"cluster_var": "state"}, "notes": "Coarser clustering as a stress test."}
        ]
      },
      "diagnostics_plan": [
        {
          "diag_spec_id": "diag/difference_in_differences/pretrends/joint_test",
          "scope": "spec",
          "linkage": "computed_for_each_core_spec"
        }
      ],
      "budgets": {
        "max_specs_core_total": 150,
        "max_specs_controls_subset": 80
      },
      "sampling": {
        "seed": 12345,
        "controls_subset_sampler": "stratified_size"
      }
    }
  ]
}
```

### Multi-design papers (important)

A single paper can have multiple baseline claim objects that are estimated using different design families (e.g., one main table is IV, another is RD).

This is well-posed: store the design family at the **baseline-group level** via `design_code`, and define `core_universe` separately for each baseline group.

See `specification_tree/SPEC_UNIVERSE_AND_SAMPLING.md` for sampling requirements and output JSON blocks.
