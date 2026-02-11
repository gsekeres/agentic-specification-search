# Specification Universe & Sampling Policy

This document defines how the spec-search protocol handles **intractably large** specification spaces in a principled, auditable way.

It is motivated by two facts:

1) Some axes (especially **covariate inclusion**) are central and combinatorial.
2) Full enumeration is often infeasible, but ad hoc sampling is not auditable.

This policy applies to both:

- **replication robustness** (`baseline`, `design/*`, `rc/*`, `infer/*`), and
- large exploratory families (`explore/*`) when they are intentionally run.

## 1) Define the “specification universe” first

For each baseline claim object (baseline group), define a **specification universe** \( \mathcal{U} \):

- a set of admissible specs, described as:
  - a set of axes with allowed variants, plus
  - explicit constraints that rule out invalid combinations.

Think of \(\mathcal{U}\) as the *cross-product* of allowed variants after constraints.

### What belongs in the universe definition

At minimum:

- baseline group identifiers (outcome/treatment/estimand/population concepts),
- the axis families included (controls, sample, FE, inference, etc.),
- per-axis variant menus,
- linkage constraints for bundled estimators (IV, AIPW/DML, synth),
- a computed or upper-bounded universe size \(|\mathcal{U}|\).

If \(|\mathcal{U}|\) is not computable exactly (due to variable-level subsets), record a conservative upper bound and the subset-generation rule.

## 1.1) Manuscript-derived constraints (recommended defaults)

The manuscript often reveals useful constraints on what should be considered a “reasonable” surface.
These constraints should be encoded in the universe definition \(\mathcal{U}\) rather than left implicit.

### Control-count envelope

For each baseline group, define the set of “main specs” the manuscript treats as canonical implementations (e.g., main-table columns/rows).
Let:

- \(k_{min}\) = min number of controls across main specs,
- \(k_{max}\) = max number of controls across main specs.

Recommended default for control-subset sampling:

- restrict draws to subset sizes in \([k_{min}, k_{max}]\) (after adding mandatory controls).

This keeps the sampled universe comparable to what the manuscript itself exposes and prevents the agent from inventing ultra-high-dimensional adjustments the paper never considers.

### Linkage constraints for bundled estimators

If the baseline estimator is bundled (IV, AIPW/DML), check whether covariate adjustment is shared across components.

- If linked, enforce joint control-set draws across components.
- If unlinked and clearly revealed, allow component-wise variation but cap the cross-product (avoid factorial explosion unless the manuscript reveals it).

See `specification_tree/REVEALED_SEARCH_SPACE.md` and `specification_tree/SPECIFICATION_SURFACE.md`.

## 2) Decide whether enumeration is feasible

Define a compute/reporting budget per baseline group:

- `max_specs_core_total` (e.g., 150),
- `max_specs_controls_subset` (e.g., 100),
- optional per-axis caps.

If \(|\mathcal{U}|\) exceeds the budget, do **not** expand combinatorially by default. Instead:

1) run a deterministic “core battery” (baseline + a small required set per axis), then
2) draw additional specs by **random sampling from \(\mathcal{U}\)** until the budget is met.

## 3) Sampling must be reproducible and coverage-aware

Sampling is not “pick random things”. It must satisfy:

### A) Reproducibility

- Use a deterministic seed derived from `{paper_id, baseline_group_id}` plus a global seed.
- Record the seed, sampler name, and draw indices in outputs.

### B) Coverage constraints

Aim to ensure (when possible):

- each axis variant is included at least once,
- each control variable in the candidate pool is included at least once and excluded at least once (for control-subset sampling),
- avoid degenerate repeats (exact same spec twice).

### C) Stratification (recommended default)

For combinatorial spaces driven by control inclusion:

- stratify draws by subset size (e.g., small/medium/large) to probe sensitivity across adjustment intensity,
- optionally stratify by blocks (demographics vs geography vs baseline outcomes, etc.).

## 4) Linkage constraints for bundled estimators are enforced inside \(\mathcal{U}\)

For multi-component estimators (IV, AIPW/DML), the universe must encode whether adjustment is **linked**:

- If linked, the sampled control set applies jointly to all components.
- If unlinked and the manuscript reveals that structure, allow component-wise variation but keep a *small*, auditable menu (avoid a full factorial unless revealed).

See `specification_tree/REVEALED_SEARCH_SPACE.md`.

## 5) How agents should record the universe + sampling

### A) Per-paper universe summary (in `SPECIFICATION_SEARCH.md`)

The search agent should write a short section:

- baseline groups found,
- the core universe axes included,
- budgets,
- sampler used,
- effective number of sampled specs.

### B) Machine-readable universe block (in `coefficient_vector_json`)

For sampled specs, include a `universe` and `sampling` block:

```json
{
  "universe": {
    "universe_id": "core_rc_v1",
    "baseline_group_id": "G1",
    "axes": ["controls", "sample", "fe", "inference"],
    "constraints": ["linked_adjustment=true"],
    "size_upper_bound": 1000000
  },
  "sampling": {
    "sampler": "stratified_controls_subsets",
    "seed": 12345,
    "draw_index": 17,
    "budget": {"max_specs_core_total": 150, "max_specs_controls_subset": 100}
  }
}
```

The realized choice set for that draw belongs in the relevant axis block, e.g. `controls.included/excluded` for control-subset sampling.

## 6) Where this is implemented in the tree

- Control-subset search lives in `specification_tree/modules/robustness/controls.md` (`rc/controls/subset/*`).
- Multi-axis sampled “joint” specs (when needed) should use `rc/joint/sampled/draw_{k}` with the realized combination recorded in JSON.
