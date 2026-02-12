# Spec Tree Refactor Notes (Progress + Open Decisions)

This is a working document to track the ongoing refactor of the specification tree into a **typed, orthogonal taxonomy** and to improve the replication/search workflow.

## Why refactor (problems observed)

### 1) Axis duplication

Pre-refactor, method files re-listed generic axes (controls/sample/SE/functional form) that also existed elsewhere. In the new structure, universal axes live in `specification_tree/modules/robustness/*` and design files should reference them rather than duplicate them.

### 2) Mixed object types under one “spec” namespace

The current “specification” list mixes:

- estimand-preserving re-estimates,
- inference-only recomputations,
- diagnostics (assumption checks),
- post-processing (MHT/FDR),
- exploration (heterogeneity/CATE/alt outcomes/treatments).

These are not the same statistical object; we need a minimal type system.

### 3) Naming drift / inconsistencies

Examples in current data/prompts:

- `robust/het/*` vs `robust/heterogeneity/*`
- `robust/se/*` living inside `clustering_variations.md`
- “functional form” files containing alternative estimators (quantiles/PPML/etc.)

### 4) “Tree distance” not aligned with statistical axes

Downstream dependence estimation uses `spec_tree_path` parsing that mostly distinguishes *file names* and a single section anchor. This is too coarse to be an economically/statistically meaningful measure of “distance” between specs.

### 5) Exploration is valuable but under-specified

Verification reports show many papers where the majority of enumerated runs are “non-core” (alt outcomes, heterogeneity), which:

- inflates raw spec counts without adding core robustness,
- makes it harder to interpret what the core evidence object is,
- motivates explicitly separating **replication robustness** from **exploration**.

## Target architecture (current decision)

See `specification_tree/ARCHITECTURE.md` for the conceptual contract.

Key decisions:

- Typed `spec_id` namespaces: `design/*`, `rc/*`, `infer/*`, `sens/*`, `diag/*`, `post/*`, `explore/*` (+ reserved `baseline`).
- Explicitly distinguish **RC (robustness checks)** from **sensitivity analysis**:
  - `rc/*` = estimand-preserving re-specifications under maintained assumptions (controls/sample/coding/etc.).
  - `sens/*` = assumption-relaxation objects (bounds, breakdown points), not ordinary regressions.
- One canonical home per axis: method/design files contain only *design-specific* variations; universal axes live in modules.
- Treat **DML as an estimation/nuisance-learning module**, not a primary method.
- Encourage “one-axis-at-a-time” robustness by default; allow a small set of justified interactions.
- Add an explicit **data pre-processing / coding** module (variable construction, dichotomization, standardization, index construction).

## Implementation plan (repo changes)

### A) Spec-tree file re-org

Implemented:

- `specification_tree/designs/` contains typed design files.
- `specification_tree/modules/{robustness,inference,diagnostics,sensitivity,exploration,postprocess,estimation}/` contains typed universal modules.
- `specification_tree/COVERAGE.md` tracks major degrees-of-freedom and remaining gaps.

### B) Prompts

Implemented:

- Added a surface-driven workflow:
  - `prompts/02_paper_classifier.md` (design classification)
  - `prompts/03_spec_surface_builder.md` (define `SPECIFICATION_SURFACE.json`)
  - `prompts/04_spec_surface_verifier.md` (pre-run audit/edit surface)
- Updated `prompts/05_spec_searcher.md` to be a **runner-only** prompt: execute the approved surface and write typed outputs (estimates table + optional diagnostics tables).
- Updated `prompts/06_post_run_verifier.md` to audit surface-driven outputs keyed on `spec_run_id` and `baseline_group_id` and to classify core vs non-core mechanically by namespace.

### C) Pipeline + data schema

Implemented (partial):

- `unified_results.csv` carries `spec_run_id` and `baseline_group_id` and buckets summary stats by `spec_id` namespace/design.
- Updated `estimation/scripts/02_build_spec_level.py` to prefer merging verification maps on `spec_run_id` when present (fallback to legacy merge keys otherwise).

## Open econometric questions (need explicit policy)

1) **Functional-form changes vs estimand changes**
   - When is a log/asinh transformation treated as “same outcome concept” vs a different estimand?
2) **Heterogeneity / CATE**
   - Default: `explore/*` unless the baseline estimand is explicitly heterogeneous (e.g., subgroup ATT).
   - Need a rule for “baseline heterogeneity claim”: subgroup-only main result vs interaction specification.
3) **IV instrument set changes**
   - Changing instruments can change the LATE; should often be `explore/*` unless instruments are explicitly equivalent in the paper.
4) **RCT ITT vs TOT**
   - TOT/LATE are different estimands; default to exploration unless baseline is TOT.
5) **Set-level post-processing**
   - MHT/FDR/spec-curve are not regressions; should be `post/*` objects computed after the fact, not “specs”.
6) **Coding + dichotomization**
   - When does thresholding/binning count as a sensitivity check vs an estimand change (exploration)?
   - Rule-of-thumb: if baseline treatment/outcome is already binary/binned, nearby cutoffs can be sensitivity; otherwise, threshold sweeps are exploration.

## Evidence from verification artifacts (what to learn from)

Some verification reports show heavy non-core shares due to:

- component outcomes vs index outcomes (alt outcomes explosion),
- extensive heterogeneity interaction runs,
- duplicate extraction across output files.

These motivate:

- explicit exploration outputs,
- deduplication/duplicate-flagging in aggregation,
- clarifying what “counts” as a core robustness spec for baseline claims.
