# Methodology

This document describes the methodological approach behind the Agentic Specification Search project.

## Motivation

Published research findings in economics often reflect only a small subset of the specifications that researchers explored during analysis. This "researcher degrees of freedom" problem makes it difficult to assess the robustness of findings without access to the full specification space.

Our goal is to make replication and robustness auditing **mechanically auditable** at scale by:

1. Defining a **typed specification tree** (orthogonal statistical object types)
2. Defining a **per-paper specification surface** (the executable universe, constraints, and budgets) *before any models run*
3. Executing the approved surface and writing standardized outputs
4. Auditing outputs to produce a conservative **verified core**
5. Feeding the verified core into mixture/dependence/counterfactual estimation

## Key Design Decisions

### 1) Baseline claim objects and baseline groups

A paper can have multiple “main” claims (different outcomes, treatments, target populations, or even designs). We represent each main claim as a **baseline group**: one claim object with:

- outcome concept
- treatment/exposure concept
- estimand concept
- target population

This separation is central because “robustness” is meaningful **within** a baseline group, not across unrelated claim objects.

See `specification_tree/CLAIM_GROUPING.md`.

### 2) Typed specification tree (orthogonal axes)

Empirical variation mixes multiple statistical object types. To keep this auditable, every row is typed by `spec_id` namespace:

- `baseline`, `design/*`, `rc/*` are estimate-like and core-eligible by default
- `infer/*` are inference-only recomputations recorded separately (not treated as additional estimate rows)
- `diag/*` (diagnostics), `sens/*` (assumption relaxations), `post/*` (set-level transforms), and `explore/*` (concept changes) are non-core by default

Design/identification families live in `specification_tree/designs/*.md` and enumerate within-design variants (`design/{design_code}/*`). Cross-design modules live in `specification_tree/modules/*`.

See `specification_tree/ARCHITECTURE.md` and `specification_tree/INDEX.md`.

### 3) DML positioning

Double/debiased machine learning (DML) is treated as an **estimation wrapper** (nuisance learning + orthogonalization), not as a primary design. It is `rc/estimation/dml/*` only when it preserves the baseline claim object.

See `specification_tree/modules/estimation/dml.md`.

### 4) Surface-driven execution (define universe before running)

**Decision**: separate “define the universe” from “run the universe”.

Per paper, an agent produces `SPECIFICATION_SURFACE.json` that:

- enumerates baseline groups and baseline specs,
- selects the core-eligible universe (`baseline`, `design/*`, `rc/*`),
- defines an inference plan (one canonical inference choice for estimate rows + optional `infer/*` variants recorded separately),
- encodes constraints (e.g., control-count envelope, linkage rules for bundles),
- defines budgets and reproducible sampling for intractable combinatorics.

This surface is reviewed and edited *pre-run* by a verifier agent. Only then does a runner execute it.

See `specification_tree/SPECIFICATION_SURFACE.md`, `specification_tree/SPEC_UNIVERSE_AND_SAMPLING.md`, and `specification_tree/REVEALED_SEARCH_SPACE.md`.

### 5) Coefficient-vector format (carry full outputs)

**Decision**: store full model outputs and audit metadata in a single structured JSON column (`coefficient_vector_json`) alongside scalar focal estimates.

**Rationale**:
- Different methods have different output structures (OLS coefficients differ from IV first stages, which differ from event study lead/lag coefficients)
- A flexible JSON format accommodates all methods without requiring method-specific columns
- Downstream analysis can parse the JSON for specific use cases
- Full model output is preserved for reproducibility

**Contract** (enforced by validators): for successful estimate-like rows, `coefficient_vector_json` is a JSON object with reserved audit keys and the full coefficient vector nested under `coefficients` to avoid key collisions. At minimum it includes:

- `coefficients`: parameter name → estimate
- `inference`: inference choice used for the scalar `std_error`/`p_value`
- `software`: runner language/version + key package versions
- `surface_hash`: deterministic hash of the `SPECIFICATION_SURFACE.json` used for the run

For failures (`run_success=0`), scalar numeric columns are kept missing and the JSON payload must include a non-empty `error` string (plus optional structured details).

Vector-producing designs (event studies, local projections, SVAR IRFs) must declare a **scalar focal parameter** in JSON and store the full path/vector.

See `specification_tree/CONTRACT.md`.

### 6) Diagnostics as separate, linkable objects

**Decision**: diagnostics are recorded in a separate table (when run) and linked to estimates via a join map, rather than being mixed into the estimate table.

Outputs (optional, when planned in the surface):

- `diagnostics_results.csv` (one row per diagnostic run; `diag/*`)
- `spec_diagnostics_map.csv` (links `spec_run_id` ↔ `diagnostic_run_id`)

This supports diagnostics that are:

- `baseline_group`-scoped (e.g., RD McCrary density at a cutoff), or
- `spec`-scoped (e.g., IV first-stage strength under a specific control set).

See `specification_tree/modules/diagnostics/design_diagnostics.md` and `specification_tree/CONTRACT.md`.

### 7) Paper selection

**Decision**: Random sample from ALL AEA papers with data in openICPSR, stratified by journal.

**Rationale**:
- No restrictions on method type, code quality, or observational vs. experimental
- Only requirement is that data files are present
- This avoids selection bias toward papers with "good" code
- Method type is determined during analysis, not during selection
- Journal stratification ensures representation across AEA outlets

## Workflow (agents and artifacts)

At a high level:

1) **Design classification** (`prompts/02_paper_classifier.md`) → `design_code`
2) **Surface build (pre-run)** (`prompts/03_spec_surface_builder.md`) → `SPECIFICATION_SURFACE.json` + `SPECIFICATION_SURFACE.md`
3) **Surface verification (pre-run)** (`prompts/04_spec_surface_verifier.md`) → edited surface + `SPEC_SURFACE_REVIEW.md`
4) **Surface execution** (`prompts/05_spec_searcher.md`) → `specification_results.csv` + `SPECIFICATION_SEARCH.md` (+ optional diagnostics tables)
   - optional `inference_results.csv` for `infer/*` recomputations
5) **Post-run verification** (`prompts/06_post_run_verifier.md`) → `data/verification/{PAPER_ID}/...`

## Output schema (core)

### `specification_results.csv` (estimate-like rows only)

Core required fields include:

- `paper_id`, `spec_run_id`, `baseline_group_id`
- `spec_id`, `spec_tree_path`
- `outcome_var`, `treatment_var`
- `coefficient`, `std_error`, `p_value` (scalar focal estimate)
- `coefficient_vector_json` (required JSON payload; may include full vectors/bundles)
- `run_success` (0/1) and `run_error` (failure reason when `run_success=0`)

See `specification_tree/CONTRACT.md` for the full contract.

### `inference_results.csv` (inference-only rows; optional)

If inference variants are computed, they are recorded separately and linked back to estimate rows via `spec_run_id`.

See `specification_tree/CONTRACT.md` for the recommended schema.

### `verification_spec_map.csv` (post-run classification)

Verification assigns each executed row:

- a baseline-group mapping,
- a core eligibility flag,
- a category and short justification.

This produces the conservative **verified core** used downstream.

## Revealed search space and linkage constraints

The surface is constrained by the manuscript’s **revealed search space**:

- **Control-count envelope**: bounds sampled control-set sizes by what the paper reports as “main” control complexity.
- **Bundled estimators**: IV/AIPW/DML/synth often share adjustment sets across components; the surface records whether adjustment is linked and enforces joint variation when linked.

See `specification_tree/REVEALED_SEARCH_SPACE.md`.

## Quality assurance (high level)

We rely on layered guardrails:

- **Pre-run**: surface verification forces explicit constraints/budgets/linkage decisions.
- **Run-time**: runner emits stable IDs (`spec_run_id`) and records rich metadata in JSON.
- **Post-run**: verification filters drift/invalid rows and labels core vs non-core conservatively.

## Limitations

1. **Software Translation**: Original Stata code must be translated to Python/R, potentially introducing errors
2. **Variable Naming**: Agent must correctly identify variables across heterogeneous naming conventions
3. **Surface specification**: Defining the surface is itself a judgment call; the verifier stage is designed to catch incoherent expansion
4. **Data Availability**: Some papers may have restricted data not in the openICPSR package
5. **Computational constraints**: Some specs/diagnostics/sensitivity objects may be infeasible in a given package

## References

Simonsohn, U., Simmons, J. P., & Nelson, L. D. (2020). Specification curve analysis. *Nature Human Behaviour*, 4(11), 1208-1214.

Athey, S., & Imbens, G. W. (2022). Design-based analysis in difference-in-differences settings with staggered adoption. *Journal of Econometrics*, 226(1), 62-79.

Young, A. (2022). Consistency without inference: Instrumental variables in practical application. *European Economic Review*, 147, 104112.

Brodeur, A., Cook, N., & Heyes, A. (2020). Methods matter: p-hacking and publication bias in causal analysis in economics. *American Economic Review*, 110(11), 3634-3660.
