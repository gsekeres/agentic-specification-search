# Verification Agent Instructions (Post-Run Audit)

Use this prompt to **audit and classify** the outputs of a completed, surface-driven specification run for paper **{PAPER_ID}**.

You do **not** run new regressions. You verify:

1) baseline claim objects (baseline groups),
2) whether executed rows preserve those claim objects (core RC) or drifted, and
3) whether any rows are invalid/mis-extracted.

---

## Inputs

- **Package directory**: `{EXTRACTED_PACKAGE_PATH}`
- Expected files:
  - `specification_results.csv` (required)
  - `inference_results.csv` (optional; if present, contains `infer/*` recomputations linked to `spec_run_id`)
  - `SPECIFICATION_SEARCH.md` (optional but usually present)
  - `SPECIFICATION_SURFACE.json` (recommended; if present, treat as the pre-run baseline-group plan)
  - `diagnostics_results.csv`, `spec_diagnostics_map.csv` (optional)

---

## Outputs (REQUIRED)

Create:

`data/verification/{PAPER_ID}/`

Write:

1) `data/verification/{PAPER_ID}/verification_baselines.json`
2) `data/verification/{PAPER_ID}/verification_spec_map.csv`
3) `data/verification/{PAPER_ID}/VERIFICATION_REPORT.md`

Do not overwrite `specification_results.csv`.

---

## Core definitions

### Baseline group (claim object)

A baseline group is one claim object:

- outcome concept
- treatment/exposure concept
- estimand concept
- target population

Surface-driven runs should already label each row with `baseline_group_id`. Your job is to verify that this grouping is coherent and adjust if needed (with explanation).

### Core-eligible namespaces (mechanical default)

These namespaces are *eligible* to be core RC **by default**:

- `baseline`
- `design/*`
- `rc/*`

Other namespaces are non-core by default (`diag/*`, `sens/*`, `post/*`, `explore/*`). In a surface-driven run, they should not appear in `specification_results.csv`.
Inference variants (`infer/*`) should appear only in `inference_results.csv` (if that file is present), not in `specification_results.csv`.

---

## Step-by-step procedure

### Step 0: Load and sanity-check results

Open `{EXTRACTED_PACKAGE_PATH}/specification_results.csv` and confirm:

- `spec_run_id` exists and is unique within the paper
- `baseline_group_id` exists (surface-driven requirement)
- `spec_id` is typed and consistent with `spec_tree_path`
- `run_success` exists (0/1) and failures have a concrete `run_error`
- numeric fields are finite for executed rows (or clearly marked missing with reason in JSON)
- no `infer/*` rows are present in `specification_results.csv`

Flag any violations as `invalid`.

### Step 1: Verify baseline groups against the surface (if present)

If `{EXTRACTED_PACKAGE_PATH}/SPECIFICATION_SURFACE.json` exists:

- check that the surface’s baseline groups match what appears in results,
- note any missing baseline groups or spurious groups,
- check linkage assumptions for bundled estimators (linked vs unlinked adjustment).

If the surface is absent, infer baseline groups conservatively from baseline-like specs.

### Step 2: Identify baseline specs inside each baseline group

For each `baseline_group_id`, identify baseline rows:

- `spec_id == "baseline"` or `spec_id` starts with `baseline__`
- or rows explicitly labeled as baseline in `SPECIFICATION_SEARCH.md` / surface baseline specs

If a baseline group has multiple baseline specs (e.g., Table 2 col1 vs col2), keep them; they are part of the revealed surface.

### Step 3: Classify each executed row

For each row, decide:

1) `is_valid` (0/1)
2) `is_baseline` (0/1)
3) `is_core_test` (0/1)
4) `category`

Default for `is_valid`:

- start with `run_success==1` (from `specification_results.csv`), then
- override to 0 if the focal estimate is clearly mis-extracted (wrong variable, wrong sign due to coding error, nonsensical p-values, etc.).

Default `category` by namespace:

- `baseline` or `design/*` → `core_method`
- `rc/controls/*` → `core_controls`
- `rc/sample/*` → `core_sample`
- `rc/fe/*` → `core_fe`
- `rc/form/*` → `core_funcform`
- `rc/preprocess/*` → `core_preprocess`
- `rc/data/*` → `core_data`
- `rc/weights/*` → `core_weights`

Then override mechanically if the row drifted away from the baseline claim object:

- outcome/treatment concept changed → non-core (`noncore_alt_outcome` / `noncore_alt_treatment`)
- clear subpopulation change (and not a baseline group) → `noncore_population_change`
- heterogeneous-effect-only output when baseline is average → `noncore_heterogeneity`

Be conservative: if unsure, mark non-core and explain.

### Step 4: Write required outputs

#### 4.1 `verification_baselines.json`

Include:

- `baseline_groups` with `claim_summary`, `expected_sign` (if inferable), baseline spec_run_ids/spec_ids, and notes.
- If you edited baseline-group assignments relative to the surface, explain in `global_notes`.

#### 4.2 `verification_spec_map.csv`

Include one row per spec-run row with at least:

- `paper_id`
- `spec_run_id`
- `spec_id`
- `spec_tree_path`
- `outcome_var`
- `treatment_var`
- `baseline_group_id`
- `closest_baseline_spec_run_id` (blank allowed)
- `is_baseline` (0/1)
- `is_valid` (0/1)
- `is_core_test` (0/1)
- `category`
- `why` (≤ 25 words; concrete)
- `confidence` (0.0–1.0)

#### 4.3 `VERIFICATION_REPORT.md`

Include:

- baseline groups found (and baseline spec_run_ids/spec_ids)
- counts: total rows, core, non-core, invalid, unclear
- category counts
- top issues (e.g., outcome/treatment drift, missing diagnostics linkage, duplicated spec_run_id)
- recommendations for improving the surface or runner script

---

## Quality bar

- Every `spec_run_id` in `specification_results.csv` appears exactly once in `verification_spec_map.csv`.
- Every baseline group referenced in the CSV exists in the JSON.
- Explanations are concrete and anchored in observable row fields.
- run `python scripts/validate_agent_outputs.py --paper-id {PAPER_ID}` and ensure it reports 0 `ERROR` issues
