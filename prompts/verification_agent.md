# Verification Agent Instructions

Use this prompt to **audit and classify** the outputs of a completed specification search for paper **{PAPER_ID}**.

Your job is *not* to run more regressions. Your job is to decide which recorded specifications are **actual tests of the paper’s core hypothesis(es)** (i.e., robustness checks or alternative implementations of the *same* estimand), and which are **not comparable** (placebos, different outcomes, different treatments, heterogeneity-only, diagnostics, etc.).

This classification is paper-specific and may involve identifying **multiple baseline specifications** (e.g., multiple “main table” columns) and assigning each robustness spec to the baseline it best corresponds to.

---

## Inputs

**Package directory**: `{EXTRACTED_PACKAGE_PATH}`

You should expect to find:
- `specification_results.csv` (required)
- `SPECIFICATION_SEARCH.md` (optional but usually present)
- Original replication code and paper documentation (optional; use only as needed)

---

## Outputs (REQUIRED)

Create a verification folder:

`agentic_specification_search/data/verification/{PAPER_ID}/`

Write three files:

1) `agentic_specification_search/data/verification/{PAPER_ID}/verification_baselines.json`

2) `agentic_specification_search/data/verification/{PAPER_ID}/verification_spec_map.csv`

3) `agentic_specification_search/data/verification/{PAPER_ID}/VERIFICATION_REPORT.md`

Do **not** overwrite or modify `specification_results.csv`.

---

## Definitions (what counts as a “test of the core hypothesis”)

### Baseline specification
A **baseline** is a specification that corresponds to a paper’s canonical claim / main result. A paper can have:
- one baseline, or
- multiple baselines (e.g., separate outcomes, subgroups, or main-table columns).

### Core test (what we want to keep later)
A specification is a **core test** if it is a meaningful alternative implementation of the *same claim* as a baseline, such as:
- controls variations (add/drop controls)
- sample restrictions (pre-registered subsamples, alternative trimming rules)
- inference variations (cluster choices, robust vs clustered, etc.)
- FE structure variations that preserve the estimand (e.g., add time FE)
- functional form changes that preserve interpretation (log vs IHS, etc.)

A specification is **not** a core test if it is primarily:
- a placebo test (fake treatment, pre-trends, unaffected outcomes)
- a different outcome that is not part of the baseline claim
- a different treatment/exposure that changes the causal object (unless the paper’s claim explicitly treats it as an equivalent definition)
- heterogeneity-only results (interaction terms) unless the heterogeneity is itself a baseline claim
- diagnostics / sanity checks / descriptive regressions that are not testing the claim

When uncertain, be conservative: classify as non-core and explain why.

---

## Step-by-step procedure

### Step 0: Load and sanity-check the results
Open `{EXTRACTED_PACKAGE_PATH}/specification_results.csv`.
Confirm the schema is the expected standardized one (columns like `spec_id`, `spec_tree_path`, `outcome_var`, `treatment_var`, `coefficient`, `std_error`, `p_value`, `fixed_effects`, `controls_desc`, `cluster_var`, `sample_desc`).

If many rows have missing or non-finite coefficient/SE, note it in the report and mark them as invalid.

### Step 1: Identify candidate baselines
Use the following precedence rules:
1. Any specs with `spec_id == "baseline"` are baseline candidates.
2. Any specs with `spec_tree_path` containing `#baseline` are baseline candidates.
3. If there are multiple “baseline-like” rows, do not force them into one: keep multiple baselines if they correspond to distinct canonical claims.

### Step 2: Define baseline claim(s)
For each baseline candidate, define a **baseline claim object**:
- What is the hypothesis being tested in words?
- What is the outcome concept?
- What is the treatment/exposure concept?
- What is the expected sign/direction (if the paper has one)?

Use `SPECIFICATION_SEARCH.md` and documentation in the package to recover the intent if needed.

Group baselines into **baseline groups** when multiple baseline specs represent the *same* claim (e.g., Table 2 col1 vs col2 differing only by controls).

### Step 3: Classify each specification row
For each `spec_id` row in `specification_results.csv`, decide:

1) Which baseline group (if any) it is *trying* to test.
2) Whether it is a **core test** of that baseline group.
3) If it is a core test, which **baseline spec** it is most comparable to (the closest baseline candidate).

Use the row’s fields to determine what changed:
- outcome change: `outcome_var`
- treatment change: `treatment_var`
- sample change: `sample_desc`
- FE change: `fixed_effects`
- controls change: `controls_desc`
- inference change: `cluster_var`
- method family change: infer from `spec_tree_path` (methods vs robustness vs placebo)

If the row appears mis-extracted (e.g., wrong coefficient chosen) or the variation clearly does not correspond to the described claim, mark `is_valid=0` and explain.

### Step 4: Write outputs

#### 4.1 `verification_baselines.json`
JSON schema (strict):
```json
{
  "paper_id": "{PAPER_ID}",
  "package_path": "{EXTRACTED_PACKAGE_PATH}",
  "verified_at": "YYYY-MM-DD",
  "verifier": "verification_agent",
  "baseline_groups": [
    {
      "baseline_group_id": "G1",
      "claim_summary": "One sentence: the canonical claim in words.",
      "expected_sign": "+ | - | 0 | unknown",
      "baseline_spec_ids": ["baseline", "baseline_table2_col1"],
      "baseline_outcome_vars": ["..."],
      "baseline_treatment_vars": ["..."],
      "notes": "Any important nuance about equivalence across baselines."
    }
  ],
  "global_notes": "High-level findings about definition issues, mismatches, etc."
}
```

#### 4.2 `verification_spec_map.csv`
CSV schema (strict; include all columns exactly):
- `paper_id`
- `spec_id`
- `spec_tree_path`
- `outcome_var`
- `treatment_var`
- `baseline_group_id` (empty if none)
- `closest_baseline_spec_id` (empty if none)
- `is_baseline` (0/1)
- `is_core_test` (0/1)
- `category` (one of: `core_controls`, `core_sample`, `core_inference`, `core_fe`, `core_funcform`, `core_method`, `noncore_placebo`, `noncore_alt_outcome`, `noncore_alt_treatment`, `noncore_heterogeneity`, `noncore_diagnostic`, `invalid`, `unclear`)
- `why` (<= 25 words; concrete reason referencing fields)
- `confidence` (0.0–1.0)

Every spec in `specification_results.csv` must appear exactly once in this CSV.

#### 4.3 `VERIFICATION_REPORT.md`
Include:
- Baseline groups found (and baseline spec_ids)
- Counts: total specs, core specs, non-core specs, invalid, unclear
- Category counts
- Top 5 most suspicious rows (wrong outcome/treatment, placebo mis-tagged, etc.)
- Concrete recommendations for fixing the underlying spec-search script if the *baseline claim itself* seems wrong.

---

## Quality bar / checks

Before finishing:
- Ensure the CSV covers every `spec_id` exactly once.
- Ensure every `baseline_group_id` referenced in the CSV exists in the JSON.
- Be conservative: only classify a spec as core if it is clearly a test of a baseline claim.

