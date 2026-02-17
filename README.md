# Agentic Specification Search

Replication package for "Editorial Screening when Science is Cheap" (Fishman & Sekeres, 2026). This repository implements an automated specification search framework that applies AI agents to run large-scale robustness analyses on empirical economics papers, then estimates mixture, dependence, and counterfactual screening models from the resulting specification-level data.

## Overview

1. **Typed specification tree** defines designs + orthogonal modules (RC, inference, diagnostics, sensitivity, exploration)
2. **Specification surface (per paper)** defines the executable universe (constraints, budgets, sampling) *before* any models run
3. **Runner agents** execute the approved surface and write structured outputs
4. **Verification agents** audit baseline groups + core eligibility and flag invalid/drifted rows
5. **Estimation pipeline** fits mixture models, estimates within-paper dependence, and computes counterfactual disclosure requirements

The current sample covers **99 papers** from AEA journals (AER, AEJ-Applied, AEJ-Policy, AEJ-Macro, AEJ-Micro, AER: Insights, AEA P&P), yielding ~7,500 specifications (~4,900 verified-core).

## Project Structure

```
agentic_specification_search/
├── README.md
├── unified_results.csv                 
├── unified_inference_results.csv        # Inference-only rows (infer/*), if present
│
├── specification_tree/                 # Specification tree definitions
│   ├── INDEX.md
│   ├── designs/                        # Design/identification families (within-design implementations)
│   └── modules/                        # Orthogonal modules (rc/*, infer/*, diag/*, sens/*, explore/*, post/*)
│
├── prompts/                            # Agent prompts
│   ├── 01_replicator.md                # (Optional) replicate author results + translate code
│   ├── 02_paper_classifier.md          # Design-family classification (pre-surface)
│   ├── 03_spec_surface_builder.md      # Build SPECIFICATION_SURFACE.json (pre-run)
│   ├── 04_spec_surface_verifier.md     # Critique/edit surface (pre-run)
│   ├── 05_spec_searcher.md             # Runner: execute approved surface (run stage)
│   ├── 06_post_run_verifier.md         # Post-run audit and core classification
│   └── 07_CLEANUP.md                   # Optional disk cleanup guidance
│
├── scripts/                            # Infrastructure scripts
│   ├── validate_agent_outputs.py       # Mechanical validation of agent outputs
│   ├── paper_replications/             # Per-paper replication scripts (optional)
│   │   └── {PAPER_ID}.py
│   └── paper_analyses/                 # Per-paper surface-driven runner scripts
│       └── {PAPER_ID}.py
│
├── estimation/                         # Estimation pipeline
│   ├── run_all.py
│   ├── scripts/
│   │   ├── 00_build_unified_results.py
│   │   ├── 00_summarize_verification.py
│   │   ├── 01a_build_i4r_claim_map.py
│   │   ├── 01b_build_i4r_oracle_claim_map.py
│   │   ├── 01_build_claim_level.py
│   │   ├── 02_build_spec_level.py
│   │   ├── 03_extract_i4r_baseline.py
│   │   ├── 04_fit_mixture.py
│   │   ├── 05_estimate_dependence.py
│   │   ├── 06_counterfactual.py
│   │   ├── 07_i4r_discrepancies.py
│   │   ├── 08_inference_audit.py
│   │   ├── 09_write_overleaf_tables.py
│   │   ├── 10_bootstrap_mixture_ci.py
│   │   ├── 11_journal_subgroup.py
│   │   ├── 12_posterior_assignment.py
│   │   ├── 13_counterfactual_montecarlo.py
│   │   ├── 14_effective_sample_size.py
│   │   ├── 15_summary_statistics.py
│   │   ├── 16_variance_decomposition.py
│   │   ├── 17_build_paper_catalog.py
│   │   ├── 18_mixture_comparison_table.py
│   │   ├── 19_sensitivity_tables.py
│   │   └── make_figures.jl
│   ├── data/                           # Intermediate datasets
│   │   ├── claim_level.csv
│   │   ├── spec_level.csv
│   │   ├── spec_level_verified.csv
│   │   ├── spec_level_verified_core.csv
│   │   └── i4r_comparison.csv
│   ├── results/                        # Estimation outputs (JSON)
│   │   ├── mixture_params_abs_t.json
│   │   ├── dependence.json
│   │   ├── counterfactual_params.json
│   │   └── ...
│   └── figures/                        # Generated figures (PDF)
│
├── data/
│   ├── metadata/
│   │   ├── aea_package_to_journal.jsonl    # Full AEA package universe
│   │   └── ...
│   ├── downloads/
│   │   ├── raw_packages/               # Downloaded ZIP files
│   │   └── extracted/                  # Unzipped replication packages
│   └── tracking/
│       ├── spec_search_status.json     # Per-paper analysis status
│       ├── completed_analyses.jsonl
│       └── download_manifest.jsonl
│
├── i4r/                                # I4R replication data (Sample A)
└── non-empirical-figures/              # Standalone theory figures
```

## Running the Pipeline

### Python environment

This repo requires **Python >= 3.10** (needed by `pyfixest`).

For a fully pinned environment (recommended for reproducing the included results):

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-py310.lock
```

For a minimal (unpinned) environment:

```bash
python -m pip install -r requirements.txt
```

### Full pipeline

```bash
cd agentic_specification_search
python estimation/run_all.py --all
```

This runs four stages in order:

1. **Data construction** (`--data`): Scripts 00-03, 07-08 build `unified_results.csv`, `claim_level.csv`, `spec_level.csv`, and I4R comparison datasets from per-paper outputs in each paper's extracted directory.

2. **Estimation** (`--est`): Scripts 04-06 (+ 09 when available).
   - `04_fit_mixture.py` — fits folded/truncated normal mixture models (K=2,3,4) to the |t|-statistic distribution
   - `05_estimate_dependence.py` — estimates AR(1) within-paper dependence parameter phi
   - `06_counterfactual.py` — computes counterfactual disclosure requirements (m_new)
   - `09_write_overleaf_tables.py` — generates LaTeX tables for the paper (requires data-stage outputs)

3. **Figures** (`--figs`): `make_figures.jl` generates all PDF figures (requires Julia with PyPlot).

4. **Extensions** (`--extensions`): Scripts 10-19 run bootstrap CIs, journal subgroup analysis, Monte Carlo validation, sensitivity tables, paper catalog, etc.

### Individual stages

```bash
python estimation/run_all.py --data         # Data construction only
python estimation/run_all.py --est          # Estimation only
python estimation/run_all.py --figs         # Figures only
python estimation/run_all.py --extensions   # Extension analyses only
```

### Adding new papers

1. Download replication packages to `data/downloads/extracted/{PAPER_ID}/`
2. (Optional) Replicate baseline results / translate code (see `prompts/01_replicator.md`)
3. Classify designs (see `prompts/02_paper_classifier.md`)
4. Build a surface (see `prompts/03_spec_surface_builder.md`)
5. Pre-run audit/edit surface (see `prompts/04_spec_surface_verifier.md`)
6. Run the approved surface (see `prompts/05_spec_searcher.md`)
7. Post-run audit + core classification (see `prompts/06_post_run_verifier.md`)
8. Validate outputs: `python scripts/validate_agent_outputs.py --paper-id {PAPER_ID}`
9. Re-run the pipeline: `python estimation/run_all.py --all`

## Specification Tree

Design files live in `specification_tree/designs/` and enumerate **within-design** estimator implementations (`design/*`).

Universal (cross-design) modules live in `specification_tree/modules/` and enumerate:

- robustness checks (`rc/*`)
- inference variants (`infer/*`) (recorded as inference outputs, not new estimates)
- diagnostics (`diag/*`)
- sensitivity analysis (`sens/*`)
- post-processing (`post/*`)
- exploration (`explore/*`)

The paper-specific executable object is `SPECIFICATION_SURFACE.json` (see `specification_tree/SPECIFICATION_SURFACE.md`). The runner executes only `baseline`, `design/*`, and `rc/*` into `specification_results.csv` using a **canonical inference choice** per baseline group. Additional inference variants (`infer/*`) are recorded separately in `inference_results.csv` (keyed by `spec_run_id`). Diagnostics (if planned) are written to `diagnostics_results.csv` and linked via `spec_diagnostics_map.csv`.

See `specification_tree/INDEX.md` for the canonical file/module index and typed namespace rules.

## Output Format

### unified_results.csv

Each row is one specification from one paper:

| Column | Description |
|--------|-------------|
| `paper_id` | Package identifier (e.g., `223561-V1`) |
| `journal` | Journal (if available from package metadata) |
| `paper_title` | Paper title (if available from package metadata) |
| `spec_run_id` | Unique executed-row identifier (within paper) |
| `baseline_group_id` | Baseline claim object identifier (from the pre-run surface) |
| `spec_id` | Typed estimate identifier (`baseline`, `design/*`, `rc/*`). Inference variants live in `inference_results.csv`. |
| `spec_tree_path` | Defining node path under `specification_tree/` (file + optional section anchor) |
| `outcome_var` | Dependent variable name |
| `treatment_var` | Treatment/exposure variable name |
| `coefficient` | Point estimate |
| `std_error` | Standard error |
| `p_value` | p-value |
| `ci_lower` | CI lower bound (optional; can be empty) |
| `ci_upper` | CI upper bound (optional; can be empty) |
| `n_obs` | Number of observations |
| `r_squared` | \(R^2\) (optional; can be empty) |
| `coefficient_vector_json` | Full output payload (JSON object; required). Must include `coefficients`, `inference`, `software`, `surface_hash` (+ axis blocks like `controls` / `sample` / `functional_form` when applicable). Failures include `error` + `error_details`. Keep top-level schema stable (use `design`/`extra` for extensions). |
| `sample_desc` | Sample description (optional) |
| `fixed_effects` | Fixed effects description (optional) |
| `controls_desc` | Controls description (optional) |
| `cluster_var` | Clustering variable(s) used (optional) |
| `run_success` | 0/1 success flag for the executed row |
| `run_error` | Short error string when `run_success=0` |
| `spec_fingerprint` | Deterministic hash of the spec signature (duplicate tracking) |
| `dup_group_size` | Number of rows sharing the same fingerprint (within paper) |
| `dup_rank` | Deterministic rank within the fingerprint group |
| `dup_canonical_spec_run_id` | The canonical `spec_run_id` for the fingerprint group |
| `dup_is_duplicate` | 0/1 flag for non-canonical duplicates (`dup_rank>1`) |

### inference_results.csv

Each row is one inference-only recomputation (`infer/*`) for a reference estimate row:

The data construction stage also concatenates these per-paper files into `unified_inference_results.csv` (same schema).

| Column | Description |
|--------|-------------|
| `paper_id` | Package identifier |
| `inference_run_id` | Unique inference-row identifier (within paper) |
| `spec_run_id` | Reference estimate-row identifier being recomputed |
| `baseline_group_id` | Baseline claim object identifier |
| `spec_id` | Typed inference identifier (`infer/*`) |
| `spec_tree_path` | Defining inference node path under `specification_tree/` |
| `coefficient` | Point estimate (typically matches the reference row) |
| `std_error` | Standard error under this inference choice |
| `p_value` | p-value under this inference choice |
| `coefficient_vector_json` | Full output payload (JSON object; required). Must include `coefficients`, `inference`, `software`, `surface_hash` (+ inference metadata/warnings). Failures include `error` + `error_details`. Keep top-level schema stable (use `design`/`extra` for extensions). |
| `run_success` | 0/1 success flag for the inference recomputation |
| `run_error` | Short error string when `run_success=0` |

## Software

All analyses use open-source tools (no Stata):

- **Python**: `pandas`, `numpy`, `statsmodels`, `linearmodels`, `pyfixest`, `scipy`
- **R**: `fixest`, `plm`, `lfe`, `did`, `synthdid`
- **Julia**: `PyPlot`, `DataFrames`, `CSV`, `JSON3`, `Distributions`, `KernelDensity`
