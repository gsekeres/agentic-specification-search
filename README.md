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
│
├── specification_tree/                 # Specification tree definitions
│   ├── INDEX.md
│   ├── designs/                        # Design/identification families (within-design implementations)
│   └── modules/                        # Orthogonal modules (rc/*, infer/*, diag/*, sens/*, explore/*, post/*)
│
├── prompts/                            # Agent prompts
│   ├── 01_downloader.md                # Package download automation
│   ├── 02_paper_classifier.md          # Design-family classification (pre-surface)
│   ├── 03_spec_surface_builder.md      # Build SPECIFICATION_SURFACE.json (pre-run)
│   ├── 04_spec_surface_verifier.md     # Critique/edit surface (pre-run)
│   ├── 05_spec_searcher.md             # Runner: execute approved surface (run stage)
│   ├── 06_post_run_verifier.md         # Post-run audit and core classification
│   └── 07_CLEANUP.md                   # Optional disk cleanup guidance
│
├── scripts/                            # Infrastructure scripts
│   └── paper_analyses/                 # Per-paper estimation scripts
│       └── {PAPER_ID}.py
│
├── estimation/                         # Estimation pipeline
│   ├── run_all.py
│   ├── scripts/
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
│   │   ├── 08_i4r_paper_audit.py
│   │   ├── 10_inference_audit.py
│   │   ├── 11_write_overleaf_tables.py
│   │   ├── 12_bootstrap_mixture_ci.py
│   │   ├── 15_journal_subgroup.py
│   │   ├── 16_posterior_assignment.py
│   │   ├── 17_dependence_heterogeneity.py
│   │   ├── 18_sign_consistency.py
│   │   ├── 19_funnel_plot.py
│   │   ├── 20_counterfactual_montecarlo.py
│   │   ├── 21_effective_sample_size.py
│   │   ├── 22_window_surface.py
│   │   ├── 23_within_paper_dispersion.py
│   │   ├── 24_summary_statistics.py
│   │   ├── 25_variance_decomposition.py
│   │   ├── 26_build_paper_catalog.py
│   │   ├── 27_mixture_comparison_table.py
│   │   ├── 27_sensitivity_tables.py
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

### Full pipeline

```bash
cd agentic_specification_search
python estimation/run_all.py --all
```

This runs four stages in order:

1. **Data construction** (`--data`): Scripts 00-10 build `claim_level.csv`, `spec_level.csv`, and I4R comparison datasets from raw `specification_results.csv` files in each paper's extracted directory.

2. **Estimation** (`--est`): Scripts 04-06, 11.
   - `04_fit_mixture.py` — fits folded/truncated normal mixture models (K=2,3,4) to the |t|-statistic distribution
   - `05_estimate_dependence.py` — estimates AR(1) within-paper dependence parameter phi
   - `06_counterfactual.py` — computes counterfactual disclosure requirements (m_new)
   - `11_write_overleaf_tables.py` — generates LaTeX tables for the paper

3. **Figures** (`--figs`): `make_figures.jl` generates all PDF figures (requires Julia with PyPlot).

4. **Extensions** (`--extensions`): Scripts 12-27 run bootstrap CIs, leave-one-out CV, journal subgroup analysis, Monte Carlo validation, sensitivity tables, paper catalog, etc.

### Individual stages

```bash
python estimation/run_all.py --data         # Data construction only
python estimation/run_all.py --est          # Estimation only
python estimation/run_all.py --figs         # Figures only
python estimation/run_all.py --extensions   # Extension analyses only
```

### Adding new papers

1. Download replication packages to `data/downloads/extracted/{PAPER_ID}/`
2. Classify designs (see `prompts/paper_classifier.md`)
3. Build a surface (see `prompts/spec_surface_builder.md`)
4. Pre-run audit/edit surface (see `prompts/spec_surface_verifier.md`)
5. Run the approved surface (see `prompts/spec_search_agent.md`)
6. Post-run audit + core classification (see `prompts/verification_agent.md`)
7. Rebuild unified results: `python scripts/create_unified_csv.py`
8. Re-run the pipeline: `python estimation/run_all.py --all`

## Specification Tree

Design files live in `specification_tree/designs/` and enumerate **within-design** estimator implementations (`design/*`).

Universal (cross-design) modules live in `specification_tree/modules/` and enumerate:

- robustness checks (`rc/*`)
- inference variants (`infer/*`)
- diagnostics (`diag/*`)
- sensitivity analysis (`sens/*`)
- post-processing (`post/*`)
- exploration (`explore/*`)

The paper-specific executable object is `SPECIFICATION_SURFACE.json` (see `specification_tree/SPECIFICATION_SURFACE.md`). The runner executes only `baseline`, `design/*`, `rc/*`, and `infer/*` into `specification_results.csv`; if diagnostics are planned, it writes `diagnostics_results.csv` separately and links them via `spec_diagnostics_map.csv`.

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
| `spec_id` | Typed specification identifier (`baseline`, `design/*`, `rc/*`, `infer/*`, etc.) |
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
| `coefficient_vector_json` | Full output payload (JSON; required; may include bundles/focal/vector outputs) |
| `sample_desc` | Sample description (optional) |
| `fixed_effects` | Fixed effects description (optional) |
| `controls_desc` | Controls description (optional) |
| `cluster_var` | Clustering variable(s) used (optional) |

## Software

All analyses use open-source tools (no Stata):

- **Python**: `pandas`, `numpy`, `statsmodels`, `linearmodels`, `pyfixest`, `scipy`
- **R**: `fixest`, `plm`, `lfe`, `did`, `synthdid`
- **Julia**: `PyPlot`, `DataFrames`, `CSV`, `JSON3`, `Distributions`, `KernelDensity`
