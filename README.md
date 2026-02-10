# Agentic Specification Search

Replication package for "Editorial Screening when Science is Cheap" (Fishman & Sekeres, 2025). This repository implements an automated specification search framework that applies AI agents to run large-scale robustness analyses on empirical economics papers, then estimates mixture, dependence, and counterfactual screening models from the resulting specification-level data.

## Overview

1. **Specification trees** define standardized robustness specifications for each empirical method (DiD, IV, RD, panel FE, etc.)
2. **AI agents** (Claude Code) systematically execute all specifications from the trees on each paper's replication package
3. **Verification agents** audit the raw results for correctness
4. **Estimation pipeline** fits mixture models, estimates within-paper dependence, and computes counterfactual disclosure requirements

The current sample covers **99 papers** from AEA journals (AER, AEJ-Applied, AEJ-Policy, AEJ-Macro, AEJ-Micro, AER: Insights, AEA P&P), yielding ~7,500 specifications (~4,900 verified-core).

## Project Structure

```
agentic_specification_search/
├── README.md
├── unified_results.csv                 # Aggregated results across all papers
│
├── specification_tree/                 # Specification tree definitions
│   ├── INDEX.md
│   ├── methods/                        # Method-specific specs
│   │   ├── difference_in_differences.md
│   │   ├── event_study.md
│   │   ├── regression_discontinuity.md
│   │   ├── instrumental_variables.md
│   │   ├── panel_fixed_effects.md
│   │   ├── cross_sectional_ols.md
│   │   ├── discrete_choice.md
│   │   └── dynamic_panel.md
│   └── robustness/                     # Universal robustness checks
│       ├── leave_one_out.md
│       ├── single_covariate.md
│       ├── sample_restrictions.md
│       ├── clustering_variations.md
│       └── functional_form.md
│
├── prompts/                            # Agent prompts
│   ├── spec_search_agent.md            # Main specification search agent
│   ├── verification_agent.md           # Result verification agent
│   ├── download_agent.md               # Package download automation
│   └── paper_classifier.md             # Method classification
│
├── scripts/                            # Infrastructure scripts
│   ├── create_unified_csv.py           # Aggregate per-paper results
│   ├── select_new_papers.py            # Stratified paper selection
│   ├── batch_download.py               # Batch download helper
│   ├── 01_collect_dois.py              # DOI collection
│   ├── 02_identify_journals.py         # Journal identification
│   └── paper_analyses/                 # Per-paper estimation scripts
│       └── {PAPER_ID}.py
│
├── estimation/                         # Estimation pipeline
│   ├── run_all.py                      # Master pipeline runner
│   ├── scripts/                        # Numbered analysis scripts
│   │   ├── 00_summarize_verification.py
│   │   ├── 01a_build_i4r_claim_map.py
│   │   ├── 01b_build_i4r_oracle_claim_map.py
│   │   ├── 01_build_claim_level.py
│   │   ├── 02_build_spec_level.py
│   │   ├── 03_extract_i4r_baseline.py
│   │   ├── 04_fit_mixture.py           # Mixture model estimation
│   │   ├── 05_estimate_dependence.py   # AR(1) dependence estimation
│   │   ├── 06_counterfactual.py        # Disclosure counterfactual
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
│   │   └── make_figures.jl             # All figures (Julia/PyPlot)
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
│   │   └── extracted/                  # Unzipped replication packages (99 papers)
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
2. Run the spec search agent (see `prompts/spec_search_agent.md`)
3. Run the verification agent (see `prompts/verification_agent.md`)
4. Rebuild unified results: `python scripts/create_unified_csv.py`
5. Re-run the pipeline: `python estimation/run_all.py --all`

## Specification Tree

| Method | File | Key Variations |
|--------|------|----------------|
| Difference-in-Differences | `methods/difference_in_differences.md` | FE, controls, samples, modern estimators |
| Event Study | `methods/event_study.md` | Window, reference period, pre-trends |
| Regression Discontinuity | `methods/regression_discontinuity.md` | Bandwidth, polynomial, kernel |
| Instrumental Variables | `methods/instrumental_variables.md` | First stage, weak IV, LIML |
| Panel Fixed Effects | `methods/panel_fixed_effects.md` | FE structure, clustering |
| Cross-Sectional OLS | `methods/cross_sectional_ols.md` | Controls, functional form |
| Discrete Choice | `methods/discrete_choice.md` | Logit/probit, marginal effects |
| Dynamic Panel | `methods/dynamic_panel.md` | GMM, lag structure |

**Robustness checks** (applied universally): leave-one-out, single-covariate, sample restrictions, clustering variations, functional form.

## Output Format

### unified_results.csv

Each row is one specification from one paper:

| Column | Description |
|--------|-------------|
| `paper_id` | Package identifier (e.g., `223561-V1`) |
| `claim_id` | Claim within the paper |
| `spec_id` | Specification identifier |
| `spec_type` | Specification tree category |
| `coefficient` | Point estimate |
| `std_error` | Standard error |
| `t_statistic` | t-statistic |
| `p_value` | p-value |
| `n_obs` | Number of observations |
| `is_original` | Whether this is the paper's original specification |
| `ran_successfully` | Whether execution completed |

## Software

All analyses use open-source tools (no Stata):

- **Python**: `pandas`, `numpy`, `statsmodels`, `linearmodels`, `pyfixest`, `scipy`
- **R**: `fixest`, `plm`, `lfe`, `did`, `synthdid`
- **Julia**: `PyPlot`, `DataFrames`, `CSV`, `JSON3`, `Distributions`, `KernelDensity`
