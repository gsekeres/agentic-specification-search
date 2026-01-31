# Agentic Specification Search

A systematic, reproducible framework for running specification curve analyses on economics research papers using AI agents.

## Overview

This project implements a **formulaic specification search** methodology where:

1. **Specification trees** define standardized specifications for each empirical method
2. **AI agents** systematically run all specifications from the trees
3. **Unified results** aggregate findings across papers for meta-analysis

## Project Structure

```
agentic_specification_search/
├── README.md                           # This file
├── unified_results.csv                 # Aggregated results from all papers
│
├── specification_tree/                 # Markdown tree defining specs
│   ├── INDEX.md                        # Master index of all spec types
│   ├── methods/                        # Method-specific specifications
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
│   ├── download_agent.md               # Package download automation
│   └── paper_classifier.md             # Method classification
│
├── scripts/                            # Python infrastructure
│   ├── datacite_fetcher.py             # Fetch metadata from DataCite
│   ├── classify_papers.py              # Classify papers by method
│   ├── manifest.py                     # Download tracking
│   ├── create_unified_csv.py           # Aggregate results
│   └── paper_analyses/                 # Per-paper estimation scripts
│       ├── _template.py
│       └── {PAPER_ID}.py
│
├── data/
│   ├── metadata/
│   │   ├── packages_2025_classified.jsonl
│   │   └── packages_all.jsonl
│   ├── downloads/
│   │   ├── raw_packages/               # Downloaded ZIP files
│   │   └── extracted/                  # Unzipped packages
│   └── tracking/
│       ├── completed_analyses.jsonl
│       └── download_manifest.jsonl
│
└── docs/
    ├── PAPER_SPECIFICATIONS.md         # Paper-level documentation
    └── METHODOLOGY.md                  # Methodology justification
```

## Quick Start

### 1. Set Up Environment

```bash
# Install Python dependencies
pip install -r requirements.txt

# Verify Claude Code is available
claude --version
```

### 2. Download Packages

```bash
# Open Chrome and log into openICPSR
# Then launch the download agent:

# Using Claude Code:
# Paste the prompt from prompts/download_agent.md
```

### 3. Run Specification Search

For each paper:

```bash
# Launch specification search agent:
# Use prompts/spec_search_agent.md with the paper's path
```

### 4. Aggregate Results

```bash
python scripts/create_unified_csv.py
```

## Specification Tree

The specification tree defines **standardized specifications** for each empirical method:

| Method | Spec File | Key Variations |
|--------|-----------|----------------|
| Difference-in-Differences | `methods/difference_in_differences.md` | FE, controls, samples, modern estimators |
| Event Study | `methods/event_study.md` | Window, reference period, pre-trends |
| Regression Discontinuity | `methods/regression_discontinuity.md` | Bandwidth, polynomial, kernel |
| Instrumental Variables | `methods/instrumental_variables.md` | First stage, weak IV, LIML |
| Panel Fixed Effects | `methods/panel_fixed_effects.md` | FE structure, clustering |
| Cross-Sectional OLS | `methods/cross_sectional_ols.md` | Controls, functional form |
| Discrete Choice | `methods/discrete_choice.md` | Logit/probit, marginal effects |
| Dynamic Panel | `methods/dynamic_panel.md` | GMM, lag structure |

**Robustness checks** (applied to all methods):

| Check | File | Description |
|-------|------|-------------|
| Leave-One-Out | `robustness/leave_one_out.md` | Drop each covariate |
| Single Covariate | `robustness/single_covariate.md` | Treatment + 1 control |
| Sample Restrictions | `robustness/sample_restrictions.md` | Subsamples, time windows |
| Clustering | `robustness/clustering_variations.md` | Different SE levels |
| Functional Form | `robustness/functional_form.md` | Logs, polynomials |

## Output Format

### unified_results.csv

| Column | Description |
|--------|-------------|
| `paper_id` | Package identifier (e.g., "223561-V1") |
| `journal` | AER, AEJ-Applied, etc. |
| `spec_id` | Specification identifier (e.g., "did/fe/twoway") |
| `spec_tree_path` | Path in specification tree |
| `coefficient` | Point estimate |
| `std_error` | Standard error |
| `p_value` | p-value |
| `coefficient_vector_json` | Full model output as JSON |
| ... | See `prompts/spec_search_agent.md` for full schema |

### coefficient_vector_json

```json
{
  "treatment": {"var": "policy", "coef": 0.05, "se": 0.02, "pval": 0.01},
  "controls": [{"var": "age", "coef": 0.1, "se": 0.05, "pval": 0.04}],
  "fixed_effects": ["state", "year"],
  "diagnostics": {"first_stage_F": 45.2}
}
```

## Paper Selection

Target: **100 papers** from AEA journals

| Journal | Target |
|---------|--------|
| AER | 25 |
| AEJ-Applied | 25 |
| AEJ-Policy | 20 |
| AEJ-Macro | 10 |
| AEJ-Micro | 10 |
| JEL/JEP | 10 |

Selection criteria:
- Has data files in openICPSR
- Random sample (not method- or topic-stratified)
- Method type determined during analysis

## Software

**We do NOT use Stata.** All analyses use open-source tools:

**Python:**
- `pandas`, `numpy`
- `statsmodels`
- `linearmodels` (panel data)
- `pyfixest` (high-dimensional fixed effects)

**R:**
- `fixest`
- `plm`, `lfe`
- `did`, `synthdid`

## References

- Simonsohn, U., Simmons, J. P., & Nelson, L. D. (2020). Specification curve analysis. *Nature Human Behaviour*, 4(11), 1208-1214.
- AEA Data and Code Availability Policy: https://www.aeaweb.org/journals/data
