# Agentic Specification Search

Replication package for **"Editorial Screening when Science is Cheap"** (Fishman & Sekeres, 2026).

**Abstract:** We build a constrained, auditable agentic workflow that constructs an ex ante specification surface for each paper and then executes the robustness universe it admits, and we apply it to 103 empirical studies published in AEA journals. Comparing our automated runtime to a conservative human benchmark, we estimate a roughly 170-fold decline in the marginal cost of running observational specifications. We study the resulting shift in behavior as a commitment equilibrium of a screening game, where journals commit *ex ante* to acceptance rules and researchers sequentially search over dependent specifications, stop strategically, and selectively disclose evidence. The induced true- and false-positive acceptance rates trace out a purity--throughput frontier. We prove a universal information-theoretic bound on this frontier, governed by the total likelihood-ratio information a researcher can accumulate before optimally stopping. We verify that the current *de facto* practice in observational research, requiring a set of robustness checks, is an optimal mechanism; but we prove that screening collapses as testing becomes cheap unless the required number of robustness checks scales at least linearly in the inverse cost of each test. We then document, using audited ex ante specification surfaces and the robustness universes they induce, that observational social science has indeed entered a cheap-testing regime. The theory implies that to maintain conventional purity at fixed throughput, the number of qualifying robustness checks must grow at least proportionally with the cost decline; under our empirical calibration the implied disclosure requirement is on the order of 7,000 checks. This raises a serious issue for observational work going forward, and we argue for the need to develop methods to interpret sets of many specifications simultaneously, as opposed to current interpretative practice, which focuses on a handful of main specifications and a small set of robustness checks.

## How the pipeline works

```
 Download         Classify        Build surface       Verify           Execute            Audit
 replication  -->  design     -->  (pre-run         -->  surface    -->  approved       -->  outputs
 package           family          commitment)          (pre-run)       surface              (post-run)
                                                                         |
                                                                         v
                                                               specification_results.csv
                                                               inference_results.csv
                                                                         |
                                                                         v
                                                               Estimation pipeline
                                                               (mixture, dependence,
                                                                counterfactual screening)
```

The key empirical object is the **specification surface**: a per-paper, ex ante commitment that fixes the admissible universe of estimand-preserving variants before any models run. It reconstructs only the analytical forks a paper reveals (controls varied, samples restricted, functional forms changed) and constrains execution to that universe. This is the empirical counterpart of the commitment object in the screening theory.

### Agent pipeline (per paper)

Each paper passes through a six-stage agentic workflow (see `prompts/`):

| Stage | Agent prompt | Input | Output |
|-------|-------------|-------|--------|
| 1. Replicate (optional) | `01_replicator.md` | Replication package | Baseline reproduction |
| 2. Classify | `02_paper_classifier.md` | Paper + code | Design family |
| 3. Build surface | `03_spec_surface_builder.md` | Design + data | `SPECIFICATION_SURFACE.json` |
| 4. Verify surface | `04_spec_surface_verifier.md` | Surface | Edited surface + review |
| 5. Execute | `05_spec_searcher.md` | Approved surface | `specification_results.csv` |
| 6. Audit | `06_post_run_verifier.md` | Results | Verified core |

### Estimation pipeline

The estimation pipeline (`estimation/run_all.py`) aggregates per-paper outputs into structural estimates:

| Stage | Scripts | What it does |
|-------|---------|-------------|
| Data construction | 00--03, 07--08 | Build `unified_results.csv`, claim-level and spec-level datasets, I4R comparisons |
| Estimation | 04--06, 09 | Fit three-type folded-Gaussian mixture, estimate AR(1) dependence, compute counterfactual disclosure |
| Figures | `make_figures.jl` | Generate all PDF figures |
| Extensions | 10--19 | Bootstrap CIs, journal subgroups, Monte Carlo validation, variance decomposition, sensitivity |

## Project structure

```
agentic_specification_search/
├── specification_tree/           # Typed specification tree (the replication ontology)
│   ├── designs/                  #   17 design families (DID, IV, RD, RCT, panel FE, ...)
│   └── modules/                  #   Universal modules (robustness, inference, diagnostics, ...)
│
├── prompts/                      # Agent prompts (stages 01-07)
│
├── scripts/                      # Infrastructure
│   ├── download_packages.py      #   Authenticated openICPSR downloader
│   ├── extract_packages.py       #   ZIP extraction
│   ├── validate_agent_outputs.py #   Output validation
│   ├── paper_replications/       #   Per-paper replication scripts
│   └── paper_analyses/           #   Per-paper surface-driven runner scripts
│
├── estimation/                   # Estimation pipeline
│   ├── run_all.py                #   Master runner (--data, --est, --figs, --extensions)
│   ├── scripts/                  #   20 numbered scripts + Julia figure generator
│   ├── data/                     #   Intermediate datasets
│   └── figures/                  #   Generated figures (PDF)
│
├── data/
│   ├── downloads/extracted/      #   Per-paper directories with data + agent outputs
│   ├── tracking/                 #   AEA universe (4,284 packages), download logs
│   └── verification/             #   Post-run verification reports
│
├── i4r/                          # I4R validation sample (104 papers, 41 on openICPSR)
├── non-empirical-figures/        # Theory figures
│
├── unified_results.csv           # All specifications, all papers
├── unified_inference_results.csv # Inference-only recomputations
├── requirements.txt              # Python dependencies (unpinned)
└── requirements-py310.lock       # Fully pinned Python 3.10 environment
```

## Specification tree

The specification tree is a typed, orthogonal decomposition of empirical variation. Every specification row has a typed `spec_id` that makes its statistical role mechanically recoverable:

| Namespace | Object type | Core-eligible | Stored in |
|-----------|------------|---------------|-----------|
| `baseline` | Paper's canonical estimate | Yes | `specification_results.csv` |
| `design/*` | Within-design estimator variant | Yes | `specification_results.csv` |
| `rc/*` | Robustness check (estimand-preserving) | Yes | `specification_results.csv` |
| `infer/*` | Inference recomputation | No | `inference_results.csv` |
| `diag/*` | Diagnostic / falsification | No | `diagnostics_results.csv` |
| `sens/*` | Sensitivity / partial identification | No | `sensitivity_results.csv` |
| `post/*` | Set-level transform (MHT, spec curve) | No | `postprocess_results.csv` |
| `explore/*` | Concept / estimand change | No | `exploration_results.csv` |

Design files (`specification_tree/designs/`) cover 17 identification families: difference-in-differences, event study, IV, RD, RCT, shift-share, synthetic control, panel FE, cross-sectional OLS, dynamic panel, discrete choice, local projection, structural VAR, structural calibration, bunching, duration/survival, and DSGE Bayesian estimation.

See `specification_tree/INDEX.md` for the full index and `specification_tree/ARCHITECTURE.md` for the conceptual contract.

## Reproducing the results

### Requirements

- **Python >= 3.10** (needed by `pyfixest`)
- **Julia** with `PyPlot`, `DataFrames`, `CSV`, `JSON3`, `Distributions`, `KernelDensity` (for figures)

### Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip

# Fully pinned (recommended):
pip install -r requirements-py310.lock

# Or minimal:
pip install -r requirements.txt
pip install curl_cffi  # for openICPSR downloads
```

### Run the full pipeline

```bash
python estimation/run_all.py --all
```

Individual stages:

```bash
python estimation/run_all.py --data         # Data construction
python estimation/run_all.py --est          # Mixture, dependence, counterfactual
python estimation/run_all.py --figs         # Figures (requires Julia)
python estimation/run_all.py --extensions   # Bootstrap CIs, subgroups, sensitivity
```

### Downloading replication packages

Packages are hosted on openICPSR behind Cloudflare. The downloader uses `curl_cffi` with Chrome TLS impersonation and Keycloak SSO authentication.

```bash
export ICPSR_EMAIL="your@email.com"
export ICPSR_PASS="your_password"

python scripts/download_packages.py --login-only              # test login
python scripts/download_packages.py --project-ids 112431      # specific packages
python scripts/download_packages.py --sample 10 --seed 42     # random sample
python scripts/download_packages.py --journal "AER" --sample 20  # filtered sample
python scripts/extract_packages.py                             # extract all
```

### Adding a new paper

1. `python scripts/download_packages.py --project-ids {PID}`
2. Run agent stages 02 through 06 (see `prompts/`)
3. `python scripts/validate_agent_outputs.py --paper-id {PAPER_ID}`
4. `python estimation/run_all.py --all`

## Output format

### unified_results.csv

Each row is one specification from one paper. Key columns:

| Column | Description |
|--------|-------------|
| `paper_id` | Package identifier (e.g., `112431-V1`) |
| `spec_id` | Typed identifier (`baseline`, `design/*`, `rc/*`) |
| `baseline_group_id` | Claim object identifier |
| `coefficient` | Point estimate |
| `std_error` | Standard error |
| `p_value` | p-value |
| `n_obs` | Number of observations |
| `coefficient_vector_json` | Full model output (JSON with `coefficients`, `inference`, `software`, `surface_hash`) |
| `run_success` | 0/1 success flag |
| `spec_fingerprint` | Deterministic hash for duplicate tracking |

Additional columns: `journal`, `paper_title`, `outcome_var`, `treatment_var`, `ci_lower`, `ci_upper`, `r_squared`, `sample_desc`, `fixed_effects`, `controls_desc`, `cluster_var`, `run_error`, `dup_*` fields.

### unified_inference_results.csv

Each row is one inference-only recomputation (`infer/*`) linked to an estimate row via `spec_run_id`. Same scalar columns plus `inference_run_id`.

## Software

All analyses use open-source tools (no Stata):

- **Python**: `pandas`, `numpy`, `statsmodels`, `linearmodels`, `pyfixest`, `scipy`, `rdrobust`, `curl_cffi`
- **R**: `fixest`, `plm`, `lfe`, `did`, `synthdid`
- **Julia**: `PyPlot`, `DataFrames`, `CSV`, `JSON3`, `Distributions`, `KernelDensity`

## Citation

```bibtex
@article{fishmanSekeres2026,
  title={Editorial Screening when Science is Cheap},
  author={Nic Fishman and Gabriel Sekeres},
  year={2026}
}
```
