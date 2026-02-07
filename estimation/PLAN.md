# Empirics Appendix Plan (QJE-style)

This file is the **engineering + empirical appendix plan** for the pipeline in
`agentic_specification_search/estimation/`. It serves two purposes:

1) Define what is "official" in the empirical build (scripts, inputs, outputs).
2) Lay out the empirical appendix structure and robustness grid.

---

## 0. Canonical empirical objects

### Samples
- **Sample A (paired replications):** 40 AEA-journal papers from \citet{brodeur2024newhope}. Unit = paper-level canonical claim.
- **Sample B (automated spec-search):** All papers for which we can generate a specification tree from a public replication package (95 papers total = Sample A plus 55 additional papers). Unit = specification node within paper.

### Units
- **Claim level:** \((i)\) indexes paper \(i\) in Sample A (one canonical claim per paper).
- **Specification level:** \((i,s)\) indexes a paper \(i\) and a spec-tree node \(s\).

### Evidence indices
- **Main index (used for mixture + counterfactual):** \(|Z|\), implemented as the absolute \(t\)-statistic for the focal coefficient under harmonized inference.
- **Robustness index (appendix):** \(Z_p\equiv -\log_{10}(p)\), where \(p\) is a two-sided \(p\)-value computed from \(|t|\) (normal approximation).

### Screening primitives to estimate
- **Type mixture:** \(k\in\{N,H,L\}\) with weights \(\pi_k\) and type-specific evidence laws.
- **Dependence primitive:** \(\phi\) summarizing within-paper correlation; effective independence \(\Delta=1-\phi\).
- **Cost ratio (calibration):** \(\lambda=\gamma^{\mathrm{new}}/\gamma^{\mathrm{old}}\) disciplined by replication-time ratios.
- **Witness window:** evidence window \(B\subset[0,\infty)\) on the \(Z\)-scale chosen by a separation criterion.

---

## 1. Official pipeline

### Entrypoint
```
python agentic_specification_search/estimation/run_all.py           # data + estimation + figures
python agentic_specification_search/estimation/run_all.py --all     # everything including extensions
python agentic_specification_search/estimation/run_all.py --extensions  # extension analyses only
```

### Stage 1: Data construction (scripts 00--10)
- `00_summarize_verification.py` — verification-agent summaries
- `01a_build_i4r_claim_map.py` — map independent reanalysis claims → automated baseline objects
- `01b_build_i4r_oracle_claim_map.py` — within-paper matched "oracle" spec mapping
- `01_build_claim_level.py` — claim-level dataset
- `02_build_spec_level.py` — spec-level dataset w/ tree metadata + verification labels
- `03_extract_i4r_baseline.py` — independent reanalysis benchmark objects
- `08_i4r_paper_audit.py` — paper-level comparability/audit flags
- `10_inference_audit.py` — static inference audit on scripts
- `07_i4r_discrepancies.py` — discrepancy table + LaTeX outputs

### Stage 2: Estimation (scripts 04--06, 11)
- `04_fit_mixture.py` — 3-component truncated-Gaussian mixture on |Z| (main); gamma mixture on Z_p (robustness); K-sensitivity (K=2,3,4)
- `05_estimate_dependence.py` — distance-based decay + AR(1) traversal + 4 alternative models
- `06_counterfactual.py` — counterfactual mapping + sensitivity grid
- `11_write_overleaf_tables.py` — manuscript-ready LaTeX tables

### Stage 3: Figures (Julia)
- `make_figures.jl` — all main-text and appendix figures via PyPlot/LaTeX

### Stage 4: Extension analyses (scripts 12--25)

**Mixture extensions:**
- `12_bootstrap_mixture_ci.py` — parametric bootstrap CIs for all mixture parameters (B=200)
- `13_bootstrap_lrt.py` — bootstrap LRT for K=2 vs 3 and K=3 vs 4 (B=200)
- `14_leave_one_out_cv.py` — leave-one-paper-out cross-validation for K={2,3,4}
- `15_subsample_stability.py` — subsample stability by journal + random halves
- `16_posterior_assignment.py` — posterior type probabilities P(k|Z) + stacked bar chart

**Dependence extensions:**
- `17_dependence_heterogeneity.py` — paper-level phi_i estimates + scatter vs n_specs
- `18_sign_consistency.py` — sign-flip rates by tree distance
- `19_funnel_plot.py` — |Z| vs precision funnel plot

**Counterfactual extensions:**
- `20_counterfactual_montecarlo.py` — Monte Carlo validation of binomial approximation
- `21_effective_sample_size.py` — n_eff = Delta * n visualization
- `22_window_surface.py` — separation score S(B) heat map
- `23_within_paper_dispersion.py` — within-paper |Z| box plots sorted by baseline

**Descriptive tables:**
- `24_summary_statistics.py` — summary statistics for Samples A and B
- `25_variance_decomposition.py` — between/within-paper ANOVA on |Z|

---

## 2. Official outputs

### Overleaf figures (`overleaf/tex/v8_figures/`)

**Main text (2:1 aspect ratio):**
- `fig_z_distributions_threeway.pdf`
- `fig_three_type_mixture_fit.pdf`
- `fig_counterfactual_old_vs_new.pdf`

**Appendix (from make_figures.jl):**
- `fig_i4r_agreement_verified.pdf`
- `fig_tstat_distributions_threeway_filters.pdf`
- `fig_mixture_diagnostics.pdf`
- `fig_k_sensitivity.pdf`
- `fig_mixture_fit_K2.pdf`, `fig_mixture_fit_K4.pdf`
- `fig_corr_distance.pdf`
- `fig_dependence_alternatives.pdf`
- `fig_counterfactual_sensitivity.pdf`

**Appendix (from extension scripts):**
- `fig_posterior_heatmap.pdf`
- `fig_bootstrap_mixture_ci.pdf`
- `fig_bootstrap_lrt.pdf`
- `fig_leave_one_out_cv.pdf`
- `fig_subsample_stability.pdf`
- `fig_phi_vs_nspecs.pdf`
- `fig_sign_consistency.pdf`
- `fig_funnel_plot.pdf`
- `fig_montecarlo_validation.pdf`
- `fig_effective_sample_size.pdf`
- `fig_window_surface.pdf`
- `fig_within_paper_dispersion.pdf`

### Overleaf tables (`overleaf/tex/v8_tables/`)
- `tab_summary_statistics.tex`
- `tab_i4r_claim_results_verified.tex`
- `tab_i4r_discrepancies_top10_verified.tex`
- `tab_mixture_params_abs_t.tex`
- `tab_mixture_params_z.tex`
- `tab_variance_decomposition.tex`
- `tab_dependence_summary.tex`
- `tab_counterfactual_sensitivity.tex`
- `tab_inference_audit_i4r.tex`

---

## 3. Appendix G structure

Matches `overleaf/tex/v8_sections/appendixG_empirics.tex`.

### G.1 Reproducibility and artifacts
- Pipeline entrypoint, 4 stages, artifact paths

### G.2 Samples and empirical objects
- Sample A/B definitions, units, summary statistics table

### G.3 Automated workflow and verification
- Replication agent protocol, verification agent, verified core definition

### G.4 Inference harmonization
- Harmonized inference object, |Z| main index, Z_p robustness, sign orientation, winsorization

### G.5 Validation on Sample A
- Claim-by-claim table, agreement diagnostics, discrepancy taxonomy, filter sensitivity

### G.6 Mixture estimation
- Baseline truncated-Gaussian fit, K-sensitivity, gamma robustness, additional robustness
- Goodness-of-fit (PP/QQ), posterior assignment, bootstrap CIs, bootstrap LRT, LOO-CV, subsample stability, variance decomposition

### G.7 Dependence estimation
- Distance-based decay, AR(1) robustness, 4 alternative models
- Dependence heterogeneity, sign consistency, funnel plot

### G.8 Counterfactual implementation
- Window selection, effective sample size, qualification, regime comparison, alternative rules, sensitivity grid
- Monte Carlo validation, n_eff visualization, window surface, within-paper dispersion

### G.9 Additional audits
- Inference audit (clustering/robust SE conventions)

### G.10 Catalog
- Complete catalog table of all figures and tables
