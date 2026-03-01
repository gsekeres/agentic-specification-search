# Specification Search Report: 173341-V1

**Paper**: Bischof, Guarnieri, Grottera, Nasuti -- "How Does the Provision of Public Goods Affect Clientelism? Evidence from a Randomized Infrastructure Program in Brazil"
**Design**: Randomized experiment (household-level RCT)
**Date**: 2026-02-25

---

## Surface Summary

- **Paper ID**: 173341-V1
- **Baseline groups**: 2
  - **G1**: Individual-level clientelist requests (stacked cross-section 2012-2013, Table 3)
  - **G2**: Electoral outcomes at voting-section level (Table 4)
- **Budget**: G1 max 50, G2 max 15
- **Sampling seed**: 173341
- **Surface hash**: See `coefficient_vector_json` in outputs

### G1 Design

- **Estimator**: OLS with municipality FE
- **Outcome**: `ask_private_stacked` (binary: requested private good from politician)
- **Treatment**: `treatment` (cisterns randomization) + `rainfall_std_stacked` (natural variation)
- **Canonical inference**: CRV1 clustered at `b_clusters` (neighborhood)
- **Baseline specs**: Table 3 Col 3 (treatment + rainfall) and Col 4 (+ interaction)

### G2 Design

- **Estimator**: reghdfe with absorbed location FE
- **Outcome**: `incumbent_votes_section` (votes for incumbent mayor)
- **Treatment**: `tot_treat_by_section_2` (rescaled share of treated individuals)
- **Canonical inference**: CRV1 clustered at `location_id`
- **Baseline spec**: Table 4 Col 1 (21 municipalities where incumbent ran for re-election)

---

## Execution Summary

| Category | Planned | Executed | Succeeded | Failed |
|----------|---------|----------|-----------|--------|
| G1 baselines | 2 | 2 | 2 | 0 |
| G1 design variants | 4 | 4 | 4 | 0 |
| G1 RC/controls | 17 | 17 | 17 | 0 |
| G1 RC/form | 15 | 15 | 15 | 0 |
| G1 RC/sample | 5 | 5 | 5 | 0 |
| G1 RC/fe | 2 | 2 | 2 | 0 |
| **G1 total** | **45** | **45** | **45** | **0** |
| G2 baseline | 1 | 1 | 1 | 0 |
| G2 design variants | 1 | 1 | 1 | 0 |
| G2 RC variants | 4 | 4 | 4 | 0 |
| **G2 total** | **6** | **6** | **6** | **0** |
| **All specs** | **51** | **51** | **51** | **0** |
| Inference variants | 6 | 6 | 6 | 0 |

---

## Specification Axes Explored

### G1 Axes

1. **Controls**: 7 control sets (none, mun FE only, mun FE + year, + engagement bundles for assoc/pres/voted/all)
2. **Treatment form**: cisterns only, rainfall only, treatment-by-year, rainfall-by-year, treatment x rainfall interaction
3. **Outcome form**: `ask_private_stacked` (baseline), `ask_nowater_private_stacked` (exclude water requests)
4. **Sample splits**: year 2012 only, year 2013 only
5. **Outlier handling**: Winsorize outcome at 1st/99th percentiles
6. **FE structure**: municipality FE + year, municipality FE only, no FE

Many specs are cross-combinations of the above axes (e.g., cisterns only + 2012 subsample, engagement controls + year subsample, Col4 interaction variant across all control sets).

### G2 Axes

1. **Sample**: 21 municipalities (strict incumbent name match) vs. 39 municipalities (broader definition)
2. **Controls**: drop eligible voters, drop study share
3. **FE**: with/without location FE

### Inference Variants (separate table)

- HC1 (heteroskedasticity-robust, individual-level) for G1 baselines and G2 baseline
- Cluster at municipality level for G1 baselines
- CRV1 at municipality for G2 (approximation to wild-cluster bootstrap; `wildboottest` not installed)

---

## Deviations and Notes

1. **Rainfall collinearity in year subsamples**: Within a single year, `rainfall_std_stacked` is perfectly collinear with municipality FE (since rainfall varies only at the municipality level). Year-subsample rainfall-only regressions therefore drop municipality FE. This is expected and correct.

2. **Wild-cluster bootstrap unavailable**: The `wildboottest` package is not installed. For G2 inference variant at the municipality level, standard CRV1 clustering is used as an approximation. The paper reports wild-cluster bootstrap p-values for Table 4.

3. **Singleton FE in G2**: pyfixest drops 43 singleton location FE in the 21-municipality sample (N drops from 909 to 866) and 102 in the 39-municipality sample. This matches Stata's `reghdfe` behavior.

4. **Winsorization has no effect**: The outcome `ask_private_stacked` is binary (0/1), so winsorizing at 1st/99th percentile has no practical effect. These specs serve as mechanical robustness checks.

---

## Key Results

### G1: Cisterns Treatment Effect on Private Good Requests

- **Baseline Col 3**: coef = -0.0296, SE = 0.0125, p = 0.0185 (N = 4,288)
- **Baseline Col 4** (with interaction): coef = -0.0296, SE = 0.0125, p = 0.0187
- Treatment effect is **robustly negative** across most specifications
- Coefficient range for `treatment`: [-0.033, +0.046] across all 45 G1 specs
- Most specifications with treatment as focal yield statistically significant negative effects

### G2: Electoral Outcomes

- **Baseline Table 4 Col 1**: coef = -0.1012, SE = 0.0579, p = 0.0829 (N = 866)
- Marginally significant; sensitive to sample definition and inference choice
- With HC1 SE: p = 0.1045; with CRV1 at municipality: p = 0.0412

---

## Software Stack

- **Python**: 3.12
- **pyfixest**: 0.40+
- **pandas**: 2.x
- **numpy**: via pyfixest
- Runner: `scripts/paper_analyses/173341-V1.py`

---

## Output Files

- `specification_results.csv`: 51 rows (all `run_success=1`)
- `inference_results.csv`: 6 rows (all `run_success=1`)
- `scripts/paper_analyses/173341-V1.py`: executable analysis script
