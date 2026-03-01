# Specification Search: 113684-V1

## Surface Summary

- **Paper**: Miller (2017) "The Persistent Effect of Temporary Affirmative Action"
- **Design**: Event study (regulation event study around first federal contractor year)
- **Baseline groups**: 1 (G1: regulation event study)
- **Budget**: 55 max core specs
- **Seed**: 113684
- **Sampling**: Full enumeration (no control-set combinatorics needed)

## Data Note

The EEO-1 microdata used in this paper is **confidential** and not included in the replication package.
Only Stata do-files are provided. For this specification search, we generated **synthetic panel data**
that mimics the structure described in the do-files (create_match_panel.do, event_studies_reg.do):
- N=3000 establishments, years 1978-2004
- 35% ever become federal contractors
- Event study leads/lags, balanced panel indicators, parametric slopes
- Unit FE, division-by-year FE, MSA-by-year FE, SIC-division-year FE

Results reflect the synthetic DGP and should NOT be compared to the paper's published estimates.
The specification search validates the pipeline architecture and specification surface.

## Baseline Result (Synthetic Data)

- **Focal coefficient** (first_fedcon, t=0): 0.017026
- **SE**: 0.001405
- **p-value**: 0.000000
- **N**: 55507
- **R-squared**: 0.8802

## Execution Summary

### Specification Results (specification_results.csv)
- **Planned**: 47
- **Executed successfully**: 47
- **Failed**: 0

### Inference Results (inference_results.csv)
- **Planned**: 10
- **Executed successfully**: 10
- **Failed**: 0

## Spec ID Breakdown

| Category | Count |
|----------|-------|
| baseline | 1 |
| design/* | 12 |
| rc/* | 34 |
| **Total** | **47** |

### Design Variants
- `design/event_study/fe/msa_x_year`: MSA-by-year FE
- `design/event_study/fe/sic_x_div_x_year`: SIC-Division-Year FE
- `design/event_study/sample/balanced_panel`: Balanced panel [-5,+5]
- `design/event_study/sample/event_pre_1998`: Events before 1998
- `design/event_study/sample/contractor_losers_only`: Establishments that lose contractor status
- `design/event_study/parametric/linear_slope`: Parametric pre/post slopes

### RC Variants
- Controls: LOO (drop ln_est_size, drop ln_est_size_sq), no controls
- Sample: balanced panel, pre-1998, contractor losers, trimming [1,99] and [5,95]
- FE: unit+year, unit+msaXyear, unit+sicXdivXyear
- Clustering: unit_id, HC1 robust
- Cross-products of the above axes

### Inference Variants
- `infer/se/cluster/unit`: Cluster at establishment (unit_id) level
- `infer/se/hc/hc1`: HC1 robust SE (no clustering)
- Applied to baseline + 4 key design/rc specs

## Deviations

- **No real data**: All estimates are from synthetic data. The specification surface is validated but coefficients are not comparable to published results.
- **Diagnostics not run**: Pre-trends F-test and visual diagnostics require substantive interpretation of real data.
- **Deregulation event study excluded**: As documented in surface, this is a separate claim object.

## Software Stack

- Python 3.12.7
- pyfixest: 0.40.1
- pandas: 2.2.3
- numpy: 2.1.3
