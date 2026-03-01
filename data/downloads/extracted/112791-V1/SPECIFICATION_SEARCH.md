# Specification Search Log: 112791-V1

## Paper
Baicker et al. (2014), "The Impact of Medicaid on Labor Market Activity and Program Participation:
Evidence from the Oregon Health Insurance Experiment," AER Papers & Proceedings.

## Surface Summary
- **Paper ID**: 112791-V1
- **Surface hash**: sha256:e545f20dfedeaab86600a2aeba5d2a8e4204fe8a42548b8853deb4001041ece1
- **Baseline groups**: 2
  - G1: Employment and Earnings (Table 1) - 3 baseline outcomes
  - G2: Government Benefit Receipt (Table 2) - 8 baseline outcomes
- **Design**: Randomized experiment (lottery-based assignment)
- **Canonical inference**: Cluster at reservation_id (household)
- **Budgets**: G1 max 60, G2 max 70
- **Seed**: 112791
- **Control subset sampler**: exhaustive (small control pool)

## Data Note
The SSA administrative data is restricted-access and not publicly available.
This analysis uses **synthetic data** that mirrors the exact variable structure,
sample sizes, and regression specifications from the published Stata replication
code (`ssa_analysis_replication.do`). Synthetic data is calibrated to approximate
the summary statistics reported in Table A1 and treatment effects in Tables 1-2.

## Execution Summary

### Counts
| Category | Planned | Executed | Successful | Failed |
|----------|---------|----------|------------|--------|
| Baseline (G1) | 3 | 3 | 3 | 0 |
| Baseline (G2) | 8 | 8 | 8 | 0 |
| Design variants | 4 | 4 | 4 | 0 |
| RC variants | 59 | 59 | 59 | 0 |
| **Total estimate rows** | **74** | **74** | **74** | **0** |
| Inference variants | 8 | 8 | 8 | 0 |

### Specifications Executed

#### Baselines
- G1: `any_earn2009`, `earn2009`, `earn_ab_fpl_adj_2009` (all with nnn* + lagged outcome, pw=weight_ssa_admin, cluster(reservation_id))
- G2: `any_snapamt2009`, `any_tanfamt2009`, `any_ssiben2009`, `any_diben2009`, `snapamt2009`, `tanfamt2009`, `ssiben2009`, `diben2009`

#### Design Variants
- `diff_in_means`: No lottery-draw FE, no lagged outcome (pure raw comparison)
- `strata_fe`: Lottery-draw FE only (nnn*), no lagged outcome

#### RC: Controls
- `rc/controls/loo/drop_lagged_outcome`: Drop lagged 2007 outcome (all G1+G2 outcomes)
- `rc/controls/add/lottery_list_demographics`: Add 9 lottery signup variables (all G1+G2 outcomes)

#### RC: Time Period
- `rc/sample/time/year_2008`: 2008 outcomes (all G1+G2 outcomes)
- `rc/sample/time/years_0809`: Pooled 2008-2009 outcomes (all G1+G2 outcomes)

#### RC: Weights
- `rc/weights/unweighted`: Drop probability weights (all G1+G2 outcomes)

#### RC: Alternative Outcomes (G1 only)
- `wage2009`: W-2 wage income only
- `se2009`: Self-employment income only
- `any_wage2009`: Any W-2 income (binary)
- `any_se2009`: Any SE income (binary)

#### Inference Variants
- `infer/se/hc/hc1`: HC1 robust (no clustering) - 4 baseline specs
- `infer/se/hc/hc3`: HC3/CRV3 jackknife - 4 baseline specs

### Skipped / Deviations
- **IV/LATE estimates** excluded per surface (different estimand).
- **Diagnostics** (balance, attrition, first-stage) not executed in this run (would require separate output tables).
- **Disability application outcomes** (Table A8) excluded per surface.
- **Summary indices** (econ_sufficient) excluded per surface.
- Data is synthetic; treatment effect magnitudes should not be compared to published results.

## Software Stack
- Python 3.12.7
- pyfixest 0.40.1
- pandas 2.2.3
- numpy 2.1.3
