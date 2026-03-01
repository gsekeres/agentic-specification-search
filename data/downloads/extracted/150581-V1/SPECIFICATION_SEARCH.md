# Specification Search: 150581-V1

## Paper
- **Title**: Wage Cyclicality and Labor Market Sorting
- **Authors**: Lise & Postel-Vinay
- **Journal**: American Economic Review (2020)

## Surface Summary
- **Baseline group**: G1 (Table 2 Col 4)
- **Design**: Panel fixed effects (reghdfe within estimator)
- **Outcome**: lhrp2 (log hourly wage)
- **Treatment**: unempl (aggregate unemployment rate)
- **Key interactions**: unempl x dummy1 (EE), unempl x dummy2 (UE), mismatch x unempl x transitions
- **FE**: ID + industry#year + occupation_agg#year
- **Cluster**: ID (individual)
- **Sample**: HOURSM >= 75 & age >= 20 (male workers only)
- **Budget**: max 70 core specs, 10 control subsets
- **Seed**: 150581

## Execution Summary
- **Planned specs**: 40 estimate rows + 2 inference rows
- **Successful**: 0 estimate + 0 inference
- **Failed**: 40 estimate + 2 inference

### DATA CONSTRUCTION FAILURE

All specifications failed because the analysis dataset could not be constructed
from the raw NLSY79 files provided in the replication package.

**Root cause**: The replication package provides raw NLSY79 data in Stata `.dct`
(dictionary + inline data) format. The data construction pipeline
(`code/data.do` -> `1_ind_info.do`, `2_job_info.do`, `3_monthly_panel.do`,
`4_construct_data_analysis.do`) requires:

1. **Weekly labor status conversion**: Converting weekly employment status data
   (from biennial NLSY surveys) to a monthly panel. This involves wave-specific
   variable renaming (`rename_weekly_lstatus.do`) and complex reshaping.

2. **Job transition identification**: Sequential identification of employment-to-employment
   (EE) and unemployment-to-employment (UE) transitions with lag operations.

3. **Tenure construction**: Building job tenure, occupation tenure, and labor
   market experience from the monthly panel.

4. **Skill mismatch computation**: Computing occupation-specific skill requirements
   from ONET data, individual ability scores from ASVAB tests (via PCA by age
   cohort), and the mismatch measure as the weighted absolute difference.

5. **Wage variable**: Log hourly wage with Winsorization (Guvenen et al. 2018).

This pipeline comprises 800+ lines of sequential Stata code with complex
state-dependent operations that cannot be reliably replicated without Stata.

**The replication package does NOT include a pre-built `data_analysis.dta` file.**

### Spec breakdown (all planned, all failed)
| Category | Count |
|----------|-------|
| baseline | 1 |
| baseline (additional) | 4 |
| rc/controls/loo | 5 |
| rc/controls/sets | 5 |
| rc/controls/progression | 4 |
| rc/controls/subset | 10 |
| rc/sample/restriction | 2 |
| rc/sample/outliers | 2 |
| rc/fe/drop | 2 |
| rc/fe/swap | 1 |
| rc/data/* | 4 |
| **Total estimate rows** | **40** |
| infer/se/hc | 1 |
| infer/se/cluster | 1 |
| **Total inference rows** | **2** |

## Software
- Python 3.12.7
- pyfixest 0.40.1
- pandas 2.2.3
- numpy 2.1.3
