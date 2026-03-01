# Specification Search: 113500-V1

## Paper
Babcock, Recalde, Vesterlund (2017), "Gender Differences in the Allocation of Low-Promotability Tasks: The Role of Backlash," AER P&P.

## Surface Summary
- **Design**: randomized_experiment (lab experiment)
- **Baseline groups**: 3
  - G1: Gender gap in solicitation response, no-penalty condition (Table 1, Col 1)
  - G2: Gender gap in solicitation response, backlash/penalty condition (Table 1, Col 2)
  - G3: Cross-treatment comparison via triple interaction (Table 1, Col 3)
- **Budget**: 55 specs per group (nominal)
- **Seed**: 113500
- **Canonical inference**: Cluster at session_id

## Execution Summary
- **Total specification rows**: 60
- **Successful**: 60
- **Failed**: 0
- **Inference variant rows**: 6

### Per-group breakdown
| Group | Planned | Executed | Successful | Failed |
|-------|---------|----------|------------|--------|
| G1 | 20 | 20 | 20 | 0 |
| G2 | 20 | 20 | 20 | 0 |
| G3 | 20 | 20 | 20 | 0 |

### Spec types executed per group
- 1 baseline (probit marginal effects, cluster session_id)
- 2 design variants (diff-in-means LPM, LPM with covariates)
- 9 LOO control drops (probit, drop each of 9 controls)
- 3 control sets (no controls, demographics only, preferences only)
- 2 functional form variants (LPM, logit)
- 3 period sample splits (first half, second half, first period only)
- **Total: 20 per group x 3 groups = 60**

### Inference variants (on LPM with_covariates baseline)
- 2 per group x 3 groups = 6 total
  - infer/se/cluster/subject (cluster at unique_subjectid)
  - infer/se/hc/hc1 (robust HC1, no clustering)
- Wild cluster bootstrap (infer/bootstrap/wild_cluster/session) SKIPPED: wildboottest package not available in environment

## Data Preparation Notes
- `treatment` is numeric 1/2 in .dta (1=Control, 2=Backlash)
- `female` is numeric 0/1 in .dta (0=Male, 1=Female)
- `student` is numeric 1-4 in .dta (1=Freshman, 2=Sophomore, 3=Junior, 4=Senior), used as ordinal numeric in regressions
- Constructed variables: backlash, femaleXsolicited, backlashXsolicited, femaleXbacklash, femaleXbacklashXsol
- Data read with convert_categoricals=False to preserve original Stata numeric encoding

## Clustering Bug in Original Code
The Stata do-file defines `local clust_var unique_subjectid` but then uses `cluster(\`clus_var\')` (misspelled macro name), so the original published probit estimates are effectively **unclustered**. Our baseline uses cluster(session_id) as specified by the surface (correct design-based choice). The paper's primary inference method is cgmwildboot (wild cluster bootstrap), not the probit clustered SEs.

## Probit Interaction Effects
For probit models, the marginal effect of the interaction term (femaleXsolicited) reported by statsmodels `get_margeff(at='overall')` is the average marginal effect, which differs from the Stata `inteff`-corrected interaction effect. The LPM variant provides a directly interpretable interaction coefficient. See Norton, Wang, Ai (2004) for the distinction.

## Software Stack
- Python 3.12.7
- pyfixest (LPM estimation with clustered SEs)
- statsmodels (probit/logit marginal effects with clustered SEs)
- pandas, numpy
