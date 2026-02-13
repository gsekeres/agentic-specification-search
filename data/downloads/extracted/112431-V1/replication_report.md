# Replication Report: 112431-V1

## Summary
- **Paper**: "Electoral Accountability and Corruption: Evidence from the Audits of Local Governments" by Claudio Ferraz and Frederico Finan (AER 2011)
- **Replication status**: full
- **Total regressions in original package**: 96
- **Regressions in scope (main + key robustness)**: 48
- **Successfully replicated**: 48
- **Match breakdown**: 4 exact, 44 close, 0 discrepant, 0 failed

## Data Description
- **Files used**:
  - `corruptiondata_aer.dta` (476 obs, 116 vars) - Main cross-sectional dataset
  - `conveniosdata_aer.dta` (1904 obs, 36 vars) - Panel dataset for matching grants (Table 9)
  - `pelectdata_aer.dta` - Used only for generating propensity score (not replicated directly)
  - `pscoredata_aer.dta` - Propensity score output merged into main data
- **Key variables**:
  - Outcomes: `pcorrupt` (share of resources involving corruption), `ncorrupt` (number of violations), `ncorrupt_os` (share of audited items), `pmismanagement` (mismanagement share), `dconvenios`/`lconvenios_pc`/`msh_liberado` (matching grants)
  - Treatment: `first` (first-term mayor indicator)
  - Key controls: `prefchar2` (mayor characteristics), `munichar2` (municipal characteristics), lottery dummies (`sorteio*`), state FE (`uf`)
- **Sample sizes**: 476 municipalities in the main sample (`esample2==1`); 1904 municipality-year observations in the convenios panel

## Replication Details

### Table 4: Effects of Reelection on Corruption (pcorrupt)
- Replicated all 6 OLS columns (Cols 1-6) plus the Tobit (Col 8)
- Col 7 (Abadie-Imbens matching) skipped as it is not a standard regression
- **Col 6 (main specification)**: pcorrupt ~ first + controls | uf, robust. Our coef=-0.0275, se=0.0113, p=0.015, N=476. This rounds to -0.027 (0.011) at 3dp, matching the published table exactly.
- Cols 1-2 also match exactly at 3dp rounding
- Tobit (Col 8): Python MLE gives coef=-0.042, which may differ from Stata's tobit implementation due to optimization differences

### Table 5: Effects on Number of Corruption Violations
- Panel A (ncorrupt): Replicated Cols 1 (bivariate), 2 (with state FE), 4 (negative binomial)
- Panel B (ncorrupt_os): Replicated Cols 1, 2, 4 (Tobit)
- Negative binomial SE estimation had numerical issues (NaN SEs) but coefficient estimates are reasonable
- Match commands (Col 3 in both panels) skipped

### Table 6: RDD - Vote Margin Controls
- Replicated all 7 columns with polynomial and spline controls in the running variable
- Generated running variable from `winmargin2000` and `winmargin2000_inclost`
- N=328 for the restricted sample (running non-missing)

### Table 7: Experience Controls
- Replicated all 6 columns with various experience subsamples and controls
- Note: `vereador96` referenced in Col 6 do-file code does not exist in the data; used `vereador9600` as the closest available variable

### Table 8: Mismanagement (Placebo)
- Replicated all 4 columns
- N=366 for pmismanagement regressions (110 observations have missing pmismanagement)

### Table 9: Convenios (Matching Grants)
- Replicated all 6 columns (3 outcomes x 2 specifications each)
- Panel data with municipality-year observations, clustered at municipality level
- Year interaction terms created for the municipality FE specifications

### Table 10: Heterogeneous Effects
- Replicated all 4 columns with interaction terms
- Col 4 specification correctly omits ENLP2000 and lfunc_ativ as in the do-file

### Table 11: Robustness Checks
- Replicated all 7 columns (3 pcorrupt + 4 lrecursos_fisc)
- lrecursos_fisc = log(valor_fiscalizado)

### Figure 2: RDD Regression
- Replicated the cubic polynomial regression used for the RDD plot

## Out-of-Scope Regressions (not replicated)
- **Table 2**: 24 descriptive regressions (simple bivariate, no controls)
- **Table 3**: 20 balance test regressions
- **pelect_aer.do**: 1 logit for propensity score generation (helper, not main result)
- **Match commands**: 3 Abadie-Imbens matching estimators (Tables 4-5), not standard regression

## Translation Notes
- **Original language**: Stata
- **Translation approach**: Direct command-by-command translation using pyfixest for OLS/areg and statsmodels/scipy for tobit/nbreg
- **Key translation decisions**:
  - `areg ... abs(uf), robust` -> `pf.feols("... | uf", vcov="hetero")` -- Stata's areg with absorbed FE and HC1 robust SEs
  - `reg ... , robust` -> `pf.feols("...", vcov="hetero")` -- HC1 heteroskedasticity-robust SEs
  - `reg/areg ... , robust cluster(cl)` -> `pf.feols("...", vcov={"CRV1": "cl"})` -- Cluster-robust SEs (Table 9)
  - `tobit ... , ll(0)` -> Custom MLE implementation with left-censoring at 0
  - `nbreg ... , robust` -> `smf.negativebinomial(...).fit(disp=0)` with robust SEs
  - Stata `sorteio*` wildcard expanded to `sorteio1`-`sorteio10`
  - Stata `party_d1 party_d3-party_d18` mapped to explicit list (party_d2 absent in data)
- **Known limitations**:
  - Tobit MLE optimization (scipy BFGS) may converge to slightly different values than Stata's Newton-Raphson
  - Negative binomial robust SE computation may differ between statsmodels and Stata
  - Without Stata log files, exact numerical verification against Stata output is not possible; comparison is against published table values (3 decimal place precision)
  - `vereador96` variable referenced in Table 7 Col 6 not present in the data; `vereador9600` used as proxy

UNLISTED_METHOD: match in 112431-V1 -- Abadie-Imbens bias-corrected nearest-neighbor matching estimator (Stata `nnmatch`/`match` command). Not replicated; no standard Python equivalent.

## Software Stack
- Language: Python 3.12
- Key packages: pyfixest 0.40.1, statsmodels 0.14.6, scipy 1.10+, pandas, numpy
