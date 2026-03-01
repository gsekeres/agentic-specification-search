# Specification Search: 112444-V1

## Paper
Reinhart, Carmen M. and Kenneth S. Rogoff (2011). "From Financial Crash to Debt Crisis."
*American Economic Review*, 101(5), 1676-1706.

## Surface Summary
- **Baseline groups**: 2
  - G1: Banking crisis contagion (focal: center variable)
  - G2: External debt crisis contagion (focal: debt_move variable)
- **Design**: Pooled OLS with HC2 robust SE (TSP HCTYPE=2)
- **Budgets**: G1 max 60, G2 max 40
- **Seed**: 112444

## Data Reconstruction
The original TSP code loads panel data from external text files (banking.txt, ext_debt.txt,
development.txt, region.txt, public_gdp.txt) that are not included in the replication package.
The panel was reconstructed from:
- **Varieties_Part_I-IV.xls**: Banking crisis (col 7) and external debt crisis (col 6) indicators
  for 70 countries, 1800-2010
- **Debt_to_GDP_Part_I-IV.xls**: Total public debt/GDP ratios

Derived variables follow TSP code:
- `bank_move`: 3-year lagged MA of banking crisis: (bank(t-1)+bank(t-2)+bank(t-3))/3
- `debt_move`: 3-year lagged MA of external debt crisis: (debt(t-1)+debt(t-2)+debt(t-3))/3
- `center`: UK/US financial center contagion: (UK_bank(t)+UK_bank(t-1)+UK_bank(t-2)+US_bank(t)+US_bank(t-1)+US_bank(t-2))/6
- `public`: 2-year change in public debt/GDP ratio
- `develop1`: Emerging market dummy (development==1)
- `develop2`: Advanced economy dummy (development==2)

## Estimation
- **Canonical inference**: HC2 robust standard errors (statsmodels OLS with cov_type='HC2')
- **FE models**: pyfixest with vcov="hetero" (HC1) for FE specifications
- **Logit/probit**: statsmodels with HC2

## Execution Summary

| Metric | Count |
|--------|-------|
| Total specs planned | 60 |
| Specs executed successfully | 60 |
| Specs failed | 0 |
| G1 specs | 35 (35 success) |
| G2 specs | 25 (25 success) |
| Inference variants | 6 |

## RC Axes Executed

### G1: Banking Crisis
- Baseline (Eq5 with public, 1824-2009)
- Controls LOO: drop public, bank_move, debt_move, develop1, develop2
- Sample periods: 1900-2009, 1946-2009
- Sample subsets: advanced only, emerging only, drop UK/US, trim public debt
- Functional form: logit, probit
- Fixed effects: country, year, country+year, region
- Joint variations: period x controls, FE x period, subsample x period, FE x cluster

### G2: External Debt Crisis
- Baseline (Eq6 with public, 1824-2009)
- Controls LOO: drop public, bank_move, center, develop1, develop2
- Sample periods: 1900-2009, 1946-2009
- Sample subsets: advanced only, emerging only
- Functional form: logit, probit
- Fixed effects: country, year, region
- Joint variations: period x controls, FE x period, FE x cluster

## Deviations from Surface
- Data reconstructed from Excel files rather than original TSP text files
- Development classification based on standard R&R grouping; exact classification
  may differ slightly from original (no definitive mapping in the replication package)
- N counts may differ slightly from TSP output due to missing value handling
  in the reconstructed dataset
- The TSP code has 71 country series (b1-b71, d1-d71) but only 70 countries
  are found in the Varieties Excel files. One country may be missing.
- center variable in year-FE models may have reduced variation due to collinearity

## Software
- Python 3.12.7
- pandas, numpy, pyfixest, statsmodels
- Surface hash: sha256:2dc33d264dcb7e4b125e1763708d920f82b59d68063cc6e3cf257244ef8c4f97
