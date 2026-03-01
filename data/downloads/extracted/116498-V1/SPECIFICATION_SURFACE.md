# Specification Surface: 116498-V1

## Paper: Is There a Link Between Foreclosures and Health? (Currie & Tekin)

## Baseline Groups

### G1: Effect of foreclosure rates on hospital admissions
- **Outcome**: Non-elective hospital admission rate per 100,000 population (nnonelective_rate)
- **Treatment**: Four quarterly lags of foreclosure rate per 100,000 (rate_fore_1 through rate_fore_4)
- **Estimand**: Sum of lag coefficients (cumulative foreclosure effect on health)
- **Population**: Zip codes in AZ, CA, FL, NJ (2005-2009), excluding top 10% seasonal/vacation zips
- **Baseline spec**: Table 3b -- areg nnonelective_rate rate_fore_1-rate_fore_4 i.county#i.t ziptrend* [aweight=gtot], absorb(nzip) vce(cluster county)
- **Key feature**: Treatment is continuous (foreclosure rate), with 4 distributed lags; the focal parameter is the sum of the 4 lag coefficients

### Additional baselines (same claim, different health outcome measures)
- npqi_rate (preventable quality indicators)
- nheart_rate (heart-related admissions)
- nmentalhealth_rate (mental health admissions)
- nrespiratory_rate (respiratory admissions)

## Core Universe

### Sample restrictions (from paper Tables 6 and 8)
- Include vacation zips (reversing the baseline exclusion)
- Early period only (t <= 12, roughly 2005-2007)
- Judicial foreclosure states only (NJ, FL)
- Non-judicial foreclosure states only (AZ, CA)
- Age subgroups: 0-19, 20-49, 50-64, 65+ (from Table 5)
- Minority-heavy zips (>70% minority)
- Majority-white zips (<10% minority)
- Low income zips (bottom tercile)
- High income zips (top tercile)

### Outlier trimming
- Trim outcome at 1st/99th percentiles
- Trim outcome at 5th/95th percentiles

### Fixed effects axes
- Drop county-by-time FE (use time FE only)
- Drop pseudo-zip trends
- Replace county-by-time with state-by-time FE

### Data construction / treatment alternatives
- Use foreclosure starts (lis pendens + notice of default) instead of completed foreclosures (Table 8)
- Use only lag 1 (contemporaneous effect proxy)
- Use lags 1 and 2 only

### Controls axes
- Add housing price index lags (zhpi_1 through zhpi_4) as in Table 4
- Add county-level unemployment rate as in Table 4

### Functional form
- log(1 + rate) transformation of outcome
- asinh transformation of outcome

### Weights
- Unweighted (drop analytic weights)

### Joint specifications (insurance type from Table 5e-g)
- Private insurance admissions
- Public insurance admissions
- Uninsured admissions

## Inference Plan
- **Canonical**: Cluster SEs at county level (matching Stata vce(cluster county))
- **Variant 1**: Cluster at zip code level (finer, more clusters)
- **Variant 2**: HC1 robust only (no clustering)

## Constraints
- No time-varying controls in the baseline; the paper's identification relies on FE structure
- The control-count envelope is effectively [0, 4] (with housing price lags or unemployment as the only added controls)
- Treatment is always the foreclosure rate lags; alternative treatments (starts) change data construction, not the claim object
- Focal parameter is always the sum of lag coefficients

## Budget
- Max core specs: 80
- No control subset sampling needed (few control additions possible)
- Total planned: approximately 35-40 specs
- Seed: 116498

## What is excluded and why
- Table 1 (PSID analysis): entirely different dataset and unit of analysis (individuals, not zip codes)
- Table 4 county-level regressions: different unit of analysis (county aggregation)
- Table 7 acute conditions (heart, respiratory, gastro, injury): exploratory sub-outcomes, not headline claims
- Cancer outcomes: paper notes these as placebo/negative control, not a claim
- Elective admissions: treated as a falsification check by the paper
