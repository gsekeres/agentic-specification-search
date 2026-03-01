# Specification Surface: 125321-V1

## Paper: Can Technology Solve the Principal-Agent Problem? Evidence from China's War on Air Pollution (Greenstone, He, Jia, Liu, AER 2022)

## Baseline Groups

### G1: Effect of automation on reported PM10 pollution

- **Outcome**: PM10 air pollution concentration (daily, station-level; residualized from station FE, month FE, and weather controls)
- **Treatment**: Automation of pollution monitoring (sharp transition from manual to automated reporting at each station)
- **Estimand**: Sharp RD local effect of automation on reported PM10 at the automation date
- **Population**: Chinese air pollution monitoring stations automated in rolling waves during 2013-2014
- **Design**: Sharp regression discontinuity in time, using `rdrobust` with local linear estimation, triangular kernel, and MSE-optimal (CCFT) bandwidth

### Baseline specification

The primary baseline is **Table 1A, Row 1, Column 2**: the residualized PM10 RD estimate with station FE, month FE, and weather controls (wind speed, rain, temperature, relative humidity). The running variable is T = date - auto_date (days from the automation date), with cutoff at 0. The rdrobust command uses p(1) q(2) kernel(tri) vce(cluster code_city).

The paper first shows a raw (unresidualized) RD estimate (Col 1), then the residualized version (Col 2) as the preferred specification. Additional columns show wave-specific and deadline-city subsamples.

### Additional baselines

- **Raw PM10** (Table 1A, Col 1): the unresidualized RD estimate (no FE or weather controls partialled out)
- **Monthly AOD** (Table 1A, Row 2): Aerosol Optical Depth measured via satellite (station-month level), an independent measure of air quality. This provides a falsification check -- AOD should not change with automation since satellites are unaffected by monitoring technology.

### Residualization approach

The paper's approach is distinctive: rather than including covariates directly in rdrobust, the authors first residualize PM10 from station FE, month FE, and weather controls using reghdfe, then estimate the RD on the residuals. This means "controls" variation in the specification surface operates through the residualization step.

## Core Universe

### Design axes (RD-specific)
- **Bandwidth**: half baseline, double baseline (paper does not report these but they are standard)
- **Polynomial order**: local linear (p=1, baseline), local quadratic (p=2)
- **Kernel**: triangular (baseline), uniform, Epanechnikov (Table B3 reports all three)
- **Procedure**: conventional (no bias correction) vs robust bias-corrected (CCFT-style)

### Controls axes
- **None**: raw PM10, no residualization
- **Weather only**: residualize from weather controls but no FE
- **Weather + FE**: station FE + month FE + weather (baseline)

### Sample axes
- **Wave 1 only**: stations automated in wave 1 (auto_date == first wave)
- **Wave 2 only**: stations automated in wave 2
- **Deadline only**: stations with auto_date == 19359 or auto_date == 19724 (deadline cities)
- **76 cities**: restrict to the original 76 pilot cities (Table B4)
- **No missing PM10**: exclude stations with high pre-period missing rates (Table B8)
- **Trim outcome 1/99 and 5/95**: winsorize PM10 at percentile bounds
- **Donut**: exclude observations within 1 or 3 days of cutoff

### Functional form axes
- **Log PM10**: log transformation of PM10 (paper reports log specification in Table 1B)
- **Level PM10**: untransformed PM10 (baseline)

### Fixed effects axes
- **Station + month FE**: baseline
- **Station + year-month FE**: finer time FE (used in Table 1B event-study columns)

### Data aggregation
- **Monthly**: collapse to station-month level (as in AOD analysis); running variable becomes n_month

## Inference Plan

- **Canonical**: Cluster SEs at city level (code_city), matching paper
- **Variant 1**: Cluster at station level (pm10_n) -- finer clustering
- **Variant 2**: HC robust only (no clustering)

## Constraints

- **Control-count envelope**: [0, 4] weather controls + FE variation through residualization
- **No linkage constraints**: single-equation RD
- **Bandwidth**: MSE-optimal by default; manual alternatives are fractions/multiples
- **Residualization is implicit**: control variation operates through the first-stage reghdfe, not through rdrobust covariates directly

## Budget

- Max core specs: 60
- Max control subset specs: 0 (controls pool is small; full enumeration feasible)
- Seed: 125321

## What is excluded and why

- **Table 1B event-study DiD estimates**: different design family (difference-in-differences with staggered adoption). These exploit deadline vs non-deadline timing in a DiD framework with leads and lags. Would require a separate baseline group with design_code="event_study" or "difference_in_differences".
- **Table 2 (online search for masks/filters)**: different outcome concept. The paper uses this as behavioral evidence that citizens responded to the information change, not as a direct pollution measurement claim. Would be explore/outcome.
- **Table 2B (DiD search for deadline cities)**: combines different outcome and different design.
- **Table B6 (PM10 variability/SD)**: different estimand (standard deviation of PM10, not level). This is a separate claim about data manipulation patterns.
- **Table B7 (SO2 and NO2 placebo outcomes)**: falsification tests, not the main claim. Would be diag/placebo.
- **Table B9 (RD density tests at air quality thresholds)**: diagnostics about bunching at regulatory thresholds, not an estimate of automation effect.
- **Table B10 (AOD-PM10 correlation)**: diagnostic about data quality, not an RD estimate.
- **Table D1 (search-sales association)**: different dataset and different question about measurement validity.

## Diagnostics plan

- **Density test**: rddensity at the automation date (standard RD validity check)
- **Weather continuity**: RD estimates on weather variables (Table B2) as placebo outcomes to verify no discontinuity in weather at automation date
