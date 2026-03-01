# Specification Surface: 120078-V1

## Paper: Can Information Reduce Ethnic Discrimination? Evidence from Airbnb

## Baseline Groups

### G1: Effect of minority host status on listing price, conditional on review accumulation (panel FE)

- **Outcome**: log_price (daily log-price of Airbnb listing)
- **Treatment**: minodummy x rev100 (minority indicator interacted with number of reviews / 100)
- **Estimand**: Within-listing effect of minority status on price as reviews accumulate
- **Population**: Airbnb listings in major US, Canadian, and European cities with at least one review, observed across multiple scraping waves
- **Baseline spec**: Table 5 Col 1 (xtreg log_price ... c.minodummy#c.rev100 + rating interactions + controls, fe i(newid) robust cl(newid), review > 0 & review < 40)
- **Focal parameter**: Coefficient on minodummy x rev100 (how minority price gap changes with information)

### Additional baselines (same claim, different sample windows)
- Table 5 Col 2: review < 60 (baseline__table5_col2)
- Table 5 Col 3: review < 80, with quadratic term (baseline__table5_col3)

## Design and Identification

The paper studies ethnic discrimination on Airbnb using panel data on listings observed across multiple scraping waves. The identification strategy relies on within-listing variation: as reviews accumulate, the informational content about a listing increases, and the ethnic price gap should narrow if discrimination is statistical (information-based) rather than taste-based.

Table 4 presents a structural model estimated via iterated NLS with bootstrapped SEs (a learning model with parameter rho governing the speed of information accumulation). Table 5 presents the reduced-form panel FE version with linear and quadratic interactions between minority status and review count.

We use the Table 5 reduced-form panel FE specification as the baseline because:
1. It is a standard panel FE regression implementable in Python
2. The structural model (Table 4) requires custom NLS iteration that is not standard
3. Table 5 is explicitly presented as a robustness check confirming the same pattern

## Core Universe

### Controls axes
- **LOO**: 13 specs (drop key individual controls from the lesX set one at a time -- focusing on substantively important controls like shared flat indicator, capacity, bedrooms, bathrooms, superhost status, verification indicators, listing description counts, picture counts)
- **Standard sets**: 5 specs (none / minimal size / property characteristics / host characteristics / full)
- **Progression**: 5 specs (build up controls step by step: bivariate, size only, size+amenities, size+amenities+host, full with counts)
- **Subset search**: 15 budgeted random draws (seed=120078)

### Sample axes
- Review upper bound: < 60, < 80, < 100 (Table 5 varies this)
- Review > 0 (drop the zero-review restriction already imposed in baseline)
- Trim log_price at 1st/99th percentile
- Trim log_price at 5th/95th percentile

### Fixed effects axes
- Drop city-wave FE (simplify to listing FE only)
- Swap neighborhood-city FE for city-wave FE

### Functional form axes
- Quadratic in rev100 (Table 5 Col 3 includes a squared term)

## Inference Plan
- **Canonical**: Cluster SEs at listing level (newid), matching paper's cl(newid)
- **Variant 1**: HC1 robust SEs (no clustering)
- **Variant 2**: Cluster SEs at neighborhood-city level (hoodcityID) -- coarser clustering

## Constraints
- Control-count envelope: [0, 52]
- No linkage constraints (single-equation panel FE)
- The interaction terms (minodummy x wave dummies, lastrat* x rev100) are always included as part of the model specification, not as optional controls
- Sample always restricted to Drev100 > 0 (listings with nonzero review change across waves)

## Budget
- Max core specs: 70
- Max control subset specs: 15
- Total planned: ~52
- Seed: 120078

## What is excluded and why
- **Table 4 structural model**: Requires custom iterated NLS with bootstrapped SEs; not a standard panel FE regression. The reduced-form Table 5 captures the same claim.
- **Table 2 (cross-sectional OLS)**: Different design (pooled cross-section with neighborhood/block FE, no listing FE). This is preliminary evidence, not the main claim.
- **Table 3 (cross-sectional by review segments)**: Same as Table 2 but split by review bins. Cross-sectional design, not panel FE.
- **Table 6 (sub-sample analysis)**: Changes target population (Arabic/Muslim only, Black only, US/Canada only, Europe only, entire flat only, shared flat only). These are heterogeneity analyses, not the main claim.
- **Tables 7-10 (mechanisms)**: Different outcomes (ethnic matching, ratings, upgrading, review accumulation). Not the same claim object.
- **Table A7-A8 (appendix)**: Different outcomes (exit probability, non-person pictures). Not the same claim object.
- **Missing variable indicators**: The $missing global includes many missing-value dummies; these are part of the control set and included in the full control set specification.
