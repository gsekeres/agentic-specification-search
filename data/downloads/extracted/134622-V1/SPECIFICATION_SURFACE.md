# Specification Surface: 134622-V1

## Paper: Immigration and Entrepreneurship in the United States (Azoulay, Jones, Kim, & Miranda)

## Data Availability Warning

**CRITICAL**: This paper relies on confidential administrative Census data (W-2 income records, Longitudinal Business Database) accessible only through Census Research Data Centers. The replication package contains only:
- Aggregated firm size distribution data (disclosure-approved)
- Fortune 500 hand-collected data (public)
- Survey of Business Owners aggregated data (disclosure-approved)

The individual-level microdata required for Table B1 regression analysis (the only regression-based output) is NOT available in the replication package. This specification surface documents the intended analysis but CANNOT BE EXECUTED without Census RDC access.

## Baseline Groups

### G1: Wage gap between immigrant- and native-founded startups
- **Outcome**: ln_real_wages (log real wages of startup employees)
- **Treatment**: ifirm_defn1 (indicator for immigrant-founded startup)
- **Estimand**: Conditional wage differential (descriptive, not causal)
- **Population**: Employees at startups (Census W-2, 2005-2010)
- **Baseline spec**: Table B1 Col 6 -- reghdfe ln_real_wages ifirm_defn1 ln_firm_size is_male is_foreign_born, absorb(age year_t5 county naics4) vce(robust)

## Core Universe (contingent on data access)

### Controls progression (Table B1 columns)
- Bivariate (no controls, no FE)
- Add year FE only
- Add county FE
- Add demographics (gender, foreign-born status)
- Add industry FE (naics4)
- Full specification with firm size

### LOO controls
- Drop ln_firm_size, is_male, is_foreign_born (3 specs)

### FE axes
- Drop industry FE
- Drop county FE
- Drop age FE

### Data construction
- Alternative immigrant definition (ifirm_defn2: majority-immigrant founded)

## Inference Plan
- **Canonical**: HC1 robust SEs
- No clustering variants documented in the paper

## Constraints
- Control-count envelope: [0, 3]
- Paper is primarily descriptive (firm size distributions, patent rates)
- Only one regression table (Appendix Table B1) uses individual-level data
- No causal identification claim -- purely conditional correlations

## Budget
- Max core specs: 50 (contingent on data availability)
- Seed: 134622

## What is excluded and why
- Figure 1 (firm size distributions): descriptive plotting, no regression
- Figure 3 (Fortune 500 immigrant share over time): descriptive, no regression
- Figure 4 (patent rates by firm size): descriptive, no regression
- SBO analysis: different dataset, descriptive tabulations
- Table B2 (power law slopes): descriptive, not a standard regression claim
- ALL of the above use only the aggregated data available in the replication package

## Feasibility Assessment
**NOT FEASIBLE** without Census RDC access. The surface is documented for completeness but cannot be executed with the available data.
