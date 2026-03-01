# Specification Surface Review: 134622-V1

## Summary of Baseline Groups
- **G1**: Conditional wage gap between immigrant- and native-founded startups
  - Descriptive claim (not causal): wage differential conditional on observables
  - Baseline from Appendix Table B1 Col 6 with full controls and FE
  - Only regression output in the paper; rest is descriptive plotting

## Changes Made
1. Added explicit data_availability_warning to the surface JSON -- the microdata is confidential Census data not included in the package.
2. Documented the full specification progression from Table B1 (6 columns with progressively more controls/FE).
3. Flagged this paper as NOT FEASIBLE for execution without Census RDC access.

## Key Constraints and Linkage Rules
- Single-equation OLS, no bundled estimator
- No causal identification claim -- the paper is explicit that this is descriptive
- Controls serve a conditioning purpose (describing the wage gap net of observables), not causal identification

## Budget/Sampling Assessment
- 50-spec budget is reasonable for the intended surface
- However, the budget is moot since the data is not available

## What's Missing
- The entire paper's primary contribution is in the descriptive analysis of firm size distributions, which involves no regression analysis and cannot be meaningfully subjected to specification search
- The one regression table (B1) is an appendix table, not the paper's main claim

## Blocking Issues
1. **DATA NOT AVAILABLE**: Individual-level Census W-2 data is confidential and not included in the replication package
2. **PRIMARILY DESCRIPTIVE**: The paper's main claims are about firm size distribution patterns, not regression coefficients

## Final Assessment
**NOT APPROVED TO RUN** due to data unavailability. The surface is documented for reference but cannot be executed. If this paper must be included in the sample, Census RDC access would be required.
