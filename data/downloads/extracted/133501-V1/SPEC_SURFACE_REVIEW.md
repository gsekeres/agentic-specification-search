# Specification Surface Review: 133501-V1

## Summary of Baseline Groups
- **G1**: Discontinuous change in mortality at minimum driving age (sharp RD)
  - Well-defined claim object: RD estimate of mortality jump at MDA cutoff
  - Running variable (age in months relative to MDA) is clean and cannot be manipulated
  - Three baseline outcomes: MVA mortality (primary), all-cause mortality, suicide/accident poisoning
  - Aggregated cell-level data limits control variable options

## Changes Made
1. No changes to baseline group definition -- well-specified for RD.
2. Verified that the design_audit correctly identifies bandwidth, kernel, polynomial order, and bias correction parameters.
3. Confirmed that the paper's Table A.10 (alternative bandwidths), Table A.11 (alternative polynomials), and Table A.12 (OLS) are all represented as design variants.
4. Added sex-specific subsamples (male/female) as rc/sample variants since the paper reports these prominently.

## Key Constraints and Linkage Rules
- No bundled estimator: single-equation RD
- No meaningful control set to vary (only firstmonth indicator available in aggregated data)
- The design variant space (bandwidth x polynomial x kernel) is the main source of specification variation
- Bias correction choice (conventional vs robust) is a meaningful design variant

## Budget/Sampling Assessment
- ~50-60 planned specs is within the 80-spec budget
- The bandwidth x polynomial x kernel grid is inherently enumerable (no combinatorial explosion)
- Sample restriction variants (male/female, MDA type, time period) add meaningful heterogeneity checks
- No control subset sampling needed (only 1 possible control)

## What's Missing (minor)
- Add Health outcomes (driving behavior, employment, school enrollment) are a different dataset -- correctly excluded from this surface but could be a separate baseline group if desired
- Donut-hole RD (excluding observations right at the cutoff) could be added as rc/sample/donut but the paper does not report this
- No outcome winsorization beyond log transform -- aggregated rates may have outliers in sparse cells

## Final Assessment
**APPROVED TO RUN.** The surface correctly focuses on the RD design variants (bandwidth, polynomial, kernel, inference procedure) as the primary specification search space. The aggregated data structure appropriately limits the control-based search. The budget is feasible and the design audit is complete.
