# Specification Surface Review: 163241-V1

## Summary of Baseline Groups
- **G1**: Effect of pay transparency on gender wage gap (DiD with staggered adoption)
  - Well-defined claim object: female#treated interaction measures how transparency reduces the gender gap
  - Baseline matches Table 4 Col 2 exactly (individual FE, prov-year-sex FE, clustered at institution)
  - Two peer group definitions (Inst-Dept, Inst-Dept-Rank) treated as alternative baselines within the same claim
  - The dual treatment definition (provincial law + peer revelation) is correctly documented

## Changes Made
1. No changes to baseline group definition -- well-specified.
2. Verified that the two peer group definitions produce substantively different treatment assignments and coefficients but measure the same underlying claim.
3. Confirmed that the cross-sectional specification (Table 4 Col 1) is treated as a design/FE variant rather than a separate baseline group -- appropriate since the claim object is the same.
4. Added data availability note: main analysis data is confidential Statistics Canada data, but log files provide exact regression output.

## Key Constraints and Linkage Rules
- No bundled estimator: single-equation TWFE regression
- Individual FE (id3) absorbs all time-invariant heterogeneity, leaving very few time-varying controls
- The prov_year_sex FE is a province-year-gender fixed effect that absorbs province-level trends by gender
- Treatment is constructed at the intersection of province policy and individual peer group composition
- Staggered adoption: Ontario (1996), Manitoba (1997), BC (2001), PEI (2005), NS (2008), NL (2009)

## Budget/Sampling Assessment
- ~30-40 planned specs is within the 80-spec budget
- No combinatorial control axis (only 1-3 controls)
- The main specification variation comes from: peer group definition, FE structure, sample restrictions, and rank subgroups
- Province-level clustering (variant) has very few clusters (~10) -- results should be interpreted cautiously

## What's Missing (minor)
- No heterogeneity-robust DiD estimator (Callaway-Sant'Anna, Sun-Abraham) -- would be valuable given staggered adoption. Could add as design variant but implementation complexity is high.
- No wild cluster bootstrap for the province-level clustering variant (few clusters problem)
- No Bacon decomposition to assess TWFE bias from staggered timing

## Potential Concerns
1. **Staggered adoption**: TWFE may be biased with heterogeneous treatment effects across cohorts. The paper does not address this explicitly, and modern DiD estimators should ideally be considered.
2. **Few clusters at province level**: Only ~10 provinces, so province-level clustering may understate standard errors.
3. **Data availability**: Main analysis data is confidential. However, the surface is constructable from the log files and do-files.

## Final Assessment
**APPROVED TO RUN** (conditional on data access). The surface correctly identifies the focal coefficient (female#treated interaction), documents the staggered adoption structure, and covers the paper's revealed robustness space (Table 5 variants, Table 6 rank subgroups). The main limitation is data availability -- the analysis data is confidential Statistics Canada data.
