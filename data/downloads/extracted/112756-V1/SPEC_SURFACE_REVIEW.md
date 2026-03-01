# Specification Surface Review: 112756-V1

## Summary of Baseline Groups
- **G1**: Effect of kelurahan (appointed) vs desa (elected) village heads on Golkar vote share
  - Well-defined claim object: within-district variation in village administrative type and 1999 election outcomes
  - Baseline is Table 2 Col 5 OLS with full controls and district FE
  - Probit variant (Table 2 Col 9) correctly treated as additional baseline
  - Rich control progression directly visible in Table 2 columns

## Changes Made
1. Grouped polynomial terms (perc_ruralHH_1996 through _4) as a single LOO unit -- they should be dropped together.
2. Added probit as a functional form RC variant rather than a separate baseline group (both estimate the same claim object, same controls).
3. Confirmed district (kab) clustering is appropriate -- kelurahan/desa classification varies within districts.
4. Added sub-district (kecamatan) FE as an FE variant -- this is a much finer level of comparison and provides a strong robustness check.

## Key Constraints and Linkage Rules
- No bundled estimator: single-equation OLS with absorbed district FE
- District FE (kab) is the primary identification structure -- dropping it fundamentally changes the comparison
- Cluster at district level matches the paper's choice and the level of FE variation
- lpopulation_1996 appears to have polynomial terms (check data) -- should be grouped

## Budget/Sampling Assessment
- ~60-70 planned specs is within the 100-spec budget
- 15 random control subset draws with seed=112756 is reproducible
- LOO covers 20 individual controls (or control groups) -- extensive sensitivity analysis
- Control progression mirrors Table 2 structure directly -- very informative

## What's Missing (minor)
- PSM estimator (Table 2 Panel C) excluded due to complexity -- would require psmatch2 equivalent
- Mechanisms analysis (Table 3) correctly excluded -- different outcome variables
- Could add Golkar continuous vote share outcome if available (instead of binary indicator)
- Balance tests between kelurahan and desa on pre-treatment characteristics could be a diagnostic

## Final Assessment
**APPROVED TO RUN.** This is an excellent candidate for specification search. The paper has a clear claim object, rich control progression visible in Table 2, large sample (~43K villages), and substantial control variation (23 controls across 3 groups). The within-district identification is well-suited to LOO and control progression analysis.
