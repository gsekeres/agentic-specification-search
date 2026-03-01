# Specification Surface Review: 112629-V1

## Summary of Baseline Groups
- **G1**: Effect of parental immigration status on children's healthcare utilization
  - Three outcomes (doctor visit, perceived health, usual place of care) all measuring the same claim: children of foreign-born parents have different healthcare utilization
  - Treatment is parental foreign-born status x child citizenship (forcit_pc and forncit_pc)
  - Correctly treated as additional baselines within one group (same claim, different outcome measures)

## Changes Made
1. Consolidated treatment definition variants: the paper uses foreign_p, foreign_p+ncitizen_c, and forcit_pc/forncit_pc across different specification columns. All are rc/data/treatment variants.
2. Verified that LOO candidates group related dummies together (e.g., all poverty dummies dropped together, all education dummies together) rather than individual dummy LOO.
3. Confirmed that race controls are optional (some specs include them, some do not) and should be LOO-droppable.
4. Noted that the data is not provided in the package -- the SAS programs build finaldata.dta from raw IHIS files. This is a data access constraint.

## Key Constraints and Linkage Rules
- No bundled estimator: single-equation logit with survey design
- Survey design (PSU clustering + strata + weights) is the canonical inference
- Logit marginal effects are the paper's reported objects; LPM coefficients are a design alternative
- Year and region FE are included in all main specifications

## Budget/Sampling Assessment
- ~55-60 planned specs is within the 80-spec budget
- 10 random control subset draws with seed=112629 is reproducible
- LOO covers 11 droppable control groups -- sufficient for sensitivity analysis

## What's Missing (minor)
- Could add probit as a design alternative (logit vs probit usually gives nearly identical AMEs)
- Could explore different survey weight variables if available in IHIS
- No sample restriction by year (could restrict to specific IHIS waves)

## Data Access Note
The raw data is not provided in the package. The SAS programs (00.00-00.03) construct finaldata.dta from IHIS microdata. The Stata do-files assume finaldata.dta exists. This means replication requires access to IHIS data and running the SAS pipeline first.

## Final Assessment
**APPROVED TO RUN** (contingent on data availability). The surface is conceptually coherent, with a clear claim object, appropriate control progression, and well-defined treatment variants. The main constraint is data access (IHIS microdata not included in the package).
