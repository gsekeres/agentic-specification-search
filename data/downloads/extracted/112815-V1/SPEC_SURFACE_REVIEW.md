# Specification Surface Review: 112815-V1

## Summary of Baseline Groups
- **G1**: Joint significance of certificates and institutions on earnings
  - The "claim object" is weak: the paper explicitly states "No causal interpretation" for its regressions
  - The regressions are ancillary to the paper's main contribution (descriptive characterization of NSPE vs HSPE markets)
  - No focal coefficient is reported -- only F-tests of joint significance

## Changes Made
1. No changes to baseline group definition, but added a prominent note that this paper is primarily descriptive.
2. Reduced budget to 15 specs (minimal) given the limited scope for meaningful specification search.
3. Confirmed that the main analysis data (BPS 2004/2009 restricted-use) is NOT provided in the package.

## Key Constraints and Linkage Rules
- No causal claim: the paper characterizes market segments through descriptive statistics
- Data access: requires NCES restricted-use data license for BPS, Barrons, and NSOPF
- Only supplementary public data files (IPEDS/ASC variables, course codes) are provided
- The regression includes ~1279 institution fixed effects as regressors (not absorbed FE), making it a saturated model

## Budget/Sampling Assessment
- 15-spec budget is appropriate given the descriptive nature
- No control subset sampling needed (the only "controls" are institution FE)
- The F-tests are the paper's only statistical claims, and they test joint significance of hundreds of indicators

## What's Missing
- This paper fundamentally lacks the structure needed for meaningful specification search
- There is no treatment-outcome relationship with a focal coefficient to vary
- The descriptive statistics (tabulations, means) cannot be expressed as regression specifications

## Data Access Constraint
**BLOCKING**: The main analysis dataset (BPS 2004/2009 restricted-use) is not provided. The do-file requires four separate NCES restricted-use datasets that must be obtained through a data license application. Without these data, no specifications can be run.

## Final Assessment
**NOT RECOMMENDED FOR SPECIFICATION SEARCH.** This is a descriptive paper with no causal claims, no focal coefficients, and restricted-access data not included in the package. The regressions are ancillary F-tests explicitly labeled as non-causal. While a minimal surface is defined for completeness, running a specification search on this paper would not produce meaningful results for the project's goals. If the paper must be processed, the surface is technically approved but the output will have very limited informational value.
