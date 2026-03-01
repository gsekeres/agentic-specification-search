# Specification Surface: 112815-V1

## Paper: The Economics of Online Postsecondary Education: MOOCs, Nonselective Education, and Highly Selective Education (Hoxby, 2014)

## Important Note: Primarily Descriptive Paper

This paper is a descriptive characterization of online postsecondary education markets. It does NOT make causal claims. The analysis consists almost entirely of:
- Tabulations (tab) of survey responses by institution type (NSPE vs HSPE)
- Summary statistics (sum) of key variables
- Two regressions (one for NSPE, one for HSPE) that exist solely for F-tests of joint significance

The paper explicitly states "No causal interpretation" for its regressions. There is no focal coefficient, no standard error of interest, and no hypothesis test of a specific treatment effect.

## Baseline Groups

### G1: Joint significance of certificates and institutions on earnings
- **Outcome**: incres09 (2009 income/earnings)
- **Treatment**: Certificate indicators (cert1*, cert2*) and institution fixed effects (instid1-instid1279)
- **Estimand**: Joint F-test -- NOT a focal coefficient
- **Population**: BPS 2004/2009 students at NSPE and HSPE institutions
- **Baseline spec**: reg incres09 cert1* cert2* instid1-instid1279; testparm cert*; testparm instid*

### Additional baselines
- HSPE earnings regression: reg incres09 instid1-instid1279 if barrons08==1 & medSAT>=90

## Core Universe (minimal)

The specification search is of very limited value for this paper because:
1. No causal claim is made
2. No focal coefficient is reported
3. The regressions exist solely for F-tests
4. The main data (BPS restricted-use) is not provided in the package

### Available axes (for completeness)
- Split by institution type (NSPE only, HSPE only)
- Drop institution fixed effects (test certificates alone)
- Drop certificate variables (test institution effects alone)

## Inference Plan
- **Canonical**: Default Stata SEs (HC1)
- No meaningful variants given the descriptive nature

## Constraints
- Data access: BPS 2004/2009 and Barrons restricted-use data require NCES license
- Only ipeds_asc_variables.dta and bps200409_course_codes_titles_definitions.dta are provided
- The do-file cannot be run without restricted-access data
- The paper's main contribution is descriptive statistics, not regression estimates

## Budget
- Max core specs: 15 (minimal, given descriptive nature)
- Seed: 112815

## What is excluded and why
- ALL tabulations and summary statistics: these are descriptive, not regression-based
- Faculty data analysis (NSOPF 2004): separate dataset, descriptive only
- Course-level analysis (transcript data): descriptive only
- The entire paper is essentially excluded from meaningful specification search because it makes no causal claims
