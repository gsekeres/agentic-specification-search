# Specification Surface: 113566-V1

## Paper Overview
- **Title**: The Effect of Grade Retention on High School Completion (Jacob & Lefgren, AEJ Applied)
- **Design**: Fuzzy Regression Discontinuity (2SLS/IV at test score cutoffs)
- **Key claim**: Grade retention in 6th grade increases dropout, while the effect for 8th graders is smaller or insignificant. Uses Chicago Public Schools' promotional gate policy as a natural experiment.

## Baseline Groups

### G1: Effect of Grade Retention on High School Dropout

**Claim object**: The LATE of grade retention (at the promotional test score cutoff) on high school dropout by Fall 2005, for students near the cutoff.

**Baseline specifications**:
- **Table2-Grade6**: Fuzzy RD IV with flexible knots, full covariates, sample [-1.0, +0.5] around cutoff. `ivreg2 dropF2005 (dret2 = gpmarg* gpind4_index_above*) [full controls] if newgrade==6 & samp_m10p5==1, cl(gp)`
- **Table2-Grade8**: Same for 8th graders (newgrade==8)
- **Table2-OlderGrade8**: Same for older 8th graders sent to transition centers (newgrade==9)

The running variable `index` is the normalized distance from the promotional test score cutoff. Treatment `dret2` indicates actual retention, instrumented by spline functions that create discontinuities at the cutoff.

## RC Axes Included

### Design variants (bandwidth, polynomial, instrument set)
- **Bandwidth**: 4 sample windows: [-1.0, +0.5], [-1.5, +1.0], [-2.0, +1.5], [-0.8, +0.3]
- **Polynomial order**: Add quadratic in index, cubic in index, cubic with split above/below cutoff
- **Instrument sets**: Flexible knots (baseline), fixed knots, pass dummy, experiment-specific pass dummies, marginal area only
- **Control richness**: Group dummies only, group + index controls, full experiment-interacted covariates

### Sample restrictions
- **Grade subsamples**: Grade 6 only, grade 8 only, older grade 8 only
- **Cohort subsamples**: 1997, 1998, 1999 separately
- **Failure type**: Failed reading only, failed math only, failed both

## What Is Excluded and Why

- **Table 4 (other outcomes)**: Graduation, private school transfer, move out of CPS -- these are different outcome concepts and belong in `explore/*`, not core.
- **Table 5 (subgroup heterogeneity)**: By cohort, failure type, school quality, race, gender, free lunch -- these are `explore/*` subgroup analyses.
- **Table 6 (intermediate mechanisms)**: Entered high school, starting age, credits earned -- different outcome concepts.
- **Aggregate analysis**: The aggregate-level analysis (collapsed by index*experiment) is an alternative estimator that changes the weighting, not a standard RD variant.
- **OLS estimates**: Table 2 OLS regressions are provided for comparison but are not the paper's preferred RD estimates. They could be included as `design/*` alternatives but are conceptually different (no IV).

## Budgets and Sampling

- **Max core specs**: 70
- **Full enumeration**: All design x sample combinations are discretely enumerable
- **No control subset sampling**: Controls are bundled in experiment-interacted blocks

## Inference Plan

- **Canonical**: Clustered at index*experiment cell level (gp), matching the paper
- **Variants**: HC1 robust (no clustering), classical (no robust)
- The paper explicitly reports non-clustered and robust-only versions as checks
