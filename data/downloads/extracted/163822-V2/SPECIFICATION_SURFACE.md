# Specification Surface: 163822-V2

## Paper: Digital Addiction (Allcott, Gentzkow, and Song, AER 2022)

## Baseline Groups

### G1: Effect of screen-time interventions on phone usage (FITSBY apps)

- **Outcome**: PD_P2_UsageFITSBY (daily minutes on Facebook, Instagram, Twitter, Snapchat, Browser, YouTube in Period 2)
- **Treatment**: B (bonus indicator: financial incentive to reduce use) and L (any limit indicator: commitment device)
- **Estimand**: ITT effect of randomly assigned bonus and limit treatments on FITSBY app usage
- **Population**: Android phone users recruited via Phone Dashboard app (2020, during COVID-19)
- **Baseline spec**: reg PD_P2_UsageFITSBY B L i.Stratifier PD_P1_UsageFITSBY, robust
- **Focal parameters**: Coefficients on B (bonus effect) and L (limit effect)

### Additional baselines (same claim, different time periods and outcome definitions)
- Usage in Period 3 (baseline__usage_p3_fitsby)
- Usage in Period 4 (baseline__usage_p4_fitsby)
- Usage in Period 5 (baseline__usage_p5_fitsby)
- Average usage Periods 2-5 (baseline__usage_p2to5_fitsby)
- Total phone usage in Period 2 (baseline__usage_p2_total)
- Average total usage Periods 2-5 (baseline__usage_p2to5_total)

### G2: Effect of screen-time interventions on well-being outcomes

- **Outcome**: index_well_N (ICW survey well-being index, normalized)
- **Treatment**: B (bonus) and L (limit), in stacked survey waves S3/S4
- **Estimand**: ITT effect of treatments on survey-based well-being measures
- **Population**: Same as G1
- **Baseline spec**: FDR Table regression -- stacked panel with survey-wave interacted controls
- **Focal parameter**: Coefficient on L (limit effect on well-being)

### Additional baselines (same claim, different well-being outcomes)
- AddictionIndex_N (baseline__addiction_index)
- SMSIndex_N (baseline__sms_index)
- PhoneUseChange_N (baseline__phone_use_change)
- LifeBetter_N (baseline__life_better)
- SWBIndex_N (baseline__swb_index)

## Design and Identification

This is a randomized controlled trial (RCT). Participants installed an Android app (Phone Dashboard) and were randomly assigned to two orthogonal treatments:
1. **Bonus (B)**: Financial incentive ($50) to reduce screen time below a target
2. **Limit (L)**: Screen-time limit as a commitment device (with varying snooze options: 0, 2, 5, 20 minutes, or no snooze)

Randomization was stratified (Stratifier variable from stochatreat package). The primary outcome is FITSBY app usage measured by the Phone Dashboard app. Secondary outcomes are survey-based well-being measures from four survey waves (S1-S4).

The experimental design makes identification straightforward: any difference in outcomes between treated and control groups is attributable to the treatment. ANCOVA-style regressions (controlling for baseline outcome and strata) improve precision.

## Core Universe

### G1 (Usage outcomes):

**Design estimator axes**
- Difference-in-means (no controls)
- With pre-treatment covariates (ANCOVA)

**Controls axes**
- **LOO**: 2 specs (drop baseline usage, drop strata dummies)
- **Standard sets**: 4 specs (none / baseline only / strata only / baseline + strata)
- **Progression**: 4 specs (no controls / strata only / baseline usage only / full)

**Sample axes**
- Trim usage at 1st/99th percentile
- Trim usage at 5th/95th percentile
- Balanced panel (respondents present in P2 and P3)
- Balanced panel (respondents present in all periods)

**Data construction axes**
- Total phone usage (not just FITSBY apps)
- Usage measured in hours instead of minutes

**Treatment definition axes**
- Detailed limit types (L_1 through L_5 for different snooze durations)

**Functional form axes**
- log(1 + usage)
- asinh(usage)

### G2 (Well-being outcomes):

**Controls axes**
- LOO: drop baseline outcome, drop strata
- Standard sets: none / strata only / full

**Sample axes**
- Trim at 1st/99th, 5th/95th percentile
- S3 only, S4 only (instead of stacked)

## Inference Plan

### G1:
- **Canonical**: HC1 robust SEs (matching paper's 'robust')
- **Variant**: HC2 SEs (small-sample correction)

### G2:
- **Canonical**: Cluster SEs at UserID (matching paper's cluster(UserID) for stacked panel)
- **Variant**: HC1 robust SEs (for single cross-section per survey)

## Constraints
- Control-count envelope: [0, 2] for G1; [0, 3] for G2
- No linkage constraints (simple OLS with random assignment)
- Both treatment indicators (B and L) always included simultaneously (orthogonal treatments)
- Stratification dummies (i.Stratifier) match the randomization design and are the natural precision controls

## Budget
- G1: Max core specs 50, no subset sampling needed (small control set)
- G2: Max core specs 40, no subset sampling needed
- Total planned: ~45 (G1) + ~25 (G2) = ~70
- Seed: 163822

## What is excluded and why
- **Structural model (StructuralModel.R)**: Different analysis entirely -- estimates a structural model of self-control and temptation for welfare calculations. Not a treatment effect regression.
- **Heterogeneous treatment effects (Heterogeneity.do)**: These vary the outcome (by app, by time period, by person characteristics). They are explorations of the same treatment, not the main ITT claim.
- **HeterogeneityInstrumental.do**: IV specifications instrumenting limit compliance. Changes the estimand from ITT to LATE/TOT.
- **Beliefs.do**: Different outcome (predicted vs actual treatment effect). Not the main usage or well-being claim.
- **CommitmentDemand.do, QualitativeEvidence.do, Temptation.do**: Descriptive analyses, not treatment effect regressions.
- **SurveyValidation.do**: Different experiment (reward for accurate prediction). Not the main treatment.
- **COVID response (COVIDResponse.do)**: Descriptive statistics about COVID, not treatment effects.
- **Detailed limit types (L_1 through L_5)**: The paper presents both simple (L = any limit) and detailed specifications. Simple is the main one; detailed is included as a data construction rc variant.
