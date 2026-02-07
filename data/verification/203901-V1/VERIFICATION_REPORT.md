# Verification Report: 203901-V1

## Paper Information
- **Paper ID**: 203901-V1
- **Journal**: AEJ: Economic Policy
- **Title**: Disrupting Drug Markets: The Effects of Crackdowns on Rogue Pain Clinics
- **Method**: Staggered Difference-in-Differences

## Baseline Groups

### G1: DEA crackdowns reduce opioid dispensing
- **Baseline spec_id**: baseline
- **Outcome**: MME_PC (morphine milligram equivalents per capita)
- **Treatment**: treatment (binary indicator for DEA enforcement action)
- **Expected sign**: Negative
- **Baseline coefficient**: -35.35 (SE = 6.50, p < 0.0001)
- **FE**: county_fips + state_year
- **Clustering**: county_fips
- **Controls**: None
- **Sample**: Counties within +/-3 years of treatment (N = 26,482)

## Summary Counts

| Category | Count |
|----------|-------|
| Total specifications | 71 |
| Baseline | 1 |
| Core tests (excluding baseline) | 56 |
| Non-core | 14 |
| Invalid | 0 |
| Unclear | 0 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_method | 1 | Baseline specification |
| core_fe | 5 | Fixed effects variations (county+year, county only, year only, none, state+year) |
| core_inference | 2 | Clustering variations (robust SE, state-level clustering) |
| core_sample | 27 | Sample restrictions (exclude states, exclude years, trim/winsorize, cohort splits, population splits, opioid baseline splits) |
| core_funcform | 6 | Functional form (log, IHS, rescaled outcome variables) |
| core_controls | 16 | Control variable additions and leave-one-out from full set, plus PDMP policy control |
| noncore_heterogeneity | 11 | Interaction models (7) and regional subsamples (4) |
| noncore_placebo | 3 | Placebo tests (pre-treatment, fake timing, unaffected outcome) |

## Top 5 Most Suspicious Rows

1. **robust/outcome/opiates_per100K** (opiatesper100K outcome): This is a mechanical rescaling of MME_PC by 10,000x. The coefficient (-353,549) and t-stat (-5.439) are identical to baseline once units are accounted for. Not a genuinely independent specification; inflates the specification count.

2. **robust/outcome/PCPV** (PCPV outcome): This is MME_PC divided by 10. The t-stat (-5.439) is identical to baseline. Same concern as above.

3. **robust/funcform/log_opiates** (log(opiatesper100K) outcome): The coefficient (-0.0692) is identical to robust/funcform/log_outcome because log(X*c) shifts only the intercept. Not independent from log_outcome.

4. **robust/sample/drop_state_FL**: Exact duplicate of robust/sample/exclude_FL. Both have coefficient -30.62, SE 6.16, N = 25,916. Appears to be an accidental duplicate in the specification search.

5. **robust/heterogeneity/high_density** and **robust/heterogeneity/high_md_pc**: These interaction model specs report the main treatment coefficient, which in the presence of an interaction term represents the effect when the moderator = 0 (low density or low MD per capita). The coefficient has a different interpretation than the average treatment effect and should not be directly compared to baseline. These are correctly classified as non-core but the coefficient could be misleading if compared naively to the baseline.

## Recommendations for Spec-Search Script

1. **Deduplicate rescaled outcomes**: The three rescaled specifications (opiatesper100K, PCPV, log_opiates) are not independent from the baseline or log_outcome specifications. Consider flagging or removing mechanical rescalings that produce identical t-statistics.

2. **Remove duplicate sample restriction**: drop_state_FL and exclude_FL are identical. The script should check for duplicates before writing results.

3. **Clarify heterogeneity interaction coefficients**: For interaction model specifications, consider reporting both the main effect and the interaction term, or explicitly noting that the reported coefficient is the conditional effect for the reference group, not the average treatment effect.

4. **Consider additional robustness dimensions**: The search could include (a) modern staggered DiD estimators (Callaway-SantAnna, Sun-Abraham) since treatment timing varies; (b) alternative weighting schemes; (c) Conley spatial standard errors.

5. **Subsample splits**: The large/small county and high/low baseline opioid splits are classified as core_sample but are borderline heterogeneity analyses. They test whether the effect holds in subgroups rather than testing the average effect. Users of this data should be aware of this distinction.
