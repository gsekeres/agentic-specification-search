# Verification Report: 113547-V1

## Paper Information
- **Paper ID**: 113547-V1
- **Title**: Stealth Consolidation: Evidence from an Amendment to the Hart-Scott-Rodino Act
- **Authors**: Thomas G. Wollmann
- **Journal**: AER: Insights
- **Method**: Triple Difference-in-Differences (OLS)
- **Verified At**: 2026-02-09

## Baseline Specifications

### Baseline Group G1: Triple DID (primary result)
- **Claim**: The 2001 HSR Act Amendment caused an increase in horizontal (same-industry) mergers among newly-exempt firms relative to non-horizontal mergers and relative to never-exempt firms.
- **Spec ID**: `baseline_triple_diff_logs`
- **Coefficient**: 0.218 (SE=0.050, t=4.33, p<0.001)
- **Expected sign**: Positive
- **Outcome**: log_mergers
- **Treatment**: post * horizontal * below (triple interaction)
- **FE**: year, year*below, year*horizontal
- **N**: 72, R2=0.997

### Baseline Group G2: Newly-exempt DID (supporting)
- **Claim**: Within newly-exempt mergers, horizontal mergers increased relative to non-horizontal mergers after the amendment.
- **Spec ID**: `baseline_newly_exempt_logs`
- **Coefficient**: 0.186 (SE=0.046, t=4.06, p<0.001)
- **Expected sign**: Positive
- **Outcome**: log_mergers
- **Treatment**: post * horizontal (DID interaction on newly-exempt subsample)
- **FE**: year
- **N**: 36, R2=0.988

## Classification Summary

| Classification | Count | Percentage |
|---------------|-------|------------|
| Baseline | 2 | 2.2% |
| Core | 65 | 69.9% |
| Non-core | 26 | 28.0% |
| **Total** | **93** | **100%** |

## Detailed Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| baseline | 2 | Primary paper specifications |
| core_sample | 43 | Sample restrictions (excl. software, excl. large mergers, time windows, LOO, excl. crisis/transition years) |
| core_controls | 6 | Alternative post-period definitions (post>2000, >2002, >2003) applied to both triple DID and newly-exempt DID |
| core_fe | 6 | Linear/quadratic time trends replacing year FE |
| core_funcform | 4 | Levels instead of logs, IHS transformation |
| core_method | 3 | Poisson count model, WLS |
| core_inference | 3 | Classical (non-robust) standard errors |
| non_core_placebo | 15 | Never-exempt control group checks, non-horizontal merger placebos, pre-period timing placebos |
| non_core_alt_outcome | 5 | Dollar value of mergers, horizontal share |
| non_core_heterogeneity | 3 | Software-only subsample (SIC 7372) |
| non_core_diagnostic | 3 | Horizontal-only DD, total mergers DD |

## Classification Logic

### Core specifications (65 specs)
A specification is classified as **core** if it tests the same fundamental hypothesis as the baseline -- that the HSR exemption caused an increase in horizontal mergers among newly-exempt firms -- while varying one or more of:

1. **Sample restrictions** (43 specs): Excluding software mergers (SIC 7372), excluding large mergers (>=150MM), excluding both, narrowing/shifting the time window (3-6 year symmetric windows, asymmetric windows), excluding crisis years (2008-2009), excluding the transition year (2001), and leave-one-year-out analysis (17 jackknife specs).

2. **Controls/treatment timing** (6 specs): Redefining the post-period cutoff to 2000, 2002, or 2003 instead of 2001.

3. **Fixed effects variation** (6 specs): Replacing year fixed effects with linear or quadratic time trends (applied to both logs and levels).

4. **Functional form** (4 specs): Using merger counts in levels instead of logs, inverse hyperbolic sine transformation, first differences of log horizontal mergers.

5. **Estimation method** (3 specs): Poisson count model, weighted least squares.

6. **Inference variation** (3 specs): Classical (non-robust) standard errors.

### Non-core specifications (26 specs)
A specification is classified as **non-core** if it tests a different hypothesis, serves as a placebo, or is diagnostic:

1. **Placebo/control group tests** (15 specs): Never-exempt DID regressions (the untreated group should show no effect), pre-period placebo timing tests (fake treatment at 1997/1998/1999), non-horizontal merger DDs (the "placebo outcome" that should not respond to the exemption), first-diff non-horizontal mergers.

2. **Alternative outcomes** (5 specs): Dollar value of mergers (testing a different claim about dollar volumes vs. counts), horizontal share DD.

3. **Heterogeneity** (3 specs): Software-only subsample analyses (testing whether the effect is concentrated in one industry, not the average treatment effect).

4. **Diagnostics** (3 specs): Horizontal-only DD in logs and levels (collapses the triple DID into a simple DD without the non-horizontal comparison), total mergers DD.

## Key Classification Decisions

1. **Never-exempt specs are non-core placebo**: Every spec with "never_exempt" in the name runs the DID on the untreated group. These are control group validity checks, not robustness tests of the main claim. They should be null.

2. **Dollar-value specs are non-core**: Switching from merger counts to dollar values fundamentally changes the claim being tested (number of horizontal mergers vs. dollar value of horizontal mergers). The SPECIFICATION_SEARCH.md notes these are a "different outcome" and the results are qualitatively different (insignificant in the triple DID).

3. **Post-period redefinitions are core controls**: Changing whether "post" means >2000, >2001, >2002, or >2003 tests the same claim under different treatment timing assumptions. These are robustness checks on the treatment definition.

4. **First-diff horizontal DD is core**: Although it collapses from a triple DID to a simple DD, the first_diff_horizontal_dd_logs tests the same directional claim (horizontal mergers among newly-exempt increased) using a different functional form (first differences). The first_diff_nonhorizontal_dd_logs is classified as non-core placebo because it tests the non-horizontal "control" outcome.

5. **Software-only is non-core heterogeneity**: These specs examine whether the effect is concentrated in one industry (SIC 7372), which is a heterogeneity decomposition rather than a test of the average treatment effect.

6. **Levels specs are core functional form**: Using raw merger counts instead of log(mergers) tests the same hypothesis with a different functional form.

## Robustness Summary

Among the 65 core specifications:
- **Positive coefficient**: 62/65 (95.4%). The 3 negative are the linear trend and quadratic trend triple DID in logs (both highly insignificant), which the SPECIFICATION_SEARCH.md explicitly discusses as a known sensitivity.
- **Significant at 5%**: 57/65 (87.7%). The 8 insignificant specs are: linear trend triple DID logs/levels, linear trend newly-exempt logs/levels, quadratic trend triple DID/newly-exempt logs, and first-diff horizontal DD logs.
- **Core robustness verdict**: STRONG. The result is robust across essentially all sample restrictions, time windows, post-period definitions, inference methods, Poisson models, IHS, and WLS. It is sensitive to replacing year FE with parametric trends, which is well-documented by the authors.

## Files Generated
- `verification_baselines.json`: Baseline identification with baseline groups, claims, and paper metadata
- `verification_spec_map.csv`: Full classification of all 93 specifications with baseline group mapping
- `VERIFICATION_REPORT.md`: This report
