# Specification Surface Review: 113513-V1

## Summary of Baseline Groups

**G1 (Economics Share vs Other Discipline Shares)**: The paper is purely descriptive -- no causal claim is made. The regression content consists of bivariate OLS of economics major share on each other discipline's share plus a year trend. This is correctly captured as a single baseline group with multiple baseline spec IDs (one per comparison discipline).

- No missing baseline groups: the paper only runs these bivariate regressions.
- The demographic subsamples (females, nonwhites) are correctly treated as RC sample restrictions rather than separate baseline groups, since the claim structure is identical.

## Design Selection

- `cross_sectional_ols` is appropriate even though the data is time-series (year-level). The paper treats it as cross-sectional OLS with robust SEs.
- `design_audit` correctly notes the descriptive/correlational nature with no causal claim.
- No design variants included, which is appropriate -- no alternative estimators (IV, matching) are meaningful for a descriptive correlation.

## RC Axes Assessment

- **Sample restrictions**: Demographic subsamples and time splits are the right axes.
- **Functional form**: Second major share and year-trend variations are appropriate.
- **Missing axes**: No weighting (the data is already population-level aggregates from IPEDS). No clustering (year-level data, only ~16 obs).

## Controls Multiverse Policy

- `controls_count_min=1, controls_count_max=1` is correct. The paper never uses multiple discipline shares simultaneously; each regression is bivariate (one discipline + year).
- No control subset sampling needed.

## Inference Plan

- HC1 canonical is correct (matches paper's `robust` option).
- Newey-West HAC is a valuable variant given potential serial correlation in the ~16 time-series observations.
- Classical OLS provides a comparison point.

## Budget Assessment

- 55 max core specs is appropriate for the naturally small specification space (11 disciplines x ~5 variants).
- Full enumeration is feasible and appropriate.

## What's Missing

- The very small sample size (~16 observations per regression) limits the informational content of specification analysis. This should be noted as a caveat in interpretation.
- No bootstrap inference variant is listed, but given N~16, finite-sample inference is a concern that HAC partly addresses.

## Verdict

**Approved to run.** The surface is faithful to the manuscript's limited regression content. The descriptive nature of the paper means the specification surface is inherently narrow, which is correctly reflected in the modest budget.
