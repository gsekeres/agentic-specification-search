# Specification Surface: 138401-V1

## Paper Overview

This paper studies the long-run effects of childhood measles vaccine exposure on adult labor market outcomes. It exploits a continuous difference-in-differences design: the introduction of the measles vaccine in the early 1960s provides cohort-level variation in childhood vaccine exposure, while cross-state variation in pre-vaccine measles incidence rates generates treatment intensity variation. The treatment variable `M12_exp_rate` interacts the state-level average 12-year pre-vaccine measles incidence rate with years of childhood vaccine exposure (0 for pre-1949 cohorts, up to 16 for post-1963 cohorts).

## Baseline Groups

### G1: Effect of measles vaccine exposure on adult labor market outcomes

**Claim object**: Childhood measles vaccine exposure improves adult labor market outcomes (income, employment, poverty) for cohorts born in higher pre-vaccine measles incidence states after vaccine introduction.

**Baseline specification** (Table 2, Column 3 -- log CPI-adjusted income): OLS regression of `ln_cpi_income` on `M12_exp_rate` with birth state FE, birth year FE, ACS survey year FE, and rich demographic interaction FE (age x black x female, bpl x black, bpl x female, bpl x black x female), plus `black` and `female` as level controls. SEs clustered at birth-state x birth-year cohort level.

**Why one baseline group**: All six outcome variables in Table 2 (CPI income, CPI income excluding zeros, log income, poverty, employment, hours worked) test the same underlying claim -- that measles vaccine exposure improved adult economic outcomes. Log income is the most natural focal outcome (continuous, well-behaved). The other five outcomes are listed as additional baseline spec_ids.

**Why not separate groups by outcome**: The outcomes are all manifestations of the same labor-market-improvement estimand. The paper presents them together in a single table and does not frame any subset as a separate claim.

## Included Robustness Checks

### Controls (Leave-one-out)

The control set is minimal (black, female as level effects), since most variation is absorbed by the rich FE structure. LOO drops each:
- Drop `black`
- Drop `female`

### Sample Restrictions (from Table 3 and additional)

1. **Exclude partial exposure cohorts**: Keep only fully exposed (exposure=16) and unexposed (exposure=0) cohorts, as in Table 3
2. **Narrow cohort window (1941-1971)**: Restrict to birth years 1941-1971, as in Table 3
3. **Narrow cohort window (1945-1975)**: Alternative cohort window
4. **Men only / Women only**: Gender-specific subsamples
5. **White only / Black only**: Race-specific subsamples
6. **Age 30-55**: Tighter age restriction for prime working age
7. **Exclude DC**: Drop District of Columbia observations
8. **Post-2005 ACS only**: Restrict to later ACS survey years
9. **Pre-2010 ACS only**: Restrict to earlier ACS survey years
10. **Employed only**: Restrict to employed individuals (for wage/hours outcomes)

### Fixed Effects (from Tables 4 and Appendix Table 4)

1. **Add Census region (4) x birth year FE** (`breg4_byear`): More flexible cohort trends at region level (Appendix Table 4)
2. **Add Census division (9) x birth year FE** (`breg9_byear`): More flexible cohort trends at division level (Table 4, Appendix Table 4)
3. **Add birth-state-specific linear cohort trends** (`bpl_cohort_trend`): State-specific trends in outcomes over cohorts (Appendix Table 4)
4. **Add mean reversion control** (`mean_reversion_control`): Interact pre-cohort mean outcome with pre-cohort indicator (Table 4)
5. **Drop bpl_black interaction FE**: Test sensitivity to removing birth-state x race interactions
6. **Drop bpl_female interaction FE**: Test sensitivity to removing birth-state x gender interactions
7. **Drop bpl_black_female interaction FE**: Test sensitivity to removing triple interactions
8. **Drop ageblackfemale FE**: Test sensitivity to removing age x race x gender interactions
9. **Simplified FE**: Keep only core DiD FE (bpl, birthyr, year) without demographic interaction FE

### Treatment Construction (from Appendix Table 2)

The paper uses `M12_exp_rate` (12-year pre-vaccine average) as baseline. Appendix Table 2 reports results using alternative averaging windows:
- `M2_exp_rate` through `M11_exp_rate`: Using 2-year through 11-year pre-vaccine measles incidence averages

These are treatment variable construction choices (data construction RC), not sample or control changes.

### Functional Form / Outcome

- **Level income** (instead of log): Use `cpi_incwage` as outcome
- **Level income excluding zeros**: Use `cpi_incwage_no0`

## Excluded from Core

- **Event study estimates** (Figure 3): These are pre-trends diagnostics / dynamic treatment effects. They belong in `diag/*` or `explore/*`.
- **Falsification/placebo test** (Table 5): Uses 1960-1970 Census data with contemporaneous outcomes for children. This is a `diag/*` placebo test.
- **Population density analysis** (Appendix Table 1): Cross-sectional correlation between density and measles rates. Diagnostic/descriptive only.
- **Figure 4 (other diseases)**: Event studies for other childhood diseases as falsification. Diagnostic.

## Inference Plan

The paper's baseline clusters at `bplcohort` (birth state x birth year). Appendix Table 3 systematically reports results with many alternative clustering choices:

- **Canonical**: Cluster at `bplcohort` (birth state x birth year)
- **Variants** (all from Appendix Table 3):
  - `bpl` (birth state only -- coarser)
  - `bplexposure` (birth state x exposure level)
  - `bpl_region4` (Census region -- only 4 clusters, questionable)
  - `bpl_region9` (Census division -- 9 clusters)
  - `statefip` (state of residence)
  - `statecohort` is not included because `birthyr` already absorbs time
  - `birthyr` (birth year only)
  - `exposure` (exposure level only)
  - HC1 robust (no clustering)

This paper has an unusually rich revealed inference search space since Appendix Table 3 explicitly tests 10+ clustering schemes across all outcomes.

## Budget and Sampling

- **Total core budget**: 80 specifications
- **Controls subset budget**: 0 (only 2 non-FE controls, LOO covers the space)
- **Seed**: 138401
- **Sampler**: not applicable (no random subset sampling needed)

Full enumeration is feasible for all axes since the control space is tiny (absorbed by FE) and the RC axes are well-defined.

## Key Linkage Constraints

- The treatment variable `M12_exp_rate` is constructed as `(avg_12yr_measles_rate * exposure) / 100000`. Alternative treatment constructions (M2-M11) change the averaging window but preserve the same interaction structure.
- The rich FE structure (`i.bpl i.birthyr i.ageblackfemale i.bpl_black i.bpl_female i.bpl_black_female`) is required for identification -- dropping birth state or birth year FE would break the DiD design. These FE should not be varied independently.
- `black` and `female` appear both as level controls and as interaction-FE components. Dropping them as level controls is a valid LOO check; dropping the FE interactions is not (it changes the comparison group).
- When restricting to a single race or gender subsample, the corresponding interaction FE should be dropped.

## Estimated Spec Count

| Category | Count |
|---|---|
| Baseline (ln_cpi_income) | 1 |
| Additional baselines (5 other outcomes) | 5 |
| Design (TWFE) | 1 |
| RC/controls/loo | 2 |
| RC/controls/sets | 1 |
| RC/sample | 10 |
| RC/fe/add | 4 |
| RC/fe/drop | 4 |
| RC/fe/simplify | 1 |
| RC/data/treatment_construction | 11 |
| RC/form/outcome | 3 |
| RC/joint | 6 |
| **Total** | **51 core specs** |

Note: The RC specs are run for the focal outcome (ln_cpi_income). The 5 additional baseline specs cover the other outcome variables at the baseline specification. At a budget of 80, additional outcome-specific RCs can be added.
