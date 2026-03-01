# Specification Surface: 151841-V1

## Paper
Beaman, Magruder, & Robinson (2023), "Targeting High Ability Entrepreneurs Using Community Information: Mechanism Design In The Field," American Economic Review.

## Baseline Groups

### G1: Heterogeneous ITT (Winner * Peer Rank)

**Claim object:**
- **Outcome concept**: Marginal returns to capital grants measured by income and profits
- **Treatment concept**: Interaction of grant receipt (Winner) with average peer-assessed quintile rank of marginal returns (Quint_Rank_NS, excluding self-rank)
- **Estimand concept**: Whether peers can identify entrepreneurs with higher returns to capital; the key estimand is the heterogeneous ITT -- the coefficient on Winner * Rank in a panel FE regression
- **Target population**: Small-scale entrepreneurs in Kolkata, India, non-attriters, survey rounds 1-4

**Design**: Randomized experiment (weighted lottery for business grants within groups of ~5 entrepreneurs)

**Baseline spec (Table 2, Panel A):**
- Outcome: Trim_Income, Trim_Profits_30Days (trimmed at 99.5th percentile of percentage change)
- Treatment: Winner_Quint_Rank_NS (Winner * average peer quintile rank, excluding self-rank)
- Model: `xtreg outcome Winner_Quint_Rank_NS Winner i.Surveyor_Code i.Survey_Version i.survey_month [weight=Propensity_Score], fe clu(GroupNumber) i(Id)`
- FE: Household (Id), Surveyor, Survey round, Survey month
- Weights: Inverse propensity score (accounts for weighted lottery)
- Clustering: Group level (GroupNumber)
- Reported with and without 26 baseline controls interacted with Winner

## Design Estimator Alternatives

1. **ANCOVA** (`design/randomized_experiment/estimator/ancova`): Replace household FE with strata FE, include baseline outcome as control. Uses post-baseline rounds only (rounds 2-4).

## Robustness Check Axes

### Controls
- **Add psychometric controls**: 17 psychometric personality questions from baseline added to 26 standard controls
- **Leave-one-out panels**: Drop each of four control panels (demographics, business type, household, business characteristics) one at a time
- **Control subsets**: Use individual panels and pairwise/triple combinations as control sets

### Data Construction (Rank variable)
- **Include self-rank**: Use Quintile_Rank instead of Quint_Rank_NS
- **Relative (zero-sum) rank**: Use Rel_Rank_NS instead of quintile (non-zero-sum)
- **Median rank**: Use Q_Rank_med_NS (median of peer reports instead of mean)
- **SD of rank interaction**: Add Winner * SD_Rank * Rank interaction

### Functional Form
- **Log outcomes**: log(Income+1), log(Profits+1) instead of trimmed levels
- **Tercile rank**: Replace continuous rank with top/middle tercile dummies

### Sample
- **All 5 waves**: Include demonetization-affected survey round 5
- **Groups of 5 only**: Restrict to groups with exactly 5 members
- **Trim 5-95**: Tighter outcome trimming at 5th-95th percentiles
- **Winsorize 1-99**: Winsorize outcome at 1st-99th percentile

### Weights
- **Unweighted**: Drop inverse propensity score weights

### Fixed Effects
- **Drop surveyor FE**: Remove Surveyor_Code from FE set
- **Drop survey month FE**: Remove survey_month from FE set
- **Strata FE instead of HH FE**: Replace household FE with randomization strata FE (high-leverage change)

### Joint Specifications
Multiple axes changed simultaneously: ANCOVA +/- controls, log outcomes +/- controls, rank variants +/- controls, sample restrictions +/- controls, unweighted + controls, tercile rank +/- controls.

## Inference Plan

**Canonical**: Cluster SE at GroupNumber (randomization group level)

**Variants**:
- HC1 robust (no clustering)
- Cluster at household (Id) level

## Constraints and Guardrails

- Controls are always interacted with Winner (not added as main effects), following the paper's approach of mean-imputing missing values and including Winner * miss_control interactions
- Control count envelope: [0, 43] -- paper shows no-control and 26-control specs; with psychometric adds 17 more
- Linked adjustment: Not applicable (single-equation estimator)
- Outcomes are trimmed at the 99.5th percentile of percentage change from the previous round, following the paper's trimming rule

## Budget and Sampling

- **Total budget**: ~100 specification rows (across two outcomes x multiple axes)
- **Control subsets**: 9 named combinations (4 individual panels + 5 pairwise/triple), not random sampling
- **Seed**: 151841
- Full enumeration is feasible given the structured panel-based control groups
