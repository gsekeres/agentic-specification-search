# Specification Surface: 173341-V1

## Paper
"Vulnerability and Clientelism"
Bobonis, Gertler, Gonzalez-Navarro, and Nichter, American Economic Review, 2022

## Overview

This paper combines a household-level RCT (cistern distribution) with natural rainfall variation to study how vulnerability affects clientelist behavior in semi-arid northeastern Brazil. The paper has two main empirical settings: (1) individual-level survey data on requests for private goods from politicians (Tables 1, 3, 5), and (2) voting-section-level electoral outcomes (Table 4). The core claim is that both reducing vulnerability (via cisterns) and positive rainfall shocks reduce clientelist requests for private goods, especially among households with existing clientelist relationships.

## Baseline Groups

### G1: Effect of cisterns treatment and rainfall on clientelist requests

**Claim object:**
- **Outcome concept**: Whether an individual requested any private good from a politician (binary, stacked 2012-2013)
- **Treatment concept**: Joint effects of cisterns treatment (randomized) and standardized rainfall shocks (natural variation)
- **Estimand concept**: ITT effects of cisterns treatment and rainfall on private good requests, with municipality FE and year FE
- **Target population**: Individuals in semi-arid municipalities of northeastern Brazil enrolled in cistern program

**Baseline spec**: Table 3, Column 3
- `ask_private_stacked ~ treatment + rainfall_std_stacked + i.mun_id + year2012, cluster(b_clusters)`
- This is the joint specification including both treatment and rainfall but not their interaction. Column 4 adds the interaction.

**Why this baseline group**: Table 3 is the paper's core results table. It builds from separate cisterns-only and rainfall-only specifications to the combined specification. The treatment effect on requests is the paper's headline claim about vulnerability reducing clientelism.

### G2: Electoral outcomes (voting section level)

**Claim object:**
- **Outcome concept**: Votes for incumbent mayor in 2012 election (at voting-section level)
- **Treatment concept**: Share of cistern-treated individuals in each voting section (rescaled from household-level randomization)
- **Estimand concept**: ITT effect of cistern treatment exposure on electoral support for incumbent, in municipalities where incumbent ran for re-election
- **Target population**: Voting sections in 21 municipalities where incumbent mayor ran for re-election

**Baseline spec**: Table 4, Column 1
- `reghdfe incumbent_votes_section tot_treat_by_section_2 tot_study_2 eligible, absorb(location_id) cluster(location_id)`
- Uses wild-cluster bootstrap at municipality level for inference.

**Why this baseline group**: Table 4 represents a fundamentally different outcome concept (electoral behavior), unit of analysis (voting section), and treatment definition (section-level treatment share). This requires a separate baseline group.

### Why not additional baseline groups?

- **Table 2 vulnerability outcomes**: These are intermediate/mechanism outcomes (happiness, health, food security) that show cisterns reduce vulnerability. They support the paper's theory but are not the main claim about clientelism.
- **Table 5 heterogeneity by clientelist relationship**: This interacts treatment with a pre-treatment measure of clientelist relationships. It changes the estimand (CHET rather than ATE). Treated as exploration.
- **Table A8 other electoral outcomes**: Alternative outcomes within the same electoral setting. Treated as functional form variants for G2.

## Revealed Search Space

The paper reveals the following specification dimensions:

1. **Treatment definitions** (Table 3 columns):
   - Cisterns only (Col 1)
   - Rainfall only (Col 2)
   - Cisterns + rainfall (Col 3)
   - Cisterns + rainfall + interaction (Col 4)
   - Cisterns by year (Col 5)
   - Rainfall by year (Col 6)

2. **Outcome variants** (Table 3 Cols 7-8):
   - Private goods excluding water
   - Public goods (placebo-like test)

3. **Controls for citizen engagement** (Appendix Table A7):
   - Member of community association + interactions
   - President of community association + interactions
   - Voted in 2008 + interactions
   - All of the above

4. **Electoral sample restriction** (Table 4):
   - 21 municipalities: incumbent mayor ran for re-election (name_match==1)
   - 39 municipalities: broader incumbency definition (name/VP/party/coalition match)

5. **Separate year analysis** (Table 3 Cols 5-6): Treatment and rainfall effects by year

## Core Universe

### G1: Clientelist Requests

#### Baseline specs (2)
- `baseline`: Table 3 Col 3 (treatment + rainfall, mun FE + year)
- `baseline__table3_col4`: Table 3 Col 4 (adds treatment x rainfall interaction)

#### Design specs (1)
- `design/randomized_experiment/estimator/diff_in_means`: Bivariate regression (treatment only, no FE)

#### RC: Controls (8)
- **Sets**: no controls, municipality FE only, mun FE + year, mun FE + year + engagement controls (4 variants: association member, president, voted, all)
- **Progression**: Bivariate, treatment only, treatment + rainfall, treatment + rainfall + interaction

#### RC: Sample (3)
- Year 2012 only (unstacked)
- Year 2013 only (unstacked)
- Winsorize outcome at 1/99

#### RC: Functional form / outcome (3)
- Private goods excluding water (ask_nowater_private_stacked)
- Public goods (ask_public_stacked) -- placebo-like
- Request and receive private good (askrec_private_stacked)

#### RC: Treatment specification (3)
- Treatment effect by year (treat_2012, treat_2013)
- Rainfall effect by year (rainfall_std_stacked_2012, rainfall_std_stacked_2013)
- Treatment x rainfall interaction

#### RC: Data level (1)
- Household-level aggregation (clientelism_household_data.dta)

#### RC: Fixed effects (2)
- No municipality FE
- Municipality FE only (no year)

### G2: Electoral Outcomes

#### Baseline specs (2)
- `baseline`: Table 4 Col 1 (21 municipalities, incumbent mayor ran)
- `baseline__table4_col2`: Table 4 Col 2 (39 municipalities, broader incumbency definition)

#### Design specs (1)
- `design/randomized_experiment/estimator/diff_in_means`: No FE

#### RC: Sample (1)
- Broad incumbency sample (39 municipalities)

#### RC: Outcome (3)
- Challenger votes
- Turnout
- Blank and null votes

#### RC: Controls (2)
- Drop eligible voters control
- Drop study respondent share control

#### RC: Fixed effects (1)
- No location FE

## Total budget: ~50 core specs (G1) + ~20 core specs (G2)

## Inference Plan

### G1
**Canonical**: Cluster-robust SE at the neighborhood level (b_clusters), matching the paper.

**Variants** (recorded in inference_results.csv):
- HC1 (robust, individual-level)
- Cluster at municipality level (coarser; very few municipalities)

### G2
**Canonical**: Cluster-robust SE at the voting location level (location_id), matching the paper.

**Variants**:
- Wild-cluster bootstrap at municipality level (as in the paper for p-values)
- HC1 (robust, no clustering)

## What Is Excluded (and why)

1. **Table 2 vulnerability outcomes**: These are mechanism/intermediate outcomes (CES-D depression, health, food security). They demonstrate that cisterns reduce vulnerability but are not the main clientelism claim.
2. **Table 5 heterogeneity by clientelist relationship**: Interacts treatment with frequent_interactor. This changes the estimand (heterogeneous treatment effect). Treated as exploration, not core RC.
3. **Table A5 wealth and treatment**: Balance/mechanism check, not the main claim.
4. **Table A6 clientelism marker and treatment**: Shows clientelism marker is not affected by treatment. Diagnostic, not outcome.
5. **Table A9 politician responses**: Different outcome concept (politician behavior vs citizen behavior). Exploration.
6. **Table A10 citizen preferences**: Preferences for goods/policy, not clientelist behavior.
7. **Figures**: Visualizations of the same regressions, not additional specifications.

## Linkage Constraints

Not applicable for G1 (simple OLS with FE). For G2, the reghdfe absorbed FE structure is fixed to location_id; the rescaling of treatment variables to voting-section level is inherent to the design.

## Data Structure

### G1 datasets:
- `clientelism_individual_data.dta`: Individual-level cross-section
- `clientelism_individual_data_stacked.dta`: Individual-level stacked (2012 + 2013)
- `clientelism_household_data.dta`: Household-level cross-section
- Key variables: treatment, rainfall_std_stacked, ask_private_stacked, frequent_interactor, mun_id, b_clusters, year2012

### G2 dataset:
- `voting_data.dta`: Voting-section-level data (2012 election)
- Key variables: incumbent_votes_section, tot_treat_by_section_2, tot_study_2, eligible, location_id, mun_id, name_match

## Key Design Features

- **Randomization**: Household-level random assignment to cistern receipt within municipalities
- **Rainfall variation**: Natural (non-experimental) variation in standardized annual rainfall
- **Clustering**: Neighborhoods (b_clusters) for survey outcomes; voting locations for electoral outcomes
- **Municipality FE**: Absorb municipality-level confounders (10 municipalities in the study)
- **Stacking**: 2012 and 2013 data are pooled with year FE to increase power
- **Rescaling for electoral outcomes**: Treatment is converted to voting-section-level shares for the electoral analysis
