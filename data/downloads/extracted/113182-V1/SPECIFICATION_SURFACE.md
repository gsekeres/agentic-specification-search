# Specification Surface: 113182-V1

## Paper Overview
- **Title**: The Logic of Insurgent Electoral Violence (Condra, Long, Shaver, Wright, 2018, AER)
- **Design**: Instrumental variables (two distinct IV strategies)
- **Data**: District-election panel (Table 2) and polling-station-road-segment cross-section (Table 3) in Afghanistan

## Baseline Groups

### G1: Attack Timing and Voter Turnout (Table 2)

**Claim object**: The causal effect of election-day morning insurgent attacks on voter turnout in Afghan districts. Wind speed at 4:30 AM is used as an instrument for the number of attacks between 5-11 AM, exploiting the fact that wind affects the accuracy of indirect fire weapons used by insurgents.

**Why a separate group**: Table 2 is the first of two core IV strategies. The unit of observation is district-election, the instrument is surface wind speed, and the outcome is turnout share. This is distinct from the spatial IV in Table 3.

**Baseline specification** (Table 2, Column 4 -- preferred):
- Second stage: `total_v2_agcho10 ~ df_5to11 + [weather controls, quadratic] + plus_wind_00Z_10_pre14D + population_2010_adj | first`, cluster(DISTID)
- First stage: `df_5to11 ~ plus_wind_00Z_10 + [same controls] | first`, cluster(DISTID)
- Instrument: `plus_wind_00Z_10` (surface wind speed at 4:30 AM, positive values)
- Sample: `no_voting_either==0 & disrupt==1` (districts that held elections and experienced disruption)
- N controls: 18 (9 linear weather + 9 quadratic weather + pre-14-day wind + population)

**Control progressions revealed by the paper** (Table 2 columns):
1. OLS only (Col 1): no weather controls, just population
2. 2SLS with linear weather (Col 2): 9 linear weather + population = 10 controls
3. 2SLS with quadratic weather (Col 3): + 9 quadratic weather = 19 controls minus population... actually 18 controls
4. 2SLS with quadratic + pre-14-day wind (Col 4, baseline): + plus_wind_00Z_10_pre14D = 18 controls
5. Alternative outcomes (Cols 5-6): Ghani turnout, Abdullah turnout

### G2: IED Deployment and Voting Near Roads (Table 3)

**Claim object**: The causal effect of IED deployment near roads on vote totals at nearby polling stations. Nighttime cloud cover is used as an instrument for IED placement, because darkness (cloud cover) facilitates covert IED emplacement.

**Why a separate group**: Table 3 uses a completely different instrument (cloud cover vs wind speed), different endogenous variable (IED deployment vs attacks), different unit of observation (polling-station-road-segment vs district-election), and different outcome (vote totals vs turnout share).

**Baseline specification** (Table 3, Column 3 -- preferred):
- Second stage: `total_votes_wins ~ post_event_indicator + [geographic controls] + march_rain + march_rain2 | distid`, robust SE (with clustered SE reported separately)
- First stage: `post_event_indicator ~ cloudz_perc_election + [same controls] | distid`
- Instrument: `cloudz_perc_election` (nighttime cloud cover percentage on election day)
- Sample: `closure_indicator==0 & stations!=0`
- N controls: 7 (geographic + rain)

**Control progressions revealed by the paper** (Table 3 columns):
1. OLS (Col 1): geographic controls only
2. 2SLS without rain (Col 2): geographic controls, no rain
3. 2SLS with rain (Col 3, baseline): + march_rain + march_rain2
4. Alternative outcomes (Cols 4-5): Ghani votes, Abdullah votes

## RC Axes Included

### G1 RC Axes

**Controls**:
- **Control progressions**: Linear weather only (Col 2), quadratic weather (Col 3), quadratic + pre-14D (Col 4, baseline). These progressions are critical for assessing the exclusion restriction.
- **LOO drops**: Drop population, drop pre-14-day wind, drop quadratic terms. Each tests sensitivity to specific exclusion-restriction arguments.

**Sample**:
- **Include all districts** (drop disrupt==1 condition): SI Tables 5 and SI-9 run on the full sample (no_voting_either==0 only)
- **Include non-disrupted districts**: Tests whether the effect is driven by the sample restriction

**Alternative outcomes** (from the SI tables):
- `total_nc_TO` (no-corruption turnout), `susp_turnout_v2` (suspicious turnout), `corruption` (corruption measure)

**Alternative treatments** (from SI Tables 11-12):
- Varying the attack time window: df_5to7, df_5to8, df_5to9, df_5to10, df_5to12
- Attacks per 60K population: df_5to11_per60k (SI Table 10)

**Estimator variants**:
- LIML (more robust to weak instruments, especially relevant with weather instrument)

### G2 RC Axes

**Controls**:
- **LOO drops**: Drop rain controls, drop pre-event 6-month indicator, drop shape_leng, drop population
- **No rain**: Run without march_rain and march_rain2 (matches Col 2)

**Sample**:
- **Include non-disrupted areas**: Parallel to G1 sample expansion

**Alternative outcomes** (from SI tables):
- `total_nc_wins` (clean vote totals, no-corruption), `ghani_nc_wins`, `abdullah_nc_wins` (candidate-specific clean votes)
- `corrupt_perc` (corruption percentage, SI Table 25)

**Estimator variants**:
- LIML

## What Is Excluded and Why

- **Tables 6-7 (ANQAR survey regressions)**: These are cross-sectional OLS regressions of survey-based security perceptions on election dissatisfaction. They are secondary/supportive analysis, not the paper's core IV claim. They use a different dataset, different identification strategy, and different estimand.
- **Figures**: Descriptive time-series plots, not regression-based.
- **Exploration/heterogeneity**: No formal heterogeneity analysis in the paper.
- **Sensitivity**: Exclusion-restriction sensitivity would be a separate analysis outside the core surface.

## Budgets and Sampling

- **G1 max core specs**: 70 (baseline + 4 additional baselines + control progressions + LOO + sample + outcome/treatment alternatives + LIML)
- **G2 max core specs**: 55 (baseline + 3 additional baselines + LOO + sample + outcome alternatives + LIML)
- **Control subsets**: Exhaustive enumeration (structured progressions matching the paper's table columns)
- **Seed**: 113182

## Inference Plan

### G1
- **Canonical**: Cluster at district level (DISTID). Matches Table 2.
- **Variants**: Robust SEs without clustering.

### G2
- **Canonical**: Heteroskedasticity-robust SEs. Table 3 reports robust as primary, with clustered (pc_clust_thiessen) SEs as robustness.
- **Variants**: Cluster at Thiessen polygon level.

## Key Linkage Constraints

- **Both IV strategies**: Controls/FE are linked across first and second stages. Any control added to the second stage must appear in the first stage. The paper maintains this linkage throughout.
- **Weather controls (G1)**: The weather variables are critical for the exclusion restriction argument. The progression (linear -> quadratic -> +pre14D) is designed to absorb potential direct effects of weather on turnout. Dropping all weather controls would violate the identification logic.
- **Geographic controls (G2)**: Road-segment characteristics (ht_route_indicator, rcv2, shape_leng) and population are essential spatial controls. The exclusion restriction depends on cloud cover affecting voting only through IED deployment, conditional on these geographic features.
