# Specification Search: 113182-V1

## Paper
Condra, Long, Shaver, Wright (2018), "The Logic of Insurgent Electoral Violence," AER.

## Surface Summary
- **Paper ID**: 113182-V1
- **Design**: Instrumental Variables (two distinct IV strategies)
- **Baseline groups**: 2 (G1: attack timing/turnout, G2: IED/vote totals)
- **Surface hash**: sha256:7c66178ad87d4ee2257773cd3a455abf68b5da3b9ca07d72c7fa072a0b10426e

### G1: Attack Timing and Voter Turnout (Table 2)
- **Instrument**: Wind speed at 4:30 AM (plus_wind_00Z_10)
- **Endogenous**: Morning attacks (df_5to11)
- **Outcome**: Voter turnout (total_v2_agcho10)
- **FE**: Election (first)
- **Cluster**: DISTID
- **Sample**: N=410 (no_voting_either==0 & disrupt==1)
- **Budget**: max 70 core specs

### G2: IED Deployment and Voting Near Roads (Table 3)
- **Instrument**: Nighttime cloud cover (cloudz_perc_election)
- **Endogenous**: IED deployment (post_event_indicator)
- **Outcome**: Total votes for winners (total_votes_wins)
- **FE**: District (distid)
- **SE**: Robust (canonical), clustered at pc_clust_thiessen (variant)
- **Sample**: N=15056 (closure_indicator==0 & stations!=0)
- **Budget**: max 55 core specs

## Execution Summary

### Counts
| Category | G1 | G2 | Total |
|----------|----|----|-------|
| Core specs (specification_results.csv) | 33 | 19 | 52 |
| Inference variants (inference_results.csv) | 2 | 4 | 6 |
| Successes | 33 | 18 | 51 |
| Failures | 0 | 1 | 1 |

### What Was Executed

**G1 Baselines (5 specs)**:
- Table 2 Col 4 (preferred, 18 controls)
- Table 2 Col 2 (linear weather, 10 controls)
- Table 2 Col 3 (quadratic weather, 17 controls)
- Table 2 Col 5 (Ghani turnout)
- Table 2 Col 6 (Abdullah turnout)

**G1 Design variants (1 spec)**:
- LIML estimator

**G1 RC variants**:
- Control progressions: linear, quadratic, quadratic+pre14D
- LOO drops: population, pre-14D wind, quadratic weather, windspeed_06Z, windspeed_12Z, temp_00Z, rain_00Z
- Sample: include all districts, include non-disrupted
- Alternative outcomes: total_nc_TO, susp_turnout_v2, corruption
- Alternative treatments: df_5to11_per60k, df_5to7, df_5to8, df_5to9, df_5to10, df_5to12
- Additional controls: pashto, terrain, DF_pretrend28d, intim, nighttime_observations

**G1 LIML + alternative outcomes (3 specs)**:
- LIML with ashrafTO_agcho10, abdullahTO_agcho10, total_nc_TO

**G2 Baselines (4 specs)**:
- Table 3 Col 3 (preferred, 7 controls with rain)
- Table 3 Col 2 (no rain, 5 controls)
- Table 3 Col 4 (Ghani wins)
- Table 3 Col 5 (Abdullah wins)

**G2 Design variants (1 spec)**:
- LIML estimator

**G2 RC variants**:
- LOO drops: rain controls, pre-event 6m, shape_leng, population_v2, ht_route_indicator, rcv2
- Sample: no_disrupt==1 subsample
- Alternative outcomes: total_nc_wins, ghani_nc_wins, abdullah_nc_wins, corrupt_perc

**G2 LIML + alternative outcomes (3 specs)**:
- LIML with total_nc_wins, ghani_nc_wins, abdullah_nc_wins

**Inference variants**:
- G1: HC1 (robust, no clustering) on baseline and Col 2
- G2: Cluster(pc_clust_thiessen) on baseline and alternative baselines

### Deviations
- None. All planned specifications were executed successfully.
- Note: LIML and 2SLS give numerically identical point estimates for just-identified IV
  (1 instrument, 1 endogenous). SEs differ slightly due to degrees-of-freedom corrections
  and FE handling (pyfixest absorbs FE; linearmodels uses dummies).

## Software Stack
- Python 3.12.7
- pyfixest 0.40.1
- linearmodels 6.1
- pandas 2.2.3
- numpy 2.1.3
- statsmodels 0.14.6

## Seed
- Surface seed: 113182
- Control subsets: exhaustive enumeration (no random sampling needed)
