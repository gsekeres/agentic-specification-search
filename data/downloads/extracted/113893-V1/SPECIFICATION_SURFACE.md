# Specification Surface: 113893-V1

## Paper Overview
- **Title**: Cross-Border Media and Nationalism: Evidence from Serbian Radio in Croatia (DellaVigna et al., 2014)
- **Design**: Instrumental variables (geographic signal strength as instrument for radio availability)
- **Data**: Municipality-level cross-sectional data (.dta) with vote shares, radio availability, signal strength, and demographic controls
- **Key context**: Uses geographic variation in Serbian radio signal strength to instrument for actual radio exposure in Croatian municipalities. Studies effect on nationalist voting and ethnic tensions.

## Baseline Groups

### G1: Nationalist Vote Share (Tables 3-4)

**Claim object**: Serbian radio availability increases vote share for the nationalist HSP party in Croatian municipalities.

**Baseline specification**:
- Formula (OLS): `Nazi_share ~ radio1 + controls_long [aweight=people_listed]`
- Formula (RF): `Nazi_share ~ s1_1 + controls_long [aweight=people_listed]`
- Outcome: `Nazi_share` (vote share for HSP nationalist party)
- Treatment: `radio1` (binary radio availability) or `s1` / `s1_1` (signal strength)
- Instrument: Signal strength (s1) predicting radio availability (radio1)
- Controls: Two sets -- controls_short (11 vars: geographic + census) and controls_long (21 vars: + war, monuments, streets, brewery)
- Region dummies (r1-r5) always included
- Weights: Population (people_listed)
- Clustering: Municipality (Opsina2)
- Focal coefficient: radio1 in OLS, or s1_1 in reduced form

**Additional baseline-like rows**:
- OLS with short controls
- Reduced form with geography only
- Reduced form with short controls

### G2: Ethnic Graffiti (Table 5)

**Claim object**: Serbian radio availability increases anti-Serb graffiti (binary indicator) in Croatian municipalities.

**Baseline specification**:
- Formula: `graffiti ~ radio1 + controls_long` (dprobit in paper, LPM for specification search)
- Binary outcome: graffiti presence

## RC Axes Included

### Controls (G1, primary)
- **Leave-one-out**: Drop each of the "manual/war" controls individually (war, mon, name_of_the_streets_c/i, pivo_S, bliz, etc.)
- **Standard sets**: Geography only (log_distance_full + r1-r5), controls_short (census + geographic), controls_long (all)
- **Additions**: Hungarian radio control (radio_hung), Croatian radio signal (eloss5050powerHKR), free-space loss (floss_1) -- from Table 6 robustness
- **Random subsets**: 15-25 random draws from the control pool (excluding mandatory region dummies)

### Sample restrictions
- Distance < 75km subsample (Table 7, restricts to areas closer to Serbian transmitters)
- Outlier trimming on vote share

### Design variants
- 2SLS (explicit IV regression)
- Continuous signal strength as treatment
- Signal strength dummies (quartile indicators s_dum_1 through s_dum_4)
- Two radio station dummies (radio1 + radio2)

### Weights
- Unweighted (removing population weighting)

### Clustering alternatives
- Municipality (baseline)
- Robust (no clustering)
- Spatial Conley SE

## What Is Excluded and Why

- **Other party vote shares (Table 4: hdz_share, sdp_share, turnout)**: Reported as supplementary outcomes. Could be explore/* but excluded from core to focus on the headline Nazi_share result.
- **2003 and 2011 election year comparisons (Table 8)**: Different election years with potentially different dynamics; better as explore/*.
- **Survey analysis (Survey-Replication.do)**: Different dataset and claim object (attitudes vs. behavior).
- **Lab experiment (Experiment-Replication.do)**: Completely different setting and identification.

## Budgets and Sampling

- **G1 max core specs**: 75
- **G2 max core specs**: 25
- **Max control subsets**: 25 (G1), 10 (G2)
- **Seed**: 113893
- **Sampling**: Controls organized into blocks (geographic, census, manual/war, region). Region dummies always included.

## Inference Plan

- **Canonical**: Cluster at municipality (Opsina2)
- **Variants**: HC1 robust, Conley spatial HAC (50km cutoff)
