# Specification Surface: 157781-V1

**Paper**: "Rebel on the Canal: Disrupted Trade Access and Social Conflict in China, 1650-1911"

**Design**: difference_in_differences

**Created**: 2026-02-24

---

## Paper Summary

This paper studies the effect of the closure of China's Grand Canal on social conflict (rebellion incidence). The Grand Canal was the primary north-south trade route in imperial China, and its effective closure in the mid-19th century (following a series of reforms and the Yellow River's course change) disrupted trade access for counties along the canal. The paper uses a difference-in-differences design comparing rebellion incidence in canal-adjacent counties vs. non-adjacent counties, before and after the canal closure.

The panel spans 575 counties across six provinces around the Grand Canal from 1650 to 1911. The main dependent variable is `ashonset_cntypop1600` (inverse hyperbolic sine of rebellion onset per capita using 1600 population). The treatment is `interaction1` = `alongcanal * reform` (Along Canal x Post).

---

## Baseline Group

### G1: Canal closure and rebellion incidence

**Claim object**:
- **Outcome**: Rebellion incidence, measured as asinh(rebellion onset per 1600 population in millions)
- **Treatment**: Along Canal x Post (interaction1 = alongcanal * reform)
- **Estimand**: DiD ATT -- the effect of canal closure on social conflict in canal-adjacent counties relative to non-adjacent counties
- **Population**: 575 counties in six provinces around the Grand Canal, 1650-1911

**Baseline specs** (Table 3, Columns 1-5):

The paper presents a progression of FE specifications as its main results:

| Column | County FE | Year FE | Pre-reform rebellion x Year FE | Province x Year FE | Prefecture trend | Controls x Post |
|--------|-----------|---------|-------------------------------|-------------------|-----------------|-----------------|
| 1      | Yes       | Yes     | No                            | No                | No              | No              |
| 2      | Yes       | Yes     | Yes                           | No                | No              | No              |
| 3      | Yes       | Yes     | Yes                           | Yes               | No              | No              |
| 4      | Yes       | Yes     | Yes                           | Yes               | Yes             | No              |
| 5      | Yes       | Yes     | Yes                           | Yes               | Yes             | Yes             |

Column 4 (full FE without controls) is the preferred specification based on discussion in the text. Column 5 adds all time-varying controls interacted with post.

**Why only one baseline group**: The paper has a single main claim (canal closure increased rebellions) tested with one outcome concept across all main tables. Tables 4-7 are robustness/mechanisms using the same claim object with alternative treatment definitions, sample restrictions, and heterogeneity decompositions.

---

## What Is Included and Why

### Design alternatives
- **TWFE**: The paper uses standard two-way FE (reghdfe). Since this is a sharp single-date treatment (not staggered adoption), TWFE is the natural and uncontroversial estimator. No modern staggered-DiD estimator alternatives are needed because treatment timing is uniform.

### Robustness checks (rc/*)

**Controls (leave-one-out from Col 5)**:
- Drop each of the 15 post-interacted controls one at a time from the full specification (Table 3 Col 5). Controls include:
  - Geography: larea_after (ln land area x post), rug_after (ruggedness x post)
  - Climate: disaster, disaster_after, flooding, drought, flooding_after, drought_after
  - Population: lpopdencnty1600_after (ln initial population density x post)
  - Agriculture: maize, maize_after, sweetpotato, sweetpotato_after, wheat_after, rice_after

**Control set progressions**:
- No controls (bivariate DiD)
- Climate-only subset (disaster, flooding, drought and their post interactions)
- Geography-only subset (larea_after, rug_after)
- Agriculture-only subset (maize, sweetpotato, wheat, rice and post interactions)
- Full controls (all 15)
- Also: the FE progression from Cols 1-5 (each adding more FE layers) is itself a revealed control/adjustment progression

**Fixed effects variations**:
- Drop ashprerebels x year FE (from Col 4 baseline)
- Drop province x year FE
- Drop prefecture trend
- Add prefecture x year FE (upgrade from prefecture linear trend to full interactions)

**Sample restrictions**:
- Drop Opium War battlefield counties (Table 7, Col 1)
- Drop Taiping Rebellion region counties (Table 7, Col 2)
- Restrict to pre-1826 / post-1826 split (alternative pre/post cutoff)
- Restrict sample period to 1700-1911 or 1750-1911

**Functional form / outcome transformations**:
- Log(1 + onset/pop1600) instead of asinh
- asinh(onset/area) -- per-area normalization instead of per-capita
- asinh(onset/cntypop) -- time-varying population normalization
- onset_any (binary: any rebellion onset)

**Treatment definition alternatives** (Table 4):
- Canal length density x Post (continuous intensity)
- Canal town share x Post (share of towns within 10km of canal)
- Distance to canal x Post (continuous, negative expected sign)

**Outlier trimming**:
- Trim Y at 1st/99th percentile

### Inference variants
- **Canonical**: Cluster at county (OBJECTID), matching paper baseline
- **Conley spatial HAC**: 500km distance cutoff, 262-year lag (as reported in paper brackets in every table)
- **Prefecture clustering**: Coarser geographic clustering
- **Province clustering**: Very coarse clustering (~6 provinces, a stress test)

---

## What Is Excluded and Why

### Table 2 (pre-trends)
Table 2 tests pre-treatment trends (alongcanal * year for 1776-1825). This is a diagnostic, not an estimate of the main DiD effect. Listed in diagnostics_plan.

### Table 5 (North vs. South triple difference)
Table 5 decomposes the effect by northern vs. southern canal sections using a triple interaction (Along Canal x Post x North). This is heterogeneity analysis that changes the estimand from an average DiD effect to a location-specific effect, so it belongs in exploration rather than core.

### Table 6 (placebo treatments)
Table 6 tests placebo treatments (along Yangtze, along old Yellow River, along coast, along courier routes). These are falsification/diagnostic exercises, not estimates of the baseline estimand.

### Table 7 Panel B (interaction with Opium/Taiping)
Panel B of Table 7 includes triple interactions with Opium War and Taiping indicators. These are heterogeneity/mechanism analyses.

### Appendix tables
Online appendix tables (A1-A7) contain additional robustness checks from the authors. These were not reviewed exhaustively but the revealed axes (alternative normalizations, sample restrictions, etc.) are captured in the rc specs above.

---

## Constraints

- **Controls count envelope**: [0, 15]. Table 3 Cols 1-4 use no explicit controls (relying entirely on FE absorption), while Col 5 adds 15 time-varying controls. This is the full revealed range.
- **FE structure**: The paper reveals a 4-level FE progression: (1) county + year, (2) + pre-reform rebellion trends, (3) + province x year, (4) + prefecture linear trends. The preferred specification is level 4.
- **Sharp single-date treatment**: All counties are treated at the same time (reform period), so there is no staggered adoption issue. This eliminates the need for heterogeneity-robust DiD estimators.
- **Spatial structure**: Counties have geographic coordinates used for Conley SE computation. The paper computes spatial HAC SEs with 500km distance and 262-year lag cutoffs.

---

## Budget and Sampling

- **Total budget**: ~100 specs. This is feasible through full enumeration:
  - 5 baseline/progression specs (Table 3 Cols 1-5)
  - 15 LOO control drops (from Col 5)
  - ~5 control subset groupings
  - ~4 FE variations
  - ~4 sample restrictions
  - ~4 outcome transformations
  - ~3 treatment alternatives
  - ~1 outlier trim
  - ~5 additional baseline_spec_ids
- No random subset sampling needed.

---

## Key Implementation Notes

1. **Data file**: `Data/Final/rebellion.dta` is the sole analysis dataset. Created by `Program/Clean/clean.do` from raw data in `Data/Raw/`.
2. **Key variables set in generalsetup.do**:
   - `$Y` = `ashonset_cntypop1600` (outcome)
   - `$X` = `interaction1` (treatment = alongcanal * reform)
   - `$ctrls` = 15 time-varying post-interaction controls
   - Panel: `xtset OBJECTID year`
3. **Estimator**: `reghdfe` for all main specifications (Stata's high-dimensional FE linear regression).
4. **Conley SEs**: The paper uses `ols_spatial_HAC` (Hsiang 2010) after residualizing with `hdfe`. This requires first projecting out FEs, then computing spatial HAC on residualized data.
5. **Variable construction**: The outcome variable `ashonset_cntypop1600` = asinh(onset_all / (cntypop1600/1000000)) is constructed in generalsetup.do. Several alternative normalizations are also constructed there.
6. **Pre-reform rebellion measure**: `ashprerebels` = asinh(total pre-reform rebellion per capita) is used as a county-level covariate interacted with year FE in the progressive FE specifications.
