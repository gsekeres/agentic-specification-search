# Specification Surface: 114542-V1

**Paper**: Cattaneo, Galiani, Gertler, Martinez & Titiunik (2009). "Housing, Health and Happiness." *American Economic Journal: Economic Policy*, 1(1), 75-105.

**Design**: Randomized Experiment (Piso Firme program)

**Date**: 2026-02-24

---

## Paper Summary

Mexico's Piso Firme program replaced dirt floors with cement floors in poor households. The paper exploits the staggered rollout: Torreon (treatment) received the program in 2000-2001 while Durango (control) had not yet received it by the 2005 survey. The analysis compares outcomes across the two cities using OLS with cluster-robust standard errors at the census block level (the unit of randomization).

The paper reports results across two datasets:
- **Household-level** (N=2,783): cement floor coverage (Table 4), satisfaction/mental health (Table 6), robustness checks (Table 7)
- **Individual/child-level** (N varies by outcome): child health outcomes (Table 5)

## Baseline Groups

### G1: Cement Floor Coverage (Housing First Stage)
- **Outcome concept**: Share of rooms with cement floors
- **Treatment**: `dpisofirme` (=1 Torreon, =0 Durango)
- **Estimand**: ITT of Piso Firme on floor quality
- **Population**: Household-level, 2005 survey
- **Baseline spec**: `reg S_shcementfloor dpisofirme, cl(idcluster)` (Table 4, Model 1)
- **Additional baselines**: Models 2-4 with progressive controls

### G2: Satisfaction and Mental Health
- **Outcome concept**: Floor satisfaction, house satisfaction, life satisfaction, depression (CES-D), perceived stress (PSS)
- **Treatment**: `dpisofirme`
- **Estimand**: ITT of Piso Firme on satisfaction/mental health
- **Population**: Household-level, 2005 survey
- **Baseline spec**: `reg S_satisfloor dpisofirme, cl(idcluster)` (Table 6, Model 1)
- **Additional baselines**: S_satishouse, S_satislife, S_cesds, S_pss

### G3: Child Health
- **Outcome concept**: Child health (parasite count, diarrhea, anemia, height-for-age, weight-for-height)
- **Treatment**: `dpisofirme`
- **Estimand**: ITT of Piso Firme on child health
- **Population**: Individual-level (children 0-5), 2005 survey
- **Baseline spec**: `reg S_parcount dpisofirme, cl(idcluster)` (Table 5, Model 1)
- **Additional baselines**: S_diarrhea, S_anemia, S_haz, S_whz

## Design

This is a randomized experiment. The paper uses simple OLS with treatment indicator, progressively adding controls (Models 1-4):
- **Model 1**: No controls (pure treatment-control comparison)
- **Model 2**: Demographics + health controls + missingness dummies
- **Model 3**: Model 2 + social program controls
- **Model 4**: Model 3 + economic controls

All regressions cluster SEs at `idcluster` (census block), matching the randomization unit.

## Control Variable Groups

### Household-level (G1, G2):
- **HH_demog1**: S_HHpeople, S_headage, S_spouseage, S_headeduc, S_spouseeduc
- **HH_demog2**: S_dem1-S_dem8
- **HH_health**: S_waterland, S_waterhouse, S_electricity, S_hasanimals, S_animalsinside, S_garbage, S_washhands
- **HH_social**: S_cashtransfers, S_milkprogram, S_foodprogram, S_seguropopular
- **HH_econ**: S_incomepc, S_assetspc
- Plus missingness dummies for each group

### Individual-level (G3):
- **CH_demog**: S_HHpeople, S_rooms, S_age, S_gender, S_childma, S_childmaage, S_childmaeduc, S_childpa, S_childpaage, S_childpaeduc
- **HH_health**: S_waterland, S_waterhouse, S_electricity, S_hasanimals, S_animalsinside, S_garbage, S_washhands
- **HH_social**: S_cashtransfers, S_milkprogram, S_foodprogram, S_seguropopular
- **HH_econ**: S_incomepc, S_assetspc
- Plus age-trimester-gender dummies (dtriage_*) and missingness dummies

## RC Axes

### Controls (primary axis)
- **Leave-one-out**: Drop each key control from Model 4 (full model) one at a time
- **Control sets**: Models 1-4 as progressive control sets
- **Random subsets**: 10 random draws from the control pool (G1 only)

### Sample
- **Trim outcomes**: 1/99 and 5/95 percentile trimming on the outcome variable

## Inference Plan

- **Canonical**: Cluster-robust SEs at census block level (`CRV1: idcluster`), matching paper
- **Variant 1**: Heteroskedasticity-robust HC1 (no clustering)
- **Variant 2**: Cluster at municipality level (`idmun`) -- coarser clustering (G1 only)

## Budget

- G1: ~60 specs (4 baselines + 1 design + ~30 RC controls + 2 sample + random subsets)
- G2: ~30 specs (5 baselines + ~15 RC controls/outcome combos + sample trims)
- G3: ~40 specs (5 baselines + ~20 RC controls/outcome combos + sample trims)
- **Total target**: ~130 specs across 3 baseline groups

## What Is Excluded and Why

- **Table 7 robustness outcomes** (S_instcement, S_instsanita, etc.): These are alternative outcomes the paper uses as placebo/falsification, not the main claim objects. Excluded from core.
- **Census balance tests** (Table 2): These are diagnostics, not estimates.
- **S_mccdts, S_pbdypct** (cognitive tests): Very high missingness (>5000 of 6693 missing), excluded as unreliable.
- **Moran's I spatial tests**: Diagnostic, not part of core specification surface.
- **S_malincom, S_palincom** (parental income): Treated as robustness outcomes in paper, not main claims.
