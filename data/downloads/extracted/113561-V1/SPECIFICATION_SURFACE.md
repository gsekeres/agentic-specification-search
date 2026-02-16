# Specification Surface: 113561-V1

**Paper**: Fong & Luttmer (2009), "What Determines Giving to Hurricane Katrina Victims? Experimental Evidence on Racial Group Loyalty"

**Design**: Randomized experiment (survey experiment)

## 1. Baseline Groups

Four baseline groups, corresponding to the four outcome measures for white respondents in Table 4:

| Group | Outcome | Treatment | Population | Baseline Table |
|-------|---------|-----------|------------|---------------|
| G1 | `giving` (actual donation, $0-100) | `picshowblack` | White respondents | Tab 4, Panel 1, Col b |
| G2 | `hypgiv_tc500` (hypothetical giving, topcoded $500) | `picshowblack` | White respondents | Tab 4, Panel 2, Col b |
| G3 | `subjsupchar` (charity support, 1-7 scale) | `picshowblack` | White respondents | Tab 4, Panel 3, Col b |
| G4 | `subjsupgov` (government support, 1-7 scale) | `picshowblack` | White respondents | Tab 4, Panel 4, Col b |

### Rationale
The paper's main claim is about how the race of Hurricane Katrina victims (randomly assigned via photos) affects white Americans' generosity and support for relief organizations. All four outcomes are treated as headline results in Table 4. Table 5 presents identical robustness checks for all four outcomes. Each outcome measures a distinct aspect of generosity, warranting separate baseline groups.

### What is excluded as baseline
- **Full sample results** (Table 4 Col a): These are not the primary claim; the paper focuses on white respondents.
- **Black respondent results** (Table 4 Col c): Different target population; would be exploration.
- **Table 3 (perception outcomes)**: per_hfhdif is a mechanism/mediator, not a giving outcome.
- **Table 6 (interaction effects)**: Heterogeneity by racial attitudes; exploration, not baseline.

## 2. Revealed Search Space

Table 5 reveals the paper's own robustness dimensions:

1. **Sample restrictions**: Main survey variant only (surveyvariant==1), Slidell only, Biloxi only, race-shown subsample only (picobscur==0)
2. **Control sets**: No demographics (minimal), baseline (nraud + cntrldems), extended (+ addcntrl1)
3. **Weights**: Paper uses tweight (survey weights); some specs use mweight; some are unweighted
4. **Alternative estimators**: Censored normal regression (giving, hypgiv_tc500), ordered probit (subjsupchar, subjsupgov)

## 3. Control Blocks

Controls are organized into semantic blocks:

- **pic_controls** (MANDATORY): `picraceb`, `picobscur` -- other treatment arms, always included
- **manipulation_audio**: `aud_econdis`, `nraudworthy`, `aud_republ`, `aud_govtben`, `aud_church`, `aud_loot`, `cityslidell`
- **survey_variant**: `var_fullstakes`, `var_racesalient`
- **demographics_age**: `age`, `age2`
- **demographics_education**: `edudo`, `edusc`, `educp`
- **demographics_income**: `lnhhinc`, `dualin`
- **demographics_marital**: `married`, `male`, `singlemale`
- **demographics_geography**: `south`
- **demographics_labor**: `work`, `disabled`, `retired`
- **charitable_giving**: `dcharkatrina`, `lcharkatrina`, `dchartot2005`, `lchartot2005`
- **extra_controls**: `hfh_effective`, `lifepriorities_help`, `lifepriorities_mony`

## 4. Core Universe (per group)

Each baseline group includes:

### Design variants
- Difference-in-means (no controls except pic_controls)

### RC: Controls
- **sets/none**: Treatment + pic_controls only (bivariate)
- **sets/minimal**: Treatment + pic_controls + manipulation_audio + survey_variant (no demographics)
- **sets/baseline**: Paper's baseline control set
- **sets/extended**: Baseline + extra_controls
- **LOO blocks**: Drop each demographic block one at a time (7 blocks)
- **Progression**: Build up from bivariate to full

### RC: Sample
- Main survey variant only (surveyvariant==1)
- Slidell only (cityslidell==1)
- Biloxi only (cityslidell==0)
- Race-shown treatment only (picobscur==0)

### RC: Weights
- Unweighted (paper's Table 5 uses both weighted and unweighted)
- Paper weights (tweight)

### RC: Functional form (G1 only)
- Topcode giving at 50 (half of max)

## 5. Inference Plan

**Canonical**: HC1 robust SE for all groups (matches Stata `robust`).

**Variant** (G1 only): HC3 as a small-sample stress test.

## 6. Budget

| Group | Planned specs |
|-------|--------------|
| G1 | ~25 |
| G2 | ~22 |
| G3 | ~20 |
| G4 | ~20 |
| **Total** | **~87** |

This exceeds the 50-spec target while remaining feasible.

## 7. Diagnostics Plan

No explicit diagnostics planned. Balance checks are standard for RCTs but are not part of the specification search. The experiment's randomization is the identification strategy.
