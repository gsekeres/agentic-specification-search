# Specification Surface: 113517-V1

## Paper
Moscarini & Postel-Vinay, "The Relative Power of Employment-to-Employment Reallocation and Unemployment Exits in Predicting Wage Growth", AER P&P 2017.

## Summary

This paper uses a two-stage estimation procedure to study the predictive power of various labor market flows (EE, UE, NE, EU, EN transitions and unemployment rate) for aggregate wage growth across demographic markets and time.

### Two-Stage Procedure
1. **First stage**: Individual-level transition indicators (e.g., eetrans_i) are regressed on controls (lagged state, union, size, occupation, industry, public sector) with market*time FE absorbed. The market*time FE are extracted as predicted flow rates.
2. **Second stage**: Predicted wage growth (itself a market*time FE from a first-stage wage-change regression) is regressed on the predicted flow rates with market FE absorbed and a linear time trend.

The focal parameter is the coefficient on `xee` (predicted EE transition rate) in the second stage, measuring how much EE reallocation predicts wage growth after controlling for other flows.

## Baseline Groups

### G1: Log Nominal Earnings Growth
- **Outcome**: xdlogern_nom (predicted delta log nominal earnings)
- **Treatment**: xee (predicted EE rate)
- **Baseline spec**: All flows specification (spec 6 from Table 1, Panel A)
- **Sample**: All employed (EZeligible=1, DWeligible=1), N~6.2M

### G2: Log Real Earnings Growth
- **Outcome**: xdlogern (predicted delta log real earnings)
- **Treatment**: xee (predicted EE rate)
- **Baseline spec**: All flows specification (spec 6 from Table 1, Panel B)
- **Sample**: Same as G1

### G3: Log Nominal Hourly Wage Growth
- **Outcome**: xdloghwr_nom (predicted delta log nominal hourly wage)
- **Treatment**: xee (predicted EE rate)
- **Baseline spec**: All flows specification (spec 6 from Table 1, Panel C)
- **Sample**: Hourly-paid workers (EZeligible_hw=1, DWeligible_hw=1), N~3.2M

### G4: Log Real Hourly Wage Growth
- **Outcome**: xdloghwr (predicted delta log real hourly wage)
- **Treatment**: xee (predicted EE rate)
- **Baseline spec**: All flows specification (spec 6 from Table 1, Panel D)
- **Sample**: Same as G3

## Rationale for 4 Baseline Groups

The paper explicitly presents results for all 4 dependent variables as separate panels in Table 1, and the patterns differ substantially across them (e.g., xee is strongly positive for nominal earnings but negative for real hourly wages). Each depvar represents a distinct outcome concept and the paper treats each as a separate headline result.

## Core Universe (per group)

Each baseline group has the same structural core universe:

### Control progressions (revealed by paper's Table 1 columns)
- **EE only** (spec 1): xee + ym_num | mkt
- **UE only** (spec 2): xue + ym_num | mkt
- **UR only** (spec 3): xur + ym_num | mkt
- **EE + UE** (spec 4): xee + xue + ym_num | mkt
- **EE + UE + UR** (spec 5): xee + xue + xur + ym_num | mkt
- **All flows** (spec 6, baseline): xee + xue + xur + xne + xen + xeu + ym_num | mkt
- **Grouped flows** (spec 7): xee + xur + xnue + xenu + ym_num | mkt

### Leave-one-out from baseline (drop one flow control)
- Drop xue, drop xur, drop xne, drop xen, drop xeu, drop ym_num

### Sample variations
- **Job stayers** (spec 8): eetrans_i==0 & lagemp>0

### Functional form
- **EE interaction** (spec 9): add xee*eetrans_i to all-flows specification

### Weights
- **Unweighted**: Drop survey weights

### Fixed effects
- **Drop market FE**: OLS without absorbed market FE

## Constraints

- **Linked adjustment**: First-stage controls are fixed. The specification search varies only the second-stage RHS variables.
- **Control count**: 1 (EE only) to 7 (all flows + time trend)
- **First-stage is invariant**: All first-stage regressions (producing xee, xue, etc.) are run once and their FE are mapped to all observations. Varying first-stage controls would be a separate (much more expensive) axis not revealed by the paper.

## Inference

- **Canonical**: Classical (IID) SE -- the paper's areg commands in second stage do not specify robust or cluster.
- **Variants**: HC1 (robust), cluster by market (mkt)

## Budget

Target ~17 specs per group x 4 groups = ~68 total core specs, plus inference variants.
