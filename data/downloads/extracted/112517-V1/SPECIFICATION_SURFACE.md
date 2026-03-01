# Specification Surface: 112517-V1

## Paper Overview
- **Title**: What Do Emissions Markets Deliver and to Whom? Evidence from Southern California's NOx Trading Program (Fowlie, Holland & Mansur, 2012, AER)
- **Design**: Difference-in-differences via nearest-neighbor matching
- **Key finding**: The RECLAIM cap-and-trade program significantly reduced NOx emissions at regulated facilities compared to similar unregulated facilities. The ATT on emissions change (from 1990/93 to 2004/05) is large and negative, indicating the program reduced emissions by roughly 20-30% more than what would have occurred absent the program. The paper also examines environmental justice implications via heterogeneous effects by neighborhood demographics.

## Baseline Groups

### G1: Emissions Reduction Effect (Table 4)

**Claim object**: The average treatment effect on the treated (ATT) of RECLAIM program participation on facility-level NOx emissions change. Identification is through a matched DiD design comparing RECLAIM-regulated facilities to nearest-neighbor matched control facilities on pre-treatment emissions, with exact industry matching.

**Baseline specification** (Table 4, Panel A, Row 1 -- levels):
- Estimator: `nnmatch` (nearest-neighbor matching) with ATT estimand
- Outcome: `DIFFNOX` = POSTNOX - PRENOX (post-period average NOx minus pre-period average NOx)
- Treatment: `dumreclaim` (indicator for RECLAIM-regulated facility)
- Matching on: `PRENOX` (pre-period average NOx emissions)
- Exact matching on: `fsic` (4-digit SIC code)
- Bias adjustment: `PRENOX`
- Number of matches: 3
- Sample: Facilities in nonattainment counties (`nonattall==1`), excluding facilities with zero or missing emissions
- Inference: Abadie-Imbens robust variance estimator

**Additional baseline-like specs**:
- Table 4 Row 2 (levels with demographics): adds `income1` and `pctminor1` as matching variables, exact matching on `PRENOX_Q` (quartile)
- Table 4 log versions: outcome is `lnDIFFNOX = ln(POSTNOX+1) - ln(PRENOX+1)`, matching on `lnPRENOX`, bias adjustment on `lnPRENOX` and `lnPRENOX^2`
- Table 4 log with demographics: combines log form with demographic matching

## RC Axes Included

### Controls (matching variables)
- **No demographics**: Match only on PRENOX (baseline Row 1)
- **With demographics**: Add income1, pctminor1 as matching variables, plus exact match on PRENOX quartile (baseline Row 2)

### Sample restrictions
- **Drop electric utilities**: Exclude facilities flagged via R2009 electric utility data (Table 4 Panel B)
- **Southern California only**: Restrict control group to Southern Cal counties (Table 6 "South" check)
- **Northern California controls**: Drop Southern Cal control facilities, keeping only Northern Cal controls (Table 6 "North" check)
- **No SCAQMD controls**: Drop all South Coast Air Quality Management District control facilities (Table 6 "No SC" check)
- **Severe nonattainment only**: Restrict to severe ozone nonattainment areas (Table 6 "Severe" check)
- **Single-facility firms only**: Drop multi-facility control firms (Table 6 "Single" check)
- **Small firms**: Restrict control group to non-RECLAIM facilities in RECLAIM industries (Table 5)

### Period definitions
- **Pre=pd1 (1990/93), Post=pd4 (2004/05)**: Paper's preferred long-run comparison
- **Pre=pd2 (1997/98), Post=pd3 (2001/02)**: Shorter, medium-term comparison
- **Pre=pd1, Post=pd3**: Alternative intermediate comparison

### Functional form
- **Levels**: DIFFNOX = POSTNOX - PRENOX (baseline)
- **Logs**: lnDIFFNOX = ln(POSTNOX+1) - ln(PRENOX+1); matching on lnPRENOX
- **Log with quadratic bias adjustment**: bias adjustment includes lnPRENOX and lnPRENOX^2

### Matching parameters
- **Number of matches**: m = 1, 2, 3 (baseline), 4, 5 (Table A2 varies this from 1 to 5)

### Estimator alternatives
- **OLS with industry FE**: `areg DIFFNOX PRENOX dumreclaim, a(fsic) r cluster(ab)` -- this is reported alongside nnmatch in the paper as a comparison
- **OLS provides TWFE-style DiD**: Controlling for PRENOX and industry FE gives a conditional DiD estimate

### Joint variations
- Log specification combined with: dropping electric utilities, shorter time period, Southern Cal, severe nonattainment, single facility, no SCAQMD
- Levels specification combined with: dropping electric utilities, shorter time period
- OLS/areg combined with: nonattainment sample in both levels and logs

## What Is Excluded and Why

- **Tables 7-8 (Environmental Justice)**: These examine heterogeneous treatment effects by neighborhood income and minority composition. They involve a complex post-matching regression with many additional interaction terms (treatment x income, treatment x minority share, etc. with match-group FE). These are treated as exploration/heterogeneity, not as separate baseline claim objects. The paper's main claim is about aggregate emissions reduction, not distributional impacts.
- **Table 5 (Small Firms)**: This restricts to small non-RECLAIM firms as controls. Included as an RC axis (sample restriction) rather than a separate baseline group.
- **Table A3-A4 (Appendix)**: Additional robustness checks. Some overlap with RC axes already included.
- **Propensity score**: The paper uses pscore-based trimming (dropping observations with extreme propensity scores) as a preprocessing step. This is embedded in the data preparation and maintained across all specifications.

## Budgets and Sampling

- **G1 max core specs**: 80
- **Max control subsets**: 10 (only 2 optional demographic matching variables)
- **Seed**: 112517
- **Full enumeration**: Feasible for control subsets and matching parameters. Random sampling for joint axis (sample x period x form x estimator) combinations to stay within budget.

## Inference Plan

- **Canonical**: Abadie-Imbens robust variance estimator for nnmatch with m=3
- **Variants**: HC1 robust SE (for OLS alternatives); cluster by air basin (for areg specifications)
- Inference variants recorded in `inference_results.csv`

## Key Linkage Constraints

- When using log outcomes, matching variables and bias adjustment must also be in logs (lnPRENOX instead of PRENOX)
- When adding demographics to matching, exact matching on PRENOX quartile must also be added (they move together in the paper)
- The propensity score trimming step is always applied before matching (not varied)
- When switching to areg/OLS estimator, PRENOX enters as a control variable and industry enters as an absorbed FE, with clustering at the air basin level
