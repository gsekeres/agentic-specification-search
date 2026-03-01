# Specification Surface: 112338-V1

## Paper Overview
- **Title**: The Effect of Medicare Part D on Pharmaceutical Prices and Utilization (Duggan & Scott Morton, 2010, AER)
- **Design**: Cross-sectional OLS
- **Key finding**: Medicare Part D increased prices for brand-name drugs with higher Medicare market share, with little impact on quantities. A one-standard-deviation increase in Medicare market share is associated with roughly a 2 percentage-point larger price increase from 2003 to 2006.

## Baseline Groups

### G1: Price Effects (Table 2)

**Claim object**: The effect of a drug's Medicare market share on its log price-per-day change from 2003 to 2006 (inflation-adjusted). The paper's preferred specification is Table 2, Column 4 (with outlier trimming and excluding cancer drugs).

**Baseline specification** (Table 2, Column 4):
- Formula: `lppd0603 ~ mcar0203mepsrx + yrs03onmkt + anygen`
- Outcome: `lppd0603` (log change in price per day, 2003-2006, CPI adjusted)
- Treatment: `mcar0203mepsrx` (Medicare market share of Rx, pooled 2002-2003 MEPS)
- Controls: `yrs03onmkt` (years on market by 2003), `anygen` (any generic competitor by 2006)
- Sample: Top 1000 brand-name, non-OTC drugs on market by 2003; outliers trimmed at [-1.1, 1.095]; cancer drugs excluded
- Weights: `meps0203scripts` (total MEPS prescription count 2002+2003)
- Inference: Robust (HC1) standard errors

**Additional baseline-like specs**:
- Table 2 Col 1: bivariate (no controls, no trimming, no cancer exclusion)
- Table 2 Col 2: add controls, no trimming, no cancer exclusion
- Table 2 Col 3: add trimming, no cancer exclusion

### G2: Quantity Effects (Table 3)

**Claim object**: The effect of a drug's Medicare market share on its log change in total doses from 2003 to 2006. The paper presents this as a companion result to the price effect.

**Baseline specification** (Table 3, Column 4):
- Formula: `ldoses0603 ~ mcar0203mepsrx + yrs03onmkt + anygen`
- Outcome: `ldoses0603` (log change in total doses, 2003-2006)
- Treatment: same as G1
- Controls: same as G1
- Sample: Same base sample; outlier bounds differ ([-3.95, 1.51]); cancer drugs excluded
- Weights and inference: same as G1

## RC Axes Included

### Controls
- **Leave-one-out**: Drop yrs03onmkt; drop anygen
- **Single additions**: Add protected class indicator, small therapeutic category, lagged price/quantity change
- **Interaction sets**: Medicare share interacted with protected class and/or small category (from Table 5)
- **Full set**: All available controls and interactions simultaneously

### Sample restrictions
- **Outlier trimming**: No trimming; paper's trimming bounds; symmetric 5/95 and 2/98 percentile trims
- **Cancer exclusion**: Include vs. exclude cancer drugs (thercat==8)
- **Rank cutoffs**: Top 200, top 300, top 500 (paper uses top 1000 and top ~292 in Table 2 Col 6)
- **Generic competition**: Drop drugs facing generic entry by 2006 (Table 3 Col 7 uses this)

### Treatment definition
- **Spending share**: Use `mcar0203mepspd` (Medicare share of spending) instead of prescription share (Table 2 Col 5)
- **Self-pay decomposition**: Split Medicare share into self-pay (`mcself0203mepsrx`) vs other Medicare (`mcoth0203mepsrx`) (Table 4)
- **Dual decomposition**: Further split out dual-eligible share (`dual0203mepsrx`) (Table 4 Col 3)

### Outcome definition
- **Level price change**: `ppd0603` instead of log
- **Log sales change**: `lsalesq0603` (combines price and quantity; Table 4 Col 7)
- **Log dose change**: `ldoses0603` (quantity, for G1 cross-check; and primary outcome for G2)

### Weights
- **Unweighted**: Drop analytic weights
- **Alternative weights**: `meps02scripts + meps03scripts` (explicit sum rather than harmonic)

### Fixed effects
- **Therapeutic category FE**: Add therapeutic category (`ther1`) fixed effects (not in paper's main tables but a natural robustness check)

### Joint variations
- No trimming + include cancer drugs
- Top 200 + interaction controls
- Spending share + exclude cancer
- Unweighted + no trimming

## What Is Excluded and Why

- **Table 5 heterogeneity regressions**: These interact Medicare share with protected class/small category indicators. These are included as RC axes (interaction control sets) rather than separate baseline groups because the paper treats them as robustness/extensions of the main effect.
- **Table 4 decomposition**: Treatment decomposition into self-pay, other-Medicare, and dual components is included as an RC axis (treatment definition variant).
- **Design variants**: No within-design alternatives (no FE to vary, no instruments, no panel structure; the data is a single cross-section of drugs).
- **Exploration**: No heterogeneity or CATE analysis beyond what is captured by the paper's interaction terms.

## Budgets and Sampling

- **G1 max core specs**: 80
- **G2 max core specs**: 60
- **Max control subsets**: 20 (G1), 15 (G2) -- small control pool makes near-exhaustive enumeration feasible
- **Seeds**: 112338 (G1), 112339 (G2)

## Inference Plan

- **Canonical**: HC1 robust standard errors (matches paper's `reg ..., robust`)
- **Variants**: Cluster by therapeutic category (ther1) -- used in Table 5; HC2; HC3
- Inference variants are recorded in `inference_results.csv`, not in `specification_results.csv`

## Key Linkage Constraints

- Medicare share interactions (`mcar0203prot`, `scmcar0203`) must be varied jointly with their constituent level terms (`protected`, `smallcat`)
- When decomposing Medicare share into self-pay/other/dual, the components must enter together (they sum to total Medicare share)
- Outlier trim bounds must be appropriate for the chosen outcome variable (different bounds for log prices vs. log doses vs. log sales)
