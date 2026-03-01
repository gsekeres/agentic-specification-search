# Specification Surface Review: 112338-V1

## Paper
Duggan & Scott Morton (2010), "The Effect of Medicare Part D on Pharmaceutical Prices and Utilization," AER.

## Summary of Baseline Groups

- **G1**: Effect of Medicare market share on log price-per-day change (2003-2006). Table 2, Col 4 is the preferred specification. Claim object is well-defined: marginal effect of drug-level Medicare Rx share on brand-name drug price growth.
- **G2**: Effect of Medicare market share on log dose change (2003-2006). Table 3, Col 4 is the preferred specification. Distinct claim object (quantity rather than price), justifying a separate baseline group.

Both groups share the same identification strategy (selection-on-observables, cross-drug variation in Medicare share), the same treatment variable, and largely the same control set and sample restrictions. The two-group structure is appropriate.

## Changes Made

### 1. Renamed sample subset rank cutoffs to match actual do-file thresholds
- G1: `rc/sample/subset/top200` renamed to `rc/sample/subset/top292` (do-file line 450 uses `imsgrouprank03<=292`, not 200)
- G2: `rc/sample/subset/top200` renamed to `rc/sample/subset/top293` (do-file line 474 uses `imsgrouprank03<=293`)
- G1: `rc/joint/top200_with_interactions` renamed to `rc/joint/top292_with_interactions`

### 2. Moved outcome cross-checks to explore/ (they change the claim object)
- G1: `rc/form/outcome/log_sales_change` -> `explore/outcome/log_sales_change` (lsalesq0603 = price x quantity, changes claim from price growth to sales growth)
- G1: `rc/form/outcome/log_doses_change` -> `explore/outcome/log_doses_change` (ldoses0603 is the G2 quantity outcome; using it in G1 changes the claim object from price to quantity)
- G2: `rc/form/outcome/log_sales_change` -> `explore/outcome/log_sales_change` (same reasoning: changes from quantity to revenue)

### 3. No changes to level price outcome
- `rc/form/outcome/level_price_change` (ppd0603) in G1 is retained as RC. The level vs. log functional form preserves the claim object (same concept of price change, different scale). Confirmed the variable is constructed in the do-file as `ppd0603 = priceperday06 - priceperday03`.

## Key Constraints and Linkage Rules

- **Interaction controls**: When adding interaction terms (mcar0203prot, scmcar0203), the constituent level terms (protected, smallcat) must also enter. The surface correctly bundles these in `rc/controls/sets/interactions_*`.
- **Treatment decomposition**: When decomposing Medicare share into self-pay/other/dual, all components must enter together (they sum to total share). Correctly noted in surface constraints.
- **Outlier trim bounds**: Different for each outcome variable. The surface correctly specifies outcome-specific bounds.
- **No bundled estimator**: Single-equation OLS, no linkage constraints.

## Variable Verification

All variable names verified against the do-file (`regs-partd-final.do`) and data files:
- `lppd0603`, `ldoses0603`, `lsalesq0603`, `ppd0603`: constructed in do-file (confirmed lines 352-374)
- `mcar0203mepsrx`, `mcar0203mepspd`, `mcself0203mepsrx`, `dual0203mepsrx`: confirmed in merge0203AA.dta columns
- `yrs03onmkt`, `anygen`: constructed in do-file (lines 356, 412)
- `protected`, `smallcat`, `ther1`: constructed in do-file (lines 219-223, 405-406)
- `mcar0203prot`, `scmcar0203`: constructed in do-file (lines 526-527)
- `meps0203scripts`: confirmed in merge0203AA.dta columns

**Data availability note**: The IMS data files (`ims0106all.dta`, `ims0106data2`) are not included in the package, but the do-file constructs all analysis variables from intermediate files that ARE present (`merge0203AA.dta`, `usp548final.dta`). The full pipeline from raw IMS data cannot be run, but the analysis regressions can be replicated from the intermediate data.

## Budget/Sampling Assessment

- G1: 80 max specs. The core universe lists 3 additional baselines + ~30 RC specs + ~4 joint specs = ~37 explicit specs, well within budget. With control subset enumeration (small pool), total reaches ~50-60 specs.
- G2: 60 max specs. The core universe lists 3 additional baselines + ~20 RC specs + 2 joint specs = ~25 explicit specs, feasible within budget.
- Combined across both groups: ~110-140 total specifications, appropriate for the paper's scope.
- Seeds are distinct (112338, 112339), reproducible.

## What's Missing (minor)

- **Table 2 Col 6**: Uses `imsgrouprank03<=292`, labeled "top 200" in the paper. The surface now uses the actual threshold. Consider also adding `rc/sample/subset/top500` for G2 (currently only in G1).
- **No sensitivity to IMS rank definition**: The rank variable `imsgrouprank03` is taken as given. No variation in the underlying rank definition is included, which is appropriate since the IMS data construction is outside the package.
- **No Oster-style unobserved confounding sensitivity**: Could be useful for a cross-sectional design with selection-on-observables, but the paper does not pursue this. Not blocking.

## Final Assessment

**APPROVED TO RUN.** The surface is conceptually coherent, faithful to the manuscript's revealed search space, and correctly separates outcome cross-checks (which change the claim object) from functional-form RC axes (which preserve it). The budget is feasible and the sampling plan is reproducible. Both baseline groups are well-defined with distinct claim objects.
