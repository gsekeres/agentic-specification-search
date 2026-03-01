# Specification Surface Review: 163822-V2

## Summary of Baseline Groups

- **G1**: Effect of screen-time interventions on phone usage (FITSBY apps)
  - Well-defined claim object: ITT effect of bonus/limit treatments on FITSBY app usage
  - Baseline spec matches the code in CommitmentResponse.do (`reg_usage_simple, fitsby`): `reg PD_P2_UsageFITSBY B L $STRATA PD_P1_UsageFITSBY, robust` where B = S3_Bonus and L = (S2_LimitType != 0)
  - Additional baselines cover usage in Periods 3-5, average across periods, and total phone usage -- all correctly typed as same claim with different outcome horizons/definitions
  - Design code `randomized_experiment` is correct: individual-level stratified RCT

- **G2**: Effect of screen-time interventions on well-being outcomes
  - Well-defined claim object: ITT effect of treatments on survey-based well-being measures
  - Baseline spec matches FDRTable.do: stacked S3/S4 panel with survey-wave-interacted strata and baseline outcomes
  - Additional baselines cover multiple well-being outcomes (addiction, SMS addiction, phone use change, life better, SWB, survey index)
  - Uses cluster(UserID) for stacked panel -- correct

## Changes Made

1. **Clarified G1 baseline notes**: Added explicit reference to the `reg_treatment` ado file call (`reg yvar B L $STRATA baseline, robust`) so the runner knows the exact Stata code being replicated.

2. **Clarified G2 FDRTable regression structure**: The FDRTable.do code does something non-obvious: it creates B3 and B4 (bonus x survey indicators), then replaces B with B4. The regression `reg yvar B B4 L indep` thus has B and B4 as collinear (Stata drops one). Updated the notes to document this behavior so the runner can replicate it correctly.

3. **Added G2 stacking note**: The G2 regressions require reshaping data from wide to long format (stacking S3 and S4 survey waves) with survey indicator S and interactions. Added a constraint note documenting this requirement.

4. No changes to baseline group definitions, RC axes, or budgets.

## Key Constraints and Linkage Rules

- No bundled estimator: simple OLS with random assignment
- G1: HC1 robust SEs (matching paper's `robust` option); G2: cluster at UserID for stacked panel
- Both treatment indicators (B and L) always included simultaneously -- they are orthogonal by design
- Stratification dummies (i.Stratifier) match the randomization design and are the natural precision controls
- For G2, the stacking structure (i.S and S-interacted controls) must be replicated -- this is not a standard single cross-section regression
- The `rc/data/treatment/detailed_limit_types` variant replaces the binary L with L_1 through L_5 (different snooze durations); this is the `gen_treatment` ado without the `simple` option

## Budget/Sampling Assessment

- G1: ~30 planned specs within 50-spec budget -- feasible
- G2: ~20 planned specs within 40-spec budget -- feasible
- No control subset sampling needed (only 2-3 controls in each group)
- Control progression covers the natural build-up: no controls, strata only, baseline usage only, full
- The diagnostic plan includes covariate balance check for G1 -- appropriate for RCT

## What's Missing (minor)

- No `rc/weights/*` variant: the CommitmentResponse.do code includes a `reg_usage_simple_balanced` program that uses entropy balancing weights (ebalance command). This could be an RC variant but is a minor robustness check for an RCT.
- The `rc/form/outcome/log1p` and `rc/form/outcome/asinh` specs change the coefficient interpretation from minutes to semi-elasticity. The surface correctly places these in `rc/form/*` but the runner should note the interpretation change when reporting.
- No attrition diagnostic is included in the diagnostics plan. Given that this is an app-based experiment with panel attrition across periods, `diag/randomized_experiment/attrition/attrition_diff` would be a useful addition but is not blocking.

## Final Assessment

**APPROVED TO RUN.** The surface is conceptually coherent, the two baseline groups are well-separated (usage vs well-being), the RCT design is correctly captured, and the budgets are feasible. The G2 stacking structure is now documented for the runner.
