# Specification Surface Review: 116280-V1

## Summary of Baseline Groups

- **G1**: Effect of state regulation on mutual organizational form (mutual ~ mlaw + controls | cluster(state))
  - Well-defined claim object: association between state insurance regulation variables and probability of mutual organizational form
  - Baseline spec matches paper's Table 2 Column 1 exactly: logit of mutual on mlaw, slaw, regulate, nfc, reform, and interactions, plus decade dummies, with SEs clustered at state level (47 clusters)
  - N = 881 observations, Pseudo R2 = 0.265, key coefficient on mlaw = -2.12 (z = -5.48)
  - Additional baseline specs (favor_spec, rcorp_spec) represent the paper's alternative regulatory measures from Tables 3-4

## Changes Made

1. **Added `reported_object: "index_coef"` to design_audit**: The baseline estimator is logit, so the reported coefficient is a logit index coefficient, not an odds ratio or average marginal effect. This is critical for coefficient comparability across estimator variants (LPM, probit).

2. **Added `n_clusters: 47` to design_audit**: Documents the number of state clusters for inference interpretation.

3. **Removed binary treatment variants from rc, moved to explore**: `rc/form/treatment/mlaw_binary`, `rc/form/treatment/slaw_binary`, and `rc/form/treatment/favor_binary` were removed from rc_spec_ids. Dichotomizing the continuous mlaw/slaw variables at the median changes the coefficient interpretation fundamentally (from a marginal effect of regulatory requirements to an average difference between high/low regulation states). These belong in `explore/*` and are documented in the constraints section.

4. **Replaced with `rc/form/treatment/favor_as_treatment`**: The favor specification (Table 3) replaces mlaw/slaw with the combined favor measure. This IS in the paper's revealed search space and is better characterized as an alternative treatment operationalization that preserves the broad claim about regulation and organizational form.

5. **Added linkage notes to constraints**: Documented four key linkage rules:
   - Reform interaction terms are mechanically linked to the reform dummy
   - Favor specs substitute for mlaw/slaw (not additive)
   - Decade dummies and rcorp are substitutes for time controls
   - clogit automatically drops 13 states with all-same outcomes

## Key Constraints and Linkage Rules

- **No bundled estimator**: Single-equation logit/LPM/probit, no linked adjustment needed
- **Reform-interaction linkage**: refmlaw, refslaw, refregulate, refnfc are mechanically defined as reform*X -- dropping reform while keeping interactions is incoherent. Similarly, reffavor is linked to the favor specification only.
- **Treatment measure substitution**: mlaw/slaw specifications and favor specifications are alternative regulatory measures from the paper. They are substitutes, not additive. The favor specs use reffavor instead of refmlaw/refslaw.
- **Time controls**: Decade dummies (ten2-ten5) and rcorp (real interest rate) serve similar time-control roles; the paper uses one or the other, never both simultaneously.
- **clogit sample restriction**: The conditional logit with group(state) automatically drops 13 states (131 observations) that have all-stock or all-mutual formations.

## Budget/Sampling Assessment

- 49 planned core specs (after removing 3 binary treatment specs) is well within the 60-spec budget
- 10 random control subset draws with seed=116280 is reproducible
- LOO covers 12 droppable controls (slaw, regulate, nfc, reform, refmlaw, refslaw, refregulate, refnfc, ten2, ten3, ten4, ten5)
- Control progression provides 5 build-up steps from bivariate to full
- Sample restrictions (trimming, state exclusion, period restrictions) provide meaningful sensitivity
- Estimator variants (LPM, probit, clogit) are well-chosen and standard

## What's Missing (minor)

- No Oster-style sensitivity analysis is planned, but this is appropriate given the paper explicitly frames the relationship as associational rather than causal
- No weights axis: the paper does not use weights, so this is correctly excluded
- Could add `rc/form/outcome/ame` (average marginal effects from logit) to make coefficients comparable to LPM, but this is a reporting choice rather than a specification choice

## Verification Against Code

Verified against MS20040270.log which contains the complete Stata output:
- Table 2: logit mutual mlaw slaw regulate nfc reform refmlaw refslaw refregulate refnfc ten2-ten5, cluster(state) -- matches baseline
- Table 3: logit with favor replacing mlaw/slaw -- confirmed in surface
- Table 4: logit with rcorp instead of decade dummies -- confirmed in surface
- Table 5: clogit with group(state) cluster(state) -- confirmed as rc/fe/add/state and rc/form/estimator/clogit_state
- All variable names verified against the data dictionary in the log file header

## Final Assessment

**APPROVED TO RUN.** The surface is conceptually coherent, faithful to the manuscript's revealed search space (Tables 2-5), and the budget is feasible. The key correction was removing binary treatment dichotomizations that change the claim object's coefficient interpretation. The logit baseline with LPM/probit/clogit alternatives provides a well-structured estimator sensitivity analysis.
