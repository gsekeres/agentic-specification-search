# Specification Surface Review: 128143-V1

**Paper**: "Yellow Vests, Pessimistic Beliefs, and Carbon Tax Aversion" (Douenne & Fabre, AEJ: EP)
**Design**: Randomized experiment (survey RCT) with manual 2SLS/IV
**Review date**: 2026-02-25

---

## Summary of Baseline Groups

### G1: Self-Interest Beliefs and Tax Acceptance (Tables 5.1-5.2)
- **Claim**: Beliefs about personal gains/losses from the carbon tax causally affect tax acceptance.
- **Baseline specs**: Table 5.2 Col 1 (IV on p10-p60 subsample, 53 pp) and Col 2 (IV on full sample, 46 pp).
- **Design**: Manual 2SLS using `traite_cible * traite_cible_conjoint` (random dividend eligibility assignment) as instrument for `non_perdant` (belief of not losing). Survey weights. Default OLS SEs from `lm()`.

### G2: Environmental Effectiveness Beliefs and Tax Approval (Tables 5.3-5.4)
- **Claim**: Beliefs about environmental effectiveness of carbon taxation causally affect tax approval.
- **Baseline specs**: Table 5.4 Col 1 (approval Yes ~ effectiveness Yes, 42 pp) and Col 3 (acceptance not-No ~ effectiveness Yes, 50 pp).
- **Design**: Manual 2SLS using `apres_modifs + info_CC` (random information treatments) as instruments. Sargan overid test does not reject (p=0.93).

---

## Changes Made

### 1. OLS specifications moved from rc/* to explore/* (IMPORTANT)

Both G1 and G2 included `rc/form/ols_instead_of_iv` as a core robustness check. However, dropping IV and running OLS fundamentally changes the identification strategy and the estimand (from LATE to OLS association). This is not an estimand-preserving robustness check; it is an alternative estimand. Moved to `explore/alternative_estimands/ols_instead_of_iv`.

### 2. Feedback subsample moved from rc/* to explore/* (G1)

`rc/sample/feedback_subsample` (Table 5.2 Col 4) uses a completely different sub-experiment (feedback variant with `simule_gagnant` as instrument instead of `traite_cible`), a different outcome variable (`taxe_feedback_approbation` instead of `taxe_cible_approbation`), and a different sample restriction (`variante_taxe_info=='f' & |simule_gain|<50`). This is not an RC variant of the same claim -- it is a separate experiment testing a related but distinct hypothesis. Moved to `explore/alternative_estimands/feedback_subsample`.

### 3. Endogenous variable redefinition moved from rc/* to explore/* (G2)

`rc/form/endogenous_not_no` changes the endogenous variable from `taxe_efficace=='Oui'` (actively believes effective) to `taxe_efficace!='Non'` (does not disbelieve). This changes the belief being instrumented and therefore the claim object. Moved to `explore/variable_definitions/endogenous_not_no`.

### 4. Control counts corrected

The original surface listed `n_controls: 20` for G1 baseline specs and `n_controls: 15` for G2 baseline specs. Verification against the code shows:

- **G1**: `variables_reg_self_interest` = 6 named + 17 demographics + 6 piecewise income spline terms = 29, plus 3 second-stage extras (cible, I(taxe_approbation=='NSP'), tax_acceptance) = 32 total for Col 1, 33 for Col 2 (adds `single` in second stage).
- **G2**: `variables_reg_ee` = 8 named + 17 demographics = 25 total.

Updated `n_controls` and `controls_count_min/max` accordingly.

### 5. Canonical inference corrected from HC1 to OLS default

The paper uses default `lm()` standard errors in R, which are classical (homoskedastic) OLS SEs, not HC1. Changed canonical from `infer/se/hc/hc1` to `infer/se/ols_default` and added HC1 as an inference variant.

### 6. Added LOO control blocks for G1

Added three LOO blocks that were missing from G1: `rc/controls/loo/hausse_depenses_par_uc`, `rc/controls/loo/simule_gain`, `rc/controls/loo/prog_na`. These are substantively meaningful controls (expense increases, estimated gains, progressivity belief) that can be dropped one at a time.

### 7. Removed redundant design spec ID for G1

`design/randomized_experiment/estimator/with_covariates` was listed in `design_spec_ids` but this is essentially the baseline estimator itself (RCT-IV with covariates). There is no meaningful alternative design implementation. Set to empty.

---

## Key Constraints and Linkage Rules

1. **Bundled IV estimator**: The same control set must appear in both first and second stage. LOO drops are joint across stages.
2. **Instrument sets are experimentally fixed**: G1 uses `traite_cible * traite_cible_conjoint` (cannot be varied). G2 uses `apres_modifs + info_CC` (cannot be varied).
3. **Manual 2SLS**: Paper uses fitted values from first-stage LPM plugged into second-stage LPM. Standard errors do NOT correct for generated-regressor uncertainty. This is a known limitation acknowledged by the authors.
4. **Survey weights**: All regressions use survey weights (`s$weight`). The unweighted variant is a robustness check.
5. **Demographic variables**: `variables_demo` contains 21 variables (defined in preparation.R line 1667). The regression control sets remove 4 of these (revenu, rev_tot, age, age_65_plus), leaving 17 demographics.

---

## Budget and Sampling Assessment

### G1 Budget (60 max):
- 2 baselines + 1 outcome form variant + 7 LOO control blocks + 1 unweighted = **11 core specs per baseline** = 22 total.
- Well within 60 budget.

### G2 Budget (50 max):
- 2 baselines + 1 outcome form variant + 4 LOO control blocks + 1 unweighted = **8 core specs per baseline** = 16 total.
- Well within 50 budget.

---

## What's Missing

1. **Proper 2SLS SE correction**: The manual 2SLS approach produces invalid standard errors (no generated-regressor correction). The paper runs `ivreg` separately for F-statistics but not for main SEs (because `ivreg` does not support survey weights). An inference variant using proper 2SLS SEs (without weights) could be added. This is partially captured by the `ivreg` diagnostics but not as an inference variant.
2. **Alternative piecewise income knots**: The paper's Appendix includes specifications with different income spline knots (30, 40, 50, 60 instead of 20, 70). These are captured in the code (sio2-sio5 in papier.R) but not in the surface. Could be added as `rc/preprocess/income_knots/*`.
3. **LIML estimator**: The code computes LIML estimates (line 414, 899). These could be `design/instrumental_variables/estimator/liml` variants but are not included.

---

## Approved to Run

**Status**: APPROVED with notes.

The surface is now cleaner after moving estimand-changing specifications out of the core RC universe. The main concern is the manual 2SLS SE issue, which is a paper-level limitation rather than a surface design problem. The control count corrections ensure the constraints envelope is accurate. The budgets are adequate for the specified RC axes.

**Important caveat for runners**: This paper's R code loads pre-computed `.RData` files and defines control vectors dynamically. The runner will need to reconstruct the control variable lists from `preparation.R` output and `papier.R` definitions. The `variables_demo` vector (21 elements) is defined in `preparation.R` line 1667.
