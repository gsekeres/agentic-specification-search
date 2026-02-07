# Verification Report: 173341-V1

**Paper**: "Vulnerability and Clientelism" by Bobonis, Gertler, Gonzalez-Navarro, and Nichter (2022), *American Economic Review*

**Verified**: 2026-02-04

**Verifier**: verification_agent

---

## 1. Baseline Groups

### G1: Cisterns Treatment Effect on Private Requests
- **Claim**: The cisterns treatment (which reduces household vulnerability) decreases private requests to politicians, indicating that vulnerability sustains clientelism.
- **Baseline spec_id**: `baseline`
- **Outcome**: `ask_private_stacked` (whether individual requested any private good from a politician, stacked across 2012 and 2013)
- **Treatment**: `treatment` (cisterns treatment assignment, binary RCT indicator)
- **Expected sign**: Negative
- **Baseline coefficient**: -0.0296 (SE = 0.0125, p = 0.018)
- **Source**: Table 3, Column 3 of the paper

### G2: Rainfall Effect on Private Requests
- **Claim**: Rainfall shocks (which increase vulnerability when negative) increase private requests to politicians.
- **Baseline spec_id**: `baseline_rainfall`
- **Outcome**: `ask_private_stacked`
- **Treatment**: `rainfall_std_stacked` (standardized rainfall, where positive = more rain = less vulnerability)
- **Expected sign**: Negative (more rain reduces vulnerability and thus reduces requests)
- **Baseline coefficient**: -0.0233 (SE = 0.0098, p = 0.018)
- **Source**: Same regression as G1 (Table 3, Column 3), different coefficient reported

---

## 2. Counts

| Category | Count |
|----------|-------|
| Total specifications | 92 |
| Baselines | 2 |
| Core tests (non-baseline) | 57 |
| Non-core specifications | 32 |
| Invalid | 1 |
| Unclear | 0 |

### Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_sample | 42 | Leave-one-out municipalities (38), year subsamples (2), winsorization (2) |
| core_controls | 5 | 2 baselines + drop rainfall, rainfall-only, add frequent interactor |
| core_funcform | 7 | Outcome variants (ask_nowater, askrec for both G1/G2), winsorized rainfall, quadratic rainfall |
| core_fe | 2 | Pooled OLS, municipality-only FE |
| core_inference | 2 | Robust HC1, municipality-level clustering |
| core_method | 1 | Probit marginal effects |
| noncore_alt_outcome | 18 | Household vulnerability outcomes (Table 2), individual political outcomes (Table 1), household outcomes with controls |
| noncore_heterogeneity | 11 | Interactions (frequent interactor, association member, high rainfall, year-specific), subsamples by demographic group |
| noncore_placebo | 3 | Public goods requests as placebo (2 duplicated regressions + rainfall on public goods) |
| invalid | 1 | Exact duplicate of baseline |

---

## 3. Top 5 Most Suspicious Rows

### 3.1. `robust/outcome/ask_private_stacked` (INVALID -- exact duplicate)
- **Issue**: This spec has the same outcome (`ask_private_stacked`), treatment (`treatment`), FE, sample, clustering, and coefficient (-0.0296) as the `baseline` spec. It is a verbatim duplicate.
- **Action**: Marked as `invalid`. This inflates the apparent number of distinct specifications.

### 3.2. `robust/placebo/public_goods` vs `robust/outcome/ask_public_stacked` (duplicate regression)
- **Issue**: Both run `ask_public_stacked ~ treatment + rainfall_std_stacked + year2012 + C(mun_id)` clustered by `b_clusters`. They produce identical coefficients (-0.00464, p=0.347). The first is labeled as a placebo test, the second as an alternative outcome. They are the same regression labeled differently.
- **Action**: Both classified as non-core (placebo), but this is a definitional issue that inflates the count.

### 3.3. `robust/control/rainfall_only` (treatment variable mismatch)
- **Issue**: This spec reports the coefficient on `rainfall_std_stacked` from a regression that omits `treatment`. It is labeled under `robustness/leave_one_out.md` but it is testing the G2 claim (rainfall effect), not G1 (treatment effect). The labeling under "control variations" is misleading; it is a G2 robustness check.
- **Action**: Classified as core test for G2. No coefficient extraction error, but the spec_tree_path labeling is misleading.

### 3.4. Household outcomes (rows 61-68, 88-90) treat mechanism outcomes as alternatives
- **Issue**: The household vulnerability outcomes (d_Happiness, d_Health, d_Child_Food_Security, d_Overall_index) from Table 2 are mechanism outcomes: they show that the treatment reduces vulnerability. The paper's main claim (Table 3) is about the downstream effect on clientelism. These are supporting evidence for the causal channel, not robustness checks of the main claim.
- **Action**: Classified as `noncore_alt_outcome`. If the research question were "does the cisterns treatment reduce vulnerability?", these would be baselines.

### 3.5. `robust/estimation/probit` (missing FE)
- **Issue**: The probit specification drops municipality FE (set to "None (probit)") and has no clustering. The baseline includes municipality FE. This is a meaningful specification change but the missing FE makes it less comparable. The pseudo R-squared (0.041) is much lower than the baseline R-squared (0.073), suggesting the FE absorb meaningful variation.
- **Action**: Classified as `core_method` with slightly lower confidence (0.85). The probit should ideally include FE to be fully comparable.

---

## 4. Recommendations for the Spec-Search Script

1. **Remove exact duplicate**: `robust/outcome/ask_private_stacked` is identical to `baseline`. The alternative-outcomes loop should skip the baseline outcome variable to avoid double-counting.

2. **Deduplicate placebo specifications**: `robust/placebo/public_goods` and `robust/outcome/ask_public_stacked` run the same regression. The code should either (a) not include `ask_public_stacked` in the alternative outcomes loop since it is a placebo, or (b) not separately run it as a placebo test.

3. **Separate G1 and G2 more clearly**: The script intermixes treatment effect robustness (G1) and rainfall effect robustness (G2). Consider separate sections or explicit labeling in the spec_id.

4. **Reduce leave-one-out municipality dominance**: The 38 leave-one-out-municipality specs dominate the specification count (41% of all specs). While informative for jackknife-style sensitivity, they are highly correlated. Consider (a) reporting summary statistics of these collectively, or (b) reducing their weight in overall robustness assessments.

5. **Add FE to probit**: The probit specification should attempt to include municipality fixed effects (or at minimum municipality dummies) to be comparable to the baseline OLS specification.

6. **Consider adding more substantively diverse robustness checks**: The current search is heavy on sample restrictions but light on alternative model specifications. Missing possibilities include: (a) logit models with FE, (b) different definitions of the treatment variable (e.g., intention-to-treat vs actual cistern receipt), (c) controlling for pre-treatment covariates, (d) wild cluster bootstrap p-values (which the paper itself reports in Table 4).
