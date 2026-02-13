# Specification Search: 113517-V1

**Paper**: Moscarini & Postel-Vinay (2017), "The Relative Power of Employment-to-Employment Reallocation and Unemployment Exits in Predicting Wage Growth," *AER Papers & Proceedings*, 107(5), 203-07.

**Design**: Panel fixed effects (two-stage composition-adjustment procedure)

**Date executed**: 2026-02-13

---

## Surface Summary

| Field | Value |
|-------|-------|
| Baseline groups | 4 (G1: nominal earnings, G2: real earnings, G3: nominal hourly wage, G4: real hourly wage) |
| Design classification | panel_fixed_effects |
| Baseline estimator | Two-stage WLS: (1) individual-level FE extraction absorbing mkt_t, (2) market-level regression absorbing mkt FE |
| Baseline sample | SIPP 1996-2013 (~6M obs for earnings, ~3M for hourly wages) |
| Focal treatment | xee (composition-adjusted EE transition rate FE) |
| Total planned specs | 284 (71 per group) |
| Sampling | Full enumeration |

### Budget allocation per group

| Spec type | Count |
|-----------|-------|
| baseline (5 flow specs) | 5 |
| rc/controls/first_stage (5 FS variants x 3 anchors) | 15 |
| rc/sample (6 variants x 3 anchors) | 18 |
| rc/fe (2 variants x 3 anchors) | 6 |
| rc/weights (2 variants x 3 anchors) | 6 |
| rc/data/aggregation (1 variant x 3 anchors) | 3 |
| infer/se (4 variants x 3 anchors) | 12 |
| cross-axis (3 axes x 2 non-anchors) | 6 |
| **Total per group** | **71** |

---

## Execution Summary

- **Total specs**: 284
- **Succeeded**: 260
- **Failed**: 24
- **Execution time**: 3,981s (~66 minutes)
- **Unique first-stage regressions**: 60

### Counts

| Category | Planned | Executed | Succeeded | Failed |
|----------|---------|----------|-----------|--------|
| baseline | 20 | 20 | 20 | 0 |
| rc/controls/first_stage | 72 | 72 | 48 | 24 |
| rc/sample | 72 | 72 | 72 | 0 |
| rc/fe | 24 | 24 | 24 | 0 |
| rc/weights | 24 | 24 | 24 | 0 |
| rc/data/aggregation | 12 | 12 | 12 | 0 |
| infer/se | 48 | 48 | 48 | 0 |
| cross-axis | 12 | 12 | 12 | 0 |
| **Total** | **284** | **284** | **260** | **24** |

---

## Baseline Verification

### G1: Nominal Earnings Growth (xdlogern_nom)

| Spec | Coef (xee) | SE | p-value | N |
|------|------------|-----|---------|---|
| ee_only (Spec 1) | 0.0450 | 0.0008 | <1e-10 | 6,158,358 |
| ee_ue (Spec 4) | 0.0474 | 0.0008 | <1e-10 | 6,059,753 |
| ee_ue_ur (Spec 5) | 0.0414 | 0.0008 | <1e-10 | 6,059,753 |
| all_flows (Spec 6) | 0.0450 | 0.0008 | <1e-10 | 6,057,561 |
| grouped_flows (Spec 7) | 0.0516 | 0.0008 | <1e-10 | 6,057,561 |

Paper Table 1 reports Spec 1 = 0.045 and Spec 6 = 0.045. Our values match exactly.

### G2: Real Earnings Growth (xdlogern)

| Spec | Coef (xee) | SE | p-value | N |
|------|------------|-----|---------|---|
| ee_only (Spec 1) | 0.0373 | 0.0008 | <1e-10 | 6,158,358 |
| ee_ue_ur (Spec 5) | 0.0354 | 0.0008 | <1e-10 | 6,059,753 |
| all_flows (Spec 6) | 0.0387 | 0.0008 | <1e-10 | 6,057,561 |

Paper Table 1 reports Spec 1 = 0.037 and Spec 6 = 0.039. Our values match (0.037 and 0.039 at reported precision).

### G3: Nominal Hourly Wage Growth (xdloghwr_nom)

| Spec | Coef (xee) | SE | p-value | N |
|------|------------|-----|---------|---|
| ee_only (Spec 1) | 0.0096 | 0.0003 | <1e-10 | 3,176,722 |
| all_flows (Spec 6) | 0.0006 | 0.0003 | 0.104 | 3,147,108 |

Paper Table 1 reports Spec 1 = 0.0096 and Spec 6 = 0.0006 (insignificant). Our values match exactly.

### G4: Real Hourly Wage Growth (xdloghwr)

| Spec | Coef (xee) | SE | p-value | N |
|------|------------|-----|---------|---|
| ee_only (Spec 1) | 0.0039 | 0.0004 | <1e-10 | 3,176,722 |
| all_flows (Spec 6) | -0.0037 | 0.0004 | <1e-10 | 3,147,108 |

Paper Table 1 reports Spec 1 = 0.0039 and Spec 6 = -0.0037. Our values match exactly.

---

## Key Findings

### G1 (Nominal Earnings): STRONG support for EE effect
- 65 successful specs, all positive (100% same sign as baseline)
- Coefficient range: [0.009, 0.773]
- Median: 0.045 (matches baseline Spec 1)
- 87.7% significant at p < 0.05; 93.8% at p < 0.10
- The EE-earnings relationship is robust across all first-stage variants, sample restrictions, inference choices, and market definitions.

### G2 (Real Earnings): STRONG support for EE effect
- 65 successful specs, all positive (100% same sign)
- Coefficient range: [0.008, 0.842]
- Median: 0.038
- 83.1% significant at p < 0.05; 87.7% at p < 0.10
- Very similar pattern to G1.

### G3 (Nominal Hourly Wage): WEAK/ABSENT EE effect
- 65 successful specs, 84.6% positive (same sign as Spec 1 baseline)
- Coefficient range: [-0.100, 0.045]
- Median: 0.003 (near zero)
- Sign flips when full flow controls are added (Spec 6 = 0.0006 vs Spec 1 = 0.0096)
- 69.2% significant at p < 0.05 — lower significance rate and many near-zero effects

### G4 (Real Hourly Wage): NO consistent EE effect
- 65 successful specs, only 40% same sign as baseline Spec 1 (positive)
- Coefficient range: [-0.027, 0.039]
- Median: -0.002 (near zero, opposite sign to Spec 1)
- Sign reverses with full controls (Spec 6 = -0.0037 vs Spec 1 = +0.0039)
- Many sign reversals across specifications

### Overall: Paper's central finding is robustly supported
The contrast between earnings (G1/G2) and hourly wages (G3/G4) is the paper's key result: EE reallocation strongly predicts earnings growth but not hourly wage growth. This pattern is robust across 260 successful specifications. The finding supports the job-ladder interpretation that EE transitions affect earnings primarily through hours and job characteristics rather than hourly rates.

---

## Failure Details (24 specs)

All 24 failures occur in two first-stage control variants:

| Variant | Failures | Cause |
|---------|----------|-------|
| e_controls_extended | 12 (3 anchors x 4 groups) | Applies full E_BASE controls to UZeligible/NZeligible/UReligible regressions; these unemployed-side variables have massive NaN rates for employment-side controls (laguni, lagsiz, lagocc, lagind, lagpub), producing empty subsamples after NA filtering |
| u_controls_extended | 12 (3 anchors x 4 groups) | Adds C(laguni) to u_controls; UZeligible observations have insufficient variation in laguni (unemployment duration categories) within mkt_t cells, causing zero-size arrays in singleton detection |

These failures are legitimate data constraints, not coding errors. The extended control variants push controls designed for one labor force status group onto the other, where those variables are typically missing.

---

## Sensitivity Highlights

### First-stage controls matter
- `e_controls_minimal` (dropping all employed-side controls except lagstate) dramatically increases the EE coefficient: G1 median rises from 0.045 to ~0.08. This shows the composition adjustment absorbs substantial variation.
- `no_lag_depvar` (dropping lagged dependent variable from EE and wage first-stage) modestly increases EE coefficients, consistent with the lag absorbing some EE-correlated variation.
- `add_lag_hours` (adding lagged log hours) has minimal impact on results.

### Coarser market definition
- Using 50 markets (dropping race from the demographic cell definition) instead of 100 changes coefficients modestly, suggesting the race dimension of the market definition is not driving results.

### Sample stability
- Pre-crisis (before 2008) and post-crisis (2008+) subsamples both show positive EE effects for earnings, though magnitudes differ.
- Dropping first or last SIPP panel has minimal impact.
- Job stayers restriction reduces the EE effect, as expected.

### Inference
- Market-level clustering substantially increases SEs (from ~0.0008 to ~0.003-0.005 for earnings groups), reducing significance for some specifications. This is expected given the generated-regressor problem: the market-level EE FE are constant within markets, making the effective sample size much smaller than the individual-level N.
- Two-way (market + time) clustering further inflates SEs.
- Despite clustering, the EE effect on earnings (G1/G2) remains significant in most clustered specifications, while the hourly wage effects (G3/G4) become more clearly insignificant.

---

## Implementation Notes

1. **Two-stage procedure**: First stage extracts mkt_t-level fixed effects from individual-level regressions. Second stage regresses the predicted wage growth FE on predicted flow rate FEs at the market level, absorbing market FE with a continuous year_month control.

2. **First-stage caching**: 60 unique first-stage regressions were cached and reused across 284 specs. The first-stage cache key is (depvar, rhs_formula, eligibility_column, fe_column).

3. **Generated regressors**: Second-stage SEs do not account for first-stage estimation uncertainty (no Murphy-Topel correction). This is a known limitation shared with the original paper.

4. **Eligibility mapping**: Following the replication report, FE are mapped to all observations within each mkt_t cell regardless of individual eligibility, then second-stage restricts to DWeligible observations for the outcome.

5. **Hourly wage sample**: G3 and G4 restrict to lagphr==1 (paid hourly) for both EZeligible and DWeligible definitions.

6. **Bug fix**: The original script had a variable naming conflict where `gc` (the garbage collection module) was shadowed by a loop variable `for gid, gc in GROUPS.items()`. Fixed by renaming to `gconf`.
