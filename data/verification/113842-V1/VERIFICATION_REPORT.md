# Verification Report: 113842-V1

## Paper Information
- **Title**: Teaching the Tax Code: Earnings Responses to an Experiment with EITC Recipients
- **Authors**: Raj Chetty and Emmanuel Saez
- **Journal**: American Economic Journal: Applied Economics, 2013, 5(1): 1-31
- **Total Specifications**: 130 (24 KS tests + 106 OLS regressions)

## Data Constraints
The underlying microdata (ts0708_total.dta) is **confidential and proprietary** H&R Block data. No new specifications could be run. All 130 specifications were extracted from the permutation results file (coeff_final.dta), which stores original coefficients and p-values for each specification alongside 2000 permutation draws. Standard errors and sample sizes are unavailable for any specification.

## Baseline Groups

### G1: Full-Sample Treatment Effect on EITC Amount (d_fdeic)
- **Claim**: The EITC information treatment has no statistically significant average effect on EITC amounts in the full sample.
- **Baseline spec**: `reg_12`
- **Expected sign**: Positive (information should increase EITC uptake)
- **Baseline coefficient**: $24.02 (p = 0.099 standard, p = 0.100 permutation)
- **Outcome**: `d_fdeic` (change in federal EITC amount, 2007 to 2008)
- **Treatment**: `treat` (binary, randomized information provision)
- **Table 3, Column 1, Row 1**

### G2: Complier Treatment Effect on EITC Amount
- **Claim**: Among clients of "complying" tax preparers (goodtax==1), the information treatment significantly increased EITC amounts by $58-67.
- **Baseline specs**: `reg_13` (no controls), `reg_18` (with controls)
- **Expected sign**: Positive
- **Baseline coefficient**: $67.26 without controls (p = 0.003); $58.05 with controls (p = 0.007)
- **Outcome**: `d_fdeic`
- **Treatment**: `treat`
- **Table 3, Columns 1-2, goodtax==1 subsample**

This is the paper's central positive finding. "Complying" tax preparers are those whose clients show above-median rates of income in the EITC-maximizing range. The treatment is effective only through these preparers, suggesting the information channel requires preparer cooperation.

### G3: Differential Treatment Effect (Interaction)
- **Claim**: The differential treatment effect between complying and non-complying tax preparers on EITC amounts is large ($90-95) and statistically significant.
- **Baseline specs**: `reg_15` (no controls), `reg_20` (with controls)
- **Expected sign**: Positive
- **Baseline coefficient**: $95.21 without controls (p = 0.009); $90.33 with controls (p = 0.007)
- **Outcome**: `d_fdeic`
- **Treatment**: `good_treat` (goodtax x treat interaction)
- **Table 3, Columns 1-2, interaction specification**

**Note**: The paper presents this as its key finding: the null average effect is explained by large positive effects among compliers offset by negative (insignificant) effects among non-compliers.

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baselines)** | **55** | |
| core_baseline | 1 | G1 baseline: full sample d_fdeic treatment effect (reg_12) |
| core_controls | 4 | Adding controls to baseline specs (reg_17, reg_18, reg_19, reg_20) |
| core_sample | 22 | Complier/non-complier subsamples + EITC slope region subsamples (phaseout/plateau/phasein x complier/non-complier) |
| core_alt_outcome | 12 | Alternative outcomes in same framework: midinc08, poor08, highinc08, d_inc, d_inc_trim -- testing treatment effect on different measures of income position |
| core_interaction | 16 | goodtax x treat interaction specifications across outcomes and subsamples |
| **Non-core tests** | **75** | |
| noncore_balance | 23 | 11 regression balance tests (Table 1) + 12 KS balance tests on pre-treatment covariates |
| noncore_alt_method | 12 | KS distributional tests on post-treatment outcomes -- nonparametric alternative to regression |
| noncore_unidentified | 40 | Permutation specs (reg_67 to reg_106) with unknown outcome variables |
| **Total** | **130** | |

## Detailed Classification Notes

### Core Tests (55 specs including 5 baselines across 3 groups)

**Baselines (5 specs)**: The paper has an unusual baseline structure because the primary result is about treatment heterogeneity rather than a simple average treatment effect. The full-sample null (G1: reg_12) is itself a finding, while the complier positive effect (G2: reg_13, reg_18) and the interaction (G3: reg_15, reg_20) are the key claims. All five baseline specs correspond to Table 3 results, which is the paper's main results table.

**Control variations (4 non-baseline core specs)**: Adding controls (inc07, inc07^2, wages07, married, deps) to the full-sample and non-complier specifications. The complier and interaction versions with controls are already classified as baselines. Key finding: controls reduce the full-sample coefficient from $24 to $17 (still insignificant), while the complier coefficient drops from $67 to $58 (still significant at p=0.007).

**Sample restrictions (22 specs)**: The largest core category, reflecting the paper's systematic exploration of heterogeneity:
- Complier vs non-complier splits for all outcomes (d_fdeic, midinc08, poor08, highinc08, d_inc, d_inc_trim)
- EITC slope region subsamples: phaseout (slope=-1), plateau (slope=0), and phasein (slope=1) for midinc08
- Cross-cuts: slope region x complier status

**Alternative outcomes (12 specs)**: The paper systematically tests treatment effects on multiple income indicators:
- `midinc08`: indicator for income in EITC plateau range (Table 3 Cols 3-4)
- `poor08`: indicator for low income (Table 3 Col 5)
- `highinc08`: indicator for high income (Table 3 Col 6)
- `d_inc`: change in total income (Table 4 Col 1)
- `d_inc_trim`: trimmed income change (Table 4 Col 2)

These are classified as core because they test the same treatment on closely related outcomes that together characterize the earnings response distribution.

**Interaction specifications (16 non-baseline core specs)**: The goodtax x treat interaction across all outcomes and slope subsamples. These are central to the paper's argument that treatment effects are heterogeneous by tax preparer compliance.

### Non-Core Tests (75 specs)

**Balance tests (23 specs)**: 11 OLS regressions (reg_1 to reg_11) testing whether pre-treatment covariates are balanced across treatment arms, plus 12 KS distributional balance tests (kst_13 to kst_24) on pre-treatment income and wages. These test the validity of randomization, not the treatment effect itself. Notable: reg_10 (return08 balance) is marginally significant (p=0.041), suggesting slight differential attrition.

**Alternative method (12 specs)**: KS distributional tests (kst_1 to kst_12) on post-treatment outcomes (d_fdeic, d_eic_wage, inc08, wages08). These test whether the entire outcome distribution shifts, not just the mean. They address a related but distinct question from the regression-based treatment effects and use a fundamentally different statistical method. The KS test on d_fdeic among compliers (kst_2, p=0.0035) corroborates the regression result.

**Unidentified specifications (40 specs)**: Specifications reg_67 through reg_106 come from unlabeled columns in the permutation file (coeff_final.dta). Their outcome variables, control sets, and subsamples cannot be determined from available data. Based on coefficient magnitudes:
- Some (reg_67-76, 87-96) have dollar-scale coefficients ($8-$215), likely testing treatment effects on dollar-denominated outcomes
- Others (reg_77-86, 97-106) have small coefficients (0.001-0.089), likely testing binary/proportion outcomes
- Several clusters (reg_90/91/95/96, all ~$103, p~0.002) likely test the same claim under minor variations

These are classified as non-core with confidence 0.50 because we cannot determine their relationship to the baseline claims.

## Robustness Assessment

### G1 (Full-sample average treatment effect): Robust null
The full-sample treatment effect on d_fdeic is consistently null:
- Without controls: $24 (p = 0.099)
- With controls: $17 (p = 0.195)
- Alternative outcomes (midinc08, highinc08): also null (p = 0.437-0.841)
- One exception: poor08 is marginally significant (p = 0.074)

The null is very stable across specifications. Both standard and permutation p-values agree closely.

### G2 (Complier treatment effect): Robust positive finding
The complier treatment effect is significant across multiple outcomes and specifications:
- d_fdeic: $67 without controls (p = 0.003), $58 with controls (p = 0.007)
- midinc08: 0.026-0.029 (p = 0.003-0.009)
- poor08: -0.015 (p = 0.003)
- By EITC slope region: significant in phaseout (p = 0.021) and phasein (p = 0.009), not in plateau (p = 0.177)
- KS test corroborates: p = 0.0035

Key sensitivity: The income change outcomes (d_inc, d_inc_trim) show negative but insignificant effects among compliers (p = 0.149-0.157), which is puzzling given the EITC increase. The paper acknowledges this may reflect self-employment income reporting rather than real labor supply changes.

### G3 (Interaction/differential effect): Robust
The interaction is significant across all main outcomes:
- d_fdeic: $90-95 (p = 0.007-0.009)
- midinc08: 0.047-0.060 (p = 0.004-0.008)
- poor08: -0.019 (p = 0.020)
- highinc08: -0.028 (p = 0.038)
- d_inc: -$420 (p = 0.017)
- d_inc_trim: -$428 (p = 0.011)
- By EITC slope: strongest in phaseout (p = 0.001), moderate in phasein (p = 0.052), weak in plateau (p = 0.092)

The interaction effect is robust, though it is strongest in the phaseout region of the EITC schedule, which is where misreporting incentives are largest.

### Cross-cutting observations
1. **Permutation vs standard inference agreement**: p-values from 2000-draw permutation tests closely match asymptotic standard errors in nearly all specifications, validating the cluster-robust inference.
2. **Self-employment channel concern**: The pattern of EITC increases without corresponding income increases raises questions about whether responses reflect real earnings changes or reporting behavior.
3. **Balance test flag**: The marginally significant differential attrition (return08, p=0.041) warrants attention, though it is not dramatically significant.
4. **Data limitation**: No standard errors or confidence intervals are available for any specification, limiting the precision of robustness assessments. All inference is based on p-values only.
