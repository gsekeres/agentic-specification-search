# Verification Report: 116157-V1

## Paper
**Title**: Runs on Money Market Mutual Funds
**Authors**: Lawrence Schmidt, Allan Timmermann, Russell Wermers
**Journal**: American Economic Review (AER), 2016
**Method**: Cross-sectional OLS

## Baseline Groups Found

| Group | Claim | Expected Sign | Baseline Spec IDs |
|-------|-------|---------------|-------------------|
| G1 | Sophisticated (low-ER) investors withdrew significantly more from MMMFs during the Sept 2008 crisis week | - | spec_001 |

The paper has a single primary claim: share classes with a higher fraction of sophisticated investors (proxied by expense ratio <= 35bp) experienced larger outflows during the crisis week following Lehman's bankruptcy. The baseline specification is Table 3, Column 3, with coefficient -35.79 (SE = 10.85, p = 0.001, N = 166).

## Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **79** |
| Baselines | 1 |
| Core tests (non-baseline) | 40 |
| Non-core tests | 38 |

## Category Breakdown

| Category | Count | Core/Non-core |
|----------|-------|---------------|
| baseline | 1 | core |
| core_controls | 12 | core |
| core_sample | 12 | core |
| core_inference | 7 | core |
| core_fe | 3 | core |
| core_funcform | 6 | core |
| noncore_alt_outcome | 6 | non-core |
| noncore_alt_treatment | 10 | non-core |
| noncore_placebo | 4 | non-core |
| noncore_heterogeneity | 13 | non-core |
| noncore_diagnostic | 5 | non-core |

## Classification Details

### Core Tests (41 specs including baseline)

**Controls (12 specs: spec_002 through spec_013)**
- No controls (spec_002): coef = -33.26, p = 0.001
- Drop each of 5 baseline controls one at a time (spec_003 through spec_007): coefficients range from -29.63 to -41.26, all significant at 1%
- Add flow volatility (spec_008), complex PIPERC (spec_009), quadratics (spec_010): all significant at 1%
- Kitchen sink with all extras (spec_011): coef = -19.99, p = 0.10 -- borderline significance with heavy overcontrolling
- Expense ratio only (spec_012): coef = -27.60, p = 0.002
- Fund characteristics only (spec_013): coef = -39.04, p < 0.001

**Sample (12 specs: spec_014 through spec_025)**
- Alternative ER cutoffs for sophistication definition (spec_014-017): 25bp, 30bp, 50bp, 150bp -- all significant
- High-ER share classes only (spec_018): coef = -12.65, p = 0.30 -- expected null, these are not sophisticated
- All institutional with interaction (spec_019): coef = -51.88, p < 0.001
- Trimming/winsorizing outliers (spec_020-022): coefficients -19.3 to -28.1, all highly significant
- Large funds only (spec_023): coef = 0.98, p = 0.95 -- effect disappears for large funds
- Small funds only (spec_024): coef = -29.47, p = 0.017
- Include retail (spec_025): identical to baseline (ER<=35bp already excludes retail)

**Inference (7 specs: spec_037 through spec_043)**
- Classical SE (spec_037): p = 0.031
- HC1/HC2/HC3 robust (spec_038-040): p < 0.001 for all
- Cluster by fundno (spec_041): p = 0.001 (identical to baseline, fundno = fundid here)
- Cluster by complex (spec_042): p = 0.001
- Bootstrap (spec_043): p < 0.001

**Fixed Effects (3 specs: spec_044, spec_045, spec_070)**
- Fund FE with ER*soph interaction (spec_044): within-fund variation, coef = 30.50, p < 0.001 (positive because higher ER = less outflow within fund)
- Complex FE with expense ratio (spec_045): coef = 10.82, p < 0.001
- Fund FE with expense ratio (spec_070): coef = 12.18, p = 0.001

**Functional Form (6 specs: spec_046, spec_047, spec_051-054)**
- WLS by share-class TNA (spec_046): coef = -31.88, p = 0.009
- WLS by fund TNA (spec_047): coef = -24.13, p = 0.044
- Level flows instead of log (spec_051): coef = -19.76, p < 0.001
- Quadratic treatment (spec_052): linear coef = -81.16, p = 0.066
- IHS transform (spec_053): coef = -0.286, p < 0.001 (different scale)
- Unstandardized controls (spec_054): coef = -33.67, p = 0.001

### Non-Core Tests (38 specs)

**Alternative Outcomes (6 specs: spec_026 through spec_031)**
- Cumulative flows measured over different crisis-day windows (Monday through Thursday, full week in levels, and extended to 9/26). These show the effect building over the crisis week from -6.0pp on Monday to -38.1pp by 9/26. All significant. Different dependent variables measuring flow dynamics rather than the main crisis-week claim.

**Alternative Treatments (10 specs: spec_032-036, spec_069, spec_071-074)**
- Using expense ratio directly (continuous, dummies, raw) or different soph_frac cutoffs (25bp, 30bp, 50bp, 150bp) with interaction terms on the full institutional sample. These test whether the sophistication effect appears under different treatment operationalizations. While related to the core claim, they test a fundamentally different treatment variable.
- Note: spec_069 duplicates spec_032 (both are std_expr on all inst with no FE). spec_071 duplicates spec_035. spec_073 duplicates spec_036.

**Placebo Tests (4 specs: spec_056 through spec_059)**
- Pre-crisis flows (spec_056): coef = 0.18, p = 0.96 -- no pre-trend
- Retail share classes (spec_057): coef = 1.35, p = 0.13 -- no retail effect
- Yield as treatment (spec_058): coef = -2.56, p = 0.69 -- yield does not predict outflows
- WAM as treatment (spec_059): coef = 9.07, p = 0.10 -- borderline, weakly positive

**Heterogeneity (13 specs: spec_048-050, spec_055, spec_060-068)**
- Quantile regressions at 25th/50th/75th percentile (spec_048-050): effect concentrated in left tail (-42.0 at p25, -17.9 at median, -0.87 at p75)
- Interaction with fund size (spec_055, spec_060 -- duplicates): main coef flips to +49.0 but interaction is -14.1, showing fund-size-dependent effect
- Interaction with yield (spec_061): main coef = -132.0, p = 0.47 (large SE); interaction insignificant
- Interaction with liquidity (spec_062): main coef = -33.7, p = 0.011; interaction insignificant
- Interaction with fund business (spec_063): main coef = -37.2, p = 0.060; interaction insignificant
- Interaction with complex PIPERC (spec_068): main coef = -35.8, p = 0.034; interaction insignificant
- Subsamples by liquidity (spec_064-065): high liq coef = -40.0 (p=0.076), low liq coef = -21.0 (p=0.082)
- Subsamples by yield (spec_066-067): high yield coef = -30.1 (p=0.072), low yield coef = -49.2 (p=0.011)

**Diagnostic/Auxiliary (5 specs: spec_075 through spec_079)**
- Table 1 first-stage regressions showing that expense ratio predicts minimum investment (spec_075-077): validates the claim that low ER proxies for institutional/sophisticated investors
- Table 4 fund-FE specifications for different flow windows (spec_078-079): within-fund results on Monday and Wednesday flows

## Top 5 Issues

### 1. spec_025 (incl_retail) is identical to spec_001 (baseline)
**Issue**: The coefficient, SE, and all statistics for spec_025 are exactly identical to spec_001. The ER<=35bp sample restriction already excludes retail share classes, so "including retail" changes nothing.
**Recommendation**: Flag as redundant/duplicate. Remove or note that ER<=35bp filter already restricts to institutional classes.

### 2. spec_060 duplicates spec_055; spec_069 duplicates spec_032; spec_071 duplicates spec_035; spec_073 duplicates spec_036
**Issue**: Multiple specifications appear twice under different spec_tree_paths. spec_060 (soph_x_fund_size) has identical coefficients to spec_055 (interact_soph_tna). spec_069 (fe/none_expr) is identical to spec_032 (treatment/std_expr). spec_071 (cutoff_25bp) is identical to spec_035 (soph_frac25_interact). spec_073 (cutoff_50bp) is identical to spec_036 (soph_frac50_interact).
**Recommendation**: Remove duplicates. The effective unique specification count is 75, not 79.

### 3. spec_023 (large funds) shows complete null
**Issue**: For large funds (TNA >= median), the coefficient is essentially zero (0.98, p = 0.95). This is a notable sensitivity: the entire sophistication-outflow effect is concentrated among smaller funds. This is an important qualification to the main finding.
**Recommendation**: This should be highlighted prominently, as it suggests the result is driven by a specific subsample.

### 4. spec_011 (kitchen sink) pushes p-value to borderline significance
**Issue**: With all available controls (quadratics, flow volatility, complex PIPERC), the coefficient drops to -19.99 with p = 0.10. While this is expected with heavy overcontrolling (loss of precision), it signals some sensitivity to the control specification.
**Recommendation**: Note as a sensitivity flag. The magnitude drops by ~44% and significance disappears.

### 5. spec_041 (cluster_fundno) appears identical to baseline (cluster_fundid)
**Issue**: spec_041 clusters by fundno and produces exactly the same coefficients and SEs as spec_001 (which clusters by fundid). This suggests fundno and fundid are the same variable in this dataset.
**Recommendation**: Remove as it adds no new information about inference robustness.

## Overall Assessment

The main result is robust across most reasonable specification choices:
- **Controls**: Significant across 11 of 12 control variations (92%), with the exception of the heavily overcontrolled kitchen-sink model (p = 0.10)
- **Inference**: Significant under all 7 SE methods (100%)
- **Functional form**: Significant across 5 of 6 functional form variations (83%), with the quadratic treatment borderline (p = 0.066)
- **Sample**: Significant in 10 of 12 sample variations (83%), with notable exceptions for high-ER share classes (expected null) and large funds (unexpected null)

The key qualification is the **fund-size sensitivity**: the effect is entirely driven by smaller funds. The placebo tests pass cleanly: no pre-crisis effect, no retail effect, and no effect of irrelevant treatments.

The specification search includes several duplicate specifications (4 pairs), reducing the effective unique count from 79 to 75. The effective core test count (excluding duplicates) is 39, and the non-core count is 35.
