# Verification Report: 113887-V2

## Paper Information
- **Title**: Testing Paternalism: Cash versus In-Kind Transfers
- **Authors**: Jesse M. Cunha
- **Journal**: AEJ: Applied Economics
- **Total Specifications**: 77

## Baseline Groups

### G1: In-Kind Treatment Effect on Total Expenditure (ik_fu)
- **Claim**: In-kind food transfers increase total per-capita household expenditure, with the DD interaction term ik_fu capturing the treatment effect relative to control.
- **Baseline spec**: `baseline`
- **Expected sign**: Positive
- **Baseline coefficient**: 51.80 (SE: 28.04, p = 0.065)
- **Outcome**: `pc_exp_total`
- **Treatment**: `ik_fu`
- **N**: 10828, R2: 0.070

### G2: Cash Treatment Effect on Total Expenditure (cash_fu)
- **Claim**: Cash transfers also increase total per-capita household expenditure, with similar magnitude to in-kind transfers though less precisely estimated.
- **Baseline spec**: `baseline_cash`
- **Expected sign**: Positive
- **Baseline coefficient**: 40.44 (SE: 32.29, p = 0.211)
- **Outcome**: `pc_exp_total`
- **Treatment**: `cash_fu`
- **N**: 10828, R2: 0.070

**Note**: G1 and G2 come from the same regression -- the baseline specification includes both ik_fu and cash_fu as DD interaction terms, with the coefficient of interest differing. The paper's primary focus is on ik_fu (G1), with cash_fu (G2) used for comparison. The 69 ik_fu specifications all map to G1; the 8 cash_fu specifications (baseline_cash + 4 cash_treatment outcome decompositions + 3 implied) map to G2.

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baselines)** | **41** | |
| core_controls | 12 | 2 baselines + no controls, hh size, demographics, housing, full baseline, 5 leave-one-out control drops |
| core_outcome | 4 | Food, non-food, in-kind basket goods, non-in-kind food decompositions |
| core_sample | 13 | Exclude ik_no_educ, ik_educ_vs_cash, balanced states, HH with/without kids, large/small HH, farmers/non-farmers, indigenous/non-indigenous, diconsa yes/no |
| core_inference | 3 | Robust SEs, cluster at municipality, cluster at state |
| core_funcform | 6 | Log, asinh, total levels, per adult-equivalent, winsorize 1%, winsorize 5% |
| core_estimation | 3 | Household FE for total expenditure, food expenditure, and in-kind expenditure |
| **Non-core tests** | **36** | |
| noncore_alt_outcome | 21 | 17 disaggregated food items (10 in-kind basket items + 7 other food categories) + 4 disaggregated non-food categories |
| noncore_heterogeneity | 6 | By education (2), baseline expenditure (2), gender of head (2) |
| noncore_placebo | 5 | 3 baseline cross-sectional balance tests (treatment var = ik), 2 placebo-like nonfood DD effects (schooling, HH items) |
| noncore_alt_treatment | 4 | Cash treatment (cash_fu) on food, non-food, in-kind, non-in-kind outcomes |
| **Total** | **77** | |

## Detailed Classification Notes

### Core Tests (41 specs including 2 baselines)

**Baselines (2 specs)**: The two baseline specifications come from the same DD regression of per-capita total expenditure on treatment group indicators, a follow-up dummy, DD interaction terms (ik_fu, cash_fu), and village-level controls (Diconsa store indicator + month-of-interview fixed effects). Standard errors are clustered at the village level (id_loc). G1 reports the ik_fu coefficient (51.80, p=0.065) and G2 reports the cash_fu coefficient (40.44, p=0.211).

**Control variations (10 non-baseline core_controls specs)**: These systematically vary the control set:
- `controls/no_controls`: No village controls -- coefficient increases to 55.63 (p=0.015), suggesting controls are not driving the result but add noise
- Progressive addition of controls: hh_size, demographics, housing, full_baseline -- adding more controls generally tightens the estimate; full_baseline yields coef=53.53, p=0.026
- Leave-one-out (5 specs): Dropping individual village controls (diconsa, month FEs) one at a time -- all coefficients remain in the 51-58 range, showing no single control is driving the result

**Outcome decompositions (4 core_outcome specs)**: These decompose the total expenditure effect into components that directly test the paper's mechanism:
- `outcome/pc_exp_food` (coef=41.51, p=0.017): Most of the total effect comes from food
- `outcome/pc_exp_nfood` (coef=10.29, p=0.446): Small, insignificant nonfood effect
- `outcome/pc_exp_inkind` (coef=44.32, p<0.001): The effect is concentrated in in-kind basket goods -- this is the paper's strongest result
- `outcome/pc_exp_n_inkind` (coef=-2.81, p=0.856): No crowd-out of non-basket food items

These four specs are classified as core because they directly test the paper's central claim about "extra-marginal" consumption: in-kind transfers increase consumption of the transferred items beyond what cash-equivalent transfers would predict, and households do not compensate by reducing spending on other goods.

**Sample restrictions (13 specs)**: These test sensitivity to sample composition:
- Pooling alternatives (2 specs): exclude_ik_no_educ and ik_educ_vs_cash drop the in-kind-without-education group (these are identical, coef=58.94, p=0.086)
- Geographic: balanced_states (coef=55.43, p=0.055)
- Household composition: with/without children, large/small HH (6 specs) -- notably, hh_with_kids yields a small, insignificant effect (9.35, p=0.773) while hh_no_kids shows a large, significant effect (87.02, p=0.005)
- Occupation: farmers (3.19, p=0.938) vs non_farmers (95.63, p=0.007) -- stark difference
- Ethnicity: indigenous (70.71, p=0.040) vs non_indigenous (41.30, p=0.192)
- Diconsa store presence: diconsa_yes (60.47, p=0.108) vs diconsa_no (38.66, p=0.347)

**Inference variations (3 specs)**: Same point estimate (51.80) with different SE computations:
- `inference/robust_se`: HC1 robust SE = 19.23, p=0.007 (much smaller than clustered SE)
- `inference/cluster_muni`: Municipality clustering, SE=27.12, p=0.056
- `inference/cluster_state`: State clustering, SE=17.78, p=0.004 (fewer clusters but smaller SE)

Note: The baseline clusters at the village level (SE=28.04). The p-value sensitivity to clustering level is notable -- robust SEs and state clustering yield p<0.01 while village and municipality clustering give p>0.05.

**Functional form (6 specs)**: Alternative transformations of the outcome:
- `functional/log_outcome` (coef=0.141, p=0.003): 14.1% increase in total expenditure -- significant
- `functional/asinh_outcome` (coef=0.142, p=0.003): Nearly identical to log
- `functional/total_levels` (coef=229.26, p=0.005): Household-level (not per capita) in pesos
- `functional/per_adult_equiv` (coef=60.01, p=0.063): Per adult-equivalent -- slightly larger but similar significance
- `functional/winsorize_1pct` (coef=47.99, p=0.031): Trimming outliers at 1% -- significant
- `functional/winsorize_5pct` (coef=42.28, p=0.014): More aggressive trimming -- even more significant

The log/asinh and winsorization results suggest the baseline specification's marginal significance (p=0.065) is partly driven by heavy tails in the expenditure distribution.

**Estimation method (3 specs)**: Household fixed effects instead of village-level controls:
- `estimation/hh_fe/pc_exp_total` (coef=52.44, p=0.018): Very similar to baseline, confirming the DD design
- `estimation/hh_fe/pc_exp_food` (coef=41.63, p=0.005): Food decomposition with HH FE
- `estimation/hh_fe/pc_exp_inkind` (coef=43.98, p<0.001): In-kind goods with HH FE -- highly robust

### Non-Core Tests (36 specs)

**Disaggregated outcomes (21 specs)**: Individual food items and nonfood subcategories:
- 10 in-kind basket items: corn flour, rice, beans, powdered milk, oil, canned fish, cookies, pasta, cereal, lentils. Powdered milk dominates (coef=23.44, p<0.001), followed by cookies (6.37, p<0.001), canned fish (4.31, p<0.001), and cereal (3.87, p<0.001). Some items show no significant effect (beans: -0.03, p=0.968; oil: 0.61, p=0.287).
- 7 other food categories: fruits/vegetables, grains, pulses, meat, dairy, cooking fat, junk food. Grains (14.27, p<0.001) and dairy (10.49, p=0.008) show significant effects, likely reflecting substitution with basket items.
- 4 nonfood subcategories: schooling, clothing, medical/hygiene, household items. All insignificant, consistent with no nonfood effect.

These are non-core because they provide granular decomposition beyond the four primary outcome categories (total, food, nonfood, inkind/non-inkind) that constitute the paper's main claim.

**Heterogeneity (6 specs)**: Subgroup splits by household characteristics:
- Education of head: low_educ (57.28, p=0.091) vs high_educ (62.35, p=0.026)
- Baseline expenditure: low (68.17, p=0.002) vs high (34.49, p=0.286)
- Gender of head: male (52.51, p=0.049) vs female (67.61, p=0.159)

These are non-core because they decompose the effect by subgroup rather than test alternative specifications of the same estimate.

**Placebo tests (5 specs)**: Tests of the research design's validity:
- 3 baseline balance tests: Cross-sectional differences at baseline between in-kind and control groups for total (-24.12, p=0.394), food (-13.86, p=0.393), and nonfood (-10.26, p=0.445) expenditure. Treatment variable is `ik` (not `ik_fu`), testing pre-treatment balance. All insignificant, supporting the RCT design.
- 2 placebo-like nonfood tests: DD effects on schooling (2.88, p=0.430) and household items (1.79, p=0.700) expenditure. These are plausible placebos because food transfers should not affect these categories.

These are non-core because they validate the design rather than estimate the treatment effect.

**Cash treatment decompositions (4 specs)**: Outcome decompositions for the cash treatment arm (cash_fu):
- Food (26.20, p=0.197), nonfood (14.24, p=0.381), in-kind basket goods (6.33, p=0.041), non-in-kind food (19.87, p=0.282)
- Notably, the cash effect on in-kind basket goods (6.33) is much smaller than the in-kind effect (44.32), supporting the "extra-marginal" interpretation
- These are non-core because they track a different treatment variable (cash_fu) from the paper's primary focus (ik_fu)

## Duplicates Identified

Two specs produce identical results:
- `sample/exclude_ik_no_educ` = `sample/ik_educ_vs_cash`: Both yield coef=58.94, SE=34.32, p=0.086. They implement the same sample restriction (dropping in-kind without education group) from different angles.

Additionally, all three inference specs share the same point estimate (51.795) because they only change SE computation. The baseline and these three specs have identical coefficients by construction.

## Robustness Assessment

The main finding -- that in-kind transfers increase total per-capita expenditure -- receives **moderate** support:

- **G1 (ik_fu on pc_exp_total)**: The baseline coefficient is 51.80 with p=0.065, marginally insignificant at the 5% level. However, across 41 core specifications tracking ik_fu, the effect is consistently positive (range: 3.19 to 229.26 in different units). Adding baseline controls (full_baseline: p=0.026), using household FE (p=0.018), log/asinh transformation (p=0.003), winsorization (p=0.014-0.031), or robust SEs (p=0.007) all yield p<0.05.

- **Sensitivity to inference**: The p-value is sensitive to the level of clustering. Village-level clustering (baseline) gives p=0.065, municipality gives p=0.056, but robust SEs give p=0.007 and state clustering gives p=0.004. The choice of clustering level materially affects the statistical conclusion.

- **Strong mechanism evidence**: The decomposition into in-kind basket goods (coef=44.32, p<0.001) is highly significant and robust across all specifications including HH FE (43.98, p<0.001). This is the paper's strongest and most robust finding.

- **Heterogeneous effects**: The treatment effect varies substantially across subgroups. Non-farmers (95.63) show much larger effects than farmers (3.19). Households without children (87.02) show larger effects than those with children (9.35). Low baseline expenditure households respond more (68.17) than high (34.49). This heterogeneity is substantively important but reduces the precision of the average effect.

- **G2 (cash_fu on pc_exp_total)**: The cash effect (40.44, p=0.211) is imprecisely estimated across all specifications. The paper cannot reject equality with the in-kind effect, but neither can it confidently establish a positive cash effect on total expenditure.

Key caveats:
1. The marginal significance of the baseline specification (p=0.065) means the statistical conclusion depends on specification choices, particularly the level of clustering and the inclusion of controls.
2. The sample split `sample/exclude_ik_no_educ` and `sample/ik_educ_vs_cash` are exact duplicates, reducing the effective number of unique specifications by 1 (76 unique out of 77 total).
3. The strong heterogeneity in treatment effects (farmers vs non-farmers, kids vs no kids) suggests the average treatment effect may mask important variation in who benefits from in-kind transfers.
