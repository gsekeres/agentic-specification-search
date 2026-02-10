# Verification Report: 113673-V1

## Paper Information
- **Title**: Vocational Training for Disadvantaged Youth in Colombia: A Long Term Follow Up
- **Authors**: Orazio Attanasio, Arlen Guarin, Carlos Medina, Costas Meghir
- **Journal**: AEJ-Applied (2017)
- **Total Specifications**: 50

## Baseline Groups

### G1: Formal Sector Income (contrib_inc_max)
- **Claim**: The Juventud en Accion vocational training program increases formal sector income for disadvantaged youth.
- **Baseline specs**: `baseline/ec/formal_income`, `baseline/es/formal_income`
- **Expected sign**: Positive
- **EC coefficient**: 26824.92 (SE: 4267.12, t=6.29, p<0.001, N=372,648)
- **ES coefficient**: 35330.67 (SE: 10766.22, t=3.28, p=0.001, N=306,696)
- **Outcome**: `contrib_inc_max` (formal sector income from social security records)
- **Treatment**: `select_h1` (EC) / `TK` (ES) -- program selection indicator

### G2: Formal Sector Employment (pareado_max)
- **Claim**: The program increases the probability of formal sector employment.
- **Baseline specs**: `baseline/ec/formal_employment`, `baseline/es/formal_employment`
- **Expected sign**: Positive
- **EC coefficient**: 0.0383 (SE: 0.0054, t=7.03, p<0.001, N=372,648)
- **ES coefficient**: 0.0424 (SE: 0.0121, t=3.50, p<0.001, N=306,696)
- **Outcome**: `pareado_max` (binary: appears in social security records)
- **Treatment**: `select_h1` (EC) / `TK` (ES)

### G3: Large Firm Employment (N200)
- **Claim**: The program increases employment in large formal firms (>200 employees), indicating access to higher-quality formal sector jobs.
- **Baseline specs**: `baseline/ec/large_firm`, `baseline/es/large_firm`
- **Expected sign**: Positive
- **EC coefficient**: 0.0271 (SE: 0.0046, t=5.91, p<0.001, N=372,648)
- **ES coefficient**: 0.0323 (SE: 0.0102, t=3.17, p=0.002, N=306,696)
- **Outcome**: `N200` (binary: employed in firm with >200 employees)
- **Treatment**: `select_h1` (EC) / `TK` (ES)

**Note**: Each baseline group contains two specifications (EC and ES) because the paper uses two independent samples from the same experiment. The Entire Cohort (EC) uses administrative records only and is unweighted. The Evaluation Sample (ES) conditions on baseline survey participation and uses sampling weights to account for survey non-response. Both samples use Course x Gender fixed effects corresponding to the randomization strata.

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baselines)** | **23** | |
| core_baseline | 9 | 6 baselines + 3 duplicates under outcome/* that exactly reproduce baselines |
| core_controls | 7 | Displacement main effects (4: p3/p5 with and without FE), full controls (1 duplicate), no-gender-interactions (2, which also restrict sample) |
| core_fe | 1 | No-FE specification for formal employment |
| core_sample | 4 | EC-only (duplicate), ES-only (duplicate), weighted ES (duplicate), unweighted EC (duplicate) |
| core_inference | 2 | Individual clustering for EC and ES (duplicates of baseline SEs) |
| **Non-core tests** | **27** | |
| noncore_heterogeneity | 16 | Gender subgroups (12) + displacement interactions (4) |
| noncore_diagnostic | 11 | Balance tests for baseline covariates (11) |
| **Total** | **50** | |

## Detailed Classification Notes

### Core Tests (23 specs including 6 baselines)

**Baselines (6 specs)**: The six primary baseline specifications estimate the treatment effect on three outcomes (formal income, formal employment, large firm employment) across two independent samples (EC, ES). All use OLS with Course x Gender fixed effects corresponding to the randomization strata. All coefficients are positive and highly significant.

**Duplicates flagged as core (10 specs)**: A notable feature of this specification set is that many "robustness" specifications are exact duplicates of baselines. Specifically:
- `outcome/formal_income/ec` = `baseline/ec/formal_income` (coef 26824.92, SE 4267.12)
- `outcome/formal_binary/ec` = `baseline/ec/formal_employment` (coef 0.0383, SE 0.0054)
- `outcome/large_firm/ec` = `baseline/ec/large_firm` (coef 0.0271, SE 0.0046)
- `sample/ec_only/formal_employment` = `baseline/ec/formal_employment`
- `sample/es_only/formal_employment` = `baseline/es/formal_employment`
- `weights/weighted/es/formal_employment` = `baseline/es/formal_employment` (ES is already weighted)
- `weights/unweighted/ec/formal_employment` = `baseline/ec/formal_employment` (EC is already unweighted)
- `controls/full_gender_interactions/ec` = `baseline/ec/formal_employment` (baseline already uses gender interactions)
- `inference/cluster_individual/ec` = `baseline/ec/formal_employment` (baseline already clusters at individual)
- `inference/cluster_individual/es` = `baseline/es/formal_employment` (baseline already clusters at individual)

These duplicates arise because the specification search extracted the same regression result under multiple category labels. They are classified as core because they reproduce the baseline exactly, but they do not provide additional information.

**Displacement main effects (4 specs)**: The displacement specifications `displacement/no_fe/p3`, `displacement/fe/p3`, `displacement/no_fe/p5`, `displacement/fe/p5` estimate the treatment main effect on formal employment while controlling for local treatment intensity (proportion of unemployed treated in the local labor market). The treatment variable is still `select_h1` and the outcome is still `pareado_max`. The versions with FE add the treatment intensity control to the baseline; the versions without FE also drop the Course x Gender FE. These are core because they test the same hypothesis (treatment --> formal employment) with additional controls.

**No-FE estimation (1 spec)**: `estimation/no_fe/formal_employment` drops the Course x Gender fixed effects. This is a core FE variation. Note this spec has the same coefficient (0.0444) as `displacement/no_fe/p3`, suggesting they are the same regression.

**Controls without gender interactions (2 specs)**: `controls/no_gender_interactions/women` and `controls/no_gender_interactions/men` run the baseline specification on gender-specific subsamples without interacting controls with gender. These serve a dual role as both control variations and sample restrictions. They are classified as core because they estimate the same treatment effect (program --> formal employment) with a modified control structure, even though the sample is also restricted. The coefficients (women: 0.034, men: 0.049) bracket the full-sample baseline (0.038), consistent with the overall treatment effect.

### Non-Core Tests (27 specs)

**Gender heterogeneity (12 specs)**: The 12 heterogeneity/gender specs estimate treatment effects separately for men and women across all three outcomes and both samples. These are non-core because they test whether the effect varies by gender rather than providing alternative estimates of the average treatment effect. Results are largely consistent: effects are positive for both genders, with women showing larger effects on employment and men showing larger effects on income in the EC sample. Two ES specs for men are marginally insignificant (p=0.059 for income and employment) reflecting smaller effective sample sizes.

**Displacement interactions (4 specs)**: The displacement interaction specs estimate the coefficient on `select_h1#c.p3_t4` or `select_h1#c.p5_t4`, which is the interaction between treatment and local treatment intensity. The treatment variable is an interaction term, not the main treatment indicator. These test whether the treatment effect is heterogeneous with respect to local market saturation (a general equilibrium displacement question), not whether the treatment has a main effect. All four interaction coefficients are statistically insignificant (p-values: 0.123, 0.553, 0.089, 0.757), consistent with no displacement.

**Balance tests (11 specs)**: These regress pre-treatment (2004) baseline characteristics on the treatment indicator to verify randomization balance. The outcome variables (empl_04, pempl_04, contract_04, dformal_04, salary_04, profit_04, days_04, hours_04, educ_lb, age_lb, dmarried_lb) are all pre-treatment covariates, not post-treatment outcomes. They test the validity of the research design, not the treatment effect itself. Two characteristics show significant imbalance: education (coef 0.255, p<0.001) and age (coef -0.192, p=0.022). The paper addresses this by including these variables as controls in all main specifications.

## Duplicates Summary

After removing exact duplicates, there are approximately 30 unique specifications:
- 6 unique baselines
- 4 unique displacement main effects (which include treatment + treatment intensity control)
- 1 unique no-FE specification (may overlap with displacement/no_fe/p3)
- 4 unique displacement interactions
- 12 unique gender heterogeneity specs
- 11 unique balance tests
- 2 unique controls-without-interactions specs
- Remaining specs are exact duplicates of baselines labeled under different categories

## Robustness Assessment

The main findings are **strongly robust** across all core specifications:

- **G1 (Formal income)**: Both EC and ES show large, positive, significant effects. The EC coefficient (26,825 pesos) and ES coefficient (35,331 pesos) are consistent in direction and general magnitude. Gender subgroups (non-core) confirm the effect for both men and women.

- **G2 (Formal employment)**: The EC baseline (0.038, t=7.03) and ES baseline (0.042, t=3.50) are highly consistent. The no-FE specification yields a slightly larger estimate (0.044, t=8.35), and displacement-control specifications (0.039-0.045) remain significant and close to baseline. The effect is robust to dropping fixed effects, adding treatment intensity controls, and varying the control structure.

- **G3 (Large firm employment)**: EC (0.027, t=5.91) and ES (0.032, t=3.17) are consistent. No additional core robustness specs test this outcome beyond the baselines and their duplicates.

Key observations:
- **Cross-sample consistency**: EC and ES produce very similar estimates despite using different data sources (admin-only vs survey-augmented), different sample sizes, and different weighting schemes. This is strong evidence that results are not driven by sample selection or measurement choices.
- **No displacement effects**: All displacement interaction terms are insignificant (p > 0.08), suggesting no general equilibrium crowding out.
- **Randomization mostly balanced**: 9 of 11 baseline characteristics show no significant differences. The two imbalanced variables (education, age) are controlled for in all main specifications.
- **Limited specification variation**: Because the microdata was not available, the specification search was limited to extracting results from existing log files. No additional control progressions, functional form variations, or alternative inference approaches could be tested beyond what the authors already reported.
