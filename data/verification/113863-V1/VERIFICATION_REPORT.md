# Verification Report: 113863-V1

## Paper Information
- **Title**: Marriage Networks, Nepotism and Labor Market Outcomes in China
- **Journal**: AEJ: Applied Economics
- **Total Specifications**: 79

## Baseline Groups

### G1: CHNS -- FIL Death on Male Hourly Income
- **Claim**: Death of a father-in-law (FIL) reduces men's log hourly real income in urban China, by severing nepotistic employment network connections.
- **Baseline spec**: `baseline`
- **Expected sign**: Negative
- **Baseline coefficient**: -0.119 (SE: 0.067, p = 0.075)
- **Outcome**: `loghrindinc_real` (log hourly real individual income)
- **Treatment**: `post_FILdie` (indicator = 1 after FIL death)
- **N**: 2,023; R-squared: 0.762
- **CHNS Table 2, Column 1**

### G2: SLCC -- FIL Death on Male Income (Replication)
- **Claim**: Replication of the FIL death effect using the independent SLCC retrospective panel dataset, confirming the negative earnings impact.
- **Baseline spec**: `slcc/baseline`
- **Expected sign**: Negative
- **Baseline coefficient**: -0.069 (SE: 0.034, p = 0.041)
- **Outcome**: `logincom_male` (log income, male)
- **Treatment**: `post_fILdie_male` (indicator = 1 after FIL death)
- **N**: 13,012; R-squared: 0.737
- **SLCC Table 2, Column 1**

**Note**: The two baseline groups correspond to two independent datasets (CHNS panel and SLCC retrospective panel) testing the same hypothesis. The CHNS result is marginally significant at 10% (p = 0.075) while the SLCC result is significant at 5% (p = 0.041). Both use individual + year*province fixed effects with panel FE estimation, though the CHNS uses individual-clustered SE while the SLCC uses robust SE (areg).

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baselines)** | **53** | |
| core_controls | 15 | 2 baselines + control variations (adding/removing demographics, education, employment, marriage, health) and MIL-death-controlled specs |
| core_fe | 8 | Individual-only, year-only, province-only, separate additive, year*province (baseline), pooled OLS, for both CHNS and SLCC |
| core_sample | 26 | Age ranges, urban/rural inclusion, event-only, time periods, winsorization, heads/non-heads, education splits, 9 province leave-one-out, excluding divorced |
| core_inference | 2 | Robust SE, province-clustered SE |
| core_method | 1 | First differences |
| core_funcform | 1 | Age fixed effects instead of polynomial |
| **Non-core tests** | **26** | |
| noncore_alt_outcome | 9 | Job change (x2), level income, hours worked (x2), food expenditure, log purchases, log rent value |
| noncore_alt_treatment | 3 | Mother-in-law (MIL) death (x2 with baseline/extended controls), SLCC MIL death |
| noncore_placebo | 3 | Female own father death (CHNS x2), female placebo (SLCC) |
| noncore_heterogeneity | 10 | Employment sector (state/collective/private x2 datasets), FIL-while-MIL-alive/dead split (x2 datasets), post-1997 reform interaction |
| noncore_diagnostic | 1 | Move/attrition indicator (selective attrition test) |
| **Total** | **79** | |

## Detailed Classification Notes

### Core Tests (53 specs including 2 baselines)

**Baselines (2 specs)**: The two primary baselines correspond to the main result (Table 2, Column 1) in each of the two datasets. Both regress log income on an indicator for post-FIL-death with individual + year*province fixed effects.

**Control variations (13 non-baseline core_controls specs)**: These systematically add or remove controls from the baseline specification:
- No controls (FE only): weaker effect (-0.093, p=0.16) showing controls matter for precision
- Partial control sets: age only, age+education, no education, marriage+health
- Full controls (including employment sector): strongest CHNS result (-0.152, p=0.020)
- Age functional form: linear, quadratic, cubic (baseline) age polynomials
- Controlling for MIL death (alt_treatment/both_fil_mil and extended): these are classified as core because post_FILdie remains the treatment of interest while post_MILdie is added as a covariate. The coefficient on post_FILdie is slightly strengthened.
- SLCC extended controls and SLCC controlling for MIL death

**Fixed effects variations (8 specs across both datasets)**: Systematic FE structure exploration:
- CHNS: Individual only, ID+Year, ID+Province (=individual only due to collinearity), ID+Year+Province (separate), Pooled OLS
- SLCC: Separate year+province FE, Individual FE only, Pooled OLS
- Pooled OLS in both datasets yields near-zero insignificant coefficients, confirming the within-individual variation is essential for identification.

**Sample restrictions (26 specs)**: The largest core category:
- Age range variations: 20-50 (wider) and 25-40 (narrower) in both CHNS and SLCC
- Urban/rural: including rural observations nearly eliminates the effect (coef = -0.011), consistent with the urban nepotism mechanism
- Event subsample: restricting to men who experience FIL death yields the strongest result (-0.235, p = 0.003)
- Time periods: pre-1997 (larger coef -0.298) vs post-1997 (near zero), consistent with declining nepotism after state sector reform
- Winsorization: 1-99 and 5-95 percentile bounds for the dependent variable
- Household head status: heads show stronger effect (-0.186, p = 0.012); non-heads show no effect
- Education splits: high and low education subsamples
- Province leave-one-out (9 specs): coefficient ranges from -0.075 (drop province 32) to -0.146 (drop province 37)
- Excluding divorced/separated individuals
- Include 1989 wave and no-singletons specs (both exact duplicates of baseline)

**Inference variations (2 specs)**: Same point estimate as baseline with different SE computation:
- Robust HC SE: slightly larger SE (0.075), p = 0.113 (loses marginal significance)
- Province-clustered SE: slightly smaller SE (0.059), p = 0.077 (similar to baseline)

**Method (1 spec)**: First differences as alternative to FE -- weaker effect (-0.096, p = 0.26)

**Functional form (1 spec)**: Age fixed effects instead of age polynomial -- slightly weaker (-0.104, p = 0.11)

### Non-Core Tests (26 specs)

**Alternative outcomes (9 specs)**: These test the FIL death effect on different dependent variables:
- Job change (2 specs, CHNS): positive but insignificant -- FIL death does not clearly increase job mobility
- Level income: negative but insignificant (log vs level does not matter qualitatively)
- Hours worked (2 specs): positive insignificant -- no labor supply response
- Household food expenditure: significant negative (-9.08, p = 0.007) -- only consumption measure showing significance
- Household purchases and rent value: negative but insignificant

These are non-core because they measure different outcomes, not alternative implementations of the main log-income specification.

**Alternative treatments (3 specs)**: These substitute a different treatment variable:
- MIL death in CHNS (2 specs): positive and insignificant (0.130 and 0.123), confirming the gender-asymmetric nepotism hypothesis (mothers-in-law do not provide network connections for men)
- MIL death in SLCC (1 spec): positive and insignificant (0.063)

These serve as placebo/falsification tests for the mechanism rather than robustness of the main effect.

**Placebo tests (3 specs)**: Formal falsification exercises:
- Female own father death on female income (CHNS, 2 specs): positive insignificant -- women's own fathers do not provide the same network connections, supporting the gender-asymmetric mechanism
- SLCC female placebo: negative but insignificant -- no effect on female income in SLCC either

**Heterogeneity (10 specs)**: Subsample analyses by group:
- Employment sector (6 specs across both datasets): State sector shows negative but mostly insignificant effects; collective sector shows large negative but imprecise effects; private sector shows no effect. Pattern is consistent with nepotism being concentrated in state/collective sectors.
- FIL death split by MIL survival (3 specs): FIL death while MIL is alive has a stronger effect (-0.174, p=0.018 in CHNS; -0.094, p=0.015 in SLCC) while FIL death after MIL already dead is insignificant
- Post-1997 reform interaction (1 spec): significant base effect (-0.232, p=0.010) with interaction term

These are non-core because they decompose the effect by subgroup rather than test alternative implementations.

**Diagnostic (1 spec)**: Move/attrition indicator tests whether FIL death causes differential panel attrition (marginally significant at 10%).

## Duplicates Identified

The following specs produce identical coefficients and SEs to the baseline:
1. `panel/sample/include_1989` = `baseline` (identical coef -0.1192, SE 0.0669)
2. `panel/sample/no_singletons` = `baseline` (identical coef -0.1192, SE 0.0669)

The following specs produce identical coefficients due to collinearity:
3. `panel/fe/unit_only` = `panel/fe/twoway_id_prov` (identical coef -0.0948, because province FE is absorbed by individual FE)
4. `panel/fe/twoway_id_year` vs `panel/fe/threeway` are near-identical (-0.1179 for both) for the same reason

After removing duplicates, there are approximately 75 unique specifications.

## Robustness Assessment

The main finding -- that FIL death reduces men's earnings -- receives **moderate** support:

**Direction is robust**: 96% of CHNS core specs for the main treatment (post_FILdie on loghrindinc_real) yield negative coefficients. The sign is very stable across control variations, FE structures, sample restrictions, and time periods.

**Statistical significance is fragile (CHNS)**: The CHNS baseline is only marginally significant (p = 0.075), and significance depends on specification choices:
- Stronger with full controls (p = 0.020), event-only subsample (p = 0.003), heads only (p = 0.012), winsorization (p = 0.064)
- Weaker with no controls (p = 0.16), unit-only FE (p = 0.19), wider age ranges, rural inclusion (p = 0.89), robust SE (p = 0.11)
- Province leave-one-out: 4/9 provinces maintain p < 0.10, 5/9 do not

**SLCC replication is more robust**: The SLCC baseline is significant at 5% (p = 0.041) and remains significant across most core SLCC specs (baseline extended p = 0.036, controlling for MIL p = 0.037, wider age range p = 0.018). It weakens with narrower age range (p = 0.29) and individual FE only (p = 0.08).

**Falsification tests support identification**: MIL death has no effect (positive insignificant), female placebos show no effect, pooled OLS shows no effect (confirming within-individual variation is key), and the effect is concentrated in state/collective sectors as predicted by the nepotism theory.

**Key sensitivities**:
- Urban vs rural: effect disappears with rural observations (urban-specific phenomenon)
- Time period: effect is driven by pre-1997 reform era; post-1997 subsample shows no effect
- Household head status: effect concentrated among heads
- Province 32: dropping this province halves the coefficient
- Robust SE vs clustered SE: marginal significance depends on clustering choice
