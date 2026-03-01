# Specification Surface: 113109-V1

## Paper Overview
- **Title**: Housing Booms and Busts, Labor Market Outcomes, and College Attendance (Charles, Hurst, & Notowidigdo, 2018, AER)
- **Design**: Instrumental variables (cross-sectional)
- **Data**: MSA-level long differences (2000-2006/2007), US Metropolitan Statistical Areas
- **Key tables**: Table 1 (employment/wages IV), Table 2 (relative wages), Table 3 (college enrollment IV), Online Appendix robustness tables
- **Instrument**: Saiz housing supply elasticity

## Baseline Groups

### G1: Housing Boom Effect on Employment (Table 1)

**Claim object**: Housing booms caused by local demand shocks (instrumented by housing supply elasticity) increased employment, particularly in construction and FIRE sectors, with larger effects for young men (18-25). The paper uses MSA-level long differences.

**Baseline specification**:
- `ivreg2 d_emp_18_25_le (hp_growth_real_00_06 = iv) controls [aw=msa_total_emp_all_2000], cluster(statefip)`
- Endogenous: `hp_growth_real_00_06` (housing demand shock = deltaP + units_growth)
- Instrument: `iv` (Saiz housing supply elasticity)
- Controls: `college_share_2000`, `female_employed_share_2000`, `pop_prev`, `share_foreign_18_55_2000`
- Weighted by MSA total employment in 2000, clustered at state level
- Multiple outcomes in Table 1: employment-pop ratio (d_emp), wages (d_wage), adjusted wages (d_ewage), construction employment (d_cons), FIRE employment (d_fire)

### G2: Housing Boom Effect on College Enrollment (Table 3)

**Claim object**: Housing booms reduced college enrollment rates for young people, as the opportunity cost of education increased via higher non-college wages.

**Baseline specifications**:
- Table 3 IV: `ivreg2 d_any_18_25_a1 (housing_demand_shock = iv) controls [aw=wgt], cluster(statefip)`
- Also bachelor's degree rate: `d_bachelor_18_25_a1`
- Weighted by population, clustered at state level

## RC Axes Included

### Controls (both groups)
- **LOO**: Drop each of 4 baseline controls individually
- **No controls**: Bivariate IV
- **Add controls**: Region FE, manufacturing share, routine share, unemployment rate (from robustness table)
- **Full model**: All baseline + all additional controls + region FE
- **Random subsets**: Stratified by size for 8-variable pool

### Sample restrictions
- **Outlier trimming**: y at [1,99] and [5,95], x at [1,99]
- **Drop extreme elasticity MSAs**: Exclude MSAs with very high or low supply elasticity
- **Drop small MSAs**: Population threshold

### Treatment definition
- **deltaP only**: Use only house price growth (not combined with units growth)
- **deltaP scaled**: Alternative scaling of the housing demand shock

### Instrument variations (from tableOA_robustness2.do)
- **iv_sig**: Saiz elasticity with significant topography only
- **iv_sig2**: Alternative significance threshold
- **iv2_poly3**: Polynomial in elasticity
- **price_rent_ratio**: Alternative instrument based on price-rent ratio
- These are critical robustness checks for IV -- they change the excluded instrument while maintaining the identification strategy.

### Estimator variants
- **LIML**: More robust to weak instruments than 2SLS

### Weight variations
- **Unweighted**: Drop analytic weights
- **Population weights**: Use population instead of employment

### Outcome variations (G2)
- **Associate degree**: Different education level
- **Older age group (26-33)**: Different age cohort (Table 3 also reports this)

### Joint variations
- Control set x instrument combinations
- Control set x sample restriction combinations

## What Is Excluded and Why

- **Table 2 (relative wages)**: Uses different age group (26-55) and different outcomes (relative wages by skill). These are supporting evidence for the mechanism, not the main employment claim.
- **OLS specifications**: The paper reports OLS alongside IV for comparison. OLS is not a separate baseline (it estimates a different parameter -- ATE vs LATE). Included as a design comparison if useful.
- **Online Appendix first-stage tables**: These are diagnostics, not separate claims.
- **Housing bust period (2006-2009)**: Different time period. Could be a separate baseline group but the boom-period result is the headline.
- **Heterogeneity by gender**: The paper shows male/female separately, but the main claim is for the combined population.

## Budgets and Sampling

- **G1**: Max 70 core specs, 20 control subsets
- **G2**: Max 50 core specs, 15 control subsets
- **Seed**: 113109
- **Sampling**: Stratified by size for the 8-variable control pool

## Inference Plan

- **Canonical**: Cluster-robust SE at state level (matching the paper)
- **Variants**: HC1 robust (no clustering), HC2 (small-sample)
- State-level clustering with ~50 clusters is standard and well-behaved

## Key Constraints

- **Linked adjustment**: Controls appear in both first and second stage (standard IV). Control variation must be applied jointly.
- **Single instrument**: Just-identified (one instrument for one endogenous variable). No overidentification tests are feasible for the baseline setup.
- **Instrument alternatives change the LATE**: Different instruments may identify effects for different subpopulations. This is correctly treated as an RC (the exclusion restriction and relevance condition are tested, not the estimand per se).
