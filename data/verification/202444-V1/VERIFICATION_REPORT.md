# Verification Report: 202444-V1

## Paper Information
- **Title**: Polity Size and Local Government Performance: Evidence from India
- **Authors**: Veda Narasimhan and Jeffrey Weaver
- **Journal**: American Economic Review (AER)
- **Paper ID**: 202444-V1
- **Verified**: 2026-02-09
- **Verifier**: verification_agent

## Summary Statistics

| Metric | Count |
|--------|-------|
| Total specifications | 186 |
| Baseline specifications | 15 |
| Core test specifications | 107 |
| Non-core specifications | 79 |
| Baseline groups | 5 |

## Baseline Groups

### G1: Education Index (Table 2, 1991 RD)
- **Claim**: Villages above the 1000-population threshold in the 1991 census have better educational outcomes
- **Expected sign**: Positive
- **Baseline outcome**: `education_index`
- **Baseline treatment**: `above_1000_pop91`
- **Baseline coefficient**: 0.095 (p < 0.001)
- **Total specs in group**: 47

### G2: Village Amenities Index (Table 2, 1991 RD)
- **Claim**: Villages above the 1000-population threshold in the 1991 census have better village-level infrastructure
- **Expected sign**: Positive
- **Baseline outcome**: `amenities_village_index`
- **Baseline treatment**: `above_1000_pop91`
- **Baseline coefficient**: 0.057 (p = 0.013)
- **Total specs in group**: 39

### G3: Household Amenities Index (Table 2, 1991 RD)
- **Claim**: Villages above the 1000-population threshold in the 1991 census have better household-level infrastructure
- **Expected sign**: Positive
- **Baseline outcome**: `amenities_hh_index`
- **Baseline treatment**: `above_1000_pop91`
- **Baseline coefficient**: 0.097 (p < 0.001)
- **Total specs in group**: 33

### G4: Mission Antyodaya Programs Index (Table 3, Panel A, 2011 RD)
- **Claim**: Villages above the 1000-population threshold in the 2011 census have better welfare program delivery
- **Expected sign**: Positive
- **Baseline outcome**: `ma_index`
- **Baseline treatment**: `above_1000_pop11`
- **Baseline coefficient**: 0.100 (p = 0.008)
- **Total specs in group**: 35

### G5: NREGS Performance Z-Score (Table 3, Panel B, 2011 RD)
- **Claim**: Villages above the 1000-population threshold in the 2011 census have better NREGS workfare performance
- **Expected sign**: Positive
- **Baseline outcome**: `performance_z`
- **Baseline treatment**: `above_1000_pop11`
- **Baseline coefficient**: 0.157 (p < 0.001)
- **Total specs in group**: 32

## Classification Breakdown

### Core Tests (107 specifications)

| Category | Count | Description |
|----------|-------|-------------|
| core_method | 5 | Baseline replications of the 5 main index outcomes |
| core_funcform | 50 | Bandwidth (25), polynomial order (10), kernel function (10), IHS transformation (5) |
| core_sample | 31 | Donut hole (15), exclude noncompliant districts (3), trimming (5), winsorization (5), All-India (3) |
| core_controls | 12 | Parametric RD with progressive control sets (bivariate, +SC proportion, +infrastructure, full) |
| core_inference | 9 | No clustering (5), block-level clustering (4) |

### Non-Core Tests (79 specifications)

| Category | Count | Description |
|----------|-------|-------------|
| noncore_alt_outcome | 34 | Baseline-only outcomes (10), component/subgroup outcomes (24) |
| noncore_placebo | 25 | Placebo cutoffs at 500/750/1500/2000 (20), balance tests on predetermined covariates (5) |
| noncore_heterogeneity | 13 | By SC proportion, literacy, GP size, solo GP status, habitations |
| noncore_alt_treatment | 5 | Alternative running variables (2011 pop for amenities, 2001 pop for amenities) |
| noncore_diagnostic | 2 | First stage: effect on GP population |

## Classification Decisions and Rationale

### Core Classification Rationale

1. **Bandwidth variations (core_funcform)**: These test the same outcome-treatment pair with different RD bandwidths (fixed 200/300/500, half-optimal, double-optimal). They directly test whether the baseline result is sensitive to bandwidth choice, a central concern in RD designs.

2. **Polynomial order (core_funcform)**: Quadratic and cubic local polynomials (baseline uses linear). These test sensitivity to the functional form of the local polynomial, another standard RD robustness check.

3. **Kernel function (core_funcform)**: Uniform and Epanechnikov kernels (baseline uses triangular). These test sensitivity to the weighting function used near the cutoff.

4. **IHS transformation (core_funcform)**: Inverse hyperbolic sine transformation of the same outcome variable. Tests sensitivity to the functional form of the dependent variable.

5. **Donut hole (core_sample)**: Excludes observations within +/- 25, 50, or 100 of the cutoff. Tests whether results are driven by potential manipulation of the running variable near the threshold.

6. **Sample restrictions (core_sample)**: Excluding noncompliant districts, trimming outcomes at 1%/99%, winsorizing at 5%/95%, and running on the All-India sample. These directly test whether the baseline effect estimate is sensitive to the sample composition.

7. **Clustering (core_inference)**: No clustering and block-level clustering (baseline clusters at GP level). Tests sensitivity of inference to the level of clustering.

8. **Control progression (core_controls)**: Parametric RD (OLS within BW=300) with progressively added controls. Tests whether the treatment effect changes as covariates are added, a standard specification test.

### Non-Core Classification Rationale

1. **Baseline-only outcomes (noncore_alt_outcome)**: The 10 baselines for individual outcomes (any_primary01, any_middle01, any_primary11, any_middle11, bpl_hhs, health_insurance_hhs, total_hhs_pensions, electricity_saubhagya, persondays_pp, expenditure_labor_pp) are classified as non-core because they measure different outcomes than the main indices. They represent individual Table 2/3 entries rather than core robustness tests.

2. **Component outcomes (noncore_alt_outcome)**: The 24 component/subgroup outcome specifications decompose the main indices into their constituent parts. While informative, these test different outcomes rather than providing direct robustness evidence for the main index results.

3. **Placebo cutoffs (noncore_placebo)**: Placebo RD at cutoffs 500, 750, 1500, and 2000. These test the identification strategy rather than the robustness of a specific coefficient estimate.

4. **Balance/predetermined tests (noncore_placebo)**: Testing whether predetermined covariates (primary/middle school in 1991, tarred road, electricity, SC proportion) show discontinuities at the 1000 threshold. These validate the RD design rather than testing the main effect.

5. **Heterogeneity (noncore_heterogeneity)**: Subsample analyses by SC proportion, literacy, GP size, solo GP status, and habitation count. These examine treatment effect heterogeneity rather than providing robustness for the average effect.

6. **Alternative running variables (noncore_alt_treatment)**: Using 2011 or 2001 population as the running variable for amenity outcomes (baseline uses 1991 pop). This is a different RD design entirely (different delimitation episode), so it tests a different natural experiment.

7. **First stage (noncore_diagnostic)**: Effect of crossing the 1000 threshold on GP population. This is a diagnostic for the RD mechanism, not a robustness test of the reduced-form effect.

## Confidence Assessment

All 186 specifications were classified with **high confidence**. The specification search documentation (`SPECIFICATION_SEARCH.md`) provided clear categorization that aligned well with the observed data patterns. The spec_id naming convention (e.g., `rd/bandwidth/fixed_200`, `robust/sample/trim_1pct`) made classification straightforward.

## Notes

- The paper uses two distinct RD designs (1991 and 2011 delimitation episodes), each with its own running variable and outcome set.
- The 15 baseline specifications span 15 distinct outcomes across Tables 2 and 3 of the paper. Only the 5 main index outcomes (education_index, amenities_village_index, amenities_hh_index, ma_index, performance_z) have dedicated robustness specifications.
- The All-India sample specifications use `above_1000_pop11` as the treatment variable even for amenity outcomes that use `above_1000_pop91` in the baseline. This reflects the fact that the All-India analysis uses the 2011 delimitation.
- The parametric RD specifications (control progression) use OLS within a fixed bandwidth of 300, unlike the baseline which uses rdrobust with MSE-optimal bandwidth.
- No specifications were classified as `invalid` or `unclear`.
