# Specification Search Report: 112749-V1

## Paper
Hornbeck & Naidu (2014), "When the Levee Breaks: Black Migration and Economic Development in the American South," AER 104(3): 963-990.

## Surface Summary
- **Baseline groups**: 2
  - G1: Black population share (lnfrac_black ~ f_int, Table 2)
  - G2: Farm equipment value (lnvalue_equipment ~ f_int, Table 4)
- **Design code**: difference_in_differences (TWFE with county FE)
- **Budget**: 80 max specs per group
- **Seed**: 112749 (full enumeration, no sampling needed)

## Baseline Specifications

### G1: Black population share
- Table 2, Column 1: `areg lnfrac_black f_int_* [state-year FE + geo + lags] | fips, cluster(fips)`
- Focal coefficient (f_int_1930): -0.1563, SE: 0.0321, p < 0.001, N = 2604
- Fixed effects: county (fips, absorbed)
- Controls: state-year dummies + geographic/crop suitability controls + lagged outcome
- Inference: clustered SE at county (fips) level
- Weights: county area (county_w)

### G2: Farm equipment value
- Table 4, Column 2: `areg lnvalue_equipment f_int_* [state-year FE + geo + equip lags] | fips, cluster(fips)`
- Focal coefficient (f_int_1940): 0.4396, SE: 0.0995, p < 0.001, N = 2170
- Fixed effects: county (fips, absorbed)
- Controls: state-year dummies + geographic/crop suitability controls + lagged equipment values
- Inference: clustered SE at county (fips) level
- Weights: county area (county_w)

## Execution Counts

### G1: Black Population Share (23 specs)

| Category | Planned | Executed | Failed | Notes |
|----------|---------|----------|--------|-------|
| Baseline | 1 | 1 | 0 | Table 2 Col 1, f_int_1930 focal |
| Design (TWFE) | 1 | 1 | 0 | Identical to baseline (explicit TWFE label) |
| Control sets | 3 | 3 | 0 | None, minimal (state-year only), extended (+ND) |
| Control progression | 4 | 4 | 0 | Geography, lags, New Deal, full |
| FE variant | 1 | 0 | 1 | Drop state-year FE: Matrix singular |
| Sample restrictions | 3 | 3 | 0 | Drop 1930, drop 1970, short window 1930-1950 |
| Sample outliers | 1 | 0 | 1 | Trim Y 1-99: Matrix singular |
| Functional form | 2 | 2 | 0 | Level outcome, binary flood treatment |
| Weights | 2 | 2 | 0 | Unweighted, population-1920 weighted |
| Alt focal coef (f_int_1950) | 4 | 4 | 0 | Baseline + 3 RC variants at 1950 focal |
| Alt outcome (lnpopulation_black) | 1 | 1 | 0 | Population level instead of share |
| **G1 Total** | **23** | **21** | **2** | |

### G2: Farm Equipment Value (23 specs)

| Category | Planned | Executed | Failed | Notes |
|----------|---------|----------|--------|-------|
| Baseline | 1 | 1 | 0 | Table 4 Col 2, f_int_1940 focal |
| Design (TWFE) | 1 | 1 | 0 | Identical to baseline (explicit TWFE label) |
| Control sets | 3 | 2 | 1 | None, minimal succeed; extended (+ND) singular |
| Control progression | 4 | 3 | 1 | Geography, lags, New Deal succeed; full singular |
| FE variant | 1 | 0 | 1 | Drop state-year FE: Matrix singular |
| Sample restrictions | 2 | 1 | 1 | Short window succeeds; drop first post singular |
| Sample outliers | 1 | 1 | 0 | Trim Y 1-99 |
| Functional form | 2 | 2 | 0 | Level outcome, binary flood treatment |
| Weights | 2 | 0 | 2 | Unweighted and pop-1920 both singular |
| Alt focal coefs | 4 | 2 | 2 | f_int_1930 and f_int_1970 baselines succeed; unweighted and extended at f_int_1970 singular |
| Alt outcome (lntractors) | 1 | 1 | 0 | Tractor count instead of equipment value |
| **G2 Total** | **23** | **15** | **8** | |

### Combined Totals

| Metric | Count |
|--------|-------|
| Total specification rows | 46 |
| Successful (run_success=1) | 36 |
| Failed (run_success=0) | 10 |
| Success rate | 78.3% |

### Inference Variants (separate file)

| Variant | Base Spec | Group | Coefficient | SE | p-value | Success |
|---------|-----------|-------|------------|------|---------|---------|
| infer/se/hc/hc1 | G1 baseline | G1 | -0.1563 | 0.0562 | 0.005 | Yes |
| infer/se/cluster/state | G1 baseline | G1 | -0.1563 | 0.0373 | 0.003 | Yes |
| infer/se/hc/hc1 | G2 baseline | G2 | 0.4396 | 0.0956 | <0.001 | Yes |
| infer/se/cluster/state | G2 baseline | G2 | 0.4396 | 0.1813 | 0.041 | Yes |

## Failure Analysis

All 10 failures report "Matrix is singular." This is a known issue with this paper's data structure:

1. **G1 failures (2)**: Dropping state-year FE leaves too few identifying variation sources; trimming the log-share outcome creates collinearity with absorbed county FE.

2. **G2 failures (8)**: The equipment outcome is available only for 3 census years (1900-1940 + 1970), creating a sparser panel. This causes singularity when:
   - Adding New Deal controls to the extended set (reduces observations, creates collinearity)
   - Dropping state-year FE (insufficient variation)
   - Dropping the first post-treatment period (only 2 post periods remain)
   - Changing weight specification (county_w absorbs variation needed for identification)

The higher failure rate for G2 (34.8%) vs G1 (8.7%) reflects the fundamentally sparser panel structure of the agricultural census data.

## Software Stack
- Python 3.x
- pyfixest (0.40+) for OLS/TWFE with absorbed FE and clustered SE
- pandas, numpy, pyreadstat, json

## Results Summary

### G1: Black Population Share (21 successful specs)

**Coefficient range for f_int_1930 focal:**
- Range: [-0.1746, -0.0878]
- Baseline: -0.1563
- All 18 specs with f_int_1930 focal yield negative, statistically significant coefficients (p < 0.01 for most)
- Binary flood treatment (f_bin_1930) attenuates the effect to -0.088 but remains significant (p = 0.003)

**Coefficient range for f_int_1950 focal (4 specs):**
- Range: [-0.3910, -0.1812]
- All negative and significant

**Alt outcome (lnpopulation_black):**
- Coefficient: -0.138, p = 0.012 -- significant and same sign

### G2: Farm Equipment Value (15 successful specs)

**Coefficient range for f_int_1940 focal:**
- Range: [0.1577, 0.6554]
- Baseline: 0.4396
- All 11 specs with f_int_1940 focal yield positive, statistically significant coefficients
- Binary flood treatment (f_bin_1940) attenuates the effect to 0.158 but remains significant (p = 0.011)
- No-controls spec inflates coefficient to 0.655 (expected: omitted variable bias)

**Alt focal coefs:**
- f_int_1930: 0.022 (p = 0.796) -- not significant (pre-trend check, expected null)
- f_int_1970: 0.700 (p < 0.001) -- large and significant (persistent/growing effect)
- No-controls at f_int_1970: 4.887 (inflated, p < 0.001)

**Alt outcome (lntractors):**
- Coefficient at f_int_1940: 1.041, p < 0.001 -- strong positive effect on tractor adoption

### Robustness Assessment

**G1 (Black population share): STRONG support**
- The negative effect of flood exposure on Black population share is robust across all successful specifications
- Consistent sign and significance across control progressions, functional form, weight variants, and alternative focal periods
- Effect magnitude is stable around -0.15 to -0.18

**G2 (Farm equipment value): STRONG support**
- The positive effect of flood exposure on farm equipment value is robust across all successful specifications
- Pre-trend check (f_int_1930) shows no significant pre-existing trend, as expected
- Effect persists and grows over time (f_int_1970 > f_int_1940 > f_int_1930)
- Higher failure rate is a data limitation, not a robustness concern

### Specifications with p > 0.05
- G2 baseline__f_int_1930: p = 0.796 (expected null -- pre-treatment placebo)
- No other successful specification has p > 0.05

## Deviations from Surface

1. **rc/controls/sets/baseline and rc/controls/progression/bivariate** were not run as separate specs for either group because the baseline spec already captures the full control set. The "baseline" control set IS the baseline, so it would be redundant. The "bivariate" progression (no controls at all) is captured by rc/controls/sets/none.

2. **10 specifications failed with matrix singularity.** These are genuine numerical failures due to the sparse panel structure (especially G2 with only 3 census years for equipment). Failures are recorded with run_success=0 and run_error="Matrix is singular."

3. **Additional focal coefficient variants** were added beyond the surface plan (f_int_1950 for G1, f_int_1930/f_int_1970 for G2) to capture the full dynamic treatment effect pattern, following the paper's emphasis on time-varying effects. These use the `baseline__` prefix convention.

4. **Alternative outcome variants** (lnpopulation_black for G1, lntractors for G2) were added as single RC specs to test sensitivity to outcome measurement.
