# Verification Report: 195142-V1

## Paper
**Title**: Place-Based Subsidies and Regional Convergence: Evidence from Germany  
**Journal**: AER  
**Paper ID**: 195142-V1  
**Verified**: 2026-02-03

## Baseline Groups

### G1: Total GRW Subsidy Receipts
- **Claim**: GRW subsidy rate reforms reduce total GRW subsidy receipts at the county level in East Germany.
- **Baseline spec_id**: baseline
- **Outcome**: D_ln_grw_total (first-differenced log total GRW subsidies)
- **Treatment**: D_ref_weighted (first-differenced weighted reform variable)
- **Coefficient**: 0.0290, SE: 0.0235, p: 0.226
- **Expected sign**: Positive (subsidy rate cuts reduce receipts)

### G2: Subsidized Investment Volume
- **Claim**: GRW subsidy rate reforms reduce subsidized investment volume at the county level in East Germany.
- **Baseline spec_id**: baseline_grw_vol
- **Outcome**: D_ln_grw_vol (first-differenced log subsidized investment volume)
- **Treatment**: D_ref_weighted
- **Coefficient**: 0.0370, SE: 0.0279, p: 0.194
- **Expected sign**: Positive

## Counts

| Category | Count |
|----------|-------|
| Total specifications | 68 |
| Baselines | 2 |
| Core tests (including baselines) | 40 |
| Non-core tests | 19 |
| Invalid | 9 |
| Unclear | 0 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_method | 5 | Baseline specs and alternative estimation approaches |
| core_fe | 5 | Fixed effects structure variations |
| core_inference | 3 | Clustering/SE variations |
| core_sample | 20 | Sample restriction and bandwidth variations |
| core_funcform | 4 | Functional form alternatives (levels, IHS, cumulative) |
| core_controls | 3 | Control variable additions |
| noncore_alt_outcome | 5 | Different outcomes (unemployment, GDP, tax, population) |
| noncore_heterogeneity | 11 | Split-sample heterogeneity analyses |
| noncore_placebo | 3 | Placebo and falsification tests |
| invalid | 9 | Exact duplicates or coding errors |

## Top 5 Most Suspicious Rows

### 1. robust/treatment/small, robust/treatment/medium, robust/treatment/large
**Issue**: All three produce coefficients and standard errors *identical* to the baseline (coeff=0.02896, SE=0.02347, p=0.226). The script constructs D_ref_small, D_ref_medium, D_ref_large by first-differencing ref_small, ref_medium, ref_large respectively. However, since ref_weighted = (ref_small + ref_medium + ref_large) / 3, and the D_ operator is linear, D_ref_weighted = (D_ref_small + D_ref_medium + D_ref_large) / 3. The identical results suggest that either (a) all three size-specific reform variables are perfectly collinear after differencing in the m30 sample (unlikely), or (b) there is a bug where the script used the wrong column. **Marked invalid.**

### 2. did/fe/state_year (duplicate of baseline)
**Issue**: This spec claims to test "State x Year FE (baseline)" but produces results identical to the baseline spec. The baseline already uses state_year FE, so this is simply a duplicate. **Marked invalid.**

### 3. robust/cluster/amr (duplicate of baseline)
**Issue**: The baseline already clusters at the AMR (labor market region) level. This spec repeats the exact same clustering and produces identical results. **Marked invalid.**

### 4. robust/outcome/grw_total and robust/outcome/grw_vol (duplicates of baselines)
**Issue**: robust/outcome/grw_total is an exact duplicate of baseline, and robust/outcome/grw_vol is an exact duplicate of baseline_grw_vol. These were apparently generated to populate the "alternative outcomes" category but simply re-ran the baseline specifications. **Marked invalid.**

### 5. robust/sample/drop_year_2017 (identical to baseline)
**Issue**: Dropping year 2017 produces results identical to the baseline (same N=1141, same coefficient). This suggests 2017 is the last year in the data and first-differencing already eliminates it (the D_ operator requires a lagged value that would be 2016). The "drop" has no effect. **Marked invalid.**

## Critical Observations

### Data Limitation
The paper studies manufacturing employment effects of GRW subsidies using confidential establishment-level data (BHP). The replication package does not include BHP data. All specifications in this search use county-level aggregate outcomes from the publicly available external_data file. **The specification search cannot test the paper primary claim about employment.**

### Baseline Interpretation
The positive coefficient on D_ref_weighted indicates that when subsidy rates are reformed (cut), total GRW subsidies increase. This is counterintuitive if "reform" means cuts. The SPECIFICATION_SEARCH.md describes this as "subsidy rate cuts are associated with reduced subsidy receipts" with expected positive sign, but this requires that the reform variable is coded such that positive values indicate rate increases, or the relationship is mechanical (higher subsidies correlate with reform eligibility). The exact coding of ref_weighted deserves scrutiny.

### Duplicate Prevalence
9 of 68 specifications (13%) are exact duplicates or invalid due to coding errors. This inflates the apparent number of distinct robustness checks. After removing duplicates, there are effectively 59 distinct specifications.

## Recommendations for Fixing the Spec-Search Script

1. **Remove duplicate specifications**: did/fe/state_year, robust/cluster/amr, robust/control/none, robust/outcome/grw_total, robust/outcome/grw_vol, and robust/sample/drop_year_2017 are exact duplicates that should be removed.

2. **Fix size-specific treatment construction**: The variables D_ref_small, D_ref_medium, D_ref_large should be constructed independently from ref_small, ref_medium, ref_large (not derived from the weighted average). Investigate why all three produce identical results; this likely indicates a data issue where ref_small = ref_medium = ref_large in the m30 sample after first-differencing.

3. **Clarify baseline claim**: The specification search describes the claim as about GRW subsidy flows, but the paper main claim is about manufacturing employment. If possible, note that these results test a secondary/mechanical relationship, not the paper headline finding.

4. **Consider adding more meaningful robustness checks**: Since the paper uses an event study design with staggered adoption, consider adding event study lead/lag coefficients (the script creates them but never uses them), and modern DiD estimators (Callaway-SantAnna, Sun-Abraham) which are particularly relevant for staggered designs.
