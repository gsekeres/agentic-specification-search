# Verification Report: 113829-V1

## Paper
- **Title**: Sweetening the Deal? Political Connections and Sugar Mills in India
- **Author**: Sandip Sukhtankar
- **Journal**: AEJ-Applied
- **Data**: Panel of 183 sugar mills in Maharashtra, India, 1993-2005

## Baseline Groups

### G1: Political connections and sugarcane prices during elections
- **Baseline spec_ids**: baseline
- **Claim**: Politically connected sugar mill chairmen manipulate cane prices during election years
- **Outcome**: rprice (real price paid per ton of sugarcane)
- **Treatment**: interall (interaction of polcon x election year)
- **Expected sign**: Negative (within mill FE specification)
- **Baseline coefficient**: -20.41, SE=11.02, p=0.066
- **FE**: mill + year
- **Controls**: capacity, monthly rainfall (12 months), rainfall deviations (12 months)
- **Clustering**: mill (tabfinal)
- **N**: 1,168
- **R-squared**: 0.891

## Summary Counts

| Metric | Count |
|--------|-------|
| **Total specifications** | 78 |
| **Core tests** | 48 |
| **Non-core tests** | 29 |
| **Invalid** | 1 |
| **Unclear** | 0 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_controls | 10 |
| core_fe | 4 |
| core_funcform | 5 |
| core_inference | 3 |
| core_sample | 26 |
| invalid | 1 |
| noncore_alt_outcome | 9 |
| noncore_alt_treatment | 10 |
| noncore_heterogeneity | 6 |
| noncore_placebo | 4 |

## Top 5 Most Suspicious Rows

1. **robust/sample/no_polcheck_mills** (INVALID): Reports capacity as treatment variable
   instead of interall. The variable interall (polcon x election) is identically zero for
   mills that never have a politically connected chairman, so the script fell back to
   reporting the coefficient on capacity. This is not a valid test of the baseline claim.

2. **robust/control/minimal**: Exact duplicate of robust/control/capacity_only -- same
   coefficient (-21.22), same SE, same p-value. Both use mill+year FE with capacity as
   the only control. Redundant but not invalid.

3. **panel/fe/twoway**: Identical coefficient and p-value to baseline (-20.41, p=0.066).
   This is because the baseline already uses mill+year (two-way) FE. The spec_id suggests
   it is a separate FE variation but it is actually identical to the baseline.

4. **robust/sample/exclude_last_year** and **robust/sample/drop_year_2005**: Both yield
   identical results to the baseline (coef=-20.41, p=0.066). Suggests 2005 has no
   variation or is already excluded from the sample.

5. **robust/placebo/party_power** (treatment=partyinstate): Shows a significant POSITIVE
   effect (coef=22.91, p=0.019). Tests whether the ruling party being in power at the
   state level affects prices. Correctly classified as non-core placebo.

## Recommendations for Spec-Search Script

1. **Fix no_polcheck_mills extraction**: When interall is collinear/zero in a subsample,
   the script should either skip the specification and note it as infeasible, or clearly
   flag that the reported coefficient is not for the treatment of interest.

2. **Remove duplicate minimal spec**: robust/control/minimal is identical to
   robust/control/capacity_only. The script should detect and deduplicate.

3. **Note redundant FE specs**: panel/fe/twoway produces identical results to baseline.
   The script could detect coefficient equality and flag duplicates.

4. **Clarify treatment variable decompositions**: The alternative treatment specs (pcinter,
   acinter, pcconnected, acconnected) test related but distinct causal objects. The script
   should clearly label these as decompositions rather than robustness checks.

5. **Consider adding the paper actual clustering**: The paper uses two-way clustering
   (mill and zone-year). The spec search approximated with mill-only clustering. Adding
   the correct two-way clustering would improve fidelity.

