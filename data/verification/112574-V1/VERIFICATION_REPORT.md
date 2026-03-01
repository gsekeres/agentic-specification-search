# Verification Report: 112574-V1

## Paper
Kuziemko & Werker (2006), "How Much Is a Seat on the Security Council Worth? Foreign Aid and Bribery at the United Nations", JDE

## Baseline Groups Found

### G1: UN voting alignment x election interaction on ODA
- **Baseline spec_run_ids**: 112574-V1_run_001 (baseline, pair+year FE), 112574-V1_run_002 (pair+donor-year FE), 112574-V1_run_003 (pair FE + macro controls)
- **Baseline spec_ids**: baseline, baseline__main_colV, baseline__main_colVI
- **Claim**: Politically aligned countries (high UN voting agreement) receive more ODA during election years -- the interaction p_unvotes_elecex captures this differential effect
- **Baseline coefficient**: 45.46 (p=0.017, N=15315) with pair + year FE, 3-way clustering
- **Expected sign**: Positive (more aligned countries get more aid around elections)

## Counts

| Metric | Count |
|--------|-------|
| Total rows in specification_results.csv | 51 |
| Valid (run_success=1) | 51 |
| Invalid (run_success=0) | 0 |
| Core tests (is_core_test=1) | 48 |
| Non-core (valid) | 3 |
| Baseline rows | 6 (3 core, 3 noncore) |
| Inference variants (inference_results.csv) | 4 |

## Category Counts

| Category | Count |
|----------|-------|
| core_sample | 17 |
| core_controls | 14 |
| core_funcform | 10 |
| core_fe | 4 |
| core_method (baseline) | 3 |
| noncore_alt_treatment | 3 |

## Robustness Assessment

### Sign consistency
- **48 of 48** core specifications (100%) produce a positive coefficient, consistent with the baseline sign.
- No sign reversals across any core specification.

### Statistical significance
- **29 of 48** core specifications (60.4%) are significant at the 5% level.
- **32 of 48** core specifications (66.7%) are significant at the 10% level.

### Levels specifications (ODA in levels)
- 38 core levels specs, all positive.
- 28 of 38 (73.7%) significant at 5%.
- Coefficient range: [8.47, 170.51]
- The US-only subsample produces the largest coefficient (170.51) but is not significant (p=0.185) due to small sample (N=3063).

### Log ODA specifications
- 10 core log-ODA specs, all positive.
- 1 of 10 (10.0%) significant at 5%.
- Coefficient range: [0.021, 0.353]
- The log transformation substantially reduces significance. Only the recipient-year FE variant reaches p<0.05 (p=0.003). Most log specs have p>0.25.

### FE structure sensitivity
- pair + year FE (baseline): coef=45.46, p=0.017
- pair + donor-year FE: coef=33.99, p=0.025
- pair FE + macro controls: coef=36.73, p=0.018
- pair + recipient-year FE: coef=51.59, p=0.027
- All four FE structures preserve significance in levels.

### Controls sensitivity
- Adding macro controls (pop, gdp2000, pop_donor, gdp2000_donor) with year FE: coef=40.19, p=0.015 -- significant.
- Leave-one-out control variants: all 4 LOO specs with year FE are significant (p=0.014-0.017). All 4 LOO specs with donor-year FE are significant (p=0.027-0.029).
- Controls sensitivity is very low -- the result is stable across all control configurations.

### Sample sensitivity
- Drop Big 3 recipients (Israel, Egypt, India): coef=20.34, p=0.013 -- significant but smaller magnitude.
- Drop Big 5 recipients (+Pakistan, Indonesia): coef=14.73, p=0.118 -- loses significance.
- US-only donor: coef=170.51, p=0.185 -- large but imprecise.
- Balanced panel: identical to baseline (panel is already balanced).
- The result is partially driven by large aid recipients. Dropping the Big 5 eliminates significance.

### Inference sensitivity (from inference_results.csv)
- **3-way cluster (baseline)**: SE=19.04, p=0.017
- **Pair cluster**: SE=22.52, p=0.044 -- significant at 5%
- **Recipient cluster**: SE=26.90, p=0.094 -- marginal at 10%
- **2-way (donor, recipient)**: SE=19.65, p=0.082 -- marginal at 10%
- **HC1**: SE=22.13, p=0.040 -- significant at 5%

The result is significant at 5% under pair-level and HC1 clustering, borderline under 2-way and recipient-only clustering. The 3-way cluster (baseline) yields the smallest SE, which is unusual -- typically adding cluster dimensions increases SEs.

### Non-core specifications
- 3 specs use p_unvotes_rt_elecex (real-time unvotes decomposition) as the treatment variable: baseline__main_colVII, colVIII, colIX. These decompose the UN voting alignment measure into real-time vs. historical components. All show small, insignificant coefficients (p=0.71-0.88), suggesting the effect is driven by persistent rather than contemporaneous voting alignment.

### Duplicate specifications noted
- rc/fe/pair_plus_year duplicates baseline (already pair+year FE)
- rc/sample/subset/balanced_panel duplicates baseline (panel already balanced)
- rc/joint/balanced_panel_pair_year duplicates baseline (same reason)
- Several joint specs with donor-year FE and LOO controls produce nearly identical results because the controls are absorbed by donor-year FE.

## Top Issues

1. **Log transformation fragility**: The result is robust in levels but largely insignificant in log form. 9 of 10 log-ODA core specs have p>0.05. This is a meaningful vulnerability since log ODA is a common alternative specification for skewed aid data.

2. **Sample sensitivity to large recipients**: Dropping the Big 5 aid recipients eliminates significance (p=0.118). The effect appears concentrated among the largest aid recipients.

3. **Inference method sensitivity**: The 3-way cluster (baseline) produces the smallest standard errors. Under recipient-only or 2-way clustering, the result is borderline. The paper's choice of 3-way clustering happens to be the most favorable for significance.

4. **Real-time unvotes decomposition insignificant**: The colVII-IX decomposition (p_unvotes_rt_elecex) shows no significant effect, suggesting the election-year aid boost responds to persistent rather than contemporaneous voting alignment. This is informative but changes the treatment concept.

5. **Duplicate specifications**: At least 3 specs mechanically duplicate the baseline, reducing the effective unique spec count to approximately 45.

## Recommendations

1. **De-duplicate** the 3 mechanical baseline duplicates.
2. **Expand log-ODA analysis**: Given the sensitivity to functional form, more log specifications with different FE structures should be explored.
3. **Add interaction with aid magnitude**: The Big-5 recipient sensitivity suggests the effect may be heterogeneous by baseline aid levels.
4. **Consider Tobit or zero-inflated models**: Aid data has many zeros and is heavily skewed; functional form choices matter substantially.

## Conclusion

The specification search reveals that the baseline result (positive interaction of UN voting alignment with election years on ODA) is **robust in levels** -- all 48 core specs are positive, and 60.4% achieve p<0.05. The result is stable across FE structures and control configurations. However, it is **sensitive to functional form** (log transformation eliminates significance in most cases), **large recipient composition** (dropping Big 5 recipients weakens significance), and **inference choices** (some clustering approaches yield borderline significance).

Overall assessment: **MODERATE-to-STRONG support** in levels, **WEAK support** in log form. The directional finding is very robust (100% sign consistency), but statistical significance depends on keeping the outcome in levels and retaining large aid recipients in the sample.
