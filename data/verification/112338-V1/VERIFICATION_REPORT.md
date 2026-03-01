# Verification Report: 112338-V1

## Paper
Duggan & Scott Morton (2010), "The Effect of Medicare Part D on Pharmaceutical Prices and Utilization", AER

## Baseline Groups Found

### G1: Price effect (Table 2)
- **Baseline spec_run_ids**: 112338-V1_spec_001 (baseline), 112338-V1_spec_002 (table2_col1), 112338-V1_spec_003 (table2_col2), 112338-V1_spec_004 (table2_col3)
- **Claim**: Effect of Medicare Part D market share (mcar0203mepsrx) on log change in drug price-per-day from 2003 to 2006 (lppd0603)
- **Baseline coefficient**: -0.1364 (SE not reported in scalar column, p=0.015, N=517)
- **Expected sign**: Negative (higher Medicare share -> lower price growth)

### G2: Quantity effect (Table 3)
- **Baseline spec_run_ids**: 112338-V1_spec_036 (baseline), 112338-V1_spec_037 (table3_col1), 112338-V1_spec_038 (table3_col2), 112338-V1_spec_039 (table3_col3)
- **Claim**: Effect of Medicare Part D market share (mcar0203mepsrx) on log change in drug doses from 2003 to 2006 (ldoses0603)
- **Baseline coefficient**: 0.3892 (p=0.218, N=517)
- **Expected sign**: Positive (higher Medicare share -> more utilization growth)

## Counts

| Metric | Count |
|--------|-------|
| Total rows in specification_results.csv | 56 |
| Valid (run_success=1) | 22 |
| Invalid (run_success=0) | 34 |
| Core tests (is_core_test=1) | 22 |
| Non-core (valid) | 0 |
| Baseline rows | 8 |
| Inference variants (inference_results.csv) | 5 (all failed) |

## Category Counts

| Category | Count |
|----------|-------|
| core_method (baseline) | 8 |
| core_sample | 8 |
| core_funcform | 6 |
| invalid_failure | 34 |

## Robustness Assessment

### G1: Price Effect

#### Sign consistency
- **11 of 11** valid specifications (100%) produce a negative coefficient, consistent with the baseline sign.

#### Statistical significance
- **11 of 11** specifications (100%) are significant at the 5% level.
- Coefficient range: [-0.2429, -0.1272]
- N range: [200, 548]

#### Key findings by axis
- **Sample restrictions**: Dropping cancer exclusion (include_cancer, no_trim_include_cancer) and using only top-292 drugs all preserve significance.
- **Treatment form**: Alternative Medicare share measures (spending share mcar0203mepspd, self-pay decomposition mcself0203mepsrx) yield significant results. The self-pay decomposed coefficient is larger in magnitude (-0.225 to -0.243).
- **Joint specs**: All successful joint specs (spending_share_no_cancer, no_trim_include_cancer) remain significant.

### G2: Quantity Effect

#### Sign consistency
- **11 of 11** valid specifications (100%) produce a positive coefficient, consistent with the baseline sign.

#### Statistical significance
- **0 of 11** specifications (0%) are significant at the 5% level.
- **2 of 11** specifications (18.2%) are significant at the 10% level (both from drop_generic_facing and its joint variant, p=0.054).
- Coefficient range: [0.2651, 0.5168]
- N range: [200, 548]

#### Key findings
- The quantity effect is consistently positive but never reaches conventional significance, mirroring the paper's Table 3 results. The closest to significance are the specifications dropping drugs facing generic competition (p=0.054).

### Inference sensitivity (inference_results.csv)
- All 5 inference variants (HC2, HC3, clustered by therapeutic category for both baselines) **failed** due to proprietary data dependencies. No inference robustness assessment is possible.

## Top Issues

1. **Extreme failure rate**: 34 of 56 specifications (60.7%) fail because the replication package lacks proprietary IMS Health data. Key variables (imsgrouprank03, meps0203scripts for weighting, therapeutic category identifiers) are not in the distributed dataset. This severely limits the specification surface.

2. **All inference variants fail**: The 5 inference alternatives all fail, preventing any assessment of inference robustness (HC2/HC3 vs HC1, clustering by therapeutic category).

3. **Missing specification axes**: Due to data constraints, no controls leave-one-out, no weighting alternatives (unweighted), no FE specifications, and no additional sample cuts (top300, top500, OTC filter relaxation) could be executed. The executed core specs are essentially only the baselines plus a few sample and treatment-form variations.

4. **Duplicate specifications**: Several specs produce identical results (include_cancer = table2_col3/table3_col3 because the baseline already drops cancer; no_trim_include_cancer = table2_col2/table3_col2 because col2 uses the full untrimmed sample). The effective unique specification count is lower than the reported core count.

5. **No explore/* specs executed**: The surface planned explore/outcome/log_sales_change and explore/outcome/log_doses_change but these do not appear in the results (likely blocked by data availability).

## Recommendations

1. **Investigate whether additional IMS variables can be reconstructed** from the available data. The high failure rate makes the specification surface very thin.
2. **Controls axis is entirely blocked**: Even basic LOO (drop yrs03onmkt, drop anygen) fails. The replication script may need to be restructured to access these variables from whatever data is available.
3. **Inference variants should be re-attempted** using only the variables available in the replication package.
4. **De-duplicate** joint specs that mechanically reproduce baseline variants (e.g., include_cancer when the baseline sample already includes cancer drugs at an earlier step).

## Conclusion

The specification search for G1 (price effect) confirms that the baseline result is **robust across all executable specifications** -- 11/11 significant at 5% with consistent negative sign. However, the surface is severely constrained by proprietary data limitations (only 22 of 56 specs succeed).

The G2 (quantity effect) baseline is not significant (p=0.218) and **no specification achieves conventional significance**, consistent with the paper's own characterization of the quantity results as suggestive.

Overall assessment: **STRONG support for G1** (price effect, all executable specs significant), **WEAK support for G2** (quantity effect, consistently positive but never significant). The assessment is limited by the high failure rate due to proprietary data.
