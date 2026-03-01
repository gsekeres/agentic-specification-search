# Verification Report: 173341-V1

## Paper
Bobonis, Gertler, Gonzalez-Navarro & Nichter (2022), "Vulnerability and Clientelism", AER

## Baseline Groups Found

### G1: Cisterns treatment effect on clientelist requests (Table 3)
- **Baseline spec_run_ids**: 173341-V1__G1__run001, 173341-V1__G1__run002
- **Baseline spec_ids**: baseline__table3_col3, baseline__table3_col4
- **Claim**: ITT effect of cistern receipt on requests for private goods from politicians (ask_private_stacked), in semi-arid Brazilian municipalities. Household-level RCT with municipality FE and year FE, stacked 2012-2013.
- **Baseline coefficient (treatment)**: -0.0296 (SE=0.0126, p=0.0185, N=4288, R2=0.073)
- **Expected sign**: Negative (cistern receipt reduces clientelist requests)

### G2: Cisterns treatment effect on incumbent votes (Table 4)
- **Baseline spec_run_id**: 173341-V1__G2__run046
- **Baseline spec_id**: baseline__table4_col1
- **Claim**: ITT effect of share of cistern-treated in voting section on incumbent mayor votes. Section-level analysis with voting location FE.
- **Baseline coefficient (tot_treat_by_section_2)**: -0.1012 (SE=0.0581, p=0.0829, N=866, R2=0.945)
- **Expected sign**: Negative (treatment reduces incumbent vote share)

## Counts

| Metric | Count |
|--------|-------|
| Total rows in specification_results.csv | 51 |
| Valid (run_success=1) | 51 |
| Invalid (run_success=0) | 0 |
| Core tests (is_core_test=1) | 51 |
| Non-core | 0 |
| Baseline rows | 3 |
| Inference variants (inference_results.csv) | 6 |

## Category Counts

| Category | Count |
|----------|-------|
| core_method (baseline + design) | 7 |
| core_controls | 16 |
| core_funcform | 16 |
| core_sample | 7 |
| core_fe | 5 |

## Robustness Assessment

### G1: Clientelist requests (45 specifications)

#### Sign consistency
- **38 of 45** specifications (84.4%) produce a negative coefficient, consistent with the baseline sign.
- 7 specifications produce positive coefficients (sign flips), all statistically insignificant:
  - 4 rows with engagement controls (voted/all variants): Adding treatment x voted or treatment x all-engagement interactions flips the sign on the treatment main effect (coefs ~+0.04, p>0.2). These are cross-product rows combining engagement controls with the Col 4 baseline.
  - 1 row: rainfall_only on 2013-only subsample (coef~0, p=0.97) -- rainfall shock has no effect in 2013.
  - 1 row: treatment_by_year with engagement controls (treat_2012 coef=+0.047, p=0.22).
  - 1 row: ask_nowater_private outcome with engagement controls (coef~0, p=0.92).

#### Statistical significance
- **19 of 45** specifications (42.2%) are significant at p < 0.05.
- **28 of 45** specifications (62.2%) are significant at p < 0.10.
- Many insignificant specs arise from year subsamples (N=1621 or N=2667) or from adding engagement controls that absorb the treatment effect.

#### Controls sensitivity
- Adding engagement controls (association membership, political participation, voted) substantially attenuates and sometimes flips the treatment effect. This is the main vulnerability for G1.
- Core control sets without engagement (none, mun_fe_only, mun_fe_year) preserve the sign and significance.

#### Sample sensitivity
- 2012-only subsample: coef=-0.027, p=0.106 (insignificant)
- 2013-only subsample: coef=-0.033, p=0.026 (significant)
- Winsorization has no impact (binary outcome).

### G2: Incumbent votes (6 specifications)

#### Sign consistency
- **5 of 6** specifications produce a negative coefficient.
- 1 sign flip: simple difference in means (no FE) produces coef=+0.028, p=0.89 -- location FE are essential for this analysis.

#### Statistical significance
- **1 of 6** significant at p < 0.05 (dropping study share control).
- **2 of 6** significant at p < 0.10 (baseline + dropping study share).
- This result is fragile: it depends on the specific control set and location FE.

### Inference sensitivity (from inference_results.csv)

**G1 (baseline__table3_col3):**
- Cluster(b_clusters): SE=0.0126, p=0.0185 (baseline)
- HC1 (robust): SE=0.0112, p=0.0081 -- more significant than baseline
- Cluster(municipality): SE=0.0144, p=0.0468 -- still significant at 5%

**G2 (baseline__table4_col1):**
- Cluster(location_id): SE=0.0581, p=0.0829 (baseline)
- HC1 (robust): SE=0.0623, p=0.1045 -- not significant at 10%
- Cluster(municipality): SE=0.0464, p=0.0412 -- significant at 5%

G1 inference is robust. G2 inference is sensitive: robust SEs make it insignificant, but municipality-level clustering (fewer clusters, but matching the paper's wild bootstrap) makes it significant.

## Top Issues

1. **Cross-product spec_id duplication**: The runner produced combinations of spec_id variations with year subsamples and engagement controls, resulting in 45 G1 rows (vs ~21 planned in the surface). While each spec_run_id is unique, the same spec_id appears multiple times within G1 (e.g., rc/controls/sets/mun_fe_year_engagement_assoc appears 4 times). This inflates the specification count but does not invalidate any row.

2. **Engagement controls attenuate treatment effect**: Adding controls for political engagement (voted, association membership, presidential association) attenuates or flips the treatment coefficient. These are plausibly mediators (cisterns reduce engagement which reduces clientelist requests), so controlling for them is a "bad control" issue. The surface notes this but the runner still included them.

3. **G2 fragility**: The electoral outcome result (Table 4) is marginally significant at baseline and loses significance in most robustness checks. With only 6 specifications and small effective sample, this claim has weak support.

4. **No outcome/treatment concept drift**: All G1 rows maintain the ask_private_stacked outcome (except ask_nowater_private, which is an operationalization variant) and cistern-related treatment variables. All G2 rows maintain incumbent_votes_section and tot_treat_by_section_2. No concept drift detected.

## Recommendations

1. The engagement control specifications should be flagged as potentially problematic (bad controls / mediators). A sensitivity note in the surface would help.
2. G2 would benefit from additional robustness checks (alternative treatment scaling, bandwidth for municipality matching), though the small sample limits what can be done.
3. Cross-product specifications should ideally have unique spec_ids (e.g., appending sample info to the control spec_id) to avoid confusion.

## Conclusion

**G1 (clientelist requests)**: The baseline result (treatment reduces private good requests by ~3pp) receives **MODERATE support**. The point estimate is stable across most specifications (median coef = -0.027), but significance is sensitive to subsample and controls. Engagement controls are a known confounder. Inference is robust across SE methods.

**G2 (incumbent votes)**: The baseline result (treatment reduces incumbent votes) receives **WEAK support**. The effect is marginally significant at baseline (p=0.083) and loses significance in most robustness checks. Only 1 of 6 specs is significant at p<0.05.

Overall assessment: **MODERATE support** for the paper's main claim (G1) and **WEAK support** for the electoral outcome claim (G2).
