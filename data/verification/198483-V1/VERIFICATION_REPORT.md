# Verification Report: 198483-V1

## Paper
**Title**: National Solidarity Program Impact on Security and Welfare in Afghanistan  
**Journal**: AEJ-Applied  
**Method**: Randomized Controlled Trial with panel data (matched pairs design)  

## Baseline Groups Found

| Group | Claim | Baseline spec_ids | Significant? |
|-------|-------|-------------------|--------------|
| G1 | NSP improves economic welfare (Anderson economic index, male) | baseline/index_Economic_Andr_M | No (p=0.226) |
| G2 | NSP improves public goods provision (Anderson public goods index) | baseline/index_PublicGoods_Andr | Yes (p=0.017) |
| G3 | NSP improves subjective economic perceptions | baseline/index_Economic_Andr_Subj | Yes (p<1e-11) |
| G4 | NSP improves attitudes toward government | baseline/index_Attitudes_Andr_M | Yes (p<0.001) |
| G5 | NSP improves perceived security (male and female) | baseline/index_Security_perc_Andr_M, baseline/index_Security_perc_Andr_F + duplicates | Yes (p<0.001, p=0.001) |
| G6 | NSP affects experienced security incidents | baseline/index_Security_exp_Andr_M + duplicate | No (p=0.974) |

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **71** |
| Baselines | 10 |
| Core tests (non-baseline) | 37 |
| Non-core tests | 24 |
| Invalid | 0 |
| Unclear | 0 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_method | 12 (includes 10 baselines + 2 alt index methods) |
| core_controls | 8 |
| core_sample | 17 |
| core_fe | 4 |
| core_funcform | 3 |
| core_inference | 3 |
| noncore_heterogeneity | 15 |
| noncore_placebo | 5 |
| noncore_alt_outcome | 4 |

## Top 5 Most Suspicious Rows

1. **did/fe/village** (spec_id: did/fe/village): Uses village fixed effects, which absorbs the treatment variable (treatment is assigned at the village level). This fundamentally changes the estimand -- the coefficient of -0.059 (p<0.001) likely reflects within-village time variation rather than the treatment effect. Classified as core_fe with low confidence (0.6).

2. **baseline/security/index_Security_perc_Andr_M** (spec_id: baseline/security/index_Security_perc_Andr_M): This is an exact duplicate of baseline/index_Security_perc_Andr_M -- identical coefficient (0.0795), SE (0.0229), and N (8962). The spec search appears to have recorded the same baseline twice under different spec_id prefixes.

3. **baseline/security/index_Security_perc_Andr_F** (spec_id: baseline/security/index_Security_perc_Andr_F): Same duplication issue as above -- identical to baseline/index_Security_perc_Andr_F.

4. **baseline/security/index_Security_exp_Andr_M** (spec_id: baseline/security/index_Security_exp_Andr_M): Same duplication issue -- identical to baseline/index_Security_exp_Andr_M.

5. **robust/control/add_MLand_owns** vs **robust/control/demographics**: These two specs have identical coefficients (0.01501), identical SEs (0.01038), identical N (5949), and identical controls (MAge, MEducation, MLand_owns). They appear to be the same specification recorded under two different spec_ids. The add_MLand_owns represents the final step of a control progression that matches the demographics specification exactly.

## Key Observations

### Structure of robustness checks
The specification search concentrates almost all robustness checks on the G1 baseline (economic welfare index), which is notably the WEAKEST of the baseline results (not significant at conventional levels). The other baseline groups (G2-G6) have almost no dedicated robustness checks -- only East heterogeneity interaction specs, which are classified as non-core.

### Heterogeneity analysis
15 of 71 specs (21%) are heterogeneity analyses. These include:
- East x Treatment interaction terms across all outcome domains
- Pashtun share and opium production interactions
- Demographic subgroup analyses (age, education, land ownership splits)

All are classified as non-core because heterogeneity is not itself a baseline claim, even though the East/non-East differential is a key paper finding.

### Placebo tests
5 placebo tests regress pre-treatment security incident counts (at various distance radii) on the treatment indicator. These appropriately test for baseline balance in security outcomes and are correctly excluded as non-core.

### Fixed effects variations
The FE variations (none, pair only, village, cluster) all test the economic index outcome. The village FE specification is particularly concerning because treatment is assigned at the village level, so village FE absorbs the treatment effect. The resulting coefficient (-0.059) is negative and significant, which is inconsistent with the baseline estimate.

## Recommendations for Spec-Search Script

1. **Remove duplicate baseline rows**: The baseline/security/* rows (51-53) are exact duplicates of rows 5-7. The script should deduplicate these.

2. **Remove the duplicate control spec**: robust/control/add_MLand_owns and robust/control/demographics produce identical results and should be deduplicated.

3. **Add robustness checks for non-G1 baselines**: The current spec search concentrates almost all robustness on the economic index (G1). The script should generate controls/sample/clustering variations for the other significant outcomes (G2 public goods, G3 subjective economic, G4 attitudes, G5 security perceptions).

4. **Flag the village FE spec more carefully**: Village FE absorbs the treatment in this RCT design. The script should note that this specification is not a valid robustness test and consider excluding it or clearly flagging it.

5. **Consider whether individual outcome components belong**: The 4 individual outcome variables (M7_93z_wins_ln, M8_91z_wins_ln, M9_05z, M9_06z) are components of the composite index. Whether these are core tests or alternative outcomes depends on the analytic perspective -- the paper treats the composite index as the primary estimand.
