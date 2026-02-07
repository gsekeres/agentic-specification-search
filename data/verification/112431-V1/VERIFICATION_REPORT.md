# Verification Report: 112431-V1

## Paper Information

- **Paper**: Electoral Accountability and Corruption: Evidence from the Audits of Local Government
- **Authors**: Ferraz and Finan (2011)
- **Journal**: American Economic Review
- **Paper ID**: 112431-V1

## Baseline Groups

### G1: Reelection incentives reduce corruption

- **Claim**: First-term mayors who face reelection incentives engage in less corruption (lower proportion of audited federal funds associated with corruption) than term-limited second-term mayors.
- **Expected sign**: Negative (first-term mayors have LESS corruption)
- **Baseline spec IDs**: `baseline` (no FE, coef=-0.0188, p=0.047), `baseline_fe` (state FE, coef=-0.0193, p=0.056)
- **Outcome**: `pcorrupt` (proportion of federal funds associated with corruption)
- **Treatment**: `first` (indicator for first-term mayor who can seek reelection)

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **96** |
| Baseline specs | 2 |
| Core tests (non-baseline) | 78 |
| Non-core specs | 16 |
| Invalid | 0 |
| Unclear | 0 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_controls | 23 | Leave-one-out, control progression, control set variations |
| core_sample | 31 | Lottery drops, state drops, pop/urban splits, trimming, winsorizing, experienced-only, male mayors |
| core_inference | 6 | Classical SE, HC1/HC2/HC3, state cluster, lottery cluster |
| core_funcform | 6 | Log/IHS transform of pcorrupt, binary corruption (LPM), count outcomes as alternative measures |
| core_method | 13 | RDD (7 specs), probit, negbin, poisson, logit, population/transfer weights |
| core_fe | 1 | Baseline with state fixed effects |
| noncore_alt_outcome | 4 | pmismanagement, dcorrupt_desvio, dcorrupt_licitacao, dcorrupt_superfat |
| noncore_heterogeneity | 8 | Interactions with political competition, media, judiciary, population, urbanization, income, PT party, same-party governor |
| noncore_placebo | 4 | Predetermined outcomes: resources audited, population, income, urbanization |

## Top 5 Most Suspicious / Borderline Rows

1. **robust/outcome/ncorrupt** (outcome=ncorrupt, classified core_funcform, confidence=0.7): Count of corruption violations rather than proportion. This is arguably a different outcome measure rather than a functional form variation. I classified it as core because it measures the same underlying construct (corruption), but the interpretation differs (count vs. share). A stricter classification would mark this as noncore_alt_outcome.

2. **robust/outcome/dcorrupt** (outcome=dcorrupt, classified core_funcform, confidence=0.65): Binary indicator for any corruption. This is an extensive-margin measure compared to the baseline intensive-margin measure (pcorrupt). The estimand changes from "how much corruption" to "whether any corruption." Borderline case; classified as core because it tests the same broad hypothesis.

3. **robust/estimation/probit** (outcome=any_corrupt, classified core_method, confidence=0.7): Uses a different outcome variable name (any_corrupt vs dcorrupt) and a different estimation method. The outcome appears to be a binary corruption indicator similar to dcorrupt. The combination of changed outcome AND changed method makes this borderline.

4. **robust/estimation/negbin** and **robust/estimation/poisson** (outcome=ncorrupt, classified core_method, confidence=0.7): These use count models for the number of corruption violations. They combine a different outcome (ncorrupt instead of pcorrupt) with a different estimation method. The double departure from baseline makes these borderline.

5. **robust/heterogeneity/male_mayors** (outcome=pcorrupt, classified core_sample, confidence=0.85): This was classified as a subsample restriction rather than heterogeneity because the controls_desc says "Subsample: male_mayors" and N=453, suggesting it drops female mayors rather than adding an interaction term. If it actually uses an interaction, it should be reclassified as noncore_heterogeneity.

## Verification Notes

### What went well
- The specification search is well-structured with clear spec_id naming conventions.
- Baseline specs are correctly identified and tagged.
- The treatment variable (first) is consistent across all 96 specifications.
- The classification into robustness categories in SPECIFICATION_SEARCH.md aligns well with the actual spec_tree_paths.

### Issues identified
1. **Inference variation coefficient mismatch**: The inference-variation rows (robust/cluster/* and robust/se/*) all share the same coefficient (-0.01756) which differs from both baseline (-0.0188) and baseline_fe (-0.0193). This suggests these specs used a different control set than either baseline. The coefficient should match one of the baselines if only the standard errors changed.

2. **Borderline outcome measures**: Several specifications classified as core_funcform or core_method use different outcome variables (ncorrupt, dcorrupt, any_corrupt). These measure the same broad concept (corruption) but with different operationalizations. A more conservative approach would classify all non-pcorrupt outcomes as noncore_alt_outcome.

3. **Control set ambiguity**: The controls_desc field for some robustness specs says things like "Dropped pref_masc" without specifying the base control set. It appears the leave-one-out specs drop from the full control set, while the baseline has no controls.

### Recommendations
- Clarify which control set the inference-variation specs use, since their coefficient does not match either baseline.
- Consider splitting the borderline outcome-measure specs (ncorrupt, dcorrupt) into their own baseline group if the paper treats them as separate claims.
- The control progression specs (robust/control/*) should specify their base set more clearly.
