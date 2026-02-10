# Verification Report: 113651-V1

## Paper Information
- **Title**: The Causal Effect of Competition on Prices and Quality: Evidence from a Field Experiment
- **Authors**: Matias Busso, Sebastian Galiani
- **Journal**: American Economic Journal: Applied Economics (2018)
- **Total Specifications**: 211

## Baseline Groups

### G1: Competition Reduces Incumbent Retailer Prices (Table 5 Panel A)
- **Claim**: Random entry of new retailers into local markets reduces incumbent retailers' prices. The ITT effect of randomized entry (Hgt0) on the log weighted price index (log_P_index) is negative. IV estimates using observed entry (Hfgt0) instrumented by randomized entry (Hgt0) are roughly 2x the ITT magnitude due to non-compliance.
- **Baseline specs**: 12 specifications (spec_ids 1-12)
- **Expected sign**: Negative
- **Headline coefficient**: -0.020 (SE: 0.007, p = 0.006) -- spec 1, ITT sample1 no controls
- **Outcome**: `log_P_index` (log weighted price index)
- **Treatment**: `Hgt0` (ITT: any randomized entry > 0) / `Hfgt0` (IV: observed entry instrumented by randomized assignment)
- **Table 5, Panel A**: 3 samples x 2 methods (ITT/IV) x 2 control sets (none/full)

**Note**: All 12 baseline specs test the same claim with variation in: (a) sample definition -- sample1 (all retailers with baseline+endline prices, N=399), sample2 (targeted neighborhoods only, N=254), sample3 (incumbents in targeted neighborhoods, N=212); (b) estimation method -- OLS ITT vs IV-2SLS; (c) controls -- none vs full (baseline prices, pre-treatment competition, province dummies, district demographics). The ITT estimates are consistently negative and significant across samples without controls. With full controls, ITT estimates remain significant. IV estimates without controls are significant but IV-with-full-controls specifications (specs 4, 8, 12) lose significance due to inflated standard errors from combining IV with many covariates in small samples.

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baselines)** | **97** | |
| Baselines | 12 | Table 5 Panel A: 3 samples x 2 methods x 2 control sets |
| core_controls | 18 | Partial control sets (base_only, dist_only, no_province) across samples and methods |
| core_sample | 20 | Consumer-level prices (Table 5B no-controls), winsorized/trimmed price indices |
| core_treatment | 12 | Table 7 dose-response: D_H_random_2 and D_H_random_3_4 (number of entrants) |
| core_funcform | 27 | Unweighted price index, non-barcode-change price, level price, pooled product prices with product FE, pooled non-barcode-change with product FE |
| core_inference | 2 | HC1 and HC3 robust standard errors (same point estimate, different SEs) |
| core_fe | 6 | Naive OLS (no IV) on observed entry Hfgt0 -- endogenous but informative comparison |
| **Non-core tests** | **114** | |
| noncore_alt_outcome | 100 | Product switching (Table 4, 16 specs), service quality (Table 8A, 16 specs), client counts (Table 8B, 8 specs), individual product prices (Appendix Table A7, 60 specs) |
| noncore_heterogeneity | 11 | Incumbents vs entrants, targeted vs non-targeted, urban interaction effects |
| noncore_placebo | 3 | CCT retailers in non-experimental districts (Table 8C spillover), non-targeted neighborhoods |
| **Total** | **211** | |

## Detailed Classification Notes

### Core Tests (97 specs including 12 baselines)

**Baselines (12 specs)**: The 12 primary baseline specifications correspond to Table 5, Panel A. Each regresses log_P_index on either Hgt0 (ITT) or Hfgt0 instrumented by Hgt0 (IV), with SEs clustered by mercado (district), with or without the full control set. The control set includes baseline log prices (log_P_index_pret), pre-treatment competition (M_col_pre_treatment), quantity index (Q_index_pret), province dummies, and district-level demographics (beneficiaries, income, education, urban status). Three sample definitions progressively restrict to the most affected retailers.

**Control variations (18 non-baseline core_controls specs)**: These systematically vary which controls are included:
- Base controls only (baseline prices, pre-treatment competition, quantity): 6 specs (ITT + IV across 3 samples)
- District controls only (province dummies, demographics): 6 specs
- Full controls minus province dummies: 6 specs

These are core because they test the same price effect with different conditioning sets, which directly probes the stability of the experimental estimate to control selection. In an RCT, ITT estimates should be stable across control sets (which they largely are), while IV estimates can be more sensitive.

**Sample variations (20 non-baseline core specs)**: These test the same price effect on different samples or with outlier treatment:
- Consumer-reported prices (Table 5B, log_P_s): 4 specs without controls -- same claim (competition reduces prices) measured from the demand side rather than supply side
- Winsorized price index (1% and 5% levels): 6 specs across 3 samples
- Trimmed price index (1% and 5% levels): 6 specs across 3 samples
- These test sensitivity to extreme price observations. All winsorized and trimmed ITT estimates remain significant.

**Treatment variations (12 specs)**: Table 7 specs use the dose-response treatment definition:
- D_H_random_2: indicator for exactly 1 randomly assigned entrant
- D_H_random_3_4: indicator for 2+ randomly assigned entrants
- Tested across 3 samples with and without controls
- These are core because they decompose the same competition-prices treatment effect by dose, testing whether more entrants produce larger price reductions. Generally D_H_random_3_4 (multiple entrants) is consistently significant while D_H_random_2 (single entrant) sometimes loses significance, especially with controls.

**Functional form variations (18 specs)**: These use alternative price index constructions:
- Unweighted price index (log_P_index_nw, Table 6): 9 specs -- removes expenditure-share weighting. The unweighted index shows smaller and often insignificant effects (e.g., spec 31 ITT sample1 p=0.16), suggesting the price effect is concentrated in products with larger expenditure shares.
- Non-barcode-change price index (L_p_nc, Table 6): 6 specs -- prices only for products where the retailer did not switch barcode. These coefficients are near zero and entirely insignificant (p=0.36 to 0.97), suggesting that the price effect operates primarily through product switching (retailers changing to cheaper brands/varieties) rather than direct price reductions on identical products. This is an important finding about the mechanism.
- Level of price index (P_index_level): 3 specs -- log vs level transformation, all significant.

**Inference variations (2 specs)**: HC1 and HC3 heteroskedasticity-robust SEs instead of cluster-robust SEs. Point estimate identical (-0.020); HC1 SE = 0.0087, HC3 SE = 0.0088 vs clustered SE = 0.0072. Results remain significant at p < 0.025 even with more conservative SEs.

**Pooled product prices with product FE (9 core_funcform specs)**: Table 6 pooled individual product-level regressions with product fixed effects (specs 203-211):
- Pooled product prices (log_price_pooled): 6 specs (ITT + IV x 3 samples). Sample2 ITT is significant (p=0.037), but sample1 is insignificant (p=0.68). The product-level pooling with FE does not weight by expenditure shares, which may explain weaker results than the aggregated index.
- Pooled non-barcode-change product prices (log_price_pooled_nbc): 3 ITT specs. All insignificant (p=0.28-0.91), consistent with the aggregated non-barcode-change results showing zero effect.

**Naive OLS / estimator variations (6 specs)**: These regress log_P_index directly on Hfgt0 (observed entry) without instrumenting, to show the OLS vs IV comparison. Naive OLS coefficients are attenuated toward zero compared to IV (e.g., -0.011 vs -0.040 for sample1), consistent with measurement error in observed entry or non-compliance attenuating the OLS estimate.

### Non-Core Tests (114 specs)

**Alternative outcomes -- product switching (16 specs, Table 4)**: Four product availability measures (change_product, change_productl, change_brand, change_variety) each tested with ITT/IV x no controls/full controls. All coefficients are small and insignificant (p = 0.31 to 0.93), showing that while competition affects prices, it does not significantly change the probability of retailers switching products, brands, or varieties. These test a different claim about market adjustment channels.

**Alternative outcomes -- service quality (16 specs, Table 8A)**: Four service quality measures from retailer and consumer surveys:
- CC_hygiene (store cleanliness): 4 specs, all insignificant
- RAS_time (shopping time): 4 specs, all insignificant
- RAS_delivery (delivery quality): 4 specs, all insignificant
- RAS_attention (service attention): 4 specs, marginally significant (p = 0.018-0.048) -- the only quality dimension affected by competition

These test the paper's secondary hypothesis about quality effects of competition, a distinct claim from price effects.

**Alternative outcomes -- client counts (8 specs, Table 8B)**: Business stealing and CCT beneficiary customer share:
- clients_best: 4 specs, all insignificant -- no significant business stealing
- sol_clients: 4 specs, mostly insignificant (ITT p=0.10-0.17, IV p=0.22-0.60)

**Individual product prices (60 specs, Appendix Table A7)**: These decompose the aggregate price effect into 15 individual food products, each tested with ITT sample1 + IV sample1 + ITT sample2 + ITT sample3 = 4 specs per product (60 total). These are non-core because they test the effect on individual product prices (a different, disaggregated outcome) rather than the aggregate price index. Results show substantial heterogeneity:
- Products with significant negative effects (consistent with main finding): Oil, Pasta, Cod, Flour (4 of 15 products)
- Products with positive coefficients (opposite sign): Milk, Sardines, Bread (Bread is actually significantly positive in sample1, p=0.036)
- Products with near-zero effects: Rice, Sugar, Eggs, Chocolate, Beans, Onions, Salami, Chicken
The aggregate price index effect is driven by a subset of products, particularly those with larger expenditure shares.

**Heterogeneity (11 specs)**: Explore whether the price effect varies across subgroups:
- Incumbents vs entrants within sample1/sample2: Incumbents show effects similar to full-sample baselines. Entrant-only specs have very small N (59 or 42) with unreliable estimates.
- Targeted vs non-targeted neighborhoods: Targeted shows significant effect; non-targeted shows no effect (p=0.31, N=136) -- consistent with localized competition effect.
- Urban interaction: 4 specs decomposing the effect by urban/rural status. Neither the main effect nor the interaction is individually significant, though the overall treatment effect absorbs into the decomposition.

**Placebo / spillover tests (2 specs)**: CCT retailers in non-experimental districts (sample4, Table 8C): ITT coef = -0.014, p = 0.31; IV coef = -0.024, p = 0.33. These serve as a placebo test -- competition was not randomized in these districts, so effects should be absent. The null result is consistent with the experimental design's validity.

## Duplicates Identified

- Specs 125-126 (het_ITT/IV_sample2_incumbents) are identical to specs 9 and 11 (T5_ITT/IV_sample3_nocontrols) -- same coefficient, SE, N=212, 70 clusters. This is because sample3 (incumbents in targeted neighborhoods) is the same subsample as "incumbents in sample2."
- Specs 94 and 130 test effectively the same question (spillover to untreated units) from different angles: spec 94 uses CCT retailers in non-experimental districts (sample4), spec 130 uses non-targeted neighborhoods within experimental districts.

After removing the 2 exact duplicates, there are 209 unique specifications.

## Robustness Assessment

The main finding -- that competition from random entry reduces incumbent retailers' prices -- is **moderately robust** across core specifications:

**Strong robustness**:
- ITT estimates are negative and significant (p < 0.05) across all 3 samples, both with and without controls: 6/6 ITT baselines are significant
- Winsorized and trimmed price indices all yield significant ITT estimates with similar magnitudes (-0.019 to -0.029)
- Level instead of log: all significant
- Alternative SE estimators (HC1, HC3): remain significant
- Dose-response (Table 7): D_H_random_3_4 (2+ entrants) is consistently significant across all samples

**Moderate sensitivity**:
- IV estimates with full controls lose significance in all 3 samples (specs 4, 8, 12 have p = 0.17-0.45) -- this reflects inflated SEs from combining IV with many controls in small samples (N=209-379, 70-72 clusters), not sign reversal
- D_H_random_2 (single entrant) loses significance with controls in 3/3 samples, suggesting marginal dose effects are imprecisely estimated
- Consumer-side IV with controls fails catastrophically (spec 53: SE = 178, spec 49: p = 0.84) -- weak instrument problem with consumer-level data

**Important mechanism finding**:
- The non-barcode-change price index (L_p_nc) shows zero effect (6 specs, all p > 0.35), while the overall price index shows significant negative effects. This reveals that the price reduction operates through **product switching** (retailers changing to cheaper brands/varieties when facing competition), not through direct price reductions on identical products. This is consistent with the paper's Table 4 results showing some product switching and with the unweighted price index showing weaker effects.

**Key sensitivity**:
- The unweighted price index (log_P_index_nw) shows ITT effects that are smaller and sometimes insignificant (3 of 9 specs significant at p < 0.05), suggesting the competitive price effect is concentrated in products with larger expenditure shares.
