# Verification Report: 120078-V1

## Paper
**Can Information Reduce Ethnic Discrimination? Evidence from Airbnb** (AEJ: Applied Economics)

## Baseline Groups

### G1: Ethnic price gap
- **Claim**: Ethnic minority hosts on Airbnb charge lower prices than non-minority hosts, as measured by the coefficient on a minority dummy (minodummy) in a regression of log listing price (log_price) on minority status with city-wave and/or neighborhood fixed effects.
- **Expected sign**: Negative
- **Baseline spec_ids**: baseline_table2_col1, baseline_table2_col2, baseline_table2_col3, baseline_table2_col4
- **Notes**: These correspond to Table 2, columns 1-4 in the paper. Column 4 (block + city-wave FE + full property controls) is the paper preferred specification. All four baselines are grouped together because they estimate the same estimand (the ethnic price gap) under progressively richer conditioning.

## Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **77** |
| Baselines | 4 |
| Core tests (incl baselines) | 67 |
| Non-core tests | 10 |
| Invalid | 0 |
| Unclear | 0 |

### Core category breakdown (including baselines)

| Core category | Count |
|---------------|-------|
| core_controls | 20 |
| core_sample | 32 |
| core_fe | 7 |
| core_inference | 5 |
| core_funcform | 3 |
| core_method | 0 |

Note: The 4 baselines are included in the counts above (2 in core_controls, 2 in core_fe). Total core = 67, non-core = 10.

### Non-core breakdown

| Non-core category | Count |
|-------------------|-------|
| noncore_heterogeneity | 6 |
| noncore_placebo | 2 |
| noncore_alt_treatment | 2 |

## Classification Rationale

### Core tests (67 specs including baselines)
The vast majority of specifications are straightforward robustness checks of the baseline claim. They use the same outcome (log_price or a monotonic transformation like ihs_price, price, log_price_std) and the same treatment (minodummy), varying only:
- **Fixed effects structure** (7 specs incl 2 baselines): no FE, wave FE, neighborhood FE, city FE, high-dimensional FE, plus block+city-wave FE baselines
- **Controls** (20 specs incl 2 baselines): leave-one-out analysis of control groups and individual controls, progressive control build-up, reviews polynomial
- **Sample restrictions** (32 specs): by review count (5), wave period (6), city subsample (6), drop-city (4), property type (2), outlier handling (4), price market (2), bedroom count (3)
- **Inference** (5 specs): different clustering levels (newid, blockID, hoodcityID, city, robust HC)
- **Functional form** (3 specs): IHS price, price levels, standardized log price

### Non-core: Heterogeneity (6 specs)
The heterogeneity specifications (robust/heterogeneity/*) include interaction terms (e.g., minodummy * superhost). The reported coefficient on minodummy in these models represents the effect for the **reference group only** (e.g., non-superhosts), not the average effect. This makes them not directly comparable to the baseline, which estimates an average effect across all groups. They are classified as noncore_heterogeneity.

### Non-core: Alternative treatments (2 specs)
- robust/treatment/arabic_african: Uses arabic_african instead of minodummy. This restricts to a specific ethnic subgroup, changing the causal object.
- robust/treatment/continuous_minority: Uses nber_mino_names (count of minority names) instead of a binary indicator. This changes both the treatment definition and the scale of the coefficient.

### Non-core: Placebo tests (2 specs)
- robust/placebo/random_treatment: Uses a randomly assigned fake minority indicator (fake_mino). This is a validity check, not a test of the core claim.
- robust/placebo/picture_change: Uses change_pics as the outcome instead of price. This tests whether minority status predicts picture changes (it should not), serving as a placebo.

## Top 5 Most Suspicious Rows

1. **robust/control/none**: This specification is numerically identical to baseline_table2_col1 (same formula: log_price ~ minodummy | citywaveID, same sample, same clustering). The coefficient is exactly the same (-0.1686). This is a duplicate, not an independent robustness check. However, it is not invalid -- just redundant.

2. **robust/cluster/city**: Clustering at the city level with only ~12 cities yields very few clusters, which may produce unreliable standard errors. The coefficient is unchanged (same point estimate as other clustering specs), but the p-value (0.0006) is notably larger than with finer clustering (p=0.0). The inference is questionable but the coefficient itself is valid.

3. **robust/sample/city_barcelona**: The coefficient is +0.004 (positive, p=0.88), opposite to the expected sign. This is not suspicious per se -- Barcelona may simply not exhibit the discrimination pattern -- but it stands out as one of only 1 positive coefficient among 77 specs.

4. **robust/sample/city_berlin**: The coefficient is -0.002 (essentially zero, p=0.94), similarly showing no discrimination effect in Berlin. Again not invalid, but notable.

5. **robust/heterogeneity/verified**: The main effect coefficient (-0.211) is substantially larger in magnitude than any baseline, suggesting that the reference group (unverified hosts) may have unusually high discrimination. This is expected given the interaction model structure but makes the coefficient not directly comparable to baselines.

## Recommendations

1. **Remove the duplicate**: robust/control/none is identical to baseline_table2_col1 and should be flagged or removed to avoid double-counting in robustness summaries.

2. **Failed specifications**: The script attempted to run panel/fe/unit (unit FE), panel/fe/twoway (unit + wave FE), robust/treatment/black (Black hosts only), panel/method/first_diff (first differences), and four structural/rho_* specifications. These all failed silently. The first-differences and structural model specs would have been valuable robustness checks. Consider debugging these failures.

3. **Heterogeneity interpretation**: If heterogeneity specs are to be included as core tests, the interaction coefficient (e.g., mino_superhost) should be summed with the main effect to get the total effect for each subgroup, rather than reporting only the main effect. Alternatively, run the baseline regression separately on each subgroup.

4. **City-level clustering**: The robust/cluster/city specification should carry a warning about few-cluster inference. Consider wild cluster bootstrap instead.

5. **Review-count subsamples**: The review-count sample splits (Table 3 style) use a slightly different sample base (drawn from df rather than df_main for the no-reviews group) and different controls. This is appropriate for the paper secondary claim about information reducing discrimination, but makes them slightly less comparable to the Table 2 baselines.
