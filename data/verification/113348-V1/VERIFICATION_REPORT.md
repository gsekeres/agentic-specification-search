# Verification Report: 113348-V1

## Paper Information
- **Title**: Technology, International Trade, and Pollution from U.S. Manufacturing
- **Authors**: Arik Levinson
- **Journal**: American Economic Review (2009)
- **Total Specifications**: 100

## Baseline Groups

### G1: Technique Effect (All Four Pollutants Average)
- **Claim**: The technique effect (within-industry cleaner production) accounts for approximately 78% of the total reduction in pollution from U.S. manufacturing between 1987 and 2001, dominating the composition effect (shifts in production across industries).
- **Baseline spec**: `baseline/all_four`
- **Expected sign**: Positive (technique fraction > 0, and specifically > composition fraction)
- **Baseline coefficient**: 0.7795 (technique fraction; no SE or p-value -- this is an accounting decomposition)
- **Outcome**: `technique_fraction_all_four`
- **Treatment**: `decomposition` (deterministic accounting identity)
- **Table 2, all four pollutants average**

### G2: Technique Effect by Individual Pollutant
- **Claim**: The technique effect dominates for each individual pollutant: SO2 (77%), NOx (60%), CO (76%), VOC (95%).
- **Baseline specs**: `baseline/SO2`, `baseline/NOx`, `baseline/CO`, `baseline/VOC`
- **Expected sign**: Positive
- **Baseline coefficients**: SO2=0.773, NOx=0.598, CO=0.758, VOC=0.948
- **Outcome**: `technique_fraction_[pollutant]`
- **Table 2, individual pollutant rows**

**Note**: This paper uses an accounting decomposition (not regression), so there are no standard errors or p-values for most specifications. The decomposition identity splits total pollution change into scale (overall output growth), composition (shifts across industries), and technique (within-industry emission intensity changes). The technique fraction is defined as (technique change) / (total pollution change), which can exceed 1 or be negative when composition effects offset or reinforce technique effects.

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baselines)** | **57** | |
| core_decomposition | 5 | 5 baselines: all_four average + SO2, NOx, CO, VOC individual pollutants |
| core_sample | 42 | Time period variations (16), leave-one-out industry exclusion (20), drop dirty/clean industries (3), include dropped industries (1), drop recession year (1), drop bottom 10% clean (1) |
| core_measurement | 9 | Alternative deflator (industry PPI), winsorize IPPS (3 levels), output measure (domestic consumption, value added), log decomposition, chained decomposition, median industry |
| core_inference | 1 | Bootstrap confidence interval (200 replications) |
| **Non-core tests** | **43** | |
| noncore_heterogeneity | 26 | Sub-period decompositions (3), rolling windows (11), sector subsets (8), trade exposure (2), size splits (2) |
| noncore_alt_outcome | 13 | Trade composition effects (3), Table 4 composition-only effects (5), cross-section regressions (4), scale trend (1) |
| noncore_alt_model | 4 | OLS log trend (2, one appears duplicate), pooled OLS panel, FE panel -- parametric regression approaches rather than decomposition |
| **Total** | **100** | |

## Detailed Classification Notes

### Core Tests (57 specs including 5 baselines)

**Baselines (5 specs)**: The five primary baseline specifications correspond to Table 2 of the paper. Four report the technique fraction for individual pollutants (SO2, NOx, CO, VOC), and one averages across all four. All use the 1987-2001 period with the full sample of ~450 four-digit SIC industries. The decomposition is: total pollution change = scale effect + composition effect + technique effect. The technique fraction is technique/(scale+composition+technique).

**Time period variations (16 core_sample specs)**: These vary the start and/or end year of the decomposition while keeping the same methodology and industry set:
- Truncating the end date from 2001 back to 1995: technique fraction ranges from 0.615 (1987-1995) to 0.808 (1987-1999), all showing technique dominance.
- Shifting the start date forward (1988-2001, 1990-2001): technique fraction 0.699-0.759, still technique-dominant.
- Extending the sample earlier (1985-2001, 1980-2001, 1975-2001, 1972-2001): technique fraction decreases from 0.798 to 0.624 as the window widens, but still exceeds 50% in all cases.
- Historical period (1972-1987): technique fraction 0.631, showing technique dominance even before the main sample period.
- Alternative windows (1980-1990, 1990-2000, first/second half): all show technique fraction 0.63-0.83.
- These are classified as core because they test the same decomposition methodology on the same outcome with different time windows, directly assessing temporal robustness of the baseline claim.

**Leave-one-out industry exclusion (20 core_sample specs)**: Drop one 2-digit SIC industry at a time (SIC 20-39). The technique fraction ranges from 0.752 (dropping SIC 33, Primary Metals) to 0.830 (dropping SIC 26, Paper). No single industry drives the baseline result. These are core because they test sample sensitivity with the same decomposition methodology.

**Other sample variations (3 core_sample specs)**: Dropping top 10% or 25% dirtiest industries, dropping bottom 10% cleanest, including previously dropped industries, dropping 1991 recession. The top-25%-dirty drop is notable: technique fraction jumps to 1.46, meaning composition effects actually increase pollution in the remaining clean industries. The include_dropped9 and drop_bottom10_clean specs are identical to baseline (suggesting no industries were actually excluded or the cleanest industries contribute negligibly).

**Measurement variations (9 core_measurement specs)**:
- **Industry-specific PPI deflator**: This is the single largest sensitivity. Using industry-specific PPI instead of the GDP deflator reduces the technique fraction from 0.78 to 0.39. The paper discusses this extensively -- industry-specific deflators absorb relative price changes that the baseline attributes to technique change.
- **Winsorize IPPS** at 1%, 5%, 10%: technique fraction increases from 0.780 to 0.806 to 0.896 -- extreme pollution intensities pull the decomposition toward technique.
- **Domestic consumption** instead of shipments: technique fraction rises to 0.903 (trade-adjusted output reduces apparent composition effects).
- **Value added** instead of shipments: technique fraction 0.777, very close to baseline.
- **Log decomposition**: multiplicative rather than additive decomposition yields 0.819, slightly higher.
- **Chained decomposition**: year-by-year chaining produces scale_comp = 0.129 (only 14 observations, one per year-pair).
- **Median industry**: 0.191 (median industry-level pollution change), a fundamentally different summary statistic.

**Inference (1 spec)**: Bootstrap CI with 200 replications: mean = 0.785, SE = 0.053, 95% CI = [0.669, 0.883]. This is the only specification providing formal statistical uncertainty, confirming the technique fraction is statistically distinguishable from zero and from 0.5 (composition dominance).

### Non-Core Tests (43 specs)

**Heterogeneity (26 specs)**: These decompose the effect by subgroup or time window and are non-core because they assess where the effect is larger/smaller rather than providing alternative estimates of the same aggregate quantity:
- **Sub-periods (3)**: 1987-1991 yields 4.03 (unstable, near-zero denominator), 1991-1996 yields 0.63, 1996-2001 yields 0.83.
- **Rolling 4-year windows (11)**: Show substantial variation (0.34 to 4.03). Early windows (1987-1991, 1988-1992) are unstable because aggregate pollution barely changed. Later windows (1994-2001) consistently show technique fractions of 0.75-0.93.
- **Sector subsets (8)**: Heavy industry (0.86), light manufacturing (0.67), durables (0.97), nondurables (0.67), chemicals (0.66), metals (0.88), machinery (1.53 -- composition effect reversed), food (0.94). All except machinery show technique dominance.
- **Trade exposure (2)**: High-trade industries (0.80) vs low-trade (0.76) -- technique effect is slightly larger for trade-exposed industries, weakly consistent with pollution-haven effects on composition.
- **Size splits (2)**: Large (0.78) vs small (0.77) -- virtually identical, ruling out size-driven heterogeneity.

**Alternative outcomes (13 specs)**: These measure different quantities:
- **Trade composition (3)**: All imports (composition change = 0.74), non-OECD imports (1.44 -- dirty-ward shift), exports (0.84). These test whether trade patterns shifted toward dirtier goods (pollution haven hypothesis), a different claim from the baseline technique-dominance finding.
- **Table 4 composition effects (5)**: Report the composition-only effect for each pollutant and the average. These are the complement of the technique fraction -- showing composition accounted for only -0.025 to -0.109 of total change.
- **Cross-section regressions (4)**: OLS regressions of log output growth on change in import penetration. All four pollutant versions produce identical coefficients (-0.860, t=-4.94), suggesting they use the same underlying regression with N=318 industries. These test the trade-composition channel via regression rather than decomposition.
- **Scale trend (1)**: OLS log trend of real output, documenting that output grew ~5% per year.

**Alternative models (4 specs)**: These use regression-based approaches rather than decomposition:
- OLS log trend of predicted pollution (2 specs, appear identical with same coefficients): growth rate of 1.18% per year (R2=0.79).
- Pooled OLS panel: year trend coefficient 0.019 (SE=0.008, t=2.42) on N=5325.
- Panel FE: year trend coefficient 0.019 (SE=0.0006, t=31.3) on N=5325 with industry FE. The dramatic reduction in SE with FE shows strong within-industry trends.

These are non-core because they use a fundamentally different methodology (regression) to address a related but distinct question (is there a time trend in pollution intensity?) rather than providing alternative decomposition estimates.

## Potential Duplicates and Anomalies

1. **robust/sample/include_dropped9** produces coefficients identical to baseline/all_four (technique_frac = 0.7795). This suggests no industries were actually excluded in the baseline, making this specification redundant.
2. **robust/sample/drop_bottom10_clean** also produces coefficients identical to baseline/all_four, suggesting the cleanest 10% of industries contribute negligibly to the aggregate decomposition.
3. **robust/sample/drop_recession_1991** is identical to baseline, likely because the decomposition only uses endpoint years (1987 and 2001), not annual data.
4. **robust/model/ols_log_trend** and **robust/model/ols_log_trend_1972_2001** produce identical coefficients (slope=0.01183, R2=0.788) despite claiming different sample periods. The description says 1972-2001 but N=15 suggests 1987-2001.
5. **Cross-section regressions** (xs_regression_so2, no2, co, voc) all produce identical coefficients (-0.860, SE=0.174), suggesting these are the same regression repeated for four pollutant labels. The sample description varies by pollutant but N=318 is identical.
6. **Rolling window 1987-1991** is identical to **subperiod 1987-1991** (both technique_frac = 4.028), which is expected since the 4-year rolling window starting in 1987 spans the same period as the 1987-1991 sub-period.

## Robustness Assessment

The main finding -- that the technique effect dominates the composition effect in explaining U.S. manufacturing pollution reduction -- is **robust** across most core specifications:

- **G1 (technique fraction, all four pollutants)**: The baseline estimate of 0.78 ranges from 0.62 (1987-1995 truncation, extended 1972-2001 period) to 0.90 (domestic consumption, 10% winsorize) across core specifications. The technique fraction exceeds 0.5 in all time-period variations, all leave-one-out industry exclusions, and all measurement alternatives except one.

- **G2 (pollutant-specific technique fractions)**: SO2 (0.77), CO (0.76), and VOC (0.95) all show strong technique dominance. NOx (0.60) is the weakest but still shows technique exceeding composition.

- **Bootstrap CI**: [0.669, 0.883] clearly excludes 0.5, providing formal statistical evidence for technique dominance.

**Key sensitivity**: The industry-specific PPI deflator reduces the technique fraction to 0.39, the only specification where composition effects exceed technique. This occurs because industry-specific deflators remove relative price changes that the GDP-deflator-based decomposition attributes to technique change. The paper discusses this as a measurement issue: if industry i's prices fell faster than GDP deflator, the GDP-deflator approach overstates real output growth in that industry, inflating the technique effect. This is not so much a robustness failure as a fundamental measurement question about what constitutes "real" output change.

**Non-core heterogeneity**: The technique effect is present across virtually all sectors (except machinery, where composition effects reverse) and in both high- and low-trade-exposure industries, suggesting the finding is not driven by a single sector or by trade-related composition shifts. Short time windows (1987-1991, 1988-1992) produce unstable fractions due to near-zero aggregate pollution changes in the denominator.
