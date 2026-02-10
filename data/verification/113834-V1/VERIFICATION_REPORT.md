# Verification Report: 113834-V1

## Paper
**Title**: Deregulation, Consolidation, and Efficiency: Evidence from U.S. Nuclear Power
**Authors**: Lucas W. Davis and Catherine Wolfram
**Journal**: American Economic Journal: Applied Economics, 2012
**Method**: Panel fixed effects (two-way FE: reactor + year-month)

## Baseline Groups Found

| Group | Claim | Expected Sign | Baseline Spec IDs |
|-------|-------|---------------|-------------------|
| G1 | Deregulation/divestiture of nuclear power plants increased operating efficiency (capacity factor) by ~10 pp | + | baseline |

This is an empirical panel fixed effects study exploiting variation in the timing of divestitures across nuclear reactors during 1999-2007. The baseline specification is Table 2, Column 3 of the paper.

- **Baseline coefficient**: 10.007 pp (paper: ~10.7 pp, difference from age variable scaling and pyfixest vs. Stata areg)
- **Baseline SE**: 2.060 (paper: ~2.0)
- **Baseline p-value**: 8.03e-06
- **Baseline N**: 36,667
- **Baseline R-squared**: 0.2212
- **Fixed effects**: reactor + year-month
- **Controls**: age^2, age^3
- **Clustering**: plant level (65 plants)

## Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **86** |
| Baselines | 1 |
| Core tests (non-baseline) | 52 |
| Non-core tests | 33 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| baseline | 1 | Table 2 Col 3 replication |
| core_fe_structure | 5 | No FE, reactor only, time only, reactor+year, plant+time |
| core_controls | 7 | No age, age^2 only, add capacity, add same_owner (x2), same_type, same_manuf |
| core_clustering | 4 | Reactor, state, NRC region, heteroskedasticity-robust |
| core_sample_restriction | 19 | State exclusions (8), region exclusions (5), temporal (6) |
| core_subsample | 10 | BWR, PWR, GE, WE, vintage splits (3), NRC region splits (3) |
| core_outcome_transform | 4 | Log power, binary >80%, binary >90%, generation MWh |
| core_trimming | 3 | Drop negative, winsorize 1/99, drop >100% |
| core_time_trend | 2 | Linear trend, quadratic trend |
| noncore_alternative_baseline | 2 | Table 2 Col 1 and Col 2 (paper build-up specifications) |
| noncore_different_unit | 1 | Plant-level aggregation (changes unit of analysis) |
| noncore_heterogeneity | 14 | Type, manufacturer, vintage, sale/transfer, price interactions |
| noncore_different_treatment | 2 | same_owner as treatment, never-divested subsample |
| noncore_monthly_decomposition | 12 | Separate regressions by calendar month |

## Classification Decisions

### Core Test Classifications (52 specs)

**Fixed effects structure (5 specs)**: These systematically vary the FE structure (no FE, reactor only, time only, reactor+year, plant+time) while keeping the same treatment and outcome. All produce positive, significant coefficients ranging from 6.0 to 14.2 pp. The no-FE specification has the largest coefficient (14.2), as expected since it omits reactor-level confounders that are positively correlated with divestiture.

**Control variables (7 specs)**: Variations in age polynomial controls and addition of consolidation measures. Removing age controls or adjusting the polynomial order barely changes the coefficient (~10.0-10.2). Adding consolidation measures (same_owner, same_type, same_manuf) reduces the coefficient to 7.7-8.6 pp but it remains significant, indicating the divestiture effect operates partly but not entirely through the consolidation channel. Note that consol/same_owner is an exact duplicate of robust/control/add_same_owner.

**Clustering alternatives (4 specs)**: The point estimate is identical (10.01) since only the SE computation changes. Clustering at reactor level (SE=1.62) and heteroskedasticity-robust (SE=0.60) give tighter SEs. Clustering at state (SE=1.88) gives similar SEs. Clustering at NRC region (SE=2.10, only ~4 clusters) gives slightly larger SEs but remains significant at 5% (p=0.017).

**Sample restrictions (19 specs)**: Eight state exclusions, five regional exclusions, and six temporal restrictions. The coefficient ranges from 7.4 (excluding Midwest) to 14.4 (excluding South). All are positive and significant. The robustness to state exclusions confirms no single state drives the result. Temporal restrictions show the effect persists across all time windows.

**Subsample splits (10 specs)**: BWR-only and GE-only produce identical results (coefficient=0.92, insignificant at p=0.82) because all GE reactors are BWR-type. These subsamples have very few divested units. PWR-only and WE-only are significant. Vintage and NRC region splits are mostly significant, with the 1975-84 vintage (p=0.10) and NRC region 1 (p=0.09) being marginal cases.

**Outcome transformations (4 specs)**: Log(power+1), binary >80%, binary >90%, and generation MWh all show positive, significant divestiture effects. These confirm the result is not sensitive to the specific outcome metric.

**Trimming/winsorization (3 specs)**: Dropping negative capacity factors, winsorizing at 1/99, and dropping >100% all produce positive significant coefficients (5.2-9.9 pp). The coefficient is smaller (5.2) when dropping negatives, as these extreme low observations are disproportionately in the pre-divestiture period.

**Time trend alternatives (2 specs)**: Linear and quadratic time trends replacing year-month FE both produce significant coefficients (~10.9-11.1 pp), confirming the result is not driven by the specific time FE structure.

### Non-Core Classifications (33 specs)

**Alternative baselines (2 specs)**: Table 2 Columns 1 and 2 are build-up specifications from the paper's own table showing the progression from simpler to more complete models. They represent intermediate steps in the paper's argument, not robustness checks of the final specification. Column 1 (time FE only) gives 6.33 pp; Column 2 (reactor + time FE, no age controls) gives 10.22 pp.

**Different unit of analysis (1 spec)**: Plant-level aggregation changes the unit from reactor to plant, creating fractional treatment values for multi-reactor plants where only some reactors are divested. This fundamentally alters the identification and produces a meaningless coefficient (-9.60, SE=26.16, p=0.71). This is correctly excluded from the core set.

**Heterogeneity interactions (14 specs)**: All heterogeneity specifications (type, manufacturer, vintage, sale/transfer, wholesale price) use different treatment variables (interaction terms like divtypePWR, divmanfGE, divV75, etc.) rather than the baseline divested indicator. These decompose the effect by subgroup but do not test robustness of the main treatment effect. All but one (CE manufacturer, p=0.10) are significant.

**Different treatment variable (2 specs)**: consol/same_owner_coef uses same_owner rather than divested as the treatment, testing a different hypothesis (consolidation effect). consol/never_divested restricts to never-divested reactors with same_owner as treatment. Neither tests robustness of the divestiture claim.

**Monthly decomposition (12 specs)**: Separate regressions for each calendar month (January through December) test seasonal heterogeneity in the divestiture effect. All 12 months show positive, significant coefficients (7.3-14.5 pp), with larger effects in May-June (refueling season) and November. These are descriptive decompositions rather than robustness checks of the pooled estimate.

## Core Specification Summary Statistics

Among the 52 core (non-baseline) specifications with divested as treatment and power as the primary outcome (excluding outcome transformations):

- **48 of 48 power-outcome core specs** have positive coefficients (100%)
- **44 of 48** are significant at 5% (91.7%)
- **45 of 48** are significant at 10% (93.8%)
- Coefficient range: 0.92 to 14.43 pp
- Median coefficient: approximately 9.9 pp
- The 4 insignificant specs are: BWR_only (0.92, p=0.82), GE_only (0.92, p=0.82, duplicate of BWR), vintage_V85 (8.63, p=0.10), nrc_1 (6.52, p=0.09)

Including the 4 outcome transformation core specs (all significant with positive coefficients), the full core set is 52 specs.

## Notable Issues

### 1. Duplicate specifications
- consol/same_owner is an exact duplicate of robust/control/add_same_owner (identical coefficient=7.666, SE=2.373, same specification)
- robust/subsample/GE_only is identical to robust/subsample/BWR_only (coefficient=0.916, p=0.82) because all GE-manufactured reactors are BWR-type

### 2. BWR/GE subsample weakness
The BWR-only and GE-only subsamples produce an insignificant coefficient of 0.92 (p=0.82). This is due to very few divested BWR/GE reactors, resulting in insufficient variation within this subsample. This does not undermine the main finding, which is identified across all reactor types, but it does indicate that the efficiency gains from divestiture were primarily realized in PWR (Westinghouse, CE, B&W) reactors.

### 3. Plant-level aggregation failure
The plant-level specification (baseline/table2_col5_plant) produces a negative, insignificant coefficient. As noted in the SPECIFICATION_SEARCH.md, this is because averaging across reactors within multi-reactor plants creates fractional treatment values, which attenuates the estimated effect. This is a methodological artifact, not evidence against the main claim.

### 4. Consolidation channel
Adding consolidation controls (same_owner, same_type, same_manuf) reduces the divestiture coefficient from ~10 to ~7.7-8.6 pp but it remains significant. This suggests approximately 20-25% of the efficiency gain operates through the consolidation channel (firms acquiring multiple plants of similar type), with the remaining 75-80% attributable to other aspects of deregulation.

### 5. Strong overall robustness
The main finding is exceptionally robust. Among core specifications, the only weaknesses appear in very small subsamples (BWR/GE only, certain vintage or NRC region splits) where there is insufficient variation in the treatment variable. The coefficient is stable across all major specification dimensions: FE structure, controls, clustering, state/region exclusions, temporal restrictions, outcome transformations, trimming, and time trend alternatives.

## Recommendations

1. **Remove duplicate specs from analysis**: consol/same_owner duplicates robust/control/add_same_owner; GE_only duplicates BWR_only. After deduplication, there are 84 unique specifications.

2. **Weight the BWR/GE finding appropriately**: The insignificant BWR/GE result reflects small-sample issues (few divested BWR reactors), not a substantive challenge to the main finding. In a specification curve analysis, this should be noted as a power issue.

3. **The heterogeneity and monthly decomposition specs should be treated as supplementary**: These 26 non-core specs provide rich descriptive detail but should not be included in a specification curve of the baseline divestiture claim because they use different treatment variables or test different hypotheses.

4. **Overall assessment: STRONG robustness**: The divestiture effect on nuclear power plant efficiency is one of the most robust findings in the sample, with 100% positive coefficients and >91% significance at 5% across 52 core specifications.
