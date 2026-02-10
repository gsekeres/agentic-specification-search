# Verification Report: 113176-V1

## Paper Information
- **Title**: Learning to Coordinate: A Study in Retail Gasoline
- **Authors**: David P. Byrne, Nicolas de Roos
- **Journal**: American Economic Review
- **Total Specifications**: 70

## Baseline Groups

### G1: Post-Coordination Margin Increase
- **Claim**: The transition to coordinated Thursday price cycling in Perth retail gasoline (circa March 2010) increased retail margins by approximately 5.3 cents per liter (cpl).
- **Baseline specs**: `baseline` (station FE only), `baseline_dow` (+ DOW), `baseline_cost` (+ cost change), `baseline_full` (+ DOW + cost)
- **Expected sign**: Positive
- **Baseline coefficient**: 5.31 cpl (SE: 0.076, p < 0.001) for the primary spec; 5.31-5.35 cpl across the four baselines
- **Outcome**: `marg` (retail margin = price - wholesale TGP)
- **Treatment**: `post_coord` (= 1 after March 2010)
- **Fixed effects**: Station
- **Sample**: Major brands (BP, Caltex, Woolworths, Coles), Perth, 2003-2015, trimmed 1%

**Note**: The paper is primarily descriptive, documenting price coordination through figures rather than formal regression tables. These specifications formalize the paper's central empirical assertion as a panel regression. There is only one baseline group because the paper has a single core claim: post-coordination margins increased. The four baseline variants differ only in control variables (DOW dummies and/or wholesale cost change) and produce near-identical estimates.

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Baseline** | **4** | 4 baseline variants (station FE only, +DOW, +cost, +DOW+cost) |
| **Core tests** | **41** | |
| core_fe | 5 | Pooled OLS, station+year FE, station+DOW FE, brand FE, brand+year FE |
| core_inference | 4 | Heteroskedastic-robust, brand-clustered, year-clustered, IID SEs |
| core_sample | 22 | Brand subsets (4), cycling/non-cycling (2), year drops (4), time windows (2), post-2005, no oil shock, weekday/day-specific (4), trim/winsor/no-trim (3), all brands |
| core_funcform | 3 | Log margin, IHS margin, % margin |
| core_treatment | 5 | Alt dates (Jan 2010, Jun 2010), both transitions, three periods, post-coord trend |
| core_controls | 2 | Add brand dummies, cycling station interaction |
| **Non-core tests** | **25** | |
| noncore_alt_outcome | 9 | Price level, price change, Thu/Wed jump probabilities, cycle-bottom margin, jump-day margin, jump size, squared margin, log price |
| noncore_alt_treatment | 1 | Post-BP-end dummy (Aug 2012, tests a different event) |
| noncore_heterogeneity | 6 | BP/Caltex/Woolworths/Coles brand interactions, Thursday-specific effect, early vs late post-coord |
| noncore_placebo | 4 | Fake treatment dates 2005-2008 in pre-period |
| noncore_alt_estimation | 5 | Unconditional quantile regressions (10th-90th) without station FE |
| **Total** | **70** | |

## Detailed Classification Notes

### Baselines (4 specs)

The four baseline specifications all estimate the same core relationship -- retail margin regressed on post_coord dummy with station FE -- with progressively richer controls. Their near-identical coefficients (5.31-5.35 cpl) confirm that DOW dummies and wholesale cost changes do not confound the before/after margin comparison. They are grouped as a single baseline group G1.

### Core Tests (41 specs)

**Fixed effects variations (5 specs)**: These systematically vary the fixed effects structure:
- Pooled OLS (no FE, coef=5.45): very close to baseline, suggesting station heterogeneity is not a major confounder for the time-varying treatment.
- Station + year FE (coef=4.66): absorbs year-level trends, reducing the estimate by ~12%. This is arguably the most informative robustness check because it addresses pre-trends.
- Station + DOW FE (coef=5.35): absorbs day-of-week effects, identical to baseline_dow by construction.
- Brand FE (coef=5.54): coarser than station FE but produces a similar estimate.
- Brand + year FE (coef=4.65): mirrors station+year, confirming year FE matter more than the granularity of entity FE.

**Inference variations (4 specs)**: All maintain the identical point estimate (5.31 cpl) but vary SE computation:
- Heteroskedasticity-robust: SE=0.011, highly significant
- Brand-clustered: SE=0.351, p=0.0006 -- still significant despite only 4 clusters (conservative)
- Year-clustered: SE=0.610, p<0.001 -- with 13 year clusters, still significant
- IID: SE=0.010, most optimistic

The brand-clustered SE is the most conservative and most relevant given the small number of brands. The result remains significant even with this very small number of clusters.

**Sample restrictions (22 specs)**: The largest category, reflecting the paper's geographic and temporal scope:
- *Brand subsets* (4): BP (4.75), Caltex (4.83), Woolworths (6.08), Coles (5.87). All four brands individually show positive, significant margin increases. The pattern is consistent with the coordination narrative: supermarket chains (Woolworths, Coles) gained more than majors (BP, Caltex).
- *Station type* (2): Cycling stations (5.12) vs non-cycling (3.68). Both positive and significant, though non-cycling stations show a smaller effect consistent with spillover or confounding.
- *Year drops* (4): Dropping individual early or late years (2003, 2004, 2013, 2014) produces coefficients of 5.04-5.36, confirming no single year drives the result.
- *Time windows* (2): Narrower windows of 2008-2012 (3.90) and 2009-2011 (3.73) reduce the estimate but remain highly significant, addressing pre-trend concerns.
- *Post-2005* (1): Dropping 2003-2004 yields 5.07.
- *No oil shock* (1): Excluding 2008-2009 yields 5.76 (higher, since oil price crash temporarily compressed margins).
- *Day-specific* (4): Weekdays (4.78), Monday (4.33), Wednesday (0.50), Thursday (9.19). The Wednesday vs Thursday pattern is strongly consistent with the coordination narrative: pre-coordination, Wednesdays were jump days; post-coordination, Thursdays became the coordinated jump day.
- *Outlier treatment* (3): 5% trim (4.40), no trim (5.59), 1% winsorize (5.55). Robust to outlier handling.
- *All brands* (1): Including independents yields 5.08, slightly below major-brand baseline.

**Functional form (3 specs)**: These transform the margin outcome while preserving the same treatment and identification:
- Log margin (0.87): consistent with an 87% proportional increase in margins for positive-margin observations.
- IHS margin (1.07): handles zero and negative margins.
- % margin (3.45pp): margin as percentage of wholesale cost increased by 3.45pp.

These are core because they transform the same outcome variable rather than measuring a fundamentally different concept.

**Treatment variations (5 specs)**: These maintain the margin outcome but alter the treatment definition:
- Alternative dates (Jan 2010: 5.05; Jun 2010: 5.27): the result is stable across plausible coordination transition dates.
- Both transitions (post_coord=4.76, post_bp_end=1.29): decomposing the effect into two coordination phases shows an additional 1.3 cpl increase after BP ended its Wednesday price leadership in Aug 2012.
- Three periods (BP-led=4.76, post-BP=6.05): consistent with the two-transition model.
- Post-coord + trend (3.85 + 0.68/year trend): allowing margins to trend upward post-coordination reduces the level shift to 3.85 but confirms ongoing margin growth.

**Control variations (2 specs)**:
- Add brand dummies (5.31): absorbed by station FE, coefficient unchanged.
- Cycling station interaction (base=4.82, interaction=0.30, p=0.23): cycling and non-cycling stations show similar margin increases; the interaction is insignificant.

### Non-Core Tests (25 specs)

**Alternative outcomes (9 specs)**: These measure fundamentally different dependent variables:
- Price level with cost control (4.86): prices rose, not just margins. Non-core because outcome is different.
- Price change dp (-0.18): coordination reduced daily price change magnitude. Negative coefficient is expected under coordination (smoother price paths).
- Thursday jump probability (+0.083): coordination increased the frequency of Thursday price jumps.
- Wednesday jump probability (-0.009): and decreased Wednesday jumps.
- Cycle-bottom margin (3.29): even trough margins rose.
- Jump-day margin (4.70): jump-day margins rose more.
- Jump size (1.48): coordinated jumps were larger.
- Squared margin (81.6): tests dispersion, not level.
- Log price (0.036): different functional form of a different outcome (price, not margin).

These provide rich supporting evidence for the coordination narrative but measure different aspects of market outcomes.

**Alternative treatment (1 spec)**: post_bp_end (Aug 2012) tests a different event -- the end of BP's Wednesday price leadership. This is a distinct empirical claim about a second coordination transition, not a robustness check of the March 2010 transition.

**Heterogeneity (6 specs)**: These decompose the effect by subgroup:
- Brand interactions (4): BP (-0.87 differential), Caltex (-0.60), Woolworths (+0.97), Coles (+0.75). Show how gains were distributed across brands.
- Thursday-specific effect: post_coord*Thursday interaction of +4.40 cpl shows the coordination effect is concentrated on Thursdays.
- Early vs late post-coord: an additional 1.24 cpl after Aug 2012, testing time-varying coordination intensity.

These decompose the effect rather than providing alternative estimates of the average effect.

**Placebo tests (4 specs)**: Fake treatment dates in the pre-period (March 2005-2008) on the pre-coordination sample only:
- 2005: 1.27 cpl (significant)
- 2006: 2.08 cpl (significant)
- 2007: 1.81 cpl (significant)
- 2008: 1.19 cpl (significant)

All four placebos are positive and significant, revealing a pre-existing upward trend in margins during 2003-2010. This is the most important caveat: the baseline 5.3 cpl estimate likely overstates the causal effect of coordination by 1-2 cpl. However, the baseline estimate (5.3 cpl) is substantially larger than any placebo (max 2.1 cpl), suggesting coordination added a meaningful additional increment. These are non-core because they test research design validity rather than provide alternative estimates.

**Quantile regressions (5 specs)**: Unconditional quantile regressions (10th through 90th percentile) without station FE:
- 10th: 4.00, 25th: 5.50, 50th: 6.30, 75th: 6.10, 90th: 5.60
- All positive and significant. The median effect (6.30) exceeds the mean (5.31), consistent with a positively skewed margin distribution.

These are non-core because they use a fundamentally different estimator (no station FE) and measure distributional effects rather than the conditional mean.

## Robustness Assessment

The main finding -- that retail margins increased significantly after the March 2010 coordination transition -- is **very robust** across all core specifications:

- **Core coefficient range**: 0.50 to 9.19 cpl, but excluding day-specific subsamples (Wednesday/Thursday), the range narrows to 3.73-5.76 cpl. All 45 core+baseline specifications produce positive, highly significant estimates (p < 0.01).
- **Year FE sensitivity**: Adding year FE reduces the estimate from 5.31 to 4.66 cpl (~12% reduction), which is the most meaningful single-dimension robustness check. The result remains strongly significant.
- **Time window sensitivity**: Narrow windows (2009-2011, 2008-2012) reduce the estimate to 3.7-3.9 cpl but remain significant. This addresses pre-trend concerns: even in the immediate vicinity of the transition, margins increased by ~4 cpl.
- **Most conservative estimate**: The post-coord + trend specification yields 3.85 cpl as the level shift, with an additional 0.68 cpl/year trend. This is probably the most defensible point estimate, accounting for the growing margins documented in the placebos.
- **Inference robustness**: The result survives brand-level clustering (4 clusters, p = 0.0006) and year-level clustering (13 clusters, p < 0.001).

### Key Caveats

1. **Pre-trend in margins**: All four placebo tests are positive and significant (1.2-2.1 cpl), indicating margins were rising before coordination. A conservative adjustment would reduce the causal estimate to roughly 3-4 cpl.
2. **No counterfactual**: This is a before/after design without an untreated control group. All Perth stations were exposed simultaneously.
3. **Treatment is a pure time variable**: Collinear with monthly FE, limiting identification strategies. Year FE can be included but month FE would absorb the treatment.
4. **Paper is primarily descriptive**: The regressions formalize the paper's narrative but the paper itself relies on figures, not regression tables. The specification search necessarily imposes a parametric structure on what the authors presented non-parametrically.
