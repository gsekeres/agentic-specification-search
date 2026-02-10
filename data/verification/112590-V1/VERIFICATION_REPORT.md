# Verification Report: 112590-V1

## Paper
**Title**: The Impact of Medical Liability Standards on Regional Variations in Physician Behavior: Evidence from the Adoption of National-Standard Rules
**Author**: Michael Frakes
**Method**: Difference-in-Differences (staggered adoption of national-standard medical liability rules)

## Baseline Groups Found

| Group | Claim | Expected Sign | Baseline Spec IDs |
|-------|-------|---------------|-------------------|
| G1 | Adoption of national-standard liability rules leads to convergence of state health outcomes toward national averages (Apgar score deviation) | + | baseline |

The paper's core finding concerns cesarean section rate convergence using restricted NHDS data not available in the replication package. The specification search operates on the available vital statistics data (Tables 9 and C-2), testing Apgar score convergence and related health outcomes.

- **Baseline coefficient**: 0.00292
- **Baseline SE**: 0.00133 (clustered at state level)
- **Baseline p-value**: 0.028
- **Baseline N**: 1,237
- **Baseline R-squared**: 0.899
- **Fixed effects**: State + Year + State-specific linear trends
- **Weights**: Number of births (analytic weights)

Note: The positive coefficient indicates states move *further* from the national Apgar mean after adopting national-standard rules, which is the opposite direction from the paper's main convergence hypothesis for cesarean rates. This is a secondary outcome in the paper.

## Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **76** |
| Baselines | 1 |
| Core tests (non-baseline) | 37 |
| Non-core tests | 38 |

## Category Breakdown

| Category | Count | Core/Non-core |
|----------|-------|---------------|
| baseline | 1 | core |
| core_controls | 11 | core |
| core_fe | 5 | core |
| core_sample | 12 | core |
| core_inference | 4 | core |
| core_funcform | 9 | core |
| noncore_alt_outcome | 19 | non-core |
| noncore_placebo | 4 | non-core |
| noncore_heterogeneity | 9 | non-core |
| noncore_diagnostic | 3 | non-core |

## Classification Details

### Core Tests (38 specs including baseline)

**Controls (11 specs)**
Systematically varied the control set while maintaining the same outcome (apgar5_diff1p), treatment (specns), fixed effects, and sample. All use the same baseline framework.
- No controls (p=0.057), minimal controls (p=0.057): slightly above 5% threshold
- Demographics only (p=0.028), tort reform only (p=0.050): significant or borderline
- Leave-one-out tests (dropping tort, demographics, education, age, race, beds): p-values range from 0.022 to 0.064
- The result is moderately sensitive to control specification. Only 5 of 11 are significant at 5%.

**Fixed Effects (5 specs)**
Tested alternative FE structures. This is the most important sensitivity dimension:
- State FE only: coef=0.00095, p=0.412 -- result disappears
- Year FE only: coef=-0.00122, p=0.521 -- sign reversal
- State+year (no trends): coef=0.00175, p=0.193 -- insignificant
- State+year, no controls, no trends: coef=0.00240, p=0.112 -- insignificant
- No FE at all: coef=-0.00209, p=0.290 -- sign reversal

The baseline result is entirely dependent on state-specific linear time trends. Without trends, the result vanishes. This is a significant fragility.

**Sample Restrictions (12 specs)**
- Early period (1977-1990): p=0.608 -- no effect in early period
- Late period (1991-2004): p=0.673 -- no effect in late period
- Dropping specific states (MD, DC, always-treated, late adopters): robust (p=0.018 to 0.037)
- Drop small states: p=0.047 (robust)
- Drop large states: p=0.219 -- effect driven by larger states
- Trim 1%/5% outliers: p=0.041/0.034 (robust but smaller coefficients)
- Post-1980: p=0.054 (borderline)
- Pre-1995: p=0.021 (significant)

Notable: the effect is not present in either early or late subperiods alone, suggesting it depends on the full panel length. The effect disappears when dropping large states.

**Inference (4 specs)**
All 4 core inference specs use the same baseline outcome and treatment but vary SE computation:
- HC1 robust: p<0.001
- Conventional: p<0.001
- HC3: p<0.001
- Cluster year: p<0.001

Without state-level clustering, the result is much more significant, indicating substantial within-state serial correlation. The baseline choice to cluster at the state level is conservative and appropriate.

**Functional Form and Estimation (9 specs)**
- IHS transformation: p=0.028 (identical to baseline; IHS approximates identity for small values)
- Standardized (z-score): p=0.028 (trivial rescaling)
- Unweighted OLS for Apgar5: p=0.085 (less significant)
- Log outcome: p=0.657 (insignificant; poor fit)
- Squared outcome: p=0.117 (insignificant)
- OLS no weights no clustering: p<0.001 (more significant)
- First differences: p=0.283 (insignificant; different identification)
- Between estimator: p=0.817 (sign reversal; no cross-sectional effect)

### Non-Core Tests (38 specs)

**Alternative Outcomes (19 specs)**
Specifications that change the dependent variable or treatment variable are classified as non-core because they test a different claim than the baseline. These include:
- Good Apgar convergence (2 specs): p=0.571, p=0.517 -- insignificant
- Neonatal mortality convergence (2 specs): p=0.389, p=0.106 -- insignificant
- Cesarean rate convergence: p=0.913 -- no effect
- Low birthweight convergence: p=0.582 -- no effect
- Preterm delivery convergence: p=0.646 -- no effect
- Level outcomes (neonatal, Apgar, good Apgar): all insignificant
- Good Apgar no controls (p=0.023): significant but different outcome + different spec
- Neonatal no controls: p=0.741 -- insignificant
- Vital2 specifications (5 specs): Use different dataset, treatment (ln_stand_cs), and outcomes (logapgar5, good2). Only logapgar_notrends is significant (p=0.049).
- Good Apgar unweighted, neonatal unweighted: different outcomes, both insignificant

**Placebo Tests (4 specs)**
- Early treatment (5-year lead): coef=-0.00260, p=0.022 -- **significant and concerning**. A pre-trend or anticipation effect.
- Low birthweight placebo: p=0.665 -- passes
- Preterm placebo: p=0.343 -- passes
- Permuted treatment: p=0.650 -- passes

**Heterogeneity (9 specs)**
Subsample analyses and heterogeneous effects. Most use different treatment (ln_stand_cs) and outcomes (logapgar5, good2, log_nn28):
- Initial low/high Apgar states with logapgar5 outcome: both insignificant
- Initial low/high states with good2 outcome: both insignificant
- Neonatal mortality heterogeneity (full, low, high): all insignificant
- Caps vs no-caps states: both insignificant
No significant heterogeneity detected in any subgroup.

**Diagnostic/Triage (3 specs)**
These test whether cesarean rate increases are associated with patient selection:
- Low birthweight among CS mothers: p=0.137
- Preterm among CS mothers: p=0.058 (borderline)
- Very preterm (<27w) among CS mothers: p=0.341

## Top 5 Issues

### 1. Complete dependence on state-specific linear trends
The baseline result (p=0.028) vanishes entirely without state-specific linear time trends. With just state + year FE, p=0.193. With only state FE, p=0.412. With no FE, the coefficient reverses sign. This is the most important finding: the result is identified entirely by within-state deviations from state-specific linear trends, which is a fragile identification strategy. If the true effect is nonlinear or the trends are misspecified, the result could be spurious.

### 2. Significant early-treatment placebo (p=0.022)
The placebo test shifting treatment 5 years earlier yields a significant coefficient with the opposite sign (coef=-0.0026, p=0.022). This suggests either a pre-trend, anticipation effects, or that the state-specific linear trends do not fully capture differential pre-treatment dynamics. This directly threatens the parallel trends assumption underlying the DiD design.

### 3. No effect in temporal subsamples
Splitting the sample into early (1977-1990, p=0.608) or late (1991-2004, p=0.673) periods yields no significant result in either half. The effect is only present in the full panel, which is unusual for a genuine treatment effect and could reflect an artifact of the trend specification over the longer window.

### 4. Effect driven by large states
Dropping large states eliminates the result (p=0.219), while dropping small states does not (p=0.047). Combined with the finding that unweighted estimates are less significant (p=0.085 vs 0.028), this indicates the result is driven by the most populous states. This is not necessarily problematic (birth-weighted analysis is appropriate), but it suggests the effect is not widespread.

### 5. No generalization to other health outcomes
Among 10 alternative outcome specifications testing good Apgar, neonatal mortality, cesarean rate, low birthweight, and preterm delivery convergence, only 1 is significant at 5% (good Apgar without controls, p=0.023, which uses a different specification from baseline). The Apgar5 convergence finding does not generalize to any other available health outcome measure with the full control set.

## Overall Assessment

The baseline result (coef=0.00292, p=0.028) for Apgar5 convergence shows substantial fragility across the specification search:

**Core robustness summary**:
- Controls: 5/11 significant at 5% (45%)
- Fixed effects: 0/5 significant at 5% (0%) -- result requires state trends
- Sample: 7/12 significant at 5% (58%)
- Inference: 4/4 significant at 5% (100%) -- but all are less conservative than baseline
- Functional form/method: 4/9 significant at 5% (44%), and 2 of the 4 (IHS, standardized) are trivial transformations

Overall, approximately 20 of 37 non-baseline core specs (54%) are significant at 5%. However, the most substantive robustness dimensions (fixed effects, temporal subsamples, estimation method) show clear fragility. The significant early-treatment placebo further undermines confidence in the causal interpretation.

It is important to note that this is a secondary result in the paper. The primary cesarean convergence finding uses restricted NHDS data not available for replication.
