# Verification Report: 112514-V1

## Paper Information
- **Title**: Screening, Competition, and Job Design
- **Authors**: Bjorn Bartling, Ernst Fehr, Klaus Schmidt
- **Journal**: American Economic Review (2012)
- **Total Specifications**: 88

## Baseline Groups

### G1: Effect of Limited Discretion Contracts on Effort
- **Claim**: Limited discretion contracts elicit significantly more effort from employees than full discretion contracts in the base treatment, with the effect concentrated at lower wages.
- **Baseline spec**: `baseline`
- **Expected sign**: Positive
- **Baseline coefficient**: 2.731 (SE: 0.415, p < 0.001)
- **Outcome**: `effort`
- **Treatment**: `limited` (limited discretion dummy)
- **Table 1, Column 1**

**Note**: This is the only baseline group. The paper's central finding is that limited discretion contracts (which restrict the employee's ability to provide extremely low effort) paradoxically elicit *higher* voluntary effort than full discretion contracts, especially at low wages. The regression is `effort ~ wage + limited + limited*wage` with subject FE and SEs clustered by subject, restricted to employees in the base treatment who accepted contracts. The positive coefficient on `limited` (2.731) represents the effort premium at zero wage. The paper's screening treatment (Table 2), which introduces competition and reputation, tests a distinct claim about how competition moderates this effect and is classified as non-core.

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baseline)** | **49** | |
| core_controls | 12 | 1 baseline + 11 control variations (period dummies, requested effort, share, quadratic wage, log wage, wage bins, kitchen sink, etc.) |
| core_sample | 20 | Period restrictions (early/late/middle/drop first/last), wage subsamples, matching groups, session drops, extreme effort removal, weighting, interior efforts, nonzero surplus |
| core_inference | 7 | Robust SE, IID SE, cluster by session, cluster by session x matching group, CRV1 (duplicate of baseline), OLS manual FE, HC3 |
| core_fe | 8 | Pooled OLS, random effects, first differences, session-period FE, two-way subject+period FE, matching group FE, quantile (median/Q25/Q75) |
| core_funcform | 3 | Level-log, cubic wage, standardized wage |
| **Non-core tests** | **39** | |
| noncore_alt_outcome | 12 | Principal profit (3 variants), agent profit (2), log effort, binary high effort, effort gap, acceptance, surplus, probit acceptance |
| noncore_alt_treatment | 1 | Only interaction term (changes treatment variable to limitedwage) |
| noncore_alt_sample | 6 | Screening treatment (3 variants), pooled base+screening, base second, screen second, screening full controls |
| noncore_placebo | 5 | Scrambled limited, lagged effort, limited predicts future wage, period 1 only, within-subject permutation |
| noncore_heterogeneity | 10 | By session order, matching group, early/late period, wage level, requested effort, share generosity, screening reputation interactions (4) |
| noncore_diagnostic | 2 | Between estimator (insignificant cross-sectional), ordered probit (no FE) |
| **Duplicates noted** | **3** | baseline = sample/base_first_sessions (identical); inference/crv1_baseline = baseline (same SE); form/log_linear = outcome/log_effort (same regression) |
| **Total** | **88** | |

## Detailed Classification Notes

### Core Tests (49 specs including 1 baseline)

**Baseline (1 spec)**: The primary baseline is Table 1, Column 1: `effort ~ wage + limited + limitedwage | subjectid`, clustered by subject, on the base treatment (type==2, treatmentid==3, acceptance==1). Coefficient on `limited` = 2.731 (SE = 0.415, p < 0.001, N = 658, R2 = 0.519).

**Control variations (11 non-baseline core_controls specs)**: These add or modify controls while keeping the same outcome, treatment, sample, and subject FE:
- No controls (just limited dummy): coefficient drops to -0.528 (insignificant) because without wage controls, the limited effect is confounded by wage selection. This is informative about the importance of the wage control.
- Table 1 Col 2 (add requestedeffort + period dummies): 2.722, nearly identical.
- No interaction term: 1.654, still highly significant.
- Add offered share: 2.698, stable.
- Period dummies: 2.706, stable.
- Quadratic wage (Table A1): 2.780, stable.
- Add requested effort only: 2.748, stable.
- Wage bins: 1.498, somewhat smaller but still significant.
- Log wage: 2.491, stable.
- Kitchen sink (all controls): 2.476, stable.
- Wage + share + requested effort: 2.435, stable.

**Sample restrictions (20 specs)**: The largest core category:
- Period subsets: early (2.754), late (2.424, p=0.001), drop period 1 (2.922), drop period 15 (2.733), drop both (2.909), middle periods 5-10 (2.456). All significant.
- Wage subsets: low wage <10 (1.752, p<0.001), medium/high >=10 (1.790, p=0.018), high >=20 (2.188, p=0.19 -- insignificant). The insignificance at high wages is consistent with the theory that limited contracts matter when wages are low.
- Matching groups: group 1 (2.282), group 2 (3.313). Both significant, stable.
- Drop extreme efforts 0/10: 1.422 (p<0.001). Smaller but significant.
- Base first sessions (400-600): 2.731 -- exact duplicate of baseline because all base treatment data comes from these sessions.
- Session drops: 2.641, 2.693, 2.948. All significant, stable.
- Interior efforts only (2-9): 1.854 (p<0.001).
- Drop zero surplus: 2.829 (p<0.001).
- Inverse subject count weighting: 2.545 (p<0.001).
- Period weighting: 2.582 (p<0.001).

**Inference variations (7 specs)**: All maintain the same point estimate (2.731) with the baseline specification except pooled_hc3 which drops FE:
- Robust SE: 0.348 (tighter than baseline 0.415).
- IID SE: 0.378.
- Cluster by session: 0.201 (only 3 sessions -- likely too few clusters for valid inference).
- Cluster by session x matching group: 0.338.
- CRV1 baseline: 0.415 (identical to baseline -- this is a duplicate).
- OLS with subject dummies + cluster: 0.432 (slightly larger SE from manual FE implementation).
- Pooled HC3 (no FE): different coefficient 2.386 (SE 0.390).

**Fixed effects / method variations (8 core specs)**:
- Pooled OLS: 2.386 (smaller without subject FE but still significant).
- Random effects: 2.658 (very close to FE 2.731, suggesting FE assumption is appropriate).
- First differences: 2.595 (close to FE, confirming within-subject identification).
- Session x Period FE: 2.501 (different FE structure, still significant).
- Two-way subject + period FE: 2.706 (nearly identical to baseline, period FE adds little).
- Matching group FE only: 2.380 (coarser FE, still significant).
- Quantile regressions (no FE): median = 3.500, Q25 = 2.000, Q75 = 3.000. All highly significant. These lack subject FE but confirm the limited effect across the effort distribution.

**Functional form (3 core specs)**:
- Level-log (effort on log wage): 1.272, significant.
- Cubic wage: 2.468, stable.
- Standardized wage: 1.721 (rescaled coefficient, same underlying regression).

### Non-Core Tests (39 specs)

**Alternative outcomes (12 specs)**: These change the dependent variable away from effort:
- Principal profit (3 specs): 5.346 to 6.535. Limited contracts increase employer profit at all specifications. These test a related but distinct claim about welfare.
- Agent profit (2 specs): -3.480 (all), -2.731 (accepted). Agent profit decreases mechanically because agent profit = wage - effort, and effort increases.
- Log effort: 0.996, same as form/log_linear (duplicate).
- High effort binary: 0.943, highly significant.
- Effort gap (effort - requested effort): 3.717. Large effect on voluntary overprovision.
- Acceptance: -0.424. Limited contracts reduce acceptance rates by 42 percentage points.
- Surplus: 3.188 (p=0.025). Limited contracts increase total surplus, but marginally.
- Probit acceptance: -0.458 (p=0.79). Probit fails because it does not account for subject FE.

These are non-core because they address different claims (profitability, welfare, acceptance) rather than robustness of the effort claim.

**Alternative treatment (1 spec)**:
- `controls/only_interaction`: Reports coefficient on `limitedwage` instead of `limited`. This changes the treatment variable interpretation entirely (marginal effect of limited per unit wage, not the level effect). Insignificant (p=0.12).

**Alternative sample / treatment arms (6 specs)**: These use the screening treatment or pool treatments:
- Screening treatment (treatmentid==1): 2.483 (p<0.001). Similar effect in the screening arm.
- Screening + reputation controls: 3.230 (Table 2 Col 2).
- Pooled base+screening: 2.662.
- Base second (treatmentid==2): 3.184. Subjects who experienced screening first show larger effect.
- Screen second (treatmentid==4): 0.892 (p=0.10, insignificant).
- Screening full controls (Table 2 Col 5): 2.693.

These are non-core because they test a different experimental treatment arm (with reputation and competition) rather than alternative implementations of the base treatment specification.

**Placebo tests (5 specs)**: These test the validity of the research design:
- Scrambled limited: -0.002 (p=0.99). Confirms real assignment drives the result.
- Lagged effort: -0.358 (p=0.43). Current contract type does not predict past effort.
- Limited predicts future wage: 0.752 (p=0.40). No reverse causality.
- Period 1 only: 0.682 (p=0.49). Insignificant with only cross-sectional variation (one observation per subject in period 1).
- Within-subject permutation: 0.597 (p=0.02). Some residual correlation due to non-random ordering of contracts within subjects.

These are non-core because they validate the design rather than provide alternative estimates of the effect.

**Heterogeneity (10 specs)**: These explore whether the effect varies by subgroups:
- Session order interaction: coefficient on limited = 2.731 (identical to baseline because base_first is collinear with the sample).
- Matching group interaction: 2.713.
- Early/late period interaction: 2.735.
- Wage level interaction: 3.207 (the interaction term would show how high wages reduce the limited effect).
- Requested effort interaction: 2.741.
- Screening reputation interactions: 4 specs exploring how the limited effect varies with agent reputation in the screening treatment. These use the screening sample, not the base treatment.
- Share generosity interaction: 2.704.

These are non-core because they decompose the effect by subgroup rather than provide alternative estimates of the same baseline specification.

**Diagnostics (2 specs)**:
- Between estimator: -2.473 (p=0.36, wrong sign). The cross-sectional (between-subject) relationship is negative and insignificant, confirming that the effect is identified from within-subject variation (subjects who receive both contract types).
- Ordered probit: -0.069 (p=0.77). The ordered probit does not include subject FE and thus cannot identify the within-subject effect. This is a method failure rather than a meaningful robustness check.

## Duplicates Identified

The following specs produce identical or near-identical results:
1. `sample/base_first_sessions` = `baseline`: Both yield coefficient 2.731, SE 0.415, N=658. The base treatment data comes exclusively from sessions 400-600, so restricting to those sessions does not change the sample.
2. `inference/crv1_baseline` = `baseline`: Identical SE computation (CRV1 clustered by subject).
3. `form/log_linear` = `outcome/log_effort`: Both regress log(effort) on wage + limited + limitedwage with subject FE. Coefficient 0.996 in both.
4. All inference specs except `inference/pooled_hc3` share the same point estimate (2.731) because they use the same model with different SE estimators.

After removing duplicates, there are approximately 85 unique specifications.

## Robustness Assessment

The main finding -- that limited discretion contracts elicit more effort in the base treatment -- is **robust** across core specifications:

- **Control sensitivity**: Coefficient ranges from 1.498 (wage bins) to 2.780 (quadratic wage) across core control variations. All remain significant at p < 0.01 except when wages are not controlled for (`controls/none`: -0.528, insignificant), which reflects confounding from wage selection, not a failure of robustness.

- **Sample stability**: Coefficient ranges from 1.422 (drop extreme efforts) to 3.313 (matching group 2) across core sample restrictions. All significant at conventional levels except high-wage-only subsample (p=0.19, N=216), which is consistent with the theoretical prediction that limited contracts bind at low wages.

- **Method consistency**: FE (2.731), RE (2.658), first differences (2.595), and pooled OLS (2.386) all yield similar and significant estimates. Quantile regressions (no FE) confirm the effect at Q25 (2.0), median (3.5), and Q75 (3.0).

- **Inference robustness**: The baseline is significant under all SE approaches. The most conservative standard error (OLS + subject cluster, 0.432) still yields p < 0.001. Clustering at the session level (only 3 clusters) is arguably too few for valid inference.

**Key sensitivities**:
- **Wage control is essential**: Without controlling for wage, the limited coefficient flips sign (-0.528) because limited contracts are selected at different wage levels than full discretion contracts.
- **High-wage subsample is insignificant**: At wages >= 20, the effect (2.188, p=0.19) loses significance. This is theoretically expected because at high wages, full discretion contracts already elicit high effort.
- **Between estimator shows wrong sign**: The cross-sectional relationship is negative (-2.473, p=0.36), confirming that identification relies entirely on within-subject variation. Subjects who receive more limited contracts on average do not show higher average effort.
- **Screening treatment effects vary**: The limited effect is similar in the screening treatment (2.483) but becomes insignificant in the "screen second" treatment arm (0.892, p=0.10), suggesting potential order effects or that competition in screening attenuates the limited discretion advantage.
- **Placebo tests pass**: Scrambled treatment (p=0.99), lagged effort (p=0.43), and future wage prediction (p=0.40) all confirm no spurious relationships.
