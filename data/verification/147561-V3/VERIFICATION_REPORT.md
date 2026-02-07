# Verification Report: 147561-V3

## Paper
**Title**: Local Elites as State Capacity: How City Chiefs Use Local Information to Increase Tax Compliance in the D.R. Congo  
**Journal**: AER  
**Method**: Randomized Controlled Trial (cross-sectional OLS)

## Baseline Groups

### G1: Local tax collectors increase tax compliance
- **Claim**: Local tax collectors (city chiefs) increase property tax compliance relative to central collectors in Kananga, D.R. Congo.
- **Expected sign**: Positive
- **Baseline spec_ids**: baseline, baseline_month_fe, baseline_full_fe, baseline_nonexempt
- **Outcome**: taxes_paid (binary)
- **Treatment**: t_l (Local vs Central assignment)
- **Baseline coefficients**: 0.017 (stratum FE), 0.017 (stratum+month FE), 0.024 (stratum+month+house FE), 0.029 (non-exempt subsample)
- **Notes**: The four baselines form a progression of increasing FE saturation. All are highly significant (p < 1e-9). The preferred specification appears to be baseline_full_fe with all three FE sets.

## Summary Counts

| Category | Count |
|----------|-------|
| Total specifications | 60 |
| Baselines | 4 |
| Core tests (including baselines) | 39 |
| Non-core | 21 |
| Invalid | 0 |
| Unclear | 0 |

## Category Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| core_fe | 8 | FE structure variations (including 4 baselines and 5 robustness FE variants) |
| core_sample | 21 | Sample restrictions (including baseline_nonexempt, house-type subsets, rate subsets, leave-one-stratum-out, leave-one-period-out) |
| core_controls | 5 | Control variable additions (rate, rate dummies, exempt, full, include_cli) |
| core_inference | 4 | Clustering/SE variations (HC1, HC2, HC3, cluster on stratum) |
| core_funcform | 1 | LPM without FE |
| noncore_alt_outcome | 7 | Different dependent variables (tax amount levels, IHS tax amount, bribery, visits) |
| noncore_heterogeneity | 9 | Demographic subgroup analyses (gender, tribe, age, occupation, house type interaction) |
| noncore_placebo | 3 | Falsification tests (salongo hours, exempt status, permuted treatment) |
| noncore_alt_treatment | 2 | Different treatment comparisons (CLI vs Central, Local vs CLI) |

## Top 5 Most Suspicious Rows

1. **robust/placebo/exempt_status** (p=0.001, coef=0.015): This placebo test shows a statistically significant positive effect of local collector assignment on exempt status. For a balance/placebo check, one would expect no effect. This could indicate that local collectors are better at identifying which properties qualify for exemptions, or it could reflect a data issue. The significance warrants investigation but does not invalidate the main result since exemption is a pre-treatment characteristic that should be balanced by randomization.

2. **robust/outcome/ihs_tax_amount** (classified noncore_alt_outcome, confidence=0.9): There is a judgment call here about whether IHS-transformed tax amount is a "different outcome" or a "functional form variation" of the same claim. I classified it as non-core because the paper's main claim is about the extensive margin (binary compliance), not the intensive margin (how much tax is paid). However, if the paper treats IHS tax amount as an alternative measure of the same claim, it could be reclassified as core_funcform.

3. **robust/treatment/include_cli** (classified core_controls, confidence=0.8): This spec adds the CLI treatment arm as a control variable while still estimating t_l (Local vs Central). It changes the sample (N=38173 vs 28751) by including CLI-assigned properties. I classified it as core_controls because the treatment of interest (t_l) is unchanged and the CLI indicator is simply an additional control. However, the sample change is non-trivial.

4. **robust/cluster/stratum** (classified core_inference, confidence=0.85): When clustering on stratum, the FE structure changes (stratum FE is dropped, month+house FE kept). This means both the inference and the FE structure differ from baseline_full_fe. The coefficient also changes (0.0225 vs 0.0235). The joint change in FE and clustering creates mild ambiguity, but the core estimand is preserved.

5. **robust/outcome/tax_amount** and **robust/outcome/tax_amount_stratum_fe** and **robust/outcome/tax_amount_no_fe**: These three specs use tax amount in levels (taxes_paid_amt) with coefficients of ~42-47 Congolese Francs. They are clearly measuring a different outcome concept (intensive margin) than the baseline (extensive margin binary), supporting classification as non-core.

## Heterogeneity Classification Rationale

The 9 heterogeneity specifications are classified as non-core. While they use the same outcome (taxes_paid) and treatment (t_l), they restrict to specific demographic subgroups (male/female, Luluwa/other tribe, young/old, gov/non-gov workers) or add interaction terms (treatment x house type). These estimate conditional average treatment effects (CATEs) for subpopulations rather than the average treatment effect (ATE) that is the paper's core claim. If the paper's main argument were about differential effects by subgroup, these would be reclassified as core.

## Recommendations

1. **No script fixes needed for baselines**: The baselines correctly capture the paper's main result with appropriate escalation of FE structure.

2. **Consider reclassifying IHS tax amount**: If the paper treats IHS(tax amount) as a robustness check for the main compliance result (rather than a separate outcome), robust/outcome/ihs_tax_amount could be reclassified as core_funcform.

3. **Investigate exempt_status placebo failure**: The significant coefficient on the exempt_status placebo test (p=0.001) is notable. This should be cross-referenced with the paper's discussion of balance tests and randomization checks.

4. **Leave-one-out strata analysis is thorough**: The 10 leave-one-stratum-out and 3 leave-one-period-out specs provide good coverage for detecting outlier-driven results. All show consistent positive significant effects.
