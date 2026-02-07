# Verification Report: 145141-V1

**Paper**: Measuring the Welfare Effects of Shame and Pride
**Authors**: Butera, Metcalfe, Morrison, and Taubinsky
**Journal**: American Economic Review (2021)
**Verified**: 2026-02-03
**Verifier**: verification_agent

---

## Baseline Groups

### G1: YMCA Experiment
- **Claim**: WTP for public recognition of gym attendance increases with the number of gym visits.
- **Baseline spec_ids**: `baseline`
- **Outcome**: `wtp` | **Treatment**: `visits` | **Expected sign**: +
- **Baseline coefficient**: 0.103 (p < 1e-14, N=4070)

### G2: Charity Experiments
- **Claim**: WTP for public recognition of charitable giving increases with the giving interval.
- **Baseline spec_ids**: `charity/baseline/linear`, `charity/baseline/quadratic`
- **Outcome**: `wtp` | **Treatment**: `interval` | **Expected sign**: +
- **Baseline coefficients**: 0.175 (linear, p < 1e-50, N=25908), 0.233 (quadratic, p < 1e-20, N=25908)

---

## Summary Counts

| Category | Count |
|----------|-------|
| **Total specifications** | **74** |
| Baselines | 3 |
| Core tests (including baselines) | 47 |
| Non-core tests | 21 |
| Placebo (non-core) | 3 |
| Invalid | 3 |

### Category Breakdown

| Category | Count |
|----------|-------|
| `core_method` (baselines) | 3 |
| `core_sample` | 23 |
| `core_funcform` | 9 |
| `core_inference` | 6 |
| `core_controls` | 6 |
| `noncore_heterogeneity` | 21 |
| `noncore_placebo` | 3 |
| `invalid` | 3 |

---

## Classification Rationale

### Core tests (44 non-baseline specs)
- **Sample restrictions (23)**: Monotonic sample, close-to-belief subsamples, close-to-past subsamples, outlier exclusions (visits > 20/15/10), individual experiment subsamples (Prolific, Berkeley, BU from pooled baseline), winsorization, and interval endpoint exclusions. These all preserve the same treatment-outcome relationship on defensible subsamples.
- **Functional form (9)**: Quadratic, log, interval-index, and cubic transformations of the treatment variable. These change how the dose-response is parameterized but test the same directional claim.
- **Inference (6)**: Robust (HC) standard errors instead of the default clustered SEs. Point estimates are identical; only inference changes.
- **Controls (6)**: Adding group-size dummies, demographic controls (age, female), or running with no controls. These test sensitivity of the treatment coefficient to control variables.

### Non-core tests (21 heterogeneity + 3 placebo = 24)
- **Heterogeneity (21)**: Subsample splits by past attendance level (YMCA), median anonymity score (charity), gender, and age. Also includes interaction models (visits x past, interval x group size). These test whether the effect differs across subgroups, not whether the main effect exists. They are not comparable to the baseline as robustness checks because they split the sample into non-overlapping groups or add interaction terms that change interpretation of the main coefficient.
- **Placebo (3)**: Using anonymous-round performance (`anom_interval`) instead of recognition-round performance as the treatment. These are identification validation tests, not tests of the main claim.

### Invalid (3)
- `ymca/sample/excl_above_7`: Identical coefficient, SE, and N to `ymca/sample/excl_above_10`, suggesting the cutoff was not properly implemented.
- `charity/sample/prolific_trimmed`: Identical results to `charity/sample/prolific_quadratic` (same N=16490), meaning the 1% trimming removed zero observations.
- `charity/sample/berkeley_trimmed`: Identical results to `charity/sample/berkeley_quadratic` (same N=6528), same issue.

---

## Top 5 Most Suspicious Rows

1. **`ymca/sample/excl_above_7`** (spec 28): Exact duplicate of `excl_above_10` -- coefficient 0.7159, SE 0.0996, N=2590 are identical despite claiming a different cutoff (7.5 vs 10). The estimation script likely has a bug where the cutoff variable was not updated.

2. **`charity/sample/prolific_trimmed`** (spec 56): Claims to be a "Prolific trimmed 1%" sample but produces identical results to the unmodified Prolific quadratic spec (coef=0.1572, SE=0.0179, N=16490). The trimming either was not applied or the 1% threshold falls outside the data range.

3. **`charity/sample/berkeley_trimmed`** (spec 57): Same issue as above for the Berkeley sample. Identical to `charity/sample/berkeley_quadratic` (coef=0.3790, SE=0.0702, N=6528).

4. **`ymca/sample/close_past_2`** (spec 9): The only YMCA sample restriction that is not significant (p=0.376). This may reflect a genuine power issue (N=1446 with a very narrow subsample), but it is worth noting that the coefficient (0.341) is still positive and larger than the baseline. Not invalid, but warrants attention.

5. **`charity/heterogeneity/bu_below_med`** (spec 42): Not significant (p=0.357) and the coefficient (0.128) is notably smaller than the BU baseline. The BU sample is already the smallest (N=2890), and splitting it by median further reduces power (N=1479). This is correctly classified as non-core heterogeneity, but the contrast with the above-median split (coef=0.553, p=0.0003) is notable.

---

## Recommendations for the Spec-Search Script

1. **Fix the `excl_above_7` cutoff**: The script at `scripts/paper_analyses/145141-V1.py` likely has a copy-paste error where the visits exclusion threshold for the "excl_above_7" specification is actually set to 10 (or an equivalent condition). Verify the actual filtering logic.

2. **Fix the trimming logic**: The 1% trimming for the charity samples appears to not remove any observations. Check whether the trimming is applied to the correct variable (wtp) and whether the percentile thresholds are computed correctly. It is possible the WTP variable has a bounded support that makes 1% trimming a no-op.

3. **Consider adding the YMCA quadratic as a co-baseline**: The current baseline is linear-only for YMCA but the paper's preferred specification appears to be quadratic. Consider whether `ymca/form/quadratic` should be a co-baseline alongside the linear spec.

4. **Heterogeneity specs could include marginal effects**: The interaction models (e.g., `ymca/interact/past_linear`) report the main coefficient on visits, which is interpretable as the effect when the interacted variable is zero. If the goal is to recover a comparable estimate, consider also reporting the marginal effect at the mean of the interacted variable.

5. **Tobit models are missing**: The original paper uses Tobit for censored WTP outcomes. Adding Tobit specifications would improve comparability with the published results and could be classified as `core_method` variants.
