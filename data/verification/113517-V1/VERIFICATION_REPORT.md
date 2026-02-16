# Verification Report: 113517-V1

## Paper
Moscarini & Postel-Vinay (2017), "The Relative Power of Employment-to-Employment Reallocation and Unemployment Exits in Predicting Wage Growth", AER P&P.

## Baseline Groups

| Group | Outcome | Baseline spec_run_id | Baseline coef | Baseline SE | Baseline p |
|-------|---------|---------------------|---------------|-------------|------------|
| G1 | xdlogern_nom (log nominal earnings growth) | 113517-V1_run0001 | 0.0476 | 0.0008 | 0.000 |
| G2 | xdlogern (log real earnings growth) | 113517-V1_run0018 | 0.0406 | 0.0008 | 0.000 |
| G3 | xdloghwr_nom (log nominal hourly wage growth) | 113517-V1_run0035 | 0.0004 | 0.0003 | 0.192 |
| G4 | xdloghwr (log real hourly wage growth) | 113517-V1_run0052 | -0.0037 | 0.0004 | 0.000 |

## Summary Counts

| Metric | Count |
|--------|-------|
| Total rows | 68 |
| Core specs | 68 |
| Non-core | 0 |
| Invalid | 0 |
| Baselines | 4 |
| Failed | 0 |

## Category Breakdown

| Category | Count |
|----------|-------|
| core_method (baseline) | 4 |
| core_controls | 48 |
| core_sample | 4 |
| core_funcform | 4 |
| core_weights | 4 |
| core_fe | 4 |

## Key Findings

### G1 (Log Nominal Earnings): STRONG positive, robust
- Baseline: coef=0.0476, highly significant (p<0.001)
- All 17 specs show positive coefficient, range [0.036, 0.123]
- Even with cluster(mkt) SE, p=0.037
- Drop-FE spec amplifies coefficient to 0.123 (pooled cross-market variation)
- Most sensitive to: dropping xen (flow control), dropping xue (flow control)

### G2 (Log Real Earnings): STRONG positive, robust
- Baseline: coef=0.0406, highly significant (p<0.001)
- All 17 specs show positive coefficient, range [0.029, 0.115]
- With cluster(mkt) SE, p=0.037
- Similar pattern to G1 but slightly smaller magnitudes

### G3 (Log Nominal Hourly Wage): WEAK/AMBIGUOUS
- Baseline: coef=0.0004, NOT significant (p=0.192)
- With cluster(mkt) SE, p=0.945 -- completely insignificant
- Sign varies across specs: positive in EE-only (0.0078), near-zero in all-flows (0.0004), negative when dropping time trend (-0.003)
- 3 of 17 specs have p>0.05 with classical SE

### G4 (Log Real Hourly Wage): MIXED, sign-sensitive
- Baseline: coef=-0.0037, significant with classical SE (p<0.001)
- With cluster(mkt) SE, p=0.564 -- NOT significant
- Sign FLIPS: positive in EE-only (+0.0025), negative in all-flows (-0.0037)
- 1 spec (drop xue) has positive coefficient and p=0.50

## Issues Found

1. **No issues with data integrity**: All 68 specs produced valid estimates.
2. **No namespace violations**: All spec_ids are properly typed as baseline, rc/*.
3. **No infer/* rows in specification_results.csv**: Clean separation.
4. **No duplicate spec_run_ids**: All unique.

## Recommendations

1. **Inference concern**: The paper uses classical (IID) SE for the second-stage regressions. Since the variation in xee, xue, etc. is at the market*time level but the observations are individual-level, classical SE dramatically understate uncertainty. With cluster(mkt) SE, the G1/G2 results remain significant (p~0.04) but G3/G4 become completely insignificant (p>0.5). This is a substantive finding.

2. **Sign sensitivity in G4**: The coefficient on xee flips from positive (EE-only) to negative (all-flows) for real hourly wages. This suggests the positive EE-wage relationship is confounded by other flows that move with EE.

3. **The paper's main claim is well-supported for earnings (G1, G2)**: Even under aggressive inference, EE reallocation significantly predicts earnings growth. The claim about hourly wages being different is also supported -- they show much weaker/ambiguous patterns.
