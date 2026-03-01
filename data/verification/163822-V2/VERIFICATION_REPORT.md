# Verification Report: 163822-V2

## Baseline Groups
- **G1**: Bonus/Limit treatments reduce phone usage (FITSBY apps)
  - Baseline spec_run_ids: ['163822-V2_run_001' through '163822-V2_run_008']
  - Baseline spec_ids: ['baseline', 'baseline', 'baseline__usage_p3_fitsby', 'baseline__usage_p4_fitsby', 'baseline__usage_p5_fitsby', 'baseline__usage_p2_total', 'baseline__usage_p2to5_fitsby', 'baseline__usage_p2to5_total']
  - Expected sign: negative (treatments reduce usage)

- **G2**: Limit treatment improves well-being
  - Baseline spec_run_ids: ['163822-V2_run_030' through '163822-V2_run_035']
  - Baseline spec_ids: ['baseline', 'baseline__addiction_index', 'baseline__sms_index', 'baseline__phone_use_change', 'baseline__life_better', 'baseline__swb_index']
  - Expected sign: positive (limit improves well-being)

## Counts
- **Total rows**: 54
- **Core**: 54
- **Non-core**: 0
- **Invalid**: 0
- **Baselines**: 14

## Category Breakdown
| Category | Count |
|----------|-------|
| core_method | 18 |
| core_controls | 15 |
| core_sample | 9 |
| core_data | 8 |
| core_funcform | 4 |

## Sign and Significance

### G1 (Phone Usage)
- All 38 specifications show negative coefficients (100%)
- Significant at 5%: 28/38 (74%)
- Coefficient range: [-55.927, -0.084]
- The 10 non-significant specs fall into clear patterns:
  - Total usage (not just FITSBY): weaker effect (p=0.19)
  - No controls / strata-only / diff-in-means: lose precision without baseline outcome control (p=0.05-0.16)
  - 5/95 trimming: marginal (p=0.098)
  - Balanced panel (all periods): marginal (p=0.055)

### G2 (Well-being)
- All 16 specifications show positive coefficients (100%)
- Significant at 5%: 15/16 (94%)
- Coefficient range: [0.040, 0.232]
- Only 1 non-significant: baseline__swb_index (SWB index, p=0.177)

## Assessment
- **MODERATE-TO-STRONG robustness**: The treatment effects are directionally robust across all specifications. G2 well-being results are strong (94% significant). G1 usage results are moderate (74% significant), with non-significance concentrated in specs that remove precision controls (expected in an RCT where baseline outcome control is standard) or change the outcome definition.
- The RCT design means control removal reduces precision but does not introduce bias, so non-significance in sparse-control specs is informative about precision, not about the treatment effect direction.
- Functional form transforms (log1p, asinh) maintain significance.
- Sample restrictions (balanced panels) maintain significance or are marginal.
- Time period variants (P3, P4, P5) all significant for G1 bonus treatment, showing persistent effects.

## Issues
1. **Duplicate spec_ids within paper**: Rows 001 and 002 both have spec_id='baseline' but differ in focal treatment (B vs L). Similarly, spec_ids like 'rc/controls/sets/none' appear in both G1 and G2. While spec_run_ids are unique, spec_ids are not unique within the paper. This is a minor schema issue.
2. **L-treatment G1 specs use non-standard spec_ids**: Rows 046-054 have spec_ids with '_L' suffixes (e.g., 'rc/data/outcome/p3_fitsby_L', 'rc/form/outcome/log1p_L') that are not in the surface core_universe but are coherent extensions for the second treatment arm.
3. **Inference mismatch between G1 and G2**: G1 uses HC1 (robust) as canonical, G2 uses cluster(UserID) as canonical. This is correct per the surface plan but worth noting.

## Recommendations
- Consider appending treatment arm to spec_id to avoid duplicates (e.g., 'baseline__p2_fitsby_B' and 'baseline__p2_fitsby_L').
- The L-treatment G1 specs (runs 046-054) should be documented in the surface core_universe for completeness.
- The SWB index non-significance in G2 may reflect a genuinely weaker effect on that particular outcome rather than a specification issue.
