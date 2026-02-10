# Verification Report: 217741-V1

## Paper Information
- **Paper ID**: 217741-V1
- **Total Specifications**: 136
- **Baseline Specifications**: 2
- **Core Test Specifications**: 114
- **Non-Core Specifications**: 22

## Baseline Groups

### G1: Occupations with higher AI exposure (Webb patent-based measure) show larger incr
- **Expected sign**: +
- **Baseline spec(s)**: `baseline_Webb`
- **Outcome**: `DHSshEmployee`
- **Treatment**: `PCT_aiW`
- **Notes**: Coefficient: 0.076 (SE=0.042, p=0.068). Marginally insignificant at 5%. WLS with sector+country FE, country-sector clustered SE.

### G2: Occupations with higher AI exposure (Felten ability-based measure) show larger i
- **Expected sign**: +
- **Baseline spec(s)**: `baseline_Felten`
- **Outcome**: `DHSshEmployee`
- **Treatment**: `PCT_aiF`
- **Notes**: Coefficient: 0.185 (SE=0.052, p<0.001). Highly significant. The Felten measure shows much stronger results than Webb.

**Global Notes**: Paper: 'AI and Women's Employment in Europe' (AER P&P 2025). Cross-sectional WLS of DHS percent change in female employment share on AI exposure percentiles, with sector and country FE. Two AI measures diverge substantially: Felten is robust and significant, Webb is marginal. 136 specs split evenly between the two measures. Leave-one-out country and occupation specs test sensitivity. Sector-specific results show strong heterogeneity.

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baselines)** | **114** | |
| core_controls | 2 | |
| core_fe | 4 | |
| core_funcform | 4 | |
| core_inference | 4 | |
| core_method | 4 | |
| core_sample | 96 | |
| **Non-core tests** | **22** | |
| noncore_alt_outcome | 2 | |
| noncore_alt_treatment | 2 | |
| noncore_heterogeneity | 18 | |
| **Total** | **136** | |

## Verification Checks

- Total specs in CSV: 136
- Unique spec_ids: 136
- All spec_ids unique: Yes
- Every spec has a category: Yes
- Every spec has is_baseline + is_core_test: Yes
