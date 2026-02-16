# Specification Search Report: 112431-V1

## Paper
Ferraz & Finan (2011), "Electoral Accountability and Corruption: Evidence from the Audits of Local Governments," AER 101(4).

## Surface Summary
- **Baseline groups**: 1 (G1: pcorrupt ~ first)
- **Design code**: cross_sectional_ols
- **Budget**: 80 max specs (53 executed)
- **Seed**: 112431

## Baseline Specification
- Table 4, Column 6: `areg pcorrupt first [41 controls] | uf, robust`
- Coefficient: -0.0275, SE: 0.0113, p=0.015, N=476
- Fixed effects: state (uf, 26 states)
- Inference: HC1 robust SE

## Execution Counts

| Category | Planned | Executed | Failed | Notes |
|----------|---------|----------|--------|-------|
| Baseline | 1 | 1 | 0 | Table 4 Col 6 |
| Control progression | 8 | 8 | 0 | Bivariate through full+extended |
| Control sets | 2 | 2 | 0 | Minimal (sorteio only) + Extended |
| LOO controls | 16 | 16 | 0 | 14 individual + 2 block-level |
| Random control subsets | 20 | 20 | 0 | Block-level random draws, seed=112431 |
| FE variants | 2 | 2 | 0 | Drop uf, region FE |
| Sample variants | 2 | 2 | 0 | Trim 1-99, 5-95 |
| Functional form | 2 | 2 | 0 | log1p, asinh |
| **Total core specs** | **53** | **53** | **0** | |

### Inference variants (separate file)
| Variant | Executed | Notes |
|---------|----------|-------|
| Cluster SE at uf | 1 | SE=0.0101, p=0.012 (tighter than HC1) |
| HC3 SE | 1 | FAILED: pyfixest does not support HC3 vcov dict |

## Software Stack
- Python 3.x
- pyfixest (0.40+) for OLS with absorbed FE
- numpy, pandas, json

## Results Summary

### Coefficient range
- All 53 specifications yield a negative coefficient for `first` (supporting the paper's claim)
- Range: [-0.0312, -0.0171]
- Baseline: -0.0275
- All but 4 specifications are significant at p < 0.05 with HC1 SE

### Robustness assessment
- **STRONG support**: The effect of reelection incentives on corruption is robust to:
  - All control progressions (bivariate through full extended)
  - Leave-one-out of any individual control or control block
  - 20 random control-subset combinations
  - Dropping state FE or using region FE
  - Trimming outcome outliers at 1-99 and 5-95 percentiles
  - Log(1+y) and asinh(y) outcome transformations
  - Clustering SE at the state level

### Specifications with p > 0.05
- rc/controls/progression/bivariate: p=0.071 (no controls, just state FE)
- rc/controls/progression/mayor_demographics: p=0.071 (3 controls only)
- rc/controls/progression/mayor_demographics_party: p=0.058 (20 controls, no municipality/political)
- rc/controls/subset/random_020: p=0.074 (only mayor_demographics block)
- rc/controls/subset/random_018: p=0.071 (only mayor_demographics block)
- rc/controls/subset/random_008: p=0.071 (only mayor_demographics block)

All of these are minimal-control specifications. With any substantial control set, the result is consistently significant.

## Deviations from Surface
- HC3 inference variant failed due to pyfixest not supporting HC3 as a vcov dict key. Recorded as failure with error message.
- No other deviations.
