# Specification Search Log: 113517-V1

## Surface Summary
- **Paper**: Moscarini & Postel-Vinay (2017), "The Relative Power of Employment-to-Employment Reallocation and Unemployment Exits in Predicting Wage Growth", AER P&P
- **Baseline groups**: 4 (G1: log nom earnings, G2: log real earnings, G3: log nom hourly wage, G4: log real hourly wage)
- **Design**: cross_sectional_ols (two-stage procedure with absorbed market FE)
- **Canonical inference**: Classical (IID) SE (paper default -- areg without robust/cluster)
- **Seed**: 113517

## Execution Summary

### Planned vs Executed
- **Total core specs**: 68 (4 groups x 17 per group)
- **Inference variants**: 8 (2 per group: HC1, cluster by mkt)
- **Failed specs**: 0

### Specs by Type
- baseline: 4
- rc/controls/progression/*: 20
- rc/controls/loo/*: 24
- rc/controls/sets/*: 4
- rc/sample/*: 4
- rc/form/*: 4
- rc/weights/*: 4
- rc/fe/*: 4

### Deviations
- Added rc/controls/sets/minimal and rc/controls/progression/ee_ur beyond the original surface to reach 50+ specs.

## Software
- Python 3.x
- pyfixest (feols with absorbed FE)
- pandas, numpy

## Notes
- Two-stage procedure: first stage extracts market*time FE from individual-level regressions, second stage regresses predicted values on each other with market FE absorbed.
- First-stage controls are invariant across all specifications.
- Data has ~6M observations; memory-efficient approach used.
- Total runtime: ~827s
