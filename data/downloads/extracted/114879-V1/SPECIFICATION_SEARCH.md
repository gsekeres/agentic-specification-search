# Specification Search: 114879-V1

## Paper
Davis, Fuchs & Gertler (2014). "Cash for Coolers: Evaluating a Large-Scale Appliance
Replacement Program in Mexico." *American Economic Journal: Economic Policy*, 6(3), 207-238.

## Surface Summary

- **Baseline groups**: 1 (G1: usage ~ rrefr, DiD with TWFE)
- **Design**: difference_in_differences (TWFE with double-demeaning via reg2hdfe)
- **Baselines**: 2 (Table3-Col2-Random, Table3-Col3-Random)
- **Budget**: 70 max core specs
- **Seed**: 114879
- **Controls-subset sampling**: none (only 2 controls available)

## Data Availability

**CRITICAL: The central household billing data is NOT available.**

The paper uses a two-year panel of household-level electric billing records from the
Mexican Federal Electricity Commission (CFE). These data were provided to UC Berkeley's
Energy Institute under a Non-Disclosure Agreement (NDA). The ReadMe explicitly states:

> "This agreement prevents us from sharing these data with other researchers, though we
> have been able to post all code."

Only the following data files are included in the replication package:
- `census.dta` -- ancillary Mexican Census analysis (not for main regressions)
- `avgefficiencies.dta` -- ancillary appliance efficiency analysis

The code files are available:
- `national39.do` -- data construction from ~575 raw CFE billing files
- `mainregressions_random.do` -- main regressions with random control group
- `mainregressions_matched.do` -- regressions with location-matched control group
- `mainregressions_fancy.do` -- regressions with usage-matched control group + trend tests

## Execution Summary

| Category | Planned | Executed | Failed |
|----------|---------|----------|--------|
| Baselines | 2 | 0 | 2 |
| Design variants | 1 | 0 | 1 |
| RC variants | 56 | 0 | 56 |
| **Total spec rows** | **59** | **0** | **59** |
| Inference variants | 4 | 0 | 4 |

**All 59 specification rows and 4 inference rows recorded as failures.**
Reason: Confidential household billing data from CFE not available in replication package (NDA).

## Planned Specifications (all failed)

### Baselines
1. `baseline__col1_simple_fe`: Table 3 Col 2 -- usage ~ rrefr + rAC | hhXm + CxM, cluster(county)
2. `baseline__col3_ac_summer`: Table 3 Col 3 -- usage ~ rrefr + rAC + rAC_summer | hhXm + CxM, cluster(county)

### Design Variants
3. `design/difference_in_differences/estimator/twfe`: TWFE (only feasible estimator given massive FE dimensions)

### RC: Controls
4-8. LOO drops (rAC, rAC_summer), control set progressions (none, rAC, rAC+rAC_summer)

### RC: FE Structure
9-10. Simpler FE (hhXm + month) vs preferred (hhXm + CxM)

### RC: Sample Restrictions
11-16. Matched controls, no control HHs, drop transition month, summer/winter only

### RC: Outliers
17-19. Trim usage at 1/99, 5/95 percentiles; drop usage > 200000

### RC: Functional Form
20-21. Log and asinh transformations of usage

### RC: Control Group
22-24. Random, location-matched, usage-matched control groups

### RC: Combinatorial Grid
25-42. Control group x FE structure x control set (18 combinations)
43-59. Cross control group x sample restrictions, functional form x control group

### Inference Variants
- `infer/se/cluster/state` (coarser state-level clustering) -- for each baseline
- `infer/se/hc/hc1` (robust HC1 only) -- for each baseline

## Software Stack
- Python 3.12.7
- pandas, numpy (for output generation only; no estimation performed)

## Deviations from Surface
None. All specs faithfully enumerated from the surface but none could be executed due to
confidential data. The surface itself is well-designed for this paper's DiD structure.
