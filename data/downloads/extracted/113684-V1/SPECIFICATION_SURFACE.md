# Specification Surface: 113684-V1

## Paper Overview
- **Title**: The Persistent Effect of Temporary Affirmative Action (Miller, 2017)
- **Design**: Event study
- **Data**: EEO-1 establishment-level panel (confidential; not included in replication package, only do-files provided)
- **Key limitation**: No data files in the package -- only Stata do-files. Specification surface is defined from code analysis only.

## Baseline Groups

### G1: Regulation Event Study (Table 2 / Figure 3A)

**Claim object**: Federal contractor status (affirmative action regulation) increases the black share of establishment employment, with effects that persist after deregulation.

**Baseline specification**:
- Formula: `f_black ~ first_fedcon_F6 + ... + first_fedcon_L6 + ln_est_size + ln_est_size_sq | unit_id + divXyear`
- Outcome: `f_black` (black share of employees)
- Treatment: Lead/lag dummies around first federal contractor year (event window [-6, +6], reference = t-1)
- Controls: `ln_est_size`, `ln_est_size_sq` (log establishment size and square)
- FE: Establishment (unit_id) + Division-by-Year (divXyear)
- Clustering: Firm level (firm_id)
- Focal parameter: Impact coefficient at t=0 (first_fedcon)

## Design Variants

The paper itself explores several within-design alternatives in Tables 2-3:
1. **Alternative FE structures**: MSA-by-year, SIC-by-Division-by-year (replacing Division-by-year)
2. **Balanced panel**: Restricting to establishments observed for full [-5,+5] window
3. **Earlier events only**: Restricting to events before 1998
4. **Contractor losers only**: Restricting to establishments that eventually lose contractor status
5. **Parametric trend**: Replacing lead/lag dummies with linear pre/post slopes

## RC Axes Included

### Controls
- **Leave-one-out**: Drop ln_est_size, drop ln_est_size_sq (only 2 controls)
- **No controls**: Omit both size controls entirely

### Sample restrictions
- Balanced panel (5-year window)
- Events before 1998
- Contractor losers subsample
- Outcome trimming at [1,99] and [5,95] percentiles

### Fixed effects
- Unit + year only (no geographic-time interaction)
- Unit + MSA-by-year
- Unit + SIC-by-Division-by-year

### Clustering alternatives
- Cluster at unit (establishment) level instead of firm
- Robust SE (no clustering)

## What Is Excluded and Why

- **Deregulation event study (Table 3)**: Separate claim object (deregulation persistence). Could be a second baseline group but the deregulation events are conceptually distinct. Excluded to keep surface focused on the main regulation effect.
- **Occupation-level estimates (Table 2-3 within-job)**: These use occupation-specific data subsets and represent within-job decompositions, not the main establishment-level claim.
- **Persistence analysis (Figure 5-6)**: Conditional on losing contractor status; separate estimand.
- **MCSUI employer discrimination data (Table 5)**: Different dataset entirely.
- **explore/***: No alternative outcome or treatment concepts to explore.

## Budgets and Sampling

- **Max core specs**: 55
- **No control-subset combinatorics**: Only 2 controls, so LOO and no-controls cover the full space.
- **Seed**: 113684
- **Full enumeration**: Feasible given small number of axes.

## Inference Plan

- **Canonical**: Cluster at firm (firm_id), matching all paper specs
- **Variants**: Cluster at unit (establishment), HC1 robust
