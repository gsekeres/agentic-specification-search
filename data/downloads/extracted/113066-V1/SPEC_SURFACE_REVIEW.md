# Specification Surface Review: 113066-V1

## Summary

The surface was reviewed against the paper's Stata do-files (AER-2015-0681_paper_dofile.do, AER-2015-0681_onlineappendix_dofile.do), data files, and the classification. This is Steinwender (2018), studying the effect of the transatlantic telegraph on cotton price convergence.

## Baseline Groups

### G1: Telegraph Effect on Cotton Price Difference (Level)
- **Status**: Correctly defined. Primary claim (telegraph reduced average price gap).
- **Design code**: `difference_in_differences` is appropriate for this before/after comparison. Though there is no cross-sectional variation, the before/after structure with Newey-West SE is the paper's identification.
- **Design audit**: Present. Notes daily time-series structure, Newey-West SE, and the before/after treatment.
- **Multiple baseline specs**: Three columns (raw, freight-adjusted, forward) correctly captured.

### G2: Telegraph Effect on Price Difference Variance
- **Status**: Correctly defined. Separate claim object (different outcome concept: variance vs level).
- **Design audit**: Correctly mirrors G1 structure with variance-specific outcome construction notes.

### Assessment of baseline group choices
- Two baseline groups for two distinct outcome concepts (level and variance of price gap) -- correct.
- Trade effects (Tables 8-9) are correctly excluded as secondary outcomes.
- IV specifications (Table 3) are correctly excluded as a different estimand.

## Checklist Results

### A) Baseline groups
- Two groups for level and variance effects -- correct.
- Tables 4-5 (ARIMA) and Tables 11-13 (structural) are correctly out of scope.

### B) Design selection
- `difference_in_differences` is the best fit for this before/after natural experiment with a single time series.
- No within-design estimator variants listed (no `design/*` spec_ids), which is appropriate -- there are no standard DiD estimator alternatives for a single interrupted time series.

### C) RC axes
- **Controls**: Appropriately limited. The paper has very few controls (only l1nyrec and freight measures). Full enumeration is trivial.
- **Sample**: Good coverage of no-trade-day exclusion, window width, and outlier trimming.
- **Outcome definitions**: The main source of variation (raw, freight-adjusted, forward). Well captured.
- **G2 variance outcomes**: Alternative dispersion measures (absolute deviation, log) are appropriate.

### D) Controls multiverse policy
- G1: `controls_count_min=0`, `controls_count_max=2` -- correct for this minimal-controls setup.
- G2: `controls_count_min=0`, `controls_count_max=1` -- correct.
- No mandatory controls -- correct.

### E) Inference plan
- Canonical Newey-West with 2 lags matches the paper -- correct.
- Bandwidth sensitivity (4 and 8 lags) is important for this time-series design. Good inclusion.
- HC1 as a comparison (no autocorrelation correction) is useful.

### F) Budgets + sampling
- G1: 55 specs, G2: 30 specs -- reasonable given the limited variation axes.
- Full enumeration of tiny control pool is appropriate.
- Seed specified (113066).

### G) Diagnostics plan
- Empty for both groups. Could consider a structural break test at the telegraph date, but this is more of a design validation than a standard diagnostic. Not blocking.

## Key Constraints and Linkage Rules
- Single binary treatment (tele = 0/1) with no intensity variation.
- Outcome construction (raw vs freight-adjusted) is the primary RC axis.
- The variance outcome (G2) requires within-period mean construction, which depends on the sample definition.

## What's Missing
- Could consider adding a placebo test using a different pre-telegraph event as the treatment date (shift the "tele" cutoff). This would be a useful diagnostic.
- The paper's Table 3 IV specifications (telegraph as instrument for information lag) could be included as `explore/*` rather than a separate baseline group.
- Trade effects (Tables 8-9) could be a third baseline group if the paper frames them as equally central, but the price convergence result is clearly the headline finding.

## Final Assessment
**Approved to run.** The surface correctly identifies two baseline claim objects (level and variance of the price gap), captures the limited but meaningful RC axes for this time-series natural experiment, and includes appropriate Newey-West bandwidth sensitivity. No blocking issues.
