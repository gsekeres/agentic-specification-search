# Specification Surface Review: 113684-V1

## Summary

The surface was reviewed against the paper's do-files (analysis.do, event_studies_reg.do, event_studies_dereg.do). This paper studies the persistent effects of affirmative action using event studies around federal contractor status changes. Data is confidential EEO-1 microdata not included in the package.

## Baseline Groups

### G1: Regulation Event Study
- **Status**: Correctly defined. The main claim is about the impact of gaining federal contractor status on black employment share.
- **Design code**: `event_study` is correct. The paper uses reghdfe with lead/lag indicators around the first_fedcon_yr event.
- **Design audit**: Present and accurate. Includes event window, FE structure, clustering, and reference period.

## Checklist Results

### A) Baseline groups
- Single baseline group for the regulation event study (Table 2 / Figure 3A) -- appropriate as the paper's headline result.
- Deregulation event study (Table 3 / Figure 3B) could be a separate baseline group but is better treated as a distinct claim about persistence rather than the main regulatory effect. The exclusion is documented and defensible.

### B) Design selection
- `event_study` is correct for the lead/lag indicator design with reghdfe.
- Design variants appropriately include the paper's own alternative FE structures, balanced panel, and parametric trend specifications.
- Design audit is concise and includes all interpretation-critical fields.

### C) RC axes
- **Controls**: Appropriately simple -- the paper uses only 2 time-varying controls (ln_est_size, ln_est_size_sq) across all specifications. LOO and no-controls variants cover the full space.
- **Sample**: Good coverage including balanced panel, pre-1998 events, and contractor losers.
- **FE**: Three alternative FE structures from the paper itself (div-year, msa-year, sic-div-year).
- **Missing**: No data construction or preprocessing axes, but this is appropriate given the confidential data constraint.

### D) Controls multiverse policy
- `controls_count_min=2` and `controls_count_max=2` -- correct for the baseline, though the RC axis allows going to 0.
- No linked adjustment needed (single-equation estimator).

### E) Inference plan
- Canonical inference (cluster at firm_id) matches all paper specifications.
- Variants (cluster at unit_id, HC1) are appropriate stress tests.

### F) Budgets + sampling
- Budget of 55 specs is reasonable given the limited control variation and the focus on design/FE/sample variants.
- No sampling needed; full enumeration is feasible.

### G) Diagnostics plan
- Pre-trends joint F-test and visual inspection are standard and appropriate for event studies.

## Key Constraints and Linkage Rules
- No bundled estimators.
- Control set is essentially fixed (2 controls); variation comes from FE structure and sample restrictions.

## What's Missing
- Deregulation event study is excluded but documented. Acceptable for a focused surface.
- No data available for actual execution; surface serves as a template.

## Final Assessment
**Approved to run (conditional on data access).** The surface correctly identifies the main claim object, defines appropriate design and RC axes matching the paper's own robustness checks, and sets reasonable budgets. The main caveat is that data is not included in the replication package.
