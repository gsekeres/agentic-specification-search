# Specification Surface Review: 112749-V1

## Summary

The specification surface for Hornbeck & Naidu (2014) defines two baseline groups:
- **G1**: Black labor/population outcomes (Table 2) -- the paper's central finding
- **G2**: Agricultural capital outcomes (Table 4) -- the mechanism/consequence channel

Both use panel FE with county FE, state-year FE, and geographic/historical controls.

## Verification Results

### A) Baseline Groups -- PASS
- G1 and G2 correspond to distinct claim objects (labor composition vs. capital substitution).
- Table 5 (farmland/land values) could be a third group but is closely related to G2 and omitting it keeps the budget reasonable.
- Table 1 (pre-differences) is correctly excluded as balance checks.
- Tables 3, 6, 7 are correctly excluded as alternative samples/treatments.

### B) Design Selection -- PASS
- `panel_fixed_effects` is the correct design code. The paper uses county FE with time-varying treatment intensity (not a standard DiD with a binary treatment-post interaction).
- The `design_audit` blocks are adequate, capturing estimator, FE structure, clustering, and weights.
- The long-difference design variant is appropriate for this setting.

### C) RC Axes -- PASS with minor notes
- Control variation axes (add/drop New Deal, geography, tenancy, plantation, propensity score) match the paper's revealed robustness structure from RefTables 1-3.
- Alternative flood measures (Red Cross data) are included, matching the paper's RobustnessMeasures tables.
- Sample restrictions are reasonable.
- **Note**: The `rc/functional_form/outcome/level_instead_log` axis changes coefficient interpretation (elasticity vs. level effect) but is a standard check. Acceptable.

### D) Controls Multiverse -- PASS
- The control count envelope (10-120) is wide enough to accommodate the variety of specifications in the paper.
- Controls are not linked across equations (each outcome has its own lagged DVs). This is correct.
- Mandatory controls: State-year FE should always be included (they are the paper's identification strategy anchor).

### E) Inference Plan -- PASS
- Canonical: county-level clustering matches the paper exactly.
- Variants: HC1 and state-level clustering are reasonable stress tests.
- Conley spatial SEs from the paper are not included as a variant (too complex to implement). This is acceptable.

### F) Budgets and Sampling -- PASS
- Target of 50-60 specs is adequate for this paper.
- Full enumeration is feasible given the discrete axes.

### G) Diagnostics -- N/A
- No diagnostics planned. This is acceptable for a panel FE design where the key identification assumption (exogeneity of flood) is not formally testable through standard diagnostics.

## Changes Made
- No changes to the surface. The surface is well-structured and ready to run.

## Decision: APPROVED TO RUN

The surface is coherent, faithful to the manuscript's revealed search space, and properly budgeted. Proceed to Stage 5.
