# Specification Surface Review: 112444-V1

## Summary

The surface was reviewed against the paper's TSP code, output files, and data structure. This is Reinhart & Rogoff (2011), a historical panel study of crisis contagion across 70 countries over up to 200 years.

## Baseline Groups

### G1: Banking Crisis Contagion
- **Status**: Correctly defined. The banking crisis equations (1, 3, 5, 7) form a coherent claim object.
- **Design code**: `panel_fixed_effects` is the classified design, though the actual implementation is pooled OLS (no country FE). This is noted in the design_audit.
- **Design audit**: Present. Notes the absence of country/time FE and the use of development-status dummies as group indicators.

### G2: External Debt Crisis Contagion
- **Status**: Correctly defined. The debt crisis equations (2, 4, 6, 8) form a separate claim object (different outcome).
- **Design code**: Same as G1.

No additional baseline groups needed. The three estimation periods (1824, 1900, 1946) are sample variations, not separate claims.

## Checklist Results

### A) Baseline groups
- Two baseline groups for two outcomes (banking crisis, debt crisis) -- correct.
- Period variations are correctly treated as RC sample restrictions, not separate baseline groups.

### B) Design selection
- `panel_fixed_effects` is the classified design. The actual estimation is pooled OLS with development dummies, which is a special case (no within-unit demeaning). The `design/panel_fixed_effects/estimator/within` variant will add actual country FE, which is a substantively important check.
- Design audit is present and informative.

### C) RC axes
- **Controls**: Appropriate. Small pool (develop1, develop2, bank_move, debt_move, center, public). LOO and add-one variations cover the space.
- **Sample**: Good coverage with period variations (matching the paper's three subperiods) and development-status subsamples.
- **Functional form**: Logit and probit are appropriate for binary outcomes; the paper itself estimates logit.
- **FE**: Adding country and year FE is a critical check since the baseline has none.
- No high-leverage axes appear to be missing.

### D) Controls multiverse policy
- G1: `controls_count_min=4`, `controls_count_max=5` -- correct for the paper's revealed range.
- G2: `controls_count_min=2`, `controls_count_max=5` -- correct.
- `linked_adjustment=false` -- correct, no bundled estimator.

### E) Inference plan
- Canonical HC2 matches TSP HCTYPE=2 -- correct.
- Clustering by country is an important variant given the panel structure.

### F) Budgets + sampling
- G1: 60 specs, G2: 40 specs -- reasonable given the small control pool and limited design complexity.
- Full enumeration is appropriate for the small control set.
- Seed specified (112444).

### G) Diagnostics plan
- Empty, which is appropriate for a pooled OLS setup with no endogeneity test baseline.

## Key Constraints and Linkage Rules
- No bundled estimators, no linked adjustment.
- Development dummies (develop1, develop2) should generally be kept together as a pair.

## What's Missing
- The TSP code references data files loaded from hardcoded Windows paths. The data may need manual assembly from the Excel files provided. This is a feasibility concern for the runner, not a surface design issue.
- The paper's three estimation periods and two equation systems (with/without public debt) are all captured.

## Final Assessment
**Approved to run.** The surface correctly identifies two baseline claim objects (banking and debt crisis contagion), defines appropriate RC axes, and sets reasonable budgets. The key innovation of adding country FE (which the paper lacks) is included. No blocking issues.
