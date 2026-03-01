# Specification Surface Review: 112451-V1

## Summary

The surface was reviewed against the paper's Stata do-files (brc_t4.do, brc_t5.do, brc_globals.do), data files, and the classification. This is Furman & Stern (2011), studying the effect of BRC deposits on article citations using a DiD design with negative binomial FE.

## Baseline Groups

### G1: Effect of BRC Deposit on Citations
- **Status**: Correctly defined. Single claim object (BRC deposit effect on citations).
- **Design code**: `difference_in_differences` is correct for the pre/post deposit comparison with article FE.
- **Design audit**: Present and comprehensive. Notes the conditional FE negative binomial estimator, bootstrap SE, and the staggered treatment timing.
- **No additional baseline groups needed**: Table 5 (substitution test) is a falsification check, not a separate main claim. The event study (Figure 2) is correctly treated as a diagnostic.

## Checklist Results

### A) Baseline groups
- Single baseline group for the BRC deposit effect -- correct.
- Table 5 is correctly excluded as a falsification/exploration check.

### B) Design selection
- `difference_in_differences` is appropriate. The design is DiD with staggered treatment timing across articles.
- `design/difference_in_differences/estimator/twfe` is included as a design variant (linear TWFE instead of negative binomial). This is appropriate.

### C) RC axes
- **Controls**: Limited but appropriate. The mandatory age/year dummies leave little room for control variation. The 3-4 substantive controls (window, age_brc1, post_brc_yrs) are correctly identified.
- **Sample**: Good coverage -- BRC-only sample, outlier trimming, short-panel exclusion.
- **Functional form**: Key axis. Poisson FE, OLS on logs, asinh transforms are appropriate alternatives to negative binomial. Grouped age dummies (matching Table 4, Cols 3-4) are included.
- **FE**: Pair FE is a natural addition for the matched design.
- **Data construction**: Matching variations are appropriate.

### D) Controls multiverse policy
- `controls_count_min=50`, `controls_count_max=55` -- reflects the reality that 50+ dummies are mandatory. Correct.
- `linked_adjustment=false` -- correct.
- The functional form policy note about IRR vs semi-elasticity interpretation is helpful.

### E) Inference plan
- Canonical bootstrap SE matches the paper -- correct.
- Cluster-robust at article level is an important variant (articles observed over many years).
- HC1 for OLS variants is appropriate.

### F) Budgets + sampling
- 60 specs total is reasonable. Most variation is from functional form and sample axes, not controls.
- Full enumeration is appropriate given the very limited control variation.
- Seed specified (112451).

### G) Diagnostics plan
- Pre-trends event study (Figure 2) is correctly included as a diagnostic with `baseline_group` scope.

## Key Constraints and Linkage Rules
- Age dummies and year dummies are always included together -- correctly enforced.
- The negative binomial FE estimator is the paper's canonical choice; alternatives (Poisson, OLS) are RC functional form variations.

## What's Missing
- Nothing material. The surface appropriately captures the limited control variation and focuses RC on functional form and sample axes where the paper has more room.
- Could consider adding `rc/data/matching/alternative_controls` for different matched control articles, but this would require re-running the matching algorithm, which may be infeasible.

## Final Assessment
**Approved to run.** The surface correctly identifies the single baseline claim object, defines appropriate RC axes with emphasis on functional form and sample variations (since control variation is limited), and includes the pre-trends diagnostic. No blocking issues.
