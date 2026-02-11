# Spec Tree Coverage Checklist

This checklist tracks whether the spec tree covers the major degrees of freedom in empirical workflows **at the right level of statistical typing**.

Legend:

- ✅ implemented (has a module/design file in the new typed structure)
- 🟡 partially (exists only in legacy files or mixed typing)
- ❌ missing (needs a module/design)

## 1) Baseline claim object (verification)

- ✅ Baseline groups with outcome/treatment/estimand/population concepts (verification protocol + prompts)
- 🟡 Explicit “baseline estimand type” field (needs to be standardized)

## 2) Design / identification families (design-specific)

Current coverage exists as typed design files under `specification_tree/designs/`:

- ✅ DiD
- ✅ Event study
- ✅ RD
- ✅ IV
- ✅ Randomized experiment (RCT / field experiment)
- ✅ Synthetic control / SDID
- ✅ Panel FE
- ✅ Cross-sectional / selection-on-observables
- ✅ Discrete choice
- ✅ Dynamic panel
- ✅ Local projections
- ✅ SVAR
- ✅ Structural calibration / moments
- ✅ Bunching
- ✅ Duration / survival
- ✅ DSGE Bayesian

Major design-family gaps to consider adding (depending on scope):

- ❌ Shift-share / Bartik designs (common in applied micro)
- ❌ Gravity/trade-style panel designs (if treated as distinct)
- ❌ Structural demand / IO (if desired beyond discrete choice)

## 3) Robustness checks (RC; estimand-preserving re-specification)

- ✅ Data pre-processing & coding (`specification_tree/modules/robustness/preprocessing.md`)
- ✅ Data construction (merges/aggregation/panel building) (`specification_tree/modules/robustness/data_construction.md`)
- ✅ Controls / adjustment set (`specification_tree/modules/robustness/controls.md`)
- ✅ Sample restrictions (`specification_tree/modules/robustness/sample.md`)
- ✅ Fixed effects (`specification_tree/modules/robustness/fixed_effects.md`)
- ✅ Functional form & transformations (`specification_tree/modules/robustness/functional_form.md`)
- ✅ Weights (`specification_tree/modules/robustness/weights.md`)

## 4) Sensitivity analysis (assumption relaxations / partial-ID)

- ✅ Unobserved confounding sensitivity / partial-ID (`specification_tree/modules/sensitivity/unobserved_confounding.md`)
- ✅ Design-assumption sensitivity (IV/DiD/RD/RCT/synth) (`specification_tree/modules/sensitivity/assumptions/`)

Additional sensitivity axes to consider:

- ❌ Spillovers/exposure mapping sensitivity (often changes treatment concept → may be exploration)
- ❌ Data-vintage sensitivity (revisions, alternative vintages)

## 5) Inference modules

- ✅ Standard errors + clustering (`specification_tree/modules/inference/standard_errors.md`)
- ✅ Resampling (bootstrap, randomization inference) (`specification_tree/modules/inference/resampling.md`)
- ❌ Bayesian inference alternatives (if in scope)

## 6) Diagnostics (assumption checks / falsification)

- ✅ Placebos (`specification_tree/modules/diagnostics/placebos.md`)
- ✅ Unified design diagnostic menu (`specification_tree/modules/diagnostics/design_diagnostics.md`)
- ✅ General regression diagnostics (`specification_tree/modules/diagnostics/regression_diagnostics.md`)

## 7) Post-processing (set-level transforms)

- ✅ Multiple testing / multiplicity (`specification_tree/modules/postprocess/multiple_testing.md`)
- ✅ Specification-curve / multiverse summaries (`specification_tree/modules/postprocess/specification_curve.md`)

## 8) Exploration (concept/estimand changes)

- ✅ Alternative variable definitions (`specification_tree/modules/exploration/variable_definitions.md`)
- ✅ Heterogeneity / subgroup effects (`specification_tree/modules/exploration/heterogeneity.md`)
- ✅ CATE estimation (`specification_tree/modules/exploration/cate_estimation.md`)
- ✅ Policy learning (`specification_tree/modules/exploration/policy_learning.md`)
- ✅ Alternative estimands (`specification_tree/modules/exploration/alternative_estimands.md`)

## 9) Estimation wrappers

- ✅ DML as nuisance-learning layer (`specification_tree/modules/estimation/dml.md`)
