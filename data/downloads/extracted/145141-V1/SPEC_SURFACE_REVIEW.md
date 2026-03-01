# Specification Surface Review: 145141-V1

## Paper: Measuring the Welfare Effects of Shame and Pride (Butera, Metcalfe, Morrison, Taubinsky, 2022)

## Summary of Baseline Groups

- **G1**: ITT effect of public recognition on YMCA attendance (image -> attendance | coherent_sample)
  - Well-defined claim object with clean RCT identification.
  - Two baseline specs: Table 2 Col 2 (with past) and Col 3 (with past + beliefs).
  - Design: individual-level randomization, BDM arm excluded from reduced-form.

- **G2**: Within-subject effect of public recognition on real-effort task performance (SR -> pts | cluster(id))
  - Three baseline specs by sample (Prolific, Berkeley, BU).
  - Within-subject design with randomized round order, 3 obs per person.
  - All regressions include ownpay (financial incentives indicator) and order dummies.

- **G3**: Shape of the WTP function for public recognition (visits/interval -> wtp | cluster(id))
  - Four baseline specs: YMCA coherent (Table 3 Col 2) + Prolific/Berkeley/BU charity (Table 6 Col 2/4/6).
  - Individual-by-interval panel. Quadratic OLS recovering curvature ratio.
  - Most specification-rich group with OLS/Tobit, linear/quadratic, multiple sample restrictions.

## Changes Made

### 1. Fixed G1 baseline_spec_ids (consistency error)
- **Before**: `baseline__table2_col1_coh` and `baseline__table2_col3_coh`.
- **After**: `baseline__table2_col2_coh` and `baseline__table2_col3_coh`.
- **Rationale**: The actual baseline specs are labeled Table2-Col2-Coh and Table2-Col3-Coh. Col 1 (no controls) is not a named baseline; it is covered by `rc/controls/sets/none`.

### 2. Fixed G2 baseline_spec_ids (missing Prolific)
- **Before**: Only `baseline__table5_col2_berkeley` and `baseline__table5_col3_bu`.
- **After**: Added `baseline__table5_col1_prolific`.
- **Rationale**: All three samples are defined as baseline specs with labels. Omitting the Prolific baseline was an oversight -- it is the largest and arguably primary charity sample.

### 3. Removed `rc/sample/outliers/drop_high_pts_3000` from G2
- **Before**: Listed as an rc_spec_id suggesting it was an optional restriction.
- **After**: Removed from rc_spec_ids; documented in design_audit as a baseline data-cleaning step.
- **Rationale**: The code (`charity_reduced_form_analysis.do` line 55: `drop if high_pts == 1`) unconditionally removes observations with points > 3000 in ANY round before all analysis. This is not an optional sample restriction -- it is applied in the baseline. Adding it as an rc variant would be misleading (it cannot be "toggled off" without changing the baseline).

### 4. Clarified G2 mandatory controls (ownpay)
- **Before**: constraints.notes mentioned ownpay as "always included" but no `mandatory_controls` field.
- **After**: Added explicit `mandatory_controls: ["ownpay"]` and `optional_pool: ["o1", "o2"]`.
- **Rationale**: `ownpay` is the financial incentives round indicator. Dropping it would confound the SR coefficient with financial incentive effects. It is a second experimental condition, not a removable precision covariate. Making this explicit prevents the runner from accidentally generating specs without ownpay.

### 5. Corrected G3 YMCA interval count
- **Before**: design_audit stated "18 attendance intervals" for YMCA.
- **After**: Corrected to "11 attendance intervals" for YMCA, "18 points intervals" for charity.
- **Rationale**: The YMCA code (`ymca_reduced_form_pr_function.do`) creates 11 intervals: 0, 1, 2, 3, 4, 5-6, 7-8, 9-12, 13-17, 18-22, 23-28. The charity experiments have 18 intervals (0-99 through 1700+). The original surface conflated these.

### 6. Added YMCA visits coding detail to G3 design_audit
- **Before**: No documentation of the visits variable coding.
- **After**: Added `ymca_visits_coding` and `charity_interval_coding` fields.
- **Rationale**: The YMCA `visits` variable uses midpoints (0, 1, 2, 3, 4, 5.5, 7.5, 10.5, 15, 20, 26.5) while the charity `interval` variable uses (original_interval + 0.5) representing hundreds of points. These coding details are essential for reproducing the regressions and computing the curvature ratio.

### 7. Scoped Tobit and interval_idx specs as YMCA-only in G3
- **Before**: `rc/form/estimator/tobit_quadratic`, `rc/form/outcome/interval_idx_quadratic` etc.
- **After**: `rc/form/estimator/tobit_quadratic_ymca`, `rc/form/outcome/interval_idx_quadratic_ymca` etc.
- **Rationale**: The charity analysis code (`charity_reduced_form_analysis.do`) does NOT run Tobit regressions. Tobit is used only in the YMCA WTP analysis (`ymca_reduced_form_pr_function.do`). Similarly, the interval_idx alternative coding is a YMCA-specific rc axis. Adding `_ymca` suffixes prevents the runner from attempting Tobit on charity data or interval_idx on charity data.

### 8. Added Tobit censoring documentation
- **Before**: No documentation of Tobit censoring bounds.
- **After**: Added `tobit_censoring` field in design_audit and `tobit_note` in functional_form_policy.
- **Rationale**: Tobit uses `ll ul` (lower/upper limit). Slider WTP limits differ by sample: Prolific +/-$10, BU/Berkeley +/-$25. This is important for replication.

### 9. Added high_pts cleaning to G3 charity sample restrictions
- **Before**: G3 charity baseline specs did not document high_pts exclusion.
- **After**: Added `& high_pts==0` to sample restrictions for charity specs.
- **Rationale**: The charity data file is loaded after the `drop if high_pts == 1` step, so the WTP regressions are also run on the cleaned data. This should be documented for consistency.

### 10. Added G1 explicit mandatory/optional control pools
- **Before**: No `mandatory_controls` or `optional_pool` fields.
- **After**: Added `mandatory_controls: []` (empty, since no controls are required for identification in an RCT) and `optional_pool: ["past", "beliefs_w_image"]`.
- **Rationale**: Makes the control policy explicit and auditable.

### 11. Added reviewed_at timestamp and estimand field to design_audits
- Minor metadata additions for auditability.

## Key Constraints and Linkage Rules

- **G1**: No bundled estimator. Single-equation OLS with robust SEs. Controls serve precision only (RCT).
- **G2**: No bundled estimator. `ownpay` is mandatory (second experimental arm indicator). Order dummies (o1, o2) are optional precision controls. Clustering at individual level is required (within-subject design).
- **G3**: No bundled estimator. Quadratic term (visits2 / c.interval#c.interval) is part of the functional form, not a removable control in the traditional sense. Tobit vs OLS is an explicit estimator choice. YMCA and charity use different x-axis variables -- specs must not be mixed across experiments.

## Budget/Sampling Assessment

- **G1**: ~20 planned specs within 50-spec budget. Full enumeration feasible.
- **G2**: ~50 planned specs within 80-spec budget. Full enumeration feasible. Main variation is across 3 samples x sample restrictions x control sets x functional forms.
- **G3**: ~65 planned specs within 80-spec budget. Full enumeration feasible. Most variation comes from YMCA-specific axes (Tobit, interval coding, beliefs restrictions, top-interval exclusion) and charity-specific axes (top interval inclusion, close-to-score, pooling).
- All budgets are reasonable. No random subset sampling needed given the small control pools.

## What's Missing (minor)

1. **QM221 sample for G2/G3**: The BU QM221 sample runs the same regressions in the code but is documented as supplementary. Could be added as an additional baseline spec, but the paper does not present it as a main result. Correctly excluded.
2. **Ln(1+visits) squared for YMCA G3**: The code generates `ln_visits` and `ln_visits2` variables and labels them. This is already captured in `rc/form/outcome/ln_visits_quadratic_ymca`.
3. **Tobit for charity G3**: The charity code does not run Tobit. If desired, one could add Tobit as an analyst-added rc axis, but since it is not revealed by the paper's code for charity data, its absence is principled.
4. **Robust_sample for G1**: The code generates `All = robust_sample` (includes incoherent WTP respondents, excludes only BDM) but comments "not included as table in paper". This is already captured in `rc/sample/definition/robust_sample`.

## Final Assessment

**APPROVED TO RUN.** The surface is conceptually coherent, faithful to the manuscript's revealed search space, and the budget is feasible. The corrections above fix consistency errors in baseline_spec_ids, clarify mandatory vs optional controls, scope YMCA-only variants correctly, and document data-cleaning steps that are part of the baseline rather than optional robustness checks.
