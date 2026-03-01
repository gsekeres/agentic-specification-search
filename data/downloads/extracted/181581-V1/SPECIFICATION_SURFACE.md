# Specification Surface: 181581-V1

## Paper: When a Doctor Falls from the Sky: The Impact of Easing Doctor Supply Constraints on Mortality (Okeke)

## Baseline Groups

### G1: Effect of deploying a doctor on 7-day neonatal mortality

- **Outcome**: mort7 (binary: infant died within 7 days of birth)
- **Treatment**: doctor (indicator for Doctor Arm assignment) and mlp (indicator for MLP Arm assignment), both included simultaneously with control as reference
- **Estimand**: ITT effect of deploying a doctor to a primary health center on 7-day infant mortality
- **Population**: Live births to women enrolled in the study at participating primary health centers in Kaduna State, Nigeria
- **Baseline spec**: Table 4 Col 1 (reghdfe mort7 mlp doctor, abs(strata) cl(fid)) -- no individual controls, strata FE only
- **Alternative baseline**: Table 4 Col 2 adds basic individual-level controls (cont_base) and quarter-of-birth FE

### Why one baseline group

The paper's central claim is that deploying doctors to primary health centers reduces early neonatal mortality. Table 4 presents this result across four specifications with increasing controls (none, basic, extended, double-lasso). The MLP arm coefficient is included in all regressions but the headline finding is the doctor arm effect. Table 5 explores dosage heterogeneity (high vs low exposure), Tables 6-7 examine mechanisms (clinical quality, facility impacts), and Table 8 presents IV estimates using doctor assignment as an instrument for provider proficiency. These are all subordinate to the single main mortality claim.

## Experimental Design

- **Randomization**: Health facilities were randomly assigned to Control, MLP Arm, or Doctor Arm within randomization strata.
- **Treatment arms**: (0) Control -- no new provider deployed; (1) MLP Arm -- mid-level provider deployed; (2) Doctor Arm -- doctor deployed.
- **Unit of observation**: Live births (child-level). Clustering at facility (fid).
- **Strata**: Randomization strata (geographic/facility blocks).
- **Enrollment**: Pregnant women enrolled at baseline (woman_bl.dta), followed up after delivery (woman_el.dta).

## Core Universe

### Design estimator implementations

- **Difference in means**: Simple regression of mort7 on treatment dummies, no FE
- **Strata FE**: Include strata FE only (matches Table 4 Col 1)
- **With covariates**: Include strata FE + individual covariates (matches Table 4 Col 2-3)

### Controls axes

Table 4 reveals a clear progression of control sets:

- **No controls** (Col 1): Just strata FE
- **Basic controls (cont_base)** (Col 2): cct, magedum (maternal age), first (first birth), hausa (ethnicity), mschool (education), auton (autonomy), car, last (last birth in facility), gest (gestational age), male
- **Extended controls (cont_base + cont_hc)** (Col 3): Adds hc_deliveries, hc_cesarean, hc_transfusion, hc_clean (facility characteristics)
- **Double-lasso** (Col 4): Machine-selected from full control set (cont_all: cont_base + cont_hc + pastdeath, hc_workers, hc_open24hrs, hc_equipment, hc_beds, hc_lab, hc_drug, hc_nopow, hc_vent, hc_cond)

Control pool for subset sampling: individual controls (cct, magedum, first, hausa, mschool, auton, car, last, gest, male) + facility controls (hc_deliveries, hc_cesarean, hc_transfusion, hc_clean, pastdeath, hc_workers, hc_open24hrs, hc_equipment, hc_beds, hc_lab, hc_drug, hc_nopow, hc_vent, hc_cond).

Axes:
- **Standard sets**: 3 specs (none, basic, extended)
- **Progression**: 4 specs (strata only -> add qtr -> add individual -> add facility)
- **LOO from extended**: 12 specs (drop each of the individual + facility controls one at a time from extended set)
- **Subset search**: 10 budgeted random draws (seed=181581)

### Fixed effects axes

- Add quarter-of-birth FE (qtr) to strata-only specification
- Drop quarter-of-birth FE from specifications that include it

### Sample axes

- **Doctor arm only**: Drop MLP arm, compare only Doctor vs Control (Table 5 approach)
- **High dose**: Restrict to pregnancies with above-median exposure to deployed provider (Table 5 Panel B)
- **Low dose**: Restrict to pregnancies with below-median exposure (Table 5 Panel A)
- **Exclude multiple births**: Restrict to singleton births

### Outcome definition axes

- **mort30**: 30-day mortality instead of 7-day mortality. Same concept (neonatal mortality) with different cutoff.

### Treatment definition axes

- **Doctor only**: Drop mlp from the regression, estimate doctor effect vs pooled control+MLP

## Inference Plan

- **Canonical**: Cluster SEs at health facility (fid) level, matching the randomization unit. Used in all Table 4 specifications.
- **Variant 1**: HC1 robust SEs without clustering
- **Variant 2**: Cluster at strata level (coarser clustering)

## Constraints

- Control-count envelope: [0, 25]. Table 4 ranges from 0 controls (Col 1) to machine-selected controls (Col 4 double-lasso, potentially up to ~25 controls).
- No linkage constraints: single-equation RCT design.
- Both mlp and doctor treatment indicators are always included together in baseline specification. This is maintained across all core specifications except the "doctor_only" treatment definition RC.
- Sample excludes non-consenting women and pregnancies without live births.

## Budget

- Max core specs: 80
- Max control subset specs: 10
- Estimated total core specs: ~38 (1 baseline + 3 design + 3 control sets + 4 progression + 12 LOO + 10 subset + 2 FE + 4 sample + 1 outcome + 1 treatment = ~41)
- Seed: 181581

## What is excluded and why

- **Table 2 (effect on supply -- staffing)**: Different outcome (staffing levels). This is a first-stage / mechanism result, not the mortality claim.
- **Table 3 (effect on doctor care)**: Intermediate outcome (whether patient received care from a doctor). Mechanism analysis.
- **Table 5 (dosage effects)**: Dosage heterogeneity is included as sample restriction RC (high/low dose), but the full Table 5 with dose-specific coefficients is a form of heterogeneity analysis.
- **Table 6 (observed quality of treatment)**: Different dataset (patient.dta), different outcomes (clinical quality indicators), different unit of analysis. Belongs in explore/variable_definitions.
- **Table 7 (qualitative impacts)**: Self-reported facility impacts, different dataset (impact.dta). Mechanism analysis.
- **Table 8 (IV analysis: provider quality -> mortality)**: Changes the estimand from ITT to IV/LATE (provider proficiency instrumented by doctor assignment). Belongs in explore/alternative_estimands. The secondary_design_codes classification notes IV as a secondary design.
- **Double-lasso estimation (dsregress, Table 4 Col 4)**: This is a machine-learning-assisted estimator that changes the estimation approach. It could be included as rc/estimation/dml but is excluded from the default core because dsregress is Stata 17-specific and not easily replicated in Python. The core surface covers the standard OLS specifications.
- **In utero death outcome (inut)**: Different outcome concept.
- **Appendix tables**: Various robustness checks and extensions that can be evaluated after the core surface is run.
