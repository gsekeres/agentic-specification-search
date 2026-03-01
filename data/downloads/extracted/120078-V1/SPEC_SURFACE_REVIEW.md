# Specification Surface Review: 120078-V1

## Summary of Baseline Groups

- **G1**: Effect of minority host status on listing price conditional on review accumulation (panel FE)
  - Well-defined claim object: within-listing effect of minority status on price as reviews accumulate
  - Baseline spec matches Table 5 Col 1 (`xtreg log_price ... c.minodummy#c.rev100 + rating interactions + lesX, fe i(newid) robust cl(newid), review > 0 & review < 40`)
  - Additional baselines (Table 5 Col 2: review < 60, Col 3: review < 80 with quadratic) are the same claim with different sample windows -- correctly included
  - Design code `panel_fixed_effects` is correct: within-listing FE regression

## Changes Made

1. **Fixed `design_audit.model_formula`**: The original formula incorrectly referenced `lastrat*_KKrho` and `minodummy_KKrho` from the Table 4 structural model. Updated to reflect the actual Table 5 reduced-form specification: `c.minodummy#ib10.wave + (c.lastrat7 ... c.minodummy)#c.rev100 + lesX + i.citywaveID | newid`.

2. **Corrected sample restriction**: Added `Drev100 > 0` to the sample restriction. The code in `2_reg.do` applies `keep if Drev100>0` at the top of the file (restricting to listings with nonzero review change across waves), in addition to the Table-5-specific `review > 0 & review < 40`.

3. **Added missing-value dummies note**: The `$lesX` global in `global_vars.do` includes `$missing` (about 20 missing-value indicator dummies for amenities, verification status, etc.). These are always included alongside the substantive property/host controls. The surface now documents this bundling. The n_controls count of 52 refers to substantive controls only; the runner should always include the missing-value dummies when any subset of property controls is included.

4. No changes to baseline group definition, RC axes, or budget.

## Key Constraints and Linkage Rules

- No bundled estimator: single-equation panel FE with absorbed listing FE
- Cluster at listing level (newid) matches paper's `robust cl(newid)`
- The interaction terms (`c.minodummy#ib10.wave`, `c.lastrat*#c.rev100`) are structural to the model and should always be included -- they are not optional controls
- Missing-value indicator dummies (`$missing`) are bundled with the property/host controls and should not be varied independently
- Year dummies in the control set (year2009-year2015) are part of `$loueur` and absorbed by the listing FE in practice; they may be collinear with citywaveID FE

## Budget/Sampling Assessment

- ~52 planned core specs within the 70-spec budget -- feasible
- 15 random control subset draws with seed=120078 is reproducible
- 13 LOO specs cover key substantive controls (shared flat, capacity, bedrooms, bathrooms, superhost, verification, description counts, pictures)
- 5 control progression specs provide informative build-up
- Review-count sample restriction variants (60, 80, 100) correctly mirror Table 5's own variations

## What's Missing (minor)

- No `rc/data/*` axis for alternative ethnicity classification (the paper uses a specific minority indicator; no alternative coding is obvious from the data)
- Table 5 Col 3 includes the quadratic term -- correctly captured as `rc/form/treatment/quadratic_rev100`
- No design estimator variants (e.g., first-difference or CRE) are included; this is appropriate given the panel structure and the paper's exclusive use of within-FE
- The `$missing` dummies could in principle be an rc/data variant (drop all missing dummies), but this is low priority and the surface correctly documents them as bundled

## Final Assessment

**APPROVED TO RUN.** The surface is conceptually coherent, faithful to the manuscript's revealed search space, and the budget is feasible. The model formula and sample restriction have been corrected to match the actual code.
