# Specification Surface Review: 112805-V1

## Summary of Baseline Groups
- **G1**: Validity of value-added models (VAM coefficient = 1 under lottery IV)
  - Claim object is well-defined: lottery-based IV coefficient on school VAM should equal 1 if VAM is unbiased
  - The paper's main "specification curve" IS the VAM construction grid (model x estimation x sample x counterfactual = 36 combinations)
  - Additional baselines (Model 1 levels, Model 2 mix, Model 2 FE) correctly treated within the same group

## Changes Made
1. Explicitly noted the linked_adjustment constraint: VAM construction and IV controls are structurally coupled. The lagged test score controls in the IV second stage should match the VAM model type (Model 2 controls for lagged scores; Model 1 does not).
2. Added grade-band sample splits (elementary vs middle school) as sample RC variants.
3. Added math-only and reading-only outcome variants.
4. Confirmed that demographic controls (race, free lunch) are unavailable in the public-use data per the README.

## Key Constraints and Linkage Rules
- **Critical linkage**: VAM construction and IV specification are structurally linked. Model 2 VAMs control for lagged scores in the VAM; the IV second stage also includes lagged scores. These should not be independently varied.
- lottery_FE is the randomization unit and must be included as FE in all specs
- onmargin==1 defines the analysis sample (non-degenerate lottery groups)
- Bootstrap inference (100 reps) is computationally expensive but matches the paper

## Budget/Sampling Assessment
- ~45-55 planned specs is within the 60-spec budget
- The 36-cell VAM grid is the natural "specification curve" -- no random sampling needed
- Additional axes (outcome type, grade split, controls) add ~10-15 more specs

## What's Missing (minor)
- Models 3 and 4 (with demographics): unavailable due to data restrictions
- Year-weighted VAM (commented out in code): would require additional estimation
- 2004 outcomes (Table A2): could be included as an explore/* extension
- Reduced form (lottery -> test scores) is a useful diagnostic but not the main claim

## Data Note
The public-use data file has scrambled school IDs and no demographic variables. Results will differ slightly from the published paper per the README. This is acceptable for specification search purposes.

## Final Assessment
**APPROVED TO RUN.** The paper's structure naturally produces a grid of specifications through the VAM construction dimensions. The surface correctly identifies the linked adjustment constraint and the randomization-based identification. The main challenge is computational (bootstrap inference with FE-IV on ~5000 students across many specs).
