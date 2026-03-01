# Specification Surface: 113744-V1

## Paper Overview
- **Title**: Promoting Healthy Choices: Information vs. Convenience (Wisdom, Downs, and Loewenstein, 2010)
- **Design**: Randomized experiment (field experiment)
- **Data**: Individual-level data from Subway restaurant experiments (N=638, SPSS format)
- **Key context**: Two studies with random assignment to calorie information and menu ordering treatments.

## Baseline Groups

### G1: Total Meal Calories (Table 3)

**Claim object**: Calorie information labeling and menu ordering (healthy default vs. unhealthy default) affect total caloric intake of Subway customers.

**Baseline specification**:
- Formula: `TotalCal ~ CalInfo + CalRef + HealthyMenu + UnhealthyMenu + Age + Female + AfrAmer`
- Outcome: `TotalCal` (total meal calories)
- Treatment: Four binary treatment indicators (CalInfo, CalRef, HealthyMenu, UnhealthyMenu)
- Controls: Age, Female, AfrAmer
- No fixed effects, no clustering
- Focal coefficient: HealthyMenu (the paper's primary finding is that menu ordering matters more than information)

**Additional baseline-like rows**:
- `baseline__table3_sandwich_cal`: Outcome = SandwichCal (sandwich calories only)
- `baseline__table3_non_sandwich_cal`: Outcome = NonSandwichCal (non-sandwich calories)

## RC Axes Included

### Controls
- **Leave-one-out**: Drop each of 3 baseline controls individually
- **Single additions**: Add Hunger, ChainFreq, DailyCal, Dieting, Overweight
- **Standard sets**: No controls (pure treatment comparison), demographics only (Age, Female, AfrAmer), demographics + eating habits (+ Hunger, ChainFreq, DailyCal, Dieting), full (all available)
- **Random subsets**: 15-20 random draws from the 7-variable optional pool

### Sample restrictions
- Outlier trimming at [1,99] and [5,95] percentiles of TotalCal
- Opened-seal only (customers who opened calorie seal for CalInfo treatment)
- Non-dieters only (Dieting==0)

### Functional form
- Log-transform of TotalCal
- Asinh-transform of TotalCal

### Preprocessing
- Winsorization of outcome at [1,99]

### Design variants
- Study 2 only (replication study)
- Study 1 only
- Pooled with Study2 fixed effect

## What Is Excluded and Why

- **ChoseLowCalSandwich outcome**: Binary outcome requiring different estimator (logit/probit); would change the focal parameter interpretation. Could be explore/* but excluded from core.
- **CalEstimate accuracy outcomes**: EstMinusActMealCal, ABSEstMinusActMealCal -- different claim object (calorie estimation accuracy vs. calorie consumption).
- **Interaction effects**: Study-by-treatment interactions are reported but represent heterogeneity, not the main ATE claim.
- **MealEnjoy outcome**: Satisfaction is a secondary outcome, not part of the main calorie consumption claim.

## Budgets and Sampling

- **Max core specs**: 65
- **Max control subsets**: 20
- **Seed**: 113744
- **Sampling**: Exhaustive for single additions and standard sets; stratified random for multi-control subsets.

## Inference Plan

- **Canonical**: HC1 robust standard errors (appropriate for individual-level experiment)
- **Variants**: Classical OLS, HC3 (small-sample)
