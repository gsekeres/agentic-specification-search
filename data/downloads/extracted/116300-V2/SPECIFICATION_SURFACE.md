# Specification Surface: 116300-V2

## Paper: Brunnermeier & Nagel -- "Do Wealth Fluctuations Generate Time-Varying Risk Aversion? Micro-Evidence on Individuals' Asset Allocation" (AER)

This paper tests whether household portfolio allocation (risky asset share) responds to wealth changes, using PSID panel data. The null hypothesis of constant relative risk aversion (CRRA) predicts zero elasticity; the alternative (habit formation / loss aversion) predicts positive.

---

## Baseline Groups

### G1: Elasticity of risky asset share to wealth changes

- **Design**: Panel fixed effects (first-differenced OLS)
- **Panel unit**: PSID family (famid)
- **Panel time**: Survey year (biennial after 1997)
- **FE structure**: Year x region interactions (via `i.year*i.region`)
- **Differencing**: First difference (all variables in changes: dpstk, dfw)
- **Outcome**: dpstk (change in stocks/(financial wealth + other debt))
- **Treatment**: dfw (change in log financial wealth)
- **Estimand**: Elasticity of risky asset share to wealth changes
- **Target population**: Non-retired PSID households who are stock market participants, with financial wealth > $10,000, non-movers

The paper's Tables 4-5 present the core OLS results. The regressions include a very large set of controls capturing household demographics, income, employment status, education-age interactions, health, and wealth composition. The key coefficient is on dfw (change in log financial wealth).

The paper also presents IV results (Tables 4-5, 2SLS) instrumenting dfw with income dummies and inheritance as instruments for wealth changes. These are a separate estimator for the same claim object.

---

## Baseline Specs

- **Table4-Col1**: dpstk on dfw + full demographic controls, year*region interactions, cluster(famid)
- **Table4-Col2**: Same + wealth composition controls (dlabfw, dhomefw, dbusfw)
- Additional baselines: dpbhstk (broader risky asset share) on dtwnocar (wealth excl. cars); dpstk_dfw_with_composition

The paper runs separate regressions for the 1984-1999 and 1999-2003 sample periods. The published code defaults to the 1999-2003 period (`keep if year > 2000`).

---

## Core Universe

### Design variants
- First-difference OLS is the baseline estimator

### RC axes
- **Controls**: LOO drops of individual controls (inheritance, family composition, employment, wealth components); control set progressions (minimal demographics, demographics + income, full, full + composition); random control subsets
- **Sample period**: 1984-1999 vs 1999-2003 (the paper's main sample split)
- **Sample restrictions**: Higher wealth thresholds (fw > $50k, tw > $50k)
- **Outlier trimming**: Trim extreme changes in wealth and asset share
- **Functional form (outcome)**: dlogpstk (log change), dlogitpstk (logit change), dpstklev (level change), dpbhstk (broader risky share including business + housing)
- **Functional form (treatment)**: dtwnocar (total wealth excl. cars) as alternative to dfw
- **Weights**: Family weights (famwt) vs unweighted

### Excluded from core
- IV/2SLS regressions (instrumenting dfw with income dummies + inheritance) -- different estimator, could be treated as design variant but the instruments are endogeneity-addressing not standard RC
- Participation probit (Table 2) -- different outcome (extensive margin entry/exit)
- Inertia tests (capital gains decomposition) -- mechanism analysis
- Spline regressions -- functional form exploration
- Lagged wealth changes (Table 7) -- different specification testing timing

---

## Constraints

- Year*region interactions are mandatory (they absorb macro trends and aggregate wealth movements)
- Large control set (20-32 variables) reflecting the rich PSID demographic + economic data
- Control-count envelope: 20-32
- The paper uses the same control set across both sample periods

---

## Inference Plan

- **Canonical**: Cluster at family level (matching `cluster(famid)`)
- **Variants**: HC1 robust; bootstrap quantile regression (paper uses bsqreg in appendix)

---

## Budget

- Total core specs: up to 80
- Controls-subset sampling: 10 random draws (stratified by size)
- The large control set makes LOO + progression + random subsets the primary combinatorial axis
