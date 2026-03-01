# Specification Surface: 116139-V1

## Paper Overview
- **Title**: Leader Punishment and Cooperation in Groups: Experimental Field Evidence from Ethiopia (Kosfeld and Rustagi, 2015 AER)
- **Design**: Randomized experiment (field experiment)
- **Data**: Five datasets from a field experiment with Ethiopian forest user groups. Leaders are classified into behavioral types (conditional cooperators, efficiency-minded conditional cooperators, selfish) based on behavior in one-shot lab-in-the-field public goods games. Group-level and individual-level data on cooperation outcomes, leader characteristics, and experimental punishment game results.
- **Key finding**: Groups led by conditional cooperators exhibit significantly higher cooperation (better forest condition) than groups led by selfish leaders. In the experimental punishment game, conditional cooperator leaders punish free-riders significantly more than selfish leaders.

## Baseline Groups

### G1: Leader Type and Group Cooperation (Table 6, Col 3)

**Claim object**: The average effect of having a conditional cooperator leader (vs selfish/other type) on group cooperation (forest condition), conditional on group characteristics and village fixed effects.

**Baseline specification** (Table 6, Column 3):
- Formula: `pct ~ leq + leqef + las + ccs + ed + pp + gs + wmk + time + fem + ginic + i.vcode, robust`
- Outcome: `pct` (forest condition percentage, measure of group cooperation)
- Treatment: Three leader type dummies (`leq` = conditional cooperator, `leqef` = conditional cooperator with efficiency concerns, `las` = selfish). The omitted category is the non-punishment type.
- Controls: 8 group-level characteristics (ccs = common property size, ed = education, pp = previous protection, gs = group size, wmk = week market, time = time of establishment, fem = female share, ginic = Gini coefficient of cattle)
- Fixed effects: Village (5 villages, i.vcode)
- SE: Heteroskedasticity-robust
- Dataset: Leader_Group_AER_2014.dta (N ~ 49 groups, though 2 excluded for missing lcode)

**Additional baseline specs** (from Table 6):
- `baseline__table6_col1`: No controls, no village FE (difference in means)
- `baseline__table6_col2`: Group controls only, no village FE
- `baseline__table6_col8`: Full controls + leader demographics (lage, ledu, lclan) + village FE
- `baseline__table6_col9`: Full controls + leader demographics + heterogeneity measures (chet, shet, ginil) + village FE

**Why this is the primary claim**: Table 6 is the paper's main result table. The paper explicitly frames the leader-type to cooperation link as its central finding. Column 3 (with controls and village FE) is the preferred specification discussed in the text.

### G2: Leader Punishment Behavior (Table 3)

**Claim object**: The effect of experimental conditions on leader punishment behavior (count of punishment points), estimated via Poisson regression with session-clustered SE.

**Baseline specification** (Table 3):
- Formula: `poisson pi i.cd1, vce(cluster fcode)`
- Outcome: `pi` (number of punishment points assigned by the leader)
- Treatment: 9 experimental condition dummies (cd1 = 2..10, base = cd1 = 1). Conditions vary the contributions of two group members across efficiency and inequality frames.
- Controls: None
- Clustering: Session/group level (fcode)
- Dataset: Leader_Pun_Poisson_AER_2014.dta

## RC Axes Included (G1)

### Controls
- **Leave-one-out (LOO)**: Drop each of the 8 baseline controls individually (ccs, ed, pp, gs, wmk, time, fem, ginic)
- **Single additions**: Add each of 6 extended controls individually (chet, shet, ginil, lage, ledu, lclan)
- **Predefined sets**: No controls (Col 1), group-only (Col 2), group + leader demographics (Col 8), full including heterogeneity (Col 9)
- **Build-up progression**: Progressive addition from minimal to full control set
- **Random subsets**: 20 random draws from the 14-control pool, stratified by size (5-12)

### Additional controls from Appendix tables
- `turnover`: Leader turnover indicator (Table A10)
- `leaderskill`: Leader ability measure (Table A5 Col 1)
- `clan1`, `clan2`, `clan3`: Clan membership dummies (Table A6)
- `peren`, `seas`: Stream type indicators (Table A6 Col 7)
- `slope`: Geographic slope (Table A6 Col 8)

### Fixed effects
- **Drop village FE**: No spatial controls (like Col 2)
- **Village FE only**: No group-level controls, only village FE

### Treatment definition
- **Leader dummy (lcode)**: Single dummy for any leader type rather than separate type dummies (Col 4)
- **Continuous punishment (lpun)**: Use leader's punishment behavior as a continuous treatment (Table A11)

### Outcome definition
- **pct2**: Alternative cooperation measure (Table 7 Col 2 uses pct2 as outcome)

### Sample restrictions
- **Drop influential observations**: Drop 2 (Col 6) or 4 (Col 7) observations identified by DFITS
- **Drop vice leaders**: Exclude vice-leaders (vlcode==0 restriction, Col 5)
- **Drop non-punishment (LNP) leaders**: Exclude lcode==0 groups (Table A11 Col 3)

### Joint variations
- **Controls x sample**: Combine influential-observation drops with control set variations

## RC Axes Included (G2)

### Model specification
- **Negative binomial**: nbreg instead of Poisson (overdispersion check, footnote 21)
- **OLS / LPM**: Linear probability model for punishment > 0

### Controls
- **Leader demographics**: Add lage, ledu, lclan (footnote 22)
- **Individual additions**: Add each of the 3 leader controls separately

### Outcome
- **pj**: Alternative punishment measure (Table A2)

### Sample (within leader types)
- **By leader type**: Run separately for leqef (lcode==0), leq (lcode==1), las (lcode==3) -- mirrors Table 5

### Design alternative
- **Difference in means**: Simple mean comparison across conditions

## What Is Excluded and Why

- **Table 5 (within/across leader types)**: Table 5 analyzes punishment patterns within and across leader types. These are subsample analyses of the punishment game, not separate baseline claims. Treated as sample restriction RCs under G2.
- **Table 7 (panel FE)**: Table 7 uses panel data (Leq_Panel_AER_2014.dta) with group FE to examine within-group variation over time. This is a supporting analysis using a different dataset and design (panel FE rather than cross-sectional). Excluded from the core surface because it targets a different estimand (within-group change) and uses a different sample/structure. Could be added as an explore/* spec.
- **Table A4 (probit selection)**: Probit models for leader selection are diagnostic/balance checks, not estimates of the main causal parameter. Excluded.
- **Table A8 (sub-sample balance)**: Balance tests across sub-samples are diagnostics. Excluded.
- **Wild cluster bootstrap (Table A7)**: The paper reports wild cluster bootstrap with 400-1200 reps. This is treated as an inference variant, not a separate specification. Noted in the inference plan but not included as a programmed variant since the wildboottest package is not available in this environment.
- **Exploration**: No formal heterogeneity analysis or alternative estimand analysis is presented. No explore/* specs.

## Budgets and Sampling

- **G1 (Cooperation)**: Max 60 core specs. 8 LOO + 6 single additions + ~4 predefined sets + 20 random subsets + ~8 appendix control additions + ~3 FE variants + ~3 treatment variants + ~1 outcome variant + ~3 sample restrictions + ~4 joint specs = ~60.
- **G2 (Punishment)**: Max 20 core specs. Model variants (2) + control additions (4) + outcome variant (1) + sample subsets (3) + design alternative (1) + baseline = ~12.
- **Combined target**: ~80 total core specs.
- **Seed**: 116139

## Inference Plan

### G1
- **Canonical**: Heteroskedasticity-robust SE (HC1), matching `reg ... , robust`
- **Variants**: Village-clustered (5 clusters -- very few, Table A7 Col 2); HC2; HC3
- The paper's Table A7 also reports wild cluster bootstrap, but this is noted as infeasible in the current environment

### G2
- **Canonical**: Clustered at session level (fcode), matching `vce(cluster fcode)`
- **Variants**: Robust SE without clustering

## Key Linkage Constraints

- No bundled estimator in G1 (simple OLS with FE)
- G2 is a single-equation Poisson model (no bundling)
- Leader type dummies (leq, leqef, las) always enter jointly in G1 (they are the treatment of interest). Dropping one type dummy is not a valid RC because it changes the estimand (the omitted category shifts).
- Village FE (5 villages) are always included in the preferred specification. Dropping village FE is an explicit RC variant that changes what confounders are absorbed.
- Small sample (N ~ 47-49 for G1, conditional on lcode != .) means that some RC combinations may have insufficient degrees of freedom. The runner should check for convergence issues.
