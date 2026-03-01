# Specification Surface: 113630-V1

## Paper Overview
- **Title**: Experimental Evidence on the Long-Run Impact of Community Based Monitoring (Bjorkman Nyqvist, de Walque & Svensson, AEJ Applied)
- **Design**: Cluster-randomized controlled trial (two experiments)
- **Key claim**: Community-based monitoring interventions (providing information and facilitating participation) reduce child mortality and improve health outcomes in Uganda. Two experiments: (1) Information & Participation (I&P, 2004-2009), (2) Participation only (P, 2006-2009). Randomization at health facility level within district strata.

## Baseline Groups

### G1_mortality: Treatment Effect on Child Mortality

**Claim object**: ITT effect of community monitoring intervention on under-5 child mortality rate.

**Baseline specifications**:
- **Table3-PanelA-ColI**: Crude under-5 death rate, I&P experiment. `areg death_year treatment if sample1==1, a(dcode) robust`
- **Table3-PanelA-ColII**: Infant death rate (under 12 months).
- **Table3-PanelA-ColIV**: Exposure-corrected death rate (deaths per 1000 person-years).
- Additional baselines: Same outcomes for the P-only experiment (Panel B).

Data is collapsed to health facility level for the main mortality analysis. District (dcode) FE as strata controls. Robust SEs (already at cluster level).

### G2_weight: Treatment Effect on Child Nutritional Status

**Claim object**: ITT effect on weight-for-age z-score.

**Baseline specifications**:
- **Table5-ColI**: Weight-for-age z-score, infants 0-12 months, I&P experiment. `areg zw1 treatment if [age/z-score/outlier filters] & sample1==1, a(dcode) cluster(hfcode)`
- **Table5-ColII**: Same for older children 13-59 months.
- Additional baselines: P-only experiment versions.

Individual-level data clustered at health facility level.

## RC Axes Included

### G1_mortality
- **Controls**: Baseline facility characteristics (avg_charge_gentreat, avgOP_baseline, hhs) and their squares -- from Panel C cross-experiment comparison
- **Sample**: I&P only, P only, neonatal deaths only, post-2007
- **Outlier trimming**: Trim extreme death rates
- **Outcome transforms**: Crude, infant, neonatal, exposure-corrected death rates
- **Design**: Difference-in-means (no strata FE), Poisson with exposure offset
- **Weights**: Exposure weighting

### G2_weight
- **Controls**: Baseline facility characteristics
- **Sample**: I&P only, P only, infants only, older children only
- **Outlier trimming**: Tighter z-score bounds (3 SD), Cortinovis growth chart outlier exclusion
- **Outcome transform**: Height-for-age z-score as alternative

## What Is Excluded and Why

- **Table 4 (births/pregnancies)**: Different outcome concept (fertility, not health).
- **Table 6 (utilization)**: Health facility utilization is a mechanism, not the primary outcome.
- **Table 7 (processes/practices)**: Intermediate health facility quality measures.
- **Table 8 (immunization)**: Different outcome concept. Could be a separate baseline group but immunization is secondary to mortality/nutrition in the paper's framing.
- **Table 9 (local actions)**: Community engagement measures.
- **Panel C (cross-experiment comparison)**: Difference-in-differences comparing I&P vs P effects. This changes the estimand and belongs in `explore/*`.
- **Poisson individual-level models**: The paper reports Poisson regressions on individual data with exposure offsets. These are alternative estimators captured in `design/*`.

## Budgets and Sampling

- **G1_mortality**: Max 60 core specs
- **G2_weight**: Max 40 core specs
- **Total**: ~100 core specs
- **Control subsets**: Exhaustive (small pool of 3-6 controls)

## Inference Plan

### G1_mortality
- **Canonical**: HC1 robust (data at cluster level)
- **Variants**: Cluster at HF level, cluster at district level

### G2_weight
- **Canonical**: Clustered at health facility level
- **Variants**: HC1 robust, cluster at district level
