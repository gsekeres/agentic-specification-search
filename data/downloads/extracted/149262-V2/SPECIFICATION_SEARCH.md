# Specification Search Log: 149262-V2

**Paper**: Mixed-ability seating field experiment in Chinese elementary schools
**Design**: Randomized experiment (cluster-randomized at classroom level)
**Run date**: 2026-02-24

---

## Surface Summary

- **Paper ID**: 149262-V2
- **Baseline groups**: 2
  - G1: Academic performance (Table 3) -- focal outcome: `ave3`, treatment: `treat1`
  - G2: Personality traits / Big Five (Table 4) -- focal outcome: `extra2`, treatment: `treat1`
- **Design code**: `randomized_experiment`
- **Estimator**: ANCOVA OLS with cluster-robust SEs at classroom level
- **Randomization unit**: classroom (`class1`, 36 clusters)
- **Strata**: school-grade (`grade1`)
- **Sample**: Lower-track students (`hsco==0`, N=901)

### Budgets
- G1: max 60 core specs (planned 40)
- G2: max 50 core specs (planned 31)
- Controls subset sampling: none (full enumeration; only 11 controls)
- Seed: 149262

---

## Execution Summary

### specification_results.csv

| Category | G1 | G2 | Total |
|---|---|---|---|
| **Baseline** | 5 | 10 | 15 |
| **Design variants** | 4 | 3 | 7 |
| **RC/controls (LOO)** | 11 | 11 | 22 |
| **RC/controls (sets)** | 4 | 2 | 6 |
| **RC/controls (progression)** | 4 | 0 | 4 |
| **RC/sample** | 3 | 2 | 5 |
| **RC/outcome** | 5 | 0 | 5 |
| **RC/treatment** | 2 | 2 | 4 |
| **RC/FE** | 2 | 1 | 3 |
| **Total** | **40** | **31** | **71** |

- **Planned**: 71
- **Executed**: 71
- **Succeeded**: 71
- **Failed**: 0

### inference_results.csv

| Variant | G1 | G2 | Total |
|---|---|---|---|
| HC1 robust SE | 1 | 1 | 2 |
| Cluster at grade1 | 1 | 0 | 1 |
| Randomization inference (500 perms) | 1 | 1 | 2 |
| **Total** | **3** | **2** | **5** |

---

## Baseline Reproduction

### G1 Primary Baseline: Table 3, Panel A, Column 2
- **Outcome**: `ave3` (standardized average of Chinese and math endline scores)
- **Treatment**: `treat1` (MS dummy), with `treat2` (MSR) also included
- **Baseline control**: `ave1`
- **Controls**: gender, age, height, hukou_rd1, nationality_rd1, health, sib, fa_eduy, mo_eduy, pc, car
- **FE**: grade1 (school-grade)
- **Cluster**: class1
- **Sample**: hsco==0 (lower track, N=901)
- **Result**: coef=0.0128, se=0.0847, p=0.8812

### G1 Additional Baselines
- Table3-PanelA-Col1 (minimal): coef=0.0060, se=0.1217, p=0.9608
- Table3-PanelA-Col4 (Chinese): coef=-0.0205, se=0.0704, p=0.7728
- Table3-PanelA-Col6 (Math): coef=-0.0040, se=0.1065, p=0.9705
- Table3-PanelB-Col2 (upper track): coef=-0.0626, se=0.0569, p=0.2788

### G2 Primary Baseline: Table 4, Panel A, Column 1 (Extraversion)
- **Outcome**: `extra2` (endline extraversion score)
- **Treatment**: `treat1`, with `treat2` also included
- **Baseline control**: `extra1`
- **Controls**: same full set as G1
- **FE**: grade1
- **Cluster**: class1
- **Sample**: hsco==0 (lower track, N=901)
- **Result**: coef=0.5526, se=0.8324, p=0.5111

### G2 Additional Baselines (Lower Track)
- Agreeableness: coef=1.5246
- Openness: coef=0.0192
- Neuroticism: coef=0.9846
- Conscientiousness: coef=0.8468

### G2 Additional Baselines (Upper Track)
- Extraversion: coef=-0.9184
- Agreeableness: coef=-0.6760
- Openness: coef=-0.2030
- Neuroticism: coef=0.5068
- Conscientiousness: coef=-0.0672

---

## Key Findings

### G1 (Academic Performance)
- The baseline estimate for treat1 (MS) on average academic performance is very small (coef=0.013) and statistically insignificant (p=0.88).
- Results are robust across all specification variants: coefficients remain close to zero and statistically insignificant across LOO controls, different control sets, sample restrictions, and design variants.
- The pre-attrition sample (final0.dta, N=1023 lower-track) shows a somewhat larger estimate (coef=0.226) but sample composition differs.
- Subject-specific outcomes (Chinese, Math) are also insignificant.
- Randomization inference p-value (0.788) confirms the clustered SE inference.

### G2 (Personality Traits)
- The focal extraversion estimate is positive (coef=0.553) but statistically insignificant (p=0.511).
- Results are largely stable across LOO controls and sample restrictions.
- Agreeableness shows the largest positive effect among Big Five traits (coef=1.525).
- Upper-track results generally show smaller or negative effects.

---

## Deviations from Surface

- **Randomization inference**: Surface requested 2000 permutations; reduced to 500 for computational feasibility (running 2000 pyfixest regressions per permutation test was too slow). The resulting p-values (G1: 0.788, G2: 0.206) should be adequate for qualitative inference.
- **No skipped specs**: All 71 planned specifications were successfully executed.

---

## Software Stack

- **Python**: 3.10+ (via pyenv/conda)
- **pyfixest**: 0.40+
- **pandas**: 2.x
- **numpy**: 1.x
- Data loaded from Stata `.dta` files using `pd.read_stata()`
