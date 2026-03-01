# Specification Surface: 149262-V2

**Paper**: Mixed-ability seating field experiment in Chinese elementary schools
**Design**: Randomized experiment (cluster-randomized at the classroom level)
**Created**: 2026-02-24

---

## Study Overview

This paper studies the effects of mixed-ability deskmate seating on academic performance and noncognitive (Big Five personality) outcomes in 36 elementary school classes across 4 schools in Longhui county, Hunan, China. Classes were randomly assigned to three arms:

- **Control**: traditional same-ability seating
- **MS (treat1)**: mixed-ability seating
- **MSR (treat2)**: mixed-ability seating with monetary rewards

Within each class, students are classified as lower track (hsco==0) or upper track (hsco==1). All main regressions are run separately by track. N=1,802 (901 per track).

Main estimation: ANCOVA-style OLS:
```
endline_outcome ~ baseline_outcome + treat1 + treat2 + covariates + grade_FE, cluster(class1)
```

---

## Baseline Groups

### G1: Academic Performance (Table 3)

- **Claim**: Mixed-ability seating affects lower-track students' academic performance
- **Focal table**: Table 3, Panel A, Column 2
- **Outcome**: ave3 (standardized average score)
- **Treatment**: treat1 (MS), with treat2 also included
- **Controls**: gender, age, height, hukou_rd1, nationality_rd1, health, sib, fa_eduy, mo_eduy, pc, car + i.grade1
- **Sample**: hsco==0 (N=901)
- **Clustering**: class1

### G2: Personality Traits (Table 4)

- **Claim**: Mixed-ability seating affects lower-track students' noncognitive skills
- **Focal table**: Table 4, Panel A (all five Big Five dimensions)
- **Primary outcome**: extra2 (extraversion); additional: agree2, open2, neur2, cons2
- **Same specification structure as G1

---

## Core Universe

### G1 (40 specs): Academic Performance
- 1 baseline + 4 additional baselines (col1, Chinese, math, upper-track)
- 4 design variants (diff-in-means, ANCOVA, with-covariates, strata-FE)
- 11 LOO controls + 4 control sets + 4 control progressions
- 3 sample variants (pooled, exclude noncompliers, pre-attrition)
- 5 outcome variants (Chinese, math, midterm, raw Chinese, raw math)
- 2 treatment variants (treat2 focal, pooled treatment)
- 2 FE variants (drop grade1, add school)

### G2 (31 specs): Personality Traits
- 1 baseline + 9 additional baselines (4 lower-track traits + 5 upper-track traits)
- 3 design variants
- 11 LOO controls + 2 control sets
- 2 sample variants
- 2 treatment variants
- 1 FE variant

### Total: 71 core specifications + 5 inference variants

---

## Inference Plan

- **Canonical**: Cluster at class1 (36 clusters)
- **Variants**: HC1 robust, cluster at grade1, randomization inference (500 permutations)
