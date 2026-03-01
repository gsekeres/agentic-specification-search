# Specification Search: 116139-V1

## Paper
Kosfeld and Rustagi (2015 AER) "Leader Punishment and Cooperation in Groups: Experimental Field Evidence from Ethiopia"

## Surface Summary
- **Baseline groups**: 2 (G1: Leader Type -> Cooperation; G2: Punishment Game)
- **G1 budget**: 60 core specs
- **G2 budget**: 20 core specs
- **Seed**: 116139
- **Design**: Randomized experiment (field experiment)

## Execution Summary

| Category | Count |
|----------|-------|
| Total specifications | 81 |
| G1 (Cooperation) | 69 |
| G2 (Punishment) | 12 |
| Baselines | 6 |
| Design variants | 2 |
| RC variants | 73 |
| Successful | 81 |
| Failed | 0 |
| Inference variants | 4 |

## G1 Specifications Breakdown

### Baselines (5)
- `baseline` (Table 6 Col 3): pct ~ leq leqef las + 8 controls | vcode, robust
- `baseline__table6_col1`: No controls, no FE
- `baseline__table6_col2`: Group controls, no FE
- `baseline__table6_col8`: + leader demographics
- `baseline__table6_col9`: Full 14 controls

### Design Variants (1)
- `design/randomized_experiment/estimator/diff_in_means`: No controls, no FE

### RC: Controls LOO (8)
- Drop each of 8 baseline controls individually (ccs, ed, pp, gs, wmk, time, fem, ginic)

### RC: Controls Single Additions (14)
- Extended: chet, shet, ginil, lage, ledu, lclanp (6)
- Appendix: turnover, leaderskill, clan1, clan2, clan3, peren, seas, slope (8)

### RC: Controls Sets (4)
- no_controls (village FE only)
- group_only (baseline controls, no FE)
- group_plus_leader (baseline + leader demographics + FE)
- full_plus_heterogeneity (all 14 controls + FE)

### RC: Controls Build-Up (4)
- stage1_demo (4 controls) -> stage2_econ (8) -> stage3_leader (11) -> stage4_full (14)

### RC: Controls Random Subsets (20)
- 20 random draws from 14-control pool, sizes 5-12, seed=116139

### RC: Fixed Effects (2)
- drop_village_fe: No FE
- village_fe_only: Only FE, no controls

### RC: Treatment Form (2)
- lcode_dummy: Factor dummies for leader classification code
- lpun_continuous: Continuous punishment score

### RC: Outcome Form (1)
- pct2: Alternative forest condition measure

### RC: Sample Restrictions (4)
- drop_influential_2: Drop 2 DFITS-influential observations
- drop_influential_4: Drop 4 DFITS-influential observations
- drop_vice_leaders: vlcode==0
- drop_lnp_leaders: lcode > 0

### RC: Joint (4)
- Controls x sample combinations (full controls with sample restrictions)

## G2 Specifications Breakdown

### Baseline (1)
- `baseline`: poisson pi i.cd1, vce(cluster fcode)

### Design Variants (1)
- OLS difference in means

### RC: Model Form (2)
- nbreg: Negative binomial
- ols_lpm: OLS linear model

### RC: Controls (4)
- Single additions: lage, ledu, lclan
- Leader demographics set: all 3 together

### RC: Outcome (1)
- pj: Alternative punishment measure

### RC: Sample Subsets (3)
- lcode==0 (Leqef only), lcode==1 (Leq only), lcode==3 (Las only)

## Inference Variants

### G1 (3)
- Baseline (Table 6 Col 3) re-estimated with:
  - Village-clustered SE (5 clusters)
  - HC2 standard errors
  - HC3 standard errors

### G2 (1)
- Baseline (Table 3 Poisson) re-estimated with:
  - Robust SE (no clustering)

## Deviations and Notes

1. **Wild cluster bootstrap**: Not available (wildboottest package not installed). Table A7 Cols 3-5 cannot be replicated.
2. **Table 7 panel FE**: Excluded (different dataset/estimand). Would require `Leq_Panel_AER_2014.dta`.
3. **pct2 outcome**: Only 25 non-missing observations (of 51), so the pct2 specification has very low power.
4. **Small sample**: N~47-51 for G1. Specifications with many controls (13-14) have very few degrees of freedom.
5. **Influential observations**: Used pre-computed `inf_lead` and `inf_lead2` flags from the dataset.
6. **G2 subsample specs**: May fail or have convergence issues for small leader-type subgroups (especially lcode==3, N=40).

## Software Stack
- Python 3.12.7
- pyfixest (OLS with FE and robust SE)
- statsmodels (Poisson, Negative Binomial)
- pandas, numpy
