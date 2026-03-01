# Specification Search: 128143-V1
**Paper**: Douenne & Fabre (2022) 'Yellow Vests, Pessimistic Beliefs, and Carbon Tax Aversion'
**Design**: Randomized experiment (survey experiment with IV/2SLS)

## Surface Summary
- Baseline groups: 2 (G1: self-interest IV, G2: environmental effectiveness IV)
- G1 budget: max 60 core, 30 control subsets
- G2 budget: max 50 core, 25 control subsets
- Seed: 128143
- Surface hash: sha256:c2f941ae7b85724f5e95849d85b2f9d027a90a378108bec889084683b40f8ba5

## G1: Self-Interest IV (Table 5.2)
- Outcome: taxe_cible_approbation != 'Non' (carbon tax acceptance)
- Endogenous: non_perdant (believes does not lose from targeted dividend)
- Instruments: traite_cible, traite_cible_conjoint, interaction
- Method: Manual 2SLS (first-stage LPM, second-stage LPM with fitted values)
- Weights: survey weights
- Subsample: percentile_revenu in [10,60] (near income threshold)
- Baseline coefficient: 0.5569 (SE=0.1307, p=0.0000, N=1969)
- G1 specs: 35 rows (35 success)

## G2: Environmental Effectiveness IV (Table 5.4)
- Outcome: taxe_approbation != 'Non' (tax acceptance) / == 'Oui' (approval)
- Endogenous: taxe_efficace == 'Oui' (believes tax is environmentally effective)
- Instruments: apres_modifs (info on EE), info_CC (info on climate change)
- Method: Manual 2SLS with survey weights
- Baseline Col1 (Yes~Yes): coef=0.4361, SE=0.1666, p=0.0089, N=3002
- Baseline Col3 (notNo~Yes): coef=0.5135, SE=0.2403, p=0.0327, N=3002
- G2 specs: 22 rows (22 success)

## Execution Summary
- Total specification rows: 57 (57 success, 0 failed)
- Inference rows: 6 (6 success, 0 failed)

### G1 Breakdown (35 specs):
- baseline: 1 (subsample [p10,p60])
- baseline__fullsample: 1
- rc/form/outcome: 2 (approval Yes on subsample + full sample)
- rc/controls/loo (subsample): 7 (taxe_efficace, tax_acceptance, piecewise_income, demographics, hausse, simule_gain, prog_na)
- rc/controls/loo (full sample): 7 (same blocks)
- rc/controls/loo (individual demo factors, subsample): 8 (sexe, statut_emploi, csp, region, diplome4, taille_agglo, fume, actualite)
- rc/controls/loo (numeric demo, subsample): 5 (taille_menage, nb_14_et_plus, nb_adultes, uc, niveau_vie)
- rc/controls/loo/single: 1
- rc/controls/loo/cible: 1
- rc/weights/unweighted: 2 (subsample + full sample)

### G2 Breakdown (22 specs):
- baseline: 1 (approval Yes ~ effectiveness Yes)
- baseline__col3: 1 (acceptance notNo ~ effectiveness Yes)
- rc/form/outcome: 1 (approval Yes variant from col3 baseline)
- rc/controls/loo (full sample, notNo outcome): 4 (income, gains, gagnant, demographics)
- rc/controls/loo (individual demo factors): 8 (sexe, statut_emploi, csp, region, diplome4, taille_agglo, fume, actualite)
- rc/controls/loo/single: 1
- rc/controls/loo (approval_yes outcome): 4
- rc/weights/unweighted: 2 (notNo + Yes outcomes)

### Inference variants:
- G1: HC1 on baseline (subsample), HC2 on baseline (subsample), HC1 on full sample
- G2: HC1 on Col3 baseline, HC2 on Col3 baseline, HC1 on Col1 baseline

## Notes
- Manual 2SLS: SEs do not correct for generated-regressor uncertainty (matches paper).
- percentile_revenu computed from ERFS 2014 reference distribution (ecdf).
- Piecewise linear income terms: knots at percentile 20 and 70.
- Factor variables (sexe, statut_emploi, csp, region, diplome4, taille_agglo, fume, actualite)
  converted to dummies with first-category dropped.
- Linked adjustment: control sets vary jointly in first and second stage.

## Software
- Python 3.12.7
- statsmodels 0.14.6
- pandas 2.2.3
- numpy 2.1.3

## Deviations
- percentile_revenu reconstructed from ERFS reference distribution rather than loaded from RData.
- Factor variable dummification may differ slightly in category ordering from R's default.
- Some numeric columns required explicit conversion from string format in the CSV.
