# Specification Surface: 126722-V1

## Paper
Lopez, Sautmann, Schaner (AEJ Applied) - "Does Patient Demand Contribute to the Overuse of Prescription Drugs?"

## Design
Randomized experiment (field RCT) in 60 health clinics (CSComs) in Mali. Two voucher treatment arms (patient voucher, doctor voucher) randomly assigned at the clinic-day level. A separate patient information treatment is cross-randomized.

## Baseline Groups

### G1: Effect of voucher treatments on malaria treatment outcomes

**Claim object**: The paper's main claim is that patient demand contributes to antimalarial overuse. The key test is whether patient vouchers vs doctor vouchers differentially affect prescribing/purchasing of antimalarials. The primary outcomes are:
- `RXtreat_sev_simple_mal` (prescribed any antimalarial) -- Table 3, Col 2
- `treat_sev_simple_mal` (purchased any antimalarial) -- Table 3, Col 3
- `RXtreat_severe_mal` (prescribed severe malaria treatment) -- Table 3, Col 4
- `treat_severe_mal` (purchased severe malaria treatment) -- Table 3, Col 5
- `used_vouchers_admin` (used voucher, admin data) -- Table 3, Col 1

**Focal treatment variables**: `patient_voucher` and `doctor_voucher` (both binary). The focal coefficient is on `patient_voucher` (effect of giving patients direct voucher access vs control).

**Baseline estimator**: OLS with date fixed effects (DD1-DD35) and manual covariate controls, clustered at the clinic level (cscomnum_OS, ~60 clusters).

**Important note on pdslasso**: The paper's Table 3 uses double-selection LASSO (pdslasso) to select controls from a large candidate set (272 CC* variables = clinic dummies + patient characteristics + pairwise interactions). Since pdslasso is not available in Python, we replicate the manual covariate approach from Table B10 (Appendix), which uses a fixed set of ~20 patient/respondent covariates without LASSO selection. The Table B8 "no controls" specification is included as a design variant.

## Core Universe

### Design variants
- **diff_in_means**: Treatment dummies + date FE only (no additional controls). Corresponds to Table B8.
- **with_covariates**: Full manual covariate set (Table B10 approach). This is the baseline.
- **strata_fe**: Add clinic FE (CL1-CL59) since randomization was within-clinic.

### RC axes
1. **Controls LOO**: Drop each of the ~13 non-indicator individual covariates one at a time
2. **Controls sets**: none, minimal (symptoms only), extended (add home survey vars)
3. **Controls progression**: Build up from bivariate to full control set
4. **Controls subset**: Random draws from the covariate pool (seed=126722, 15 draws)
5. **Sample**: Trim outcome at 1%/99% and 5%/95%
6. **Sample restriction**: Limit to home survey subsample
7. **FE**: Add clinic FE; Drop date FE

### Inference plan
- **Canonical**: Cluster at clinic level (cscomnum_OS), CRV1 -- matches paper
- **Variant**: HC1 robust (no clustering) as stress test

## Constraints
- Controls count: 0 to 20 (from no-controls spec to full manual set)
- Binary outcomes: no functional form transforms (outcome is 0/1)
- Linked adjustment: false (simple OLS, no bundled estimator)

## Budget
- Target: ~65 specifications across the 5 outcome variables
- 15 random control subset draws
- Full enumeration of LOO, progression, sets, sample, and FE axes
