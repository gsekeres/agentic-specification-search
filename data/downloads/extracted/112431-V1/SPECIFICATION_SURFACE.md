# Specification Surface: 112431-V1 (Ferraz & Finan 2011)

## Paper Summary

Ferraz & Finan (2011) study whether electoral accountability reduces corruption in Brazilian municipalities. They exploit quasi-random audits by the CGU (Controladoria Geral da Uniao) and compare corruption levels between first-term mayors (who can seek reelection) and second-term, term-limited mayors. The main finding is that first-term mayors have significantly less corruption (pcorrupt is ~2.7pp lower), consistent with reelection incentives disciplining incumbents.

## Baseline Groups

### G1: Effect of reelection incentives on corruption (share of resources)

- **Outcome concept**: Share of audited resources found to involve corruption (`pcorrupt`)
- **Treatment concept**: First-term mayor indicator (`first`) -- captures reelection incentives
- **Estimand concept**: Conditional average difference in corruption between first-term and second-term mayors, controlling for observables + state FE
- **Target population**: 476 Brazilian municipalities audited by CGU lottery (esample2==1)
- **Baseline specification**: Table 4, Column 6 -- `areg pcorrupt first [full controls] | uf, robust`
  - Coefficient: -0.0275, SE: 0.0113, p=0.015
  - 41 controls + state FE, HC1 robust SE

**Why only one baseline group**: The paper's main claim is about `pcorrupt ~ first`. Tables 5 (ncorrupt, ncorrupt_os), Table 9 (convenios), and Table 8 (pmismanagement placebo) test related but distinct outcome concepts. Tables 10-11 test heterogeneity and robustness but preserve the same core claim object. We focus the specification surface on the main pcorrupt claim as it is the headline result.

## Control Structure

The paper reveals a clear control progression across Table 4 columns:

| Set | Controls | N controls |
|-----|----------|-----------|
| None (T4C1) | bivariate | 0 |
| Mayor demographics (T4C2) | pref_masc, pref_idade_tse, pref_escola + 17 party dummies | 20 |
| + Municipality (T4C3) | + lpop, purb, p_secundario, mun_novo, lpib02, gini_ipea, lrec_trans | 27 |
| + Political (T4C4) | + p_cad_pref, vereador_eleit, ENLP2000, comarca | 31 |
| + Lottery (T4C5) | + sorteio1-sorteio10 | 41 |
| + State FE (T4C6 = baseline) | same 41 controls + uf absorbed | 41 + uf FE |
| Extended (T5/T7/T8) | + lfunc_ativ and/or lrec_fisc | 42-43 |

**Control blocks** (for block-level variation):
1. `mayor_demographics`: pref_masc, pref_idade_tse, pref_escola (3 vars)
2. `party_dummies`: party_d1, party_d3-party_d18 (17 vars, move as block)
3. `municipality_chars`: lpop, purb, p_secundario, mun_novo, lpib02, gini_ipea (6 vars)
4. `fiscal`: lrec_trans (1 var)
5. `political`: p_cad_pref, vereador_eleit, ENLP2000, comarca (4 vars)
6. `audit_controls`: sorteio1-sorteio10 (10 vars, lottery dummies -- design controls)
7. `additional_fiscal`: lfunc_ativ, lrec_fisc (2 vars, used in extended specs)

**Mandatory controls**: None strictly mandatory, but lottery dummies (sorteio*) are strongly recommended as they account for the audit lottery design.

## Core Universe

### Design variants
- `design/cross_sectional_ols/estimator/ols` -- the paper's design class (OLS with absorbed FE)

### RC: Controls
- **Progression**: 8 build-up steps from bivariate to full+extended
- **Leave-one-out (LOO)**: Drop each control block (or individual variable for non-block vars) from baseline. 15 LOO specs.
- **Random subsets**: 20 random control-subset draws (stratified by size within the [0, 43] envelope, seed=112431)

### RC: Fixed Effects
- `rc/fe/drop/uf` -- no state FE (pooled OLS, as in T4C5)
- `rc/fe/add/region` -- replace state FE with broader region FE (5 Brazilian regions)

### RC: Sample
- `rc/sample/outliers/trim_y_1_99` -- trim pcorrupt at 1st/99th percentile
- `rc/sample/outliers/trim_y_5_95` -- trim pcorrupt at 5th/95th percentile

### RC: Functional Form
- `rc/form/outcome/log1p` -- log(1+pcorrupt) outcome transformation (pcorrupt is a share [0,1])
- `rc/form/outcome/asinh` -- asinh(pcorrupt) transformation

## Inference Plan

- **Canonical**: HC1 robust SE (matches paper's `robust` option). Used for all estimate rows.
- **Variants** (written to inference_results.csv):
  - Cluster SE at state level (uf) -- matches FE grouping, 26 clusters
  - HC3 -- small-sample leverage correction

## Budget

- Target: ~60 core specifications
- Control progression: 8 specs
- LOO: 15 specs
- Random subsets: 20 specs
- FE variants: 2 specs
- Sample variants: 2 specs
- Functional form variants: 2 specs
- Baseline: 1 spec
- **Total planned**: ~50 specs (well within 80-spec budget)

## What Is Excluded (and Why)

- **Table 5 outcomes** (ncorrupt, ncorrupt_os): Different outcome concept -- would be separate baseline groups if included
- **Table 8** (pmismanagement): Placebo outcome -- belongs in diagnostics, not core
- **Table 9** (convenios): Different dataset, panel structure, different outcome -- separate design
- **Tables 10-11** (heterogeneity/robustness interactions): Interaction terms change the interpretation of the `first` coefficient -- these are explore/heterogeneity
- **Table 6** (RDD polynomial controls): These are additional control-set variations captured in the control progression
- **Tobit/NegBin estimators** (T4C8, T5A-C4, T5B-C4): Alternative estimators for censored/count outcomes -- would be design variants for different baseline groups
