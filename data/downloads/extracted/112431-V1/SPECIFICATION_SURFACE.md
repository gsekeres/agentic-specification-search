# Specification Surface: 112431-V1 (Ferraz & Finan, AER 2011)

**Paper**: "Electoral Accountability and Corruption: Evidence from the Audits of Local Governments"

**Date created**: 2026-02-13

**Design classification**: `cross_sectional_ols` (high confidence)

---

## 1. Baseline Groups

This paper makes **three** distinct outcome claims that are treated as headline results, all sharing the same treatment concept (first-term mayor status) and target population (476 CGU-audited Brazilian municipalities). They correspond to three different corruption measures.

### G1: Share of resources involving corruption (`pcorrupt`)

- **Outcome concept**: Share of audited federal resources associated with corrupt practices (embezzlement, fraud, overbilling).
- **Treatment concept**: `first` = 1 if the mayor is in their first term (eligible for reelection), = 0 if in their second (final) term (term-limited).
- **Estimand concept**: Conditional ATE of first-term status on corruption share, adjusting for observables + state FE.
- **Target population**: 476 municipalities in the main sample (`esample2==1`).
- **Baseline spec**: Table 4, Col 6: `areg pcorrupt first [40 controls] if esample2==1, robust abs(uf)`. Coefficient = -0.0275, SE = 0.0113, p = 0.015.

This is the paper's primary claim and receives the most detailed specification surface.

### G2: Number of corruption violations (`ncorrupt`)

- **Outcome concept**: Raw count of corruption violations found by auditors.
- **Baseline spec**: Table 5A, Col 2: `areg ncorrupt first lrec_fisc [controls] | uf`. Coefficient = -0.471, SE = 0.148, p = 0.002.
- Includes `lrec_fisc` (log audited resources) as a mandatory control since audit scale mechanically determines how many violations can be found.

### G3: Corruption violations as share of audited items (`ncorrupt_os`)

- **Outcome concept**: Number of corruption violations divided by total service orders (normalized by audit intensity).
- **Baseline spec**: Table 5B, Col 2: `areg ncorrupt_os first lrec_fisc lfunc_ativ [controls] | uf`. Coefficient = -0.0105, SE = 0.0044, p = 0.017.
- Includes both `lrec_fisc` and `lfunc_ativ` as mandatory controls.

### Why these three groups (and not more)?

- **Tables 4-5** present the paper's three headline corruption measures. They are reported prominently and interpreted as the main evidence.
- **Table 6** (RDD polynomials) is a robustness check for G1, not a separate claim. It lives in the G1 surface as functional-form RC.
- **Table 7** (experience controls) is a robustness check for G1 addressing an identification concern (confounding by political experience).
- **Table 8** (`pmismanagement`) is a **placebo test** (mismanagement should not respond to reelection incentives). This is a diagnostic, not a baseline claim.
- **Table 9** (convenios/matching grants) uses a different dataset (panel) and different outcome concept. It is a secondary mechanism result, not a core corruption claim. Excluded from the core surface.
- **Table 10** (heterogeneous effects) is exploration -- interaction terms with moderators (judicial presence, media, voter registration, political competition). Not a baseline claim.
- **Table 11** (robustness checks) tests manipulation concerns. Cols 1-3 test pcorrupt robustness; Col 4+ tests whether audit resources (`lrecursos_fisc`) respond to treatment (placebo/manipulation check).

---

## 2. Revealed Search Space Analysis

The paper reveals a rich specification surface through its own table structure:

### Control set progression (Table 4, Cols 1-6)
The paper's own build-up sequence defines 6 natural control sets:

1. **Bivariate**: treatment only (0 controls)
2. **Mayor characteristics** (`prefchar2`): gender, age, education, 16 party dummies (19 vars)
3. **+ Municipal characteristics** (`munichar2`) + fiscal: population, urbanization, education, new municipality, GDP, inequality, transfer revenue (7 vars added = 26 total)
4. **+ Political competition**: voter registration rate, council composition, effective number of parties, judicial district (4 vars added = 30 total)
5. **+ Lottery dummies** (`sorteio1`-`sorteio10`): audit lottery round indicators (10 vars added = 40 total)
6. **+ State FE** (`uf`): absorbed, same 40 controls

This progression is the paper's main revealed control-set variation.

### Additional controls appearing in later tables
- `lfunc_ativ` (log number of activities audited): appears in Tables 5-11 but not in Table 4 main spec
- `lrec_fisc` (log audited resources): mandatory for count outcomes, also outcome in Table 11B
- Experience variables (`exp_prefeito`, `nexp`, `nexp2`): Table 7 only

### Fixed effects
- **State FE** (`uf`, 26 states): used in all main specs (Tables 4 Col 6 onward)
- **No FE**: Table 4 Cols 1-5
- Dropping state FE is a natural robustness check

### Sample restrictions revealed
- Full sample: `esample2==1`, N=476
- Running variable non-missing: N=328 (Table 6)
- Experience subsamples: N=312, 294, 286, 310 (Table 7)
- `pmismanagement` non-missing: N=366 (Table 8)

### Inference
- All main specs use HC1 robust standard errors
- Table 9 (panel) uses cluster-robust SEs at municipality level, but this is a different baseline group

---

## 3. Core-Eligible Universe (G1: pcorrupt)

### A. Control sets (revealed progression)

| spec_id | Description | Source |
|---------|-------------|--------|
| `rc/controls/sets/none` | Bivariate (treatment + FE only) | Table 4, Col 1 analogue with FE |
| `rc/controls/sets/prefchar2` | Mayor characteristics only + FE | Table 4, Col 2 analogue with FE |
| `rc/controls/sets/prefchar2_munichar2_fiscal` | + municipal chars + fiscal | Table 4, Col 3 analogue with FE |
| `rc/controls/sets/prefchar2_munichar2_fiscal_political` | + political competition | Table 4, Col 4 analogue with FE |
| `rc/controls/sets/prefchar2_munichar2_fiscal_political_sorteio` | + lottery dummies | Table 4, Col 5 analogue with FE (= baseline without FE) |
| `rc/controls/sets/full_with_lfunc_ativ` | All controls + lfunc_ativ | Full set used in Tables 6-11 |

**Count**: 6 specs

### B. Leave-one-out at the block level

Drop one control block at a time from the baseline set (40 controls, state FE):

| spec_id | Dropped block | Vars removed |
|---------|---------------|--------------|
| `rc/controls/loo_block/drop_prefchar2_continuous` | Mayor demographics (continuous) | 3 vars |
| `rc/controls/loo_block/drop_prefchar2_party` | Party dummies | 16 vars |
| `rc/controls/loo_block/drop_munichar2` | Municipal characteristics | 6 vars |
| `rc/controls/loo_block/drop_fiscal` | Fiscal transfer revenue | 1 var |
| `rc/controls/loo_block/drop_political` | Political competition | 4 vars |
| `rc/controls/loo_block/drop_sorteio` | Audit lottery dummies | 10 vars |

**Count**: 6 specs

### C. Leave-one-out at the individual variable level (key continuous/non-dummy controls)

Drop one individual control variable from the baseline set. Restricted to the 14 non-dummy, non-block-tied controls to keep the battery interpretable:

| spec_id | Variable dropped |
|---------|-----------------|
| `rc/controls/loo/drop_pref_masc` | Mayor gender |
| `rc/controls/loo/drop_pref_idade_tse` | Mayor age |
| `rc/controls/loo/drop_pref_escola` | Mayor education |
| `rc/controls/loo/drop_lpop` | Log population |
| `rc/controls/loo/drop_purb` | Urban share |
| `rc/controls/loo/drop_p_secundario` | Secondary education share |
| `rc/controls/loo/drop_mun_novo` | New municipality |
| `rc/controls/loo/drop_lpib02` | Log GDP per capita |
| `rc/controls/loo/drop_gini_ipea` | Gini coefficient |
| `rc/controls/loo/drop_lrec_trans` | Log transfer revenue |
| `rc/controls/loo/drop_p_cad_pref` | Voter registration rate |
| `rc/controls/loo/drop_vereador_eleit` | Council composition |
| `rc/controls/loo/drop_ENLP2000` | Effective number of parties |
| `rc/controls/loo/drop_comarca` | Judicial district indicator |

**Count**: 14 specs

### D. Exhaustive block-combination subset search

With 6 control blocks, the full block-combination space has 2^6 = 64 combinations (including the empty set = bivariate and the full set = baseline). This is feasible for full enumeration.

Each combination includes the treatment variable + state FE + the union of selected blocks.

**Count**: 64 specs (includes baseline and bivariate as special cases; net new = 64 - 2 already counted = 62)

### E. Fixed effects variants

| spec_id | Description |
|---------|-------------|
| `rc/fe/drop/uf` | Drop state FE (pooled OLS with all controls) |
| `rc/fe/add/region` | Replace state FE with broader region FE (if feasible) |

**Count**: 2 specs

### F. Sample restrictions

| spec_id | Description | Motivation |
|---------|-------------|------------|
| `rc/sample/outliers/trim_y_1_99` | Trim pcorrupt to [1%, 99%] | Outlier sensitivity |
| `rc/sample/outliers/trim_y_5_95` | Trim pcorrupt to [5%, 95%] | Stronger trimming |
| `rc/sample/outliers/cooksd_4_over_n` | Drop high-influence observations | Influence diagnostic |
| `rc/sample/restriction/running_nonmissing` | Restrict to municipalities with non-missing running variable | Table 6 sample (N=328) |
| `rc/sample/restriction/pmismanagement_nonmissing` | Restrict to municipalities with non-missing pmismanagement | Table 8 sample (N=366) |

**Count**: 5 specs

### G. Functional form (RDD-style polynomial controls)

These are robustness checks that add polynomial controls in the margin of victory (running variable), following Table 6's approach. They are RC (not design changes) because the paper treats them as parametric controls, not formal RDD estimation.

| spec_id | Description | Source |
|---------|-------------|--------|
| `rc/form/model/rdd_polynomial_linear` | Add linear running variable control | Table 6, Col 2 |
| `rc/form/model/rdd_polynomial_quadratic` | Add quadratic in running variable | Table 6, Col 3 |
| `rc/form/model/rdd_polynomial_cubic` | Add cubic in running variable | Table 6, Col 4 |
| `rc/form/outcome/asinh` | asinh(pcorrupt) instead of pcorrupt | Handles zero-inflated outcome |

Note: RDD polynomial specs necessarily restrict to running_nonmissing sample (N=328).

**Count**: 4 specs

### H. Estimator variant

| spec_id | Description | Source |
|---------|-------------|--------|
| `rc/estimation/tobit_ll0` | Tobit with left-censoring at 0 | Table 4, Col 8 |

**Count**: 1 spec

### I. Inference variants

| spec_id | Description | Motivation |
|---------|-------------|------------|
| `infer/se/hc/classical` | Classical (homoskedastic) SEs | Baseline comparison |
| `infer/se/hc/hc2` | HC2 robust SEs | Small-sample correction |
| `infer/se/hc/hc3` | HC3 robust SEs | Conservative small-sample correction |
| `infer/se/cluster/uf` | Cluster SEs at state level | Natural clustering unit for state FE (26 clusters) |

**Count**: 4 specs

### G1 Total Budget

| Category | Count |
|----------|-------|
| Baseline | 1 |
| Control sets (progression) | 6 |
| LOO blocks | 6 |
| LOO individual variables | 14 |
| Block-combination subsets | 62 (net new) |
| FE variants | 2 |
| Sample restrictions | 5 |
| Functional form | 4 |
| Estimator variant | 1 |
| Inference variants | 4 |
| **Total** | **105** |

This is within the 150-spec budget. Full enumeration is feasible; no random sampling required.

---

## 4. Core-Eligible Universe (G2: ncorrupt, G3: ncorrupt_os)

G2 and G3 share the same treatment, population, and identification strategy as G1 but use different outcome measures. They receive a smaller surface because the paper's own robustness analysis is more limited for these outcomes (Table 5 shows only bivariate vs. full-controls).

### Per group: ~25 specs each

| Category | Count |
|----------|-------|
| Baseline | 1 |
| Control sets (none, full_with_lfunc_ativ) | 2 |
| LOO blocks | 6 |
| Sample trimming (y 1/99, y 5/95) | 2 |
| Functional form (asinh for G2 only) | 0-1 |
| Inference (HC2, HC3) | 2 |
| **Total G2** | **~14** |
| **Total G3** | **~13** |

### G2 note on mandatory controls
- `lrec_fisc` (log audited resources) is a **mandatory control** for G2 because audit scale mechanically determines how many violations can be found. Dropping it changes the estimand concept.

### G3 note on mandatory controls
- Both `lrec_fisc` and `lfunc_ativ` are mandatory controls for G3, as the outcome is a ratio of violations to audited items.

---

## 5. What Is Excluded from the Core Surface (and Why)

### Table 8 (Mismanagement placebo)
- **pmismanagement** regressions are a **falsification test**: mismanagement (as opposed to corruption) should not respond to reelection incentives because it reflects incompetence rather than strategic behavior.
- Classification: `diag/cross_sectional_ols/placebo/mismanagement` (diagnostic, not core estimate).

### Table 9 (Matching grants / convenios)
- Uses a different dataset (`conveniosdata_aer.dta`, panel structure), different outcomes (`dconvenios`, `lconvenios_pc`, `msh_liberado`), and a DiD-like identification strategy with year interactions.
- This is a **mechanism result** (do first-term mayors extract more matching grants?), not a direct corruption measurement.
- Classification: `explore/outcome/matching_grants` (exploration; different outcome concept and estimand).

### Table 10 (Heterogeneous effects)
- Interaction terms (first x comarca, first x media, first x voter_registration, first x political_competition) test whether the corruption-reducing effect of reelection incentives is moderated by institutional features.
- Classification: `explore/heterogeneity/*` (exploration; the baseline estimand is an average effect).

### Table 11 Cols 1-3 (Additional robustness for pcorrupt)
- These test specific manipulation concerns (audit timing, PT mayors, same-party governor) by adding interaction terms.
- While these are in principle RC, they test very specific threats rather than general robustness. They could be included as `rc/controls/add_interaction/*` but are excluded from the core surface to keep it focused on the standard robustness dimensions.

### Table 11 Cols 4-7 (lrecursos_fisc manipulation test)
- Tests whether first-term mayors can manipulate the amount of resources audited (lrecursos_fisc = log(valor_fiscalizado)).
- Classification: `diag/cross_sectional_ols/placebo/audit_resources` (diagnostic/manipulation test).

### Alternative estimators not in the paper
- IPW, AIPW, matching estimators are available in the design file (`cross_sectional_ols.md`) but are excluded because (a) the paper does not use them as main estimates (matching is in Col 7 of Table 4 but was not replicated due to software constraints), and (b) the treatment is binary but the identification argument relies on selection-on-observables, making these estimators interesting but not the paper's revealed approach.

---

## 6. Constraints and Guardrails

### Control-count envelope
- **G1**: [0, 41] controls (Table 4 reveals the full 0-to-40 progression; Tables 6-11 add lfunc_ativ for 41).
- **G2**: [0, 41] controls (Table 5 shows bivariate and full).
- **G3**: [0, 42] controls (Table 5 shows bivariate and full with lfunc_ativ + lrec_fisc).

### Linked adjustment
- Not applicable: the baseline estimator is a single OLS regression (not bundled).

### FE constraints
- State FE (`uf`, 26 levels) is the paper's main absorption. Adding finer FE (municipality-level) is not feasible in a cross-section (N=476 municipalities, each observed once).
- Dropping state FE is a legitimate RC (tested indirectly in Table 4, Cols 1-5).

### Sample size constraint
- N=476 is small. The specification surface should avoid specs that dramatically reduce the sample unless the paper itself does so (e.g., running_nonmissing N=328 is paper-revealed).

---

## 7. Implementation Notes for the Spec-Search Agent

1. **State FE**: The baseline uses `areg ... abs(uf)` which is equivalent to `pf.feols("... | uf")`. All G1 core specs except `rc/fe/drop/uf` should include state FE.

2. **Robust SEs**: The baseline uses HC1 (`vcov="hetero"` in pyfixest). Inference variants change only the SE computation, not the point estimate.

3. **Control blocks**: The 6 control blocks (prefchar2_continuous, prefchar2_party, munichar2, fiscal, political, sorteio) should be treated as atomic units for block-level LOO and exhaustive block combinations.

4. **Party dummies**: The 16 party dummies (party_d1, party_d3-party_d18; note party_d2 is absent from data) should always be included or excluded as a block.

5. **Sorteio dummies**: The 10 lottery-round dummies (sorteio1-sorteio10) should be included or excluded as a block.

6. **RDD polynomial specs**: These require restricting the sample to `running != NaN` (N=328) and adding the generated `running` variable (and its powers). The `running` variable must be generated from `winmargin2000` and `winmargin2000_inclost` as in the do-file.

7. **Tobit**: The Tobit left-censored at 0 spec requires a separate MLE implementation (not pyfixest). Use scipy/statsmodels.
