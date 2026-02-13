# Specification Surface: 113561-V1

**Paper**: "What Determines Giving to Hurricane Katrina Victims? Experimental Evidence on Racial Group Loyalty"
**Authors**: Christina M. Fong and Erzo F.P. Luttmer (AEJ: Applied Economics, 2009)
**Design**: Randomized experiment (online survey experiment)
**Created**: 2026-02-13

---

## 1. Paper Overview

This paper uses an online survey experiment conducted through Knowledge Networks to study racial group loyalty in charitable giving. Respondents were randomly assigned to view pictures of Hurricane Katrina victims that varied in the portrayed race of the victims (black victims, white victims, or race-obscured pictures). The experiment also independently randomized audio descriptions along several dimensions (economic disadvantage, crime, willingness to help, etc.). The primary outcome is dictator-game giving (0-100 scale), with secondary outcomes including hypothetical giving, support for charity spending, and support for government spending.

### Key Design Features
- **Randomization**: Individual-level random assignment to picture treatment arms and audio manipulations
- **Treatment**: Three picture manipulation dummies (`picshowblack`, `picraceb`, `picobscur`) with the omitted category being "white victims, race clearly shown"
- **Focal coefficient**: Always on `picshowblack` (showing black victims vs showing white victims with race visible)
- **Estimation**: WLS with survey weights (`tweight`) and HC1 robust standard errors
- **Sample**: N=1343 (after excluding those who could not hear audio and 5 with missing giving data); ~915 white, ~247 black respondents

### Revealed Search Space (from manuscript Tables 3-6)

The paper reveals the following degrees of freedom:

| Axis | Variations Revealed | Table(s) |
|------|---------------------|----------|
| **Outcomes** | giving, hypgiv_tc500, subjsupchar, subjsupgov | Tables 3-5 |
| **Sample** | Full, white-only, black-only, main variant, Slidell, Biloxi, race-shown only | Tables 4, 5 |
| **Controls** | Full manip + demographics, nraud + demographics, no demographics, extra controls | Tables 3-5 |
| **Manipulation coding** | Separate worthiness dummies ($manip, 12 vars) vs aggregated nraudworthy ($nraud, 9 vars) | Tables 3 vs 4 |
| **Weights** | tweight (baseline), mweight (main variant only), unweighted (Table 3 cols 5-6, Table 6) | Tables 3, 5, 6 |
| **Alternative estimators** | OLS, censored normal regression, ordered probit | Table 5 |
| **Heterogeneity interactions** | By respondent race, ethnic closeness, social contact, perceived opportunities | Tables 3, 6 |

---

## 2. Baseline Groups

### Why Four Baseline Groups

The paper presents four outcome variables as parallel main results in Table 4, and devotes the entire Table 5 to robustness checks across all four outcomes for white respondents. This structure indicates that all four outcomes are treated as headline claims. The paper's interpretive focus is on **white respondents** (in-group bias), so the baseline groups target the white subsample.

The full-sample and black-respondent results are included as sample robustness checks within the white-respondent baseline groups (for the full sample) or excluded from core (for black respondents), following the paper's own emphasis.

### G1: Experimental Giving (White Respondents)

- **Outcome concept**: Experimental charitable giving (dictator game, 0-100 scale)
- **Treatment concept**: Pictures show black Katrina victims vs white victims
- **Estimand**: ITT effect on giving for white respondents
- **Baseline spec**: Table 4, Row 1, White column (reg_id=8): coef = -4.198, p = 0.370, N = 915
- **Secondary baseline**: Table 3, Col 2, Full sample (reg_id=2): coef = -2.301, p = 0.550, N = 1343

This is the paper's primary claim object. The experimental giving variable is the real-stakes dictator game outcome.

### G2: Hypothetical Giving (White Respondents)

- **Outcome concept**: Hypothetical charitable giving (topcoded at $500)
- **Baseline spec**: Table 4, Row 2, White column (reg_id=11): coef = -2.181, p = 0.591, N = 913

### G3: Subjective Support for Charity Spending (White Respondents)

- **Outcome concept**: Subjective support for charity spending (1-7 Likert scale)
- **Baseline spec**: Table 4, Row 3, White column (reg_id=14): coef = -0.221, p = 0.167, N = 907

### G4: Subjective Support for Government Spending (White Respondents)

- **Outcome concept**: Subjective support for government spending (1-7 Likert scale)
- **Baseline spec**: Table 4, Row 4, White column (reg_id=17): coef = -0.435, p = 0.026, N = 913
- **Note**: This is the only baseline result that is statistically significant at conventional levels for white respondents.

---

## 3. Core-Eligible Universe (per baseline group)

All four baseline groups share the same core universe structure, with minor differences in outcome-specific preprocessing. The specifications are organized by axis following one-axis-at-a-time variation from the baseline.

### A. Design Estimator Implementations

| Spec ID | Description |
|---------|-------------|
| `design/randomized_experiment/estimator/diff_in_means` | Simple difference in means by treatment arm (no covariates, no weights) |
| `design/randomized_experiment/estimator/with_covariates` | OLS with covariates (the paper's approach; for G1 this is the baseline) |

### B. Controls Robustness (`rc/controls/*`)

#### B.1 Control sets

| Spec ID | Description | Maps to |
|---------|-------------|---------|
| `rc/controls/sets/none` | Treatment dummies only (bivariate) | Pure experimental estimate |
| `rc/controls/sets/minimal` | Treatment + manipulation controls (nraud, 9 vars) | Table 5 row 5 logic |
| `rc/controls/sets/baseline` | Treatment + nraud + cntrldems (29 vars) | Table 4 baseline |
| `rc/controls/sets/extended` | Treatment + nraud + cntrldems + addcntrl1 (32 vars) | Table 5 row 6 |

#### B.2 Control progression (building up from bivariate to full)

| Spec ID | Description | N controls |
|---------|-------------|------------|
| `rc/controls/progression/bivariate` | Treatment dummies only | 0 |
| `rc/controls/progression/manipulation_only` | + nraud manipulation controls | 9 |
| `rc/controls/progression/manipulation_plus_demographics` | + core demographics (age, education, income, marriage, gender, geography, labor) | 23 |
| `rc/controls/progression/manipulation_plus_demographics_plus_charity` | + charitable giving history | 27 |
| `rc/controls/progression/full` | + extra controls (hfh_effective, lifepriorities) | 30 |

Note: On the white subsample, `black` and `other` dummies are collinear (all zeros) and automatically dropped.

#### B.3 Manipulation coding

| Spec ID | Description |
|---------|-------------|
| `rc/controls/manipulation_coding/nraudworthy` | Use aggregated nraudworthy (baseline for Tables 4-5) |
| `rc/controls/manipulation_coding/separate_worthiness` | Use separate worthiness dummies (Table 3 approach: aud_prephur, aud_crime, aud_helpoth, aud_contrib separate instead of nraudworthy) |

#### B.4 Leave-one-out (LOO) controls

Drop each non-mandatory control one at a time from the baseline set. On the white subsample, there are 18 droppable variables (14 demographics + 4 charitable giving):

`age/age2`, `edudo`, `edusc`, `educp`, `lnhhinc`, `dualin`, `married`, `male`, `singlemale`, `south`, `work`, `disabled`, `retired`, `dcharkatrina`, `lcharkatrina`, `dchartot2005`, `lchartot2005`

Note: `age` and `age2` should be dropped together as a pair. This yields 17 LOO specs.

### C. Sample Robustness (`rc/sample/*`)

| Spec ID | Description | Maps to |
|---------|-------------|---------|
| `rc/sample/subpopulation/full_sample` | Run on full sample (white + black + other) | Table 4 "All" column |
| `rc/sample/subpopulation/main_variant_only` | Main survey variant only (surveyvariant==1), use mweight | Table 5 row 2 |
| `rc/sample/subpopulation/slidell_only` | Slidell respondents only (cityslidell==1) | Table 5 row 3 |
| `rc/sample/subpopulation/biloxi_only` | Biloxi respondents only (cityslidell==0) | Table 5 row 4 |
| `rc/sample/subpopulation/race_shown_only` | Exclude race-obscured treatment (picobscur==0) | Table 5 row 8 |

Note: The Slidell/Biloxi split and race-shown restriction are within-population robustness for the white respondent target. The full-sample spec extends the target population but is included as RC because the paper presents full-sample results as a parallel baseline.

### D. Weights Robustness (`rc/weights/*`)

| Spec ID | Description | Maps to |
|---------|-------------|---------|
| `rc/weights/main/paper_weights` | tweight (baseline) | Tables 3-5 baseline |
| `rc/weights/main/unweighted` | Unweighted OLS | Table 3 cols 5-6 (race subsamples), Table 6 |

### E. Outcome Preprocessing (`rc/preprocess/*`) -- G1 and G2 only

| Spec ID | Description | Applicable to |
|---------|-------------|--------------|
| `rc/preprocess/outcome/winsor_1_99` | Winsorize outcome at 1st/99th percentiles | G1 (giving) |
| `rc/preprocess/outcome/topcode_giving_at_99` | Topcode giving at 99th percentile | G1 (giving) |
| `rc/preprocess/outcome/topcode_hypgiv_at_250` | Alternative topcode for hypothetical giving at $250 | G2 (hypgiv_tc500) |
| `rc/preprocess/outcome/no_topcode` | Use raw hypothgiving without topcoding | G2 (hypgiv_tc500) |

### F. Inference Variants (`infer/*`)

| Spec ID | Description |
|---------|-------------|
| `infer/se/hc/hc1` | HC1 robust SE (baseline) |
| `infer/se/hc/hc2` | HC2 robust SE |
| `infer/se/hc/hc3` | HC3 robust SE (small-sample leverage correction) |
| `infer/se/hc/classical` | Classical (homoskedastic) SE |

---

## 4. Constraints

### Control-Count Envelope

- **Minimum**: 11 controls (Table 5 row 5: nraud + race dummies, no demographics)
- **Maximum**: 32 controls (Table 5 row 6: nraud + cntrldems + addcntrl1; or Table 3: manip + cntrldems)
- **Baseline**: 29 controls (nraud + cntrldems, Table 4)

### Treatment Arms Are Fixed

The three treatment dummies (`picshowblack`, `picraceb`, `picobscur`) are always included and are not counted as "controls." They are part of the experimental design specification. The focal coefficient is always on `picshowblack`.

### No Bundled Estimator

This is a simple OLS/WLS regression of outcomes on treatment indicators and covariates. There is no bundled estimator (no IV, AIPW, DML, or synth). `linked_adjustment = false`.

### One-Axis-at-a-Time

Specifications vary one axis at a time from the baseline. No combinatorial cross-products of controls x sample x weights are generated unless the paper reveals them (it does not). The total spec count is the sum across axes, not a factorial.

---

## 5. Budget and Sampling

### Estimated Spec Counts per Baseline Group

| Axis | G1 (giving) | G2 (hypgiv) | G3 (subjchar) | G4 (subjgov) |
|------|-------------|-------------|---------------|---------------|
| Baseline | 1 | 1 | 1 | 1 |
| Design estimator (diff in means) | 1 | 1 | 1 | 1 |
| Control sets (none, minimal, extended) | 3 | 3 | 3 | 3 |
| Control progression | 5 | 3 | 3 | 3 |
| Manipulation coding | 1 | 1 | 1 | 1 |
| LOO controls | 17 | 17 | 17 | 17 |
| Sample variants | 5 | 5 | 5 | 5 |
| Weights (unweighted) | 1 | 1 | 1 | 1 |
| Outcome preprocessing | 2 | 2 | 0 | 0 |
| Inference variants | 3 | 3 | 3 | 3 |
| **Total** | **~39** | **~37** | **~35** | **~35** |

**Grand total across all baseline groups: ~146 specifications**

### Sampling

Full enumeration is feasible for all baseline groups. No random sampling is needed. The spec counts are well within typical budgets.

---

## 6. Excluded from Core (and Why)

### A. Table 3 Interaction Specifications (Columns 3-6)

Table 3 columns 3-6 introduce interactions between picture treatment and respondent characteristics (respondent race, subjective identification with racial group). These change the estimand from a simple ITT to a conditional/heterogeneous treatment effect. Classified as `explore/heterogeneity/*`.

### B. Table 6 Interaction Analyses

Table 6 examines treatment effect heterogeneity by three racial attitude variables (ethnic closeness, social contact with blacks, perceived economic opportunities for blacks). These are explicitly heterogeneity explorations. Furthermore, the `interactd` program in Table 6 runs **unweighted** OLS (no survey weights), which is a different estimator specification than the baseline. Classified as `explore/heterogeneity/*`.

### C. Censored Regression and Ordered Probit (Table 5, Row 7)

- **Censored normal regression** (for giving and hypothetical giving): Alternative estimator for bounded/censored outcomes. Difficult to replicate exactly in Python; SEs from Hessian rather than sandwich.
- **Ordered probit** (for subjective support outcomes): Alternative estimator for ordinal outcomes. Does not support pweights in Python statsmodels; Hessian was numerically unstable.

These are excluded from core for feasibility reasons and because they target a latent variable model (different parameter).

### D. Black Respondent Results

Table 4 black columns and Table 5C provide parallel results for black respondents (N~247). These target a different population. The small sample size limits statistical power. Could be added as separate baseline groups if desired, but excluded from the default core to keep the surface focused on the paper's interpretive emphasis (white respondents).

### E. Full-Sample Results as Separate Claims

Full-sample results (Table 4 "All" columns) pool white and black respondents. Rather than a separate baseline group, the full sample is included as a sample RC within each G1-G4 baseline group (`rc/sample/subpopulation/full_sample`).

---

## 7. Diagnostics Plan (Not Part of Core)

| Diagnostic | Scope | Description |
|-----------|-------|-------------|
| `diag/randomized_experiment/balance/covariates` | Baseline group | Balance check of baseline covariates across picshowblack treatment arms |

No attrition diagnostic is needed because attrition is minimal (only 5 observations with missing giving data out of 1348 who passed the audio check).

---

## 8. Implementation Notes

### Variable Construction

Several variables must be constructed from raw data before running specifications:
- `per_hfhdif = per_hfhblk - per_hfhwht` (not used in core, only in manipulation check)
- `hypgiv_tc500`: topcode hypothgiving at 500
- `nraudworthy = aud_helpoth - aud_crime + aud_contrib + aud_prephur`
- Race dummies: `white = ppethm==1`, `black = ppethm==2`, `other = 1 - black - white`
- Log income from categories: `lnhhinc` from `ppincimp` (midpoint method)
- Age: `age = ppage`, `age2 = ppage^2`
- Education dummies from `ppeducat`
- Various other recodes (see do-file section 1)

### Weight Handling

All baseline specifications use `[pw=tweight]` which in Stata implements WLS with HC1 robust SE. In Python, this corresponds to `statsmodels.WLS(weights=tweight)` with `cov_type='HC1'`.

For the unweighted RC, use standard OLS with HC1.

For the main-variant-only sample RC, use `mweight` instead of `tweight`.

### Collinearity

When running on white-only subsample, `black` and `other` are constant (all zeros) and must be dropped. Similarly, on the race-shown-only subsample (`picobscur==0`), `picobscur` is constant zero and `picraceb == picshowblack`, creating perfect collinearity -- automatically drop one.

### Focal Coefficient

The focal coefficient for all specifications is always on `picshowblack`. This is the treatment indicator for "pictures show black victims" and captures the racial bias effect.
