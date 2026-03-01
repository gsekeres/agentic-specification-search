# Specification Surface Review: 113500-V1

**Paper**: Babcock, Recalde, Vesterlund (2017), "Gender Differences in the Allocation of Low-Promotability Tasks: The Role of Backlash," AER P&P
**Design**: randomized_experiment (lab experiment)
**Date reviewed**: 2026-02-25

---

## Summary of Baseline Groups

| Group | Claim Object | Sample | Status |
|-------|-------------|--------|--------|
| G1 | Gender gap in solicitation response, no-penalty condition (Table 1, Col 1) | treatment=="Control" & invest_group>0 | Verified with corrections |
| G2 | Gender gap in solicitation response, backlash/penalty condition (Table 1, Col 2) | treatment=="Backlash" & invest_group>0 | Verified with corrections |
| G3 | Cross-treatment comparison via triple interaction (Table 1, Col 3) | invest_group>0 (both conditions) | Verified with corrections |

**Changes made**: Corrected treatment condition coding, variable type notes, and clustering bug documentation. See details below.

---

## A) Baseline Groups

- G1, G2, G3 are correctly separated:
  - G1 (no-penalty) and G2 (backlash) are separate treatment conditions with separate probit regressions.
  - G3 pools both conditions with a triple interaction. This targets a different estimand (difference-in-differences across treatments).
- The three groups correspond exactly to the three columns of Table 1 in the paper.
- No missing baseline groups. The penalty choice analysis (Section 6) and request allocation analysis (Table B2) are correctly excluded as different outcomes/populations.

### CRITICAL DATA CODING ISSUE

The surface originally used numeric treatment coding (`treatment==1` for no-penalty, `treatment==2` for backlash). **In the actual data (`final_dataset.dta`), `treatment` is a categorical variable with values "Control" and "Backlash", not numeric 1/2.** Similarly, `female` is categorical ("Female"/"Male"), and `student` is categorical ("Freshman"/"Sophomore"/"Junior"/"Senior").

**Corrections made to JSON**:
- G1 sample condition: `invest_group > 0 & treatment=="Control"` (not `treatment==1`)
- G2 sample condition: `invest_group > 0 & treatment=="Backlash"` (not `treatment==2`)
- Added data preparation notes: `female`, `backlash`, `femaleXsolicited`, `backlashXsolicited`, `femaleXbacklash`, `femaleXbacklashXsol` must all be constructed from the raw categorical variables before analysis.

### Clustering Bug in Original Code

The Stata do-file defines `local clust_var unique_subjectid` (line 227) but then uses `cluster(\`clus_var')` (without the 't') in the probit commands (lines 229, 233, 239, 243, 249, 253). This means the cluster macro is empty and Stata runs the probit **without clustering**. The surface specifies `cluster_var: session_id` which is the correct design-based choice (randomization at the session level), but the original published estimates may actually not be clustered. The `cgmwildboot` results in the paper appear to be the primary inference method, which does implement proper clustering. This discrepancy is documented in the JSON.

---

## B) Design Selection

- `design_code: randomized_experiment` is correct. This is a lab experiment with random assignment at the session level.
- Design audit blocks are populated with: randomization_unit (session), estimand (ITT), cluster_var (session_id).
- The probit estimator is the paper's choice; LPM and logit are included as design variants.

## C) RC Axes

- **LOO control drops** (9 specs per group): Drop each of the 9 controls individually. This is comprehensive.
- **Control sets**: No controls, demographics only, preferences only. Good structured groupings.
- **Functional form**: LPM and logit as alternatives to probit. The LPM variant is particularly useful because the interaction coefficient is directly interpretable (no inteff correction needed). Correct.
- **Period splits**: First half, second half, first period only. These test learning effects across the 10-round experiment. Appropriate.
- **Design variants**: Difference-in-means and with-covariates. Appropriate.

### Issue: `student` variable is categorical, not numeric
The `student` variable in the data has values "Freshman", "Sophomore", "Junior", "Senior" (categorical). The paper's do-file does not include `student` as a control -- it appears in the JSON surface but needs to be noted that it must be either converted to a numeric indicator or used as a factor. The original Stata code loads student as a categorical variable. Looking at the do-file, the control list is: `period risk_seeking1 social1 age non_caucasian student usborn business other`. In Stata, if `student` is a string/categorical, Stata would either fail or automatically create dummies. In Python (pyfixest), this needs explicit handling. Added note to JSON.

### Issue: `non_caucasian` vs `caucasian`
Both `caucasian` and `non_caucasian` exist in the data. The surface correctly uses `non_caucasian` (matching the do-file). Verified.

## D) Controls Multiverse Policy

- `controls_count_min: 0` (no controls, just treatment/gender/interaction terms). Correct.
- `controls_count_max: 9` (all 9 individual-level covariates). Correct.
- `linked_adjustment: false`. Correct -- no bundled estimator.
- The interaction terms (femaleXsolicited for G1/G2; full triple set for G3) are correctly documented as mandatory for the estimand.

## E) Inference Plan

- **Canonical**: Cluster at session_id. This is correct (session is the randomization unit).
- **Variants**: Subject-level clustering, HC1 robust, wild cluster bootstrap. All appropriate.
- **Small clusters**: The paper has only ~8-12 sessions per treatment condition. This is correctly flagged. Standard clustered SEs may perform poorly; wild cluster bootstrap (cgmwildboot in Stata) is the paper's preferred inference method.
- **Practical note**: The `wildboottest` Python package may not be available in the analysis environment. If not, fall back to standard clustered SEs at the session level with a note about small-cluster concerns.

## F) Budgets and Sampling

- 55 specs per group (3 groups x 55 = 165 nominal). This is adequate.
- Exhaustive control subset enumeration. Correct for 9 controls.
- Seed: 113500.

## G) Diagnostics Plan

- Balance of covariates across sessions/treatments. Appropriate.
- No other design diagnostics needed for a lab experiment with known random assignment.

---

## Key Constraints and Linkage Rules

1. **Interaction terms are mandatory**: femaleXsolicited (G1/G2) and the full triple-interaction set (G3) cannot be dropped. They define the estimand.
2. **Period control**: Subjects play 10 rounds. The `period` variable controls for time trends/learning. It is in the baseline but can be dropped as a robustness check.
3. **Probit marginal effects**: For the probit specification, the interaction effect requires inteff/inteff3 correction. The LPM variant avoids this issue.
4. **Data preparation**: Several variables must be constructed from raw data:
   - `backlash = (treatment == "Backlash")` (binary numeric)
   - `female_num = (female == "Female")` (binary numeric)
   - `femaleXsolicited = female_num * solicited`
   - `backlashXsolicited = backlash * solicited`
   - `femaleXbacklash = female_num * backlash`
   - `femaleXbacklashXsol = female_num * backlash * solicited`
   - `student_num` needs conversion from categorical to numeric indicator
   - `non_caucasian` is already numeric (float32)

---

## Changes Made to JSON

1. **G1**: Corrected `sample_condition` from `invest_group > 0 (green players only) & treatment==1` to `invest_group > 0 & treatment=="Control"`.
2. **G2**: Corrected `sample_condition` from `invest_group > 0 (green players) & treatment==2` to `invest_group > 0 & treatment=="Backlash"`.
3. **G3**: Corrected `sample_condition` to note treatment variable is categorical.
4. **G1 baseline_spec**: Corrected `sample` field from `invest_group > 0 & treatment==1` to `invest_group > 0 & treatment=="Control"`.
5. **G2 baseline_spec**: Corrected `sample` field from `invest_group > 0 & treatment==2` to `invest_group > 0 & treatment=="Backlash"`.
6. Added `data_preparation_notes` to all three groups documenting the categorical variable conversions needed.
7. Documented the clustering bug in the original code.

---

## Final Assessment

**APPROVED TO RUN** with the corrections above. The surface is well-structured with three distinct baseline groups mapping exactly to the paper's Table 1 columns. The RC axes are comprehensive (controls LOO, functional form, period splits, design variants). The main practical issues are:
1. Categorical variable conversion (treatment, female, student) -- documented in JSON.
2. Small number of clusters for session-level clustering -- wild cluster bootstrap is preferred but may require package availability.
3. Probit interaction effects require inteff correction; LPM is the cleaner specification for the analysis pipeline.
