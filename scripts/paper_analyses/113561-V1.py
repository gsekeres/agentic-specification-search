"""
Specification Search Script for Fong & Luttmer (2009)
"What Determines Giving to Hurricane Katrina Victims? Experimental Evidence on Racial Group Loyalty"
American Economic Journal: Applied Economics, 1(2), 64-87.

Paper ID: 113561-V1

Executes the approved SPECIFICATION_SURFACE.json:
  - G1: giving ~ picshowblack (Table 4 Panel 1, white respondents)
  - G2: hypgiv_tc500 ~ picshowblack (Table 4 Panel 2, white respondents)
  - G3: subjsupchar ~ picshowblack (Table 4 Panel 3, white respondents)
  - G4: subjsupgov ~ picshowblack (Table 4 Panel 4, white respondents)

Outputs:
  - specification_results.csv
  - inference_results.csv
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
PAPER_ID = "113561-V1"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PACKAGE_DIR = os.path.join(REPO_ROOT, "data", "downloads", "extracted", PAPER_ID)
SEED = 113561

# =============================================================================
# Load and prepare data (same as replication script)
# =============================================================================
df = pd.read_stata(os.path.join(PACKAGE_DIR, "katrina.dta"), convert_categoricals=False)

# Sample selection
df = df[df['soundcheck'] == 3].copy()
df = df[df['giving'].notna()].copy()

# --- Data cleaning (matching replication script exactly) ---
df['var_racesalient'] = (df['surveyvariant'] == 2).astype(int)
df['var_fullstakes'] = (df['surveyvariant'] == 3).astype(int)

df['hypgiv_tc500'] = df['hypothgiving'].copy()
df.loc[df['hypgiv_tc500'] > 500, 'hypgiv_tc500'] = 500

df['white'] = (df['ppethm'] == 1).astype(int)
df['black'] = (df['ppethm'] == 2).astype(int)
df['other'] = (1 - df['black'] - df['white']).astype(int)

df['age'] = df['ppage']
df['age2'] = df['ppage'] ** 2
df['dualin'] = df['ppdualin']

df['edudo'] = (df['ppeducat'] == 1).astype(int)
df['eduhs'] = (df['ppeducat'] == 2).astype(int)
df['edusc'] = (df['ppeducat'] == 3).astype(int)
df['educp'] = (df['ppeducat'] == 4).astype(int)

df['lnhhsz'] = np.log(df['pphhsize'])

inc_map = {
    1: np.log(2500), 2: np.log((5000+7499)/2), 3: np.log((7500+9999)/2),
    4: np.log((10000+12499)/2), 5: np.log((12500+14999)/2), 6: np.log((15000+19999)/2),
    7: np.log((20000+24999)/2), 8: np.log((25000+29999)/2), 9: np.log((30000+34999)/2),
    10: np.log((35000+39999)/2), 11: np.log((40000+49999)/2), 12: np.log((50000+59999)/2),
    13: np.log((60000+74999)/2), 14: np.log((75000+84999)/2), 15: np.log((85000+99999)/2),
    16: np.log((100000+124999)/2), 17: np.log((125000+149999)/2), 18: np.log((150000+174999)/2),
    19: np.log(350000),
}
df['lnhhinc'] = df['ppincimp'].map(inc_map)

df['married'] = (df['ppmarit'] == 1).astype(int)
df['male'] = (df['ppgender'] == 1).astype(int)
df['singlemale'] = (df['male'] & ~df['married'].astype(bool)).astype(int)

df['nrtheast'] = (df['ppreg4'] == 1).astype(int)
df['midwest'] = (df['ppreg4'] == 2).astype(int)
df['south'] = (df['ppreg4'] == 3).astype(int)
df['west'] = (df['ppreg4'] == 4).astype(int)

df['work'] = (df['ppwork'] <= 4).astype(int)
df['retired'] = (df['ppwork'] == 6).astype(int)
df['disabled'] = (df['ppwork'] == 7).astype(int)
df['unempl'] = (df['ppwork'] == 5).astype(int)
df['notwork'] = ((df['ppwork'] == 8) | (df['ppwork'] == 9)).astype(int)

df['dcharkatrina'] = ((df['charkatrina'] > 0) & (df['charkatrina'].notna())).astype(int)
df['lcharkatrina'] = np.log(df['charkatrina'].replace(0, np.nan))
df['lcharkatrina'] = df['lcharkatrina'].fillna(0)

df['dchartot2005'] = ((df['chartot2005'] > 0) & (df['chartot2005'].notna())).astype(int)
df['lchartot2005'] = np.log(df['chartot2005'].replace(0, np.nan))
df['lchartot2005'] = df['lchartot2005'].fillna(0)

for val, rank_label in [(2, 'help'), (6, 'mony')]:
    col = f'lifepriorities_{rank_label}'
    mask = df['lifepriorities5'].notna()
    df[col] = np.nan
    for i, pri_col in enumerate(['lifepriorities1', 'lifepriorities2', 'lifepriorities3',
                                   'lifepriorities4', 'lifepriorities5'], 1):
        weight = 6 - i
        base_contrib = (df[pri_col] == val).astype(float) * weight
        df.loc[mask, col] = df.loc[mask, col].fillna(0) + base_contrib[mask]
    df.loc[mask, col] = df.loc[mask, col] + 1

df['nraudworthy'] = df['aud_helpoth'] - df['aud_crime'] + df['aud_contrib'] + df['aud_prephur']

# Topcoded giving for RC
df['giving_tc50'] = df['giving'].clip(upper=50)
df['hypgiv_tc200'] = df['hypothgiving'].clip(upper=200)

print(f"Data loaded: {len(df)} observations total")
print(f"White respondents: {(df['white']==1).sum()}")

# =============================================================================
# Control block definitions
# =============================================================================
PIC_CONTROLS = ['picraceb', 'picobscur']

MANIP_AUDIO = ['aud_econdis', 'nraudworthy', 'aud_republ', 'aud_govtben',
               'aud_church', 'aud_loot', 'cityslidell']

SURVEY_VARIANT = ['var_fullstakes', 'var_racesalient']

DEMO_AGE = ['age', 'age2']
DEMO_EDUCATION = ['edudo', 'edusc', 'educp']
DEMO_INCOME = ['lnhhinc', 'dualin']
DEMO_MARITAL = ['married', 'male', 'singlemale']
DEMO_GEOGRAPHY = ['south']
DEMO_LABOR = ['work', 'disabled', 'retired']

CHARITABLE = ['dcharkatrina', 'lcharkatrina', 'dchartot2005', 'lchartot2005']

EXTRA_CONTROLS = ['hfh_effective', 'lifepriorities_help', 'lifepriorities_mony']

# Baseline controls for white sample (Table 4 / Table 5 baseline)
BASELINE_CONTROLS = (PIC_CONTROLS + MANIP_AUDIO + SURVEY_VARIANT +
                     DEMO_AGE + DEMO_EDUCATION + DEMO_INCOME + DEMO_MARITAL +
                     DEMO_GEOGRAPHY + DEMO_LABOR + CHARITABLE)

EXTENDED_CONTROLS = BASELINE_CONTROLS + EXTRA_CONTROLS

MINIMAL_CONTROLS = PIC_CONTROLS + MANIP_AUDIO + SURVEY_VARIANT

# LOO block map: block_name -> variables to drop from baseline
LOO_BLOCKS = {
    'age_block': DEMO_AGE,
    'education_block': DEMO_EDUCATION,
    'income_block': DEMO_INCOME,
    'labor_block': DEMO_LABOR,
    'charitable_block': CHARITABLE,
    'geography_block': DEMO_GEOGRAPHY,
    'marital_block': DEMO_MARITAL,
}

# =============================================================================
# Helper: run a single OLS/WLS specification via pyfixest
# =============================================================================
results = []
inference_results = []
run_counter = 0
infer_counter = 0


def run_spec(outcome_var, treatment_var, controls, data, baseline_group_id,
             spec_id, spec_tree_path, weight_var=None, sample_desc="white respondents",
             controls_desc="", fixed_effects="none", cluster_var="",
             coef_vector_extra=None):
    """Run one OLS/WLS specification and record results."""
    global run_counter
    run_counter += 1
    spec_run_id = f"{PAPER_ID}__run_{run_counter:04d}"

    try:
        # Build formula
        rhs_vars = [treatment_var] + controls
        rhs_str = " + ".join(rhs_vars)
        formula = f"{outcome_var} ~ {rhs_str}"

        # Prepare data: drop NaN in all relevant columns
        all_vars = [outcome_var] + rhs_vars
        if weight_var:
            all_vars.append(weight_var)
        dfreg = data[all_vars].dropna().copy()

        if len(dfreg) < 10:
            raise ValueError(f"Too few observations: {len(dfreg)}")

        # Run model
        if weight_var:
            model = pf.feols(formula, data=dfreg, vcov="hetero", weights=weight_var)
        else:
            model = pf.feols(formula, data=dfreg, vcov="hetero")

        # Extract results
        coef = float(model.coef()[treatment_var])
        se = float(model.se()[treatment_var])
        pval = float(model.pvalue()[treatment_var])
        ci = model.confint()
        ci_lower = float(ci.loc[treatment_var].iloc[0])
        ci_upper = float(ci.loc[treatment_var].iloc[1])
        n_obs = int(model._N)
        r2 = float(model._r2)

        # Coefficient vector
        coef_dict = {k: round(float(v), 8) for k, v in model.coef().items()}
        if coef_vector_extra:
            coef_dict.update(coef_vector_extra)

        row = {
            'paper_id': PAPER_ID,
            'spec_run_id': spec_run_id,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'baseline_group_id': baseline_group_id,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': round(coef, 8),
            'std_error': round(se, 8),
            'p_value': round(pval, 8),
            'ci_lower': round(ci_lower, 8),
            'ci_upper': round(ci_upper, 8),
            'n_obs': n_obs,
            'r_squared': round(r2, 8),
            'coefficient_vector_json': json.dumps(coef_dict),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'run_success': 1,
            'run_error': '',
        }
        results.append(row)
        print(f"  [{spec_run_id}] {spec_id}: {outcome_var} ~ {treatment_var} = {coef:.6f} (SE={se:.6f}, p={pval:.4f}), N={n_obs}")
        return row

    except Exception as e:
        row = {
            'paper_id': PAPER_ID,
            'spec_run_id': spec_run_id,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'baseline_group_id': baseline_group_id,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': np.nan,
            'std_error': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_obs': np.nan,
            'r_squared': np.nan,
            'coefficient_vector_json': json.dumps(coef_vector_extra or {}),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'run_success': 0,
            'run_error': str(e)[:200],
        }
        results.append(row)
        print(f"  [{spec_run_id}] {spec_id}: FAILED - {e}")
        return row


def run_inference_variant(base_row, data, infer_spec_id, infer_tree_path,
                          vcov_type="HC3", weight_var=None):
    """Re-estimate under alternative inference and record to inference_results."""
    global infer_counter
    infer_counter += 1
    inference_run_id = f"{PAPER_ID}__infer_{infer_counter:04d}"

    try:
        outcome_var = base_row['outcome_var']
        treatment_var = base_row['treatment_var']
        baseline_group_id = base_row['baseline_group_id']

        # Reconstruct control list from coefficient_vector_json
        coef_dict = json.loads(base_row['coefficient_vector_json'])
        rhs_vars = [k for k in coef_dict.keys() if k != 'Intercept']
        rhs_str = " + ".join(rhs_vars)
        formula = f"{outcome_var} ~ {rhs_str}"

        all_vars = [outcome_var] + rhs_vars
        if weight_var:
            all_vars.append(weight_var)
        dfreg = data[all_vars].dropna().copy()

        if weight_var:
            model = pf.feols(formula, data=dfreg, vcov=vcov_type, weights=weight_var)
        else:
            model = pf.feols(formula, data=dfreg, vcov=vcov_type)

        coef = float(model.coef()[treatment_var])
        se = float(model.se()[treatment_var])
        pval = float(model.pvalue()[treatment_var])
        ci = model.confint()
        ci_lower = float(ci.loc[treatment_var].iloc[0])
        ci_upper = float(ci.loc[treatment_var].iloc[1])
        n_obs = int(model._N)
        r2 = float(model._r2)

        coef_v = {k: round(float(v), 8) for k, v in model.coef().items()}
        coef_v['inference'] = {'vcov_type': vcov_type}

        irow = {
            'paper_id': PAPER_ID,
            'inference_run_id': inference_run_id,
            'spec_run_id': base_row['spec_run_id'],
            'spec_id': infer_spec_id,
            'spec_tree_path': infer_tree_path,
            'baseline_group_id': baseline_group_id,
            'coefficient': round(coef, 8),
            'std_error': round(se, 8),
            'p_value': round(pval, 8),
            'ci_lower': round(ci_lower, 8),
            'ci_upper': round(ci_upper, 8),
            'n_obs': n_obs,
            'r_squared': round(r2, 8),
            'coefficient_vector_json': json.dumps(coef_v),
            'run_success': 1,
            'run_error': '',
        }
        inference_results.append(irow)
        print(f"    [INFER {inference_run_id}] {infer_spec_id}: SE={se:.6f}, p={pval:.4f}")
        return irow

    except Exception as e:
        irow = {
            'paper_id': PAPER_ID,
            'inference_run_id': inference_run_id,
            'spec_run_id': base_row['spec_run_id'],
            'spec_id': infer_spec_id,
            'spec_tree_path': infer_tree_path,
            'baseline_group_id': base_row['baseline_group_id'],
            'coefficient': np.nan,
            'std_error': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_obs': np.nan,
            'r_squared': np.nan,
            'coefficient_vector_json': json.dumps({'error': str(e)[:200]}),
            'run_success': 0,
            'run_error': str(e)[:200],
        }
        inference_results.append(irow)
        print(f"    [INFER {inference_run_id}] {infer_spec_id}: FAILED - {e}")
        return irow


# =============================================================================
# White respondent subsample
# =============================================================================
df_white = df[df['white'] == 1].copy()
print(f"\nWhite respondent sample: N={len(df_white)}")

# =============================================================================
# Define outcome groups
# =============================================================================
OUTCOME_GROUPS = {
    'G1': {'outcome': 'giving', 'label': 'actual giving'},
    'G2': {'outcome': 'hypgiv_tc500', 'label': 'hypothetical giving'},
    'G3': {'outcome': 'subjsupchar', 'label': 'charity support'},
    'G4': {'outcome': 'subjsupgov', 'label': 'government support'},
}

TREATMENT = 'picshowblack'

# =============================================================================
# Execute specification surface
# =============================================================================
for gid, ginfo in OUTCOME_GROUPS.items():
    outcome = ginfo['outcome']
    label = ginfo['label']
    print(f"\n{'='*70}")
    print(f"Baseline Group {gid}: {outcome} ({label})")
    print(f"{'='*70}")

    # -------------------------------------------------------------------------
    # 1. BASELINE: Table 4, white respondents, weighted
    # -------------------------------------------------------------------------
    print(f"\n--- Baseline ---")
    baseline_row = run_spec(
        outcome_var=outcome,
        treatment_var=TREATMENT,
        controls=BASELINE_CONTROLS,
        data=df_white,
        baseline_group_id=gid,
        spec_id="baseline",
        spec_tree_path="designs/randomized_experiment.md#baseline",
        weight_var="tweight",
        sample_desc="white respondents",
        controls_desc="nraud + demographics + charitable giving (weighted)",
    )

    # -------------------------------------------------------------------------
    # 2. DESIGN VARIANT: difference in means (no controls except pic)
    # -------------------------------------------------------------------------
    print(f"\n--- Design: Difference in Means ---")
    run_spec(
        outcome_var=outcome,
        treatment_var=TREATMENT,
        controls=PIC_CONTROLS,
        data=df_white,
        baseline_group_id=gid,
        spec_id="design/randomized_experiment/estimator/diff_in_means",
        spec_tree_path="designs/randomized_experiment.md#a-itt-implementations-estimand-preserving-under-random-assignment",
        weight_var="tweight",
        sample_desc="white respondents",
        controls_desc="pic controls only (diff-in-means, weighted)",
    )

    # -------------------------------------------------------------------------
    # 3. RC: CONTROL SETS
    # -------------------------------------------------------------------------
    print(f"\n--- RC: Control Sets ---")

    # sets/none: bivariate (treatment + pic controls, unweighted)
    run_spec(
        outcome_var=outcome,
        treatment_var=TREATMENT,
        controls=PIC_CONTROLS,
        data=df_white,
        baseline_group_id=gid,
        spec_id="rc/controls/sets/none",
        spec_tree_path="modules/robustness/controls.md#a-standard-control-sets",
        sample_desc="white respondents",
        controls_desc="pic controls only (unweighted)",
    )

    # sets/minimal: pic + manipulation + survey variant (matches Table 5 s5 "no demographics")
    run_spec(
        outcome_var=outcome,
        treatment_var=TREATMENT,
        controls=MINIMAL_CONTROLS,
        data=df_white,
        baseline_group_id=gid,
        spec_id="rc/controls/sets/minimal",
        spec_tree_path="modules/robustness/controls.md#a-standard-control-sets",
        weight_var="tweight",
        sample_desc="white respondents",
        controls_desc="pic + manipulation + survey variant (no demographics, weighted)",
    )

    # sets/extended: baseline + extra controls (matches Table 5 s6)
    run_spec(
        outcome_var=outcome,
        treatment_var=TREATMENT,
        controls=EXTENDED_CONTROLS,
        data=df_white,
        baseline_group_id=gid,
        spec_id="rc/controls/sets/extended",
        spec_tree_path="modules/robustness/controls.md#a-standard-control-sets",
        weight_var="tweight",
        sample_desc="white respondents",
        controls_desc="nraud + demographics + charitable + extra controls (weighted)",
    )

    # -------------------------------------------------------------------------
    # 4. RC: LOO BLOCKS
    # -------------------------------------------------------------------------
    print(f"\n--- RC: LOO Blocks ---")
    for block_name, block_vars in LOO_BLOCKS.items():
        loo_controls = [c for c in BASELINE_CONTROLS if c not in block_vars]
        run_spec(
            outcome_var=outcome,
            treatment_var=TREATMENT,
            controls=loo_controls,
            data=df_white,
            baseline_group_id=gid,
            spec_id=f"rc/controls/loo/drop_{block_name}",
            spec_tree_path="modules/robustness/controls.md#b-leave-one-out-controls-loo",
            weight_var="tweight",
            sample_desc="white respondents",
            controls_desc=f"baseline minus {block_name} (weighted)",
            coef_vector_extra={"controls": {"family": "loo", "dropped": block_vars,
                               "block_name": block_name}},
        )

    # -------------------------------------------------------------------------
    # 5. RC: CONTROL PROGRESSION
    # -------------------------------------------------------------------------
    print(f"\n--- RC: Control Progression ---")

    # Bivariate (treatment only + pic)
    run_spec(
        outcome_var=outcome,
        treatment_var=TREATMENT,
        controls=PIC_CONTROLS,
        data=df_white,
        baseline_group_id=gid,
        spec_id="rc/controls/progression/bivariate",
        spec_tree_path="modules/robustness/controls.md#d-control-progression-build-up",
        weight_var="tweight",
        sample_desc="white respondents",
        controls_desc="pic controls only (progression start, weighted)",
    )

    # Manipulation only
    run_spec(
        outcome_var=outcome,
        treatment_var=TREATMENT,
        controls=PIC_CONTROLS + MANIP_AUDIO + SURVEY_VARIANT,
        data=df_white,
        baseline_group_id=gid,
        spec_id="rc/controls/progression/manipulation_only",
        spec_tree_path="modules/robustness/controls.md#d-control-progression-build-up",
        weight_var="tweight",
        sample_desc="white respondents",
        controls_desc="pic + manipulation + survey variant (weighted)",
    )

    # Manipulation + demographics (no charitable giving)
    run_spec(
        outcome_var=outcome,
        treatment_var=TREATMENT,
        controls=(PIC_CONTROLS + MANIP_AUDIO + SURVEY_VARIANT +
                  DEMO_AGE + DEMO_EDUCATION + DEMO_INCOME + DEMO_MARITAL +
                  DEMO_GEOGRAPHY + DEMO_LABOR),
        data=df_white,
        baseline_group_id=gid,
        spec_id="rc/controls/progression/manipulation_plus_demographics",
        spec_tree_path="modules/robustness/controls.md#d-control-progression-build-up",
        weight_var="tweight",
        sample_desc="white respondents",
        controls_desc="pic + manipulation + survey variant + demographics (no charitable, weighted)",
    )

    # Full = extended
    run_spec(
        outcome_var=outcome,
        treatment_var=TREATMENT,
        controls=EXTENDED_CONTROLS,
        data=df_white,
        baseline_group_id=gid,
        spec_id="rc/controls/progression/full",
        spec_tree_path="modules/robustness/controls.md#d-control-progression-build-up",
        weight_var="tweight",
        sample_desc="white respondents",
        controls_desc="all controls including extra (progression end, weighted)",
    )

    # -------------------------------------------------------------------------
    # 6. RC: SAMPLE RESTRICTIONS
    # -------------------------------------------------------------------------
    print(f"\n--- RC: Sample Restrictions ---")

    # Main survey variant only (surveyvariant==1)
    df_main = df_white[df_white['surveyvariant'] == 1].copy()
    run_spec(
        outcome_var=outcome,
        treatment_var=TREATMENT,
        controls=BASELINE_CONTROLS,
        data=df_main,
        baseline_group_id=gid,
        spec_id="rc/sample/restriction/main_survey_only",
        spec_tree_path="modules/robustness/sample.md#d-data-quality-and-eligibility-filters",
        weight_var="mweight",
        sample_desc="white, main survey variant only",
        controls_desc="nraud + demographics + charitable giving (mweight)",
        coef_vector_extra={"sample": {"rule": "surveyvariant==1", "weight": "mweight"}},
    )

    # Slidell only
    df_slidell = df_white[df_white['cityslidell'] == 1].copy()
    run_spec(
        outcome_var=outcome,
        treatment_var=TREATMENT,
        controls=BASELINE_CONTROLS,
        data=df_slidell,
        baseline_group_id=gid,
        spec_id="rc/sample/restriction/slidell_only",
        spec_tree_path="modules/robustness/sample.md#d-data-quality-and-eligibility-filters",
        weight_var="tweight",
        sample_desc="white, Slidell only",
        controls_desc="nraud + demographics + charitable giving (weighted)",
        coef_vector_extra={"sample": {"rule": "cityslidell==1"}},
    )

    # Biloxi only
    df_biloxi = df_white[df_white['cityslidell'] == 0].copy()
    run_spec(
        outcome_var=outcome,
        treatment_var=TREATMENT,
        controls=BASELINE_CONTROLS,
        data=df_biloxi,
        baseline_group_id=gid,
        spec_id="rc/sample/restriction/biloxi_only",
        spec_tree_path="modules/robustness/sample.md#d-data-quality-and-eligibility-filters",
        weight_var="tweight",
        sample_desc="white, Biloxi only",
        controls_desc="nraud + demographics + charitable giving (weighted)",
        coef_vector_extra={"sample": {"rule": "cityslidell==0"}},
    )

    # Race-shown only (picobscur==0)
    df_raceshown = df_white[df_white['picobscur'] == 0].copy()
    run_spec(
        outcome_var=outcome,
        treatment_var=TREATMENT,
        controls=BASELINE_CONTROLS,
        data=df_raceshown,
        baseline_group_id=gid,
        spec_id="rc/sample/restriction/race_shown_only",
        spec_tree_path="modules/robustness/sample.md#d-data-quality-and-eligibility-filters",
        weight_var="tweight",
        sample_desc="white, race-shown treatment only",
        controls_desc="nraud + demographics + charitable giving (weighted)",
        coef_vector_extra={"sample": {"rule": "picobscur==0"}},
    )

    # -------------------------------------------------------------------------
    # 7. RC: WEIGHTS
    # -------------------------------------------------------------------------
    print(f"\n--- RC: Weights ---")

    # Unweighted
    run_spec(
        outcome_var=outcome,
        treatment_var=TREATMENT,
        controls=BASELINE_CONTROLS,
        data=df_white,
        baseline_group_id=gid,
        spec_id="rc/weights/main/unweighted",
        spec_tree_path="modules/robustness/weights.md#a-main-weight-choices",
        sample_desc="white respondents",
        controls_desc="nraud + demographics + charitable giving (unweighted)",
        coef_vector_extra={"weights": {"spec_id": "rc/weights/main/unweighted",
                           "weight_var": "none"}},
    )

    # Paper weights (same as baseline, but explicitly labeled)
    run_spec(
        outcome_var=outcome,
        treatment_var=TREATMENT,
        controls=BASELINE_CONTROLS,
        data=df_white,
        baseline_group_id=gid,
        spec_id="rc/weights/main/paper_weights",
        spec_tree_path="modules/robustness/weights.md#a-main-weight-choices",
        weight_var="tweight",
        sample_desc="white respondents",
        controls_desc="nraud + demographics + charitable giving (tweight)",
        coef_vector_extra={"weights": {"spec_id": "rc/weights/main/paper_weights",
                           "weight_var": "tweight"}},
    )

    # -------------------------------------------------------------------------
    # 8. RC: FUNCTIONAL FORM (G1, G2 only)
    # -------------------------------------------------------------------------
    if gid == 'G1':
        print(f"\n--- RC: Functional Form (topcode giving at 50) ---")
        run_spec(
            outcome_var='giving_tc50',
            treatment_var=TREATMENT,
            controls=BASELINE_CONTROLS,
            data=df_white,
            baseline_group_id=gid,
            spec_id="rc/form/outcome/topcode_giving_at_50",
            spec_tree_path="modules/robustness/functional_form.md#a-outcome-transformations",
            weight_var="tweight",
            sample_desc="white respondents",
            controls_desc="nraud + demographics + charitable giving (weighted, giving topcoded at 50)",
            coef_vector_extra={"functional_form": {"outcome_transform": "topcode_50",
                               "interpretation": "Same concept, reduced influence of max gifts"}},
        )

    if gid == 'G2':
        print(f"\n--- RC: Functional Form (topcode hypgiv at 200) ---")
        run_spec(
            outcome_var='hypgiv_tc200',
            treatment_var=TREATMENT,
            controls=BASELINE_CONTROLS,
            data=df_white,
            baseline_group_id=gid,
            spec_id="rc/form/outcome/topcode_hypgiv_at_200",
            spec_tree_path="modules/robustness/functional_form.md#a-outcome-transformations",
            weight_var="tweight",
            sample_desc="white respondents",
            controls_desc="nraud + demographics + charitable giving (weighted, hypgiv topcoded at 200)",
            coef_vector_extra={"functional_form": {"outcome_transform": "topcode_200",
                               "interpretation": "Same concept, tighter topcode than baseline 500"}},
        )

    # -------------------------------------------------------------------------
    # 9. INFERENCE VARIANTS (G1 only: HC3)
    # -------------------------------------------------------------------------
    if gid == 'G1' and baseline_row and baseline_row['run_success'] == 1:
        print(f"\n--- Inference Variant: HC3 ---")
        run_inference_variant(
            base_row=baseline_row,
            data=df_white,
            infer_spec_id="infer/se/hc/hc3",
            infer_tree_path="modules/inference/standard_errors.md#hc3",
            vcov_type="HC3",
            weight_var="tweight",
        )


# =============================================================================
# Write outputs
# =============================================================================
print(f"\n{'='*70}")
print("Writing outputs")
print(f"{'='*70}")

# specification_results.csv
spec_cols = ['paper_id', 'spec_run_id', 'spec_id', 'spec_tree_path',
             'baseline_group_id', 'outcome_var', 'treatment_var',
             'coefficient', 'std_error', 'p_value', 'ci_lower', 'ci_upper',
             'n_obs', 'r_squared', 'coefficient_vector_json',
             'sample_desc', 'fixed_effects', 'controls_desc', 'cluster_var',
             'run_success', 'run_error']

results_df = pd.DataFrame(results)
results_df[spec_cols].to_csv(os.path.join(PACKAGE_DIR, 'specification_results.csv'), index=False)
print(f"Wrote {len(results_df)} rows to specification_results.csv")

# inference_results.csv
if inference_results:
    infer_cols = ['paper_id', 'inference_run_id', 'spec_run_id', 'spec_id',
                  'spec_tree_path', 'baseline_group_id',
                  'coefficient', 'std_error', 'p_value', 'ci_lower', 'ci_upper',
                  'n_obs', 'r_squared', 'coefficient_vector_json',
                  'run_success', 'run_error']
    infer_df = pd.DataFrame(inference_results)
    infer_df[infer_cols].to_csv(os.path.join(PACKAGE_DIR, 'inference_results.csv'), index=False)
    print(f"Wrote {len(infer_df)} rows to inference_results.csv")

# Summary stats
print(f"\n--- Summary ---")
print(f"Total specification rows: {len(results_df)}")
for gid in ['G1', 'G2', 'G3', 'G4']:
    n = (results_df['baseline_group_id'] == gid).sum()
    n_success = ((results_df['baseline_group_id'] == gid) & (results_df['run_success'] == 1)).sum()
    print(f"  {gid}: {n} specs ({n_success} successful)")
print(f"Total inference rows: {len(inference_results)}")
n_failed = (results_df['run_success'] == 0).sum()
print(f"Failed specs: {n_failed}")
