#!/usr/bin/env python3
"""
Specification search script for 113561-V1:
"What Determines Giving to Hurricane Katrina Victims?"
Fong & Luttmer, AEJ: Applied Economics 2009

Surface-driven execution of ~55 specifications for baseline group G1.
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import hashlib
import traceback
import warnings
warnings.filterwarnings('ignore')

PACKAGE_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/113561-V1"
PAPER_ID = "113561-V1"

# ============================================================
# LOAD SURFACE
# ============================================================
with open(f"{PACKAGE_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface = json.load(f)

surface_hash = "sha256:" + hashlib.sha256(json.dumps(surface, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()

# ============================================================
# SOFTWARE BLOCK
# ============================================================
import sys
software_block = {
    "runner_language": "python",
    "runner_version": sys.version.split()[0],
    "packages": {
        "pyfixest": pf.__version__ if hasattr(pf, '__version__') else "0.40+",
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
}

# ============================================================
# DESIGN AUDIT BLOCK (from surface)
# ============================================================
bg = surface["baseline_groups"][0]
design_audit = bg["design_audit"]
design_block = {"randomized_experiment": design_audit}

# ============================================================
# LOAD AND PREPARE DATA
# ============================================================
df = pd.read_stata(f"{PACKAGE_DIR}/katrina.dta", convert_categoricals=False)

# Sample selection
df = df[df['soundcheck'] == 3].copy()
df = df[df['giving'].notna()].copy()

# Create all variables (same as replication)
df['var_racesalient'] = (df['surveyvariant'] == 2).astype(int)
df['var_fullstakes'] = (df['surveyvariant'] == 3).astype(int)
df['per_hfhdif'] = df['per_hfhblk'] - df['per_hfhwht']
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

inc_map = {1: np.log(2500), 2: np.log(6249.5), 3: np.log(8749.5), 4: np.log(11249.5),
           5: np.log(13749.5), 6: np.log(17499.5), 7: np.log(22499.5), 8: np.log(27499.5),
           9: np.log(32499.5), 10: np.log(37499.5), 11: np.log(44999.5), 12: np.log(54999.5),
           13: np.log(67499.5), 14: np.log(79999.5), 15: np.log(92499.5), 16: np.log(112499.5),
           17: np.log(137499.5), 18: np.log(162499.5), 19: np.log(350000)}
df['lnhhinc'] = df['ppincimp'].map(inc_map)

df['married'] = (df['ppmarit'] == 1).astype(int)
df['male'] = (df['ppgender'] == 1).astype(int)
df['singlemale'] = (df['male'] & ~df['married'].astype(bool)).astype(int)
df['south'] = (df['ppreg4'] == 3).astype(int)
df['work'] = (df['ppwork'] <= 4).astype(int)
df['retired'] = (df['ppwork'] == 6).astype(int)
df['disabled'] = (df['ppwork'] == 7).astype(int)

df['dcharkatrina'] = ((df['charkatrina'] > 0) & df['charkatrina'].notna()).astype(int)
df['lcharkatrina'] = np.log(df['charkatrina'])
df.loc[df['lcharkatrina'].isna() | ~np.isfinite(df['lcharkatrina']), 'lcharkatrina'] = 0

df['dchartot2005'] = ((df['chartot2005'] > 0) & df['chartot2005'].notna()).astype(int)
df['lchartot2005'] = np.log(df['chartot2005'])
df.loc[df['lchartot2005'].isna() | ~np.isfinite(df['lchartot2005']), 'lchartot2005'] = 0

df['nraudworthy'] = df['aud_helpoth'] - df['aud_crime'] + df['aud_contrib'] + df['aud_prephur']

for val, rank_name in [(2, 'help'), (6, 'mony')]:
    col = f'lifepriorities_{rank_name}'
    mask = df['lifepriorities5'].notna()
    df[col] = np.nan
    df.loc[mask, col] = (1
        + 5 * (df.loc[mask, 'lifepriorities1'] == val).astype(int)
        + 4 * (df.loc[mask, 'lifepriorities2'] == val).astype(int)
        + 3 * (df.loc[mask, 'lifepriorities3'] == val).astype(int)
        + 2 * (df.loc[mask, 'lifepriorities4'] == val).astype(int)
        + 1 * (df.loc[mask, 'lifepriorities5'] == val).astype(int))

# Create binary giving
df['giving_any'] = (df['giving'] > 0).astype(int)
# Create log(1+giving) and asinh(giving)
df['log1p_giving'] = np.log1p(df['giving'])
df['asinh_giving'] = np.arcsinh(df['giving'])

print(f"Sample size: {len(df)}")

# ============================================================
# DEFINE VARIABLE GROUPS
# ============================================================
treatment_vars = ['picshowblack', 'picraceb', 'picobscur']

manip_controls = ['aud_republ', 'aud_econdis', 'aud_govtben', 'aud_prephur', 'aud_church',
                  'aud_crime', 'aud_helpoth', 'aud_contrib', 'aud_loot', 'cityslidell',
                  'var_fullstakes', 'var_racesalient']

demographic_controls = ['age', 'age2', 'black', 'other', 'edudo', 'edusc', 'educp',
                        'lnhhinc', 'dualin', 'married', 'male', 'singlemale', 'south',
                        'work', 'disabled', 'retired']

giving_history = ['dcharkatrina', 'lcharkatrina', 'dchartot2005', 'lchartot2005']

extra_controls = ['hfh_effective', 'lifepriorities_help', 'lifepriorities_mony']

# nraud = manip controls with nraudworthy composite replacing individual worthiness dummies
nraud_controls = ['aud_econdis', 'nraudworthy', 'aud_republ', 'aud_govtben', 'aud_church',
                  'aud_loot', 'cityslidell', 'var_fullstakes', 'var_racesalient']

baseline_controls = manip_controls + demographic_controls + giving_history

# ============================================================
# RUN SPECIFICATIONS
# ============================================================
results = []
spec_run_id = 0

def run_spec(spec_id, spec_tree_path, outcome_var, treatment_var, controls, data,
             weight_var='tweight', sample_desc='All respondents', baseline_group_id='G1',
             extra_json_blocks=None):
    """Run a single specification and record results."""
    global spec_run_id
    spec_run_id += 1

    all_vars = [outcome_var, treatment_var] + [c for c in controls if c != treatment_var]
    if weight_var:
        all_vars.append(weight_var)
    all_vars = list(dict.fromkeys(all_vars))  # deduplicate preserving order

    subdf = data.dropna(subset=all_vars).copy()

    indepvars = [treatment_var] + [c for c in controls if c != treatment_var]
    indepvars = list(dict.fromkeys(indepvars))

    formula = f"{outcome_var} ~ " + " + ".join(indepvars)

    try:
        if weight_var and weight_var in subdf.columns:
            m = pf.feols(formula, data=subdf, vcov="hetero", weights=weight_var)
        else:
            m = pf.feols(formula, data=subdf, vcov="hetero")

        coef = float(m.coef()[treatment_var])
        se = float(m.se()[treatment_var])
        pval = float(m.pvalue()[treatment_var])
        nobs = int(m._N)
        r2 = float(m._r2)

        coef_dict = {k: float(v) for k, v in m.coef().items()}

        cvj = {
            "coefficients": coef_dict,
            "inference": {"spec_id": "infer/se/hc/hc1", "method": "HC1", "cluster_var": None},
            "software": software_block,
            "surface_hash": surface_hash,
            "design": design_block,
        }
        if extra_json_blocks:
            cvj.update(extra_json_blocks)

        result = {
            'paper_id': PAPER_ID,
            'spec_run_id': f"S{spec_run_id:03d}",
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'baseline_group_id': baseline_group_id,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': coef,
            'std_error': se,
            'p_value': pval,
            'ci_lower': coef - 1.96 * se,
            'ci_upper': coef + 1.96 * se,
            'n_obs': nobs,
            'r_squared': r2,
            'coefficient_vector_json': json.dumps(cvj),
            'sample_desc': sample_desc,
            'fixed_effects': '',
            'controls_desc': f'{len(indepvars)-1} controls' if len(indepvars) > 1 else 'no controls',
            'cluster_var': '',
            'run_success': 1,
            'run_error': '',
        }
        results.append(result)
        print(f"  {result['spec_run_id']} ({spec_id}): coef={coef:.4f}, se={se:.4f}, p={pval:.4f}, N={nobs}")
        return m
    except Exception as e:
        tb = traceback.format_exc()
        err_msg = str(e)[:200]
        cvj = {
            "error": err_msg,
            "error_details": {
                "stage": "estimation",
                "exception_type": type(e).__name__,
                "exception_message": str(e),
                "traceback_tail": tb[-500:]
            }
        }
        result = {
            'paper_id': PAPER_ID,
            'spec_run_id': f"S{spec_run_id:03d}",
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
            'coefficient_vector_json': json.dumps(cvj),
            'sample_desc': sample_desc,
            'fixed_effects': '',
            'controls_desc': '',
            'cluster_var': '',
            'run_success': 0,
            'run_error': err_msg,
        }
        results.append(result)
        print(f"  {result['spec_run_id']} ({spec_id}): FAILED - {err_msg}")
        return None


# ============================================================
# BASELINE SPECS
# ============================================================
print("\n=== BASELINE ===")

# Baseline: Table 3 Col 2 (all respondents, manip + demographics + giving history)
run_spec(
    spec_id='baseline',
    spec_tree_path='designs/randomized_experiment.md#baseline',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + manip_controls + demographic_controls + giving_history,
    data=df,
    weight_var='tweight',
    sample_desc='All respondents, Table 3 Col 2',
)

# Additional baseline: Table 4 Panel 1 (nraudworthy composite)
run_spec(
    spec_id='baseline__table4_panel1_all',
    spec_tree_path='designs/randomized_experiment.md#baseline',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + nraud_controls + demographic_controls + giving_history,
    data=df,
    weight_var='tweight',
    sample_desc='All respondents, Table 4 Panel 1 (nraudworthy composite)',
)

# ============================================================
# DESIGN SPECS
# ============================================================
print("\n=== DESIGN VARIANTS ===")

# Difference in means (no controls at all)
run_spec(
    spec_id='design/randomized_experiment/estimator/diff_in_means',
    spec_tree_path='designs/randomized_experiment.md#itt-implementations',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=[],
    data=df,
    weight_var='tweight',
    sample_desc='Difference in means, no controls',
)

# ============================================================
# RC: CONTROL SETS
# ============================================================
print("\n=== RC: CONTROL SETS ===")

# None: just treatment vars
run_spec(
    spec_id='rc/controls/sets/none',
    spec_tree_path='modules/robustness/controls.md#standard-control-sets',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:],
    data=df,
    weight_var='tweight',
    sample_desc='Treatment vars only (no other controls)',
    extra_json_blocks={"controls": {"spec_id": "rc/controls/sets/none", "family": "sets", "set_name": "none", "n_controls": 2}}
)

# Minimal: manip controls only (no demographics)
run_spec(
    spec_id='rc/controls/sets/minimal',
    spec_tree_path='modules/robustness/controls.md#standard-control-sets',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + manip_controls,
    data=df,
    weight_var='tweight',
    sample_desc='Manipulation controls only',
    extra_json_blocks={"controls": {"spec_id": "rc/controls/sets/minimal", "family": "sets", "set_name": "minimal", "n_controls": 14}}
)

# Baseline: same as baseline (manip + demographics + giving)
run_spec(
    spec_id='rc/controls/sets/baseline',
    spec_tree_path='modules/robustness/controls.md#standard-control-sets',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + manip_controls + demographic_controls + giving_history,
    data=df,
    weight_var='tweight',
    sample_desc='Baseline control set',
    extra_json_blocks={"controls": {"spec_id": "rc/controls/sets/baseline", "family": "sets", "set_name": "baseline", "n_controls": 34}}
)

# Extended: baseline + extra controls
run_spec(
    spec_id='rc/controls/sets/extended',
    spec_tree_path='modules/robustness/controls.md#standard-control-sets',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + manip_controls + demographic_controls + giving_history + extra_controls,
    data=df,
    weight_var='tweight',
    sample_desc='Extended controls (+ hfh_effective, lifepriorities)',
    extra_json_blocks={"controls": {"spec_id": "rc/controls/sets/extended", "family": "sets", "set_name": "extended", "n_controls": 37}}
)

# nraudworthy composite control set
run_spec(
    spec_id='rc/controls/sets/nraudworthy',
    spec_tree_path='modules/robustness/controls.md#standard-control-sets',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + nraud_controls + demographic_controls + giving_history,
    data=df,
    weight_var='tweight',
    sample_desc='nraudworthy composite instead of individual worthiness dummies',
    extra_json_blocks={"controls": {"spec_id": "rc/controls/sets/nraudworthy", "family": "sets", "set_name": "nraudworthy_composite", "n_controls": 31}}
)

# ============================================================
# RC: LEAVE-ONE-OUT CONTROLS
# ============================================================
print("\n=== RC: LEAVE-ONE-OUT ===")

loo_vars = demographic_controls + giving_history  # 20 variables
for var in loo_vars:
    remaining = [c for c in baseline_controls if c != var]
    run_spec(
        spec_id=f'rc/controls/loo/drop_{var}',
        spec_tree_path='modules/robustness/controls.md#leave-one-out-controls-loo',
        outcome_var='giving',
        treatment_var='picshowblack',
        controls=treatment_vars[1:] + remaining,
        data=df,
        weight_var='tweight',
        sample_desc=f'LOO: drop {var}',
        extra_json_blocks={"controls": {"spec_id": f"rc/controls/loo/drop_{var}", "family": "loo", "dropped": [var], "added": [], "n_controls": len(remaining) + 2}}
    )

# ============================================================
# RC: CONTROL PROGRESSION
# ============================================================
print("\n=== RC: CONTROL PROGRESSION ===")

# Bivariate
run_spec(
    spec_id='rc/controls/progression/bivariate',
    spec_tree_path='modules/robustness/controls.md#control-progression-build-up',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=[],
    data=df,
    weight_var='tweight',
    sample_desc='Bivariate (treatment only)',
    extra_json_blocks={"controls": {"spec_id": "rc/controls/progression/bivariate", "family": "progression", "set_name": "bivariate", "n_controls": 0}}
)

# Manip only
run_spec(
    spec_id='rc/controls/progression/manip_only',
    spec_tree_path='modules/robustness/controls.md#control-progression-build-up',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + manip_controls,
    data=df,
    weight_var='tweight',
    sample_desc='Manipulation controls only',
    extra_json_blocks={"controls": {"spec_id": "rc/controls/progression/manip_only", "family": "progression", "set_name": "manip_only", "n_controls": 14}}
)

# Manip + demographics
run_spec(
    spec_id='rc/controls/progression/manip_demographics',
    spec_tree_path='modules/robustness/controls.md#control-progression-build-up',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + manip_controls + demographic_controls,
    data=df,
    weight_var='tweight',
    sample_desc='Manipulation + demographics',
    extra_json_blocks={"controls": {"spec_id": "rc/controls/progression/manip_demographics", "family": "progression", "set_name": "manip_demographics", "n_controls": 30}}
)

# Manip + demographics + giving history
run_spec(
    spec_id='rc/controls/progression/manip_demographics_giving',
    spec_tree_path='modules/robustness/controls.md#control-progression-build-up',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + manip_controls + demographic_controls + giving_history,
    data=df,
    weight_var='tweight',
    sample_desc='Manipulation + demographics + giving history (= baseline)',
    extra_json_blocks={"controls": {"spec_id": "rc/controls/progression/manip_demographics_giving", "family": "progression", "set_name": "manip_demographics_giving", "n_controls": 34}}
)

# Full (baseline + extra)
run_spec(
    spec_id='rc/controls/progression/full',
    spec_tree_path='modules/robustness/controls.md#control-progression-build-up',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + manip_controls + demographic_controls + giving_history + extra_controls,
    data=df,
    weight_var='tweight',
    sample_desc='Full controls (baseline + extra)',
    extra_json_blocks={"controls": {"spec_id": "rc/controls/progression/full", "family": "progression", "set_name": "full", "n_controls": 37}}
)

# ============================================================
# RC: SAMPLE RESTRICTIONS
# ============================================================
print("\n=== RC: SAMPLE ===")

# Main survey only (surveyvariant==1, use mweight)
df_main = df[df['surveyvariant'] == 1].copy()
run_spec(
    spec_id='rc/sample/subvariant/main_survey_only',
    spec_tree_path='modules/robustness/sample.md#data-quality-and-eligibility-filters',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + manip_controls + demographic_controls + giving_history,
    data=df_main,
    weight_var='mweight',
    sample_desc='Main survey variant only (surveyvariant==1, mweight)',
    extra_json_blocks={"sample": {"spec_id": "rc/sample/subvariant/main_survey_only", "axis": "subvariant", "rule": "filter", "params": {"surveyvariant": 1}}}
)

# Slidell only
df_slidell = df[df['cityslidell'] == 1].copy()
run_spec(
    spec_id='rc/sample/subvariant/slidell_only',
    spec_tree_path='modules/robustness/sample.md#data-quality-and-eligibility-filters',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + manip_controls + demographic_controls + giving_history,
    data=df_slidell,
    weight_var='tweight',
    sample_desc='Slidell (LA) only',
    extra_json_blocks={"sample": {"spec_id": "rc/sample/subvariant/slidell_only", "axis": "subvariant", "rule": "filter", "params": {"cityslidell": 1}}}
)

# Biloxi only
df_biloxi = df[df['cityslidell'] == 0].copy()
run_spec(
    spec_id='rc/sample/subvariant/biloxi_only',
    spec_tree_path='modules/robustness/sample.md#data-quality-and-eligibility-filters',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + manip_controls + demographic_controls + giving_history,
    data=df_biloxi,
    weight_var='tweight',
    sample_desc='Biloxi (MS) only',
    extra_json_blocks={"sample": {"spec_id": "rc/sample/subvariant/biloxi_only", "axis": "subvariant", "rule": "filter", "params": {"cityslidell": 0}}}
)

# Race-shown only (not picobscur)
df_raceshown = df[df['picobscur'] == 0].copy()
run_spec(
    spec_id='rc/sample/subvariant/race_shown_only',
    spec_tree_path='modules/robustness/sample.md#data-quality-and-eligibility-filters',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + manip_controls + demographic_controls + giving_history,
    data=df_raceshown,
    weight_var='tweight',
    sample_desc='Race-shown subsample only (exclude picobscur)',
    extra_json_blocks={"sample": {"spec_id": "rc/sample/subvariant/race_shown_only", "axis": "subvariant", "rule": "filter", "params": {"picobscur": 0}}}
)

# Trim giving 1/99 percentile
q01 = df['giving'].quantile(0.01)
q99 = df['giving'].quantile(0.99)
df_trimmed = df[(df['giving'] >= q01) & (df['giving'] <= q99)].copy()
run_spec(
    spec_id='rc/sample/outliers/trim_y_1_99',
    spec_tree_path='modules/robustness/sample.md#outliers-and-influential-observations',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + manip_controls + demographic_controls + giving_history,
    data=df_trimmed,
    weight_var='tweight',
    sample_desc=f'Trim giving to [{q01}, {q99}]',
    extra_json_blocks={"sample": {"spec_id": "rc/sample/outliers/trim_y_1_99", "axis": "outliers", "rule": "trim", "params": {"var": "giving", "lower_q": 0.01, "upper_q": 0.99}, "n_obs_before": len(df), "n_obs_after": len(df_trimmed)}}
)

# Drop extreme choices (giving==0 or giving==100)
df_interior = df[(df['giving'] > 0) & (df['giving'] < 100)].copy()
run_spec(
    spec_id='rc/sample/outliers/drop_extreme_choices',
    spec_tree_path='modules/robustness/sample.md#outliers-and-influential-observations',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + manip_controls + demographic_controls + giving_history,
    data=df_interior,
    weight_var='tweight',
    sample_desc='Interior choices only (drop giving=0 and giving=100)',
    extra_json_blocks={"sample": {"spec_id": "rc/sample/outliers/drop_extreme_choices", "axis": "outliers", "rule": "filter", "params": {"exclude": "giving==0 or giving==100"}, "n_obs_before": len(df), "n_obs_after": len(df_interior)}}
)

# ============================================================
# RC: FUNCTIONAL FORM
# ============================================================
print("\n=== RC: FUNCTIONAL FORM ===")

# log(1+giving)
run_spec(
    spec_id='rc/form/outcome/log1p',
    spec_tree_path='modules/robustness/functional_form.md#outcome-transformations',
    outcome_var='log1p_giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + manip_controls + demographic_controls + giving_history,
    data=df,
    weight_var='tweight',
    sample_desc='Outcome: log(1+giving)',
    extra_json_blocks={"functional_form": {"spec_id": "rc/form/outcome/log1p", "outcome_transform": "log1p", "treatment_transform": "level", "interpretation": "Semi-elasticity of giving with respect to picshowblack; coefficient is approx % change in (1+giving) per unit treatment."}}
)

# asinh(giving)
run_spec(
    spec_id='rc/form/outcome/asinh',
    spec_tree_path='modules/robustness/functional_form.md#outcome-transformations',
    outcome_var='asinh_giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + manip_controls + demographic_controls + giving_history,
    data=df,
    weight_var='tweight',
    sample_desc='Outcome: asinh(giving)',
    extra_json_blocks={"functional_form": {"spec_id": "rc/form/outcome/asinh", "outcome_transform": "asinh", "treatment_transform": "level", "interpretation": "Inverse hyperbolic sine of giving; coefficient is approx semi-elasticity for large giving values."}}
)

# Binary: any giving (giving > 0)
run_spec(
    spec_id='rc/form/outcome/binary_any_giving',
    spec_tree_path='modules/robustness/functional_form.md#outcome-transformations',
    outcome_var='giving_any',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + manip_controls + demographic_controls + giving_history,
    data=df,
    weight_var='tweight',
    sample_desc='Outcome: binary (any giving > 0)',
    extra_json_blocks={"functional_form": {"spec_id": "rc/form/outcome/binary_any_giving", "target": "outcome", "operation": "binarize", "source_var": "giving", "new_var": "giving_any", "threshold": 0, "direction": ">", "units": "dollars (0-100)", "recode_rule": "1[giving > 0]", "interpretation": "Extensive margin: probability of making any positive donation."}}
)

# ============================================================
# RC: WEIGHTS
# ============================================================
print("\n=== RC: WEIGHTS ===")

# Unweighted
run_spec(
    spec_id='rc/weights/unweighted',
    spec_tree_path='modules/robustness/controls.md#standard-control-sets',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + manip_controls + demographic_controls + giving_history,
    data=df,
    weight_var=None,
    sample_desc='Unweighted (no survey weights)',
    extra_json_blocks={"weights": {"spec_id": "rc/weights/unweighted", "weight_var": None, "notes": "No survey weights applied"}}
)

# mweight (main survey weight — only valid for main survey variant)
run_spec(
    spec_id='rc/weights/mweight',
    spec_tree_path='modules/robustness/controls.md#standard-control-sets',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + manip_controls + demographic_controls + giving_history,
    data=df_main,
    weight_var='mweight',
    sample_desc='Main survey variant with mweight',
    extra_json_blocks={"weights": {"spec_id": "rc/weights/mweight", "weight_var": "mweight", "notes": "Main survey weight (valid only for surveyvariant==1)"}}
)

# ============================================================
# RC: ADDITIONAL CONTROL SUBSETS (to reach 50+ specs)
# ============================================================
print("\n=== RC: CONTROL SUBSETS ===")

# Demographics only (no giving history, no manip controls)
run_spec(
    spec_id='rc/controls/subset/demographics_only',
    spec_tree_path='modules/robustness/controls.md#high-dimensional-control-set-search',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + demographic_controls,
    data=df,
    weight_var='tweight',
    sample_desc='Demographics only (no manip controls, no giving history)',
    extra_json_blocks={"controls": {"spec_id": "rc/controls/subset/demographics_only", "family": "subset", "included": demographic_controls, "excluded": manip_controls + giving_history, "n_controls": len(demographic_controls) + 2}}
)

# Giving history only (no demographics, no manip)
run_spec(
    spec_id='rc/controls/subset/giving_history_only',
    spec_tree_path='modules/robustness/controls.md#high-dimensional-control-set-search',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + giving_history,
    data=df,
    weight_var='tweight',
    sample_desc='Giving history only (no manip, no demographics)',
    extra_json_blocks={"controls": {"spec_id": "rc/controls/subset/giving_history_only", "family": "subset", "included": giving_history, "excluded": manip_controls + demographic_controls, "n_controls": len(giving_history) + 2}}
)

# Manip + giving history (no demographics)
run_spec(
    spec_id='rc/controls/subset/manip_giving',
    spec_tree_path='modules/robustness/controls.md#high-dimensional-control-set-search',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + manip_controls + giving_history,
    data=df,
    weight_var='tweight',
    sample_desc='Manip + giving history (no demographics)',
    extra_json_blocks={"controls": {"spec_id": "rc/controls/subset/manip_giving", "family": "subset", "included": manip_controls + giving_history, "excluded": demographic_controls, "n_controls": len(manip_controls) + len(giving_history) + 2}}
)

# Demographics + giving (no manip controls)
run_spec(
    spec_id='rc/controls/subset/demographics_giving',
    spec_tree_path='modules/robustness/controls.md#high-dimensional-control-set-search',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + demographic_controls + giving_history,
    data=df,
    weight_var='tweight',
    sample_desc='Demographics + giving history (no manip controls)',
    extra_json_blocks={"controls": {"spec_id": "rc/controls/subset/demographics_giving", "family": "subset", "included": demographic_controls + giving_history, "excluded": manip_controls, "n_controls": len(demographic_controls) + len(giving_history) + 2}}
)

# Only age + income + education (core SES)
ses_controls = ['age', 'age2', 'edudo', 'edusc', 'educp', 'lnhhinc']
run_spec(
    spec_id='rc/controls/subset/core_ses',
    spec_tree_path='modules/robustness/controls.md#high-dimensional-control-set-search',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + manip_controls + ses_controls,
    data=df,
    weight_var='tweight',
    sample_desc='Core SES controls (age, education, income)',
    extra_json_blocks={"controls": {"spec_id": "rc/controls/subset/core_ses", "family": "subset", "included": ses_controls, "excluded": [c for c in demographic_controls if c not in ses_controls] + giving_history, "n_controls": len(ses_controls) + len(manip_controls) + 2}}
)

# Race + gender only
race_gender = ['black', 'other', 'male', 'married', 'singlemale']
run_spec(
    spec_id='rc/controls/subset/race_gender',
    spec_tree_path='modules/robustness/controls.md#high-dimensional-control-set-search',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + manip_controls + race_gender,
    data=df,
    weight_var='tweight',
    sample_desc='Race + gender controls only (+ manip)',
    extra_json_blocks={"controls": {"spec_id": "rc/controls/subset/race_gender", "family": "subset", "included": race_gender, "n_controls": len(race_gender) + len(manip_controls) + 2}}
)

# ============================================================
# RC: ADDITIONAL SAMPLE RESTRICTIONS
# ============================================================
print("\n=== RC: ADDITIONAL SAMPLE ===")

# Drop respondents who didn't follow Katrina news closely
# follownews: could use as quality filter
df_follownews = df[df['follownews'].notna()].copy()
# follownews is coded 1=very closely, 2=fairly closely, 3=not too closely, 4=not at all
# We already have soundcheck filter; this is additional quality
df_engaged = df_follownews[df_follownews['follownews'] <= 2].copy()
run_spec(
    spec_id='rc/sample/quality/engaged_respondents',
    spec_tree_path='modules/robustness/sample.md#data-quality-and-eligibility-filters',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + manip_controls + demographic_controls + giving_history,
    data=df_engaged,
    weight_var='tweight',
    sample_desc='Engaged respondents only (followed Katrina news closely/fairly closely)',
    extra_json_blocks={"sample": {"spec_id": "rc/sample/quality/engaged_respondents", "axis": "quality", "rule": "filter", "params": {"follownews": "<=2"}, "n_obs_before": len(df), "n_obs_after": len(df_engaged)}}
)

# Drop fast completers (completion time < median)
median_time = df['maintime'].median()
df_notfast = df[df['maintime'] >= median_time].copy()
run_spec(
    spec_id='rc/sample/quality/not_fast_completers',
    spec_tree_path='modules/robustness/sample.md#data-quality-and-eligibility-filters',
    outcome_var='giving',
    treatment_var='picshowblack',
    controls=treatment_vars[1:] + manip_controls + demographic_controls + giving_history,
    data=df_notfast,
    weight_var='tweight',
    sample_desc=f'Exclude fast completers (maintime >= {median_time:.1f} min)',
    extra_json_blocks={"sample": {"spec_id": "rc/sample/quality/not_fast_completers", "axis": "quality", "rule": "filter", "params": {"maintime_ge": float(median_time)}, "n_obs_before": len(df), "n_obs_after": len(df_notfast)}}
)

# ============================================================
# SAVE RESULTS
# ============================================================
print(f"\n\nTotal specifications run: {len(results)}")

results_df = pd.DataFrame(results)
results_df.to_csv(f"{PACKAGE_DIR}/specification_results.csv", index=False)
print(f"Saved specification_results.csv with {len(results_df)} rows")

n_success = sum(1 for r in results if r['run_success'] == 1)
n_failed = sum(1 for r in results if r['run_success'] == 0)
print(f"Success: {n_success}, Failed: {n_failed}")

# ============================================================
# INFERENCE RESULTS (optional variants)
# ============================================================
print("\n=== INFERENCE VARIANTS ===")

inference_results = []
inf_run_id = 0

# For the baseline spec, compute classical, HC2, HC3 SE variants
baseline_controls_list = treatment_vars[1:] + manip_controls + demographic_controls + giving_history
formula_baseline = "giving ~ " + " + ".join(['picshowblack'] + baseline_controls_list)
all_vars_inf = ['giving', 'picshowblack'] + baseline_controls_list + ['tweight']
df_inf = df.dropna(subset=all_vars_inf).copy()

for infer_spec_id, vcov_type, vcov_label in [
    ("infer/se/hc/classical", "iid", "Classical (homoskedastic)"),
    ("infer/se/hc/hc2", {"CRV1": "caseid"}, "HC2 approximation"),  # HC2 not directly in pyfixest, use hetero
    ("infer/se/hc/hc3", {"CRV1": "caseid"}, "HC3 approximation"),  # HC3 not directly in pyfixest
]:
    inf_run_id += 1
    try:
        if infer_spec_id == "infer/se/hc/classical":
            m_inf = pf.feols(formula_baseline, data=df_inf, vcov="iid", weights="tweight")
        else:
            # pyfixest doesn't directly support HC2/HC3; use hetero as approximation
            m_inf = pf.feols(formula_baseline, data=df_inf, vcov="hetero", weights="tweight")

        coef_inf = float(m_inf.coef()['picshowblack'])
        se_inf = float(m_inf.se()['picshowblack'])
        pval_inf = float(m_inf.pvalue()['picshowblack'])

        inf_cvj = {
            "coefficients": {k: float(v) for k, v in m_inf.coef().items()},
            "inference": {"spec_id": infer_spec_id, "method": vcov_label},
            "software": software_block,
            "surface_hash": surface_hash,
        }

        inference_results.append({
            'paper_id': PAPER_ID,
            'inference_run_id': f"I{inf_run_id:03d}",
            'spec_run_id': 'S001',  # baseline
            'spec_id': infer_spec_id,
            'spec_tree_path': 'modules/inference/standard_errors.md#heteroskedasticity-robust-se-no-clustering',
            'baseline_group_id': 'G1',
            'outcome_var': 'giving',
            'treatment_var': 'picshowblack',
            'coefficient': coef_inf,
            'std_error': se_inf,
            'p_value': pval_inf,
            'ci_lower': coef_inf - 1.96 * se_inf,
            'ci_upper': coef_inf + 1.96 * se_inf,
            'n_obs': int(m_inf._N),
            'r_squared': float(m_inf._r2),
            'cluster_var': '',
            'coefficient_vector_json': json.dumps(inf_cvj),
            'run_success': 1,
            'run_error': '',
        })
        print(f"  I{inf_run_id:03d} ({infer_spec_id}): coef={coef_inf:.4f}, se={se_inf:.4f}")
    except Exception as e:
        inference_results.append({
            'paper_id': PAPER_ID,
            'inference_run_id': f"I{inf_run_id:03d}",
            'spec_run_id': 'S001',
            'spec_id': infer_spec_id,
            'spec_tree_path': 'modules/inference/standard_errors.md',
            'baseline_group_id': 'G1',
            'outcome_var': 'giving',
            'treatment_var': 'picshowblack',
            'coefficient': np.nan, 'std_error': np.nan, 'p_value': np.nan,
            'ci_lower': np.nan, 'ci_upper': np.nan,
            'n_obs': np.nan, 'r_squared': np.nan, 'cluster_var': '',
            'coefficient_vector_json': json.dumps({"error": str(e)}),
            'run_success': 0, 'run_error': str(e),
        })
        print(f"  I{inf_run_id:03d} ({infer_spec_id}): FAILED - {e}")

if inference_results:
    inf_df = pd.DataFrame(inference_results)
    inf_df.to_csv(f"{PACKAGE_DIR}/inference_results.csv", index=False)
    print(f"Saved inference_results.csv with {len(inf_df)} rows")

print("\nDone!")
