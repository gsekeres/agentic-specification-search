"""
Specification Search Script for Ferraz & Finan (2011)
"Electoral Accountability and Corruption: Evidence from the Audits of Local Governments"
American Economic Review, 101(4), 1274-1311.

Paper ID: 112431-V1

Surface-driven execution:
  - G1: pcorrupt ~ first (Table 4 Col 6 baseline)
  - Cross-sectional OLS with state FE, HC1 robust SE
  - 51 specifications total

Outputs:
  - specification_results.csv (baseline, design/*, rc/* rows)
  - inference_results.csv (infer/* rows)
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import sys
import hashlib
import traceback
import warnings
warnings.filterwarnings('ignore')

# Add scripts dir for utilities
sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

DATA_DIR = "data/downloads/extracted/112431-V1"
PAPER_ID = "112431-V1"

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

# Load data
df_raw = pd.read_stata(f"{DATA_DIR}/corruptiondata_aer.dta")
df_raw = df_raw[df_raw['esample2'] == 1].copy()
for col in df_raw.columns:
    if df_raw[col].dtype == np.float32:
        df_raw[col] = df_raw[col].astype(np.float64)

# Define variable groups
party_vars = ['party_d1'] + [f'party_d{i}' for i in range(3, 19)]
prefchar2 = ['pref_masc', 'pref_idade_tse', 'pref_escola'] + party_vars
munichar2 = ['lpop', 'purb', 'p_secundario', 'mun_novo', 'lpib02', 'gini_ipea']
sorteio_vars = [f'sorteio{i}' for i in range(1, 11)]
political = ['p_cad_pref', 'vereador_eleit', 'ENLP2000', 'comarca']
baseline_controls = prefchar2 + munichar2 + ['lrec_trans'] + political + sorteio_vars

design_audit = surface_obj["baseline_groups"][0]["design_audit"]
inference_canonical = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]

results = []
inference_results = []
spec_run_counter = 0


def run_spec(spec_id, spec_tree_path, baseline_group_id, outcome_var, treatment_var,
             controls, fixed_effects_str, fe_formula, data, vcov,
             sample_desc, controls_desc, cluster_var="",
             axis_block_name=None, axis_block=None, notes=""):
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        controls_str = " + ".join(controls) if controls else ""
        if controls_str and fe_formula:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str} | {fe_formula}"
        elif controls_str:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str}"
        elif fe_formula:
            formula = f"{outcome_var} ~ {treatment_var} | {fe_formula}"
        else:
            formula = f"{outcome_var} ~ {treatment_var}"

        m = pf.feols(formula, data=data, vcov=vcov)

        coef_val = float(m.coef().get(treatment_var, np.nan))
        se_val = float(m.se().get(treatment_var, np.nan))
        pval = float(m.pvalue().get(treatment_var, np.nan))

        try:
            ci = m.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]]) if treatment_var in ci.index else np.nan
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]]) if treatment_var in ci.index else np.nan
        except:
            ci_lower = np.nan
            ci_upper = np.nan

        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except:
            r2 = np.nan

        all_coefs = {k: float(v) for k, v in m.coef().items()}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                      "method": "robust", "type": "HC1"},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"cross_sectional_ols": design_audit},
            axis_block_name=axis_block_name,
            axis_block=axis_block,
        )

        results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": nobs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": fixed_effects_str,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 1,
            "run_error": ""
        })
        return run_id, coef_val, se_val, pval, nobs

    except Exception as e:
        err_msg = str(e)[:240]
        err_details = error_details_from_exception(e, stage="estimation")
        payload = make_failure_payload(
            error=err_msg,
            error_details=err_details,
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH
        )

        results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": fixed_effects_str,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ===== BASELINE =====
run_spec("baseline", "designs/cross_sectional_ols.md#baseline", "G1",
         "pcorrupt", "first", baseline_controls,
         "uf (state)", "uf", df_raw, "hetero",
         "esample2==1 (N=476)", "prefchar2 + munichar2 + lrec_trans + political + sorteio")

# ===== ADDITIONAL BASELINES =====
run_spec("baseline__ncorrupt", "designs/cross_sectional_ols.md#baseline", "G1",
         "ncorrupt", "first", ["lrec_fisc"] + baseline_controls,
         "uf (state)", "uf", df_raw, "hetero",
         "esample2==1", "lrec_fisc + prefchar2 + munichar2 + lrec_trans + political + sorteio")

run_spec("baseline__ncorrupt_os", "designs/cross_sectional_ols.md#baseline", "G1",
         "ncorrupt_os", "first", ["lrec_fisc"] + baseline_controls + ["lfunc_ativ"],
         "uf (state)", "uf", df_raw, "hetero",
         "esample2==1", "lrec_fisc + prefchar2 + munichar2 + lrec_trans + lfunc_ativ + political + sorteio")

# ===== RC: CONTROLS LOO =====
loo_candidates = ['pref_masc', 'pref_idade_tse', 'pref_escola',
                  'lpop', 'purb', 'p_secundario', 'mun_novo', 'lpib02', 'gini_ipea',
                  'p_cad_pref', 'vereador_eleit', 'ENLP2000', 'lrec_trans']

for var in loo_candidates:
    ctrl = [c for c in baseline_controls if c != var]
    run_spec(f"rc/controls/loo/drop_{var}",
             "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
             "pcorrupt", "first", ctrl,
             "uf (state)", "uf", df_raw, "hetero",
             "esample2==1", f"baseline minus {var}",
             axis_block_name="controls",
             axis_block={"spec_id": f"rc/controls/loo/drop_{var}", "family": "loo",
                        "dropped": [var], "added": [], "n_controls": len(ctrl)})

# ===== RC: CONTROL SETS =====
run_spec("rc/controls/sets/none",
         "modules/robustness/controls.md#standard-control-sets", "G1",
         "pcorrupt", "first", [],
         "uf (state)", "uf", df_raw, "hetero",
         "esample2==1", "none (bivariate + state FE)",
         axis_block_name="controls",
         axis_block={"spec_id": "rc/controls/sets/none", "family": "sets",
                    "dropped": baseline_controls, "added": [], "n_controls": 0, "set_name": "none"})

run_spec("rc/controls/sets/minimal",
         "modules/robustness/controls.md#standard-control-sets", "G1",
         "pcorrupt", "first", prefchar2,
         "uf (state)", "uf", df_raw, "hetero",
         "esample2==1", "prefchar2 only",
         axis_block_name="controls",
         axis_block={"spec_id": "rc/controls/sets/minimal", "family": "sets",
                    "dropped": [], "added": [], "n_controls": len(prefchar2), "set_name": "minimal"})

extended = baseline_controls + ['lfunc_ativ', 'lrec_fisc']
run_spec("rc/controls/sets/extended",
         "modules/robustness/controls.md#standard-control-sets", "G1",
         "pcorrupt", "first", extended,
         "uf (state)", "uf", df_raw, "hetero",
         "esample2==1", "baseline + lfunc_ativ + lrec_fisc",
         axis_block_name="controls",
         axis_block={"spec_id": "rc/controls/sets/extended", "family": "sets",
                    "dropped": [], "added": ["lfunc_ativ", "lrec_fisc"], "n_controls": len(extended),
                    "set_name": "extended"})

# ===== RC: CONTROL PROGRESSION =====
prog_steps = [
    ("rc/controls/progression/bivariate", [], "bivariate"),
    ("rc/controls/progression/mayor_chars", prefchar2, "mayor characteristics"),
    ("rc/controls/progression/municipal_chars", prefchar2 + munichar2, "mayor + municipal"),
    ("rc/controls/progression/political", prefchar2 + munichar2 + ['lrec_trans'] + political, "mayor + municipal + political"),
    ("rc/controls/progression/lottery_dummies", prefchar2 + munichar2 + ['lrec_trans'] + political + sorteio_vars, "mayor + municipal + political + sorteio"),
    ("rc/controls/progression/full", baseline_controls + ['lfunc_ativ', 'lrec_fisc'], "all available controls"),
]

for spec_id, ctrls, desc in prog_steps:
    run_spec(spec_id, "modules/robustness/controls.md#control-progression-build-up", "G1",
             "pcorrupt", "first", ctrls,
             "uf (state)", "uf", df_raw, "hetero",
             "esample2==1", desc,
             axis_block_name="controls",
             axis_block={"spec_id": spec_id, "family": "progression",
                        "dropped": [], "added": [], "n_controls": len(ctrls), "set_name": desc})

# ===== RC: CONTROL SUBSET SEARCH =====
rng = np.random.RandomState(112431)
subset_pool = ['pref_masc', 'pref_idade_tse', 'pref_escola',
               'lpop', 'purb', 'p_secundario', 'mun_novo', 'lpib02', 'gini_ipea',
               'lrec_trans', 'p_cad_pref', 'vereador_eleit', 'ENLP2000', 'comarca']
mandatory = party_vars + sorteio_vars

for draw_i in range(1, 21):
    k = rng.randint(3, len(subset_pool) + 1)
    chosen = list(rng.choice(subset_pool, size=k, replace=False))
    ctrls = mandatory + chosen
    excluded = [v for v in subset_pool if v not in chosen]
    spec_id = f"rc/controls/subset/random_{draw_i:03d}"
    run_spec(spec_id, "modules/robustness/controls.md#subset-generation-specids", "G1",
             "pcorrupt", "first", ctrls,
             "uf (state)", "uf", df_raw, "hetero",
             "esample2==1", f"random subset draw {draw_i}",
             axis_block_name="controls",
             axis_block={"spec_id": spec_id, "family": "subset", "method": "random",
                        "seed": 112431, "draw_index": draw_i,
                        "pool": subset_pool, "mandatory": mandatory,
                        "included": chosen, "excluded": excluded,
                        "n_controls": len(ctrls)})

# ===== RC: SAMPLE =====
q01 = df_raw['pcorrupt'].quantile(0.01)
q99 = df_raw['pcorrupt'].quantile(0.99)
df_trim1 = df_raw[(df_raw['pcorrupt'] >= q01) & (df_raw['pcorrupt'] <= q99)].copy()
n_before = len(df_raw)
run_spec("rc/sample/outliers/trim_y_1_99",
         "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
         "pcorrupt", "first", baseline_controls,
         "uf (state)", "uf", df_trim1, "hetero",
         f"trim pcorrupt [1%,99%]", "baseline controls",
         axis_block_name="sample",
         axis_block={"spec_id": "rc/sample/outliers/trim_y_1_99", "axis": "outliers",
                    "rule": "trim", "params": {"var": "pcorrupt", "lower_q": 0.01, "upper_q": 0.99},
                    "n_obs_before": n_before, "n_obs_after": len(df_trim1)})

q05 = df_raw['pcorrupt'].quantile(0.05)
q95 = df_raw['pcorrupt'].quantile(0.95)
df_trim5 = df_raw[(df_raw['pcorrupt'] >= q05) & (df_raw['pcorrupt'] <= q95)].copy()
run_spec("rc/sample/outliers/trim_y_5_95",
         "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
         "pcorrupt", "first", baseline_controls,
         "uf (state)", "uf", df_trim5, "hetero",
         f"trim pcorrupt [5%,95%]", "baseline controls",
         axis_block_name="sample",
         axis_block={"spec_id": "rc/sample/outliers/trim_y_5_95", "axis": "outliers",
                    "rule": "trim", "params": {"var": "pcorrupt", "lower_q": 0.05, "upper_q": 0.95},
                    "n_obs_before": n_before, "n_obs_after": len(df_trim5)})

# ===== RC: FIXED EFFECTS =====
run_spec("rc/fe/drop/uf", "modules/robustness/fixed_effects.md#dropping-fe-relative-to-baseline", "G1",
         "pcorrupt", "first", baseline_controls,
         "none", "", df_raw, "hetero",
         "esample2==1", "baseline controls, no state FE",
         axis_block_name="fixed_effects",
         axis_block={"spec_id": "rc/fe/drop/uf", "family": "drop",
                    "added": [], "dropped": ["uf"],
                    "baseline_fe": ["uf"], "new_fe": []})

df_raw['nsorteio_cat'] = df_raw['nsorteio'].astype(str)
run_spec("rc/fe/add/nsorteio", "modules/robustness/fixed_effects.md#additive-fe-variations-relative-to-baseline", "G1",
         "pcorrupt", "first", [c for c in baseline_controls if not c.startswith('sorteio')],
         "uf + nsorteio", "uf + nsorteio_cat", df_raw, "hetero",
         "esample2==1", "baseline controls (minus sorteio dummies) + uf + nsorteio FE",
         axis_block_name="fixed_effects",
         axis_block={"spec_id": "rc/fe/add/nsorteio", "family": "add",
                    "added": ["nsorteio"], "dropped": [],
                    "baseline_fe": ["uf"], "new_fe": ["uf", "nsorteio"]})

# ===== RC: FUNCTIONAL FORM =====
df_raw['pcorrupt_asinh'] = np.arcsinh(df_raw['pcorrupt'])
run_spec("rc/form/outcome/asinh", "modules/robustness/functional_form.md#outcome-transformations", "G1",
         "pcorrupt_asinh", "first", baseline_controls,
         "uf (state)", "uf", df_raw, "hetero",
         "esample2==1", "baseline controls",
         axis_block_name="functional_form",
         axis_block={"spec_id": "rc/form/outcome/asinh",
                    "outcome_transform": "asinh", "treatment_transform": "level",
                    "interpretation": "Inverse hyperbolic sine of corruption share; approx log for large values, handles zeros."})

df_raw['pcorrupt_log1p'] = np.log1p(df_raw['pcorrupt'])
run_spec("rc/form/outcome/log1p", "modules/robustness/functional_form.md#outcome-transformations", "G1",
         "pcorrupt_log1p", "first", baseline_controls,
         "uf (state)", "uf", df_raw, "hetero",
         "esample2==1", "baseline controls",
         axis_block_name="functional_form",
         axis_block={"spec_id": "rc/form/outcome/log1p",
                    "outcome_transform": "log1p", "treatment_transform": "level",
                    "interpretation": "Log(1+pcorrupt); handles zeros. Coefficient is approx semi-elasticity."})

# ===== INFERENCE VARIANTS =====
baseline_run_id = f"{PAPER_ID}_run_001"
controls_str = " + ".join(baseline_controls)
m = pf.feols(f"pcorrupt ~ first + {controls_str} | uf", data=df_raw, vcov={"CRV1": "uf"})
all_coefs = {k: float(v) for k, v in m.coef().items()}
ci = m.confint()
payload = make_success_payload(
    coefficients=all_coefs,
    inference={"spec_id": "infer/se/cluster/uf", "method": "cluster",
              "cluster_vars": ["uf"], "n_clusters": {"uf": int(df_raw['uf'].nunique())}},
    software=SW_BLOCK,
    surface_hash=SURFACE_HASH,
    design={"cross_sectional_ols": design_audit}
)

inference_results.append({
    "paper_id": PAPER_ID,
    "inference_run_id": f"{PAPER_ID}_infer_001",
    "spec_run_id": baseline_run_id,
    "spec_id": "infer/se/cluster/uf",
    "spec_tree_path": "modules/inference/standard_errors.md#single-level-clustering",
    "baseline_group_id": "G1",
    "coefficient": float(m.coef()["first"]),
    "std_error": float(m.se()["first"]),
    "p_value": float(m.pvalue()["first"]),
    "ci_lower": float(ci.loc["first", ci.columns[0]]),
    "ci_upper": float(ci.loc["first", ci.columns[1]]),
    "n_obs": int(m._N),
    "r_squared": float(m._r2),
    "coefficient_vector_json": json.dumps(payload),
    "cluster_var": "uf",
    "run_success": 1,
    "run_error": ""
})

# ===== WRITE OUTPUTS =====
spec_df = pd.DataFrame(results)
spec_df.to_csv(f"{DATA_DIR}/specification_results.csv", index=False)

if inference_results:
    infer_df = pd.DataFrame(inference_results)
    infer_df.to_csv(f"{DATA_DIR}/inference_results.csv", index=False)

print(f"Wrote {len(results)} specification results and {len(inference_results)} inference results")
