"""
Specification Search Script for Kosfeld and Rustagi (2015)
"Leader Punishment and Cooperation in Groups:
 Experimental Field Evidence from Ethiopia"
American Economic Review, 105(2), 747-783.

Paper ID: 116139-V1

Surface-driven execution:
  - G1: pct ~ leq + leqef + las + controls | vcode (OLS, Table 6)
  - G2: pi ~ C(cd1) (Poisson, Table 3)
  - Strict adherence to surface core_universe spec_ids

Outputs:
  - specification_results.csv (baseline, design/*, rc/* rows)
  - inference_results.csv (infer/* rows)
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import statsmodels.formula.api as smf
import statsmodels.api as sm
import json
import sys
import warnings
import traceback
warnings.filterwarnings('ignore')

# Add scripts dir for utilities
sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

PAPER_ID = "116139-V1"
DATA_DIR = "data/downloads/extracted/116139-V1"
OUTPUT_DIR = DATA_DIR

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

# Design audit blocks from surface
G1_DESIGN_AUDIT = surface_obj["baseline_groups"][0]["design_audit"]
G2_DESIGN_AUDIT = surface_obj["baseline_groups"][1]["design_audit"]
G1_INFERENCE_CANONICAL = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]
G2_INFERENCE_CANONICAL = surface_obj["baseline_groups"][1]["inference_plan"]["canonical"]

# =============================================================================
# Load Data
# =============================================================================

df_group = pd.read_stata(f"{DATA_DIR}/data/Leader_Group_AER_2014.dta")
df_group['vcode'] = df_group['vcode'].astype(int)

df_pun = pd.read_stata(f"{DATA_DIR}/data/Leader_Pun_Poisson_AER_2014.dta")
df_pun['cd1'] = df_pun['cd1'].astype(int)
df_pun['fcode'] = df_pun['fcode'].astype(int)

# =============================================================================
# G1 variable definitions
# =============================================================================

TREATMENT_VARS = ["leq", "leqef", "las"]
BASELINE_CONTROLS = ["ccs", "ed", "pp", "gs", "wmk", "time", "fem", "ginic"]
EXTENDED_CONTROLS_LEADER = ["lage", "ledu", "lclanp"]
EXTENDED_CONTROLS_HET = ["chet", "shet", "ginil"]
APPENDIX_CONTROLS = ["turnover", "leaderskill", "clan1", "clan2", "clan3", "peren", "seas", "slope"]
ALL_CONTROLS = BASELINE_CONTROLS + EXTENDED_CONTROLS_HET + EXTENDED_CONTROLS_LEADER

# Results storage
spec_rows = []
infer_rows = []
run_counter = 0


def next_run_id():
    global run_counter
    run_counter += 1
    return f"{PAPER_ID}__run_{run_counter:04d}"


# =============================================================================
# Helper: Run G1 OLS specification
# =============================================================================

def run_g1_ols(spec_id, spec_tree_path, controls, fe_var="vcode",
               treatment_vars=None, outcome_var="pct",
               sample_filter=None, sample_desc="full sample",
               axis_block_name=None, axis_block=None,
               extra=None, func_form=None, notes=None):
    """Run a G1 OLS spec and return a row dict."""
    rid = next_run_id()
    row = {
        "paper_id": PAPER_ID,
        "spec_run_id": rid,
        "spec_id": spec_id,
        "spec_tree_path": spec_tree_path,
        "baseline_group_id": "G1",
        "outcome_var": outcome_var,
        "treatment_var": "leq",
    }
    if treatment_vars is None:
        treatment_vars = TREATMENT_VARS

    try:
        df = df_group.copy()
        if sample_filter is not None:
            df = df[sample_filter(df)].copy()

        rhs = " + ".join(treatment_vars + controls)
        if fe_var:
            formula = f"{outcome_var} ~ {rhs} | {fe_var}"
        else:
            formula = f"{outcome_var} ~ {rhs}"

        m = pf.feols(formula, data=df, vcov="hetero")

        focal_var = treatment_vars[0]
        coefs = {v: float(m.coef()[v]) for v in m.coef().index}
        ses = {v: float(m.se()[v]) for v in m.se().index}
        pvals = {v: float(m.pvalue()[v]) for v in m.pvalue().index}

        ci = m.confint()
        ci_lower = float(ci.loc[focal_var, ci.columns[0]])
        ci_upper = float(ci.loc[focal_var, ci.columns[1]])

        payload = make_success_payload(
            coefficients=coefs,
            inference={"spec_id": G1_INFERENCE_CANONICAL["spec_id"],
                       "params": G1_INFERENCE_CANONICAL.get("params", {}),
                       "se_type": "HC1"},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"randomized_experiment": G1_DESIGN_AUDIT},
        )
        if axis_block_name and axis_block:
            payload[axis_block_name] = axis_block
        if func_form:
            payload["functional_form"] = func_form
        if extra:
            payload["extra"] = extra
        if notes:
            payload["notes"] = notes

        row.update({
            "coefficient": float(m.coef()[focal_var]),
            "std_error": float(m.se()[focal_var]),
            "p_value": float(m.pvalue()[focal_var]),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": int(m._N),
            "r_squared": float(m._r2),
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": fe_var if fe_var else "",
            "controls_desc": ", ".join(controls) if controls else "none",
            "cluster_var": "",
            "run_success": 1,
            "run_error": "",
        })
    except Exception as e:
        err_str = str(e)[:240]
        payload = make_failure_payload(
            error=err_str,
            error_details=error_details_from_exception(e, stage="estimation"),
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
        )
        row.update({
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": fe_var if fe_var else "",
            "controls_desc": ", ".join(controls) if controls else "none",
            "cluster_var": "",
            "run_success": 0,
            "run_error": err_str,
        })
    return row


# =============================================================================
# Helper: Run G2 Poisson specification
# =============================================================================

def run_g2_poisson(spec_id, spec_tree_path, controls=None, outcome_var="pi",
                   cluster_var="fcode", sample_filter=None, sample_desc="full sample",
                   model_type="poisson", axis_block_name=None, axis_block=None,
                   func_form=None, extra=None, notes=None):
    """Run a G2 Poisson/NB/OLS spec and return a row dict."""
    rid = next_run_id()
    row = {
        "paper_id": PAPER_ID,
        "spec_run_id": rid,
        "spec_id": spec_id,
        "spec_tree_path": spec_tree_path,
        "baseline_group_id": "G2",
        "outcome_var": outcome_var,
        "treatment_var": "cd1",
    }
    if controls is None:
        controls = []

    try:
        df = df_pun.copy()
        if sample_filter is not None:
            df = df[sample_filter(df)].copy()

        ctrl_str = " + " + " + ".join(controls) if controls else ""
        formula = f"{outcome_var} ~ C(cd1, Treatment(reference=1)){ctrl_str}"

        if model_type == "poisson":
            if cluster_var:
                m = smf.poisson(formula, data=df).fit(
                    cov_type='cluster', cov_kwds={'groups': df[cluster_var]}, disp=0
                )
            else:
                m = smf.poisson(formula, data=df).fit(cov_type='HC1', disp=0)
        elif model_type == "nbreg":
            if cluster_var:
                m = smf.negativebinomial(formula, data=df).fit(
                    cov_type='cluster', cov_kwds={'groups': df[cluster_var]},
                    disp=0, maxiter=200
                )
            else:
                m = smf.negativebinomial(formula, data=df).fit(
                    cov_type='HC1', disp=0, maxiter=200
                )
        elif model_type == "ols":
            if cluster_var:
                m = smf.ols(formula, data=df).fit(
                    cov_type='cluster', cov_kwds={'groups': df[cluster_var]}
                )
            else:
                m = smf.ols(formula, data=df).fit(cov_type='HC1')

        # Focal coefficient: cd1=5 (first inequality condition, representative)
        focal_key = "C(cd1, Treatment(reference=1))[T.5]"
        if focal_key not in m.params.index:
            # Fallback: first non-intercept coefficient
            non_int = [k for k in m.params.index if k != "Intercept"]
            focal_key = non_int[0] if non_int else m.params.index[0]

        coefs = {v: float(m.params[v]) for v in m.params.index}
        ses = {v: float(m.bse[v]) for v in m.bse.index}
        pvals = {v: float(m.pvalues[v]) for v in m.pvalues.index}

        ci = m.conf_int()
        ci_lower = float(ci.loc[focal_key, 0])
        ci_upper = float(ci.loc[focal_key, 1])

        r2 = float(m.rsquared) if hasattr(m, 'rsquared') and not np.isnan(m.rsquared) else np.nan
        if hasattr(m, 'prsquared'):
            r2 = float(m.prsquared)

        infer_spec = G2_INFERENCE_CANONICAL if cluster_var else {"spec_id": "infer/se/hc/robust", "params": {}}

        payload = make_success_payload(
            coefficients=coefs,
            inference={"spec_id": infer_spec["spec_id"],
                       "params": infer_spec.get("params", {}),
                       "se_type": "cluster" if cluster_var else "HC1"},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"randomized_experiment": G2_DESIGN_AUDIT},
        )
        if axis_block_name and axis_block:
            payload[axis_block_name] = axis_block
        if func_form:
            payload["functional_form"] = func_form
        if extra:
            payload["extra"] = extra
        if notes:
            payload["notes"] = notes

        row.update({
            "coefficient": float(m.params[focal_key]),
            "std_error": float(m.bse[focal_key]),
            "p_value": float(m.pvalues[focal_key]),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": int(m.nobs),
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": "",
            "controls_desc": ", ".join(controls) if controls else "none",
            "cluster_var": cluster_var if cluster_var else "",
            "run_success": 1,
            "run_error": "",
        })
    except Exception as e:
        err_str = str(e)[:240]
        payload = make_failure_payload(
            error=err_str,
            error_details=error_details_from_exception(e, stage="estimation"),
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
        )
        row.update({
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": "",
            "controls_desc": ", ".join(controls) if controls else "none",
            "cluster_var": cluster_var if cluster_var else "",
            "run_success": 0,
            "run_error": err_str,
        })
    return row


# =============================================================================
# G1 BASELINE SPECIFICATIONS
# =============================================================================

print("=" * 60)
print("G1: Baseline specifications")
print("=" * 60)

# --- Primary baseline: Table 6 Col 3 ---
r = run_g1_ols(
    spec_id="baseline",
    spec_tree_path="specification_tree/designs/randomized_experiment.md#ols-with-covariates",
    controls=BASELINE_CONTROLS,
    fe_var="vcode",
    sample_desc="lcode non-missing, N~51",
    notes="Table 6 Col 3: pct ~ leq leqef las + 8 controls | vcode, robust"
)
spec_rows.append(r)
print(f"  baseline: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")

# --- Additional baseline: Table 6 Col 1 (no controls, no FE) ---
r = run_g1_ols(
    spec_id="baseline__table6_col1",
    spec_tree_path="specification_tree/designs/randomized_experiment.md#difference-in-means",
    controls=[],
    fe_var=None,
    sample_desc="Table 6 Col 1: no controls, no FE",
    notes="Table 6 Col 1: pct ~ leq leqef las, robust"
)
spec_rows.append(r)
print(f"  Col1: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")

# --- Additional baseline: Table 6 Col 2 (controls, no FE) ---
r = run_g1_ols(
    spec_id="baseline__table6_col2",
    spec_tree_path="specification_tree/designs/randomized_experiment.md#ols-with-covariates",
    controls=BASELINE_CONTROLS,
    fe_var=None,
    sample_desc="Table 6 Col 2: group controls, no FE",
    notes="Table 6 Col 2: pct ~ leq leqef las + 8 controls, robust"
)
spec_rows.append(r)
print(f"  Col2: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")

# --- Additional baseline: Table 6 Col 8 (controls + leader demographics + FE) ---
r = run_g1_ols(
    spec_id="baseline__table6_col8",
    spec_tree_path="specification_tree/designs/randomized_experiment.md#ols-with-covariates",
    controls=BASELINE_CONTROLS + EXTENDED_CONTROLS_LEADER,
    fe_var="vcode",
    sample_desc="Table 6 Col 8: + leader demographics",
    notes="Table 6 Col 8: pct ~ leq leqef las + 8 baseline + lage ledu lclanp | vcode, robust"
)
spec_rows.append(r)
print(f"  Col8: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")

# --- Additional baseline: Table 6 Col 9 (full controls + FE) ---
r = run_g1_ols(
    spec_id="baseline__table6_col9",
    spec_tree_path="specification_tree/designs/randomized_experiment.md#ols-with-covariates",
    controls=ALL_CONTROLS,
    fe_var="vcode",
    sample_desc="Table 6 Col 9: full controls",
    notes="Table 6 Col 9: pct ~ leq leqef las + all 14 controls | vcode, robust"
)
spec_rows.append(r)
print(f"  Col9: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")


# =============================================================================
# G1 DESIGN VARIANTS
# =============================================================================

print("\n" + "=" * 60)
print("G1: Design variants")
print("=" * 60)

# --- diff_in_means: no controls, no FE (same as Col 1 but tagged as design variant) ---
r = run_g1_ols(
    spec_id="design/randomized_experiment/estimator/diff_in_means",
    spec_tree_path="specification_tree/designs/randomized_experiment.md#difference-in-means",
    controls=[],
    fe_var=None,
    sample_desc="diff in means: no controls, no FE",
)
spec_rows.append(r)
print(f"  diff_in_means: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")

# --- with_covariates: same as baseline (already covered, but listed for completeness) ---
# Skip: identical to baseline, not adding duplicate


# =============================================================================
# G1 RC: CONTROLS LOO (Leave-One-Out)
# =============================================================================

print("\n" + "=" * 60)
print("G1: RC Controls LOO")
print("=" * 60)

for ctrl in BASELINE_CONTROLS:
    remaining = [c for c in BASELINE_CONTROLS if c != ctrl]
    r = run_g1_ols(
        spec_id=f"rc/controls/loo/{ctrl}",
        spec_tree_path="specification_tree/modules/robustness/controls.md#leave-one-out",
        controls=remaining,
        fe_var="vcode",
        sample_desc=f"LOO: dropped {ctrl}",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/loo/{ctrl}", "family": "loo",
                    "dropped": [ctrl], "n_controls": len(remaining)},
    )
    spec_rows.append(r)
    print(f"  loo/{ctrl}: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")


# =============================================================================
# G1 RC: CONTROLS SINGLE ADDITIONS (extended controls)
# =============================================================================

print("\n" + "=" * 60)
print("G1: RC Controls Single Additions")
print("=" * 60)

extended_single_adds = {
    "chet": "chet",
    "shet": "shet",
    "ginil": "ginil",
    "lage": "lage",
    "ledu": "ledu",
    "lclanp": "lclanp",
}

for label, var in extended_single_adds.items():
    ctrls = BASELINE_CONTROLS + [var]
    r = run_g1_ols(
        spec_id=f"rc/controls/single/add_{label}",
        spec_tree_path="specification_tree/modules/robustness/controls.md#single-addition",
        controls=ctrls,
        fe_var="vcode",
        sample_desc=f"single add: + {var}",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/single/add_{label}", "family": "single_addition",
                    "added": [var], "n_controls": len(ctrls)},
    )
    spec_rows.append(r)
    print(f"  add_{label}: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")

# Appendix controls single additions
appendix_single_adds = {
    "turnover": "turnover",
    "leaderskill": "leaderskill",
    "clan1": "clan1",
    "clan2": "clan2",
    "clan3": "clan3",
    "peren": "peren",
    "seas": "seas",
    "slope": "slope",
}

for label, var in appendix_single_adds.items():
    ctrls = BASELINE_CONTROLS + [var]
    r = run_g1_ols(
        spec_id=f"rc/controls/single/add_{label}",
        spec_tree_path="specification_tree/modules/robustness/controls.md#single-addition",
        controls=ctrls,
        fe_var="vcode",
        sample_desc=f"single add (appendix): + {var}",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/single/add_{label}", "family": "single_addition",
                    "added": [var], "n_controls": len(ctrls)},
    )
    spec_rows.append(r)
    print(f"  add_{label}: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")


# =============================================================================
# G1 RC: CONTROLS SETS (predefined)
# =============================================================================

print("\n" + "=" * 60)
print("G1: RC Controls Sets")
print("=" * 60)

# --- no_controls ---
r = run_g1_ols(
    spec_id="rc/controls/sets/no_controls",
    spec_tree_path="specification_tree/modules/robustness/controls.md#predefined-set",
    controls=[],
    fe_var="vcode",
    sample_desc="no group controls, village FE only",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/no_controls", "family": "predefined_set",
                "set_name": "no_controls", "n_controls": 0},
)
spec_rows.append(r)
print(f"  no_controls (w/ FE): coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")

# --- group_only (baseline controls, no FE) ---
r = run_g1_ols(
    spec_id="rc/controls/sets/group_only",
    spec_tree_path="specification_tree/modules/robustness/controls.md#predefined-set",
    controls=BASELINE_CONTROLS,
    fe_var=None,
    sample_desc="group controls only, no village FE",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/group_only", "family": "predefined_set",
                "set_name": "group_only", "n_controls": 8},
)
spec_rows.append(r)
print(f"  group_only: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")

# --- group_plus_leader ---
r = run_g1_ols(
    spec_id="rc/controls/sets/group_plus_leader",
    spec_tree_path="specification_tree/modules/robustness/controls.md#predefined-set",
    controls=BASELINE_CONTROLS + EXTENDED_CONTROLS_LEADER,
    fe_var="vcode",
    sample_desc="group + leader demographics + FE",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/group_plus_leader", "family": "predefined_set",
                "set_name": "group_plus_leader", "n_controls": 11},
)
spec_rows.append(r)
print(f"  group+leader: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")

# --- full_plus_heterogeneity ---
r = run_g1_ols(
    spec_id="rc/controls/sets/full_plus_heterogeneity",
    spec_tree_path="specification_tree/modules/robustness/controls.md#predefined-set",
    controls=ALL_CONTROLS,
    fe_var="vcode",
    sample_desc="all 14 controls + FE (Col 9)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/full_plus_heterogeneity", "family": "predefined_set",
                "set_name": "full_plus_heterogeneity", "n_controls": 14},
)
spec_rows.append(r)
print(f"  full+het: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")


# =============================================================================
# G1 RC: CONTROLS BUILD-UP PROGRESSION
# =============================================================================

print("\n" + "=" * 60)
print("G1: RC Controls Build-Up Progression")
print("=" * 60)

build_up_stages = [
    ("stage1_demo", ["ccs", "ed", "pp", "gs"]),
    ("stage2_econ", ["ccs", "ed", "pp", "gs", "wmk", "time", "fem", "ginic"]),
    ("stage3_leader", ["ccs", "ed", "pp", "gs", "wmk", "time", "fem", "ginic", "lage", "ledu", "lclanp"]),
    ("stage4_full", ALL_CONTROLS),
]

for idx, (stage_label, ctrls) in enumerate(build_up_stages):
    r = run_g1_ols(
        spec_id=f"rc/controls/progression/build_up",
        spec_tree_path="specification_tree/modules/robustness/controls.md#progressive-build-up",
        controls=ctrls,
        fe_var="vcode",
        sample_desc=f"build-up: {stage_label} ({len(ctrls)} controls)",
        axis_block_name="controls",
        axis_block={"spec_id": "rc/controls/progression/build_up",
                    "family": "progressive_build_up",
                    "stage": stage_label, "stage_index": idx + 1,
                    "included": ctrls, "n_controls": len(ctrls)},
    )
    # Make spec_run_id unique by appending stage info to the spec_id in results
    r["spec_id"] = f"rc/controls/progression/build_up"
    spec_rows.append(r)
    print(f"  build_up/{stage_label}: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")


# =============================================================================
# G1 RC: CONTROLS RANDOM SUBSETS
# =============================================================================

print("\n" + "=" * 60)
print("G1: RC Controls Random Subsets")
print("=" * 60)

ALL_POOL = ALL_CONTROLS  # 14 controls
rng = np.random.default_rng(seed=116139)
for i in range(20):
    n_draw = rng.integers(5, 13)  # 5 to 12 controls
    draw = sorted(rng.choice(ALL_POOL, size=n_draw, replace=False).tolist())
    r = run_g1_ols(
        spec_id=f"rc/controls/subset/random_{i+1:02d}",
        spec_tree_path="specification_tree/modules/robustness/controls.md#random-subset",
        controls=draw,
        fe_var="vcode",
        sample_desc=f"random subset {i+1} ({len(draw)} controls)",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/subset/random_{i+1:02d}", "family": "random_subset",
                    "draw_index": i + 1, "included": draw, "n_controls": len(draw)},
    )
    spec_rows.append(r)
    print(f"  random_{i+1:02d}: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")


# =============================================================================
# G1 RC: FIXED EFFECTS
# =============================================================================

print("\n" + "=" * 60)
print("G1: RC Fixed Effects")
print("=" * 60)

# --- drop village FE ---
r = run_g1_ols(
    spec_id="rc/fe/drop_village_fe",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#drop-fe",
    controls=BASELINE_CONTROLS,
    fe_var=None,
    sample_desc="drop village FE",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/drop_village_fe", "dropped": "vcode"},
)
spec_rows.append(r)
print(f"  drop_fe: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")

# --- village FE only (no group controls) ---
r = run_g1_ols(
    spec_id="rc/fe/village_fe_only",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#fe-only",
    controls=[],
    fe_var="vcode",
    sample_desc="village FE only, no group controls",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/village_fe_only", "variant": "fe_only"},
)
spec_rows.append(r)
print(f"  fe_only: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")


# =============================================================================
# G1 RC: TREATMENT FORM
# =============================================================================

print("\n" + "=" * 60)
print("G1: RC Treatment Form")
print("=" * 60)

# --- lcode dummy (single leader dummy instead of 3 type dummies) ---
# Table 6 Col 4: i.lcode replaces leq leqef las
# lcode: 0=LNP, 1=Leq, 2=Leqef, 3=Las. Use dummies for i.lcode.
rid = next_run_id()
try:
    df_t = df_group.copy()
    df_t['lcode_int'] = df_t['lcode'].fillna(-1).astype(int)
    df_t = df_t[df_t['lcode_int'] >= 0].copy()
    rhs_ctrl = " + ".join(BASELINE_CONTROLS)
    # i.lcode: dummies for lcode, base=0
    formula = f"pct ~ C(lcode_int, Treatment(reference=0)) + {rhs_ctrl} | vcode"
    m = pf.feols(formula, data=df_t, vcov="hetero")
    # focal: lcode==1 (Leq type)
    focal_key = [k for k in m.coef().index if "T.1]" in k or "lcode_int" in k and "1" in k]
    if focal_key:
        fk = focal_key[0]
    else:
        fk = m.coef().index[0]
    coefs = {v: float(m.coef()[v]) for v in m.coef().index}
    ci = m.confint()
    payload = make_success_payload(
        coefficients=coefs,
        inference={"spec_id": G1_INFERENCE_CANONICAL["spec_id"], "params": {}, "se_type": "HC1"},
        software=SW_BLOCK, surface_hash=SURFACE_HASH,
        design={"randomized_experiment": G1_DESIGN_AUDIT},
    )
    payload["functional_form"] = {
        "spec_id": "rc/form/treatment/lcode_dummy",
        "interpretation": "Factor dummies for leader classification code (lcode) instead of separate behavioral type dummies",
        "treatment_definition": "C(lcode, Treatment(reference=0))"
    }
    spec_rows.append({
        "paper_id": PAPER_ID, "spec_run_id": rid,
        "spec_id": "rc/form/treatment/lcode_dummy",
        "spec_tree_path": "specification_tree/modules/robustness/functional_form.md#treatment-recoding",
        "baseline_group_id": "G1",
        "outcome_var": "pct", "treatment_var": "lcode",
        "coefficient": float(m.coef()[fk]), "std_error": float(m.se()[fk]),
        "p_value": float(m.pvalue()[fk]),
        "ci_lower": float(ci.loc[fk, ci.columns[0]]),
        "ci_upper": float(ci.loc[fk, ci.columns[1]]),
        "n_obs": int(m._N), "r_squared": float(m._r2),
        "coefficient_vector_json": json.dumps(payload),
        "sample_desc": "lcode dummies (Table 6 Col 4 analog)",
        "fixed_effects": "vcode", "controls_desc": ", ".join(BASELINE_CONTROLS),
        "cluster_var": "", "run_success": 1, "run_error": "",
    })
    print(f"  lcode_dummy: coef={float(m.coef()[fk]):.4f}, p={float(m.pvalue()[fk]):.4f}")
except Exception as e:
    print(f"  lcode_dummy: FAILED - {e}")
    err_str = str(e)[:240]
    payload = make_failure_payload(error=err_str, error_details=error_details_from_exception(e, stage="estimation"),
                                  software=SW_BLOCK, surface_hash=SURFACE_HASH)
    spec_rows.append({
        "paper_id": PAPER_ID, "spec_run_id": rid,
        "spec_id": "rc/form/treatment/lcode_dummy",
        "spec_tree_path": "specification_tree/modules/robustness/functional_form.md#treatment-recoding",
        "baseline_group_id": "G1",
        "outcome_var": "pct", "treatment_var": "lcode",
        "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
        "ci_lower": np.nan, "ci_upper": np.nan,
        "n_obs": np.nan, "r_squared": np.nan,
        "coefficient_vector_json": json.dumps(payload),
        "sample_desc": "", "fixed_effects": "vcode",
        "controls_desc": ", ".join(BASELINE_CONTROLS),
        "cluster_var": "", "run_success": 0, "run_error": err_str,
    })


# --- lpun continuous treatment ---
rid = next_run_id()
try:
    df_t = df_group.copy()
    rhs_ctrl = " + ".join(BASELINE_CONTROLS)
    formula = f"pct ~ lpun + {rhs_ctrl} | vcode"
    m = pf.feols(formula, data=df_t, vcov="hetero")
    coefs = {v: float(m.coef()[v]) for v in m.coef().index}
    ci = m.confint()
    payload = make_success_payload(
        coefficients=coefs,
        inference={"spec_id": G1_INFERENCE_CANONICAL["spec_id"], "params": {}, "se_type": "HC1"},
        software=SW_BLOCK, surface_hash=SURFACE_HASH,
        design={"randomized_experiment": G1_DESIGN_AUDIT},
    )
    payload["functional_form"] = {
        "spec_id": "rc/form/treatment/lpun_continuous",
        "interpretation": "Leader punishment behavior (continuous) replaces categorical leader type dummies",
        "treatment_definition": "lpun (continuous punishment score)"
    }
    spec_rows.append({
        "paper_id": PAPER_ID, "spec_run_id": rid,
        "spec_id": "rc/form/treatment/lpun_continuous",
        "spec_tree_path": "specification_tree/modules/robustness/functional_form.md#treatment-recoding",
        "baseline_group_id": "G1",
        "outcome_var": "pct", "treatment_var": "lpun",
        "coefficient": float(m.coef()["lpun"]), "std_error": float(m.se()["lpun"]),
        "p_value": float(m.pvalue()["lpun"]),
        "ci_lower": float(ci.loc["lpun", ci.columns[0]]),
        "ci_upper": float(ci.loc["lpun", ci.columns[1]]),
        "n_obs": int(m._N), "r_squared": float(m._r2),
        "coefficient_vector_json": json.dumps(payload),
        "sample_desc": "continuous lpun treatment (Table A11 analog)",
        "fixed_effects": "vcode", "controls_desc": ", ".join(BASELINE_CONTROLS),
        "cluster_var": "", "run_success": 1, "run_error": "",
    })
    print(f"  lpun_continuous: coef={float(m.coef()['lpun']):.4f}, p={float(m.pvalue()['lpun']):.4f}")
except Exception as e:
    print(f"  lpun_continuous: FAILED - {e}")
    err_str = str(e)[:240]
    payload = make_failure_payload(error=err_str, error_details=error_details_from_exception(e, stage="estimation"),
                                  software=SW_BLOCK, surface_hash=SURFACE_HASH)
    spec_rows.append({
        "paper_id": PAPER_ID, "spec_run_id": rid,
        "spec_id": "rc/form/treatment/lpun_continuous",
        "spec_tree_path": "specification_tree/modules/robustness/functional_form.md#treatment-recoding",
        "baseline_group_id": "G1",
        "outcome_var": "pct", "treatment_var": "lpun",
        "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
        "ci_lower": np.nan, "ci_upper": np.nan,
        "n_obs": np.nan, "r_squared": np.nan,
        "coefficient_vector_json": json.dumps(payload),
        "sample_desc": "", "fixed_effects": "vcode",
        "controls_desc": ", ".join(BASELINE_CONTROLS),
        "cluster_var": "", "run_success": 0, "run_error": err_str,
    })


# =============================================================================
# G1 RC: OUTCOME FORM
# =============================================================================

print("\n" + "=" * 60)
print("G1: RC Outcome Form")
print("=" * 60)

# --- pct2 as outcome ---
r = run_g1_ols(
    spec_id="rc/form/outcome/pct2",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#outcome-recoding",
    controls=BASELINE_CONTROLS,
    fe_var="vcode",
    outcome_var="pct2",
    sample_desc="alternative outcome pct2 (forest condition measure 2)",
    func_form={
        "spec_id": "rc/form/outcome/pct2",
        "interpretation": "Alternative forest condition percentage measure (pct2)",
        "outcome_definition": "pct2"
    },
)
spec_rows.append(r)
print(f"  pct2: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")


# =============================================================================
# G1 RC: SAMPLE RESTRICTIONS
# =============================================================================

print("\n" + "=" * 60)
print("G1: RC Sample Restrictions")
print("=" * 60)

# --- Drop 2 influential observations (inf_lead == 1) ---
r = run_g1_ols(
    spec_id="rc/sample/outliers/drop_influential_2",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outlier-exclusion",
    controls=BASELINE_CONTROLS,
    fe_var="vcode",
    sample_filter=lambda df: df['inf_lead'] == 0,
    sample_desc="drop 2 DFITS-influential obs (Table 6 Col 6)",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/drop_influential_2",
                "dropped_criterion": "DFITS > 2*sqrt((k+1)/N)", "n_dropped": 2},
)
spec_rows.append(r)
print(f"  drop_inf_2: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")

# --- Drop 4 influential observations (inf_lead2 == 1) ---
r = run_g1_ols(
    spec_id="rc/sample/outliers/drop_influential_4",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outlier-exclusion",
    controls=BASELINE_CONTROLS,
    fe_var="vcode",
    sample_filter=lambda df: df['inf_lead2'] == 0,
    sample_desc="drop 4 DFITS-influential obs (Table 6 Col 7)",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/drop_influential_4",
                "dropped_criterion": "DFITS > threshold", "n_dropped": 4},
)
spec_rows.append(r)
print(f"  drop_inf_4: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")

# --- Drop vice leaders (vlcode==0 only) ---
r = run_g1_ols(
    spec_id="rc/sample/subset/drop_vice_leaders",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subsample",
    controls=BASELINE_CONTROLS,
    fe_var="vcode",
    sample_filter=lambda df: df['vlcode'] == 0,
    sample_desc="vlcode==0: drop vice leaders (Table 6 Col 5)",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subset/drop_vice_leaders",
                "restriction": "vlcode == 0", "description": "Exclude vice-leaders"},
)
spec_rows.append(r)
print(f"  drop_vice: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")

# --- Drop LNP (non-punishment) leaders: lcode > 0 ---
r = run_g1_ols(
    spec_id="rc/sample/subset/drop_lnp_leaders",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subsample",
    controls=BASELINE_CONTROLS,
    fe_var="vcode",
    sample_filter=lambda df: df['lcode'] > 0,
    sample_desc="lcode > 0: exclude LNP (non-punishment) leaders",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subset/drop_lnp_leaders",
                "restriction": "lcode > 0", "description": "Exclude non-punishment type leaders"},
)
spec_rows.append(r)
print(f"  drop_lnp: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")


# =============================================================================
# G1 RC: JOINT (controls x sample)
# =============================================================================

print("\n" + "=" * 60)
print("G1: RC Joint (controls x sample)")
print("=" * 60)

joint_combos = [
    ("drop_inf2__full_controls", ALL_CONTROLS, lambda df: df['inf_lead'] == 0,
     "full controls, drop 2 influential"),
    ("drop_inf4__full_controls", ALL_CONTROLS, lambda df: df['inf_lead2'] == 0,
     "full controls, drop 4 influential"),
    ("drop_vice__full_controls", ALL_CONTROLS, lambda df: df['vlcode'] == 0,
     "full controls, drop vice leaders"),
    ("drop_lnp__no_controls", [], lambda df: df['lcode'] > 0,
     "no controls, drop LNP leaders"),
]

for label, ctrls, filt, desc in joint_combos:
    r = run_g1_ols(
        spec_id=f"rc/joint/controls_sample/{label}",
        spec_tree_path="specification_tree/modules/robustness/joint.md#controls-x-sample",
        controls=ctrls,
        fe_var="vcode",
        sample_filter=filt,
        sample_desc=desc,
        axis_block_name="joint",
        axis_block={"spec_id": f"rc/joint/controls_sample/{label}",
                    "axes": ["controls", "sample"], "description": desc},
    )
    spec_rows.append(r)
    print(f"  joint/{label}: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")


# =============================================================================
# G2 BASELINE SPECIFICATION
# =============================================================================

print("\n" + "=" * 60)
print("G2: Baseline specification (Poisson)")
print("=" * 60)

r = run_g2_poisson(
    spec_id="baseline",
    spec_tree_path="specification_tree/designs/randomized_experiment.md#poisson",
    sample_desc="full sample, N=510",
    notes="Table 3: poisson pi i.cd1, vce(cluster fcode)"
)
spec_rows.append(r)
print(f"  baseline: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")


# =============================================================================
# G2 DESIGN VARIANTS
# =============================================================================

print("\n" + "=" * 60)
print("G2: Design variants")
print("=" * 60)

# --- Diff in means: OLS with session-clustered SE (canonical inference) ---
r = run_g2_poisson(
    spec_id="design/randomized_experiment/estimator/diff_in_means",
    spec_tree_path="specification_tree/designs/randomized_experiment.md#difference-in-means",
    model_type="ols",
    cluster_var="fcode",
    sample_desc="OLS difference in means (session-clustered SE)",
    notes="OLS with condition dummies, clustered SE at session level"
)
spec_rows.append(r)
print(f"  diff_in_means: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")


# =============================================================================
# G2 RC: MODEL FORM
# =============================================================================

print("\n" + "=" * 60)
print("G2: RC Model Form")
print("=" * 60)

# --- Negative binomial ---
r = run_g2_poisson(
    spec_id="rc/form/model/nbreg",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#model-specification",
    model_type="nbreg",
    sample_desc="negative binomial (overdispersion check, fn 21)",
    func_form={
        "spec_id": "rc/form/model/nbreg",
        "interpretation": "Negative binomial model to check for overdispersion relative to Poisson",
        "model_type": "negative_binomial"
    },
)
spec_rows.append(r)
print(f"  nbreg: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")

# --- OLS/LPM ---
r = run_g2_poisson(
    spec_id="rc/form/model/ols_lpm",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#model-specification",
    model_type="ols",
    sample_desc="OLS/LPM (linear model for count outcome)",
    func_form={
        "spec_id": "rc/form/model/ols_lpm",
        "interpretation": "OLS linear model for punishment count (vs Poisson baseline)",
        "model_type": "ols"
    },
)
spec_rows.append(r)
print(f"  ols_lpm: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")


# =============================================================================
# G2 RC: CONTROLS
# =============================================================================

print("\n" + "=" * 60)
print("G2: RC Controls")
print("=" * 60)

# --- Single additions ---
for var in ["lage", "ledu", "lclan"]:
    r = run_g2_poisson(
        spec_id=f"rc/controls/single/add_{var}",
        spec_tree_path="specification_tree/modules/robustness/controls.md#single-addition",
        controls=[var],
        sample_desc=f"+ {var}",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/single/add_{var}", "family": "single_addition",
                    "added": [var], "n_controls": 1},
    )
    spec_rows.append(r)
    print(f"  add_{var}: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")

# --- All leader demographics together ---
r = run_g2_poisson(
    spec_id="rc/controls/sets/leader_demographics",
    spec_tree_path="specification_tree/modules/robustness/controls.md#predefined-set",
    controls=["lage", "ledu", "lclan"],
    sample_desc="+ lage ledu lclan (fn 22)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/leader_demographics", "family": "predefined_set",
                "set_name": "leader_demographics", "n_controls": 3},
)
spec_rows.append(r)
print(f"  leader_demographics: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")


# =============================================================================
# G2 RC: OUTCOME
# =============================================================================

print("\n" + "=" * 60)
print("G2: RC Outcome")
print("=" * 60)

# --- pj as outcome ---
r = run_g2_poisson(
    spec_id="rc/form/outcome/pj",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#outcome-recoding",
    outcome_var="pj",
    sample_desc="alternative punishment outcome pj (Table A2)",
    func_form={
        "spec_id": "rc/form/outcome/pj",
        "interpretation": "Alternative punishment measure (pj instead of pi)",
        "outcome_definition": "pj"
    },
)
spec_rows.append(r)
print(f"  pj: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")


# =============================================================================
# G2 RC: SAMPLE SUBSETS (by leader type)
# =============================================================================

print("\n" + "=" * 60)
print("G2: RC Sample Subsets (by leader type)")
print("=" * 60)

leader_type_subsets = [
    ("lcode_0_only", lambda df: df['lcode'] == 0, "Leqef leaders only (lcode==0)"),
    ("lcode_1_only", lambda df: df['lcode'] == 1, "Leq leaders only (lcode==1)"),
    ("lcode_3_only", lambda df: df['lcode'] == 3, "Las leaders only (lcode==3)"),
]

for label, filt, desc in leader_type_subsets:
    r = run_g2_poisson(
        spec_id=f"rc/sample/subset/{label}",
        spec_tree_path="specification_tree/modules/robustness/sample.md#subsample",
        sample_filter=filt,
        sample_desc=desc,
        axis_block_name="sample",
        axis_block={"spec_id": f"rc/sample/subset/{label}",
                    "restriction": label, "description": desc},
    )
    spec_rows.append(r)
    if r['run_success']:
        print(f"  {label}: coef={r['coefficient']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")
    else:
        print(f"  {label}: FAILED - {r['run_error'][:80]}")


# =============================================================================
# INFERENCE VARIANTS (G1 and G2)
# =============================================================================

print("\n" + "=" * 60)
print("Inference variants")
print("=" * 60)

# --- G1 inference variants on baseline spec ---
# Find baseline spec_run_id
baseline_g1_row = [r for r in spec_rows if r['spec_id'] == 'baseline' and r['baseline_group_id'] == 'G1'][0]
baseline_g1_rid = baseline_g1_row['spec_run_id']

infer_counter = 0


def next_infer_id():
    global infer_counter
    infer_counter += 1
    return f"{PAPER_ID}__infer_{infer_counter:04d}"


# --- G1: Village-clustered SE ---
rid = next_infer_id()
try:
    formula = f"pct ~ leq + leqef + las + {' + '.join(BASELINE_CONTROLS)} | vcode"
    m = pf.feols(formula, data=df_group, vcov={"CRV1": "vcode"})
    coefs = {v: float(m.coef()[v]) for v in m.coef().index}
    ci = m.confint()
    payload = make_success_payload(
        coefficients=coefs,
        inference={"spec_id": "infer/se/cluster/village", "params": {"cluster_var": "vcode"},
                   "se_type": "CRV1", "n_clusters": 5},
        software=SW_BLOCK, surface_hash=SURFACE_HASH,
        design={"randomized_experiment": G1_DESIGN_AUDIT},
    )
    infer_rows.append({
        "paper_id": PAPER_ID, "inference_run_id": rid,
        "spec_run_id": baseline_g1_rid,
        "spec_id": "infer/se/cluster/village",
        "spec_tree_path": "specification_tree/modules/inference/standard_errors.md#cluster",
        "baseline_group_id": "G1",
        "coefficient": float(m.coef()["leq"]), "std_error": float(m.se()["leq"]),
        "p_value": float(m.pvalue()["leq"]),
        "ci_lower": float(ci.loc["leq", ci.columns[0]]),
        "ci_upper": float(ci.loc["leq", ci.columns[1]]),
        "n_obs": int(m._N), "r_squared": float(m._r2),
        "coefficient_vector_json": json.dumps(payload),
        "run_success": 1, "run_error": "",
    })
    print(f"  G1 cluster/village: SE={float(m.se()['leq']):.4f}, p={float(m.pvalue()['leq']):.4f}")
except Exception as e:
    print(f"  G1 cluster/village: FAILED - {e}")
    payload = make_failure_payload(error=str(e)[:240], error_details=error_details_from_exception(e, stage="inference"),
                                  software=SW_BLOCK, surface_hash=SURFACE_HASH)
    infer_rows.append({
        "paper_id": PAPER_ID, "inference_run_id": rid,
        "spec_run_id": baseline_g1_rid,
        "spec_id": "infer/se/cluster/village",
        "spec_tree_path": "specification_tree/modules/inference/standard_errors.md#cluster",
        "baseline_group_id": "G1",
        "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
        "ci_lower": np.nan, "ci_upper": np.nan,
        "n_obs": np.nan, "r_squared": np.nan,
        "coefficient_vector_json": json.dumps(payload),
        "run_success": 0, "run_error": str(e)[:240],
    })

# --- G1: HC2 ---
# HC2/HC3 not supported with absorbed FE in pyfixest; use explicit village dummies
rid = next_infer_id()
try:
    formula = f"pct ~ leq + leqef + las + {' + '.join(BASELINE_CONTROLS)} + C(vcode)"
    m = pf.feols(formula, data=df_group, vcov="HC2")
    coefs = {v: float(m.coef()[v]) for v in m.coef().index}
    ci = m.confint()
    payload = make_success_payload(
        coefficients=coefs,
        inference={"spec_id": "infer/se/hc/hc2", "params": {}, "se_type": "HC2"},
        software=SW_BLOCK, surface_hash=SURFACE_HASH,
        design={"randomized_experiment": G1_DESIGN_AUDIT},
    )
    infer_rows.append({
        "paper_id": PAPER_ID, "inference_run_id": rid,
        "spec_run_id": baseline_g1_rid,
        "spec_id": "infer/se/hc/hc2",
        "spec_tree_path": "specification_tree/modules/inference/standard_errors.md#hc2",
        "baseline_group_id": "G1",
        "coefficient": float(m.coef()["leq"]), "std_error": float(m.se()["leq"]),
        "p_value": float(m.pvalue()["leq"]),
        "ci_lower": float(ci.loc["leq", ci.columns[0]]),
        "ci_upper": float(ci.loc["leq", ci.columns[1]]),
        "n_obs": int(m._N), "r_squared": float(m._r2),
        "coefficient_vector_json": json.dumps(payload),
        "run_success": 1, "run_error": "",
    })
    print(f"  G1 HC2: SE={float(m.se()['leq']):.4f}, p={float(m.pvalue()['leq']):.4f}")
except Exception as e:
    print(f"  G1 HC2: FAILED - {e}")
    payload = make_failure_payload(error=str(e)[:240], error_details=error_details_from_exception(e, stage="inference"),
                                  software=SW_BLOCK, surface_hash=SURFACE_HASH)
    infer_rows.append({
        "paper_id": PAPER_ID, "inference_run_id": rid,
        "spec_run_id": baseline_g1_rid,
        "spec_id": "infer/se/hc/hc2",
        "spec_tree_path": "specification_tree/modules/inference/standard_errors.md#hc2",
        "baseline_group_id": "G1",
        "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
        "ci_lower": np.nan, "ci_upper": np.nan,
        "n_obs": np.nan, "r_squared": np.nan,
        "coefficient_vector_json": json.dumps(payload),
        "run_success": 0, "run_error": str(e)[:240],
    })

# --- G1: HC3 ---
rid = next_infer_id()
try:
    formula = f"pct ~ leq + leqef + las + {' + '.join(BASELINE_CONTROLS)} + C(vcode)"
    m = pf.feols(formula, data=df_group, vcov="HC3")
    coefs = {v: float(m.coef()[v]) for v in m.coef().index}
    ci = m.confint()
    payload = make_success_payload(
        coefficients=coefs,
        inference={"spec_id": "infer/se/hc/hc3", "params": {}, "se_type": "HC3"},
        software=SW_BLOCK, surface_hash=SURFACE_HASH,
        design={"randomized_experiment": G1_DESIGN_AUDIT},
    )
    infer_rows.append({
        "paper_id": PAPER_ID, "inference_run_id": rid,
        "spec_run_id": baseline_g1_rid,
        "spec_id": "infer/se/hc/hc3",
        "spec_tree_path": "specification_tree/modules/inference/standard_errors.md#hc3",
        "baseline_group_id": "G1",
        "coefficient": float(m.coef()["leq"]), "std_error": float(m.se()["leq"]),
        "p_value": float(m.pvalue()["leq"]),
        "ci_lower": float(ci.loc["leq", ci.columns[0]]),
        "ci_upper": float(ci.loc["leq", ci.columns[1]]),
        "n_obs": int(m._N), "r_squared": float(m._r2),
        "coefficient_vector_json": json.dumps(payload),
        "run_success": 1, "run_error": "",
    })
    print(f"  G1 HC3: SE={float(m.se()['leq']):.4f}, p={float(m.pvalue()['leq']):.4f}")
except Exception as e:
    print(f"  G1 HC3: FAILED - {e}")
    payload = make_failure_payload(error=str(e)[:240], error_details=error_details_from_exception(e, stage="inference"),
                                  software=SW_BLOCK, surface_hash=SURFACE_HASH)
    infer_rows.append({
        "paper_id": PAPER_ID, "inference_run_id": rid,
        "spec_run_id": baseline_g1_rid,
        "spec_id": "infer/se/hc/hc3",
        "spec_tree_path": "specification_tree/modules/inference/standard_errors.md#hc3",
        "baseline_group_id": "G1",
        "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
        "ci_lower": np.nan, "ci_upper": np.nan,
        "n_obs": np.nan, "r_squared": np.nan,
        "coefficient_vector_json": json.dumps(payload),
        "run_success": 0, "run_error": str(e)[:240],
    })

# --- G2: Robust SE (no clustering) ---
baseline_g2_row = [r for r in spec_rows if r['spec_id'] == 'baseline' and r['baseline_group_id'] == 'G2'][0]
baseline_g2_rid = baseline_g2_row['spec_run_id']

rid = next_infer_id()
try:
    formula = "pi ~ C(cd1, Treatment(reference=1))"
    m = smf.poisson(formula, data=df_pun).fit(cov_type='HC1', disp=0)
    focal_key = "C(cd1, Treatment(reference=1))[T.5]"
    coefs = {v: float(m.params[v]) for v in m.params.index}
    ci = m.conf_int()
    payload = make_success_payload(
        coefficients=coefs,
        inference={"spec_id": "infer/se/hc/robust", "params": {}, "se_type": "HC1"},
        software=SW_BLOCK, surface_hash=SURFACE_HASH,
        design={"randomized_experiment": G2_DESIGN_AUDIT},
    )
    infer_rows.append({
        "paper_id": PAPER_ID, "inference_run_id": rid,
        "spec_run_id": baseline_g2_rid,
        "spec_id": "infer/se/hc/robust",
        "spec_tree_path": "specification_tree/modules/inference/standard_errors.md#robust",
        "baseline_group_id": "G2",
        "coefficient": float(m.params[focal_key]), "std_error": float(m.bse[focal_key]),
        "p_value": float(m.pvalues[focal_key]),
        "ci_lower": float(ci.loc[focal_key, 0]),
        "ci_upper": float(ci.loc[focal_key, 1]),
        "n_obs": int(m.nobs), "r_squared": float(m.prsquared) if hasattr(m, 'prsquared') else np.nan,
        "coefficient_vector_json": json.dumps(payload),
        "run_success": 1, "run_error": "",
    })
    print(f"  G2 robust: SE={float(m.bse[focal_key]):.4f}, p={float(m.pvalues[focal_key]):.4f}")
except Exception as e:
    print(f"  G2 robust: FAILED - {e}")
    payload = make_failure_payload(error=str(e)[:240], error_details=error_details_from_exception(e, stage="inference"),
                                  software=SW_BLOCK, surface_hash=SURFACE_HASH)
    infer_rows.append({
        "paper_id": PAPER_ID, "inference_run_id": rid,
        "spec_run_id": baseline_g2_rid,
        "spec_id": "infer/se/hc/robust",
        "spec_tree_path": "specification_tree/modules/inference/standard_errors.md#robust",
        "baseline_group_id": "G2",
        "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
        "ci_lower": np.nan, "ci_upper": np.nan,
        "n_obs": np.nan, "r_squared": np.nan,
        "coefficient_vector_json": json.dumps(payload),
        "run_success": 0, "run_error": str(e)[:240],
    })


# =============================================================================
# WRITE OUTPUTS
# =============================================================================

print("\n" + "=" * 60)
print("Writing outputs")
print("=" * 60)

# --- specification_results.csv ---
spec_df = pd.DataFrame(spec_rows)
spec_df.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)
print(f"  specification_results.csv: {len(spec_df)} rows")
print(f"    G1 rows: {(spec_df['baseline_group_id'] == 'G1').sum()}")
print(f"    G2 rows: {(spec_df['baseline_group_id'] == 'G2').sum()}")
print(f"    Successful: {spec_df['run_success'].sum()}")
print(f"    Failed: {(spec_df['run_success'] == 0).sum()}")

# --- inference_results.csv ---
# Add missing required columns based on baseline_group_id
for row in infer_rows:
    if row["baseline_group_id"] == "G1":
        row["outcome_var"] = "pct"
        row["treatment_var"] = "leq"
        row["cluster_var"] = row.get("cluster_var", "")
    elif row["baseline_group_id"] == "G2":
        row["outcome_var"] = "pi"
        row["treatment_var"] = "cd1"
        row["cluster_var"] = row.get("cluster_var", "fcode")
infer_df = pd.DataFrame(infer_rows)
infer_df.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)
print(f"  inference_results.csv: {len(infer_df)} rows")
print(f"    Successful: {infer_df['run_success'].sum()}")

# --- Verify unique spec_run_ids ---
assert spec_df['spec_run_id'].nunique() == len(spec_df), "Duplicate spec_run_ids!"
print("  All spec_run_ids unique: OK")

# --- SPECIFICATION_SEARCH.md ---
n_g1 = (spec_df['baseline_group_id'] == 'G1').sum()
n_g2 = (spec_df['baseline_group_id'] == 'G2').sum()
n_success = int(spec_df['run_success'].sum())
n_fail = int((spec_df['run_success'] == 0).sum())

# Compute category counts
baseline_count = spec_df[spec_df['spec_id'].str.startswith('baseline')].shape[0]
design_count = spec_df[spec_df['spec_id'].str.startswith('design/')].shape[0]
rc_count = spec_df[spec_df['spec_id'].str.startswith('rc/')].shape[0]

md = f"""# Specification Search: {PAPER_ID}

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
| Total specifications | {len(spec_df)} |
| G1 (Cooperation) | {n_g1} |
| G2 (Punishment) | {n_g2} |
| Baselines | {baseline_count} |
| Design variants | {design_count} |
| RC variants | {rc_count} |
| Successful | {n_success} |
| Failed | {n_fail} |
| Inference variants | {len(infer_df)} |

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
- Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}
- pyfixest (OLS with FE and robust SE)
- statsmodels (Poisson, Negative Binomial)
- pandas, numpy
"""

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write(md)
print(f"  SPECIFICATION_SEARCH.md: written")

# Print final summary
print("\n" + "=" * 60)
print("DONE")
print(f"Total specs: {len(spec_df)}, Inference: {len(infer_df)}")
print(f"Success rate: {n_success}/{len(spec_df)} ({100*n_success/len(spec_df):.1f}%)")
print("=" * 60)
