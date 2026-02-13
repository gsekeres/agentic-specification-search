"""
Specification Search Script for Ferraz & Finan (2011)
"Electoral Accountability and Corruption: Evidence from the Audits of Local Governments"
American Economic Review, 101(4), 1274-1311.

Paper ID: 112431-V1

Executes the approved SPECIFICATION_SURFACE.json:
  - G1: pcorrupt ~ first  (Table 4 Col 6 baseline)
  - G2: ncorrupt ~ first  (Table 5A Col 2 baseline)
  - G3: ncorrupt_os ~ first (Table 5B Col 2 baseline)

Outputs:
  - specification_results.csv
  - diagnostics_results.csv
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json
import os
import itertools
import warnings
warnings.filterwarnings('ignore')

from scipy.optimize import minimize, approx_fprime
from scipy.stats import norm, t as t_dist

# =============================================================================
# Configuration
# =============================================================================
PACKAGE_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/112431-V1"
PAPER_ID = "112431-V1"
SEED = 112431

# =============================================================================
# Load data
# =============================================================================
df = pd.read_stata(os.path.join(PACKAGE_DIR, "corruptiondata_aer.dta"))
dfs = df[df['esample2'] == 1].copy()

print(f"Loaded data: {len(df)} total rows, {len(dfs)} in estimation sample (esample2==1)")

# =============================================================================
# Variable group definitions (matching Stata globals / surface control blocks)
# =============================================================================
prefchar2_continuous = ["pref_masc", "pref_idade_tse", "pref_escola"]
prefchar2_party = ["party_d1", "party_d3", "party_d4", "party_d5", "party_d6",
                   "party_d7", "party_d8", "party_d9", "party_d10", "party_d11",
                   "party_d12", "party_d13", "party_d14", "party_d15", "party_d16",
                   "party_d17", "party_d18"]
munichar2 = ["lpop", "purb", "p_secundario", "mun_novo", "lpib02", "gini_ipea"]
fiscal = ["lrec_trans"]
political = ["p_cad_pref", "vereador_eleit", "ENLP2000", "comarca"]
sorteio = [f"sorteio{i}" for i in range(1, 11)]
audit_scale = ["lfunc_ativ", "lrec_fisc"]

# Block definitions for combinatorial enumeration
CONTROL_BLOCKS = {
    "prefchar2_continuous": prefchar2_continuous,
    "prefchar2_party": prefchar2_party,
    "munichar2": munichar2,
    "fiscal": fiscal,
    "political": political,
    "sorteio": sorteio,
}

BLOCK_NAMES = list(CONTROL_BLOCKS.keys())  # order matters for enumeration

uf_dummies = [c for c in df.columns if c.startswith("uf_d")]

# G1 baseline controls (all 6 blocks, no audit_scale)
G1_BASELINE_CONTROLS = (prefchar2_continuous + prefchar2_party + munichar2 +
                        fiscal + political + sorteio)

# G2 baseline controls (mandatory: lrec_fisc + all 6 blocks)
G2_BASELINE_CONTROLS = (["lrec_fisc"] + prefchar2_continuous + prefchar2_party +
                        munichar2 + fiscal + political + sorteio)

# G3 baseline controls (mandatory: lrec_fisc + lfunc_ativ + all 6 blocks)
G3_BASELINE_CONTROLS = (["lrec_fisc", "lfunc_ativ"] + prefchar2_continuous +
                        prefchar2_party + munichar2 + fiscal + political + sorteio)

# =============================================================================
# Prepare running variable for RDD specs
# =============================================================================
dfs['wm'] = np.nan
dfs.loc[dfs['reeleito'] == 1, 'wm'] = dfs.loc[dfs['reeleito'] == 1, 'winmargin2000']
dfs.loc[dfs['incumbent'] == 1, 'wm'] = dfs.loc[dfs['incumbent'] == 1, 'winmargin2000_inclost']
dfs['running'] = dfs['wm'].copy()
dfs.loc[dfs['incumbent'] == 1, 'running'] = -dfs.loc[dfs['incumbent'] == 1, 'wm']
dfs['running2'] = dfs['running'] ** 2
dfs['running3'] = dfs['running'] ** 3

# Experience variables for add-experience specs
dfs['nexp'] = (dfs['exp_prefeito'].fillna(0) + dfs['vereador9600'].fillna(0)) * 4
dfs['nexp2'] = dfs['nexp'] ** 2

# =============================================================================
# Helper functions
# =============================================================================
results = []
run_counter = 0


def fml(y, rhs, absorb=None):
    """Build pyfixest formula."""
    rhs_str = " + ".join(rhs)
    if absorb:
        return f"{y} ~ {rhs_str} | {absorb}"
    else:
        return f"{y} ~ {rhs_str}"


def run_ols(outcome, treatment, controls, data, fe=None, vcov_type="hetero",
            cluster_var=None, spec_id=None, spec_tree_path=None,
            baseline_group_id=None, controls_desc="", sample_desc="esample2==1",
            fe_desc="", extra_json=None):
    """Run OLS regression and return result dict."""
    global run_counter
    run_counter += 1
    spec_run_id = f"{PAPER_ID}_run{run_counter:04d}"

    rhs = [treatment] + controls
    if cluster_var:
        vcov = {"CRV1": cluster_var}
    else:
        vcov = vcov_type

    try:
        formula = fml(outcome, rhs, absorb=fe)
        m = pf.feols(formula, data=data, vcov=vcov)

        coef = float(m.coef()[treatment])
        se = float(m.se()[treatment])
        pval = float(m.pvalue()[treatment])
        nobs = int(m._N)
        r2 = float(m._r2)

        # Fallback to statsmodels if pyfixest returns NaN SE (happens with HC2 + many dummies)
        if np.isnan(se) or np.isnan(pval):
            all_vars = [outcome] + rhs
            clean = data[all_vars].dropna()
            y = clean[outcome]
            X = sm.add_constant(clean[rhs])
            m_sm = sm.OLS(y, X).fit(cov_type=vcov_type if isinstance(vcov_type, str) and vcov_type != "hetero" else "HC2")
            coef = float(m_sm.params[treatment])
            se = float(m_sm.bse[treatment])
            pval = float(m_sm.pvalues[treatment])
            nobs = int(m_sm.nobs)
            r2 = float(m_sm.rsquared)

        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

        coef_dict = {k: float(v) for k, v in m.coef().items()}
        if extra_json:
            coef_dict.update(extra_json)

        row = {
            "paper_id": PAPER_ID,
            "spec_run_id": spec_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome,
            "treatment_var": treatment,
            "coefficient": round(coef, 8),
            "std_error": round(se, 8),
            "p_value": round(pval, 8),
            "ci_lower": round(ci_lower, 8),
            "ci_upper": round(ci_upper, 8),
            "n_obs": nobs,
            "r_squared": round(r2, 6),
            "coefficient_vector_json": json.dumps(coef_dict),
            "sample_desc": sample_desc,
            "fixed_effects": fe_desc if fe_desc else (fe if fe else ""),
            "controls_desc": controls_desc,
            "cluster_var": cluster_var if cluster_var else "",
        }
        results.append(row)
        print(f"  {spec_run_id}: {spec_id} | coef={coef:.6f} se={se:.6f} p={pval:.4f} N={nobs}")
        return row

    except Exception as e:
        row = {
            "paper_id": PAPER_ID,
            "spec_run_id": spec_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome,
            "treatment_var": treatment,
            "coefficient": "",
            "std_error": "",
            "p_value": "",
            "ci_lower": "",
            "ci_upper": "",
            "n_obs": "",
            "r_squared": "",
            "coefficient_vector_json": json.dumps({"error": str(e)}),
            "sample_desc": sample_desc,
            "fixed_effects": fe_desc if fe_desc else (fe if fe else ""),
            "controls_desc": controls_desc,
            "cluster_var": cluster_var if cluster_var else "",
        }
        results.append(row)
        print(f"  {spec_run_id}: {spec_id} | FAILED: {e}")
        return row


def run_tobit(outcome, treatment, controls, data, fe_dummies=None, lower=0,
              spec_id=None, spec_tree_path=None, baseline_group_id=None,
              controls_desc="", sample_desc="esample2==1", fe_desc=""):
    """Run tobit regression via manual MLE and return result dict."""
    global run_counter
    run_counter += 1
    spec_run_id = f"{PAPER_ID}_run{run_counter:04d}"

    try:
        rhs_vars = [treatment] + controls
        if fe_dummies:
            rhs_vars = rhs_vars + fe_dummies
        all_vars = [outcome] + rhs_vars
        data_clean = data[all_vars].dropna()
        y = data_clean[outcome].values.astype(float)
        X = sm.add_constant(data_clean[rhs_vars])
        col_names = list(X.columns)
        X_arr = X.values.astype(float)
        n, k = X_arr.shape

        # OLS starting values
        ols_res = np.linalg.lstsq(X_arr, y, rcond=None)
        beta_init = ols_res[0]
        resid = y - X_arr @ beta_init
        sigma_init = max(np.std(resid), 0.01)
        init = np.concatenate([beta_init, [np.log(sigma_init)]])

        def negll(params):
            beta = params[:-1]
            log_sigma = params[-1]
            sigma = np.exp(log_sigma)
            xb = X_arr @ beta
            censored = (y <= lower)
            ll = 0.0
            if censored.any():
                ll += np.sum(norm.logcdf((lower - xb[censored]) / sigma))
            if (~censored).any():
                ll += np.sum(-0.5 * np.log(2 * np.pi) - log_sigma
                             - 0.5 * ((y[~censored] - xb[~censored]) / sigma) ** 2)
            return -ll

        res = minimize(negll, init, method='BFGS', options={'maxiter': 10000, 'gtol': 1e-8})

        # Numerical Hessian for SEs
        eps = 1e-5
        k_total = len(res.x)
        H = np.zeros((k_total, k_total))
        for i in range(k_total):
            def grad_i(p, idx=i):
                return approx_fprime(p, negll, eps)[idx]
            H[i] = approx_fprime(res.x, grad_i, eps)
        try:
            se_all = np.sqrt(np.diag(np.linalg.inv(H)))
        except:
            se_all = np.full(k_total, np.nan)

        beta = res.x[:-1]
        se = se_all[:-1]

        treat_idx = col_names.index(treatment)
        coef = float(beta[treat_idx])
        se_val = float(se[treat_idx])
        pval = float(2 * (1 - norm.cdf(abs(coef / se_val)))) if se_val > 0 else np.nan
        ci_lower = coef - 1.96 * se_val
        ci_upper = coef + 1.96 * se_val

        coef_dict = {col_names[i]: float(beta[i]) for i in range(len(col_names))}

        row = {
            "paper_id": PAPER_ID,
            "spec_run_id": spec_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome,
            "treatment_var": treatment,
            "coefficient": round(coef, 8),
            "std_error": round(se_val, 8),
            "p_value": round(pval, 8) if not np.isnan(pval) else "",
            "ci_lower": round(ci_lower, 8),
            "ci_upper": round(ci_upper, 8),
            "n_obs": n,
            "r_squared": "",
            "coefficient_vector_json": json.dumps(coef_dict),
            "sample_desc": sample_desc,
            "fixed_effects": fe_desc,
            "controls_desc": controls_desc,
            "cluster_var": "",
        }
        results.append(row)
        print(f"  {spec_run_id}: {spec_id} | coef={coef:.6f} se={se_val:.6f} p={pval:.4f} N={n}")
        return row

    except Exception as e:
        row = {
            "paper_id": PAPER_ID,
            "spec_run_id": spec_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome,
            "treatment_var": treatment,
            "coefficient": "",
            "std_error": "",
            "p_value": "",
            "ci_lower": "",
            "ci_upper": "",
            "n_obs": "",
            "r_squared": "",
            "coefficient_vector_json": json.dumps({"error": str(e)}),
            "sample_desc": sample_desc,
            "fixed_effects": fe_desc,
            "controls_desc": controls_desc,
            "cluster_var": "",
        }
        results.append(row)
        print(f"  {spec_run_id}: {spec_id} | FAILED: {e}")
        return row


def run_nbreg(outcome, treatment, controls, data, fe_dummies=None,
              spec_id=None, spec_tree_path=None, baseline_group_id=None,
              controls_desc="", sample_desc="esample2==1", fe_desc=""):
    """Run negative binomial regression via statsmodels."""
    global run_counter
    run_counter += 1
    spec_run_id = f"{PAPER_ID}_run{run_counter:04d}"

    try:
        rhs_vars = [treatment] + controls
        if fe_dummies:
            rhs_vars = rhs_vars + fe_dummies
        all_vars = [outcome] + rhs_vars
        data_clean = data[all_vars].dropna()

        formula_str = f"{outcome} ~ " + " + ".join(rhs_vars)
        m = smf.negativebinomial(formula_str, data=data_clean).fit(disp=0, maxiter=500, cov_type="HC1")

        coef = float(m.params[treatment])
        se_val = float(m.bse[treatment])
        pval = float(m.pvalues[treatment])
        nobs = int(m.nobs)
        r2 = float(m.prsquared) if hasattr(m, 'prsquared') else ""
        ci_lower = coef - 1.96 * se_val
        ci_upper = coef + 1.96 * se_val

        coef_dict = {k: float(v) for k, v in m.params.items()}

        row = {
            "paper_id": PAPER_ID,
            "spec_run_id": spec_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome,
            "treatment_var": treatment,
            "coefficient": round(coef, 8),
            "std_error": round(se_val, 8),
            "p_value": round(pval, 8),
            "ci_lower": round(ci_lower, 8),
            "ci_upper": round(ci_upper, 8),
            "n_obs": nobs,
            "r_squared": round(r2, 6) if isinstance(r2, float) else "",
            "coefficient_vector_json": json.dumps(coef_dict),
            "sample_desc": sample_desc,
            "fixed_effects": fe_desc,
            "controls_desc": controls_desc,
            "cluster_var": "",
        }
        results.append(row)
        print(f"  {spec_run_id}: {spec_id} | coef={coef:.6f} se={se_val:.6f} p={pval:.4f} N={nobs}")
        return row

    except Exception as e:
        row = {
            "paper_id": PAPER_ID,
            "spec_run_id": spec_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome,
            "treatment_var": treatment,
            "coefficient": "",
            "std_error": "",
            "p_value": "",
            "ci_lower": "",
            "ci_upper": "",
            "n_obs": "",
            "r_squared": "",
            "coefficient_vector_json": json.dumps({"error": str(e)}),
            "sample_desc": sample_desc,
            "fixed_effects": fe_desc,
            "controls_desc": controls_desc,
            "cluster_var": "",
        }
        results.append(row)
        print(f"  {spec_run_id}: {spec_id} | FAILED: {e}")
        return row


def blocks_to_controls(block_indices):
    """Convert a tuple of block inclusion indicators (0/1) to a flat list of control variables."""
    controls = []
    for i, include in enumerate(block_indices):
        if include:
            controls.extend(CONTROL_BLOCKS[BLOCK_NAMES[i]])
    return controls


def blocks_to_desc(block_indices):
    """Convert block indices to human-readable description."""
    included = [BLOCK_NAMES[i] for i, v in enumerate(block_indices) if v]
    if not included:
        return "none (empty set)"
    return " + ".join(included)


def trim_sample(data, outcome, lower_pct, upper_pct):
    """Trim sample by percentiles of outcome variable."""
    lo = data[outcome].quantile(lower_pct / 100.0)
    hi = data[outcome].quantile(upper_pct / 100.0)
    return data[(data[outcome] >= lo) & (data[outcome] <= hi)].copy()


def cooksd_filter(data, outcome, treatment, controls, fe_col, threshold_mult=4):
    """Remove high Cook's D observations (4/N threshold). Preserves all columns."""
    rhs = [treatment] + controls
    # Include FE dummies in the OLS for Cook's D computation
    fe_dum_cols = [c for c in data.columns if c.startswith("uf_d")]
    all_vars = list(set([outcome] + rhs + fe_dum_cols + ([fe_col] if fe_col else [])))
    clean = data.dropna(subset=[outcome] + rhs)
    y = clean[outcome].values
    X_mat = sm.add_constant(clean[rhs + fe_dum_cols].values)
    try:
        ols = sm.OLS(y, X_mat).fit()
        influence = ols.get_influence()
        cooks = influence.cooks_distance[0]
        n = len(y)
        mask = cooks < (threshold_mult / n)
        return clean[mask].copy()
    except:
        return clean.copy()


# =============================================================================
# ==================== BASELINE GROUP G1: pcorrupt ============================
# =============================================================================
print("\n" + "=" * 70)
print("BASELINE GROUP G1: pcorrupt ~ first")
print("=" * 70)

# --- G1 Baseline ---
print("\n--- G1 Baseline (Table 4 Col 6) ---")
run_ols("pcorrupt", "first", G1_BASELINE_CONTROLS, dfs, fe="uf",
        spec_id="baseline", spec_tree_path="baseline",
        baseline_group_id="G1", fe_desc="uf",
        controls_desc="prefchar2_continuous + prefchar2_party + munichar2 + fiscal + political + sorteio")

# --- G1 Design ---
print("\n--- G1 Design (OLS, same as baseline) ---")
run_ols("pcorrupt", "first", G1_BASELINE_CONTROLS, dfs, fe="uf",
        spec_id="design/cross_sectional_ols/estimator/ols",
        spec_tree_path="designs/cross_sectional_ols.md#estimators",
        baseline_group_id="G1", fe_desc="uf",
        controls_desc="baseline controls")

# --- G1 RC: Control Sets ---
print("\n--- G1 RC: Control Sets ---")

# none (bivariate + state FE)
run_ols("pcorrupt", "first", [], dfs, fe="uf",
        spec_id="rc/controls/sets/none",
        spec_tree_path="modules/robustness/controls.md#control-sets",
        baseline_group_id="G1", fe_desc="uf",
        controls_desc="none (bivariate)")

# prefchar2 only
run_ols("pcorrupt", "first", prefchar2_continuous + prefchar2_party, dfs, fe="uf",
        spec_id="rc/controls/sets/prefchar2",
        spec_tree_path="modules/robustness/controls.md#control-sets",
        baseline_group_id="G1", fe_desc="uf",
        controls_desc="prefchar2_continuous + prefchar2_party")

# prefchar2 + munichar2 + fiscal
run_ols("pcorrupt", "first", prefchar2_continuous + prefchar2_party + munichar2 + fiscal,
        dfs, fe="uf",
        spec_id="rc/controls/sets/prefchar2_munichar2_fiscal",
        spec_tree_path="modules/robustness/controls.md#control-sets",
        baseline_group_id="G1", fe_desc="uf",
        controls_desc="prefchar2 + munichar2 + fiscal")

# prefchar2 + munichar2 + fiscal + political
run_ols("pcorrupt", "first",
        prefchar2_continuous + prefchar2_party + munichar2 + fiscal + political,
        dfs, fe="uf",
        spec_id="rc/controls/sets/prefchar2_munichar2_fiscal_political",
        spec_tree_path="modules/robustness/controls.md#control-sets",
        baseline_group_id="G1", fe_desc="uf",
        controls_desc="prefchar2 + munichar2 + fiscal + political")

# prefchar2 + munichar2 + fiscal + political + sorteio (= baseline without FE... but we use FE)
run_ols("pcorrupt", "first",
        prefchar2_continuous + prefchar2_party + munichar2 + fiscal + political + sorteio,
        dfs, fe="uf",
        spec_id="rc/controls/sets/prefchar2_munichar2_fiscal_political_sorteio",
        spec_tree_path="modules/robustness/controls.md#control-sets",
        baseline_group_id="G1", fe_desc="uf",
        controls_desc="all 6 blocks (= baseline)")

# full with audit scale
run_ols("pcorrupt", "first", G1_BASELINE_CONTROLS + audit_scale, dfs, fe="uf",
        spec_id="rc/controls/sets/full_with_audit_scale",
        spec_tree_path="modules/robustness/controls.md#control-sets",
        baseline_group_id="G1", fe_desc="uf",
        controls_desc="baseline + lfunc_ativ + lrec_fisc")

# --- G1 RC: LOO Blocks ---
print("\n--- G1 RC: LOO Blocks ---")
for block_name in BLOCK_NAMES:
    remaining = [v for bn, bv in CONTROL_BLOCKS.items()
                 for v in bv if bn != block_name]
    run_ols("pcorrupt", "first", remaining, dfs, fe="uf",
            spec_id=f"rc/controls/loo_block/drop_{block_name}",
            spec_tree_path="modules/robustness/controls.md#leave-one-out-block",
            baseline_group_id="G1", fe_desc="uf",
            controls_desc=f"baseline minus {block_name}")

# --- G1 RC: LOO Key Variables ---
print("\n--- G1 RC: LOO Key Variables ---")
loo_vars = ["pref_masc", "pref_idade_tse", "pref_escola",
            "lpop", "purb", "p_secundario", "mun_novo", "lpib02", "gini_ipea",
            "lrec_trans", "p_cad_pref", "vereador_eleit", "ENLP2000", "comarca"]
for var in loo_vars:
    remaining = [v for v in G1_BASELINE_CONTROLS if v != var]
    run_ols("pcorrupt", "first", remaining, dfs, fe="uf",
            spec_id=f"rc/controls/loo/drop_{var}",
            spec_tree_path="modules/robustness/controls.md#leave-one-out-controls-loo",
            baseline_group_id="G1", fe_desc="uf",
            controls_desc=f"baseline minus {var}")

# --- G1 RC: Add Experience Controls ---
print("\n--- G1 RC: Add Experience Controls ---")
# exp_prefeito
run_ols("pcorrupt", "first", G1_BASELINE_CONTROLS + audit_scale + ["exp_prefeito"],
        dfs, fe="uf",
        spec_id="rc/controls/add/exp_prefeito",
        spec_tree_path="modules/robustness/controls.md#add-controls",
        baseline_group_id="G1", fe_desc="uf",
        controls_desc="baseline + audit_scale + exp_prefeito")

# nexp + nexp2
run_ols("pcorrupt", "first", G1_BASELINE_CONTROLS + audit_scale + ["nexp", "nexp2"],
        dfs, fe="uf",
        spec_id="rc/controls/add/nexp_nexp2",
        spec_tree_path="modules/robustness/controls.md#add-controls",
        baseline_group_id="G1", fe_desc="uf",
        controls_desc="baseline + audit_scale + nexp + nexp2")

# --- G1 RC: Exhaustive Block Combinations (2^6 = 64) ---
print("\n--- G1 RC: Exhaustive Block Combinations ---")
n_blocks = len(BLOCK_NAMES)
for combo_idx in range(2 ** n_blocks):
    bits = tuple((combo_idx >> i) & 1 for i in range(n_blocks))
    ctrl = blocks_to_controls(bits)
    desc = blocks_to_desc(bits)

    # Tag overlaps with existing specs
    if combo_idx == 0:
        # Empty set = same as rc/controls/sets/none -- skip duplicate
        continue
    if all(b == 1 for b in bits):
        # Full set = same as baseline -- skip duplicate
        continue

    extra = {"block_combination": {BLOCK_NAMES[i]: bool(bits[i]) for i in range(n_blocks)},
             "combo_index": combo_idx}

    run_ols("pcorrupt", "first", ctrl, dfs, fe="uf",
            spec_id=f"rc/controls/subset/exhaustive_blocks",
            spec_tree_path="modules/robustness/controls.md#control-subset-enumeration",
            baseline_group_id="G1", fe_desc="uf",
            controls_desc=desc, extra_json=extra)

# --- G1 RC: FE Variants ---
print("\n--- G1 RC: FE Variants ---")

# Drop state FE
run_ols("pcorrupt", "first", G1_BASELINE_CONTROLS, dfs, fe=None,
        spec_id="rc/fe/drop/uf",
        spec_tree_path="modules/robustness/fixed_effects.md#drop-fe",
        baseline_group_id="G1", fe_desc="none",
        controls_desc="baseline controls, no FE")

# Add region FE instead of state FE
# Create region indicator from state codes
if 'region' not in dfs.columns:
    # Brazilian states to regions mapping
    # North: AC AM AP PA RO RR TO -> uf codes: AC=12,AM=13,AP=16,PA=15,RO=11,RR=14,TO=17
    # Northeast: AL BA CE MA PB PE PI RN SE -> uf codes
    # Southeast: ES MG RJ SP
    # South: PR RS SC
    # Central-West: DF GO MS MT
    # Use the first digit of state code as a proxy for region
    # Actually, just use C(uf) with fewer categories - use region dummies
    # For simplicity, create region from uf categories
    state_to_region = {
        'AC': 'N', 'AM': 'N', 'AP': 'N', 'PA': 'N', 'RO': 'N', 'RR': 'N', 'TO': 'N',
        'AL': 'NE', 'BA': 'NE', 'CE': 'NE', 'MA': 'NE', 'PB': 'NE', 'PE': 'NE',
        'PI': 'NE', 'RN': 'NE', 'SE': 'NE',
        'ES': 'SE', 'MG': 'SE', 'RJ': 'SE', 'SP': 'SE',
        'PR': 'S', 'RS': 'S', 'SC': 'S',
        'DF': 'CO', 'GO': 'CO', 'MS': 'CO', 'MT': 'CO'
    }
    if 'uf' in dfs.columns:
        dfs['region'] = dfs['uf'].map(state_to_region)
    else:
        # Use uf_dummies to infer region - create a simple numeric region
        # Just partition by available uf values
        dfs['region'] = 'unknown'

run_ols("pcorrupt", "first", G1_BASELINE_CONTROLS, dfs, fe="region",
        spec_id="rc/fe/add/region",
        spec_tree_path="modules/robustness/fixed_effects.md#fe-variants",
        baseline_group_id="G1", fe_desc="region",
        controls_desc="baseline controls, region FE instead of state FE")

# --- G1 RC: Sample Variants ---
print("\n--- G1 RC: Sample Variants ---")

# Trim 1-99
trimmed_1_99 = trim_sample(dfs, "pcorrupt", 1, 99)
run_ols("pcorrupt", "first", G1_BASELINE_CONTROLS, trimmed_1_99, fe="uf",
        spec_id="rc/sample/outliers/trim_y_1_99",
        spec_tree_path="modules/robustness/sample.md#trimming",
        baseline_group_id="G1", fe_desc="uf",
        controls_desc="baseline controls",
        sample_desc="esample2==1, pcorrupt trimmed 1-99 pctile")

# Trim 5-95
trimmed_5_95 = trim_sample(dfs, "pcorrupt", 5, 95)
run_ols("pcorrupt", "first", G1_BASELINE_CONTROLS, trimmed_5_95, fe="uf",
        spec_id="rc/sample/outliers/trim_y_5_95",
        spec_tree_path="modules/robustness/sample.md#trimming",
        baseline_group_id="G1", fe_desc="uf",
        controls_desc="baseline controls",
        sample_desc="esample2==1, pcorrupt trimmed 5-95 pctile")

# Cook's D
cooksd_data = cooksd_filter(dfs, "pcorrupt", "first", G1_BASELINE_CONTROLS, "uf")
run_ols("pcorrupt", "first", G1_BASELINE_CONTROLS, cooksd_data, fe="uf",
        spec_id="rc/sample/outliers/cooksd_4_over_n",
        spec_tree_path="modules/robustness/sample.md#cooks-distance",
        baseline_group_id="G1", fe_desc="uf",
        controls_desc="baseline controls",
        sample_desc="esample2==1, Cook's D < 4/N")

# Running variable non-missing (RDD sample)
running_nonmissing = dfs[dfs['running'].notna()].copy()
run_ols("pcorrupt", "first", G1_BASELINE_CONTROLS, running_nonmissing, fe="uf",
        spec_id="rc/sample/restriction/running_nonmissing",
        spec_tree_path="modules/robustness/sample.md#sample-restriction",
        baseline_group_id="G1", fe_desc="uf",
        controls_desc="baseline controls",
        sample_desc="esample2==1 & running~=. (N~328)")

# pmismanagement non-missing
pmis_nonmissing = dfs[dfs['pmismanagement'].notna()].copy()
run_ols("pcorrupt", "first", G1_BASELINE_CONTROLS, pmis_nonmissing, fe="uf",
        spec_id="rc/sample/restriction/pmismanagement_nonmissing",
        spec_tree_path="modules/robustness/sample.md#sample-restriction",
        baseline_group_id="G1", fe_desc="uf",
        controls_desc="baseline controls",
        sample_desc="esample2==1 & pmismanagement non-missing")

# --- G1 RC: Functional Form ---
print("\n--- G1 RC: Functional Form ---")

# asinh(pcorrupt)
dfs['asinh_pcorrupt'] = np.arcsinh(dfs['pcorrupt'])
run_ols("asinh_pcorrupt", "first", G1_BASELINE_CONTROLS, dfs, fe="uf",
        spec_id="rc/form/outcome/asinh",
        spec_tree_path="modules/robustness/functional_form.md#outcome-transform",
        baseline_group_id="G1", fe_desc="uf",
        controls_desc="baseline controls, asinh(pcorrupt)")

# RDD polynomial: linear (joint: restrict to running_nonmissing + add running)
run_ols("pcorrupt", "first", G1_BASELINE_CONTROLS + ["running"],
        running_nonmissing, fe="uf",
        spec_id="rc/form/model/rdd_polynomial_linear",
        spec_tree_path="modules/robustness/functional_form.md#rdd-polynomial",
        baseline_group_id="G1", fe_desc="uf",
        controls_desc="baseline + linear running",
        sample_desc="esample2==1 & running~=.")

# RDD polynomial: quadratic
run_ols("pcorrupt", "first", G1_BASELINE_CONTROLS + ["running", "running2"],
        running_nonmissing, fe="uf",
        spec_id="rc/form/model/rdd_polynomial_quadratic",
        spec_tree_path="modules/robustness/functional_form.md#rdd-polynomial",
        baseline_group_id="G1", fe_desc="uf",
        controls_desc="baseline + quadratic running",
        sample_desc="esample2==1 & running~=.")

# RDD polynomial: cubic
run_ols("pcorrupt", "first", G1_BASELINE_CONTROLS + ["running", "running2", "running3"],
        running_nonmissing, fe="uf",
        spec_id="rc/form/model/rdd_polynomial_cubic",
        spec_tree_path="modules/robustness/functional_form.md#rdd-polynomial",
        baseline_group_id="G1", fe_desc="uf",
        controls_desc="baseline + cubic running",
        sample_desc="esample2==1 & running~=.")

# --- G1 RC: Estimator Variant (Tobit) ---
print("\n--- G1 RC: Tobit ---")
tobit_controls = G1_BASELINE_CONTROLS + audit_scale
run_tobit("pcorrupt", "first", tobit_controls, dfs,
          fe_dummies=uf_dummies, lower=0,
          spec_id="rc/estimation/tobit_ll0",
          spec_tree_path="modules/estimation/tobit.md",
          baseline_group_id="G1", fe_desc="uf (dummies)",
          controls_desc="baseline + audit_scale + uf dummies")

# --- G1 Inference Variants ---
print("\n--- G1 Inference Variants ---")

# Classical (non-robust) SE
run_ols("pcorrupt", "first", G1_BASELINE_CONTROLS, dfs, fe="uf",
        vcov_type="iid",
        spec_id="infer/se/hc/classical",
        spec_tree_path="modules/inference/standard_errors.md#classical",
        baseline_group_id="G1", fe_desc="uf",
        controls_desc="baseline controls, classical SE")

# HC2 -- pyfixest does not support HC2/HC3 with absorbed FE, use explicit dummies
run_ols("pcorrupt", "first", G1_BASELINE_CONTROLS + uf_dummies, dfs, fe=None,
        vcov_type="HC2",
        spec_id="infer/se/hc/hc2",
        spec_tree_path="modules/inference/standard_errors.md#hc2",
        baseline_group_id="G1", fe_desc="uf (dummies)",
        controls_desc="baseline controls + uf dummies, HC2 SE")

# HC3
run_ols("pcorrupt", "first", G1_BASELINE_CONTROLS + uf_dummies, dfs, fe=None,
        vcov_type="HC3",
        spec_id="infer/se/hc/hc3",
        spec_tree_path="modules/inference/standard_errors.md#hc3",
        baseline_group_id="G1", fe_desc="uf (dummies)",
        controls_desc="baseline controls + uf dummies, HC3 SE")

# Cluster at state level
run_ols("pcorrupt", "first", G1_BASELINE_CONTROLS, dfs, fe="uf",
        cluster_var="uf",
        spec_id="infer/se/cluster/uf",
        spec_tree_path="modules/inference/standard_errors.md#cluster",
        baseline_group_id="G1", fe_desc="uf",
        controls_desc="baseline controls, clustered SE at state level")


# =============================================================================
# ==================== BASELINE GROUP G2: ncorrupt ============================
# =============================================================================
print("\n" + "=" * 70)
print("BASELINE GROUP G2: ncorrupt ~ first")
print("=" * 70)

# --- G2 Baseline ---
print("\n--- G2 Baseline (Table 5A Col 2) ---")
run_ols("ncorrupt", "first", G2_BASELINE_CONTROLS, dfs, fe="uf",
        spec_id="baseline", spec_tree_path="baseline",
        baseline_group_id="G2", fe_desc="uf",
        controls_desc="lrec_fisc + prefchar2 + munichar2 + fiscal + political + sorteio")

# --- G2 Design ---
print("\n--- G2 Design ---")
run_ols("ncorrupt", "first", G2_BASELINE_CONTROLS, dfs, fe="uf",
        spec_id="design/cross_sectional_ols/estimator/ols",
        spec_tree_path="designs/cross_sectional_ols.md#estimators",
        baseline_group_id="G2", fe_desc="uf",
        controls_desc="baseline controls")

# --- G2 RC: Control Sets ---
print("\n--- G2 RC: Control Sets ---")

# Mandatory only (lrec_fisc + state FE)
run_ols("ncorrupt", "first", ["lrec_fisc"], dfs, fe="uf",
        spec_id="rc/controls/sets/mandatory_only",
        spec_tree_path="modules/robustness/controls.md#control-sets",
        baseline_group_id="G2", fe_desc="uf",
        controls_desc="lrec_fisc only (mandatory)")

# Full with lfunc_ativ
run_ols("ncorrupt", "first", G2_BASELINE_CONTROLS + ["lfunc_ativ"], dfs, fe="uf",
        spec_id="rc/controls/sets/full_with_lfunc_ativ",
        spec_tree_path="modules/robustness/controls.md#control-sets",
        baseline_group_id="G2", fe_desc="uf",
        controls_desc="baseline + lfunc_ativ")

# --- G2 RC: LOO Blocks ---
print("\n--- G2 RC: LOO Blocks ---")
for block_name in BLOCK_NAMES:
    remaining_optional = [v for bn, bv in CONTROL_BLOCKS.items()
                         for v in bv if bn != block_name]
    controls_g2 = ["lrec_fisc"] + remaining_optional
    run_ols("ncorrupt", "first", controls_g2, dfs, fe="uf",
            spec_id=f"rc/controls/loo_block/drop_{block_name}",
            spec_tree_path="modules/robustness/controls.md#leave-one-out-block",
            baseline_group_id="G2", fe_desc="uf",
            controls_desc=f"lrec_fisc + baseline minus {block_name}")

# --- G2 RC: Sample Variants ---
print("\n--- G2 RC: Sample Variants ---")

# Trim 1-99
trimmed_1_99_nc = trim_sample(dfs, "ncorrupt", 1, 99)
run_ols("ncorrupt", "first", G2_BASELINE_CONTROLS, trimmed_1_99_nc, fe="uf",
        spec_id="rc/sample/outliers/trim_y_1_99",
        spec_tree_path="modules/robustness/sample.md#trimming",
        baseline_group_id="G2", fe_desc="uf",
        controls_desc="baseline controls",
        sample_desc="esample2==1, ncorrupt trimmed 1-99 pctile")

# Trim 5-95
trimmed_5_95_nc = trim_sample(dfs, "ncorrupt", 5, 95)
run_ols("ncorrupt", "first", G2_BASELINE_CONTROLS, trimmed_5_95_nc, fe="uf",
        spec_id="rc/sample/outliers/trim_y_5_95",
        spec_tree_path="modules/robustness/sample.md#trimming",
        baseline_group_id="G2", fe_desc="uf",
        controls_desc="baseline controls",
        sample_desc="esample2==1, ncorrupt trimmed 5-95 pctile")

# --- G2 RC: Functional Form ---
print("\n--- G2 RC: Functional Form ---")

# asinh(ncorrupt)
dfs['asinh_ncorrupt'] = np.arcsinh(dfs['ncorrupt'])
run_ols("asinh_ncorrupt", "first", G2_BASELINE_CONTROLS, dfs, fe="uf",
        spec_id="rc/form/outcome/asinh",
        spec_tree_path="modules/robustness/functional_form.md#outcome-transform",
        baseline_group_id="G2", fe_desc="uf",
        controls_desc="baseline controls, asinh(ncorrupt)")

# --- G2 RC: Estimator Variant (Negative Binomial) ---
print("\n--- G2 RC: Negative Binomial ---")
# NB with uf dummies (Table 5A Col 4 approach: sorteio2-10, uf_dummies)
sorteio_2_10 = [f"sorteio{i}" for i in range(2, 11)]
nbreg_controls = (["lrec_fisc"] + prefchar2_continuous + prefchar2_party +
                  munichar2 + fiscal + ["lfunc_ativ"] + political + sorteio_2_10)
run_nbreg("ncorrupt", "first", nbreg_controls, dfs,
          fe_dummies=uf_dummies,
          spec_id="rc/estimation/nbreg",
          spec_tree_path="modules/estimation/count_models.md#negative-binomial",
          baseline_group_id="G2", fe_desc="uf (dummies)",
          controls_desc="full controls + uf dummies (NegBin)")

# --- G2 Inference Variants ---
print("\n--- G2 Inference Variants ---")

run_ols("ncorrupt", "first", G2_BASELINE_CONTROLS + uf_dummies, dfs, fe=None,
        vcov_type="HC2",
        spec_id="infer/se/hc/hc2",
        spec_tree_path="modules/inference/standard_errors.md#hc2",
        baseline_group_id="G2", fe_desc="uf (dummies)",
        controls_desc="baseline controls + uf dummies, HC2 SE")

run_ols("ncorrupt", "first", G2_BASELINE_CONTROLS + uf_dummies, dfs, fe=None,
        vcov_type="HC3",
        spec_id="infer/se/hc/hc3",
        spec_tree_path="modules/inference/standard_errors.md#hc3",
        baseline_group_id="G2", fe_desc="uf (dummies)",
        controls_desc="baseline controls + uf dummies, HC3 SE")


# =============================================================================
# ==================== BASELINE GROUP G3: ncorrupt_os =========================
# =============================================================================
print("\n" + "=" * 70)
print("BASELINE GROUP G3: ncorrupt_os ~ first")
print("=" * 70)

# --- G3 Baseline ---
print("\n--- G3 Baseline (Table 5B Col 2) ---")
run_ols("ncorrupt_os", "first", G3_BASELINE_CONTROLS, dfs, fe="uf",
        spec_id="baseline", spec_tree_path="baseline",
        baseline_group_id="G3", fe_desc="uf",
        controls_desc="lrec_fisc + lfunc_ativ + prefchar2 + munichar2 + fiscal + political + sorteio")

# --- G3 Design ---
print("\n--- G3 Design ---")
run_ols("ncorrupt_os", "first", G3_BASELINE_CONTROLS, dfs, fe="uf",
        spec_id="design/cross_sectional_ols/estimator/ols",
        spec_tree_path="designs/cross_sectional_ols.md#estimators",
        baseline_group_id="G3", fe_desc="uf",
        controls_desc="baseline controls")

# --- G3 RC: Control Sets ---
print("\n--- G3 RC: Control Sets ---")

# Mandatory only (lrec_fisc + lfunc_ativ + state FE)
run_ols("ncorrupt_os", "first", ["lrec_fisc", "lfunc_ativ"], dfs, fe="uf",
        spec_id="rc/controls/sets/mandatory_only",
        spec_tree_path="modules/robustness/controls.md#control-sets",
        baseline_group_id="G3", fe_desc="uf",
        controls_desc="lrec_fisc + lfunc_ativ only (mandatory)")

# Baseline (full controls)
run_ols("ncorrupt_os", "first", G3_BASELINE_CONTROLS, dfs, fe="uf",
        spec_id="rc/controls/sets/baseline",
        spec_tree_path="modules/robustness/controls.md#control-sets",
        baseline_group_id="G3", fe_desc="uf",
        controls_desc="full baseline (42 controls)")

# --- G3 RC: LOO Blocks ---
print("\n--- G3 RC: LOO Blocks ---")
for block_name in BLOCK_NAMES:
    remaining_optional = [v for bn, bv in CONTROL_BLOCKS.items()
                         for v in bv if bn != block_name]
    controls_g3 = ["lrec_fisc", "lfunc_ativ"] + remaining_optional
    run_ols("ncorrupt_os", "first", controls_g3, dfs, fe="uf",
            spec_id=f"rc/controls/loo_block/drop_{block_name}",
            spec_tree_path="modules/robustness/controls.md#leave-one-out-block",
            baseline_group_id="G3", fe_desc="uf",
            controls_desc=f"lrec_fisc + lfunc_ativ + baseline minus {block_name}")

# --- G3 RC: Sample Variants ---
print("\n--- G3 RC: Sample Variants ---")

# Trim 1-99
trimmed_1_99_nos = trim_sample(dfs, "ncorrupt_os", 1, 99)
run_ols("ncorrupt_os", "first", G3_BASELINE_CONTROLS, trimmed_1_99_nos, fe="uf",
        spec_id="rc/sample/outliers/trim_y_1_99",
        spec_tree_path="modules/robustness/sample.md#trimming",
        baseline_group_id="G3", fe_desc="uf",
        controls_desc="baseline controls",
        sample_desc="esample2==1, ncorrupt_os trimmed 1-99 pctile")

# Trim 5-95
trimmed_5_95_nos = trim_sample(dfs, "ncorrupt_os", 5, 95)
run_ols("ncorrupt_os", "first", G3_BASELINE_CONTROLS, trimmed_5_95_nos, fe="uf",
        spec_id="rc/sample/outliers/trim_y_5_95",
        spec_tree_path="modules/robustness/sample.md#trimming",
        baseline_group_id="G3", fe_desc="uf",
        controls_desc="baseline controls",
        sample_desc="esample2==1, ncorrupt_os trimmed 5-95 pctile")

# --- G3 RC: Estimator Variant (Tobit) ---
print("\n--- G3 RC: Tobit ---")
run_tobit("ncorrupt_os", "first", G3_BASELINE_CONTROLS, dfs,
          fe_dummies=uf_dummies, lower=0,
          spec_id="rc/estimation/tobit_ll0",
          spec_tree_path="modules/estimation/tobit.md",
          baseline_group_id="G3", fe_desc="uf (dummies)",
          controls_desc="baseline + uf dummies, tobit ll(0)")

# --- G3 Inference Variants ---
print("\n--- G3 Inference Variants ---")

run_ols("ncorrupt_os", "first", G3_BASELINE_CONTROLS + uf_dummies, dfs, fe=None,
        vcov_type="HC2",
        spec_id="infer/se/hc/hc2",
        spec_tree_path="modules/inference/standard_errors.md#hc2",
        baseline_group_id="G3", fe_desc="uf (dummies)",
        controls_desc="baseline controls + uf dummies, HC2 SE")

run_ols("ncorrupt_os", "first", G3_BASELINE_CONTROLS + uf_dummies, dfs, fe=None,
        vcov_type="HC3",
        spec_id="infer/se/hc/hc3",
        spec_tree_path="modules/inference/standard_errors.md#hc3",
        baseline_group_id="G3", fe_desc="uf (dummies)",
        controls_desc="baseline controls + uf dummies, HC3 SE")


# =============================================================================
# ==================== DIAGNOSTICS ============================================
# =============================================================================
print("\n" + "=" * 70)
print("DIAGNOSTICS: Balance test for G1")
print("=" * 70)

diag_results = []
# Balance test: regress each control on treatment (first)
balance_vars = (prefchar2_continuous + munichar2 + fiscal + political +
                ["exp_prefeito", "lpib02", "gini_ipea"])
# Deduplicate
balance_vars = list(dict.fromkeys(balance_vars))

balance_rows = []
for bvar in balance_vars:
    try:
        bdata = dfs[[bvar, "first"]].dropna()
        if len(bdata) < 10:
            continue
        m = pf.feols(f"{bvar} ~ first", data=bdata, vcov="hetero")
        coef = float(m.coef()["first"])
        se = float(m.se()["first"])
        pval = float(m.pvalue()["first"])
        balance_rows.append({
            "variable": bvar,
            "coefficient": round(coef, 6),
            "std_error": round(se, 6),
            "p_value": round(pval, 4),
            "n_obs": int(m._N),
            "significant_05": pval < 0.05
        })
    except Exception as e:
        balance_rows.append({
            "variable": bvar,
            "coefficient": "",
            "std_error": "",
            "p_value": "",
            "n_obs": "",
            "significant_05": "",
            "error": str(e)
        })

n_sig = sum(1 for r in balance_rows if r.get("significant_05") == True)
n_total = len(balance_rows)

diag_results.append({
    "paper_id": PAPER_ID,
    "diagnostic_run_id": f"{PAPER_ID}_diag001",
    "diag_spec_id": "diag/cross_sectional_ols/balance/treatment_balance",
    "spec_tree_path": "modules/diagnostics/balance.md",
    "diagnostic_scope": "baseline_group",
    "diagnostic_context_id": "G1_balance",
    "diagnostic_json": json.dumps({
        "baseline_group_id": "G1",
        "n_variables_tested": n_total,
        "n_significant_05": n_sig,
        "fraction_significant": round(n_sig / n_total, 4) if n_total > 0 else "",
        "variable_results": balance_rows
    })
})
print(f"  Balance test: {n_sig}/{n_total} variables significant at 5%")


# =============================================================================
# ==================== WRITE OUTPUTS ==========================================
# =============================================================================
print("\n" + "=" * 70)
print("WRITING OUTPUTS")
print("=" * 70)

# 1. specification_results.csv
results_df = pd.DataFrame(results)
csv_path = os.path.join(PACKAGE_DIR, "specification_results.csv")
results_df.to_csv(csv_path, index=False)
print(f"Wrote {len(results_df)} rows to specification_results.csv")

# Count by group
for gid in ["G1", "G2", "G3"]:
    n = (results_df['baseline_group_id'] == gid).sum()
    print(f"  {gid}: {n} specs")

# Count by type
for prefix in ["baseline", "design/", "rc/", "infer/"]:
    n = results_df['spec_id'].str.startswith(prefix).sum()
    print(f"  {prefix}: {n} specs")

# 2. diagnostics_results.csv
diag_df = pd.DataFrame(diag_results)
diag_path = os.path.join(PACKAGE_DIR, "diagnostics_results.csv")
diag_df.to_csv(diag_path, index=False)
print(f"Wrote {len(diag_df)} rows to diagnostics_results.csv")

# 3. Check uniqueness
assert results_df['spec_run_id'].nunique() == len(results_df), "spec_run_id not unique!"
print("All spec_run_ids are unique.")

# 4. Summary statistics
n_success = results_df[results_df['coefficient'] != ''].shape[0]
n_fail = results_df[results_df['coefficient'] == ''].shape[0]
print(f"\nExecution summary: {n_success} succeeded, {n_fail} failed, {len(results_df)} total")

# Print coefficient range for baselines
for gid in ["G1", "G2", "G3"]:
    bl = results_df[(results_df['baseline_group_id'] == gid) & (results_df['spec_id'] == 'baseline')]
    if len(bl) > 0 and bl.iloc[0]['coefficient'] != '':
        print(f"  {gid} baseline: coef={bl.iloc[0]['coefficient']}, se={bl.iloc[0]['std_error']}, p={bl.iloc[0]['p_value']}")
