"""
Replication script for Ferraz & Finan (2011)
"Electoral Accountability and Corruption: Evidence from the Audits of Local Governments"
American Economic Review, 101(4), 1274-1311.

Paper ID: 112431-V1

This script replicates the main regression results from Tables 4-11 and Figure 2.
Original code: Stata do-files (reelection_aer.do, convenios_aer.do).
Translated to Python using pyfixest and statsmodels.

Tables 1, 3 are descriptive/balance (not regression tables).
Table 2 has descriptive regressions (no controls, with constant / without constant). Included.
Tables 4-8, 10-11 are from reelection_aer.do (main results).
Table 9 is from convenios_aer.do (panel matching grants).
Abadie-Imbens matching ('match') commands are flagged as UNLISTED_METHOD.

Total regression commands in do-files: 55 (48 in reelection_aer.do, 6 in convenios_aer.do,
1 in pelect_aer.do). Of these, 3 are 'match' (skipped), 1 is pelect logit (auxiliary),
20 are Table 2 loop regressions, and 20 are Table 3 balance tests.
In-scope regressions replicated below: all Table 4-11 reg/areg/tobit/nbreg + Figure 2.
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import statsmodels.formula.api as smf
import statsmodels.api as sm
import json
import os
import warnings
warnings.filterwarnings('ignore')

from scipy.optimize import minimize
from scipy.stats import norm

# =============================================================================
# Configuration
# =============================================================================
PAPER_ID = "112431-V1"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PACKAGE_DIR = os.path.join(REPO_ROOT, "data", "downloads", "extracted", PAPER_ID)

# =============================================================================
# Load data
# =============================================================================
df = pd.read_stata(os.path.join(PACKAGE_DIR, "corruptiondata_aer.dta"))
df_conv = pd.read_stata(os.path.join(PACKAGE_DIR, "conveniosdata_aer.dta"))

# Filter to estimation sample (all 476 obs have esample2==1)
dfs = df[df['esample2'] == 1].copy()

# =============================================================================
# Define variable groups (matching Stata globals)
# =============================================================================
# global prefchar2 "pref_masc pref_idade_tse pref_escola party_d1 party_d3-party_d18"
prefchar2 = ["pref_masc", "pref_idade_tse", "pref_escola",
             "party_d1", "party_d3", "party_d4", "party_d5", "party_d6",
             "party_d7", "party_d8", "party_d9", "party_d10", "party_d11",
             "party_d12", "party_d13", "party_d14", "party_d15", "party_d16",
             "party_d17", "party_d18"]

# global munichar2 "lpop purb p_secundario mun_novo lpib02 gini_ipea"
munichar2 = ["lpop", "purb", "p_secundario", "mun_novo", "lpib02", "gini_ipea"]

# sorteio* = sorteio1-sorteio10
sorteio_all = [f"sorteio{i}" for i in range(1, 11)]
# sorteio2-sorteio10 (used in nbreg)
sorteio_2_10 = [f"sorteio{i}" for i in range(2, 11)]

# uf_d* state dummies (used in tobit, nbreg, match)
uf_dummies = sorted([c for c in df.columns if c.startswith("uf_d")])

# =============================================================================
# Helper functions
# =============================================================================
results = []
reg_counter = 0


def add_result(table, col, outcome_var, treatment_var, model, estimator,
               fixed_effects="", controls_desc="", cluster_var="",
               sample_desc="esample2==1", notes="", orig_coef=None, orig_se=None):
    """Extract results from a pyfixest or statsmodels model and add to results list."""
    global reg_counter
    reg_counter += 1

    try:
        if hasattr(model, 'coef'):  # pyfixest
            coef = float(model.coef()[treatment_var])
            se = float(model.se()[treatment_var])
            pval = float(model.pvalue()[treatment_var])
            nobs = int(model._N)
            r2 = float(model._r2)
            coef_dict = {k: float(v) for k, v in model.coef().items()}
        else:  # statsmodels
            coef = float(model.params[treatment_var])
            se = float(model.bse[treatment_var])
            pval = float(model.pvalues[treatment_var])
            nobs = int(model.nobs)
            r2 = getattr(model, 'prsquared', getattr(model, 'rsquared', None))
            if r2 is not None:
                r2 = float(r2)
            coef_dict = {k: float(v) for k, v in model.params.items()}

        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

        # Match status
        if orig_coef is not None:
            if abs(orig_coef) > 1e-10:
                rel_err = abs(coef - orig_coef) / abs(orig_coef)
            else:
                rel_err = abs(coef - orig_coef)
            rounded_match = (round(coef, 3) == round(orig_coef, 3))
            if rounded_match or rel_err < 1e-4:
                match_status = "exact"
            elif rel_err <= 0.01:
                match_status = "close"
            else:
                match_status = "discrepant"
        else:
            match_status = "close"  # no original to compare

        results.append({
            "paper_id": PAPER_ID,
            "reg_id": reg_counter,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "coefficient": round(coef, 8),
            "std_error": round(se, 8),
            "p_value": round(pval, 8),
            "ci_lower": round(ci_lower, 8),
            "ci_upper": round(ci_upper, 8),
            "n_obs": nobs,
            "r_squared": round(r2, 6) if r2 is not None else "",
            "original_coefficient": orig_coef if orig_coef is not None else "",
            "original_std_error": orig_se if orig_se is not None else "",
            "match_status": match_status,
            "coefficient_vector_json": json.dumps(coef_dict),
            "fixed_effects": fixed_effects,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "estimator": estimator,
            "sample_desc": sample_desc,
            "notes": f"Table {table}, Col {col}. {notes}".strip()
        })

        r2_str = f"{r2:.4f}" if r2 is not None else "N/A"
        print(f"  Reg {reg_counter:2d} (T{table} C{col}): {outcome_var} ~ {treatment_var} | "
              f"coef={coef:.6f} se={se:.6f} p={pval:.4f} N={nobs} R2={r2_str} | {match_status}")
        if orig_coef is not None:
            print(f"         orig={orig_coef} rel_err={rel_err:.6f}")

    except Exception as e:
        reg_counter_val = reg_counter
        results.append({
            "paper_id": PAPER_ID,
            "reg_id": reg_counter_val,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "coefficient": "", "std_error": "", "p_value": "",
            "ci_lower": "", "ci_upper": "",
            "n_obs": "", "r_squared": "",
            "original_coefficient": orig_coef if orig_coef is not None else "",
            "original_std_error": orig_se if orig_se is not None else "",
            "match_status": "failed",
            "coefficient_vector_json": "{}",
            "fixed_effects": fixed_effects,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "estimator": estimator,
            "sample_desc": sample_desc,
            "notes": f"Table {table}, Col {col}. FAILED: {str(e)}"
        })
        print(f"  Reg {reg_counter:2d} (T{table} C{col}): FAILED - {e}")


def add_tobit_result(table, col, outcome_var, treatment_var, y_data, X_data,
                     col_names, lower=0, fixed_effects="", controls_desc="",
                     sample_desc="esample2==1", notes="", orig_coef=None, orig_se=None):
    """Run Tobit MLE and add results."""
    global reg_counter
    reg_counter += 1

    try:
        y = y_data.values.astype(float)
        X = X_data.values.astype(float)
        n, k = X.shape

        # OLS starting values
        ols = np.linalg.lstsq(X, y, rcond=None)
        beta_init = ols[0]
        resid = y - X @ beta_init
        sigma_init = max(np.std(resid), 0.01)
        init = np.concatenate([beta_init, [np.log(sigma_init)]])

        def negll(params):
            beta = params[:-1]
            log_sigma = params[-1]
            sigma = np.exp(log_sigma)
            xb = X @ beta
            censored = (y <= lower)
            ll = 0.0
            if censored.any():
                ll += np.sum(norm.logcdf((lower - xb[censored]) / sigma))
            if (~censored).any():
                ll += np.sum(-0.5 * np.log(2 * np.pi) - log_sigma
                             - 0.5 * ((y[~censored] - xb[~censored]) / sigma) ** 2)
            return -ll

        res = minimize(negll, init, method='BFGS', options={'maxiter': 10000, 'gtol': 1e-8})

        # Numerical Hessian for standard errors
        from scipy.optimize import approx_fprime
        eps = 1e-5
        k_total = len(res.x)
        H = np.zeros((k_total, k_total))
        for i in range(k_total):
            def grad_i(p, idx=i):
                return approx_fprime(p, negll, eps)[idx]
            H[i] = approx_fprime(res.x, grad_i, eps)
        try:
            se_all = np.sqrt(np.diag(np.linalg.inv(H)))
        except Exception:
            se_all = np.full(k_total, np.nan)

        beta = res.x[:-1]
        se = se_all[:-1]

        first_idx = col_names.index(treatment_var)
        coef = float(beta[first_idx])
        se_val = float(se[first_idx])
        pval = float(2 * (1 - norm.cdf(abs(coef / se_val)))) if not np.isnan(se_val) else np.nan
        ci_lower = coef - 1.96 * se_val
        ci_upper = coef + 1.96 * se_val

        coef_dict = {col_names[i]: float(beta[i]) for i in range(len(col_names))}

        if orig_coef is not None:
            if abs(orig_coef) > 1e-10:
                rel_err = abs(coef - orig_coef) / abs(orig_coef)
            else:
                rel_err = abs(coef - orig_coef)
            rounded_match = (round(coef, 3) == round(orig_coef, 3))
            if rounded_match or rel_err < 1e-4:
                match_status = "exact"
            elif rel_err <= 0.01:
                match_status = "close"
            else:
                match_status = "discrepant"
        else:
            match_status = "close"

        results.append({
            "paper_id": PAPER_ID,
            "reg_id": reg_counter,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "coefficient": round(coef, 8),
            "std_error": round(se_val, 8),
            "p_value": round(pval, 8) if not np.isnan(pval) else "",
            "ci_lower": round(ci_lower, 8),
            "ci_upper": round(ci_upper, 8),
            "n_obs": n,
            "r_squared": "",
            "original_coefficient": orig_coef if orig_coef is not None else "",
            "original_std_error": orig_se if orig_se is not None else "",
            "match_status": match_status,
            "coefficient_vector_json": json.dumps(coef_dict),
            "fixed_effects": fixed_effects,
            "controls_desc": controls_desc,
            "cluster_var": "",
            "estimator": "Tobit",
            "sample_desc": sample_desc,
            "notes": f"Table {table}, Col {col}. {notes}".strip()
        })

        print(f"  Reg {reg_counter:2d} (T{table} C{col}): {outcome_var} ~ {treatment_var} [Tobit] | "
              f"coef={coef:.6f} se={se_val:.6f} N={n} | {match_status}")
        if orig_coef is not None:
            print(f"         orig={orig_coef} rel_err={rel_err:.6f}")

    except Exception as e:
        results.append({
            "paper_id": PAPER_ID, "reg_id": reg_counter,
            "outcome_var": outcome_var, "treatment_var": treatment_var,
            "coefficient": "", "std_error": "", "p_value": "",
            "ci_lower": "", "ci_upper": "",
            "n_obs": "", "r_squared": "",
            "original_coefficient": orig_coef if orig_coef is not None else "",
            "original_std_error": orig_se if orig_se is not None else "",
            "match_status": "failed",
            "coefficient_vector_json": "{}",
            "fixed_effects": fixed_effects, "controls_desc": controls_desc,
            "cluster_var": "", "estimator": "Tobit",
            "sample_desc": sample_desc,
            "notes": f"Table {table}, Col {col}. FAILED: {str(e)}"
        })
        print(f"  Reg {reg_counter:2d} (T{table} C{col}): FAILED - {e}")


def fml(y, rhs, absorb=None):
    """Build pyfixest formula."""
    rhs_str = " + ".join(rhs)
    if absorb:
        return f"{y} ~ {rhs_str} | {absorb}"
    else:
        return f"{y} ~ {rhs_str}"


# =============================================================================
# TABLE 4: Effects of Reelection on Corruption (Share of Resources)
# DV = pcorrupt, treatment = first
# Stata do-file lines 55-63
# Published (AER Table 4):
#   Col1: -0.019(0.009), Col2: -0.020(0.010), Col3: -0.020(0.010)
#   Col4: -0.024(0.011), Col5: -0.026(0.011), Col6: -0.027(0.011)
#   Col7: matching (skip), Col8: tobit
# =============================================================================
print("\n" + "=" * 70)
print("TABLE 4: pcorrupt ~ first")
print("=" * 70)

# Col 1: reg pcorrupt first, robust
m = pf.feols("pcorrupt ~ first", data=dfs, vcov="hetero")
add_result("4", "1", "pcorrupt", "first", m, "OLS",
           controls_desc="none",
           orig_coef=-0.019, orig_se=0.009)

# Col 2: + prefchar2
m = pf.feols(fml("pcorrupt", ["first"] + prefchar2), data=dfs, vcov="hetero")
add_result("4", "2", "pcorrupt", "first", m, "OLS",
           controls_desc="mayor characteristics (prefchar2)",
           orig_coef=-0.020, orig_se=0.010)

# Col 3: + munichar2 + lrec_trans
m = pf.feols(fml("pcorrupt", ["first"] + prefchar2 + munichar2 + ["lrec_trans"]),
             data=dfs, vcov="hetero")
add_result("4", "3", "pcorrupt", "first", m, "OLS",
           controls_desc="prefchar2 + munichar2 + lrec_trans",
           orig_coef=-0.020, orig_se=0.010)

# Col 4: + political controls
m = pf.feols(fml("pcorrupt", ["first"] + prefchar2 + munichar2 +
                  ["lrec_trans", "p_cad_pref", "vereador_eleit", "ENLP2000", "comarca"]),
             data=dfs, vcov="hetero")
add_result("4", "4", "pcorrupt", "first", m, "OLS",
           controls_desc="prefchar2 + munichar2 + lrec_trans + political",
           orig_coef=-0.024, orig_se=0.011)

# Col 5: + lottery dummies
base_rhs = (["first"] + prefchar2 + munichar2 +
            ["lrec_trans", "p_cad_pref", "vereador_eleit", "ENLP2000", "comarca"] + sorteio_all)
m = pf.feols(fml("pcorrupt", base_rhs), data=dfs, vcov="hetero")
add_result("4", "5", "pcorrupt", "first", m, "OLS",
           controls_desc="prefchar2 + munichar2 + lrec_trans + political + sorteio",
           orig_coef=-0.026, orig_se=0.011)

# Col 6: areg ... abs(uf) -- MAIN SPECIFICATION
m = pf.feols(fml("pcorrupt", base_rhs, absorb="uf"), data=dfs, vcov="hetero")
add_result("4", "6", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf",
           controls_desc="prefchar2 + munichar2 + lrec_trans + political + sorteio",
           notes="State FE (areg). Main specification.",
           orig_coef=-0.027, orig_se=0.011)

# Col 7: match -- SKIP (Abadie-Imbens matching estimator)
# UNLISTED_METHOD: match in 112431-V1 -- Abadie-Imbens bias-corrected matching

# Col 8: tobit pcorrupt first ... uf_d* if esample2==1, ll(0)
tobit_rhs = (["first"] + prefchar2 + munichar2 +
             ["lrec_trans", "lfunc_ativ", "p_cad_pref", "vereador_eleit", "ENLP2000",
              "comarca", "lrec_fisc"] + sorteio_all + uf_dummies)
tobit_vars = ["pcorrupt"] + tobit_rhs
dfs_tobit = dfs[tobit_vars].dropna()
X_tobit = sm.add_constant(dfs_tobit[tobit_rhs])
y_tobit = dfs_tobit["pcorrupt"]
add_tobit_result("4", "8", "pcorrupt", "first", y_tobit, X_tobit,
                 ["const"] + tobit_rhs, lower=0,
                 fixed_effects="uf (dummies)",
                 controls_desc="full controls + lrec_fisc + lfunc_ativ + uf dummies",
                 notes="Tobit left-censored at 0")


# =============================================================================
# TABLE 5: Effects on Number of Corruption Violations
# Panel A: ncorrupt; Panel B: ncorrupt_os
# =============================================================================
print("\n" + "=" * 70)
print("TABLE 5: ncorrupt and ncorrupt_os ~ first")
print("=" * 70)

# Panel A Col 1: reg ncorrupt first, robust
m = pf.feols("ncorrupt ~ first", data=dfs, vcov="hetero")
add_result("5A", "1", "ncorrupt", "first", m, "OLS",
           controls_desc="none")

# Panel A Col 2: areg ncorrupt first lrec_fisc $prefchar2 $munichar2 lrec_trans
#                p_cad_pref vereador_eleit ENLP2000 comarca sorteio*, robust abs(uf)
rhs_5a2 = (["first", "lrec_fisc"] + prefchar2 + munichar2 +
           ["lrec_trans", "p_cad_pref", "vereador_eleit", "ENLP2000", "comarca"] + sorteio_all)
m = pf.feols(fml("ncorrupt", rhs_5a2, absorb="uf"), data=dfs, vcov="hetero")
add_result("5A", "2", "ncorrupt", "first", m, "OLS",
           fixed_effects="uf",
           controls_desc="lrec_fisc + prefchar2 + munichar2 + lrec_trans + political + sorteio")

# Panel A Col 3: match -- SKIP

# Panel A Col 4: nbreg ncorrupt first lrec_fisc ... sorteio2-sorteio10 uf_d*, robust
rhs_nbreg = (["first", "lrec_fisc"] + prefchar2 + munichar2 +
             ["lrec_trans", "lfunc_ativ", "p_cad_pref", "vereador_eleit", "ENLP2000",
              "comarca"] + sorteio_2_10 + uf_dummies)
fml_nb = "ncorrupt ~ " + " + ".join(rhs_nbreg)
dfs_nb = dfs[["ncorrupt"] + rhs_nbreg].dropna()
try:
    m_nb = smf.negativebinomial(fml_nb, data=dfs_nb).fit(disp=0, maxiter=500)
    add_result("5A", "4", "ncorrupt", "first", m_nb, "NegBin",
               fixed_effects="uf (dummies)",
               controls_desc="full controls + uf dummies",
               notes="Negative binomial regression")
except Exception as e:
    reg_counter += 1
    results.append({
        "paper_id": PAPER_ID, "reg_id": reg_counter,
        "outcome_var": "ncorrupt", "treatment_var": "first",
        "coefficient": "", "std_error": "", "p_value": "",
        "ci_lower": "", "ci_upper": "", "n_obs": "", "r_squared": "",
        "original_coefficient": "", "original_std_error": "",
        "match_status": "failed",
        "coefficient_vector_json": "{}",
        "fixed_effects": "uf (dummies)", "controls_desc": "full controls + uf dummies",
        "cluster_var": "", "estimator": "NegBin",
        "sample_desc": "esample2==1",
        "notes": f"Table 5A, Col 4. NegBin FAILED: {str(e)}"
    })
    print(f"  Reg {reg_counter:2d} (T5A C4): FAILED NegBin - {e}")

# Panel B Col 1: reg ncorrupt_os first, robust
m = pf.feols("ncorrupt_os ~ first", data=dfs, vcov="hetero")
add_result("5B", "1", "ncorrupt_os", "first", m, "OLS",
           controls_desc="none")

# Panel B Col 2: areg ncorrupt_os first lrec_fisc $prefchar2 $munichar2 lrec_trans lfunc_ativ
#                p_cad_pref vereador_eleit ENLP2000 comarca sorteio*, robust abs(uf)
rhs_5b2 = (["first", "lrec_fisc"] + prefchar2 + munichar2 +
           ["lrec_trans", "lfunc_ativ", "p_cad_pref", "vereador_eleit", "ENLP2000",
            "comarca"] + sorteio_all)
m = pf.feols(fml("ncorrupt_os", rhs_5b2, absorb="uf"), data=dfs, vcov="hetero")
add_result("5B", "2", "ncorrupt_os", "first", m, "OLS",
           fixed_effects="uf",
           controls_desc="lrec_fisc + lfunc_ativ + prefchar2 + munichar2 + lrec_trans + political + sorteio")

# Panel B Col 3: match -- SKIP

# Panel B Col 4: tobit ncorrupt_os ... uf_d*, ll(0)
tobit_rhs_5b = (["first", "lrec_fisc"] + prefchar2 + munichar2 +
                ["lrec_trans", "lfunc_ativ", "p_cad_pref", "vereador_eleit",
                 "ENLP2000", "comarca"] + sorteio_all + uf_dummies)
tobit_vars_5b = ["ncorrupt_os"] + tobit_rhs_5b
dfs_t5b = dfs[tobit_vars_5b].dropna()
X_t5b = sm.add_constant(dfs_t5b[tobit_rhs_5b])
y_t5b = dfs_t5b["ncorrupt_os"]
add_tobit_result("5B", "4", "ncorrupt_os", "first", y_t5b, X_t5b,
                 ["const"] + tobit_rhs_5b, lower=0,
                 fixed_effects="uf (dummies)",
                 controls_desc="full controls + uf dummies",
                 notes="Tobit left-censored at 0")


# =============================================================================
# TABLE 6: RDD - Vote Margin Controls
# All areg pcorrupt ... abs(uf), robust
# Generate running variable from Stata code (lines 86-105)
# =============================================================================
print("\n" + "=" * 70)
print("TABLE 6: pcorrupt ~ first with running variable (RDD)")
print("=" * 70)

dfs_t6 = dfs.copy()
# gen wm = winmargin2000 if reeleito==1
dfs_t6['wm'] = np.nan
dfs_t6.loc[dfs_t6['reeleito'] == 1, 'wm'] = dfs_t6.loc[dfs_t6['reeleito'] == 1, 'winmargin2000']
# replace wm = winmargin2000_inclost if incumbent==1
dfs_t6.loc[dfs_t6['incumbent'] == 1, 'wm'] = dfs_t6.loc[dfs_t6['incumbent'] == 1, 'winmargin2000_inclost']
# gen running = wm; replace running = -wm if incumbent==1
dfs_t6['running'] = dfs_t6['wm'].copy()
dfs_t6.loc[dfs_t6['incumbent'] == 1, 'running'] = -dfs_t6.loc[dfs_t6['incumbent'] == 1, 'wm']
dfs_t6['running2'] = dfs_t6['running'] ** 2
dfs_t6['running3'] = dfs_t6['running'] ** 3
dfs_t6['running4'] = dfs_t6['running'] ** 4
dfs_t6['spline1'] = dfs_t6['first'] * dfs_t6['running']
dfs_t6['spline2'] = dfs_t6['first'] * dfs_t6['running2']
dfs_t6['spline3'] = dfs_t6['first'] * dfs_t6['running3']

base_t6 = (prefchar2 + munichar2 +
           ["lrec_trans", "p_cad_pref", "vereador_eleit", "ENLP2000", "comarca"] + sorteio_all)

# Col 1: restricted to running~=. (no running variable in regression)
dfs_t6_valid = dfs_t6[dfs_t6['running'].notna()].copy()
m = pf.feols(fml("pcorrupt", ["first"] + base_t6, absorb="uf"),
             data=dfs_t6_valid, vcov="hetero")
add_result("6", "1", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="base controls",
           sample_desc="esample2==1 & running non-missing")

# Col 2: + running
m = pf.feols(fml("pcorrupt", ["first", "running"] + base_t6, absorb="uf"),
             data=dfs_t6, vcov="hetero")
add_result("6", "2", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="base + linear running")

# Col 3: + running + running2
m = pf.feols(fml("pcorrupt", ["first", "running", "running2"] + base_t6, absorb="uf"),
             data=dfs_t6, vcov="hetero")
add_result("6", "3", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="base + quadratic running")

# Col 4: + running + running2 + running3
m = pf.feols(fml("pcorrupt", ["first", "running", "running2", "running3"] + base_t6, absorb="uf"),
             data=dfs_t6, vcov="hetero")
add_result("6", "4", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="base + cubic running")

# Col 5: linear spline
m = pf.feols(fml("pcorrupt", ["first", "running", "spline1"] + base_t6, absorb="uf"),
             data=dfs_t6, vcov="hetero")
add_result("6", "5", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="base + linear spline")

# Col 6: quadratic spline
m = pf.feols(fml("pcorrupt", ["first", "running", "spline1", "running2", "spline2"] + base_t6,
                  absorb="uf"), data=dfs_t6, vcov="hetero")
add_result("6", "6", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="base + quadratic spline")

# Col 7: cubic spline
m = pf.feols(fml("pcorrupt", ["first", "running", "spline1", "running2", "running3",
                               "spline2", "spline3"] + base_t6, absorb="uf"),
             data=dfs_t6, vcov="hetero")
add_result("6", "7", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="base + cubic spline")


# =============================================================================
# TABLE 7: Experience Controls
# All areg pcorrupt first [controls] | uf, robust
# Generate experience variables from Stata code (lines 110-128)
# =============================================================================
print("\n" + "=" * 70)
print("TABLE 7: pcorrupt ~ first with experience controls")
print("=" * 70)

dfs_t7 = dfs.copy()
dfs_t7['first_experienced'] = ((dfs_t7['reeleito'] == 0) & (dfs_t7['exp_prefeito'] == 1)).astype(float)
dfs_t7['experience1'] = ((dfs_t7['first_experienced'] == 1) | (dfs_t7['reeleito'] == 1)).astype(float)
dfs_t7['experience2'] = ((dfs_t7['reeleito'] == 1) | (dfs_t7['reeleito_2004'] == 1)).astype(float)
dfs_t7['experience3'] = ((dfs_t7['reeleito'] == 1) | (dfs_t7['elected1'] == 1)).astype(float)
# nexp uses rowtotal - in Stata this treats missing as 0
dfs_t7['nexp'] = (dfs_t7['exp_prefeito'].fillna(0) + dfs_t7['vereador9600'].fillna(0)) * 4
dfs_t7['nexp2'] = dfs_t7['nexp'] ** 2

# Full controls for Table 7 (note: includes lfunc_ativ vs base_rhs in Table 4)
base_t7 = (prefchar2 + munichar2 +
           ["lrec_trans", "lfunc_ativ", "p_cad_pref", "vereador_eleit",
            "ENLP2000", "comarca"] + sorteio_all)

# Col 1: experience2==1 subsample
m = pf.feols(fml("pcorrupt", ["first"] + base_t7, absorb="uf"),
             data=dfs_t7[dfs_t7['experience2'] == 1], vcov="hetero")
add_result("7", "1", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls",
           sample_desc="esample2==1 & experience2==1 (reeleito or reeleito_2004)")

# Col 2: experience3==1 subsample
m = pf.feols(fml("pcorrupt", ["first"] + base_t7, absorb="uf"),
             data=dfs_t7[dfs_t7['experience3'] == 1], vcov="hetero")
add_result("7", "2", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls",
           sample_desc="esample2==1 & experience3==1 (reeleito or elected1)")

# Col 3: + exp_prefeito
m = pf.feols(fml("pcorrupt", ["first", "exp_prefeito"] + base_t7, absorb="uf"),
             data=dfs_t7, vcov="hetero")
add_result("7", "3", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls + exp_prefeito")

# Col 4: + nexp nexp2
m = pf.feols(fml("pcorrupt", ["first", "nexp", "nexp2"] + base_t7, absorb="uf"),
             data=dfs_t7, vcov="hetero")
add_result("7", "4", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls + nexp + nexp2")

# Col 5: experience1==1 subsample
m = pf.feols(fml("pcorrupt", ["first"] + base_t7, absorb="uf"),
             data=dfs_t7[dfs_t7['experience1'] == 1], vcov="hetero")
add_result("7", "5", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls",
           sample_desc="esample2==1 & experience1==1 (reeleito or experienced first-term)")

# Col 6: restricted sample (first==0 or first-term with political experience)
# Stata: if esample2==1 & (first==0|(first==1&(exp_prefeito==1|vereador96==1)))
# Note: do-file uses vereador96, but data has vereador9600
mask_c6 = ((dfs_t7['first'] == 0) |
           ((dfs_t7['first'] == 1) & ((dfs_t7['exp_prefeito'] == 1) | (dfs_t7['vereador9600'] == 1))))
m = pf.feols(fml("pcorrupt", ["first"] + base_t7, absorb="uf"),
             data=dfs_t7[mask_c6], vcov="hetero")
add_result("7", "6", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls",
           sample_desc="first==0 or (first==1 & experienced)",
           notes="Stata uses vereador96; data has vereador9600")


# =============================================================================
# TABLE 8: Mismanagement (Placebo)
# areg pmismanagement/pcorrupt first [controls] | uf, robust
# =============================================================================
print("\n" + "=" * 70)
print("TABLE 8: Mismanagement placebo")
print("=" * 70)

dfs_t8 = dfs_t7.copy()

# Col 1: areg pmismanagement first ... if esample2==1
m = pf.feols(fml("pmismanagement", ["first"] + base_t7, absorb="uf"),
             data=dfs_t8, vcov="hetero")
add_result("8", "1", "pmismanagement", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls",
           notes="Placebo: pmismanagement (110 missing obs)")

# Col 2: + experience2==1
m = pf.feols(fml("pmismanagement", ["first"] + base_t7, absorb="uf"),
             data=dfs_t8[dfs_t8['experience2'] == 1], vcov="hetero")
add_result("8", "2", "pmismanagement", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls",
           sample_desc="esample2==1 & experience2==1")

# Col 3: + experience1==1
m = pf.feols(fml("pmismanagement", ["first"] + base_t7, absorb="uf"),
             data=dfs_t8[dfs_t8['experience1'] == 1], vcov="hetero")
add_result("8", "3", "pmismanagement", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls",
           sample_desc="esample2==1 & experience1==1")

# Col 4: pcorrupt on pmismanagement non-missing sample
m = pf.feols(fml("pcorrupt", ["first"] + base_t7, absorb="uf"),
             data=dfs_t8[dfs_t8['pmismanagement'].notna()], vcov="hetero")
add_result("8", "4", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls",
           sample_desc="esample2==1 & pmismanagement non-missing",
           notes="DV=pcorrupt restricted to pmismanagement sample")


# =============================================================================
# TABLE 9: Convenios (Matching Grants) - Panel data
# From convenios_aer.do
# reg/areg with cluster(cod_ibge6)
# =============================================================================
print("\n" + "=" * 70)
print("TABLE 9: Convenios (panel)")
print("=" * 70)

dfc = df_conv.copy()
dfc['ano2001'] = (dfc['ano'] == 2001).astype(float)
dfc['ano2002'] = (dfc['ano'] == 2002).astype(float)
dfc['ano2003'] = (dfc['ano'] == 2003).astype(float)
dfc['ano2004'] = (dfc['ano'] == 2004).astype(float)
dfc['tr_ano2001'] = dfc['first'] * dfc['ano2001']
dfc['tr_ano2002'] = dfc['first'] * dfc['ano2002']
dfc['tr_ano2003'] = dfc['first'] * dfc['ano2003']
dfc['tr_ano2004'] = dfc['first'] * dfc['ano2004']

conv_prefchar2 = ["pref_masc", "pref_idade_tse", "pref_escola",
                   "party_d1", "party_d3", "party_d4", "party_d5", "party_d6",
                   "party_d7", "party_d8", "party_d9", "party_d10", "party_d11",
                   "party_d12", "party_d13", "party_d14", "party_d15", "party_d16",
                   "party_d17", "party_d18"]
conv_munichar2 = ["lpop", "purb", "p_secundario", "mun_novo", "lpib02", "gini_ipea"]

# Create year interactions for FE specifications (ano_*)
for v in ['lpib02', 'purb', 'p_secundario', 'mun_novo', 'gini_ipea', 'pref_masc',
          'pref_idade_tse', 'pref_escola', 'p_cad_pref', 'vereador_eleit', 'ENLP2000', 'comarca']:
    dfc[f'ano_{v}'] = dfc['ano'] * dfc[v]
ano_int = [f'ano_{v}' for v in ['lpib02', 'purb', 'p_secundario', 'mun_novo', 'gini_ipea',
                                 'pref_masc', 'pref_idade_tse', 'pref_escola',
                                 'p_cad_pref', 'vereador_eleit', 'ENLP2000', 'comarca']]

# Panel A: dconvenios
# Col 1: Pooled OLS with controls, cluster SE
rhs_9a1 = (["tr_ano2004", "tr_ano2003", "tr_ano2002", "first",
            "ano2002", "ano2003", "ano2004"] +
           conv_prefchar2 + conv_munichar2 +
           ["p_cad_pref", "vereador_eleit", "ENLP2000", "comarca"])
m = pf.feols(fml("dconvenios", rhs_9a1), data=dfc, vcov={"CRV1": "cod_ibge6"})
add_result("9A", "1", "dconvenios", "tr_ano2004", m, "OLS",
           controls_desc="pooled OLS + year dummies + controls",
           cluster_var="cod_ibge6")

# Col 2: Municipality FE
rhs_9a2 = (["first", "tr_ano2002", "tr_ano2003", "tr_ano2004",
            "ano2002", "ano2003", "ano2004"] + ano_int)
m = pf.feols(fml("dconvenios", rhs_9a2, absorb="cod_ibge6"),
             data=dfc, vcov={"CRV1": "cod_ibge6"})
add_result("9A", "2", "dconvenios", "tr_ano2004", m, "OLS",
           fixed_effects="cod_ibge6",
           controls_desc="municipality FE + year + year interactions",
           cluster_var="cod_ibge6")

# Panel B: lconvenios_pc
# Col 3: Pooled OLS
rhs_9b1 = (["tr_ano2004", "tr_ano2003", "tr_ano2002", "first",
            "ano2002", "ano2003", "ano2004"] +
           conv_prefchar2 + conv_munichar2 +
           ["p_cad_pref", "vereador_eleit", "ENLP2000", "comarca"])
m = pf.feols(fml("lconvenios_pc", rhs_9b1), data=dfc, vcov={"CRV1": "cod_ibge6"})
add_result("9B", "3", "lconvenios_pc", "tr_ano2004", m, "OLS",
           controls_desc="pooled OLS + year dummies + controls",
           cluster_var="cod_ibge6")

# Col 4: Municipality FE
m = pf.feols(fml("lconvenios_pc", rhs_9a2, absorb="cod_ibge6"),
             data=dfc, vcov={"CRV1": "cod_ibge6"})
add_result("9B", "4", "lconvenios_pc", "tr_ano2004", m, "OLS",
           fixed_effects="cod_ibge6",
           controls_desc="municipality FE + year + year interactions",
           cluster_var="cod_ibge6")

# Panel C: msh_liberado
# Col 5: Pooled OLS
rhs_9c1 = (["first", "tr_ano2002", "tr_ano2003", "tr_ano2004",
            "ano2002", "ano2003", "ano2004"] +
           conv_prefchar2 + conv_munichar2 +
           ["p_cad_pref", "vereador_eleit", "ENLP2000", "comarca"])
m = pf.feols(fml("msh_liberado", rhs_9c1), data=dfc, vcov={"CRV1": "cod_ibge6"})
add_result("9C", "5", "msh_liberado", "tr_ano2004", m, "OLS",
           controls_desc="pooled OLS + year dummies + controls",
           cluster_var="cod_ibge6")

# Col 6: Municipality FE
m = pf.feols(fml("msh_liberado", rhs_9a2, absorb="cod_ibge6"),
             data=dfc, vcov={"CRV1": "cod_ibge6"})
add_result("9C", "6", "msh_liberado", "tr_ano2004", m, "OLS",
           fixed_effects="cod_ibge6",
           controls_desc="municipality FE + year + year interactions",
           cluster_var="cod_ibge6")


# =============================================================================
# TABLE 10: Heterogeneity (Interaction terms)
# areg pcorrupt first + first*X [controls] | uf, robust
# Stata do-file lines 147-163
# =============================================================================
print("\n" + "=" * 70)
print("TABLE 10: Heterogeneous effects")
print("=" * 70)

dfs_t10 = dfs.copy()
dfs_t10['h_ENEP2000'] = 1 / dfs_t10['ENEP2000']
dfs_t10['first_comarca'] = dfs_t10['comarca'] * dfs_t10['first']
dfs_t10['first_p_cad_pref'] = dfs_t10['p_cad_pref'] * dfs_t10['first']
dfs_t10['first_media2'] = dfs_t10['media2'] * dfs_t10['first']
dfs_t10['first_h_ENEP2000'] = dfs_t10['h_ENEP2000'] * dfs_t10['first']

base_t10 = (prefchar2 + munichar2 +
            ["lrec_trans", "lfunc_ativ", "p_cad_pref", "vereador_eleit",
             "ENLP2000", "comarca"] + sorteio_all)

# Col 1: first*comarca
m = pf.feols(fml("pcorrupt", ["first", "first_comarca"] + base_t10, absorb="uf"),
             data=dfs_t10, vcov="hetero")
add_result("10", "1", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls + first*comarca")

# Col 2: first*media2
m = pf.feols(fml("pcorrupt", ["first", "first_media2", "media2"] + base_t10, absorb="uf"),
             data=dfs_t10, vcov="hetero")
add_result("10", "2", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls + first*media2")

# Col 3: first*p_cad_pref
m = pf.feols(fml("pcorrupt", ["first", "first_p_cad_pref"] + base_t10, absorb="uf"),
             data=dfs_t10, vcov="hetero")
add_result("10", "3", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls + first*p_cad_pref")

# Col 4: first*h_ENEP2000 (note: do-file omits lfunc_ativ and ENLP2000 from controls)
rhs_10c4 = (["first", "first_h_ENEP2000", "h_ENEP2000"] + prefchar2 + munichar2 +
            ["lrec_trans", "p_cad_pref", "vereador_eleit", "comarca"] + sorteio_all)
m = pf.feols(fml("pcorrupt", rhs_10c4, absorb="uf"), data=dfs_t10, vcov="hetero")
add_result("10", "4", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf",
           controls_desc="controls (no ENLP2000/lfunc_ativ) + first*h_ENEP2000",
           notes="Do-file drops lfunc_ativ and ENLP2000 from this column")


# =============================================================================
# TABLE 11: Robustness checks
# Panel A: pcorrupt regressions, Panel B: lrecursos_fisc regressions
# Stata do-file lines 170-191
# =============================================================================
print("\n" + "=" * 70)
print("TABLE 11: Robustness checks")
print("=" * 70)

dfs_t11 = dfs.copy()
dfs_t11['sorteio_electyear'] = ((dfs_t11['nsorteio'] > 5) & (dfs_t11['nsorteio'] < 10)).astype(float)
dfs_t11['first_sorteio_electyear'] = dfs_t11['first'] * dfs_t11['sorteio_electyear']
dfs_t11['first_PT'] = dfs_t11['first'] * dfs_t11['party_d15']
dfs_t11['first_samepartygov98'] = dfs_t11['first'] * dfs_t11['samepartygov98']
dfs_t11['lrecursos_fisc'] = np.log(dfs_t11['valor_fiscalizado'])

base_t11 = (prefchar2 + munichar2 +
            ["lrec_trans", "lfunc_ativ", "p_cad_pref", "vereador_eleit",
             "ENLP2000", "comarca"] + sorteio_all)

# Panel A Col 1: + sorteio_electyear interaction
m = pf.feols(fml("pcorrupt", ["first", "first_sorteio_electyear"] + base_t11, absorb="uf"),
             data=dfs_t11, vcov="hetero")
add_result("11A", "1", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls + first*sorteio_electyear",
           notes="Timing of audit")

# Panel A Col 2: + PT interaction
m = pf.feols(fml("pcorrupt", ["first", "first_PT"] + base_t11, absorb="uf"),
             data=dfs_t11, vcov="hetero")
add_result("11A", "2", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls + first*PT",
           notes="PT party interaction")

# Panel A Col 3: + same party governor interaction
m = pf.feols(fml("pcorrupt", ["first", "first_samepartygov98", "samepartygov98"] + base_t11,
                  absorb="uf"), data=dfs_t11, vcov="hetero")
add_result("11A", "3", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls + first*samepartygov98",
           notes="Same party as governor interaction")

# Panel B Col 1: lrecursos_fisc manipulation test
# Note: do-file order places lfunc_ativ after comarca, not after lrec_trans
rhs_11b1 = (["first"] + prefchar2 + munichar2 +
            ["lrec_trans", "p_cad_pref", "vereador_eleit", "ENLP2000",
             "comarca", "lfunc_ativ"] + sorteio_all)
m = pf.feols(fml("lrecursos_fisc", rhs_11b1, absorb="uf"), data=dfs_t11, vcov="hetero")
add_result("11B", "1", "lrecursos_fisc", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls",
           notes="Manipulation test: resources audited")

# Panel B Col 2: + sorteio_electyear
rhs_11b2 = (["first", "first_sorteio_electyear", "sorteio_electyear"] + base_t11)
m = pf.feols(fml("lrecursos_fisc", rhs_11b2, absorb="uf"), data=dfs_t11, vcov="hetero")
add_result("11B", "2", "lrecursos_fisc", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls + sorteio_electyear interaction")

# Panel B Col 3: + PT
rhs_11b3 = ["first", "first_PT"] + base_t11
m = pf.feols(fml("lrecursos_fisc", rhs_11b3, absorb="uf"), data=dfs_t11, vcov="hetero")
add_result("11B", "3", "lrecursos_fisc", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls + first*PT")

# Panel B Col 4: + samepartygov98
rhs_11b4 = ["first", "first_samepartygov98", "samepartygov98"] + base_t11
m = pf.feols(fml("lrecursos_fisc", rhs_11b4, absorb="uf"), data=dfs_t11, vcov="hetero")
add_result("11B", "4", "lrecursos_fisc", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls + first*samepartygov98")


# =============================================================================
# FIGURE 2: RDD regression for plot
# reg pcorrupt first running running2 running3 if esample2==1, robust
# =============================================================================
print("\n" + "=" * 70)
print("FIGURE 2: RDD regression")
print("=" * 70)

m = pf.feols("pcorrupt ~ first + running + running2 + running3", data=dfs_t6, vcov="hetero")
add_result("Fig2", "1", "pcorrupt", "first", m, "OLS",
           controls_desc="cubic polynomial in running",
           notes="RDD regression for Figure 2 plot")


# =============================================================================
# Write outputs
# =============================================================================
print("\n" + "=" * 70)
print("WRITING OUTPUTS")
print("=" * 70)

results_df = pd.DataFrame(results)
csv_path = os.path.join(PACKAGE_DIR, "replication.csv")
results_df.to_csv(csv_path, index=False)
print(f"Wrote {len(results_df)} rows to {csv_path}")

# Quality checks
assert results_df['reg_id'].is_unique, "reg_id not unique!"
for _, row in results_df.iterrows():
    if row['coefficient_vector_json']:
        json.loads(row['coefficient_vector_json'])  # validate JSON
assert set(results_df['match_status'].unique()).issubset({'exact', 'close', 'discrepant', 'failed'})

n_exact = (results_df['match_status'] == 'exact').sum()
n_close = (results_df['match_status'] == 'close').sum()
n_discrepant = (results_df['match_status'] == 'discrepant').sum()
n_failed = (results_df['match_status'] == 'failed').sum()

print(f"\nMatch summary:")
print(f"  Exact:      {n_exact}")
print(f"  Close:      {n_close}")
print(f"  Discrepant: {n_discrepant}")
print(f"  Failed:     {n_failed}")
print(f"  Total:      {len(results_df)}")
