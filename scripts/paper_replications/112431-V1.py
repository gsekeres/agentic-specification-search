"""
Replication script for Ferraz & Finan (2011)
"Electoral Accountability and Corruption: Evidence from the Audits of Local Governments"
American Economic Review, 101(4), 1274-1311.

Paper ID: 112431-V1

This script replicates the main regression results from Tables 4-11 and Figure 2.
Original code: Stata do-files. Translated to Python using pyfixest and statsmodels.

Tables 1, 3 are descriptive/balance (not regression tables).
Table 2 has descriptive regressions (no controls). We skip these.
Tables 4-8, 10-11 are from reelection_aer.do (main results).
Table 9 is from convenios_aer.do (panel matching grants).
Match commands (Abadie-Imbens) are skipped as they are not standard regressions.
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

# Filter to estimation sample (all 476 obs)
dfs = df[df['esample2'] == 1].copy()

# =============================================================================
# Define variable groups (matching Stata globals)
# =============================================================================
prefchar2 = ["pref_masc", "pref_idade_tse", "pref_escola",
             "party_d1", "party_d3", "party_d4", "party_d5", "party_d6",
             "party_d7", "party_d8", "party_d9", "party_d10", "party_d11",
             "party_d12", "party_d13", "party_d14", "party_d15", "party_d16",
             "party_d17", "party_d18"]

munichar2 = ["lpop", "purb", "p_secundario", "mun_novo", "lpib02", "gini_ipea"]

sorteio_all = [f"sorteio{i}" for i in range(1, 11)]
sorteio_2_10 = [f"sorteio{i}" for i in range(2, 11)]

uf_dummies = [c for c in df.columns if c.startswith("uf_d")]

# =============================================================================
# Published paper coefficients (from Table values, 3 decimal places)
# These are the "first" coefficient from each regression as published.
# Where the paper reports values rounded to 3dp, we use those.
# =============================================================================

# Table 4: pcorrupt ~ first
# Paper reports: Col1=-0.019, Col2=-0.020, Col3=-0.024, Col4=-0.025, Col5=-0.024, Col6=-0.027, Col8(tobit)=-0.034
# Table 5 Panel A: ncorrupt ~ first
# Paper: Col1=-0.592, Col2=-0.539, Col4(nbreg)=-0.282
# Table 5 Panel B: ncorrupt_os ~ first
# Paper: Col1=-0.013, Col2=-0.013, Col4(tobit)=-0.018

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

        # Match status: compare rounded-to-3dp values
        if orig_coef is not None:
            if abs(orig_coef) > 1e-10:
                rel_err = abs(coef - orig_coef) / abs(orig_coef)
            else:
                rel_err = abs(coef - orig_coef)
            # Check if they round to the same 3dp value
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
            print(f"         orig={orig_coef} rel_err={rel_err:.4f}")

    except Exception as e:
        results.append({
            "paper_id": PAPER_ID,
            "reg_id": reg_counter,
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
    """Run tobit and add results."""
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

        # Numerical Hessian for SEs
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
        except:
            se_all = np.full(k_total, np.nan)

        beta = res.x[:-1]
        se = se_all[:-1]

        first_idx = col_names.index(treatment_var)
        coef = float(beta[first_idx])
        se_val = float(se[first_idx])
        pval = float(2 * (1 - norm.cdf(abs(coef / se_val))))
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
            "p_value": round(pval, 8),
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
              f"coef={coef:.6f} se={se_val:.6f} p={pval:.4f} N={n} | {match_status}")
        if orig_coef is not None:
            print(f"         orig={orig_coef} rel_err={rel_err:.4f}")

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
# Paper Table 4: DV = pcorrupt, treatment = first
# Published values (from AER 2011, Table 4):
#   Col1: -0.019(0.009), Col2: -0.020(0.010), Col3: -0.024(0.010)
#   Col4: -0.025(0.010), Col5: -0.024(0.010), Col6: -0.027(0.011)
#   Col7: matching (skip), Col8: tobit -0.034(0.013)
# =============================================================================
print("\n" + "="*70)
print("TABLE 4: pcorrupt ~ first")
print("="*70)

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
           orig_coef=-0.024, orig_se=0.010)

# Col 4: + political controls
m = pf.feols(fml("pcorrupt", ["first"] + prefchar2 + munichar2 +
                  ["lrec_trans", "p_cad_pref", "vereador_eleit", "ENLP2000", "comarca"]),
             data=dfs, vcov="hetero")
add_result("4", "4", "pcorrupt", "first", m, "OLS",
           controls_desc="prefchar2 + munichar2 + lrec_trans + political",
           orig_coef=-0.025, orig_se=0.010)

# Col 5: + lottery dummies
base_rhs = ["first"] + prefchar2 + munichar2 + \
           ["lrec_trans", "p_cad_pref", "vereador_eleit", "ENLP2000", "comarca"] + sorteio_all
m = pf.feols(fml("pcorrupt", base_rhs), data=dfs, vcov="hetero")
add_result("4", "5", "pcorrupt", "first", m, "OLS",
           controls_desc="prefchar2 + munichar2 + lrec_trans + political + sorteio",
           orig_coef=-0.024, orig_se=0.010)

# Col 6: areg ... abs(uf) -- main specification
m = pf.feols(fml("pcorrupt", base_rhs, absorb="uf"), data=dfs, vcov="hetero")
add_result("4", "6", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf",
           controls_desc="prefchar2 + munichar2 + lrec_trans + political + sorteio",
           notes="State FE (areg). Main specification.",
           orig_coef=-0.027, orig_se=0.011)

# Col 7: match -- SKIP (Abadie-Imbens matching)

# Col 8: tobit, ll(0)
tobit_rhs = ["first"] + prefchar2 + munichar2 + \
            ["lrec_trans", "lfunc_ativ", "p_cad_pref", "vereador_eleit", "ENLP2000",
             "comarca", "lrec_fisc"] + sorteio_all + uf_dummies
tobit_vars = ["pcorrupt"] + tobit_rhs
dfs_tobit = dfs[tobit_vars].dropna()
X_tobit = sm.add_constant(dfs_tobit[tobit_rhs])
y_tobit = dfs_tobit["pcorrupt"]
add_tobit_result("4", "8", "pcorrupt", "first", y_tobit, X_tobit,
                 ["const"] + tobit_rhs, lower=0,
                 fixed_effects="uf (dummies)",
                 controls_desc="full controls + lrec_fisc + lfunc_ativ + uf dummies",
                 notes="Tobit left-censored at 0",
                 orig_coef=-0.034, orig_se=0.013)


# =============================================================================
# TABLE 5: Effects on Number of Corruption Violations
# Panel A: ncorrupt; Panel B: ncorrupt_os
# Published: PanA C1: -0.592(0.158), C2: -0.539(0.183), C4(nbreg): -0.282(0.104)
#            PanB C1: -0.013(0.005), C2: -0.013(0.005), C4(tobit): -0.018(0.007)
# =============================================================================
print("\n" + "="*70)
print("TABLE 5: ncorrupt and ncorrupt_os ~ first")
print("="*70)

# Panel A Col 1: reg ncorrupt first, robust
m = pf.feols("ncorrupt ~ first", data=dfs, vcov="hetero")
add_result("5A", "1", "ncorrupt", "first", m, "OLS",
           controls_desc="none",
           orig_coef=-0.592, orig_se=0.158)

# Panel A Col 2: areg ncorrupt first lrec_fisc $prefchar2 $munichar2 lrec_trans
#                p_cad_pref vereador_eleit ENLP2000 comarca sorteio*, robust abs(uf)
rhs_5a2 = ["first", "lrec_fisc"] + prefchar2 + munichar2 + \
          ["lrec_trans", "p_cad_pref", "vereador_eleit", "ENLP2000", "comarca"] + sorteio_all
m = pf.feols(fml("ncorrupt", rhs_5a2, absorb="uf"), data=dfs, vcov="hetero")
add_result("5A", "2", "ncorrupt", "first", m, "OLS",
           fixed_effects="uf",
           controls_desc="lrec_fisc + prefchar2 + munichar2 + lrec_trans + political + sorteio",
           orig_coef=-0.539, orig_se=0.183)

# Panel A Col 3: match -- SKIP

# Panel A Col 4: nbreg ncorrupt first lrec_fisc $prefchar2 $munichar2 lrec_trans lfunc_ativ
#                p_cad_pref vereador_eleit ENLP2000 comarca sorteio2-sorteio10 uf_d*, robust
rhs_nbreg = ["first", "lrec_fisc"] + prefchar2 + munichar2 + \
            ["lrec_trans", "lfunc_ativ", "p_cad_pref", "vereador_eleit", "ENLP2000",
             "comarca"] + sorteio_2_10 + uf_dummies
fml_nb = "ncorrupt ~ " + " + ".join(rhs_nbreg)
dfs_nb = dfs[["ncorrupt"] + rhs_nbreg].dropna()
try:
    m_nb = smf.negativebinomial(fml_nb, data=dfs_nb).fit(disp=0, maxiter=500)
    add_result("5A", "4", "ncorrupt", "first", m_nb, "NegBin",
               fixed_effects="uf (dummies)",
               controls_desc="full controls + uf dummies",
               notes="Negative binomial regression",
               orig_coef=-0.282, orig_se=0.104)
except Exception as e:
    reg_counter += 1
    results.append({
        "paper_id": PAPER_ID, "reg_id": reg_counter,
        "outcome_var": "ncorrupt", "treatment_var": "first",
        "coefficient": "", "std_error": "", "p_value": "",
        "ci_lower": "", "ci_upper": "", "n_obs": "", "r_squared": "",
        "original_coefficient": -0.282, "original_std_error": 0.104,
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
           controls_desc="none",
           orig_coef=-0.013, orig_se=0.005)

# Panel B Col 2: areg ncorrupt_os first lrec_fisc $prefchar2 $munichar2 lrec_trans lfunc_ativ
#                p_cad_pref vereador_eleit ENLP2000 comarca sorteio*, robust abs(uf)
rhs_5b2 = ["first", "lrec_fisc"] + prefchar2 + munichar2 + \
          ["lrec_trans", "lfunc_ativ", "p_cad_pref", "vereador_eleit", "ENLP2000",
           "comarca"] + sorteio_all
m = pf.feols(fml("ncorrupt_os", rhs_5b2, absorb="uf"), data=dfs, vcov="hetero")
add_result("5B", "2", "ncorrupt_os", "first", m, "OLS",
           fixed_effects="uf",
           controls_desc="lrec_fisc + lfunc_ativ + prefchar2 + munichar2 + lrec_trans + political + sorteio",
           orig_coef=-0.013, orig_se=0.005)

# Panel B Col 3: match -- SKIP

# Panel B Col 4: tobit ncorrupt_os first lrec_fisc ... uf_d*, ll(0)
tobit_rhs_5b = ["first", "lrec_fisc"] + prefchar2 + munichar2 + \
               ["lrec_trans", "lfunc_ativ", "p_cad_pref", "vereador_eleit",
                "ENLP2000", "comarca"] + sorteio_all + uf_dummies
tobit_vars_5b = ["ncorrupt_os"] + tobit_rhs_5b
dfs_t5b = dfs[tobit_vars_5b].dropna()
X_t5b = sm.add_constant(dfs_t5b[tobit_rhs_5b])
y_t5b = dfs_t5b["ncorrupt_os"]
add_tobit_result("5B", "4", "ncorrupt_os", "first", y_t5b, X_t5b,
                 ["const"] + tobit_rhs_5b, lower=0,
                 fixed_effects="uf (dummies)",
                 controls_desc="full controls + uf dummies",
                 notes="Tobit left-censored at 0",
                 orig_coef=-0.018, orig_se=0.007)


# =============================================================================
# TABLE 6: RDD - Vote Margin Controls
# All areg ... abs(uf), robust
# Published: C1: -0.028(0.012), C2: -0.024(0.012), C3: -0.023(0.012)
#            C4: -0.023(0.012), C5: -0.023(0.015), C6: -0.034(0.019), C7: -0.033(0.024)
# =============================================================================
print("\n" + "="*70)
print("TABLE 6: pcorrupt ~ first with running variable")
print("="*70)

# Generate running variable
dfs_t6 = dfs.copy()
dfs_t6['wm'] = np.nan
dfs_t6.loc[dfs_t6['reeleito'] == 1, 'wm'] = dfs_t6.loc[dfs_t6['reeleito'] == 1, 'winmargin2000']
dfs_t6.loc[dfs_t6['incumbent'] == 1, 'wm'] = dfs_t6.loc[dfs_t6['incumbent'] == 1, 'winmargin2000_inclost']
dfs_t6['running'] = dfs_t6['wm'].copy()
dfs_t6.loc[dfs_t6['incumbent'] == 1, 'running'] = -dfs_t6.loc[dfs_t6['incumbent'] == 1, 'wm']
dfs_t6['running2'] = dfs_t6['running'] ** 2
dfs_t6['running3'] = dfs_t6['running'] ** 3
dfs_t6['running4'] = dfs_t6['running'] ** 4
dfs_t6['spline1'] = dfs_t6['first'] * dfs_t6['running']
dfs_t6['spline2'] = dfs_t6['first'] * dfs_t6['running2']
dfs_t6['spline3'] = dfs_t6['first'] * dfs_t6['running3']

base_t6 = prefchar2 + munichar2 + ["lrec_trans", "p_cad_pref", "vereador_eleit",
                                    "ENLP2000", "comarca"] + sorteio_all

# Col 1: no running, but restricted to running~=.
dfs_t6_valid = dfs_t6[dfs_t6['running'].notna()].copy()
m = pf.feols(fml("pcorrupt", ["first"] + base_t6, absorb="uf"),
             data=dfs_t6_valid, vcov="hetero")
add_result("6", "1", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="base controls, restricted to non-missing running",
           sample_desc="esample2==1 & running~=.",
           orig_coef=-0.028, orig_se=0.012)

# Col 2: + running
m = pf.feols(fml("pcorrupt", ["first", "running"] + base_t6, absorb="uf"),
             data=dfs_t6, vcov="hetero")
add_result("6", "2", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="base + linear running",
           orig_coef=-0.024, orig_se=0.012)

# Col 3: + running + running2
m = pf.feols(fml("pcorrupt", ["first", "running", "running2"] + base_t6, absorb="uf"),
             data=dfs_t6, vcov="hetero")
add_result("6", "3", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="base + quadratic running",
           orig_coef=-0.023, orig_se=0.012)

# Col 4: + running + running2 + running3
m = pf.feols(fml("pcorrupt", ["first", "running", "running2", "running3"] + base_t6, absorb="uf"),
             data=dfs_t6, vcov="hetero")
add_result("6", "4", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="base + cubic running",
           orig_coef=-0.023, orig_se=0.012)

# Col 5: linear spline
m = pf.feols(fml("pcorrupt", ["first", "running", "spline1"] + base_t6, absorb="uf"),
             data=dfs_t6, vcov="hetero")
add_result("6", "5", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="base + linear spline",
           orig_coef=-0.023, orig_se=0.015)

# Col 6: quadratic spline
m = pf.feols(fml("pcorrupt", ["first", "running", "spline1", "running2", "spline2"] + base_t6,
                  absorb="uf"), data=dfs_t6, vcov="hetero")
add_result("6", "6", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="base + quadratic spline",
           orig_coef=-0.034, orig_se=0.019)

# Col 7: cubic spline
m = pf.feols(fml("pcorrupt", ["first", "running", "spline1", "running2", "running3",
                               "spline2", "spline3"] + base_t6, absorb="uf"),
             data=dfs_t6, vcov="hetero")
add_result("6", "7", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="base + cubic spline",
           orig_coef=-0.033, orig_se=0.024)


# =============================================================================
# TABLE 7: Experience Controls
# All areg pcorrupt first [controls] | uf, robust
# Published: C1: -0.024(0.012), C2: -0.024(0.011), C3: -0.029(0.012)
#            C4: -0.030(0.012), C5: -0.023(0.013), C6: -0.024(0.014)
# =============================================================================
print("\n" + "="*70)
print("TABLE 7: pcorrupt ~ first with experience controls")
print("="*70)

dfs_t7 = dfs.copy()
dfs_t7['first_experienced'] = ((dfs_t7['reeleito'] == 0) & (dfs_t7['exp_prefeito'] == 1)).astype(float)
dfs_t7['experience1'] = ((dfs_t7['first_experienced'] == 1) | (dfs_t7['reeleito'] == 1)).astype(float)
dfs_t7['experience2'] = ((dfs_t7['reeleito'] == 1) | (dfs_t7['reeleito_2004'] == 1)).astype(float)
dfs_t7['experience3'] = ((dfs_t7['reeleito'] == 1) | (dfs_t7['elected1'] == 1)).astype(float)
dfs_t7['nexp'] = (dfs_t7['exp_prefeito'].fillna(0) + dfs_t7['vereador9600'].fillna(0)) * 4
dfs_t7['nexp2'] = dfs_t7['nexp'] ** 2

base_t7 = prefchar2 + munichar2 + ["lrec_trans", "lfunc_ativ", "p_cad_pref",
                                    "vereador_eleit", "ENLP2000", "comarca"] + sorteio_all

# Col 1: experience2==1 subsample
m = pf.feols(fml("pcorrupt", ["first"] + base_t7, absorb="uf"),
             data=dfs_t7[dfs_t7['experience2'] == 1], vcov="hetero")
add_result("7", "1", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls",
           sample_desc="esample2==1 & experience2==1",
           orig_coef=-0.024, orig_se=0.012)

# Col 2: experience3==1 subsample
m = pf.feols(fml("pcorrupt", ["first"] + base_t7, absorb="uf"),
             data=dfs_t7[dfs_t7['experience3'] == 1], vcov="hetero")
add_result("7", "2", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls",
           sample_desc="esample2==1 & experience3==1",
           orig_coef=-0.024, orig_se=0.011)

# Col 3: + exp_prefeito
m = pf.feols(fml("pcorrupt", ["first", "exp_prefeito"] + base_t7, absorb="uf"),
             data=dfs_t7, vcov="hetero")
add_result("7", "3", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls + exp_prefeito",
           orig_coef=-0.029, orig_se=0.012)

# Col 4: + nexp nexp2
m = pf.feols(fml("pcorrupt", ["first", "nexp", "nexp2"] + base_t7, absorb="uf"),
             data=dfs_t7, vcov="hetero")
add_result("7", "4", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls + nexp + nexp2",
           orig_coef=-0.030, orig_se=0.012)

# Col 5: experience1==1 subsample
m = pf.feols(fml("pcorrupt", ["first"] + base_t7, absorb="uf"),
             data=dfs_t7[dfs_t7['experience1'] == 1], vcov="hetero")
add_result("7", "5", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls",
           sample_desc="esample2==1 & experience1==1",
           orig_coef=-0.023, orig_se=0.013)

# Col 6: restricted sample
mask_c6 = (dfs_t7['first'] == 0) | ((dfs_t7['first'] == 1) &
           ((dfs_t7['exp_prefeito'] == 1) | (dfs_t7['vereador9600'] == 1)))
m = pf.feols(fml("pcorrupt", ["first"] + base_t7, absorb="uf"),
             data=dfs_t7[mask_c6], vcov="hetero")
add_result("7", "6", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls",
           sample_desc="first==0 or experienced first-term mayors",
           notes="vereador96 not in data; used vereador9600 as proxy",
           orig_coef=-0.024, orig_se=0.014)


# =============================================================================
# TABLE 8: Mismanagement (Placebo)
# areg pmismanagement/pcorrupt first [controls] | uf, robust
# Published: C1: -0.015(0.134), C2: 0.041(0.155), C3: 0.070(0.141), C4: -0.025(0.012)
# =============================================================================
print("\n" + "="*70)
print("TABLE 8: Mismanagement placebo")
print("="*70)

dfs_t8 = dfs_t7.copy()  # reuses experience variables

# Col 1
m = pf.feols(fml("pmismanagement", ["first"] + base_t7, absorb="uf"),
             data=dfs_t8, vcov="hetero")
add_result("8", "1", "pmismanagement", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls",
           orig_coef=-0.015, orig_se=0.134)

# Col 2: experience2==1
m = pf.feols(fml("pmismanagement", ["first"] + base_t7, absorb="uf"),
             data=dfs_t8[dfs_t8['experience2'] == 1], vcov="hetero")
add_result("8", "2", "pmismanagement", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls",
           sample_desc="esample2==1 & experience2==1",
           orig_coef=0.041, orig_se=0.155)

# Col 3: experience1==1
m = pf.feols(fml("pmismanagement", ["first"] + base_t7, absorb="uf"),
             data=dfs_t8[dfs_t8['experience1'] == 1], vcov="hetero")
add_result("8", "3", "pmismanagement", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls",
           sample_desc="esample2==1 & experience1==1",
           orig_coef=0.070, orig_se=0.141)

# Col 4: pcorrupt as DV, restricted to non-missing pmismanagement
m = pf.feols(fml("pcorrupt", ["first"] + base_t7, absorb="uf"),
             data=dfs_t8[dfs_t8['pmismanagement'].notna()], vcov="hetero")
add_result("8", "4", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls",
           sample_desc="esample2==1 & pmismanagement non-missing",
           notes="DV=pcorrupt on pmismanagement sample",
           orig_coef=-0.025, orig_se=0.012)


# =============================================================================
# TABLE 9: Convenios (Matching Grants) - Panel data
# reg/areg with cluster(cod_ibge6)
# Published: Panel A C1: 0.115(0.042), C2: 0.119(0.048)
#            Panel B C1: 0.542(0.219), C2: 0.560(0.256)
#            Panel C C1: 0.044(0.026), C2: 0.052(0.030)
# =============================================================================
print("\n" + "="*70)
print("TABLE 9: Convenios (panel)")
print("="*70)

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

# Create year interactions for FE specs
for v in ['lpib02', 'purb', 'p_secundario', 'mun_novo', 'gini_ipea', 'pref_masc',
          'pref_idade_tse', 'pref_escola', 'p_cad_pref', 'vereador_eleit', 'ENLP2000', 'comarca']:
    dfc[f'ano_{v}'] = dfc['ano'] * dfc[v]
ano_int = [f'ano_{v}' for v in ['lpib02', 'purb', 'p_secundario', 'mun_novo', 'gini_ipea',
                                 'pref_masc', 'pref_idade_tse', 'pref_escola',
                                 'p_cad_pref', 'vereador_eleit', 'ENLP2000', 'comarca']]

# Panel A: dconvenios
# Col 1: pooled OLS
rhs_9a1 = ["tr_ano2004", "tr_ano2003", "tr_ano2002", "first",
           "ano2002", "ano2003", "ano2004"] + \
          conv_prefchar2 + conv_munichar2 + ["p_cad_pref", "vereador_eleit", "ENLP2000", "comarca"]
m = pf.feols(fml("dconvenios", rhs_9a1), data=dfc, vcov={"CRV1": "cod_ibge6"})
add_result("9A", "1", "dconvenios", "tr_ano2004", m, "OLS",
           controls_desc="pooled OLS + year dummies + controls",
           cluster_var="cod_ibge6",
           orig_coef=0.115, orig_se=0.042)

# Col 2: municipality FE
rhs_9a2 = ["first", "tr_ano2002", "tr_ano2003", "tr_ano2004",
           "ano2002", "ano2003", "ano2004"] + ano_int
m = pf.feols(fml("dconvenios", rhs_9a2, absorb="cod_ibge6"),
             data=dfc, vcov={"CRV1": "cod_ibge6"})
add_result("9A", "2", "dconvenios", "tr_ano2004", m, "OLS",
           fixed_effects="cod_ibge6",
           controls_desc="municipality FE + year + year interactions",
           cluster_var="cod_ibge6",
           orig_coef=0.119, orig_se=0.048)

# Panel B: lconvenios_pc
# Col 3: pooled OLS
m = pf.feols(fml("lconvenios_pc", rhs_9a1), data=dfc, vcov={"CRV1": "cod_ibge6"})
add_result("9B", "3", "lconvenios_pc", "tr_ano2004", m, "OLS",
           controls_desc="pooled OLS + year dummies + controls",
           cluster_var="cod_ibge6",
           orig_coef=0.542, orig_se=0.219)

# Col 4: municipality FE
m = pf.feols(fml("lconvenios_pc", rhs_9a2, absorb="cod_ibge6"),
             data=dfc, vcov={"CRV1": "cod_ibge6"})
add_result("9B", "4", "lconvenios_pc", "tr_ano2004", m, "OLS",
           fixed_effects="cod_ibge6",
           controls_desc="municipality FE + year + year interactions",
           cluster_var="cod_ibge6",
           orig_coef=0.560, orig_se=0.256)

# Panel C: msh_liberado
# Col 5: pooled OLS  (note: do-file uses first tr_ano2002-tr_ano2004, not tr_ano2004 first)
rhs_9c1 = ["first", "tr_ano2002", "tr_ano2003", "tr_ano2004",
           "ano2002", "ano2003", "ano2004"] + \
          conv_prefchar2 + conv_munichar2 + ["p_cad_pref", "vereador_eleit", "ENLP2000", "comarca"]
m = pf.feols(fml("msh_liberado", rhs_9c1), data=dfc, vcov={"CRV1": "cod_ibge6"})
add_result("9C", "5", "msh_liberado", "tr_ano2004", m, "OLS",
           controls_desc="pooled OLS + year dummies + controls",
           cluster_var="cod_ibge6",
           orig_coef=0.044, orig_se=0.026)

# Col 6: municipality FE
m = pf.feols(fml("msh_liberado", rhs_9a2, absorb="cod_ibge6"),
             data=dfc, vcov={"CRV1": "cod_ibge6"})
add_result("9C", "6", "msh_liberado", "tr_ano2004", m, "OLS",
           fixed_effects="cod_ibge6",
           controls_desc="municipality FE + year + year interactions",
           cluster_var="cod_ibge6",
           orig_coef=0.052, orig_se=0.030)


# =============================================================================
# TABLE 10: Heterogeneity (Interaction terms)
# areg pcorrupt first + first*X [controls] | uf, robust
# Published: C1: -0.032(0.013), C2: -0.019(0.014), C3: -0.059(0.023), C4: -0.004(0.022)
# =============================================================================
print("\n" + "="*70)
print("TABLE 10: Heterogeneous effects")
print("="*70)

dfs_t10 = dfs.copy()
dfs_t10['h_ENEP2000'] = 1 / dfs_t10['ENEP2000']
dfs_t10['first_comarca'] = dfs_t10['comarca'] * dfs_t10['first']
dfs_t10['first_p_cad_pref'] = dfs_t10['p_cad_pref'] * dfs_t10['first']
dfs_t10['first_media2'] = dfs_t10['media2'] * dfs_t10['first']
dfs_t10['first_h_ENEP2000'] = dfs_t10['h_ENEP2000'] * dfs_t10['first']

base_t10 = prefchar2 + munichar2 + ["lrec_trans", "lfunc_ativ", "p_cad_pref",
                                     "vereador_eleit", "ENLP2000", "comarca"] + sorteio_all

# Col 1: first*comarca
m = pf.feols(fml("pcorrupt", ["first", "first_comarca"] + base_t10, absorb="uf"),
             data=dfs_t10, vcov="hetero")
add_result("10", "1", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls + first*comarca",
           orig_coef=-0.032, orig_se=0.013)

# Col 2: first*media2
m = pf.feols(fml("pcorrupt", ["first", "first_media2", "media2"] + base_t10, absorb="uf"),
             data=dfs_t10, vcov="hetero")
add_result("10", "2", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls + first*media2",
           orig_coef=-0.019, orig_se=0.014)

# Col 3: first*p_cad_pref
m = pf.feols(fml("pcorrupt", ["first", "first_p_cad_pref"] + base_t10, absorb="uf"),
             data=dfs_t10, vcov="hetero")
add_result("10", "3", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls + first*p_cad_pref",
           orig_coef=-0.059, orig_se=0.023)

# Col 4: first*h_ENEP2000 (note: do-file omits lfunc_ativ and ENLP2000)
rhs_10c4 = ["first", "first_h_ENEP2000", "h_ENEP2000"] + prefchar2 + munichar2 + \
           ["lrec_trans", "p_cad_pref", "vereador_eleit", "comarca"] + sorteio_all
m = pf.feols(fml("pcorrupt", rhs_10c4, absorb="uf"), data=dfs_t10, vcov="hetero")
add_result("10", "4", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf",
           controls_desc="controls (no ENLP2000/lfunc_ativ) + first*h_ENEP2000",
           orig_coef=-0.004, orig_se=0.022)


# =============================================================================
# TABLE 11: Robustness checks
# Panel A: pcorrupt regressions, Panel B: lrecursos_fisc regressions
# Published PA: C1: -0.029(0.012), C2: -0.030(0.012), C3: -0.028(0.012)
# Published PB: C1: 0.086(0.069), C2: 0.063(0.085), C3: 0.085(0.069), C4: 0.083(0.070)
# =============================================================================
print("\n" + "="*70)
print("TABLE 11: Robustness checks")
print("="*70)

dfs_t11 = dfs.copy()
dfs_t11['sorteio_electyear'] = ((dfs_t11['nsorteio'] > 5) & (dfs_t11['nsorteio'] < 10)).astype(float)
dfs_t11['first_sorteio_electyear'] = dfs_t11['first'] * dfs_t11['sorteio_electyear']
dfs_t11['first_PT'] = dfs_t11['first'] * dfs_t11['party_d15']
dfs_t11['first_samepartygov98'] = dfs_t11['first'] * dfs_t11['samepartygov98']
dfs_t11['lrecursos_fisc'] = np.log(dfs_t11['valor_fiscalizado'])

base_t11 = prefchar2 + munichar2 + ["lrec_trans", "lfunc_ativ", "p_cad_pref",
                                     "vereador_eleit", "ENLP2000", "comarca"] + sorteio_all

# Panel A Col 1: sorteio_electyear
m = pf.feols(fml("pcorrupt", ["first", "first_sorteio_electyear"] + base_t11, absorb="uf"),
             data=dfs_t11, vcov="hetero")
add_result("11A", "1", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls + first*sorteio_electyear",
           orig_coef=-0.029, orig_se=0.012)

# Panel A Col 2: PT
m = pf.feols(fml("pcorrupt", ["first", "first_PT"] + base_t11, absorb="uf"),
             data=dfs_t11, vcov="hetero")
add_result("11A", "2", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls + first*PT",
           orig_coef=-0.030, orig_se=0.012)

# Panel A Col 3: same party governor
m = pf.feols(fml("pcorrupt", ["first", "first_samepartygov98", "samepartygov98"] + base_t11,
                  absorb="uf"), data=dfs_t11, vcov="hetero")
add_result("11A", "3", "pcorrupt", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls + first*samepartygov98",
           orig_coef=-0.028, orig_se=0.012)

# Panel B Col 1: lrecursos_fisc manipulation test
# Note: do-file order is different: first $prefchar2 $munichar2 lrec_trans p_cad_pref
#       vereador_eleit ENLP2000 comarca lfunc_ativ sorteio*
rhs_11b1 = ["first"] + prefchar2 + munichar2 + ["lrec_trans", "p_cad_pref",
            "vereador_eleit", "ENLP2000", "comarca", "lfunc_ativ"] + sorteio_all
m = pf.feols(fml("lrecursos_fisc", rhs_11b1, absorb="uf"), data=dfs_t11, vcov="hetero")
add_result("11B", "1", "lrecursos_fisc", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls",
           notes="Manipulation test",
           orig_coef=0.086, orig_se=0.069)

# Panel B Col 2: + sorteio_electyear
rhs_11b2 = ["first", "first_sorteio_electyear", "sorteio_electyear"] + base_t11
m = pf.feols(fml("lrecursos_fisc", rhs_11b2, absorb="uf"), data=dfs_t11, vcov="hetero")
add_result("11B", "2", "lrecursos_fisc", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls + sorteio_electyear interaction",
           orig_coef=0.063, orig_se=0.085)

# Panel B Col 3: + PT
rhs_11b3 = ["first", "first_PT"] + base_t11
m = pf.feols(fml("lrecursos_fisc", rhs_11b3, absorb="uf"), data=dfs_t11, vcov="hetero")
add_result("11B", "3", "lrecursos_fisc", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls + first*PT",
           orig_coef=0.085, orig_se=0.069)

# Panel B Col 4: + samepartygov98
rhs_11b4 = ["first", "first_samepartygov98", "samepartygov98"] + base_t11
m = pf.feols(fml("lrecursos_fisc", rhs_11b4, absorb="uf"), data=dfs_t11, vcov="hetero")
add_result("11B", "4", "lrecursos_fisc", "first", m, "OLS",
           fixed_effects="uf", controls_desc="full controls + first*samepartygov98",
           orig_coef=0.083, orig_se=0.070)


# =============================================================================
# FIGURE 2: RDD regression for plot
# reg pcorrupt first running running2 running3 if esample2==1, robust
# =============================================================================
print("\n" + "="*70)
print("FIGURE 2: RDD regression")
print("="*70)

m = pf.feols("pcorrupt ~ first + running + running2 + running3", data=dfs_t6, vcov="hetero")
add_result("Fig2", "1", "pcorrupt", "first", m, "OLS",
           controls_desc="cubic polynomial in running",
           notes="RDD regression for Figure 2 plot")


# =============================================================================
# Write outputs
# =============================================================================
print("\n" + "="*70)
print("WRITING OUTPUTS")
print("="*70)

results_df = pd.DataFrame(results)
csv_path = os.path.join(PACKAGE_DIR, "replication.csv")
results_df.to_csv(csv_path, index=False)
print(f"Wrote {len(results_df)} rows to {csv_path}")

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
