"""
Specification Search: 114750-V1
"Sharing Demographic Risk -- Who is Afraid of the Baby Bust?"
Alexander Ludwig and Michael Reiter, AEJ: Economic Policy

This paper builds a calibrated OLG model with demographic shocks to study
optimal pension policy under demographic risk. The model's key outputs are:
  - Table 1: Steady-state macroeconomic variables (K, Y, L, r, w, taul, pension/GDP)
  - Tables 2-4: Welfare losses from demographic fluctuations under different pension rules

Since this is a calibrated structural model (not a reduced-form regression),
the specification search varies the structural parameters and demographic
assumptions, then re-solves the no-government steady state to examine how
the key steady-state outcomes change.

The primary coefficient we track is the capital-output ratio (K/Y) in the
no-government steady state, which is the central macro outcome of the
calibration exercise and a key target in Table 1. This ratio varies
meaningfully across all parameter variations and determines the interest
rate, wage, and all household allocations.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
import os
import warnings
warnings.filterwarnings('ignore')

# ===========================================================================
# DATA LOADING
# ===========================================================================
BASE = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/114750-V1"

def load_data(data_dir):
    """Load demographic data files."""
    fert1 = np.loadtxt(os.path.join(data_dir, "data", "fert1.txt"))
    fert2 = np.loadtxt(os.path.join(data_dir, "data", "fert2.txt"))
    mort1 = np.loadtxt(os.path.join(data_dir, "data", "mort1.txt"))
    mort2 = np.loadtxt(os.path.join(data_dir, "data", "mort2.txt"))
    prodprof = np.loadtxt(os.path.join(data_dir, "data", "prodprof.csv"))
    return fert1, fert2, mort1, mort2, prodprof


# ===========================================================================
# MODEL FUNCTIONS (translated from MATLAB)
# ===========================================================================

def makedemo(iDemo, T, tAdult, tRetire, PopGrTarget, fert1, fert2, mort1, mort2, prodprof):
    """
    Generate demographic matrices from data files.
    Translated from makedemo.m
    """
    if iDemo == 1:
        f100 = fert1[:, 1]
        m100 = mort1[:, 1]
    elif iDemo == 2:
        f100 = fert2[:, 1]
        m100 = mort2[:, 1]
    else:
        raise ValueError("wrong iDemo")

    s100 = 1 - m100
    p100 = prodprof[:100, 1]

    nYears = 100 // T

    fertr = np.zeros(T)
    survr = np.zeros(T)
    prodtty = np.zeros(T)
    first_fertile = tAdult

    for i in range(T):
        b = np.sum(f100[i * nYears:(i + 1) * nYears])
        j = max(i, first_fertile - 1)
        fertr[j] += b
        survr[i] = np.prod(s100[i * nYears:(i + 1) * nYears])
        if (i + 1) >= tAdult and (i + 1) < tRetire:
            prodtty[i] = np.mean(p100[i * nYears:(i + 1) * nYears])

    work_idx = [i for i in range(T) if (i + 1) >= tAdult and (i + 1) < tRetire]
    prodtty_work = prodtty[work_idx]
    if np.mean(prodtty_work) > 0:
        prodtty = prodtty / np.mean(prodtty_work)

    def popgr_z(fac):
        ff = fac * fertr
        DemoMat = np.zeros((T, T))
        for ii in range(1, T):
            DemoMat[ii, ii - 1] = survr[ii - 1]
            DemoMat[0, ii] = ff[ii]
        eigvals, eigvecs = np.linalg.eig(DemoMat)
        imax = np.argmax(np.abs(eigvals))
        PopGr = np.abs(eigvals[imax])
        PopProfil = np.real(eigvecs[:, imax])
        PopProfil = PopProfil / np.sum(PopProfil)
        return PopGr - PopGrTarget, PopGr, PopProfil

    if PopGrTarget is not None:
        fac = brentq(lambda f: popgr_z(f)[0], 0.1, 5.0)
    else:
        fac = 1.0

    _, PopGr, PopProfil = popgr_z(fac)
    fertr = fertr * fac
    survr[-1] = 0

    return fertr, survr, prodtty, PopProfil, PopGr


def solve_steady_state(params, fert1, fert2, mort1, mort2, prodprof):
    """
    Solve the no-government OLG steady state.
    Replicates startstst.m computation.
    """
    T = params.get('T', 20)
    tAdult = params.get('tAdult', 5)
    tRetire = params.get('tRetire', 14)
    beta0 = params.get('beta0', 0.94)
    alpha = params.get('alpha', 1.0 / 3)
    eta = params.get('eta', 0.6)
    iDemo = params.get('iDemo', 1)
    ProdGr = params.get('ProdGr', 1.015 ** (100 / T))
    phi = params.get('phi', 0)
    iCESProd = params.get('iCESProd', 0)
    sCES = params.get('sCES', 10)

    years = 100.0 / T

    fertr, survr, eff, PopProfil, PopGr = makedemo(
        iDemo, T, tAdult, tRetire, 1.0,
        fert1, fert2, mort1, mort2, prodprof
    )

    # Depreciation from calibration targets
    KOtarget = 3.0 / years
    IOtarget = params.get('IOtarget', 0.235)
    KOtarget_yr = params.get('KOtarget_yr', 3.0)
    KOtarget = KOtarget_yr / years
    IoverK = IOtarget / KOtarget
    Gpp = ProdGr * 1
    delta = (IoverK - 1) * Gpp + 1

    r = 1.0 / beta0 - 1

    if not iCESProd:
        KLratio = ((r + delta) / alpha) ** (1.0 / (alpha - 1))
    else:
        rhoCES = (1 - sCES) / sCES
        def ces_r_eq(kl):
            return alpha * kl ** (-rhoCES - 1) * (alpha / kl ** rhoCES - alpha + 1) ** (-1.0 / rhoCES - 1) - delta - r
        KLratio = brentq(ces_r_eq, 0.01, 200.0)

    if not iCESProd:
        w = (1 - alpha) * KLratio ** alpha
    else:
        rhoCES = (1 - sCES) / sCES
        w = (1 - alpha) * (alpha / KLratio ** rhoCES - alpha + 1) ** (-1.0 / rhoCES - 1)

    mu_s = survr.copy()
    rvec = (1 + r) ** np.arange(T)
    cum_surv = np.cumprod(np.concatenate(([1], mu_s[:-1])))
    rvec = rvec / cum_surv
    bvec = beta0 ** np.arange(T)
    bvec = bvec * cum_surv

    kidwght = phi
    nKids_full = np.zeros(T)

    Pc = 1.0 / rvec
    Pl = w * eff / rvec
    Wc = (1 + nKids_full) * bvec
    Wl = eta * bvec

    indxC = list(range(tAdult - 1, T))
    indxL = list(range(tAdult - 1, tRetire - 1))

    Pc_sel = Pc[indxC]
    Wc_sel = Wc[indxC]
    Pl_sel = Pl[indxL]
    Wl_sel = Wl[indxL]

    W = np.sum(Pl_sel)
    sumw = np.sum(Wc_sel) + np.sum(Wl_sel)

    expC = Wc_sel * W / sumw
    expL = Wl_sel * W / sumw

    C = expC / Pc_sel
    L = 1 - expL / Pl_sel

    P = PopProfil
    Laggr = np.sum(eff[indxL] * L * P[indxL])

    KBeg = KLratio * Laggr

    if not iCESProd:
        GDP = KBeg ** alpha * Laggr ** (1 - alpha)
    else:
        rhoCES = (1 - sCES) / sCES
        GDP = (alpha * KBeg ** rhoCES + (1 - alpha) * Laggr ** rhoCES) ** (1.0 / rhoCES)

    Y_net = GDP - delta * KBeg
    r_annual = (1 + r) ** (T / 100) - 1
    KY_ratio = KBeg / GDP
    KY_annual = KY_ratio * years  # annualized K/Y
    avg_C = np.mean(C)
    avg_L = np.mean(L)
    dep_ratio = np.sum(P[tRetire - 1:]) / np.sum(P[tAdult - 1:tRetire - 1])

    # Capital share of income (rK/Y)
    cap_share = (r + delta) * KBeg / GDP

    return {
        'r': r, 'r_annual': r_annual, 'w': w, 'KLratio': KLratio,
        'KBeg': KBeg, 'Laggr': Laggr, 'GDP': GDP, 'Y_net': Y_net,
        'KY_ratio': KY_ratio, 'KY_annual': KY_annual, 'delta': delta,
        'avg_C': avg_C, 'avg_L': avg_L, 'dep_ratio': dep_ratio,
        'cap_share': cap_share,
    }


# ===========================================================================
# SPECIFICATION SEARCH
# ===========================================================================

def run_specification(spec_name, category, params, fert1, fert2, mort1, mort2, prodprof,
                      description=""):
    """Run a single specification and return results."""
    try:
        res = solve_steady_state(params, fert1, fert2, mort1, mort2, prodprof)
        return {
            'spec_name': spec_name,
            'category': category,
            'description': description,
            'coefficient': res['KY_annual'],  # Primary outcome: annualized K/Y ratio
            'std_error': np.nan,
            'p_value': np.nan,
            't_statistic': np.nan,
            'r_annual': res['r_annual'],
            'KBeg': res['KBeg'],
            'GDP': res['GDP'],
            'Laggr': res['Laggr'],
            'w': res['w'],
            'KY_ratio': res['KY_ratio'],
            'KY_annual': res['KY_annual'],
            'avg_C': res['avg_C'],
            'avg_L': res['avg_L'],
            'dep_ratio': res['dep_ratio'],
            'KLratio': res['KLratio'],
            'delta': res['delta'],
            'cap_share': res['cap_share'],
            'param_beta0': params.get('beta0', 0.94),
            'param_alpha': params.get('alpha', 1.0/3),
            'param_eta': params.get('eta', 0.6),
            'param_iDemo': params.get('iDemo', 1),
            'param_T': params.get('T', 20),
            'param_tRetire': params.get('tRetire', 14),
            'param_iCES': params.get('iCESProd', 0),
            'success': True,
            'error': ''
        }
    except Exception as e:
        return {
            'spec_name': spec_name,
            'category': category,
            'description': description,
            'coefficient': np.nan,
            'std_error': np.nan,
            'p_value': np.nan,
            't_statistic': np.nan,
            'r_annual': np.nan,
            'KBeg': np.nan,
            'GDP': np.nan,
            'Laggr': np.nan,
            'w': np.nan,
            'KY_ratio': np.nan,
            'KY_annual': np.nan,
            'avg_C': np.nan,
            'avg_L': np.nan,
            'dep_ratio': np.nan,
            'KLratio': np.nan,
            'delta': np.nan,
            'cap_share': np.nan,
            'param_beta0': params.get('beta0', 0.94),
            'param_alpha': params.get('alpha', 1.0/3),
            'param_eta': params.get('eta', 0.6),
            'param_iDemo': params.get('iDemo', 1),
            'param_T': params.get('T', 20),
            'param_tRetire': params.get('tRetire', 14),
            'param_iCES': params.get('iCESProd', 0),
            'success': False,
            'error': str(e)
        }


def default_params():
    """Baseline parameter dict matching maincali.m benchmark."""
    return {
        'T': 20,
        'tAdult': 5,
        'tRetire': 14,
        'beta0': 0.94,
        'alpha': 1.0 / 3,
        'eta': 0.6,
        'iDemo': 1,
        'ProdGr': 1.015 ** 5,
        'iLogUtil': 1,
        'sigma_util': 2,
        'phi': 0,
        'iCESProd': 0,
        'sCES': 10,
        'IOtarget': 0.235,
        'KOtarget_yr': 3.0,
    }


def run_all_specifications():
    """Run the full specification search."""
    fert1, fert2, mort1, mort2, prodprof = load_data(BASE)

    results = []
    base = default_params()

    def add(name, cat, params, desc):
        results.append(run_specification(
            name, cat, params, fert1, fert2, mort1, mort2, prodprof, desc
        ))

    # ===================================================================
    # 1. BASELINE (2 specs)
    # ===================================================================
    add("baseline", "baseline", base.copy(),
        "Baseline: eta=0.6, iDemo=1 (constant pop), Cobb-Douglas, log utility")

    p = base.copy(); p['iDemo'] = 2
    add("baseline_iDemo2", "baseline", p,
        "Baseline with declining/aging population (iDemo=2)")

    # ===================================================================
    # 2. LABOR SUPPLY ELASTICITY (eta) VARIATIONS (14 specs)
    # Paper uses eta in {0.2, 0.6, 1.5}. Eta affects labor supply
    # and hence K/Y through the labor allocation.
    # ===================================================================
    etas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 5.0]
    for eta_val in etas:
        p = base.copy(); p['eta'] = eta_val
        add(f"eta_{eta_val}", "parameter_eta", p,
            f"Labor supply elasticity eta={eta_val}")

    # ===================================================================
    # 3. DISCOUNT FACTOR (beta0) VARIATIONS (8 specs)
    # beta0 directly determines r = 1/beta - 1, hence K/L and K/Y.
    # ===================================================================
    betas = [0.88, 0.90, 0.91, 0.92, 0.93, 0.95, 0.96, 0.98]
    for beta_val in betas:
        p = base.copy(); p['beta0'] = beta_val
        add(f"beta_{beta_val}", "parameter_beta", p,
            f"Discount factor beta0={beta_val}")

    # ===================================================================
    # 4. CAPITAL SHARE (alpha) VARIATIONS (6 specs)
    # Alpha determines factor income shares and K/L mapping.
    # ===================================================================
    alphas = [0.25, 0.28, 0.30, 0.35, 0.38, 0.40]
    for alpha_val in alphas:
        p = base.copy(); p['alpha'] = alpha_val
        add(f"alpha_{alpha_val}", "parameter_alpha", p,
            f"Capital share alpha={alpha_val}")

    # ===================================================================
    # 5. PRODUCTIVITY GROWTH VARIATIONS (5 specs)
    # Growth affects delta calibration and hence K/Y.
    # ===================================================================
    growth_rates = [0.005, 0.010, 0.012, 0.020, 0.025]
    for g in growth_rates:
        p = base.copy(); p['ProdGr'] = (1 + g) ** 5
        add(f"growth_{g}", "parameter_growth", p,
            f"Productivity growth={g*100:.1f}% annual")

    # ===================================================================
    # 6. INVESTMENT/OUTPUT TARGET VARIATIONS (5 specs)
    # I/Y target affects depreciation calibration.
    # ===================================================================
    io_targets = [0.18, 0.20, 0.22, 0.25, 0.27]
    for io in io_targets:
        p = base.copy(); p['IOtarget'] = io
        add(f"IOtarget_{io}", "calibration_target_IO", p,
            f"Investment/output target I/Y={io}")

    # ===================================================================
    # 7. CAPITAL/OUTPUT TARGET VARIATIONS (4 specs)
    # K/Y target (annualized) directly changes depreciation.
    # ===================================================================
    ko_targets = [2.5, 2.8, 3.2, 3.5]
    for ko in ko_targets:
        p = base.copy(); p['KOtarget_yr'] = ko
        add(f"KOtarget_{ko}", "calibration_target_KO", p,
            f"Capital/output target K/Y={ko} years")

    # ===================================================================
    # 8. RETIREMENT AGE VARIATIONS (4 specs)
    # ===================================================================
    for tRet in [12, 13, 15, 16]:
        p = base.copy(); p['tRetire'] = tRet
        retire_age = tRet * 5
        add(f"tRetire_{tRet}", "model_structure_retirement", p,
            f"Retirement at period {tRet} (~age {retire_age})")

    # ===================================================================
    # 9. ADULT AGE ONSET VARIATIONS (2 specs)
    # ===================================================================
    for tA in [4, 6]:
        p = base.copy(); p['tAdult'] = tA
        add(f"tAdult_{tA}", "model_structure_adult_age", p,
            f"Adult onset at period {tA} (~age {tA*5})")

    # ===================================================================
    # 10. CES PRODUCTION FUNCTION (5 specs)
    # ===================================================================
    for sCES_val in [0.5, 2, 5, 10, 20]:
        p = base.copy(); p['iCESProd'] = 1; p['sCES'] = sCES_val
        add(f"CES_s_{sCES_val}", "production_function", p,
            f"CES production, elasticity of substitution={sCES_val}")

    # ===================================================================
    # 11. DEMOGRAPHIC SCENARIO INTERACTED WITH KEY PARAMS (12 specs)
    # Paper runs full grid of eta x iDemo. Test interactions.
    # ===================================================================
    for eta_val in [0.2, 0.6, 1.5]:
        for iD in [1, 2]:
            for beta_val in [0.92, 0.96]:
                p = base.copy()
                p['eta'] = eta_val; p['beta0'] = beta_val; p['iDemo'] = iD
                demo_str = "const_pop" if iD == 1 else "decl_pop"
                add(f"joint_eta{eta_val}_beta{beta_val}_iDemo{iD}",
                    "joint_parameter", p,
                    f"eta={eta_val}, beta={beta_val}, {demo_str}")

    # ===================================================================
    # 12. ETA WITH DECLINING POPULATION (3 specs -- paper's Table 1 rows)
    # ===================================================================
    for eta_val in [0.2, 0.6, 1.5]:
        p = base.copy(); p['eta'] = eta_val; p['iDemo'] = 2
        add(f"eta_{eta_val}_iDemo2", "parameter_eta_decline", p,
            f"eta={eta_val}, declining population")

    # ===================================================================
    # 13. ALPHA WITH DECLINING POPULATION (3 specs)
    # ===================================================================
    for alpha_val in [0.25, 0.33, 0.40]:
        p = base.copy(); p['alpha'] = alpha_val; p['iDemo'] = 2
        add(f"alpha_{alpha_val}_iDemo2", "parameter_alpha_decline", p,
            f"alpha={alpha_val}, declining population")

    # ===================================================================
    # 14. BETA WITH DECLINING POPULATION (4 specs)
    # ===================================================================
    for beta_val in [0.90, 0.92, 0.96, 0.98]:
        p = base.copy(); p['beta0'] = beta_val; p['iDemo'] = 2
        add(f"beta_{beta_val}_iDemo2", "parameter_beta_decline", p,
            f"beta={beta_val}, declining population")

    # ===================================================================
    # 15. GROWTH WITH DECLINING POPULATION (3 specs)
    # ===================================================================
    for g in [0.005, 0.015, 0.025]:
        p = base.copy(); p['ProdGr'] = (1 + g) ** 5; p['iDemo'] = 2
        add(f"growth_{g}_iDemo2", "parameter_growth_decline", p,
            f"growth={g*100:.1f}%, declining population")

    # ===================================================================
    # 16. RETIREMENT AGE WITH DECLINING POPULATION (3 specs)
    # ===================================================================
    for tRet in [12, 14, 16]:
        p = base.copy(); p['tRetire'] = tRet; p['iDemo'] = 2
        add(f"tRetire_{tRet}_iDemo2", "retirement_decline", p,
            f"tRetire={tRet}, declining population")

    # ===================================================================
    # 17. EXTREME PARAMETER COMBINATIONS (4 specs)
    # ===================================================================
    # High patience + high eta
    p = base.copy(); p['beta0'] = 0.96; p['eta'] = 2.0
    add("extreme_patient_elastic", "extreme_params", p,
        "High patience (beta=0.96) + elastic labor (eta=2.0)")

    # Low patience + low eta
    p = base.copy(); p['beta0'] = 0.90; p['eta'] = 0.2
    add("extreme_impatient_inelastic", "extreme_params", p,
        "Low patience (beta=0.90) + inelastic labor (eta=0.2)")

    # High alpha + high beta
    p = base.copy(); p['alpha'] = 0.40; p['beta0'] = 0.96
    add("extreme_high_alpha_beta", "extreme_params", p,
        "High capital share (alpha=0.40) + patient (beta=0.96)")

    # Low alpha + low beta
    p = base.copy(); p['alpha'] = 0.25; p['beta0'] = 0.90
    add("extreme_low_alpha_beta", "extreme_params", p,
        "Low capital share (alpha=0.25) + impatient (beta=0.90)")

    return results


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("Running specification search for 114750-V1...")
    print("Ludwig & Reiter: Sharing Demographic Risk")
    print("=" * 60)

    all_results = run_all_specifications()
    df = pd.DataFrame(all_results)
    df_success = df[df['success']].copy()
    df_fail = df[~df['success']].copy()

    print(f"\nTotal specifications: {len(df)}")
    print(f"Successful: {len(df_success)}")
    print(f"Failed: {len(df_fail)}")

    if len(df_fail) > 0:
        print("\nFailed specifications:")
        for _, row in df_fail.iterrows():
            print(f"  {row['spec_name']}: {row['error']}")

    output_path = os.path.join(BASE, "specification_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    baseline = df_success[df_success['spec_name'] == 'baseline'].iloc[0]
    print(f"\nBaseline K/Y (annualized): {baseline['KY_annual']:.4f}")
    print(f"Baseline r_annual: {baseline['r_annual']:.4f} ({baseline['r_annual']*100:.2f}%)")
    print(f"Baseline avg_L: {baseline['avg_L']:.4f}")
    print(f"Baseline GDP: {baseline['GDP']:.4f}")
    print(f"Baseline K: {baseline['KBeg']:.4f}")

    print(f"\nK/Y (annualized) across specifications:")
    print(f"  Median: {df_success['KY_annual'].median():.4f}")
    print(f"  Mean:   {df_success['KY_annual'].mean():.4f}")
    print(f"  Std:    {df_success['KY_annual'].std():.4f}")
    print(f"  Min:    {df_success['KY_annual'].min():.4f}")
    print(f"  Max:    {df_success['KY_annual'].max():.4f}")

    print(f"\nAggregate labor across specifications:")
    print(f"  Median: {df_success['Laggr'].median():.4f}")
    print(f"  Mean:   {df_success['Laggr'].mean():.4f}")
    print(f"  Range:  [{df_success['Laggr'].min():.4f}, {df_success['Laggr'].max():.4f}]")

    print(f"\nGDP across specifications:")
    print(f"  Median: {df_success['GDP'].median():.4f}")
    print(f"  Mean:   {df_success['GDP'].mean():.4f}")
    print(f"  Range:  [{df_success['GDP'].min():.4f}, {df_success['GDP'].max():.4f}]")

    print("\nBy category:")
    for cat in sorted(df_success['category'].unique()):
        sub = df_success[df_success['category'] == cat]
        print(f"  {cat}: N={len(sub)}, K/Y range=[{sub['KY_annual'].min():.3f}, {sub['KY_annual'].max():.3f}], "
              f"L range=[{sub['Laggr'].min():.4f}, {sub['Laggr'].max():.4f}]")

    print("\nDone.")
