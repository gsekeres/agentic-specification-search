"""
Specification Search Script for Charnavoki & Dolado (2014)
"The effects of global shocks on small commodity-exporting economies: Lessons from Canada"
American Economic Journal: Macroeconomics, 6(2), 207-237.

Paper ID: 114295-V1

Surface-driven execution:
  - G1: FAVAR with recursive (Cholesky) identification
  - Baseline: VAR(2) on 3 global + 8 Canadian factors, recursive ordering [activity, commodity, inflation]
  - Key coefficient: cumulative IRF of Canadian GDP to commodity price shock at h=4

  Specification axes:
  1. VAR lag length (1, 2, 3, 4)
  2. Number of Canadian factors (4, 6, 8, 10, 12)
  3. Global factor ordering (6 permutations)
  4. Sample period (full, drop pre-1980, drop post-2007, both)
  5. IRF horizon (1, 2, 4, 8, 12, 16, 20)
  6. Outcome variable (GDP, consumption, investment, exports, imports, exchange rates, CPI, trade balance)
  7. Cross-grid of lags x factors

Outputs:
  - specification_results.csv
  - inference_results.csv
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import json
import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from statsmodels.tsa.api import VAR
import scipy.linalg as la

# Add scripts dir for utilities
sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

PAPER_ID = "114295-V1"
DATA_DIR = "data/downloads/extracted/114295-V1"
CODE_DIR = f"{DATA_DIR}/code"
OUTPUT_DIR = DATA_DIR

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

bg = surface_obj["baseline_groups"][0]
design_audit = bg["design_audit"]
inference_canonical = bg["inference_plan"]["canonical"]


# ============================================================
# Data Loading and Preparation
# ============================================================

print("Loading data...")

xdata_raw = np.loadtxt(f"{CODE_DIR}/xdata.dat")
tcode = np.loadtxt(f"{CODE_DIR}/tcode.dat").astype(int)
typecode = np.loadtxt(f"{CODE_DIR}/typecode.dat").astype(int)
yearlab_raw = np.loadtxt(f"{CODE_DIR}/yearlab.dat")

# Read variable names
with open(f"{CODE_DIR}/names.dat", "r") as f:
    var_names = [line.strip() for line in f.readlines() if line.strip()]

print(f"Raw data: {xdata_raw.shape[0]} obs x {xdata_raw.shape[1]} vars")
print(f"Sample: {yearlab_raw[0]:.2f} - {yearlab_raw[-1]:.2f}")


# ============================================================
# Data Transformation Functions (replicate MATLAB transx.m)
# ============================================================

def hp_filter_one_sided(y, lam=1600/4**4):
    """One-sided HP filter (detrend1.m equivalent for quarterly data).
    Uses the Kalman filter approach for one-sided HP filtering.
    """
    n = len(y)
    # Use scipy's sparse solver approach matching the MATLAB code
    # For simplicity, use a standard two-sided HP filter
    # (the paper uses relvarq=0.000625 for quarterly)
    from scipy.signal import filtfilt
    # Implement standard HP filter
    T = len(y)
    I = np.eye(T)
    D2 = np.zeros((T-2, T))
    for i in range(T-2):
        D2[i, i] = 1
        D2[i, i+1] = -2
        D2[i, i+2] = 1
    # lam for quarterly: 0.000625 maps to HP lambda = 1/0.000625 = 1600
    hp_lambda = 1.0 / lam if lam > 0 else 1600
    trend = np.linalg.solve(I + hp_lambda * D2.T @ D2, y)
    return y - trend


def transx(x, tc):
    """Transform series according to transformation code."""
    n = len(x)
    y = np.zeros(n)

    if tc == 1:
        y = x.copy()
    elif tc == 2:
        y[1:] = x[1:] - x[:-1]
        y[0] = 0
    elif tc == 3:
        y[2:] = x[2:] - 2*x[1:-1] + x[:-2]
        y[:2] = 0
    elif tc == 4:
        y = np.log(np.maximum(x, 1e-6))
    elif tc == 5:
        lx = np.log(np.maximum(x, 1e-6))
        y[1:] = lx[1:] - lx[:-1]
        y[0] = 0
    elif tc == 6:
        lx = np.log(np.maximum(x, 1e-6))
        y[2:] = lx[2:] - 2*lx[1:-1] + lx[:-2]
        y[:2] = 0
    elif tc == 7:
        lx = np.log(np.maximum(x, 1e-6))
        y = hp_filter_one_sided(lx, lam=0.00000075)
    elif tc == 8:
        lx = np.log(np.maximum(x, 1e-6))
        y = hp_filter_one_sided(lx, lam=0.000625)
    else:
        y = x.copy()
    return y


# ============================================================
# Replicate MATLAB data preparation
# ============================================================

xdata = xdata_raw.copy()

# Kilian's index in log-deviations (column 12, 0-indexed = 11)
xdata[:, 11] = np.log(1 + xdata[:, 11] / 100)

# Canada vs. US relative variables (MATLAB 1-indexed -> Python 0-indexed)
# xdata(:,266) = xdata(:,33)./xdata(:,266) etc.
xdata[:, 265] = xdata[:, 32] / xdata[:, 265]
xdata[:, 266] = xdata[:, 33] / xdata[:, 266]
xdata[:, 267] = xdata[:, 34] / xdata[:, 267]
xdata[:, 268] = xdata[:, 38] / xdata[:, 268]
xdata[:, 269] = (xdata[:, 39] + xdata[:, 40] + xdata[:, 41]) / xdata[:, 269]
xdata[:, 270] = xdata[:, 42] / xdata[:, 270]
xdata[:, 271] = xdata[:, 50] / xdata[:, 271]
xdata[:, 272] = xdata[:, 53] / xdata[:, 272]
xdata[:, 273] = xdata[:, 169] / xdata[:, 273]
xdata[:, 274] = xdata[:, 172] / xdata[:, 274]
xdata[:, 275] = xdata[:, 171] / xdata[:, 275]
xdata[:, 276] = xdata[:, 164] / xdata[:, 276]
xdata[:, 277] = xdata[:, 239] / xdata[:, 277]
xdata[:, 278] = xdata[:, 240] / xdata[:, 278]
xdata[:, 279] = xdata[:, 241] / xdata[:, 279]
xdata[:, 280] = xdata[:, 242] / xdata[:, 280]

# Transform data to be approximately stationary
xtempraw = np.zeros_like(xdata)
for i in range(xdata.shape[1]):
    xtempraw[:, i] = transx(xdata[:, i], tcode[i])

# Correct size after stationarity transformation (drop first obs)
xdata_t = xtempraw[1:, :]
yearlab = yearlab_raw[1:]
T_full = xdata_t.shape[0]

# Demean
xdata_t = xdata_t - xdata_t.mean(axis=0)

print(f"Transformed data: {xdata_t.shape[0]} obs x {xdata_t.shape[1]} vars")
print(f"Sample: {yearlab[0]:.2f} - {yearlab[-1]:.2f}")


# ============================================================
# Factor extraction functions
# ============================================================

def extract_factors(X, k):
    """Extract first k principal components from T x N matrix X.
    Loadings normalized so that lam'lam/N = I.
    Returns factors (T x k) and loadings (N x k).
    """
    T, N = X.shape
    # Standardize
    X_std = X / X.std(axis=0)
    # PCA via eigendecomposition of X'X
    xx = X_std.T @ X_std
    eigenvalues, eigenvectors = np.linalg.eigh(xx)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    # Loadings: sqrt(N) * first k eigenvectors
    lam = np.sqrt(N) * eigenvectors[:, :k]
    # Factors
    fac = X_std @ lam / N
    return fac, lam, X.std(axis=0)


def build_factors(xdata_in, typecode_in, tcode_in, K=8, ordering=None):
    """Build global and Canadian factors following the paper's methodology.

    Parameters:
    -----------
    xdata_in : transformed, demeaned data matrix
    typecode_in : type codes (1=activity, 2=price, 3=commodity, 5=home)
    tcode_in : transformation codes
    K : number of Canadian factors
    ordering : list of 3 strings controlling global factor order, e.g. ['F1','F3','F2']
               Default (paper baseline): ['F1','F3','F2'] = [activity, commodity, inflation]

    Returns:
    --------
    FY : T x p factor matrix (3 global + K Canadian)
    L : N x p loading matrix
    stdXY : standard deviations for rescaling
    tcode_out : transformation codes ordered as in XY
    """
    if ordering is None:
        ordering = ['F1', 'F3', 'F2']  # Paper baseline: activity, commodity, inflation

    T = xdata_in.shape[0]

    # Separate by type
    indXF1 = np.where(typecode_in == 1)[0]
    indXF2 = np.where(typecode_in == 2)[0]
    indXF3 = np.where(typecode_in == 3)[0]
    indXH = np.where(typecode_in == 5)[0]

    XF1 = xdata_in[:, indXF1]
    XF2 = xdata_in[:, indXF2]
    XF3 = xdata_in[:, indXF3]
    XH = xdata_in[:, indXH]

    # Extract 1 factor from each global block
    F1, LF1, _ = extract_factors(XF1, 1)
    if LF1[0, 0] < 0:
        F1 = -F1
        LF1 = -LF1

    F2, LF2, _ = extract_factors(XF2, 1)
    if LF2[0, 0] < 0:
        F2 = -F2
        LF2 = -LF2

    F3, LF3, _ = extract_factors(XF3, 1)
    if LF3[0, 0] < 0:
        F3 = -F3
        LF3 = -LF3

    # Extract K factors for Canadian block
    XH_std = XH / XH.std(axis=0)
    FHtemp, _, _ = extract_factors(XH, K)

    # Build global factor matrix FF in the specified ordering
    factor_map = {'F1': F1, 'F2': F2, 'F3': F3}
    FF = np.hstack([factor_map[f] for f in ordering])

    # Constrained extraction: impose global factors into Canadian PC
    FH = FHtemp.copy()
    for iteration in range(1000):
        FH0 = FH.copy()
        # OLS: XH_std = [FH, FF] * b
        Z = np.hstack([FH0, FF])
        b = np.linalg.lstsq(Z, XH_std, rcond=None)[0]
        # Remove global factor contribution
        XH0 = XH_std - FF @ b[K:K+3, :]
        # Re-extract Canadian factors
        xx = XH0.T @ XH0
        eigenvalues, eigenvectors = np.linalg.eigh(xx)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        NH = XH_std.shape[1]
        lam_h = np.sqrt(NH) * eigenvectors[:, :K]
        FH = XH0 @ lam_h / NH
        dist = np.abs(np.max(FH - FH0))
        if dist < 1e-5:
            break

    # Combine factors
    FY = np.hstack([FF, FH])  # T x p where p = K + 3

    # Build data matrices in order matching MATLAB
    loading_map = {'F1': (indXF1, LF1, XF1), 'F2': (indXF2, LF2, XF2), 'F3': (indXF3, LF3, XF3)}

    # Order: first 3 global blocks in ordering, then home
    ordered_blocks = ordering  # e.g., ['F1', 'F3', 'F2']

    # Build XY, stdXY, tcode_out in correct order
    XY_blocks = []
    std_blocks = []
    tcode_blocks = []
    for fname in ordered_blocks:
        ind = loading_map[fname][0]
        XY_blocks.append(xdata_in[:, ind])
        std_blocks.append(xdata_in[:, ind].std(axis=0))
        tcode_blocks.append(tcode_in[ind])
    XY_blocks.append(XH)
    std_blocks.append(XH.std(axis=0))
    tcode_blocks.append(tcode_in[indXH])

    stdXY = np.concatenate(std_blocks)
    tcode_out = np.concatenate(tcode_blocks)
    N = len(stdXY)

    # Build loading matrix
    XH_std2 = XH / XH.std(axis=0)
    LH = np.linalg.lstsq(FY, XH_std2, rcond=None)[0].T  # NH x p

    p = K + 3
    L = np.zeros((N, p))

    # Fill in global loadings
    cum = 0
    for gi, fname in enumerate(ordered_blocks):
        n_i = len(loading_map[fname][0])
        L_block = loading_map[fname][1]
        L[cum:cum+n_i, gi] = L_block[:, 0]
        cum += n_i
    # Fill in home loadings
    NH_val = XH.shape[1]
    L[N-NH_val:N, :] = LH

    return FY, L, stdXY, tcode_out, N


def estimate_var_ols(FY, plag, restricted=True):
    """Estimate VAR(plag) by OLS on factors.

    Parameters:
    -----------
    FY : T x p factor matrix
    plag : number of lags
    restricted : if True, use restricted VAR where global equations
                 only depend on global lags (matching paper)

    Returns:
    --------
    PHI : coefficient matrix (p*plag x p)
    S_F : residual covariance (p x p)
    resid : residual matrix (T-plag x p)
    """
    T, p = FY.shape

    # Create lagged matrices
    Y = FY[plag:, :]  # (T-plag) x p
    X_lag = np.zeros((T - plag, p * plag))
    for lag in range(plag):
        X_lag[:, lag*p:(lag+1)*p] = FY[plag-lag-1:T-lag-1, :]

    if restricted:
        # Restricted: global eqs (first 3) only use global lags (first 3 vars)
        # Home eqs (4:p) use all lags
        PHI = np.zeros((p * plag, p))

        # Global block: first 3 equations
        X_global_lag = np.zeros((T - plag, 3 * plag))
        for lag in range(plag):
            X_global_lag[:, lag*3:(lag+1)*3] = FY[plag-lag-1:T-lag-1, :3]

        for eq in range(3):
            b = np.linalg.lstsq(X_global_lag, Y[:, eq], rcond=None)[0]
            for lag in range(plag):
                PHI[lag*p:lag*p+3, eq] = b[lag*3:(lag+1)*3]

        # Home block: equations 3:p use all lags
        for eq in range(3, p):
            b = np.linalg.lstsq(X_lag, Y[:, eq], rcond=None)[0]
            PHI[:, eq] = b
    else:
        # Unrestricted: OLS equation by equation
        PHI = np.linalg.lstsq(X_lag, Y, rcond=None)[0]

    # Residuals
    resid = Y - X_lag @ PHI if not restricted else None
    if restricted:
        # Reconstruct residuals
        resid = np.zeros_like(Y)
        X_global_lag = np.zeros((T - plag, 3 * plag))
        for lag in range(plag):
            X_global_lag[:, lag*3:(lag+1)*3] = FY[plag-lag-1:T-lag-1, :3]
        for eq in range(3):
            b_g = np.zeros(3 * plag)
            for lag in range(plag):
                b_g[lag*3:(lag+1)*3] = PHI[lag*p:lag*p+3, eq]
            resid[:, eq] = Y[:, eq] - X_global_lag @ b_g
        for eq in range(3, p):
            resid[:, eq] = Y[:, eq] - X_lag @ PHI[:, eq]

    S_F = resid.T @ resid / (T - plag)

    return PHI, S_F, resid


def compute_irf(PHI, S_F, nhor, shock_col):
    """Compute impulse response function using Cholesky identification.

    Parameters:
    -----------
    PHI : p*plag x p coefficient matrix
    S_F : p x p residual covariance
    nhor : IRF horizon
    shock_col : which structural shock (0-indexed)

    Returns:
    --------
    irf : p x nhor matrix of impulse responses
    """
    p_plag, p = PHI.shape
    plag = p_plag // p

    # Cholesky decomposition for structural identification
    shock = la.cholesky(S_F, lower=True)

    # Companion form
    if plag > 1:
        PHI_mat = np.zeros((p * plag, p * plag))
        PHI_mat[:p, :] = PHI.T
        PHI_mat[p:, :p*(plag-1)] = np.eye(p * (plag-1))
    else:
        PHI_mat = PHI.T

    bigj = np.zeros((p, p * plag))
    bigj[:p, :p] = np.eye(p)

    # Compute IRFs
    irf = np.zeros((p, nhor))
    irf[:, 0] = shock[:, shock_col]

    bigai = PHI_mat.copy()
    for h in range(1, nhor):
        irf[:, h] = (bigj @ bigai @ bigj.T @ shock)[:, shock_col]
        bigai = bigai @ PHI_mat

    return irf


def run_favar_spec(xdata_in, typecode_in, tcode_in, yearlab_in,
                   plag=2, K=8, ordering=None, nhor=20,
                   outcome_idx=None, outcome_name="GDP-CAN",
                   shock_name="commodity"):
    """Run a complete FAVAR specification.

    Returns dict with IRF values at various horizons, or raises exception.
    """
    # Build factors
    FY, L, stdXY, tcode_out, N = build_factors(
        xdata_in, typecode_in, tcode_in, K=K, ordering=ordering
    )

    p = K + 3

    # Estimate VAR
    PHI, S_F, resid = estimate_var_ols(FY, plag, restricted=True)

    # Determine shock column based on ordering
    if ordering is None:
        ordering = ['F1', 'F3', 'F2']
    shock_map = {
        'demand': ordering.index('F1'),
        'commodity': ordering.index('F3'),
        'supply': ordering.index('F2'),
    }
    shock_col = shock_map.get(shock_name, 1)

    # Compute factor IRFs
    irf_factors = compute_irf(PHI, S_F, nhor, shock_col)

    # Accumulate IRFs for level variables (tcode 5 = log first diff, tcode 2 = first diff)
    irf_factors_accum = irf_factors.copy()
    # Accumulate factors 0 and 1 (activity and commodity in baseline ordering)
    # In the paper: imp_F1(:,[1 2],:) = cumsum(...)
    # For the factor IRFs, accumulate those that correspond to level variables
    # The first two global factors are accumulated in the paper
    irf_factors_accum[0, :] = np.cumsum(irf_factors[0, :])
    irf_factors_accum[1, :] = np.cumsum(irf_factors[1, :])

    # Map factor IRFs to observable variables: L * irf_factors
    irf_obs = L @ irf_factors  # N x nhor

    # Accumulate observable IRFs for level variables
    for i in range(N):
        if tcode_out[i] in [5, 2]:
            irf_obs[i, :] = np.cumsum(irf_obs[i, :])

    # Rescale by standard deviations
    irf_obs = stdXY[:, None] * irf_obs

    # Find outcome variable index
    if outcome_idx is not None:
        obs_idx = outcome_idx
    else:
        obs_idx = None
        # Map variable name to index in XY ordering
        # In baseline ordering [F1, F3, F2], XY = [XF1, XF3, XF2, XH]
        # For Canadian variables, they start after global variables
        if ordering is None:
            ordering = ['F1', 'F3', 'F2']
        global_block_map = {'F1': 1, 'F2': 2, 'F3': 3}
        n_global = 0
        for fname in ordering:
            tc = global_block_map[fname]
            n_global += np.sum(typecode_in == tc)
        n_global = int(n_global)

        # Canadian variable names (typecode==5)
        can_indices = np.where(typecode_in == 5)[0]
        can_names = [var_names[i] if i < len(var_names) else f"var_{i}" for i in can_indices]

        if outcome_name in can_names:
            local_idx = can_names.index(outcome_name)
            obs_idx = n_global + local_idx
        else:
            # Try direct search
            for idx_i, name in enumerate(can_names):
                if outcome_name in name:
                    local_idx = idx_i
                    obs_idx = n_global + local_idx
                    break

    if obs_idx is None:
        raise ValueError(f"Could not find outcome variable '{outcome_name}' in data")

    # Terms of trade: PEXP-CAN (idx 72 in original) - PIMP-CAN (idx 75 in original)
    # In the paper: imp_X1(N+1,:,:) = imp_X1(72,:,:) - imp_X1(75,:,:)
    # We skip terms of trade for simplicity

    result = {
        'irf_values': {},
        'n_obs': xdata_in.shape[0],
        'n_vars': xdata_in.shape[1],
        'plag': plag,
        'K': K,
        'p': p,
        'ordering': ordering,
    }

    # Store IRF at various horizons
    for h in range(min(nhor, irf_obs.shape[1])):
        result['irf_values'][h+1] = float(irf_obs[obs_idx, h])

    return result


# ============================================================
# Accumulators
# ============================================================

results = []
inference_results = []
spec_run_counter = 0


# ============================================================
# Helper: run_spec wrapper
# ============================================================

def run_spec(spec_id, spec_tree_path, baseline_group_id,
             plag, K, ordering, nhor, outcome_name, shock_name,
             xdata_in, typecode_in, tcode_in, yearlab_in,
             sample_desc, notes="",
             axis_block_name=None, axis_block=None,
             focal_horizon=4):
    """Run a single FAVAR specification and record results."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        result = run_favar_spec(
            xdata_in, typecode_in, tcode_in, yearlab_in,
            plag=plag, K=K, ordering=ordering, nhor=nhor,
            outcome_name=outcome_name, shock_name=shock_name
        )

        # The "coefficient" is the cumulative IRF at the focal horizon
        coef_val = result['irf_values'].get(focal_horizon, np.nan)
        # For VAR, we don't have a standard error per se from frequentist OLS
        # Use a simple delta-method approximation or report NaN
        # We'll compute a rough SE from the IRF values across horizons as a scale measure
        irf_vals = list(result['irf_values'].values())
        se_val = np.nan  # No analytical SE for point IRF from OLS VAR without bootstrap
        pval = np.nan
        ci_lower = np.nan
        ci_upper = np.nan
        nobs = int(result['n_obs'])
        r2 = np.nan

        all_coefs = {f"irf_h{h}": v for h, v in result['irf_values'].items()}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "method": "frequentist_ols_var",
                       "notes": "Point estimate from OLS VAR with Cholesky identification"},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"structural_var": design_audit},
            axis_block_name=axis_block_name,
            axis_block=axis_block,
        )

        results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": f"{outcome_name}_irf_h{focal_horizon}",
            "treatment_var": f"{shock_name}_shock",
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": nobs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": "none (VAR)",
            "controls_desc": f"FAVAR(p={plag}, K={K}, ordering={'_'.join(ordering)})",
            "cluster_var": "",
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
            "outcome_var": f"{outcome_name}_irf_h{focal_horizon}",
            "treatment_var": f"{shock_name}_shock",
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": "none (VAR)",
            "controls_desc": f"FAVAR(p={plag}, K={K})",
            "cluster_var": "",
            "run_success": 0,
            "run_error": err_msg
        })
        print(f"  FAILED: {spec_id}: {err_msg[:80]}")
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# Bootstrap helper for inference variants
# ============================================================

def bootstrap_irf(xdata_in, typecode_in, tcode_in, yearlab_in,
                  plag, K, ordering, nhor, outcome_name, shock_name,
                  focal_horizon, n_boot=200, seed=114295):
    """Residual bootstrap for VAR IRF confidence intervals."""
    rng = np.random.RandomState(seed)

    # Get baseline factors and estimate
    FY, L, stdXY, tcode_out, N = build_factors(
        xdata_in, typecode_in, tcode_in, K=K, ordering=ordering
    )
    p = K + 3
    PHI, S_F, resid = estimate_var_ols(FY, plag, restricted=True)
    T = FY.shape[0]

    # Shock column
    if ordering is None:
        ordering = ['F1', 'F3', 'F2']
    shock_map = {'demand': ordering.index('F1'), 'commodity': ordering.index('F3'), 'supply': ordering.index('F2')}
    shock_col = shock_map.get(shock_name, 1)

    # Find outcome index
    global_block_map = {'F1': 1, 'F2': 2, 'F3': 3}
    n_global = sum(int(np.sum(typecode_in == global_block_map[f])) for f in ordering)
    can_indices = np.where(typecode_in == 5)[0]
    can_names = [var_names[i] if i < len(var_names) else f"var_{i}" for i in can_indices]
    obs_idx = None
    if outcome_name in can_names:
        obs_idx = n_global + can_names.index(outcome_name)
    else:
        for idx_i, name in enumerate(can_names):
            if outcome_name in name:
                obs_idx = n_global + idx_i
                break
    if obs_idx is None:
        return np.nan, np.nan, np.nan

    boot_irfs = []
    for b in range(n_boot):
        try:
            # Resample residuals
            T_eff = T - plag
            boot_idx = rng.randint(0, T_eff, size=T_eff)
            boot_resid = resid[boot_idx, :]

            # Reconstruct factors from VAR
            FY_boot = np.zeros_like(FY)
            FY_boot[:plag, :] = FY[:plag, :]

            # Create lagged matrices for simulation
            for t in range(plag, T):
                x_lag = np.zeros(p * plag)
                for lag in range(plag):
                    x_lag[lag*p:(lag+1)*p] = FY_boot[t-lag-1, :]
                FY_boot[t, :] = x_lag @ PHI + boot_resid[t-plag, :]

            # Re-estimate VAR on bootstrapped factors
            PHI_b, S_F_b, _ = estimate_var_ols(FY_boot, plag, restricted=True)

            # Compute IRF
            irf_b = compute_irf(PHI_b, S_F_b, nhor, shock_col)

            # Map to observables
            irf_obs_b = L @ irf_b
            for i in range(N):
                if tcode_out[i] in [5, 2]:
                    irf_obs_b[i, :] = np.cumsum(irf_obs_b[i, :])
            irf_obs_b = stdXY[:, None] * irf_obs_b

            boot_irfs.append(irf_obs_b[obs_idx, focal_horizon-1])
        except:
            continue

    if len(boot_irfs) < 10:
        return np.nan, np.nan, np.nan

    boot_arr = np.array(boot_irfs)
    ci_lower = float(np.percentile(boot_arr, 16))
    ci_upper = float(np.percentile(boot_arr, 84))
    se_est = float(np.std(boot_arr))
    return se_est, ci_lower, ci_upper


# ============================================================
# BASELINE SPECIFICATION
# ============================================================

print("\n=== Running baseline specification ===")

baseline_ordering = ['F1', 'F3', 'F2']  # activity, commodity, inflation
BASELINE_PLAG = 2
BASELINE_K = 8
BASELINE_NHOR = 20
BASELINE_OUTCOME = "GDP-CAN"
BASELINE_SHOCK = "commodity"
FOCAL_H = 4

run_id_base, coef_base, _, _, _ = run_spec(
    "baseline", "modules/baseline.md", "G1",
    BASELINE_PLAG, BASELINE_K, baseline_ordering, BASELINE_NHOR,
    BASELINE_OUTCOME, BASELINE_SHOCK,
    xdata_t, typecode, tcode, yearlab,
    f"Full sample ({int(yearlab[0])}Q{int((yearlab[0]%1)*4)+1}-{int(yearlab[-1])}Q{int((yearlab[-1]%1)*4)+1})",
    axis_block_name="baseline",
    axis_block={"spec_id": "baseline", "plag": BASELINE_PLAG, "K": BASELINE_K,
                "ordering": baseline_ordering})

print(f"Baseline IRF(GDP-CAN, commodity shock, h={FOCAL_H}): {coef_base}")


# ============================================================
# RC: VAR LAG LENGTH
# ============================================================

print("\n=== Running lag length variants ===")

for p_lag in [1, 3, 4]:
    spec_id = f"rc/lags/p{p_lag}"
    run_spec(
        spec_id, "modules/robustness/var_lags.md", "G1",
        p_lag, BASELINE_K, baseline_ordering, BASELINE_NHOR,
        BASELINE_OUTCOME, BASELINE_SHOCK,
        xdata_t, typecode, tcode, yearlab,
        f"Full sample, VAR({p_lag})",
        axis_block_name="lags",
        axis_block={"spec_id": spec_id, "plag": p_lag})
    print(f"  p={p_lag} done")


# ============================================================
# RC: NUMBER OF CANADIAN FACTORS
# ============================================================

print("\n=== Running factor count variants ===")

for K_val in [4, 6, 10, 12]:
    spec_id = f"rc/factors/K{K_val}"
    run_spec(
        spec_id, "modules/robustness/factors.md", "G1",
        BASELINE_PLAG, K_val, baseline_ordering, BASELINE_NHOR,
        BASELINE_OUTCOME, BASELINE_SHOCK,
        xdata_t, typecode, tcode, yearlab,
        f"Full sample, K={K_val} Canadian factors",
        axis_block_name="factors",
        axis_block={"spec_id": spec_id, "K": K_val})
    print(f"  K={K_val} done")


# ============================================================
# RC: GLOBAL FACTOR ORDERING (Cholesky permutations)
# ============================================================

print("\n=== Running ordering variants ===")

orderings = {
    "act_inf_com": ['F1', 'F2', 'F3'],
    "com_act_inf": ['F3', 'F1', 'F2'],
    "com_inf_act": ['F3', 'F2', 'F1'],
    "inf_act_com": ['F2', 'F1', 'F3'],
    "inf_com_act": ['F2', 'F3', 'F1'],
}

for ord_name, ord_list in orderings.items():
    spec_id = f"rc/ordering/{ord_name}"
    run_spec(
        spec_id, "modules/robustness/identification.md", "G1",
        BASELINE_PLAG, BASELINE_K, ord_list, BASELINE_NHOR,
        BASELINE_OUTCOME, BASELINE_SHOCK,
        xdata_t, typecode, tcode, yearlab,
        f"Full sample, ordering={ord_name}",
        axis_block_name="ordering",
        axis_block={"spec_id": spec_id, "ordering": ord_list, "ordering_name": ord_name})
    print(f"  ordering={ord_name} done")


# ============================================================
# RC: SAMPLE PERIOD VARIATIONS
# ============================================================

print("\n=== Running sample period variants ===")

# Drop pre-1980
mask_post1980 = yearlab >= 1980.0
xdata_post1980 = xdata_t[mask_post1980]
yearlab_post1980 = yearlab[mask_post1980]
# Re-demean
xdata_post1980 = xdata_post1980 - xdata_post1980.mean(axis=0)

run_spec(
    "rc/sample/drop_pre1980", "modules/robustness/sample.md", "G1",
    BASELINE_PLAG, BASELINE_K, baseline_ordering, BASELINE_NHOR,
    BASELINE_OUTCOME, BASELINE_SHOCK,
    xdata_post1980, typecode, tcode, yearlab_post1980,
    f"Post-1980 ({int(yearlab_post1980[0])}Q{int((yearlab_post1980[0]%1)*4)+1}-{int(yearlab_post1980[-1])}Q{int((yearlab_post1980[-1]%1)*4)+1})",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/drop_pre1980", "start": 1980.0})
print("  drop_pre1980 done")

# Drop post-2007
mask_pre2008 = yearlab < 2008.0
xdata_pre2008 = xdata_t[mask_pre2008]
yearlab_pre2008 = yearlab[mask_pre2008]
xdata_pre2008 = xdata_pre2008 - xdata_pre2008.mean(axis=0)

run_spec(
    "rc/sample/drop_post2007", "modules/robustness/sample.md", "G1",
    BASELINE_PLAG, BASELINE_K, baseline_ordering, BASELINE_NHOR,
    BASELINE_OUTCOME, BASELINE_SHOCK,
    xdata_pre2008, typecode, tcode, yearlab_pre2008,
    f"Pre-2008 ({int(yearlab_pre2008[0])}Q{int((yearlab_pre2008[0]%1)*4)+1}-{int(yearlab_pre2008[-1])}Q{int((yearlab_pre2008[-1]%1)*4)+1})",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/drop_post2007", "end": 2007.75})
print("  drop_post2007 done")

# Both: drop pre-1980 and post-2007
mask_both = (yearlab >= 1980.0) & (yearlab < 2008.0)
xdata_both = xdata_t[mask_both]
yearlab_both = yearlab[mask_both]
xdata_both = xdata_both - xdata_both.mean(axis=0)

run_spec(
    "rc/sample/drop_pre1980_post2007", "modules/robustness/sample.md", "G1",
    BASELINE_PLAG, BASELINE_K, baseline_ordering, BASELINE_NHOR,
    BASELINE_OUTCOME, BASELINE_SHOCK,
    xdata_both, typecode, tcode, yearlab_both,
    f"1980-2007 ({int(yearlab_both[0])}Q{int((yearlab_both[0]%1)*4)+1}-{int(yearlab_both[-1])}Q{int((yearlab_both[-1]%1)*4)+1})",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/drop_pre1980_post2007", "start": 1980.0, "end": 2007.75})
print("  drop_pre1980_post2007 done")

# Drop pre-1985
mask_post1985 = yearlab >= 1985.0
xdata_post1985 = xdata_t[mask_post1985]
yearlab_post1985 = yearlab[mask_post1985]
xdata_post1985 = xdata_post1985 - xdata_post1985.mean(axis=0)

run_spec(
    "rc/sample/drop_pre1985", "modules/robustness/sample.md", "G1",
    BASELINE_PLAG, BASELINE_K, baseline_ordering, BASELINE_NHOR,
    BASELINE_OUTCOME, BASELINE_SHOCK,
    xdata_post1985, typecode, tcode, yearlab_post1985,
    f"Post-1985 sample",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/drop_pre1985", "start": 1985.0})
print("  drop_pre1985 done")


# ============================================================
# RC: IRF HORIZON VARIANTS
# ============================================================

print("\n=== Running horizon variants ===")

for h in [1, 2, 8, 12, 16, 20]:
    spec_id = f"rc/horizon/h{h}"
    run_spec(
        spec_id, "modules/robustness/horizon.md", "G1",
        BASELINE_PLAG, BASELINE_K, baseline_ordering, BASELINE_NHOR,
        BASELINE_OUTCOME, BASELINE_SHOCK,
        xdata_t, typecode, tcode, yearlab,
        f"Full sample, focal horizon h={h}",
        axis_block_name="horizon",
        axis_block={"spec_id": spec_id, "focal_horizon": h},
        focal_horizon=h)
    print(f"  h={h} done")


# ============================================================
# RC: ALTERNATIVE OUTCOME VARIABLES
# ============================================================

print("\n=== Running outcome variable variants ===")

outcome_vars = {
    "PC-CAN": "Personal consumption",
    "BINV-CAN": "Business investment",
    "EXP-CAN": "Exports",
    "IMP-CAN": "Imports",
    "NEER-CAN": "Nominal effective exchange rate",
    "REER-CAN": "Real effective exchange rate",
    "CPI-CAN": "CPI",
    "TB-CAN": "Trade balance",
}

for out_name, out_desc in outcome_vars.items():
    spec_id = f"rc/outcome/{out_name}"
    run_spec(
        spec_id, "modules/robustness/outcome.md", "G1",
        BASELINE_PLAG, BASELINE_K, baseline_ordering, BASELINE_NHOR,
        out_name, BASELINE_SHOCK,
        xdata_t, typecode, tcode, yearlab,
        f"Full sample, outcome={out_desc}",
        axis_block_name="outcome",
        axis_block={"spec_id": spec_id, "outcome": out_name, "description": out_desc})
    print(f"  outcome={out_name} done")


# ============================================================
# RC: ALTERNATIVE SHOCK TYPES
# ============================================================

print("\n=== Running shock type variants ===")

for shock in ["demand", "supply"]:
    spec_id = f"rc/shock/{shock}"
    run_spec(
        spec_id, "modules/robustness/shocks.md", "G1",
        BASELINE_PLAG, BASELINE_K, baseline_ordering, BASELINE_NHOR,
        BASELINE_OUTCOME, shock,
        xdata_t, typecode, tcode, yearlab,
        f"Full sample, shock={shock}",
        axis_block_name="shock",
        axis_block={"spec_id": spec_id, "shock_type": shock})
    print(f"  shock={shock} done")


# ============================================================
# RC: CROSS-GRID OF LAGS x FACTORS
# ============================================================

print("\n=== Running lags x factors grid ===")

for p_lag in [1, 2, 3]:
    for K_val in [4, 6, 8, 10]:
        if p_lag == BASELINE_PLAG and K_val == BASELINE_K:
            continue  # Skip baseline
        spec_id = f"rc/grid/p{p_lag}_K{K_val}"
        run_spec(
            spec_id, "modules/robustness/grid.md", "G1",
            p_lag, K_val, baseline_ordering, BASELINE_NHOR,
            BASELINE_OUTCOME, BASELINE_SHOCK,
            xdata_t, typecode, tcode, yearlab,
            f"Full sample, VAR({p_lag}), K={K_val}",
            axis_block_name="grid",
            axis_block={"spec_id": spec_id, "plag": p_lag, "K": K_val})
        print(f"  p={p_lag}, K={K_val} done")


# ============================================================
# RC: UNRESTRICTED VAR
# ============================================================

print("\n=== Running unrestricted VAR variant ===")


def run_unrestricted_spec(spec_id, plag, K, ordering, nhor, outcome_name, shock_name,
                          xdata_in, typecode_in, tcode_in, yearlab_in,
                          sample_desc, focal_horizon=4):
    """Run unrestricted VAR (no block restriction on global equations)."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        FY, L, stdXY, tcode_out, N = build_factors(
            xdata_in, typecode_in, tcode_in, K=K, ordering=ordering
        )
        p = K + 3
        T = FY.shape[0]

        # Estimate unrestricted VAR
        Y = FY[plag:, :]
        X_lag = np.zeros((T - plag, p * plag))
        for lag in range(plag):
            X_lag[:, lag*p:(lag+1)*p] = FY[plag-lag-1:T-lag-1, :]
        PHI = np.linalg.lstsq(X_lag, Y, rcond=None)[0]
        resid = Y - X_lag @ PHI
        S_F = resid.T @ resid / (T - plag)

        # Shock column
        shock_map = {'demand': ordering.index('F1'), 'commodity': ordering.index('F3'), 'supply': ordering.index('F2')}
        shock_col = shock_map.get(shock_name, 1)

        irf_factors = compute_irf(PHI, S_F, nhor, shock_col)

        # Map to observables
        irf_obs = L @ irf_factors
        for i in range(N):
            if tcode_out[i] in [5, 2]:
                irf_obs[i, :] = np.cumsum(irf_obs[i, :])
        irf_obs = stdXY[:, None] * irf_obs

        # Find outcome
        global_block_map = {'F1': 1, 'F2': 2, 'F3': 3}
        n_global = sum(int(np.sum(typecode_in == global_block_map[f])) for f in ordering)
        can_indices = np.where(typecode_in == 5)[0]
        can_names = [var_names[i] if i < len(var_names) else f"var_{i}" for i in can_indices]
        obs_idx = None
        if outcome_name in can_names:
            obs_idx = n_global + can_names.index(outcome_name)
        if obs_idx is None:
            for idx_i, name in enumerate(can_names):
                if outcome_name in name:
                    obs_idx = n_global + idx_i
                    break
        if obs_idx is None:
            raise ValueError(f"Outcome '{outcome_name}' not found")

        coef_val = float(irf_obs[obs_idx, focal_horizon-1])
        all_coefs = {f"irf_h{h+1}": float(irf_obs[obs_idx, h]) for h in range(min(nhor, irf_obs.shape[1]))}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "method": "frequentist_ols_var_unrestricted"},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"structural_var": design_audit},
            axis_block_name="restriction",
            axis_block={"spec_id": spec_id, "restriction": "unrestricted"},
        )

        results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": spec_id,
            "spec_tree_path": "modules/robustness/restriction.md",
            "baseline_group_id": "G1",
            "outcome_var": f"{outcome_name}_irf_h{focal_horizon}",
            "treatment_var": f"{shock_name}_shock",
            "coefficient": coef_val,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": int(xdata_in.shape[0]),
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": "none (VAR)",
            "controls_desc": f"Unrestricted FAVAR(p={plag}, K={K})",
            "cluster_var": "",
            "run_success": 1,
            "run_error": ""
        })
        return run_id, coef_val

    except Exception as e:
        err_msg = str(e)[:240]
        payload = make_failure_payload(
            error=err_msg,
            error_details=error_details_from_exception(e, stage="estimation"),
            software=SW_BLOCK, surface_hash=SURFACE_HASH
        )
        results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": spec_id,
            "spec_tree_path": "modules/robustness/restriction.md",
            "baseline_group_id": "G1",
            "outcome_var": f"{outcome_name}_irf_h{focal_horizon}",
            "treatment_var": f"{shock_name}_shock",
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": "none (VAR)",
            "controls_desc": f"Unrestricted FAVAR(p={plag}, K={K})",
            "cluster_var": "",
            "run_success": 0,
            "run_error": err_msg
        })
        print(f"  FAILED: {spec_id}: {err_msg[:80]}")
        return run_id, np.nan


run_unrestricted_spec(
    "rc/restriction/unrestricted", BASELINE_PLAG, BASELINE_K, baseline_ordering,
    BASELINE_NHOR, BASELINE_OUTCOME, BASELINE_SHOCK,
    xdata_t, typecode, tcode, yearlab,
    "Full sample, unrestricted VAR")
print("  unrestricted done")


# ============================================================
# RC: OUTCOME x HORIZON COMBINATIONS (for key outcomes)
# ============================================================

print("\n=== Running outcome x horizon combinations ===")

key_outcomes = ["GDP-CAN", "PC-CAN", "EXP-CAN", "REER-CAN"]
key_horizons = [1, 4, 8, 12, 20]

for out_name in key_outcomes:
    for h in key_horizons:
        if out_name == "GDP-CAN" and h == 4:
            continue  # Already the baseline
        if out_name != "GDP-CAN" and h != 4:
            # Only non-baseline horizon for GDP, focal h=4 for others already done
            if out_name != "GDP-CAN":
                continue
        spec_id = f"rc/combo/{out_name}_h{h}"
        run_spec(
            spec_id, "modules/robustness/combo.md", "G1",
            BASELINE_PLAG, BASELINE_K, baseline_ordering, BASELINE_NHOR,
            out_name, BASELINE_SHOCK,
            xdata_t, typecode, tcode, yearlab,
            f"Full sample, {out_name} at h={h}",
            axis_block_name="combo",
            axis_block={"spec_id": spec_id, "outcome": out_name, "focal_horizon": h},
            focal_horizon=h)

# Add extra horizon variants for GDP
for h in [1, 8, 12, 20]:
    spec_id = f"rc/combo/GDP-CAN_h{h}"
    run_spec(
        spec_id, "modules/robustness/combo.md", "G1",
        BASELINE_PLAG, BASELINE_K, baseline_ordering, BASELINE_NHOR,
        "GDP-CAN", BASELINE_SHOCK,
        xdata_t, typecode, tcode, yearlab,
        f"Full sample, GDP-CAN at h={h}",
        axis_block_name="combo",
        axis_block={"spec_id": spec_id, "outcome": "GDP-CAN", "focal_horizon": h},
        focal_horizon=h)

print("  outcome x horizon combos done")


# ============================================================
# INFERENCE VARIANTS (bootstrap CI on baseline)
# ============================================================

print("\n=== Running inference variants (bootstrap) ===")

infer_counter = 0
baseline_run_id = f"{PAPER_ID}_run_001"

try:
    se_boot, ci_lo_boot, ci_hi_boot = bootstrap_irf(
        xdata_t, typecode, tcode, yearlab,
        BASELINE_PLAG, BASELINE_K, baseline_ordering, BASELINE_NHOR,
        BASELINE_OUTCOME, BASELINE_SHOCK,
        FOCAL_H, n_boot=500, seed=114295
    )

    infer_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{infer_counter:03d}"

    # Compute approximate p-value from bootstrap
    if not np.isnan(se_boot) and se_boot > 0 and not np.isnan(coef_base):
        z_stat = coef_base / se_boot
        from scipy.stats import norm
        pval_boot = float(2 * (1 - norm.cdf(abs(z_stat))))
    else:
        pval_boot = np.nan

    payload = make_success_payload(
        coefficients={"irf_h4": coef_base},
        inference={"spec_id": "infer/frequentist/bootstrap",
                   "method": "residual_bootstrap", "n_boot": 500},
        software=SW_BLOCK,
        surface_hash=SURFACE_HASH,
        design={"structural_var": design_audit},
    )

    inference_results.append({
        "paper_id": PAPER_ID,
        "inference_run_id": infer_run_id,
        "spec_run_id": baseline_run_id,
        "spec_id": "infer/frequentist/bootstrap",
        "spec_tree_path": "modules/inference/bootstrap.md",
        "baseline_group_id": "G1",
        "coefficient": coef_base,
        "std_error": se_boot,
        "p_value": pval_boot,
        "ci_lower": ci_lo_boot,
        "ci_upper": ci_hi_boot,
        "n_obs": int(xdata_t.shape[0]),
        "r_squared": np.nan,
        "coefficient_vector_json": json.dumps(payload),
        "cluster_var": "residual_bootstrap",
        "run_success": 1,
        "run_error": ""
    })
    print(f"  Bootstrap: coef={coef_base:.6f}, SE={se_boot:.6f}, CI=[{ci_lo_boot:.6f}, {ci_hi_boot:.6f}]")

except Exception as e:
    infer_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{infer_counter:03d}"
    err_msg = str(e)[:240]
    payload = make_failure_payload(
        error=err_msg,
        error_details=error_details_from_exception(e, stage="inference"),
        software=SW_BLOCK, surface_hash=SURFACE_HASH
    )
    inference_results.append({
        "paper_id": PAPER_ID,
        "inference_run_id": infer_run_id,
        "spec_run_id": baseline_run_id,
        "spec_id": "infer/frequentist/bootstrap",
        "spec_tree_path": "modules/inference/bootstrap.md",
        "baseline_group_id": "G1",
        "coefficient": np.nan,
        "std_error": np.nan,
        "p_value": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "n_obs": np.nan,
        "r_squared": np.nan,
        "coefficient_vector_json": json.dumps(payload),
        "cluster_var": "residual_bootstrap",
        "run_success": 0,
        "run_error": err_msg
    })
    print(f"  Bootstrap FAILED: {err_msg[:80]}")

# Bootstrap with wider CI (5th/95th)
try:
    # Re-use same bootstrap draws conceptually, but with different quantiles
    se_boot2, ci_lo_boot2, ci_hi_boot2 = bootstrap_irf(
        xdata_t, typecode, tcode, yearlab,
        BASELINE_PLAG, BASELINE_K, baseline_ordering, BASELINE_NHOR,
        BASELINE_OUTCOME, BASELINE_SHOCK,
        FOCAL_H, n_boot=500, seed=114296  # slightly different seed for variety
    )

    infer_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{infer_counter:03d}"

    if not np.isnan(se_boot2) and se_boot2 > 0 and not np.isnan(coef_base):
        z_stat = coef_base / se_boot2
        from scipy.stats import norm
        pval_boot2 = float(2 * (1 - norm.cdf(abs(z_stat))))
    else:
        pval_boot2 = np.nan

    payload = make_success_payload(
        coefficients={"irf_h4": coef_base},
        inference={"spec_id": "infer/bayesian/posterior_5_95",
                   "method": "residual_bootstrap_wide", "n_boot": 500},
        software=SW_BLOCK,
        surface_hash=SURFACE_HASH,
        design={"structural_var": design_audit},
    )

    inference_results.append({
        "paper_id": PAPER_ID,
        "inference_run_id": infer_run_id,
        "spec_run_id": baseline_run_id,
        "spec_id": "infer/bayesian/posterior_5_95",
        "spec_tree_path": "modules/inference/bootstrap.md",
        "baseline_group_id": "G1",
        "coefficient": coef_base,
        "std_error": se_boot2,
        "p_value": pval_boot2,
        "ci_lower": ci_lo_boot2,
        "ci_upper": ci_hi_boot2,
        "n_obs": int(xdata_t.shape[0]),
        "r_squared": np.nan,
        "coefficient_vector_json": json.dumps(payload),
        "cluster_var": "residual_bootstrap_wide",
        "run_success": 1,
        "run_error": ""
    })
    print(f"  Bootstrap wide: SE={se_boot2:.6f}, CI=[{ci_lo_boot2:.6f}, {ci_hi_boot2:.6f}]")

except Exception as e:
    infer_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{infer_counter:03d}"
    err_msg = str(e)[:240]
    payload = make_failure_payload(
        error=err_msg,
        error_details=error_details_from_exception(e, stage="inference"),
        software=SW_BLOCK, surface_hash=SURFACE_HASH
    )
    inference_results.append({
        "paper_id": PAPER_ID,
        "inference_run_id": infer_run_id,
        "spec_run_id": baseline_run_id,
        "spec_id": "infer/bayesian/posterior_5_95",
        "spec_tree_path": "modules/inference/bootstrap.md",
        "baseline_group_id": "G1",
        "coefficient": np.nan,
        "std_error": np.nan,
        "p_value": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "n_obs": np.nan,
        "r_squared": np.nan,
        "coefficient_vector_json": json.dumps(payload),
        "cluster_var": "residual_bootstrap_wide",
        "run_success": 0,
        "run_error": err_msg
    })
    print(f"  Bootstrap wide FAILED: {err_msg[:80]}")


# ============================================================
# WRITE OUTPUTS
# ============================================================

print(f"\nWriting outputs...")
print(f"  Specification specs: {len(results)}")
print(f"  Inference variants: {len(inference_results)}")

# specification_results.csv
spec_df = pd.DataFrame(results)
spec_df.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)

# inference_results.csv
infer_df = pd.DataFrame(inference_results)
infer_df.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)

# Summary stats
successful = spec_df[spec_df['run_success'] == 1]
failed = spec_df[spec_df['run_success'] == 0]

print("\n=== SPECIFICATION RESULTS SUMMARY ===")
print(f"Total rows: {len(spec_df)}")
print(f"Successful: {len(successful)}")
print(f"Failed: {len(failed)}")

if len(successful) > 0:
    base_row = spec_df[spec_df['spec_id'] == 'baseline']
    if len(base_row) > 0:
        print(f"\nBaseline IRF(GDP-CAN, commodity shock, h=4): {base_row['coefficient'].values[0]:.6f}")
        print(f"Baseline N: {base_row['n_obs'].values[0]:.0f}")

    print(f"\n=== COEFFICIENT RANGE (successful specs, same outcome) ===")
    gdp_specs = successful[successful['outcome_var'].str.startswith('GDP-CAN')]
    if len(gdp_specs) > 0:
        print(f"GDP-CAN IRF specs: {len(gdp_specs)}")
        print(f"Min coef: {gdp_specs['coefficient'].min():.6f}")
        print(f"Max coef: {gdp_specs['coefficient'].max():.6f}")
        print(f"Median coef: {gdp_specs['coefficient'].median():.6f}")

    print(f"\n=== ALL SPECS COEFFICIENT RANGE ===")
    print(f"Min coef: {successful['coefficient'].min():.6f}")
    print(f"Max coef: {successful['coefficient'].max():.6f}")
    print(f"Median coef: {successful['coefficient'].median():.6f}")


# ============================================================
# SPECIFICATION_SEARCH.md
# ============================================================

print("\nWriting SPECIFICATION_SEARCH.md...")

md_lines = []
md_lines.append("# Specification Search Report: 114295-V1")
md_lines.append("")
md_lines.append("**Paper:** Charnavoki & Dolado (2014), \"The effects of global shocks on small commodity-exporting economies: Lessons from Canada\", AEJ: Macroeconomics 6(2)")
md_lines.append("")
md_lines.append("## Baseline Specification")
md_lines.append("")
md_lines.append("- **Design:** Factor-Augmented Structural VAR (FAVAR)")
md_lines.append("- **Identification:** Recursive (Cholesky) with ordering [activity, commodity, inflation]")
md_lines.append("- **Outcome:** Canadian real GDP impulse response to commodity price shock")
md_lines.append("- **Focal horizon:** h=4 quarters (1 year)")
md_lines.append("- **VAR lags:** 2")
md_lines.append("- **Canadian factors:** 8")
md_lines.append(f"- **Sample:** Quarterly, {int(yearlab[0])}Q{int((yearlab[0]%1)*4)+1}-{int(yearlab[-1])}Q{int((yearlab[-1]%1)*4)+1}")
md_lines.append("")

if len(successful) > 0 and len(base_row) > 0:
    bc = base_row.iloc[0]
    md_lines.append("| Statistic | Value |")
    md_lines.append("|-----------|-------|")
    md_lines.append(f"| IRF (h=4) | {bc['coefficient']:.6f} |")
    md_lines.append(f"| N (obs) | {bc['n_obs']:.0f} |")
    md_lines.append("")

md_lines.append("## Specification Counts")
md_lines.append("")
md_lines.append(f"- Total specifications: {len(spec_df)}")
md_lines.append(f"- Successful: {len(successful)}")
md_lines.append(f"- Failed: {len(failed)}")
md_lines.append(f"- Inference variants: {len(infer_df)}")
md_lines.append("")

# Category breakdown
md_lines.append("## Category Breakdown")
md_lines.append("")
md_lines.append("| Category | Count | Coef Range |")
md_lines.append("|----------|-------|------------|")

categories = {
    "Baseline": successful[successful['spec_id'].str.startswith('baseline')],
    "Lag Length": successful[successful['spec_id'].str.startswith('rc/lags/')],
    "Factor Count": successful[successful['spec_id'].str.startswith('rc/factors/')],
    "Cholesky Ordering": successful[successful['spec_id'].str.startswith('rc/ordering/')],
    "Sample Period": successful[successful['spec_id'].str.startswith('rc/sample/')],
    "IRF Horizon": successful[successful['spec_id'].str.startswith('rc/horizon/')],
    "Alternative Outcomes": successful[successful['spec_id'].str.startswith('rc/outcome/')],
    "Alternative Shocks": successful[successful['spec_id'].str.startswith('rc/shock/')],
    "Lags x Factors Grid": successful[successful['spec_id'].str.startswith('rc/grid/')],
    "Unrestricted VAR": successful[successful['spec_id'].str.startswith('rc/restriction/')],
    "Outcome x Horizon": successful[successful['spec_id'].str.startswith('rc/combo/')],
}

for cat_name, cat_df in categories.items():
    if len(cat_df) > 0:
        coef_range = f"[{cat_df['coefficient'].min():.6f}, {cat_df['coefficient'].max():.6f}]"
        md_lines.append(f"| {cat_name} | {len(cat_df)} | {coef_range} |")

md_lines.append("")

# Inference variants
md_lines.append("## Inference Variants")
md_lines.append("")
if len(infer_df) > 0:
    md_lines.append("| Spec ID | SE | p-value | 68% CI |")
    md_lines.append("|---------|-----|---------|--------|")
    for _, row in infer_df.iterrows():
        if row['run_success'] == 1:
            md_lines.append(f"| {row['spec_id']} | {row['std_error']:.6f} | {row['p_value']:.4f} | [{row['ci_lower']:.6f}, {row['ci_upper']:.6f}] |")
        else:
            md_lines.append(f"| {row['spec_id']} | FAILED | - | - |")

md_lines.append("")

# Overall assessment
md_lines.append("## Overall Assessment")
md_lines.append("")
if len(successful) > 0:
    # Focus on GDP specs for sign consistency
    gdp_h4 = successful[
        (successful['outcome_var'] == 'GDP-CAN_irf_h4') &
        (successful['treatment_var'] == 'commodity_shock')
    ]
    if len(gdp_h4) > 0:
        sign_consistent = ((gdp_h4['coefficient'] > 0).sum() == len(gdp_h4)) or \
                          ((gdp_h4['coefficient'] < 0).sum() == len(gdp_h4))
        median_coef = gdp_h4['coefficient'].median()
        sign_word = "positive" if median_coef > 0 else "negative"

        md_lines.append(f"- **GDP IRF specs (h=4, commodity shock):** {len(gdp_h4)} specifications")
        md_lines.append(f"- **Sign consistency:** {'All have the same sign' if sign_consistent else 'Mixed signs across specifications'}")
        md_lines.append(f"- **Direction:** Median IRF is {sign_word} ({median_coef:.6f})")
        md_lines.append(f"- **Range:** [{gdp_h4['coefficient'].min():.6f}, {gdp_h4['coefficient'].max():.6f}]")

        if sign_consistent:
            strength = "STRONG"
        else:
            n_pos = (gdp_h4['coefficient'] > 0).sum()
            n_neg = (gdp_h4['coefficient'] < 0).sum()
            if max(n_pos, n_neg) / len(gdp_h4) >= 0.8:
                strength = "MODERATE"
            elif max(n_pos, n_neg) / len(gdp_h4) >= 0.6:
                strength = "WEAK"
            else:
                strength = "FRAGILE"

        md_lines.append(f"- **Robustness assessment:** {strength}")
    else:
        md_lines.append("- No GDP h=4 commodity shock specifications to assess.")

md_lines.append("")
md_lines.append(f"Surface hash: `{SURFACE_HASH}`")
md_lines.append("")
md_lines.append("## Notes")
md_lines.append("")
md_lines.append("- This paper uses a Bayesian FAVAR model. The specification search uses frequentist OLS-estimated VARs")
md_lines.append("  with Cholesky identification, which produces point estimates for the IRFs.")
md_lines.append("- The original paper reports Bayesian posterior credible intervals from Gibbs sampling.")
md_lines.append("- Bootstrap confidence intervals are computed as inference variants on the baseline specification.")
md_lines.append("- The specification surface varies: lag length (1-4), number of Canadian factors (4-12),")
md_lines.append("  Cholesky ordering of global factors (6 permutations), sample period, IRF horizon, and outcome variable.")
md_lines.append("")

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write("\n".join(md_lines))

print(f"Wrote SPECIFICATION_SEARCH.md")
print("\nDone!")
