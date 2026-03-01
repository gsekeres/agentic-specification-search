"""
Specification Search Script for Gortz, Tsoukalas & Zanetti (2018)
"News Shocks under Financial Frictions"
American Economic Journal: Macroeconomics, 10(4), 1-31.

Paper ID: 130141-V1

Surface-driven execution:
  - G1: Structural VAR with Minnesota prior, MFEVD identification of TFP news shock
  - Baseline: VAR(5) on [TFP, GDP, Consumption, Hours, GZ_spread, SP500, Inflation]
  - Key coefficient: IRF of GDP to TFP news shock at h=8

  Specification axes:
  1. VAR lag length (2, 3, 4, 5, 6, 8)
  2. Financial variable swap (GZ spread, EBP, default risk, bank equity, BAA, none)
  3. Adding Investment to the VAR
  4. Sample period (full 1984-2017, pre-GR 1984-2007Q3, post-1990)
  5. Identification scheme (MFEVD news vs max-FEV financial)
  6. IRF horizon for measurement (1, 4, 8, 12, 16, 20)
  7. Outcome variable (GDP, TFP, Consumption, Hours, SP500, Inflation, Investment)
  8. Variable ordering in VAR
  9. Number of posterior draws
 10. Grid: lags x financial variable, lags x sample

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

from scipy import linalg as la

# Add scripts dir for utilities
sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

PAPER_ID = "130141-V1"
DATA_DIR = "data/downloads/extracted/130141-V1"
VAR_DIR = f"{DATA_DIR}/Data-and-Codes/VAR results"
OUTPUT_DIR = DATA_DIR
EXCEL_PATH = f"{VAR_DIR}/finaldata072015.xlsx"

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

print("Loading VAR data from Excel...")
df_raw = pd.read_excel(EXCEL_PATH, sheet_name='84Q1-2017Q1', header=0, skiprows=[1])

# Keep only first 14 columns (data columns + date)
df_raw = df_raw.iloc[:, :14]
df_raw.columns = ['date', 'TFP', 'GDP', 'Investment', 'Consumption', 'Hours',
                   'SP500', 'Inflation', 'ebp', 'gzspr', 'defaultrisk',
                   'RMV_banks', 'LOOS', 'BAA']

# Drop rows with all NaN
df_raw = df_raw.dropna(subset=['date'])
print(f"Loaded data: {len(df_raw)} observations, 13 variables")

# Convert to numpy array matching MATLAB column order:
# 1:TFP, 2:GDP, 3:Investment, 4:Consumption, 5:Hours, 6:SP500, 7:Inflation,
# 8:ebp, 9:gzspr, 10:defaultrisk, 11:RMV_banks, 12:LOOS, 13:BAA
VAR_NAMES_ALL = ['TFP', 'GDP', 'Investment', 'Consumption', 'Hours', 'SP500',
                  'Inflation', 'ebp', 'gzspr', 'defaultrisk', 'RMV_banks', 'LOOS', 'BAA']
VAR_COL_MAP = {name: i for i, name in enumerate(VAR_NAMES_ALL)}

Ydata_full = df_raw[VAR_NAMES_ALL].values.astype(np.float64)
dates = pd.to_datetime(df_raw['date'])

# Build date labels like MATLAB yearlab
yearlab_full = np.array([d.year + (d.month - 1) / 12.0 for d in dates])

print(f"Sample: {dates.iloc[0].strftime('%Y-Q%q') if hasattr(dates.iloc[0], 'strftime') else dates.iloc[0]} to {dates.iloc[-1]}")
print(f"Shape: {Ydata_full.shape}")

# Check for NaN - handle LOOS which starts in 1990Q2
print(f"Non-null counts per variable:")
for i, name in enumerate(VAR_NAMES_ALL):
    n_valid = np.sum(~np.isnan(Ydata_full[:, i]))
    print(f"  {name}: {n_valid}/{len(Ydata_full)}")


# ============================================================
# VAR Helper Functions (translated from MATLAB)
# ============================================================

def mlag2(Y, p):
    """Create lagged matrix like MATLAB mlag2."""
    T, M = Y.shape
    Ylag = np.zeros((T, M * p))
    for lag in range(1, p + 1):
        Ylag[lag:, (lag-1)*M:lag*M] = Y[:T-lag, :]
    return Ylag


def minneprc(y, x, p_lag, quarterly, const, lev, prior, lam_params):
    """Minnesota prior: returns diagonal of inv(H) and inv(H)*bprior.

    Translated from minneprc.m
    """
    T, nvar = y.shape
    k = const + nvar * p_lag

    lambda0 = prior
    lambda1 = lam_params[0]
    lambda2 = lam_params[1]
    lambda3 = 1.0
    lambda4 = 1e2

    # Lag decay
    if quarterly == 1:
        ld = np.arange(1, p_lag + 1, dtype=float) ** (-lambda3)
    else:
        j = np.ceil(p_lag / 3.0) ** (-lambda3)
        b_val = 0.0
        if p_lag > 1:
            b_val = (np.log(1) - np.log(j)) / (1 - p_lag)
        a_val = np.exp(-b_val)
        ld = a_val * np.exp(b_val * np.arange(1, p_lag + 1, dtype=float))

    # Squared inverse for own lags (lambda2 adjustment done later)
    ld = (lambda0 * lambda1 * lambda2 * ld) ** (-2)

    # Scale factors from univariate AR regressions
    s = np.zeros(nvar)
    for i in range(nvar):
        # Build regressor for univariate AR
        xi = x[:, k-1:k].copy() if const else np.empty((T, 0))  # constant
        for j_lag in range(p_lag):
            xi = np.column_stack([xi, x[:, i + j_lag * nvar:i + j_lag * nvar + 1]])
        bsh = np.linalg.lstsq(xi, y[:, i], rcond=None)[0]
        u = y[:, i] - xi @ bsh
        s[i] = np.dot(u, u) / T

    # Build H diagonal
    if lambda4 > 0:
        if const:
            H = np.concatenate([np.kron(ld, s), [(lambda0 * lambda4) ** (-2)]])
        else:
            H = np.kron(ld, s)
    else:
        if const:
            H = np.concatenate([np.kron(ld, s), [0.0]])
        else:
            H = np.kron(ld, s)

    # Stack and distinguish own lag vs other
    hm = np.zeros(k * nvar)
    bm = np.zeros(k * nvar)

    for i in range(nvar):
        hadd = H.copy()
        for j_lag in range(p_lag):
            hadd[i + j_lag * nvar] = (lambda2 ** 2) * H[i + j_lag * nvar]
        hm[k * i:k * (i + 1)] = hadd

        # Unit root prior on own first lag
        if lev[i] == 1:
            bm[k * i + i] = hm[k * i + i] * 1.0

    return hm, bm


def diagrv(mat, vec):
    """Replace diagonal of mat with vec (like Gauss diagrv)."""
    result = mat.copy()
    np.fill_diagonal(result, vec)
    return result


def vec(A):
    """Vectorize a matrix column-major (Fortran order)."""
    return A.flatten(order='F')


def IRFVAR(A, A0inv, p_lag, h):
    """Compute IRFs following MATLAB IRFVAR.m"""
    q = A0inv.shape[0]
    J = np.zeros((q, q * p_lag))
    J[:, :q] = np.eye(q)

    # h=0
    IRF = (J @ np.eye(q * p_lag) @ J.T @ A0inv).reshape(q * q, 1, order='F')

    A_power = np.eye(q * p_lag)
    for i in range(1, h + 1):
        A_power = A_power @ A
        irf_col = (J @ A_power @ J.T @ A0inv).reshape(q * q, 1, order='F')
        IRF = np.hstack([IRF, irf_col])

    return IRF


def estimate_var_bayesian(Y, p_lag, lev, ndraws=500, seed=0):
    """Estimate Bayesian VAR with Minnesota prior (BIC-selected hyperparameters).

    Returns posterior draws of IRFs for the news shock.
    Translated from bvarnews.m (ident=2 case).
    """
    T_raw, M = Y.shape
    K = M * p_lag + 1  # number of regressors (including constant)
    nhor = 40

    # Generate lagged Y matrix
    ylag = mlag2(Y, p_lag)
    ylag = ylag[p_lag:, :]

    y = Y[p_lag:, :]
    T = y.shape[0]
    x = np.column_stack([ylag, np.ones(T)])

    # BIC grid search for Minnesota prior hyperparameters
    lambda1_grid = np.concatenate([
        np.arange(1e-5, 9e-5 + 1e-4, 1e-4),
        np.arange(1e-4, 9e-4 + 1e-4, 1e-4),
        np.arange(0.001, 0.009 + 0.001, 0.001),
        np.arange(0.01, 0.5 + 0.01, 0.01)
    ])
    lambda2_grid = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])

    # Create full grid
    l1_rep = np.tile(lambda1_grid, len(lambda2_grid))
    l2_rep = np.repeat(lambda2_grid, len(lambda1_grid))
    lambda_grid = np.column_stack([l1_rep, l2_rep])

    best_bic = np.inf
    best_idx = 0

    for nmod in range(len(lambda_grid)):
        hm, bm = minneprc(y, x, p_lag, 1, 1, lev, 1, lambda_grid[nmod])
        H_inv = diagrv(np.eye(K * M), hm)
        XtX = np.kron(np.eye(M), x.T @ x)
        xxx = np.linalg.inv(XtX + H_inv)
        bb = xxx @ (vec(x.T @ y) + H_inv @ bm)
        b = bb.reshape(K, M, order='F')
        res = y - x @ b
        vmat = (1.0 / T) * (res.T @ res)

        det_val = np.linalg.det(vmat)
        if det_val > 0:
            bic_val = np.log(det_val) + (np.log(T) / T) * b.size
            if bic_val < best_bic:
                best_bic = bic_val
                best_idx = nmod

    # Re-estimate at optimal
    hm, bm = minneprc(y, x, p_lag, 1, 1, lev, 1, lambda_grid[best_idx])
    H_inv = diagrv(np.eye(K * M), hm)
    XtX = np.kron(np.eye(M), x.T @ x)
    xxx = np.linalg.inv(XtX + H_inv)
    bb = xxx @ (vec(x.T @ y) + H_inv @ bm)
    b = bb.reshape(K, M, order='F')
    res = y - x @ b
    vmat = (1.0 / T) * (res.T @ res)

    sxx = np.linalg.cholesky(xxx)
    sinv = np.linalg.cholesky(np.linalg.inv(vmat))

    # Posterior draws
    rng = np.random.RandomState(seed)
    randmatrix_all = rng.randn(M, T, ndraws)
    randvec_all = rng.randn(K * M, ndraws)

    impulses_news = np.zeros((M, nhor, ndraws))

    for irep in range(ndraws):
        # Draw from Normal-Inverse-Wishart
        RANW = randmatrix_all[:, :, irep] / np.sqrt(T)
        RANTR = RANW.T @ sinv
        try:
            WISH = np.linalg.inv(RANTR.T @ RANTR)
        except np.linalg.LinAlgError:
            continue
        try:
            SWISH = np.linalg.cholesky(WISH)
        except np.linalg.LinAlgError:
            continue

        RANC = randvec_all[:, irep]

        # Build V matrix for coefficient draws
        V = np.zeros((M * K, M * K))
        for i in range(M):
            for j in range(M):
                V[i*K:(i+1)*K, j*K:(j+1)*K] = SWISH[i, j] * sxx[i*K:(i+1)*K, i*K:(i+1)*K]

        SHOCK = V @ RANC
        bbdraw = vec(b) + SHOCK
        bdraw = bbdraw.reshape(K, M, order='F')

        # Companion form
        B_comp = np.zeros((M * p_lag, M * p_lag))
        B_comp[:M, :] = bdraw[:-1, :].T  # exclude constant row
        if p_lag > 1:
            B_comp[M:, :M*(p_lag-1)] = np.eye(M * (p_lag - 1))

        # Cholesky of drawn covariance
        vmatdraw = WISH
        try:
            shock_mat = np.linalg.cholesky(vmatdraw)
        except np.linalg.LinAlgError:
            continue

        # IRFs
        irf = IRFVAR(B_comp, shock_mat, p_lag, nhor)

        # MFEVD identification (ident=2): find shock maximizing FEV of TFP at horizon nhor
        # V9 contains the cumulated contributions to TFP FEV from non-first shocks
        # IRF rows: (M+1), (2M+1), ..., ((M-1)*M+1) -> shocks 2..M effect on variable 1
        # In Python: row indices are M, 2M, ..., (M-1)*M  (0-indexed: M, 2M, ..., (M-1)*M)
        # From MATLAB: V9 = IRF((M+1:M:(M-1)*M+1),1:nhor)' -> 1-indexed
        # In 0-indexed: rows M, 2M, ..., (M-1)*M, columns 0:nhor-1
        # Then V9 is (nhor x (M-1))
        row_indices = np.arange(M, M * M, M)  # shocks 2..M effect on var 1 (TFP)
        V9 = irf[row_indices, :nhor].T  # nhor x (M-1)

        # MFEVD method: maximize at horizon nhor
        WWW = np.zeros((M - 1, M - 1))
        j = nhor - 1  # 0-indexed last horizon
        WWW += np.outer(V9[j, :], V9[j, :])

        eigenvalues, eigenvectors = np.linalg.eigh(WWW)
        max_idx = np.argmax(eigenvalues)
        q_vec = eigenvectors[:, max_idx]
        gamma = np.concatenate([[0.0], q_vec])

        # Compute news IRF
        newshock = np.zeros((M, nhor))
        for i_var in range(M):
            # IRF rows for variable i_var: 0, M, 2M, ..., (M-1)*M  -> i_var, M+i_var, 2M+i_var, ...
            row_idx_var = np.arange(i_var, M * M, M)
            newshock[i_var, :] = irf[row_idx_var, :nhor].T @ gamma

        # Sign normalization: TFP should increase in the long run
        if newshock[0, nhor - 1] < 0:
            newshock = -newshock

        impulses_news[:, :, irep] = newshock

    return impulses_news, T


def estimate_var_bayesian_financial(Y, p_lag, lev, ndraws=500, seed=0):
    """Estimate Bayesian VAR with MFEVD identification for first variable (financial shock).

    Used for ident=3 case: financial variable placed first in VAR.
    Maximizes FEV of first variable over medium horizon (6-32 quarters).
    """
    T_raw, M = Y.shape
    K = M * p_lag + 1
    nhor = 40

    ylag = mlag2(Y, p_lag)
    ylag = ylag[p_lag:, :]

    y = Y[p_lag:, :]
    T = y.shape[0]
    x = np.column_stack([ylag, np.ones(T)])

    # BIC grid search (same as bvarnewsU.m but with different lev and lambda grids)
    lambda1_grid = np.concatenate([
        np.arange(1e-6, 9e-5 + 1e-4, 1e-4),
        np.arange(1e-4, 9e-4 + 1e-4, 1e-4),
        np.arange(0.001, 0.09 + 0.001, 0.001)
    ])
    lambda2_grid = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 5.0])

    l1_rep = np.tile(lambda1_grid, len(lambda2_grid))
    l2_rep = np.repeat(lambda2_grid, len(lambda1_grid))
    lambda_grid = np.column_stack([l1_rep, l2_rep])

    best_bic = np.inf
    best_idx = 0

    for nmod in range(len(lambda_grid)):
        hm, bm = minneprc(y, x, p_lag, 1, 1, lev, 1, lambda_grid[nmod])
        H_inv = diagrv(np.eye(K * M), hm)
        XtX = np.kron(np.eye(M), x.T @ x)
        try:
            xxx = np.linalg.inv(XtX + H_inv)
        except:
            continue
        bb = xxx @ (vec(x.T @ y) + H_inv @ bm)
        b = bb.reshape(K, M, order='F')
        res = y - x @ b
        vmat = (1.0 / T) * (res.T @ res)

        det_val = np.linalg.det(vmat)
        if det_val > 0:
            bic_val = np.log(det_val) + (np.log(T) / T) * b.size
            if bic_val < best_bic:
                best_bic = bic_val
                best_idx = nmod

    hm, bm = minneprc(y, x, p_lag, 1, 1, lev, 1, lambda_grid[best_idx])
    H_inv = diagrv(np.eye(K * M), hm)
    XtX = np.kron(np.eye(M), x.T @ x)
    xxx = np.linalg.inv(XtX + H_inv)
    bb = xxx @ (vec(x.T @ y) + H_inv @ bm)
    b = bb.reshape(K, M, order='F')
    res = y - x @ b
    vmat = (1.0 / T) * (res.T @ res)

    sxx = np.linalg.cholesky(xxx)
    sinv = np.linalg.cholesky(np.linalg.inv(vmat))

    rng = np.random.RandomState(seed)
    randmatrix_all = rng.randn(M, T, ndraws)
    randvec_all = rng.randn(K * M, ndraws)

    impulses_news = np.zeros((M, nhor, ndraws))

    for irep in range(ndraws):
        RANW = randmatrix_all[:, :, irep] / np.sqrt(T)
        RANTR = RANW.T @ sinv
        try:
            WISH = np.linalg.inv(RANTR.T @ RANTR)
        except:
            continue
        try:
            SWISH = np.linalg.cholesky(WISH)
        except:
            continue

        RANC = randvec_all[:, irep]
        V = np.zeros((M * K, M * K))
        for i in range(M):
            for j in range(M):
                V[i*K:(i+1)*K, j*K:(j+1)*K] = SWISH[i, j] * sxx[i*K:(i+1)*K, i*K:(i+1)*K]

        SHOCK = V @ RANC
        bbdraw = vec(b) + SHOCK
        bdraw = bbdraw.reshape(K, M, order='F')

        B_comp = np.zeros((M * p_lag, M * p_lag))
        B_comp[:M, :] = bdraw[:-1, :].T
        if p_lag > 1:
            B_comp[M:, :M*(p_lag-1)] = np.eye(M * (p_lag - 1))

        vmatdraw = WISH
        try:
            shock_mat = np.linalg.cholesky(vmatdraw)
        except:
            continue

        irf = IRFVAR(B_comp, shock_mat, p_lag, nhor)

        # ident=3 (MFEVD-Uhlig): maximize FEV of first variable over medium run (6-32 quarters)
        # V9 = IRF((1:M:(M-1)*M+M),1:nhor)' in MATLAB -> all shocks' effect on var 1
        row_indices = np.arange(0, M * M, M)  # shocks 1..M effect on var 1
        V9 = irf[row_indices, :nhor].T  # nhor x M

        WWW = np.zeros((M, M))
        for j in range(5, 32):  # MATLAB 6:32 -> Python 5:32
            WWW += (32 + 1 - (j+1)) * np.outer(V9[j, :], V9[j, :])

        eigenvalues, eigenvectors = np.linalg.eigh(WWW)
        max_idx = np.argmax(eigenvalues)
        gamma = eigenvectors[:, max_idx]

        newshock = np.zeros((M, nhor))
        for i_var in range(M):
            row_idx_var = np.arange(i_var, M * M, M)
            newshock[i_var, :] = irf[row_idx_var, :nhor].T @ gamma

        # Sign: first variable should increase
        if newshock[0, nhor - 1] < 0:
            newshock = -newshock

        impulses_news[:, :, irep] = newshock

    return impulses_news, T


def compute_median_irf(impulses_news):
    """Compute median and 16/84 percentile IRFs from posterior draws."""
    median_irf = np.nanmedian(impulses_news, axis=2)
    low_irf = np.nanpercentile(impulses_news, 16, axis=2)
    high_irf = np.nanpercentile(impulses_news, 84, axis=2)
    return median_irf, low_irf, high_irf


# ============================================================
# Accumulators
# ============================================================

results = []
inference_results = []
spec_run_counter = 0

# Variable names for the baseline model ordering
# Baseline: model = [1, 2, 4, 5, 9, 6, 7] (MATLAB 1-indexed)
# -> TFP, GDP, Consumption, Hours, gzspr, SP500, Inflation
BASELINE_MODEL_INDICES = [0, 1, 3, 4, 8, 5, 6]  # 0-indexed into VAR_NAMES_ALL
BASELINE_VAR_NAMES = [VAR_NAMES_ALL[i] for i in BASELINE_MODEL_INDICES]
BASELINE_LEV = [1, 1, 1, 0, 0, 0, 0]  # From MATLAB code: lev = [1,1,1,0,0,0,0, 1,1,1]
BASELINE_PLAGS = 5
BASELINE_NDRAWS = 500
FOCAL_HORIZON = 8  # quarters (2 years)


# ============================================================
# Helper: run_spec
# ============================================================

def run_spec(spec_id, spec_tree_path, baseline_group_id,
             var_indices, var_names, lev_vec, p_lag, ndraws,
             ident_type, outcome_idx, outcome_name,
             data_matrix, yearlab_arr,
             sample_desc, focal_horizon=8,
             axis_block_name=None, axis_block=None, notes="",
             seed=0):
    """Run a single VAR specification and record results.

    ident_type: 'news' for MFEVD news shock (ident=2), 'financial' for max-FEV (ident=3)
    outcome_idx: index into the VAR variables for the outcome variable
    """
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        # Select data columns
        Y = data_matrix[:, var_indices].copy()

        # Check for NaN - find first/last valid observation
        valid_mask = ~np.any(np.isnan(Y), axis=1)
        first_valid = np.argmax(valid_mask)
        last_valid = len(valid_mask) - 1 - np.argmax(valid_mask[::-1])
        Y = Y[first_valid:last_valid+1, :]
        yl = yearlab_arr[first_valid:last_valid+1]

        M = Y.shape[1]
        T_eff = Y.shape[0]

        if T_eff < p_lag + 20:
            raise ValueError(f"Too few observations: {T_eff} after removing NaN")

        # Estimate
        if ident_type == 'news':
            impulses, T_est = estimate_var_bayesian(Y, p_lag, lev_vec, ndraws=ndraws, seed=seed)
        elif ident_type == 'financial':
            impulses, T_est = estimate_var_bayesian_financial(Y, p_lag, lev_vec, ndraws=ndraws, seed=seed)
        else:
            raise ValueError(f"Unknown ident_type: {ident_type}")

        # Compute median IRF
        median_irf, low_irf, high_irf = compute_median_irf(impulses)

        # Extract focal coefficient
        h_idx = min(focal_horizon - 1, median_irf.shape[1] - 1)
        coef_val = float(median_irf[outcome_idx, h_idx])
        ci_lower = float(low_irf[outcome_idx, h_idx])
        ci_upper = float(high_irf[outcome_idx, h_idx])

        # Approximate SE from posterior spread
        se_val = float((ci_upper - ci_lower) / 2.0)

        # Approximate p-value
        if se_val > 0:
            z_stat = abs(coef_val) / se_val
            from scipy.stats import norm
            pval = float(2 * (1 - norm.cdf(z_stat)))
        else:
            pval = np.nan

        # All IRF coefficients
        all_coefs = {}
        for h in range(min(40, median_irf.shape[1])):
            all_coefs[f"irf_h{h+1}"] = float(median_irf[outcome_idx, h])

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "method": "bayesian_posterior", "n_draws": ndraws},
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
            "treatment_var": "TFP_news_shock" if ident_type == 'news' else "financial_shock",
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": int(T_eff),
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": "none (VAR)",
            "controls_desc": f"VAR({p_lag}), {M} vars, {ident_type} ident, {ndraws} draws",
            "cluster_var": "",
            "run_success": 1,
            "run_error": ""
        })
        return run_id, coef_val

    except Exception as e:
        err_msg = str(e)[:240]
        tb = traceback.format_exc()
        payload = make_failure_payload(
            error=err_msg,
            error_details=error_details_from_exception(e, stage="estimation"),
            software=SW_BLOCK, surface_hash=SURFACE_HASH
        )
        results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": f"{outcome_name}_irf_h{focal_horizon}",
            "treatment_var": "TFP_news_shock" if ident_type == 'news' else "financial_shock",
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
            "controls_desc": f"VAR, {ident_type} ident",
            "cluster_var": "",
            "run_success": 0,
            "run_error": err_msg
        })
        print(f"  FAILED: {spec_id}: {err_msg[:100]}")
        print(f"  Traceback: {tb[-200:]}")
        return run_id, np.nan


# ============================================================
# BASELINE SPECIFICATION
# ============================================================

print("\n=== Running baseline specification ===")
print(f"Variables: {BASELINE_VAR_NAMES}")
print(f"Lags: {BASELINE_PLAGS}, Draws: {BASELINE_NDRAWS}")

# GDP is at index 1 in the baseline VAR ordering
GDP_IDX_BASELINE = 1

run_id_base, coef_base = run_spec(
    "baseline", "modules/baseline.md", "G1",
    BASELINE_MODEL_INDICES, BASELINE_VAR_NAMES, BASELINE_LEV,
    BASELINE_PLAGS, BASELINE_NDRAWS, 'news',
    GDP_IDX_BASELINE, "GDP",
    Ydata_full, yearlab_full,
    "Full sample 1984Q1-2017Q1",
    focal_horizon=FOCAL_HORIZON,
    axis_block_name="baseline",
    axis_block={"spec_id": "baseline", "plag": BASELINE_PLAGS,
                "variables": BASELINE_VAR_NAMES},
    seed=0)

print(f"Baseline IRF(GDP, TFP news shock, h={FOCAL_HORIZON}): {coef_base}")


# ============================================================
# RC: VAR LAG LENGTH
# ============================================================

print("\n=== Running lag length variants ===")

for p_lag in [2, 3, 4, 6, 8]:
    spec_id = f"rc/lags/p{p_lag}"
    _, coef = run_spec(
        spec_id, "modules/robustness/var_lags.md", "G1",
        BASELINE_MODEL_INDICES, BASELINE_VAR_NAMES, BASELINE_LEV,
        p_lag, BASELINE_NDRAWS, 'news',
        GDP_IDX_BASELINE, "GDP",
        Ydata_full, yearlab_full,
        f"Full sample, VAR({p_lag})",
        focal_horizon=FOCAL_HORIZON,
        axis_block_name="lags",
        axis_block={"spec_id": spec_id, "plag": p_lag},
        seed=p_lag)
    print(f"  p={p_lag}: coef={coef}")


# ============================================================
# RC: FINANCIAL VARIABLE SWAP
# ============================================================

print("\n=== Running financial variable variants ===")

# Financial variable alternatives:
# Baseline uses gzspr (index 8 in VAR_NAMES_ALL)
# Alternatives: ebp (7), defaultrisk (9), RMV_banks (10), BAA (12), none
financial_variants = {
    'ebp': (7, [0, 1, 3, 4, 7, 5, 6], [1, 1, 1, 0, 0, 0, 0]),
    'defaultrisk': (9, [0, 1, 3, 4, 9, 5, 6], [1, 1, 1, 0, 0, 0, 0]),
    'RMV_banks': (10, [0, 1, 3, 4, 10, 5, 6], [1, 1, 1, 0, 1, 0, 0]),
    'BAA': (12, [0, 1, 3, 4, 12, 5, 6], [1, 1, 1, 0, 0, 0, 0]),
    'none': (None, [0, 1, 3, 4, 5, 6], [1, 1, 1, 0, 0, 0]),  # 6-variable VAR without financial
}

for fin_name, (fin_idx, model_indices, lev_vec) in financial_variants.items():
    spec_id = f"rc/financial_var/{fin_name}"
    var_names = [VAR_NAMES_ALL[i] for i in model_indices]
    # GDP is at index 1 in all these orderings
    gdp_idx = 1
    _, coef = run_spec(
        spec_id, "modules/robustness/financial_variable.md", "G1",
        model_indices, var_names, lev_vec,
        BASELINE_PLAGS, BASELINE_NDRAWS, 'news',
        gdp_idx, "GDP",
        Ydata_full, yearlab_full,
        f"Full sample, financial var={fin_name}",
        focal_horizon=FOCAL_HORIZON,
        axis_block_name="financial_var",
        axis_block={"spec_id": spec_id, "financial_var": fin_name},
        seed=hash(fin_name) % (2**31))
    print(f"  financial_var={fin_name}: coef={coef}")


# ============================================================
# RC: ADD INVESTMENT TO VAR (8-variable model)
# ============================================================

print("\n=== Running 8-variable model with Investment ===")

# Add Investment: [TFP, GDP, Investment, Consumption, Hours, gzspr, SP500, Inflation]
model_8var = [0, 1, 2, 3, 4, 8, 5, 6]
var_names_8 = [VAR_NAMES_ALL[i] for i in model_8var]
lev_8 = [1, 1, 1, 1, 0, 0, 0, 0]

_, coef = run_spec(
    "rc/add_investment", "modules/robustness/variables.md", "G1",
    model_8var, var_names_8, lev_8,
    BASELINE_PLAGS, BASELINE_NDRAWS, 'news',
    1, "GDP",  # GDP is still at index 1
    Ydata_full, yearlab_full,
    "Full sample, 8-variable model with Investment",
    focal_horizon=FOCAL_HORIZON,
    axis_block_name="variables",
    axis_block={"spec_id": "rc/add_investment", "n_vars": 8},
    seed=42)
print(f"  8-variable model: coef={coef}")


# ============================================================
# RC: SAMPLE PERIOD VARIATIONS
# ============================================================

print("\n=== Running sample period variants ===")

# Pre-Great Recession: 1984Q1 - 2007Q3
mask_pre_gr = yearlab_full < 2007.75
Ydata_preGR = Ydata_full[mask_pre_gr]
yearlab_preGR = yearlab_full[mask_pre_gr]

_, coef = run_spec(
    "rc/sample/pre_GR", "modules/robustness/sample.md", "G1",
    BASELINE_MODEL_INDICES, BASELINE_VAR_NAMES, BASELINE_LEV,
    BASELINE_PLAGS, BASELINE_NDRAWS, 'news',
    GDP_IDX_BASELINE, "GDP",
    Ydata_preGR, yearlab_preGR,
    "Pre-GR sample 1984Q1-2007Q3",
    focal_horizon=FOCAL_HORIZON,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/pre_GR", "end": "2007Q3"},
    seed=100)
print(f"  Pre-GR: coef={coef}")

# Post-1990 (LOOS starts 1990Q2)
mask_post1990 = yearlab_full >= 1990.0
Ydata_post1990 = Ydata_full[mask_post1990]
yearlab_post1990 = yearlab_full[mask_post1990]

_, coef = run_spec(
    "rc/sample/post1990", "modules/robustness/sample.md", "G1",
    BASELINE_MODEL_INDICES, BASELINE_VAR_NAMES, BASELINE_LEV,
    BASELINE_PLAGS, BASELINE_NDRAWS, 'news',
    GDP_IDX_BASELINE, "GDP",
    Ydata_post1990, yearlab_post1990,
    "Post-1990 sample 1990Q1-2017Q1",
    focal_horizon=FOCAL_HORIZON,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/post1990", "start": "1990Q1"},
    seed=200)
print(f"  Post-1990: coef={coef}")

# Pre-GR with 3 lags (as in paper for shorter samples)
_, coef = run_spec(
    "rc/sample/pre_GR_p3", "modules/robustness/sample.md", "G1",
    BASELINE_MODEL_INDICES, BASELINE_VAR_NAMES, BASELINE_LEV,
    3, BASELINE_NDRAWS, 'news',
    GDP_IDX_BASELINE, "GDP",
    Ydata_preGR, yearlab_preGR,
    "Pre-GR sample 1984Q1-2007Q3, VAR(3)",
    focal_horizon=FOCAL_HORIZON,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/pre_GR_p3", "end": "2007Q3", "plag": 3},
    seed=101)
print(f"  Pre-GR p=3: coef={coef}")


# ============================================================
# RC: IDENTIFICATION SCHEME
# ============================================================

print("\n=== Running alternative identification (max-FEV financial) ===")

# ident=3: EBP placed first, maximize FEV of EBP (financial shock)
# model = [8, 1, 2, 4, 5, 6, 7] in MATLAB -> [ebp, TFP, GDP, Cons, Hours, SP500, Inflation]
model_fin_first = [7, 0, 1, 3, 4, 5, 6]  # ebp first
var_names_fin = [VAR_NAMES_ALL[i] for i in model_fin_first]
lev_fin = [0, 1, 1, 1, 0, 0, 0]  # lev from bvarnewsU.m: [0,1,1,1,0,1,0]

_, coef = run_spec(
    "rc/ident/MFEVD_financial", "modules/robustness/identification.md", "G1",
    model_fin_first, var_names_fin, lev_fin,
    BASELINE_PLAGS, BASELINE_NDRAWS, 'financial',
    2, "GDP",  # GDP is at index 2 in this ordering
    Ydata_full, yearlab_full,
    "Full sample, max-FEV financial shock (EBP first)",
    focal_horizon=FOCAL_HORIZON,
    axis_block_name="identification",
    axis_block={"spec_id": "rc/ident/MFEVD_financial", "ident": "max_FEV_EBP"},
    seed=300)
print(f"  Max-FEV financial: coef={coef}")

# GZ spread placed first for max-FEV
model_gz_first = [8, 0, 1, 3, 4, 5, 6]
var_names_gz = [VAR_NAMES_ALL[i] for i in model_gz_first]

_, coef = run_spec(
    "rc/ident/MFEVD_GZfirst", "modules/robustness/identification.md", "G1",
    model_gz_first, var_names_gz, lev_fin,
    BASELINE_PLAGS, BASELINE_NDRAWS, 'financial',
    2, "GDP",
    Ydata_full, yearlab_full,
    "Full sample, max-FEV financial shock (GZ spread first)",
    focal_horizon=FOCAL_HORIZON,
    axis_block_name="identification",
    axis_block={"spec_id": "rc/ident/MFEVD_GZfirst", "ident": "max_FEV_GZ"},
    seed=301)
print(f"  Max-FEV GZ: coef={coef}")


# ============================================================
# RC: IRF HORIZON FOR MEASUREMENT
# ============================================================

print("\n=== Running IRF horizon variants ===")

# We already computed baseline at h=8; now vary the focal horizon
# But we need to re-run to get different h values from the same IRFs
# For efficiency, just change focal_horizon parameter

for h in [1, 4, 12, 16, 20]:
    spec_id = f"rc/horizon/h{h}"
    _, coef = run_spec(
        spec_id, "modules/robustness/horizon.md", "G1",
        BASELINE_MODEL_INDICES, BASELINE_VAR_NAMES, BASELINE_LEV,
        BASELINE_PLAGS, BASELINE_NDRAWS, 'news',
        GDP_IDX_BASELINE, "GDP",
        Ydata_full, yearlab_full,
        f"Full sample, h={h}",
        focal_horizon=h,
        axis_block_name="horizon",
        axis_block={"spec_id": spec_id, "focal_horizon": h},
        seed=0)
    print(f"  h={h}: coef={coef}")


# ============================================================
# RC: OUTCOME VARIABLE
# ============================================================

print("\n=== Running outcome variable variants ===")

# Baseline VAR: [TFP(0), GDP(1), Consumption(2), Hours(3), gzspr(4), SP500(5), Inflation(6)]
outcome_variants = {
    'TFP': 0,
    'Consumption': 2,
    'Hours': 3,
    'gzspr': 4,
    'SP500': 5,
    'Inflation': 6,
}

for out_name, out_idx in outcome_variants.items():
    spec_id = f"rc/outcome/{out_name}"
    _, coef = run_spec(
        spec_id, "modules/robustness/outcome.md", "G1",
        BASELINE_MODEL_INDICES, BASELINE_VAR_NAMES, BASELINE_LEV,
        BASELINE_PLAGS, BASELINE_NDRAWS, 'news',
        out_idx, out_name,
        Ydata_full, yearlab_full,
        f"Full sample, outcome={out_name}",
        focal_horizon=FOCAL_HORIZON,
        axis_block_name="outcome",
        axis_block={"spec_id": spec_id, "outcome": out_name},
        seed=0)
    print(f"  outcome={out_name}: coef={coef}")

# Also GDP from the 8-variable model with Investment
_, coef = run_spec(
    "rc/outcome/Investment_from_8var", "modules/robustness/outcome.md", "G1",
    model_8var, var_names_8, lev_8,
    BASELINE_PLAGS, BASELINE_NDRAWS, 'news',
    2, "Investment",  # Investment is at index 2 in 8var model
    Ydata_full, yearlab_full,
    "Full sample, 8-var model, outcome=Investment",
    focal_horizon=FOCAL_HORIZON,
    axis_block_name="outcome",
    axis_block={"spec_id": "rc/outcome/Investment_from_8var", "outcome": "Investment"},
    seed=42)
print(f"  outcome=Investment (8var): coef={coef}")


# ============================================================
# RC: VARIABLE ORDERING (move financial variable)
# ============================================================

print("\n=== Running ordering variants ===")

# Move financial variable to first position
# [gzspr, TFP, GDP, Consumption, Hours, SP500, Inflation]
model_fin_first_news = [8, 0, 1, 3, 4, 5, 6]
var_names_reord = [VAR_NAMES_ALL[i] for i in model_fin_first_news]
lev_reord = [0, 1, 1, 1, 0, 0, 0]

_, coef = run_spec(
    "rc/ordering/fin_first", "modules/robustness/ordering.md", "G1",
    model_fin_first_news, var_names_reord, lev_reord,
    BASELINE_PLAGS, BASELINE_NDRAWS, 'news',
    2, "GDP",  # GDP at index 2
    Ydata_full, yearlab_full,
    "Full sample, GZ spread placed first",
    focal_horizon=FOCAL_HORIZON,
    axis_block_name="ordering",
    axis_block={"spec_id": "rc/ordering/fin_first", "ordering": "gzspr_first"},
    seed=400)
print(f"  fin_first: coef={coef}")

# Move SP500 before financial variable
# [TFP, GDP, Consumption, Hours, SP500, gzspr, Inflation]
model_sp_before_fin = [0, 1, 3, 4, 5, 8, 6]
var_names_spbf = [VAR_NAMES_ALL[i] for i in model_sp_before_fin]
lev_spbf = [1, 1, 1, 0, 0, 0, 0]

_, coef = run_spec(
    "rc/ordering/sp500_before_fin", "modules/robustness/ordering.md", "G1",
    model_sp_before_fin, var_names_spbf, lev_spbf,
    BASELINE_PLAGS, BASELINE_NDRAWS, 'news',
    1, "GDP",
    Ydata_full, yearlab_full,
    "Full sample, SP500 before GZ spread",
    focal_horizon=FOCAL_HORIZON,
    axis_block_name="ordering",
    axis_block={"spec_id": "rc/ordering/sp500_before_fin", "ordering": "sp500_before_gzspr"},
    seed=401)
print(f"  sp500_before_fin: coef={coef}")


# ============================================================
# RC: MORE POSTERIOR DRAWS
# ============================================================

print("\n=== Running increased draws variant ===")

_, coef = run_spec(
    "rc/ndraws/1000", "modules/robustness/draws.md", "G1",
    BASELINE_MODEL_INDICES, BASELINE_VAR_NAMES, BASELINE_LEV,
    BASELINE_PLAGS, 1000, 'news',
    GDP_IDX_BASELINE, "GDP",
    Ydata_full, yearlab_full,
    "Full sample, 1000 posterior draws",
    focal_horizon=FOCAL_HORIZON,
    axis_block_name="draws",
    axis_block={"spec_id": "rc/ndraws/1000", "ndraws": 1000},
    seed=0)
print(f"  1000 draws: coef={coef}")


# ============================================================
# RC: GRID - LAGS x FINANCIAL VARIABLE
# ============================================================

print("\n=== Running lags x financial variable grid ===")

grid_fin_vars = {
    'ebp': (7, [0, 1, 3, 4, 7, 5, 6], [1, 1, 1, 0, 0, 0, 0]),
    'defaultrisk': (9, [0, 1, 3, 4, 9, 5, 6], [1, 1, 1, 0, 0, 0, 0]),
}

for p_lag in [3, 4]:
    for fin_name, (fin_idx, model_indices, lev_vec) in grid_fin_vars.items():
        spec_id = f"rc/grid/p{p_lag}_{fin_name}"
        var_names = [VAR_NAMES_ALL[i] for i in model_indices]
        _, coef = run_spec(
            spec_id, "modules/robustness/grid.md", "G1",
            model_indices, var_names, lev_vec,
            p_lag, BASELINE_NDRAWS, 'news',
            1, "GDP",
            Ydata_full, yearlab_full,
            f"Full sample, VAR({p_lag}), financial={fin_name}",
            focal_horizon=FOCAL_HORIZON,
            axis_block_name="grid",
            axis_block={"spec_id": spec_id, "plag": p_lag, "financial_var": fin_name},
            seed=p_lag * 100 + hash(fin_name) % 100)
        print(f"  p={p_lag}, fin={fin_name}: coef={coef}")


# ============================================================
# RC: GRID - LAGS x SAMPLE
# ============================================================

print("\n=== Running lags x sample grid ===")

for p_lag in [3, 4]:
    spec_id = f"rc/grid/p{p_lag}_preGR"
    _, coef = run_spec(
        spec_id, "modules/robustness/grid.md", "G1",
        BASELINE_MODEL_INDICES, BASELINE_VAR_NAMES, BASELINE_LEV,
        p_lag, BASELINE_NDRAWS, 'news',
        GDP_IDX_BASELINE, "GDP",
        Ydata_preGR, yearlab_preGR,
        f"Pre-GR sample, VAR({p_lag})",
        focal_horizon=FOCAL_HORIZON,
        axis_block_name="grid",
        axis_block={"spec_id": spec_id, "plag": p_lag, "sample": "pre_GR"},
        seed=p_lag * 200)
    print(f"  p={p_lag}, pre-GR: coef={coef}")


# ============================================================
# RC: OUTCOME x HORIZON COMBINATIONS
# ============================================================

print("\n=== Running outcome x horizon combinations ===")

combo_outcomes = {"GDP": GDP_IDX_BASELINE, "TFP": 0, "Consumption": 2}
combo_horizons = [4, 12, 20]

for out_name, out_idx in combo_outcomes.items():
    for h in combo_horizons:
        if out_name == "GDP" and h == FOCAL_HORIZON:
            continue  # Skip baseline
        spec_id = f"rc/combo/{out_name}_h{h}"
        _, coef = run_spec(
            spec_id, "modules/robustness/combo.md", "G1",
            BASELINE_MODEL_INDICES, BASELINE_VAR_NAMES, BASELINE_LEV,
            BASELINE_PLAGS, BASELINE_NDRAWS, 'news',
            out_idx, out_name,
            Ydata_full, yearlab_full,
            f"Full sample, {out_name} at h={h}",
            focal_horizon=h,
            axis_block_name="combo",
            axis_block={"spec_id": spec_id, "outcome": out_name, "focal_horizon": h},
            seed=0)
        print(f"  {out_name} h={h}: coef={coef}")


# ============================================================
# RC: ADDITIONAL FINANCIAL x SAMPLE COMBOS
# ============================================================

print("\n=== Running financial x sample combinations ===")

# EBP in pre-GR sample
_, coef = run_spec(
    "rc/combo/ebp_preGR", "modules/robustness/combo.md", "G1",
    [0, 1, 3, 4, 7, 5, 6], [VAR_NAMES_ALL[i] for i in [0, 1, 3, 4, 7, 5, 6]],
    [1, 1, 1, 0, 0, 0, 0],
    BASELINE_PLAGS, BASELINE_NDRAWS, 'news',
    1, "GDP",
    Ydata_preGR, yearlab_preGR,
    "Pre-GR sample, EBP as financial var",
    focal_horizon=FOCAL_HORIZON,
    axis_block_name="combo",
    axis_block={"spec_id": "rc/combo/ebp_preGR", "financial_var": "ebp", "sample": "pre_GR"},
    seed=500)
print(f"  EBP + pre-GR: coef={coef}")

# BAA in pre-GR sample
_, coef = run_spec(
    "rc/combo/BAA_preGR", "modules/robustness/combo.md", "G1",
    [0, 1, 3, 4, 12, 5, 6], [VAR_NAMES_ALL[i] for i in [0, 1, 3, 4, 12, 5, 6]],
    [1, 1, 1, 0, 0, 0, 0],
    BASELINE_PLAGS, BASELINE_NDRAWS, 'news',
    1, "GDP",
    Ydata_preGR, yearlab_preGR,
    "Pre-GR sample, BAA as financial var",
    focal_horizon=FOCAL_HORIZON,
    axis_block_name="combo",
    axis_block={"spec_id": "rc/combo/BAA_preGR", "financial_var": "BAA", "sample": "pre_GR"},
    seed=501)
print(f"  BAA + pre-GR: coef={coef}")

# 6-variable (no financial) in pre-GR sample
_, coef = run_spec(
    "rc/combo/noFin_preGR", "modules/robustness/combo.md", "G1",
    [0, 1, 3, 4, 5, 6], [VAR_NAMES_ALL[i] for i in [0, 1, 3, 4, 5, 6]],
    [1, 1, 1, 0, 0, 0],
    BASELINE_PLAGS, BASELINE_NDRAWS, 'news',
    1, "GDP",
    Ydata_preGR, yearlab_preGR,
    "Pre-GR sample, no financial variable",
    focal_horizon=FOCAL_HORIZON,
    axis_block_name="combo",
    axis_block={"spec_id": "rc/combo/noFin_preGR", "financial_var": "none", "sample": "pre_GR"},
    seed=502)
print(f"  No fin + pre-GR: coef={coef}")

# Max-FEV financial on pre-GR sample
_, coef = run_spec(
    "rc/combo/maxFEV_preGR", "modules/robustness/combo.md", "G1",
    [7, 0, 1, 3, 4, 5, 6], [VAR_NAMES_ALL[i] for i in [7, 0, 1, 3, 4, 5, 6]],
    [0, 1, 1, 1, 0, 0, 0],
    BASELINE_PLAGS, BASELINE_NDRAWS, 'financial',
    2, "GDP",
    Ydata_preGR, yearlab_preGR,
    "Pre-GR sample, max-FEV financial (EBP first)",
    focal_horizon=FOCAL_HORIZON,
    axis_block_name="combo",
    axis_block={"spec_id": "rc/combo/maxFEV_preGR", "ident": "max_FEV_EBP", "sample": "pre_GR"},
    seed=503)
print(f"  Max-FEV + pre-GR: coef={coef}")

# Baseline with LOOS (starts 1990Q2)
# [TFP, GDP, Consumption, Hours, LOOS, SP500, Inflation]
_, coef = run_spec(
    "rc/financial_var/LOOS", "modules/robustness/financial_variable.md", "G1",
    [0, 1, 3, 4, 11, 5, 6], [VAR_NAMES_ALL[i] for i in [0, 1, 3, 4, 11, 5, 6]],
    [1, 1, 1, 0, 0, 0, 0],
    BASELINE_PLAGS, BASELINE_NDRAWS, 'news',
    1, "GDP",
    Ydata_full, yearlab_full,
    "Full sample, LOOS as financial var (post-1990)",
    focal_horizon=FOCAL_HORIZON,
    axis_block_name="financial_var",
    axis_block={"spec_id": "rc/financial_var/LOOS", "financial_var": "LOOS"},
    seed=504)
print(f"  LOOS: coef={coef}")


# ============================================================
# INFERENCE VARIANTS
# ============================================================

print("\n=== Running inference variants ===")

infer_counter = 0
baseline_run_id = f"{PAPER_ID}_run_001"

# Wider credible intervals (5th/95th percentiles)
# We re-estimate the baseline and extract different quantiles
try:
    Y_base = Ydata_full[:, BASELINE_MODEL_INDICES].copy()
    valid_mask = ~np.any(np.isnan(Y_base), axis=1)
    first_valid = np.argmax(valid_mask)
    last_valid = len(valid_mask) - 1 - np.argmax(valid_mask[::-1])
    Y_base = Y_base[first_valid:last_valid+1, :]

    impulses_base, T_base = estimate_var_bayesian(
        Y_base, BASELINE_PLAGS, BASELINE_LEV, ndraws=BASELINE_NDRAWS, seed=0)

    # 5/95 percentiles
    low5 = np.nanpercentile(impulses_base, 5, axis=2)
    high95 = np.nanpercentile(impulses_base, 95, axis=2)
    median_irf = np.nanmedian(impulses_base, axis=2)

    h_idx = FOCAL_HORIZON - 1
    coef_infer = float(median_irf[GDP_IDX_BASELINE, h_idx])
    ci_lo_wide = float(low5[GDP_IDX_BASELINE, h_idx])
    ci_hi_wide = float(high95[GDP_IDX_BASELINE, h_idx])
    se_wide = float((ci_hi_wide - ci_lo_wide) / (2 * 1.645))  # 90% CI -> SE

    if se_wide > 0:
        z_stat = abs(coef_infer) / se_wide
        from scipy.stats import norm
        pval_wide = float(2 * (1 - norm.cdf(z_stat)))
    else:
        pval_wide = np.nan

    infer_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{infer_counter:03d}"

    payload = make_success_payload(
        coefficients={"irf_h8": coef_infer},
        inference={"spec_id": "infer/bayesian/posterior_5_95",
                   "method": "bayesian_posterior_wide", "quantiles": [0.05, 0.95]},
        software=SW_BLOCK,
        surface_hash=SURFACE_HASH,
        design={"structural_var": design_audit},
    )

    inference_results.append({
        "paper_id": PAPER_ID,
        "inference_run_id": infer_run_id,
        "spec_run_id": baseline_run_id,
        "spec_id": "infer/bayesian/posterior_5_95",
        "spec_tree_path": "modules/inference/bayesian.md",
        "baseline_group_id": "G1",
        "coefficient": coef_infer,
        "std_error": se_wide,
        "p_value": pval_wide,
        "ci_lower": ci_lo_wide,
        "ci_upper": ci_hi_wide,
        "n_obs": int(Y_base.shape[0]),
        "r_squared": np.nan,
        "coefficient_vector_json": json.dumps(payload),
        "cluster_var": "bayesian_posterior",
        "run_success": 1,
        "run_error": ""
    })
    print(f"  Wide CI: coef={coef_infer:.6f}, CI=[{ci_lo_wide:.6f}, {ci_hi_wide:.6f}]")

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
        "spec_tree_path": "modules/inference/bayesian.md",
        "baseline_group_id": "G1",
        "coefficient": np.nan,
        "std_error": np.nan,
        "p_value": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "n_obs": np.nan,
        "r_squared": np.nan,
        "coefficient_vector_json": json.dumps(payload),
        "cluster_var": "bayesian_posterior",
        "run_success": 0,
        "run_error": err_msg
    })
    print(f"  Wide CI FAILED: {err_msg[:80]}")

# Frequentist OLS VAR with bootstrap
try:
    # Simple OLS VAR (no prior) with bootstrap
    Y_base2 = Ydata_full[:, BASELINE_MODEL_INDICES].copy()
    valid_mask2 = ~np.any(np.isnan(Y_base2), axis=1)
    first_valid2 = np.argmax(valid_mask2)
    last_valid2 = len(valid_mask2) - 1 - np.argmax(valid_mask2[::-1])
    Y_ols = Y_base2[first_valid2:last_valid2+1, :]

    T_ols, M_ols = Y_ols.shape
    p_ols = BASELINE_PLAGS

    ylag_ols = mlag2(Y_ols, p_ols)
    ylag_ols = ylag_ols[p_ols:, :]
    y_ols = Y_ols[p_ols:, :]
    x_ols = np.column_stack([ylag_ols, np.ones(y_ols.shape[0])])

    b_ols = np.linalg.lstsq(x_ols, y_ols, rcond=None)[0]
    res_ols = y_ols - x_ols @ b_ols
    vmat_ols = res_ols.T @ res_ols / y_ols.shape[0]

    # Cholesky
    shock_ols = np.linalg.cholesky(vmat_ols)

    # Companion form
    B_ols = np.zeros((M_ols * p_ols, M_ols * p_ols))
    B_ols[:M_ols, :] = b_ols[:-1, :].T
    if p_ols > 1:
        B_ols[M_ols:, :M_ols*(p_ols-1)] = np.eye(M_ols * (p_ols - 1))

    nhor_ols = 40
    irf_ols = IRFVAR(B_ols, shock_ols, p_ols, nhor_ols)

    # MFEVD identification
    row_indices = np.arange(M_ols, M_ols * M_ols, M_ols)
    V9_ols = irf_ols[row_indices, :nhor_ols].T
    WWW_ols = np.zeros((M_ols - 1, M_ols - 1))
    WWW_ols += np.outer(V9_ols[nhor_ols - 1, :], V9_ols[nhor_ols - 1, :])
    eigenvalues_ols, eigenvectors_ols = np.linalg.eigh(WWW_ols)
    q_vec_ols = eigenvectors_ols[:, np.argmax(eigenvalues_ols)]
    gamma_ols = np.concatenate([[0.0], q_vec_ols])

    newshock_ols = np.zeros((M_ols, nhor_ols))
    for i_var in range(M_ols):
        row_idx_var = np.arange(i_var, M_ols * M_ols, M_ols)
        newshock_ols[i_var, :] = irf_ols[row_idx_var, :nhor_ols].T @ gamma_ols
    if newshock_ols[0, nhor_ols - 1] < 0:
        newshock_ols = -newshock_ols

    coef_ols = float(newshock_ols[GDP_IDX_BASELINE, FOCAL_HORIZON - 1])

    # Bootstrap
    n_boot = 500
    rng_boot = np.random.RandomState(130141)
    boot_coefs = []
    T_eff_ols = y_ols.shape[0]

    for b_iter in range(n_boot):
        try:
            boot_idx = rng_boot.randint(0, T_eff_ols, size=T_eff_ols)
            boot_resid = res_ols[boot_idx, :]
            Y_boot = np.zeros_like(Y_ols)
            Y_boot[:p_ols, :] = Y_ols[:p_ols, :]
            for t in range(p_ols, T_ols):
                x_lag_t = np.zeros(M_ols * p_ols)
                for lag in range(p_ols):
                    x_lag_t[lag*M_ols:(lag+1)*M_ols] = Y_boot[t-lag-1, :]
                x_t = np.concatenate([x_lag_t, [1.0]])
                Y_boot[t, :] = x_t @ b_ols + boot_resid[t - p_ols, :]

            ylag_b = mlag2(Y_boot, p_ols)[p_ols:, :]
            y_b = Y_boot[p_ols:, :]
            x_b = np.column_stack([ylag_b, np.ones(y_b.shape[0])])
            b_b = np.linalg.lstsq(x_b, y_b, rcond=None)[0]
            res_b = y_b - x_b @ b_b
            vmat_b = res_b.T @ res_b / y_b.shape[0]
            shock_b = np.linalg.cholesky(vmat_b)

            B_b = np.zeros((M_ols * p_ols, M_ols * p_ols))
            B_b[:M_ols, :] = b_b[:-1, :].T
            if p_ols > 1:
                B_b[M_ols:, :M_ols*(p_ols-1)] = np.eye(M_ols * (p_ols - 1))

            irf_b = IRFVAR(B_b, shock_b, p_ols, nhor_ols)
            row_idx_b = np.arange(M_ols, M_ols * M_ols, M_ols)
            V9_b = irf_b[row_idx_b, :nhor_ols].T
            WWW_b = np.outer(V9_b[nhor_ols - 1, :], V9_b[nhor_ols - 1, :])
            ev_b, evec_b = np.linalg.eigh(WWW_b)
            gamma_b = np.concatenate([[0.0], evec_b[:, np.argmax(ev_b)]])

            ns_b = np.zeros((M_ols, nhor_ols))
            for i_var in range(M_ols):
                row_idx_var = np.arange(i_var, M_ols * M_ols, M_ols)
                ns_b[i_var, :] = irf_b[row_idx_var, :nhor_ols].T @ gamma_b
            if ns_b[0, nhor_ols - 1] < 0:
                ns_b = -ns_b

            boot_coefs.append(float(ns_b[GDP_IDX_BASELINE, FOCAL_HORIZON - 1]))
        except:
            continue

    if len(boot_coefs) >= 50:
        boot_arr = np.array(boot_coefs)
        ci_lo_boot = float(np.percentile(boot_arr, 16))
        ci_hi_boot = float(np.percentile(boot_arr, 84))
        se_boot = float(np.std(boot_arr))
        if se_boot > 0:
            z_boot = abs(coef_ols) / se_boot
            from scipy.stats import norm
            pval_boot = float(2 * (1 - norm.cdf(z_boot)))
        else:
            pval_boot = np.nan
    else:
        ci_lo_boot = np.nan
        ci_hi_boot = np.nan
        se_boot = np.nan
        pval_boot = np.nan

    infer_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{infer_counter:03d}"

    payload = make_success_payload(
        coefficients={"irf_h8": coef_ols},
        inference={"spec_id": "infer/frequentist/bootstrap",
                   "method": "residual_bootstrap", "n_boot": n_boot},
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
        "coefficient": coef_ols,
        "std_error": se_boot,
        "p_value": pval_boot,
        "ci_lower": ci_lo_boot,
        "ci_upper": ci_hi_boot,
        "n_obs": int(T_ols),
        "r_squared": np.nan,
        "coefficient_vector_json": json.dumps(payload),
        "cluster_var": "residual_bootstrap",
        "run_success": 1,
        "run_error": ""
    })
    print(f"  Bootstrap: coef={coef_ols:.6f}, SE={se_boot:.6f}, CI=[{ci_lo_boot:.6f}, {ci_hi_boot:.6f}]")

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
        print(f"\nBaseline IRF(GDP, TFP news shock, h={FOCAL_HORIZON}): {base_row['coefficient'].values[0]:.6f}")
        print(f"Baseline N: {base_row['n_obs'].values[0]:.0f}")

    gdp_specs = successful[successful['outcome_var'].str.startswith('GDP')]
    if len(gdp_specs) > 0:
        print(f"\n=== GDP IRF SPECS ===")
        print(f"Count: {len(gdp_specs)}")
        print(f"Min coef: {gdp_specs['coefficient'].min():.6f}")
        print(f"Max coef: {gdp_specs['coefficient'].max():.6f}")
        print(f"Median coef: {gdp_specs['coefficient'].median():.6f}")

    print(f"\n=== ALL SPECS ===")
    print(f"Min coef: {successful['coefficient'].min():.6f}")
    print(f"Max coef: {successful['coefficient'].max():.6f}")
    print(f"Median coef: {successful['coefficient'].median():.6f}")


# ============================================================
# SPECIFICATION_SEARCH.md
# ============================================================

print("\nWriting SPECIFICATION_SEARCH.md...")

md_lines = []
md_lines.append("# Specification Search Report: 130141-V1")
md_lines.append("")
md_lines.append("**Paper:** Gortz, Tsoukalas & Zanetti (2018), \"News Shocks under Financial Frictions\", AEJ: Macroeconomics 10(4)")
md_lines.append("")
md_lines.append("## Baseline Specification")
md_lines.append("")
md_lines.append("- **Design:** Structural Bayesian VAR with Minnesota prior (BIC-selected hyperparameters)")
md_lines.append("- **Identification:** Medium-run MFEVD (maximize forecast error variance of TFP at h=40)")
md_lines.append("- **Outcome:** GDP impulse response to TFP news shock")
md_lines.append(f"- **Focal horizon:** h={FOCAL_HORIZON} quarters")
md_lines.append(f"- **VAR lags:** {BASELINE_PLAGS}")
md_lines.append(f"- **Variables:** {', '.join(BASELINE_VAR_NAMES)}")
md_lines.append("- **Sample:** Quarterly, 1984Q1-2017Q1 (133 observations)")
md_lines.append(f"- **Posterior draws:** {BASELINE_NDRAWS}")
md_lines.append("")

if len(successful) > 0 and len(base_row) > 0:
    bc = base_row.iloc[0]
    md_lines.append("| Statistic | Value |")
    md_lines.append("|-----------|-------|")
    md_lines.append(f"| IRF (h={FOCAL_HORIZON}) | {bc['coefficient']:.6f} |")
    md_lines.append(f"| SE (posterior) | {bc['std_error']:.6f} |")
    md_lines.append(f"| 68% CI | [{bc['ci_lower']:.6f}, {bc['ci_upper']:.6f}] |")
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
    "Financial Variable": successful[successful['spec_id'].str.startswith('rc/financial_var/')],
    "Add Investment": successful[successful['spec_id'].str.startswith('rc/add_investment')],
    "Sample Period": successful[successful['spec_id'].str.startswith('rc/sample/')],
    "Identification": successful[successful['spec_id'].str.startswith('rc/ident/')],
    "IRF Horizon": successful[successful['spec_id'].str.startswith('rc/horizon/')],
    "Outcome Variable": successful[successful['spec_id'].str.startswith('rc/outcome/')],
    "Variable Ordering": successful[successful['spec_id'].str.startswith('rc/ordering/')],
    "Posterior Draws": successful[successful['spec_id'].str.startswith('rc/ndraws/')],
    "Grid (lags x fin)": successful[successful['spec_id'].str.contains('grid.*p.*_')],
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
    md_lines.append("| Spec ID | SE | p-value | CI |")
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
    gdp_h8 = successful[
        (successful['outcome_var'] == f'GDP_irf_h{FOCAL_HORIZON}') &
        (successful['treatment_var'] == 'TFP_news_shock')
    ]
    if len(gdp_h8) > 0:
        sign_consistent = ((gdp_h8['coefficient'] > 0).sum() == len(gdp_h8)) or \
                          ((gdp_h8['coefficient'] < 0).sum() == len(gdp_h8))
        median_coef = gdp_h8['coefficient'].median()
        sign_word = "positive" if median_coef > 0 else "negative"

        md_lines.append(f"- **GDP IRF specs (h={FOCAL_HORIZON}, TFP news shock):** {len(gdp_h8)} specifications")
        md_lines.append(f"- **Sign consistency:** {'All have the same sign' if sign_consistent else 'Mixed signs across specifications'}")
        md_lines.append(f"- **Direction:** Median IRF is {sign_word} ({median_coef:.6f})")
        md_lines.append(f"- **Range:** [{gdp_h8['coefficient'].min():.6f}, {gdp_h8['coefficient'].max():.6f}]")

        if sign_consistent:
            strength = "STRONG"
        else:
            n_pos = (gdp_h8['coefficient'] > 0).sum()
            n_neg = (gdp_h8['coefficient'] < 0).sum()
            if max(n_pos, n_neg) / len(gdp_h8) >= 0.8:
                strength = "MODERATE"
            elif max(n_pos, n_neg) / len(gdp_h8) >= 0.6:
                strength = "WEAK"
            else:
                strength = "FRAGILE"

        md_lines.append(f"- **Robustness assessment:** {strength}")
    else:
        md_lines.append(f"- No GDP h={FOCAL_HORIZON} TFP news shock specifications to assess.")

md_lines.append("")
md_lines.append(f"Surface hash: `{SURFACE_HASH}`")
md_lines.append("")
md_lines.append("## Notes")
md_lines.append("")
md_lines.append("- This paper uses a Bayesian VAR with Minnesota prior. Hyperparameters are selected via BIC grid search.")
md_lines.append("- TFP news shock identified via medium-run MFEVD: the rotation that maximizes the forecast error variance")
md_lines.append("  of TFP at the VAR horizon, subject to zero contemporaneous effect on TFP (Barsky-Sims type).")
md_lines.append("- The specification search varies: lag length (2-8), financial variable in VAR (GZ spread, EBP, default risk,")
md_lines.append("  bank equity, BAA spread, or none), sample period (full vs. pre-Great Recession vs. post-1990),")
md_lines.append("  identification scheme (MFEVD news vs. max-FEV financial), IRF horizon, outcome variable, and variable ordering.")
md_lines.append("- Posterior draws from Normal-Inverse-Wishart distribution, with credible intervals at 16th/84th percentiles.")
md_lines.append("")

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write("\n".join(md_lines))

print(f"Wrote SPECIFICATION_SEARCH.md")
print("\nDone!")
