#!/usr/bin/env python3
"""
06_counterfactual.py
====================

Compute counterfactual screening under cost shift.

Given:
- Mixture parameters (pi_k, mu_k, sigma_k) for k in {N, H, L}
  using sigma=1 fixed truncated-normal mixture
- Dependence parameter phi (hence Delta = 1 - phi) from AR(1) under by_category ordering
- Cost ratio lambda = gamma^new / gamma^old from timing data

Compute:
- Window optimization: B=[z_lo, +inf) with z_lo optimized
- Calibration: binary-search N_eff_old such that FDR(m=50, N_eff_old)=0.05
- Main result: m_new for new regime
- Disclosure scaling: m_old=1..100
- Sensitivity: lambda, phi (all 5 orderings), z_lo, mixture variants
- Operating-point curves

All FDR computations use null-only FDR: FDR = pi_N * Q_N / Q_bar.

Output:
  - estimation/results/counterfactual.csv
  - estimation/results/counterfactual_params.json
  - estimation/results/counterfactual_dependence_sensitivity.csv
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "estimation" / "results"
MIXTURE_FILE = RESULTS_DIR / "mixture_params_abs_t.json"
DEPENDENCE_FILE = RESULTS_DIR / "dependence.json"
TIMING_CSV = BASE_DIR / "timing_analyses.csv"
OUTPUT_CSV = RESULTS_DIR / "counterfactual.csv"
OUTPUT_JSON = RESULTS_DIR / "counterfactual_params.json"
OUTPUT_DEP_SENS = RESULTS_DIR / "counterfactual_dependence_sensitivity.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
M_OLD_BASELINE = 50
FDR_TARGET = 0.05
Z_HI = None  # No upper bound on |t| (tail-probability thresholding)
LAMBDA_FALLBACK = 1 / 172

# Orderings to include in sensitivity
ORDERING_KEYS = ["spec_order", "lex_path", "bfs", "dfs", "by_category"]

def _is_infinite_upper_bound(z_hi) -> bool:
    if z_hi is None:
        return True
    try:
        return not np.isfinite(float(z_hi))
    except Exception:
        return True


def _fmt_z_hi(z_hi) -> str:
    return "+inf" if _is_infinite_upper_bound(z_hi) else f"{float(z_hi):.2f}"


def _z_hi_to_scalar(z_hi) -> float:
    """For CSV-friendly numeric columns: use NaN for +inf/None."""
    return float("nan") if _is_infinite_upper_bound(z_hi) else float(z_hi)


def _z_hi_to_json(z_hi) -> float | None:
    """For strict JSON: use null for +inf/None."""
    return None if _is_infinite_upper_bound(z_hi) else float(z_hi)


# ---------------------------------------------------------------------------
# Lambda from timing data
# ---------------------------------------------------------------------------
def load_lambda_from_timing() -> tuple[float, dict]:
    """Load lambda from timing_analyses.csv.

    Lambda = mean(total_agent_time_s) / (40 * 3600).
    Returns (lambda_value, timing_metadata_dict).
    """
    meta: dict = {"timing_csv": str(TIMING_CSV), "n_papers": 0}
    if not TIMING_CSV.exists():
        print(f"  Warning: timing CSV not found at {TIMING_CSV}, using fallback lambda={LAMBDA_FALLBACK:.6f}")
        meta["status"] = "csv_missing"
        return LAMBDA_FALLBACK, meta

    df = pd.read_csv(TIMING_CSV)
    # Exclude failed papers (e.g. timeouts) for clean cost estimate
    if "spec_search_success" in df.columns:
        df = df[df["spec_search_success"].astype(str).str.strip().str.lower() == "true"]
    times = pd.to_numeric(df.get("total_agent_time_s", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(times) == 0:
        print(f"  Warning: no valid total_agent_time_s in timing CSV, using fallback lambda={LAMBDA_FALLBACK:.6f}")
        meta["status"] = "no_valid_rows"
        return LAMBDA_FALLBACK, meta

    mean_time = float(times.mean())
    lam = mean_time / (40 * 3600)
    lam_per_paper = times / (40 * 3600)
    meta.update({
        "status": "ok",
        "n_papers": int(len(times)),
        "mean_agent_time_s": float(mean_time),
        "median_agent_time_s": float(times.median()),
        "min_agent_time_s": float(times.min()),
        "max_agent_time_s": float(times.max()),
        "lambda_computed": float(lam),
        "lambda_p25": float(lam_per_paper.quantile(0.25)),
        "lambda_p50": float(lam_per_paper.quantile(0.50)),
        "lambda_p75": float(lam_per_paper.quantile(0.75)),
        "lambda_p90": float(lam_per_paper.quantile(0.90)),
    })
    return lam, meta


# ---------------------------------------------------------------------------
# Parameter loading
# ---------------------------------------------------------------------------
def load_parameters():
    """Load mixture (sigma=1 fixed) and dependence parameters."""
    # Load sigma=1 fixed mixture params (preferred)
    if not MIXTURE_FILE.exists():
        raise RuntimeError(f"Mixture file not found: {MIXTURE_FILE}")

    with open(MIXTURE_FILE, 'r') as f:
        mixture_all = json.load(f)

    params = mixture_all.get("spec_level", {}).get("baseline_only_sigma_fixed_1")
    if params is None:
        raise RuntimeError("baseline_only_sigma_fixed_1 mixture params not found in mixture_params_abs_t.json")
    mixture_source = "spec_level:baseline_only_sigma_fixed_1"

    # Load dependence params
    if not DEPENDENCE_FILE.exists():
        raise RuntimeError(f"Dependence file not found: {DEPENDENCE_FILE}")

    with open(DEPENDENCE_FILE, 'r') as f:
        dependence = json.load(f)

    phi = dependence.get('preferred', {}).get('phi', 0.187)
    Delta = dependence.get('preferred', {}).get('Delta', 0.813)

    # Also load all mixture variants for sensitivity
    mixture_variants = {}

    # sigma-free (baseline-only)
    sf = mixture_all.get("spec_level", {}).get("baseline_only", None)
    if sf is not None:
        mixture_variants["sigma_free"] = sf

    # sigma=1 (baseline)
    mixture_variants["sigma_fixed_1"] = params

    # K=2
    k2 = mixture_all.get("k_sensitivity", {}).get("K=2", {}).get("truncnorm")
    if k2 is not None:
        mixture_variants["K2"] = k2

    # K=4
    k4 = mixture_all.get("k_sensitivity", {}).get("K=4", {}).get("truncnorm")
    if k4 is not None:
        mixture_variants["K4"] = k4

    return params, phi, Delta, dependence, mixture_source, mixture_variants


# ---------------------------------------------------------------------------
# Core probability computations
# ---------------------------------------------------------------------------
def compute_pass_probabilities(B, params):
    """
    Compute pass probabilities for each type given window B.
    B = (z_lo, z_hi) on the |z| scale.
    Supports truncnorm and gaussian distributions.
    """
    z_lo, z_hi = B
    probs = {}
    dist = str(params.get("distribution", "gaussian"))

    # Determine component keys (K=2 uses Low/High, K=4 uses N/H1/H2/L, K=3 uses N/H/L)
    pi_keys = list(params['pi'].keys())

    for k in pi_keys:
        mu_k = float(params['mu'][k])
        sigma_k = float(max(params['sigma'][k], 1e-8))

        if dist == "truncnorm":
            lo = float(params.get("truncation_lo", 0.0))

            def tcdf(x):
                if x is None or not np.isfinite(x):
                    return 1.0
                x = float(x)
                if x <= lo:
                    return 0.0
                alpha = (lo - mu_k) / sigma_k
                denom = float(max(stats.norm.sf(alpha), 1e-12))
                return float((stats.norm.cdf((x - mu_k) / sigma_k) - stats.norm.cdf(alpha)) / denom)

            cdf_hi = tcdf(z_hi)
            cdf_lo = 0.0 if (z_lo is None or float(z_lo) <= lo) else tcdf(z_lo)
            p_k = float(max(0.0, min(1.0, cdf_hi - cdf_lo)))
        else:
            cdf_hi = 1.0 if (z_hi is None or (isinstance(z_hi, float) and not np.isfinite(z_hi))) else float(stats.norm.cdf((float(z_hi) - mu_k) / sigma_k))
            cdf_lo = 0.0 if (z_lo is None or float(z_lo) <= -np.inf) else float(stats.norm.cdf((float(z_lo) - mu_k) / sigma_k))
            p_k = float(max(0.0, min(1.0, cdf_hi - cdf_lo)))

        probs[k] = p_k

    return probs


def binomial_tail(n, m, p):
    """P(Bin(n, p) >= m)."""
    if p <= 0:
        return 0.0 if m > 0 else 1.0
    if p >= 1:
        return 1.0
    if n < m:
        return 0.0
    return float(stats.binom.sf(m - 1, int(n), p))


def fdr_null_only(n_eff, m, pass_probs, params):
    """
    Compute null-only FDR = pi_N * Q_N / Q_bar.
    For K=3 mixtures with keys N, H, L.
    """
    pi = params['pi']
    Q = {k: binomial_tail(n_eff, m, pass_probs[k]) for k in pi.keys()}
    Q_bar = sum(pi[k] * Q[k] for k in pi.keys())
    if Q_bar <= 0:
        return np.nan, Q, Q_bar

    # Null-only: only N-type in numerator
    null_key = 'N' if 'N' in pi else list(pi.keys())[0]
    fdr = (pi[null_key] * Q[null_key]) / Q_bar
    return float(fdr), Q, float(Q_bar)


def screening_metrics(n_eff, m, pass_probs, params):
    """Compute full screening metrics."""
    pi = params['pi']
    Q = {k: binomial_tail(n_eff, m, pass_probs[k]) for k in pi.keys()}
    Q_bar = sum(pi[k] * Q[k] for k in pi.keys())

    null_key = 'N' if 'N' in pi else list(pi.keys())[0]
    fdr_null = (pi[null_key] * Q[null_key]) / Q_bar if Q_bar > 0 else np.nan

    result = {'Q_bar': Q_bar, 'FDR_null': fdr_null}
    for k in pi.keys():
        result[f'Q_{k}'] = Q[k]
    return result


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def _calibrate_n_eff(m_target, pass_probs, params, fdr_target=0.05):
    """Find n_eff such that FDR_null(m_target, n_eff) ≈ fdr_target.

    Binary search: FDR is monotonically increasing in n_eff for fixed m.
    """
    # First find upper bound where FDR > target
    lo, hi = m_target, m_target
    for n in range(m_target, 100001):
        fdr, _, _ = fdr_null_only(n, m_target, pass_probs, params)
        if np.isfinite(fdr) and fdr > fdr_target:
            hi = n
            break
    else:
        return 100000  # FDR never exceeds target

    # Binary search between lo and hi
    lo = m_target
    while lo < hi - 1:
        mid = (lo + hi) // 2
        fdr, _, _ = fdr_null_only(mid, m_target, pass_probs, params)
        if not np.isfinite(fdr) or fdr <= fdr_target:
            lo = mid
        else:
            hi = mid

    return lo


def find_min_m(n_eff, pass_probs, params, fdr_target=0.05):
    """Find smallest m such that FDR_null(m, n_eff) <= fdr_target."""
    for m in range(1, int(n_eff) + 1):
        fdr, Q, Q_bar = fdr_null_only(n_eff, m, pass_probs, params)
        if np.isfinite(fdr) and fdr <= fdr_target:
            return {'m': int(m), 'FDR_null': float(fdr), 'Q_bar': float(Q_bar)}
    return None


# ---------------------------------------------------------------------------
# Window optimization
# ---------------------------------------------------------------------------
def optimize_window(params, fdr_target=0.05, m_old=M_OLD_BASELINE):
    """
    Grid-search z_lo in [0.5, 5.0] with no upper bound (z_hi=None).
    Maximize p_H subject to FDR=0.05 being achievable at m_old.
    """
    z_lo_grid = np.arange(0.5, 5.01, 0.05)
    best = None
    best_p_H = -1.0

    for z_lo in z_lo_grid:
        B = (float(z_lo), Z_HI)
        p = compute_pass_probabilities(B, params)
        p_H = p.get('H', 0)
        if p_H < 0.01:
            continue

        # Check FDR is achievable: can we find n_eff where FDR(m_old, n_eff)=target?
        n_eff = _calibrate_n_eff(m_old, p, params, fdr_target)
        fdr_check, _, _ = fdr_null_only(n_eff, m_old, p, params)
        if not np.isfinite(fdr_check) or fdr_check > fdr_target * 1.1:
            continue

        if p_H > best_p_H:
            best_p_H = p_H
            best = {
                'z_lo': float(z_lo),
                'z_hi': Z_HI,
                'B': B,
                'pass_probabilities': p,
                'calibrated_n_eff': n_eff,
            }

    if best is None:
        # Fallback: standard significance threshold
        B = (1.96, Z_HI)
        p = compute_pass_probabilities(B, params)
        n_eff = _calibrate_n_eff(m_old, p, params, fdr_target)
        best = {
            'z_lo': 1.96,
            'z_hi': Z_HI,
            'B': B,
            'pass_probabilities': p,
            'calibrated_n_eff': n_eff,
        }

    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Computing Counterfactual Screening")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load parameters
    # ------------------------------------------------------------------
    print("\nLoading parameters...")
    params, phi, Delta, dependence, mixture_source, mixture_variants = load_parameters()

    print(f"\nMixture parameters (sigma=1 fixed):")
    print(f"  source = {mixture_source}")
    for k in ['N', 'H', 'L']:
        print(f"  {k}: pi={params['pi'][k]:.3f}, mu={params['mu'][k]:.3f}, "
              f"sigma={params['sigma'][k]:.3f}")

    print(f"\nDependence parameters (preferred = {dependence.get('preferred', {}).get('ordering', 'N/A')}):")
    print(f"  phi = {phi:.4f}")
    print(f"  Delta = {Delta:.4f}")

    # Load lambda
    print("\nLoading lambda from timing CSV...")
    LAMBDA_BASELINE, timing_meta = load_lambda_from_timing()
    print(f"  lambda = {LAMBDA_BASELINE:.6f} (1/{1/LAMBDA_BASELINE:.1f})")

    # ------------------------------------------------------------------
    # Evidence window: tail threshold B=[1.96, +inf)
    # ------------------------------------------------------------------
    B = (1.96, Z_HI)
    p_pass = compute_pass_probabilities(B, params)
    print(f"\nEvidence window B = ({float(B[0]):.2f}, {_fmt_z_hi(B[1])})")
    print(f"  Pass probabilities: p_N={p_pass['N']:.4f}, p_H={p_pass['H']:.4f}, p_L={p_pass['L']:.4f}")

    # ------------------------------------------------------------------
    # Main calibration
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Main Calibration (m_old={M_OLD_BASELINE}, FDR_target={FDR_TARGET})")
    print("=" * 60)

    # Binary-search N_eff_old: FDR(m=50, N_eff_old) = 0.05
    N_eff_old = _calibrate_n_eff(M_OLD_BASELINE, p_pass, params, FDR_TARGET)
    fdr_at_cal, _, _ = fdr_null_only(N_eff_old, M_OLD_BASELINE, p_pass, params)
    print(f"  N_eff_old = {N_eff_old}")
    print(f"  FDR at calibration: {fdr_at_cal:.6f}")

    # Interpretation: n_old = N_eff_old / Delta
    n_old_implied = N_eff_old / Delta
    print(f"  Implied n_old = N_eff_old / Delta = {n_old_implied:.1f}")

    # New regime
    N_eff_new = int(np.ceil(N_eff_old / LAMBDA_BASELINE))
    print(f"  N_eff_new = ceil(N_eff_old / lambda) = {N_eff_new}")

    # Find m_new
    new_sol = find_min_m(N_eff_new, p_pass, params, FDR_TARGET)
    if new_sol is None:
        raise RuntimeError("Could not find m_new satisfying FDR target!")
    m_new = new_sol['m']
    print(f"  m_new = {m_new}")
    print(f"  m_new / m_old = {m_new / M_OLD_BASELINE:.2f}")

    # Degradation: FDR in new regime using old m
    fdr_degrad, _, _ = fdr_null_only(N_eff_new, M_OLD_BASELINE, p_pass, params)
    print(f"  FDR(new regime, old m={M_OLD_BASELINE}) = {fdr_degrad:.4f}")

    # ------------------------------------------------------------------
    # Disclosure scaling: m_old = 1..100
    # ------------------------------------------------------------------
    print(f"\n{'-'*40}")
    print("Disclosure Scaling (m_old = 1..100)")
    print("-" * 40)

    scaling_rows = []
    for m_old in range(1, 101):
        n_eff_old_m = _calibrate_n_eff(m_old, p_pass, params, FDR_TARGET)
        n_eff_new_m = int(np.ceil(n_eff_old_m / LAMBDA_BASELINE))
        sol = find_min_m(n_eff_new_m, p_pass, params, FDR_TARGET)
        m_new_m = sol['m'] if sol else None
        scaling_rows.append({
            'm_old': m_old,
            'n_eff_old': n_eff_old_m,
            'n_eff_new': n_eff_new_m,
            'm_new': m_new_m,
            'ratio': float(m_new_m / m_old) if m_new_m else None,
        })
        if m_old in [1, 5, 10, 20, 50, 100]:
            print(f"  m_old={m_old}: n_eff_old={n_eff_old_m}, m_new={m_new_m}, "
                  f"ratio={m_new_m/m_old:.1f}" if m_new_m else f"  m_old={m_old}: infeasible")

    scaling_df = pd.DataFrame(scaling_rows)

    # ------------------------------------------------------------------
    # Sensitivity: lambda
    # ------------------------------------------------------------------
    print(f"\n{'-'*40}")
    print("Sensitivity: Lambda")
    print("-" * 40)

    lambda_grid = [1/50, 1/100, LAMBDA_BASELINE, 1/250, 1/500]
    # Add quantile lambdas from timing data
    for qk in ["lambda_p25", "lambda_p50", "lambda_p75", "lambda_p90"]:
        qv = timing_meta.get(qk)
        if qv is not None:
            lambda_grid.append(qv)
    lambda_grid = sorted(set(lambda_grid), reverse=True)

    lambda_sens = []
    for lam in lambda_grid:
        n_eff_new_l = int(np.ceil(N_eff_old / lam))
        sol = find_min_m(n_eff_new_l, p_pass, params, FDR_TARGET)
        m_new_l = sol['m'] if sol else None
        lambda_sens.append({
            'lambda': float(lam),
            'lambda_inv': float(1/lam),
            'n_eff_new': n_eff_new_l,
            'm_new': m_new_l,
            'ratio': float(m_new_l / M_OLD_BASELINE) if m_new_l else None,
        })
        print(f"  lambda=1/{1/lam:.0f}: m_new={m_new_l}, ratio={m_new_l/M_OLD_BASELINE:.1f}" if m_new_l else f"  lambda=1/{1/lam:.0f}: infeasible")

    # ------------------------------------------------------------------
    # Sensitivity: phi (all 5 non-random orderings)
    # ------------------------------------------------------------------
    print(f"\n{'-'*40}")
    print("Sensitivity: Phi (AR(1) orderings)")
    print("-" * 40)

    ar1_ords = dependence.get("ar1_orderings", {})
    preferred_ordering = dependence.get("preferred", {}).get("ordering", "by_category")

    phi_sens = []
    dep_rows = []
    for key in ORDERING_KEYS:
        if key not in ar1_ords:
            continue
        ord_data = ar1_ords[key]
        phi_v = float(ord_data.get("phi", np.nan))
        Delta_v = float(ord_data.get("Delta", np.nan))
        phi_ci_lo = ord_data.get("phi_ci_lower", None)
        phi_ci_hi = ord_data.get("phi_ci_upper", None)
        if not np.isfinite(phi_v):
            continue

        # Phi enters for interpretation only: n_old = N_eff_old / Delta_v
        n_old_interp = N_eff_old / Delta_v
        n_new_interp = N_eff_new / Delta_v

        phi_sens.append({
            'ordering': key,
            'phi': phi_v,
            'Delta': Delta_v,
            'phi_ci_lower': float(phi_ci_lo) if phi_ci_lo is not None else None,
            'phi_ci_upper': float(phi_ci_hi) if phi_ci_hi is not None else None,
            'is_preferred': key == preferred_ordering,
            'n_old_implied': float(n_old_interp),
            'n_new_implied': float(n_new_interp),
        })

        # Dependence sensitivity rows for the counterfactual appendix table.
        dep_rows.append({
            "dependence_label": key,
            "phi": float(phi_v),
            "Delta": float(Delta_v),
            "lambda": float(LAMBDA_BASELINE),
            "FDR_target": float(FDR_TARGET),
            "n_eff_old": int(N_eff_old),
            "n_eff_new": int(N_eff_new),
            "m_old": int(M_OLD_BASELINE),
            "m_new": int(m_new),
            "m_ratio": float(m_new / M_OLD_BASELINE),
            "n_old_implied": float(n_old_interp),
        })

        print(f"  {key}: phi={phi_v:.4f}, Delta={Delta_v:.4f}, "
              f"n_old_implied={n_old_interp:.1f}"
              + (" [preferred]" if key == preferred_ordering else ""))

    # Also add CI bounds from preferred ordering
    pref_data = ar1_ords.get(preferred_ordering, {})
    ci_lo = pref_data.get("phi_ci_lower", None)
    ci_hi = pref_data.get("phi_ci_upper", None)
    if ci_lo is not None and ci_hi is not None:
        for label, phi_ci in [("preferred_ci_low_phi", float(ci_lo)),
                               ("preferred_ci_high_phi", float(ci_hi))]:
            Delta_ci = 1 - phi_ci
            n_old_ci = N_eff_old / Delta_ci
            phi_sens.append({
                'ordering': label,
                'phi': phi_ci,
                'Delta': Delta_ci,
                'is_preferred': False,
                'n_old_implied': float(n_old_ci),
            })
            dep_rows.append({
                "dependence_label": label,
                "phi": float(phi_ci),
                "Delta": float(Delta_ci),
                "lambda": float(LAMBDA_BASELINE),
                "FDR_target": float(FDR_TARGET),
                "n_eff_old": int(N_eff_old),
                "n_eff_new": int(N_eff_new),
                "m_old": int(M_OLD_BASELINE),
                "m_new": int(m_new),
                "m_ratio": float(m_new / M_OLD_BASELINE),
                "n_old_implied": float(n_old_ci),
            })

    dep_df = pd.DataFrame(dep_rows)

    # ------------------------------------------------------------------
    # Sensitivity: z_lo
    # ------------------------------------------------------------------
    print(f"\n{'-'*40}")
    print("Sensitivity: z_lo")
    print("-" * 40)

    z_lo_grid = [0.5, 1.0, 1.5, 1.96, 2.5, 3.0, 4.0, 5.0]
    z_lo_sens = []
    for z_lo_v in z_lo_grid:
        B_v = (z_lo_v, Z_HI)
        p_v = compute_pass_probabilities(B_v, params)
        n_eff_old_v = _calibrate_n_eff(M_OLD_BASELINE, p_v, params, FDR_TARGET)
        n_eff_new_v = int(np.ceil(n_eff_old_v / LAMBDA_BASELINE))
        sol = find_min_m(n_eff_new_v, p_v, params, FDR_TARGET)
        m_new_v = sol['m'] if sol else None
        z_lo_sens.append({
            'z_lo': z_lo_v,
            'p_N': float(p_v['N']),
            'p_H': float(p_v['H']),
            'p_L': float(p_v['L']),
            'n_eff_old': n_eff_old_v,
            'n_eff_new': n_eff_new_v,
            'm_new': m_new_v,
            'ratio': float(m_new_v / M_OLD_BASELINE) if m_new_v else None,
        })
        print(f"  z_lo={z_lo_v:.2f}: m_new={m_new_v}, ratio={m_new_v/M_OLD_BASELINE:.1f}" if m_new_v else f"  z_lo={z_lo_v:.2f}: infeasible")

    # ------------------------------------------------------------------
    # Sensitivity: mixture variants
    # ------------------------------------------------------------------
    print(f"\n{'-'*40}")
    print("Sensitivity: Mixture Variants")
    print("-" * 40)

    mix_sens = []
    for mix_name, mix_params in mixture_variants.items():
        # Compute pass probs with this mixture's parameters
        p_m = compute_pass_probabilities(B, mix_params)

        # Only run for K=3 mixtures with N, H, L keys
        pi_keys = list(mix_params['pi'].keys())
        if 'N' not in pi_keys:
            print(f"  {mix_name}: skipping (no N key, keys={pi_keys})")
            continue

        n_eff_old_m = _calibrate_n_eff(M_OLD_BASELINE, p_m, mix_params, FDR_TARGET)
        n_eff_new_m = int(np.ceil(n_eff_old_m / LAMBDA_BASELINE))
        sol = find_min_m(n_eff_new_m, p_m, mix_params, FDR_TARGET)
        m_new_m = sol['m'] if sol else None
        mix_sens.append({
            'mixture': mix_name,
            'pi': {k: float(v) for k, v in mix_params['pi'].items()},
            'mu': {k: float(v) for k, v in mix_params['mu'].items()},
            'sigma': {k: float(v) for k, v in mix_params['sigma'].items()},
            'n_eff_old': n_eff_old_m,
            'n_eff_new': n_eff_new_m,
            'm_new': m_new_m,
            'ratio': float(m_new_m / M_OLD_BASELINE) if m_new_m else None,
        })
        print(f"  {mix_name}: m_new={m_new_m}, ratio={m_new_m/M_OLD_BASELINE:.1f}" if m_new_m else f"  {mix_name}: infeasible")

    # ------------------------------------------------------------------
    # Operating-point curves
    # ------------------------------------------------------------------
    print(f"\n{'-'*40}")
    print("Operating Point Curves")
    print("-" * 40)

    op_rows = []
    for regime, n_eff in [('old', N_eff_old), ('new', N_eff_new)]:
        for m in range(1, n_eff + 1):
            met = screening_metrics(n_eff, m, p_pass, params)
            if met['Q_bar'] <= 0:
                continue
            op_rows.append({
                'regime': regime,
                'n_eff': int(n_eff),
                'm': int(m),
                **met,
            })
    print(f"  Generated {len(op_rows)} operating point rows")

    # ------------------------------------------------------------------
    # Build main results CSV
    # ------------------------------------------------------------------
    results_rows = []
    for lam_entry in lambda_sens:
        lam = lam_entry['lambda']
        n_eff_new_l = int(np.ceil(N_eff_old / lam))

        # Find m for old and new at this lambda
        old_sol = find_min_m(N_eff_old, p_pass, params, FDR_TARGET)
        new_sol_l = find_min_m(n_eff_new_l, p_pass, params, FDR_TARGET)
        if old_sol is None or new_sol_l is None:
            continue

        fdr_deg, _, _ = fdr_null_only(n_eff_new_l, old_sol['m'], p_pass, params)

        results_rows.append({
            'lambda': float(lam),
            'FDR_target': float(FDR_TARGET),
            'B_lo': float(B[0]),
            'B_hi': _z_hi_to_scalar(B[1]),
            'n_eff_old': N_eff_old,
            'n_eff_new': n_eff_new_l,
            'm_old': old_sol['m'],
            'm_new': new_sol_l['m'],
            'm_ratio': float(new_sol_l['m'] / old_sol['m']),
            'FDR_null_old': old_sol['FDR_null'],
            'FDR_null_new': new_sol_l['FDR_null'],
            'FDR_null_new_with_old_m': float(fdr_deg) if np.isfinite(fdr_deg) else None,
        })

    results_df = pd.DataFrame(results_rows)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Saving results...")
    print("=" * 60)

    # Save main results CSV
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved {OUTPUT_CSV}")

    # Save dependence sensitivity CSV
    if len(dep_df) > 0:
        dep_df.to_csv(OUTPUT_DEP_SENS, index=False)
        print(f"  Saved {OUTPUT_DEP_SENS}")

    # Save operating points CSV
    op_df = pd.DataFrame(op_rows)
    op_df.to_csv(RESULTS_DIR / "operating_points.csv", index=False)
    print(f"  Saved {RESULTS_DIR / 'operating_points.csv'}")

    # Save flat JSON (no nullfdr_variant nesting)
    summary = {
        "mixture_source": mixture_source,
        "mixture_params": {
            "pi": params['pi'],
            "mu": params['mu'],
            "sigma": params['sigma'],
            "distribution": params.get("distribution", "truncnorm"),
            "truncation_lo": params.get("truncation_lo", 0.0),
        },
        "dependence": {
            "preferred_ordering": preferred_ordering,
            "phi": phi,
            "Delta": Delta,
        },
        "evidence_window": {
            "z_lo": float(B[0]),
            "z_hi": _z_hi_to_json(B[1]),
        },
        "pass_probabilities": {k: float(v) for k, v in p_pass.items()},
        "cost_parameters": {
            "lambda_baseline": float(LAMBDA_BASELINE),
            "lambda_grid": [float(x) for x in lambda_grid],
            "lambda_p25": float(timing_meta.get("lambda_p25", LAMBDA_BASELINE)),
            "lambda_p50": float(timing_meta.get("lambda_p50", LAMBDA_BASELINE)),
            "lambda_p75": float(timing_meta.get("lambda_p75", LAMBDA_BASELINE)),
            "lambda_p90": float(timing_meta.get("lambda_p90", LAMBDA_BASELINE)),
        },
        "timing_source": timing_meta,
        "calibration": {
            "m_old_baseline": int(M_OLD_BASELINE),
            "fdr_target": float(FDR_TARGET),
            "calibrated_n_eff_old": int(N_eff_old),
            "fdr_at_calibration": float(fdr_at_cal),
            "n_old_implied": float(n_old_implied),
        },
        "main_result": {
            "m_old": int(M_OLD_BASELINE),
            "m_new": int(m_new),
            "m_ratio": float(m_new / M_OLD_BASELINE),
            "n_eff_old": int(N_eff_old),
            "n_eff_new": int(N_eff_new),
            "fdr_degradation": float(fdr_degrad) if np.isfinite(fdr_degrad) else None,
        },
        "scaling": scaling_df.to_dict('records'),
        "sensitivity": {
            "lambda": lambda_sens,
            "phi": phi_sens,
            "z_lo": z_lo_sens,
            "mixture": mix_sens,
        },
        "operating_points": op_rows,
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved {OUTPUT_JSON}")

    print(f"\n{'='*60}")
    print("HEADLINE RESULT")
    print("=" * 60)
    print(f"  m_old = {M_OLD_BASELINE}")
    print(f"  m_new = {m_new}")
    print(f"  ratio = {m_new/M_OLD_BASELINE:.1f}x")
    print(f"  lambda = {LAMBDA_BASELINE:.6f} (1/{1/LAMBDA_BASELINE:.0f})")
    print(f"  B = [{float(B[0]):.2f}, {_fmt_z_hi(B[1])}]")
    print(f"  N_eff_old = {N_eff_old}, N_eff_new = {N_eff_new}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
