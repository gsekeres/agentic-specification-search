#!/usr/bin/env python3
"""
04_fit_mixture.py
=================

Fit a three-type mixture model for the evidence index used in the screening counterfactuals.

Main index (paper):
  |Z| := |t|,
the absolute t-statistic for the focal coefficient under harmonized inference.
This index is nonnegative, with:
  |Z| = 0     (no evidence / null),
  |Z| ≈ 1.96  (two-sided p=0.05 under normal approximation),
  |Z| = 10    (very high significance).

Model family (main):
  |Z| | k ~ TruncNormal(mu_k, sigma_k; lo=0),   k in {N, H, L}.
We estimate a 3-component truncated-Gaussian mixture by maximum likelihood with multiple restarts.

Outputs:
  - estimation/results/mixture_params_abs_t.json      (main; |t|, truncated-normal mixture)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import foldnorm, norm

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "estimation" / "data"
RESULTS_DIR = BASE_DIR / "estimation" / "results"

CLAIM_LEVEL_FILE = DATA_DIR / "claim_level.csv"
SPEC_LEVEL_FILE = DATA_DIR / "spec_level.csv"
SPEC_LEVEL_VERIFIED_CORE_FILE = DATA_DIR / "spec_level_verified_core.csv"
I4R_COMPARISON_FILE = DATA_DIR / "i4r_comparison.csv"

OUTPUT_ABS_T = RESULTS_DIR / "mixture_params_abs_t.json"

# Winsorization thresholds (estimation only)
WINSORIZE_T_THRESHOLD = 20.0
WINSORIZE_T_SENSITIVITY = [15.0, 10.0]
TRIM_ABS_T_CUTOFFS = [10.0]


def _winsorize_pos(x: np.ndarray, threshold: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = np.clip(x, 0.0, None)
    return np.minimum(x, float(threshold))


def _softplus(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def _inv_softplus(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = np.clip(y, 1e-12, None)
    return np.log(np.expm1(y))


def _softmax(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = w - np.max(w)
    e = np.exp(w)
    s = float(np.sum(e))
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(w) / len(w)
    return e / s


def _component_labels(n_components: int) -> list[str]:
    """Return canonical component labels sorted by ascending mean."""
    if n_components == 2:
        return ["Low", "High"]
    elif n_components == 3:
        return ["N", "H", "L"]
    elif n_components == 4:
        return ["N", "H1", "H2", "L"]
    else:
        return [f"C{i}" for i in range(n_components)]


def _foldnorm_logpdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = float(max(sigma, 1e-8))
    mu = float(max(mu, 0.0))
    c = mu / sigma
    return foldnorm.logpdf(x, c=c, scale=sigma)


def _truncnorm_logpdf(x: np.ndarray, mu: float, sigma: float, lo: float = 0.0) -> np.ndarray:
    """
    Log-pdf of a N(mu,sigma^2) distribution truncated to [lo, +inf).

    Parameterization uses the *untruncated* normal's (mu, sigma).
    """
    sigma = float(max(sigma, 1e-8))
    mu = float(mu)
    lo = float(lo)
    x = np.asarray(x, dtype=float)
    z = (x - mu) / sigma
    logpdf = norm.logpdf(z) - np.log(sigma)
    logZ = norm.logsf((lo - mu) / sigma)  # log(1 - Phi((lo-mu)/sigma))
    return logpdf - logZ


def fit_truncnorm_mixture(
    data: np.ndarray,
    n_components: int = 3,
    n_init: int = 40,
    random_state: int = 42,
    max_iter: int = 800,
    lo: float = 0.0,
    sigma_constraint: str | None = None,
) -> dict:
    """
    Fit a truncated-normal mixture by direct maximum likelihood with multiple restarts.
    Used for the main build when the evidence index is |Z| (absolute t-statistic).

    Component k:
        X_k ~ N(mu_k, sigma_k^2) truncated to [lo, +inf).

    sigma_constraint:
        None       — default, sigma free (sigma >= 1e-6 via softplus)
        "fixed_1"  — fix sigma_k = 1.0 for all k; only optimize pi and mu
        "geq_1"    — sigma >= 1.0 via softplus(raw) + 1.0
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    x = np.clip(x, float(lo), None)
    if x.size < max(10, n_components * 3):
        raise ValueError("Not enough observations to fit mixture")

    qs = np.linspace(0.15, 0.85, n_components)
    mu0 = np.quantile(x, qs)
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 1.0
    sig0 = np.clip(sd * np.ones(n_components), 0.25, None)

    if sigma_constraint == "fixed_1":
        # Only optimize weights and means; sigma fixed at 1.0
        def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            K = n_components
            w = theta[:K]
            mu_raw = theta[K : 2 * K]
            pi = _softmax(w)
            mu = _softplus(mu_raw)
            sigma = np.ones(K)
            return pi, mu, sigma

        theta0 = np.concatenate([np.zeros(n_components), _inv_softplus(mu0)])
    elif sigma_constraint == "geq_1":
        # sigma = softplus(raw) + 1.0, so sigma >= 1.0
        def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            K = n_components
            w = theta[:K]
            mu_raw = theta[K : 2 * K]
            sig_raw = theta[2 * K : 3 * K]
            pi = _softmax(w)
            mu = _softplus(mu_raw)
            sigma = _softplus(sig_raw) + 1.0
            return pi, mu, sigma

        theta0 = np.concatenate([np.zeros(n_components), _inv_softplus(mu0), _inv_softplus(sig0)])
    else:
        # Default: sigma free
        def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            K = n_components
            w = theta[:K]
            mu_raw = theta[K : 2 * K]
            sig_raw = theta[2 * K : 3 * K]
            pi = _softmax(w)
            mu = _softplus(mu_raw)
            sigma = _softplus(sig_raw) + 1e-6
            return pi, mu, sigma

        theta0 = np.concatenate([np.zeros(n_components), _inv_softplus(mu0), _inv_softplus(sig0)])

    def nll(theta: np.ndarray) -> float:
        pi, mu, sigma = unpack(theta)
        log_pi = np.log(np.clip(pi, 1e-12, 1.0))
        logpdf = []
        for k in range(n_components):
            logpdf.append(_truncnorm_logpdf(x, float(mu[k]), float(sigma[k]), lo=float(lo)))
        logpdf = np.stack(logpdf, axis=1)
        ll = logsumexp(logpdf + log_pi[None, :], axis=1).sum()
        if not np.isfinite(ll):
            return 1e18
        return float(-ll)

    rng = np.random.default_rng(int(random_state))

    best_res = None
    best_val = None
    for _ in range(int(n_init)):
        jitter = rng.normal(0, 0.6, size=theta0.size)
        theta_init = theta0 + jitter
        res = minimize(nll, theta_init, method="L-BFGS-B", options={"maxiter": int(max_iter)})
        if not res.success:
            continue
        val = float(res.fun)
        if best_val is None or val < best_val:
            best_val = val
            best_res = res

    if best_res is None:
        best_res = minimize(nll, theta0, method="L-BFGS-B", options={"maxiter": int(max_iter)})

    pi_hat, mu_hat, sig_hat = unpack(best_res.x)
    order = np.argsort(mu_hat)
    pi_hat = pi_hat[order]
    mu_hat = mu_hat[order]
    sig_hat = sig_hat[order]

    labels = _component_labels(n_components)
    log_like = float(-nll(best_res.x))
    n = int(x.size)
    # Parameter count depends on constraint
    if sigma_constraint == "fixed_1":
        p = (n_components - 1) + n_components  # weights + means only
    else:
        p = (n_components - 1) + 2 * n_components  # weights + means + sigmas
    aic = 2 * p - 2 * log_like
    bic = p * np.log(n) - 2 * log_like

    return {
        "distribution": "truncnorm",
        "truncation_lo": float(lo),
        "sigma_constraint": sigma_constraint,
        "pi": {labels[i]: float(pi_hat[i]) for i in range(n_components)},
        "mu": {labels[i]: float(mu_hat[i]) for i in range(n_components)},
        "sigma": {labels[i]: float(sig_hat[i]) for i in range(n_components)},
        "log_likelihood": float(log_like),
        "aic": float(aic),
        "bic": float(bic),
        "n_obs": int(n),
        "n_params": int(p),
        "converged": bool(getattr(best_res, "success", False)),
        "n_iter": int(getattr(best_res, "nit", -1)),
        "optimizer_message": str(getattr(best_res, "message", "")),
    }


def fit_foldnorm_mixture(
    data: np.ndarray,
    n_components: int = 3,
    n_init: int = 40,
    random_state: int = 42,
    max_iter: int = 800,
    sigma_constraint: str | None = None,
    fix_null_mean_zero: bool = False,
) -> dict:
    """
    Fit folded-normal mixture by direct maximum likelihood with multiple restarts.
    Used for appendix robustness on |t|.

    sigma_constraint:
        None       — default, sigma free
        "fixed_1"  — fix sigma_k = 1.0 for all k; only optimize pi and mu

    fix_null_mean_zero:
        If True, fix the first (lowest) component mean to 0 and optimize K-1 free means.
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    x = np.abs(x)
    if x.size < max(10, n_components * 3):
        raise ValueError("Not enough observations to fit mixture")

    fix_sigma = sigma_constraint == "fixed_1"
    n_free_mu = n_components - 1 if fix_null_mean_zero else n_components

    qs = np.linspace(0.15, 0.85, n_components)
    mu0 = np.quantile(x, qs)
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 1.0
    sig0 = np.clip(sd * np.ones(n_components), 0.25, None)

    mu0_free = mu0[1:] if fix_null_mean_zero else mu0

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        K = n_components
        w = theta[:K]
        pi = _softmax(w)
        if fix_null_mean_zero:
            mu_raw = theta[K : K + n_free_mu]
            mu = np.concatenate([[0.0], _softplus(mu_raw)])
        else:
            mu_raw = theta[K : K + n_free_mu]
            mu = _softplus(mu_raw)
        if fix_sigma:
            sigma = np.ones(K)
        else:
            sig_raw = theta[K + n_free_mu :]
            sigma = _softplus(sig_raw) + 1e-6
        return pi, mu, sigma

    def nll(theta: np.ndarray) -> float:
        pi, mu, sigma = unpack(theta)
        log_pi = np.log(np.clip(pi, 1e-12, 1.0))
        logpdf = []
        for k in range(n_components):
            logpdf.append(_foldnorm_logpdf(x, float(mu[k]), float(sigma[k])))
        logpdf = np.stack(logpdf, axis=1)
        ll = logsumexp(logpdf + log_pi[None, :], axis=1).sum()
        if not np.isfinite(ll):
            return 1e18
        return float(-ll)

    rng = np.random.default_rng(int(random_state))
    if fix_sigma:
        theta0 = np.concatenate([np.zeros(n_components), _inv_softplus(mu0_free)])
    else:
        theta0 = np.concatenate([np.zeros(n_components), _inv_softplus(mu0_free), _inv_softplus(sig0)])

    best_res = None
    best_val = None
    for _ in range(int(n_init)):
        jitter = rng.normal(0, 0.6, size=theta0.size)
        theta_init = theta0 + jitter
        res = minimize(nll, theta_init, method="L-BFGS-B", options={"maxiter": int(max_iter)})
        if not res.success:
            continue
        val = float(res.fun)
        if best_val is None or val < best_val:
            best_val = val
            best_res = res

    if best_res is None:
        best_res = minimize(nll, theta0, method="L-BFGS-B", options={"maxiter": int(max_iter)})

    pi_hat, mu_hat, sig_hat = unpack(best_res.x)
    order = np.argsort(mu_hat)
    pi_hat = pi_hat[order]
    mu_hat = mu_hat[order]
    sig_hat = sig_hat[order]

    labels = _component_labels(n_components)
    log_like = float(-nll(best_res.x))
    n = int(x.size)
    if fix_sigma:
        p = (n_components - 1) + n_free_mu  # weights + free means only
    else:
        p = (n_components - 1) + n_free_mu + n_components  # weights + free means + sigmas
    aic = 2 * p - 2 * log_like
    bic = p * np.log(n) - 2 * log_like

    return {
        "distribution": "foldnorm",
        "sigma_constraint": sigma_constraint,
        "fix_null_mean_zero": fix_null_mean_zero,
        "pi": {labels[i]: float(pi_hat[i]) for i in range(n_components)},
        "mu": {labels[i]: float(mu_hat[i]) for i in range(n_components)},
        "sigma": {labels[i]: float(sig_hat[i]) for i in range(n_components)},
        "log_likelihood": float(log_like),
        "aic": float(aic),
        "bic": float(bic),
        "n_obs": int(n),
        "n_params": int(p),
        "converged": bool(getattr(best_res, "success", False)),
        "n_iter": int(getattr(best_res, "nit", -1)),
        "optimizer_message": str(getattr(best_res, "message", "")),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # MAIN (|t|): truncated-Gaussian mixture on nonnegative support
    # =========================================================================
    abs_t_results: dict = {"index": "abs_t", "winsorize_threshold": WINSORIZE_T_THRESHOLD}

    if I4R_COMPARISON_FILE.exists():
        cmp = pd.read_csv(I4R_COMPARISON_FILE)
        t_i4r = pd.to_numeric(cmp.get("t_i4r"), errors="coerce").to_numpy(dtype=float)
        t_i4r = _winsorize_pos(np.abs(t_i4r), WINSORIZE_T_THRESHOLD)
        if np.isfinite(t_i4r).sum() >= 10:
            abs_t_results["i4r_benchmark"] = {"primary": fit_truncnorm_mixture(t_i4r, n_init=30, random_state=42, lo=0.0)}

    if SPEC_LEVEL_FILE.exists():
        spec_df = pd.read_csv(SPEC_LEVEL_FILE)
        t_col = "Z_abs" if "Z_abs" in spec_df.columns else ("Z" if "Z" in spec_df.columns else "t_stat")

        t_all = pd.to_numeric(spec_df[t_col], errors="coerce").to_numpy(dtype=float)
        t_all = _winsorize_pos(np.abs(t_all), WINSORIZE_T_THRESHOLD)

        baseline_specs = spec_df[spec_df["spec_tree_path"].astype(str).str.contains("#baseline", na=False)]
        if len(baseline_specs) == 0 and "spec_id" in spec_df.columns:
            baseline_specs = spec_df[spec_df["spec_id"].astype(str) == "baseline"]
        t_base = pd.to_numeric(baseline_specs[t_col], errors="coerce").to_numpy(dtype=float)
        t_base = _winsorize_pos(np.abs(t_base), WINSORIZE_T_THRESHOLD)

        out_t: dict = {
            "equal_weight": fit_truncnorm_mixture(t_all, n_init=25, random_state=42, lo=0.0),
            "baseline_only": (
                fit_truncnorm_mixture(t_base, n_init=25, random_state=42, lo=0.0) if np.isfinite(t_base).sum() >= 10 else None
            ),
        }

        winsor_sens_t: dict = {}
        for thr in WINSORIZE_T_SENSITIVITY:
            tb = _winsorize_pos(t_base, float(thr))
            if np.isfinite(tb).sum() >= 10:
                winsor_sens_t[f"winsor_{int(thr)}"] = fit_truncnorm_mixture(tb, n_init=20, random_state=42, lo=0.0)
        out_t["winsor_sensitivity"] = winsor_sens_t

        trim_sens: dict = {}
        for cutoff in TRIM_ABS_T_CUTOFFS:
            t_trim = t_base[np.isfinite(t_base) & (t_base <= float(cutoff))]
            if len(t_trim) >= 10:
                trim_sens[f"trim_abs_le_{int(cutoff)}"] = fit_truncnorm_mixture(t_trim, n_init=20, random_state=42, lo=0.0)
                # Constrained-sigma variants
                print(f"  Fitting sigma_fixed_1 on trimmed |t|<={int(cutoff)} (n={len(t_trim)})...")
                trim_sens[f"trim_abs_le_{int(cutoff)}_sigma_fixed_1"] = fit_truncnorm_mixture(
                    t_trim, n_init=40, random_state=42, lo=0.0, sigma_constraint="fixed_1"
                )
                print(f"  Fitting sigma_geq_1 on trimmed |t|<={int(cutoff)} (n={len(t_trim)})...")
                trim_sens[f"trim_abs_le_{int(cutoff)}_sigma_geq_1"] = fit_truncnorm_mixture(
                    t_trim, n_init=40, random_state=42, lo=0.0, sigma_constraint="geq_1"
                )
        out_t["trim_sensitivity"] = trim_sens

        abs_t_results["spec_level"] = out_t

    # =========================================================================
    # Verified-core spec-level |t| fits (truncated-normal)
    # =========================================================================
    if SPEC_LEVEL_VERIFIED_CORE_FILE.exists():
        vdf = pd.read_csv(SPEC_LEVEL_VERIFIED_CORE_FILE)
        t_col_vc = "Z_abs" if "Z_abs" in vdf.columns else ("Z" if "Z" in vdf.columns else "t_stat")

        t_core = pd.to_numeric(vdf[t_col_vc], errors="coerce").to_numpy(dtype=float)
        t_core = _winsorize_pos(np.abs(t_core), WINSORIZE_T_THRESHOLD)

        out_vc: dict = {"z_column": t_col_vc}
        out_vc["all_core"] = fit_truncnorm_mixture(t_core, n_init=25, random_state=42, lo=0.0)

        params_baseline_vc = None
        if "v_is_baseline" in vdf.columns:
            base = vdf[pd.to_numeric(vdf["v_is_baseline"], errors="coerce").fillna(0).astype(int) == 1]
            if len(base) > 0:
                tb = pd.to_numeric(base[t_col_vc], errors="coerce").to_numpy(dtype=float)
                tb = _winsorize_pos(np.abs(tb), WINSORIZE_T_THRESHOLD)
                if np.isfinite(tb).sum() >= 10:
                    params_baseline_vc = fit_truncnorm_mixture(tb, n_init=25, random_state=42, lo=0.0)
        out_vc["baseline_only"] = params_baseline_vc

        abs_t_results["spec_level_verified_core"] = out_vc

    # =========================================================================
    # K-SENSITIVITY (K=2,3,4): truncated-normal on verified-core all specs
    # =========================================================================
    print("\n--- K-sensitivity analysis ---")
    k_sensitivity: dict = {}

    # Data source: verified-core all specs (largest clean sample)
    vc_data_t = None
    if SPEC_LEVEL_VERIFIED_CORE_FILE.exists():
        vdf = pd.read_csv(SPEC_LEVEL_VERIFIED_CORE_FILE)
        t_col_vc = "Z_abs" if "Z_abs" in vdf.columns else ("Z" if "Z" in vdf.columns else "t_stat")
        vc_data_t = pd.to_numeric(vdf[t_col_vc], errors="coerce").to_numpy(dtype=float)
        vc_data_t = _winsorize_pos(np.abs(vc_data_t), WINSORIZE_T_THRESHOLD)

    for K in [2, 3, 4]:
        k_key = f"K={K}"
        k_sensitivity[k_key] = {}

        if vc_data_t is not None and np.isfinite(vc_data_t).sum() >= max(10, K * 3):
            print(f"  Fitting truncnorm K={K} on verified-core |t| (n={np.isfinite(vc_data_t).sum()})...")
            k_sensitivity[k_key]["truncnorm"] = fit_truncnorm_mixture(
                vc_data_t, n_components=K, n_init=30, random_state=42, lo=0.0
            )

    # Print AIC/BIC comparison
    print("\n  K-sensitivity AIC/BIC comparison (truncnorm on |t|):")
    for k_key in sorted(k_sensitivity.keys()):
        tn = k_sensitivity[k_key].get("truncnorm", {})
        if tn:
            print(f"    {k_key}: AIC={tn['aic']:.1f}, BIC={tn['bic']:.1f}, logL={tn['log_likelihood']:.1f}")

    abs_t_results["k_sensitivity"] = k_sensitivity

    # =========================================================================
    # FOLDED-NORMAL ROBUSTNESS (σ=1, μ_N=0): K=2,3,4 on verified-core |t|≤10
    # =========================================================================
    print("\n--- Folded-normal robustness (sigma=1, mu_N=0) ---")
    folded_robustness: dict = {}

    if SPEC_LEVEL_VERIFIED_CORE_FILE.exists():
        vdf_fn = pd.read_csv(SPEC_LEVEL_VERIFIED_CORE_FILE)
        t_col_fn = "Z_abs" if "Z_abs" in vdf_fn.columns else ("Z" if "Z" in vdf_fn.columns else "t_stat")
        t_fn = pd.to_numeric(vdf_fn[t_col_fn], errors="coerce").to_numpy(dtype=float)
        t_fn = np.abs(t_fn)
        t_fn = t_fn[np.isfinite(t_fn) & (t_fn <= 10.0)]

        for K in [2, 3, 4]:
            if len(t_fn) >= max(10, K * 3):
                print(f"  Fitting foldnorm K={K} (sigma=1, mu_N=0) on |t|<=10 (n={len(t_fn)})...")
                folded_robustness[f"K={K}"] = fit_foldnorm_mixture(
                    t_fn, n_components=K, n_init=50, random_state=42,
                    sigma_constraint="fixed_1", fix_null_mean_zero=True,
                )
                res = folded_robustness[f"K={K}"]
                print(f"    AIC={res['aic']:.1f}, BIC={res['bic']:.1f}, logL={res['log_likelihood']:.1f}")

    abs_t_results["folded_normal_robustness"] = folded_robustness

    # =========================================================================
    # MU_FREE SIGMA=1 COMPARISON: both families, K=2,3,4, full + |t|≤10
    # =========================================================================
    print("\n--- mu_free sigma=1 comparison (truncnorm + foldnorm, K=2,3,4) ---")
    mu_free_comparison: dict = {}

    if SPEC_LEVEL_VERIFIED_CORE_FILE.exists():
        vdf_mf = pd.read_csv(SPEC_LEVEL_VERIFIED_CORE_FILE)
        t_col_mf = "Z_abs" if "Z_abs" in vdf_mf.columns else ("Z" if "Z" in vdf_mf.columns else "t_stat")
        t_mf_raw = pd.to_numeric(vdf_mf[t_col_mf], errors="coerce").to_numpy(dtype=float)
        t_mf_full = _winsorize_pos(np.abs(t_mf_raw), WINSORIZE_T_THRESHOLD)
        t_mf_trim = np.abs(t_mf_raw)
        t_mf_trim = t_mf_trim[np.isfinite(t_mf_trim) & (t_mf_trim <= 10.0)]

        samples = [("full", t_mf_full), ("trim10", t_mf_trim)]

        for K in [2, 3, 4]:
            for sample_name, sample_data in samples:
                if len(sample_data) < max(10, K * 3):
                    continue

                # Truncated normal: sigma=1 fixed, mu free (default behavior)
                key_tn = f"truncnorm_K={K}_{sample_name}"
                print(f"  Fitting {key_tn} (n={len(sample_data)})...")
                mu_free_comparison[key_tn] = fit_truncnorm_mixture(
                    sample_data, n_components=K, n_init=50, random_state=42,
                    lo=0.0, sigma_constraint="fixed_1",
                )
                res_tn = mu_free_comparison[key_tn]
                print(f"    AIC={res_tn['aic']:.1f}, BIC={res_tn['bic']:.1f}, logL={res_tn['log_likelihood']:.1f}")

                # Folded normal: sigma=1 fixed, mu_N FREE (fix_null_mean_zero=False)
                key_fn = f"foldnorm_K={K}_{sample_name}"
                print(f"  Fitting {key_fn} (n={len(sample_data)})...")
                mu_free_comparison[key_fn] = fit_foldnorm_mixture(
                    sample_data, n_components=K, n_init=50, random_state=42,
                    sigma_constraint="fixed_1", fix_null_mean_zero=False,
                )
                res_fn = mu_free_comparison[key_fn]
                print(f"    AIC={res_fn['aic']:.1f}, BIC={res_fn['bic']:.1f}, logL={res_fn['log_likelihood']:.1f}")

        # Print comparison table
        print("\n  mu_free sigma=1 comparison AIC/BIC:")
        print(f"  {'Model':<30s} {'n':>5s} {'AIC':>10s} {'BIC':>10s} {'logL':>10s}")
        for key in sorted(mu_free_comparison.keys()):
            r = mu_free_comparison[key]
            print(f"  {key:<30s} {r['n_obs']:>5d} {r['aic']:>10.1f} {r['bic']:>10.1f} {r['log_likelihood']:>10.1f}")

    abs_t_results["mu_free_sigma1_comparison"] = mu_free_comparison

    # =========================================================================
    # SYSTEMATIC GRID: K x sigma_constraint x sample (truncated-normal)
    # 3 sigma specs (free, fixed_1, geq_1) x 3 K (2,3,4) x 2 samples (full, trim10)
    # All on verified-core data for consistency.
    # =========================================================================
    print("\n--- Systematic grid: K x sigma x sample ---")
    systematic_grid: dict = {}

    if SPEC_LEVEL_VERIFIED_CORE_FILE.exists():
        vdf_sg = pd.read_csv(SPEC_LEVEL_VERIFIED_CORE_FILE)
        t_col_sg = "Z_abs" if "Z_abs" in vdf_sg.columns else ("Z" if "Z" in vdf_sg.columns else "t_stat")
        t_sg_raw = pd.to_numeric(vdf_sg[t_col_sg], errors="coerce").to_numpy(dtype=float)
        t_sg_full = _winsorize_pos(np.abs(t_sg_raw), WINSORIZE_T_THRESHOLD)
        t_sg_trim = np.abs(t_sg_raw)
        t_sg_trim = t_sg_trim[np.isfinite(t_sg_trim) & (t_sg_trim <= 10.0)]

        samples_sg = [("full", t_sg_full), ("trim10", t_sg_trim)]
        sigma_specs = [("free", None), ("fixed_1", "fixed_1"), ("geq_1", "geq_1")]

        for K in [2, 3, 4]:
            for sigma_name, sigma_arg in sigma_specs:
                for sample_name, sample_data in samples_sg:
                    if len(sample_data) < max(10, K * 3):
                        continue
                    key = f"K={K}_sigma={sigma_name}_{sample_name}"
                    print(f"  Fitting {key} (n={len(sample_data)})...")
                    systematic_grid[key] = fit_truncnorm_mixture(
                        sample_data, n_components=K, n_init=40, random_state=42,
                        lo=0.0, sigma_constraint=sigma_arg,
                    )
                    res = systematic_grid[key]
                    print(f"    AIC={res['aic']:.1f}, BIC={res['bic']:.1f}, logL={res['log_likelihood']:.1f}")

        # Print summary table
        print("\n  Systematic grid AIC/BIC summary:")
        print(f"  {'Key':<35s} {'n':>5s} {'AIC':>10s} {'BIC':>10s} {'logL':>10s}")
        for key in sorted(systematic_grid.keys()):
            r = systematic_grid[key]
            print(f"  {key:<35s} {r['n_obs']:>5d} {r['aic']:>10.1f} {r['bic']:>10.1f} {r['log_likelihood']:>10.1f}")

    abs_t_results["systematic_grid"] = systematic_grid

    OUTPUT_ABS_T.write_text(json.dumps(abs_t_results, indent=2) + "\n")
    print(f"Wrote {OUTPUT_ABS_T}")


if __name__ == "__main__":
    main()
