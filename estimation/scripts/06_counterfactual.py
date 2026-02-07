#!/usr/bin/env python3
"""
06_counterfactual.py
====================

Compute counterfactual screening under cost shift.

Given:
- Mixture parameters (pi_k, mu_k, sigma_k) for k in {N, H, L}
- Dependence parameter phi (hence Delta = 1 - phi)
- Cost ratio lambda = gamma^new / gamma^old

Compute:
- Mapping (B, m) -> qualification rates and FDR
- Required disclosure (m) to maintain fixed screening targets under gamma shift

Output: estimation/results/counterfactual.csv, estimation/results/counterfactual_params.json
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
OUTPUT_CSV = RESULTS_DIR / "counterfactual.csv"
OUTPUT_JSON = RESULTS_DIR / "counterfactual_params.json"
OUTPUT_DEP_SENS = RESULTS_DIR / "counterfactual_dependence_sensitivity.csv"
OUTPUT_ALTERNATIVES = RESULTS_DIR / "counterfactual_alternatives.json"

# Cost parameters
LAMBDA_BASELINE = 1 / 14  # gamma^new / gamma^old (14 days -> 1 day)
LAMBDA_GRID = [1/20, 1/14, 1/7]  # Sensitivity range

# Baseline testing horizon in the old regime (normalized gamma_old = 1).
# This is a calibration constant: with gamma_new = lambda*gamma_old, horizons scale like 1/gamma.
# Note: must be large enough so the model can attain common (rho,FDR) targets even when
# effective independence Delta is small (high within-paper dependence).
N_OLD_BASELINE = 33

# FDR targets
RHO_TARGETS = [0.05, 0.10, 0.20]
FDR_TARGETS = [0.05, 0.10, 0.20]

# Default operating point used for window selection and for the operating-point curve plots.
RHO_PLOT = 0.10
FDR_PLOT_TARGET = 0.05

# Evidence window search on the |Z| scale.
# Main build uses |Z|=|t|, so p=0.05 corresponds to |t|≈1.96 under the normal approximation.
# Upper bound capped at 10.0 consistent with the Z ≤ 10 data trimming used in estimation.
Z_LO_MIN = 1.959963984540054
Z_LO_GRID = np.linspace(Z_LO_MIN, 10.0, 45)
Z_HI_GRID = np.linspace(Z_LO_MIN + 0.5, 10.0, 60)


def _gamma_shape_scale(mu: float, sigma: float) -> tuple[float, float]:
    mu = float(max(mu, 1e-8))
    sigma = float(max(sigma, 1e-8))
    shape = (mu / sigma) ** 2
    scale = (sigma**2) / mu
    return float(max(shape, 1e-8)), float(max(scale, 1e-8))


def load_parameters():
    """Load mixture and dependence parameters."""
    # Load mixture params
    mixture_source = None
    if MIXTURE_FILE.exists():
        with open(MIXTURE_FILE, 'r') as f:
            mixture = json.load(f)
        # Preference order:
        # 1. Spec-level baseline trimmed to Z ≤ 10 (correct intuitive K=3 mix)
        # 2. K=3 truncnorm from verified-core (fallback)
        # 3. Spec-level baseline-only
        # 4. Verified-core baseline-only
        # 5. Claim-level primary
        trim10 = mixture.get("spec_level", {}).get("trim_sensitivity", {}).get("trim_abs_le_10")
        k3_vc = mixture.get("k_sensitivity", {}).get("K=3", {}).get("truncnorm")
        sl_base = mixture.get("spec_level", {}).get("baseline_only")
        vc_base = mixture.get("spec_level_verified_core", {}).get("baseline_only")
        cl_prim = mixture.get("claim_level", {}).get("primary", {})

        if trim10 is not None:
            params = trim10
            mixture_source = "spec_level:trim_sensitivity:trim_abs_le_10"
        elif k3_vc is not None:
            params = k3_vc
            mixture_source = "k_sensitivity:K=3:truncnorm (verified_core all specs)"
        elif sl_base is not None:
            params = sl_base
            mixture_source = "spec_level:baseline_only"
        elif vc_base is not None:
            params = vc_base
            mixture_source = "spec_level_verified_core:baseline_only"
        else:
            params = cl_prim
            mixture_source = "claim_level:primary"
    else:
        # Default parameters based on typical economics literature
        params = {
            'pi': {'N': 0.30, 'H': 0.40, 'L': 0.30},
            'mu': {'N': 0.5, 'H': 2.5, 'L': 4.5},
            'sigma': {'N': 1.2, 'H': 1.0, 'L': 1.5},
        }
        print("Warning: Using default mixture parameters")
        mixture_source = "default"

    # Load dependence params
    dependence = None
    if DEPENDENCE_FILE.exists():
        with open(DEPENDENCE_FILE, 'r') as f:
            dependence = json.load(f)
        phi = dependence.get('preferred', {}).get('phi', 0.6)
        Delta = dependence.get('preferred', {}).get('Delta', 0.4)
    else:
        phi = 0.6
        Delta = 0.4
        print("Warning: Using default dependence parameters")

    return params, phi, Delta, dependence, mixture_source


def compute_pass_probabilities(B, params):
    """
    Compute pass probabilities for each type given window B.

    B = (z_lo, z_hi) on the index scale.
    """
    z_lo, z_hi = B
    probs = {}
    dist = str(params.get("distribution", "gaussian"))

    for k in ['N', 'H', 'L']:
        mu_k = params['mu'][k]
        sigma_k = params['sigma'][k]
        if dist == "gamma":
            a, sc = _gamma_shape_scale(float(mu_k), float(sigma_k))
            cdf_hi = 1.0 if (z_hi is None or not np.isfinite(z_hi)) else float(stats.gamma.cdf(float(z_hi), a=a, scale=sc))
            cdf_lo = 0.0 if (z_lo is None or float(z_lo) <= 0) else float(stats.gamma.cdf(float(z_lo), a=a, scale=sc))
            p_k = cdf_hi - cdf_lo
        elif dist == "foldnorm":
            mu_k = float(mu_k)
            sigma_k = float(max(sigma_k, 1e-8))
            c = mu_k / sigma_k
            cdf_hi = 1.0 if (z_hi is None or not np.isfinite(z_hi)) else float(stats.foldnorm.cdf(float(z_hi), c=c, scale=sigma_k))
            cdf_lo = 0.0 if (z_lo is None or float(z_lo) <= 0) else float(stats.foldnorm.cdf(float(z_lo), c=c, scale=sigma_k))
            p_k = cdf_hi - cdf_lo
        elif dist == "truncnorm":
            mu_k = float(mu_k)
            sigma_k = float(max(sigma_k, 1e-8))
            lo = float(params.get("truncation_lo", 0.0))

            def tcdf(x: float | None) -> float:
                if x is None:
                    return 1.0
                x = float(x)
                if not np.isfinite(x):
                    return 1.0
                if x <= lo:
                    return 0.0
                alpha = (lo - mu_k) / sigma_k
                denom = float(max(stats.norm.sf(alpha), 1e-12))
                return float((stats.norm.cdf((x - mu_k) / sigma_k) - stats.norm.cdf(alpha)) / denom)

            cdf_hi = tcdf(z_hi)
            cdf_lo = 0.0 if (z_lo is None or float(z_lo) <= lo) else tcdf(z_lo)
            p_k = float(max(0.0, min(1.0, cdf_hi - cdf_lo)))
        else:
            p_k = stats.norm.cdf((z_hi - mu_k) / sigma_k) - stats.norm.cdf((z_lo - mu_k) / sigma_k)
        probs[k] = p_k

    return probs


def binomial_tail(n, m, p):
    """
    P(Bin(n, p) >= m) using exact computation.
    """
    if p <= 0:
        return 0.0 if m > 0 else 1.0
    if p >= 1:
        return 1.0
    if n < m:
        return 0.0

    # Use survival function
    return float(stats.binom.sf(m - 1, int(n), p))


def choose_witness_window(
    params,
    z_lo_grid=Z_LO_GRID,
    z_hi_grid=Z_HI_GRID,
    p_H_min=0.05,
    *,
    n_eff: int | None = None,
    rho_target: float | None = None,
    fdr_target: float | None = None,
):
    """
    Choose a two-sided "significant but not too significant" evidence window:
        B = [z_lo, z_hi]

    Objective: maximize p_H (power for H-type papers) subject to FDR
    feasibility at the baseline operating point.  This selects the widest
    window that captures the most H-type mass while maintaining screening
    quality — the upper bound naturally excludes L-type (inflated) results.
    """
    pi = params['pi']
    pi_bad = pi['N'] + pi['L']
    best = None
    best_p_H = -1.0

    for z_lo in z_lo_grid:
        z_lo = float(z_lo)
        for z_hi in z_hi_grid:
            z_hi = float(z_hi)
            if z_hi <= z_lo:
                continue
            B = (z_lo, z_hi)
            p = compute_pass_probabilities(B, params)
            p_H = p['H']
            if p_H < p_H_min:
                continue
            if n_eff is not None and fdr_target is not None:
                # Require FDR ≤ target at m=1 in old regime.  This ensures the
                # window provides good screening at the baseline threshold, so
                # the counterfactual story is: m=1 works pre-agentic, but FDR
                # degrades under agentic testing, and higher m restores it.
                met = screening_metrics(int(n_eff), 1, p, params)
                if not np.isfinite(met['FDR_total']):
                    continue
                if met['FDR_total'] > float(fdr_target):
                    continue
                if rho_target is not None and met['Q_bar'] < float(rho_target):
                    continue
            # Maximize H-type power (widest feasible window)
            if p_H > best_p_H:
                best_p_H = p_H
                p_bad = (pi['N'] * p['N'] + pi['L'] * p['L']) / max(pi_bad, 1e-12)
                score = np.log((p_H + 1e-12) / (p_bad + 1e-12))
                best = {
                    'B': B,
                    'score': float(score),
                    'pass_probabilities': p,
                }

    if best is None:
        # Fallback: one-sided significance threshold
        B = (float(Z_LO_MIN), float(np.inf))
        return {
            'B': B,
            'score': None,
            'pass_probabilities': compute_pass_probabilities(B, params),
        }
    return best


def qualification_rates(n_eff, m, pass_probs):
    """Qualification probability Q_k = P(Bin(n_eff, p_k) >= m) for each type k."""
    return {k: binomial_tail(n_eff, m, pass_probs[k]) for k in ['N', 'H', 'L']}


def screening_metrics(n_eff, m, pass_probs, params):
    """
    Compute qualification rates and implied FDRs for a given (n_eff, m, B).
    """
    Q = qualification_rates(n_eff, m, pass_probs)
    pi = params['pi']
    Q_bar = pi['N'] * Q['N'] + pi['H'] * Q['H'] + pi['L'] * Q['L']
    num_bad = pi['N'] * Q['N'] + pi['L'] * Q['L']
    num_null = pi['N'] * Q['N']
    fdr_total = num_bad / Q_bar if Q_bar > 0 else np.nan
    fdr_null = num_null / Q_bar if Q_bar > 0 else np.nan
    return {
        'Q_N': Q['N'],
        'Q_H': Q['H'],
        'Q_L': Q['L'],
        'Q_bar': Q_bar,
        'FDR_total': fdr_total,
        'FDR_null': fdr_null,
    }


def find_min_m_for_targets(n_eff, pass_probs, params, rho_target, fdr_target, fdr_mode="total"):
    """
    For a fixed evidence window B (hence pass_probs) and fixed n_eff, find the
    smallest m such that:
      (i) enough papers qualify to fill capacity: Q_bar >= rho_target
      (ii) FDR <= fdr_target

    fdr_mode:
        "total" — FDR = (pi_N*Q_N + pi_L*Q_L) / Q_bar  (N and L/E are false discoveries)
        "null_only" — FDR = pi_N*Q_N / Q_bar  (only N-types are false discoveries)

    Journal acceptance probability can then be set to a = rho_target / Q_bar.
    """
    fdr_key = 'FDR_null' if fdr_mode == "null_only" else 'FDR_total'
    # Upper bound on m: beyond n_eff, qualification is identically zero.
    for m in range(1, int(n_eff) + 1):
        met = screening_metrics(n_eff, m, pass_probs, params)
        if not np.isfinite(met[fdr_key]):
            continue
        if met['Q_bar'] < rho_target:
            continue
        if met[fdr_key] <= fdr_target:
            a = float(min(1.0, rho_target / met['Q_bar']))
            return {
                'm': int(m),
                'a': a,
                **met
            }
    return None


def share_based_threshold(n_eff, pass_probs, params, rho_target, fdr_target, tau_grid=None):
    """
    Share-based screening: require m/n >= tau (fraction threshold).
    Find minimum tau such that FDR <= fdr_target and Q_bar >= rho_target.
    """
    if tau_grid is None:
        tau_grid = np.linspace(0.01, 1.0, 200)

    for tau in tau_grid:
        m = int(max(1, np.ceil(tau * n_eff)))
        if m > n_eff:
            continue
        met = screening_metrics(n_eff, m, pass_probs, params)
        if not np.isfinite(met['FDR_total']):
            continue
        if met['Q_bar'] < rho_target:
            continue
        if met['FDR_total'] <= fdr_target:
            a = float(min(1.0, rho_target / met['Q_bar']))
            return {
                'tau': float(tau),
                'm': int(m),
                'a': a,
                **met,
            }
    return None


def likelihood_ratio_screening(n_eff, pass_probs, params, rho_target, fdr_target):
    """
    Likelihood-ratio screening: for each paper compute posterior P(k=H | n_in_B, n_eff).
    Uses binomial likelihoods. Find threshold on posterior P(H) such that
    FDR <= fdr_target and capacity >= rho_target.

    Returns the required posterior threshold and equivalent m.
    """
    pi = params['pi']
    p_N, p_H, p_L = pass_probs['N'], pass_probs['H'], pass_probs['L']

    # For each possible count m_obs in {0, ..., n_eff}, compute posterior P(H)
    posteriors = []
    for m_obs in range(n_eff + 1):
        # Binomial likelihood for each type
        lik_N = stats.binom.pmf(m_obs, n_eff, p_N) if p_N > 0 else (1.0 if m_obs == 0 else 0.0)
        lik_H = stats.binom.pmf(m_obs, n_eff, p_H) if p_H > 0 else (1.0 if m_obs == 0 else 0.0)
        lik_L = stats.binom.pmf(m_obs, n_eff, p_L) if p_L > 0 else (1.0 if m_obs == 0 else 0.0)

        marginal = pi['N'] * lik_N + pi['H'] * lik_H + pi['L'] * lik_L
        if marginal > 0:
            post_H = (pi['H'] * lik_H) / marginal
        else:
            post_H = 0.0
        posteriors.append({'m_obs': m_obs, 'post_H': post_H, 'marginal': marginal})

    # Sort by posterior P(H) descending and find threshold
    # Accept papers with posterior > threshold, choosing threshold to satisfy (rho, FDR)
    posteriors.sort(key=lambda x: -x['post_H'])

    # Try thresholds from highest to lowest posterior
    thresholds = sorted(set(p['post_H'] for p in posteriors), reverse=True)

    for threshold in thresholds:
        # Accept all m_obs with post_H >= threshold
        accepted = [p for p in posteriors if p['post_H'] >= threshold]
        if not accepted:
            continue

        # Compute aggregate qualification rate and FDR
        # P(accepted | type k) = sum over m_obs where post_H >= threshold of Bin(n_eff, p_k, m_obs)
        Q = {}
        for k, p_k in [('N', p_N), ('H', p_H), ('L', p_L)]:
            Q[k] = sum(
                stats.binom.pmf(p['m_obs'], n_eff, p_k)
                for p in accepted
            )

        Q_bar = pi['N'] * Q['N'] + pi['H'] * Q['H'] + pi['L'] * Q['L']
        if Q_bar <= 0:
            continue

        num_bad = pi['N'] * Q['N'] + pi['L'] * Q['L']
        fdr = num_bad / Q_bar

        if Q_bar >= rho_target and fdr <= fdr_target:
            a = float(min(1.0, rho_target / Q_bar))
            # Find equivalent m: the minimum count that achieves this posterior
            equiv_m = min(p['m_obs'] for p in accepted)
            return {
                'posterior_threshold': float(threshold),
                'equivalent_m': int(equiv_m),
                'a': a,
                'Q_N': float(Q['N']),
                'Q_H': float(Q['H']),
                'Q_L': float(Q['L']),
                'Q_bar': float(Q_bar),
                'FDR_total': float(fdr),
            }
    return None


def main():
    print("=" * 60)
    print("Computing Counterfactual Screening")
    print("=" * 60)

    # Load parameters
    print("\nLoading parameters...")
    params, phi, Delta, dependence, mixture_source = load_parameters()

    print(f"\nMixture parameters:")
    print(f"  source = {mixture_source}")
    for k in ['N', 'H', 'L']:
        print(f"  {k}: pi={params['pi'][k]:.3f}, mu={params['mu'][k]:.3f}, "
              f"sigma={params['sigma'][k]:.3f}")

    print(f"\nDependence parameters:")
    print(f"  phi = {phi:.3f}")
    print(f"  Delta = {Delta:.3f}")

    # Choose evidence window B
    print("\nChoosing evidence window (significant but not too significant)...")
    n_eff_old_baseline = int(max(1, np.round(Delta * int(N_OLD_BASELINE))))
    window = choose_witness_window(
        params,
        n_eff=n_eff_old_baseline,
        rho_target=min(RHO_TARGETS),
        fdr_target=FDR_PLOT_TARGET,
    )
    B = window['B']
    p_pass = window['pass_probabilities']
    print(f"  B = ({B[0]:.2f}, {B[1] if np.isfinite(B[1]) else np.inf})")
    print(f"  Pass probabilities: p_H={p_pass['H']:.3f}, p_N={p_pass['N']:.3f}, p_L={p_pass['L']:.3f}")

    # =========================================================================
    # Main counterfactual: old vs new regime
    # =========================================================================
    print("\n" + "-" * 40)
    print("Counterfactual: Old vs New Regime")
    print("-" * 40)

    results = []

    gamma_old = 1.0 / N_OLD_BASELINE

    for lam in LAMBDA_GRID:
        gamma_new = float(lam) * gamma_old
        n_old = int(N_OLD_BASELINE)
        n_new = int(np.ceil(N_OLD_BASELINE / lam))

        n_eff_old = int(max(1, np.round(Delta * n_old)))
        n_eff_new = int(max(1, np.round(Delta * n_new)))

        for rho_target in RHO_TARGETS:
            for fdr_target in FDR_TARGETS:
                old_sol = find_min_m_for_targets(n_eff_old, p_pass, params, rho_target, fdr_target)
                new_sol = find_min_m_for_targets(n_eff_new, p_pass, params, rho_target, fdr_target)

                if old_sol is None or new_sol is None:
                    continue

                # Degradation: FDR in new regime using old regime's m
                degrad = screening_metrics(n_eff_new, old_sol['m'], p_pass, params)

                results.append({
                    'lambda': float(lam),
                    'rho_target': float(rho_target),
                    'FDR_target': float(fdr_target),
                    'B_lo': float(B[0]),
                    'B_hi': float(B[1]) if np.isfinite(B[1]) else np.nan,
                    'n_old': n_old,
                    'n_new': n_new,
                    'n_eff_old': n_eff_old,
                    'n_eff_new': n_eff_new,
                    'm_old': old_sol['m'],
                    'm_new': new_sol['m'],
                    'm_ratio': float(new_sol['m'] / old_sol['m']),
                    'theoretical_ratio': float(1 / lam),
                    'a_old': old_sol['a'],
                    'a_new': new_sol['a'],
                    'FDR_total_old': old_sol['FDR_total'],
                    'FDR_total_new': new_sol['FDR_total'],
                    'FDR_new_with_old_m': float(degrad['FDR_total']),
                })

                print(
                    f"  λ={lam:.3f} (n_old={n_old}, n_new={n_new}; n_eff_old={n_eff_old}, n_eff_new={n_eff_new}) "
                    f"ρ={rho_target:.2f}, FDR={fdr_target:.2f}: m_old={old_sol['m']}, m_new={new_sol['m']}"
                )

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # =========================================================================
    # Sensitivity: Dependence choice
    # =========================================================================
    print("\n" + "-" * 40)
    print("Sensitivity: Dependence Choice")
    print("-" * 40)

    dep_rows = []
    dep_variants = []
    dep_variants.append(("preferred", float(phi), float(Delta)))

    if dependence is not None:
        # Distance-based (rho(d)=phi^d) fit
        phi_dist = dependence.get("distance_based", {}).get("decay_fit", {}).get("phi", np.nan)
        if np.isfinite(phi_dist):
            dep_variants.append(("distance_based", float(phi_dist), float(1 - float(phi_dist))))

        # AR(1) CI endpoints (interpreting CI over pooled phi)
        ar1_pooled = dependence.get("ar1", {}).get("pooled", {})
        phi_lo = ar1_pooled.get("phi_ci_lower", np.nan)
        phi_hi = ar1_pooled.get("phi_ci_upper", np.nan)
        if np.isfinite(phi_lo) and np.isfinite(phi_hi):
            dep_variants.append(("ar1_ci_low_Delta", float(phi_hi), float(1 - float(phi_hi))))
            dep_variants.append(("ar1_ci_high_Delta", float(phi_lo), float(1 - float(phi_lo))))

    # Deduplicate by label
    seen = set()
    dep_variants_uniq = []
    for lab, ph, de in dep_variants:
        if lab in seen:
            continue
        seen.add(lab)
        dep_variants_uniq.append((lab, ph, de))

    for dep_label, phi_v, Delta_v in dep_variants_uniq:
        for lam in LAMBDA_GRID:
            n_old = int(N_OLD_BASELINE)
            n_new = int(np.ceil(N_OLD_BASELINE / lam))
            n_eff_old = int(max(1, np.round(Delta_v * n_old)))
            n_eff_new = int(max(1, np.round(Delta_v * n_new)))

            for rho_target in RHO_TARGETS:
                for fdr_target in FDR_TARGETS:
                    old_sol = find_min_m_for_targets(n_eff_old, p_pass, params, rho_target, fdr_target)
                    new_sol = find_min_m_for_targets(n_eff_new, p_pass, params, rho_target, fdr_target)
                    if old_sol is None or new_sol is None:
                        continue
                    dep_rows.append(
                        {
                            "dependence_label": dep_label,
                            "phi": float(phi_v),
                            "Delta": float(Delta_v),
                            "lambda": float(lam),
                            "rho_target": float(rho_target),
                            "FDR_target": float(fdr_target),
                            "n_old": int(n_old),
                            "n_new": int(n_new),
                            "n_eff_old": int(n_eff_old),
                            "n_eff_new": int(n_eff_new),
                            "m_old": int(old_sol["m"]),
                            "m_new": int(new_sol["m"]),
                            "m_ratio": float(new_sol["m"] / max(old_sol["m"], 1)),
                            "FDR_total_old": float(old_sol["FDR_total"]),
                            "FDR_total_new": float(new_sol["FDR_total"]),
                        }
                    )

        # Short printout for the headline operating point
        dep_df = pd.DataFrame(dep_rows)
        if dep_df.empty:
            print(f"  {dep_label}: no feasible (rho,FDR) solutions for this Delta")
            continue
        headline = dep_df[
            (dep_df["dependence_label"] == dep_label)
            & (dep_df["lambda"] == float(LAMBDA_BASELINE))
            & (dep_df["rho_target"] == float(RHO_PLOT))
            & (dep_df["FDR_target"] == float(FDR_PLOT_TARGET))
        ]
        if len(headline) == 1:
            r = headline.iloc[0]
            print(
                f"  {dep_label}: Δ={Delta_v:.3f} => (λ={LAMBDA_BASELINE:.3f}, ρ={RHO_PLOT:.2f}, FDR={FDR_PLOT_TARGET:.2f}) "
                f"m_old={int(r['m_old'])}, m_new={int(r['m_new'])} (ratio={r['m_ratio']:.2f})"
            )

    dep_df = pd.DataFrame(dep_rows)
    if dep_df.empty:
        raise RuntimeError("Dependence sensitivity produced no feasible rows; check mixture/dependence calibration.")

    # =========================================================================
    # Operating point curves (for plotting)
    # =========================================================================
    print("\n" + "-" * 40)
    print("Operating Point Curves (baseline λ)")
    print("-" * 40)

    op_rows = []
    rho_plot = float(RHO_PLOT)
    for regime, gamma in [('old', gamma_old), ('new', float(LAMBDA_BASELINE) * gamma_old)]:
        n = int(N_OLD_BASELINE) if regime == 'old' else int(np.ceil(N_OLD_BASELINE / LAMBDA_BASELINE))
        n_eff = int(max(1, np.round(Delta * n)))
        for m in range(1, n_eff + 1):
            met = screening_metrics(n_eff, m, p_pass, params)
            if met['Q_bar'] <= 0:
                continue
            a = float(min(1.0, rho_plot / met['Q_bar']))
            q_H = a * met['Q_H']
            q_0 = a * (params['pi']['N'] * met['Q_N'] + params['pi']['L'] * met['Q_L']) / max(params['pi']['N'] + params['pi']['L'], 1e-12)
            op_rows.append({
                'regime': regime,
                'gamma': float(gamma),
                'n': int(n),
                'n_eff': int(n_eff),
                'rho_target': float(rho_plot),
                'm': int(m),
                'a': a,
                'q_H': float(q_H),
                'q_0': float(q_0),
                **met,
            })
    op_df = pd.DataFrame(op_rows)

    # =========================================================================
    # COUNTERFACTUAL ALTERNATIVES
    # =========================================================================
    print("\n" + "=" * 60)
    print("Counterfactual Alternatives")
    print("=" * 60)

    alternatives: dict = {}

    # --- A. Window alternatives ---
    print("\n--- A. Window Alternatives ---")

    # A1. One-sided window: B = [1.96, inf)
    B_onesided = (float(Z_LO_MIN), float(np.inf))
    p_onesided = compute_pass_probabilities(B_onesided, params)
    alt_a1_rows = []
    for lam in LAMBDA_GRID:
        n_old = int(N_OLD_BASELINE)
        n_new = int(np.ceil(N_OLD_BASELINE / lam))
        n_eff_old = int(max(1, np.round(Delta * n_old)))
        n_eff_new = int(max(1, np.round(Delta * n_new)))
        for rho_target in RHO_TARGETS:
            for fdr_target in FDR_TARGETS:
                old_sol = find_min_m_for_targets(n_eff_old, p_onesided, params, rho_target, fdr_target)
                new_sol = find_min_m_for_targets(n_eff_new, p_onesided, params, rho_target, fdr_target)
                if old_sol and new_sol:
                    alt_a1_rows.append({
                        'lambda': float(lam), 'rho_target': rho_target, 'FDR_target': fdr_target,
                        'B_lo': B_onesided[0], 'B_hi': None,
                        'n_eff_old': n_eff_old, 'n_eff_new': n_eff_new,
                        'm_old': old_sol['m'], 'm_new': new_sol['m'],
                        'm_ratio': float(new_sol['m'] / max(old_sol['m'], 1)),
                    })
    alternatives['A1_one_sided_window'] = {
        'description': 'B = [1.96, inf): naive significance threshold, no upper bound',
        'pass_probabilities': {k: float(v) for k, v in p_onesided.items()},
        'results': alt_a1_rows,
    }
    print(f"  A1 one-sided: {len(alt_a1_rows)} feasible (rho,FDR,lambda) combos")

    # A2. Fixed lower + grid upper: B = [1.96, z_hi] for z_hi in grid
    z_hi_sens_grid = [3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
    alt_a2_rows = []
    for z_hi_val in z_hi_sens_grid:
        B_a2 = (float(Z_LO_MIN), float(z_hi_val))
        p_a2 = compute_pass_probabilities(B_a2, params)
        n_old = int(N_OLD_BASELINE)
        n_new = int(np.ceil(N_OLD_BASELINE / LAMBDA_BASELINE))
        n_eff_old = int(max(1, np.round(Delta * n_old)))
        n_eff_new = int(max(1, np.round(Delta * n_new)))
        old_sol = find_min_m_for_targets(n_eff_old, p_a2, params, RHO_PLOT, FDR_PLOT_TARGET)
        new_sol = find_min_m_for_targets(n_eff_new, p_a2, params, RHO_PLOT, FDR_PLOT_TARGET)
        alt_a2_rows.append({
            'z_hi': z_hi_val,
            'p_N': float(p_a2['N']), 'p_H': float(p_a2['H']), 'p_L': float(p_a2['L']),
            'm_old': old_sol['m'] if old_sol else None,
            'm_new': new_sol['m'] if new_sol else None,
            'm_ratio': float(new_sol['m'] / max(old_sol['m'], 1)) if old_sol and new_sol else None,
        })
    alternatives['A2_fixed_lower_grid_upper'] = {
        'description': 'B = [1.96, z_hi] at rho=0.10, FDR=0.10, lambda=1/14',
        'results': alt_a2_rows,
    }
    print(f"  A2 fixed lower + grid upper: {len(alt_a2_rows)} z_hi values")

    # A3. Wider optimal window: relax p_H_min from 0.05 to 0.01
    window_wider = choose_witness_window(
        params, p_H_min=0.01,
        n_eff=n_eff_old_baseline, rho_target=RHO_PLOT, fdr_target=FDR_PLOT_TARGET,
    )
    B_wider = window_wider['B']
    p_wider = window_wider['pass_probabilities']
    alt_a3_rows = []
    for lam in LAMBDA_GRID:
        n_old = int(N_OLD_BASELINE)
        n_new = int(np.ceil(N_OLD_BASELINE / lam))
        n_eff_old_a3 = int(max(1, np.round(Delta * n_old)))
        n_eff_new_a3 = int(max(1, np.round(Delta * n_new)))
        old_sol = find_min_m_for_targets(n_eff_old_a3, p_wider, params, RHO_PLOT, FDR_PLOT_TARGET)
        new_sol = find_min_m_for_targets(n_eff_new_a3, p_wider, params, RHO_PLOT, FDR_PLOT_TARGET)
        if old_sol and new_sol:
            alt_a3_rows.append({
                'lambda': float(lam),
                'B_lo': float(B_wider[0]),
                'B_hi': float(B_wider[1]) if np.isfinite(B_wider[1]) else None,
                'm_old': old_sol['m'], 'm_new': new_sol['m'],
                'm_ratio': float(new_sol['m'] / max(old_sol['m'], 1)),
            })
    alternatives['A3_wider_optimal_window'] = {
        'description': 'Optimal window with p_H_min=0.01 (relaxed from 0.05)',
        'B': [float(B_wider[0]), float(B_wider[1]) if np.isfinite(B_wider[1]) else None],
        'pass_probabilities': {k: float(v) for k, v in p_wider.items()},
        'results': alt_a3_rows,
    }
    print(f"  A3 wider optimal: B=({B_wider[0]:.2f}, {B_wider[1] if np.isfinite(B_wider[1]) else 'inf'}), {len(alt_a3_rows)} combos")

    # --- B. Calibration alternatives ---
    print("\n--- B. Calibration Alternatives ---")

    # B4. n_old sensitivity
    n_old_grid = [20, 33, 50]
    alt_b4_rows = []
    for n_old_val in n_old_grid:
        for lam in LAMBDA_GRID:
            n_new_val = int(np.ceil(n_old_val / lam))
            n_eff_old_b4 = int(max(1, np.round(Delta * n_old_val)))
            n_eff_new_b4 = int(max(1, np.round(Delta * n_new_val)))
            for rho_target in RHO_TARGETS:
                for fdr_target in FDR_TARGETS:
                    old_sol = find_min_m_for_targets(n_eff_old_b4, p_pass, params, rho_target, fdr_target)
                    new_sol = find_min_m_for_targets(n_eff_new_b4, p_pass, params, rho_target, fdr_target)
                    if old_sol and new_sol:
                        alt_b4_rows.append({
                            'n_old': n_old_val, 'lambda': float(lam),
                            'rho_target': rho_target, 'FDR_target': fdr_target,
                            'n_eff_old': n_eff_old_b4, 'n_eff_new': n_eff_new_b4,
                            'm_old': old_sol['m'], 'm_new': new_sol['m'],
                            'm_ratio': float(new_sol['m'] / max(old_sol['m'], 1)),
                        })
    alternatives['B4_n_old_sensitivity'] = {
        'description': 'Sensitivity to baseline horizon n_old',
        'n_old_grid': n_old_grid,
        'results': alt_b4_rows,
    }
    # Print headline for each n_old
    for n_old_val in n_old_grid:
        headline = [r for r in alt_b4_rows
                    if r['n_old'] == n_old_val and abs(r['lambda'] - LAMBDA_BASELINE) < 1e-6
                    and r['rho_target'] == RHO_PLOT and r['FDR_target'] == FDR_PLOT_TARGET]
        if headline:
            h = headline[0]
            print(f"  B4 n_old={n_old_val}: m_old={h['m_old']}, m_new={h['m_new']}, ratio={h['m_ratio']:.2f}")

    # B5. Delta alternatives from dependence models
    alt_b5_rows = []
    delta_variants = [("preferred", float(Delta))]

    if dependence is not None:
        alt_models = dependence.get("alternative_models", {})
        for model_name, model_data in alt_models.items():
            delta_val = model_data.get("Delta", model_data.get("Delta_at_d1", np.nan))
            if np.isfinite(delta_val) and delta_val > 0:
                delta_variants.append((model_name, float(delta_val)))

        # Also add AR(1) pooled
        ar1_delta = dependence.get("ar1", {}).get("pooled", {}).get("Delta", np.nan)
        if np.isfinite(ar1_delta):
            delta_variants.append(("ar1_pooled", float(ar1_delta)))

    for dep_label, delta_val in delta_variants:
        for lam in LAMBDA_GRID:
            n_old = int(N_OLD_BASELINE)
            n_new = int(np.ceil(N_OLD_BASELINE / lam))
            n_eff_old_b5 = int(max(1, np.round(delta_val * n_old)))
            n_eff_new_b5 = int(max(1, np.round(delta_val * n_new)))
            old_sol = find_min_m_for_targets(n_eff_old_b5, p_pass, params, RHO_PLOT, FDR_PLOT_TARGET)
            new_sol = find_min_m_for_targets(n_eff_new_b5, p_pass, params, RHO_PLOT, FDR_PLOT_TARGET)
            if old_sol and new_sol:
                alt_b5_rows.append({
                    'dependence_model': dep_label, 'Delta': delta_val, 'lambda': float(lam),
                    'n_eff_old': n_eff_old_b5, 'n_eff_new': n_eff_new_b5,
                    'm_old': old_sol['m'], 'm_new': new_sol['m'],
                    'm_ratio': float(new_sol['m'] / max(old_sol['m'], 1)),
                })
    alternatives['B5_Delta_alternatives'] = {
        'description': 'Delta from alternative dependence models',
        'variants': {lab: dv for lab, dv in delta_variants},
        'results': alt_b5_rows,
    }
    for dep_label, delta_val in delta_variants:
        headline = [r for r in alt_b5_rows
                    if r['dependence_model'] == dep_label and abs(r['lambda'] - LAMBDA_BASELINE) < 1e-6]
        if headline:
            h = headline[0]
            print(f"  B5 {dep_label} (Delta={delta_val:.3f}): m_old={h['m_old']}, m_new={h['m_new']}, ratio={h['m_ratio']:.2f}")

    # --- C. Screening rule alternatives ---
    print("\n--- C. Screening Rule Alternatives ---")

    # C6. Share-based rule: require m/n >= tau
    alt_c6_rows = []
    for lam in LAMBDA_GRID:
        n_old = int(N_OLD_BASELINE)
        n_new = int(np.ceil(N_OLD_BASELINE / lam))
        n_eff_old_c6 = int(max(1, np.round(Delta * n_old)))
        n_eff_new_c6 = int(max(1, np.round(Delta * n_new)))
        for rho_target in RHO_TARGETS:
            for fdr_target in FDR_TARGETS:
                old_sol = share_based_threshold(n_eff_old_c6, p_pass, params, rho_target, fdr_target)
                new_sol = share_based_threshold(n_eff_new_c6, p_pass, params, rho_target, fdr_target)
                if old_sol and new_sol:
                    alt_c6_rows.append({
                        'lambda': float(lam), 'rho_target': rho_target, 'FDR_target': fdr_target,
                        'n_eff_old': n_eff_old_c6, 'n_eff_new': n_eff_new_c6,
                        'tau_old': old_sol['tau'], 'tau_new': new_sol['tau'],
                        'tau_ratio': float(new_sol['tau'] / max(old_sol['tau'], 1e-12)),
                        'm_old': old_sol['m'], 'm_new': new_sol['m'],
                        'm_ratio': float(new_sol['m'] / max(old_sol['m'], 1)),
                    })
    alternatives['C6_share_based'] = {
        'description': 'Share-based rule: require m/n >= tau. Key: does tau change with n?',
        'results': alt_c6_rows,
    }
    headline_c6 = [r for r in alt_c6_rows
                   if abs(r['lambda'] - LAMBDA_BASELINE) < 1e-6
                   and r['rho_target'] == RHO_PLOT and r['FDR_target'] == FDR_PLOT_TARGET]
    if headline_c6:
        h = headline_c6[0]
        print(f"  C6 share-based: tau_old={h['tau_old']:.3f}, tau_new={h['tau_new']:.3f}, ratio={h['tau_ratio']:.2f}")

    # C7. Likelihood-ratio screening
    alt_c7_rows = []
    for lam in LAMBDA_GRID:
        n_old = int(N_OLD_BASELINE)
        n_new = int(np.ceil(N_OLD_BASELINE / lam))
        n_eff_old_c7 = int(max(1, np.round(Delta * n_old)))
        n_eff_new_c7 = int(max(1, np.round(Delta * n_new)))
        for rho_target in RHO_TARGETS:
            for fdr_target in FDR_TARGETS:
                old_sol = likelihood_ratio_screening(n_eff_old_c7, p_pass, params, rho_target, fdr_target)
                new_sol = likelihood_ratio_screening(n_eff_new_c7, p_pass, params, rho_target, fdr_target)
                if old_sol and new_sol:
                    alt_c7_rows.append({
                        'lambda': float(lam), 'rho_target': rho_target, 'FDR_target': fdr_target,
                        'n_eff_old': n_eff_old_c7, 'n_eff_new': n_eff_new_c7,
                        'posterior_threshold_old': old_sol['posterior_threshold'],
                        'posterior_threshold_new': new_sol['posterior_threshold'],
                        'equiv_m_old': old_sol['equivalent_m'],
                        'equiv_m_new': new_sol['equivalent_m'],
                        'FDR_old': old_sol['FDR_total'],
                        'FDR_new': new_sol['FDR_total'],
                    })
    alternatives['C7_likelihood_ratio'] = {
        'description': 'Likelihood-ratio screening: accept if posterior P(H) > threshold. Bayes-optimal rule.',
        'results': alt_c7_rows,
    }
    headline_c7 = [r for r in alt_c7_rows
                   if abs(r['lambda'] - LAMBDA_BASELINE) < 1e-6
                   and r['rho_target'] == RHO_PLOT and r['FDR_target'] == FDR_PLOT_TARGET]
    if headline_c7:
        h = headline_c7[0]
        print(f"  C7 LR-screening: post_thresh_old={h['posterior_threshold_old']:.3f}, "
              f"post_thresh_new={h['posterior_threshold_new']:.3f}, "
              f"equiv_m_old={h['equiv_m_old']}, equiv_m_new={h['equiv_m_new']}")

    # =========================================================================
    # D. Null-only FDR with fixed window and constrained-σ mixture
    # =========================================================================
    print("\n" + "=" * 60)
    print("D. Null-only FDR counterfactual (σ=1 fixed, B=[1.96,10])")
    print("=" * 60)

    # Load sigma_fixed_1 mixture params
    constrained_params = None
    constrained_source = None
    if MIXTURE_FILE.exists():
        with open(MIXTURE_FILE, 'r') as f:
            mixture_all = json.load(f)
        constrained_params = (
            mixture_all
            .get("spec_level", {})
            .get("trim_sensitivity", {})
            .get("trim_abs_le_10_sigma_fixed_1")
        )
        if constrained_params is not None:
            constrained_source = "spec_level:trim_sensitivity:trim_abs_le_10_sigma_fixed_1"

    if constrained_params is None:
        print("  WARNING: sigma_fixed_1 params not found, falling back to baseline params")
        constrained_params = params
        constrained_source = mixture_source + " (fallback)"

    print(f"  Mixture source: {constrained_source}")
    for k in ['N', 'H', 'L']:
        print(f"    {k}: pi={constrained_params['pi'][k]:.3f}, "
              f"mu={constrained_params['mu'][k]:.3f}, "
              f"sigma={constrained_params['sigma'][k]:.3f}")

    # Fixed evidence window
    B_fixed = (float(Z_LO_MIN), 10.0)
    p_pass_fixed = compute_pass_probabilities(B_fixed, constrained_params)
    print(f"  Fixed window B = ({B_fixed[0]:.2f}, {B_fixed[1]:.2f})")
    print(f"  Pass probs: p_N={p_pass_fixed['N']:.4f}, p_H={p_pass_fixed['H']:.4f}, p_L={p_pass_fixed['L']:.4f}")

    # Run counterfactual with null-only FDR
    nullfdr_results = []
    for lam in LAMBDA_GRID:
        n_old = int(N_OLD_BASELINE)
        n_new = int(np.ceil(N_OLD_BASELINE / lam))
        n_eff_old = int(max(1, np.round(Delta * n_old)))
        n_eff_new = int(max(1, np.round(Delta * n_new)))

        for rho_target in RHO_TARGETS:
            for fdr_target in FDR_TARGETS:
                old_sol = find_min_m_for_targets(
                    n_eff_old, p_pass_fixed, constrained_params,
                    rho_target, fdr_target, fdr_mode="null_only"
                )
                new_sol = find_min_m_for_targets(
                    n_eff_new, p_pass_fixed, constrained_params,
                    rho_target, fdr_target, fdr_mode="null_only"
                )
                if old_sol is None or new_sol is None:
                    continue

                degrad = screening_metrics(n_eff_new, old_sol['m'], p_pass_fixed, constrained_params)

                nullfdr_results.append({
                    'lambda': float(lam),
                    'rho_target': float(rho_target),
                    'FDR_target': float(fdr_target),
                    'B_lo': float(B_fixed[0]),
                    'B_hi': float(B_fixed[1]),
                    'n_old': n_old,
                    'n_new': n_new,
                    'n_eff_old': n_eff_old,
                    'n_eff_new': n_eff_new,
                    'm_old': old_sol['m'],
                    'm_new': new_sol['m'],
                    'm_ratio': float(new_sol['m'] / old_sol['m']),
                    'a_old': old_sol['a'],
                    'a_new': new_sol['a'],
                    'FDR_null_old': old_sol['FDR_null'],
                    'FDR_null_new': new_sol['FDR_null'],
                    'FDR_null_new_with_old_m': float(degrad['FDR_null']),
                })

                print(
                    f"  λ={lam:.3f} (n_eff_old={n_eff_old}, n_eff_new={n_eff_new}) "
                    f"ρ={rho_target:.2f}, FDR_null={fdr_target:.2f}: "
                    f"m_old={old_sol['m']}, m_new={new_sol['m']}"
                )

    nullfdr_df = pd.DataFrame(nullfdr_results)

    # Operating-point curves for null-only FDR variant
    nullfdr_op_rows = []
    for regime, gamma in [('old', gamma_old), ('new', float(LAMBDA_BASELINE) * gamma_old)]:
        n = int(N_OLD_BASELINE) if regime == 'old' else int(np.ceil(N_OLD_BASELINE / LAMBDA_BASELINE))
        n_eff = int(max(1, np.round(Delta * n)))
        for m in range(1, n_eff + 1):
            met = screening_metrics(n_eff, m, p_pass_fixed, constrained_params)
            if met['Q_bar'] <= 0:
                continue
            a = float(min(1.0, float(RHO_PLOT) / met['Q_bar']))
            nullfdr_op_rows.append({
                'regime': regime,
                'n_eff': int(n_eff),
                'm': int(m),
                'a': a,
                **met,
            })

    # =========================================================================
    # Save results
    # =========================================================================
    print("\n" + "-" * 40)
    print("Saving results...")
    print("-" * 40)

    # Save counterfactual results
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved counterfactual results to {OUTPUT_CSV}")

    # Save dependence sensitivity results
    if len(dep_df) > 0:
        dep_df.to_csv(OUTPUT_DEP_SENS, index=False)
        print(f"  Saved dependence sensitivity to {OUTPUT_DEP_SENS}")

    # Save operating point curves
    op_df.to_csv(RESULTS_DIR / "operating_points.csv", index=False)
    print(f"  Saved operating points to {RESULTS_DIR / 'operating_points.csv'}")

    # Save parameters and summary
    summary = {
        "mixture_source": mixture_source,
        'mixture_params': {
            'pi': params['pi'],
            'mu': params['mu'],
            'sigma': params['sigma'],
        },
        'dependence': {
            'phi': phi,
            'Delta': Delta,
        },
        'evidence_window': {
            'z_lo': B[0],
            'z_hi': B[1] if np.isfinite(B[1]) else None,
        },
        'pass_probabilities': p_pass,
        'cost_parameters': {
            'lambda_baseline': LAMBDA_BASELINE,
            'lambda_grid': LAMBDA_GRID,
        },
        'horizon': {
            'n_old_baseline': N_OLD_BASELINE,
            'rho_plot': rho_plot,
            'fdr_plot_target': float(FDR_PLOT_TARGET),
            'gamma_old': float(gamma_old),
        },
        'counterfactual_summary': results_df.to_dict('records') if len(results_df) > 0 else [],
        # Null-only FDR variant (σ=1 fixed, B=[1.96,10])
        'nullfdr_variant': {
            'description': 'Null-only FDR with sigma=1 fixed mixture and B=[1.96,10]',
            'fdr_mode': 'null_only',
            'mixture_source': constrained_source,
            'mixture_params': {
                'pi': constrained_params['pi'],
                'mu': constrained_params['mu'],
                'sigma': constrained_params['sigma'],
            },
            'evidence_window': {
                'z_lo': float(B_fixed[0]),
                'z_hi': float(B_fixed[1]),
            },
            'pass_probabilities': {k: float(v) for k, v in p_pass_fixed.items()},
            'counterfactual_summary': nullfdr_df.to_dict('records') if len(nullfdr_df) > 0 else [],
            'operating_points': nullfdr_op_rows,
        },
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved parameters to {OUTPUT_JSON}")

    # Save alternatives
    with open(OUTPUT_ALTERNATIVES, 'w') as f:
        json.dump(alternatives, f, indent=2)
    print(f"  Saved counterfactual alternatives to {OUTPUT_ALTERNATIVES}")

    print("\nDone!")


if __name__ == "__main__":
    main()
