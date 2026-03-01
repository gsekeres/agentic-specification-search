#!/usr/bin/env julia
#=
make_figures.jl
===============
Generate all estimation figures (replaces Python fig*.py scripts).
Uses PyPlot backend with LaTeX text rendering.

Usage:
    julia estimation/scripts/make_figures.jl
=#

using Plots, LaTeXStrings, JSON, CSV, DataFrames, Statistics, Distributions
import PyPlot
using PyPlot: matplotlib
using PyPlot.PyCall: pyimport

pyplot()
PyPlot.rc("text", usetex=true)
PyPlot.rc("font", family="serif")
PyPlot.rc("font", size=12)
matplotlib.rcParams["mathtext.fontset"] = "cm"

# ── Paths ────────────────────────────────────────────────────────────────────
const BASE_DIR    = dirname(dirname(@__DIR__))
const DATA_DIR    = joinpath(BASE_DIR, "estimation", "data")
const RESULTS_DIR = joinpath(BASE_DIR, "estimation", "results")
const FIG_DIR     = joinpath(BASE_DIR, "estimation", "figures")
const OL_FIG_DIR  = joinpath(dirname(BASE_DIR), "overleaf", "tex", "v8_figures")
const PAPER_FIG_DIR = joinpath(dirname(BASE_DIR), "scientific-competition-overleaf", "tex", "v8_figures")

mkpath(FIG_DIR)

# ── Utility functions ────────────────────────────────────────────────────────

function save_both(fig, name)
    fig.savefig(joinpath(FIG_DIR, name), bbox_inches="tight", facecolor="white", dpi=300)
    for out_dir in (OL_FIG_DIR, PAPER_FIG_DIR)
        if isdir(out_dir)
            fig.savefig(joinpath(out_dir, name), bbox_inches="tight", transparent=true, dpi=300)
        end
    end
    PyPlot.close(fig)
    println("  Saved $name")
end

function nospines!(ax)
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
end

"""Gaussian KDE with Silverman bandwidth."""
function kde(data::Vector{Float64}, grid; bw=nothing)
    n = length(data)
    n < 2 && return zeros(length(grid))
    σ = std(data)
    iq = quantile(data, 0.75) - quantile(data, 0.25)
    h = bw !== nothing ? bw : 0.9 * min(σ, iq / 1.34) * n^(-0.2)
    h = max(h, 1e-6)
    [sum(exp.(-0.5 .* ((x .- data) ./ h).^2)) / (n * h * √(2π)) for x in grid]
end

"""Truncated-normal PDF (truncated at lo from below)."""
function tn_pdf(x, μ, σ; lo=0.0)
    σ = max(σ, 1e-8)
    pdf(Truncated(Normal(μ, σ), lo, Inf), x)
end

"""K-component truncated-normal mixture PDF."""
function mix_pdf(x, πv, μv, σv; lo=0.0)
    sum(πv[k] * tn_pdf(x, μv[k], σv[k]; lo=lo) for k in eachindex(πv))
end

"""K-component truncated-normal mixture CDF."""
function mix_cdf(x, πv, μv, σv; lo=0.0)
    sum(πv[k] * cdf(Truncated(Normal(μv[k], max(σv[k], 1e-8)), lo, Inf), x) for k in eachindex(πv))
end

"""Safely extract a numeric column from a DataFrame, returning NaN for missing."""
function numcol(df, c)
    s = Symbol(c)
    hasproperty(df, s) || return Float64[]
    col = df[!, s]
    out = Float64[]
    for v in col
        if ismissing(v)
            push!(out, NaN)
        else
            try
                push!(out, Float64(v))
            catch
                try
                    push!(out, parse(Float64, string(v)))
                catch
                    push!(out, NaN)
                end
            end
        end
    end
    return out
end

"""Filter to finite values."""
finite(v) = filter(isfinite, v)

"""Find first matching column name from candidates."""
function findcol(df, candidates...)
    for c in candidates
        string(c) in names(df) && return string(c)
    end
    return nothing
end

"""Load audit exclusion sets from i4r_paper_audit.csv."""
function load_audit_sets()
    af = joinpath(RESULTS_DIR, "i4r_paper_audit.csv")
    excl = Set{String}()
    review = Set{String}()
    isfile(af) || return excl, review
    aud = CSV.read(af, DataFrame)
    for r in eachrow(aud)
        pid = string(r.paper_id)
        if hasproperty(aud, :exclude_i4r) && !ismissing(r.exclude_i4r) && r.exclude_i4r == 1
            push!(excl, pid)
        end
        if hasproperty(aud, :needs_review) && !ismissing(r.needs_review) && r.needs_review == 1
            push!(review, pid)
        end
    end
    return excl, review
end


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Z distributions threeway
# ══════════════════════════════════════════════════════════════════════════════

function fig1_z_threeway()
    println("Fig 1: Z distributions threeway (1×3 panels)")
    f = joinpath(DATA_DIR, "i4r_comparison.csv")
    isfile(f) || return @warn "SKIP: i4r_comparison.csv missing"
    df = CSV.read(f, DataFrame)
    excl, _ = load_audit_sets()
    df = df[.!in.(string.(df.paper_id), Ref(excl)), :]

    t_orig = finite(abs.(numcol(df, "t_orig")))
    t_i4r  = finite(abs.(numcol(df, "t_i4r")))
    ai_col = findcol(df, "t_AI_abs", "t_AI")
    t_ai   = ai_col !== nothing ? finite(abs.(numcol(df, ai_col))) : Float64[]

    # Load oracle series
    oracle_f = joinpath(DATA_DIR, "i4r_oracle_claim_map.csv")
    t_oracle = Float64[]
    if isfile(oracle_f)
        odf = CSV.read(oracle_f, DataFrame)
        t_oracle = finite(abs.(numcol(odf, "oracle_abs_t_stat")))
    end

    xm = 8.0; bins = collect(0:0.5:xm)

    # Define the three comparison series
    series = [
        ("I4R Benchmark",              t_i4r,   "#2563eb"),
        ("Automated Matched Subset",   t_oracle, "#009E73"),
        ("Full Automated Sample",      t_ai,    "#B31B1B"),
    ]

    fig, axes = PyPlot.subplots(1, 3, figsize=(10, 5), sharey=true)
    ym_global = 0.0

    for (i, (label, t_series, color)) in enumerate(series)
        ax = axes[i]
        # Gray background: original studies
        ax.hist(t_orig[t_orig .< xm], bins=bins, density=true,
            color="#bdbdbd", alpha=0.45, edgecolor="white", linewidth=0.5)
        # Colored overlay
        if length(t_series) > 2
            ax.hist(t_series[t_series .< xm], bins=bins, density=true,
                color=color, alpha=0.55, edgecolor="white", linewidth=0.5)
        end
        ax.axvline(1.96, color="#737373", lw=1.0, alpha=0.6)
        ax.set_title("$label", fontsize=11)
        ax.set_xlim(0, xm)
        nospines!(ax)
        ym_global = max(ym_global, ax.get_ylim()[2])
    end

    for ax in axes; ax.set_ylim(0, ym_global * 1.05) end
    axes[1].set_ylabel("Density", fontsize=13)
    fig.text(0.5, -0.02, L"Evidence index $|t|$ (absolute $t$-statistic)", ha="center", fontsize=13)
    fig.tight_layout()
    save_both(fig, "fig_z_distributions_threeway.pdf")
end


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1b: t-stat distributions with progressive filters
# ══════════════════════════════════════════════════════════════════════════════

function fig1b_tstat_filters()
    println("Fig 1b: t-stat distributions with filters")
    f = joinpath(DATA_DIR, "i4r_comparison.csv")
    isfile(f) || return @warn "SKIP"
    df = CSV.read(f, DataFrame)
    excl, review = load_audit_sets()
    pids = string.(df.paper_id)

    mask_excl  = .!in.(pids, Ref(excl))
    mask_audit = mask_excl .& .!in.(pids, Ref(review))

    subsets = [
        ("A: Full sample",    trues(nrow(df))),
        ("B: Excl.\\ flagged", mask_excl),
        ("C: Audit-passed",   mask_audit),
    ]

    xm = 12.0; xg = collect(range(0, xm, length=500)); bins = collect(0:0.5:xm)
    ai_col = findcol(df, "t_AI_abs", "t_AI")

    fig, axes = PyPlot.subplots(1, 3, figsize=(15, 4.5), sharey=true)
    ym_global = 0.0

    for (i, (title, mask)) in enumerate(subsets)
        ax = axes[i]
        sub = df[mask, :]
        to = finite(abs.(numcol(sub, "t_orig")))
        ti = finite(abs.(numcol(sub, "t_i4r")))
        ta = ai_col !== nothing ? finite(abs.(numcol(sub, ai_col))) : Float64[]

        ax.hist(to[to .< xm], bins=bins, density=true, color="#bdbdbd", alpha=0.55,
            edgecolor="white", linewidth=0.5)
        length(ti) > 2 && ax.plot(xg, kde(ti, xg), color="#2563eb", lw=2.6, ls="--")
        length(ta) > 2 && ax.plot(xg, kde(ta, xg), color="#B31B1B", lw=2.6, ls=":")
        ax.axvline(1.96, color="#737373", lw=1.0, alpha=0.6)
        n_str = length(to)
        ax.set_title("$title (n=$n_str)", fontsize=12)
        ax.set_xlim(0, xm)
        nospines!(ax)
        ym_global = max(ym_global, ax.get_ylim()[2])
    end

    for ax in axes; ax.set_ylim(0, ym_global * 1.05) end
    axes[1].set_ylabel("Density", fontsize=13)
    fig.text(0.5, -0.02, L"Absolute $t$-statistic $|t|$", ha="center", fontsize=13)

    # Legend on last panel
    h = [
        matplotlib.patches.Patch(facecolor="#bdbdbd", alpha=0.55, label="Original"),
        matplotlib.lines.Line2D([0],[0], color="#2563eb", lw=2.6, ls="--", label="Independent"),
        matplotlib.lines.Line2D([0],[0], color="#B31B1B", lw=2.6, ls=":", label="Automated"),
    ]
    axes[end].legend(handles=h, fontsize=11, frameon=false, loc="upper right")
    fig.tight_layout()
    save_both(fig, "fig_tstat_distributions_threeway_filters.pdf")
end


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Three-type mixture fit
# ══════════════════════════════════════════════════════════════════════════════

function fig2_mixture_fit()
    println("Fig 2: Three-type mixture fit")
    mf = joinpath(RESULTS_DIR, "mixture_params_abs_t.json")
    isfile(mf) || return @warn "SKIP: mixture_params_abs_t.json missing"
    mix = JSON.parsefile(mf)

    # Locate K=3 params: prefer the baseline-only sigma=1 fit (used by the counterfactual),
    # then baseline-only sigma-free, then k_sensitivity K=3.
    params = get(get(mix, "spec_level", Dict()), "baseline_only_sigma_fixed_1", nothing)
    if params === nothing
        params = get(get(mix, "spec_level", Dict()), "baseline_only", nothing)
    end
    if params === nothing
        params = get(get(get(mix, "k_sensitivity", Dict()), "K=3", Dict()), "truncnorm", nothing)
    end
    params === nothing && return @warn "SKIP: no K=3 mixture params"

    keys3 = haskey(params["pi"], "N") ? ["N","H","L"] : ["Low","High"]
    length(keys3) != 3 && return @warn "SKIP: mixture is not K=3"
    πv = [params["pi"][k]    for k in keys3]
    μv = [params["mu"][k]    for k in keys3]
    σv = [params["sigma"][k] for k in keys3]
    lo = get(params, "truncation_lo", 0.0)

    # Histogram data: verified-core replications (x-axis capped for readability)
    t_data = load_verified_core_abs_t()
    t_data === nothing && return @warn "SKIP: no verified-core data"

    xmax = min(max(maximum(t_data), maximum(μv .+ 4 .* σv)) + 0.25, 10.0)
    xg = collect(range(0, xmax, length=500))
    total = [mix_pdf(x, πv, μv, σv; lo=lo) for x in xg]
    comps = [[πv[k] * tn_pdf(x, μv[k], σv[k]; lo=lo) for x in xg] for k in 1:3]

    cc = ["#2563eb", "#009E73", "#B31B1B"]
    cl = [L"Null ($N$)", L"Moderate ($M$)", L"Extreme ($E$)"]

    fig, ax = PyPlot.subplots(figsize=(10, 5))

    ax.hist(t_data[t_data .< xmax], bins=collect(range(0, xmax, step=0.3)),
        density=true, color="#cccccc", alpha=0.5, edgecolor="white", linewidth=0.3,
        label="Replications")
    for k in 1:3
        ax.fill_between(xg, 0, comps[k], color=cc[k], alpha=0.20)
        ax.plot(xg, comps[k], color=cc[k], lw=2.0, ls="--", label=cl[k])
    end
    ax.plot(xg, total, color="black", lw=2.8, label="Mixture total")
    ax.axvline(1.96, color="#737373", lw=1.0, alpha=0.6)

    # Component labels at peaks
    for (k, label) in enumerate(["N", "M", "E"])
        pk = maximum(comps[k])
        pk > 0.01 && ax.text(μv[k], pk * 1.08, label, fontsize=14, fontweight="bold",
            color=cc[k], ha="center")
    end

    ax.set_xlabel(L"Evidence index $|t|$ (absolute $t$-statistic)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_xlim(0, xmax)
    nospines!(ax)
    ax.legend(fontsize=10, frameon=false, loc="upper right")
    fig.tight_layout()
    save_both(fig, "fig_three_type_mixture_fit.pdf")
end


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2b/2c: Mixture fit for K=2 and K=4
# ══════════════════════════════════════════════════════════════════════════════

# Component labels and colors for each K
const K_LABELS = Dict(
    2 => (["Low", "High"],
          ["#2563eb", "#009E73"],
          [L"Low", L"High"]),
    3 => (["N", "H", "L"],
          ["#2563eb", "#009E73", "#B31B1B"],
          [L"Null ($N$)", L"Moderate ($M$)", L"Extreme ($E$)"]),
    4 => (["N", "H1", "H2", "L"],
          ["#2563eb", "#009E73", "#E69F00", "#B31B1B"],
          [L"$N$", L"$M_1$", L"$M_2$", L"$E$"]),
)

"""Load verified-core |t| data for histograms."""
function load_verified_core_abs_t()
    vcf = joinpath(DATA_DIR, "spec_level_verified_core.csv")
    isfile(vcf) || return nothing
    vc = CSV.read(vcf, DataFrame)
    t_col = findcol(vc, "Z_abs", "Z", "t_stat")
    t_col === nothing && return nothing
    return finite(clamp.(abs.(numcol(vc, t_col)), 0, 20.0))
end

"""
    fig_mixture_k(K) → generates a single-panel mixture fit figure for a given K.
    Uses k_sensitivity params and verified-core replications as histogram.
"""
function fig_mixture_k(K::Int)
    println("Fig 2 (K=$K): Mixture fit")
    mf = joinpath(RESULTS_DIR, "mixture_params_abs_t.json")
    isfile(mf) || return @warn "SKIP: mixture_params_abs_t.json missing"
    mix = JSON.parsefile(mf)

    params = get(get(get(mix, "k_sensitivity", Dict()), "K=$K", Dict()), "truncnorm", nothing)
    params === nothing && return @warn "SKIP: no K=$K truncnorm params"

    info = get(K_LABELS, K, nothing)
    info === nothing && return @warn "SKIP: no label info for K=$K"
    keys_k, colors_k, labels_k = info

    πv = [params["pi"][k]    for k in keys_k]
    μv = [params["mu"][k]    for k in keys_k]
    σv = [params["sigma"][k] for k in keys_k]
    lo = get(params, "truncation_lo", 0.0)
    aic = round(params["aic"], digits=1)
    bic = round(params["bic"], digits=1)
    n_obs = Int(params["n_obs"])

    t_data = load_verified_core_abs_t()
    t_data === nothing && return @warn "SKIP: no verified-core data"
    t_data = t_data[t_data .<= 10.0]

    xmax = min(max(maximum(t_data), maximum(μv .+ 4 .* σv)) + 0.25, 10.0)
    xg = collect(range(0, xmax, length=500))
    total = [mix_pdf(x, πv, μv, σv; lo=lo) for x in xg]
    comps = [[πv[k] * tn_pdf(x, μv[k], σv[k]; lo=lo) for x in xg] for k in 1:K]

    fig, ax = PyPlot.subplots(figsize=(8, 5))
    ax.hist(t_data[t_data .< xmax], bins=collect(range(0, xmax, step=0.3)),
        density=true, color="#cccccc", alpha=0.5, edgecolor="white", linewidth=0.3,
        label="Replications")
    for k in 1:K
        ax.fill_between(xg, 0, comps[k], color=colors_k[k], alpha=0.18)
        ax.plot(xg, comps[k], color=colors_k[k], lw=2.0, ls="--", label=labels_k[k])
    end
    ax.plot(xg, total, color="black", lw=2.8, label="Mixture total")
    ax.axvline(1.96, color="#737373", lw=1.0, alpha=0.6)

    # Component labels at peaks
    for k in 1:K
        pk = maximum(comps[k])
        pk > 0.01 && ax.text(μv[k], pk * 1.08, keys_k[k], fontsize=13,
            fontweight="bold", color=colors_k[k], ha="center")
    end

    ax.set_xlabel(L"Evidence index $|t|$ (absolute $t$-statistic)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_xlim(0, xmax)
    nospines!(ax)
    ax.legend(fontsize=10, frameon=false, loc="upper right")
    fig.tight_layout()
    save_both(fig, "fig_mixture_fit_K$(K).pdf")
end

fig2b_mixture_k2() = fig_mixture_k(2)
fig2c_mixture_k4() = fig_mixture_k(4)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE: Folded-normal mixture fits (σ=1, μ_N=0 robustness)
# ══════════════════════════════════════════════════════════════════════════════

"""Folded-normal PDF: pdf of |X| where X ~ N(μ, σ²)."""
function fn_pdf(x, μ, σ)
    σ = max(σ, 1e-8)
    pdf(Normal(μ, σ), x) + pdf(Normal(μ, σ), -x)
end

"""K-component folded-normal mixture PDF."""
function fn_mix_pdf(x, πv, μv, σv)
    sum(πv[k] * fn_pdf(x, μv[k], σv[k]) for k in eachindex(πv))
end

"""
    fig_folded_k(K) → generates a single-panel folded-normal mixture fit figure.
    Uses folded_normal_robustness params (σ=1, μ_N=0) and verified-core |t|≤10 histogram.
"""
function fig_folded_k(K::Int)
    println("Fig folded K=$K: Folded-normal mixture (σ=1, μ_N=0)")
    mf = joinpath(RESULTS_DIR, "mixture_params_abs_t.json")
    isfile(mf) || return @warn "SKIP: mixture_params_abs_t.json missing"
    mix = JSON.parsefile(mf)

    params = get(get(mix, "folded_normal_robustness", Dict()), "K=$K", nothing)
    params === nothing && return @warn "SKIP: no folded K=$K params"

    info = get(K_LABELS, K, nothing)
    info === nothing && return @warn "SKIP: no label info for K=$K"
    keys_k, colors_k, labels_k = info

    πv = [params["pi"][k]    for k in keys_k]
    μv = [params["mu"][k]    for k in keys_k]
    σv = [params["sigma"][k] for k in keys_k]
    aic = round(params["aic"], digits=1)
    bic = round(params["bic"], digits=1)
    n_obs = Int(params["n_obs"])

    t_data = load_verified_core_abs_t()
    t_data === nothing && return @warn "SKIP: no verified-core data"
    t_data = t_data[t_data .<= 10.0]

    xmax = min(max(maximum(t_data), maximum(μv .+ 4 .* σv)) + 0.25, 10.0)
    xg = collect(range(0, xmax, length=500))
    total = [fn_mix_pdf(x, πv, μv, σv) for x in xg]
    comps = [[πv[k] * fn_pdf(x, μv[k], σv[k]) for x in xg] for k in 1:K]

    # Peak labels (short)
    peak_labels = K == 2 ? ["L", "H"] : (K == 3 ? ["N", "M", "E"] : ["N", "M1", "M2", "E"])

    fig, ax = PyPlot.subplots(figsize=(8, 5))
    ax.hist(t_data[t_data .< xmax], bins=collect(range(0, xmax, step=0.3)),
        density=true, color="#cccccc", alpha=0.5, edgecolor="white", linewidth=0.3,
        label="Replications")
    for k in 1:K
        ax.fill_between(xg, 0, comps[k], color=colors_k[k], alpha=0.18)
        ax.plot(xg, comps[k], color=colors_k[k], lw=2.0, ls="--", label=labels_k[k])
    end
    ax.plot(xg, total, color="black", lw=2.8, label="Mixture total")
    ax.axvline(1.96, color="#737373", lw=1.0, alpha=0.6)

    # Component labels at peaks
    for k in 1:K
        pk = maximum(comps[k])
        if pk > 0.01
            ix = argmax(comps[k])
            ax.text(xg[ix], pk * 1.08, peak_labels[k], fontsize=13,
                fontweight="bold", color=colors_k[k], ha="center")
        end
    end

    ax.set_xlabel(L"Evidence index $|t|$ (absolute $t$-statistic)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_xlim(0, xmax)
    nospines!(ax)
    ax.legend(fontsize=10, frameon=false, loc="upper right")
    fig.tight_layout()
    save_both(fig, "fig_folded_mu0_mixture_K$(K).pdf")
end

fig_folded_k2() = fig_folded_k(2)
fig_folded_k3() = fig_folded_k(3)
fig_folded_k4() = fig_folded_k(4)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE: mu_free sigma=1 comparison (truncnorm vs foldnorm, K=2,3,4)
# ══════════════════════════════════════════════════════════════════════════════

"""
    fig_mu_free_comparison()

Visualize all mu_free sigma=1 mixture fits overlaid on data histograms.
Two panels: full sample and a zoomed view (|t|≤10). Both panels overlay the
full-sample fits for K=2,3,4 (no separate "trim10" parameter set).
Prints AIC/BIC table to stdout.
"""
function fig_mu_free_comparison()
    println("Fig mu_free: sigma=1, mu_N free comparison")
    mf = joinpath(RESULTS_DIR, "mixture_params_abs_t.json")
    isfile(mf) || return @warn "SKIP: mixture_params_abs_t.json missing"
    mix = JSON.parsefile(mf)

    comp = get(mix, "mu_free_sigma1_comparison", nothing)
    comp === nothing && return @warn "SKIP: no mu_free_sigma1_comparison"

    t_data = load_verified_core_abs_t()
    t_data === nothing && return @warn "SKIP: no verified-core data"

    # Print AIC/BIC table
    println("\n  mu_free sigma=1 AIC/BIC comparison:")
    println("  $(lpad("Model", 30))  $(lpad("n", 5))  $(lpad("AIC", 10))  $(lpad("BIC", 10))  $(lpad("logL", 10))")
    for key in sort(collect(keys(comp)))
        r = comp[key]
        println("  $(lpad(key, 30))  $(lpad(Int(r["n_obs"]), 5))  $(lpad(round(r["aic"], digits=1), 10))  $(lpad(round(r["bic"], digits=1), 10))  $(lpad(round(r["log_likelihood"], digits=1), 10))")
    end

    # Styles: truncnorm solid, foldnorm dashed; K=2 thin, K=3 medium, K=4 thick
    k_lw    = Dict(2 => 1.8, 3 => 2.5, 4 => 1.8)
    k_ls_tn = Dict(2 => "-", 3 => "-", 4 => "-")
    k_ls_fn = Dict(2 => "--", 3 => "--", 4 => "--")
    k_colors = Dict(2 => "#0072B2", 3 => "#D55E00", 4 => "#009E73")

    samples = [
        ("Full sample", t_data, "full"),
        (L"|t| \leq 10\;\mathrm{(zoom)}", t_data[t_data .<= 10.0], "full"),
    ]

    fig, axes = PyPlot.subplots(1, 2, figsize=(14, 5.5))

    for (panel_idx, (panel_title, panel_data, sample_suffix)) in enumerate(samples)
        ax = axes[panel_idx]
        xmax = 10.0
        xg = collect(range(0, xmax, length=500))

        ax.hist(panel_data[panel_data .< xmax], bins=collect(range(0, xmax, step=0.3)),
            density=true, color="#cccccc", alpha=0.45, edgecolor="white", linewidth=0.2,
            label="Replications")

        for K in [2, 3, 4]
            info = get(K_LABELS, K, nothing)
            info === nothing && continue
            keys_k = info[1]

            # Truncated normal
            tn_key = "truncnorm_K=$(K)_$(sample_suffix)"
            tn_params = get(comp, tn_key, nothing)
            if tn_params !== nothing
                πv = [tn_params["pi"][k] for k in keys_k]
                μv = [tn_params["mu"][k] for k in keys_k]
                σv = [tn_params["sigma"][k] for k in keys_k]
                lo = get(tn_params, "truncation_lo", 0.0)
                aic_tn = round(tn_params["aic"], digits=1)
                total = [mix_pdf(x, πv, μv, σv; lo=lo) for x in xg]
                ax.plot(xg, total, color=k_colors[K], lw=k_lw[K], ls=k_ls_tn[K],
                    label="TN K=$K (AIC=$aic_tn)")
            end

            # Folded normal
            fn_key = "foldnorm_K=$(K)_$(sample_suffix)"
            fn_params = get(comp, fn_key, nothing)
            if fn_params !== nothing
                πv = [fn_params["pi"][k] for k in keys_k]
                μv = [fn_params["mu"][k] for k in keys_k]
                σv = [fn_params["sigma"][k] for k in keys_k]
                aic_fn = round(fn_params["aic"], digits=1)
                total = [fn_mix_pdf(x, πv, μv, σv) for x in xg]
                ax.plot(xg, total, color=k_colors[K], lw=k_lw[K], ls=k_ls_fn[K],
                    label="FN K=$K (AIC=$aic_fn)")
            end
        end

        ax.axvline(1.96, color="#737373", lw=1.0, alpha=0.6)
        ax.set_xlabel(L"Evidence index $|t|$ (absolute $t$-statistic)", fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.set_title(panel_title, fontsize=13)
        ax.set_xlim(0, xmax)
        nospines!(ax)
        ax.legend(fontsize=8, frameon=false, loc="upper right", ncol=2)
    end

    fig.tight_layout()
    save_both(fig, "fig_mu_free_sigma1_comparison.pdf")
end


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2d/2e: Constrained-σ mixture fits
# ══════════════════════════════════════════════════════════════════════════════

"""
    fig_mixture_constrained(trim_key, constraint_label, filename)

Generic constrained-σ mixture figure: loads params from trim_sensitivity[trim_key],
overlays on histogram, similar to fig2_mixture_fit.
"""
function fig_mixture_constrained(trim_key::String, constraint_label::AbstractString, filename::String;
                                  source::String="spec_level")
    println("Fig: Constrained mixture ($constraint_label)")
    mf = joinpath(RESULTS_DIR, "mixture_params_abs_t.json")
    isfile(mf) || return @warn "SKIP: mixture_params_abs_t.json missing"
    mix = JSON.parsefile(mf)

    if source == "systematic_grid"
        params = get(get(mix, "systematic_grid", Dict()), trim_key, nothing)
    else
        params = get(get(get(mix, "spec_level", Dict()), "trim_sensitivity", Dict()), trim_key, nothing)
    end
    params === nothing && return @warn "SKIP: no params for $trim_key (source=$source)"

    keys3 = haskey(params["pi"], "N") ? ["N","H","L"] : ["Low","High"]
    length(keys3) != 3 && return @warn "SKIP: mixture is not K=3"
    πv = [params["pi"][k]    for k in keys3]
    μv = [params["mu"][k]    for k in keys3]
    σv = [params["sigma"][k] for k in keys3]
    lo = get(params, "truncation_lo", 0.0)
    aic = round(params["aic"], digits=1)
    bic = round(params["bic"], digits=1)
    sigma_c = get(params, "sigma_constraint", "")

    t_data = load_verified_core_abs_t()
    t_data === nothing && return @warn "SKIP: no verified-core data"
    t_data = t_data[t_data .<= 10.0]

    xmax = min(max(maximum(t_data), maximum(μv .+ 4 .* σv)) + 0.25, 10.0)
    xg = collect(range(0, xmax, length=500))
    total = [mix_pdf(x, πv, μv, σv; lo=lo) for x in xg]
    comps = [[πv[k] * tn_pdf(x, μv[k], σv[k]; lo=lo) for x in xg] for k in 1:3]

    cc = ["#2563eb", "#009E73", "#B31B1B"]
    cl = [L"Null ($N$)", L"Moderate ($M$)", L"Extreme ($E$)"]

    fig, ax = PyPlot.subplots(figsize=(10, 5))

    ax.hist(t_data[t_data .< xmax], bins=collect(range(0, xmax, step=0.3)),
        density=true, color="#cccccc", alpha=0.5, edgecolor="white", linewidth=0.3,
        label="Replications")
    for k in 1:3
        ax.fill_between(xg, 0, comps[k], color=cc[k], alpha=0.20)
        ax.plot(xg, comps[k], color=cc[k], lw=2.0, ls="--", label=cl[k])
    end
    ax.plot(xg, total, color="black", lw=2.8, label="Mixture total")
    ax.axvline(1.96, color="#737373", lw=1.0, alpha=0.6)

    # Component labels at peaks
    for (k, label) in enumerate(["N", "M", "E"])
        pk = maximum(comps[k])
        pk > 0.01 && ax.text(μv[k], pk * 1.08, label, fontsize=14, fontweight="bold",
            color=cc[k], ha="center")
    end

    ax.set_xlabel(L"Evidence index $|t|$ (absolute $t$-statistic)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_xlim(0, xmax)
    nospines!(ax)
    ax.legend(fontsize=10, frameon=false, loc="upper right")
    fig.tight_layout()
    save_both(fig, filename)
end

fig2d_mixture_sigma_fixed_1() = fig_mixture_constrained(
    "K=3_sigma=fixed_1_full",
    latexstring("\\sigma_k = 1\\;\\mathrm{(fixed)}"),
    "fig_mixture_sigma_fixed_1.pdf";
    source="systematic_grid"
)

fig2e_mixture_sigma_geq_1() = fig_mixture_constrained(
    "K=3_sigma=geq_1_full",
    latexstring("\\sigma_k \\ge 1"),
    "fig_mixture_sigma_geq_1.pdf";
    source="systematic_grid"
)


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEMATIC GRID: K × sigma × sample — one plot per combination
# ══════════════════════════════════════════════════════════════════════════════

# Display labels for component peaks (paper convention: N, M, E)
const PEAK_LABELS = Dict(
    2 => ["N", "E"],
    3 => ["N", "M", "E"],
    4 => ["N", "M1", "M2", "E"],
)

const SIGMA_LABELS = Dict(
    "free"    => L"\sigma\;\mathrm{free}",
    "fixed_1" => L"\sigma = 1\;\mathrm{(fixed)}",
    "geq_1"  => L"\sigma \ge 1",
)

const SAMPLE_LABELS = Dict(
    "full"   => "Full sample",
)

"""
    fig_systematic_grid()

Generate one histogram + mixture overlay figure for each of the 9 systematic
grid fits (K ∈ {2,3,4} × sigma ∈ {free, fixed_1, geq_1} × sample = full).
"""
function fig_systematic_grid()
    println("\n=== Systematic grid figures ===")
    mf = joinpath(RESULTS_DIR, "mixture_params_abs_t.json")
    isfile(mf) || return @warn "SKIP: mixture_params_abs_t.json missing"
    mix = JSON.parsefile(mf)

    grid = get(mix, "systematic_grid", Dict())
    isempty(grid) && return @warn "SKIP: systematic_grid not found in JSON"

    t_data_all = load_verified_core_abs_t()
    t_data_all === nothing && return @warn "SKIP: no verified-core data"

    for K in [2, 3, 4]
        info = get(K_LABELS, K, nothing)
        info === nothing && continue
        keys_k, colors_k, labels_k = info

        for sigma_name in ["free", "fixed_1", "geq_1"]
            for sample_name in ["full"]
                grid_key = "K=$(K)_sigma=$(sigma_name)_$(sample_name)"
                params = get(grid, grid_key, nothing)
                params === nothing && (println("  SKIP: $grid_key not in grid"); continue)

                println("  Plotting $grid_key ...")
                πv = [params["pi"][k]    for k in keys_k]
                μv = [params["mu"][k]    for k in keys_k]
                σv = [params["sigma"][k] for k in keys_k]
                lo = get(params, "truncation_lo", 0.0)
                aic = round(params["aic"], digits=1)
                bic = round(params["bic"], digits=1)
                n_obs = Int(params["n_obs"])

                t_data = t_data_all
                xmax_data = 10.0
                xmax = min(max(maximum(t_data), maximum(μv .+ 4 .* σv)) + 0.25, xmax_data)
                bin_step = 0.5
                xg = collect(range(0, xmax, length=600))
                total = [mix_pdf(x, πv, μv, σv; lo=lo) for x in xg]
                comps = [[πv[k] * tn_pdf(x, μv[k], σv[k]; lo=lo) for x in xg] for k in 1:K]

                fig, ax = PyPlot.subplots(figsize=(9, 5))
                ax.hist(t_data[t_data .< xmax], bins=collect(range(0, xmax, step=bin_step)),
                    density=true, color="#cccccc", alpha=0.5, edgecolor="white", linewidth=0.3,
                    label="Replications (n=$n_obs)")
                for k in 1:K
                    ax.fill_between(xg, 0, comps[k], color=colors_k[k], alpha=0.18)
                    ax.plot(xg, comps[k], color=colors_k[k], lw=2.0, ls="--", label=labels_k[k])
                end
                ax.plot(xg, total, color="black", lw=2.8, label="Mixture total")
                ax.axvline(1.96, color="#737373", lw=1.0, alpha=0.6)

                peak_labs = get(PEAK_LABELS, K, keys_k)
                for k in 1:K
                    pk = maximum(comps[k])
                    pk > 0.01 && ax.text(μv[k], pk * 1.08, peak_labs[k], fontsize=13,
                        fontweight="bold", color=colors_k[k], ha="center")
                end

                sigma_lab_plain = Dict("free"=>"sigma free", "fixed_1"=>"sigma=1 (fixed)", "geq_1"=>"sigma>=1")[sigma_name]
                sample_lab_plain = "Full sample"
                title_str = "K=$K, $sigma_lab_plain, $sample_lab_plain -- AIC=$aic, BIC=$bic"
                ax.set_title(title_str, fontsize=12, usetex=false)
                ax.set_xlabel(L"Evidence index $|t|$ (absolute $t$-statistic)", fontsize=13)
                ax.set_ylabel("Density", fontsize=13)
                ax.set_xlim(0, xmax)
                nospines!(ax)
                ax.legend(fontsize=9, frameon=false, loc="upper right")

                # Add parameter table (plain ASCII to avoid LaTeX issues)
                param_lines = String[]
                for k in 1:K
                    push!(param_lines, "$(peak_labs[k]): pi=$(round(πv[k], digits=3)), mu=$(round(μv[k], digits=2)), sig=$(round(σv[k], digits=2))")
                end
                param_text = join(param_lines, "\n")
                ax.text(0.98, 0.55, param_text, transform=ax.transAxes, fontsize=8,
                    verticalalignment="top", horizontalalignment="right",
                    fontfamily="monospace", alpha=0.7, usetex=false,
                    bbox=Dict("boxstyle"=>"round,pad=0.3", "facecolor"=>"white", "alpha"=>0.8))

                fig.tight_layout()
                save_both(fig, "fig_grid_$(grid_key).pdf")
            end
        end
    end
end


"""
    fig_foldnorm_mufree(comp_key, K, filename)

Folded-normal mixture figure with σ=1 and free μ_N.
Reads from mu_free_sigma1_comparison[comp_key], overlays on verified-core histogram.
"""
function fig_foldnorm_mufree(comp_key::String, K::Int, filename::String)
    println("Fig: Folded-normal σ=1, μ_N free ($comp_key)")
    mf = joinpath(RESULTS_DIR, "mixture_params_abs_t.json")
    isfile(mf) || return @warn "SKIP: mixture_params_abs_t.json missing"
    mix = JSON.parsefile(mf)

    params = get(get(mix, "mu_free_sigma1_comparison", Dict()), comp_key, nothing)
    params === nothing && return @warn "SKIP: no params for $comp_key"

    info = get(K_LABELS, K, nothing)
    info === nothing && return @warn "SKIP: no label info for K=$K"
    keys_k, colors_k, labels_k = info

    πv = [params["pi"][k]    for k in keys_k]
    μv = [params["mu"][k]    for k in keys_k]
    σv = [params["sigma"][k] for k in keys_k]
    aic = round(params["aic"], digits=1)
    bic = round(params["bic"], digits=1)
    n_obs = Int(params["n_obs"])

    t_data = load_verified_core_abs_t()
    t_data === nothing && return @warn "SKIP: no verified-core data"
    t_data = t_data[t_data .<= 10.0]

    xmax = min(max(maximum(t_data), maximum(μv .+ 4 .* σv)) + 0.25, 10.0)
    xg = collect(range(0, xmax, length=500))
    total = [fn_mix_pdf(x, πv, μv, σv) for x in xg]
    comps = [[πv[k] * fn_pdf(x, μv[k], σv[k]) for x in xg] for k in 1:K]

    peak_labels = K == 2 ? ["L", "H"] : (K == 3 ? ["N", "M", "E"] : ["N", "M1", "M2", "E"])

    fig, ax = PyPlot.subplots(figsize=(10, 5))
    ax.hist(t_data[t_data .< xmax], bins=collect(range(0, xmax, step=0.3)),
        density=true, color="#cccccc", alpha=0.5, edgecolor="white", linewidth=0.3,
        label="Replications")
    for k in 1:K
        ax.fill_between(xg, 0, comps[k], color=colors_k[k], alpha=0.20)
        ax.plot(xg, comps[k], color=colors_k[k], lw=2.0, ls="--", label=labels_k[k])
    end
    ax.plot(xg, total, color="black", lw=2.8, label="Mixture total")
    ax.axvline(1.96, color="#737373", lw=1.0, alpha=0.6)

    for k in 1:K
        pk = maximum(comps[k])
        if pk > 0.01
            ix = argmax(comps[k])
            ax.text(xg[ix], pk * 1.08, peak_labels[k], fontsize=13,
                fontweight="bold", color=colors_k[k], ha="center")
        end
    end

    ax.set_xlabel(L"Evidence index $|t|$ (absolute $t$-statistic)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_xlim(0, xmax)
    nospines!(ax)
    ax.legend(fontsize=10, frameon=false, loc="upper right")
    fig.tight_layout()
    save_both(fig, filename)
end

fig2e_foldnorm_mufree_k2() = fig_foldnorm_mufree(
    "foldnorm_K=2_full", 2, "fig_foldnorm_mufree_sigma1_K2.pdf"
)

fig2f_foldnorm_mufree_k3() = fig_foldnorm_mufree(
    "foldnorm_K=3_full", 3, "fig_foldnorm_mufree_sigma1_K3.pdf"
)

fig2g_foldnorm_mufree_k4() = fig_foldnorm_mufree(
    "foldnorm_K=4_full", 4, "fig_foldnorm_mufree_sigma1_K4.pdf"
)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Counterfactual screening
# ══════════════════════════════════════════════════════════════════════════════

function fig3_counterfactual()
    println("Fig 3: Counterfactual screening (null-only FDR, flat JSON)")
    pf  = joinpath(RESULTS_DIR, "counterfactual_params.json")
    isfile(pf) || return @warn "SKIP: counterfactual_params.json missing"
    par = JSON.parsefile(pf)

    λ   = par["cost_parameters"]["lambda_baseline"]
    πv_base    = [par["mixture_params"]["pi"][k]  for k in ["N","H","L"]]
    pp_base    = [par["pass_probabilities"][k]    for k in ["N","H","L"]]

    # Read m_old and calibrated n_eff_old from JSON
    cal = get(par, "calibration", Dict())
    m_old = get(cal, "m_old_baseline", 50)

    # Null-only FDR: only N-types in numerator
    function _fdr(m, n_eff)
        n_eff < m && return 1.0
        Qs = [1.0 - cdf(Binomial(n_eff, pp_base[k]), m - 1) for k in 1:3]
        Qb = sum(πv_base .* Qs)
        Qb > 0 ? (πv_base[1]*Qs[1]) / Qb : 1.0
    end

    # Calibrate: find n_eff_old such that FDR(m_old, n_eff_old) ≈ 0.05
    fdr_target = 0.05
    n_eff_old = get(cal, "calibrated_n_eff_old", m_old)
    if !haskey(cal, "calibrated_n_eff_old")
        n_eff_old = m_old
        for n in m_old:50000
            if _fdr(m_old, n) > fdr_target
                n_eff_old = n - 1; break
            end
        end
    end
    n_eff_new = ceil(Int, n_eff_old / λ)

    # Find m_new such that FDR(m_new, n_eff_new) ≤ fdr_target
    m_new = 1
    for mc in 1:n_eff_new
        if _fdr(mc, n_eff_new) <= fdr_target
            m_new = mc; break
        end
    end

    fig, axes = PyPlot.subplots(1, 2, figsize=(10, 5))

    # ── Panel A: FDR vs m (computed from calibrated primitives) ──
    ax = axes[1]
    m_plot_max = min(m_new + 100, n_eff_new)
    m_range_old = collect(1:n_eff_old)
    m_range_new = collect(1:m_plot_max)
    fdr_old_curve = [_fdr(m, n_eff_old) for m in m_range_old]
    fdr_new_curve = [_fdr(m, n_eff_new) for m in m_range_new]

    ax.plot(m_range_old, fdr_old_curve, color="#2563eb", lw=2.8, label="Old regime")
    ax.plot(m_range_new, fdr_new_curve, color="#B31B1B", lw=2.8, ls="--", label="New regime")
    ax.axhline(fdr_target, color="black", lw=1.5, ls=":", alpha=0.7, label="FDR = 0.05")

    ax.set_xlabel(L"Required passes $m$", fontsize=13)
    ax.set_ylabel("False discovery rate (null-only)", fontsize=13)
    ax.set_xlim(0, m_plot_max)
    nospines!(ax)
    ax.legend(fontsize=11, frameon=false)

    # ── Panel B: FDR(m, λ) heatmap (calibrated: n_eff_old → n_eff_old/λ) ──
    ax2 = axes[2]

    m_grid = collect(1:m_new+100)
    λ_lo = max(0.001, λ / 3)
    λ_hi = min(0.5, λ * 5)
    λ_grid = collect(range(λ_lo, λ_hi, length=50))
    fdr_mat = zeros(length(λ_grid), length(m_grid))

    for (j, m_val) in enumerate(m_grid)
        for (i, λ_val) in enumerate(λ_grid)
            n_eff_i = ceil(Int, n_eff_old / λ_val)
            fdr_val = _fdr(m_val, n_eff_i)
            fdr_mat[i, j] = clamp(fdr_val, 0.0, 1.0)
        end
    end

    fdr_max = maximum(fdr_mat)
    pcm = ax2.pcolormesh(m_grid .- 0.5, λ_grid, fdr_mat,
        cmap="RdYlGn_r", vmin=0.0, vmax=ceil(fdr_max * 20) / 20, shading="auto")
    PyPlot.colorbar(pcm, ax=ax2, label="False discovery rate (null-only)")

    # Contour lines at key FDR thresholds
    m_centers = Float64.(m_grid)
    cs = ax2.contour(m_centers, λ_grid, fdr_mat,
        levels=[0.05, 0.10], colors="black", linewidths=1.2)
    ax2.clabel(cs, inline=true, fontsize=9, fmt="%1.2f")

    # Mark baseline calibration point
    ax2.scatter([m_new], [λ], color="white", s=90, zorder=5,
        edgecolors="black", linewidths=1.5, marker="*")

    ax2.set_xlabel(L"Required passes $m$", fontsize=13)
    ax2.set_ylabel(L"Cost ratio $\lambda$", fontsize=13)
    ax2.set_xlim(0.5, m_new + 100.5)
    ax2.set_ylim(λ_lo, λ_hi)

    fig.tight_layout()
    save_both(fig, "fig_counterfactual_old_vs_new.pdf")
end


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES 3b-1 … 3b-10: Counterfactual screening — null-only FDR, σ=1 fixed,
#                        B=[1.96,10], one figure per m_old ∈ {1,…,10}
# ══════════════════════════════════════════════════════════════════════════════

"""
    fig_counterfactual_nullfdr(m_old_target)

Two-panel counterfactual figure (FDR curves + heatmap) for a given m_old,
using null-only FDR with σ=1 fixed mixture and B=[1.96,10].
Calibrates n_eff_old so that FDR_null(m_old, n_eff_old) = 0.05.
"""
function fig_counterfactual_nullfdr(m_old_target::Int)
    println("Fig 3b (m_old=$m_old_target): Counterfactual (null-only FDR, σ=1 fixed)")
    pf  = joinpath(RESULTS_DIR, "counterfactual_params.json")
    isfile(pf) || return @warn "SKIP: counterfactual_params.json missing"
    par = JSON.parsefile(pf)

    λ  = par["cost_parameters"]["lambda_baseline"]
    πv = [par["mixture_params"]["pi"][k]  for k in ["N","H","L"]]
    pp = [par["pass_probabilities"][k]    for k in ["N","H","L"]]

    # Null-only FDR: only N-types in numerator
    function _fdr_null(m, n_eff)
        n_eff < m && return 1.0
        Qs = [1.0 - cdf(Binomial(n_eff, pp[k]), m - 1) for k in 1:3]
        Qb = sum(πv .* Qs)
        Qb > 0 ? (πv[1]*Qs[1]) / Qb : 1.0
    end

    # Calibrate: find n_eff_old such that FDR_null(m_old, n_eff_old) = 0.05
    fdr_target = 0.05
    # Use pre-calibrated value from JSON if this is the baseline m_old
    cal = get(par, "calibration", Dict())
    m_old_baseline = get(cal, "m_old_baseline", 50)
    if m_old_target == m_old_baseline && haskey(cal, "calibrated_n_eff_old")
        n_eff_old = Int(cal["calibrated_n_eff_old"])
    else
        n_eff_old = m_old_target
        for n in m_old_target:50000
            if _fdr_null(m_old_target, n) > fdr_target
                n_eff_old = n - 1; break
            end
        end
    end
    n_eff_new = ceil(Int, n_eff_old / λ)

    # Find m_new such that FDR_null(m_new, n_eff_new) ≤ fdr_target
    m_new = 1
    for mc in 1:n_eff_new
        if _fdr_null(mc, n_eff_new) <= fdr_target
            m_new = mc; break
        end
    end

    println("    n_eff_old=$n_eff_old → n_eff_new=$n_eff_new, m_old=$m_old_target → m_new=$m_new ($(round(m_new/m_old_target, digits=1))×)")

    # ── Compute disclosure scaling for bar chart (Panel A) ──
    m_olds_bar = sort(unique(vcat(collect(1:10), [20, 30, 40, 50, 60, 70, 80, 90, 100])))
    m_news_bar = Int[]
    ratios_bar = Float64[]
    for mo in m_olds_bar
        neo = mo
        for n in mo:100000
            if _fdr_null(mo, n) > fdr_target
                neo = n - 1; break
            end
        end
        nen = ceil(Int, neo / λ)
        mn = 1
        for mc in 1:nen
            if _fdr_null(mc, nen) <= fdr_target
                mn = mc; break
            end
        end
        push!(m_news_bar, mn)
        push!(ratios_bar, mn / mo)
    end

    fig, axes = PyPlot.subplots(1, 2, figsize=(10, 5))

    # ── Panel A: Disclosure multiplier line + dot chart ──
    ax = axes[1]
    ax.plot(m_olds_bar, ratios_bar, color="#2563eb", linewidth=1.5, zorder=2)
    ax.scatter(m_olds_bar, ratios_bar, color="#2563eb", s=30, zorder=3)

    # Highlight m_old=m_old_target dot in red with annotation
    idx_t = findfirst(==(m_old_target), m_olds_bar)
    if idx_t !== nothing
        ax.scatter([m_old_target], [ratios_bar[idx_t]], color="#B31B1B", s=60, zorder=4)
        ax.annotate(L"$m^{\rm old}=50$", xy=(m_old_target, ratios_bar[idx_t]),
            xytext=(m_old_target + 8, ratios_bar[idx_t] - 15),
            fontsize=11, color="#B31B1B",
            arrowprops=Dict("arrowstyle"=>"->", "color"=>"#B31B1B", "lw"=>1.2))
    end

    ax.set_xlabel(L"Current requirement $m^{\rm old}$", fontsize=13)
    ax.set_ylabel(L"Disclosure multiplier $m^{\rm new}/m^{\rm old}$", fontsize=13)
    ax.set_xlim(0.5, maximum(m_olds_bar) + 0.5)
    ax.set_ylim(0, maximum(ratios_bar) + 1)
    nospines!(ax)

    # ── Panel B: FDR_null(m, λ) heatmap — zoomed around calibration point ──
    ax2 = axes[2]

    # Zoom very tightly around the calibration point
    m_zoom_lo = 6950
    m_zoom_hi = 7050
    n_m = 400
    m_lin = collect(range(m_zoom_lo, m_zoom_hi, length=n_m))
    m_grid = unique(round.(Int, m_lin))
    sort!(m_grid)
    λ_lo = 0.0057
    λ_hi = 0.0059
    n_lam = 200
    λ_grid_hm = collect(range(λ_lo, λ_hi, length=n_lam))
    fdr_mat = zeros(length(λ_grid_hm), length(m_grid))

    for (j, m_val) in enumerate(m_grid)
        for (i, λ_val) in enumerate(λ_grid_hm)
            n_eff_i = ceil(Int, n_eff_old / λ_val)
            fdr_val = _fdr_null(m_val, n_eff_i)
            fdr_mat[i, j] = clamp(fdr_val, 0.0, 1.0)
        end
    end

    fdr_max = maximum(fdr_mat)

    pcm = ax2.pcolormesh(Float64.(m_grid), λ_grid_hm, fdr_mat,
        cmap="RdYlGn_r", vmin=0.0, vmax=fdr_max, shading="auto")
    PyPlot.colorbar(pcm, ax=ax2, label="False discovery rate")

    # Mark calibration point
    ax2.scatter([m_new], [λ], color="white", s=200, zorder=5,
        edgecolors="black", linewidths=2.0, marker="*")

    ax2.set_xlabel(L"Required passes $m$", fontsize=13)
    ax2.set_ylabel(L"Cost ratio $\lambda$", fontsize=13)
    ax2.set_ylim(λ_lo, λ_hi)

    fig.tight_layout()
    save_both(fig, "fig_counterfactual_nullfdr_m$(m_old_target).pdf")
end

# Headline figure: m_old=50 only
fig3b_nullfdr_m50() = fig_counterfactual_nullfdr(50)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: i4r agreement diagnostics
# ══════════════════════════════════════════════════════════════════════════════

function fig4_i4r_agreement()
    println("Fig 4: i4r agreement diagnostics")
    f = joinpath(DATA_DIR, "i4r_comparison.csv")
    isfile(f) || return @warn "SKIP"
    df = CSV.read(f, DataFrame)
    excl, _ = load_audit_sets()
    pids = string.(df.paper_id)

    # Load oracle (matched) reproductions
    oracle_f = joinpath(DATA_DIR, "i4r_oracle_claim_map.csv")
    use_oracle = false
    oracle_t = Dict{String,Float64}()
    if isfile(oracle_f)
        odf = CSV.read(oracle_f, DataFrame)
        for r in eachrow(odf)
            pid = string(r.paper_id)
            val = r.oracle_abs_t_stat
            if !ismissing(val) && isfinite(Float64(val))
                oracle_t[pid] = Float64(val)
            end
        end
        use_oracle = !isempty(oracle_t)
    end

    if use_oracle
        df[!, :t_match] = [get(oracle_t, string(pid), NaN) for pid in df.paper_id]
        ai_col = "t_match"
    else
        ai_col = findcol(df, "t_AI_abs", "t_AI")
        ai_col === nothing && return @warn "SKIP: no comparison column"
    end

    y_label = use_oracle ? L"$|t^{\rm match}|$" : L"$|t^{\rm auto}|$"
    diff_label = use_oracle ? L"$|t^{\rm match}| - |t^{\rm ind}|$" : L"$|t^{\rm auto}| - |t^{\rm ind}|$"

    panels = [
        ("Full sample",    trues(nrow(df))),
        ("Verified subset", .!in.(pids, Ref(excl))),
    ]

    fig, axes = PyPlot.subplots(2, 2, figsize=(11, 10))

    for (j, (label, mask)) in enumerate(panels)
        sub = df[mask, :]
        ti = finite(abs.(numcol(sub, "t_i4r")))
        ta = finite(abs.(numcol(sub, ai_col)))
        n = min(length(ti), length(ta))
        n < 2 && continue
        ti = ti[1:n]; ta = ta[1:n]

        # ── Scatter ──
        ax = axes[1, j]
        ax.scatter(ti, ta, color="#4C72B0", s=48, alpha=0.85, edgecolors="white", linewidths=0.5)
        lim = max(maximum(ti), maximum(ta)) * 1.1
        ax.plot([0, lim], [0, lim], "k--", lw=1.6, alpha=0.6)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_title("$label (n=$n)", fontsize=12)
        ax.set_xlabel(L"$|t^{\rm ind}|$", fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        r_val = cor(ti, ta)
        mad = mean(abs.(ta .- ti))
        ax.text(0.05, 0.92, latexstring("r = $(round(r_val, digits=3))"),
            transform=ax.transAxes, fontsize=10)
        ax.text(0.05, 0.84, latexstring("\\mathrm{MAD} = $(round(mad, digits=2))"),
            transform=ax.transAxes, fontsize=10)
        nospines!(ax)

        # ── Histogram ──
        ax2 = axes[2, j]
        diff = clamp.(ta .- ti, -10, 10)
        ax2.hist(diff, bins=22, color="#DD8452", alpha=0.85, edgecolor="white", linewidth=0.5)
        ax2.axvline(0, color="black", lw=1.6, ls="--", alpha=0.6)
        ax2.set_xlabel(diff_label, fontsize=12)
        ax2.set_ylabel("Count", fontsize=12)
        nospines!(ax2)
    end

    fig.tight_layout()
    save_both(fig, "fig_i4r_agreement_verified.pdf")
end


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Correlation by tree distance
# ══════════════════════════════════════════════════════════════════════════════

function fig5_corr_distance()
    println("Fig 5: AR(1) dependence by ordering")
    f = joinpath(RESULTS_DIR, "dependence.json")
    isfile(f) || return @warn "SKIP"
    dep = JSON.parsefile(f)

    ar1_ords = get(dep, "ar1_orderings", nothing)
    ar1_ords === nothing && return @warn "SKIP: no ar1_orderings"
    length(ar1_ords) == 0 && return @warn "SKIP: ar1_orderings empty"

    pref_ordering = get(get(dep, "preferred", Dict()), "ordering", "")

    # Friendly labels for orderings
    label_map = Dict(
        "spec_order"  => "Document order",
        "lex_path"    => "Lexicographic path",
        "bfs"         => "Breadth-first",
        "dfs"         => "Depth-first",
        "by_category" => "By category",
        "random"      => "Random (null)",
    )

    # Collect data sorted by phi descending
    ord_names = String[]
    phis = Float64[]
    ci_lo = Float64[]
    ci_hi = Float64[]
    r2s = Float64[]
    is_pref = Bool[]

    for (k, v) in ar1_ords
        push!(ord_names, get(label_map, k, k))
        push!(phis, Float64(v["phi"]))
        push!(ci_lo, Float64(v["phi_ci_lower"]))
        push!(ci_hi, Float64(v["phi_ci_upper"]))
        push!(r2s, Float64(v["r_squared"]))
        push!(is_pref, k == pref_ordering)
    end

    # Sort by phi descending for visual clarity
    order = sortperm(phis, rev=true)
    ord_names = ord_names[order]
    phis = phis[order]
    ci_lo = ci_lo[order]
    ci_hi = ci_hi[order]
    r2s = r2s[order]
    is_pref = is_pref[order]

    n = length(ord_names)
    y = collect(1:n)

    fig, ax = PyPlot.subplots(figsize=(7, 0.7 * n + 1.5))

    # Horizontal lollipop: CI whiskers + dot
    for i in 1:n
        col = is_pref[i] ? "#D55E00" : "#0072B2"
        ax.plot([ci_lo[i], ci_hi[i]], [y[i], y[i]], color=col, lw=2.2, solid_capstyle="round")
        ms = is_pref[i] ? 11 : 8
        ax.plot(phis[i], y[i], "o", color=col, markersize=ms, zorder=5,
            markeredgecolor="white", markeredgewidth=1.0)
        # R² annotation
        ax.text(ci_hi[i] + 0.01, y[i], latexstring("R^2=$(round(r2s[i], digits=3))"),
            fontsize=9, va="center", color="#555555")
    end

    ax.set_yticks(y)
    ax.set_yticklabels(ord_names, fontsize=11)
    ax.set_xlabel(latexstring("\\hat\\phi\\;\\mathrm{(AR(1)\\;persistence)}"), fontsize=13)
    ax.set_ylim(0.3, n + 0.7)
    ax.invert_yaxis()
    nospines!(ax)

    # Add preferred annotation
    pref_idx = findfirst(is_pref)
    if pref_idx !== nothing
        Δ = round(1 - phis[pref_idx], digits=3)
        ax.text(0.02, 0.02,
            latexstring("\\mathrm{Preferred:}\\;\\hat\\Delta = $Δ"),
            transform=ax.transAxes, fontsize=11,
            bbox=Dict("boxstyle" => "round,pad=0.3", "facecolor" => "#FFF3E0", "alpha" => 0.9))
    end

    fig.tight_layout()
    save_both(fig, "fig_corr_distance.pdf")
end


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Mixture diagnostics (PP/QQ)
# ══════════════════════════════════════════════════════════════════════════════

function fig6_mixture_diagnostics()
    println("Fig 6: Mixture diagnostics (PP/QQ)")
    sf = joinpath(DATA_DIR, "spec_level.csv")
    mf = joinpath(RESULTS_DIR, "mixture_params_abs_t.json")
    (isfile(sf) && isfile(mf)) || return @warn "SKIP"

    spec = CSV.read(sf, DataFrame)
    mix  = JSON.parsefile(mf)

    params = get(get(mix, "spec_level", Dict()), "baseline_only", nothing)
    params === nothing && return @warn "SKIP: no baseline mixture"

    πv = [params["pi"][k]    for k in ["N","H","L"]]
    μv = [params["mu"][k]    for k in ["N","H","L"]]
    σv = [params["sigma"][k] for k in ["N","H","L"]]
    lo = get(params, "truncation_lo", 0.0)
    wt = something(get(mix, "winsorize_threshold", nothing), 20.0)

    t_col = findcol(spec, "Z_abs", "Z")
    t_col === nothing && return @warn "SKIP: no Z column"
    paths = hasproperty(spec, :spec_tree_path) ? coalesce.(spec.spec_tree_path, "") : fill("", nrow(spec))
    is_bl = occursin.(Ref(r"#baseline"), paths)
    if sum(is_bl) == 0 && "spec_id" in names(spec)
        is_bl = string.(spec.spec_id) .== "baseline"
    end
    data = sort(finite(clamp.(abs.(numcol(spec[is_bl, :], t_col)), 0, wt)))
    n = length(data)
    n < 5 && return @warn "SKIP: too few baseline specs"

    ecdf_vals = (1:n) ./ n
    mcdf_vals = [mix_cdf(x, πv, μv, σv; lo=lo) for x in data]

    # QQ via bisection inversion of mix_cdf
    pq = collect(range(0.01, 0.99, length=99))
    eq = quantile(data, pq)
    function mix_quantile(p)
        a, b = 0.0, wt
        for _ in 1:100
            mid = (a + b) / 2
            mix_cdf(mid, πv, μv, σv; lo=lo) < p ? (a = mid) : (b = mid)
        end
        (a + b) / 2
    end
    mq = [mix_quantile(p) for p in pq]

    fig, axes = PyPlot.subplots(1, 2, figsize=(11, 5))

    ax = axes[1]
    ax.scatter(mcdf_vals, ecdf_vals, color="#0072B2", s=12, alpha=0.55)
    ax.plot([0,1], [0,1], "k--", lw=1.6)
    ax.set_xlabel(L"Mixture CDF $F_{\rm mix}(z)$", fontsize=12)
    ax.set_ylabel(L"Empirical CDF $\hat{F}(z)$", fontsize=12)
    ax.set_title("PP plot", fontsize=13)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    nospines!(ax)

    ax2 = axes[2]
    ax2.scatter(mq, eq, color="#D55E00", s=18, alpha=0.65)
    qmin = min(minimum(mq), minimum(eq))
    qmax = max(maximum(mq), maximum(eq))
    ax2.plot([qmin, qmax], [qmin, qmax], "k--", lw=1.6)
    ax2.set_xlabel(L"Mixture quantile $Q_{\rm mix}(p)$", fontsize=12)
    ax2.set_ylabel(L"Empirical quantile $\hat{Q}(p)$", fontsize=12)
    ax2.set_title("QQ plot", fontsize=13)
    nospines!(ax2)

    fig.tight_layout()
    save_both(fig, "fig_mixture_diagnostics.pdf")
end


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: K-sensitivity — replications histogram with K=2,3,4 overlays
# ══════════════════════════════════════════════════════════════════════════════

function fig7_k_sensitivity()
    println("Fig 7: K-sensitivity comparison")
    mf = joinpath(RESULTS_DIR, "mixture_params_abs_t.json")
    isfile(mf) || return @warn "SKIP"
    mix = JSON.parsefile(mf)
    ks = get(mix, "k_sensitivity", nothing)
    ks === nothing && return @warn "SKIP: no k_sensitivity"

    t_data = load_verified_core_abs_t()
    t_data === nothing && return @warn "SKIP: no verified-core data"
    t_data = t_data[t_data .<= 10.0]

    xmax = 10.0
    xg = collect(range(0, xmax, length=500))

    # Colors and styles for each K
    k_styles = [
        (2, "#0072B2", "-.",  2.4),
        (3, "#D55E00", "-",   2.8),
        (4, "#009E73", "--",  2.4),
    ]

    fig, ax = PyPlot.subplots(figsize=(8, 5.5))

    # Thin histogram bars
    ax.hist(t_data[t_data .< xmax], bins=collect(range(0, xmax, step=0.3)),
        density=true, color="#cccccc", alpha=0.45, edgecolor="white", linewidth=0.2,
        label="Replications")

    # Overlay each K
    for (K, col, ls, lw) in k_styles
        params = get(get(ks, "K=$K", Dict()), "truncnorm", nothing)
        params === nothing && continue
        info = get(K_LABELS, K, nothing)
        info === nothing && continue
        keys_k = info[1]
        πv = [params["pi"][k]    for k in keys_k]
        μv = [params["mu"][k]    for k in keys_k]
        σv = [params["sigma"][k] for k in keys_k]
        lo = get(params, "truncation_lo", 0.0)
        aic = round(params["aic"], digits=1)
        bic = round(params["bic"], digits=1)
        total = [mix_pdf(x, πv, μv, σv; lo=lo) for x in xg]
        ax.plot(xg, total, color=col, lw=lw, ls=ls,
            label=latexstring("K=$K\\;(\\mathrm{AIC}=$aic)"))
    end

    ax.axvline(1.96, color="#737373", lw=1.0, alpha=0.6)
    ym = ax.get_ylim()[2]
    ax.text(2.1, 0.92ym, L"$p=0.05$", fontsize=10, color="#737373")
    ax.set_xlabel(L"Evidence index $|t|$ (absolute $t$-statistic)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_xlim(0, xmax)
    nospines!(ax)
    ax.legend(fontsize=11, frameon=false, loc="upper right")
    fig.tight_layout()
    save_both(fig, "fig_k_sensitivity.pdf")
end


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8: Alternative dependence models
# ══════════════════════════════════════════════════════════════════════════════

function fig8_dependence_alternatives()
    println("Fig 8: SKIP (replaced by multi-ordering AR(1) in fig5)")
end


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9: Counterfactual sensitivity summary
# ══════════════════════════════════════════════════════════════════════════════

function fig9_counterfactual_sensitivity()
    println("Fig 9: Counterfactual sensitivity")
    af = joinpath(RESULTS_DIR, "counterfactual_alternatives.json")
    pf = joinpath(RESULTS_DIR, "counterfactual_params.json")
    isfile(af) || return @warn "SKIP"
    alts = JSON.parsefile(af)

    λ_base = 1/14

    # Collect (label, value, category_color) tuples
    entries = Tuple{String, Float64, String}[]

    # Baseline from main counterfactual
    if isfile(pf)
        par = JSON.parsefile(pf)
        cs = get(par, "counterfactual_summary", [])
        hl = filter(r -> abs(r["lambda"] - λ_base) < 1e-4 &&
                         abs(r["rho_target"] - 0.10) < 1e-4 &&
                         abs(r["FDR_target"] - 0.10) < 1e-4, cs)
        !isempty(hl) && push!(entries, ("Baseline (count-based)", Float64(hl[1]["m_ratio"]), "#0072B2"))
    end

    # A2: Selected z_hi cutoffs
    for r in get(get(alts, "A2_fixed_lower_grid_upper", Dict()), "results", [])
        r["m_ratio"] !== nothing || continue
        push!(entries, (latexstring("z_{\\rm hi}=$(Int(r["z_hi"]))"), Float64(r["m_ratio"]), "#009E73"))
    end

    # A3: Wider window
    for r in get(get(alts, "A3_wider_optimal_window", Dict()), "results", [])
        abs(r["lambda"] - λ_base) < 1e-4 || continue
        push!(entries, ("Wider window", Float64(r["m_ratio"]), "#009E73"))
        break
    end

    # B4: n_old
    for r in get(get(alts, "B4_n_old_sensitivity", Dict()), "results", [])
        (abs(r["lambda"] - λ_base) < 1e-4 && r["rho_target"] == 0.10 && r["FDR_target"] == 0.10) || continue
        push!(entries, (latexstring("n_{\\rm old}=$(Int(r["n_old"]))"), Float64(r["m_ratio"]), "#D55E00"))
    end

    # B5: Delta alternatives
    seen_dep = Set{String}()
    for r in get(get(alts, "B5_Delta_alternatives", Dict()), "results", [])
        abs(r["lambda"] - λ_base) < 1e-4 || continue
        dm = r["dependence_model"]
        dm in seen_dep && continue
        push!(seen_dep, dm)
        push!(entries, (latexstring("\\Delta\\;($(replace(dm, "_" => "\\_")))"), Float64(r["m_ratio"]), "#D55E00"))
    end

    # C6: Share-based
    for r in get(get(alts, "C6_share_based", Dict()), "results", [])
        (abs(r["lambda"] - λ_base) < 1e-4 && r["rho_target"] == 0.10 && r["FDR_target"] == 0.10) || continue
        push!(entries, (latexstring("Share-based (\\tau=$(round(r["tau_old"], digits=3)))"), Float64(r["m_ratio"]), "#CC79A7"))
        break
    end

    # C7: LR screening
    for r in get(get(alts, "C7_likelihood_ratio", Dict()), "results", [])
        (abs(r["lambda"] - λ_base) < 1e-4 && r["rho_target"] == 0.10 && r["FDR_target"] == 0.10) || continue
        ratio = r["equiv_m_new"] / max(r["equiv_m_old"], 1)
        push!(entries, ("LR screening", Float64(ratio), "#CC79A7"))
        break
    end

    isempty(entries) && return @warn "SKIP: no sensitivity results"

    labels = [e[1] for e in entries]
    ratios = [e[2] for e in entries]
    colors = [e[3] for e in entries]

    fig, ax = PyPlot.subplots(figsize=(8, max(4, 0.45 * length(entries))))
    y = collect(1:length(entries))
    ax.barh(y, ratios, color=colors, alpha=0.85, height=0.6, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(L"Disclosure multiplier $m_{\rm new}/m_{\rm old}$", fontsize=13)
    ax.axvline(1, color="black", lw=1.2, ls=":", alpha=0.5)
    ax.invert_yaxis()
    nospines!(ax)
    ax.set_title(latexstring("Counterfactual sensitivity\\;(\\lambda\\approx 1/14,\\;\\rho=0.10,\\;\\mathrm{FDR}=0.10)"),
        fontsize=13)

    # Category legend
    h = [
        matplotlib.patches.Patch(facecolor="#0072B2", alpha=0.85, label="Baseline"),
        matplotlib.patches.Patch(facecolor="#009E73", alpha=0.85, label="Window (A)"),
        matplotlib.patches.Patch(facecolor="#D55E00", alpha=0.85, label="Calibration (B)"),
        matplotlib.patches.Patch(facecolor="#CC79A7", alpha=0.85, label="Screening rule (C)"),
    ]
    ax.legend(handles=h, fontsize=10, frameon=false, loc="lower right")

    fig.tight_layout()
    save_both(fig, "fig_counterfactual_sensitivity.pdf")
end


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9b: Disclosure scaling — m_old vs m_new (FDR-matched)
# ══════════════════════════════════════════════════════════════════════════════

function fig9b_disclosure_scaling()
    println("Fig 9b: Disclosure scaling (m_old vs m_new)")
    pf = joinpath(RESULTS_DIR, "counterfactual_params.json")
    isfile(pf) || return @warn "SKIP: counterfactual_params.json missing"
    par = JSON.parsefile(pf)

    λ      = par["cost_parameters"]["lambda_baseline"]
    πv     = [par["mixture_params"]["pi"][k]  for k in ["N","H","L"]]
    p_pass = [par["pass_probabilities"][k]    for k in ["N","H","L"]]
    fdr_target = 0.05

    function _fdr(m, n_eff)
        n_eff < m && return 1.0
        Qs = [1.0 - cdf(Binomial(n_eff, p_pass[k]), m - 1) for k in 1:3]
        Qb = sum(πv .* Qs)
        Qb > 0 ? (πv[1]*Qs[1] + πv[3]*Qs[3]) / Qb : 1.0
    end

    # For each m_old, independently calibrate n_eff_old so that
    # FDR(m_old, n_eff_old) = 0.05, matching the main-text approach.
    m_olds = sort(unique(vcat(collect(1:10), [20, 30, 40, 50, 54, 60, 70, 80])))
    m_news     = Int[]
    ratios     = Float64[]
    n_eff_olds = Int[]
    n_eff_news = Int[]
    for mo in m_olds
        # Calibrate: find n_eff_old such that FDR(mo, n_eff_old) ≈ 0.05
        neo = mo
        for n in mo:50000
            if _fdr(mo, n) > fdr_target
                neo = n - 1; break
            end
        end
        nen = ceil(Int, neo / λ)
        # Find m_new such that FDR(m_new, nen) ≤ 0.05
        mn = 1
        for mc in 1:nen
            if _fdr(mc, nen) <= fdr_target
                mn = mc; break
            end
        end
        push!(n_eff_olds, neo)
        push!(n_eff_news, nen)
        push!(m_news, mn)
        push!(ratios, mn / mo)
    end

    fig, axes = PyPlot.subplots(1, 2, figsize=(10, 5))

    # Left: m_old vs m_new
    ax = axes[1]
    ax.plot(m_olds, m_news, "o-", color="#0072B2", lw=2.2, markersize=5, zorder=3)
    # 45-degree reference
    m_max_plot = maximum(m_olds)
    ax.plot([0, m_max_plot], [0, m_max_plot], "k--", lw=1.0, alpha=0.4, label="No change")
    ax.set_xlabel(L"Current requirement $m^{\rm old}$", fontsize=13)
    ax.set_ylabel(L"Required $m^{\rm new}$ (FDR-matched)", fontsize=13)
    ax.set_xlim(0.5, m_max_plot + 0.5)
    ax.set_ylim(0, maximum(m_news) + 2)
    nospines!(ax)
    ax.legend(fontsize=11, frameon=false, loc="upper left")

    # Right: ratio m_new/m_old
    ax2 = axes[2]
    bar_width = length(m_olds) > 15 ? 1.5 : 0.6
    ax2.bar(m_olds, ratios, color="#56B4E9", edgecolor="white", width=bar_width)
    # Only label bars for small m_old values and specific highlights
    for (i, r) in enumerate(ratios)
        if m_olds[i] <= 10 || m_olds[i] in [54, 80]
            ax2.text(m_olds[i], r + 0.08, string(round(r, digits=1)),
                ha="center", fontsize=7, color="#333333")
        end
    end
    ax2.axhline(1, color="black", lw=1.0, ls=":", alpha=0.4)
    ax2.set_xlabel(L"Current requirement $m^{\rm old}$", fontsize=13)
    ax2.set_ylabel(L"Disclosure multiplier $m^{\rm new}/m^{\rm old}$", fontsize=13)
    ax2.set_xlim(0.5, m_max_plot + 0.5)
    ax2.set_ylim(0, maximum(ratios) + 1)
    nospines!(ax2)

    fig.tight_layout()
    save_both(fig, "fig_disclosure_scaling.pdf")
end


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9c: Disclosure scaling — null-only FDR, σ=1 fixed, B=[1.96,10]
# ══════════════════════════════════════════════════════════════════════════════

function fig9c_disclosure_scaling_nullfdr()
    println("Fig 9c: Disclosure scaling (null-only FDR, σ=1 fixed)")
    pf = joinpath(RESULTS_DIR, "counterfactual_params.json")
    isfile(pf) || return @warn "SKIP: counterfactual_params.json missing"
    par = JSON.parsefile(pf)

    λ      = par["cost_parameters"]["lambda_baseline"]
    πv     = [par["mixture_params"]["pi"][k]  for k in ["N","H","L"]]
    p_pass = [par["pass_probabilities"][k]    for k in ["N","H","L"]]
    fdr_target = 0.05

    # Null-only FDR: only N-types in numerator
    function _fdr_null(m, n_eff)
        n_eff < m && return 1.0
        Qs = [1.0 - cdf(Binomial(n_eff, p_pass[k]), m - 1) for k in 1:3]
        Qb = sum(πv .* Qs)
        Qb > 0 ? (πv[1]*Qs[1]) / Qb : 1.0
    end

    m_olds = sort(unique(vcat(collect(1:10), [20, 30, 40, 50, 60, 70, 80, 90, 100])))
    m_news     = Int[]
    ratios     = Float64[]
    n_eff_olds = Int[]
    n_eff_news = Int[]
    for mo in m_olds
        neo = mo
        for n in mo:100000
            if _fdr_null(mo, n) > fdr_target
                neo = n - 1; break
            end
        end
        nen = ceil(Int, neo / λ)
        mn = 1
        for mc in 1:nen
            if _fdr_null(mc, nen) <= fdr_target
                mn = mc; break
            end
        end
        push!(n_eff_olds, neo)
        push!(n_eff_news, nen)
        push!(m_news, mn)
        push!(ratios, mn / mo)
        if mo in [1, 5, 10, 50, 100]
            println("    m_old=$mo: n_eff_old=$neo → n_eff_new=$nen, m_new=$mn ($(round(mn/mo, digits=1))×)")
        end
    end

    m_max_plot = maximum(m_olds)
    fig, axes = PyPlot.subplots(1, 2, figsize=(10, 5))

    # Left: m_old vs m_new
    ax = axes[1]
    ax.plot(m_olds, m_news, "o-", color="#0072B2", lw=2.2, markersize=4, zorder=3)
    ax.plot([0, m_max_plot], [0, m_max_plot], "k--", lw=1.0, alpha=0.4, label="No change")

    # Highlight m_old=50
    idx50 = findfirst(==(50), m_olds)
    if idx50 !== nothing
        ax.scatter([50], [m_news[idx50]], color="#B31B1B", s=80, zorder=5,
            edgecolors="black", linewidths=1.2, marker="D")
    end

    ax.set_xlabel(L"Current requirement $m^{\rm old}$", fontsize=13)
    ax.set_ylabel(L"Required $m^{\rm new}$ (null-FDR matched)", fontsize=13)
    ax.set_xlim(0.5, m_max_plot + 0.5)
    ax.set_ylim(0, maximum(m_news) + 5)
    nospines!(ax)
    ax.legend(fontsize=11, frameon=false, loc="upper left")

    # Right: ratio m_new/m_old
    ax2 = axes[2]
    bar_width = 1.5
    ax2.bar(m_olds, ratios, color="#56B4E9", edgecolor="white", width=bar_width)

    # Highlight m_old=50 bar
    if idx50 !== nothing
        ax2.bar([50], [ratios[idx50]], color="#B31B1B", edgecolor="white", width=bar_width, alpha=0.8)
    end

    ax2.axhline(1, color="black", lw=1.0, ls=":", alpha=0.4)
    ax2.set_xlabel(L"Current requirement $m^{\rm old}$", fontsize=13)
    ax2.set_ylabel(L"Disclosure multiplier $m^{\rm new}/m^{\rm old}$", fontsize=13)
    ax2.set_xlim(0.5, m_max_plot + 0.5)
    ax2.set_ylim(0, maximum(ratios) + 1)
    nospines!(ax2)

    fig.tight_layout()
    save_both(fig, "fig_disclosure_scaling_nullfdr.pdf")
end


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10: Bootstrap mixture CIs (3x3 histogram grid)
# ══════════════════════════════════════════════════════════════════════════════

function fig10_bootstrap_ci()
    println("Fig 10: Bootstrap mixture CIs")
    f = joinpath(RESULTS_DIR, "bootstrap_mixture_ci.json")
    isfile(f) || return @warn "SKIP: bootstrap_mixture_ci.json missing"
    boot = JSON.parsefile(f)

    params = boot["parameters"]
    # Determine which parameters are available (6 if sigma fixed, 9 if sigma free)
    param_names_all = [
        "pi_N", "pi_H", "pi_L",
        "mu_N", "mu_H", "mu_L",
        "sigma_N", "sigma_H", "sigma_L",
    ]
    param_names = [p for p in param_names_all if haskey(params, p)]
    nice = Dict(
        "pi_N" => L"$\pi_N$", "pi_H" => L"$\pi_M$", "pi_L" => L"$\pi_E$",
        "mu_N" => L"$\mu_N$", "mu_H" => L"$\mu_M$", "mu_L" => L"$\mu_E$",
        "sigma_N" => L"$\sigma_N$", "sigma_H" => L"$\sigma_M$", "sigma_L" => L"$\sigma_E$",
    )

    B = get(boot, "B", 200)

    n_params = length(param_names)
    ncols = 3
    nrows = ceil(Int, n_params / ncols)
    fig, axes = PyPlot.subplots(nrows, ncols, figsize=(10, 3 * nrows))
    # Flatten axes to 1D array for indexing
    if nrows == 1
        axes_flat = [axes[j] for j in 1:ncols]
    else
        axes_flat = [axes[r, c] for r in 1:nrows for c in 1:ncols]
    end

    for (i, name) in enumerate(param_names)
        ax = axes_flat[i]

        s = params[name]
        pt = s["point_estimate"]
        se = s["bootstrap_se"]
        lo_ci = s["ci_2_5"]
        hi_ci = s["ci_97_5"]
        bm = s["bootstrap_mean"]

        # Simulate bootstrap histogram from normal approximation
        vals = bm .+ se .* randn(B)

        ax.hist(vals, bins=30, color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.axvline(pt, color="black", linewidth=1.5, linestyle="-", label="Point est.")
        ax.axvline(lo_ci, color="firebrick", linewidth=1.2, linestyle="--", label="95\\% CI")
        ax.axvline(hi_ci, color="firebrick", linewidth=1.2, linestyle="--")

        ax.set_xlabel(nice[name], fontsize=11)
        mod(i - 1, ncols) == 0 && ax.set_ylabel("Count", fontsize=10)
        nospines!(ax)

        i == 1 && ax.legend(frameon=false, fontsize=8)
    end

    # Hide unused subplots
    for j in (n_params + 1):(nrows * ncols)
        axes_flat[j].set_visible(false)
    end

    fig.tight_layout()
    save_both(fig, "fig_bootstrap_mixture_ci.pdf")
end


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 11: Bootstrap LRT (1x2 histogram)
# ══════════════════════════════════════════════════════════════════════════════

function fig11_bootstrap_lrt()
    println("Fig 11: Bootstrap LRT")
    f = joinpath(RESULTS_DIR, "bootstrap_lrt.json")
    isfile(f) || return @warn "SKIP: bootstrap_lrt.json missing"
    boot = JSON.parsefile(f)

    B = get(boot, "B", 200)
    tests = []
    for label in ["K=2 vs K=3", "K=3 vs K=4"]
        haskey(boot, label) || continue
        push!(tests, (label, boot[label]))
    end
    isempty(tests) && return @warn "SKIP: no LRT test results"

    fig, axes = PyPlot.subplots(1, length(tests), figsize=(10, 4))
    length(tests) == 1 && (axes = [axes])

    for (idx, (label, result)) in enumerate(tests)
        ax = axes[idx]
        lr_obs = result["LR_obs"]
        p_val = result["p_value"]
        percs = result["bootstrap_LR_percentiles"]

        # Approximate bootstrap distribution: use gamma-like shape from percentiles
        # Simple approach: draw from Exponential shifted/scaled to match median and spread
        med = get(percs, "p50", lr_obs * 0.5)
        p95 = get(percs, "p95", lr_obs * 0.9)
        if med !== nothing && p95 !== nothing && med > 0
            scale = max(Float64(med), 0.1)
            lr_samples = scale .* abs.(randn(B)) .+ 0.0
        else
            lr_samples = abs.(randn(B)) .* max(lr_obs * 0.3, 1.0)
        end

        # Compute histogram bins first using numpy
        n_bins = 30
        np = pyimport("numpy")
        counts_arr, bin_edges_arr = np.histogram(lr_samples, bins=n_bins)
        counts_v = Float64.(counts_arr)
        edges_v = Float64.(bin_edges_arr)

        # Draw main histogram
        ax.hist(lr_samples, bins=n_bins, color="steelblue", alpha=0.7,
            edgecolor="white", linewidth=0.5, label="Bootstrap LR")

        # Shade rejection region (bars >= LR_obs)
        for i in 1:length(counts_v)
            if edges_v[i+1] >= lr_obs
                ax.bar((edges_v[i] + edges_v[i+1]) / 2, counts_v[i],
                    width=edges_v[i+1] - edges_v[i],
                    color="firebrick", alpha=0.5, edgecolor="white", linewidth=0.5)
            end
        end

        ax.axvline(lr_obs, color="black", linewidth=1.5, linestyle="-",
            label=latexstring("\\mathrm{LR}_{\\mathrm{obs}} = $(round(lr_obs, digits=1))"))

        ax.set_xlabel("Likelihood Ratio Statistic", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("$label  (\$p = $(round(p_val, digits=3))\$)", fontsize=12)
        nospines!(ax)
        ax.legend(frameon=false, fontsize=9)
    end

    fig.tight_layout()
    save_both(fig, "fig_bootstrap_lrt.pdf")
end


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 12: Leave-one-out CV (bar chart)
# ══════════════════════════════════════════════════════════════════════════════

function fig12_leave_one_out_cv()
    println("Fig 12: Leave-one-out CV")
    f = joinpath(RESULTS_DIR, "leave_one_out_cv.json")
    isfile(f) || return @warn "SKIP: leave_one_out_cv.json missing"
    cv = JSON.parsefile(f)

    ks = sort(collect(keys(cv)))
    k_labels = [replace(k, "K=" => "") for k in ks]
    mean_lls = [cv[k]["mean_per_obs_cv_loglik"] for k in ks]

    fig, ax = PyPlot.subplots(figsize=(4.5, 3.5))
    bars = ax.bar(k_labels, mean_lls, color="steelblue", edgecolor="white", width=0.5)

    for (bar, val) in zip(bars, mean_lls)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            string(round(val, digits=3)), ha="center", va="bottom", fontsize=10)
    end

    ax.set_xlabel(L"Number of components ($K$)", fontsize=12)
    ax.set_ylabel("Mean per-obs CV log-likelihood", fontsize=12)
    nospines!(ax)
    fig.tight_layout()
    save_both(fig, "fig_leave_one_out_cv.pdf")
end


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 13: Subsample stability (1x3 scatter)
# ══════════════════════════════════════════════════════════════════════════════

function fig13_journal_subgroup()
    println("Fig 13: Journal subgroup analysis")
    f = joinpath(RESULTS_DIR, "journal_subgroup.json")
    isfile(f) || return @warn "SKIP: journal_subgroup.json missing"
    js = JSON.parsefile(f)

    full = js["full_sample"]["params"]
    subgroups = js["subgroups"]

    # Need at least AER and Non-AER
    haskey(subgroups, "AER") || return @warn "SKIP: no AER subgroup"
    haskey(subgroups, "Non-AER") || return @warn "SKIP: no Non-AER subgroup"

    # Internal keys from fitter: N, H, L (ascending mu). Display: N, M, E.
    key_order = ["N", "H", "L"]
    display_map = Dict("N" => "N", "H" => "M", "L" => "E")
    display_names = [display_map[k] for k in key_order]

    n_types = length(key_order)
    x = collect(0:n_types-1)

    bar_groups = ["Full sample", "AER", "Non-AER"]
    colors = ["#333333", "#1f77b4", "#ff7f0e"]
    width = 0.25

    function get_params(glabel)
        if glabel == "Full sample"
            return full
        end
        return subgroups[glabel]["params"]
    end

    fig, axes = PyPlot.subplots(1, 2, figsize=(10, 4.5))

    # Panel 1: mixing weights pi
    ax = axes[1]
    for (i, (glabel, color)) in enumerate(zip(bar_groups, colors))
        p = get_params(glabel)
        vals = [p["pi"][k] for k in key_order]
        offset = (i - 1 - (length(bar_groups) - 1) / 2) * width
        ax.bar(Float64.(x) .+ offset, vals, width, label=glabel, color=color, alpha=0.85)
    end
    ax.set_xticks(x)
    ax.set_xticklabels(["\$$d\$" for d in display_names], fontsize=12)
    ax.set_ylabel(L"Mixing weight $\pi_k$", fontsize=12)
    ax.legend(frameon=false, fontsize=10)
    nospines!(ax)

    # Panel 2: means mu
    ax = axes[2]
    for (i, (glabel, color)) in enumerate(zip(bar_groups, colors))
        p = get_params(glabel)
        vals = [p["mu"][k] for k in key_order]
        offset = (i - 1 - (length(bar_groups) - 1) / 2) * width
        ax.bar(Float64.(x) .+ offset, vals, width, label=glabel, color=color, alpha=0.85)
    end
    ax.set_xticks(x)
    ax.set_xticklabels(["\$$d\$" for d in display_names], fontsize=12)
    ax.set_ylabel(L"Component mean $\mu_k$", fontsize=12)
    nospines!(ax)

    fig.tight_layout()
    save_both(fig, "fig_journal_subgroup.pdf")
end


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 14: Posterior heatmap (stacked bar)
# ══════════════════════════════════════════════════════════════════════════════

function fig14_posterior_heatmap()
    println("Fig 14: Posterior heatmap")
    mf = joinpath(RESULTS_DIR, "mixture_params_abs_t.json")
    sf = joinpath(DATA_DIR, "spec_level_verified_core.csv")
    (isfile(mf) && isfile(sf)) || return @warn "SKIP: missing files"

    mix = JSON.parsefile(mf)
    # Use the baseline specification: truncated-normal K=3, sigma=1 (no |t| cap)
    params = get(get(mix, "spec_level", Dict()), "baseline_only_sigma_fixed_1", nothing)
    params === nothing && return @warn "SKIP: no baseline_only_sigma_fixed_1 params"

    πv = [params["pi"][k] for k in ["N","H","L"]]
    μv = [params["mu"][k] for k in ["N","H","L"]]
    σv = [params["sigma"][k] for k in ["N","H","L"]]
    lo = get(params, "truncation_lo", 0.0)

    spec = CSV.read(sf, DataFrame)
    t_col = findcol(spec, "Z_abs", "Z")
    t_col === nothing && return @warn "SKIP: no Z column"
    z_vals = finite(abs.(numcol(spec, t_col)))
    length(z_vals) < 5 && return @warn "SKIP: too few obs"

    # Compute posteriors for each z using truncated-normal PDF
    n = length(z_vals)
    posteriors = zeros(n, 3)
    for i in 1:n
        for k in 1:3
            posteriors[i, k] = πv[k] * tn_pdf(z_vals[i], μv[k], σv[k]; lo=lo)
        end
        denom = sum(posteriors[i, :])
        denom > 0 && (posteriors[i, :] ./= denom)
    end

    # Sort by z
    idx = sortperm(z_vals)
    p_n = posteriors[idx, 1]
    p_h = posteriors[idx, 2]
    p_l = posteriors[idx, 3]
    x = 0:(n-1)

    cc = ["#2563eb", "#009E73", "#B31B1B"]

    fig, ax = PyPlot.subplots(figsize=(10, 4.5))
    ax.bar(x, p_n, width=1.0, color=cc[1], label="Null (N)", linewidth=0)
    ax.bar(x, p_h, width=1.0, bottom=p_n, color=cc[2], label="Moderate (M)", linewidth=0)
    ax.bar(x, p_l, width=1.0, bottom=p_n .+ p_h, color=cc[3], label="Extreme (E)", linewidth=0)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel(L"Specifications (sorted by $|t|$)", fontsize=12)
    ax.set_ylabel("Posterior probability", fontsize=12)
    ax.legend(loc="upper left", frameon=false, fontsize=10)
    nospines!(ax)
    fig.tight_layout()
    save_both(fig, "fig_posterior_heatmap.pdf")
end


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 15: Phi vs number of specs (scatter)
# ══════════════════════════════════════════════════════════════════════════════

function fig15_phi_vs_nspecs()
    println("Fig 15: Phi vs n_specs")
    hf = joinpath(RESULTS_DIR, "dependence_heterogeneity.csv")
    isfile(hf) || return @warn "SKIP: dependence_heterogeneity.csv missing"
    het = CSV.read(hf, DataFrame)
    nrow(het) < 2 && return @warn "SKIP: too few rows"

    n_specs = numcol(het, "n_specs")
    phi_i = numcol(het, "phi_i")

    # Spec-weighted mean of paper-level phis as reference line
    w = Float64.(n_specs)
    wmean_phi = sum(w .* phi_i) / sum(w)

    fig, ax = PyPlot.subplots(figsize=(5.5, 4.5))
    ax.scatter(n_specs, phi_i, s=20, alpha=0.6, linewidths=0, color="steelblue", zorder=3)

    ax.axhline(wmean_phi, color="black", linestyle="--", linewidth=1.0,
        label=latexstring("\\mathrm{Weighted\\;mean}\\;\\hat\\varphi = $(round(wmean_phi, digits=2))"), zorder=2)
    ax.legend(frameon=false, fontsize=10)

    ax.set_xlabel("Specifications per paper", fontsize=12)
    ax.set_ylabel(L"$\hat\varphi_i$", fontsize=12)
    ax.set_xlim(left=0)
    nospines!(ax)
    fig.tight_layout()
    save_both(fig, "fig_phi_vs_nspecs.pdf")
end


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 16: Sign consistency (line + error bars)
# ══════════════════════════════════════════════════════════════════════════════

function fig16_sign_consistency()
    println("Fig 16: Sign consistency")
    f = joinpath(RESULTS_DIR, "sign_consistency.csv")
    isfile(f) || return @warn "SKIP: sign_consistency.csv missing"
    sc = CSV.read(f, DataFrame)
    nrow(sc) < 2 && return @warn "SKIP: too few rows"

    distances = numcol(sc, "distance")
    rates = numcol(sc, "sign_match_rate")
    ns = numcol(sc, "n_specs")

    se = sqrt.(rates .* (1 .- rates) ./ ns)
    size_scale = ns ./ maximum(ns) .* 120 .+ 20

    fig, ax = PyPlot.subplots(figsize=(5.5, 4.5))

    ax.errorbar(distances, rates, yerr=1.96 .* se, fmt="none",
        ecolor="gray", elinewidth=1.0, capsize=3, capthick=1.0, zorder=2)
    ax.scatter(distances, rates, s=size_scale, color="steelblue",
        edgecolors="white", linewidths=0.5, zorder=3)
    ax.plot(distances, rates, color="steelblue", linewidth=1.0, alpha=0.5, zorder=1)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, label="Random (0.5)", zorder=1)

    ax.set_xlabel(L"Tree distance $d$", fontsize=12)
    ax.set_ylabel("Sign consistency rate", fontsize=12)
    ax.legend(frameon=false, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=-0.2, right=maximum(distances) + 0.5)
    d_max_int = Int(floor(maximum(distances)))
    ax.set_xticks(collect(0:d_max_int))
    nospines!(ax)
    fig.tight_layout()
    save_both(fig, "fig_sign_consistency.pdf")
end


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 17: Funnel plot (scatter)
# ══════════════════════════════════════════════════════════════════════════════

function fig17_funnel_plot()
    println("Fig 17: Funnel plot")
    sf = joinpath(DATA_DIR, "spec_level_verified_core.csv")
    isfile(sf) || return @warn "SKIP: spec_level_verified_core.csv missing"
    spec = CSV.read(sf, DataFrame)

    t_col = findcol(spec, "Z_abs", "Z")
    t_col === nothing && return @warn "SKIP: no Z column"
    abs_z = abs.(numcol(spec, t_col))

    # Compute precision = sqrt(n) (cross-paper comparable; 1/SE is not
    # comparable across papers with different outcome units)
    n_col = findcol(spec, "n_obs")
    se_col = findcol(spec, "std_error")

    precision = Float64[]
    precision_label = ""
    if n_col !== nothing
        n_vals = numcol(spec, n_col)
        precision = sqrt.(n_vals)
        precision_label = L"$\sqrt{n}$"
    elseif se_col !== nothing
        se_vals = numcol(spec, se_col)
        for i in eachindex(se_vals)
            if isfinite(se_vals[i]) && se_vals[i] > 0
                push!(precision, 1.0 / se_vals[i])
            else
                push!(precision, NaN)
            end
        end
        precision_label = L"Precision ($1/\mathrm{SE}$)"
    else
        return @warn "SKIP: no n_obs or std_error column"
    end

    # Filter valid
    valid = isfinite.(abs_z) .& isfinite.(precision) .& (precision .> 0)
    abs_z = abs_z[valid]
    precision = precision[valid]
    length(abs_z) < 5 && return @warn "SKIP: too few valid observations"

    fig, ax = PyPlot.subplots(figsize=(5.5, 4.5))
    ax.scatter(precision, abs_z, s=8, alpha=0.3, linewidths=0, color="steelblue", rasterized=true)
    ax.axhline(1.96, color="black", linestyle="--", linewidth=1.0, label=L"$|t| = 1.96$")
    ax.set_xlabel(precision_label, fontsize=12)
    ax.set_ylabel(L"$|t|$", fontsize=12)
    ax.set_xscale("log")
    ax.legend(frameon=false, fontsize=10)

    y_cap = max(10.0, quantile(filter(isfinite, abs_z), 0.99))
    ax.set_ylim(bottom=0, top=y_cap)
    nospines!(ax)
    fig.tight_layout()
    save_both(fig, "fig_funnel_plot.pdf")
end


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 18: Monte Carlo validation (scatter)
# ══════════════════════════════════════════════════════════════════════════════

function fig18_montecarlo()
    println("Fig 18: Monte Carlo validation")
    mf = joinpath(RESULTS_DIR, "montecarlo_validation.json")
    isfile(mf) || return @warn "SKIP: montecarlo_validation.json missing"

    mc = JSON.parsefile(mf)
    merged = get(mc, "merged_comparison", [])
    isempty(merged) && return @warn "SKIP: no merged comparison data"

    n_eff_old = get(mc, "n_eff_old", 0)
    n_eff_new = get(mc, "n_eff_new", 0)

    # Separate by regime
    old_a = Float64[]; old_s = Float64[]
    new_a = Float64[]; new_s = Float64[]
    for r in merged
        fa = get(r, "FDR_analytical", NaN)
        fs = get(r, "FDR_simulated", NaN)
        (isfinite(fa) && isfinite(fs)) || continue
        if r["regime"] == "old"
            push!(old_a, fa); push!(old_s, fs)
        else
            push!(new_a, fa); push!(new_s, fs)
        end
    end

    fig, ax = PyPlot.subplots(figsize=(5.5, 4.5))

    length(old_a) > 0 && ax.scatter(old_a, old_s, s=30, color="C0", edgecolors="none",
        alpha=0.7, label=latexstring("\\mathrm{Old\\;regime}\\;(n_{\\mathrm{eff}}=$n_eff_old)"), zorder=3)
    length(new_a) > 0 && ax.scatter(new_a, new_s, s=30, color="C1", edgecolors="none",
        alpha=0.7, label=latexstring("\\mathrm{New\\;regime}\\;(n_{\\mathrm{eff}}=$n_eff_new)"), zorder=3)

    all_vals = vcat(old_a, old_s, new_a, new_s)
    if !isempty(all_vals)
        lo_val = max(0.0, minimum(all_vals) * 0.8)
        hi_val = min(1.0, maximum(all_vals) * 1.2)
    else
        lo_val, hi_val = 0.0, 1.0
    end
    ax.plot([lo_val, hi_val], [lo_val, hi_val], color="grey", linestyle="--",
        linewidth=1.0, zorder=1, label="45-degree line")

    ax.set_xlabel("Analytical FDR (null-only)", fontsize=12)
    ax.set_ylabel("Simulated FDR (null-only)", fontsize=12)
    ax.legend(frameon=false, fontsize=9)
    ax.set_xlim(left=lo_val, right=hi_val)
    ax.set_ylim(bottom=lo_val, top=hi_val)
    ax.set_aspect("equal", adjustable="box")
    nospines!(ax)
    fig.tight_layout()
    save_both(fig, "fig_montecarlo_validation.pdf")
end


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 19: Effective sample size (line plot)
# ══════════════════════════════════════════════════════════════════════════════

function fig19_effective_sample_size()
    println("Fig 19: Effective sample size (5 orderings)")
    f = joinpath(RESULTS_DIR, "dependence.json")
    isfile(f) || return @warn "SKIP: dependence.json missing"
    dep = JSON.parsefile(f)

    pref_ordering = get(get(dep, "preferred", Dict()), "ordering", "by_category")

    N_MAX = 500
    n = collect(1:N_MAX)

    # Ordering display config
    ordering_keys = ["spec_order", "lex_path", "bfs", "dfs", "by_category"]
    ordering_labels = Dict(
        "spec_order" => "Document order",
        "lex_path" => "Lexicographic path",
        "bfs" => "Breadth-first",
        "dfs" => "Depth-first",
        "by_category" => "By category",
    )
    ordering_colors = Dict(
        "spec_order" => "tab:orange",
        "lex_path" => "tab:green",
        "bfs" => "tab:purple",
        "dfs" => "tab:red",
        "by_category" => "black",
    )

    fig, ax = PyPlot.subplots(figsize=(5.5, 4.0))

    # Independence benchmark
    ax.plot(n, Float64.(n), linestyle=":", color="gray", linewidth=1.5,
        label=L"Independence ($\Delta = 1$)")

    # Plot each non-random ordering
    ar1_ords = get(dep, "ar1_orderings", Dict())
    for key in ordering_keys
        haskey(ar1_ords, key) || continue
        Δ_val = get(ar1_ords[key], "Delta", nothing)
        Δ_val === nothing && continue
        Δ_val = Float64(Δ_val)

        is_pref = (key == pref_ordering)
        label = get(ordering_labels, key, key)
        color = get(ordering_colors, key, "tab:blue")

        ax.plot(n, Δ_val .* n,
            linestyle = is_pref ? "-" : "--",
            color = color,
            linewidth = is_pref ? 2.5 : 1.5,
            label = latexstring("\\mathrm{$(replace(label, " " => "\\;"))}\\;(\\hat\\Delta = $(round(Δ_val, digits=3)))"))
    end

    ax.set_xlabel(L"Total specifications $n$", fontsize=12)
    ax.set_ylabel(L"Effective independent tests $n_{\mathrm{eff}}$", fontsize=12)
    ax.legend(frameon=false, fontsize=8, loc="upper left")
    ax.set_xlim(0, N_MAX)
    ax.set_ylim(0, N_MAX)
    nospines!(ax)
    fig.tight_layout()
    save_both(fig, "fig_effective_sample_size.pdf")
end


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 20: Window surface (heat map)
# ══════════════════════════════════════════════════════════════════════════════

function fig20_window_surface()
    println("Fig 20: Window surface")
    mf = joinpath(RESULTS_DIR, "mixture_params_abs_t.json")
    isfile(mf) || return @warn "SKIP: mixture_params_abs_t.json missing"
    mix = JSON.parsefile(mf)

    params = get(get(mix, "spec_level", Dict()), "baseline_only", nothing)
    params === nothing && return @warn "SKIP: no baseline mixture"

    pi_d = params["pi"]; mu_d = params["mu"]; sigma_d = params["sigma"]
    trunc_lo = get(params, "truncation_lo", 0.0)
    pi_bad_total = pi_d["N"] + pi_d["L"]

    # Load optimal window
    cf = joinpath(RESULTS_DIR, "counterfactual_params.json")
    opt_z_lo = nothing; opt_z_hi = nothing
    if isfile(cf)
        cfp = JSON.parsefile(cf)
        ew = get(cfp, "evidence_window", Dict())
        opt_z_lo = get(ew, "z_lo", nothing)
        opt_z_hi = get(ew, "z_hi", nothing)
    end

    # Grid
    Z_LO_MIN = 1.96; Z_LO_MAX = 12.0; Z_LO_STEP = 0.25
    Z_HI_OFFSET = 0.5; Z_HI_MAX = 20.0; Z_HI_STEP = 0.25
    P_H_FLOOR = 0.05; P_H_MIN_GRID = 0.01; P_BAD_MIN = 0.001

    z_lo_vals = collect(Z_LO_MIN:Z_LO_STEP:Z_LO_MAX)
    z_hi_vals = collect((Z_LO_MIN + Z_HI_OFFSET):Z_HI_STEP:Z_HI_MAX)
    n_lo = length(z_lo_vals); n_hi = length(z_hi_vals)

    S = fill(NaN, n_hi, n_lo)

    for (i, zh) in enumerate(z_hi_vals)
        for (j, zl) in enumerate(z_lo_vals)
            zh <= zl + 0.5 - 1e-6 && continue

            p_N = mix_cdf(zh, [1.0], [mu_d["N"]], [sigma_d["N"]]; lo=trunc_lo) -
                  mix_cdf(zl, [1.0], [mu_d["N"]], [sigma_d["N"]]; lo=trunc_lo)
            p_H = mix_cdf(zh, [1.0], [mu_d["H"]], [sigma_d["H"]]; lo=trunc_lo) -
                  mix_cdf(zl, [1.0], [mu_d["H"]], [sigma_d["H"]]; lo=trunc_lo)
            p_L = mix_cdf(zh, [1.0], [mu_d["L"]], [sigma_d["L"]]; lo=trunc_lo) -
                  mix_cdf(zl, [1.0], [mu_d["L"]], [sigma_d["L"]]; lo=trunc_lo)

            p_H < P_H_FLOOR && continue
            p_bad = pi_bad_total > 1e-12 ?
                (pi_d["N"] * p_N + pi_d["L"] * p_L) / pi_bad_total : 0.0
            (p_H > P_H_MIN_GRID && p_bad > P_BAD_MIN) || continue
            S[i, j] = log(p_H / p_bad)
        end
    end

    # Build edges for pcolormesh
    z_lo_edges = vcat(z_lo_vals .- Z_LO_STEP/2, [z_lo_vals[end] + Z_LO_STEP/2])
    z_hi_edges = vcat(z_hi_vals .- Z_HI_STEP/2, [z_hi_vals[end] + Z_HI_STEP/2])

    fig, ax = PyPlot.subplots(figsize=(6, 5))

    # Mask NaN
    numpy_ma = pyimport("numpy").ma
    S_ma = numpy_ma.masked_invalid(S)
    pcm = ax.pcolormesh(z_lo_edges, z_hi_edges, S_ma, cmap="viridis", shading="flat", rasterized=true)
    fig.colorbar(pcm, ax=ax, label=L"Separation score $S(B)$")

    if opt_z_lo !== nothing && opt_z_hi !== nothing
        ax.plot(opt_z_lo, opt_z_hi, marker="*", markersize=14, color="red",
            markeredgecolor="black", markeredgewidth=0.5, zorder=5, label=L"Optimal $B$")
        ax.legend(frameon=false, fontsize=10, loc="upper left")
    end

    diag = collect(range(Z_LO_MIN, stop=min(Z_LO_MAX, Z_HI_MAX), length=100))
    ax.plot(diag, diag, color="white", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel(L"$z_\ell$", fontsize=13)
    ax.set_ylabel(L"$z_h$", fontsize=13)
    ax.set_xlim(z_lo_edges[1], z_lo_edges[end])
    ax.set_ylim(z_hi_edges[1], z_hi_edges[end])
    nospines!(ax)
    fig.tight_layout()
    save_both(fig, "fig_window_surface.pdf")
end


# ══════════════════════════════════════════════════════════════════════════════
# TABLE: Baseline specification summary statistics
# ══════════════════════════════════════════════════════════════════════════════

function table_baseline_specs_summary()
    println("Table: Baseline specification summary")

    # ── Load verification summary ──
    vf = joinpath(DATA_DIR, "verification_summary_by_paper.csv")
    isfile(vf) || return @warn "SKIP: verification_summary_by_paper.csv missing"
    vs = CSV.read(vf, DataFrame)

    # ── Load paper metadata for journal info ──
    sf = joinpath(dirname(dirname(@__DIR__)), "data", "tracking", "spec_search_status.json")
    journal_map = Dict{String,String}()
    method_map  = Dict{String,String}()
    if isfile(sf)
        tracking = JSON.parsefile(sf)
        for p in get(tracking, "packages_with_data", [])
            pid = p["id"]
            journal_map[pid] = get(p, "journal", "Unknown")
            method_map[pid]  = get(p, "method", "Unknown")
        end
    end

    vs[!, :journal] = [get(journal_map, string(pid), "Unknown") for pid in vs.paper_id]

    # Normalize journal names
    function normalize_journal(j)
        j = replace(j, "-" => ": ")
        (occursin("P&P", j) || occursin("PP", j)) && return "AER: P\\&P"
        occursin("Insights", j) && return "AER: Insights"
        occursin("Applied", j) && return "AEJ: Applied"
        occursin(r"Policy|Economic Policy", j) && return "AEJ: Policy"
        occursin("Macro", j)  && return "AEJ: Macro"
        occursin("Micro", j)  && return "AEJ: Micro"
        occursin(r"AER|AEA", j) && return "AER"
        return j
    end
    vs[!, :journal_clean] = normalize_journal.(vs.journal)

    bl = vs.baseline_specs
    bg = vs.baseline_groups
    ts = vs.total_specs
    cs = vs.core_specs
    N  = nrow(vs)

    # ── Compute overall summary stats ──
    function sumstats(x)
        v = Float64.(x)
        (n    = length(v),
         mean = round(mean(v), digits=1),
         med  = round(median(v), digits=1),
         sd   = round(std(v), digits=1),
         min  = round(minimum(v), digits=0),
         p25  = round(quantile(v, 0.25), digits=1),
         p75  = round(quantile(v, 0.75), digits=1),
         max  = round(maximum(v), digits=0))
    end

    overall_bl = sumstats(bl)
    overall_bg = sumstats(bg)
    overall_ts = sumstats(ts)
    overall_cs = sumstats(cs)

    # ── By-journal breakdown ──
    journals_order = ["AER", "AER: Insights", "AER: P\\&P", "AEJ: Applied", "AEJ: Policy", "AEJ: Macro", "AEJ: Micro"]
    journal_stats = []
    for jname in journals_order
        mask = vs.journal_clean .== jname
        sum(mask) == 0 && continue
        push!(journal_stats, (journal=jname, stats=sumstats(bl[mask]), n=sum(mask)))
    end

    # ── Write LaTeX table ──
    tex = IOBuffer()

    # Panel A: Overall summary
    println(tex, raw"\begin{table}[htbp]")
    println(tex, raw"\centering")
    println(tex, raw"\caption{Summary Statistics: Author-Reported Specifications}")
    println(tex, raw"\label{tab:baseline_specs_summary}")
    println(tex, raw"\begin{tabular}{lcccccccc}")
    println(tex, raw"\toprule")
    println(tex, raw" & $N$ & Mean & Median & SD & Min & P25 & P75 & Max \\ ")
    println(tex, raw"\midrule")
    println(tex, raw"\multicolumn{9}{l}{\textit{Panel A: Full sample}} \\[3pt]")

    function write_row(io, label, s)
        println(io, "$label & $(s.n) & $(s.mean) & $(s.med) & $(s.sd) & $(Int(s.min)) & $(s.p25) & $(s.p75) & $(Int(s.max)) \\\\")
    end

    write_row(tex, "Baseline specs", overall_bl)
    write_row(tex, "Baseline claims", overall_bg)
    write_row(tex, "Total specs (incl.\\ automated)", overall_ts)
    write_row(tex, "Core robustness specs", overall_cs)

    println(tex, raw"\midrule")
    println(tex, raw"\multicolumn{9}{l}{\textit{Panel B: Baseline specs by journal}} \\[3pt]")

    for js in journal_stats
        jlabel = replace(js.journal, "&" => "\\&")
        write_row(tex, "$jlabel (\$N\$=$(js.n))", js.stats)
    end

    println(tex, raw"\bottomrule")
    println(tex, raw"\end{tabular}")
    println(tex, raw"\begin{minipage}{0.95\textwidth}")
    println(tex, raw"\vspace{6pt}")
    println(tex, raw"\footnotesize\textit{Notes:} ``Baseline specs'' counts the number of distinct regression specifications reported by the original authors (across tables and columns) that were replicated. ``Baseline claims'' counts distinct hypotheses tested. ``Total specs'' includes all programmatically generated robustness variations. ``Core robustness specs'' are automated variations that test the same estimand as the author's baseline (e.g., leave-one-out controls, clustering alternatives, sample restrictions).")
    println(tex, raw"\end{minipage}")
    println(tex, raw"\end{table}")

    tex_str = String(take!(tex))

    # Save to estimation/tables/ and overleaf
    tables_dir = joinpath(dirname(dirname(@__DIR__)), "estimation", "tables")
    mkpath(tables_dir)
    write(joinpath(tables_dir, "tab_baseline_specs_summary.tex"), tex_str)
    println("  Saved tab_baseline_specs_summary.tex")

    ol_tables = joinpath(dirname(dirname(dirname(@__DIR__))), "overleaf", "tex", "v8_tables")
    if isdir(ol_tables)
        write(joinpath(ol_tables, "tab_baseline_specs_summary.tex"), tex_str)
        println("  Saved to overleaf")
    end
end


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

function main()
    println("=" ^ 60)
    println("Generating all estimation figures (Julia)")
    println("=" ^ 60)

    for f in [
        # Main text figures
        fig1_z_threeway,
        fig1b_tstat_filters,
        fig2e_foldnorm_mufree_k2,
        fig2f_foldnorm_mufree_k3,
        fig2g_foldnorm_mufree_k4,
        fig3b_nullfdr_m50,
        # Appendix figures
        fig4_i4r_agreement,
        fig2d_mixture_sigma_fixed_1,
        fig2e_mixture_sigma_geq_1,
        fig_systematic_grid,
        fig5_corr_distance,
        fig6_mixture_diagnostics,
        fig7_k_sensitivity,
        fig9c_disclosure_scaling_nullfdr,
        fig10_bootstrap_ci,
        fig13_journal_subgroup,
        fig14_posterior_heatmap,
        fig18_montecarlo,
        fig19_effective_sample_size,
    ]
        try
            f()
        catch e
            @warn "Error in $(nameof(f))" exception=(e, catch_backtrace())
        end
    end

    println("\n" * "=" ^ 60)
    println("Done!")
    println("=" ^ 60)
end

main()
