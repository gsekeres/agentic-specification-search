#!/usr/bin/env python3
"""
Create a manuscript specification-curve figure for one verified-core paper.

The default target is Drobner (2022), 139262-V1, chosen because it has a
single verified baseline group with a rich set of estimand-preserving
specifications and visible estimate dispersion.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "estimation" / "data"
FIG_DIR = BASE_DIR / "estimation" / "figures"
OVERLEAF_FIG_DIR = ROOT_DIR / "overleaf" / "v9" / "figures"

PAPER_ID = "139262-V1"
FIG_NAME = "fig_spec_curve_drobner_verified_core.pdf"

BLUE = "#2563eb"
RED = "#B31B1B"
GRAY = "#6b7280"
LIGHT_GRAY = "#d1d5db"
TEXT = "#111827"


def _clean(s: object) -> str:
    if pd.isna(s):
        return ""
    return str(s).strip()


def _sample_cell(sample_desc: str) -> str:
    sample_desc = _clean(sample_desc)
    return {
        "No-Resolution, Bad news": "No-res., bad",
        "No-Resolution, DiD (good vs bad)": "No-res., DiD",
        "No-Resolution, Good news": "No-res., good",
        "Resolution, Bad news": "Res., bad",
        "Resolution, DiD (good vs bad)": "Res., DiD",
        "Resolution, Good news": "Res., good",
    }.get(sample_desc, sample_desc or "Other")


def _control_set(controls_desc: str) -> str:
    controls_desc = _clean(controls_desc)
    if controls_desc in ("", "none", "signal + interaction"):
        return "No added controls"
    if "session FE" in controls_desc:
        return "Full controls + FE"
    if controls_desc == "rank + sumpoints + age + gender + prior":
        return "Full controls"
    if controls_desc.startswith("+ "):
        return "Single control"
    return "Other controls"


def _sample_rule(spec_id: str) -> str:
    spec_id = _clean(spec_id)
    if "exclude_wrong_and_zero_adjustments" in spec_id:
        return "Drop wrong/zero"
    if "exclude_wrong_adjustments" in spec_id:
        return "Drop wrong"
    if "exclude_extreme_ranks" in spec_id:
        return "Middle ranks"
    if "trim_y_5_95" in spec_id:
        return "Trim 5/95"
    return "Full sample"


def _fe_rule(fixed_effects: str) -> str:
    fixed_effects = _clean(fixed_effects)
    return "Session FE" if fixed_effects == "session" else "No FE"


def _family(category: str) -> str:
    return {
        "core_method": "Baseline/method",
        "core_controls": "Controls",
        "core_sample": "Sample",
        "core_fe": "Fixed effects",
        "core_joint": "Joint",
    }.get(_clean(category), "Other")


def _load_data() -> pd.DataFrame:
    spec = pd.read_csv(DATA_DIR / "spec_level_verified_core.csv")
    raw = pd.read_csv(
        BASE_DIR
        / "data"
        / "downloads"
        / "extracted"
        / PAPER_ID
        / "specification_results.csv"
    )

    keep_cols = ["spec_run_id", "sample_desc", "ci_lower", "ci_upper"]
    df = spec[spec["paper_id"] == PAPER_ID].merge(raw[keep_cols], on="spec_run_id", how="left")
    df = df[np.isfinite(df["coefficient"]) & np.isfinite(df["std_error"])].copy()
    df = df[df["v_is_valid"].fillna(0).astype(int).eq(1)]
    df = df[df["v_is_core_test"].fillna(0).astype(int).eq(1)]

    df["ci_low"] = df["ci_lower"].where(df["ci_lower"].notna(), df["coefficient"] - 1.96 * df["std_error"])
    df["ci_high"] = df["ci_upper"].where(df["ci_upper"].notna(), df["coefficient"] + 1.96 * df["std_error"])
    df["supported"] = (df["coefficient"] > 0) & (df["p_value_eff"] < 0.05)

    df["cell"] = df["sample_desc"].map(_sample_cell)
    df["control_set"] = df["controls_desc"].map(_control_set)
    df["sample_rule"] = df["spec_id"].map(_sample_rule)
    df["fe_rule"] = df["fixed_effects"].map(_fe_rule)
    df["family"] = df["v_category"].map(_family)

    df = df.sort_values(["coefficient", "ci_low", "spec_run_id"]).reset_index(drop=True)
    df["x"] = np.arange(len(df))
    return df


def _dimension_rows() -> list[tuple[str, str, list[str]]]:
    return [
        ("Cell", "cell", ["No-res., bad", "No-res., DiD", "No-res., good", "Res., bad", "Res., DiD", "Res., good"]),
        ("Controls", "control_set", ["No added controls", "Single control", "Full controls", "Full controls + FE"]),
        ("Sample", "sample_rule", ["Full sample", "Drop wrong", "Drop wrong/zero", "Middle ranks", "Trim 5/95"]),
        ("Fixed effects", "fe_rule", ["No FE", "Session FE"]),
        ("Family", "family", ["Baseline/method", "Controls", "Sample", "Fixed effects", "Joint"]),
    ]


def make_figure() -> None:
    df = _load_data()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OVERLEAF_FIG_DIR.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "mathtext.fontset": "cm",
            "axes.edgecolor": "#4b5563",
            "axes.linewidth": 0.6,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
        }
    )

    rows: list[tuple[str, str, str, int]] = []
    y = 0
    group_spans: list[tuple[str, float, float]] = []
    for label, col, cats in _dimension_rows():
        start = y
        for cat in cats:
            rows.append((label, col, cat, y))
            y += 1
        group_spans.append((label, start - 0.35, y - 0.65))
        y += 0.65

    fig = plt.figure(figsize=(7.4, 5.7))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 2.25], hspace=0.04)
    ax_top = fig.add_subplot(gs[0])
    ax_strike = fig.add_subplot(gs[1], sharex=ax_top)

    colors = np.where(df["supported"], BLUE, RED)
    for _, row in df.iterrows():
        x = row["x"]
        color = BLUE if row["supported"] else RED
        ax_top.vlines(x, row["ci_low"], row["ci_high"], color=color, alpha=0.28, linewidth=0.65)
    ax_top.scatter(df["x"], df["coefficient"], c=colors, s=9, linewidths=0, zorder=3)
    ax_top.axhline(0, color=GRAY, linewidth=0.7, alpha=0.8)
    ax_top.set_ylabel("Estimate")
    ax_top.tick_params(axis="x", labelbottom=False, length=0)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ypad = 0.08 * (df["ci_high"].max() - df["ci_low"].min())
    ax_top.set_ylim(df["ci_low"].min() - ypad, df["ci_high"].max() + ypad)

    n = len(df)
    for _, _, _, ypos in rows:
        ax_strike.hlines(ypos, -0.5, n - 0.5, color=LIGHT_GRAY, linewidth=0.35, alpha=0.8)

    for _, col, cat, ypos in rows:
        mask = df[col].eq(cat)
        if mask.any():
            row_colors = np.where(df.loc[mask, "supported"], BLUE, RED)
            ax_strike.scatter(
                df.loc[mask, "x"],
                np.full(mask.sum(), ypos),
                marker="|",
                s=30,
                c=row_colors,
                linewidths=0.85,
            )
        ax_strike.text(n + 0.9, ypos, cat, va="center", ha="left", fontsize=6.3)

    for label, y0, y1 in group_spans:
        mid = (y0 + y1) / 2
        ax_strike.plot([-3.0, -3.0], [y0, y1], color=GRAY, linewidth=0.65, clip_on=False)
        ax_strike.plot([-3.0, -0.7], [y0, y0], color=GRAY, linewidth=0.65, clip_on=False)
        ax_strike.plot([-3.0, -0.7], [y1, y1], color=GRAY, linewidth=0.65, clip_on=False)
        ax_strike.text(-3.6, mid, label, va="center", ha="right", fontsize=6.4)

    ax_strike.set_ylim(rows[-1][3] + 0.8, -0.8)
    ax_strike.set_yticks([])
    ax_strike.set_xlabel("Specification (sorted by estimate)")
    ax_strike.spines["top"].set_visible(False)
    ax_strike.spines["right"].set_visible(False)
    ax_strike.spines["left"].set_visible(False)
    ax_strike.set_xlim(-0.5, n - 0.5)
    ax_strike.tick_params(axis="x", labelsize=7, length=2)

    handles = [
        mpl.patches.Patch(facecolor=BLUE, edgecolor="none", label="Supported"),
        mpl.patches.Patch(facecolor=RED, edgecolor="none", label="Not supported"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=True,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.01),
    )

    for out_dir in (FIG_DIR, OVERLEAF_FIG_DIR):
        fig.savefig(out_dir / FIG_NAME, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved {FIG_NAME} for {PAPER_ID} with {len(df)} verified-core specifications.")


if __name__ == "__main__":
    make_figure()
