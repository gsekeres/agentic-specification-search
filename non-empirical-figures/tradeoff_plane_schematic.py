#!/usr/bin/env python3
"""
Generate a schematic "throughput–FDR plane" figure in (q_H, L(eps)) coordinates.

Outputs:
  - tex/v8_figures/tradeoff_plane_schematic.pdf
  - tex/v8_figures/tradeoff_plane_schematic.png
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


def main() -> None:
    # Put Matplotlib cache in a writable directory (avoids warnings on some systems).
    mpl_config_dir = Path(tempfile.gettempdir()) / "mplconfig-scientific-competition-overleaf"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
    os.environ.setdefault("XDG_CACHE_HOME", str(mpl_config_dir))

    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    out_dir = Path(__file__).resolve().parent
    pdf_path = out_dir / "tradeoff_plane_schematic.pdf"
    png_path = out_dir / "tradeoff_plane_schematic.png"

    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "axes.labelpad": 2.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
        }
    )

    # Schematic master bound: q_H * L <= K.
    # (Parameters chosen for visual clarity; no quantitative claim.)
    q_h = np.linspace(0.08, 1.0, 600)
    K = 4.0
    L_max = K / q_h

    fig, ax = plt.subplots(figsize=(6.6, 2.6), constrained_layout=True)
    ax.fill_between(q_h, 0.0, L_max, color="0.92", zorder=1)
    ax.plot(q_h, L_max, color="black", lw=2.6, zorder=3, label=r"Universal bound ($q_H\,L \lesssim \kappa$)")

    # Illustrative slices.
    L_target = 2.0
    ax.axhline(L_target, color="#2b6cb0", lw=1.8, ls="--", zorder=2, label=r"Fixed FDR target ($L$ fixed)")
    q_target = 0.35
    ax.axvline(q_target, color="0.55", lw=1.8, ls="--", zorder=2, label=r"Fixed recall ($q_H$ fixed)")

    # Illustrative RC attainment point on the boundary.
    q_rc = 0.5
    L_rc = K / q_rc
    ax.plot([q_rc], [L_rc], marker="o", ms=4.5, color="black", zorder=4)
    ax.text(
        q_rc + 0.04,
        L_rc - 0.35,
        "RC\n(attains bound)",
        fontsize=8.0,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        zorder=5,
    )

    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=8.3,
        handlelength=2.8,
        labelspacing=0.35,
        borderaxespad=0.4,
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 8.5)
    ax.set_xlabel(r"Recall $q_H$")
    ax.set_ylabel(r"Purity stringency $L(\varepsilon)=\log(1/\eta(\varepsilon))$")
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 2.0, 4.0, 6.0, 8.0])
    fmt = FuncFormatter(lambda x, _pos: f"{x:g}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()