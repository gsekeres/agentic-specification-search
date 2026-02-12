#!/usr/bin/env python3
"""
10_inference_audit.py
=====================

Lightweight static audit of paper analysis scripts for a common failure mode:
the script *records* a cluster variable but does not actually apply clustered
standard errors in the statsmodels estimation branch.

This cannot prove correctness, but it helps triage the agentic-vs-i4r
distribution mismatch by identifying "likely inflated t-stats" cases.

Writes:
  - estimation/results/inference_audit.csv
  - estimation/results/inference_audit_i4r.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def _bool(re_pat: str, s: str) -> int:
    return 1 if re.search(re_pat, s, flags=re.IGNORECASE | re.MULTILINE) else 0


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _classify(text: str) -> dict[str, int]:
    # Cluster "plumbing" indicators
    has_cov_type_cluster = _bool(r"cov_type\s*=\s*['\"]cluster['\"]", text)
    has_cov_kwds_groups = int(_bool(r"\bcov_kwds\b", text) == 1 and _bool(r"\bgroups\b", text) == 1)
    has_get_robustcov_cluster = int(
        _bool(r"get_robustcov_results", text) == 1 and _bool(r"cov_type\s*=\s*['\"]cluster['\"]", text) == 1
    )

    # Robust-only indicators
    has_hc_fit = _bool(r"cov_type\s*=\s*['\"]HC\d+['\"]", text) or _bool(r"cov_type\s*=\s*vcov_type", text)

    # pyfixest indicators
    has_pyfixest = _bool(r"import\s+pyfixest", text) or _bool(r"\bpf\.", text)
    has_pyfixest_cluster = _bool(r"vcov\s*=\s*\{\s*['\"]CRV\d+['\"]\s*:\s*[^}]+\}", text)

    # cluster arg presence (very coarse)
    mentions_cluster_var = _bool(r"cluster_(var|col)|\bcluster\b", text)

    # Suspected failure mode: a script has both (i) a pyfixest clustered path and
    # (ii) a statsmodels robust path *without* any cluster cov_kwds/groups usage.
    # In those scripts, any formulas that fall back to statsmodels can silently
    # ignore clustering.
    suspected_cluster_ignored_in_sm_branch = int(
        (mentions_cluster_var == 1)
        and (has_pyfixest_cluster == 1)
        and (has_hc_fit == 1)
        and (has_cov_type_cluster == 0)
        and (has_cov_kwds_groups == 0)
        and (has_get_robustcov_cluster == 0)
    )

    return {
        "mentions_cluster_var": mentions_cluster_var,
        "has_cov_type_cluster": has_cov_type_cluster,
        "has_cov_kwds_groups": has_cov_kwds_groups,
        "has_get_robustcov_cluster": has_get_robustcov_cluster,
        "has_hc_fit": has_hc_fit,
        "has_pyfixest": has_pyfixest,
        "has_pyfixest_cluster": has_pyfixest_cluster,
        "suspected_cluster_ignored_in_sm_branch": suspected_cluster_ignored_in_sm_branch,
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]  # agentic_specification_search
    scripts_dir = base_dir / "scripts" / "paper_analyses"
    data_dir = base_dir / "estimation" / "data"
    results_dir = base_dir / "estimation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    i4r_map_path = data_dir / "i4r_claim_map.csv"
    if not i4r_map_path.exists():
        raise FileNotFoundError(f"Missing {i4r_map_path}. Run estimation/run_all.py --data first.")

    i4r_ids = set(pd.read_csv(i4r_map_path)["paper_id"].astype(str).tolist())

    rows: list[dict] = []
    for path in sorted(scripts_dir.glob("*.py")):
        paper_id = path.stem
        text = _read(path)
        feats = _classify(text)
        rows.append(
            {
                "paper_id": paper_id,
                "path": str(path),
                "is_i4r": int(paper_id in i4r_ids),
                **feats,
            }
        )

    out = pd.DataFrame(rows).sort_values(["is_i4r", "suspected_cluster_ignored_in_sm_branch", "paper_id"], ascending=[False, False, True])
    out_csv = results_dir / "inference_audit.csv"
    out.to_csv(out_csv, index=False)

    out_i4r = out[out["is_i4r"] == 1].copy()
    out_i4r_csv = results_dir / "inference_audit_i4r.csv"
    out_i4r.to_csv(out_i4r_csv, index=False)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_i4r_csv}")


if __name__ == "__main__":
    main()
