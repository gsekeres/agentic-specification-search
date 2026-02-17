#!/usr/bin/env python3
"""
agent_output_utils.py
====================

Small utilities for producing contract-compliant runner outputs:

- `coefficient_vector_json` reserved-key schema
- deterministic `surface_hash`
- standardized `software` + `error_details` blocks

This module is intentionally lightweight so paper-specific runner scripts can
import it without pulling in non-stdlib dependencies.
"""

from __future__ import annotations

import hashlib
import json
import sys
import traceback
from importlib import metadata
from typing import Any


PACKAGE_VERSION_CANDIDATES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "statsmodels",
    "linearmodels",
    "pyfixest",
    "scipy",
    "rdrobust",
    "openpyxl",
    "pyreadstat",
)

ALLOWED_PAYLOAD_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        # Required contract keys
        "coefficients",
        "inference",
        "software",
        "surface_hash",
        # Failure plumbing
        "error",
        "error_details",
        "partial",
        # Core RC axis blocks
        "controls",
        "sample",
        "fixed_effects",
        "preprocess",
        "estimation",
        "weights",
        "data_construction",
        "functional_form",
        "joint",
        # Common audit/metadata blocks
        "focal",
        "bundle",
        "warnings",
        "notes",
        # Flexible extension points
        "design",
        "extra",
        "universe",
        "sampling",
    }
)


def safe_single_line(s: str, *, max_len: int = 240) -> str:
    s = (s or "").replace("\r", " ").replace("\n", " ").strip()
    if len(s) > max_len:
        return s[: max_len - 1].rstrip() + "…"
    return s


def surface_hash(surface: dict[str, Any]) -> str:
    canon = json.dumps(surface, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    h = hashlib.sha256(canon.encode("utf-8")).hexdigest()
    return f"sha256:{h}"


def software_block(*, package_candidates: tuple[str, ...] = PACKAGE_VERSION_CANDIDATES) -> dict[str, Any]:
    pkgs: dict[str, str] = {}
    for name in package_candidates:
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        except Exception:
            continue
    return {
        "runner_language": "python",
        "runner_version": ".".join(map(str, sys.version_info[:3])),
        "packages": pkgs,
    }


def error_details_from_exception(e: BaseException, *, stage: str = "unknown", tb_lines: int = 18) -> dict[str, Any]:
    tb = "".join(traceback.format_exception(type(e), e, e.__traceback__)).strip()
    tail = "\n".join(tb.splitlines()[-tb_lines:]).strip()
    return {
        "stage": safe_single_line(stage, max_len=80) or "unknown",
        "exception_type": type(e).__name__ or "unknown",
        "exception_message": safe_single_line(str(e), max_len=500) or "unknown",
        "traceback_tail": tail,
    }


def move_unknown_payload_keys_to_extra(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Keep top-level schema stable by moving unknown keys under payload['extra'].
    """
    extra = payload.get("extra")
    if not isinstance(extra, dict):
        extra = {}
    unknown = [k for k in list(payload.keys()) if k not in ALLOWED_PAYLOAD_TOP_LEVEL_KEYS]
    for k in unknown:
        extra[k] = payload.pop(k)
    if extra:
        payload["extra"] = extra
    return payload


def make_success_payload(
    *,
    coefficients: dict[str, Any],
    inference: dict[str, Any],
    software: dict[str, Any],
    surface_hash: str,
    blocks: dict[str, Any] | None = None,
    axis_block: dict[str, Any] | None = None,
    axis_block_name: str | None = None,
    design: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    warnings: list[dict[str, Any]] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "coefficients": coefficients if isinstance(coefficients, dict) else {},
        "inference": inference if isinstance(inference, dict) else {},
        "software": software if isinstance(software, dict) else {},
        "surface_hash": str(surface_hash or ""),
    }
    if isinstance(blocks, dict) and blocks:
        for k, v in blocks.items():
            if k in {"coefficients", "inference", "software", "surface_hash"}:
                continue
            payload[k] = v
    if axis_block_name and isinstance(axis_block, dict) and axis_block:
        payload[axis_block_name] = axis_block
    if isinstance(design, dict) and design:
        payload["design"] = design
    if isinstance(extra, dict) and extra:
        payload["extra"] = extra
    if isinstance(warnings, list) and warnings:
        payload["warnings"] = warnings
    if isinstance(notes, str) and notes.strip():
        payload["notes"] = notes.strip()
    return move_unknown_payload_keys_to_extra(payload)


def make_failure_payload(
    *,
    error: str,
    error_details: dict[str, Any],
    inference: dict[str, Any] | None = None,
    software: dict[str, Any] | None = None,
    surface_hash: str | None = None,
    partial: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error": safe_single_line(error or "run_success=0", max_len=500),
        "error_details": error_details if isinstance(error_details, dict) else {},
    }
    if isinstance(inference, dict) and inference:
        payload["inference"] = inference
    if isinstance(software, dict) and software:
        payload["software"] = software
    if surface_hash:
        payload["surface_hash"] = str(surface_hash)
    if isinstance(partial, dict) and partial:
        payload["partial"] = partial
    if isinstance(extra, dict) and extra:
        payload["extra"] = extra
    return move_unknown_payload_keys_to_extra(payload)
