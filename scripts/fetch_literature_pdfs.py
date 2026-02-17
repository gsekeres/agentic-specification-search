#!/usr/bin/env python3
"""
Fetch a small, curated set of openly accessible PDFs used to motivate the repo's
"specification surface" approach (forking paths / multiverse / specification curve / prereg).

Sources live in: docs/literature/sources.json
Outputs go to:   docs/literature/pdfs/   (ignored by git)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = ROOT / "docs" / "literature" / "sources.json"
DEFAULT_DEST = ROOT / "docs" / "literature" / "pdfs"
DEFAULT_MANIFEST = DEFAULT_DEST / "manifest.json"


@dataclass(frozen=True)
class Source:
    key: str
    title: str
    year: int
    pdf_url: str
    local_filename: str


def _load_sources(path: Path) -> list[Source]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: list[Source] = []
    for s in raw.get("sources", []):
        out.append(
            Source(
                key=str(s["key"]),
                title=str(s.get("title", "")).strip(),
                year=int(s.get("year", 0) or 0),
                pdf_url=str(s["pdf_url"]).strip(),
                local_filename=str(s.get("local_filename", f"{s['key']}.pdf")).strip(),
            )
        )
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _looks_like_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(8)
        return head.startswith(b"%PDF-")
    except Exception:
        return False


def _download(url: str, out_path: Path, timeout_s: int = 30) -> None:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "agentic-specification-search/1.0 (fetch_literature_pdfs.py)",
            "Accept": "application/pdf,*/*;q=0.8",
        },
        method="GET",
    )
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = resp.read()
    except urllib.error.URLError as e:
        raise RuntimeError(f"Download failed: {url} ({e})") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(data)
    if not _looks_like_pdf(tmp):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded content does not look like a PDF: {url}")
    tmp.replace(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Download curated surface-justification PDFs (open links only).")
    ap.add_argument("--sources", type=str, default=str(DEFAULT_SOURCES), help="Path to sources.json")
    ap.add_argument("--dest", type=str, default=str(DEFAULT_DEST), help="Destination directory for PDFs")
    ap.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST), help="Write manifest JSON here")
    ap.add_argument("--no-manifest", action="store_true", help="Do not write a manifest file")
    ap.add_argument("--list", action="store_true", help="List available source keys and exit")
    ap.add_argument("--key", action="append", default=[], help="Only fetch these keys (repeatable)")
    ap.add_argument("--force", action="store_true", help="Redownload even if the file exists")
    ap.add_argument("--timeout", type=int, default=45, help="Per-download timeout in seconds")
    args = ap.parse_args()

    sources_path = Path(args.sources)
    dest_dir = Path(args.dest)
    manifest_path = Path(args.manifest)

    if not sources_path.exists():
        print(f"Missing sources file: {sources_path}", file=sys.stderr)
        return 2

    sources = _load_sources(sources_path)
    if args.list:
        for s in sources:
            print(f"{s.key}\t{s.year}\t{s.title}")
        return 0

    want = set([k.strip() for k in args.key if k.strip()])
    if want:
        sources = [s for s in sources if s.key in want]
        missing = want - set([s.key for s in sources])
        if missing:
            print(f"Unknown keys: {sorted(missing)}", file=sys.stderr)
            return 2

    manifest: dict[str, dict] = {}
    errors: dict[str, str] = {}
    for s in sources:
        out_path = dest_dir / s.local_filename
        if out_path.exists() and not args.force:
            if not _looks_like_pdf(out_path):
                print(f"WARN: {out_path} exists but is not a PDF; redownloading.")
            else:
                print(f"SKIP: {s.key} (exists)")
                if not args.no_manifest:
                    manifest[s.key] = {
                        "key": s.key,
                        "title": s.title,
                        "year": s.year,
                        "pdf_url": s.pdf_url,
                        "local_path": str(out_path.relative_to(ROOT)),
                        "bytes": out_path.stat().st_size,
                        "sha256": _sha256(out_path),
                        "downloaded_at": None,
                    }
                continue

        print(f"GET : {s.key} -> {out_path.name}")
        t0 = time.time()
        try:
            _download(s.pdf_url, out_path, timeout_s=int(args.timeout))
            dt = time.time() - t0
            size = out_path.stat().st_size
            sha = _sha256(out_path)
            print(f"OK  : {s.key} ({size/1024:.1f} KiB) in {dt:.2f}s")
            if not args.no_manifest:
                manifest[s.key] = {
                    "key": s.key,
                    "title": s.title,
                    "year": s.year,
                    "pdf_url": s.pdf_url,
                    "local_path": str(out_path.relative_to(ROOT)),
                    "bytes": size,
                    "sha256": sha,
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                }
        except Exception as e:
            msg = str(e)
            errors[s.key] = msg
            print(f"FAIL: {s.key} ({msg})", file=sys.stderr)
            if not args.no_manifest:
                manifest[s.key] = {
                    "key": s.key,
                    "title": s.title,
                    "year": s.year,
                    "pdf_url": s.pdf_url,
                    "local_path": str(out_path.relative_to(ROOT)),
                    "bytes": None,
                    "sha256": None,
                    "downloaded_at": None,
                    "error": msg,
                }
            continue

    if not args.no_manifest:
        dest_dir.mkdir(parents=True, exist_ok=True)
        manifest_obj = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sources_file": str(sources_path.relative_to(ROOT)) if sources_path.is_absolute() else str(sources_path),
            "files": manifest,
        }
        manifest_path.write_text(json.dumps(manifest_obj, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote manifest: {manifest_path}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
