#!/usr/bin/env python3
"""
Batch timing script for spec_search and verification agents.

For each paper with data:
1. Extract ZIP to dummy_analyses/
2. Run spec_search_agent via Claude CLI and record wall-clock time
3. Run verification_agent via Claude CLI and record wall-clock time
4. Merge timing into timing_analyses.csv (preserving regression counts)
5. Clean up dummy_analyses/

Usage:
  python run_timing_batch.py                       # Run all papers, 10 parallel
  python run_timing_batch.py --paper 113888-V1     # Run single paper
  python run_timing_batch.py --dry-run             # List papers without running
  python run_timing_batch.py --parallel 5          # Run 5 at a time
  python run_timing_batch.py --model opus          # Use a specific model (default: opus)
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from count_regressions import count_package
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
RAW_PACKAGES = BASE_DIR / "data" / "downloads" / "raw_packages"
DUMMY_DIR = BASE_DIR / "dummy_analyses"
OUTPUT_CSV = BASE_DIR / "timing_analyses.csv"
STATUS_FILE = BASE_DIR / "data" / "tracking" / "spec_search_status.json"
SPEC_PROMPT_FILE = BASE_DIR / "prompts" / "spec_search_agent.md"
VERIF_PROMPT_FILE = BASE_DIR / "prompts" / "verification_agent.md"

# ── Agent timeout (2 hours per agent call) ─────────────────────────────────
AGENT_TIMEOUT = 7200

# Thread lock for CSV writes
csv_lock = threading.Lock()


def get_papers_with_data() -> set:
    """Return set of base paper IDs (numeric part) that have data."""
    with open(STATUS_FILE) as f:
        status = json.load(f)
    return {p["id"].split("-")[0] for p in status["packages_with_data"]}


def get_processable_zips() -> list:
    """Return sorted list of ZIP stems that correspond to papers with data."""
    with_data = get_papers_with_data()
    zips = sorted(
        z.stem
        for z in RAW_PACKAGES.glob("*.zip")
        if z.stem.split("-")[0] in with_data
    )
    return zips


def already_timed() -> set:
    """Return paper_ids that already have timing data in CSV."""
    if not OUTPUT_CSV.exists():
        return set()
    done = set()
    with open(OUTPUT_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("paper_id", "")
            if pid and row.get("spec_search_time_s", "").strip():
                done.add(pid)
    return done


def extract_package(paper_id: str) -> Path | None:
    """Extract ZIP to dummy_analyses/{paper_id}/ and return the path."""
    zip_path = RAW_PACKAGES / f"{paper_id}.zip"
    if not zip_path.exists():
        return None

    out_dir = DUMMY_DIR / paper_id
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)

    return out_dir


def cleanup_paper(paper_id: str):
    """Remove everything created for this paper in dummy_analyses."""
    target = DUMMY_DIR / paper_id
    if target.exists():
        shutil.rmtree(target)
    # Also clean up any verification output the agent may have created
    verif_dir = BASE_DIR / "data" / "verification" / paper_id
    if verif_dir.exists():
        shutil.rmtree(verif_dir)
    # Clean up any analysis script the agent may have created
    script_file = BASE_DIR / "scripts" / "paper_analyses" / f"{paper_id}.py"
    if script_file.exists():
        script_file.unlink()


def build_spec_search_prompt(paper_id: str, package_dir: Path) -> str:
    """Build the spec_search_agent prompt with paths filled in."""
    with open(SPEC_PROMPT_FILE) as f:
        template = f.read()

    prompt = template.replace("{PAPER_ID}", paper_id)
    prompt = prompt.replace("{EXTRACTED_PACKAGE_PATH}", str(package_dir))

    override = f"""
IMPORTANT OVERRIDES FOR TIMING RUN:
- Save specification_results.csv to: {package_dir}/specification_results.csv
- Save SPECIFICATION_SEARCH.md to: {package_dir}/SPECIFICATION_SEARCH.md
- Save the Python script to: {package_dir}/{paper_id}.py
- Do NOT update data/tracking/spec_search_status.json
- Do NOT run the cleanup script
- All outputs go into {package_dir}/ only
"""
    prompt = override + "\n" + prompt
    return prompt


def build_verification_prompt(paper_id: str, package_dir: Path) -> str:
    """Build the verification_agent prompt with paths filled in."""
    with open(VERIF_PROMPT_FILE) as f:
        template = f.read()

    prompt = template.replace("{PAPER_ID}", paper_id)
    prompt = prompt.replace("{EXTRACTED_PACKAGE_PATH}", str(package_dir))

    override = f"""
IMPORTANT OVERRIDES FOR TIMING RUN:
- Save ALL verification outputs to: {package_dir}/verification/
  (NOT to data/verification/{paper_id}/)
- verification_baselines.json -> {package_dir}/verification/verification_baselines.json
- verification_spec_map.csv -> {package_dir}/verification/verification_spec_map.csv
- VERIFICATION_REPORT.md -> {package_dir}/verification/VERIFICATION_REPORT.md
- All outputs go into {package_dir}/ only
"""
    prompt = override + "\n" + prompt
    return prompt


def run_claude_agent(prompt: str, model: str = "opus",
                     timeout: int = AGENT_TIMEOUT) -> tuple[float, bool, str]:
    """
    Run a Claude agent via CLI and return (elapsed_seconds, success, output).
    Uses subscription (no API key needed).
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, dir=tempfile.gettempdir()
    ) as tmp:
        tmp.write(prompt)
        prompt_file = tmp.name

    try:
        cmd = [
            "claude",
            "-p",
            "--model", model,
            "--dangerously-skip-permissions",
        ]

        start = time.time()
        with open(prompt_file) as pf:
            result = subprocess.run(
                cmd,
                stdin=pf,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(BASE_DIR),
            )
        elapsed = time.time() - start
        success = result.returncode == 0
        output = result.stdout[-2000:] if result.stdout else ""
        if not success:
            output += "\nSTDERR: " + (result.stderr[-1000:] if result.stderr else "")
        return elapsed, success, output

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return elapsed, False, "TIMEOUT"
    finally:
        os.unlink(prompt_file)


def update_csv_timing(paper_id: str, spec_time: float, spec_ok: bool,
                      verif_time: float, verif_ok: bool,
                      n_regressions: int | None = None):
    """Update the existing CSV row for this paper with timing data."""
    with csv_lock:
        # Read all rows
        rows = []
        with open(OUTPUT_CSV, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                rows.append(row)

        # Update the matching row
        found = False
        for row in rows:
            if row["paper_id"] == paper_id:
                row["spec_search_time_s"] = f"{spec_time:.1f}"
                row["verification_time_s"] = f"{verif_time:.1f}"
                row["total_agent_time_s"] = f"{spec_time + verif_time:.1f}"
                row["spec_search_success"] = str(spec_ok)
                row["verification_success"] = str(verif_ok)
                if n_regressions is not None and not row.get("n_regressions_original", "").strip():
                    row["n_regressions_original"] = str(n_regressions)
                found = True
                break

        if not found:
            # Append new row if not in CSV
            rows.append({
                "paper_id": paper_id,
                "n_regressions_original": str(n_regressions) if n_regressions is not None else "",
                "spec_search_time_s": f"{spec_time:.1f}",
                "verification_time_s": f"{verif_time:.1f}",
                "total_agent_time_s": f"{spec_time + verif_time:.1f}",
                "spec_search_success": str(spec_ok),
                "verification_success": str(verif_ok),
            })

        # Write back
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def process_paper(paper_id: str, model: str, idx: int, total: int) -> dict:
    """Process a single paper: extract, time both agents, cleanup. Returns result dict."""
    label = f"[{idx}/{total}] {paper_id}"
    print(f"{label}: Starting...")

    # 1. Extract
    package_dir = extract_package(paper_id)
    if package_dir is None:
        print(f"{label}: SKIPPED (no ZIP)")
        return {"paper_id": paper_id, "status": "skipped"}

    # 1b. Count regressions in original code
    n_regs = count_package(str(package_dir))["total"]
    print(f"{label}: {n_regs} regressions in original code")

    # 2. Run spec_search_agent
    print(f"{label}: Running spec_search_agent...")
    prompt = build_spec_search_prompt(paper_id, package_dir)
    spec_time, spec_ok, spec_out = run_claude_agent(prompt, model=model)
    spec_status = "OK" if spec_ok else "FAILED"
    print(f"{label}: Spec search: {spec_time:.0f}s ({spec_status})")

    # 3. Run verification_agent
    print(f"{label}: Running verification_agent...")
    prompt = build_verification_prompt(paper_id, package_dir)
    verif_time, verif_ok, verif_out = run_claude_agent(prompt, model=model)
    verif_status = "OK" if verif_ok else "FAILED"
    print(f"{label}: Verification: {verif_time:.0f}s ({verif_status})")

    # 4. Record timing
    update_csv_timing(paper_id, spec_time, spec_ok, verif_time, verif_ok, n_regressions=n_regs)
    total_time = spec_time + verif_time
    print(f"{label}: DONE - Total: {total_time:.0f}s ({total_time/60:.1f} min)")

    # 5. Cleanup
    cleanup_paper(paper_id)

    return {
        "paper_id": paper_id,
        "spec_time": spec_time,
        "spec_ok": spec_ok,
        "verif_time": verif_time,
        "verif_ok": verif_ok,
        "total_time": total_time,
        "status": "done",
    }


def main():
    parser = argparse.ArgumentParser(description="Time spec_search + verification agents")
    parser.add_argument("--paper", help="Process a single paper ID")
    parser.add_argument("--dry-run", action="store_true", help="List papers without running")
    parser.add_argument("--model", default="opus",
                        help="Claude model to use (default: opus)")
    parser.add_argument("--parallel", type=int, default=10,
                        help="Number of papers to process in parallel (default: 10)")
    parser.add_argument("--delay", type=int, default=0,
                        help="Seconds to sleep between submitting new papers (default: 0)")
    args = parser.parse_args()

    DUMMY_DIR.mkdir(exist_ok=True)

    if args.paper:
        papers = [args.paper]
    else:
        papers = get_processable_zips()

    done = already_timed()
    papers = [p for p in papers if p not in done]

    if args.dry_run:
        print(f"Would process {len(papers)} papers (parallel={args.parallel}, model={args.model}):")
        for p in papers:
            print(f"  {p}")
        return

    print(f"Processing {len(papers)} papers")
    print(f"  Model: {args.model}")
    print(f"  Parallel: {args.parallel}")
    print(f"  Delay: {args.delay}s between submissions")
    print(f"  Output: {OUTPUT_CSV}")
    print(f"  Working dir: {DUMMY_DIR}")
    print()

    # Process in parallel batches with staggered submission
    results = []
    batch_start = time.time()

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {}
        for i, paper_id in enumerate(papers, 1):
            future = executor.submit(process_paper, paper_id, args.model, i, len(papers))
            futures[future] = paper_id
            if args.delay > 0 and i < len(papers):
                time.sleep(args.delay)

        for future in as_completed(futures):
            paper_id = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"ERROR processing {paper_id}: {e}")
                results.append({"paper_id": paper_id, "status": "error", "error": str(e)})

    batch_elapsed = time.time() - batch_start

    # Summary
    completed = [r for r in results if r["status"] == "done"]
    failed = [r for r in results if r["status"] not in ("done", "skipped")]
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE")
    print(f"{'='*60}")
    print(f"  Papers processed: {len(completed)}/{len(papers)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Wall-clock time: {batch_elapsed:.0f}s ({batch_elapsed/60:.1f} min)")
    if completed:
        avg_total = sum(r["total_time"] for r in completed) / len(completed)
        print(f"  Avg time per paper: {avg_total:.0f}s ({avg_total/60:.1f} min)")
        spec_ok = sum(1 for r in completed if r["spec_ok"])
        verif_ok = sum(1 for r in completed if r["verif_ok"])
        print(f"  Spec search success: {spec_ok}/{len(completed)}")
        print(f"  Verification success: {verif_ok}/{len(completed)}")
    print(f"\nResults in {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
