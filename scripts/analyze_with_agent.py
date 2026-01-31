"""
Analyze Study with Claude Code Agent

Invokes Claude Code CLI to:
1. Examine a study's README, code, and data
2. Write a custom specification curve script (1000+ specs)
3. Run the analysis
4. Save standardized results

Usage:
    python workflow/analyze_with_agent.py --project-id 240901
    python workflow/analyze_with_agent.py --project-id 240901 --cleanup
"""

import argparse
import json
import subprocess
import shutil
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
REGISTRY_DIR = BASE_DIR / "registry"
DATA_DIR = BASE_DIR / "data" / "downloads"
SCRIPTS_DIR = BASE_DIR / "workflow" / "generated_scripts"

AGENT_PROMPT_TEMPLATE = '''Analyze AEA replication package {project_id} and build a specification curve.

## Paths
- Data: {data_dir}
- Output CSV: {output_path}
- Script to write: {script_path}

## Task

1. **Examine the study**: Read the README and main analysis code to understand:
   - Core causal hypothesis
   - Treatment variable
   - Outcome variables
   - Panel structure (entity/time identifiers)
   - Controls used

2. **Write a Python script** running 1000+ specifications testing the core hypothesis.

   Vary across:
   - Outcomes (all related measures)
   - Covariate sets (none, partial, full, extended)
   - Sample restrictions (full, time subperiods, subsets)
   - Fixed effects (two-way, entity-only, time-only)
   - Clustering (entity, robust)

3. **Required output columns**:
   project_id, spec_id, outcome, treatment, cov_label, sample_label, fe_label, cluster, coef, se, tstat, pval, nobs

4. **Run the script** to generate results.

Use linearmodels.panel.PanelOLS. Handle errors gracefully. Print progress every 100 specs.
'''


def get_prompt(project_id: str, data_dir: Path, output_path: Path, script_path: Path) -> str:
    return AGENT_PROMPT_TEMPLATE.format(
        project_id=project_id,
        data_dir=data_dir,
        output_path=output_path,
        script_path=script_path
    )


def record_to_registry(registry_file: Path, record: dict):
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_file, 'a') as f:
        f.write(json.dumps(record) + '\n')


def cleanup_study(project_id: str):
    """Delete raw data after analysis."""
    extracted_dir = DATA_DIR / "extracted" / project_id
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)
        print(f"Deleted: {extracted_dir}")

    raw_dir = DATA_DIR / "raw_packages"
    if raw_dir.exists():
        for f in raw_dir.glob(f"{project_id}*.zip"):
            f.unlink()
            print(f"Deleted: {f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze study with Claude Code')
    parser.add_argument('--project-id', required=True, help='OpenICPSR project ID')
    parser.add_argument('--cleanup', action='store_true', help='Delete raw data after analysis')
    parser.add_argument('--skip-if-exists', action='store_true', help='Skip if results exist')
    parser.add_argument('--dry-run', action='store_true', help='Print prompt without running')

    args = parser.parse_args()
    project_id = args.project_id

    # Paths
    data_dir = DATA_DIR / "extracted" / project_id
    output_path = RESULTS_DIR / "spec_curves" / f"{project_id}_specs.csv"
    script_path = SCRIPTS_DIR / f"analyze_{project_id}.py"

    # Skip if done
    if args.skip_if_exists and output_path.exists():
        print(f"Results exist: {output_path}")
        return

    # Check data exists
    if not data_dir.exists():
        print(f"Error: Data not found at {data_dir}")
        record_to_registry(
            REGISTRY_DIR / "excluded_studies.jsonl",
            {'project_id': project_id, 'reason': 'data_not_found', 'timestamp': datetime.now().isoformat()}
        )
        return

    # Create directories
    output_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.parent.mkdir(parents=True, exist_ok=True)

    prompt = get_prompt(project_id, data_dir, output_path, script_path)

    print(f"Analyzing study {project_id}")
    print(f"  Data: {data_dir}")
    print(f"  Output: {output_path}")

    if args.dry_run:
        print("\n--- PROMPT ---")
        print(prompt)
        print("--- END ---")
        return

    # Invoke Claude Code CLI
    result = subprocess.run(
        ['claude', '--dangerously-skip-permissions', '-p', prompt],
        cwd=str(BASE_DIR),
        capture_output=False
    )

    if result.returncode != 0:
        print(f"Claude Code exited with code {result.returncode}")
        record_to_registry(
            REGISTRY_DIR / "excluded_studies.jsonl",
            {'project_id': project_id, 'reason': 'agent_error', 'timestamp': datetime.now().isoformat()}
        )
        return

    # Cleanup if requested and results exist
    if output_path.exists():
        record_to_registry(
            REGISTRY_DIR / "completed_studies.jsonl",
            {'project_id': project_id, 'timestamp': datetime.now().isoformat(), 'output': str(output_path)}
        )
        if args.cleanup:
            print("Cleaning up raw data...")
            cleanup_study(project_id)
    else:
        print(f"Warning: Expected output not found at {output_path}")


if __name__ == "__main__":
    main()
