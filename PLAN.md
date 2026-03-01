# Plan: Scale to 100 Fully Analyzed Papers

## Current State
- **5 papers fully analyzed**: 111185-V1, 112431-V1, 112749-V1, 113517-V1, 113561-V1
- **5 newly downloaded** (test batch, not yet analyzed): 113182-V1, 112474-V1, 114542-V1, 114295-V1, 114098-V1
- **Need: 95 more** fully analyzed papers (100 total)

## Target Composition
- **40 i4R papers** on openICPSR (41 minus 173441 "Bubbles, Crashes, and Economic Growth" = 40)
- **55 additional AEA papers** from the 4,284-package universe (random/stratified sample)
- **5 already done** count toward the 100

## i4R Project IDs to Download (40 papers)

All from `i4r/papers_with_replication_urls.csv`, excluding 173441:

```
140921 120078 149481 126722 136741 131981 128521 120483
140161 130784 134041 125201 148301 125821 149262 140121
138922 138401 128143 120568 130141 125321 167841 180741
171681 174501 147561 158401 145141 150323 157781 149882
151841 181166 184041 146041 173341 181581 139262 150581
```

## Phase 1: Download i4R Packages

```bash
export ICPSR_EMAIL="gs754@cornell.edu"
export ICPSR_PASS="Summerduckpond19!"

python scripts/download_packages.py \
  --project-ids 140921 120078 149481 126722 136741 131981 128521 120483 \
    140161 130784 134041 125201 148301 125821 149262 140121 \
    138922 138401 128143 120568 130141 125321 167841 180741 \
    171681 174501 147561 158401 145141 150323 157781 149882 \
    151841 181166 184041 146041 173341 181581 139262 150581 \
  --delay 8 --skip-extract
```

Estimated: 2-10 GB of ZIPs. Downloads tracked in `data/tracking/download_tracking.jsonl`.

## Phase 2: Download 55 Additional AEA Papers

Pick 55 papers from the universe, excluding already-downloaded and i4R papers. Stratify across journals for representativeness.

```bash
python scripts/download_packages.py --sample 55 --seed 123 --delay 8 --skip-extract
```

The `--sample` flag draws from the full universe but automatically skips papers already in tracking. If we want journal stratification, use multiple filtered runs:

```bash
# ~20 from AER
python scripts/download_packages.py --journal "American Economic Review" --sample 20 --seed 100 --delay 8 --skip-extract
# ~15 from AEJ journals
python scripts/download_packages.py --journal "American Economic Journal" --sample 15 --seed 101 --delay 8 --skip-extract
# ~10 from AER: Insights
python scripts/download_packages.py --journal "Insights" --sample 10 --seed 102 --delay 8 --skip-extract
# ~10 from JEP/JEL/P&P
python scripts/download_packages.py --journal "Perspectives" --sample 5 --seed 103 --delay 8 --skip-extract
python scripts/download_packages.py --journal "Proceedings" --sample 5 --seed 104 --delay 8 --skip-extract
```

Adjust counts to reach 55 total new papers (100 - 5 existing - 40 i4R = 55).

## Phase 3: Analyze in Rolling Batches of ~10

For each batch of ~10 papers:

### Step 1: Extract
```bash
python scripts/extract_packages.py --paper-id {PAPER_ID}
```

### Step 2: Run Agent Pipeline (per paper)

Each paper goes through prompts 02 → 03 → 04 → 05 → 06:

1. **02_paper_classifier.md** — Identify design family (cross-sectional OLS, DID, IV, RCT, etc.)
2. **03_spec_surface_builder.md** — Build `SPECIFICATION_SURFACE.json` (budget, constraints, sampling)
3. **04_spec_surface_verifier.md** — Critique/edit surface before running
4. **05_spec_searcher.md** — Execute approved surface → `specification_results.csv` + `inference_results.csv`
5. **06_post_run_verifier.md** — Post-run audit + core classification

Target: 50+ specifications per paper.

### Step 3: Validate
```bash
python scripts/validate_agent_outputs.py --paper-id {PAPER_ID}
```

### Step 4: Clean Up Raw Data (Preserve Disk)

After analysis, delete raw data files from `extracted/{PAPER_ID}/` but **keep** agent outputs (git-tracked):
- Keep: `SPECIFICATION_SURFACE.json`, `SPECIFICATION_SURFACE.md`, `SPEC_SURFACE_REVIEW.md`, `specification_results.csv`, `inference_results.csv`, `SPECIFICATION_SEARCH.md`
- Delete: everything else (original .dta, .csv, .do, .R, .py data files)

Also delete the raw ZIP after extraction is confirmed:
```bash
rm data/downloads/raw_packages/{PAPER_ID}.zip
```

This reduces disk from ~200 MB/paper to ~1 MB/paper.

### Step 5: Move to Next Batch

Repeat steps 1-4 for the next 10 papers.

## Phase 4: Rebuild Estimation Pipeline

After all 100 papers are analyzed:

```bash
python estimation/run_all.py --all
```

This rebuilds `unified_results.csv`, fits mixtures, estimates dependence, and generates tables/figures.

## Batch Schedule

| Batch | Papers | Priority | Notes |
|-------|--------|----------|-------|
| 0 | 5 existing (111185, 112431, 112749, 113517, 113561) | Done | Already fully analyzed |
| 1 | 10 i4R papers | Highest | Start with AEJ:Applied + AEJ:EP papers |
| 2 | 10 i4R papers | Highest | AER:Insights papers |
| 3 | 10 i4R papers | Highest | More AER:Insights + AER papers |
| 4 | 10 i4R papers | Highest | Remaining i4R papers |
| 5 | 5 test downloads + 5 new AEA | High | Analyze the 5 already-downloaded + 5 new |
| 6-9 | 10 AEA papers each (×4) | Medium | Fill to 100 with stratified AEA sample |
| 10 | 5 AEA papers | Medium | Final batch to 100 |

## Resource Estimates

- **Disk**: ~200 MB/paper during analysis, ~1 MB/paper after cleanup. Peak: ~2 GB for a batch of 10.
- **RAM**: Agent analysis runs one paper at a time; peak ~2-4 GB depending on dataset size.
- **Network**: ~5 GB total download for 95 packages at ~50 MB average.
- **Time**: ~5-10 min download per paper (with delays), ~15-30 min analysis per paper.

## Key Commands Reference

```bash
# Download
python scripts/download_packages.py --project-ids PID1 PID2 --delay 8 --skip-extract
python scripts/download_packages.py --login-only
python scripts/download_packages.py --dry-run --project-ids PID1 PID2

# Extract
python scripts/extract_packages.py --paper-id {PAPER_ID}
python scripts/extract_packages.py  # all unextracted

# Validate
python scripts/validate_agent_outputs.py --paper-id {PAPER_ID}
python scripts/validate_agent_outputs.py --all

# Universe
python data/tracking/build_aea_universe.py --stats-only
python data/tracking/build_aea_universe.py --force-refresh

# Estimation
python estimation/run_all.py --all
python estimation/run_all.py --data
```
