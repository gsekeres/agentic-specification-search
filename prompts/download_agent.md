# Download Agent Prompt

Use this prompt when launching agents to download AEA replication packages from openICPSR.

---

## Prerequisites

1. Chrome open and logged into https://www.openicpsr.org/
2. Chrome download settings configured:
   - Download location: `agentic_specification_search/data/downloads/raw_packages/`
   - "Ask where to save" disabled
   - Popups allowed for openicpsr.org (the download button triggers a popup)

---

## Prompt Template

```
Download AEA replication packages from openICPSR using Chrome MCP tools.

I'm logged into openICPSR in Chrome. Download packages from the metadata file.

**Metadata file**: /Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/metadata/packages_2025_classified.jsonl

**Filter criteria**: Download packages where data files are present (dta, csv, xlsx, etc.)

**Size limit**: Stop if total downloads exceed {SIZE_LIMIT_GB} GB

For each matching package:

1. **Navigate** to the project page:
   `https://www.openicpsr.org/openicpsr/project/{project_id}/version/V1/view`

2. **Check page status** using `chrome_read_page`:
   - If "restricted" or "request information" → log as restricted, skip
   - If "Download this project" button exists → continue

3. **Click "Download this project"** button

4. **On terms page**: Scroll down, click "I AGREE" button (has `id="downloadButton"`)

5. **Wait for download** using `chrome_handle_download` (timeout 5 minutes for large files)

6. **Log result** to `data/tracking/download_manifest.jsonl`:
   ```json
   {"doi": "...", "project_id": "...", "status": "downloaded|restricted|error", "filename": "...", "bytes": ..., "timestamp": "..."}
   ```

7. **Wait 2-3 seconds** between downloads (rate limiting)

If a download fails, log the error and continue to the next package.
```

---

## URL Patterns

| What | URL |
|------|-----|
| Project page | `https://www.openicpsr.org/openicpsr/project/{project_id}/version/V1/view` |
| Download terms | `https://www.openicpsr.org/openicpsr/project/{project_id}/version/V1/download/terms?path=/openicpsr/{project_id}/fcr:versions/V1&type=project` |

---

## Page States

| State | Indicator | Action |
|-------|-----------|--------|
| Public data | "Download this project" button | Download |
| Restricted | "Request Information" or "restricted" text | Log as restricted, skip |
| Missing | 404 or "not found" | Log as missing, skip |
| Rate limited | CAPTCHA or warning | Pause 5 minutes |

---

## Manifest Format

Each download attempt logged to `data/tracking/download_manifest.jsonl`:

```json
{
  "doi": "10.3886/E240901V1",
  "project_id": "240901",
  "version": "V1",
  "status": "downloaded",
  "filename": "240901-V1.zip",
  "bytes": 52428800,
  "timestamp": "2026-01-29T10:30:00Z",
  "error": null
}
```

Status values: `downloaded`, `restricted`, `error`, `missing`

---

## Post-Download

After downloads complete:

1. Unzip to `data/downloads/extracted/{project_id}-V1/`
2. Proceed to specification search (see `prompts/spec_search_agent.md`)
3. Optionally delete raw zip files after extraction

---

## Paper Selection Strategy (Target: 100 Papers)

### Journal Quotas

| Journal | Target |
|---------|--------|
| AER | 25 |
| AEJ-Applied | 25 |
| AEJ-Policy | 20 |
| AEJ-Macro | 10 |
| AEJ-Micro | 10 |
| JEL/JEP | 10 |

### Selection Criteria

1. **Any AEA paper with data in openICPSR** (no method/code restrictions)
2. **Random sample** from available papers (not topic-stratified)
3. **Journal quotas maintained** (AER vs other AEA journals)

### Selection Process

```python
# Pseudo-code for paper selection
1. Query DataCite for ALL AEA packages in openICPSR
2. Filter: has data files (dta, csv, xlsx, etc.) - no other restrictions
3. Random sample stratified by journal to meet quotas
4. Download all selected packages
5. Agent determines method type during analysis
```
