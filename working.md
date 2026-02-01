# AEA Package Collection & Journal Identification

## Overview

This directory contains scripts to collect all openICPSR package DOIs and identify which ones are linked to AEA (American Economic Association) journal articles.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/01_collect_dois.py` | Collects ALL openICPSR DOIs (prefix 10.3886) from DataCite API |
| `scripts/02_identify_journals.py` | Filters to AEA packages and maps each to its journal |
| `scripts/create_unified_csv.py` | Utility for later CSV generation |

## Output Files

| File | Size | Records | Description |
|------|------|---------|-------------|
| `data/metadata/icpsr_openicpsr_packages.jsonl` | 19M | 35,110 | Full ICPSR universe |
| `data/metadata/aea_package_to_journal.jsonl` | 1.2M | 7,409 | AEA packages only |

### Schema: `icpsr_openicpsr_packages.jsonl`

```json
{
  "package_doi": "10.3886/E123456V1",
  "base_id": "E123456",
  "version": "V1",
  "title": "Replication data for: ...",
  "publisher": "ICPSR - Interuniversity Consortium for Political and Social Research",
  "created": "2023-01-15",
  "updated": "2023-06-20",
  "landing_page_url": "https://www.openicpsr.org/openicpsr/project/123456/version/V1/view",
  "related_identifiers": [
    {"relatedIdentifier": "10.1257/aer.20210123", "relatedIdentifierType": "DOI", "relationType": "IsSupplementTo"}
  ]
}
```

### Schema: `aea_package_to_journal.jsonl`

```json
{
  "package_doi": "10.3886/E123456V1",
  "article_dois": ["10.1257/aer.20210123"],
  "journal_code": "AER",
  "confidence": "high",
  "source": "datacite_relatedIdentifier"
}
```

## Journal Distribution

| Journal | Packages |
|---------|----------|
| AER | 3,295 |
| AEJ: Policy | 956 |
| AEJ: Applied | 954 |
| AER: P&P | 769 |
| AEJ: Macro | 708 |
| JEP | 309 |
| AEJ: Micro | 298 |
| AER: Insights | 90 |
| JEL | 30 |
| **Total AEA** | **7,409** |

## Journal Code Mapping

A package is identified as "AEA" if it has a `relatedIdentifier` pointing to a `10.1257/*` DOI:

| DOI Prefix | Journal Code |
|------------|--------------|
| `10.1257/aer.` | AER |
| `10.1257/pandp.` | AER: P&P |
| `10.1257/aeri.` | AER: Insights |
| `10.1257/pol.` | AEJ: Policy |
| `10.1257/mic.` | AEJ: Micro |
| `10.1257/mac.` | AEJ: Macro |
| `10.1257/app.` | AEJ: Applied |
| `10.1257/jep.` | JEP |
| `10.1257/jel.` | JEL |

## Cache Files

| File | Size | Purpose |
|------|------|---------|
| `data/cache/datacite_responses.db` | 127M | SQLite cache of all DataCite API responses |
| `data/cache/aea_migration_table.csv` | 302K | Fallback mapping from AEA GitHub repo |

## Usage

### Collect all ICPSR DOIs

```bash
python scripts/01_collect_dois.py
```

- Fetches all DOIs with prefix `10.3886` from DataCite
- Uses cursor pagination (~73 API calls for ~35k packages)
- Implements rate limiting (800 requests / 5 minutes)
- Handles HTTP 429 with exponential backoff
- Resumable via SQLite cache

### Identify AEA journals

```bash
python scripts/02_identify_journals.py
```

- Reads `icpsr_openicpsr_packages.jsonl`
- Filters to packages with `10.1257/*` relatedIdentifiers
- Falls back to AEA migration table for older packages
- Outputs only AEA packages (non-AEA excluded)
- Resumable via output file

## Technical Details

### Rate Limiting

DataCite "identified" tier allows 1000 requests / 5 minutes. Scripts use:
- Token bucket limiter: 800 requests / 5 minutes (safe margin)
- HTTP 429 handler: honors `Retry-After` header with exponential backoff + jitter

### Resumability

Both scripts are resumable:
- `01_collect_dois.py`: Caches API responses and tracks cursor position in SQLite
- `02_identify_journals.py`: Loads existing output file and skips already-processed packages

Re-runs complete in ~1-2 seconds when cache is populated.

### DOI Filtering

File-level DOIs (e.g., `E123456V1-12345`) are excluded. Only package-level DOIs are retained.

## Data Sources

1. **DataCite API**: Primary source for DOIs and metadata
   - Endpoint: `https://api.datacite.org/dois`
   - Query: `prefix:10.3886`

2. **AEA Migration Table**: Fallback for older packages
   - URL: `https://raw.githubusercontent.com/AEADataEditor/aea-supplement-migration/master/data/generated/table.aea.icpsr.mapping.csv`
   - Provides 2,562 mappings (all redundant with DataCite data)

## Statistics

- Total ICPSR packages: 35,110
- AEA-linked packages: 7,409 (21.1%)
- Non-AEA packages: 27,701 (78.9%)
- All AEA packages identified via DataCite relatedIdentifiers
- AEA migration table provided 2,562 entries (100% redundant)

## Last Updated

2026-01-31

---

## Specification Search Results

### Overview

Systematic specification searches were conducted on 10 AEA replication packages to assess the robustness of their main findings. Each package was analyzed using a specification tree methodology that varies bandwidth, controls, sample restrictions, clustering, and functional form choices.

### Summary Statistics

| Metric | Value |
|--------|-------|
| Total papers analyzed | 10 |
| Total specifications run | 847 |
| Average specs per paper | 84.7 |
| Significance rate (p < 0.05) | 30.2% |

### Papers Analyzed

| Paper ID | Journal | Title | Specs | Method |
|----------|---------|-------|-------|--------|
| 113597-V2 | AEJ: Applied | The Impacts of Microfinance: Evidence from Joint-Liability Lending in Mongolia | 158 | Cross-sectional OLS |
| 116136-V2 | AER | Yours, Mine, and Ours: Do Divorce Laws Affect the Intertemporal Behavior of Married Couples? | 28 | Panel FE / DiD |
| 183985-V1 | AER: P&P | Emoticons as Performance Feedback for College Students | 34 | RCT / ANCOVA |
| 184341-V2 | AER: P&P | Emotional and Behavioral Impacts of Telementoring and Homeschooling Support | 86 | RCT / ANCOVA |
| 193625-V1 | AEJ: Policy | The Great Recession and the Widening Income Gap Between Alumni | 50 | DiD / Event Study |
| 207766-V1 | AER | Organized Voters: Elections and Public Funding of Nonprofits | 106 | Regression Discontinuity |
| 214201-V1 | AER | Mission Motivation and Public Sector Performance: Evidence from Pakistan | 25 | RCT / ANCOVA |
| 217741-V1 | AER: P&P | AI and Women's Employment in Europe | 152 | Panel FE / DiD |
| 223321-V1 | AER: P&P | Immigrant Age at Arrival and Intergenerational Transmission of Ethnic ID | 57 | Cross-sectional OLS/WLS |
| 230401-V1 | AER | Are Loans to Minority-Owned Firms Mispriced? | 151 | Cross-sectional OLS |

### Journal Distribution

| Journal | Specifications |
|---------|---------------|
| AER: P&P | 329 (38.8%) |
| AER | 310 (36.6%) |
| AEJ: Applied | 158 (18.7%) |
| AEJ: Policy | 50 (5.9%) |

### Method Distribution (Top Categories)

| Specification Type | Count |
|-------------------|-------|
| Leave-one-out control analysis | 182 |
| Single covariate analysis | 143 |
| Sample restrictions | 139 |
| Control set variations | 46 |
| Clustering variations | 44 |
| Bandwidth selection (RD) | 20 |

### Output Files

| File | Description |
|------|-------------|
| `unified_results.csv` | Combined results from all 847 specifications |
| `data/downloads/extracted/*/specification_results.csv` | Per-paper specification results |
| `data/downloads/extracted/*/SPECIFICATION_SEARCH.md` | Per-paper summary reports |
| `data/tracking/completed_analyses.jsonl` | Tracking of processed papers |

### Robustness Findings Summary

**Highly Robust** (consistently significant across specifications):
- 207766-V1 (Organized Voters): RD effect on congruent nonprofit funding (85% of specs significant)
- 214201-V1 (Mission Pakistan): Treatment effects on public sector performance

**Moderately Robust** (sensitive to specification choices):
- 116136-V2 (Divorce Laws): Effects vary by outcome variable and fixed effects structure
- 217741-V1 (AI Employment): Effects heterogeneous across demographic subgroups

**Fragile** (significance depends heavily on specification):
- 223321-V1 (Immigrant Age): Main effect becomes insignificant with full controls; mediated by intermarriage and English proficiency
- 230401-V1 (Minority Loans): Some coefficients flip sign across specifications

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/create_unified_csv.py` | Aggregates all specification_results.csv into unified output |
| `prompts/spec_search_agent.md` | Instructions for specification search agents |
| `prompts/CLEANUP.md` | Instructions for cleaning up data folders after analysis |
