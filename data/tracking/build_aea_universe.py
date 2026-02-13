"""
Build AEA_universe.jsonl: enumerate all AEA replication packages on openICPSR.

Strategy:
  1. DataCite query for all openICPSR DOIs (base + versioned) with an explicit
     relatedIdentifier linking to a 10.1257/* AEA paper DOI.
  2. Group by project ID, resolve latest version from versioned DOIs.
  3. Fetch BibTeX + structured metadata from Crossref for each AEA paper DOI.
  4. Write AEA_universe.jsonl.

All HTTP responses are cached in a local SQLite database so reruns are cheap.

Usage:
    python data/tracking/build_aea_universe.py
"""

import json
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DB = REPO_ROOT / "data" / "cache" / "datacite_responses.db"
OUTPUT_JSONL = REPO_ROOT / "data" / "tracking" / "AEA_universe.jsonl"

CROSSREF_MAILTO = "gsekeres@uchicago.edu"  # polite pool

# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------

def get_cache_conn():
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS http_cache (
            url TEXT PRIMARY KEY,
            response TEXT,
            fetched_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    return conn


def cached_get(url, conn, headers=None, force=False):
    """GET with SQLite cache. Returns response body as string."""
    if not force:
        row = conn.execute("SELECT response FROM http_cache WHERE url = ?", (url,)).fetchone()
        if row:
            return row[0]

    req = urllib.request.Request(url)
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    req.add_header("User-Agent", f"AEA-Universe-Builder/1.0 (mailto:{CROSSREF_MAILTO})")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} for {url}")
        body = json.dumps({"error": e.code, "url": url})
    except Exception as e:
        print(f"  Error fetching {url}: {e}")
        body = json.dumps({"error": str(e), "url": url})

    conn.execute(
        "INSERT OR REPLACE INTO http_cache (url, response) VALUES (?, ?)",
        (url, body),
    )
    conn.commit()
    return body


# ---------------------------------------------------------------------------
# DataCite pagination
# ---------------------------------------------------------------------------

DATACITE_BASE = "https://api.datacite.org/dois"


def datacite_query_all(query_params, conn, label=""):
    """Paginate through DataCite cursor-based results. Returns list of record dicts."""
    records = []
    page = 1
    params = dict(query_params)
    params.setdefault("page[size]", "1000")
    params["page[cursor]"] = "1"

    while True:
        qs = urllib.parse.urlencode(params, safe="*:()+")
        url = f"{DATACITE_BASE}?{qs}"
        print(f"  [{label}] Page {page}: {url[:120]}...")
        body = cached_get(url, conn)
        data = json.loads(body)

        if "error" in data and "data" not in data:
            print(f"  Error in response: {data}")
            break

        batch = data.get("data", [])
        records.extend(batch)
        print(f"    Got {len(batch)} records (total so far: {len(records)})")

        # Check for next page
        next_url = data.get("links", {}).get("next")
        if not next_url or len(batch) == 0:
            break

        # Extract cursor from next URL
        parsed = urllib.parse.urlparse(next_url)
        next_params = urllib.parse.parse_qs(parsed.query)
        if "page[cursor]" in next_params:
            params["page[cursor]"] = next_params["page[cursor]"][0]
        else:
            break
        page += 1
        time.sleep(0.3)

    total = data.get("meta", {}).get("total", "?")
    print(f"  [{label}] Done. {len(records)} records fetched (API reports {total} total)")
    return records


# ---------------------------------------------------------------------------
# Extract fields from DataCite records
# ---------------------------------------------------------------------------

def extract_aea_doi(record):
    """Search ALL relatedIdentifiers for a 10.1257/* DOI. Returns DOI string or None."""
    attrs = record.get("attributes", {})
    related = attrs.get("relatedIdentifiers") or []
    for ri in related:
        rid = (ri.get("relatedIdentifier") or "").strip().lower()
        if rid.startswith("10.1257/"):
            return rid
    return None


def extract_project_id(record):
    """Extract numeric project ID from DOI like 10.3886/e183985 or 10.3886/e183985v2."""
    doi = record.get("attributes", {}).get("doi", "") or record.get("id", "")
    doi = doi.lower()
    m = re.search(r"10\.3886/e(\d+)", doi)
    return m.group(1) if m else None


def extract_version(record):
    """Extract version number from DOI like 10.3886/e183985v2. Returns int or None."""
    doi = record.get("attributes", {}).get("doi", "") or record.get("id", "")
    doi = doi.lower()
    m = re.search(r"10\.3886/e\d+v(\d+)", doi)
    return int(m.group(1)) if m else None


def is_base_doi(record):
    """Check if this is a base DOI (no version suffix)."""
    doi = record.get("attributes", {}).get("doi", "") or record.get("id", "")
    return bool(re.match(r"^10\.3886/e\d+$", doi.lower()))


# ---------------------------------------------------------------------------
# Enumerate AEA packages from DataCite
# ---------------------------------------------------------------------------

def enumerate_packages(conn):
    """
    Fetch all openICPSR DOIs (base + versioned) that link to a 10.1257 AEA DOI.
    Groups by project ID and resolves latest version.
    Returns dict: project_id -> info dict.
    """
    print("\n=== DataCite: fetching all openICPSR DOIs linked to 10.1257/* ===")
    all_records = datacite_query_all(
        {
            "client-id": "gesis.icpsr",
            "query": "relatedIdentifiers.relatedIdentifier:10.1257*",
            "page[size]": "1000",
        },
        conn,
        label="aea-linked",
    )

    # Group by project ID, track max version and AEA DOI
    projects = {}
    for r in all_records:
        pid = extract_project_id(r)
        if not pid:
            continue

        aea_doi = extract_aea_doi(r)
        version = extract_version(r)
        base = is_base_doi(r)
        attrs = r.get("attributes", {})

        if pid not in projects:
            projects[pid] = {
                "project_id": pid,
                "aea_doi": None,
                "max_version": 1,
                "title": None,
                "authors": [],
                "year": None,
                "icpsr_doi": None,
                "icpsr_url": None,
            }

        p = projects[pid]

        # Collect AEA DOI from any record (base or versioned) for this project
        if aea_doi and not p["aea_doi"]:
            p["aea_doi"] = aea_doi

        # Track max version
        if version and version > p["max_version"]:
            p["max_version"] = version

        # Prefer metadata from base DOI record, fall back to any versioned record
        if base or p["title"] is None:
            title = (attrs.get("titles") or [{}])[0].get("title", "")
            title_clean = re.sub(
                r'^"?(Data\s+and\s+Code\s+for:\s*|Data\s+for:\s*|Code\s+for:\s*)',
                "",
                title,
                flags=re.IGNORECASE,
            ).strip().strip('"')

            creators = attrs.get("creators") or []
            authors = []
            for c in creators:
                parts = []
                if c.get("givenName"):
                    parts.append(c["givenName"])
                if c.get("familyName"):
                    parts.append(c["familyName"])
                if parts:
                    authors.append(" ".join(parts))
                elif c.get("name"):
                    authors.append(c["name"])

            p["title"] = title_clean or p["title"]
            p["authors"] = authors or p["authors"]
            p["year"] = attrs.get("publicationYear") or p["year"]
            p["icpsr_url"] = attrs.get("url") or p["icpsr_url"]
            if base:
                p["icpsr_doi"] = attrs.get("doi")

    # Keep only projects that have an AEA DOI
    aea_projects = {pid: p for pid, p in projects.items() if p["aea_doi"]}

    print(f"\n  Total unique projects: {len(projects)}")
    print(f"  With AEA DOI: {len(aea_projects)}")

    return aea_projects


# ---------------------------------------------------------------------------
# Fetch BibTeX + structured metadata from Crossref
# ---------------------------------------------------------------------------

def fetch_crossref_metadata(paper_doi, conn):
    """Fetch structured metadata and BibTeX for a paper DOI from Crossref."""
    result = {
        "journal": None,
        "volume": None,
        "issue": None,
        "pages": None,
        "bibtex": None,
        "cr_title": None,
        "cr_authors": [],
        "cr_year": None,
    }

    # Structured metadata
    url = f"https://api.crossref.org/works/{paper_doi}?mailto={CROSSREF_MAILTO}"
    body = cached_get(url, conn)
    try:
        data = json.loads(body)
        if "error" not in data:
            msg = data.get("message", {})
            result["journal"] = (msg.get("container-title") or [""])[0]
            result["volume"] = msg.get("volume")
            result["issue"] = msg.get("issue")
            result["pages"] = msg.get("page")
            result["cr_title"] = (msg.get("title") or [""])[0]
            cr_authors = []
            for a in msg.get("author") or []:
                parts = []
                if a.get("given"):
                    parts.append(a["given"])
                if a.get("family"):
                    parts.append(a["family"])
                if parts:
                    cr_authors.append(" ".join(parts))
            result["cr_authors"] = cr_authors
            for date_field in ["published-print", "published-online", "created"]:
                dp = msg.get(date_field, {}).get("date-parts", [[None]])[0]
                if dp and dp[0]:
                    result["cr_year"] = dp[0]
                    break
    except (json.JSONDecodeError, KeyError):
        pass

    # BibTeX via content negotiation
    bibtex_url = f"https://api.crossref.org/works/{paper_doi}/transform/application/x-bibtex"
    bibtex_body = cached_get(bibtex_url, conn, headers={"Accept": "application/x-bibtex"})
    if bibtex_body and not bibtex_body.startswith("{"):
        result["bibtex"] = bibtex_body.strip()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Building AEA Universe")
    print("=" * 60)

    conn = get_cache_conn()

    # Step 1: Enumerate all AEA packages from DataCite
    aea_projects = enumerate_packages(conn)

    # Step 2: Fetch Crossref metadata for each unique AEA paper DOI
    unique_dois = sorted(set(p["aea_doi"] for p in aea_projects.values()))
    print(f"\n=== Fetching Crossref metadata for {len(unique_dois)} unique AEA DOIs ===")

    crossref_cache = {}
    for i, doi in enumerate(unique_dois):
        cr = fetch_crossref_metadata(doi, conn)
        crossref_cache[doi] = cr
        if (i + 1) % 200 == 0:
            print(f"  Fetched {i+1}/{len(unique_dois)}")
        time.sleep(0.05)

    print(f"  Done fetching Crossref metadata")

    # Step 3: Build output records
    print(f"\n=== Building output records ===")
    records = []
    for pid in sorted(aea_projects.keys(), key=int):
        p = aea_projects[pid]
        cr = crossref_cache.get(p["aea_doi"], {})

        title = cr.get("cr_title") or p["title"] or ""
        authors = cr.get("cr_authors") or p["authors"] or []
        year = cr.get("cr_year") or p["year"]

        max_ver = p["max_version"]
        icpsr_url = (
            f"https://www.openicpsr.org/openicpsr/project/{pid}"
            f"/version/V{max_ver}/view"
        )

        record = {
            "paper_doi": p["aea_doi"],
            "icpsr_doi": p.get("icpsr_doi") or f"10.3886/e{pid}",
            "icpsr_project_id": pid,
            "icpsr_url": icpsr_url,
            "icpsr_version": f"V{max_ver}",
            "title": title,
            "authors": authors,
            "journal": cr.get("journal", ""),
            "year": year,
            "volume": cr.get("volume"),
            "issue": cr.get("issue"),
            "pages": cr.get("pages"),
            "bibtex": cr.get("bibtex"),
        }
        records.append(record)

    # Step 4: Write JSONL
    print(f"\n=== Writing {len(records)} records to {OUTPUT_JSONL} ===")
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary stats
    journals = {}
    for r in records:
        j = r["journal"] or "Unknown"
        journals[j] = journals.get(j, 0) + 1

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total AEA replication packages: {len(records)}")
    print(f"\nBy journal:")
    for j, count in sorted(journals.items(), key=lambda x: -x[1]):
        print(f"  {j}: {count}")

    versioned = sum(1 for r in records if r["icpsr_version"] != "V1")
    print(f"\nPackages with version > V1: {versioned}")
    missing_bibtex = sum(1 for r in records if not r.get("bibtex"))
    print(f"Missing BibTeX: {missing_bibtex}")

    conn.close()
    print(f"\nDone. Output: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
