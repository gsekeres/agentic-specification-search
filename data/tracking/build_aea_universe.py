"""
Build AEA_universe.jsonl: enumerate all AEA replication packages on openICPSR.

Strategy:
  1. Scrape the openICPSR AEA search page to get all project IDs + titles + authors.
     (The search embeds a Solr JSON response in the HTML.)
  2. For each project, query DataCite for the AEA paper DOI and latest version.
  3. Fetch BibTeX + structured metadata from Crossref for each AEA paper DOI.
  4. Write AEA_universe.jsonl.

All HTTP responses are cached in a local SQLite database so reruns are cheap.

Usage:
    python data/tracking/build_aea_universe.py
    python data/tracking/build_aea_universe.py --force-refresh      # re-fetch openICPSR + DataCite
    python data/tracking/build_aea_universe.py --force-all          # re-fetch everything
    python data/tracking/build_aea_universe.py --stats-only         # print summary
    python data/tracking/build_aea_universe.py --incremental        # only new Crossref
    python data/tracking/build_aea_universe.py --max-cache-age-days 30
"""

import argparse
import json
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    from curl_cffi import requests as cffi_requests

    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DB = REPO_ROOT / "data" / "cache" / "datacite_responses.db"
OUTPUT_JSONL = REPO_ROOT / "data" / "tracking" / "AEA_universe.jsonl"

CROSSREF_MAILTO = "gsekeres@uchicago.edu"  # polite pool

ICPSR_SEARCH_URL = (
    "https://www.openicpsr.org/openicpsr/search/aea/studies"
    "?start={start}&ARCHIVE=aea&sort=score+desc&rows={rows}&q=*"
)

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


def cached_get(url, conn, headers=None, force=False, max_age_days=None):
    """GET with SQLite cache. Returns response body as string."""
    if not force:
        if max_age_days is not None:
            row = conn.execute(
                "SELECT response, fetched_at FROM http_cache WHERE url = ?", (url,)
            ).fetchone()
            if row:
                try:
                    fetched = datetime.fromisoformat(row[1])
                    age_days = (
                        datetime.now(timezone.utc)
                        - fetched.replace(tzinfo=timezone.utc)
                    ).days
                    if age_days <= max_age_days:
                        return row[0]
                except (TypeError, ValueError):
                    return row[0]  # if no timestamp, use cached
        else:
            row = conn.execute(
                "SELECT response FROM http_cache WHERE url = ?", (url,)
            ).fetchone()
            if row:
                return row[0]

    req = urllib.request.Request(url)
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    req.add_header(
        "User-Agent", f"AEA-Universe-Builder/1.0 (mailto:{CROSSREF_MAILTO})"
    )

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
        "INSERT OR REPLACE INTO http_cache (url, response, fetched_at) VALUES (?, ?, ?)",
        (url, body, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    return body


# ---------------------------------------------------------------------------
# Step 1: Scrape openICPSR AEA search to enumerate all project IDs
# ---------------------------------------------------------------------------


def _extract_solr_json(html):
    """Extract the embedded Solr JSON response from the openICPSR search HTML."""
    match = re.search(r"searchResults\s*:\s*(\{\"response\")", html)
    if not match:
        return None
    start = match.start(1)
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(html, start)
    return obj


def enumerate_icpsr_projects(conn, force=False):
    """
    Paginate through the openICPSR AEA search to collect all project IDs.
    Uses curl_cffi to bypass Cloudflare, with results cached in SQLite.
    Returns dict: project_id (str) -> {title, authors}.
    """
    if not HAS_CURL_CFFI:
        print("  WARNING: curl_cffi not installed. Cannot scrape openICPSR.")
        print("  Install with: pip install curl_cffi")
        return {}

    projects = {}
    page_size = 100
    start = 0

    print("\n=== Scraping openICPSR AEA search ===")

    while True:
        url = ICPSR_SEARCH_URL.format(start=start, rows=page_size)
        cache_key = f"icpsr_search:{start}:{page_size}"

        # Check cache
        cached = None
        if not force:
            row = conn.execute(
                "SELECT response FROM http_cache WHERE url = ?", (cache_key,)
            ).fetchone()
            if row:
                cached = row[0]

        if cached:
            html = cached
        else:
            sess = cffi_requests.Session(impersonate="chrome")
            resp = sess.get(url, timeout=60)
            if resp.status_code != 200:
                print(f"  HTTP {resp.status_code} at start={start}, stopping.")
                break
            html = resp.text
            conn.execute(
                "INSERT OR REPLACE INTO http_cache (url, response, fetched_at) VALUES (?, ?, ?)",
                (cache_key, html, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            time.sleep(1.5)  # polite delay

        data = _extract_solr_json(html)
        if not data:
            print(f"  Could not parse results at start={start}, stopping.")
            break

        num_found = data["response"].get("numFound", 0)
        docs = data["response"].get("docs", [])

        for doc in docs:
            pid = str(doc.get("ID", ""))
            if not pid:
                continue
            title_raw = doc.get("TITLE", "")
            title = re.sub(
                r'^"?(Data\s+and\s+Code\s+for:\s*|Data\s+for:\s*|Code\s+for:\s*|'
                r"Code\s+and\s+(?:Unrestricted\s+)?Data\s+for[:\s]*)",
                "",
                title_raw,
                flags=re.IGNORECASE,
            ).strip().strip('"')
            authors = doc.get("AUTHOR", [])
            projects[pid] = {"title": title, "authors": authors}

        print(
            f"  Page start={start}: got {len(docs)} docs "
            f"(total so far: {len(projects)}, numFound: {num_found})"
        )

        start += page_size
        if start >= num_found or len(docs) == 0:
            break

    print(f"  Done. {len(projects)} unique projects enumerated.")
    return projects


# ---------------------------------------------------------------------------
# Step 2: DataCite — get AEA paper DOI + version for each project
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
    print(
        f"  [{label}] Done. {len(records)} records fetched (API reports {total} total)"
    )
    return records


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


def build_datacite_index(conn):
    """
    Fetch all openICPSR DOIs linked to AEA papers from DataCite.
    Returns dict: project_id -> {aea_doi, max_version}.
    """
    print("\n=== DataCite: fetching openICPSR DOIs linked to 10.1257/* ===")
    all_records = datacite_query_all(
        {
            "client-id": "gesis.icpsr",
            "query": "relatedIdentifiers.relatedIdentifier:10.1257*",
            "page[size]": "1000",
        },
        conn,
        label="aea-linked",
    )

    index = {}
    for r in all_records:
        pid = extract_project_id(r)
        if not pid:
            continue
        aea_doi = extract_aea_doi(r)
        version = extract_version(r)

        if pid not in index:
            index[pid] = {"aea_doi": None, "max_version": 1}
        if aea_doi and not index[pid]["aea_doi"]:
            index[pid]["aea_doi"] = aea_doi
        if version and version > index[pid]["max_version"]:
            index[pid]["max_version"] = version

    print(f"  Indexed {len(index)} projects from DataCite")
    return index


def datacite_lookup_single(pid, conn):
    """Look up a single project on DataCite to find AEA DOI + version."""
    # Try base DOI
    url = f"https://api.datacite.org/dois/10.3886/e{pid}"
    body = cached_get(url, conn)
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None, 1

    if "error" in data and "data" not in data:
        return None, 1

    record = data.get("data", {})
    attrs = record.get("attributes", {})

    # Check relatedIdentifiers for AEA DOI
    aea_doi = None
    for ri in attrs.get("relatedIdentifiers") or []:
        rid = (ri.get("relatedIdentifier") or "").strip().lower()
        if rid.startswith("10.1257/"):
            aea_doi = rid
            break

    # Check for versioned DOIs to find max version
    max_version = 1
    for ri in attrs.get("relatedIdentifiers") or []:
        rid = (ri.get("relatedIdentifier") or "").strip().lower()
        m = re.search(r"10\.3886/e\d+v(\d+)", rid)
        if m:
            v = int(m.group(1))
            if v > max_version:
                max_version = v

    return aea_doi, max_version


# ---------------------------------------------------------------------------
# Step 3: Crossref metadata
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
    bibtex_url = (
        f"https://api.crossref.org/works/{paper_doi}/transform/application/x-bibtex"
    )
    bibtex_body = cached_get(bibtex_url, conn, headers={"Accept": "application/x-bibtex"})
    if bibtex_body and not bibtex_body.startswith("{"):
        result["bibtex"] = bibtex_body.strip()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def print_stats(records=None):
    """Print summary stats from existing JSONL or provided records."""
    if records is None:
        if not OUTPUT_JSONL.exists():
            print("No universe file found. Run without --stats-only first.")
            return
        records = []
        with open(OUTPUT_JSONL) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    journals = {}
    for r in records:
        j = r.get("journal") or "Unknown"
        journals[j] = journals.get(j, 0) + 1

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total AEA replication packages: {len(records)}")
    print(f"\nBy journal:")
    for j, count in sorted(journals.items(), key=lambda x: -x[1]):
        print(f"  {j}: {count}")

    versioned = sum(1 for r in records if r.get("icpsr_version", "V1") != "V1")
    print(f"\nPackages with version > V1: {versioned}")
    missing_doi = sum(1 for r in records if not r.get("paper_doi"))
    print(f"Missing paper DOI: {missing_doi}")
    missing_bibtex = sum(1 for r in records if not r.get("bibtex"))
    print(f"Missing BibTeX: {missing_bibtex}")

    # Year distribution
    years = {}
    for r in records:
        y = r.get("year")
        if y:
            years[y] = years.get(y, 0) + 1
    if years:
        print("\nBy year (top 10):")
        for y, count in sorted(years.items(), key=lambda x: -x[1])[:10]:
            print(f"  {y}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Build AEA_universe.jsonl from openICPSR + DataCite + Crossref"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-fetch openICPSR + DataCite pages (bypass cache)",
    )
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Re-fetch everything (openICPSR + DataCite + Crossref, bypass all cache)",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only fetch Crossref metadata for project IDs not already in output",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print summary from existing JSONL and exit",
    )
    parser.add_argument(
        "--max-cache-age-days",
        type=int,
        default=None,
        help="Treat cache entries older than N days as stale",
    )
    args = parser.parse_args()

    if args.stats_only:
        print_stats()
        return

    force_icpsr = args.force_refresh or args.force_all
    force_crossref = args.force_all
    max_age = args.max_cache_age_days

    print("=" * 60)
    print("Building AEA Universe")
    if force_icpsr:
        print("  (force-refreshing openICPSR + DataCite)")
    if force_crossref:
        print("  (force-refreshing all Crossref queries)")
    if max_age is not None:
        print(f"  (max cache age: {max_age} days)")
    print("=" * 60)

    conn = get_cache_conn()

    # If force-refreshing, invalidate relevant cache entries
    if force_icpsr:
        deleted_icpsr = conn.execute(
            "DELETE FROM http_cache WHERE url LIKE 'icpsr_search:%'"
        ).rowcount
        deleted_dc = conn.execute(
            "DELETE FROM http_cache WHERE url LIKE '%api.datacite.org%'"
        ).rowcount
        conn.commit()
        print(
            f"  Cleared {deleted_icpsr} openICPSR + {deleted_dc} DataCite cache entries"
        )

    # Step 1: Enumerate all AEA projects from openICPSR search
    icpsr_projects = enumerate_icpsr_projects(conn, force=force_icpsr)

    # Step 2: Build DataCite index (AEA DOIs + versions)
    dc_index = build_datacite_index(conn)

    # For projects not in DataCite index, do individual lookups
    missing_from_dc = set(icpsr_projects.keys()) - set(dc_index.keys())
    if missing_from_dc:
        print(
            f"\n=== DataCite: individual lookups for {len(missing_from_dc)} "
            f"projects not in bulk index ==="
        )
        looked_up = 0
        for pid in sorted(missing_from_dc, key=int):
            aea_doi, max_ver = datacite_lookup_single(pid, conn)
            dc_index[pid] = {"aea_doi": aea_doi, "max_version": max_ver}
            looked_up += 1
            if looked_up % 100 == 0:
                print(f"  Looked up {looked_up}/{len(missing_from_dc)}")
            time.sleep(0.05)
        print(f"  Done. Looked up {looked_up} projects.")

    # Step 3: Fetch Crossref metadata for projects that have an AEA DOI
    all_pids = sorted(icpsr_projects.keys(), key=int)
    unique_dois = sorted(
        set(
            dc_index[pid]["aea_doi"]
            for pid in all_pids
            if pid in dc_index and dc_index[pid].get("aea_doi")
        )
    )

    # In incremental mode, skip DOIs we already have Crossref data for
    if args.incremental and OUTPUT_JSONL.exists():
        existing_dois = set()
        with open(OUTPUT_JSONL) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if rec.get("journal"):
                        existing_dois.add(rec["paper_doi"])
        new_dois = [d for d in unique_dois if d not in existing_dois]
        print(
            f"\n=== Incremental: {len(new_dois)} new DOIs "
            f"(skipping {len(unique_dois) - len(new_dois)} cached) ==="
        )
        dois_to_fetch = new_dois
    else:
        dois_to_fetch = unique_dois

    print(f"\n=== Fetching Crossref metadata for {len(dois_to_fetch)} AEA DOIs ===")

    crossref_cache = {}
    for i, doi in enumerate(dois_to_fetch):
        cr = fetch_crossref_metadata(doi, conn)
        crossref_cache[doi] = cr
        if (i + 1) % 200 == 0:
            print(f"  Fetched {i+1}/{len(dois_to_fetch)}")
        time.sleep(0.05)

    # For incremental mode, also load existing Crossref data for skipped DOIs
    if args.incremental and OUTPUT_JSONL.exists():
        with open(OUTPUT_JSONL) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                doi = rec.get("paper_doi")
                if doi and doi not in crossref_cache:
                    crossref_cache[doi] = {
                        "journal": rec.get("journal"),
                        "volume": rec.get("volume"),
                        "issue": rec.get("issue"),
                        "pages": rec.get("pages"),
                        "bibtex": rec.get("bibtex"),
                        "cr_title": rec.get("title"),
                        "cr_authors": rec.get("authors", []),
                        "cr_year": rec.get("year"),
                    }

    # Ensure all DOIs have Crossref data (fetch any remaining)
    for doi in unique_dois:
        if doi not in crossref_cache:
            cr = fetch_crossref_metadata(doi, conn)
            crossref_cache[doi] = cr
            time.sleep(0.05)

    print("  Done fetching Crossref metadata")

    # Step 4: Build output records
    print(f"\n=== Building output records ===")

    with_doi = 0
    without_doi = 0
    records = []
    for pid in all_pids:
        icpsr = icpsr_projects[pid]
        dc = dc_index.get(pid, {})
        aea_doi = dc.get("aea_doi")
        max_ver = dc.get("max_version", 1)
        cr = crossref_cache.get(aea_doi, {}) if aea_doi else {}

        title = cr.get("cr_title") or icpsr["title"] or ""
        authors = cr.get("cr_authors") or icpsr["authors"] or []
        year = cr.get("cr_year")

        icpsr_url = (
            f"https://www.openicpsr.org/openicpsr/project/{pid}"
            f"/version/V{max_ver}/view"
        )

        record = {
            "paper_doi": aea_doi,
            "icpsr_doi": f"10.3886/e{pid}",
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

        if aea_doi:
            with_doi += 1
        else:
            without_doi += 1

    print(f"  With AEA paper DOI: {with_doi}")
    print(f"  Without AEA paper DOI: {without_doi}")

    # Step 5: Write JSONL
    print(f"\n=== Writing {len(records)} records to {OUTPUT_JSONL} ===")
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print_stats(records)

    conn.close()
    print(f"\nDone. Output: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
