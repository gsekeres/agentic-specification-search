"""
26_build_paper_catalog.py
Build a full paper catalog with citations for all 95 replicated papers.
Uses CrossRef API to look up author/journal/DOI info.
Outputs: BibTeX entries and LaTeX table for appendix G.3.
"""

import json, csv, re, time, urllib.request, urllib.parse, urllib.error
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "estimation" / "data"
META = ROOT / "data" / "metadata"
I4R  = ROOT / "i4r"
RESULTS = ROOT / "estimation" / "results"
OVERLEAF = Path(__file__).resolve().parents[3] / "overleaf" / "tex"

# ── 1. Read verification_summary to get our 95 paper_ids ──────────────────
verification = {}
with open(DATA / "verification_summary_by_paper.csv") as f:
    reader = csv.DictReader(f)
    for r in reader:
        verification[r["paper_id"]] = r

print(f"Verification papers: {len(verification)}")

# ── 2. Read claim_level.csv (40 I4R papers with title, journal, year) ─────
claims = {}
with open(DATA / "claim_level.csv") as f:
    reader = csv.DictReader(f)
    for r in reader:
        claims[r["paper_id"]] = r

# ── 3. Read sample metadata JSONLs ────────────────────────────────────────
sample_meta = {}
for fn in ["sample_40_new.jsonl", "sample_additional.jsonl", "sample_2025_to_download.jsonl"]:
    try:
        with open(META / fn) as f:
            for line in f:
                d = json.loads(line.strip())
                pid = d.get("paper_id", "")
                if pid:
                    sample_meta[pid] = d
    except FileNotFoundError:
        pass

# ── 4. Read papers_with_replication_urls.csv ──────────────────────────────
url_data = {}
with open(I4R / "papers_with_replication_urls.csv") as f:
    reader = csv.DictReader(f)
    for r in reader:
        url = r.get("replication_url", "")
        m = re.search(r"/project/(\d+)/version/(V\d+)/view", url)
        if m:
            pid = f"{m.group(1)}-{m.group(2)}"
            url_data[pid] = r

# ── 5. Read ICPSR metadata for remaining papers ──────────────────────────
icpsr_meta = {}
needed_bases = {pid.split("-")[0]: pid for pid in verification}
with open(META / "icpsr_openicpsr_packages.jsonl") as f:
    for line in f:
        d = json.loads(line.strip())
        base = d.get("base_id", "").replace("E", "")
        if base in needed_bases:
            pid = needed_bases[base]
            title = d.get("title", "")
            clean = re.sub(
                r'^(Data and [Cc]ode for:?\s*|Replication [Dd]ata for:?\s*|Supplementary data for:?\s*)',
                "", title
            ).strip().strip('"').strip()
            doi = d.get("package_doi", "")
            url = d.get("landing_page_url", "")
            if pid not in icpsr_meta or len(clean) > len(icpsr_meta[pid].get("title", "")):
                icpsr_meta[pid] = {"title": clean, "doi": doi, "url": url, "raw_title": title}

# ── 6. Join all sources ──────────────────────────────────────────────────
papers = {}
for pid in sorted(verification.keys()):
    title = ""
    journal = ""
    year = ""
    repl_doi = ""
    repl_url = ""
    n_specs = int(verification[pid].get("total_specs", 0))
    n_core = int(verification[pid].get("core_specs", 0))

    # Source: claim_level
    if pid in claims:
        title = claims[pid].get("title", "")
        journal = claims[pid].get("journal", "")
        year = claims[pid].get("year", "")

    # Source: sample_meta
    if pid in sample_meta:
        d = sample_meta[pid]
        if not title:
            t = d.get("title", "")
            t = re.sub(
                r'^(Data and [Cc]ode for:?\s*|Replication [Dd]ata for:?\s*|Supplementary data for:?\s*)',
                "", t
            ).strip().strip('"').strip()
            title = t
        if not journal:
            journal = d.get("journal", "")
        repl_doi = d.get("package_doi", "") or d.get("doi", "")
        repl_url = d.get("landing_page_url", "") or d.get("url", "")

    # Source: URL data
    if pid in url_data:
        d = url_data[pid]
        if not title:
            title = d.get("paper_title", "")
        if not journal:
            journal = d.get("journal_name", "")
        if not year:
            year = d.get("year", "")
        if not repl_url:
            repl_url = d.get("replication_url", "")

    # Source: ICPSR metadata
    if pid in icpsr_meta:
        d = icpsr_meta[pid]
        if not title:
            title = d["title"]
        if not repl_doi:
            repl_doi = d.get("doi", "")
        if not repl_url:
            repl_url = d.get("url", "")

    # Build replication URL from paper_id if still missing
    if not repl_url:
        base, ver = pid.split("-")
        repl_url = f"https://www.openicpsr.org/openicpsr/project/{base}/version/{ver}/view"

    papers[pid] = {
        "title": title,
        "journal": journal,
        "year": year,
        "repl_doi": repl_doi,
        "repl_url": repl_url,
        "n_specs": n_specs,
        "n_core": n_core,
        "in_sample_a": pid in claims,
    }

missing_title = [pid for pid, d in papers.items() if not d["title"]]
missing_journal = [pid for pid, d in papers.items() if not d["journal"]]
print(f"Papers: {len(papers)}, missing title: {len(missing_title)}, missing journal: {len(missing_journal)}")


# ── 7. CrossRef lookup ───────────────────────────────────────────────────
def crossref_lookup(title, journal_hint=""):
    """Query CrossRef by title, return best match dict or None."""
    q = urllib.parse.quote(title)
    url = f"https://api.crossref.org/works?query.bibliographic={q}&rows=3&mailto=gsekeres@uchicago.edu"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PaperCatalog/1.0 (mailto:gsekeres@uchicago.edu)"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        items = data.get("message", {}).get("items", [])
        if not items:
            return None

        # Pick best match: prefer exact title match
        best = None
        best_score = -1
        for item in items:
            cr_title = " ".join(item.get("title", [""])).lower().strip()
            our_title = title.lower().strip()
            # Simple similarity: check if titles are close
            score = item.get("score", 0)
            if cr_title == our_title:
                score += 1000
            elif our_title in cr_title or cr_title in our_title:
                score += 500
            if score > best_score:
                best_score = score
                best = item
        return best
    except Exception as e:
        print(f"  CrossRef error: {e}")
        return None


def format_authors_bibtex(item):
    """Extract authors from CrossRef item, return BibTeX author string."""
    authors = item.get("author", [])
    if not authors:
        return ""
    parts = []
    for a in authors:
        family = a.get("family", "")
        given = a.get("given", "")
        if family:
            parts.append(f"{family}, {given}" if given else family)
    return " and ".join(parts)


def format_authors_short(item):
    """Extract short author string (e.g., 'Smith and Jones' or 'Smith et al.')."""
    authors = item.get("author", [])
    if not authors:
        return ""
    families = [a.get("family", "") for a in authors if a.get("family")]
    if len(families) == 1:
        return families[0]
    elif len(families) == 2:
        return f"{families[0]} and {families[1]}"
    else:
        return f"{families[0]} et al."


cache_path = RESULTS / "crossref_cache.json"
if cache_path.exists():
    with open(cache_path) as f:
        crossref_cache = json.load(f)
    print(f"Loaded CrossRef cache: {len(crossref_cache)} entries")
else:
    crossref_cache = {}

print("\nQuerying CrossRef for author/journal/DOI info...")
for i, (pid, info) in enumerate(sorted(papers.items())):
    if pid in crossref_cache:
        continue
    title = info["title"]
    if not title:
        print(f"  [{i+1}/{len(papers)}] {pid}: NO TITLE, skipping")
        continue
    print(f"  [{i+1}/{len(papers)}] {pid}: {title[:60]}...")
    result = crossref_lookup(title, info.get("journal", ""))
    if result:
        crossref_cache[pid] = {
            "authors_bibtex": format_authors_bibtex(result),
            "authors_short": format_authors_short(result),
            "cr_title": " ".join(result.get("title", [""])),
            "cr_journal": result.get("container-title", [""])[0] if result.get("container-title") else "",
            "cr_year": str(result.get("published-print", result.get("published-online", result.get("issued", {})))
                         .get("date-parts", [[""]])[0][0]),
            "cr_doi": result.get("DOI", ""),
            "cr_volume": result.get("volume", ""),
            "cr_issue": result.get("issue", ""),
            "cr_pages": result.get("page", ""),
        }
    else:
        crossref_cache[pid] = None
        print(f"    -> NOT FOUND on CrossRef")
    time.sleep(0.3)  # Be polite

# Save cache
with open(cache_path, "w") as f:
    json.dump(crossref_cache, f, indent=2)
print(f"\nSaved CrossRef cache: {len(crossref_cache)} entries")

# ── 8. Fill in missing data from CrossRef ─────────────────────────────────
for pid, info in papers.items():
    cr = crossref_cache.get(pid)
    if cr is None:
        continue
    if not info["journal"] and cr.get("cr_journal"):
        info["journal"] = cr["cr_journal"]
    if not info["year"] and cr.get("cr_year"):
        info["year"] = cr["cr_year"]
    info["authors_bibtex"] = cr.get("authors_bibtex", "")
    info["authors_short"] = cr.get("authors_short", "")
    info["pub_doi"] = cr.get("cr_doi", "")
    info["cr_volume"] = cr.get("cr_volume", "")
    info["cr_issue"] = cr.get("cr_issue", "")
    info["cr_pages"] = cr.get("cr_pages", "")
    # Use CrossRef journal name if better
    if cr.get("cr_journal") and not info["journal"]:
        info["journal"] = cr["cr_journal"]


# ── 9. Generate BibTeX entries ────────────────────────────────────────────
def make_bib_key(pid, info):
    """Generate a BibTeX key like 'smith2022local'."""
    short = info.get("authors_short", "")
    first_author = short.split(" ")[0].split(",")[0] if short else pid.split("-")[0]
    first_author = re.sub(r"[^a-zA-Z]", "", first_author).lower()
    year = info.get("year", "")
    title_word = ""
    if info.get("title"):
        words = re.findall(r"[A-Za-z]+", info["title"])
        # Skip common words
        skip = {"the", "a", "an", "of", "in", "on", "and", "for", "to", "from", "with", "by", "at", "is", "are", "how", "do", "does", "can", "when", "why", "what", "evidence"}
        for w in words:
            if w.lower() not in skip:
                title_word = w.lower()
                break
    return f"{first_author}{year}{title_word}"


def escape_bibtex(s):
    """Escape special characters for BibTeX."""
    s = s.replace("&", r"\&")
    return s


bib_entries = []
for pid in sorted(papers.keys(), key=lambda p: (papers[p].get("journal", ""), papers[p].get("year", ""), p)):
    info = papers[pid]
    key = make_bib_key(pid, info)
    info["bib_key"] = key

    # Published paper entry
    authors = info.get("authors_bibtex", "")
    title = info.get("title", pid)
    journal = info.get("journal", "")
    year = info.get("year", "")
    doi = info.get("pub_doi", "")
    volume = info.get("cr_volume", "")
    issue = info.get("cr_issue", "")
    pages = info.get("cr_pages", "")

    entry = f"@article{{{key},\n"
    if authors:
        entry += f"  author = {{{authors}}},\n"
    entry += f"  title = {{{escape_bibtex(title)}}},\n"
    if journal:
        entry += f"  journal = {{{escape_bibtex(journal)}}},\n"
    if year:
        entry += f"  year = {{{year}}},\n"
    if volume:
        entry += f"  volume = {{{volume}}},\n"
    if issue:
        entry += f"  number = {{{issue}}},\n"
    if pages:
        entry += f"  pages = {{{pages}}},\n"
    if doi:
        entry += f"  doi = {{{doi}}},\n"
    entry += "}\n"
    bib_entries.append(entry)

    # Replication package entry
    repl_key = f"{key}_data"
    info["repl_bib_key"] = repl_key
    repl_doi = info.get("repl_doi", "")
    repl_url = info.get("repl_url", "")
    repl_entry = f"@misc{{{repl_key},\n"
    if authors:
        repl_entry += f"  author = {{{authors}}},\n"
    repl_entry += f"  title = {{Replication data for: {escape_bibtex(title)}}},\n"
    if year:
        repl_entry += f"  year = {{{year}}},\n"
    repl_entry += f"  publisher = {{openICPSR}},\n"
    if repl_doi:
        repl_entry += f"  doi = {{{repl_doi}}},\n"
    if repl_url:
        repl_entry += f"  url = {{{repl_url}}},\n"
    repl_entry += "}\n"
    bib_entries.append(repl_entry)

# Write BibTeX file
bib_path = OVERLEAF / "v8_sections" / "replicated_papers.bib"
with open(bib_path, "w") as f:
    f.write("% Auto-generated BibTeX entries for replicated papers\n")
    f.write("% Generated by 26_build_paper_catalog.py\n\n")
    f.write("\n".join(bib_entries))
print(f"\nWrote {len(bib_entries)} BibTeX entries to {bib_path}")


# ── 10. Generate LaTeX table ─────────────────────────────────────────────
# Standardize journal names
JOURNAL_MAP = {
    # AEA journals (short forms from our metadata)
    "AEJ: Applied": "AEJ: Applied",
    "AEJ: Policy": "AEJ: Policy",
    "AEJ: Macro": "AEJ: Macro",
    "AEJ: Micro": "AEJ: Micro",
    "AER": "AER",
    "AER: Insights": "AER: Insights",
    # AEA journals (full CrossRef names)
    "AEJ: Applied Economics": "AEJ: Applied",
    "AEJ: Economic Policy": "AEJ: Policy",
    "AEJ: Macroeconomics": "AEJ: Macro",
    "AEJ: Microeconomics": "AEJ: Micro",
    "American Economic Journal: Applied Economics": "AEJ: Applied",
    "American Economic Journal: Economic Policy": "AEJ: Policy",
    "American Economic Journal: Macroeconomics": "AEJ: Macro",
    "American Economic Journal: Microeconomics": "AEJ: Micro",
    "American Economic Review": "AER",
    "The American Economic Review": "AER",
    "American Economic Review: Insights": "AER: Insights",
    "AEA Papers and Proceedings": "AEA P\\&P",
    # Other top journals
    "The Quarterly Journal of Economics": "QJE",
    "QJE": "QJE",
    "Journal of Political Economy": "JPE",
    "The Journal of Political Economy": "JPE",
    "Review of Economic Studies": "REStud",
    "The Review of Economic Studies": "REStud",
    "The Economic Journal": "EJ",
    "Economic Journal": "EJ",
    "Econometrica": "ECMA",
    "American Journal of Political Science": "AJPS",
    "American Political Science Review": "APSR",
    "The Journal of Politics": "J. Politics",
    "Journal of Politics": "J. Politics",
    # CrossRef variants
    "AEA Randomized Controlled Trials": None,  # Skip, use our metadata
    "SSRN Electronic Journal": None,  # Skip
}


def standardize_journal(j):
    if j in JOURNAL_MAP:
        v = JOURNAL_MAP[j]
        return v if v is not None else j
    # Try partial match
    for k, v in JOURNAL_MAP.items():
        if v is not None and (k.lower() in j.lower() or j.lower() in k.lower()):
            return v
    return j


# Sort papers: by journal, then year, then title
sorted_pids = sorted(papers.keys(), key=lambda p: (
    standardize_journal(papers[p].get("journal", "ZZZ")),
    papers[p].get("year", "9999"),
    papers[p].get("title", ""),
))

# Build LaTeX longtable
lines = []
lines.append(r"\begin{longtable}{p{0.55\textwidth}lccp{0.08\textwidth}}")
lines.append(r"\caption{Full catalog of replicated papers. ``Sample~A'' marks the 40 papers with paired I4R reproductions. Specs (core) reports total verified-core specifications.}")
lines.append(r"\label{tab:paper_catalog} \\")
lines.append(r"\toprule")
lines.append(r"Paper & Journal & Year & Specs (core) & Sample~A \\")
lines.append(r"\midrule")
lines.append(r"\endfirsthead")
lines.append(r"\toprule")
lines.append(r"Paper & Journal & Year & Specs (core) & Sample~A \\")
lines.append(r"\midrule")
lines.append(r"\endhead")
lines.append(r"\midrule")
lines.append(r"\multicolumn{5}{r}{\emph{Continued on next page}} \\")
lines.append(r"\endfoot")
lines.append(r"\bottomrule")
lines.append(r"\endlastfoot")

for pid in sorted_pids:
    info = papers[pid]
    key = info.get("bib_key", "")
    repl_key = info.get("repl_bib_key", "")
    journal = standardize_journal(info.get("journal", ""))
    year = info.get("year", "")
    n_core = info.get("n_core", 0)
    sample_a = r"\checkmark" if info.get("in_sample_a") else ""

    # Format: \citet{key} [\href{repl_url}{data}]
    repl_url = info.get("repl_url", "")
    if key:
        paper_col = rf"\citet{{{key}}} [\href{{{repl_url}}}{{data}}]"
    else:
        title = info.get("title", pid)
        paper_col = rf"{title} [\href{{{repl_url}}}{{data}}]"

    lines.append(rf"{paper_col} & {journal} & {year} & {n_core} & {sample_a} \\")

lines.append(r"\end{longtable}")

table_path = OVERLEAF / "v8_tables" / "tab_paper_catalog.tex"
with open(table_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote LaTeX table to {table_path}")

# ── 11. Summary ──────────────────────────────────────────────────────────
n_with_authors = sum(1 for p in papers.values() if p.get("authors_bibtex"))
n_with_journal = sum(1 for p in papers.values() if p.get("journal"))
n_with_doi = sum(1 for p in papers.values() if p.get("pub_doi"))
print(f"\nSummary:")
print(f"  Papers: {len(papers)}")
print(f"  With authors: {n_with_authors}")
print(f"  With journal: {n_with_journal}")
print(f"  With pub DOI: {n_with_doi}")
print(f"  In Sample A: {sum(1 for p in papers.values() if p.get('in_sample_a'))}")

# Save full catalog as JSON for reference
catalog_path = RESULTS / "paper_catalog.json"
with open(catalog_path, "w") as f:
    json.dump(papers, f, indent=2, default=str)
print(f"  Saved catalog JSON: {catalog_path}")
