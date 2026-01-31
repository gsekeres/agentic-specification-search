"""
DataCite API Fetcher for AEA openICPSR Deposits

Fetches DOIs and metadata for AEA replication packages via the DataCite API.
This avoids scraping the JavaScript-heavy openICPSR search page.

Usage:
    python datacite_fetcher.py --limit 100 --output metadata.jsonl
"""

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


@dataclass
class AEAPackageMetadata:
    """Metadata for an AEA openICPSR replication package."""
    doi: str
    project_id: str
    version: str
    title: str
    publication_year: int
    publisher: str
    resource_type: str
    creators: list[str]
    subjects: list[str]
    description: str
    url: str
    download_url: str
    registered: str
    updated: str
    # Classification fields (to be filled by agent)
    is_observational: Optional[bool] = None
    has_public_data: Optional[bool] = None
    primary_software: Optional[str] = None
    journal: Optional[str] = None
    data_size_mb: Optional[float] = None


def parse_doi(doi: str) -> tuple[str, str]:
    """
    Parse an AEA openICPSR DOI to extract project_id and version.

    Example: 10.3886/E114448V1 -> ('114448', 'V1')
    """
    match = re.search(r'10\.3886/E(\d+)(V\d+)?$', doi, re.IGNORECASE)
    if not match:
        return ('', '')
    project_id = match.group(1)
    version = match.group(2) or 'V1'
    return (project_id, version)


def build_download_url(project_id: str, version: str) -> str:
    """Build the openICPSR download terms URL for a project."""
    return (
        f"https://www.openicpsr.org/openicpsr/project/{project_id}/version/{version}"
        f"/download/terms?path=/openicpsr/{project_id}/fcr:versions/{version}&type=project"
    )


def build_project_url(project_id: str) -> str:
    """Build the openICPSR project page URL."""
    return f"https://www.openicpsr.org/openicpsr/project/{project_id}/version/V1/view"


class DataCiteFetcher:
    """Fetches AEA replication package metadata from DataCite API."""

    BASE_URL = "https://api.datacite.org/dois"
    PAGE_SIZE = 100  # DataCite max is 1000

    def __init__(self, rate_limit_seconds: float = 0.5):
        self.rate_limit = rate_limit_seconds
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, url: str) -> dict:
        """Make a rate-limited request to the DataCite API."""
        self._rate_limit()

        req = Request(url)
        req.add_header('Accept', 'application/json')
        req.add_header('User-Agent', 'AEA-Spec-Curve-Pipeline/1.0')

        try:
            with urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode('utf-8'))
        except HTTPError as e:
            print(f"HTTP Error {e.code}: {url}")
            raise
        except URLError as e:
            print(f"URL Error: {e.reason}")
            raise

    def _is_aea_package(self, attrs: dict) -> bool:
        """
        Check if a DataCite record is an AEA replication package.

        AEA packages typically have titles like:
        - "Replication data for: <paper title>"
        - "Data and Code for: <paper title>"
        - "Replication Package for '<paper title>'"

        And reference AEA journals (AER, AEJ, JEL, JEP, etc.)
        """
        titles = attrs.get('titles', [])
        title = titles[0].get('title', '').lower() if titles else ''

        # Check for replication package patterns
        aea_patterns = [
            'replication data for:',
            'replication package for',
            'data and code for:',
            'replication files for:',
            'data for:',  # Common AEA pattern
        ]

        # Check for AEA journal references
        aea_journals = [
            'american economic review',
            'aer insights',
            'aej: applied',
            'aej: economic policy',
            'aej: macroeconomics',
            'aej: microeconomics',
            'journal of economic literature',
            'journal of economic perspectives',
        ]

        # Title-based detection
        for pattern in aea_patterns:
            if pattern in title:
                return True

        # Journal reference detection
        for journal in aea_journals:
            if journal in title:
                return True

        # Check description
        descriptions = attrs.get('descriptions', [])
        for desc in descriptions:
            desc_text = desc.get('description', '').lower() if isinstance(desc, dict) else str(desc).lower()
            for journal in aea_journals:
                if journal in desc_text:
                    return True

        return False

    def fetch_aea_dois(
        self,
        limit: Optional[int] = None,
        sort: str = "-created",
        resource_type: Optional[str] = None
    ) -> list[AEAPackageMetadata]:
        """
        Fetch AEA openICPSR DOIs from DataCite.

        Strategy: Query all 10.3886 (ICPSR) DOIs and filter client-side
        for AEA replication packages based on title patterns.

        Args:
            limit: Maximum number of records to fetch (None = all)
            sort: Sort order ('-created' = newest first, 'created' = oldest first)
            resource_type: Filter by resource type (e.g., 'Dataset')

        Returns:
            List of AEAPackageMetadata objects
        """
        # Query for ICPSR prefix with "replication" in title (catches most AEA packages)
        # We'll filter more precisely client-side
        query = 'prefix:10.3886 AND titles.title:replication'

        if resource_type:
            query += f' AND types.resourceTypeGeneral:"{resource_type}"'

        results = []
        page = 1
        total_pages = None
        seen_dois = set()  # Avoid duplicates across versions

        while True:
            # Build URL with pagination - use URL encoding
            encoded_query = query.replace(' ', '%20').replace(':', '%3A').replace('"', '%22')
            url = (
                f"{self.BASE_URL}?"
                f"query={encoded_query}"
                f"&page[size]={self.PAGE_SIZE}"
                f"&page[number]={page}"
                f"&sort={sort}"
            )

            print(f"Fetching page {page}..." + (f" of {total_pages}" if total_pages else ""))

            try:
                data = self._make_request(url)
            except Exception as e:
                print(f"Error fetching page {page}: {e}")
                break

            # Parse metadata from results
            for item in data.get('data', []):
                attrs = item.get('attributes', {})
                doi = attrs.get('doi', '')

                project_id, version = parse_doi(doi)
                if not project_id:
                    continue

                # Skip if we've already seen this project (different versions)
                base_doi = f"10.3886/E{project_id}"
                if base_doi in seen_dois:
                    continue
                seen_dois.add(base_doi)

                # Filter for AEA packages (client-side filtering)
                if not self._is_aea_package(attrs):
                    continue

                # Extract creators
                creators = []
                for creator in attrs.get('creators', []):
                    name = creator.get('name', '')
                    if not name:
                        given = creator.get('givenName', '')
                        family = creator.get('familyName', '')
                        name = f"{given} {family}".strip()
                    if name:
                        creators.append(name)

                # Extract subjects/keywords
                subjects = []
                for subject in attrs.get('subjects', []):
                    if isinstance(subject, dict):
                        subjects.append(subject.get('subject', ''))
                    else:
                        subjects.append(str(subject))

                # Extract description
                descriptions = attrs.get('descriptions', [])
                description = ''
                for desc in descriptions:
                    if isinstance(desc, dict):
                        description = desc.get('description', '')
                        break
                    else:
                        description = str(desc)
                        break

                metadata = AEAPackageMetadata(
                    doi=doi,
                    project_id=project_id,
                    version=version,
                    title=attrs.get('titles', [{}])[0].get('title', '') if attrs.get('titles') else '',
                    publication_year=attrs.get('publicationYear', 0),
                    publisher=attrs.get('publisher', ''),
                    resource_type=attrs.get('types', {}).get('resourceTypeGeneral', ''),
                    creators=creators,
                    subjects=subjects,
                    description=description[:1000] if description else '',  # Truncate long descriptions
                    url=build_project_url(project_id),
                    download_url=build_download_url(project_id, version),
                    registered=attrs.get('registered', ''),
                    updated=attrs.get('updated', ''),
                )

                results.append(metadata)

            # Check pagination
            meta = data.get('meta', {})
            total = meta.get('total', 0)
            total_pages = (total + self.PAGE_SIZE - 1) // self.PAGE_SIZE

            print(f"  Got {len(data.get('data', []))} records, total: {total}")

            # Check if we've reached the limit
            if limit and len(results) >= limit:
                results = results[:limit]
                break

            # Check if we've fetched all pages
            if page >= total_pages:
                break

            page += 1

        return results

    def fetch_recent(self, n: int = 100) -> list[AEAPackageMetadata]:
        """Fetch the n most recently registered AEA packages."""
        return self.fetch_aea_dois(limit=n, sort="-created")


def save_metadata(
    metadata_list: list[AEAPackageMetadata],
    output_path: Path,
    format: str = "jsonl"
):
    """Save metadata to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        with open(output_path, 'w') as f:
            for m in metadata_list:
                f.write(json.dumps(asdict(m)) + '\n')
    elif format == "json":
        with open(output_path, 'w') as f:
            json.dump([asdict(m) for m in metadata_list], f, indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")

    print(f"Saved {len(metadata_list)} records to {output_path}")


def load_metadata(path: Path) -> list[AEAPackageMetadata]:
    """Load metadata from file."""
    if path.suffix == '.jsonl':
        metadata = []
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                metadata.append(AEAPackageMetadata(**data))
        return metadata
    elif path.suffix == '.json':
        with open(path) as f:
            data = json.load(f)
            return [AEAPackageMetadata(**d) for d in data]
    else:
        raise ValueError(f"Unknown file format: {path.suffix}")


def main():
    parser = argparse.ArgumentParser(
        description='Fetch AEA openICPSR metadata from DataCite API'
    )
    parser.add_argument(
        '--limit', '-n',
        type=int,
        default=100,
        help='Maximum number of records to fetch (default: 100)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('data/metadata/aea_packages.jsonl'),
        help='Output file path (default: data/metadata/aea_packages.jsonl)'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['json', 'jsonl'],
        default='jsonl',
        help='Output format (default: jsonl)'
    )
    parser.add_argument(
        '--sort',
        choices=['newest', 'oldest'],
        default='newest',
        help='Sort order (default: newest first)'
    )

    args = parser.parse_args()

    sort = '-created' if args.sort == 'newest' else 'created'

    print(f"Fetching up to {args.limit} AEA packages from DataCite...")
    print(f"Sort: {args.sort} first")
    print()

    fetcher = DataCiteFetcher()
    metadata = fetcher.fetch_aea_dois(limit=args.limit, sort=sort)

    print()
    print(f"Fetched {len(metadata)} packages")

    # Print summary
    years = [m.publication_year for m in metadata if m.publication_year]
    if years:
        print(f"Publication years: {min(years)} - {max(years)}")

    # Save
    save_metadata(metadata, args.output, args.format)


if __name__ == '__main__':
    main()
