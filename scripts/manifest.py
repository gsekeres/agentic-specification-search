"""
Download Manifest Manager

Tracks download status for AEA replication packages.
Supports resuming interrupted download sessions.

Usage:
    from download.manifest import DownloadManifest

    manifest = DownloadManifest("data/downloads/manifest.jsonl")
    manifest.record_download(doi, status="downloaded", filename="...", bytes=123)
    manifest.save()
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class DownloadRecord:
    """Record of a single download attempt."""
    doi: str
    project_id: str
    version: str
    status: str  # downloaded, restricted, error, missing, skipped
    filename: Optional[str] = None
    bytes: Optional[int] = None
    timestamp: str = ""
    error: Optional[str] = None
    duration_seconds: Optional[float] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"


class DownloadManifest:
    """Manages download status tracking."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.records: dict[str, DownloadRecord] = {}  # keyed by DOI
        self._load()

    def _load(self):
        """Load existing manifest from disk."""
        if not self.path.exists():
            return

        with open(self.path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    record = DownloadRecord(**data)
                    self.records[record.doi] = record

    def save(self):
        """Save manifest to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.path, 'w') as f:
            for record in self.records.values():
                f.write(json.dumps(asdict(record)) + '\n')

    def record_download(
        self,
        doi: str,
        project_id: str,
        version: str,
        status: str,
        filename: Optional[str] = None,
        bytes: Optional[int] = None,
        error: Optional[str] = None,
        duration_seconds: Optional[float] = None
    ):
        """Record a download attempt."""
        record = DownloadRecord(
            doi=doi,
            project_id=project_id,
            version=version,
            status=status,
            filename=filename,
            bytes=bytes,
            error=error,
            duration_seconds=duration_seconds
        )
        self.records[doi] = record
        # Auto-save after each record
        self.save()

    def get_status(self, doi: str) -> Optional[str]:
        """Get status for a DOI, or None if not in manifest."""
        record = self.records.get(doi)
        return record.status if record else None

    def is_downloaded(self, doi: str) -> bool:
        """Check if a DOI has been successfully downloaded."""
        return self.get_status(doi) == "downloaded"

    def get_pending(self, all_dois: list[str]) -> list[str]:
        """Get DOIs that haven't been attempted yet."""
        return [doi for doi in all_dois if doi not in self.records]

    def get_failed(self) -> list[str]:
        """Get DOIs that failed and might be worth retrying."""
        return [
            doi for doi, record in self.records.items()
            if record.status == "error"
        ]

    def summary(self) -> dict:
        """Get summary statistics."""
        statuses = {}
        total_bytes = 0

        for record in self.records.values():
            statuses[record.status] = statuses.get(record.status, 0) + 1
            if record.bytes:
                total_bytes += record.bytes

        return {
            "total": len(self.records),
            "by_status": statuses,
            "total_bytes": total_bytes,
            "total_gb": round(total_bytes / (1024**3), 2)
        }

    def print_summary(self):
        """Print a human-readable summary."""
        summary = self.summary()
        print(f"\nDownload Manifest Summary")
        print("=" * 40)
        print(f"Total records: {summary['total']}")
        print(f"Total size: {summary['total_gb']:.2f} GB")
        print()
        print("By status:")
        for status, count in sorted(summary['by_status'].items()):
            print(f"  {status}: {count}")


def main():
    """Test manifest functionality."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "test_manifest.jsonl"
        manifest = DownloadManifest(manifest_path)

        # Record some downloads
        manifest.record_download(
            doi="10.3886/E114448V1",
            project_id="114448",
            version="V1",
            status="downloaded",
            filename="114448_V1.zip",
            bytes=15_000_000
        )

        manifest.record_download(
            doi="10.3886/E114449V1",
            project_id="114449",
            version="V1",
            status="restricted"
        )

        manifest.record_download(
            doi="10.3886/E114450V1",
            project_id="114450",
            version="V1",
            status="error",
            error="Timeout after 60s"
        )

        manifest.print_summary()

        # Test loading
        manifest2 = DownloadManifest(manifest_path)
        assert len(manifest2.records) == 3
        assert manifest2.is_downloaded("10.3886/E114448V1")
        assert not manifest2.is_downloaded("10.3886/E114449V1")

        print("\nManifest tests passed!")


if __name__ == '__main__':
    main()
