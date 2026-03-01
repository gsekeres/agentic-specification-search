"""
Shared constants and helpers for the OpenICPSR download pipeline.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_PACKAGES_DIR = REPO_ROOT / "data" / "downloads" / "raw_packages"
EXTRACTED_DIR = REPO_ROOT / "data" / "downloads" / "extracted"
UNIVERSE_JSONL = REPO_ROOT / "data" / "tracking" / "AEA_universe.jsonl"
DOWNLOAD_TRACKING = REPO_ROOT / "data" / "tracking" / "download_tracking.jsonl"
COOKIE_PATH = REPO_ROOT / "data" / "cache" / "session_cookies.json"

# ---------------------------------------------------------------------------
# Download result
# ---------------------------------------------------------------------------

@dataclass
class DownloadResult:
    paper_id: str
    project_id: str
    version: str
    paper_doi: Optional[str] = None
    download_url: Optional[str] = None
    download_status: str = "pending"  # success, not_found, auth_failure, server_error, network_error, invalid_zip, timeout
    local_path: Optional[str] = None
    sha256: Optional[str] = None
    file_size_bytes: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: Optional[str] = None
    retry_count: int = 0

    def to_dict(self):
        return asdict(self)

    def to_json_line(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_paper_id(project_id: str, version: str) -> str:
    """e.g., '112431' + 'V1' -> '112431-V1'"""
    v = version if version.startswith("V") else f"V{version}"
    return f"{project_id}-{v}"


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file, streaming."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid_zip(path: Path) -> bool:
    """Check ZIP magic bytes (PK\\x03\\x04)."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"PK\x03\x04"
    except (OSError, IOError):
        return False


def load_universe() -> list[dict]:
    """Load AEA_universe.jsonl. Returns list of record dicts."""
    if not UNIVERSE_JSONL.exists():
        return []
    records = []
    with open(UNIVERSE_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_tracking() -> list[dict]:
    """Load download_tracking.jsonl. Returns list of record dicts."""
    if not DOWNLOAD_TRACKING.exists():
        return []
    records = []
    with open(DOWNLOAD_TRACKING) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_tracking(result: DownloadResult):
    """Append a single download result to the tracking JSONL."""
    DOWNLOAD_TRACKING.parent.mkdir(parents=True, exist_ok=True)
    with open(DOWNLOAD_TRACKING, "a") as f:
        f.write(result.to_json_line() + "\n")
