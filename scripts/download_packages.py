"""
Download openICPSR replication packages with authenticated session.

Uses curl_cffi to impersonate Chrome's TLS fingerprint, bypassing Cloudflare.

openICPSR uses Keycloak SSO. Flow:
  GET /login -> redirect to Keycloak -> POST credentials ->
  follow redirects back -> authenticated session.

Download URL pattern:
  https://www.openicpsr.org/openicpsr/project/{PID}/version/{VER}/download/project
    ?dirPath=/openicpsr/{PID}/fcr:versions/{VER}

Usage:
    # Test login
    python scripts/download_packages.py --login-only

    # Dry run
    python scripts/download_packages.py --sample 5 --seed 42 --dry-run

    # Download specific packages
    python scripts/download_packages.py --project-ids 112431 113517

    # Download a random sample
    python scripts/download_packages.py --sample 10 --seed 42

    # Filter by journal/year
    python scripts/download_packages.py --journal "American Economic Review" --year-min 2015
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from getpass import getpass
from pathlib import Path

from curl_cffi import requests as cffi_requests

from download_utils import (
    COOKIE_PATH,
    DOWNLOAD_TRACKING,
    EXTRACTED_DIR,
    RAW_PACKAGES_DIR,
    DownloadResult,
    append_tracking,
    is_valid_zip,
    load_tracking,
    load_universe,
    make_paper_id,
)

ICPSR_BASE = "https://www.openicpsr.org"
DOWNLOAD_URL_TEMPLATE = (
    "{base}/openicpsr/project/{pid}/version/{ver}/download/project"
    "?dirPath=/openicpsr/{pid}/fcr:versions/{ver}"
)


# ---------------------------------------------------------------------------
# Keycloak auth session (curl_cffi with Chrome TLS impersonation)
# ---------------------------------------------------------------------------

class ICPSRSession:
    """Authenticated session for openICPSR using Keycloak SSO."""

    def __init__(self):
        self.session = cffi_requests.Session(impersonate="chrome")
        self.authenticated = False

    def login(self, email: str, password: str, max_attempts: int = 3) -> bool:
        """
        Authenticate via Keycloak SSO.

        Flow:
          1. GET /openicpsr/login -> redirects to Keycloak login page
          2. Parse form action URL from Keycloak HTML
          3. POST credentials to Keycloak
          4. Follow redirects back to openICPSR

        Retries with backoff if Keycloak rate-limits.
        """
        for attempt in range(max_attempts):
            if attempt > 0:
                wait = 5 * (2 ** (attempt - 1))
                print(f"  Retrying login in {wait}s...")
                time.sleep(wait)
                # Fresh session to clear any tainted cookies
                self.session = cffi_requests.Session(impersonate="chrome")

            print("Authenticating with openICPSR...")

            # Step 1: Hit the login endpoint, follow redirects to Keycloak
            login_url = f"{ICPSR_BASE}/openicpsr/login"
            try:
                resp = self.session.get(login_url, allow_redirects=True, timeout=30)
                resp.raise_for_status()
            except Exception as e:
                print(f"  ERROR: Could not reach login page: {e}")
                continue

            # Step 2: Parse the Keycloak form action URL
            action_match = re.search(r'action="([^"]*)"', resp.text)
            if not action_match:
                print("  ERROR: Could not find Keycloak login form action URL")
                print(f"  Final URL was: {resp.url}")
                continue

            keycloak_action = action_match.group(1).replace("&amp;", "&")
            print(f"  Keycloak endpoint: {keycloak_action[:80]}...")

            # Step 3: POST credentials
            try:
                resp = self.session.post(
                    keycloak_action,
                    data={"username": email, "password": password},
                    allow_redirects=True,
                    timeout=30,
                )
            except Exception as e:
                print(f"  ERROR: Login POST failed: {e}")
                continue

            if "Invalid username or password" in resp.text:
                print("  ERROR: Invalid credentials (may be rate-limited, retrying...)")
                continue

            # Step 4: Verify auth
            try:
                check = self.session.get(
                    f"{ICPSR_BASE}/openicpsr/workspace",
                    allow_redirects=False,
                    timeout=15,
                )
            except Exception as e:
                print(f"  WARNING: Auth check failed ({e}), assuming OK")
                self.authenticated = True
                return True

            if check.status_code == 200:
                self.authenticated = True
                print("  Login successful!")
                return True
            elif check.status_code == 302:
                location = check.headers.get("Location", "")
                if "login" in location.lower():
                    print("  ERROR: Authentication failed (redirected to login)")
                    continue
                self.authenticated = True
                print("  Login successful!")
                return True
            else:
                print(f"  WARNING: Unexpected status {check.status_code}, assuming auth OK")
                self.authenticated = True
                return True

        print("  ERROR: All login attempts failed")
        return False

    def download_project(
        self, project_id: str, version: str, dest_dir: Path
    ) -> DownloadResult:
        """Download a single project ZIP. Returns a DownloadResult."""
        paper_id = make_paper_id(project_id, version)
        ver = version if version.startswith("V") else f"V{version}"
        url = DOWNLOAD_URL_TEMPLATE.format(
            base=ICPSR_BASE, pid=project_id, ver=ver
        )

        result = DownloadResult(
            paper_id=paper_id,
            project_id=project_id,
            version=ver,
            download_url=url,
        )

        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{paper_id}.zip"

        try:
            # Stream download
            resp = self.session.get(url, stream=True, timeout=300)

            if resp.status_code == 404:
                result.download_status = "not_found"
                result.error = "HTTP 404"
                return result
            elif resp.status_code in (401, 403):
                result.download_status = "auth_failure"
                result.error = f"HTTP {resp.status_code}"
                return result
            elif resp.status_code >= 500:
                result.download_status = "server_error"
                result.error = f"HTTP {resp.status_code}"
                return result
            elif resp.status_code != 200:
                result.download_status = "server_error"
                result.error = f"HTTP {resp.status_code}"
                return result

            # Check content type — HTML means login redirect
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" in content_type:
                result.download_status = "auth_failure"
                result.error = "Got HTML instead of ZIP (likely redirected to login)"
                return result

            # Write to temp file with streaming hash
            tmp_path = dest_path.with_suffix(".zip.tmp")
            h = hashlib.sha256()
            total_size = 0
            try:
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
                        h.update(chunk)
                        total_size += len(chunk)
            except Exception as e:
                tmp_path.unlink(missing_ok=True)
                result.download_status = "network_error"
                result.error = str(e)
                return result

            # Validate ZIP
            if not is_valid_zip(tmp_path):
                tmp_path.unlink(missing_ok=True)
                result.download_status = "invalid_zip"
                result.error = "Downloaded file is not a valid ZIP"
                return result

            # Rename to final path
            tmp_path.rename(dest_path)

            result.download_status = "success"
            result.local_path = str(dest_path.relative_to(dest_dir.parent.parent.parent))
            result.sha256 = h.hexdigest()
            result.file_size_bytes = total_size
            return result

        except Exception as e:
            err_str = str(e)
            if "timeout" in err_str.lower():
                result.download_status = "timeout"
            else:
                result.download_status = "network_error"
            result.error = err_str
            return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_credentials():
    """Get ICPSR credentials from env vars or interactive prompt."""
    email = os.environ.get("ICPSR_EMAIL")
    password = os.environ.get("ICPSR_PASS")

    if not email:
        email = input("openICPSR email: ").strip()
    if not password:
        password = getpass("openICPSR password: ")

    return email, password


def select_packages(universe: list[dict], args) -> list[dict]:
    """Filter and select packages based on CLI args."""
    packages = list(universe)

    if args.project_ids:
        id_set = set(args.project_ids)
        packages = [p for p in packages if p["icpsr_project_id"] in id_set]

    if args.journal:
        j_lower = args.journal.lower()
        packages = [p for p in packages if j_lower in (p.get("journal") or "").lower()]

    if args.year_min:
        packages = [p for p in packages if (p.get("year") or 0) >= args.year_min]
    if args.year_max:
        packages = [p for p in packages if (p.get("year") or 9999) <= args.year_max]

    if args.sample and args.sample < len(packages):
        rng = random.Random(args.seed)
        packages = rng.sample(packages, args.sample)

    if args.batch_size and args.batch_size < len(packages):
        packages = packages[: args.batch_size]

    return packages


def main():
    parser = argparse.ArgumentParser(
        description="Download openICPSR replication packages"
    )
    parser.add_argument("--login-only", action="store_true", help="Test login and exit")
    parser.add_argument(
        "--project-ids", nargs="+", help="Download specific project IDs"
    )
    parser.add_argument("--sample", type=int, help="Random sample of N packages")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--journal", help="Filter by journal name (substring match)")
    parser.add_argument("--year-min", type=int, help="Minimum publication year")
    parser.add_argument("--year-max", type=int, help="Maximum publication year")
    parser.add_argument(
        "--delay", type=float, default=5.0, help="Seconds between downloads (default: 5)"
    )
    parser.add_argument("--batch-size", type=int, help="Stop after N successful downloads")
    parser.add_argument("--force", action="store_true", help="Re-download existing packages")
    parser.add_argument("--dry-run", action="store_true", help="Show what would download")
    parser.add_argument(
        "--skip-extract", action="store_true", help="Don't extract after download"
    )
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Max retry attempts (default: 3)"
    )
    args = parser.parse_args()

    # Load universe
    universe = load_universe()
    if not universe:
        print("ERROR: No universe data. Run build_aea_universe.py first.")
        sys.exit(1)
    print(f"Loaded {len(universe)} packages from universe")

    # Select packages
    packages = select_packages(universe, args)
    if not packages and not args.login_only:
        print("No packages match the given filters.")
        sys.exit(0)

    # Build set of already-downloaded paper IDs (from tracking)
    tracking = load_tracking()
    downloaded = set()
    for t in tracking:
        if t.get("download_status") == "success":
            downloaded.add(t["paper_id"])

    # Dry run — no auth needed
    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN: would download {len(packages)} packages")
        print(f"{'='*60}")
        for p in packages:
            pid = p["icpsr_project_id"]
            ver = p.get("icpsr_version", "V1")
            paper_id = make_paper_id(pid, ver)
            status = "SKIP (exists)" if paper_id in downloaded and not args.force else "DOWNLOAD"
            journal = p.get("journal", "?")
            year = p.get("year", "?")
            print(f"  [{status}] {paper_id}  {journal} ({year})  {p.get('title', '')[:60]}")
        return

    # Session setup
    sess = ICPSRSession()
    email, password = get_credentials()
    if not sess.login(email, password):
        print("Login failed. Exiting.")
        sys.exit(1)
    if args.login_only:
        print("Login test successful.")
        return

    # Download loop
    RAW_PACKAGES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Downloading {len(packages)} packages")
    print(f"{'='*60}")

    success_count = 0
    fail_count = 0
    skip_count = 0

    for i, pkg in enumerate(packages):
        pid = pkg["icpsr_project_id"]
        ver = pkg.get("icpsr_version", "V1")
        paper_id = make_paper_id(pid, ver)

        # Skip if already downloaded (unless --force)
        if paper_id in downloaded and not args.force:
            zip_path = RAW_PACKAGES_DIR / f"{paper_id}.zip"
            if zip_path.exists():
                print(f"[{i+1}/{len(packages)}] SKIP {paper_id} (already downloaded)")
                skip_count += 1
                continue

        print(f"[{i+1}/{len(packages)}] Downloading {paper_id}...")

        # Retry loop with exponential backoff
        result = None
        for attempt in range(args.max_retries):
            result = sess.download_project(pid, ver, RAW_PACKAGES_DIR)
            result.paper_doi = pkg.get("paper_doi")
            result.retry_count = attempt

            if result.download_status == "success":
                break

            # Re-authenticate once on auth failure
            if result.download_status == "auth_failure" and attempt == 0:
                print("  Auth failure, re-authenticating...")
                if sess.login(email, password):
                    continue
                else:
                    break

            # Don't retry on 404
            if result.download_status == "not_found":
                break

            # Exponential backoff for retryable errors
            if attempt < args.max_retries - 1:
                wait = (2 ** attempt) * 2 + random.uniform(0, 2)
                print(f"  Retry {attempt+1}/{args.max_retries} in {wait:.1f}s ({result.download_status}: {result.error})")
                time.sleep(wait)

        # Record result
        append_tracking(result)

        if result.download_status == "success":
            size_mb = (result.file_size_bytes or 0) / (1024 * 1024)
            print(f"  OK ({size_mb:.1f} MB)")
            success_count += 1
            downloaded.add(paper_id)

            # Extract if requested
            if not args.skip_extract:
                from extract_packages import extract_package

                zip_path = RAW_PACKAGES_DIR / f"{paper_id}.zip"
                dest = EXTRACTED_DIR / paper_id
                extract_package(zip_path, dest)
        else:
            print(f"  FAILED: {result.download_status} - {result.error}")
            fail_count += 1

        # Check batch size limit
        if args.batch_size and success_count >= args.batch_size:
            print(f"\nBatch size limit ({args.batch_size}) reached.")
            break

        # Rate limiting
        if i < len(packages) - 1:
            delay = args.delay + random.uniform(0, 2)
            time.sleep(delay)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Downloaded: {success_count}")
    print(f"  Failed:     {fail_count}")
    print(f"  Skipped:    {skip_count}")
    print(f"  Tracking:   {DOWNLOAD_TRACKING}")


if __name__ == "__main__":
    main()
