#!/usr/bin/env python3
"""
Check if eCFR Title 8 has been updated since the last ingestion.

=== HOW IT WORKS ===

The eCFR API exposes a /titles endpoint that returns metadata for every CFR title,
including a `latest_issue_date` field. We compare this date against the one we stored
during the last ingestion. If the date is newer, regulations have changed and we need
to re-ingest.

This script is designed to be called from a GitHub Actions workflow:
  - Exit code 0 + prints "CHANGED"  → re-ingestion needed
  - Exit code 0 + prints "UNCHANGED" → no action needed
  - Exit code 1                      → error (API down, etc.)

The last-known date is stored in a simple text file:
    knowledge_base/last_ecfr_update.txt

=== WHY NOT JUST RE-INGEST EVERY TIME? ===

Re-ingestion involves:
  1. Fetching ~2MB of XML from eCFR
  2. Parsing and chunking ~400+ sections
  3. Embedding all chunks (~30 seconds of GPU/CPU time)
  4. Committing the updated index to git

That's wasted work if nothing changed. The check is a single lightweight API call
(~200ms, ~5KB response). Title 8 typically updates a few times per year, so ~99%
of weekly checks will be "UNCHANGED" and skip all that work.
"""

import sys
import os
import json
import logging
import requests
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
LAST_UPDATE_FILE = PROJECT_ROOT / "knowledge_base" / "last_ecfr_update.txt"

TITLE_NUMBER = 8  # Aliens and Nationality


def get_ecfr_title_date(title_number: int = TITLE_NUMBER) -> str:
    """
    Fetch the latest issue date for a CFR title from the eCFR API.

    The /titles endpoint returns an array of all 50 CFR titles with metadata.
    We find Title 8 and return its latest_issue_date.

    Returns:
        Date string in YYYY-MM-DD format

    Raises:
        RuntimeError if the API is unreachable or title not found
    """
    url = "https://www.ecfr.gov/api/versioner/v1/titles"
    logger.info(f"Checking eCFR for Title {title_number} latest issue date...")

    response = requests.get(url, timeout=15)
    response.raise_for_status()
    data = response.json()

    for title in data.get("titles", []):
        if title.get("number") == title_number:
            date = title.get("latest_issue_date") or title.get("up_to_date_as_of")
            if date:
                logger.info(f"eCFR Title {title_number} latest issue date: {date}")
                return date

    raise RuntimeError(f"Title {title_number} not found in eCFR /titles response")


def get_stored_date() -> str | None:
    """
    Read the last-known eCFR update date from disk.

    Returns None if the file doesn't exist (first run).
    """
    if LAST_UPDATE_FILE.exists():
        stored = LAST_UPDATE_FILE.read_text().strip()
        logger.info(f"Stored last update date: {stored}")
        return stored
    logger.info("No stored date found (first run)")
    return None


def save_date(date: str) -> None:
    """Write the current eCFR date to disk for future comparisons."""
    LAST_UPDATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_UPDATE_FILE.write_text(date + "\n")
    logger.info(f"Saved new date: {date}")


def main():
    """
    Compare the current eCFR date against the stored date.

    Outputs one of:
      CHANGED   - regulations updated, re-ingestion needed
      UNCHANGED - no change since last check

    Also sets the date file so subsequent runs can compare.
    """
    try:
        current_date = get_ecfr_title_date()
    except Exception as e:
        logger.error(f"Failed to check eCFR API: {e}")
        sys.exit(1)

    stored_date = get_stored_date()

    if stored_date is None or current_date > stored_date:
        # New date or first run — regulations may have changed
        if stored_date:
            logger.info(f"eCFR updated: {stored_date} → {current_date}")
        else:
            logger.info(f"First run — setting baseline date: {current_date}")

        save_date(current_date)
        print("CHANGED")
    else:
        logger.info(f"No change (stored: {stored_date}, current: {current_date})")
        print("UNCHANGED")


if __name__ == "__main__":
    main()
