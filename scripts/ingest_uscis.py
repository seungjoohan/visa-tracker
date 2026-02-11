"""
USCIS Policy Manual & eCFR Ingestion Script

=== WHAT THIS SCRIPT DOES ===

This is the CLI entry point for populating the RAG knowledge base with
immigration policy documents. It fetches content from public government APIs
and websites, chunks it, embeds it, and stores it in our FAISS vector store.

=== DATA SOURCES ===

1. eCFR API (ecfr.gov): The official electronic Code of Federal Regulations.
   - Free, public, no API key needed
   - Returns structured XML/JSON of federal regulations
   - We fetch 8 CFR 214.2(h) which contains all H-1B regulations

2. Local files (knowledge_base/raw/): For manually curated content.
   - Place .txt files with policy content in this directory
   - The script will chunk and ingest them

=== HOW TO RUN ===

    # From the project root:
    python scripts/ingest_uscis.py

    # Or to only ingest local files:
    python scripts/ingest_uscis.py --local-only

    # To clear the knowledge base first:
    python scripts/ingest_uscis.py --clear
"""

import sys
import os
import argparse
import logging
import re
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

# Add project root to Python path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.policy import SourceType
from app.services.vector_store import VectorStore
from app.services.knowledge_ingester import KnowledgeIngester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
INDEX_DIR = PROJECT_ROOT / "knowledge_base" / "index"
RAW_DIR = PROJECT_ROOT / "knowledge_base" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "knowledge_base" / "processed"


def _get_ecfr_latest_date(title_number: int = 8) -> str:
    """
    Query the eCFR API for the most recent issue date of a given CFR title.

    === WHY IS THIS NEEDED? ===
    The eCFR /full/{date}/title-N.json endpoint requires a date that is on or
    before the title's most recent issue date. If you request today's date but
    the eCFR hasn't published today yet, you get a 400 error like:
        "The requested date 2026-02-11 is past the title's most recent issue
         date of 2026-02-09"

    The /titles endpoint returns metadata for all 50 CFR titles, including
    each title's latest issue date. We query that first to get a valid date.

    Args:
        title_number: The CFR title number (8 = Aliens and Nationality)

    Returns:
        Date string in YYYY-MM-DD format (the latest available issue date)
    """
    titles_url = "https://www.ecfr.gov/api/versioner/v1/titles"
    logger.info(f"Querying eCFR for latest issue date of Title {title_number}...")

    response = requests.get(titles_url, timeout=15)
    response.raise_for_status()
    titles_data = response.json()

    # The response is a list of title objects; find the one matching our title number
    for title in titles_data.get("titles", []):
        if title.get("number") == title_number:
            latest_date = title.get("latest_issue_date", title.get("up_to_date_as_of"))
            if latest_date:
                logger.info(f"eCFR Title {title_number} latest issue date: {latest_date}")
                return latest_date

    # Fallback: use yesterday's date (safe bet — eCFR usually lags 1-2 days)
    from datetime import timedelta
    fallback = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    logger.warning(
        f"Could not find latest date for Title {title_number}, "
        f"falling back to {fallback}"
    )
    return fallback


def fetch_ecfr_h1b() -> list[dict]:
    """
    Fetch H-1B regulations from the eCFR API.

    === WHAT IS eCFR? ===
    The Electronic Code of Federal Regulations is the official, up-to-date
    version of all federal regulations. 8 CFR Part 214 contains the rules
    for nonimmigrant visa holders (H-1B, L-1, O-1, F-1, etc.).

    === API DETAILS ===
    The eCFR API (https://www.ecfr.gov/api/versioner/v1/full/) returns the
    full text of CFR sections. We request the XML/text content for
    Title 8, Part 214 (nonimmigrant classes).

    The date parameter must be <= the title's most recent issue date,
    so we first query /titles to discover the latest valid date.

    Returns:
        List of section dicts ready for ingestion
    """
    sections = []

    # eCFR API endpoint for full text retrieval
    # Title 8 = Aliens and Nationality
    # Part 214 = Nonimmigrant Classes
    base_url = "https://www.ecfr.gov/api/versioner/v1/full"

    logger.info("Fetching H-1B regulations from eCFR API...")

    try:
        # Step 1: Get the latest valid date for Title 8
        latest_date = _get_ecfr_latest_date(title_number=8)

        # Step 2: Fetch Title 8, Part 214 as XML
        # The eCFR /full/ endpoint only serves XML (not JSON).
        # XML is the native format for legal documents — CFR has been
        # published in XML since the 2000s.
        response = requests.get(
            f"{base_url}/{latest_date}/title-8.xml",
            params={"part": "214"},
            headers={"Accept": "application/xml"},
            timeout=60,
        )
        response.raise_for_status()

        # Parse XML response into sections
        sections = _parse_ecfr_xml(response.text)
        logger.info(f"Fetched {len(sections)} sections from eCFR")

    except requests.RequestException as e:
        logger.warning(f"Failed to fetch from eCFR API: {e}")
        logger.info("Falling back to local H-1B content...")
        sections = _get_fallback_h1b_content()

    return sections


def _parse_ecfr_xml(xml_text: str) -> list[dict]:
    """
    Parse the eCFR XML response into sections for ingestion.

    === eCFR XML STRUCTURE ===
    The XML follows the CFR's hierarchical structure:
        <TITLE> → <SUBTITLE> → <CHAPTER> → <SUBCHAPTER> → <PART> → <SUBPART> → <SECTION>

    Each <SECTION> has:
        - <SECTNO> e.g., "§ 214.2"
        - <SUBJECT> e.g., "Special requirements for admission..."
        - <P> paragraphs containing the actual regulatory text

    We extract each <SECTION> as a separate ingestion section, building
    a context breadcrumb from its parent elements (Part > Subpart > Section).
    """
    sections = []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.error(f"Failed to parse eCFR XML: {e}")
        return []

    # Find all <DIV8> elements (sections in the eCFR XML schema)
    # or <SECTION> elements depending on the XML variant
    # eCFR uses a mix of <DIVn> tags where n indicates hierarchy depth
    # DIV5 = Part, DIV6 = Subpart, DIV8 = Section

    # Strategy: recursively walk the tree and collect text from section-level elements
    def _get_text_recursive(element) -> str:
        """Extract all text from an element and its children."""
        parts = []
        if element.text and element.text.strip():
            parts.append(element.text.strip())
        for child in element:
            child_text = _get_text_recursive(child)
            if child_text:
                parts.append(child_text)
            if child.tail and child.tail.strip():
                parts.append(child.tail.strip())
        return " ".join(parts)

    def _walk_tree(element, context_path: list[str]):
        """Recursively walk the XML tree, collecting sections with context."""
        tag = element.tag

        # Extract heading/title for context breadcrumb
        heading = ""
        for head_tag in ["HEAD", "SUBJECT", "SECTNO"]:
            head_el = element.find(head_tag)
            if head_el is not None and head_el.text:
                heading = head_el.text.strip()
                break

        current_path = context_path + [heading] if heading else context_path

        # Check if this is a section-level element (contains regulatory text)
        # We look for elements that have <P> (paragraph) children
        paragraphs = element.findall(".//P")

        # If this element directly contains paragraphs and has a heading,
        # treat it as a section to ingest
        direct_paragraphs = [p for p in element if p.tag == "P"]

        if direct_paragraphs and heading:
            # Extract text from all direct paragraphs
            text_parts = []
            for p in direct_paragraphs:
                p_text = _get_text_recursive(p)
                if p_text:
                    text_parts.append(p_text)

            text = "\n\n".join(text_parts)

            # Extract CFR reference from SECTNO if present
            sectno_el = element.find("SECTNO")
            cfr_ref = None
            if sectno_el is not None and sectno_el.text:
                # Convert "§ 214.2" to "8 CFR 214.2"
                cfr_ref = "8 CFR " + sectno_el.text.replace("§", "").strip()

            if len(text) > 50:
                sections.append({
                    "title": heading,
                    "context_header": " > ".join(["8 CFR Title 8"] + current_path),
                    "text": text,
                    "cfr_reference": cfr_ref,
                    "url": "https://www.ecfr.gov/current/title-8/part-214",
                })

        # Recurse into children
        for child in element:
            if child.tag not in ("P", "HEAD", "SUBJECT", "SECTNO", "CITA", "AUTH"):
                _walk_tree(child, current_path)

    _walk_tree(root, [])

    logger.info(f"Parsed {len(sections)} sections from eCFR XML")
    return sections


def _get_fallback_h1b_content() -> list[dict]:
    """
    Hardcoded H-1B policy content as fallback when API is unavailable.

    === WHY HAVE A FALLBACK? ===
    External APIs can be down, rate-limited, or change their response format.
    Having curated fallback content ensures the knowledge base always has
    a minimum viable set of information. This content is sourced from
    publicly available USCIS and CFR summaries.

    In a production system, you'd periodically refresh this from the API
    and only fall back to cached content.
    """
    return [
        {
            "title": "H-1B Specialty Occupation Overview",
            "context_header": "USCIS Policy Manual > H-1B > Overview",
            "text": (
                "The H-1B nonimmigrant visa category allows U.S. employers to temporarily "
                "employ foreign workers in specialty occupations. A specialty occupation "
                "requires the theoretical and practical application of a body of highly "
                "specialized knowledge and a bachelor's degree or higher in the specific "
                "specialty, or its equivalent, as a minimum for entry into the occupation. "
                "The H-1B visa has an annual statutory cap of 65,000 visas, with an "
                "additional 20,000 visas available for beneficiaries who have earned a "
                "master's degree or higher from a U.S. institution of higher education. "
                "Cap-exempt employers include institutions of higher education, nonprofit "
                "research organizations, and governmental research organizations."
            ),
            "cfr_reference": "8 CFR 214.2(h)",
            "url": "https://www.uscis.gov/working-in-the-united-states/h-1b-specialty-occupations",
        },
        {
            "title": "H-1B Specialty Occupation Requirements",
            "context_header": "USCIS Policy Manual > H-1B > Specialty Occupation Definition",
            "text": (
                "To qualify as a specialty occupation, the position must meet one of the "
                "following criteria: (1) A bachelor's or higher degree or its equivalent is "
                "normally the minimum requirement for entry into the particular position; "
                "(2) The degree requirement is common to the industry in parallel positions "
                "among similar organizations or the job is so complex or unique that it can "
                "be performed only by an individual with a degree; (3) The employer normally "
                "requires a degree or its equivalent for the position; or (4) The nature of "
                "the specific duties are so specialized and complex that knowledge required "
                "to perform the duties is usually associated with the attainment of a "
                "bachelor's or higher degree. USCIS examines the proffered position to "
                "determine whether it qualifies as a specialty occupation, not the "
                "credentials of the beneficiary."
            ),
            "cfr_reference": "8 CFR 214.2(h)(4)(ii)",
            "url": "https://www.uscis.gov/working-in-the-united-states/h-1b-specialty-occupations",
        },
        {
            "title": "H-1B Labor Condition Application (LCA)",
            "context_header": "USCIS Policy Manual > H-1B > LCA Requirements",
            "text": (
                "Before filing an H-1B petition, the employer must obtain a certified Labor "
                "Condition Application (LCA) from the Department of Labor. The LCA requires "
                "the employer to attest that: (1) It will pay the H-1B worker at least the "
                "prevailing wage or the employer's actual wage for the occupation, whichever "
                "is higher; (2) The employment of the H-1B worker will not adversely affect "
                "the working conditions of similarly employed workers; (3) There is no strike "
                "or lockout at the place of employment; and (4) Notice of the LCA filing has "
                "been provided to the union bargaining representative or posted at the "
                "worksite. The prevailing wage is determined based on the occupational "
                "classification and geographic area of intended employment."
            ),
            "cfr_reference": "20 CFR 655 Subpart H",
            "url": "https://www.dol.gov/agencies/eta/foreign-labor/wages/lca",
        },
        {
            "title": "H-1B Duration of Stay and Extensions",
            "context_header": "USCIS Policy Manual > H-1B > Duration and Extensions",
            "text": (
                "H-1B status is initially granted for up to three years and can be extended "
                "for a maximum total stay of six years. After six years, the foreign worker "
                "generally must leave the United States for one year before being eligible "
                "for a new H-1B petition. However, there are important exceptions under "
                "the American Competitiveness in the Twenty-First Century Act (AC21). "
                "If an employer has filed a PERM labor certification or I-140 immigrant "
                "petition at least 365 days before the six-year limit, the H-1B worker "
                "may be eligible for one-year extensions beyond the six-year period. "
                "Additionally, if an I-140 has been approved, the worker may be eligible "
                "for three-year extensions. These provisions allow workers stuck in green "
                "card backlogs to maintain their H-1B status indefinitely."
            ),
            "cfr_reference": "8 CFR 214.2(h)(13)",
            "url": "https://www.uscis.gov/working-in-the-united-states/h-1b-specialty-occupations",
        },
        {
            "title": "H-1B Cap and Lottery Process",
            "context_header": "USCIS Policy Manual > H-1B > Cap and Lottery",
            "text": (
                "The H-1B visa category is subject to an annual numerical limitation (cap) "
                "of 65,000 visas per fiscal year, plus an additional 20,000 for beneficiaries "
                "with a U.S. master's degree or higher (the master's cap exemption). When "
                "USCIS receives more petitions than available visas, it conducts a random "
                "selection process (lottery). Starting with FY2025, USCIS implemented a "
                "beneficiary-centric selection process where each unique beneficiary is "
                "entered into the lottery only once, regardless of how many employers "
                "submit registrations on their behalf. This change was designed to reduce "
                "the advantage that beneficiaries with multiple registrations had in "
                "previous lottery systems. Cap-exempt employers include institutions of "
                "higher education, nonprofit entities related to or affiliated with "
                "institutions of higher education, and nonprofit research organizations "
                "or governmental research organizations."
            ),
            "cfr_reference": "8 CFR 214.2(h)(8)",
            "url": "https://www.uscis.gov/working-in-the-united-states/h-1b-specialty-occupations",
        },
        {
            "title": "H-1B Portability (AC21)",
            "context_header": "USCIS Policy Manual > H-1B > Portability and Job Changes",
            "text": (
                "Under the American Competitiveness in the Twenty-First Century Act (AC21), "
                "an H-1B worker may begin working for a new employer as soon as the new "
                "employer files an H-1B petition on their behalf (H-1B portability). The "
                "worker does not need to wait for the new petition to be approved before "
                "starting employment with the new employer. To qualify for H-1B portability, "
                "the worker must: (1) have been lawfully admitted to the United States, "
                "(2) have a pending or approved H-1B petition, and (3) the new employer must "
                "file a non-frivolous H-1B petition before the worker's authorized stay "
                "expires. If the worker changes employers, the new employer becomes the "
                "sponsoring employer and must file a new LCA and H-1B petition."
            ),
            "cfr_reference": "INA 214(n) / AC21 Section 105",
            "url": "https://www.uscis.gov/working-in-the-united-states/h-1b-specialty-occupations",
        },
        {
            "title": "H-4 Dependent Visa and Employment Authorization",
            "context_header": "USCIS Policy Manual > H-4 > Employment Authorization",
            "text": (
                "H-4 status is available to the spouse and unmarried children under 21 of "
                "H-1B visa holders. Certain H-4 dependent spouses are eligible for "
                "employment authorization (H-4 EAD). To be eligible for an H-4 EAD, the "
                "H-1B principal must be the beneficiary of an approved I-140 petition or "
                "must have been granted H-1B status under sections 106(a) and (b) of AC21. "
                "The H-4 EAD allows the spouse to work for any employer in the United States. "
                "The H-4 EAD is valid for the same period as the H-4 nonimmigrant's "
                "authorized period of stay. The H-4 spouse must maintain valid H-4 status "
                "to be eligible for the EAD."
            ),
            "cfr_reference": "8 CFR 214.2(h)(9)(iv)",
            "url": "https://www.uscis.gov/working-in-the-united-states/h-4-ead",
        },
        {
            "title": "F-1 OPT and STEM OPT Extension",
            "context_header": "USCIS Policy Manual > F-1 > OPT and STEM Extension",
            "text": (
                "Optional Practical Training (OPT) allows F-1 students to gain practical "
                "work experience in their field of study. Students may apply for 12 months "
                "of OPT after completing their degree. Students who graduated with a degree "
                "in a STEM (Science, Technology, Engineering, or Mathematics) field may apply "
                "for a 24-month STEM OPT extension, for a total of 36 months of OPT. "
                "To qualify for the STEM OPT extension, the student must: (1) have a STEM "
                "degree from a SEVP-certified school, (2) be employed by an employer enrolled "
                "in E-Verify, and (3) have a formal training plan (Form I-983) signed by "
                "the employer. The employer must report to the school if the student's "
                "employment is terminated. STEM OPT students must work at least 20 hours "
                "per week and report to their DSO every 6 months."
            ),
            "cfr_reference": "8 CFR 214.2(f)(10)",
            "url": "https://studyinthestates.dhs.gov/stem-opt-hub",
        },
        {
            "title": "Employment-Based Green Card Process Overview",
            "context_header": "USCIS Policy Manual > Green Card > EB Process Overview",
            "text": (
                "The employment-based green card process typically involves three main steps: "
                "(1) PERM Labor Certification: The employer must test the U.S. labor market "
                "to demonstrate that there are no qualified U.S. workers available for the "
                "position. This involves recruiting, advertising, and filing an application "
                "with the Department of Labor. (2) I-140 Immigrant Petition: After PERM "
                "approval, the employer files Form I-140 with USCIS to establish that the "
                "foreign worker qualifies for the position and that the employer can pay "
                "the offered wage. (3) I-485 Adjustment of Status: Once a visa number is "
                "available (based on the priority date and per-country limits), the foreign "
                "worker files Form I-485 to adjust their status to permanent resident. "
                "EB-1 categories (extraordinary ability, outstanding professors, multinational "
                "managers) skip the PERM step. EB-2 NIW (National Interest Waiver) also "
                "skips PERM and does not require an employer sponsor."
            ),
            "cfr_reference": "8 CFR 204.5",
            "url": "https://www.uscis.gov/green-card/green-card-eligibility/green-card-for-employment-based-immigrants",
        },
        {
            "title": "Priority Dates and Visa Bulletin",
            "context_header": "USCIS Policy Manual > Green Card > Priority Dates and Visa Bulletin",
            "text": (
                "The priority date is the date that establishes the foreign worker's place "
                "in the green card queue. For PERM-based cases, the priority date is the "
                "date the PERM application was filed. For EB-1 and EB-2 NIW cases, it is "
                "the date the I-140 was filed. The Department of State publishes the monthly "
                "Visa Bulletin which shows the priority dates that are currently being "
                "processed for each employment-based category and country of chargeability. "
                "There are two charts: (1) Final Action Dates, which indicate when a visa "
                "can be issued or adjustment of status approved, and (2) Dates for Filing, "
                "which indicate when applicants can submit their I-485 applications. "
                "Per-country limits cap each country at approximately 7% of the total "
                "available employment-based visas, leading to significant backlogs for "
                "applicants born in India and China."
            ),
            "cfr_reference": "INA 203(b)",
            "url": "https://travel.state.gov/content/travel/en/legal/visa-law0/visa-bulletin.html",
        },
    ]


def ingest_local_files(ingester: KnowledgeIngester) -> int:
    """
    Ingest any .txt files from the knowledge_base/raw/ directory.

    This allows you to manually add policy content by placing text files
    in the raw directory. Each file becomes a section in the knowledge base.

    File naming convention:
        {source_type}_{title}.txt
        e.g., uscis_policy_manual_h1b_overview.txt

    Returns:
        Number of chunks ingested
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = list(RAW_DIR.glob("*.txt"))
    if not txt_files:
        logger.info(f"No .txt files found in {RAW_DIR}")
        return 0

    sections = []
    for txt_file in txt_files:
        text = txt_file.read_text(encoding="utf-8")
        title = txt_file.stem.replace("_", " ").title()

        sections.append({
            "title": title,
            "context_header": f"Local > {title}",
            "text": text,
        })

    chunks = ingester.ingest_sections(
        sections=sections,
        source_type=SourceType.USCIS_POLICY_MANUAL,
        replace_existing=False,  # Don't replace API-sourced content
    )

    return len(chunks)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest immigration policy documents into the RAG knowledge base"
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only ingest local files from knowledge_base/raw/",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the entire knowledge base before ingesting",
    )
    args = parser.parse_args()

    # Ensure directories exist
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize vector store and ingester
    store = VectorStore(index_dir=str(INDEX_DIR))

    if args.clear:
        logger.info("Clearing existing knowledge base...")
        store.clear()
        store.save()
    else:
        # Try to load existing index
        store.load()

    ingester = KnowledgeIngester(store)

    total_chunks = 0

    # Step 1: Ingest from eCFR API (unless --local-only)
    if not args.local_only:
        logger.info("=== Ingesting H-1B policy content ===")
        sections = fetch_ecfr_h1b()
        if sections:
            chunks = ingester.ingest_sections(
                sections=sections,
                source_type=SourceType.USCIS_POLICY_MANUAL,
                base_url="https://www.uscis.gov/policy-manual",
            )
            total_chunks += len(chunks)

    # Step 2: Ingest local files
    logger.info("=== Checking for local files ===")
    local_chunks = ingest_local_files(ingester)
    total_chunks += local_chunks

    # Print stats
    stats = store.get_stats()
    print("\n" + "=" * 60)
    print("Knowledge Base Ingestion Complete")
    print("=" * 60)
    print(f"  Total chunks in store: {stats.total_chunks}")
    print(f"  Chunks by source:")
    for source, count in stats.chunks_by_source.items():
        print(f"    - {source}: {count}")
    print(f"  Embedding model: {stats.embedding_model}")
    print(f"  Embedding dimension: {stats.embedding_dimension}")
    print(f"  Index directory: {INDEX_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
