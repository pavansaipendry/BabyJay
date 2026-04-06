"""
EECS Leadership Scraper — Phase 3
=================================
Extracts department chair, associate chairs, and center directors from
eecs.ku.edu/faculty, and adds a minimal stub for the Garmin EE/CompE
scholarship from ku.academicworks.com (the real body is JS-rendered behind
sign-in, so we only capture the title + portal URL).

Outputs:
    data/admissions/eecs_leadership.json
    data/financial_aid/eecs_named_scholarships.json
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parent.parent
UA = "Mozilla/5.0 (BabyJay Phase3 scraper; contact eecs-info@ku.edu)"
NOW = lambda: datetime.now(timezone.utc).isoformat()

EMAIL_RE = re.compile(r"[\w._%+-]+@(?:ku\.edu|eecs\.ku\.edu)", re.IGNORECASE)
PHONE_RE = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")
OFFICE_RE = re.compile(
    r"\b\d{2,4}[A-Z]?\s+(?:Eaton|Nichols|Learned|Snow|Strong)(?:\s*Hall)?\b",
    re.IGNORECASE,
)

LEADERSHIP_KEYWORDS = (
    "Department Chair",
    "Associate Chair",
    "Graduate Director",
    "Undergraduate Director",
    "Director of the Institute",
    "Director of the Center",
)


def fetch(url: str, timeout: int = 30) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  fetch error {url}: {e}")
        return None


# ---------------------------------------------------------------------------
# Leadership
# ---------------------------------------------------------------------------

def _parse_leadership_card(card) -> Optional[Dict]:
    """Pull name/title/email/phone/office from a KU EECS faculty card div.
    Returns None if the card isn't a person card (no email)."""
    full_text = card.get_text(" ", strip=True)
    if not full_text:
        return None
    email_m = EMAIL_RE.search(full_text)
    if not email_m:
        return None
    email = email_m.group(0)

    # Name = text before the first title/rank keyword. Handles cases where
    # the card has no <strong>/<h4> tag wrapping the name.
    TITLE_START = re.compile(
        r"\s+(?:University\s+Distinguished\s+Professor|Distinguished\s+Professor|"
        r"Professor|Associate\s+Professor|Assistant\s+Professor|"
        r"AT&T\s+Foundation|Deane\s+E\.|Gary\s+Minden|H\.J\.|Teaching\s+Professor|"
        r"Lecturer|Senior\s+Lecturer|Director|Chair|Associate\s+Chair)"
    )
    m = TITLE_START.search(full_text)
    if m:
        name = full_text[: m.start()].strip(" ,")
    else:
        name_tag = card.find(["strong", "h4", "h5"])
        name = name_tag.get_text(strip=True) if name_tag and name_tag.get_text(strip=True) else full_text.split(",")[0].strip()

    # Title = everything between the name and the email, roughly
    title = ""
    idx = full_text.find(email)
    if idx > 0:
        pre = full_text[:idx]
        if name and pre.startswith(name):
            pre = pre[len(name):]
        title = re.sub(r"\s+", " ", pre).strip(" ,")

    phone_m = PHONE_RE.search(full_text)
    office_m = OFFICE_RE.search(full_text)
    return {
        "name": name,
        "title": title,
        "email": email,
        "phone": phone_m.group(0) if phone_m else "",
        "office": office_m.group(0) if office_m else "",
    }


def scrape_leadership() -> None:
    print("[1/2] Scraping eecs.ku.edu/faculty for leadership roles")
    html = fetch("https://eecs.ku.edu/faculty")
    if html is None:
        print("  FAILED to fetch")
        return
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    cards = soup.find_all(
        "div",
        class_=lambda c: c is not None and "col-11" in c and "col-lg-6" in c,
    )

    leaders: List[Dict] = []
    seen_emails: set = set()
    for card in cards:
        parsed = _parse_leadership_card(card)
        if not parsed:
            continue
        title = parsed.get("title", "")
        if not any(kw in title for kw in LEADERSHIP_KEYWORDS):
            continue
        if parsed["email"] in seen_emails:
            continue
        seen_emails.add(parsed["email"])
        leaders.append(parsed)

    # Classify each leader role for easy lookup
    by_role: Dict[str, Dict] = {}
    for lead in leaders:
        title = lead["title"]
        if "Department Chair" in title:
            by_role["department_chair"] = lead
        if "Associate Chair for Graduate" in title or "Graduate Director" in title:
            by_role["associate_chair_graduate"] = lead
        if "Associate Chair for Undergraduate" in title or "Undergraduate Director" in title:
            by_role["associate_chair_undergraduate"] = lead
        if "Director of the Institute for Information Sciences" in title:
            by_role["i2s_director"] = lead
        if "Director of the Center for Remote Sensing" in title:
            by_role["cresis_director"] = lead

    out = {
        "source_url": "https://eecs.ku.edu/faculty",
        "last_scraped": NOW(),
        "leadership": leaders,
        "by_role": by_role,
    }
    out_path = REPO_ROOT / "data" / "admissions" / "eecs_leadership.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  wrote {out_path.relative_to(REPO_ROOT)} — {len(leaders)} leaders, roles: {sorted(by_role)}")


# ---------------------------------------------------------------------------
# Named scholarships — Academic Works
# ---------------------------------------------------------------------------

NAMED_SCHOLARSHIPS = [
    {
        "name": "Garmin Electrical and Computer Engineering Scholarships",
        "opportunity_id": 28818,
        "url": "https://ku.academicworks.com/opportunities/28818",
        "eligibility": "Cumulative GPA >= 3.8; Electrical Engineering or Computer Engineering students",
        "notes": "Students are automatically considered based on a completed KU General Application in UKASH.",
    },
    {
        "name": "Summerfield Scholarship",
        "opportunity_id": None,
        "url": "https://admissions.ku.edu/cost-aid/scholarships",
        "eligibility": "Incoming freshmen with top credentials (merit-based, KU's first merit scholarship est. 1929 by Solon Summerfield). Often awarded to EECS-bound students.",
        "award": "$4,500/year for up to 4 years",
        "notes": "Apply via the freshman admission application. Priority deadline Nov 1.",
    },
    {
        "name": "Watkins-Berger Scholarship",
        "opportunity_id": None,
        "url": "https://admissions.ku.edu/cost-aid/scholarships",
        "eligibility": "Incoming freshmen, merit-based, similar criteria to Summerfield.",
        "award": "$4,500/year for up to 4 years",
        "notes": "Apply via the freshman admission application.",
    },
    {
        "name": "Jayhawk CyberCorps Scholarship for Service (SFS)",
        "opportunity_id": None,
        "url": "https://sfs.ku.edu",
        "eligibility": "Undergraduate or graduate Cybersecurity Engineering / CS students willing to commit to federal cybersecurity service post-graduation.",
        "award": "Full tuition + monthly stipend + book/travel allowance",
        "notes": "Federally funded CyberCorps program. Students commit to one year of federal cybersecurity work per year of scholarship.",
    },
]


def scrape_named_scholarships() -> None:
    print("\n[2/2] Verifying Academic Works Garmin opportunity + writing named scholarships")
    # Confirm the Garmin page exists and capture its title
    html = fetch("https://ku.academicworks.com/opportunities/28818")
    garmin_title = None
    if html:
        soup = BeautifulSoup(html, "html.parser")
        t = soup.find("title")
        if t:
            garmin_title = t.get_text(strip=True)

    # Annotate Garmin entry with the verified title if available
    for s in NAMED_SCHOLARSHIPS:
        if s["name"].startswith("Garmin") and garmin_title:
            s["verified_title"] = garmin_title
            s["verified_at"] = NOW()

    out = {
        "last_scraped": NOW(),
        "portal": {
            "name": "UKASH (KU Award & Scholarship Hub)",
            "url": "https://ku.academicworks.com",
            "notes": "Central KU portal for scholarship applications. Complete the General Application to be auto-considered for many awards.",
        },
        "named_scholarships": NAMED_SCHOLARSHIPS,
    }
    out_path = REPO_ROOT / "data" / "financial_aid" / "eecs_named_scholarships.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(
        f"  wrote {out_path.relative_to(REPO_ROOT)} — {len(NAMED_SCHOLARSHIPS)} named, "
        f"portal={out['portal']['url']}, garmin_verified={bool(garmin_title)}"
    )


def main() -> None:
    print("EECS Phase 3 Scraper")
    print("=" * 60)
    scrape_leadership()
    scrape_named_scholarships()
    print("\nDone.")


if __name__ == "__main__":
    main()
