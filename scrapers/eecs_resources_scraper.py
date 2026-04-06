"""
EECS Resources Scraper — Phase 2
================================
Comprehensive scrape of every EECS resource beyond the 12 degree programs
already captured in Phase 1. Covers:

  1. 11 research clusters  → data/research/eecs_research_clusters.json
  2. Facilities           → data/buildings/eecs_facilities.json
  3. Graduate program     → data/admissions/eecs_graduate.json
     (masters-program, phd-program, special-grad-admissions,
     graduate-funding, deficiency-courses)
  4. UG admissions        → data/admissions/eecs_undergraduate.json
  5. Academic experience / student orgs
                          → data/student_organizations/eecs_academic_experience.json
  6. External centers     → data/research/eecs_centers.json
     (i2s-research.ku.edu, cresis.ku.edu)
  7. External student life→ data/student_organizations/eecs_external_orgs.json
     (kuacm.club, hackku.org, ukansas-wic.github.io, hkn chapter)
  8. Engineering scholarships + career
                          → data/financial_aid/eecs_scholarships.json
                          → data/career/eecs_career.json

Every external fetch is wrapped in a try/except so one flaky site does not
kill the run. Sites are scraped with urllib + BeautifulSoup, no headless
browser, so any JS-rendered content will show up as empty / placeholder.

Run:
    python scrapers/eecs_resources_scraper.py
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"

UA = "Mozilla/5.0 (BabyJay EECS Phase2 scraper; contact eecs-info@ku.edu)"

NOW = lambda: datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# fetch helper
# ---------------------------------------------------------------------------

def fetch(url: str, timeout: int = 30, attempts: int = 2) -> Optional[str]:
    """Return HTML or None. Never raises."""
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception:
            if i == attempts - 1:
                return None
            time.sleep(1 + i)
    return None


def soupify(url: str) -> Optional[BeautifulSoup]:
    html = fetch(url)
    if html is None:
        return None
    return BeautifulSoup(html, "html.parser")


def main_content(soup: BeautifulSoup) -> BeautifulSoup:
    """Return the 'main content' of a Drupal-ish EECS page, stripped of
    nav/header/footer/scripts so only article text remains."""
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    main = (
        soup.find("main")
        or soup.find(class_="region-content")
        or soup.find(id="content")
        or soup.find(class_="node__content")
        or soup
    )
    return main


def clean(txt: str) -> str:
    return re.sub(r"\s+\n", "\n", re.sub(r"[ \t]+", " ", txt)).strip()


def write_json(rel_path: str, payload) -> None:
    p = DATA / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {p.relative_to(REPO_ROOT)}  ({p.stat().st_size} bytes)")


# ---------------------------------------------------------------------------
# 1. Research clusters
# ---------------------------------------------------------------------------

RESEARCH_CLUSTERS = [
    ("Applied Electromagnetics",            "https://eecs.ku.edu/applied-electromagnetics"),
    ("Communication Systems",               "https://eecs.ku.edu/communications-systems"),
    ("Computational Science and Engineering","https://eecs.ku.edu/computational-science-and-engineering"),
    ("Computer Systems Design",             "https://eecs.ku.edu/computer-systems-design"),
    ("Computing in the Biosciences",        "https://eecs.ku.edu/computing-biosciences"),
    ("Cybersecurity",                       "https://eecs.ku.edu/cybersecurity"),
    ("Language and Semantics",              "https://eecs.ku.edu/language-and-semantics"),
    ("Radar Systems and Remote Sensing",    "https://eecs.ku.edu/radar-systems-and-remote-sensing"),
    ("RF Systems Engineering",              "https://eecs.ku.edu/rf-systems-engineering"),
    ("Signal Processing",                   "https://eecs.ku.edu/signal-processing"),
    ("Theory of Computing",                 "https://eecs.ku.edu/theory-computing"),
]


EMAIL_RE = re.compile(r"[\w._%+-]+@(?:ku\.edu|eecs\.ku\.edu)", re.IGNORECASE)
PHONE_RE = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")
ROOM_RE = re.compile(
    r"\b\d{2,4}[A-Z]?\s+(?:Eaton|Nichols|Learned|Snow|Strong)(?:\s*Hall)?\b",
    re.IGNORECASE,
)


def _parse_faculty_card(card) -> Dict:
    """Extract a single faculty entry from a KU EECS cluster-page card div.

    Cards are the `col-11 col-lg-6 pt-3 pt-sm-0` class (found via DOM scan).
    Inside, the layout is:
        <strong>Name</strong>
        <div>Title lines</div>
        <a href="mailto:...">email</a>
        phone
        website
        office
        "Primary Research Interests"
        <ul><li>…</li></ul>
    We extract defensively so small layout differences don't crash us.
    """
    # Name — usually the first <strong> or <h4> inside the card
    name_tag = card.find(["strong", "h4", "h5"])
    name = clean(name_tag.get_text(" ", strip=True)) if name_tag else ""
    if not name:
        # Fallback — first non-empty text node
        text = card.get_text("\n", strip=True)
        name = (text.splitlines() or [""])[0]

    # email via mailto link or plain text
    email = ""
    mailto = card.find("a", href=re.compile(r"^mailto:", re.IGNORECASE))
    if mailto:
        email = mailto["href"].split(":", 1)[1].strip()
    else:
        m = EMAIL_RE.search(card.get_text(" ", strip=True))
        if m:
            email = m.group(0)

    full_text = card.get_text("\n", strip=True)
    phone_m = PHONE_RE.search(full_text)
    phone = phone_m.group(0) if phone_m else ""
    room_m = ROOM_RE.search(full_text)
    office = clean(room_m.group(0)) if room_m else ""

    # website — first http link that isn't a mailto
    website = ""
    for a in card.find_all("a", href=True):
        href = a["href"]
        if href.startswith("http"):
            website = href
            break

    # Research interests bullet list (if any)
    interests: List[str] = []
    for header in card.find_all(["h5", "h6", "strong", "b"]):
        if "research" in header.get_text(strip=True).lower():
            ul = header.find_next(["ul", "ol"])
            if ul:
                interests = [clean(li.get_text(" ", strip=True)) for li in ul.find_all("li")]
                break

    # Title — lines between the name and the email/phone, excluding interests
    lines = [ln for ln in full_text.splitlines() if ln.strip()]
    title_lines: List[str] = []
    for ln in lines:
        if ln == name:
            continue
        if EMAIL_RE.search(ln) or PHONE_RE.search(ln) or ROOM_RE.search(ln):
            continue
        if ln.startswith("http"):
            continue
        if "Primary Research Interests" in ln or "research" in ln.lower()[:20]:
            break
        title_lines.append(ln)
    title = " ".join(title_lines[:3]).strip()

    return {
        "name": name,
        "title": title,
        "email": email,
        "phone": phone,
        "website": website,
        "office": office,
        "research_interests": interests,
    }


def parse_cluster(name: str, url: str) -> Dict:
    soup = soupify(url)
    if soup is None:
        return {"name": name, "url": url, "error": "fetch_failed"}
    # Strip chrome but keep body
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    # Heading + overview paragraph (usually 1st <p> after h1)
    overview = ""
    first_p = soup.find("p")
    if first_p:
        overview = clean(first_p.get_text(" ", strip=True))

    # Program Objectives — h2 "Program Objectives" followed by a ul
    objectives: List[str] = []
    for header in soup.find_all(re.compile(r"h[1-6]")):
        if "objective" in header.get_text(strip=True).lower():
            ul = header.find_next(["ul", "ol"])
            if ul:
                objectives = [clean(li.get_text(" ", strip=True)) for li in ul.find_all("li")]
            break

    # Faculty — DOM-structured extraction via the `col-11 col-lg-6` cards
    faculty_cards = soup.find_all(
        "div",
        class_=lambda c: c is not None and "col-11" in c and "col-lg-6" in c,
    )
    faculty: List[Dict] = []
    for card in faculty_cards:
        card_text = card.get_text(" ", strip=True)
        if not EMAIL_RE.search(card_text):
            continue  # skip non-faculty cards (equipment, facility, etc.)
        fac = _parse_faculty_card(card)
        if fac.get("name"):
            faculty.append(fac)

    # Dedupe by email (cards may repeat across sections)
    seen = set()
    dedup: List[Dict] = []
    for f in faculty:
        key = f.get("email") or f.get("name")
        if key and key not in seen:
            seen.add(key)
            dedup.append(f)
    faculty = dedup

    # Flat raw text for LLM fallback
    text = soup.get_text("\n", strip=True)

    return {
        "name": name,
        "url": url,
        "overview": overview,
        "program_objectives": objectives,
        "faculty": faculty,
        "raw_text": text[:4000],
        "last_scraped": NOW(),
    }


def scrape_research_clusters() -> None:
    print("\n[1/8] Research clusters")
    out: List[Dict] = []
    for name, url in RESEARCH_CLUSTERS:
        print(f"  - {name}")
        data = parse_cluster(name, url)
        out.append(data)
        time.sleep(0.3)
    write_json(
        "research/eecs_research_clusters.json",
        {"total": len(out), "last_scraped": NOW(), "clusters": out},
    )


# ---------------------------------------------------------------------------
# 2. Facilities
# ---------------------------------------------------------------------------

def scrape_facilities() -> None:
    print("\n[2/8] Facilities")
    soup = soupify("https://eecs.ku.edu/facilities")
    if soup is None:
        print("  fetch failed")
        return
    m = main_content(soup)
    text = clean(m.get_text("\n", strip=True))

    # Best-effort sectioning by h2/h3 headers
    sections: List[Dict] = []
    for header in m.find_all(re.compile(r"h[2-6]")):
        title = clean(header.get_text(" ", strip=True))
        if not title:
            continue
        body_parts: List[str] = []
        sibling = header.find_next_sibling()
        while sibling and not re.match(r"h[1-6]", sibling.name or ""):
            body_parts.append(clean(sibling.get_text(" ", strip=True)))
            sibling = sibling.find_next_sibling()
        if any(body_parts):
            sections.append({"heading": title, "body": "\n".join(b for b in body_parts if b)})

    write_json(
        "buildings/eecs_facilities.json",
        {
            "url": "https://eecs.ku.edu/facilities",
            "last_scraped": NOW(),
            "full_text": text,
            "sections": sections,
        },
    )


# ---------------------------------------------------------------------------
# 3. Graduate details
# ---------------------------------------------------------------------------

GRAD_PAGES = {
    "masters_program":           "https://eecs.ku.edu/masters-program",
    "phd_program":                "https://eecs.ku.edu/phd-program",
    "special_grad_admissions":    "https://eecs.ku.edu/special-graduate-admissions",
    "graduate_funding":           "https://eecs.ku.edu/graduate-funding",
    "deficiency_courses":         "https://eecs.ku.edu/deficiency-courses",
}


def scrape_grad_details() -> None:
    print("\n[3/8] Graduate program details")
    out: Dict[str, Dict] = {}
    for key, url in GRAD_PAGES.items():
        print(f"  - {key}")
        soup = soupify(url)
        if soup is None:
            out[key] = {"url": url, "error": "fetch_failed"}
            continue
        m = main_content(soup)
        txt = clean(m.get_text("\n", strip=True))
        # Simple structured extraction
        deadlines = re.findall(
            r"(?:priority|final)[^.]*?(?:\b[A-Z][a-z]+\s+\d{1,2}(?:,\s*\d{4})?)",
            txt,
        )
        gpa_mentions = re.findall(r"(?:GPA\s+of\s+)?(\d\.\d)\+?", txt)
        toefl_mentions = re.findall(r"TOEFL[^.\n]{0,60}", txt)
        gre_mentions = re.findall(r"GRE[^.\n]{0,60}", txt)

        out[key] = {
            "url": url,
            "full_text": txt,
            "deadlines_preview": list(dict.fromkeys(deadlines))[:15],
            "gpa_mentions": list(dict.fromkeys(gpa_mentions))[:10],
            "toefl_mentions": list(dict.fromkeys(toefl_mentions))[:10],
            "gre_mentions": list(dict.fromkeys(gre_mentions))[:10],
            "last_scraped": NOW(),
        }
        time.sleep(0.3)
    write_json("admissions/eecs_graduate.json", out)


# ---------------------------------------------------------------------------
# 4. Undergraduate admissions
# ---------------------------------------------------------------------------

def scrape_undergrad_admissions() -> None:
    print("\n[4/8] Undergraduate admissions")
    soup = soupify("https://eecs.ku.edu/admission-undergraduate")
    if soup is None:
        print("  fetch failed")
        return
    m = main_content(soup)
    txt = clean(m.get_text("\n", strip=True))
    write_json(
        "admissions/eecs_undergraduate.json",
        {
            "url": "https://eecs.ku.edu/admission-undergraduate",
            "last_scraped": NOW(),
            "full_text": txt,
        },
    )


# ---------------------------------------------------------------------------
# 5. Academic experience (student orgs + tutoring pointers)
# ---------------------------------------------------------------------------

def scrape_academic_experience() -> None:
    print("\n[5/8] Academic experience")
    soup = soupify("https://eecs.ku.edu/academic-experience")
    if soup is None:
        print("  fetch failed")
        return
    m = main_content(soup)
    txt = clean(m.get_text("\n", strip=True))

    # Extract outbound links (student org sites etc.)
    links = [
        {"text": clean(a.get_text(" ", strip=True)), "href": a.get("href")}
        for a in m.find_all("a", href=True)
        if a.get("href", "").startswith(("http", "https"))
    ]

    write_json(
        "student_organizations/eecs_academic_experience.json",
        {
            "url": "https://eecs.ku.edu/academic-experience",
            "last_scraped": NOW(),
            "full_text": txt,
            "outbound_links": links,
        },
    )


# ---------------------------------------------------------------------------
# 6. External centers — I2S and CReSIS
# ---------------------------------------------------------------------------

def scrape_external_centers() -> None:
    print("\n[6/8] External centers (I2S + CReSIS)")
    centers: List[Dict] = []
    for name, url in [
        ("I2S — Institute for Information Sciences (formerly ITTC)", "https://i2s-research.ku.edu/"),
        ("CReSIS — Center for Remote Sensing of Ice Sheets",         "https://cresis.ku.edu/"),
    ]:
        soup = soupify(url)
        if soup is None:
            centers.append({"name": name, "url": url, "error": "fetch_failed"})
            continue
        m = main_content(soup)
        txt = clean(m.get_text("\n", strip=True))
        title = (soup.find("h1") or soup.find("title"))
        centers.append({
            "name": name,
            "url": url,
            "title": clean(title.get_text(" ", strip=True)) if title else name,
            "full_text": txt[:6000],
            "last_scraped": NOW(),
        })
        time.sleep(0.3)
    write_json(
        "research/eecs_centers.json",
        {"total": len(centers), "last_scraped": NOW(), "centers": centers},
    )


# ---------------------------------------------------------------------------
# 7. External student orgs — HackKU, KU ACM, WiC, HKN
# ---------------------------------------------------------------------------

def scrape_external_student_orgs() -> None:
    print("\n[7/8] External student orgs")
    orgs: List[Dict] = []
    for name, url in [
        ("KU ACM",                         "https://kuacm.club/"),
        ("KU ACM Tutoring",                "https://kuacm.club/tutoring/"),
        ("HackKU",                         "https://hackku.org/"),
        ("KU Women in Computing (KUWIC)",  "https://ukansas-wic.github.io/"),
        ("HKN Gamma Iota Chapter (KU)",    "https://hkn.ieee.org/hkn-chapters/all-chapters/gamma-iota-chapter/"),
        ("IEEE KU student branch",         "https://ieee.eecs.ku.edu/"),
    ]:
        soup = soupify(url)
        if soup is None:
            orgs.append({"name": name, "url": url, "error": "fetch_failed"})
            continue
        m = main_content(soup)
        txt = clean(m.get_text("\n", strip=True))
        title = (soup.find("h1") or soup.find("title"))
        orgs.append({
            "name": name,
            "url": url,
            "title": clean(title.get_text(" ", strip=True)) if title else name,
            "full_text": txt[:4000],
            "last_scraped": NOW(),
        })
        time.sleep(0.3)
    write_json(
        "student_organizations/eecs_external_orgs.json",
        {"total": len(orgs), "last_scraped": NOW(), "organizations": orgs},
    )


# ---------------------------------------------------------------------------
# 8. Engineering scholarships + career center
# ---------------------------------------------------------------------------

def scrape_scholarships_and_career() -> None:
    print("\n[8/8] Scholarships + career")
    items: Dict[str, Dict] = {}
    for key, name, url in [
        ("ug_scholarships",    "School of Engineering undergraduate scholarships",
         "https://engr.ku.edu/scholarships"),
        ("grad_scholarships",  "School of Engineering graduate scholarships",
         "https://engr.ku.edu/graduate-scholarships"),
        ("eecs_scholarship_stub", "EECS scholarship page",
         "https://eecs.ku.edu/scholarship-tuition"),
        ("engineering_career_center", "Engineering Career Center",
         "https://ecc.ku.edu/"),
        ("career_fair",        "KU Engineering & Computing Career Fair",
         "https://ecc.ku.edu/university-kansas-engineering-computing-career-fair"),
    ]:
        soup = soupify(url)
        if soup is None:
            items[key] = {"name": name, "url": url, "error": "fetch_failed"}
            continue
        m = main_content(soup)
        txt = clean(m.get_text("\n", strip=True))
        items[key] = {
            "name": name,
            "url": url,
            "full_text": txt[:5000],
            "last_scraped": NOW(),
        }
        time.sleep(0.3)

    # Split scholarships vs career into two files
    scholarships = {k: v for k, v in items.items() if "scholarship" in k}
    career = {k: v for k, v in items.items() if "career" in k or "career_fair" in k}
    write_json("financial_aid/eecs_scholarships.json", scholarships)
    write_json("career/eecs_career.json", career)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    print("EECS Resources Scraper — Phase 2")
    print("=" * 60)
    scrape_research_clusters()
    scrape_facilities()
    scrape_grad_details()
    scrape_undergrad_admissions()
    scrape_academic_experience()
    scrape_external_centers()
    scrape_external_student_orgs()
    scrape_scholarships_and_career()
    print("\nDone.")


if __name__ == "__main__":
    main()
