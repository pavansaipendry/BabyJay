"""
EECS Programs Scraper
=====================
Re-scrapes the 12 EECS-relevant degree programs from catalog.ku.edu to capture
the structured content that the previous general scraper truncated / lost.

Reads program names + URLs from data/programs/by_school/engineering.json and
re-fetches each catalog page, targeting the four tab containers:
    - admissionstextcontainer
    - requirementstextcontainer
    - plantextcontainer
    - learningoutcomestextcontainer

Parses each tab into structured fields (credit hours, GPA requirements, course
lists, 4-year plan table, learning outcomes) and writes a new file at
    data/programs/eecs_programs_detailed.json
without modifying any existing data.

Run:
    python scrapers/eecs_programs_scraper.py
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILE = REPO_ROOT / "data" / "programs" / "by_school" / "engineering.json"
OUT_FILE = REPO_ROOT / "data" / "programs" / "eecs_programs_detailed.json"

EECS_KEYWORDS = (
    "Computer Science",
    "Computer Engineering",
    "Electrical Engineering",
    "Cybersecurity",
    "Applied Computing",
    "Electrical Engineering and Computer Science",
)

UA = "Mozilla/5.0 (BabyJay EECS scraper; contact eecs-info@ku.edu)"


def _fetch(url: str, attempts: int = 3) -> str:
    last_err: Optional[Exception] = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as e:
            last_err = e
            time.sleep(1 + i)
    raise RuntimeError(f"failed to fetch {url}: {last_err}")


def _clean_text(s: str) -> str:
    """Collapse whitespace and strip."""
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{2,}", "\n", s)).strip()


def _extract_total_hours(text: str) -> Optional[int]:
    """Find something like 'Total Hours 126' or 'Total Credit Hours: 127'."""
    m = re.search(r"Total\s+(?:Credit\s+)?Hours[^\d]{0,10}(\d{2,3})", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _extract_gpa(text: str) -> Dict[str, float]:
    """Pull GPA numbers out of admissions text. Best-effort."""
    out: Dict[str, float] = {}
    # "3.0+ high school GPA"
    m = re.search(r"(\d\.\d)\+?\s+high\s+school\s+GPA", text, re.IGNORECASE)
    if m:
        out["high_school"] = float(m.group(1))
    # "KU GPA of 2.5" / "overall KU GPA of 2.5" / "a 2.5 KU GPA"
    m = re.search(r"(?:KU\s+GPA[^0-9]{0,15}(\d\.\d)|(\d\.\d)\s+KU\s+GPA)", text, re.IGNORECASE)
    if m:
        out["ku"] = float(m.group(1) or m.group(2))
    # "3.0 minimum GPA" (grad)
    m = re.search(r"minimum\s+GPA[^0-9]{0,15}(\d\.\d)", text, re.IGNORECASE)
    if m:
        out.setdefault("minimum", float(m.group(1)))
    return out


def _extract_requirements_courses(req_tab_text: str) -> Dict[str, List[str]]:
    """Bucket course codes in the requirements tab into rough categories.

    Categories we try to detect by surrounding headers (best-effort):
      math, basic_science, core_major, electives, core34_general_ed
    """
    buckets: Dict[str, List[str]] = {
        "math": [],
        "basic_science": [],
        "core_major": [],
        "electives": [],
    }
    # Split by common section headers used in the KU catalog
    sections = re.split(
        r"\n(?=(?:Mathematics|Basic Science|Computer Science Required Courses|"
        r"Electrical Engineering Required Courses|Computer Engineering Required Courses|"
        r"Required Courses|Major Requirements|Electives|Senior Electives|"
        r"Core 34 General Education))",
        req_tab_text,
    )
    CODE_RE = re.compile(r"\b([A-Z]{2,5})\s?(\d{3,4})\b")
    for sec in sections:
        if not sec.strip():
            continue
        # Get the first line as a probable header
        header = sec.splitlines()[0].strip().lower() if sec.splitlines() else ""
        codes = [f"{m.group(1)} {m.group(2)}" for m in CODE_RE.finditer(sec)]
        # Deduplicate while preserving order
        seen = set()
        codes_dedup = [c for c in codes if not (c in seen or seen.add(c))]
        if "math" in header:
            buckets["math"].extend(codes_dedup)
        elif "basic science" in header or "natural science" in header:
            buckets["basic_science"].extend(codes_dedup)
        elif "elective" in header:
            buckets["electives"].extend(codes_dedup)
        elif "required" in header or "major" in header:
            buckets["core_major"].extend(codes_dedup)
    # Dedup final lists
    for k, v in buckets.items():
        seen = set()
        buckets[k] = [c for c in v if not (c in seen or seen.add(c))]
    return buckets


def _extract_four_year_plan_from_table(plan_div) -> Optional[Dict[str, Dict[str, List[str]]]]:
    """Parse the 4-year plan HTML table directly.

    Catalog tables look like:
        [Freshman]           ← single-cell year header row
        [Fall, Hours, Spring, Hours]  ← column headers
        [EECS 101, 1, EECS 140, 4]   ← course rows
        ...
        [empty, 15, empty, 15]        ← term hour totals
        [Sophomore]
        ...

    Since text flattening loses the column structure, we read the <tr>s and
    split by cell index: column 0 = Fall, column 2 = Spring.
    """
    if plan_div is None:
        return None
    table = plan_div.find("table")
    if table is None:
        return None

    year_headers = {"Freshman", "Sophomore", "Junior", "Senior"}
    CODE_RE = re.compile(r"\b([A-Z]{2,5})\s?(\d{3,4})\b")

    plan: Dict[str, Dict[str, List[str]]] = {}
    current_year: Optional[str] = None

    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True).replace("\xa0", " ") for c in tr.find_all(["td", "th"])]
        if not cells:
            continue
        # Year header row (single cell with a year name)
        first_cell = cells[0].strip()
        if first_cell in year_headers:
            current_year = first_cell
            plan.setdefault(current_year, {"fall": [], "spring": []})
            continue
        if current_year is None:
            continue
        # Course row — column 0 = fall course, column 2 = spring course
        fall_code_cell = cells[0] if len(cells) > 0 else ""
        spring_code_cell = cells[2] if len(cells) > 2 else ""

        fall_m = CODE_RE.search(fall_code_cell)
        if fall_m:
            code = f"{fall_m.group(1)} {fall_m.group(2)}"
            if code not in plan[current_year]["fall"]:
                plan[current_year]["fall"].append(code)

        spring_m = CODE_RE.search(spring_code_cell)
        if spring_m:
            code = f"{spring_m.group(1)} {spring_m.group(2)}"
            if code not in plan[current_year]["spring"]:
                plan[current_year]["spring"].append(code)

    return plan if plan else None


def _extract_learning_outcomes(text: str) -> List[str]:
    """Learning outcomes are usually bullet-like — one per line after the header."""
    # The container text already dropped HTML structure; each outcome is usually
    # a sentence ending with a period.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Drop the intro line "At the completion of this program, students will be able to:"
    cleaned = []
    for ln in lines:
        if "At the completion of this program" in ln:
            continue
        # Skip obvious nav chrome
        if any(chrome in ln for chrome in ("Catalog Home", "Print Options", "Skip to Content")):
            continue
        if len(ln) > 15:  # drop short fragments
            cleaned.append(ln)
    return cleaned


def _scrape_program(name: str, url: str) -> Dict:
    html = _fetch(url)
    soup = BeautifulSoup(html, "html.parser")

    def _tab_text(*tab_ids: str) -> str:
        """Try multiple container id variants (the KU catalog uses
        requirements/degreerequirements/certificaterequirements on different
        pages — same for plan/degreeplan)."""
        for tid in tab_ids:
            div = soup.find(id=tid)
            if div:
                txt = _clean_text(div.get_text(separator="\n", strip=True))
                if txt:
                    return txt
        return ""

    admissions_txt = _tab_text("admissionstextcontainer")
    requirements_txt = _tab_text(
        "requirementstextcontainer",
        "degreerequirementstextcontainer",
        "certificaterequirementstextcontainer",
    )
    plan_txt = _tab_text("plantextcontainer", "degreeplantextcontainer")
    outcomes_txt = _tab_text("learningoutcomestextcontainer")

    # Grab the plan div object (not just text) for HTML-table parsing
    plan_div = soup.find(id="plantextcontainer") or soup.find(id="degreeplantextcontainer")

    total_hours = _extract_total_hours(requirements_txt) or _extract_total_hours(plan_txt)
    gpa_reqs = _extract_gpa(admissions_txt)
    course_buckets = _extract_requirements_courses(requirements_txt)
    four_year_plan = _extract_four_year_plan_from_table(plan_div)
    outcomes = _extract_learning_outcomes(outcomes_txt)

    # Collect ALL course codes that appear in the requirements tab as the
    # authoritative flat list (useful for search-by-course-code).
    CODE_RE = re.compile(r"\b([A-Z]{2,5})\s?(\d{3,4})\b")
    all_codes_seen = set()
    all_codes: List[str] = []
    for m in CODE_RE.finditer(requirements_txt):
        code = f"{m.group(1)} {m.group(2)}"
        if code not in all_codes_seen:
            all_codes_seen.add(code)
            all_codes.append(code)

    return {
        "name": name,
        "url": url,
        "total_credit_hours": total_hours,
        "gpa_requirements": gpa_reqs,
        "required_courses_all": all_codes,
        "course_buckets": course_buckets,
        "four_year_plan": four_year_plan,
        "learning_outcomes": outcomes,
        "raw_admissions_text": admissions_txt,
        "raw_requirements_text": requirements_txt,
        "raw_plan_text": plan_txt,
        "raw_outcomes_text": outcomes_txt,
        "last_scraped": datetime.utcnow().isoformat() + "Z",
    }


def main() -> None:
    if not SOURCE_FILE.exists():
        raise SystemExit(f"source file missing: {SOURCE_FILE}")
    source = json.loads(SOURCE_FILE.read_text())
    all_progs = source.get("programs", [])
    eecs_progs = [
        p for p in all_progs
        if any(kw in p.get("name", "") for kw in EECS_KEYWORDS)
    ]
    print(f"Found {len(eecs_progs)} EECS-relevant programs to re-scrape\n")

    scraped: List[Dict] = []
    for p in eecs_progs:
        name = p.get("name")
        urls = p.get("urls") or []
        # The canonical base URL is the first entry without a fragment
        base_url = next((u for u in urls if "#" not in u), urls[0] if urls else None)
        if not base_url:
            print(f"  SKIP {name}: no URL")
            continue
        print(f"  scraping: {name}")
        try:
            data = _scrape_program(name, base_url)
            scraped.append(data)
            print(
                f"    total_hours={data['total_credit_hours']} "
                f"gpa={data['gpa_requirements']} "
                f"courses={len(data['required_courses_all'])} "
                f"plan_years={len(data['four_year_plan'] or {})} "
                f"outcomes={len(data['learning_outcomes'])}"
            )
        except Exception as e:
            print(f"    ERROR: {e}")
        time.sleep(0.5)  # be polite

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(
        json.dumps(
            {
                "total_programs": len(scraped),
                "last_scraped": datetime.utcnow().isoformat() + "Z",
                "programs": scraped,
            },
            indent=2,
        )
    )
    print(f"\nWrote {len(scraped)} programs to {OUT_FILE}")


if __name__ == "__main__":
    main()
