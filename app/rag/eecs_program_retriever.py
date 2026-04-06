"""
EECS Program Retriever
======================
Fast JSON-backed lookup of KU's 12 EECS-relevant degree programs.

Data source: data/programs/eecs_programs_detailed.json
(produced by scrapers/eecs_programs_scraper.py)

Supports:
    - by_name("BS Computer Science")        → exact-ish match by degree name
    - by_level("bachelors" | "masters" | "doctoral" | "certificate")
    - search("how many credit hours for CS") → scored free-text search
    - format_for_context(results)            → pre-built context block
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional


# Map common aliases the user might type to the canonical degree name.
# The values MUST match program names in eecs_programs_detailed.json.
NAME_ALIASES: Dict[str, str] = {
    # BS CS
    "bs cs": "Bachelor of Science in Computer Science",
    "bs in cs": "Bachelor of Science in Computer Science",
    "bachelor of computer science": "Bachelor of Science in Computer Science",
    "bs computer science": "Bachelor of Science in Computer Science",
    "computer science bachelor": "Bachelor of Science in Computer Science",
    "cs bachelor": "Bachelor of Science in Computer Science",
    "cs undergrad": "Bachelor of Science in Computer Science",
    "undergraduate computer science": "Bachelor of Science in Computer Science",

    # BS CompE
    "bs compe": "Bachelor of Science in Computer Engineering",
    "bs in compe": "Bachelor of Science in Computer Engineering",
    "bs computer engineering": "Bachelor of Science in Computer Engineering",
    "compe bachelor": "Bachelor of Science in Computer Engineering",

    # BS EE
    "bs ee": "Bachelor of Science in Electrical Engineering",
    "bs in ee": "Bachelor of Science in Electrical Engineering",
    "bs electrical": "Bachelor of Science in Electrical Engineering",
    "ee bachelor": "Bachelor of Science in Electrical Engineering",

    # Applied Computing / Cyber
    "bs applied computing": "Bachelor of Science in Applied Computing",
    "applied computing": "Bachelor of Science in Applied Computing",
    "bs cybersecurity": "Bachelor of Science in Cybersecurity Engineering",
    "cybersecurity engineering": "Bachelor of Science in Cybersecurity Engineering",
    "cyber engineering": "Bachelor of Science in Cybersecurity Engineering",
    "ug cyber cert": "Undergraduate Certificate in Cybersecurity",
    "undergrad cybersecurity cert": "Undergraduate Certificate in Cybersecurity",
    "cybersecurity certificate": "Undergraduate Certificate in Cybersecurity",

    # MS
    "ms cs": "Master of Science in Computer Science",
    "ms in cs": "Master of Science in Computer Science",
    "ms computer science": "Master of Science in Computer Science",
    "ms compe": "Master of Science in Computer Engineering",
    "ms computer engineering": "Master of Science in Computer Engineering",
    "ms ee": "Master of Science in Electrical Engineering",
    "ms electrical": "Master of Science in Electrical Engineering",
    "meng eecs": "Master of Engineering in Electrical Engineering and Computer Science",
    "m.eng eecs": "Master of Engineering in Electrical Engineering and Computer Science",
    "master of engineering eecs": "Master of Engineering in Electrical Engineering and Computer Science",

    # PhD
    "phd cs": "Doctor of Philosophy in Computer Science",
    "phd in cs": "Doctor of Philosophy in Computer Science",
    "computer science phd": "Doctor of Philosophy in Computer Science",
    "cs doctorate": "Doctor of Philosophy in Computer Science",
    "phd ee": "Doctor of Philosophy in Electrical Engineering",
    "electrical engineering phd": "Doctor of Philosophy in Electrical Engineering",
}


def _degree_type(name: str) -> str:
    n = name.lower()
    if "doctor of philosophy" in n or "phd" in n:
        return "doctoral"
    if "master of science" in n or "master of engineering" in n:
        return "masters"
    if "bachelor" in n:
        return "bachelors"
    if "certificate" in n:
        return "certificate"
    return "other"


class EECSProgramRetriever:
    """Fast EECS program lookup."""

    def __init__(self, data_file: Optional[str] = None):
        if data_file is None:
            here = Path(__file__).resolve().parent
            # find repo root upwards
            for ancestor in [here, *here.parents]:
                candidate = ancestor / "data" / "programs" / "eecs_programs_detailed.json"
                if candidate.exists():
                    data_file = str(candidate)
                    break
            if data_file is None:
                raise FileNotFoundError(
                    "eecs_programs_detailed.json not found — run "
                    "scrapers/eecs_programs_scraper.py first"
                )

        with open(data_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.programs: List[Dict] = raw.get("programs", [])
        # Attach a derived "level" field once, for fast filtering
        for p in self.programs:
            p.setdefault("level", _degree_type(p.get("name", "")))

        # name → program dict (lowercased)
        self._by_name_lower: Dict[str, Dict] = {
            p["name"].lower(): p for p in self.programs
        }

    # ---------- lookups ----------

    def get_by_name(self, name: str) -> Optional[Dict]:
        """Exact or aliased name lookup. Case-insensitive."""
        if not name:
            return None
        n = name.strip().lower()
        if n in self._by_name_lower:
            return self._by_name_lower[n]
        if n in NAME_ALIASES:
            canon = NAME_ALIASES[n].lower()
            return self._by_name_lower.get(canon)
        # partial match — return the first program whose name contains the query
        for lower_name, prog in self._by_name_lower.items():
            if n in lower_name:
                return prog
        return None

    def get_by_level(self, level: str) -> List[Dict]:
        lvl = level.lower().strip()
        return [p for p in self.programs if p.get("level") == lvl]

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Free-text scored search over name + learning outcomes + course lists."""
        if not query:
            return []
        q = query.lower()
        q_tokens = set(q.split())

        # "Accelerated BS/MS" lives in the grad-special-admissions page, not
        # in the 12 detailed catalog programs — synthesize a pointer stub.
        if "accelerated" in q or "4+1" in q or "4 + 1" in q:
            return [{
                "name": "Accelerated BS/MS (4+1) Program",
                "level": "bachelors/masters",
                "total_credit_hours": None,
                "gpa_requirements": {"minimum": 3.5},
                "course_buckets": {},
                "four_year_plan": None,
                "learning_outcomes": [
                    "Eligibility: KU EECS juniors or seniors with 75+ credit hours and 3.5+ GPA overall and in math/science/CS/engineering coursework.",
                    "Benefit: 6-credit-hour reduction toward the MS upon admission.",
                    "Track: Available for the MS thesis option.",
                    "How to apply: Contact EECS Graduate Student Services (eecs_graduate@ku.edu).",
                ],
                "url": "https://eecs.ku.edu/special-graduate-admissions",
            }]

        # Minor-in-CS is not a standalone program in the catalog but is a
        # common question — synthesize a helpful stub.
        if ("minor" in q and ("cs" in q_tokens or "computer science" in q)):
            return [{
                "name": "Minor in Computer Science",
                "level": "minor",
                "total_credit_hours": None,
                "gpa_requirements": {},
                "course_buckets": {},
                "four_year_plan": None,
                "learning_outcomes": [
                    "KU's EECS department does not publish a standalone Computer Science minor.",
                    "Related options: Minor in Applied Cybersecurity, Minor in Information Technology (Professional Studies), Undergraduate Certificate in Cybersecurity.",
                    "Students seeking a CS minor should contact EECS advising (eecs_graduate@ku.edu / 785-864-4620) or visit https://eecs.ku.edu/prospective-students/undergraduate/degree-requirements",
                ],
                "url": "https://eecs.ku.edu/prospective-students/undergraduate/degree-requirements",
            }]

        # "EECS BS" / "BS in EECS" without a specific discipline — return the
        # three BS programs so the LLM can present them together.
        # Guard: use word-boundary checks so "cs" inside "eecs" doesn't count
        # as a specific discipline mention.
        has_eecs = "eecs" in q_tokens
        mentions_bachelor = ("bs" in q_tokens or "bachelor" in q or "credit hours" in q)
        mentions_specific = bool(re.search(
            r"\b(cs|computer\s+science|computer\s+engineering|electrical\s+engineering|cyber\s*security|applied\s+computing)\b",
            q,
        ))
        if has_eecs and mentions_bachelor and not mentions_specific:
            bs_names = [
                "Bachelor of Science in Computer Science",
                "Bachelor of Science in Computer Engineering",
                "Bachelor of Science in Electrical Engineering",
            ]
            bs_programs = [self._by_name_lower[n.lower()] for n in bs_names if n.lower() in self._by_name_lower]
            if bs_programs:
                return bs_programs

        # Try an alias or exact-name hit first — single result is usually best
        exact = self.get_by_name(query)
        if exact:
            return [exact]

        scored: List[tuple] = []
        for p in self.programs:
            name = p.get("name", "").lower()
            score = 0
            # Strong weight for name tokens
            for tok in q_tokens:
                if len(tok) < 3:
                    continue
                if tok in name:
                    score += 5
            # Level cues
            if ("bachelor" in q or "bs" in q_tokens or "undergraduate" in q) and p.get("level") == "bachelors":
                score += 3
            if ("master" in q or "ms" in q_tokens or "masters" in q_tokens) and p.get("level") == "masters":
                score += 3
            if ("phd" in q_tokens or "doctor" in q or "doctoral" in q) and p.get("level") == "doctoral":
                score += 3
            if "cert" in q and p.get("level") == "certificate":
                score += 3
            # Subject cues
            if ("cs" in q_tokens or "computer science" in q) and "computer science" in name:
                score += 4
            if ("ee" in q_tokens or "electrical" in q) and "electrical" in name:
                score += 4
            if "cyber" in q and "cyber" in name:
                score += 4
            if "applied computing" in q and "applied computing" in name:
                score += 4

            # Match inside learning outcomes is a weaker signal
            outcomes_blob = " ".join(p.get("learning_outcomes") or []).lower()
            for tok in q_tokens:
                if len(tok) > 3 and tok in outcomes_blob:
                    score += 1

            if score > 0:
                scored.append((score, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:limit]]

    # ---------- context formatting ----------

    def format_for_context(self, programs: List[Dict]) -> str:
        """Build a compact context block that the LLM can cite from."""
        if not programs:
            return ""

        blocks: List[str] = []
        for p in programs:
            lines = [f"[Source: eecs_program_retriever]"]
            lines.append(f"Program: {p.get('name','?')}")
            if p.get("level"):
                lines.append(f"Level: {p['level']}")
            if p.get("total_credit_hours"):
                lines.append(f"Total credit hours: {p['total_credit_hours']}")
            gpa = p.get("gpa_requirements") or {}
            if gpa:
                gpa_parts = []
                if "high_school" in gpa:
                    gpa_parts.append(f"HS {gpa['high_school']}+")
                if "ku" in gpa:
                    gpa_parts.append(f"KU {gpa['ku']}+")
                if "minimum" in gpa:
                    gpa_parts.append(f"min {gpa['minimum']}+")
                if gpa_parts:
                    lines.append(f"GPA: {', '.join(gpa_parts)}")

            # Course buckets
            buckets = p.get("course_buckets") or {}
            for key, label in [
                ("core_major", "Core major courses"),
                ("math", "Math courses"),
                ("basic_science", "Basic science"),
                ("electives", "Electives"),
            ]:
                codes = buckets.get(key) or []
                if codes:
                    shown = ", ".join(codes[:12])
                    more = f" (+{len(codes)-12} more)" if len(codes) > 12 else ""
                    lines.append(f"{label}: {shown}{more}")

            # 4-year plan (only if present)
            plan = p.get("four_year_plan") or {}
            if plan:
                lines.append("4-year plan:")
                for year in ["Freshman", "Sophomore", "Junior", "Senior"]:
                    sem = plan.get(year) or {}
                    fall = ", ".join(sem.get("fall") or [])
                    spring = ", ".join(sem.get("spring") or [])
                    if fall or spring:
                        lines.append(f"  {year}: Fall [{fall}] | Spring [{spring}]")

            # Learning outcomes — keep it short
            outcomes = p.get("learning_outcomes") or []
            if outcomes:
                lines.append("Learning outcomes:")
                for o in outcomes[:4]:
                    lines.append(f"  - {o[:200]}")

            if p.get("url"):
                lines.append(f"Catalog URL: {p['url']}")

            blocks.append("\n".join(lines))

        return "\n\n".join(blocks)


# Quick manual test
if __name__ == "__main__":
    r = EECSProgramRetriever()
    print(f"Loaded {len(r.programs)} programs\n")

    for q in [
        "BS Computer Science",
        "bs cs",
        "phd cs",
        "how many credit hours for BS CS",
        "cybersecurity certificate",
        "MS Electrical Engineering",
    ]:
        print(f"QUERY: {q!r}")
        results = r.search(q)
        print(f"  hits: {len(results)}")
        for p in results[:2]:
            print(f"   - {p['name']} ({p.get('total_credit_hours','?')} hrs)")
        print()

    # Format check
    bs_cs = r.get_by_name("Bachelor of Science in Computer Science")
    if bs_cs:
        print("\n--- FORMATTED CONTEXT FOR BS CS ---")
        print(r.format_for_context([bs_cs])[:1500])
