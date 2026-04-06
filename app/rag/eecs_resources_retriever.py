"""
EECS Resources Retriever
========================
Loads all Phase 2 scraped data and answers scoped queries about:

    - research     (11 clusters + 2 centers: I2S, CReSIS)
    - facility     (Eaton Hall, Computing Commons, EECS Shop, labs)
    - grad_admissions (masters, phd, funding, deficiency, special)
    - ug_admissions   (freshman + transfer)
    - student_org     (ACM, HackKU, KUWIC, HKN, IEEE, academic-experience page)
    - scholarship     (UG + grad engineering scholarships)
    - career          (ECC, career fair)

Each search returns a list of result dicts plus a pre-built context string.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional


def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for ancestor in [here, *here.parents]:
        if (ancestor / "data").is_dir():
            return ancestor
    raise FileNotFoundError("could not locate data/ dir")


def _load(rel: str) -> Optional[dict]:
    p = _repo_root() / "data" / rel
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


class EECSResourcesRetriever:
    def __init__(self) -> None:
        self.clusters = _load("research/eecs_research_clusters.json") or {}
        self.centers = _load("research/eecs_centers.json") or {}
        self.facilities = _load("buildings/eecs_facilities.json") or {}
        self.grad = _load("admissions/eecs_graduate.json") or {}
        self.ug = _load("admissions/eecs_undergraduate.json") or {}
        self.leadership = _load("admissions/eecs_leadership.json") or {}
        self.acad_exp = _load("student_organizations/eecs_academic_experience.json") or {}
        self.external_orgs = _load("student_organizations/eecs_external_orgs.json") or {}
        self.scholarships = _load("financial_aid/eecs_scholarships.json") or {}
        self.named_scholarships = _load("financial_aid/eecs_named_scholarships.json") or {}
        self.financial_aid_general = _load("financial_aid/financial_aid.json") or {}
        self.career = _load("career/eecs_career.json") or {}
        self.offices = _load("offices/offices.json") or {}

    # ---------------- leadership / chair ----------------

    def search_leadership(self, query: str) -> List[Dict]:
        """Return one or more leadership entries (chair / director / etc.)."""
        q = query.lower()
        by_role = (self.leadership.get("by_role") or {})
        if not by_role:
            return []

        # Specific role keywords
        if "chair" in q and "associate" not in q and "undergraduate" not in q and "grad" not in q:
            lead = by_role.get("department_chair")
            if lead:
                return [{"role": "Department Chair", **lead}]
        if "graduate" in q or "grad " in q or "masters" in q or "phd" in q:
            lead = by_role.get("associate_chair_graduate")
            if lead:
                return [{"role": "Associate Chair for Graduate Studies", **lead}]
        if "undergrad" in q:
            lead = by_role.get("associate_chair_undergraduate")
            if lead:
                return [{"role": "Associate Chair for Undergraduate Studies", **lead}]
        if "i2s" in q or "ittc" in q or "institute" in q:
            lead = by_role.get("i2s_director")
            if lead:
                return [{"role": "I2S Director", **lead}]
        if "cresis" in q or "remote sensing" in q:
            lead = by_role.get("cresis_director")
            if lead:
                return [{"role": "CReSIS Director", **lead}]

        # "leadership" / "who runs EECS" → return all
        return [{"role": role.replace("_", " ").title(), **info} for role, info in by_role.items()]

    def format_leadership_context(self, leaders: List[Dict]) -> str:
        if not leaders:
            return ""
        blocks = []
        for l in leaders:
            lines = [f"[Source: eecs_resources_retriever]"]
            lines.append(f"Role: {l.get('role','?')}")
            lines.append(f"Name: {l.get('name','?')}")
            if l.get("title"):
                lines.append(f"Title: {l['title']}")
            if l.get("email"):
                lines.append(f"Email: {l['email']}")
            if l.get("phone"):
                lines.append(f"Phone: {l['phone']}")
            if l.get("office"):
                lines.append(f"Office: {l['office']}")
            blocks.append("\n".join(lines))
        if self.leadership.get("source_url"):
            blocks.append(f"Source URL: {self.leadership['source_url']}")
        return "\n\n".join(blocks)

    # ---------------- advising / offices ----------------

    def search_advising(self, query: str) -> List[Dict]:
        """Surface EECS Department Office + Engineering Career Center + any
        other KU office whose services mention advising. Data already lives
        in data/offices/offices.json from an earlier scrape."""
        offices_data = self.offices.get("offices") or []
        q = query.lower()
        hits: List[Dict] = []

        # Always surface the two EECS-critical offices
        for o in offices_data:
            name = (o.get("name") or "").lower()
            if "eecs department office" in name or "engineering career center" in name:
                hits.append(o)

        # Then any office whose services contain "advising" and that matches
        # extra query tokens
        if "advising" in q or "advisor" in q or "appointment" in q or "advis" in q:
            for o in offices_data:
                services = [s.lower() for s in (o.get("services") or [])]
                blob = (o.get("name","") + " " + o.get("description","") + " " + " ".join(services)).lower()
                if "advising" in blob and o not in hits:
                    hits.append(o)

        return hits[:5]

    def format_advising_context(self, offices: List[Dict]) -> str:
        if not offices:
            return ""
        blocks = []
        for o in offices:
            lines = [f"[Source: eecs_resources_retriever]"]
            lines.append(f"Office: {o.get('name','?')}")
            loc_bits = [x for x in [o.get("building",""), o.get("room","")] if x]
            if loc_bits:
                lines.append(f"Location: {', '.join(loc_bits)}")
            if o.get("address"):
                lines.append(f"Address: {o['address']}")
            if o.get("phone"):
                lines.append(f"Phone: {o['phone']}")
            if o.get("email"):
                lines.append(f"Email: {o['email']}")
            if o.get("website"):
                lines.append(f"Website: {o['website']}")
            if o.get("hours"):
                lines.append(f"Hours: {o['hours']}")
            services = o.get("services") or []
            if services:
                lines.append(f"Services: {', '.join(services)}")
            if o.get("description"):
                lines.append(f"About: {o['description']}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    # ---------------- research ----------------

    CLUSTER_KEYWORDS: Dict[str, List[str]] = {
        "Applied Electromagnetics": ["electromagnetics", "em", "applied electromagnetics"],
        "Communication Systems": ["communication", "networks", "wireless", "telecom"],
        "Computational Science and Engineering": ["computational", "cse", "simulation", "scientific computing"],
        "Computer Systems Design": ["systems design", "compilers", "architecture", "operating systems"],
        "Computing in the Biosciences": ["bioinformatics", "biosciences", "computing biosciences", "bioscience"],
        "Cybersecurity": ["cybersecurity", "security", "cyber", "cryptography"],
        "Language and Semantics": ["language", "semantics", "type system", "verification"],
        "Radar Systems and Remote Sensing": ["radar", "remote sensing", "ice", "cresis"],
        "RF Systems Engineering": ["rf", "radio frequency", "antenna"],
        "Signal Processing": ["signal processing", "dsp", "filter"],
        "Theory of Computing": ["theory", "algorithms", "complexity", "computational theory"],
    }

    def search_research(self, query: str, limit: int = 3) -> List[Dict]:
        """Find matching research clusters + centers."""
        q = query.lower()
        results: List[Dict] = []

        clusters = (self.clusters.get("clusters") or [])
        centers = (self.centers.get("centers") or [])

        # Specific centers first
        for c in centers:
            name = c.get("name", "").lower()
            if ("i2s" in q or "ittc" in q) and "information sciences" in name:
                results.append({"_type": "center", **c})
            if "cresis" in q or ("ice" in q and "sheet" in q):
                if "cresis" in name or "remote sensing" in name:
                    results.append({"_type": "center", **c})

        # Cluster keyword matches (scored)
        scored: List[tuple] = []
        for c in clusters:
            name = c.get("name", "")
            keys = self.CLUSTER_KEYWORDS.get(name, [name.lower()])
            score = 0
            for k in keys:
                if k in q:
                    score += len(k)  # longer match = better
            # Also score against overview text
            overview = c.get("overview", "").lower()
            for tok in q.split():
                if len(tok) > 4 and tok in overview:
                    score += 1
            if score > 0:
                scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        for _, c in scored[:limit]:
            results.append({"_type": "cluster", **c})

        return results[:limit]

    def format_research_context(self, results: List[Dict]) -> str:
        if not results:
            return ""
        blocks: List[str] = []
        for r in results:
            t = r.get("_type")
            lines = [f"[Source: eecs_resources_retriever]"]
            if t == "cluster":
                lines.append(f"Research cluster: {r.get('name','?')}")
                if r.get("overview"):
                    lines.append(f"Overview: {r['overview'][:600]}")
                objs = r.get("program_objectives") or []
                if objs:
                    lines.append("Program objectives:")
                    for o in objs[:6]:
                        lines.append(f"  - {o[:180]}")
                fac = r.get("faculty") or []
                if fac:
                    lines.append(f"Associated faculty ({len(fac)}):")
                    for f in fac[:10]:
                        name = f.get("name", "")
                        email = f.get("email", "")
                        office = f.get("office", "")
                        bits = [p for p in [email, office] if p]
                        suffix = f" ({' | '.join(bits)})" if bits else ""
                        lines.append(f"  - {name}{suffix}")
                if r.get("url"):
                    lines.append(f"URL: {r['url']}")
            elif t == "center":
                lines.append(f"Research center: {r.get('name','?')}")
                if r.get("title"):
                    lines.append(f"Title: {r['title']}")
                if r.get("full_text"):
                    lines.append(f"Description: {r['full_text'][:800]}")
                if r.get("url"):
                    lines.append(f"URL: {r['url']}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    # ---------------- facilities ----------------

    def search_facility(self, query: str) -> List[Dict]:
        """Return facility sections that match the query."""
        q = query.lower()
        sections = (self.facilities.get("sections") or [])
        matches: List[Dict] = []
        for s in sections:
            text = (s.get("heading", "") + " " + s.get("body", "")).lower()
            if any(tok in text for tok in q.split() if len(tok) > 3):
                matches.append(s)
        # If nothing specific, return the whole facilities page
        if not matches:
            return [{"heading": "EECS Facilities", "body": (self.facilities.get("full_text") or "")[:2000]}]
        return matches[:5]

    def format_facility_context(self, sections: List[Dict]) -> str:
        if not sections:
            return ""
        blocks: List[str] = []
        for s in sections:
            lines = [f"[Source: eecs_resources_retriever]"]
            lines.append(f"Facility: {s.get('heading','?')}")
            body = (s.get("body") or "")[:900]
            if body:
                lines.append(body)
            blocks.append("\n".join(lines))
        if self.facilities.get("url"):
            blocks.append(f"URL: {self.facilities['url']}")
        return "\n\n".join(blocks)

    # ---------------- grad admissions ----------------

    def search_grad(self, query: str) -> List[Dict]:
        """Return grad-program sections that match the query."""
        q = query.lower()
        out: List[Dict] = []

        # Heuristic routing based on keywords in the query
        wants_phd = any(k in q for k in ("phd", "ph.d", "doctoral", "doctorate"))
        wants_ms = any(k in q for k in ("master", "ms ", "m.s", "masters"))
        wants_deadline = any(k in q for k in ("deadline", "due", "applic", "when"))
        wants_accel = any(k in q for k in ("accelerated", "4+1", "4 + 1", "early", "special"))
        wants_funding = any(k in q for k in ("funding", "assistantship", "gta", "gra", "fellowship", "stipend"))
        wants_defic = any(k in q for k in ("deficiency", "prereq"))

        if wants_phd:
            if self.grad.get("phd_program"):
                out.append({"key": "phd_program", **self.grad["phd_program"]})
        if wants_ms:
            if self.grad.get("masters_program"):
                out.append({"key": "masters_program", **self.grad["masters_program"]})
        if wants_deadline and not (wants_phd or wants_ms):
            # Deadline question with no specific level — return both
            if self.grad.get("masters_program"):
                out.append({"key": "masters_program", **self.grad["masters_program"]})
            if self.grad.get("phd_program"):
                out.append({"key": "phd_program", **self.grad["phd_program"]})
        if wants_accel:
            if self.grad.get("special_grad_admissions"):
                out.append({"key": "special_grad_admissions", **self.grad["special_grad_admissions"]})
        if wants_funding:
            if self.grad.get("graduate_funding"):
                out.append({"key": "graduate_funding", **self.grad["graduate_funding"]})
        if wants_defic:
            if self.grad.get("deficiency_courses"):
                out.append({"key": "deficiency_courses", **self.grad["deficiency_courses"]})

        # Default: return masters + phd overview
        if not out:
            for key in ("masters_program", "phd_program"):
                if self.grad.get(key):
                    out.append({"key": key, **self.grad[key]})

        # Dedup while preserving order
        seen = set()
        dedup = []
        for r in out:
            if r["key"] not in seen:
                seen.add(r["key"])
                dedup.append(r)
        return dedup

    def format_grad_context(self, results: List[Dict]) -> str:
        if not results:
            return ""
        blocks: List[str] = []
        for r in results:
            lines = [f"[Source: eecs_resources_retriever]"]
            lines.append(f"EECS Grad Page: {r.get('key','?').replace('_',' ').title()}")
            text = (r.get("full_text") or "")[:1800]
            if text:
                lines.append(text)
            if r.get("url"):
                lines.append(f"URL: {r['url']}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    # ---------------- ug admissions ----------------

    def get_ug_admissions(self) -> Dict:
        return self.ug or {}

    def format_ug_context(self, data: Dict) -> str:
        if not data:
            return ""
        text = (data.get("full_text") or "")[:1800]
        if not text:
            return ""
        return (
            "[Source: eecs_resources_retriever]\n"
            "EECS Undergraduate Admissions\n"
            f"{text}\n"
            f"URL: {data.get('url','')}"
        )

    # ---------------- student orgs ----------------

    # Map keyword triggers to the org entries most likely to have the answer.
    ORG_KEYWORD_MAP = {
        "hackathon": ["HackKU"],
        "hackku":    ["HackKU"],
        "tutoring":  ["KU ACM Tutoring", "KU ACM"],
        "tutor":     ["KU ACM Tutoring", "KU ACM"],
        "acm":       ["KU ACM", "KU ACM Tutoring"],
        "kuacm":     ["KU ACM", "KU ACM Tutoring"],
        "ieee":      ["IEEE KU student branch", "HKN Gamma Iota Chapter (KU)"],
        "hkn":       ["HKN Gamma Iota Chapter (KU)"],
        "kuwic":     ["KU Women in Computing (KUWIC)"],
        "wic":       ["KU Women in Computing (KUWIC)"],
        "women":     ["KU Women in Computing (KUWIC)"],
    }

    def search_student_orgs(self, query: str) -> List[Dict]:
        """Return matching student orgs (internal + external)."""
        q = query.lower()
        orgs = (self.external_orgs.get("organizations") or [])
        by_name = {o.get("name", ""): o for o in orgs}

        # Priority: explicit keyword triggers
        results: List[Dict] = []
        triggered: set = set()
        for trigger, targets in self.ORG_KEYWORD_MAP.items():
            if trigger in q:
                for t in targets:
                    if t in by_name and t not in triggered:
                        triggered.add(t)
                        results.append(by_name[t])

        # Fallback: free-text name/body match
        if not results:
            for o in orgs:
                name = o.get("name", "").lower()
                body = (o.get("full_text") or "").lower()
                if any(tok in name or tok in body for tok in q.split() if len(tok) > 2):
                    results.append(o)

        # Last resort: academic-experience page as a catch-all
        if not results and self.acad_exp.get("full_text"):
            results.append(
                {
                    "name": "EECS Academic Experience page",
                    "url": self.acad_exp.get("url"),
                    "full_text": self.acad_exp.get("full_text"),
                }
            )
        return results[:5]

    def format_orgs_context(self, orgs: List[Dict]) -> str:
        if not orgs:
            return ""
        blocks: List[str] = []
        for o in orgs:
            lines = [f"[Source: eecs_resources_retriever]"]
            lines.append(f"Student org: {o.get('name','?')}")
            if o.get("error"):
                lines.append(f"(site fetch failed — verify at {o.get('url','?')})")
            elif o.get("full_text"):
                lines.append(o["full_text"][:1200])
            if o.get("url"):
                lines.append(f"URL: {o['url']}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    # ---------------- scholarships ----------------

    def search_scholarships(self, query: str) -> List[Dict]:
        """Return a mixed list of:
            - named EECS-relevant scholarships (Garmin, Summerfield, etc.)
            - UKASH portal pointer
            - general engineering scholarship page summaries
        """
        q = query.lower()
        out: List[Dict] = []

        named = (self.named_scholarships.get("named_scholarships") or [])
        # If the user mentioned a specific named scholarship, prioritize it
        matched = []
        for s in named:
            if any(tok in s.get("name","").lower() for tok in q.split() if len(tok) > 3):
                matched.append({"_type": "named", **s})
        if matched:
            out.extend(matched)
        else:
            # Otherwise include the top 3 named scholarships as candidates
            for s in named[:4]:
                out.append({"_type": "named", **s})

        # Add the UKASH portal pointer
        portal = self.named_scholarships.get("portal") or {}
        if portal:
            out.append({"_type": "portal", **portal})

        # Add general School of Engineering scholarship summaries
        for k, v in (self.scholarships or {}).items():
            if v.get("full_text"):
                out.append({"_type": "general", "key": k, **v})

        # Add freshman scholarship tiers from the campus financial_aid file
        fa = (self.financial_aid_general.get("financial_aid") or {})
        freshman = ((fa.get("scholarships") or {}).get("freshman_scholarships") or {})
        if freshman:
            out.append({"_type": "freshman_tiers", "data": freshman})

        return out[:8]

    def format_scholarships_context(self, items: List[Dict]) -> str:
        if not items:
            return ""
        blocks: List[str] = []
        for it in items:
            t = it.get("_type", "general")
            lines = [f"[Source: eecs_resources_retriever]"]
            if t == "named":
                lines.append(f"Named scholarship: {it.get('name','?')}")
                if it.get("eligibility"):
                    lines.append(f"Eligibility: {it['eligibility']}")
                if it.get("award"):
                    lines.append(f"Award: {it['award']}")
                if it.get("notes"):
                    lines.append(f"Notes: {it['notes']}")
                if it.get("url"):
                    lines.append(f"URL: {it['url']}")
            elif t == "portal":
                lines.append(f"Scholarship portal: {it.get('name','?')}")
                if it.get("url"):
                    lines.append(f"URL: {it['url']}")
                if it.get("notes"):
                    lines.append(f"Notes: {it['notes']}")
            elif t == "general":
                lines.append(f"School of Engineering scholarship page: {it.get('name','?')}")
                if it.get("full_text"):
                    lines.append(it["full_text"][:900])
                if it.get("url"):
                    lines.append(f"URL: {it['url']}")
            elif t == "freshman_tiers":
                lines.append("KU freshman scholarship tiers (merit-based, auto-considered):")
                d = it.get("data") or {}
                if d.get("deadline"):
                    lines.append(f"Deadline: {d['deadline']}")
                if d.get("based_on"):
                    lines.append(f"Based on: {d['based_on']}")
                ks = d.get("kansas_resident_awards") or {}
                if ks:
                    lines.append("Kansas resident awards:")
                    for gpa, amt in ks.items():
                        lines.append(f"  - {gpa.replace('_',' ')}: {amt}")
                oos = d.get("out_of_state_awards") or {}
                if oos:
                    lines.append("Out-of-state awards:")
                    for gpa, amt in oos.items():
                        lines.append(f"  - {gpa.replace('_',' ')}: {amt}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    # ---------------- career ----------------

    def search_career(self, query: str) -> List[Dict]:
        out = []
        for k, v in (self.career or {}).items():
            if v.get("full_text") or not v.get("error"):
                out.append({"key": k, **v})
        return out

    def format_career_context(self, items: List[Dict]) -> str:
        if not items:
            return ""
        blocks = []
        for it in items:
            lines = [f"[Source: eecs_resources_retriever]"]
            lines.append(f"Career resource: {it.get('name','?')}")
            if it.get("full_text"):
                lines.append(it["full_text"][:1200])
            if it.get("url"):
                lines.append(f"URL: {it['url']}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)


# quick smoke test
if __name__ == "__main__":
    r = EECSResourcesRetriever()
    print("Loaded:")
    print(f"  clusters: {len(r.clusters.get('clusters', []))}")
    print(f"  centers:  {len(r.centers.get('centers', []))}")
    print(f"  grad:     {list(r.grad.keys())}")
    print(f"  orgs:     {len(r.external_orgs.get('organizations', []))}")
    print(f"  scholarships: {list(r.scholarships.keys())}")
    print(f"  career:   {list(r.career.keys())}")

    for q in [
        "cybersecurity research at KU",
        "tell me about ITTC",
        "cresis",
        "Eaton Hall labs",
        "EECS PhD funding",
        "HackKU",
        "KU ACM",
        "EECS scholarships",
    ]:
        print(f"\n--- {q!r} ---")
        if "research" in q or "ittc" in q.lower() or "cresis" in q.lower() or "cyber" in q.lower():
            res = r.search_research(q)
            ctx = r.format_research_context(res)
        elif "eaton" in q.lower() or "labs" in q.lower() or "facil" in q.lower():
            res = r.search_facility(q)
            ctx = r.format_facility_context(res)
        elif "phd" in q.lower() or "grad" in q.lower() or "funding" in q.lower() or "master" in q.lower():
            res = r.search_grad(q)
            ctx = r.format_grad_context(res)
        elif "hack" in q.lower() or "acm" in q.lower() or "ieee" in q.lower():
            res = r.search_student_orgs(q)
            ctx = r.format_orgs_context(res)
        elif "scholarship" in q.lower():
            res = r.search_scholarships(q)
            ctx = r.format_scholarships_context(res)
        else:
            ctx = ""
        print(ctx[:600] if ctx else "(no context)")
