"""
Context Builder for BabyJay RAG
================================
Builds optimized context for the LLM by:
1. Compressing retrieved data to only include relevant fields
2. Adding source tags for citation
3. Deduplicating overlapping information
4. Respecting token budgets
"""

import re
from typing import List, Dict, Any, Optional


class ContextBuilder:
    """Builds compressed, cited context from retrieved results."""

    def __init__(self, max_chars: int = 4000):
        self.max_chars = max_chars

    def build(self, query: str, route_result: Dict[str, Any]) -> str:
        """
        Build optimized context from router results.

        Args:
            query: The user's original query
            route_result: Dict from QueryRouter.route() with results, context, source, query_info

        Returns:
            Compressed context string with source citations
        """
        results = route_result.get("results", [])
        source = route_result.get("source", "unknown")
        query_info = route_result.get("query_info", {})
        intent = query_info.get("intent", "general")

        # Retrievers that pre-build rich context — pass through unchanged.
        if intent in (
            "eecs_research_info",
            "eecs_facility_info",
            "eecs_student_org_info",
            "eecs_grad_admissions_info",
            "eecs_scholarship_info",
            "eecs_career_info",
            "eecs_leadership_info",
            "eecs_advising_info",
        ) or source == "offices_retriever":
            return route_result.get("context", "")

        if not results:
            return ""

        # Determine which fields matter for this query type
        relevant_fields = self._get_relevant_fields(query, intent)

        # Build context blocks with citations
        blocks = []
        seen_names = set()

        for i, result in enumerate(results):
            # Skip duplicates
            name = self._get_name(result, intent)
            if name and name.lower() in seen_names:
                continue
            if name:
                seen_names.add(name.lower())

            block = self._compress_result(result, intent, relevant_fields, source)
            if block:
                blocks.append(block)

        # Join and truncate
        context = "\n\n".join(blocks)
        if len(context) > self.max_chars:
            context = context[:self.max_chars] + "\n\n[...additional results omitted]"

        return context

    def _get_relevant_fields(self, query: str, intent: str) -> set:
        """Determine which fields are relevant based on the query."""
        q = query.lower()
        # Word-tokenized set — avoids "ai" matching "main", "ml" matching "html".
        q_tokens = set(re.findall(r"[a-z0-9]+", q))

        def has_any(words):
            """Match if any listed word appears as a token, tolerating simple
            plural forms (word → words/es) in either direction."""
            for w in words:
                if w in q_tokens:
                    return True
                # singular listed, plural in query
                if (w + "s") in q_tokens or (w + "es") in q_tokens:
                    return True
                # plural listed, singular in query
                if w.endswith("s") and w[:-1] in q_tokens:
                    return True
                if w.endswith("es") and w[:-2] in q_tokens:
                    return True
            return False

        # Always include name/title
        fields = {"name", "title", "department"}

        if intent == "faculty_search":
            # Always include contact + location — they're compact and almost always useful
            fields.update({"email", "department", "research", "office", "building", "phone"})
            if has_any(["email", "contact", "reach"]):
                fields.update({"email", "phone"})
            if has_any(["research", "work", "study", "interested", "ml", "ai"]):
                fields.update({"research"})

        elif intent == "course_info":
            fields.update({"course_code", "credits", "description"})
            if has_any(["prerequisite", "prereq", "prereqs", "need", "before"]):
                fields.add("prerequisites")
            if has_any(["credit", "credits", "hour", "hours"]):
                fields.add("credits")
            if has_any(["level", "graduate", "undergraduate"]):
                fields.add("level")

        elif intent == "dining_info":
            fields.update({"building", "type", "hours"})

        elif intent == "transit_info":
            fields.update({"route_number", "description", "stops"})

        elif intent == "housing_info":
            fields.update({"type", "description", "amenities"})

        elif intent == "financial_info":
            fields.update({"amount", "description", "requirements"})

        else:
            # General — include everything
            fields.update({"description", "email", "phone", "building"})

        return fields

    def _get_name(self, result: Dict, intent: str) -> Optional[str]:
        """Extract the identifying name from a result."""
        if isinstance(result, dict):
            md = result.get("metadata") or {}
            return (result.get("name") or md.get("name")
                    or result.get("course_code") or md.get("course_code")
                    or result.get("title") or md.get("title")
                    or result.get("route_name") or md.get("route_name"))
        return None

    def _compress_result(self, result: Dict, intent: str,
                         relevant_fields: set, source: str) -> str:
        """Compress a single result to only include relevant fields."""

        if intent == "faculty_search":
            return self._format_faculty(result, relevant_fields, source)
        elif intent == "course_info":
            return self._format_course(result, relevant_fields, source)
        elif intent == "dining_info":
            return self._format_dining(result, relevant_fields, source)
        elif intent == "transit_info":
            return self._format_transit(result, relevant_fields, source)
        elif intent == "housing_info":
            return self._format_housing(result, relevant_fields, source)
        elif intent == "eecs_program_info":
            return self._format_eecs_program(result, source)
        else:
            return self._format_generic(result, source)

    def _format_faculty(self, r: Dict, fields: set, source: str) -> str:
        """Format a faculty result with only relevant fields.

        Works for both JSON-lookup results (fields at the top level) and
        vector-fallback results (fields nested under "metadata"). If we can't
        pull a name from either, fall back to the raw content blob so the LLM
        at least has something to ground on.
        """
        md = r.get("metadata") or {}

        def pick(key: str):
            return r.get(key) or md.get(key)

        name = pick("name")
        if not name:
            # Vector fallback with no metadata name — use content verbatim
            content = r.get("content")
            if content:
                return f"[Source: {source}]\n{str(content)[:400]}"
            return ""

        lines = [f"[Source: {source}]"]
        lines.append(f"Professor: {name}")
        lines.append(f"Department: {pick('department') or pick('department_key') or 'N/A'}")

        if "email" in fields and pick("email"):
            lines.append(f"Email: {pick('email')}")
        if "phone" in fields and pick("phone"):
            lines.append(f"Phone: {pick('phone')}")
        if "office" in fields or "building" in fields:
            office = pick("office") or ""
            building = pick("building") or ""
            if office or building:
                lines.append(f"Office: {office} {building}".strip())
        if "research" in fields:
            research = pick("research_interests") or pick("research") or []
            if isinstance(research, list):
                research_str = ", ".join(research[:5])
            else:
                research_str = str(research)[:300]
            if research_str:
                lines.append(f"Research: {research_str}")
        # Profile URL — make it available to the LLM so it can cite the
        # faculty page in the "Sources:" footer.
        profile_url = pick("profile_url") or pick("website")
        if profile_url:
            lines.append(f"URL: {profile_url}")

        return "\n".join(lines)

    def _format_course(self, r: Dict, fields: set, source: str) -> str:
        """Format a course result with only relevant fields."""
        lines = [f"[Source: {source}]"]
        code = r.get("course_code", "")
        title = r.get("title", "")
        lines.append(f"Course: {code} - {title}" if code else f"Course: {title}")

        if "credits" in fields and r.get("credits"):
            lines.append(f"Credits: {r['credits']}")
        if "level" in fields and r.get("level"):
            lines.append(f"Level: {r['level']}")
        if "description" in fields and r.get("description"):
            desc = r["description"][:200]
            lines.append(f"Description: {desc}")
        if "prerequisites" in fields and r.get("prerequisites"):
            lines.append(f"Prerequisites: {r['prerequisites']}")

        # Synthesize a KU catalog search URL from the course code — the
        # course JSON doesn't carry URLs but this search link reliably
        # resolves to the course page on catalog.ku.edu.
        if code:
            catalog_url = "https://catalog.ku.edu/search/?P=" + code.replace(" ", "+")
            lines.append(f"URL: {catalog_url}")

        return "\n".join(lines)

    def _format_dining(self, r: Dict, fields: set, source: str) -> str:
        lines = [f"[Source: {source}]"]
        lines.append(f"Name: {r.get('name', 'Unknown')}")
        if r.get("building"):
            lines.append(f"Building: {r['building']}")
        if r.get("type"):
            lines.append(f"Type: {r['type']}")
        if "hours" in fields and r.get("hours"):
            hours = r["hours"]
            if isinstance(hours, dict):
                hours_parts = [f"{k}: {v}" for k, v in hours.items() if v]
                lines.append(f"Hours: {'; '.join(hours_parts[:3])}")
            else:
                lines.append(f"Hours: {hours}")
        lines.append("Source URL: https://dining.ku.edu")
        return "\n".join(lines)

    def _format_transit(self, r: Dict, fields: set, source: str) -> str:
        lines = [f"[Source: {source}]"]
        lines.append(f"Route: {r.get('route_name', r.get('name', 'Unknown'))}")
        if r.get("route_number"):
            lines.append(f"Number: {r['route_number']}")
        if r.get("description"):
            lines.append(f"Info: {r['description'][:150]}")
        lines.append("Source URL: https://lawrenceks.org/transit/")
        return "\n".join(lines)

    def _format_housing(self, r: Dict, fields: set, source: str) -> str:
        lines = [f"[Source: {source}]"]
        lines.append(f"Name: {r.get('name', 'Unknown')}")
        if r.get("type"):
            lines.append(f"Type: {r['type']}")
        if r.get("description"):
            lines.append(f"Info: {r['description'][:200]}")
        lines.append("Source URL: https://housing.ku.edu")
        return "\n".join(lines)

    def _format_eecs_program(self, p: Dict, source: str) -> str:
        """Format an EECS degree program result with curriculum + plan."""
        lines = [f"[Source: {source}]"]
        lines.append(f"Program: {p.get('name','?')}")
        if p.get("level"):
            lines.append(f"Level: {p['level']}")
        if p.get("total_credit_hours"):
            lines.append(f"Total credit hours: {p['total_credit_hours']}")

        gpa = p.get("gpa_requirements") or {}
        if gpa:
            parts = []
            if "high_school" in gpa:
                parts.append(f"HS {gpa['high_school']}+")
            if "ku" in gpa:
                parts.append(f"KU {gpa['ku']}+")
            if "minimum" in gpa:
                parts.append(f"min {gpa['minimum']}+")
            if parts:
                lines.append(f"GPA requirements: {', '.join(parts)}")

        buckets = p.get("course_buckets") or {}
        for key, label in [
            ("core_major", "Core major courses"),
            ("math", "Math courses"),
            ("basic_science", "Basic science"),
            ("electives", "Electives"),
        ]:
            codes = buckets.get(key) or []
            if codes:
                shown = ", ".join(codes[:15])
                more = f" (+{len(codes)-15} more)" if len(codes) > 15 else ""
                lines.append(f"{label}: {shown}{more}")

        plan = p.get("four_year_plan") or {}
        if plan:
            lines.append("4-year plan:")
            for year in ["Freshman", "Sophomore", "Junior", "Senior"]:
                sem = plan.get(year) or {}
                fall = ", ".join(sem.get("fall") or [])
                spring = ", ".join(sem.get("spring") or [])
                if fall or spring:
                    lines.append(f"  {year} — Fall: {fall or '(n/a)'}  |  Spring: {spring or '(n/a)'}")

        outcomes = p.get("learning_outcomes") or []
        if outcomes:
            lines.append("Learning outcomes:")
            for o in outcomes[:4]:
                lines.append(f"  - {o[:180]}")

        if p.get("url"):
            lines.append(f"Catalog URL: {p['url']}")

        return "\n".join(lines)

    def _format_generic(self, r: Dict, source: str) -> str:
        """Format any result generically."""
        if isinstance(r, dict):
            # If it has a "content" key (from vector search), use that
            if "content" in r:
                content = r["content"][:400]
                return f"[Source: {source}]\n{content}"

            # Otherwise format key fields
            lines = [f"[Source: {source}]"]
            for key in ["name", "title", "description", "content"]:
                if r.get(key):
                    lines.append(f"{key.title()}: {str(r[key])[:200]}")
            return "\n".join(lines)

        return f"[Source: {source}]\n{str(r)[:400]}"
