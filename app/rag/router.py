"""
Query Router for BabyJay
========================
Routes classified queries to the appropriate retriever.

Flow:
    User Query → Classifier → Router → Retriever → Re-rank → Context Build → Results

Fast paths (JSON-based, ~1-50ms):
    - faculty_search → FacultyRetriever
    - course_info → CourseRetriever
    - dining_info → CampusRetriever
    - transit_info → CampusRetriever
    - housing_info → CampusRetriever
    - financial_info → CampusRetriever

Fallback (ChromaDB, ~500-1000ms):
    - admission_info, calendar_info, library_info, etc.
"""

import os
from typing import Dict, Any, List, Optional
from pathlib import Path

from .classifier import QueryClassifier
from .faculty_retriever import FacultyRetriever
from .faculty_search import FacultySearcher
from .campus_retriever import CampusRetriever
from .course_retriever import CourseRetriever
from .context_builder import ContextBuilder
from .query_decomposer import QueryDecomposer
from .eecs_program_retriever import EECSProgramRetriever
from .eecs_resources_retriever import EECSResourcesRetriever

# Lazy import reranker to avoid slowing startup
_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from .reranker import Reranker
            _reranker = Reranker()
        except Exception:
            _reranker = None
    return _reranker


class QueryRouter:
    """Routes queries to appropriate retrievers based on classification."""

    # Minimum number of results needed before reranking pays off
    _RERANK_MIN = 2

    def _rerank(self, query: str, results: List, top_k: int = 10) -> List:
        """Apply cross-encoder reranking when we have enough candidates."""
        if not results or len(results) < self._RERANK_MIN:
            return results
        reranker = _get_reranker()
        if reranker is None:
            return results
        return reranker.rerank(query, results, top_k=top_k)

    def __init__(self):
        """Initialize router with classifier and retrievers."""
        self.classifier = QueryClassifier()
        self.faculty_retriever = FacultyRetriever()   # kept for exact name lookup + dept list
        self.faculty_searcher = FacultySearcher()     # ChromaDB semantic search — primary path
        self.campus_retriever = CampusRetriever()
        self.course_retriever = CourseRetriever()
        self.context_builder = ContextBuilder()
        self.decomposer = QueryDecomposer()
        try:
            self.eecs_program_retriever: Optional[EECSProgramRetriever] = EECSProgramRetriever()
        except FileNotFoundError:
            # Data file not yet scraped — degrade gracefully
            self.eecs_program_retriever = None
        try:
            self.eecs_resources_retriever: Optional[EECSResourcesRetriever] = EECSResourcesRetriever()
        except Exception:
            self.eecs_resources_retriever = None

        # Fallback retriever (vector search) - imported lazily to avoid circular imports
        self._vector_retriever = None
    
    @property
    def vector_retriever(self):
        """Lazy load vector retriever to avoid circular imports."""
        if self._vector_retriever is None:
            from .retriever import Retriever
            self._vector_retriever = Retriever()
        return self._vector_retriever
    
    def route(self, query: str, use_vector_fallback: bool = True) -> Dict[str, Any]:
        """
        Route a query to the appropriate retriever.
        
        Args:
            query: User's query
            use_vector_fallback: Whether to fall back to vector search if specialized retriever fails
            
        Returns:
            Dictionary with results, context, source, and query_info
        """
        # Step 0: Check if query needs decomposition (multi-part questions)
        if self.decomposer.should_decompose(query):
            sub_queries = self.decomposer.decompose(query)
            if len(sub_queries) > 1:
                sub_results = []
                for sq in sub_queries:
                    sr = self.route(sq, use_vector_fallback=use_vector_fallback)
                    sub_results.append(sr)
                return self.decomposer.merge_sub_results(sub_results)

        # Step 0b: Detect multi-domain queries (e.g. "courses AND faculty", "dining AND transit")
        # When a single query clearly spans two domains, route each domain and merge.
        multi_domain = self._detect_multi_domain(query)
        if multi_domain:
            sub_results = [self.route(sq, use_vector_fallback=use_vector_fallback)
                           for sq in multi_domain]
            return self.decomposer.merge_sub_results(sub_results)

        # Step 1: Classify the query (preprocessing is handled by chat.py before routing)
        classification = self.classifier.classify(query)
        intent = classification.get("intent", "general")
        entities = classification.get("entities", {})
        scope = classification.get("scope", "top_results")
        
        # Step 2: Route to appropriate retriever
        if intent == "eecs_program_info":
            return self._route_eecs_program(query, entities, scope, classification, use_vector_fallback)

        if intent in (
            "eecs_research_info",
            "eecs_facility_info",
            "eecs_student_org_info",
            "eecs_grad_admissions_info",
            "eecs_scholarship_info",
            "eecs_career_info",
            "eecs_leadership_info",
            "eecs_advising_info",
        ):
            return self._route_eecs_resources(intent, query, classification, use_vector_fallback)

        if intent == "faculty_search":
            return self._route_faculty(query, entities, scope, classification, use_vector_fallback)

        elif intent == "course_info":
            return self._route_courses(query, entities, scope, classification, use_vector_fallback)
        
        elif intent == "dining_info":
            return self._route_dining(query, entities, scope, classification)
        
        elif intent == "housing_info":
            return self._route_housing(query, entities, scope, classification)
        
        elif intent == "transit_info":
            return self._route_transit(query, entities, scope, classification)
        
        elif intent == "financial_info":
            return self._route_tuition(query, entities, scope, classification)
        
        elif intent == "building_info":
            # Try direct offices.json lookup first, fall back to vector
            office_result = self._route_offices(query, classification)
            if office_result.get("context"):
                return office_result
            if use_vector_fallback:
                return self._route_vector_fallback(query, intent, classification)
            return self._empty_result(classification)

        elif intent in ["admission_info", "library_info", "recreation_info",
                        "safety_info", "calendar_info"]:
            # These don't have specialized retrievers yet - use vector search
            if use_vector_fallback:
                return self._route_vector_fallback(query, intent, classification)
            else:
                return self._empty_result(classification)
        
        else:
            # General or unknown intent - use vector search
            if use_vector_fallback:
                return self._route_vector_fallback(query, intent, classification)
            else:
                return self._empty_result(classification)
    
    # Pairs of (domain_a_keywords, domain_b_keywords, sub-query-a-hint, sub-query-b-hint)
    _MULTI_DOMAIN_PAIRS = [
        # Course + Faculty
        (r"\b(course|courses|class|classes|prerequisite|prereq|EECS\s*\d{3})\b",
         r"\b(professor|faculty|researcher|who teaches|email.*professor|professor.*email)\b",
         "What courses are related to {topic}?",
         "Which faculty work on {topic}?"),
        # Tuition + Housing
        (r"\b(tuition|cost|per credit|credit hour cost)\b",
         r"\b(housing|residence hall|dorm|dormitory|on.campus living)\b",
         "What is the tuition rate?",
         "What housing options are available?"),
        # Dining + Transit
        (r"\b(dining|eat|food|cafeteria|meal|restaurant)\b",
         r"\b(bus|transit|route|transportation|shuttle|get to campus|ride)\b",
         "Where can I eat on campus?",
         "What transit routes serve campus?"),
        # Calendar + Course
        (r"\b(when does|semester start|semester begin|spring|fall|calendar)\b",
         r"\b(course|courses|class|classes|EECS\s*\d{3})\b",
         "When does the semester start?",
         "What courses are available?"),
        # Calendar + Tuition (drop deadline + refund)
        (r"\b(drop|withdraw|withdrawal|last day to drop|deadline)\b",
         r"\b(refund|tuition|financial|payment)\b",
         "What are the drop/withdrawal deadlines?",
         "What is the tuition refund policy?"),
        # Faculty + Housing
        (r"\b(professor|faculty|researcher|email.*professor|professor.*email)\b",
         r"\b(housing|residence hall|dorm|dormitory|on.campus living|graduate housing)\b",
         "Who are the faculty?",
         "What housing is available?"),
    ]

    def _detect_multi_domain(self, query: str) -> Optional[List[str]]:
        """
        Detect if a query clearly spans two distinct domains and can be split.
        Returns [sub_query_a, sub_query_b] if split is viable, else None.

        Strategy: split on ", and" or "; and" or " and " (when between two
        distinct domain signals). Each half must be ≥10 chars to be useful.
        """
        import re

        # Split candidates: ", and", "; and", " and " at a clause boundary
        split_patterns = [
            r',\s+and\s+',       # "..., and ..."
            r';\s+(?:and\s+)?',  # "...; ..." or "...; and ..."
        ]

        for sp in split_patterns:
            parts = re.split(sp, query, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                a, b = parts[0].strip(), parts[1].strip()
                if len(a) >= 10 and len(b) >= 10:
                    # Check that a and b hit different domain patterns
                    a_domains = {i for i, (pa, pb, _, _) in enumerate(self._MULTI_DOMAIN_PAIRS)
                                 if re.search(pa, a, re.IGNORECASE)}
                    b_domains = {i for i, (pa, pb, _, _) in enumerate(self._MULTI_DOMAIN_PAIRS)
                                 if re.search(pb, b, re.IGNORECASE)}
                    if a_domains & b_domains:  # At least one pair matched
                        # Ensure b is a complete question
                        if not b[0].isupper():
                            b = b[0].upper() + b[1:]
                        if not b.endswith('?'):
                            b = b + '?'
                        return [a, b]

        return None

    def _route_offices(self, query: str, classification: Dict) -> Dict[str, Any]:
        """Direct JSON lookup against offices.json — no ChromaDB needed."""
        import json
        data_path = Path(__file__).resolve().parent.parent.parent / "data" / "offices" / "offices.json"
        if not data_path.exists():
            return self._empty_result(classification)
        try:
            raw = json.loads(data_path.read_text())
            offices = raw if isinstance(raw, list) else raw.get("offices", [])
        except Exception:
            return self._empty_result(classification)

        q = query.lower()
        matches = []
        for o in offices:
            name = o.get("name", "").lower()
            desc = o.get("description", "").lower()
            services = " ".join(o.get("services", [])).lower()
            if any(word in name or word in desc or word in services
                   for word in q.split() if len(word) > 2):
                matches.append(o)

        if not matches:
            return self._empty_result(classification)

        blocks = []
        for o in matches[:3]:
            lines = ["[Source: offices_retriever]",
                     f"Office: {o.get('name', '?')}"]
            if o.get("building"):
                lines.append(f"Building: {o['building']}" + (f", Room {o['room']}" if o.get("room") else ""))
            if o.get("address"):
                lines.append(f"Address: {o['address']}")
            if o.get("phone"):
                lines.append(f"Phone: {o['phone']}")
            if o.get("email"):
                lines.append(f"Email: {o['email']}")
            if o.get("hours"):
                lines.append(f"Hours: {o['hours']}")
            if o.get("website"):
                lines.append(f"Source URL: {o['website']}")
            if o.get("services"):
                lines.append(f"Services: {', '.join(o['services'][:5])}")
            blocks.append("\n".join(lines))

        context = "\n\n".join(blocks)
        return {
            "results": matches[:3],
            "context": context,
            "source": "offices_retriever",
            "query_info": classification,
            "result_count": len(matches)
        }

    def _empty_result(self, classification: Dict) -> Dict[str, Any]:
        """Return empty result when no retriever is available."""
        return {
            "results": [],
            "context": "",
            "source": "none",
            "query_info": classification,
            "result_count": 0
        }
    
    def _route_eecs_resources(self, intent: str, query: str,
                              classification: Dict, use_vector_fallback: bool) -> Dict[str, Any]:
        """Dispatch Phase 2 EECS scoped queries to the resources retriever."""
        rr = self.eecs_resources_retriever
        if rr is None:
            if use_vector_fallback:
                return self._route_vector_fallback(query, intent, classification)
            return self._empty_result(classification)

        results: List = []
        context = ""
        if intent == "eecs_research_info":
            results = rr.search_research(query)
            context = rr.format_research_context(results)
        elif intent == "eecs_facility_info":
            results = rr.search_facility(query)
            context = rr.format_facility_context(results)
        elif intent == "eecs_student_org_info":
            results = rr.search_student_orgs(query)
            context = rr.format_orgs_context(results)
        elif intent == "eecs_grad_admissions_info":
            results = rr.search_grad(query)
            context = rr.format_grad_context(results)
        elif intent == "eecs_scholarship_info":
            results = rr.search_scholarships(query)
            context = rr.format_scholarships_context(results)
        elif intent == "eecs_career_info":
            results = rr.search_career(query)
            context = rr.format_career_context(results)
        elif intent == "eecs_leadership_info":
            results = rr.search_leadership(query)
            context = rr.format_leadership_context(results)
        elif intent == "eecs_advising_info":
            results = rr.search_advising(query)
            context = rr.format_advising_context(results)

        if not context and use_vector_fallback:
            return self._route_vector_fallback(query, intent, classification)

        return {
            "results": results,
            "context": context,
            "source": "eecs_resources_retriever",
            "query_info": classification,
            "result_count": len(results),
        }

    def _route_eecs_program(self, query: str, entities: Dict, scope: str,
                            classification: Dict, use_vector_fallback: bool) -> Dict[str, Any]:
        """Route EECS degree-program queries to the dedicated JSON retriever."""
        if self.eecs_program_retriever is None:
            # Data file missing — fall back to vector search
            if use_vector_fallback:
                return self._route_vector_fallback(query, "eecs_program_info", classification)
            return self._empty_result(classification)

        results = self.eecs_program_retriever.search(query, limit=5)
        if not results and use_vector_fallback:
            return self._route_vector_fallback(query, "eecs_program_info", classification)

        context = self.eecs_program_retriever.format_for_context(results)
        return {
            "results": results,
            "context": context,
            "source": "eecs_program_retriever",
            "query_info": classification,
            "result_count": len(results),
        }

    def _route_faculty(self, query: str, entities: Dict, scope: str,
                       classification: Dict, use_vector_fallback: bool) -> Dict[str, Any]:
        """
        Route faculty queries.

        Retrieval strategy:
          1. Exact name  → FacultyRetriever JSON (fast, precise string match)
          2. Dept-only complete list → FacultyRetriever JSON (guaranteed full list)
          3. Everything else (research area, semantic query, dept+topic)
             → FacultySearcher ChromaDB (vector + metadata filter)
          4. No results → vector fallback
        """
        department = entities.get("department")
        research_area = entities.get("research_area")
        name = entities.get("name")

        results = []
        source = "faculty_searcher"
        # For complete lists use 100; for research-area semantic queries use 30
        # (EECS has 67 faculty, scores cluster tightly — need wide net)
        top_k = 100 if scope == "complete_list" else 30

        # ── Strategy 1: Exact / partial name lookup ───────────────────────
        if name and not department and not research_area:
            results = self.faculty_retriever.search_by_name(name, limit=top_k)
            source = "faculty_retriever"

        # ── Strategy 2: Department-only "complete list" request ───────────
        elif department and not research_area and scope == "complete_list":
            results = self.faculty_retriever.get_department_faculty(
                department, limit=None
            )
            source = "faculty_retriever"

        # ── Strategy 3: Semantic (research area / dept+topic / open query) ─
        else:
            # Build the query term for the vector search
            search_q = research_area or query
            # Expand department abbreviations to partial names that match ChromaDB metadata
            _DEPT_MAP = {
                "eecs": "Electrical Engineering",
                "cs": "Computer Science",
                "math": "Mathematics",
                "physics": "Physics",
                "bio": "Biology",
                "chem": "Chemistry",
            }
            dept_filter = _DEPT_MAP.get((department or "").lower(), department)
            results_raw = self.faculty_searcher.search(
                search_q, top_k=top_k, department_filter=dept_filter
            )
            # Supplement with JSON keyword match on research_interests so that
            # faculty whose embedding rank is low but clearly match the topic
            # are always included (e.g. Branicky for "machine learning").
            if research_area or (department and not name):
                kw = (research_area or search_q).lower()
                kw_terms = [t.strip() for t in kw.replace("/", " ").split() if len(t) > 2]
                json_matches = self.faculty_retriever.search_by_research_keywords(
                    kw_terms, department_key=department, limit=10
                )
                # Merge — add JSON matches not already in results_raw
                seen_names = {r.get("name", "").lower() for r in results_raw}
                for jm in json_matches:
                    if jm.get("name", "").lower() not in seen_names:
                        results_raw.append({
                            "name": jm.get("name", ""),
                            "department": jm.get("department", ""),
                            "email": jm.get("email", ""),
                            "phone": jm.get("phone", ""),
                            "office": jm.get("office", ""),
                            "building": jm.get("building", ""),
                            "profile_url": jm.get("profile_url", ""),
                            "score": 0.5,
                            "document": " ".join(str(r) for r in jm.get("research_interests", [])),
                        })
            # Normalise FacultySearcher dicts to the format format_for_context expects
            results = [
                {
                    "name":               r.get("name", ""),
                    "department":         r.get("department", ""),
                    "department_key":     r.get("department", "").lower().replace(" ", "_"),
                    "email":              r.get("email", ""),
                    "phone":              r.get("phone", ""),
                    "office":             r.get("office", ""),
                    "building":           r.get("building", ""),
                    "profile_url":        r.get("profile_url", ""),
                    # research_interests expected as a list by format_for_context
                    "research_interests": [r.get("document", "")] if r.get("document") else [],
                    "content":            r.get("document", ""),
                    "relevance_score":    r.get("score", 0),
                }
                for r in results_raw
            ]

        # ── Strategy 4: Nothing found → vector fallback ───────────────────
        if not results and use_vector_fallback:
            return self._route_vector_fallback(query, "faculty_search", classification)

        # Deduplicate by name
        seen: set = set()
        deduped: List = []
        for r in results:
            n = r.get("name", "").lower()
            if n and n in seen:
                continue
            seen.add(n)
            deduped.append(r)

        # Rerank semantically before formatting
        deduped = self._rerank(query, deduped, top_k=top_k)

        context = self.faculty_retriever.format_for_context(deduped)

        return {
            "results": deduped,
            "context": context,
            "source": source,
            "query_info": classification,
            "result_count": len(deduped),
        }
    
    def _route_courses(self, query: str, entities: Dict, scope: str,
                       classification: Dict, use_vector_fallback: bool) -> Dict[str, Any]:
        """
        Route course queries.

        Retrieval strategy:
          1. Exact course code  → CourseRetriever JSON (fast, authoritative)
          2. Everything else    → Retriever ChromaDB (vector + BM25 hybrid, source:"course")
          3. No results         → vector fallback
        """
        course_code = entities.get("course_code")
        limit = 50 if scope == "complete_list" else 10

        # ── Strategy 1: Exact course code (e.g. "EECS 168") ──────────────
        if course_code:
            course = self.course_retriever.get_course(course_code)
            if course:
                results = [course]
                context = self.course_retriever.format_for_context(results)
                return {
                    "results": results,
                    "context": context,
                    "source": "course_retriever",
                    "query_info": classification,
                    "result_count": len(results),
                }

        # ── Strategy 2: Semantic search via ChromaDB + BM25 ──────────────
        raw = self.vector_retriever.search_courses(query, n_results=limit)

        # Rerank
        raw = self._rerank(query, raw, top_k=min(limit, 10))

        if not raw and use_vector_fallback:
            return self._route_vector_fallback(query, "course_info", classification)

        # Build context via ContextBuilder (handles [Source: ...] citations)
        context = self.context_builder.build(
            query,
            {"results": raw, "source": "course_vector", "query_info": classification},
        )

        return {
            "results": raw,
            "context": context,
            "source": "course_vector",
            "query_info": classification,
            "result_count": len(raw),
        }
    
    def _route_campus_vector(
        self,
        query: str,
        source_filter: str,
        classification: Dict,
        n_results: int,
    ) -> Dict[str, Any]:
        """
        Shared ChromaDB retrieval for all campus data domains.

        Replaces CampusRetriever substring matching with vector + BM25 hybrid
        search, then applies cross-encoder reranking for precision.
        """
        raw = self.vector_retriever.search(
            query, n_results=n_results, source_filter=source_filter
        )
        raw = self._rerank(query, raw, top_k=min(n_results, 10))

        context = self.context_builder.build(
            query,
            {"results": raw, "source": source_filter, "query_info": classification},
        )

        return {
            "results": raw,
            "context": context,
            "source": f"vector_{source_filter}",
            "query_info": classification,
            "result_count": len(raw),
        }

    def _route_dining(self, query: str, entities: Dict, scope: str,
                      classification: Dict) -> Dict[str, Any]:
        n = 20 if scope == "complete_list" else 5
        return self._route_campus_vector(query, "dining", classification, n)

    def _route_housing(self, query: str, entities: Dict, scope: str,
                       classification: Dict) -> Dict[str, Any]:
        n = 50 if scope == "complete_list" else 10
        return self._route_campus_vector(query, "housing", classification, n)

    def _route_transit(self, query: str, entities: Dict, scope: str,
                       classification: Dict) -> Dict[str, Any]:
        n = 20 if scope == "complete_list" else 5
        return self._route_campus_vector(query, "transit", classification, n)

    def _route_tuition(self, query: str, entities: Dict, scope: str,
                       classification: Dict) -> Dict[str, Any]:
        """
        Route financial queries.

        Searches both 'tuition' and 'financial_aid' sources because the
        intent covers both cost questions (tuition rates) and aid questions
        (FAFSA, grants, scholarships). Results are merged and reranked.
        """
        n = 30 if scope == "complete_list" else 10
        raw_tuition = self.vector_retriever.search(query, n_results=n, source_filter="tuition")
        raw_aid = self.vector_retriever.search(query, n_results=n, source_filter="financial_aid")
        merged = raw_tuition + raw_aid
        merged = self._rerank(query, merged, top_k=min(n, 10))

        context = self.context_builder.build(
            query,
            {"results": merged, "source": "tuition", "query_info": classification},
        )
        return {
            "results": merged,
            "context": context,
            "source": "vector_tuition",
            "query_info": classification,
            "result_count": len(merged),
        }
    
    def _route_vector_fallback(self, query: str, intent: str,
                                classification: Dict) -> Dict[str, Any]:
        """Fallback to vector search with re-ranking for better precision."""
        intent_to_method = {
            "faculty_search": "faculty",
            "dining_info": "dining",
            "housing_info": "housing",
            "transit_info": "transit",
            "course_info": "courses",
            "building_info": "buildings",
            "admission_info": "admissions",
            "financial_info": "tuition",
            "library_info": "libraries",
            "recreation_info": "recreation",
            "safety_info": "campus_safety",
            "calendar_info": "calendar",
        }

        results = self.vector_retriever.smart_search(query)

        result_key = intent_to_method.get(intent, "")
        specific_results = results.get(result_key) or []
        specific_results = self._rerank(query, specific_results, top_k=5)

        # Build compressed context with [Source: ...] citations via the shared
        # ContextBuilder. Fall back to smart_search's pre-built text only if the
        # builder produced nothing (e.g., no results under this intent).
        builder_payload = {
            "results": specific_results,
            "source": "vector_retriever",
            "query_info": classification,
        }
        context = self.context_builder.build(query, builder_payload)
        if not context:
            context = results.get("context", "")

        return {
            "results": specific_results,
            "context": context,
            "source": "vector_retriever",
            "query_info": classification,
            "result_count": len(specific_results)
        }


# Quick test
if __name__ == "__main__":
    import time
    
    print("Testing Query Router")
    print("=" * 60)
    
    router = QueryRouter()
    
    test_queries = [
        # Faculty queries (FacultyRetriever)
        ("EECS professors", "faculty_retriever"),
        ("professors doing robotics", "faculty_retriever"),
        
        # Course queries (CourseRetriever) - NEW!
        ("EECS 168", "course_retriever"),
        ("prerequisites for AE 345", "course_retriever"),
        ("machine learning courses", "course_retriever"),
        ("graduate EECS courses", "course_retriever"),
        ("machien lerning", "course_retriever"),  # Typo test
        
        # Dining queries (CampusRetriever)
        ("where can I eat on campus", "campus_retriever"),
        
        # Transit queries (CampusRetriever)
        ("bus routes to KU", "campus_retriever"),
        
        # Housing queries (CampusRetriever)
        ("dorm options", "campus_retriever"),
        
        # Tuition queries (CampusRetriever)
        ("how much is tuition", "campus_retriever"),
    ]
    
    total_time = 0
    passed = 0
    failed = 0
    
    for query, expected_source in test_queries:
        print(f"\nQuery: '{query}'")
        start = time.time()
        result = router.route(query, use_vector_fallback=False)
        elapsed = (time.time() - start) * 1000
        total_time += elapsed
        
        intent = result["query_info"].get("intent", "unknown")
        source = result["source"]
        count = result["result_count"]
        
        if source == expected_source:
            status = "✓"
            passed += 1
        else:
            status = "✗"
            failed += 1
        
        print(f"  {status} Intent: {intent}")
        print(f"  {status} Source: {source} (expected: {expected_source})")
        print(f"  Results: {count}")
        print(f"  Time: {elapsed:.1f}ms")
        
        # Show sample results for courses
        if source == "course_retriever" and result["results"]:
            sample = result["results"][0]
            print(f"  Sample: {sample.get('course_code', 'N/A')} - {sample.get('title', 'N/A')[:40]}")
    
    print("\n" + "=" * 60)
    print(f"Passed: {passed}/{len(test_queries)}")
    print(f"Total time: {total_time:.1f}ms")
    print(f"Average: {total_time/len(test_queries):.1f}ms per query")