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

    def __init__(self):
        """Initialize router with classifier and retrievers."""
        self.classifier = QueryClassifier()
        self.faculty_retriever = FacultyRetriever()
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
        """Route faculty search queries."""
        department = entities.get("department")
        research_area = entities.get("research_area")
        name = entities.get("name")
        
        results = []
        
        # Strategy 1: Search by name if provided
        if name and not department and not research_area:
            results = self.faculty_retriever.search_by_name(name)
        
        # Strategy 2: Department + optional research filter
        elif department:
            if research_area:
                results = self.faculty_retriever.search(
                    department=department,
                    research_area=research_area,
                    scope=scope
                )
            else:
                results = self.faculty_retriever.get_department_faculty(
                    department,
                    limit=None if scope == "complete_list" else 10
                )
        
        # Strategy 3: Research area across all departments
        elif research_area:
            results = self.faculty_retriever.search(
                research_area=research_area,
                scope=scope
            )
        
        # Strategy 4: Fallback to vector search
        if not results and use_vector_fallback:
            return self._route_vector_fallback(query, "faculty_search", classification)
        
        # Format results
        context = self.faculty_retriever.format_for_context(results)
        
        return {
            "results": results,
            "context": context,
            "source": "faculty_retriever",
            "query_info": classification,
            "result_count": len(results)
        }
    
    def _route_courses(self, query: str, entities: Dict, scope: str,
                       classification: Dict, use_vector_fallback: bool) -> Dict[str, Any]:
        """Route course queries using CourseRetriever."""
        course_code = entities.get("course_code")
        subject = entities.get("subject")
        level = entities.get("level")
        credits = entities.get("credits")
        
        results = []
        limit = 50 if scope == "complete_list" else 10
        
        # Strategy 1: Exact course code lookup
        if course_code:
            course = self.course_retriever.get_course(course_code)
            if course:
                results = [course]
        
        # Strategy 2: Subject + optional level filter
        elif subject:
            results = self.course_retriever.search_by_subject(subject, limit=limit)
            # Filter by level if specified
            if level and results:
                results = [c for c in results if c.get("level", "").lower() == level.lower()]
        
        # Strategy 3: Level only
        elif level:
            results = self.course_retriever.search_by_level(level, limit=limit)
        
        # Strategy 4: General search.
        # For long conversational queries ("I'm trying to enroll in a machine learning
        # course, any recommendations?"), extract the key topic phrase so the keyword
        # search finds the right courses rather than noise-matching unrelated ones.
        if not results:
            import re as _re
            _tech_topic_re = _re.compile(
                r"\b(machine\s+learning|deep\s+learning|artificial\s+intelligence|"
                r"data\s+structures?|algorithms?|operating\s+systems?|computer\s+networks?|"
                r"computer\s+vision|natural\s+language\s+processing|nlp|robotics|"
                r"embedded\s+systems?|software\s+engineering|cybersecurity|compilers?|"
                r"data\s+science|neural\s+networks?|reinforcement\s+learning)\b",
                _re.IGNORECASE,
            )
            _topic_match = _tech_topic_re.search(query)
            search_term = _topic_match.group(0) if _topic_match else query
            results = self.course_retriever.search(search_term, limit=limit)
        
        # Strategy 5: Fallback to vector search if still no results
        if not results and use_vector_fallback:
            return self._route_vector_fallback(query, "course_info", classification)
        
        # Format results
        context = self.course_retriever.format_for_context(results)
        
        return {
            "results": results,
            "context": context,
            "source": "course_retriever",
            "query_info": classification,
            "result_count": len(results)
        }
    
    def _route_dining(self, query: str, entities: Dict, scope: str,
                      classification: Dict) -> Dict[str, Any]:
        """Route dining queries using CampusRetriever."""
        limit = 20 if scope == "complete_list" else 5
        
        results = self.campus_retriever.search_dining(query, limit=limit)
        context = self.campus_retriever.format_dining_context(results)
        
        return {
            "results": results,
            "context": context,
            "source": "campus_retriever",
            "query_info": classification,
            "result_count": len(results)
        }
    
    def _route_housing(self, query: str, entities: Dict, scope: str,
                       classification: Dict) -> Dict[str, Any]:
        """Route housing queries using CampusRetriever."""
        limit = 50 if scope == "complete_list" else 10
        
        results = self.campus_retriever.search_housing(query, limit=limit)
        context = self.campus_retriever.format_housing_context(results)
        
        return {
            "results": results,
            "context": context,
            "source": "campus_retriever",
            "query_info": classification,
            "result_count": len(results)
        }
    
    def _route_transit(self, query: str, entities: Dict, scope: str,
                       classification: Dict) -> Dict[str, Any]:
        """Route transit queries using CampusRetriever."""
        limit = 20 if scope == "complete_list" else 5
        
        # Check if asking specifically about KU routes
        query_lower = query.lower()
        if 'ku' in query_lower or 'campus' in query_lower:
            results = self.campus_retriever.get_ku_transit()[:limit]
        else:
            results = self.campus_retriever.search_transit(query, limit=limit)
        
        context = self.campus_retriever.format_transit_context(results)
        
        return {
            "results": results,
            "context": context,
            "source": "campus_retriever",
            "query_info": classification,
            "result_count": len(results)
        }
    
    def _route_tuition(self, query: str, entities: Dict, scope: str,
                       classification: Dict) -> Dict[str, Any]:
        """Route tuition/financial queries using CampusRetriever."""
        limit = 30 if scope == "complete_list" else 10
        
        results = self.campus_retriever.search_tuition(query, limit=limit)
        context = self.campus_retriever.format_tuition_context(results)
        
        return {
            "results": results,
            "context": context,
            "source": "campus_retriever",
            "query_info": classification,
            "result_count": len(results)
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

        # Re-rank vector results for better precision
        reranker = _get_reranker()
        if reranker and specific_results and len(specific_results) > 3:
            specific_results = reranker.rerank(query, specific_results, top_k=5)

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