"""
Query Router for BabyJay
========================
Routes classified queries to the appropriate retriever.

Flow:
    User Query → Classifier → Router → Retriever → Results

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


class QueryRouter:
    """Routes queries to appropriate retrievers based on classification."""
    
    def __init__(self):
        """Initialize router with classifier and retrievers."""
        self.classifier = QueryClassifier()
        self.faculty_retriever = FacultyRetriever()
        self.campus_retriever = CampusRetriever()
        self.course_retriever = CourseRetriever()
        
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
        # Step 0: Preprocess query (fix typos, expand synonyms)
        from .query_preprocessor import QueryPreprocessor
        preprocessor = QueryPreprocessor()
        prep_result = preprocessor.preprocess(query)
        processed_query = prep_result.get("processed", query) or query

        # Step 1: Classify the query
        classification = self.classifier.classify(processed_query)
        intent = classification.get("intent", "general")
        entities = classification.get("entities", {})
        scope = classification.get("scope", "top_results")
        
        # Step 2: Route to appropriate retriever
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
        
        elif intent in ["admission_info", "library_info", "recreation_info", 
                        "safety_info", "calendar_info", "building_info"]:
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
    
    def _empty_result(self, classification: Dict) -> Dict[str, Any]:
        """Return empty result when no retriever is available."""
        return {
            "results": [],
            "context": "",
            "source": "none",
            "query_info": classification,
            "result_count": 0
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
        
        # Strategy 4: General search (uses preprocessor for typos/synonyms)
        if not results:
            results = self.course_retriever.search(query, limit=limit)
        
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
        """Fallback to vector search for queries without specialized retrievers."""
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
        context = results.get("context", "")
        
        result_key = intent_to_method.get(intent, "")
        specific_results = results.get(result_key, [])
        
        return {
            "results": specific_results,
            "context": context,
            "source": "vector_retriever",
            "query_info": classification,
            "result_count": len(specific_results) if specific_results else 0
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