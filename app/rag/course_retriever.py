"""
Course Retriever for BabyJay
============================
Fast, flexible course search across all fields.

The retriever finds relevant courses. The LLM answers the question.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from .query_preprocessor import QueryPreprocessor


class CourseRetriever:
    """Flexible course search across all fields."""
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            current = Path(__file__).parent
            while current != current.parent:
                if (current / "data" / "courses").exists():
                    data_dir = str(current / "data")
                    break
                current = current.parent
            else:
                raise FileNotFoundError("Could not find data/courses directory")
        
        self.data_dir = Path(data_dir)
        self.courses_file = self.data_dir / "courses" / "all_courses.json"
        
        # Cache
        self._all_courses: List[Dict] = []
        self._loaded = False
        self._subject_index: Dict[str, List[Dict]] = {}
        self._level_index: Dict[str, List[Dict]] = {}
        self._code_index: Dict[str, Dict] = {}  # course_code -> course (for exact lookup)
        
        # Query preprocessor (initialized after loading courses)
        self._preprocessor: Optional[QueryPreprocessor] = None
    
    def _load_all_courses(self):
        """Load all courses from single file (fast)."""
        if self._loaded:
            return
        
        with open(self.courses_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._all_courses = data.get("courses", [])
        
        # Build indexes
        for course in self._all_courses:
            # Index by subject
            subject = course.get("subject", "").upper()
            if subject:
                if subject not in self._subject_index:
                    self._subject_index[subject] = []
                self._subject_index[subject].append(course)
            
            # Index by level
            level = course.get("level", "").lower()
            if level:
                if level not in self._level_index:
                    self._level_index[level] = []
                self._level_index[level].append(course)
            
            # Index by course code (exact lookup)
            code = course.get("course_code", "")
            if code:
                self._code_index[code.upper()] = course
        
        # Initialize preprocessor with valid subject codes
        self._preprocessor = QueryPreprocessor(set(self._subject_index.keys()))
        
        self._loaded = True
    
    def search(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Flexible search - finds courses matching query across all fields.
        Includes typo correction and synonym expansion.
        """
        self._load_all_courses()
        
        # Preprocess query (normalize, expand synonyms, fix typos)
        prep_result = self._preprocessor.preprocess(query)
        processed_query = prep_result["processed"]
        
        # If empty after processing, return empty results
        if not processed_query:
            return []
        
        query_lower = processed_query.lower().strip()
        query_terms = query_lower.split()
        
        # Check for specific course code (e.g., "AE 345", "EECS 168")
        course_code_match = re.match(r'^([A-Za-z]{2,4})\s+(\d{3,4})$', processed_query.strip(), re.IGNORECASE)
        if course_code_match:
            subject = course_code_match.group(1).upper()
            number = course_code_match.group(2)
            code = f"{subject} {number}"
            if code in self._code_index:
                return [self._code_index[code]]
        
        # Check for subject-only query (e.g., "EECS", "AE")
        if len(query_terms) == 1 and re.match(r'^[A-Za-z]{2,4}$', query_terms[0]):
            subject = query_terms[0].upper()
            if subject in self._subject_index:
                return self._subject_index[subject][:limit]
        
        # General search across all fields
        scored_results = []
        
        for course in self._all_courses:
            score = self._score_course(course, query_terms, query_lower)
            if score > 0:
                scored_results.append((score, course))
        
        # Sort by score (highest first)
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # If no results with processed query, try original query
        if not scored_results and processed_query != query.lower().strip():
            original_terms = query.lower().strip().split()
            for course in self._all_courses:
                score = self._score_course(course, original_terms, query.lower())
                if score > 0:
                    scored_results.append((score, course))
            scored_results.sort(key=lambda x: x[0], reverse=True)
        
        return [course for score, course in scored_results[:limit]]
    
    def _score_course(self, course: Dict, query_terms: List[str], full_query: str) -> int:
        """Score how well a course matches the query."""
        score = 0
        
        # Fields to search with weights
        fields = {
            "course_code": 10,
            "title": 8,
            "subject": 6,
            "description": 3,
            "department": 4,
            "school": 3,
            "level": 5,
            "prerequisites": 2,
        }
        
        for field, weight in fields.items():
            value = course.get(field)
            if not value:
                continue
            
            value_lower = str(value).lower()
            
            # Check each query term
            for term in query_terms:
                if term in value_lower:
                    score += weight
                    
                    # Bonus for exact word match
                    if re.search(rf'\b{re.escape(term)}\b', value_lower):
                        score += weight // 2
        
        # Check credits if query mentions a number
        credits = course.get("credits")
        if credits:
            for term in query_terms:
                if term.isdigit() and int(term) == credits:
                    score += 5
        
        # Level matching
        level = course.get("level", "").lower()
        if "undergraduate" in full_query and level == "undergraduate":
            score += 5
        if "graduate" in full_query and level == "graduate":
            score += 5
        if "grad " in full_query and level == "graduate":
            score += 5
        if "undergrad" in full_query and level == "undergraduate":
            score += 5
        
        return score
    
    def search_by_subject(self, subject: str, limit: int = 50) -> List[Dict]:
        """Get courses by subject code (e.g., 'EECS', 'AE')."""
        self._load_all_courses()
        subject_upper = subject.upper()
        return self._subject_index.get(subject_upper, [])[:limit]
    
    def search_by_level(self, level: str, limit: int = 50) -> List[Dict]:
        """Get courses by level (undergraduate/graduate)."""
        self._load_all_courses()
        level_lower = level.lower()
        return self._level_index.get(level_lower, [])[:limit]
    
    def get_course(self, course_code: str) -> Optional[Dict]:
        """Get a specific course by code (e.g., 'AE 345')."""
        self._load_all_courses()
        
        # Normalize course code
        match = re.match(r'([A-Za-z]{2,4})\s*(\d{3,4})', course_code.strip())
        if not match:
            return None
        
        code = f"{match.group(1).upper()} {match.group(2)}"
        return self._code_index.get(code)
    
    def get_prerequisites(self, course_code: str) -> Optional[str]:
        """Get prerequisites for a specific course."""
        course = self.get_course(course_code)
        if course:
            return course.get("prerequisites")
        return None
    
    def format_for_context(self, courses: List[Dict]) -> str:
        """Format courses for LLM context."""
        if not courses:
            return ""
        
        lines = ["=== COURSE INFORMATION ==="]
        
        for c in courses[:15]:  # Limit to 15 courses for context
            desc = c.get('description', 'N/A')
            if len(desc) > 200:
                desc = desc[:200] + "..."
            
            prereq = c.get('prerequisites', 'None')
            coreq = c.get('corequisites', '')
            coreq_str = f"\nCorequisites: {coreq}" if coreq else ""
            
            lines.append(f"""
Course: {c.get('course_code', 'N/A')} - {c.get('title', 'Unknown')}
Credits: {c.get('credits', 'N/A')}
Level: {c.get('level', 'N/A')}
School: {c.get('school', 'N/A')}
Description: {desc}
Prerequisites: {prereq}{coreq_str}
""")
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict:
        """Get statistics about course data."""
        self._load_all_courses()
        
        return {
            "total_courses": len(self._all_courses),
            "subjects": len(self._subject_index),
            "subjects_list": sorted(list(self._subject_index.keys()))[:20]
        }


# Quick test
if __name__ == "__main__":
    import time
    
    print("Testing Course Retriever")
    print("=" * 60)
    
    start = time.time()
    retriever = CourseRetriever()
    
    # Stats
    stats = retriever.get_stats()
    load_time = (time.time() - start) * 1000
    
    print(f"\nData loaded in {load_time:.0f}ms")
    print(f"  Total Courses: {stats['total_courses']}")
    print(f"  Subjects: {stats['subjects']}")
    print(f"  Sample subjects: {stats['subjects_list'][:10]}")
    
    # Test searches - including typos and synonyms
    test_queries = [
        # Exact matches
        ("AE 345", "Exact course code"),
        ("EECS", "Subject code"),
        
        # Synonyms
        ("ML courses", "ML → machine learning"),
        ("AI intro", "AI → artificial intelligence, intro → introduction"),
        ("prereqs for AE 211", "prereqs → prerequisites"),
        ("grad computer science", "grad → graduate"),
        
        # Typos
        ("machien learning", "Typo: machien → machine"),
        ("introducton to programming", "Typo: introducton → introduction"),
        ("compter science", "Typo: compter → computer"),
        
        # Combined
        ("ML prereqs", "Synonym: both ML and prereqs"),
        
        # Edge cases
        ("EECS168", "Combined course code format"),
        ("3 credit engineering", "Credit filter"),
    ]
    
    print("\n" + "=" * 60)
    print("Search Tests:")
    
    for query, description in test_queries:
        start = time.time()
        results = retriever.search(query, limit=5)
        elapsed = (time.time() - start) * 1000
        
        print(f"\nQuery: '{query}' ({description})")
        print(f"  Results: {len(results)} in {elapsed:.1f}ms")
        
        for r in results[:3]:
            title = r.get('title', 'Unknown')[:40]
            print(f"    - {r.get('course_code')}: {title}")
    
    # Test specific course lookup
    print("\n" + "=" * 60)
    print("Specific Course Lookup:")
    
    course = retriever.get_course("AE 345")
    if course:
        print(f"  {course.get('course_code')}: {course.get('title')}")
        print(f"  Credits: {course.get('credits')}")
        print(f"  Prerequisites: {course.get('prerequisites')}")
    
    print("\n" + "=" * 60)
    print("All tests complete!")