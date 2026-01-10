"""
Retriever Module - Search for relevant information
COMPREHENSIVE FIX: Handles complex queries, department+topic separation, smart extraction
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

from .embeddings import EMBEDDING_MODEL, get_project_root
from .faculty_search import FacultySearcher


class Retriever:
    def __init__(self, persist_directory: str = None):
        if persist_directory is None:
            project_root = get_project_root()
            persist_directory = str(project_root / "data" / "vectordb")
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=EMBEDDING_MODEL
        )
        self.collection = self.client.get_collection(
            name="babyjay_knowledge",
            embedding_function=self.embedding_fn
        )
        self.faculty_searcher = FacultySearcher()
    
    def search(self, query: str, n_results: int = 5, source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        where_filter = {"source": source_filter} if source_filter else None
        results = self.collection.query(
            query_texts=[query], n_results=n_results, where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        formatted = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                formatted.append({
                    "content": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "relevance_score": 1 - results['distances'][0][i] if results['distances'] else 0
                })
        return formatted
    
    def search_dining(self, query: str, n_results: int = 3): return self.search(query, n_results, "dining")
    def search_courses(self, query: str, n_results: int = 5): return self.search(query, n_results, "course")
    def search_admissions(self, query: str, n_results: int = 5): return self.search(query, n_results, "admission")
    def search_calendar(self, query: str, n_results: int = 5): return self.search(query, n_results, "calendar")
    def search_faqs(self, query: str, n_results: int = 5): return self.search(query, n_results, "faq")
    def search_tuition(self, query: str, n_results: int = 5): return self.search(query, n_results, "tuition")
    def search_financial_aid(self, query: str, n_results: int = 5): return self.search(query, n_results, "financial_aid")
    def search_housing(self, query: str, n_results: int = 5): return self.search(query, n_results, "housing")
    def search_libraries(self, query: str, n_results: int = 5): return self.search(query, n_results, "libraries")
    def search_recreation(self, query: str, n_results: int = 5): return self.search(query, n_results, "recreation")
    def search_campus_safety(self, query: str, n_results: int = 5): return self.search(query, n_results, "campus_safety")
    def search_student_organizations(self, query: str, n_results: int = 5): return self.search(query, n_results, "student_organizations")
    
    def search_transit(self, query: str, n_results: int = 3):
        routes = self.search(query, n_results, "transit")
        stops = self.search(query, n_results, "transit_stop")
        all_results = routes + stops
        all_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return all_results[:n_results]
    
    def search_faculty_enhanced(self, query: str, n_results: int = 5, department: str = None):
        results = self.faculty_searcher.search(query, top_k=n_results, department_filter=department)
        formatted = []
        for r in results:
            content = f"""Professor: {r['name']}
                Department: {r['department']}
                Email: {r['email']}
                Phone: {r['phone']}
                Office: {r['office']} {r['building']}
                Profile: {r['profile_url']}
                Research: {r.get('document', 'N/A')}"""
            formatted.append({
                "content": content,
                "metadata": {"source": "faculty", "name": r['name'], "department": r['department'],
                            "email": r['email'], "phone": r['phone'], "office": r['office'],
                            "building": r['building'], "profile_url": r['profile_url']},
                "relevance_score": r['score']
            })
        return formatted
    
    def search_faculty(self, query: str, n_results: int = 5):
        return self.search_faculty_enhanced(query, n_results)
    
    def _detect_departments_from_query(self, query: str) -> List[str]:
        """Detect department names/keywords in query."""
        query_lower = query.lower()
        dept_map = {
            "Electrical": ["eecs", "computer science", "electrical engineering", " cs ", "software"],
            "Business": ["business", "marketing", "finance", "accounting", "mba"],
            "Physics": ["physics", "astronomy", "astrophysics"],
            "Chemistry": ["chemistry", " chem "],
            "Math": ["math", "mathematics", "statistics"],
            "Psychology": ["psychology", " psych "],
            "Engineering": ["mechanical", "civil", "aerospace", "bioengineering"],
            "Law": ["law school", "legal studies"],
            "Music": ["music", "band", "orchestra"],
            "Journalism": ["journalism", "communications"],
            "Pharmacy": ["pharmacy", "pharmaceutical"],
            "Social Welfare": ["social work", "social welfare"],
        }
        matches = [dept for dept, kws in dept_map.items() if any(kw in query_lower for kw in kws)]
        
        # De-dupe, keep order
        seen = set()
        out = []
        for m in matches:
            if m not in seen:
                seen.add(m)
                out.append(m)
        return out
    
    def _extract_department_and_topic(self, query: str) -> Tuple[Optional[str], str]:
        """
        SMART extraction: separate department filter from research topic.
        
        Examples:
        - "ML professors" → (None, "machine learning")
        - "Physics professors doing ML" → ("Physics", "machine learning")
        - "EECS faculty interested in robotics" → ("Electrical", "robotics")
        
        Returns:
            (department_filter, research_topic)
        """
        query_lower = query.lower()
        
        # Detect department
        departments = self._detect_departments_from_query(query)
        dept_filter = departments[0] if departments else None
        
        # Now extract research topic, being MUCH more careful
        topic = query_lower
        
        # Step 1: Remove department keywords (they're handled separately)
        dept_keywords = [
            'eecs', 'computer science', 'electrical engineering', 'business', 'physics',
            'chemistry', 'math', 'mathematics', 'engineering', 'department', 'school of'
        ]
        for kw in dept_keywords:
            topic = topic.replace(kw, ' ')
        
        # Step 2: Remove query structure words (but keep research terms!)
        structure_words = [
            'find', 'show me', 'list', 'give me', 'i want', 'looking for',
            'professors', 'professor', 'faculty', 'researchers', 'researcher',
            'who are', 'who do', 'who does', 'who studies', 'who works on',
            'interested in', 'working on', 'doing research on', 'doing', 'research on'
        ]
        for word in structure_words:
            topic = topic.replace(word, ' ')
        
        # Step 3: Expand common abbreviations (CRITICAL!)
        expansions = {
            ' ml ': ' machine learning ',
            'ml ': 'machine learning ',  # Start of string
            ' ml': ' machine learning',  # End of string
            ' ai ': ' artificial intelligence ',
            'ai ': 'artificial intelligence ',
            ' ai': ' artificial intelligence',
            ' dl ': ' deep learning ',
            ' nlp ': ' natural language processing ',
            ' cv ': ' computer vision ',
        }
        
        for abbr, full in expansions.items():
            topic = topic.replace(abbr, full)
        
        # Step 4: Clean up whitespace
        topic = ' '.join(topic.split())
        
        # Step 5: Validation - if topic is too short or empty, use original query
        if not topic or len(topic) < 3:
            # Fallback: just remove "find", "professors" and keep the rest
            topic = query_lower
            for word in ['find', 'show me', 'professors', 'professor', 'faculty']:
                topic = topic.replace(word, ' ')
            topic = ' '.join(topic.split())
        
        return (dept_filter, topic.strip())
    
    def _contains_word(self, text: str, word: str) -> bool:
        """Check if word exists as a whole word (including common plural forms) in text."""
        text_lower = f" {text.lower()} "
        word_lower = word.lower()
        
        # Check exact match
        if f" {word_lower} " in text_lower:
            return True
        
        # Check common plural forms
        if f" {word_lower}s " in text_lower:  # professor → professors
            return True
        if f" {word_lower}es " in text_lower:  # class → classes
            return True
            
        return False
    
    def _contains_any_word(self, text: str, words: List[str]) -> bool:
        """Check if any word from the list exists in text (with word boundaries)."""
        return any(self._contains_word(text, word) for word in words)

    def _wants_complete_list(self, query: str) -> bool:
        """Detect if user wants a complete list, not just top results."""
        q = query.lower()
        complete_indicators = [
            'all ', 'every ', 'complete list', 'full list', 
            'list all', 'show all', 'all of the', 'how many'
        ]
        return any(indicator in q for indicator in complete_indicators)
    
    def smart_search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        q = query.lower()
        
        # Check if user wants complete list
        if self._wants_complete_list(query):
            n_results = 50  # Return up to 50 for complete list requests

        # --- intent flags with word boundaries ---
        is_dining = self._contains_any_word(q, ['eat', 'food', 'dining', 'restaurant', 'hungry', 'lunch', 'dinner', 'breakfast', 'coffee', 'cafe', 'meal'])
        is_transit = self._contains_any_word(q, ['bus', 'route', 'transit', 'stop', 'ride', 'transportation', 'safebus'])
        is_course = self._contains_any_word(q, ['course', 'class', 'prerequisite', 'credit', 'enroll', 'major', 'degree', 'syllabus'])
        is_building = self._contains_any_word(q, ['hall', 'building', 'eaton', 'strong', 'wescoe', 'fraser', 'watson', 'anschutz', 'union', 'capitol', 'malott'])
        is_office = self._contains_any_word(q, ['office', 'iss', 'registrar', 'career center', 'counseling', 'caps', 'health center', 'parking', 'it help'])
        is_professor = self._contains_any_word(q, ['professor', 'prof', 'faculty', 'teacher', 'instructor', 'research', 'who teaches', 'dr.', 'dr', 'advisor', 'dean'])
        
        # Faculty research keywords - these can be substrings since they're technical terms
        is_faculty_research = any(term in q for term in [
            'machine learning', 'artificial intelligence', 'ai ', 'ml ', 'data science', 'robotics',
            'neural network', 'deep learning', 'computer vision', 'cybersecurity', 'quantum',
            'particle physics', 'climate', 'genetics', 'who does research', 'who studies', 'expert in',
            'interested in', 'working on'
        ])
        
        is_admission = self._contains_any_word(q, ['admission', 'apply', 'application', 'deadline', 'requirement', 'transfer', 'freshman', 'sat', 'act', 'acceptance'])
        is_calendar = self._contains_any_word(q, ['calendar', 'semester', 'finals', 'break', 'holiday', 'classes start', 'classes end', 'spring break', 'thanksgiving'])
        is_faq = self._contains_any_word(q, ['wifi', 'password', 'canvas', 'outlook', 'print', 'ku card', 'vpn', 'how do i', 'how to', 'login'])
        is_tuition = self._contains_any_word(q, ['tuition', 'cost', 'fee', 'credit hour', 'how much', 'payment', 'billing'])
        is_financial_aid = self._contains_any_word(q, ['financial aid', 'fafsa', 'scholarship', 'grant', 'pell', 'loan'])
        is_housing = self._contains_any_word(q, ['housing', 'dorm', 'residence hall', 'apartment', 'meal plan', 'roommate'])
        is_library = self._contains_any_word(q, ['library', 'study room', 'study space', 'borrow', 'checkout'])
        is_recreation = self._contains_any_word(q, ['gym', 'rec', 'recreation', 'ambler', 'fitness', 'workout', 'intramural'])
        is_safety = self._contains_any_word(q, ['safety', 'emergency', 'police', 'kupd', '911', 'security', 'escort'])
        is_student_orgs = self._contains_any_word(q, ['club', 'organization', 'greek', 'fraternity', 'sorority', 'rush'])

        is_faculty = is_professor or is_faculty_research
        if is_faculty:
            is_course = False  # avoid course route for professor-type questions

        results = {
            "dining": [],
            "transit": [],
            "courses": [],
            "buildings": [],
            "offices": [],
            "professors": [],
            "admissions": [],
            "calendar": [],
            "faqs": [],
            "tuition": [],
            "financial_aid": [],
            "housing": [],
            "libraries": [],
            "recreation": [],
            "campus_safety": [],
            "student_organizations": [],
            "faculty": [],
            "context": ""
        }

        # --- per-domain retrieval ---
        if is_dining:
            results["dining"] = self.search(query, min(3, n_results), "dining")
        if is_transit:
            results["transit"] = self.search_transit(query, min(3, n_results))
        if is_course:
            results["courses"] = self.search(query, min(3, n_results), "course")
        if is_building:
            results["buildings"] = self.search(query, min(3, n_results), "building")
        if is_office:
            results["offices"] = self.search(query, min(3, n_results), "office")
        if is_admission:
            results["admissions"] = self.search(query, n_results, "admission")
        if is_calendar:
            results["calendar"] = self.search(query, n_results, "calendar")
        if is_faq:
            results["faqs"] = self.search(query, n_results, "faq")
        if is_tuition:
            results["tuition"] = self.search(query, n_results, "tuition")
        if is_financial_aid:
            results["financial_aid"] = self.search(query, n_results, "financial_aid")
        if is_housing:
            results["housing"] = self.search(query, n_results, "housing")
        if is_library:
            results["libraries"] = self.search(query, n_results, "libraries")
        if is_recreation:
            results["recreation"] = self.search(query, n_results, "recreation")
        if is_safety:
            results["campus_safety"] = self.search(query, n_results, "campus_safety")
        if is_student_orgs:
            results["student_organizations"] = self.search(query, n_results, "student_organizations")

        # --- faculty retrieval (IMPROVED!) ---
        if is_faculty:
            # SMART extraction: separate department from topic
            dept_filter, research_topic = self._extract_department_and_topic(query)
            
            # If we detected a department, use it
            if dept_filter:
                merged = self.search_faculty_enhanced(research_topic, n_results, dept_filter)
            else:
                # No department specified, search all
                merged = self.search_faculty_enhanced(research_topic, n_results, None)
            
            # Dedupe by name only (same person can be in multiple departments)
            seen = set()
            deduped = []
            for r in merged:
                md = r.get("metadata", {}) or {}
                name = md.get("name") or r.get("content", "")[:80]
                if name in seen:
                    continue
                seen.add(name)
                deduped.append(r)

            results["faculty"] = deduped[:n_results]
            
            # FALLBACK: If no results and we had a department filter, try without it
            if not results["faculty"] and dept_filter:
                fallback_results = self.search_faculty_enhanced(research_topic, n_results, None)
                seen = set()
                deduped = []
                for r in merged:
                    md = r.get("metadata", {}) or {}
                    name = md.get("name") or r.get("content", "")[:80]
                    if name in seen:
                        continue
                    seen.add(name)
                    deduped.append(r)
                results["faculty"] = deduped[:n_results]

        # --- build context ---
        ctx = []
        for key, label in [
            ("dining", "DINING"),
            ("transit", "TRANSIT"),
            ("courses", "COURSE"),
            ("buildings", "BUILDING"),
            ("offices", "OFFICE"),
            ("admissions", "ADMISSION"),
            ("calendar", "CALENDAR"),
            ("faqs", "FAQ"),
            ("tuition", "TUITION"),
            ("financial_aid", "FINANCIAL AID"),
            ("housing", "HOUSING"),
            ("libraries", "LIBRARY"),
            ("recreation", "RECREATION"),
            ("campus_safety", "CAMPUS SAFETY"),
            ("student_organizations", "STUDENT ORGANIZATIONS"),
            ("faculty", "FACULTY"),
        ]:
            if results.get(key):
                ctx.append(f"=== {label} INFORMATION ===")
                ctx.extend([r["content"] for r in results[key]])

        results["context"] = "\n\n".join(ctx)
        return results