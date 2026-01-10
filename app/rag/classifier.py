"""
Query Classifier for BabyJay
============================
Classifies user queries to determine intent, entities, and routing.

Uses a hybrid approach:
1. Fast regex/keyword matching for simple queries (no API call)
2. LLM classification for complex/ambiguous queries
"""

import os
import re
import json
from typing import Dict, Any, Optional, List
from openai import OpenAI


class QueryClassifier:
    """Classifies queries to determine intent and extract entities."""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Common subject codes at KU (for course detection)
        self.subject_codes = {
            "EECS", "AE", "ME", "CE", "CHEM", "PHSX", "MATH", "BIOL", "PSYC",
            "ECON", "ENGL", "HIST", "POLS", "SOC", "PHIL", "GEOL", "ASTR",
            "ACCT", "FIN", "MGMT", "MKTG", "JOUR", "COMS", "MUSC", "THEA",
            "DANC", "ART", "ARCH", "LAW", "PHARM", "NURS", "PE", "EDUC",
            "CS", "ECE", "BME", "CHE", "ARCE", "ATMO", "GEOG", "ANTH",
            "LING", "SPAN", "FREN", "GERM", "CLAS", "EALC", "SLAV", "AFST",
            "WGSS", "AMST", "GIST", "HSES", "SPLH", "OT", "PT", "RSI",
            "HDSC", "BIOS", "STAT", "MSCR", "PHTX", "MDCM", "NRSG", "PRNU"
        }
        
        # Department keyword mappings
        self.department_aliases = {
            "eecs": ["eecs", "computer science", "electrical engineering", "cs", "ece", "cse", "software engineering"],
            "business": ["business", "business school", "mba", "marketing", "finance", "accounting", "management"],
            "physics": ["physics", "astronomy", "astrophysics", "particle physics"],
            "chemistry": ["chemistry", "chem"],
            "math": ["math", "mathematics", "statistics", "stats"],
            "psychology": ["psychology", "psych"],
            "mechanical_engineering": ["mechanical engineering", "mechanical", "mech eng"],
            "civil_environmental_architectural_engineering": ["civil engineering", "civil", "environmental engineering", "architectural engineering"],
            "aerospace_engineering": ["aerospace", "aerospace engineering"],
            "bioengineering": ["bioengineering", "biomedical engineering", "bme"],
            "chemical_petroleum_engineering": ["chemical engineering", "petroleum engineering", "petroleum"],
            "law": ["law", "law school", "legal"],
            "journalism": ["journalism", "mass communications", "media"],
            "music": ["music", "band", "orchestra", "choir"],
            "pharmacy": ["pharmacy", "pharmaceutical"],
            "social_welfare": ["social work", "social welfare"],
            "education_human_sciences": ["education", "teaching"],
            "architecture": ["architecture"],
            "english": ["english", "literature", "writing"],
            "history": ["history"],
            "economics": ["economics", "econ"],
            "political_science": ["political science", "politics", "government"],
            "sociology": ["sociology"],
            "anthropology": ["anthropology"],
            "philosophy": ["philosophy"],
            "geology": ["geology", "earth science"],
            "biology": ["biology", "bio"],
            "ecology_evolutionary_biology": ["ecology", "evolutionary biology", "evolution"],
            "molecular_biosciences": ["molecular biology", "biosciences", "biochemistry"],
        }
        
        # Research area keywords
        self.research_areas = {
            "machine learning": ["machine learning", "ml", "deep learning", "neural network", "ai", "artificial intelligence"],
            "robotics": ["robotics", "robot", "autonomous"],
            "computer vision": ["computer vision", "cv", "image processing", "image recognition"],
            "natural language processing": ["natural language processing", "nlp", "text mining", "language model"],
            "cybersecurity": ["cybersecurity", "security", "cryptography", "privacy"],
            "data science": ["data science", "data mining", "big data", "analytics"],
            "quantum computing": ["quantum computing", "quantum", "quantum physics"],
            "networking": ["networking", "networks", "wireless", "communications"],
            "databases": ["database", "databases", "sql", "data management"],
            "software engineering": ["software engineering", "software development"],
            "human computer interaction": ["hci", "human computer interaction", "user experience", "ux"],
        }
        
        # Intent patterns - ORDER MATTERS (more specific patterns first)
        self.intent_patterns = {
            "housing_info": [
                r"\b(housing|dorm|dormitory|residence hall|apartment|living|roommate)\b",
                r"\b(scholarship hall|scholarship halls)\b",
                r"\b(corbin|ellsworth|hashinger|lewis|oswald|templin|downs|gsp)\b",
                r"\bwhere.*(live|stay)\b",
            ],
            "faculty_search": [
                r"\b(professor|professors|faculty|researcher|researchers|instructor|instructors|teacher|teachers)\b",
                r"\b(who teaches|who does research|expert in|specialist in)\b",
                r"\b(research in|working on|studies|studying)\b.*\b(professor|faculty)?\b",
            ],
            "course_info": [
                # Course code patterns (EECS 168, AE 345, etc.)
                r"\b[A-Z]{2,4}\s*\d{3,4}\b",
                # Keywords
                r"\b(course|courses|class|classes)\b",
                r"\b(prerequisite|prerequisites|prereq|prereqs|corequisite|corequisites)\b",
                r"\b(credit|credits|credit hour|credit hours)\b",
                r"\b(enroll|enrollment|register|registration)\b",
                r"\b(syllabus|curriculum)\b",
                r"\b(learning|programming|calculus|physics|chemistry|biology|engineering)\b",
                r"\b(learning|programming|calculus|physics|chemistry|biology|engineering)\b",
                r"\b(learning|programming|calculus|physics|chemistry|biology|engineering)\b",
                # Level + course indicators
                r"\b(undergraduate|graduate|grad|undergrad)\s+(course|courses|class|classes|level)\b",
                # Question patterns
                r"\bwhat.*(course|class|prerequisite|prereq)\b",
                r"\b(take|taking|need to take)\b.*\b(course|class)\b",
                r"\bprereq.*for\b",
            ],
            "dining_info": [
                r"\b(eat|food|dining|restaurant|hungry|lunch|dinner|breakfast|coffee|cafe|cafeteria|meal)\b",
                r"\bwhere.*(eat|food|hungry)\b",
            ],
            "transit_info": [
                r"\b(bus|transit|route|transportation|shuttle|parking|safebus|ride)\b",
            ],
            "building_info": [
                r"\b(building|where is|location of|find|directions to)\b",
            ],
            "admission_info": [
                r"\b(admission|admissions|apply|application|deadline|requirement|transfer|freshman|accept)\b",
            ],
            "financial_info": [
                r"\b(tuition|cost|fee|fees|how much|payment|billing)\b",
                r"\b(financial aid|fafsa|scholarship|grant|pell|loan)\b",
                r"\b(undergraduate|graduate).*(cost|tuition|fee|fees|price)\b",
                r"\b(cost|tuition|fee|fees|price).*(undergraduate|graduate)\b",
            ],
            "library_info": [
                r"\b(library|libraries|study room|study space|borrow|checkout)\b",
            ],
            "recreation_info": [
                r"\b(gym|rec center|recreation|fitness|workout|intramural|ambler)\b",
            ],
            "safety_info": [
                r"\b(safety|emergency|police|security|escort|911|kupd)\b",
            ],
            "calendar_info": [
                r"\b(calendar|semester|finals|break|holiday|when does)\b",
            ],
        }
        
        # Scope indicators
        self.complete_list_indicators = [
            r"\ball\b", r"\bevery\b", r"\bcomplete list\b", r"\bfull list\b",
            r"\blist all\b", r"\bshow all\b", r"\bhow many\b", r"\bentire\b"
        ]
    
    def classify(self, query: str, use_llm_fallback: bool = True) -> Dict[str, Any]:
        """
        Classify a query and extract entities.
        """
        query_lower = query.lower().strip()
        
        # Step 1: Detect intent using regex (fast)
        intent, intent_confidence = self._detect_intent_regex(query_lower, query)
        
        # Step 2: Extract entities based on intent
        if intent == "faculty_search":
            entities = self._extract_faculty_entities(query_lower)
        elif intent == "course_info":
            entities = self._extract_course_entities(query)
        else:
            entities = {}
        
        # Step 3: Detect scope
        scope = self._detect_scope(query_lower)
        
        # Step 4: If low confidence and LLM fallback enabled, use LLM
        if intent_confidence < 0.7 and use_llm_fallback:
            llm_result = self._classify_with_llm(query)
            if llm_result and llm_result.get("confidence", 0) > intent_confidence:
                return llm_result
        
        return {
            "intent": intent,
            "entities": entities,
            "scope": scope,
            "confidence": intent_confidence,
            "method": "regex",
            "original_query": query
        }
    
    def _detect_intent_regex(self, query_lower: str, query_original: str) -> tuple:
        """Detect intent using regex patterns. Returns (intent, confidence)."""
        scores = {}
        
        # Check for course codes first (high priority)
        if re.search(r"\b[A-Z]{2,4}\s*\d{3,4}\b", query_original):
            scores["course_info"] = 3  # High score for explicit course code
        
        for intent, patterns in self.intent_patterns.items():
            score = scores.get(intent, 0)
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    score += 1
            if score > 0:
                scores[intent] = score
        
        if not scores:
            return ("general", 0.3)
        
        # Get highest scoring intent
        best_intent = max(scores, key=scores.get)
        max_score = scores[best_intent]
        
        # Calculate confidence based on match strength
        # confidence = min(0.5 + (max_score * 0.15), 0.95)
        confidence = min(0.6 + (max_score * 0.15), 0.95)
        
        return (best_intent, confidence)
    
    def _extract_faculty_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities for faculty search (department, research area, name)."""
        entities = {}
        
        # Extract department
        for dept_key, aliases in self.department_aliases.items():
            for alias in aliases:
                pattern = rf"\b{re.escape(alias)}\b"
                if re.search(pattern, query, re.IGNORECASE):
                    entities["department"] = dept_key
                    break
            if "department" in entities:
                break
        
        # Extract research area
        for area, keywords in self.research_areas.items():
            for keyword in keywords:
                pattern = rf"\b{re.escape(keyword)}\b"
                if re.search(pattern, query, re.IGNORECASE):
                    entities["research_area"] = area
                    break
            if "research_area" in entities:
                break
        
        # Extract potential name
        name_patterns = [
            r"(?:dr\.?|professor|prof\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"([A-Z][a-z]+\s+[A-Z][a-z]+)(?:'s)?\s+(?:office|email|research|class)",
        ]
        for pattern in name_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                entities["name"] = match.group(1).strip()
                break
        
        return entities
    
    def _extract_course_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities for course search (subject, course_code, level, credits)."""
        entities = {}
        query_lower = query.lower()
        
        # Extract course code (e.g., "EECS 168", "AE345", "EECS168")
        course_code_match = re.search(r"\b([A-Z]{2,4})\s*(\d{3,4})\b", query, re.IGNORECASE)
        if course_code_match:
            subject = course_code_match.group(1).upper()
            number = course_code_match.group(2)
            entities["course_code"] = f"{subject} {number}"
            entities["subject"] = subject
        
        # Extract subject code only (if no full course code found)
        if "subject" not in entities:
            for code in self.subject_codes:
                pattern = rf"\b{code}\b"
                if re.search(pattern, query, re.IGNORECASE):
                    entities["subject"] = code
                    break
        
        # Extract level
        if re.search(r"\b(graduate|grad)\b", query_lower) and not re.search(r"\bundergrad", query_lower):
            entities["level"] = "graduate"
        elif re.search(r"\b(undergraduate|undergrad)\b", query_lower):
            entities["level"] = "undergraduate"
        
        # Extract credit hours
        credit_match = re.search(r"\b(\d)\s*(?:credit|cr|hour)", query_lower)
        if credit_match:
            entities["credits"] = int(credit_match.group(1))
        
        return entities
    
    def _detect_scope(self, query: str) -> str:
        """Detect if user wants a complete list or just top results."""
        for pattern in self.complete_list_indicators:
            if re.search(pattern, query, re.IGNORECASE):
                return "complete_list"
        return "top_results"
    
    def _classify_with_llm(self, query: str) -> Optional[Dict[str, Any]]:
        """Use LLM for complex/ambiguous query classification."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """Classify the user query for a university chatbot. Return JSON only.

Intents: faculty_search, dining_info, housing_info, transit_info, course_info, building_info, admission_info, financial_info, library_info, recreation_info, safety_info, calendar_info, general

For faculty_search, extract:
- department: eecs, business, physics, chemistry, math, psychology, etc.
- research_area: machine learning, robotics, cybersecurity, etc.
- name: if asking about specific person

For course_info, extract:
- course_code: e.g., "EECS 168", "AE 345"
- subject: e.g., "EECS", "AE", "MATH"
- level: "undergraduate" or "graduate"
- credits: number of credit hours if mentioned

Return format:
{"intent": "...", "entities": {...}, "scope": "top_results|complete_list", "confidence": 0.0-1.0}"""
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                temperature=0,
                max_tokens=200
            )
            
            result_text = response.choices[0].message.content.strip()
            
            if result_text.startswith("```"):
                result_text = re.sub(r"```json?\n?", "", result_text)
                result_text = result_text.rstrip("`")
            
            result = json.loads(result_text)
            result["method"] = "llm"
            result["original_query"] = query
            return result
            
        except Exception as e:
            return None
    
    def get_department_key(self, alias: str) -> Optional[str]:
        """Get the canonical department key from an alias."""
        alias_lower = alias.lower().strip()
        for dept_key, aliases in self.department_aliases.items():
            if alias_lower in [a.lower() for a in aliases]:
                return dept_key
        return None
    
    def get_all_departments(self) -> List[str]:
        """Get list of all department keys."""
        return list(self.department_aliases.keys())


# Quick test
if __name__ == "__main__":
    print("Testing Query Classifier")
    print("=" * 60)
    
    classifier = QueryClassifier()
    
    test_queries = [
        # Faculty queries
        ("EECS professors", "faculty_search"),
        ("ML researchers in business school", "faculty_search"),
        
        # Course queries
        ("EECS 168", "course_info"),
        ("prerequisites for AE 345", "course_info"),
        ("machine learning courses", "course_info"),
        ("graduate EECS courses", "course_info"),
        ("3 credit math courses", "course_info"),
        ("what courses should I take", "course_info"),
        
        # Other queries
        ("where can I eat on campus", "dining_info"),
        ("bus routes", "transit_info"),
        ("scholarship halls", "housing_info"),
        ("how much is tuition", "financial_info"),
    ]
    
    for query, expected in test_queries:
        result = classifier.classify(query, use_llm_fallback=False)
        status = "✓" if result["intent"] == expected else "✗"
        print(f"\n{status} Query: '{query}'")
        print(f"  Expected: {expected}, Got: {result['intent']} (conf: {result['confidence']:.2f})")
        if result["entities"]:
            print(f"  Entities: {result['entities']}")
    
    print("\n" + "=" * 60)
    print("All tests complete!")