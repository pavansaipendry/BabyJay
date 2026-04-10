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
import anthropic


class QueryClassifier:
    """Classifies queries to determine intent and extract entities."""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
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
            # Highest priority — specific EECS degree-program questions.
            # These patterns require BOTH a degree / program word AND an EECS
            # subject marker so we don't steal dining or housing queries.
            # Phase 3 — leadership + advising must come FIRST so they beat
            # more generic patterns.
            "eecs_leadership_info": [
                r"\b(chair|chairperson)\s+of\s+(?:the\s+)?eecs\b",
                r"\beecs\s+(?:department\s+)?chair\b",
                r"\bwho(?:'s|\s+is)\s+(?:the\s+)?(?:eecs\s+)?chair\b",
                r"\bassociate\s+chair\b",
                r"\beecs\s+(?:department\s+)?(?:leadership|head|director)\b",
                r"\bwho\s+(?:runs|leads|directs|heads|is\s+in\s+charge\s+of)\s+(?:eecs|i2s|ittc|cresis|the\s+institute)\b",
                r"\bwho\s+(?:runs|leads|directs|heads)\s+(?:the\s+)?(?:department|institute|center)\b",
                r"\b(graduate|undergrad(?:uate)?)\s+(?:studies\s+)?director\b",
                r"\b(i2s|ittc|cresis)\s+director\b",
                r"\bdirector\s+of\s+(?:the\s+)?(?:i2s|ittc|cresis|institute|center)\b",
            ],
            "eecs_advising_info": [
                r"\beecs\s+advis(?:or|ing|ers?)\b",
                r"\badvis(?:or|ing|ers?)\b.*\beecs\b",
                r"\b(book|schedule|make)\s+an?\s+advising\b",
                r"\b(who\s+advises|undergrad(?:uate)?\s+advisor)\b",
                r"\beecs\s+(?:department\s+)?office\b",
                r"\bcontact\s+eecs\b",
                r"\beecs\s+(?:main\s+)?contact\b",
            ],
            # Phase 2 — EECS scoped intents. Ordering matters: more specific
            # location/facility patterns come BEFORE the broad research
            # patterns so "labs in Eaton Hall" routes to facility_info.
            "eecs_facility_info": [
                r"\b(eaton\s+hall|nichols\s+hall)\b",
                r"\b(computing\s+commons|eecs\s+shop|eecs\s+(?:hardware\s+)?labs?)\b",
                r"\b(where\s+is\s+eecs|eecs\s+building|eecs\s+location)\b",
                # "labs in Eaton" / "labs inside Nichols" etc.
                r"\blabs?\b.*\b(eaton|nichols)\b",
            ],
            "eecs_research_info": [
                r"\b(ittc|i2s|cresis)\b",
                r"\b(research\s+cluster|research\s+group|research\s+area|research\s+center|research\s+facilit)",
                r"\beecs\b.*\bresearch\b",
                r"\bresearch\b.*\beecs\b",
                # Cluster name mentions — include "cybersecurity" so standalone
                # "cybersecurity research at KU" routes here.
                r"\b(applied\s+electromagnetics|communication\s+systems|computational\s+science|computer\s+systems\s+design|computing\s+in\s+the\s+biosciences|cybersecurity|language\s+and\s+semantics|radar\s+systems|remote\s+sensing|rf\s+systems|signal\s+processing|theory\s+of\s+computing)\b.*\bresearch\b",
                r"\bresearch\b.*\b(cybersecurity|radar|signal\s+processing|rf|electromagnetics|bioscience)\b",
            ],
            "eecs_student_org_info": [
                r"\b(ku\s+acm|acm\s+chapter|acm\s+at\s+ku|kuacm)\b",
                r"\b(hackku|hack\s+ku|ku\s+hackathon|hackathon)\b",
                r"\b(ieee\s+(?:ku|at\s+ku|chapter|student))\b",
                r"\b(hkn|eta\s+kappa\s+nu)\b",
                r"\b(ku\s+wic|women\s+in\s+computing|kuwic)\b",
                r"\b(upsilon\s+pi\s+epsilon|upe)\b",
                r"\b(jayhackers|information\s+security\s+club)\b",
                r"\b(eecs\s+(?:student\s+)?(?:club|organization|org|tutoring))\b",
                # "tutoring for EECS 168" — tutoring at KU is run by KU ACM,
                # so route tutoring to orgs. Match even when a course code
                # appears (so it beats course_info's +3).
                r"\btutoring\b",
                r"\b(peer\s+help|help\s+with\s+(?:eecs|cs))\b",
            ],
            "eecs_grad_admissions_info": [
                r"\beecs\b.*\b(grad|graduate|phd|ph\.d\.|masters?|m\.s|admission|deadline|apply)\b",
                r"\b(grad|graduate|phd|ph\.d\.|masters?|m\.s)\b.*\beecs\b",
                r"\b(grad(?:uate)?\s+(?:funding|assistantship|gta|gra|fellowship|stipend))\b",
                r"\b(deficiency\s+courses?)\b",
                r"\b(special\s+graduate\s+admissions?)\b",
                r"\b(accelerated\s+(?:bs[/\s]*ms|masters|4\s*\+\s*1))\b",
                r"\b(graduate\s+application\s+deadline|grad\s+app\s+deadline)\b",
            ],
            "eecs_scholarship_info": [
                r"\b(eecs|engineering)\s+scholarships?\b",
                r"\b(garmin|summerfield|watkins[-\s]?berger|jayhawk\s+sfs|cybercorps|ukash)\b",
                r"\bscholarships?\b.*\b(eecs|cs|compe|engineering|computer\s+science|computer\s+engineering|electrical\s+engineering|cyber)\b",
                r"\b(eecs|cs|compe|computer\s+science|computer\s+engineering|electrical\s+engineering)\b.*\bscholarships?\b",
            ],
            "eecs_career_info": [
                r"\b(engineering\s+career\s+center|ecc)\b",
                r"\b(career\s+fair)\b.*\b(engineering|eecs|ku)\b",
                r"\b(internship|co[-\s]?op)\b.*\b(eecs|cs|engineering)\b",
                r"\b(companies\s+recruit)\b.*\b(ku|eecs)\b",
                r"\bhandshake\b",
            ],
            "eecs_program_info": [
                # "BS CS requirements", "bachelor of computer science"
                r"\b(bs|b\.s\.|bachelor|masters?|m\.s\.|ms|phd|ph\.d\.|doctorate|doctoral|m\.eng|meng)\b.*\b(computer\s+science|computer\s+engineering|electrical\s+engineering|cybersecurity(?:\s+engineering)?|applied\s+computing|cs|compe|ee|eecs)\b",
                r"\b(computer\s+science|computer\s+engineering|electrical\s+engineering|cybersecurity(?:\s+engineering)?|applied\s+computing|eecs)\b.*\b(degree|program|requirements?|curriculum|plan|credit\s+hours?|prereq|core|electives?|major|minor)\b",
                # Reverse order — "credit hours for BS CS", "requirements for EECS BS"
                r"\b(credit\s+hours?|total\s+hours?|degree|program|requirements?|curriculum|plan|major)\b.*\b(eecs|bs|ms|phd|bachelor|master|doctor|cs|compe|ee|computer\s+science|computer\s+engineering|electrical\s+engineering|cybersecurity|applied\s+computing)\b",
                r"\b(4[-\s]?year\s+plan|four[-\s]?year\s+plan|degree\s+plan|course\s+sequence|core\s+courses)\b.*\b(cs|compe|ee|eecs|computer|electrical|cyber)\b",
                r"\b(learning\s+outcomes)\b.*\b(cs|compe|ee|eecs|computer|electrical|cyber)\b",
                r"\b(accelerated\s+(?:bs[/\s]*ms|masters|4\+1))\b",
                # "cybersecurity certificate" / "data science graduate certificate" / "cybersecurity cert"
                r"\b(cybersecurity|data\s+science|rf\s+systems?)\s+(?:undergraduate\s+|ug\s+|graduate\s+|grad\s+)?cert(?:ificate)?\b",
                r"\b(undergraduate\s+certificate|ug\s+certificate|graduate\s+certificate|grad\s+certificate)\s+in\s+(cybersecurity|data\s+science|rf)",
                # direct "BS CS" / "MS CS" / "PhD CS" shorthand
                r"\b(bs|ms|phd|meng)\s+(cs|compe|ee|eecs|computer\s+science|electrical\s+engineering)\b",
            ],
            "housing_info": [
                r"\b(housing|dorm|dormitory|residence hall|apartment|living|roommate)\b",
                r"\b(scholarship hall|scholarship halls)\b",
                r"\b(corbin|ellsworth|hashinger|lewis|oswald|templin|downs|gsp)\b",
                r"\bwhere.*(live|stay)\b",
            ],
            "faculty_search": [
                r"\b(professor|professors|faculty|researcher|researchers|instructor|instructors|teacher|teachers)\b",
                r"\b(who teaches|who does research|expert in|specialist in)\b",
                # Require an actual faculty noun later in the sentence — avoids
                # scoring "research in quantum" as a faculty query.
                r"\b(research(?:ing)?\s+(?:in|on)|working\s+on|studies|studying)\b.*\b(professor|professors|faculty|researcher|researchers)\b",
                # "tell me about [Name] in EECS / as a professor" and "who is [Name] in EECS"
                r"\b(tell me about|who is|who's|about)\b.+\b(eecs|professor|prof|faculty|department|dept|researcher|phd)\b",
                r"\b(eecs|professor|prof|faculty|phd)\b.+\b(tell me about|who is|who's)\b",
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
                # Only count raw field names as course signals when they appear
                # next to a course-y noun — otherwise "physics professor" scores
                # course_info as well as faculty_search.
                r"\b(calculus|physics|chemistry|biology|engineering)\s+(course|courses|class|classes|major|department|prereq)",
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
                # Guard: don't trust an LLM faculty_search classification unless
                # the original query actually contains a faculty cue. Prevents
                # bare department names like "history of KU" or "physics" from
                # dumping the entire department roster at the user.
                if llm_result.get("intent") == "faculty_search" and not self._has_faculty_cue(query_lower):
                    llm_result["intent"] = "general"
                    llm_result["entities"] = {}
                return llm_result

        return {
            "intent": intent,
            "entities": entities,
            "scope": scope,
            "confidence": intent_confidence,
            "method": "regex",
            "original_query": query
        }

    # Compiled once — any of these tokens in the query is a lexical "faculty cue"
    _FACULTY_CUE_RE = re.compile(
        r"\b(professor|professors|prof|profs|faculty|researcher|researchers|"
        r"instructor|instructors|teacher|teachers|advisor|advisors|dean|deans|"
        r"who\s+teaches|who\s+does\s+research|expert\s+in|specialist\s+in|dr\.?|"
        r"eecs|phd|ph\.d)\b",
        re.IGNORECASE,
    )

    def _has_faculty_cue(self, query_lower: str) -> bool:
        return bool(self._FACULTY_CUE_RE.search(query_lower))
    
    def _detect_intent_regex(self, query_lower: str, query_original: str) -> tuple:
        """Detect intent using regex patterns. Returns (intent, confidence)."""
        scores = {}

        # Hard overrides — these semantics are unambiguous enough to skip
        # the scoring contest entirely.
        if re.search(r"\btutoring\b", query_lower) and re.search(
            r"\b(eecs|cs|course|class|\d{3,4})\b", query_lower
        ):
            return ("eecs_student_org_info", 0.9)
        if re.search(r"\b(cs|computer\s+science)\s+minor\b", query_lower) or \
           re.search(r"\bminor\s+in\s+(?:cs|computer\s+science)\b", query_lower):
            return ("eecs_program_info", 0.9)
        if re.search(r"\b(cs|eecs|computer\s+science|computer\s+engineering|electrical\s+engineering)\s+(?:phd|ph\.d|doctoral|doctorate|masters?)\b", query_lower):
            return ("eecs_grad_admissions_info", 0.9)
        if re.search(r"\b(phd|ph\.d|doctoral|doctorate|masters?|m\.s)\s+in\s+(?:cs|eecs|computer\s+science|computer\s+engineering|electrical\s+engineering)\b", query_lower):
            return ("eecs_grad_admissions_info", 0.9)

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
        
        # Extract potential name. Require a Dr/Prof prefix OR a clear possessive
        # ("Jane Doe's office") — otherwise "Rock Chalk office" got extracted as
        # a fake name and broke name-based faculty lookups.
        name_patterns = [
            # "Dr. Jane" / "Professor Jane Doe" — prefix is mandatory
            r"(?:dr\.?|professor|prof\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            # "Jane Doe's office" — must have the 's possessive
            r"([A-Z][a-z]+\s+[A-Z][a-z]+)'s\s+(?:office|email|research|class|phone|number)",
        ]
        for pattern in name_patterns:
            match = re.search(pattern, query)  # case-sensitive on purpose
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
            classify_system = """Classify the user query for a university chatbot. Return JSON only.

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

            response = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                system=classify_system,
                messages=[{"role": "user", "content": query}],
                temperature=0,
                max_tokens=200,
            )

            result_text = response.content[0].text.strip()

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