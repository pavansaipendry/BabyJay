"""
BabyJay Chat - COMPLETE VERSION + TOPIC-TO-COURSE RESOLVER
============================================================
NEW: When topic search fails (e.g., "Deep Reinforcement Learning"),
uses RAG to find related course codes and retries live lookup.

Flow:
  User: "seats for Deep Reinforcement Learning"
    → KU search "Deep Reinforcement Learning" → No results
    → RAG search finds EECS 700 mentions reinforcement learning
    → Retry KU search with "EECS 700" → Found! 14 seats available
"""

import os
import json
import uuid
import re
import time as time_module
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Tuple

from openai import OpenAI
from app.rag.router import QueryRouter
from app.rag.rlhf_optimizer import RLHFOptimizer
from dotenv import load_dotenv

load_dotenv()

# Import live course lookup
try:
    from app.tools.live_course_lookup import lookup_course, format_sections_for_chat
    LIVE_LOOKUP_AVAILABLE = True
except ImportError:
    LIVE_LOOKUP_AVAILABLE = False
    print("[WARNING] Live course lookup not available")

# Import embedding-based intent detector
try:
    from app.tools.intent_detector import LiveCourseIntentDetector
    INTENT_DETECTOR_AVAILABLE = True
except ImportError:
    INTENT_DETECTOR_AVAILABLE = False
    print("[WARNING] Intent detector not available, using regex only")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ==================== TOPIC TO COURSE MAPPING (FALLBACK) ====================
# Used when RAG doesn't find a match. Maps common topics to course codes.
# This is manually maintained but provides reliable fallback.

TOPIC_TO_COURSE_FALLBACK = {
    # Machine Learning & AI
    "machine learning": ["EECS 658", "EECS 836"],
    "deep learning": ["EECS 738", "EECS 700"],
    "reinforcement learning": ["EECS 700"],
    "deep reinforcement learning": ["EECS 700"],
    "artificial intelligence": ["EECS 649", "EECS 738"],
    "neural network": ["EECS 738", "EECS 700"],
    "neural networks": ["EECS 738", "EECS 700"],
    "computer vision": ["EECS 841"],
    "natural language processing": ["EECS 731"],
    "nlp": ["EECS 731"],
    "data science": ["EECS 731", "EECS 658"],
    "robotics": ["EECS 690", "EECS 700"],
    "mobile robotics": ["EECS 700"],
    
    # Other CS topics
    "algorithms": ["EECS 660", "EECS 700"],
    "high performance computing": ["EECS 700"],
    "hpc": ["EECS 700"],
    "cyber physical systems": ["EECS 700"],
    "program synthesis": ["EECS 700"],
    "software engineering": ["EECS 448"],
    "database": ["EECS 647"],
    "databases": ["EECS 647"],
    "operating systems": ["EECS 678"],
    "computer networks": ["EECS 780"],
    "cybersecurity": ["EECS 710"],
    "security": ["EECS 710"],
    
    # Business/Analytics
    "supply chain": ["BSAN 460", "SCM 401"],
    "business analytics": ["BSAN 440", "BSAN 460"],
    "data analytics": ["BSAN 440"],
}


# ==================== GREETING PATTERNS ====================

GREETING_PATTERNS = [
    "hi", "hello", "hey", "sup", "yo", "hola", "howdy",
    "good morning", "good afternoon", "good evening", "morning",
    "hi there", "hello there", "hey there",
    "what's up", "whats up", "wassup", "how's it going", "hows it going",
    "how are you", "how are you doing", "how you doing", "how're you",
    "how do you do", "what's good", "whats good", "how's everything",
    "hows everything", "how's life", "hows life", "how are things",
    "thanks", "thank you", "thx", "ty", "thanks a lot", "thank you so much",
    "bye", "goodbye", "see ya", "later", "take care", "cya", "peace",
]

GREETING_REGEX_PATTERNS = [
    r"^how('?s| is| are)?\s*(it going|you|things|everything|life|u|ya).*$",
    r"^what'?s?\s*(up|good|happening|going on).*$",
    r"^hey+\s*(there)?[!?.]*$",
    r"^hi+[!?.]*$",
    r"^hello+[!?.]*$",
    r"^yo+[!?.]*$",
]

SYSTEM_PROMPT = """You are BabyJay, KU's friendly campus assistant. You ONLY help with KU-related topics.

SCOPE - You CAN answer:
- KU courses, prerequisites, credits, schedules
- KU professors, research, departments
- Campus services: dining, housing, transit, library, recreation
- Admissions, tuition, financial aid
- KU buildings, locations, offices
- Student organizations, events at KU

SCOPE - You CANNOT answer:
- General coding/programming questions (unless about a KU course)
- Math homework, physics problems 
- Questions unrelated to KU
- General knowledge questions

If someone asks an off-topic question, politely redirect:
"I'm BabyJay, KU's campus assistant! I can help with courses, professors, campus services, and anything KU-related. For general programming help, try resources like Stack Overflow or W3Schools. Is there anything about KU I can help you with?"

PERSONALITY:
- Be warm and friendly, like a helpful upperclassman
- Use casual but professional language
- Keep responses concise unless the user asks for details
- Occasionally add KU spirit (Rock Chalk!) but NOT every response - maybe 1 in 5 times

RULES:
1. If context is provided, USE IT. Answer from the context.
2. If NO context is provided, say you don't have that information and suggest who to contact.
3. Be conversational - brief for simple questions, detailed for complex ones.
4. Never say "I don't have information" if context was actually provided.
5. NEVER make up professor names or information. ONLY use professors explicitly mentioned in the context provided.
6. If you mention a professor's name, that exact name MUST appear in the context.
7. For technical topics (ML, AI, programming, algorithms), prefer EECS/CS courses over other departments unless user specifies otherwise.
8. If multiple courses match from different schools, list the most relevant one first (e.g., EECS for Machine Learning, not HDSC).


RESPONSE STYLE:
- NEVER use numbered lists or bullet points
- NEVER use bold (**text**) or headers
- Write in natural flowing paragraphs like a friend texting
- For multiple items, use prose: "You could check out X, Y, or Z" not "1. X 2. Y 3. Z"
- Keep it brief - 2-4 sentences for simple questions, short paragraphs and points for complex ones
- Sound like a helpful person chatting, not a database outputting results
- VARY your opening phrases - don't always start the same way

COURSE SELECTION PRIORITY:
- For Machine Learning, AI, programming, algorithms, data structures → prefer EECS/CS courses
- For statistics/data analysis → prefer STAT/MATH courses  
- For health/medical data → prefer HDSC/medicine courses
- When multiple courses match, prioritize by relevance to the likely student (CS student asking about ML = EECS course)
- If unsure which department user wants, mention the EECS option first for technical topics

IMPORTANT: When the user filters by department (e.g., "EECS only", "Just Business"), show ONLY professors from that exact department."""


# ==================== SMART CACHE SYSTEM ====================

class SmartCourseCache:
    """Two-layer cache for course lookups."""

    def __init__(self, data_ttl: int = 60, response_ttl: int = 300):
        self.data_cache: Dict[str, Dict] = {}
        self.response_cache: Dict[str, Dict] = {}
        self.data_ttl = data_ttl
        self.response_ttl = response_ttl

    def _normalize_course(self, course: str) -> str:
        return re.sub(r"[^a-z0-9]", "", course.lower())

    def _make_data_key(self, course: str, career: str) -> str:
        return f"{self._normalize_course(course)}:{career.lower()}"

    def _make_response_key(self, course: str, career: str, query_type: str = "general") -> str:
        return f"{self._normalize_course(course)}:{career.lower()}:{query_type}"

    def _extract_query_type(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["who teach", "instructor", "professor", "taught by"]):
            return "instructor"
        if any(w in q for w in ["seat", "enroll", "full", "open", "available", "space"]):
            return "seats"
        if any(w in q for w in ["when", "time", "schedule", "meet"]):
            return "schedule"
        if any(w in q for w in ["where", "location", "room", "building"]):
            return "location"
        return "general"

    def get_data(self, course: str, career: str) -> Optional[Dict]:
        key = self._make_data_key(course, career)
        if key in self.data_cache:
            entry = self.data_cache[key]
            age = time_module.time() - entry["timestamp"]
            if age < self.data_ttl:
                return entry["data"]
            del self.data_cache[key]
        return None

    def set_data(self, course: str, career: str, data: Dict):
        key = self._make_data_key(course, career)
        self.data_cache[key] = {"data": data, "timestamp": time_module.time()}

    def get_response(self, course: str, career: str, query: str) -> Optional[str]:
        query_type = self._extract_query_type(query)
        key = self._make_response_key(course, career, query_type)
        if key in self.response_cache:
            entry = self.response_cache[key]
            age = time_module.time() - entry["timestamp"]
            if age < self.response_ttl:
                return entry["response"]
            del self.response_cache[key]
        return None

    def set_response(self, course: str, career: str, query: str, response: str):
        query_type = self._extract_query_type(query)
        key = self._make_response_key(course, career, query_type)
        self.response_cache[key] = {
            "response": response,
            "query_type": query_type,
            "timestamp": time_module.time()
        }

    def clear(self):
        self.data_cache = {}
        self.response_cache = {}

    def stats(self) -> Dict:
        return {
            "data_entries": len(self.data_cache),
            "response_entries": len(self.response_cache),
            "data_ttl": self.data_ttl,
            "response_ttl": self.response_ttl
        }


_smart_cache = SmartCourseCache(data_ttl=60, response_ttl=300)


class ConversationStore:
    """Handles persistent storage of conversations using Redis."""

    def __init__(self, redis_host="localhost", redis_port=6379, use_redis=True):
        self.use_redis = use_redis and REDIS_AVAILABLE
        self.redis_client = None
        self.memory_store = {}

        if self.use_redis:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    decode_responses=True,
                    socket_connect_timeout=2
                )
                self.redis_client.ping()
            except (redis.ConnectionError, redis.TimeoutError):
                self.use_redis = False

    def save_message(self, session_id: str, role: str, content: str, ttl_days: int = 30):
        message = {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
        if self.use_redis:
            key = f"chat:history:{session_id}"
            self.redis_client.rpush(key, json.dumps(message))
            self.redis_client.expire(key, 86400 * ttl_days)
        else:
            if session_id not in self.memory_store:
                self.memory_store[session_id] = []
            self.memory_store[session_id].append(message)

    def load_history(self, session_id: str, max_messages: int = 100) -> List[Dict]:
        if self.use_redis:
            key = f"chat:history:{session_id}"
            messages = self.redis_client.lrange(key, -max_messages, -1)
            return [json.loads(msg) for msg in messages]
        return self.memory_store.get(session_id, [])[-max_messages:]

    def clear_history(self, session_id: str):
        if self.use_redis:
            self.redis_client.delete(f"chat:history:{session_id}")
        elif session_id in self.memory_store:
            del self.memory_store[session_id]


class BabyJayChat:
    def __init__(self, session_id: Optional[str] = None, use_redis: bool = True, debug: bool = False):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.router = QueryRouter()
        self.session_id = session_id or str(uuid.uuid4())
        self.store = ConversationStore(use_redis=use_redis)
        self.recent_context = ""
        self.last_search_query = ""
        self.debug = debug

        self.waiting_for_clarification = False
        self.clarification_context = None
        self.original_ambiguous_query = None
        self.active_department_filter = None
        self.last_mentioned_course: Optional[str] = None
        self.rlhf_optimizer = RLHFOptimizer(debug=debug)

        # Initialize intent detector
        self.intent_detector: Optional[LiveCourseIntentDetector] = None
        if INTENT_DETECTOR_AVAILABLE:
            try:
                self.intent_detector = LiveCourseIntentDetector(debug=debug)
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Intent detector init failed: {e}")

        self._conversation_history: List[Dict] = []
        self._load_from_store()

    def _load_from_store(self):
        stored_history = self.store.load_history(self.session_id)
        self._conversation_history = [{"role": msg["role"], "content": msg["content"]} for msg in stored_history]

    def _save_message(self, role: str, content: str):
        self._conversation_history.append({"role": role, "content": content})
        self.store.save_message(self.session_id, role, content)

    @property
    def conversation_history(self) -> List[Dict]:
        return self._conversation_history

    def clear_history(self):
        self._conversation_history = []
        self.store.clear_history(self.session_id)
        self.recent_context = ""
        self.last_search_query = ""
        self.active_department_filter = None
        self.last_mentioned_course = None

    # ==================== TOPIC TO COURSE RESOLVER ====================

    def _resolve_topic_to_courses(self, topic: str) -> List[str]:
        """
        Use fallback mapping + RAG to find course codes for a topic.
        
        Priority:
        1. EXACT match in fallback mapping (most reliable)
        2. PARTIAL match in fallback mapping
        3. RAG search (least reliable for specific topics)
        
        Example:
            "Deep Reinforcement Learning" → ["EECS 700"]
            "Machine Learning" → ["EECS 658", "EECS 836"]
        """
        topic_lower = topic.lower().strip()
        found_courses = []
        
        if self.debug:
            print(f"[DEBUG] Resolving topic '{topic}' to course codes...")
        
        # ==================== METHOD 1: HARDCODED FALLBACK (PRIORITY) ====================
        # Check this FIRST because it's most reliable for known topics
        
        # Try exact match first
        if topic_lower in TOPIC_TO_COURSE_FALLBACK:
            found_courses = TOPIC_TO_COURSE_FALLBACK[topic_lower].copy()
            if self.debug:
                print(f"[DEBUG] Fallback EXACT match: {found_courses}")
            return found_courses[:3]
        
        # Try partial match (topic contains key OR key contains topic)
        for key, courses in TOPIC_TO_COURSE_FALLBACK.items():
            if key in topic_lower or topic_lower in key:
                for course in courses:
                    if course not in found_courses:
                        found_courses.append(course)
        
        if found_courses:
            if self.debug:
                print(f"[DEBUG] Fallback PARTIAL match: {found_courses}")
            return found_courses[:3]
        
        # ==================== METHOD 2: RAG SEARCH (FALLBACK) ====================
        # Only use RAG if we didn't find anything in the fallback mapping
        
        try:
            search_queries = [
                f"{topic} course",
                f"{topic} class",
                topic,
            ]
            
            all_context = ""
            for query in search_queries:
                results = self.router.route(query)
                context = results.get("context", "")
                if context:
                    all_context += "\n" + context
                    if self.debug:
                        print(f"[DEBUG] RAG search '{query}': {len(context)} chars")
            
            if all_context:
                # Extract course codes like "EECS 700", "MATH 125"
                course_codes = re.findall(r'\b([A-Z]{2,4})\s*(\d{3,4})\b', all_context)
                
                # Filter: Only include courses from relevant departments for the topic
                relevant_depts = self._get_relevant_departments(topic_lower)
                
                for code in course_codes:
                    dept = code[0].upper()
                    course = f"{dept} {code[1]}"
                    
                    # If we have relevant departments, filter by them
                    if relevant_depts:
                        if dept in relevant_depts and course not in found_courses:
                            found_courses.append(course)
                    else:
                        # No filtering, accept all
                        if course not in found_courses:
                            found_courses.append(course)
                
                if self.debug and found_courses:
                    print(f"[DEBUG] RAG found courses: {found_courses[:5]}")
        
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] RAG search failed: {e}")
        
        return found_courses[:3]
    
    def _get_relevant_departments(self, topic: str) -> List[str]:
        """
        Return list of department codes relevant to a topic.
        Used to filter RAG results.
        """
        topic = topic.lower()
        
        # CS/Engineering topics
        cs_keywords = [
            "machine learning", "deep learning", "artificial intelligence", "ai",
            "neural network", "computer vision", "nlp", "natural language",
            "data science", "robotics", "algorithm", "programming", "software",
            "database", "network", "security", "cyber", "reinforcement learning"
        ]
        if any(kw in topic for kw in cs_keywords):
            return ["EECS", "CS", "CE", "MATH", "STAT"]
        
        # Business topics
        business_keywords = ["supply chain", "analytics", "business", "finance", "marketing", "management"]
        if any(kw in topic for kw in business_keywords):
            return ["BSAN", "ACCT", "FIN", "MGMT", "MKTG", "SCM"]
        
        # Math/Stats topics
        math_keywords = ["calculus", "statistics", "probability", "linear algebra", "differential"]
        if any(kw in topic for kw in math_keywords):
            return ["MATH", "STAT"]
        
        # Physics topics
        physics_keywords = ["physics", "quantum", "mechanics", "thermodynamics"]
        if any(kw in topic for kw in physics_keywords):
            return ["PHSX", "ASTR"]
        
        # No specific filtering
        return []

    def _looks_like_person_name(self, text: str) -> bool:
        """
        Check if text looks like a person's name rather than a course topic.
        
        Person names:
          - "kulkarni", "zoroya", "wang", "smith"
          - "jeff zoroya", "john smith"
        
        Course topics:
          - "machine learning", "deep learning", "artificial intelligence"
          - "EECS 700", "MATH 125"
        """
        text = text.lower().strip()
        
        # If it's a course code, definitely not a person
        if re.match(r'^[a-z]{2,4}\s*\d{3,4}$', text, re.IGNORECASE):
            return False
        
        # Known course topics - definitely not a person
        known_topics = [
            "machine learning", "deep learning", "artificial intelligence", "ai",
            "reinforcement learning", "neural network", "computer vision",
            "natural language", "nlp", "data science", "robotics", "algorithms",
            "database", "operating system", "computer network", "cybersecurity",
            "software engineering", "web development", "mobile", "cloud",
            "calculus", "statistics", "physics", "chemistry", "biology",
            "psychology", "economics", "business", "marketing", "finance",
            "supply chain", "analytics", "accounting",
        ]
        if text in known_topics or any(topic in text for topic in known_topics):
            return False
        
        # Check if it's a single word (likely last name) or two words (first + last name)
        words = text.split()
        if len(words) == 1:
            # Single word - likely a last name if it's not a known topic
            # and doesn't contain numbers
            if not any(c.isdigit() for c in text):
                return True
        elif len(words) == 2:
            # Two words - could be "first last" name
            # But also could be "machine learning" - already filtered above
            if not any(c.isdigit() for c in text):
                return True
        
        return False

    # ==================== LIVE COURSE LOOKUP ====================

    def _needs_live_course_lookup(self, query: str) -> Optional[Dict]:
        """Detect if query needs live course data."""
        if not LIVE_LOOKUP_AVAILABLE:
            return None

        original_query = query
        working_query = query
        q = working_query.lower()

        # General info questions should NOT trigger live lookup
        general_info_patterns = [
            "what is", "what's", "tell me about", "what do you know",
            "describe", "explain", "information about", "details about",
            "prerequisites", "prereqs", "credits", "description", "about"
        ]
        live_keywords = ["seats", "enroll", "open", "available", "who teaches", 
                        "instructor", "schedule", "when does", "section"]

        # If asking general info WITHOUT live keywords, skip live lookup
        if any(p in q for p in general_info_patterns):
            if not any(lk in q for lk in live_keywords):
                if self.debug:
                    print("[DEBUG] General info query, skipping live lookup")
                return None

        # ==================== EARLY EXIT: Person/Faculty Queries ====================
        # "who is professor X?" should go to faculty retriever, not live lookup
        # "who teaches EECS 700?" should go to live lookup
        
        person_query_patterns = [
            r"who\s+is\s+(?:prof(?:essor)?\.?\s+)?([a-z]+(?:\s+[a-z]+)?)\??$",  # who is prof kulkarni
            r"tell\s+me\s+about\s+(?:prof(?:essor)?\.?\s+)?([a-z]+)",  # tell me about prof X
            r"(?:prof(?:essor)?\.?\s+)([a-z]+(?:\s+[a-z]+)?)\s*\??$",  # prof kulkarni?
        ]
        
        for pattern in person_query_patterns:
            match = re.search(pattern, q, re.IGNORECASE)
            if match:
                potential_name = match.group(1).strip()
                # Check if it looks like a person name (not a course topic)
                if self._looks_like_person_name(potential_name):
                    if self.debug:
                        print(f"[DEBUG] Detected faculty query for '{potential_name}', skipping live lookup")
                    return None  # Let it fall through to faculty retriever

        # Handle follow-up references
        phrase_refs = [
            "this course", "that course", "this class", "that class",
            "the course", "the class", "that one", "this one",
        ]
        short_pronoun_match = re.fullmatch(r"\s*(it|that|this|that one|this one)\s*\??\s*", q)

        if self.last_mentioned_course and (short_pronoun_match or any(ref in q for ref in phrase_refs)):
            working_query = f"{self.last_mentioned_course} {query}"
            q = working_query.lower()

        # ==================== EMBEDDING-BASED DETECTION ====================
        
        if self.intent_detector:
            try:
                intent_result = self.intent_detector.detect(working_query)
                if self.debug:
                    print(f"[DEBUG] Embedding detection: {intent_result}")
                
                if intent_result.get("needs_live") and intent_result.get("topic"):
                    topic = intent_result["topic"]
                    
                    # Check if topic contains a course code
                    course_match = re.search(r'\b([A-Z]{2,4})\s*(\d{3,4})\b', topic, re.IGNORECASE)
                    if course_match:
                        num = int(course_match.group(2))
                        career = "Graduate" if num >= 500 else "Undergraduate"
                        topic = f"{course_match.group(1).upper()} {course_match.group(2)}"
                        is_topic_search = False
                    else:
                        advanced_topics = [
                            "machine learning", "deep learning", "ai", "artificial intelligence",
                            "neural network", "data science", "nlp", "computer vision", "robotics",
                            "reinforcement learning", "deep reinforcement learning"
                        ]
                        career = "Graduate" if any(t in topic.lower() for t in advanced_topics) else "Undergraduate"
                        is_topic_search = True
                    
                    if any(w in q for w in ["graduate", "grad level", "masters", "phd"]):
                        career = "Graduate"
                    elif any(w in q for w in ["undergraduate", "undergrad", "bachelors"]):
                        career = "Undergraduate"
                    
                    return {
                        "course": topic,
                        "career": career,
                        "original_query": original_query,
                        "is_topic_search": is_topic_search,
                        "detection_method": intent_result.get("method", "embedding"),
                        "confidence": intent_result.get("confidence", 0.0)
                    }
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Embedding detection error: {e}")

        # ==================== REGEX FALLBACK ====================
        
        if self.debug:
            print("[DEBUG] Using regex fallback for intent detection")

        forced_live = False
        course_only_match = re.match(r"^([A-Z]{2,4}\s*\d{3,4})\s*\??$", working_query.strip(), re.IGNORECASE)
        if course_only_match:
            forced_live = True

        if re.search(r"(spring|fall|summer)\s+\d{4}", q):
            forced_live = True

        live_indicators = [
            "who teaches", "who is teaching", "instructor for", "taught by",
            "seats", "open seats", "seats available", "availability", "enroll",
            "is there space", "can i enroll", "is it full", "is it open",
            "class schedule", "when is", "what time", "schedule for",
            "this semester", "current semester", "next semester",
            "next spring", "next fall", "section", "sections",
        ]

        needs_live = forced_live or any(indicator in q for indicator in live_indicators)

        course_match = re.search(r"\b([A-Z]{2,4})\s*(\d{3,4})\b", working_query, re.IGNORECASE)
        if course_match:
            question_words = ["who", "when", "where", "how many", "is there",
                           "seats", "teach", "instructor", "schedule", "section", "available"]
            if any(w in q for w in question_words):
                needs_live = True

        topic_patterns = [
            r"who\s+teaches?\s+(.+?)(?:\s+next|\s+this|\s+class|\s+course|\?|$)",
            r"who\s+is\s+teaching\s+(.+?)(?:\s+next|\s+this|\?|$)",
            r"seats?\s+(?:in|for)\s+(.+?)(?:\s+next|\s+this|\?|$)",
            r"(?:available|open)\s+(?:seats?\s+)?(?:in|for)\s+(.+?)(?:\?|$)",
            r"space\s+in\s+(.+?)(?:\?|$)",
            r"enroll\s+(?:in|for)\s+(.+?)(?:\?|$)",
        ]

        topic_search = None
        for pattern in topic_patterns:
            match = re.search(pattern, q, re.IGNORECASE)
            if match:
                topic_search = match.group(1).strip()
                topic_search = re.sub(r"\b(course|class|courses|classes|the|a|an)\b", "", topic_search).strip()
                if len(topic_search) > 2:
                    needs_live = True
                    break

        if not needs_live:
            return None

        if course_match:
            subject = course_match.group(1).upper()
            number = course_match.group(2)
            course_code = f"{subject} {number}"
            num = int(number)
            career = "Graduate" if num >= 500 else "Undergraduate"
            is_topic_search = False
        elif topic_search:
            course_code = topic_search.title()
            advanced_topics = ["machine learning", "deep learning", "ai", "artificial intelligence",
                             "neural network", "data science", "nlp", "computer vision", "robotics",
                             "reinforcement learning", "deep reinforcement learning"]
            career = "Graduate" if any(t in topic_search.lower() for t in advanced_topics) else "Undergraduate"
            is_topic_search = True
        else:
            return None

        if any(w in q for w in ["graduate", "grad level", "masters", "phd"]):
            career = "Graduate"
        elif any(w in q for w in ["undergraduate", "undergrad", "bachelors"]):
            career = "Undergraduate"

        return {
            "course": course_code,
            "career": career,
            "original_query": original_query,
            "is_topic_search": is_topic_search,
            "detection_method": "regex",
            "confidence": 0.8
        }

    def _handle_live_course_query(self, query_info: Dict) -> str:
        """Fetch live course data with topic-to-course resolution."""
        course = query_info["course"]
        career = query_info["career"]
        original_query = query_info.get("original_query", "")
        is_topic_search = query_info.get("is_topic_search", False)

        # Save for follow-up references
        if not is_topic_search:
            self.last_mentioned_course = course

        if self.debug:
            method = query_info.get("detection_method", "unknown")
            confidence = query_info.get("confidence", 0)
            print(f"[DEBUG] Live lookup: {course} ({career}) [method={method}, conf={confidence:.2f}]")

        # 1) Cached response?
        cached_response = _smart_cache.get_response(course, career, original_query)
        if cached_response:
            if self.debug:
                print(f"[DEBUG] Response cache HIT for {course}")
            return cached_response

        # 2) Cached raw data?
        result = _smart_cache.get_data(course, career)
        if result:
            if self.debug:
                print(f"[DEBUG] Data cache HIT for {course}")
        else:
            if self.debug:
                print("[DEBUG] Fetching fresh data from KU...")
            result = lookup_course(course, career=career)

            if not result.get("success"):
                return (
                    "Sorry, I couldn't fetch live course data right now. "
                    f"Error: {result.get('error', 'Unknown error')}. "
                    "You can check directly at classes.ku.edu"
                )

            # If no results, try other career level
            if result.get("total_sections", 0) == 0:
                other_career = "Undergraduate" if career == "Graduate" else "Graduate"
                if self.debug:
                    print(f"[DEBUG] No results, trying {other_career}")
                result = lookup_course(course, career=other_career)

                # ==================== TOPIC-TO-COURSE RESOLUTION ====================
                # If still no results AND this was a topic search, use RAG to find course codes
                if result.get("total_sections", 0) == 0 and is_topic_search:
                    if self.debug:
                        print(f"[DEBUG] Topic search failed, resolving '{course}' to course codes...")
                    
                    resolved_courses = self._resolve_topic_to_courses(course)
                    
                    if resolved_courses:
                        if self.debug:
                            print(f"[DEBUG] Resolved to: {resolved_courses}")
                        
                        # Try each resolved course code
                        for resolved_course in resolved_courses:
                            # Determine career from course number
                            num_match = re.search(r'\d+', resolved_course)
                            if num_match:
                                num = int(num_match.group())
                                resolved_career = "Graduate" if num >= 500 else "Undergraduate"
                            else:
                                resolved_career = career
                            
                            if self.debug:
                                print(f"[DEBUG] Trying resolved course: {resolved_course} ({resolved_career})")
                            
                            result = lookup_course(resolved_course, career=resolved_career)
                            
                            if result.get("total_sections", 0) > 0:
                                # Found sections! Update course info
                                course = resolved_course
                                career = resolved_career
                                self.last_mentioned_course = course
                                
                                if self.debug:
                                    print(f"[DEBUG] Found {result['total_sections']} sections for {course}")
                                break
                            
                            # Try other career level
                            other_career = "Undergraduate" if resolved_career == "Graduate" else "Graduate"
                            result = lookup_course(resolved_course, career=other_career)
                            
                            if result.get("total_sections", 0) > 0:
                                course = resolved_course
                                career = other_career
                                self.last_mentioned_course = course
                                
                                if self.debug:
                                    print(f"[DEBUG] Found {result['total_sections']} sections for {course} ({other_career})")
                                break
                
                # If still no results after resolution, return not found
                if result.get("total_sections", 0) == 0:
                    topic_hint = ""
                    if is_topic_search:
                        topic_hint = f" I also searched for related course codes but couldn't find any sections."
                    
                    return (
                        f"No sections found for **{query_info['course']}** in {result.get('semester', 'this semester')}.{topic_hint} "
                        "The course might not be offered right now, or check classes.ku.edu directly."
                    )

                career = result.get("career", career)

            _smart_cache.set_data(course, career, result)

        raw_data = format_sections_for_chat(result)

        # LLM format response
        try:
            # Include original topic in prompt if we resolved to a different course
            topic_context = ""
            if is_topic_search and course != query_info["course"]:
                topic_context = f"\nNote: The user asked about '{query_info['course']}' which is offered as {course}."
            
            llm_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are BabyJay, KU's friendly campus assistant.

The user asked about courses and I fetched LIVE data from classes.ku.edu.
Write a helpful, CONVERSATIONAL response.{topic_context}

STYLE RULES:
1. VARY your opening - don't use fixed phrases like "Hey there!"
2. No bullet points / no numbered lists
3. Flowing paragraphs, like chatting
4. Focus on what matters to their question (instructor, seats, time)
5. Keep it concise
6. End with one helpful tip
7. No emojis
8. If the user asked about a topic (like "Deep Reinforcement Learning") and we found it 
   under a course code (like EECS 700), mention BOTH so they know what to search for.
"""
                    },
                    {
                        "role": "user",
                        "content": f"""User asked: "{original_query}"

Live data from KU:
{raw_data}

Write a friendly, conversational response (no lists)."""
                    }
                ],
                temperature=0.8,
                max_tokens=600
            )
            response = llm_response.choices[0].message.content.strip()

            _smart_cache.set_response(course, career, original_query, response)
            return response

        except Exception as e:
            if self.debug:
                print(f"[DEBUG] LLM formatting failed: {e}")
            return raw_data

    # ==================== GREETING HANDLING ====================

    def _is_greeting(self, query: str) -> bool:
        q = query.lower().strip().rstrip("?!.")
        if q in GREETING_PATTERNS:
            return True
        for pattern in GREETING_REGEX_PATTERNS:
            if re.match(pattern, q, re.IGNORECASE):
                return True
        for greeting in GREETING_PATTERNS:
            if q.startswith(greeting) and len(q) < len(greeting) + 20:
                return True
        return False

    def _generate_greeting_response(self, query: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are BabyJay, KU's friendly campus assistant.

Respond to the user's greeting naturally and uniquely. Be warm, casual, and helpful.

RULES:
- Keep it SHORT (1-2 sentences max)
- If they ask "how are you" - respond naturally
- Don't always introduce yourself
- Vary your opening completely
- Only occasionally add "Rock Chalk!" (maybe 1 in 5 times)
- Ask what they need help with
- Match their energy
- NO emojis
"""
                    },
                    {"role": "user", "content": query}
                ],
                temperature=0.9,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Greeting LLM error: {e}")
            return "Hey! What can I help you with today?"

    def _is_about_bot(self, query: str) -> bool:
        q = query.lower()
        return any(phrase in q for phrase in [
            "who are you", "what are you", "tell me about yourself",
            "what can you do", "what do you do", "your name",
            "who invented you", "who created you", "who made you",
            "who built you", "who developed you", "who designed you",
            "are you ai", "are you a bot", "are you human",
            "are you real", "are you chatgpt", "are you gpt",
            "what is babyjay", "babyjay about", "about babyjay", 
        ])

    def _generate_about_response(self, query: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are BabyJay, KU's campus assistant.

Introduce yourself naturally and briefly (3-5 sentences). Mention you can help with:
courses (including LIVE seat availability), professors, campus life, transit, and tuition/aid.

No bullet points. No emojis."""
                    },
                    {"role": "user", "content": query}
                ],
                temperature=0.8,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] About bot LLM error: {e}")
            return (
                "I'm BabyJay, KU's campus assistant! I can help you with courses, professors, dining, housing, transit, and more. "
                "I can even check live seat availability and who's teaching a course this semester."
            )

    # ==================== QUERY VALIDATION & CLEANING ====================

    def _validate_query(self, query: str) -> bool:
        if not query or not query.strip():
            return False
        return any(c.isalnum() for c in query)

    def _needs_cleaning(self, query: str) -> bool:
        q = query.lower()

        messy_indicators = [
            "machien" in q, "machie" in q, "artifical" in q, "robtics" in q,
            "quantim" in q, "quantom" in q, "profesors" in q,
            "proffessor" in q, "reseach" in q, "compter" in q,
            " 2 " in q, " 4 " in q, " u " in q, " r " in q,
            " abt " in q, " wnt " in q, " noe " in q, " wat " in q,
            (" prof " in q and len(q) < 30),
            "  " in q, "???" in q, "!!!" in q,
            "avilable" in q, "availble" in q,
        ]

        if any(messy_indicators):
            return True

        if len(q.split()) <= 2 and q.replace(" ", "").isalpha():
            return False

        return False

    def _clean_query_hybrid(self, query: str) -> Tuple[str, bool, str]:
        if not self._needs_cleaning(query):
            return query, False, "none"

        # STEP 1: local
        try:
            from app.rag.query_preprocessor import QueryPreprocessor
            preprocessor = QueryPreprocessor()
            result = preprocessor.preprocess(query)

            local_cleaned = result.get("processed", query)
            corrections = result.get("corrections", [])

            if corrections and local_cleaned != query:
                if self.debug:
                    print(f"[DEBUG] Local fix: '{query}' → '{local_cleaned}'")
                return local_cleaned, True, "local"
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Local preprocessing failed: {e}")

        # STEP 2: LLM
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Fix typos, grammar, and text speak. Return ONLY the corrected query, nothing else."},
                    {"role": "user", "content": query}
                ],
                temperature=0,
                max_tokens=100
            )
            llm_cleaned = response.choices[0].message.content.strip()

            if llm_cleaned and len(llm_cleaned) < len(query) * 3:
                if self.debug:
                    print(f"[DEBUG] LLM fix: '{query}' → '{llm_cleaned}'")
                return llm_cleaned, True, "llm"
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] LLM cleaning failed: {e}")

        return query, False, "none"

    # ==================== FOLLOW-UP & CLARIFICATION ====================

    def _is_simple_followup(self, question: str) -> bool:
        q = question.lower().strip()
        if len(question.split()) > 8:
            return False
        if q in ["it", "it?"] and self._conversation_history:
            return True
        simple_indicators = [
            "his", "her", "their", "that", "this", "what about", "how about",
            "its", "it's", "sorry", "my bad", "actually", "i meant", "i mean",
            "no wait", "not that", "instead", "oops", "wait no", "correction",
            "not the", "no not", "i meant", "that course", "this course",
            "the course", "about it", "about that"
        ]
        return any(ind in q for ind in simple_indicators)

    def _is_department_filter(self, question: str) -> bool:
        q = question.lower()
        return any(p in q for p in ["only", "just"]) and any(k in q for k in [
            "eecs", "electrical", "computer science", "business", "physics",
            "chemistry", "math", "engineering", "department"
        ])

    def _is_ambiguous(self, query: str) -> bool:
        q = query.lower().strip()
        words = q.split()

        if self._is_simple_followup(query):
            return False
        if self._is_department_filter(query):
            return False
        if len(words) >= 3:
            return False

        vague_terms = [
            "professors", "faculty", "research", "researchers",
            "courses", "classes", "teach", "teaching",
            "help", "info", "information"
        ]
        if q in vague_terms:
            return True

        if len(words) == 2 and words[0] in vague_terms:
            return words[1] in ["here", "there", "available", "at", "ku"]

        return False

    def _generate_clarification_question(self, query: str) -> str:
        q = query.lower().strip()
        if q in ["professors", "faculty", "researchers"]:
            return "What research area or department are you interested in? (e.g., machine learning, robotics, Business school, EECS)"
        if q in ["research", "studies"]:
            return "What research topic or field would you like to know about?"
        if q in ["courses", "classes"]:
            return "Which subject or department? (e.g., computer science, business, physics)"
        if q in ["help", "info", "information"]:
            return "What would you like to know about? I can help with faculty, courses, dining, housing, transit, and more."
        return f"Could you provide more details about '{query}'? For example, which department or research area?"

    def _is_clarification_answer(self, query: str) -> bool:
        return self.waiting_for_clarification and len(query.strip()) > 0

    def _process_clarification_answer(self, answer: str) -> str:
        original = (self.original_ambiguous_query or "").lower().strip()
        answer = answer.strip()

        if original in ["professors", "faculty", "researchers"]:
            dept_keywords = ["eecs", "business", "physics", "chemistry", "math", "engineering", "psychology"]
            if any(kw in answer.lower() for kw in dept_keywords):
                combined_query = f"{answer} {original}"
            else:
                combined_query = f"{original} {answer}"
        elif original in ["research", "courses", "classes"]:
            combined_query = f"{answer} {original}"
        else:
            combined_query = f"{original} {answer}"

        if self.debug:
            print(f"[DEBUG] Combined: '{self.original_ambiguous_query}' + '{answer}' = '{combined_query}'")

        self.waiting_for_clarification = False
        self.original_ambiguous_query = None
        self.clarification_context = None

        return self.ask(combined_query, use_history=True)

    def _extract_department(self, question: str) -> Optional[str]:
        q = question.lower()
        if any(kw in q for kw in ["eecs", "computer science", "electrical engineering", " cs "]):
            return "Electrical Engineering and Computer Science"
        if "business" in q:
            return "School of Business"
        if "physics" in q:
            return "Department of Physics and Astronomy"
        if "chemistry" in q or " chem " in q:
            return "Department of Chemistry"
        if "math" in q:
            return "Department of Mathematics"
        if "psychology" in q or " psych " in q:
            return "Department of Psychology"
        if "mechanical" in q:
            return "Mechanical Engineering"
        return None

    def _filter_context_by_department(self, context: str, department: str) -> str:
        if not context or "=== FACULTY INFORMATION ===" not in context:
            return context

        lines = context.split("\n")
        filtered_lines = []
        include_current = False
        buffer = []

        for line in lines:
            if line.strip() == "=== FACULTY INFORMATION ===":
                filtered_lines.append(line)
                include_current = False
                continue

            if line.strip().startswith("===") and "FACULTY" not in line:
                if include_current:
                    filtered_lines.extend(buffer)
                buffer = []
                include_current = False
                filtered_lines.append(line)
                continue

            if line.strip().startswith("Professor:"):
                if include_current:
                    filtered_lines.extend(buffer)
                buffer = [line]
                include_current = False
                continue

            if line.strip().startswith("Department:"):
                current_dept = line.strip().replace("Department:", "").strip()
                buffer.append(line)
                include_current = department.lower() in current_dept.lower()
                continue

            buffer.append(line)

        if include_current:
            filtered_lines.extend(buffer)

        return "\n".join(filtered_lines)

    def _expand_followup_question(self, question: str) -> str:
        if not self._validate_query(question):
            return question
        if not self._conversation_history:
            return question
        if len(question.strip()) < 3:
            return question

        recent_history = self._conversation_history[-6:]
        history_text = "\n".join(
            [f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content'][:200]}" for msg in recent_history]
        )
        if not history_text or len(history_text) < 10:
            return question

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Rewrite the follow-up question to include full context. Return only the rewritten question."},
                    {"role": "user", "content": f"Conversation:\n{history_text}\n\nFollow-up: {question}\n\nRewritten:"}
                ],
                temperature=0,
                max_tokens=100
            )
            expanded = response.choices[0].message.content.strip()
            if expanded and len(expanded) > len(question):
                return expanded
            return question
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Expansion failed: {e}")
            return question

    # ==================== MAIN ASK METHOD ====================

    def ask(self, question: str, use_history: bool = True) -> str:
        start_time = time_module.time()

        if not question or not question.strip():
            return "I'd be happy to help! What would you like to know about KU?"

        if not self._validate_query(question):
            return "I didn't quite understand that. Could you rephrase your question?"

        question = question.strip()

        # Greeting
        if self._is_greeting(question):
            if self.debug:
                print("[DEBUG] Greeting detected, generating unique response")
            response = self._generate_greeting_response(question)
            if self.debug:
                print(f"[DEBUG] Greeting response time: {(time_module.time() - start_time) * 1000:.0f}ms")
            return response

        # About bot
        if self._is_about_bot(question):
            if self.debug:
                print("[DEBUG] About bot question detected")
            return self._generate_about_response(question)

        # Clarification answer
        if self._is_clarification_answer(question):
            return self._process_clarification_answer(question)

        # Ambiguous -> ask
        if self._is_ambiguous(question):
            clarification_q = self._generate_clarification_question(question)
            self.waiting_for_clarification = True
            self.original_ambiguous_query = question
            self.clarification_context = {"original_query": question, "timestamp": datetime.now().isoformat()}
            if self.debug:
                print(f"[DEBUG] Ambiguous query: '{question}' → asking clarification")
            return clarification_q

        # Cleaning
        cleaned, was_cleaned, method = self._clean_query_hybrid(question)
        search_question = cleaned

        try:
            from app.rag.query_preprocessor import QueryPreprocessor
            preprocessor = QueryPreprocessor()
            prep_result = preprocessor.preprocess(search_question)
            if prep_result.get("processed"):
                search_question = prep_result["processed"]
                if self.debug and prep_result.get("corrections"):
                    print(f"[DEBUG] Expanded: {prep_result['corrections']}")
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Preprocessor error: {e}")

        if self.debug and was_cleaned:
            print(f"[DEBUG] Cleaned ({method}): '{question}' → '{cleaned}'")

        # Live lookup
        live_query_info = self._needs_live_course_lookup(search_question)
        if live_query_info:
            if self.debug:
                print(f"[DEBUG] Live course query: {live_query_info}")
            response = self._handle_live_course_query(live_query_info)
            if use_history:
                self._save_message("user", question)
                self._save_message("assistant", response)
            if self.debug:
                print(f"[DEBUG] Total time: {(time_module.time() - start_time) * 1000:.0f}ms")
            return response

        # ==================== SEARCH (RAG) ====================
        context = ""

        # Department filter request
        if self._is_department_filter(search_question) and self.last_search_query:
            department = self._extract_department(search_question)
            if department:
                if self.debug:
                    print(f"[DEBUG] Department filter: {department}")
                search_query = f"{self.last_search_query} {department}"
                try:
                    results = self.router.route(search_query)
                    context = results.get("context", "")
                    context = self._filter_context_by_department(context, department)
                except Exception as e:
                    if self.debug:
                        print(f"[DEBUG] Search failed: {e}")
        else:
            search_query = search_question

            # Follow-up expansion
            if self._is_simple_followup(search_question) and use_history:
                try:
                    expanded = self._expand_followup_question(search_question)
                    if expanded != search_question:
                        if self.debug:
                            print(f"[DEBUG] Expanded: '{search_question}' → '{expanded}'")
                        search_query = expanded
                        
                        # RE-CHECK for live lookup after expansion
                        live_query_info = self._needs_live_course_lookup(expanded)
                        if live_query_info:
                            if self.debug:
                                print(f"[DEBUG] Expanded query triggers live lookup: {live_query_info}")
                            response = self._handle_live_course_query(live_query_info)
                            if use_history:
                                self._save_message("user", question)
                                self._save_message("assistant", response)
                            return response
                except Exception as e:
                    if self.debug:
                        print(f"[DEBUG] Expansion error: {e}")

            # Track faculty queries
            if any(kw in search_question.lower() for kw in ["professor", "faculty", "research", "ml", "ai", "machine learning"]):
                self.last_search_query = search_query

            # Route
            try:
                if self.debug:
                    print(f"[DEBUG] Routing query: '{search_query}'")
                results = self.router.route(search_query)
                context = results.get("context", "")
                source = results.get("source", "unknown")
                result_count = results.get("result_count", 0)
                if self.debug:
                    print(f"[DEBUG] Router: {source}, {result_count} results, {len(context)} chars")
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Router error: {e}")

            # Retry with KU prefix
            if not context:
                retry_query = f"KU {search_query}"
                if self.debug:
                    print(f"[DEBUG] Retry: '{retry_query}'")
                try:
                    retry_results = self.router.route(retry_query)
                    context = retry_results.get("context", "")
                except Exception as e:
                    if self.debug:
                        print(f"[DEBUG] Retry error: {e}")

            # Follow-up fallback to recent context
            if self._is_simple_followup(search_question) and self.recent_context and not context:
                context = self.recent_context
                if self.debug:
                    print("[DEBUG] Using recent_context")

        if context:
            self.recent_context = context

        # Build LLM messages
        enhanced_prompt = self.rlhf_optimizer.enhance_prompt(SYSTEM_PROMPT, question)
        messages = [{"role": "system", "content": enhanced_prompt}]
        if use_history and self._conversation_history:
            messages.extend(self._conversation_history[-10:])

        if context:
            user_msg = f"Here's information from KU's database:\n\n{context}\n\nUser's question: {question}\n\nAnswer based on the information above."
        else:
            user_msg = f"User's question: {question}\n\nI don't have specific info. Suggest where they might find it."

        messages.append({"role": "user", "content": user_msg})

        try:
            if self.debug:
                print("[DEBUG] Calling OpenAI...")
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.5,
                max_tokens=1500
            )
            assistant_msg = response.choices[0].message.content
            if use_history:
                self._save_message("user", question)
                self._save_message("assistant", assistant_msg)
            if self.debug:
                print(f"[DEBUG] Total time: {(time_module.time() - start_time) * 1000:.0f}ms")
            return assistant_msg
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] LLM error: {e}")
            return "Sorry, I encountered an error. Please try again!"


def main():
    """Interactive CLI."""
    print("=" * 60)
    print("BabyJay - KU Campus Assistant (with Topic-to-Course Resolver)")
    print("=" * 60)

    chat = BabyJayChat(debug=True)
    print(f"\nSession: {chat.session_id[:8]}...")
    print("\nCommands: 'quit', 'clear', 'debug on/off', 'cache stats'")
    print("\nTry: 'seats for Deep Reinforcement Learning?' (will resolve to EECS 700)")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            cmd = user_input.lower()
            if cmd == "quit":
                print("Goodbye!")
                break
            if cmd == "clear":
                chat.clear_history()
                _smart_cache.clear()
                print("History and cache cleared")
                continue
            if cmd == "debug on":
                chat.debug = True
                if chat.intent_detector:
                    chat.intent_detector.debug = True
                print("Debug ON")
                continue
            if cmd == "debug off":
                chat.debug = False
                if chat.intent_detector:
                    chat.intent_detector.debug = False
                print("Debug OFF")
                continue
            if cmd == "cache stats":
                stats = _smart_cache.stats()
                print(f"Cache: {stats['data_entries']} data, {stats['response_entries']} responses")
                continue

            response = chat.ask(user_input)
            print(f"\nBabyJay: {response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
