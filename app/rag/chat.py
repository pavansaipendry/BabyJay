"""
BabyJay Chat - KU Campus Assistant
====================================
RAG-based chatbot for the University of Kansas.
Routes queries through classifier → router → retrievers → context builder → LLM.
"""

import os
import json
import uuid
import re
import time as time_module
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import anthropic
from app.rag.router import QueryRouter
from app.rag.rlhf_optimizer import RLHFOptimizer
from app.rag.context_builder import ContextBuilder
from dotenv import load_dotenv

load_dotenv()

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HAIKU_MODEL = "claude-haiku-4-5-20251001"


def _call_haiku(client: anthropic.Anthropic, system: str, messages: List[Dict],
                temperature: float = 0.5, max_tokens: int = 1500) -> str:
    """Unified helper to call Claude Haiku. Converts OpenAI-style messages to Anthropic format."""
    # Separate system prompt from conversation messages
    conversation = [m for m in messages if m["role"] != "system"]

    # Merge consecutive same-role messages (Anthropic requires alternating roles)
    merged = []
    for msg in conversation:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n\n" + msg["content"]
        else:
            merged.append({"role": msg["role"], "content": msg["content"]})

    # Ensure conversation starts with user message
    if not merged or merged[0]["role"] != "user":
        merged.insert(0, {"role": "user", "content": "Hello"})

    response = client.messages.create(
        model=HAIKU_MODEL,
        system=system,
        messages=merged,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.content[0].text


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
- Anything not specifically about the University of Kansas

If someone asks an off-topic question, politely redirect them to appropriate resources and ask if there's anything KU-related you can help with. Vary your wording each time.

WHAT "CONTEXT" MEANS:
- Context = information retrieved from KU's database (courses, faculty, campus services) provided in the user message
- If context is provided, it appears after "Here's information from KU's database:"
- Conversation history = previous messages in this chat session (use for follow-ups)
- Live data = real-time info from classes.ku.edu (seats, schedules)

PERSONALITY:
- Be warm and friendly, like a helpful upperclassman
- Use casual but professional language
- Keep responses concise unless the user asks for details
- Occasionally add KU spirit (Rock Chalk!) but NOT every response - maybe 1 in 5 times

RULES:
1. If context is provided, USE IT. Answer ONLY from the context. Do NOT add information beyond what the context contains.
2. If NO context is provided, say you don't have that specific information right now and suggest checking ku.edu or the relevant KU office. Do NOT guess or answer from general knowledge — even if you think you know the answer, your knowledge may be outdated or wrong.
3. Be conversational - brief for simple questions, detailed for complex ones.
4. Never say "I don't have information" if context was actually provided.
5. NEVER make up professor names, office locations, phone numbers, URLs, email addresses, or any specific KU details. ONLY use information explicitly present in the context provided.
6. If you mention a professor's name, that exact name MUST appear in the context.
7. For technical topics (ML, AI, programming, algorithms), prefer EECS/CS courses over other departments unless user specifies otherwise.
8. If multiple courses match from different schools, list the most relevant one first (e.g., EECS for Machine Learning, not HDSC).
9. If the context seems unrelated to the user's question, ignore the context and say you don't have relevant information for that specific question.
10. NEVER fabricate specific numbers (GPA requirements, acceptance rates, tuition amounts, scholarship values) unless they appear in the provided context.
11. Context includes [Source: ...] tags. Do NOT mention these tags in your response.
12. SOURCES FOOTER — After your response, add a blank line then a Sources section ONLY if the context contains real URLs starting with https://. Format exactly like this:

Sources:
- https://eecs.ku.edu/faculty
- https://catalog.ku.edu/engineering/electrical-engineering-computer-science/

    Rules (non-negotiable):
    - ONLY include URLs that start with https:// and appear VERBATIM in the provided context.
    - NEVER invent, guess, or construct a URL. If you did not see it in the context, do not include it.
    - One URL per line with a dash prefix. No bullets, no angle brackets.
    - Deduplicate. Cap at 5 URLs.
    - If no https:// URL appears in the context, skip the Sources section entirely — do not write "Sources:" at all.

RESPONSE STYLE:
- NEVER use ** or __ for bold. Never use markdown bold/italic.
- NEVER use # or ## for headings.
- For simple questions (1 item): Answer in 1-3 plain sentences.
- For multiple items (3+ professors, courses, options): Use a dash list, one item per line.
- Format guide:
  - "Where is the ISS office?" → 2-3 sentences with the key facts
  - "List ML professors" → dash list, one professor per line with their research area
  - "What are the dining options?" → short intro sentence, then dash list per location
  - "Who teaches X?" → prose if 1-2, dash list if 3+
- Each Sources URL must be on its own line, separated from the answer by a blank line.
- VARY your opening phrases - don't always start the same way.
- Sound helpful and friendly, not robotic.

COURSE SELECTION PRIORITY:
- For Machine Learning, AI, programming, algorithms, data structures → prefer EECS/CS courses
- For statistics/data analysis → prefer STAT/MATH courses
- For health/medical data → prefer HDSC/medicine courses
- When multiple courses match, prioritize by relevance to the likely student (CS student asking about ML = EECS course)
- If unsure which department user wants, mention the EECS option first for technical topics

DEPARTMENT FILTERING:
- When the user filters by department (e.g., "EECS only", "Just Business"), show ONLY professors/courses from that exact department.
- If the filter yields 0 results, say so clearly and suggest the closest alternatives (e.g., "No EECS courses matched, but you might check out these CS courses...")."""


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
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.router = QueryRouter()
        self.context_builder = ContextBuilder()
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

    # ==================== GREETING HANDLING ====================

    def _is_greeting(self, query: str) -> bool:
        q = query.lower().strip().rstrip("?!.")
        if q in GREETING_PATTERNS:
            return True
        for pattern in GREETING_REGEX_PATTERNS:
            if re.match(pattern, q, re.IGNORECASE):
                return True
        for greeting in GREETING_PATTERNS:
            # Must match as a whole word — "hi" should NOT match "his", "history", "hire"
            if q.startswith(greeting) and len(q) < len(greeting) + 20:
                # Check that the greeting is a complete word (followed by space, end, or punctuation)
                rest = q[len(greeting):]
                if not rest or rest[0] in ' !?,.\t':
                    return True
        return False

    def _generate_greeting_response(self, query: str) -> str:
        try:
            greeting_system = """You are BabyJay, KU's friendly campus assistant.

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
            return _call_haiku(
                self.client, greeting_system,
                [{"role": "user", "content": query}],
                temperature=0.9, max_tokens=100
            )
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
            about_system = """You are BabyJay, KU's campus assistant.

Introduce yourself naturally and briefly (3-5 sentences). Mention you can help with:
courses (including LIVE seat availability), professors, campus life, transit, and tuition/aid.

No bullet points. No emojis."""
            return _call_haiku(
                self.client, about_system,
                [{"role": "user", "content": query}],
                temperature=0.8, max_tokens=200
            )
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

    def _is_off_topic(self, query: str) -> bool:
        """Detect clearly non-KU questions before running the full pipeline."""
        q = query.lower().strip()

        # Allow if it contains KU-related keywords
        ku_keywords = [
            "ku", "kansas", "jayhawk", "rock chalk", "lawrence",
            "professor", "prof", "faculty", "course", "class", "eecs", "campus",
            "dorm", "dining", "bus", "transit", "tuition", "admission", "enroll",
            "library", "rec center", "gym", "scholarship", "financial aid",
            "housing", "building", "department", "degree", "major", "minor",
            "semester", "finals", "midterm", "gpa", "registrar", "advising",
            "babyjay", "baby jay", "parking", "student", "undergraduate", "graduate",
        ]
        if any(kw in q for kw in ku_keywords):
            return False

        # Detect clearly off-topic patterns
        off_topic_patterns = [
            r"^(what|who|when|where|how)\s+(is|are|was|were|did|do|does)\s+(the\s+)?(capital|president|population|weather|tallest|biggest|fastest|oldest)",
            r"\b(recipe|cook|bake|ingredient)\b",
            r"\b(movie|film|actor|actress|netflix|spotify|song|album|singer)\b",
            r"\b(stock|bitcoin|crypto|invest|trading)\b",
            r"\b(write me|write a|generate a|create a)\s+(poem|story|essay|song|code|script|email)\b",
            r"^(solve|calculate|compute|what is \d)",
            r"\b(translate|translation)\b",
            r"\b(nba|nfl|mlb|fifa|world cup|super bowl|olympics)\b",
            r"\b(diet|weight loss|workout plan|medical advice|symptom|diagnos)\b",
            r"\b(tell me a joke|fun fact|riddle|trivia)\b",
        ]
        for pattern in off_topic_patterns:
            if re.search(pattern, q, re.IGNORECASE):
                return True

        return False

    # Known typos + text-speak tokens that signal a query needs cleaning.
    _MESSY_TOKENS = {
        "machien", "machie", "artifical", "robtics", "quantim", "quantom",
        "profesors", "proffessor", "reseach", "compter", "avilable", "availble",
        "lerning", "introducton",
        # text-speak (matched as whole words below)
        "abt", "wnt", "noe", "wat", "u", "r", "ur", "tho", "bc", "cuz",
    }

    def _needs_cleaning(self, query: str) -> bool:
        q = query.lower()

        # Fast structural signals (excess punctuation / double spaces)
        if "  " in q or "???" in q or "!!!" in q:
            return True

        # Tokenized check — avoids "u" matching "unique" etc.
        tokens = re.findall(r"[a-z0-9']+", q)
        if any(tok in self._MESSY_TOKENS for tok in tokens):
            return True

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

            # Only count as "cleaned" if the preprocessor actually recorded a
            # correction (synonym or typo). A bare lowercase/whitespace diff is
            # not a real fix and shouldn't bypass the LLM cleaning fallback.
            if corrections:
                if self.debug:
                    print(f"[DEBUG] Local fix: '{query}' → '{local_cleaned}'")
                return local_cleaned, True, "local"
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Local preprocessing failed: {e}")

        # STEP 2: LLM
        try:
            llm_cleaned = _call_haiku(
                self.client,
                "Fix typos, grammar, and text speak. Return ONLY the corrected query, nothing else.",
                [{"role": "user", "content": query}],
                temperature=0, max_tokens=100
            )

            if llm_cleaned and len(llm_cleaned) < len(query) * 3:
                if self.debug:
                    print(f"[DEBUG] LLM fix: '{query}' → '{llm_cleaned}'")
                return llm_cleaned, True, "llm"
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] LLM cleaning failed: {e}")

        return query, False, "none"

    # ==================== FOLLOW-UP & CLARIFICATION ====================

    # Compiled once at class load — word-boundary match so "his" does not match "history",
    # "her" does not match "where", "that" does not match "thatcher", etc.
    _FOLLOWUP_RE = re.compile(
        r"\b("
        r"his|her|their|its|it'?s|that|this|these|those|them|they|"
        r"what\s+about|how\s+about|about\s+it|about\s+that|"
        r"sorry|my\s+bad|actually|i\s+meant|i\s+mean|"
        r"no\s+wait|wait\s+no|not\s+that|not\s+the|no\s+not|instead|oops|correction|"
        r"that\s+course|this\s+course|the\s+course"
        r")\b",
        re.IGNORECASE,
    )

    # Pronouns that always need prior-context resolution, regardless of length
    _PRONOUN_RE = re.compile(
        r"\b(he|she|him|her|his|hers|they|them|their|it\b|that professor|this professor|"
        r"that course|this course|that program|this program)\b",
        re.IGNORECASE,
    )

    def _is_simple_followup(self, question: str) -> bool:
        q = question.lower().strip()
        if q in ["it", "it?"] and self._conversation_history:
            return True
        # Always treat pronoun-containing questions as follow-ups so we expand them
        if self._conversation_history and self._PRONOUN_RE.search(q):
            return True
        # For short questions, check the broader follow-up pattern
        if len(question.split()) > 8:
            return False
        return bool(self._FOLLOWUP_RE.search(q))

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
        """Return True only if we are waiting for a clarification AND the new
        query looks like an actual answer (short, with no question/sentence
        structure).

        If the user ignored the clarification and asked a new full question,
        the caller should clear the waiting state and treat it as a new query
        — otherwise the two get mis-combined (e.g. "professors" + "history of
        KU" → "professors history of KU" → dumps the history department).
        """
        if not self.waiting_for_clarification:
            return False
        q = query.strip()
        if not q:
            return False
        if len(q.split()) > 4:
            return False

        q_lower = q.lower()
        # Sentence / question markers — if present at all, this is a new
        # question, not a clarification answer. Uses word-boundary for the
        # small words so "of" won't match "office" but WILL match "history of".
        if "?" in q_lower:
            return False
        if re.search(
            r"\b(what|who|where|when|why|how|which|is|are|at|for|to|do|does|tell|show|give|list)\b",
            q_lower,
        ):
            return False
        # Non-answer nouns — things the user might ask about but that are not
        # valid answers to "which department/research area?".
        if re.search(
            r"\b(hours|schedule|tuition|cost|apply|application|deadline|"
            r"bus|transit|dining|food|housing|dorm|library|parking|gym|"
            r"calendar|finals|holiday|break)\b",
            q_lower,
        ):
            return False
        return True

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

        # Recurse with use_history=False so we don't double-save messages.
        # The outer caller (API route or CLI) owns history persistence.
        return self.ask(combined_query, use_history=False)

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

        recent_history = self._conversation_history[-10:]
        history_text = "\n".join(
            [f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content'][:200]}" for msg in recent_history]
        )
        if not history_text or len(history_text) < 10:
            return question

        try:
            expanded = _call_haiku(
                self.client,
                "Rewrite the follow-up question to include full context. Return only the rewritten question.",
                [{"role": "user", "content": f"Conversation:\n{history_text}\n\nFollow-up: {question}\n\nRewritten:"}],
                temperature=0, max_tokens=100
            )
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

        # Follow-ups with conversation history skip greeting/about/off-topic checks entirely
        # "his number?" should NOT be caught by greeting detector just because it starts with "hi"
        _is_followup = self._conversation_history and self._is_simple_followup(question)

        if not _is_followup:
            # Greeting
            if self._is_greeting(question):
                if self.debug:
                    print("[DEBUG] Greeting detected, generating unique response")
                response = self._generate_greeting_response(question)
                if use_history:
                    self._save_message("user", question)
                    self._save_message("assistant", response)
                if self.debug:
                    print(f"[DEBUG] Greeting response time: {(time_module.time() - start_time) * 1000:.0f}ms")
                return response

            # About bot
            if self._is_about_bot(question):
                if self.debug:
                    print("[DEBUG] About bot question detected")
                response = self._generate_about_response(question)
                if use_history:
                    self._save_message("user", question)
                    self._save_message("assistant", response)
                return response

            # Off-topic detection — skip the full pipeline for clearly non-KU questions
            if self._is_off_topic(question):
                if self.debug:
                    print("[DEBUG] Off-topic query detected, skipping pipeline")
                response = (
                    "That's a bit outside my area! I'm all about KU — courses, professors, "
                    "campus life, dining, housing, and more. Is there anything KU-related I can help you with?"
                )
                if use_history:
                    self._save_message("user", question)
                    self._save_message("assistant", response)
                return response
        else:
            if self.debug:
                print(f"[DEBUG] Follow-up detected: '{question}' — skipping greeting/about/off-topic")

        # Clarification answer
        if self._is_clarification_answer(question):
            return self._process_clarification_answer(question)
        elif self.waiting_for_clarification:
            # User ignored the clarification and asked something new — clear
            # stale state so we don't accidentally combine later.
            self.waiting_for_clarification = False
            self.original_ambiguous_query = None
            self.clarification_context = None

        # Ambiguous -> ask
        if self._is_ambiguous(question):
            clarification_q = self._generate_clarification_question(question)
            self.waiting_for_clarification = True
            self.original_ambiguous_query = question
            self.clarification_context = {"original_query": question, "timestamp": datetime.now().isoformat()}
            if self.debug:
                print(f"[DEBUG] Ambiguous query: '{question}' → asking clarification")
            return clarification_q

        # Cleaning and preprocessing (single pass)
        cleaned, was_cleaned, method = self._clean_query_hybrid(question)
        search_question = cleaned

        if self.debug and was_cleaned:
            print(f"[DEBUG] Cleaned ({method}): '{question}' → '{cleaned}'")

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
                except Exception as e:
                    if self.debug:
                        print(f"[DEBUG] Expansion error: {e}")

            # Track faculty queries (word-boundary match — "ai" must NOT match "main", "train", etc.)
            faculty_kw_re = re.compile(
                r"\b(professors?|faculty|researchers?|research|ml|ai|machine\s+learning)\b",
                re.IGNORECASE,
            )
            if faculty_kw_re.search(search_question):
                self.last_search_query = search_query

            # Route
            try:
                if self.debug:
                    print(f"[DEBUG] Routing query: '{search_query}'")
                results = self.router.route(search_query)
                # Use context builder for compressed, cited context
                context = self.context_builder.build(search_question, results)
                if not context:
                    context = results.get("context", "")
                source = results.get("source", "unknown")
                result_count = results.get("result_count", 0)
                if self.debug:
                    print(f"[DEBUG] Router: {source}, {result_count} results, {len(context)} chars")
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Router error: {e}")

            # Retry with KU prefix — only if the query looks KU-related but missed
            if not context and not self._is_off_topic(search_question):
                # Only add KU prefix if query doesn't already contain it (word boundary — "kurt" must not count as "ku")
                if not re.search(r"\b(ku|kansas)\b", search_query, re.IGNORECASE):
                    retry_query = f"KU {search_query}"
                    if self.debug:
                        print(f"[DEBUG] Retry: '{retry_query}'")
                    try:
                        retry_results = self.router.route(retry_query)
                        context = retry_results.get("context", "")
                    except Exception as e:
                        if self.debug:
                            print(f"[DEBUG] Retry error: {e}")

            # Follow-up fallback to recent context — only if it's actually a follow-up
            _is_pronoun_followup = bool(
                self._conversation_history and self._PRONOUN_RE.search(search_question)
            )
            if self._is_simple_followup(search_question):
                if not context and self.recent_context:
                    # No fresh retrieval — fall back to last known context
                    context = self.recent_context
                    if self.debug:
                        print("[DEBUG] Using recent_context for follow-up")
                elif context:
                    if _is_pronoun_followup and self.recent_context and len(self.recent_context) > len(context):
                        # Pronoun-based follow-up (he/she/his/her) found fresh context,
                        # but the pronoun refers to an entity already in recent_context.
                        # Prefer the richer prior context so "his office?" doesn't
                        # replace professor data with unrelated building-info results.
                        if self.debug:
                            print("[DEBUG] Pronoun follow-up: keeping richer recent_context over fresh context")
                        context = self.recent_context
                    else:
                        # Non-pronoun follow-up shifting to a new topic, or fresh
                        # context is richer — update recent_context.
                        self.recent_context = context
            else:
                # New topic — update or clear recent_context
                self.recent_context = context if context else ""

        # Cap context size to avoid token overflow (max ~4000 chars)
        MAX_CONTEXT_CHARS = 4000
        if context and len(context) > MAX_CONTEXT_CHARS:
            # Cut at the last newline before the limit so we don't break mid-sentence or mid-URL
            cut = context.rfind("\n", 0, MAX_CONTEXT_CHARS)
            if cut == -1:
                cut = MAX_CONTEXT_CHARS
            context = context[:cut] + "\n\n[...additional results truncated for brevity]"

        # Build LLM messages. _call_haiku takes the system prompt as a separate
        # parameter, so we don't prepend it to the messages list here.
        enhanced_prompt = self.rlhf_optimizer.enhance_prompt(SYSTEM_PROMPT, question)
        messages: List[Dict] = []

        # Cap conversation history to last 6 messages to control token usage
        if use_history and self._conversation_history:
            messages.extend(self._conversation_history[-10:])

        if context:
            user_msg = (
                f"Here's information from KU's database (use ONLY this to answer — do not add anything beyond what is here):\n\n"
                f"{context}\n\n"
                f"User's question: {question}\n\n"
                f"Answer based ONLY on the information above. If the context doesn't answer their question, say so."
            )
        else:
            user_msg = (
                f"User's question: {question}\n\n"
                f"I don't have specific information about this in my database right now. "
                f"Let the user know you don't have that info and suggest they check ku.edu or contact the relevant KU office. "
                f"Do NOT guess or make up any specific details like names, numbers, URLs, or office locations."
            )

        messages.append({"role": "user", "content": user_msg})

        try:
            if self.debug:
                print("[DEBUG] Calling Claude Haiku...")
            assistant_msg = _call_haiku(
                self.client, enhanced_prompt, messages,
                temperature=0.5, max_tokens=1500
            )
            if use_history:
                self._save_message("user", question)
                self._save_message("assistant", assistant_msg)
            if self.debug:
                print(f"[DEBUG] Total time: {(time_module.time() - start_time) * 1000:.0f}ms")
            return assistant_msg
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] LLM error: {e}")
            # Return raw context if available instead of generic error
            if context:
                return f"I found some information but had trouble formatting it. Here's what I have:\n\n{context[:1500]}"
            return "Sorry, I'm having trouble right now. Please try again in a moment!"


def main():
    """Interactive CLI."""
    print("=" * 60)
    print("BabyJay - KU Campus Assistant")
    print("=" * 60)

    chat = BabyJayChat(debug=True)
    print(f"\nSession: {chat.session_id[:8]}...")
    print("\nCommands: 'quit', 'clear', 'debug on/off'")
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
                print("History cleared")
                continue
            if cmd == "debug on":
                chat.debug = True
                print("Debug ON")
                continue
            if cmd == "debug off":
                chat.debug = False
                print("Debug OFF")
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
