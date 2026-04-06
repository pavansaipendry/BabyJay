# BabyJay Function Reference
**Every function in every file — what it does, when it fires, parameters, returns, internals.**

---

## Table of Contents
1. [api_routes.py — HTTP Layer](#1-api_routespy--http-layer)
2. [chat.py — Brain / Orchestrator](#2-chatpy--brain--orchestrator)
3. [classifier.py — Intent Detection](#3-classifierpy--intent-detection)
4. [router.py — Query Routing](#4-routerpy--query-routing)
5. [query_decomposer.py — Multi-Part Splitting](#5-query_decomposerpy--multi-part-splitting)
6. [query_preprocessor.py — Typo / Synonym Fix](#6-query_preprocessorpy--typo--synonym-fix)
7. [retriever.py — Vector + Hybrid Search](#7-retrieverpy--vector--hybrid-search)
8. [bm25_scorer.py — Keyword Scoring](#8-bm25_scorerpy--keyword-scoring)
9. [faculty_retriever.py — Faculty JSON Search](#9-faculty_retrieverpy--faculty-json-search)
10. [course_retriever.py — Course JSON Search](#10-course_retrieverpy--course-json-search)
11. [campus_retriever.py — Dining/Transit/Housing/Tuition](#11-campus_retrieverpy--diningtransithousingtuition)
12. [faculty_search.py — ChromaDB Faculty Search](#12-faculty_searchpy--chromadb-faculty-search)
13. [context_builder.py — Context Compression](#13-context_builderpy--context-compression)
14. [reranker.py — LLM Re-ranking](#14-rerankerpy--llm-re-ranking)
15. [rlhf_optimizer.py — Feedback Learning](#15-rlhf_optimizerpy--feedback-learning)
16. [embeddings.py — Data → ChromaDB Loader](#16-embeddingspy--data--chromadb-loader)
17. [openai_embeddings.py — Embedding Function](#17-openai_embeddingspy--embedding-function)
18. [regenerate_faculty_embeddings.py — Faculty Re-embed Script](#18-regenerate_faculty_embeddingspy--faculty-re-embed-script)

---

## 1. `api_routes.py` — HTTP Layer

**File:** `app/routers/api_routes.py` (245 lines)
**Purpose:** FastAPI endpoints. Receives HTTP requests, manages chat instances, saves to Supabase.

### Module-Level

| Name | Type | Description |
|------|------|-------------|
| `router` | `APIRouter` | FastAPI router with prefix `/api` |
| `_chat_instances` | `Dict[str, BabyJayChat]` | In-memory cache of chat instances keyed by conversation_id. Preserves follow-up state (department filters, clarification, last_mentioned_course) across HTTP requests. |
| `_MAX_CACHED_INSTANCES` | `int` | 200. When exceeded, evicts the oldest instance (FIFO). |

### `_get_or_create_chat(conversation_id: str) → BabyJayChat`
**Line 23 | Called by:** `chat()` endpoint
- Checks `_chat_instances` dict for existing instance
- If found → returns it (preserves follow-up state)
- If not found → creates new `BabyJayChat(session_id=conversation_id, use_redis=False, debug=False)`
- Evicts oldest if cache ≥ 200

### `POST /api/chat` → `chat(request, user, db)`
**Line 73 | Called by:** Frontend when user sends message (authenticated)
- **Params:** `request.message` (str, 1-5000 chars), `request.conversation_id` (optional str)
- **Auth:** JWT token required via `get_current_user`
- **Flow:**
  1. If no `conversation_id` → creates new conversation in Supabase via `db.create_conversation()`
  2. Gets/creates persistent chat instance via `_get_or_create_chat()`
  3. Loads DB history into instance if empty (`db.get_recent_messages()`)
  4. Calls `chat_instance.ask(message, use_history=False)` — history managed externally
  5. Appends user + assistant messages to instance's `_conversation_history`
  6. Saves both messages to Supabase via `db.add_message()`
  7. Generates title for new conversations
- **Returns:** `ChatResponse(response, conversation_id, title)`

### `POST /api/chat/anonymous` → `chat_anonymous(request)`
**Line 140 | Called by:** Frontend for unauthenticated users
- Creates a **new** BabyJayChat each time (no state preserved)
- Calls `chat_instance.ask(message, use_history=False)`
- Returns response with `conversation_id=None, title=None`

### `GET /api/conversations` → `list_conversations(user, db, limit=50)`
**Line 159 | Called by:** Frontend sidebar
- Returns list of user's conversations from Supabase, sorted by most recent

### `GET /api/conversations/{conversation_id}` → `get_conversation(conversation_id, user, db)`
**Line 179 | Called by:** Frontend when opening a conversation
- Returns conversation metadata + all messages

### `PUT /api/conversations/{conversation_id}` → `update_conversation(conversation_id, request, user, db)`
**Line 211 | Called by:** Frontend rename
- Updates conversation title in Supabase

### `DELETE /api/conversations/{conversation_id}` → `delete_conversation(conversation_id, user, db)`
**Line 226 | Called by:** Frontend delete
- Deletes conversation + all messages from Supabase

### `GET /api/health` → `health_check()`
**Line 242 | Called by:** Health monitoring
- Returns `{"status": "healthy", "service": "babyjay-api"}`

### Pydantic Models
| Model | Fields | Used By |
|-------|--------|---------|
| `ChatRequest` | `message` (str, 1-5000), `conversation_id` (optional str) | `/chat`, `/chat/anonymous` |
| `ChatResponse` | `response` (str), `conversation_id` (optional str), `title` (optional str) | All chat endpoints |
| `ConversationResponse` | `id`, `title`, `created_at`, `updated_at` | Conversation list/detail |
| `MessageResponse` | `id`, `role`, `content`, `created_at` | Conversation detail |
| `ConversationDetailResponse` | `conversation` (ConversationResponse), `messages` (List[MessageResponse]) | GET conversation |
| `UpdateTitleRequest` | `title` (str) | PUT conversation |

---

## 2. `chat.py` — Brain / Orchestrator

**File:** `app/rag/chat.py` (853 lines)
**Purpose:** The main orchestrator. Decides what to do with each query: greeting? follow-up? off-topic? ambiguous? Then cleans, routes, retrieves, builds context, and calls the LLM.

### Module-Level

#### `_call_haiku(client, system, messages, temperature=0.5, max_tokens=1500) → str`
**Line 34 | Called by:** Every LLM call in chat.py
- Unified helper to call Claude Haiku via Anthropic API
- Separates system prompt from conversation messages
- **Merges consecutive same-role messages** (Anthropic requires alternating roles)
- Ensures conversation starts with user message (inserts "Hello" if needed)
- Returns `response.content[0].text`

#### Constants
| Name | Line | Description |
|------|------|-------------|
| `GREETING_PATTERNS` | 64 | List of 30+ greeting strings ("hi", "hello", "thanks", "bye", etc.) |
| `GREETING_REGEX_PATTERNS` | 76 | 6 regex patterns for greeting variations |
| `SYSTEM_PROMPT` | 85 | The full system prompt for BabyJay — scope rules, personality, response style, course priority, department filtering rules. ~150 lines. |
| `HAIKU_MODEL` | 31 | `"claude-haiku-4-5-20251001"` |

### Class: `ConversationStore`
**Line 154 | Purpose:** Handles persistent storage using Redis (with in-memory fallback)

#### `__init__(self, redis_host="localhost", redis_port=6379, use_redis=True)`
**Line 157**
- Tries to connect to Redis. If fails → falls back to `memory_store` dict

#### `save_message(self, session_id: str, role: str, content: str, ttl_days: int = 30)`
**Line 174 | Called by:** `BabyJayChat._save_message()`
- Redis: `RPUSH chat:history:{session_id}` with 30-day TTL
- Memory: appends to `memory_store[session_id]` list

#### `load_history(self, session_id: str, max_messages: int = 100) → List[Dict]`
**Line 185 | Called by:** `BabyJayChat._load_from_store()`
- Redis: `LRANGE` last N messages
- Memory: slice last N from list

#### `clear_history(self, session_id: str)`
**Line 192 | Called by:** `BabyJayChat.clear_history()`
- Redis: `DELETE` key. Memory: `del` from dict

### Class: `BabyJayChat`
**Line 199 | Purpose:** The main chat class. One instance per conversation.

#### `__init__(self, session_id=None, use_redis=True, debug=False)`
**Line 200**
- Creates: Anthropic client, QueryRouter, ContextBuilder, ConversationStore, RLHFOptimizer
- Generates UUID session_id if none provided
- State variables:
  - `recent_context` — last successful context (for follow-up fallback)
  - `last_search_query` — last faculty-related search (for department filtering)
  - `waiting_for_clarification` — bool flag for ambiguous queries
  - `clarification_context` — stores original ambiguous query
  - `active_department_filter` — stores active department filter
  - `last_mentioned_course` — tracks last course mentioned
  - `_conversation_history` — list of {role, content} dicts
- Calls `_load_from_store()` to restore history

#### `_load_from_store(self)`
**Line 220 | Called by:** `__init__`
- Loads conversation history from ConversationStore into `_conversation_history`

#### `_save_message(self, role: str, content: str)`
**Line 224 | Called by:** `ask()` (when `use_history=True`)
- Appends to `_conversation_history` AND saves to ConversationStore

#### `conversation_history` (property)
**Line 228 | Called by:** External code
- Returns `_conversation_history`

#### `clear_history(self)`
**Line 232 | Called by:** CLI "clear" command
- Resets all state: history, context, search query, department filter, last course

---

### Greeting / About / Off-topic Detection

#### `_is_greeting(self, query: str) → bool`
**Line 242 | Called by:** `ask()` (only if NOT a follow-up)
- **Step 1:** Exact match against `GREETING_PATTERNS` list
- **Step 2:** Regex match against `GREETING_REGEX_PATTERNS`
- **Step 3:** Startswith match with **word boundary check** — "hi" won't match "his" because it checks that the next character after the greeting is a space, punctuation, or end of string
- The word boundary fix prevents "his number?" from being detected as a greeting

#### `_generate_greeting_response(self, query: str) → str`
**Line 258 | Called by:** `ask()` when greeting detected
- Calls Haiku with a greeting-specific system prompt (temperature=0.9 for variety)
- Rules: keep short, vary opening, match energy, occasional "Rock Chalk!"
- Fallback: "Hey! What can I help you with today?"

#### `_is_about_bot(self, query: str) → bool`
**Line 284 | Called by:** `ask()` (only if NOT a follow-up)
- Checks for phrases like "who are you", "what can you do", "are you ai", etc.

#### `_generate_about_response(self, query: str) → str`
**Line 296 | Called by:** `ask()` when about-bot detected
- Calls Haiku with about-bot system prompt (temperature=0.8)
- Introduces BabyJay with capabilities

---

### Query Validation & Cleaning

#### `_validate_query(self, query: str) → bool`
**Line 319 | Called by:** `ask()`, `_expand_followup_question()`
- Returns False if empty/whitespace-only or has no alphanumeric characters

#### `_is_off_topic(self, query: str) → bool`
**Line 324 | Called by:** `ask()` (only if NOT a follow-up)
- **Step 1:** If query contains ANY KU keyword (37 keywords like "ku", "professor", "campus", "eecs", etc.) → NOT off-topic
- **Step 2:** Checks 10 regex patterns for clearly off-topic content: recipes, movies, stocks, code generation, math solving, translation, pro sports, medical advice, jokes/trivia
- Returns True only if Step 2 matches AND Step 1 didn't match

#### `_needs_cleaning(self, query: str) → bool`
**Line 360 | Called by:** `_clean_query_hybrid()`
- Checks for common typos ("machien", "artifical", "robtics", etc.)
- Checks for text speak (" 2 ", " u ", " abt ", " wat ", etc.)
- Checks for formatting issues (multiple spaces, excessive punctuation)
- Returns True if any messy indicator matches

#### `_clean_query_hybrid(self, query: str) → Tuple[str, bool, str]`
**Line 382 | Called by:** `ask()`
- Returns `(cleaned_query, was_cleaned, method)`
- **Step 1:** If `_needs_cleaning()` is False → return original
- **Step 2:** Try local preprocessing via `QueryPreprocessor.preprocess()` — catches typos, synonyms
- **Step 3:** If local fix failed, try LLM cleaning via Haiku (temperature=0, "Fix typos, grammar, and text speak")
- Method is "none", "local", or "llm"

---

### Follow-up & Clarification

#### `_is_simple_followup(self, question: str) → bool`
**Line 424 | Called by:** `ask()` (early in the flow, BEFORE greeting check)
- Returns True if:
  - Query ≤ 8 words AND contains follow-up indicators ("his", "her", "their", "what about", "that course", etc.)
  - Special case: "it" or "it?" with conversation history
- This is critical — when True, the greeting/about/off-topic checks are SKIPPED

#### `_is_department_filter(self, question: str) → bool`
**Line 439 | Called by:** `ask()`, `_is_ambiguous()`
- True if query contains "only"/"just" AND a department keyword ("eecs", "business", etc.)
- Example: "EECS only" → True

#### `_is_ambiguous(self, query: str) → bool`
**Line 446 | Called by:** `ask()`
- Returns True for vague single-word queries like "professors", "courses", "help"
- Skips if it's a follow-up, department filter, or ≥3 words

#### `_generate_clarification_question(self, query: str) → str`
**Line 470 | Called by:** `ask()` when ambiguous
- Returns a specific clarification prompt based on the vague term
- "professors" → "What research area or department are you interested in?"

#### `_is_clarification_answer(self, query: str) → bool`
**Line 482 | Called by:** `ask()`
- True if `waiting_for_clarification` is True AND query is non-empty

#### `_process_clarification_answer(self, answer: str) → str`
**Line 485 | Called by:** `ask()`
- Combines original ambiguous query with the user's clarification answer
- "professors" + "EECS machine learning" → "EECS machine learning professors"
- Resets clarification state and recursively calls `self.ask()` with the combined query

#### `_extract_department(self, question: str) → Optional[str]`
**Line 509 | Called by:** `ask()` for department filtering
- Maps keywords to full department names: "eecs" → "Electrical Engineering and Computer Science"
- Covers 7 departments

#### `_filter_context_by_department(self, context: str, department: str) → str`
**Line 527 | Called by:** `ask()` for department filtering
- Parses faculty context line-by-line, only keeps professors whose "Department:" line matches the filter
- Used when user says "EECS only" after a broad faculty search

#### `_expand_followup_question(self, question: str) → str`
**Line 570 | Called by:** `ask()` for follow-up queries
- Takes last 6 messages of conversation history
- Calls Haiku: "Rewrite the follow-up question to include full context"
- Example: "his number?" + history about Prof Kulkarni → "What is Professor Kulkarni's phone number?"
- If expansion fails or is shorter than original → returns original

---

### Main Ask Method

#### `ask(self, question: str, use_history: bool = True) → str`
**Line 602 | Called by:** API routes, CLI
- **THE MAIN FUNCTION.** Everything converges here. Full flow:

1. **Validation** (lines 605-611): Empty check, `_validate_query()`
2. **Follow-up detection** (line 615): Check `_is_simple_followup()` FIRST
3. **Decision tree** (lines 617-654):
   - If NOT follow-up → check greeting → about bot → off-topic
   - If IS follow-up → skip all three, go straight to pipeline
4. **Clarification** (lines 657-668): Check `_is_clarification_answer()`, then `_is_ambiguous()`
5. **Cleaning** (line 671): `_clean_query_hybrid()` — typo fix
6. **Search/RAG** (lines 678-754):
   - If department filter + previous search → re-route with department
   - If follow-up → expand with `_expand_followup_question()`
   - Route via `self.router.route(search_query)`
   - Build context via `self.context_builder.build()`
   - If no context → retry with "KU " prefix
   - If follow-up and no context → fall back to `recent_context`
   - Cap context at 4000 chars
7. **LLM call** (lines 762-805):
   - Enhance prompt via RLHF optimizer
   - Build messages with conversation history (last 6 messages)
   - If context → "Here's information from KU's database... Answer based ONLY on this"
   - If no context → "I don't have specific information... say so"
   - Call Haiku (temperature=0.5, max_tokens=1500)
   - Save messages if `use_history=True`
   - On error: return raw context if available, else generic error

### `main()`
**Line 808 | Called by:** `python -m app.rag.chat`
- Interactive CLI loop. Supports "quit", "clear", "debug on/off" commands

---

## 3. `classifier.py` — Intent Detection

**File:** `app/rag/classifier.py` (387 lines)
**Purpose:** Classifies user queries into intents (faculty_search, course_info, dining_info, etc.) and extracts entities.

### Class: `QueryClassifier`

#### `__init__(self)`
**Line 21**
- Initializes Anthropic client
- Populates:
  - `subject_codes` — set of 60+ KU subject codes (EECS, AE, ME, etc.)
  - `department_aliases` — dict mapping dept keys to alias lists (27 departments)
  - `research_areas` — dict mapping research topics to keyword lists (11 areas)
  - `intent_patterns` — ordered dict of regex patterns for 12 intents
  - `complete_list_indicators` — regex patterns for "all", "every", "complete list", etc.

#### `classify(self, query: str, use_llm_fallback: bool = True) → Dict[str, Any]`
**Line 153 | Called by:** `QueryRouter.route()`
- **Step 1:** `_detect_intent_regex()` — fast regex matching
- **Step 2:** Extract entities based on intent (faculty or course)
- **Step 3:** `_detect_scope()` — top_results or complete_list
- **Step 4:** If confidence < 0.7 AND `use_llm_fallback` → `_classify_with_llm()`
- **Returns:** `{intent, entities, scope, confidence, method, original_query}`

#### `_detect_intent_regex(self, query_lower: str, query_original: str) → tuple`
**Line 188 | Called by:** `classify()`
- Checks for course codes first (high priority score = 3)
- Iterates all 12 intent patterns, counts regex matches as score
- Best intent = highest score
- Confidence formula: `min(0.6 + max_score * 0.15, 0.95)`
- No matches → `("general", 0.3)`

#### `_extract_faculty_entities(self, query: str) → Dict[str, Any]`
**Line 217 | Called by:** `classify()` when intent is faculty_search
- Extracts `department` by matching against 27 department alias lists
- Extracts `research_area` by matching against 11 research area keyword lists
- Extracts `name` using regex: "Dr./Prof. Firstname Lastname" patterns

#### `_extract_course_entities(self, query: str) → Dict[str, Any]`
**Line 254 | Called by:** `classify()` when intent is course_info
- Extracts `course_code` via regex `[A-Z]{2,4}\s*\d{3,4}` (e.g., "EECS 168")
- Extracts `subject` by matching against 60+ subject codes
- Extracts `level` — "graduate" or "undergraduate"
- Extracts `credits` — number before "credit/cr/hour"

#### `_detect_scope(self, query: str) → str`
**Line 288 | Called by:** `classify()`
- Checks for complete list indicators ("all", "every", "full list", etc.)
- Returns "complete_list" or "top_results"

#### `_classify_with_llm(self, query: str) → Optional[Dict[str, Any]]`
**Line 295 | Called by:** `classify()` when regex confidence < 0.7
- Calls Haiku (temperature=0, max_tokens=200)
- System prompt instructs JSON output with intent, entities, scope, confidence
- Parses JSON response, adds `method: "llm"`
- Returns None on any error

#### `get_department_key(self, alias: str) → Optional[str]`
**Line 338 | Called by:** External code
- Converts a department alias ("eecs", "cs") to canonical key ("eecs")

#### `get_all_departments(self) → List[str]`
**Line 346 | Called by:** External code
- Returns list of all department keys

---

## 4. `router.py` — Query Routing

**File:** `app/rag/router.py` (416 lines)
**Purpose:** Routes classified queries to the right retriever. The traffic cop of the system.

### Module-Level

#### `_get_reranker() → Optional[Reranker]`
**Line 36 | Called by:** `_route_vector_fallback()`
- Lazy singleton pattern — imports and creates Reranker on first call
- Returns None if import fails

### Class: `QueryRouter`

#### `__init__(self)`
**Line 50**
- Creates: QueryClassifier, FacultyRetriever, CampusRetriever, CourseRetriever, ContextBuilder, QueryDecomposer
- Lazy-loads vector retriever via property

#### `vector_retriever` (property)
**Line 62 | Called by:** `_route_vector_fallback()`
- Lazy imports and creates `Retriever()` on first access
- Avoids circular imports

#### `route(self, query: str, use_vector_fallback: bool = True) → Dict[str, Any]`
**Line 70 | Called by:** `BabyJayChat.ask()`
- **THE MAIN ROUTING FUNCTION.**
- **Step 0:** Check if query needs decomposition (multi-part questions)
  - If `decomposer.should_decompose()` → split into sub-queries → route each → merge results
- **Step 1:** Classify via `self.classifier.classify(query)`
- **Step 2:** Route based on intent:
  - `faculty_search` → `_route_faculty()`
  - `course_info` → `_route_courses()`
  - `dining_info` → `_route_dining()`
  - `housing_info` → `_route_housing()`
  - `transit_info` → `_route_transit()`
  - `financial_info` → `_route_tuition()`
  - admission/library/recreation/safety/calendar/building → `_route_vector_fallback()`
  - general/unknown → `_route_vector_fallback()`
- **Returns:** `{results, context, source, query_info, result_count}`

#### `_empty_result(self, classification: Dict) → Dict[str, Any]`
**Line 131 | Called by:** Route methods when no retriever available
- Returns `{results: [], context: "", source: "none", ...}`

#### `_route_faculty(self, query, entities, scope, classification, use_vector_fallback) → Dict`
**Line 141 | Called by:** `route()` when intent is faculty_search
- **Strategy 1:** Name only → `faculty_retriever.search_by_name(name)`
- **Strategy 2:** Department + optional research → `faculty_retriever.search()` or `get_department_faculty()`
- **Strategy 3:** Research area only → `faculty_retriever.search(research_area=area)`
- **Strategy 4:** No results → fall back to vector search
- Formats context via `faculty_retriever.format_for_context()`

#### `_route_courses(self, query, entities, scope, classification, use_vector_fallback) → Dict`
**Line 190 | Called by:** `route()` when intent is course_info
- **Strategy 1:** Exact course code → `course_retriever.get_course(code)`
- **Strategy 2:** Subject + optional level → `course_retriever.search_by_subject()` + level filter
- **Strategy 3:** Level only → `course_retriever.search_by_level()`
- **Strategy 4:** General search → `course_retriever.search(query)`
- **Strategy 5:** No results → fall back to vector search
- Limit: 50 for complete_list, 10 for top_results

#### `_route_dining(self, query, entities, scope, classification) → Dict`
**Line 237 | Called by:** `route()` when intent is dining_info
- `campus_retriever.search_dining(query)` → `format_dining_context()`
- Limit: 20 for complete_list, 5 for top_results

#### `_route_housing(self, query, entities, scope, classification) → Dict`
**Line 253 | Called by:** `route()` when intent is housing_info
- `campus_retriever.search_housing(query)` → `format_housing_context()`

#### `_route_transit(self, query, entities, scope, classification) → Dict`
**Line 269 | Called by:** `route()` when intent is transit_info
- If "ku" or "campus" in query → `campus_retriever.get_ku_transit()`
- Otherwise → `campus_retriever.search_transit(query)`
- Formats via `format_transit_context()`

#### `_route_tuition(self, query, entities, scope, classification) → Dict`
**Line 291 | Called by:** `route()` when intent is financial_info
- `campus_retriever.search_tuition(query)` → `format_tuition_context()`

#### `_route_vector_fallback(self, query, intent, classification) → Dict`
**Line 307 | Called by:** `route()` for intents without specialized retrievers
- Calls `self.vector_retriever.smart_search(query)`
- Maps intent to result key (e.g., "faculty_search" → "faculty")
- If > 3 results → re-ranks via `_get_reranker().rerank(query, results, top_k=5)`
- Returns vector search context with source citations

---

## 5. `query_decomposer.py` — Multi-Part Splitting

**File:** `app/rag/query_decomposer.py` (227 lines)
**Purpose:** Splits complex multi-part questions into simple sub-queries for better retrieval.

### Module-Level Patterns
| Name | Purpose | Example Match |
|------|---------|---------------|
| `COMPARISON_PATTERNS` | "Compare X and Y", "X vs Y" | "Compare EECS 168 and EECS 268" |
| `MULTI_ENTITY_PATTERNS` | "Tell me about X, Y, and Z" | "Tell me about EECS 168, EECS 268" |
| `LIST_PATTERNS` | "Prerequisites for X, Y, Z" | "Prerequisites for AE 345 and AE 510" |
| `COURSE_CODE_RE` | Compiled regex `[A-Z]{2,5}\s*\d{3,4}` | Finds "EECS 168" in any text |

### Class: `QueryDecomposer`

#### `should_decompose(self, query: str) → bool`
**Line 47 | Called by:** `QueryRouter.route()` (Step 0)
- Returns False if < 15 chars
- Checks comparison patterns, multiple course codes (≥2), multi-entity patterns, list patterns
- Pure regex, zero API calls

#### `decompose(self, query: str) → List[str]`
**Line 77 | Called by:** `QueryRouter.route()` when should_decompose is True
- **Strategy 1:** Multiple course codes → extract question part, create sub-query per code
  - "Compare EECS 168 and EECS 268 prerequisites" → ["EECS 168 prerequisites", "EECS 268 prerequisites"]
- **Strategy 2:** Comparison patterns → extract entities, infer question type
- **Strategy 3:** List patterns → extract prefix + entities
- **Strategy 4:** Multi-entity patterns → split entities
- Falls back to `[query]` if nothing works

#### `_extract_question_part(self, query, codes) → str`
**Line 140 | Called by:** `decompose()` (Strategy 1)
- Removes course codes and comparison words from query
- Returns what's left as the "question" (e.g., "prerequisites")

#### `_infer_question_type(self, query) → str`
**Line 159 | Called by:** `decompose()` (Strategy 2)
- Maps keywords to question types: "prerequisite" → "prerequisites", "credit" → "credits", etc.

#### `_extract_prefix(self, query, match) → str`
**Line 174 | Called by:** `decompose()` (Strategy 3)
- Extracts the question prefix before the entity list (e.g., "prerequisites for")

#### `merge_sub_results(self, sub_results: List[dict]) → dict`
**Line 184 | Called by:** `QueryRouter.route()` after decomposed sub-queries are routed
- Collects all results, contexts, and sources from sub-query results
- Deduplicates by content/name/course_code (first 100 chars)
- Joins contexts with "---" separator
- Returns merged dict with `intent: "multi_part"`

---

## 6. `query_preprocessor.py` — Typo / Synonym Fix

**File:** `app/rag/query_preprocessor.py` (412 lines)
**Purpose:** Fixes typos, expands synonyms, normalizes queries before search.

### Class: `QueryPreprocessor`

#### `__init__(self, valid_subject_codes: Set[str] = None)`
**Line 21**
- Populates `synonyms` dict (35+ entries): "ml" → "machine learning", "prereqs" → "prerequisites", etc.
- Populates `protected_words` set: common English words that shouldn't be corrected
- Populates `course_vocabulary` set: domain words for fuzzy matching ("engineering", "calculus", etc.)
- Sets `valid_subject_codes` (used for fuzzy code matching)

#### `set_subject_codes(self, codes: Set[str])`
**Line 110 | Called by:** External code after course data loaded
- Updates valid subject codes for fuzzy matching

#### `preprocess(self, query: str) → Dict`
**Line 114 | Called by:** `BabyJayChat._clean_query_hybrid()`, `CourseRetriever.search()`
- Full pipeline:
  1. `_normalize()` — lowercase, remove special chars
  2. `_detect_codes()` — protect course/subject codes from correction
  3. `_apply_synonyms()` — expand abbreviations
  4. `_apply_fuzzy_correction()` — fix remaining typos
- **Returns:** `{original, normalized, processed, corrections: [{original, corrected, type}]}`

#### `_normalize(self, query: str) → str`
**Line 162 | Called by:** `preprocess()`
- Removes emojis, special characters (keeps alphanumeric, spaces, basic punctuation)
- Normalizes whitespace
- Lowercases

#### `_detect_codes(self, query: str) → Tuple[List[str], List[str]]`
**Line 176 | Called by:** `preprocess()`
- Finds course codes like "EECS 168" → keeps them uppercase and protected
- Fuzzy-matches potential subject codes against valid codes
- Returns `(tokens_with_protected_codes, detected_codes)`

#### `_fuzzy_match_subject_code(self, token: str) → Optional[str]`
**Line 234 | Called by:** `_detect_codes()`
- Uses edit distance to match misspelled subject codes (threshold ≥ 80%)
- "eeecs" → "EECS"

#### `_apply_synonyms(self, tokens: List[str]) → Tuple[List[str], List[Dict]]`
**Line 256 | Called by:** `preprocess()`
- Handles multi-word synonyms first (e.g., "machine learning" → already correct)
- Then single-word synonyms (e.g., "ml" → "machine learning")
- Returns processed tokens + list of corrections made

#### `_apply_fuzzy_correction(self, tokens: List[str]) → Tuple[List[str], List[Dict]]`
**Line 306 | Called by:** `preprocess()`
- Skips protected words, subject codes, numbers
- Calls `_fuzzy_correct_word()` on remaining tokens

#### `_fuzzy_correct_word(self, word: str) → Optional[str]`
**Line 339 | Called by:** `_apply_fuzzy_correction()`
- Skips if word < 4 chars
- Uses fuzzy matching against `course_vocabulary` with threshold ≥ 85%
- Returns corrected word or None

---

## 7. `retriever.py` — Vector + Hybrid Search

**File:** `app/rag/retriever.py` (422 lines)
**Purpose:** The vector search layer. Queries ChromaDB for semantic search and merges with BM25 for hybrid results.

### Class: `Retriever`

#### `__init__(self, persist_directory: str = None)`
**Line 18**
- Connects to ChromaDB at `data/vectordb/`
- Gets `babyjay_knowledge` collection with OpenAI embedding function
- Creates `FacultySearcher` for the separate faculty collection
- Initializes empty `_bm25_cache` dict

#### `_get_bm25_index(self, source_filter: Optional[str] = None) → Optional[BM25Scorer]`
**Line 37 | Called by:** `search()`
- Lazy builds BM25 index for a given source filter
- Pulls up to 500 documents from ChromaDB for that source
- Creates `BM25Scorer`, indexes documents, caches by source key
- Returns cached scorer on subsequent calls

#### `search(self, query, n_results=5, source_filter=None, min_relevance=0.25) → List[Dict]`
**Line 68 | Called by:** `smart_search()`, all `search_*` methods
- **Vector search:** Queries ChromaDB, filters by source, converts distance to score, drops below min_relevance
- **BM25 search:** Gets/builds BM25 index, searches same query
- **Hybrid merge:** If both have results → `hybrid_merge(vector_results, bm25_results)` using RRF
- Falls back to vector-only if BM25 has no results
- **Returns:** List of `{content, metadata, relevance_score}` dicts

#### Domain-Specific Search Methods (all call `search()` with a source filter)
| Method | Source Filter | Default n_results |
|--------|-------------|-------------------|
| `search_dining` | "dining" | 3 |
| `search_courses` | "course" | 5 |
| `search_admissions` | "admission" | 5 |
| `search_calendar` | "calendar" | 5 |
| `search_faqs` | "faq" | 5 |
| `search_tuition` | "tuition" | 5 |
| `search_financial_aid` | "financial_aid" | 5 |
| `search_housing` | "housing" | 5 |
| `search_libraries` | "libraries" | 5 |
| `search_recreation` | "recreation" | 5 |
| `search_campus_safety` | "campus_safety" | 5 |
| `search_student_organizations` | "student_organizations" | 5 |

#### `search_transit(self, query, n_results=3) → List[Dict]`
**Line 111 | Called by:** `smart_search()`
- Searches BOTH "transit" and "transit_stop" sources, merges and sorts by score

#### `search_faculty_enhanced(self, query, n_results=5, department=None) → List[Dict]`
**Line 118 | Called by:** `smart_search()`, `search_faculty()`
- Uses `FacultySearcher` (ChromaDB faculty collection)
- Formats results with full professor details (name, dept, email, phone, office, research)

#### `search_faculty(self, query, n_results=5) → List[Dict]`
**Line 138**
- Alias for `search_faculty_enhanced()` without department filter

#### `_detect_departments_from_query(self, query) → List[str]`
**Line 141 | Called by:** `smart_search()`
- Maps 12 department categories to keyword lists
- Returns matching department names

#### `_extract_department_and_topic(self, query) → Tuple[Optional[str], str]`
**Line 169 | Called by:** `smart_search()`
- Separates department filter from research topic
- Removes department keywords, structure words from query
- Expands abbreviations (ml → machine learning, ai → artificial intelligence)
- Returns `(department_filter, research_topic)`

#### `_contains_word(self, text, word) → bool`
**Line 237**
- Word boundary check with plural support

#### `_contains_any_word(self, text, words) → bool`
**Line 254**
- Any-of word boundary check

#### `_wants_complete_list(self, query) → bool`
**Line 258**
- Detects "all", "every", "complete list" patterns

#### `smart_search(self, query, n_results=5) → Dict[str, Any]`
**Line 267 | Called by:** `QueryRouter._route_vector_fallback()`
- **THE BIG MULTI-DOMAIN SEARCH.** Used as fallback when specialized retrievers don't exist.
- Detects 16 domain flags using keyword matching (is_dining, is_transit, is_faculty, etc.)
- Faculty overrides course (avoids mis-routing professor questions)
- Runs domain-specific searches for each detected flag
- Faculty search uses smart department/topic extraction
- Builds combined context string with section headers
- **Returns:** Dict with all domain results + combined context string

---

## 8. `bm25_scorer.py` — Keyword Scoring

**File:** `app/rag/bm25_scorer.py` (208 lines)
**Purpose:** BM25 keyword scoring to complement vector search. Vector misses exact terms; BM25 catches them.

### Class: `BM25Scorer`

#### `__init__(self, k1=1.5, b=0.75)`
**Line 25**
- `k1` — term frequency saturation (higher = more weight on repeated terms)
- `b` — length normalization (0 = none, 1 = full)

#### `_tokenize(self, text: str) → List[str]`
**Line 40 | Called by:** `index_documents()`, `score()`
- Lowercase, split on non-alphanumeric
- Removes 70+ stopwords (the, a, is, are, etc.)
- Drops tokens ≤ 1 char

#### `index_documents(self, documents: List[Dict], content_key="content")`
**Line 58 | Called by:** `Retriever._get_bm25_index()`
- Tokenizes each document
- Builds document frequency counts (how many docs contain each term)
- Calculates average document length
- Sets `_indexed = True`

#### `_idf(self, term: str) → float`
**Line 87 | Called by:** `score_document()`
- BM25 IDF formula: `log((N - df + 0.5) / (df + 0.5) + 1)`
- +1 smoothing prevents negative IDF

#### `score_document(self, query_tokens: List[str], doc_idx: int) → float`
**Line 93 | Called by:** `score()`
- BM25 scoring: `idf * (tf * (k1+1)) / (tf + k1 * (1 - b + b * dl/avgdl))`
- Sums across all query terms

#### `score(self, query: str, top_k=10) → List[Tuple[int, float]]`
**Line 112 | Called by:** `search()`
- Tokenizes query, scores all documents, returns sorted (doc_idx, score) pairs

#### `search(self, query: str, top_k=5) → List[Dict]`
**Line 135 | Called by:** `Retriever.search()`
- Calls `score()`, returns document dicts with added `bm25_score` field

### Standalone Function

#### `hybrid_merge(vector_results, bm25_results, vector_weight=0.6, bm25_weight=0.4, top_k=5) → List[Dict]`
**Line 151 | Called by:** `Retriever.search()`
- **Reciprocal Rank Fusion (RRF)** — merges results by rank position, not raw scores
- RRF formula: `weight * (1 / (k + rank + 1))` where k=60
- Deduplicates by first 200 chars of content
- Default: 60% weight on vector, 40% on BM25
- Returns merged results with `hybrid_score` field

---

## 9. `faculty_retriever.py` — Faculty JSON Search

**File:** `app/rag/faculty_retriever.py` (462 lines)
**Purpose:** Fast faculty search using pre-loaded JSON data (not ChromaDB). Sub-50ms response times.

### Class: `FacultyRetriever`

#### Class Attribute: `TOPIC_DEPARTMENT_AFFINITY`
**Line 161 | Used by:** `_filter_by_research()`
- Dict mapping 30+ research topics to their "home" department sets
- Example: `"machine learning" → {"eecs", "electrical_engineering_and_computer_science"}`
- Used to rank professors by how relevant their department is to the research topic

#### `__init__(self, data_dir=None)`
**Line 19**
- Sets data directory, calls `_load_data()`

#### `_load_data(self)`
**Line 48 | Called by:** `__init__`
- Loads `faculty_combined.json` (single file with all departments)
- Builds flat `_all_faculty` list with department info attached to each faculty member
- Builds `_by_department` dict for department-based lookup

#### `get_all_departments(self) → List[Dict]`
**Line 66 | Called by:** External code
- Returns list of `{key, name, faculty_count}` for all departments

#### `get_department_faculty(self, dept_key, limit=None) → List[Dict]`
**Line 79 | Called by:** `QueryRouter._route_faculty()`
- Case-insensitive fuzzy match on department key
- Returns all faculty in that department (or up to limit)

#### `search(self, department=None, research_area=None, limit=None, scope="top_results") → List[Dict]`
**Line 109 | Called by:** `QueryRouter._route_faculty()`
- If department specified → get department faculty
- If research_area specified → `_filter_by_research()`
- Otherwise → all faculty
- Deduplicates by name (lowercased)
- Default limit: 10 for top_results, None for complete_list

#### `_filter_by_research(self, faculty: List[Dict], research_area: str) → List[Dict]`
**Line 276 | Called by:** `search()`
- Searches all faculty whose research interests or document text mentions the research area
- Ranks by **department affinity** — professors from departments that "own" the topic rank higher
- Example: for "machine learning", EECS professors rank above Business professors
- Falls back to simple text matching if no affinity mapping exists

#### `search_by_name(self, name, limit=5) → List[Dict]`
**Line 337 | Called by:** `QueryRouter._route_faculty()`
- Searches name, email, and department fields
- Case-insensitive partial matching
- Deduplicates by name

#### `format_for_context(self, faculty_list: List[Dict]) → str`
**Line 369 | Called by:** `QueryRouter._route_faculty()`
- Formats as "Professor: Name\nDepartment: ...\nEmail: ...\nOffice: ..."
- Adds "[Source: faculty_directory]" tag
- Limits research interests to first 5

#### `get_stats(self) → Dict`
**Line 394**
- Returns `{total_departments, total_faculty}`

---

## 10. `course_retriever.py` — Course JSON Search

**File:** `app/rag/course_retriever.py` (326 lines)
**Purpose:** Fast course search with typo correction and multi-field scoring. Sub-50ms.

### Class: `CourseRetriever`

#### `__init__(self, data_dir=None)`
**Line 20**
- Calls `_load_all_courses()`

#### `_load_all_courses(self)`
**Line 44 | Called by:** `__init__`
- Loads `courses_combined.json`
- Builds 3 indexes: `_by_subject`, `_by_level`, `_by_code`
- Creates `QueryPreprocessor` with valid subject codes from loaded data

#### `search(self, query, limit=20) → List[Dict]`
**Line 80 | Called by:** `QueryRouter._route_courses()`
- Preprocesses query via `QueryPreprocessor.preprocess()` (typo fix, synonym expansion)
- Tokenizes into query terms
- Scores every course via `_score_course()`
- Returns sorted by score descending

#### `_score_course(self, course, query_terms, full_query) → int`
**Line 135 | Called by:** `search()`
- Weighted scoring across fields:
  - `course_code` match: +10 per term
  - `title` match: +5 per term
  - `description` match: +2 per term
  - `subject` match: +3 per term
  - `prerequisites` match: +1 per term
  - Full query in title: +8 bonus
  - Full query in description: +3 bonus

#### `search_by_subject(self, subject, limit=50) → List[Dict]`
**Line 187 | Called by:** `QueryRouter._route_courses()`
- Direct lookup from `_by_subject` index, returns up to limit

#### `search_by_level(self, level, limit=50) → List[Dict]`
**Line 193 | Called by:** `QueryRouter._route_courses()`
- Direct lookup from `_by_level` index

#### `get_course(self, course_code) → Optional[Dict]`
**Line 199 | Called by:** `QueryRouter._route_courses()`
- Normalizes code (uppercase, single space), direct lookup from `_by_code`

#### `get_prerequisites(self, course_code) → Optional[str]`
**Line 211 | Called by:** External code
- Calls `get_course()`, returns prerequisites field

#### `format_for_context(self, courses: List[Dict]) → str`
**Line 218 | Called by:** `QueryRouter._route_courses()`
- Formats as "Course: CODE - Title\nCredits: ...\nLevel: ...\nDescription: ..."
- Caps at 15 courses, adds "[Source: course_catalog]" tags

#### `get_stats(self) → Dict`
**Line 245**
- Returns `{total_courses, total_subjects, subjects_list}`

---

## 11. `campus_retriever.py` — Dining/Transit/Housing/Tuition

**File:** `app/rag/campus_retriever.py` (396 lines)
**Purpose:** Fast JSON-based retriever for campus services. Sub-50ms.

### Class: `CampusRetriever`

#### `__init__(self, data_dir=None)`
**Line 25**
- Sets data directory, initializes empty `_cache` dict

#### `_load_json(self, filename) → Dict`
**Line 40 | Called by:** All search methods
- Loads JSON file from data directory
- Caches result in `_cache[filename]` to avoid reloading

#### `_partial_match(self, text, query) → bool`
**Line 55 | Called by:** All search methods
- Checks if any query term (split by space) appears in text (case-insensitive)

#### `search_dining(self, query=None, limit=10) → List[Dict]`
**Line 66 | Called by:** `QueryRouter._route_dining()`
- Loads `dining.json`, returns all locations if no query
- Filters by name, type, building, description matching

#### `get_all_dining(self) → List[Dict]`
**Line 82**

#### `format_dining_context(self, locations) → str`
**Line 87 | Called by:** `QueryRouter._route_dining()`
- Formats with name, building, type, hours (top 3 days)

#### `search_transit(self, query=None, limit=10) → List[Dict]`
**Line 109 | Called by:** `QueryRouter._route_transit()`
- Loads `transit.json`, filters by route name, number, description
- Special: if "ku" or "campus" in query → prioritizes KU routes (serves_ku/campus_only first)

#### `get_all_transit(self)` / `get_ku_transit(self) → List[Dict]`
**Lines 135, 140**
- get_ku_transit filters for serves_ku or campus_only routes

#### `format_transit_context(self, routes) → str`
**Line 146**

#### `search_housing(self, query=None, limit=10) → List[Dict]`
**Line 167 | Called by:** `QueryRouter._route_housing()`
- Loads `housing.json` — handles nested structure (residence_halls, scholarship_halls, apartments)
- Flattens into unified list with name, type, description, amenities, rates

#### `format_housing_context(self, housing) → str`
**Line 217**
- Includes room types and rates if available

#### `search_tuition(self, query=None, limit=10) → List[Dict]`
**Line 250 | Called by:** `QueryRouter._route_tuition()`
- Loads `tuition.json`, flattens nested tuition structure into searchable items
- Extracts base rates, fees, cost of attendance, payment info

#### `format_tuition_context(self, fees) → str`
**Line 287**
- Organizes by category (tuition, fees, cost, payment)

#### `search(self, data_type, query=None, limit=10) → Dict[str, Any]`
**Line 306**
- Generic dispatcher: routes to search_dining/transit/housing/tuition based on data_type string

---

## 12. `faculty_search.py` — ChromaDB Faculty Search

**File:** `app/rag/faculty_search.py` (220 lines)
**Purpose:** Semantic search across 2,207 KU faculty using ChromaDB with OpenAI embeddings. Used as fallback by Retriever.

### Class: `FacultySearcher`

#### `__init__(self, data_dir=None)`
**Line 32**
- Calls `_connect()`

#### `_connect(self)`
**Line 44 | Called by:** `__init__`
- Connects to ChromaDB at `data/vectordb/`
- Gets "faculty" collection with OpenAI `text-embedding-3-small` embedding function

#### `search(self, query, top_k=5, department_filter=None) → List[Dict]`
**Line 60 | Called by:** `Retriever.search_faculty_enhanced()`
- If department_filter → fetches 10x results and post-filters
- Queries ChromaDB with the query text
- Converts distance to similarity: `score = 1 / (1 + distance)`
- Returns list of `{id, name, department, email, office, phone, building, profile_url, score, document}`

#### `get_faculty_by_name(self, name) → Optional[Dict]`
**Line 118**
- Searches for name, returns top result only if name appears in result

#### `get_department_faculty(self, department, limit=50) → List[Dict]`
**Line 133**
- Queries ChromaDB with department name, filters results by department field

#### `stats(self) → Dict`
**Line 164**
- Returns `{total_faculty, collection_name, storage, embeddings}`

---

## 13. `context_builder.py` — Context Compression

**File:** `app/rag/context_builder.py` (228 lines)
**Purpose:** Compresses retrieved results into optimized context for the LLM — only relevant fields, with source citations.

### Class: `ContextBuilder`

#### `__init__(self, max_chars=4000)`
**Line 18**

#### `build(self, query, route_result) → str`
**Line 21 | Called by:** `BabyJayChat.ask()`
- Gets results, source, intent from route_result
- Determines relevant fields based on query + intent
- Compresses each result, deduplicates by name
- Joins blocks, truncates at max_chars
- Returns formatted context string

#### `_get_relevant_fields(self, query, intent) → set`
**Line 66 | Called by:** `build()`
- Always includes: name, title, department
- Faculty: adds email, department, research. If "office"/"where" in query → adds office, building, phone
- Course: adds course_code, credits, description. If "prerequisite" in query → adds prerequisites
- Dining: building, type, hours
- Transit: route_number, description, stops
- Housing: type, description, amenities
- Financial: amount, description, requirements

#### `_get_name(self, result, intent) → Optional[str]`
**Line 109 | Called by:** `build()`
- Extracts identifying name from result (name, course_code, title, or route_name)

#### `_compress_result(self, result, intent, relevant_fields, source) → str`
**Line 116 | Called by:** `build()`
- Dispatches to intent-specific formatter

#### `_format_faculty(self, r, fields, source) → str`
**Line 133**
- "[Source: faculty_retriever]\nProfessor: ...\nDepartment: ..."
- Only includes fields that are in `relevant_fields` AND have values

#### `_format_course(self, r, fields, source) → str`
**Line 159**
- "Course: CODE - Title\nCredits: ...\nDescription: ..." (truncated to 200 chars)

#### `_format_dining(self, r, fields, source) → str`
**Line 178**

#### `_format_transit(self, r, fields, source) → str`
**Line 194**

#### `_format_housing(self, r, fields, source) → str`
**Line 203**

#### `_format_generic(self, r, source) → str`
**Line 212 | Called by:** `_compress_result()` for general/unknown intents
- If result has "content" key → uses it (truncated to 400 chars)
- Otherwise formats name, title, description fields

---

## 14. `reranker.py` — LLM Re-ranking

**File:** `app/rag/reranker.py` (99 lines)
**Purpose:** Re-scores retrieved results using Claude Haiku as a cross-encoder. Dramatically improves which result is #1.

### Class: `Reranker`

#### `__init__(self)`
**Line 28**
- Creates Anthropic client

#### `rerank(self, query, results, top_k=5) → List[Dict]`
**Line 31 | Called by:** `QueryRouter._route_vector_fallback()` (when > 3 results)
- Skips if ≤ 3 results (not worth the API call)
- Caps at 15 documents for scoring
- Truncates each document to 300 chars
- Calls Haiku: "Score each document's relevance to the query from 0-10. Return JSON array."
- Parses scores from response, attaches `rerank_score` to each result
- Sorts by rerank_score descending, returns top_k
- **On any error → returns original order** (safe fallback)

---

## 15. `rlhf_optimizer.py` — Feedback Learning

**File:** `app/rag/rlhf_optimizer.py` (442 lines)
**Purpose:** Learns from user feedback (thumbs up/down) to improve responses. Lightweight RLHF without fine-tuning.

### Class: `RLHFOptimizer`

#### `__init__(self, cache_ttl=300, debug=False)`
**Line 48**
- Connects to Supabase (if available)
- `cache_ttl` — how often to refresh patterns (default 5 minutes)
- Defines `query_types` dict for classifying queries into categories

#### `_classify_query(self, query) → str`
**Line 76 | Called by:** `enhance_prompt()`, `get_query_guidance()`
- Simple keyword matching to classify query type (course_info, faculty_search, live_lookup, campus_info, general)

#### `_fetch_feedback(self, limit=500) → List[Dict]`
**Line 86 | Called by:** `_analyze_patterns()`
- Fetches last 30 days of feedback from Supabase `feedback` table

#### `_analyze_patterns(self) → Dict[str, Any]`
**Line 108 | Called by:** `enhance_prompt()`, `get_query_guidance()`
- Checks cache (5-minute TTL)
- Groups feedback by query type, calculates approval rates
- Identifies problem areas (approval < 70% with ≥ 3 samples)
- Calls `_extract_lessons()` and `_extract_success_patterns()`
- Returns `{lessons, problem_queries, success_patterns, total_feedback, overall_approval}`

#### `_extract_lessons(self, feedback) → List[str]`
**Line 182 | Called by:** `_analyze_patterns()`
- Analyzes negative feedback for common patterns:
  - Wrong entity type, missing information, too generic, outdated, format issues
- Each pattern needs ≥ 2 instances to trigger a lesson
- Also analyzes response length correlation (too long vs too short)

#### `_extract_success_patterns(self, feedback) → List[str]`
**Line 232 | Called by:** `_analyze_patterns()`
- Analyzes positive feedback for common elements:
  - Specific numbers (>70% of positive responses)
  - Professor name mentions (>50%)
  - Optimal response length (100-500 chars)

#### `_calculate_approval(self, feedback) → float`
**Line 258 | Called by:** `_analyze_patterns()`
- `positive_count / total * 100`

#### `enhance_prompt(self, base_prompt, query) → str`
**Line 266 | Called by:** `BabyJayChat.ask()`
- **Main integration point.** Called before every LLM call.
- Gets learned patterns, appends to system prompt:
  - "LEARNED FROM USER FEEDBACK:" section with up to 5 lessons
  - "WHAT USERS APPRECIATE:" section with up to 3 success patterns
  - Warning if this query type has low approval rate
- Returns enhanced prompt (or original if no patterns)

#### `get_query_guidance(self, query) → Optional[str]`
**Line 311 | Called by:** `integrate_rlhf_with_chat()` helper
- Returns specific guidance for similar past failures

#### `log_response(self, query, response, query_type=None)`
**Line 335**
- Debug logging of response for analysis

#### `get_stats(self) → Dict[str, Any]`
**Line 349**
- Returns system stats: total feedback, approval rate, lessons count, etc.

### Standalone Function

#### `integrate_rlhf_with_chat(chat_instance, optimizer=None) → chat_instance`
**Line 368**
- Monkey-patches `chat.ask()` to add RLHF guidance and logging
- Not currently used in production (enhancement is done directly in `BabyJayChat.ask()`)

---

## 16. `embeddings.py` — Data → ChromaDB Loader

**File:** `app/rag/embeddings.py` (900+ lines)
**Purpose:** One-time script to load ALL data (dining, transit, courses, buildings, offices, professors, admissions, calendar, FAQs, tuition, financial aid, housing) into ChromaDB.

### Helper Functions

| Function | Line | Description |
|----------|------|-------------|
| `get_project_root()` | 28 | Returns project root Path |
| `load_json_file(filepath)` | 33 | Load and return JSON file contents |
| `format_hours(hours)` | 39 | Format hours dict to readable string |

### Document Preparation Functions
Each returns `(documents: List[str], metadatas: List[Dict], ids: List[str])`:

| Function | Line | Source Type | Data File |
|----------|------|-------------|-----------|
| `prepare_dining_documents(data)` | 52 | "dining" | dining.json |
| `prepare_transit_documents(data)` | 86 | "transit" | transit.json |
| `prepare_course_documents(data)` | 123 | "course" | courses.json |
| `prepare_building_documents(data)` | 163 | "building" | buildings.json |
| `prepare_office_documents(data)` | 200 | "office" | offices.json |
| `prepare_professor_documents(data)` | 240 | "professor" | professors.json |
| `prepare_admission_documents(data)` | 282 | "admission" | admissions.json |
| `prepare_calendar_documents(data)` | 365 | "calendar" | calendar.json |
| `prepare_faq_documents(data)` | 475 | "faq" | faqs.json |
| `prepare_tuition_documents(data)` | 553 | "tuition" | tuition.json |
| `prepare_financial_aid_documents(data)` | 719 | "financial_aid" | financial_aid.json |
| `prepare_housing_documents(data)` | 893 | "housing" | housing.json |

Each function:
1. Reads the specific data format from JSON
2. Builds a human-readable document string for embedding (e.g., "Dining Location: Mrs. E's\nType: dining hall\nBuilding: ...")
3. Extracts metadata fields for filtering
4. Generates unique IDs (e.g., "dining_1", "course_EECS_168")

The admission/calendar/tuition/financial_aid functions also handle nested sub-documents (deadlines, refund schedules, graduation dates, scholarship tiers, etc.)

### `initialize_database()` (not shown but referenced)
- Calls all prepare functions, loads results into ChromaDB `babyjay_knowledge` collection
- Run once during setup

---

## 17. `openai_embeddings.py` — Embedding Function

**File:** `app/rag/openai_embeddings.py` (42 lines)
**Purpose:** Custom embedding function compatible with ChromaDB.

### Class: `OpenAIEmbeddingFunction`

#### `__init__(self, model="text-embedding-3-small")`
**Line 17**

#### `__call__(self, input: list[str]) → list[list[float]]`
**Line 19 | Called by:** ChromaDB internally when adding/querying documents
- Calls OpenAI API `client.embeddings.create()`
- Returns list of embedding vectors

### Module-Level
| Name | Description |
|------|-------------|
| `openai_ef` | Singleton instance of OpenAIEmbeddingFunction |
| `get_embedding_function()` | Returns the singleton |

---

## 18. `regenerate_faculty_embeddings.py` — Faculty Re-embed Script

**File:** `app/rag/regenerate_faculty_embeddings.py` (154 lines)
**Purpose:** One-time script to re-embed all 2,207 faculty members with OpenAI embeddings.

### `main()`
**Line 35**
1. Loads `faculty_documents.json` (2,207 faculty)
2. Connects to ChromaDB
3. Deletes old "faculty" collection
4. Creates new collection with OpenAI `text-embedding-3-small`
5. Prepares documents using `searchable_text` field
6. Adds in batches of 100 (to avoid API rate limits)
7. Verifies collection count
8. Runs test searches ("machine learning", "deep learning", etc.)

---

## Quick Reference: What Calls What

```
User Message (HTTP)
  └─ api_routes.chat()
       └─ _get_or_create_chat() → BabyJayChat instance
       └─ BabyJayChat.ask(message)
            ├─ _is_simple_followup()     ← checked FIRST
            ├─ _is_greeting()            ← skipped if follow-up
            ├─ _is_about_bot()           ← skipped if follow-up
            ├─ _is_off_topic()           ← skipped if follow-up
            ├─ _is_ambiguous()
            ├─ _clean_query_hybrid()
            │    └─ QueryPreprocessor.preprocess()
            ├─ _expand_followup_question()  ← calls Haiku
            ├─ QueryRouter.route()
            │    ├─ QueryDecomposer.should_decompose() / decompose()
            │    ├─ QueryClassifier.classify()
            │    │    ├─ _detect_intent_regex()
            │    │    ├─ _extract_*_entities()
            │    │    └─ _classify_with_llm()  ← if low confidence
            │    ├─ _route_faculty()
            │    │    └─ FacultyRetriever.search() / search_by_name()
            │    ├─ _route_courses()
            │    │    └─ CourseRetriever.search() / get_course()
            │    ├─ _route_dining/housing/transit/tuition()
            │    │    └─ CampusRetriever.search_*()
            │    └─ _route_vector_fallback()
            │         ├─ Retriever.smart_search()
            │         │    ├─ search() → ChromaDB + BM25 hybrid
            │         │    │    └─ hybrid_merge() (RRF)
            │         │    └─ search_faculty_enhanced()
            │         │         └─ FacultySearcher.search() → ChromaDB
            │         └─ Reranker.rerank()  ← calls Haiku
            ├─ ContextBuilder.build()
            ├─ RLHFOptimizer.enhance_prompt()
            └─ _call_haiku()  ← final LLM call
```

---

## Data Files

| File | Contents | Used By |
|------|----------|---------|
| `data/faculty_combined.json` | All faculty by department | FacultyRetriever |
| `data/courses_combined.json` | All courses | CourseRetriever |
| `data/dining.json` | Dining locations | CampusRetriever |
| `data/transit.json` | Bus routes | CampusRetriever |
| `data/housing.json` | Housing options | CampusRetriever |
| `data/tuition.json` | Tuition & fees | CampusRetriever |
| `data/vectordb/` | ChromaDB persistent storage | Retriever, FacultySearcher |
| `data/faculty_documents.json` | Faculty with searchable_text | regenerate_faculty_embeddings |

---

*Generated by Claude Code. Covers all 18 Python files in app/rag/ and app/routers/.*
