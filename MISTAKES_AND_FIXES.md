# BabyJay — Mistakes & Fixes Log

Every response-quality issue found and fixed, organized by category.

---

## HALLUCINATION PREVENTION

### Fix 1: System Prompt Rule 2 — Hallucination Backdoor
**File:** `app/rag/chat.py` (SYSTEM_PROMPT)
**Mistake:** Rule 2 said "If NO context is provided but you're confident about stable KU info, you may answer." This let GPT fabricate KU-specific details (wrong office hours, made-up phone numbers, outdated policies) from its training data.
**Fix:** Changed to: "If NO context is provided, say you don't have that specific information right now and suggest checking ku.edu. Do NOT guess or answer from general knowledge."
**Corner case covered:** User asks "What's the admissions office phone number?" with no context → old behavior: GPT guesses a number. New behavior: directs to ku.edu.

### Fix 2: No Off-Topic Gate Before Pipeline
**File:** `app/rag/chat.py` — new `_is_off_topic()` method + check in `ask()`
**Mistake:** Non-KU questions like "What's the capital of France?" ran through the full pipeline (preprocessor → classifier → vector search → LLM), wasting API calls and sometimes returning irrelevant KU data as "context."
**Fix:** Added `_is_off_topic()` with regex patterns for clearly non-KU queries (recipes, movies, stocks, math problems, trivia, etc.) that short-circuits before the pipeline. KU-related keywords bypass the filter.
**Corner cases covered:**
- "Write me a poem" → blocked (off-topic)
- "Write me a poem about KU" → allowed (contains "KU")
- "What GPA do I need?" → allowed (contains "GPA" which is student-related)
- "What's the weather in Kansas?" → blocked (weather, not KU-specific)
- "How do I write a for loop?" → blocked (general coding)
- "How do I write a for loop in EECS 168?" → allowed (contains "EECS")

### Fix 3: Low-Relevance Vector Results Treated as Truth
**File:** `app/rag/retriever.py` — `search()` method
**Mistake:** ChromaDB returned results with no minimum relevance threshold. A query about "best pizza in New York" could return KU dining results with 0.2 similarity, which got injected as context. The LLM saw "Here's information from KU's database:" and tried to answer using irrelevant data.
**Fix:** Added `min_relevance=0.25` parameter. Results below the threshold are filtered out before they reach the LLM.
**Corner case covered:** "Tell me about MIT courses" → vector search returns KU courses with low similarity → filtered out → LLM correctly says "I don't have that info."

### Fix 4: "Suggest Where They Might Find It" Encourages Fabrication
**File:** `app/rag/chat.py` — no-context user message
**Mistake:** When no context was found, the prompt told the LLM: "Suggest where they might find it." The LLM would then make up office names, URLs, phone numbers, and email addresses.
**Fix:** Changed to: "Let the user know you don't have that info and suggest they check ku.edu or contact the relevant KU office. Do NOT guess or make up any specific details like names, numbers, URLs, or office locations."
**Corner case covered:** "What's the pharmacy school admission deadline?" with no context → old behavior: GPT invents "March 15th" and a fake URL. New behavior: says to check ku.edu.

### Fix 5: Live Course Data Formatted at High Temperature
**File:** `app/rag/chat.py` — `_handle_live_course_query()`
**Mistake:** Real-time factual data (seat counts, instructor names, schedules) was passed through GPT at `temperature=0.8`. The model could paraphrase numbers wrong: "14 seats" → "around a dozen seats" or swap instructor names between sections.
**Fix:** Lowered temperature from 0.8 to 0.4 for live course formatting.
**Corner case covered:** "How many seats in EECS 168?" → raw data says 14 → at temp 0.8 model might say "about 15" → at temp 0.4 it sticks to "14."

### Fix 6: Context Confidence Signal Missing
**File:** `app/rag/chat.py` — context injection prompt
**Mistake:** Context was injected as "Here's information from KU's database:" with no relevance indication. The LLM treated a 0.3 similarity match the same as a 0.95 match.
**Fix:** Changed prompt to: "use ONLY this to answer — do not add anything beyond what is here" and added "If the context doesn't answer their question, say so."
**Corner case covered:** User asks about parking, retriever returns vaguely related transit data → LLM now says "I don't have specific parking info" instead of trying to answer from transit context.

### Fix 7: New System Prompt Rules Against Fabrication
**File:** `app/rag/chat.py` (SYSTEM_PROMPT)
**Mistake:** No explicit rules against fabricating specific data types.
**Fix:** Added Rules 9 and 10:
- Rule 9: "If the context seems unrelated to the user's question, ignore the context"
- Rule 10: "NEVER fabricate specific numbers (GPA requirements, acceptance rates, tuition amounts, scholarship values) unless they appear in the provided context"
**Corner cases covered:**
- "What GPA do I need for EECS?" → no GPA data in context → won't guess "3.0"
- "How much is a scholarship worth?" → no amount in context → won't invent "$5,000"

---

## CONVERSATION MEMORY

### Fix 8: Greetings & About-Bot Not Saved to History
**File:** `app/rag/chat.py` — `ask()` method
**Mistake:** When user said "hi" or "who are you?", the response was returned without calling `_save_message()`. These exchanges disappeared from conversation history, so the LLM had no memory of them.
**Fix:** Added `_save_message()` calls for both greeting and about-bot responses.
**Corner case covered:** User says "hi" → bot responds → user says "what did I just say?" → old behavior: no memory of greeting. New behavior: greeting is in history.

### Fix 9: New BabyJayChat Instance Per Request — All State Lost
**File:** `app/routers/api_routes.py`
**Mistake:** Every API call created a fresh `BabyJayChat()`. Instance-level state (`last_mentioned_course`, `waiting_for_clarification`, `active_department_filter`, `last_search_query`) was lost between requests. Follow-ups like "what about that course?" never worked in production.
**Fix:** Added `_chat_instances` dict keyed by `conversation_id` with `_get_or_create_chat()` function. Instances persist across requests (up to 200, with FIFO eviction). State is preserved so follow-ups, clarifications, and course references work.
**Corner cases covered:**
- "Tell me about EECS 168" → "what about the instructor?" → now works (last_mentioned_course preserved)
- "professors" → bot asks clarification → "EECS" → now correctly combines (clarification state preserved)
- "ML professors" → "only EECS" → department filter now works (active_department_filter preserved)

### Fix 10: Double Message Saving
**File:** `app/routers/api_routes.py`
**Mistake:** The API route saved messages to Supabase (lines 73, 96), AND `chat.ask()` with `use_history=True` also saved them internally via `_save_message()`. Messages were duplicated in both the database and the internal history.
**Fix:** Call `chat.ask()` with `use_history=False` and manage history manually in the route. Messages are saved once to Supabase (source of truth) and once to the instance's internal history (for LLM context).
**Corner case covered:** Long conversation → old behavior: each message appears twice in DB → doubles storage and confuses history loading.

### Fix 11: Stale `recent_context` Leaks Between Topics
**File:** `app/rag/chat.py`
**Mistake:** `recent_context` was never cleared when the user changed topics. If user asked about dining, then asked "what about parking?", the follow-up fallback would serve old dining context.
**Fix:** When a new (non-follow-up) query arrives, `recent_context` is updated to the new context or cleared if no results found. Old context only used for actual follow-ups.
**Corner case covered:** "Where can I eat?" → "Tell me about the bus" → old behavior: dining data bleeds into transit answer. New behavior: transit gets its own retrieval.

### Fix 12: Conversation History Uncapped — Token Overflow
**File:** `app/rag/chat.py`
**Mistake:** Last 10 messages sent as history with no token counting. 10 long messages = 10K+ tokens eating into context window and inflating cost.
**Fix:** Reduced to last 6 messages. Combined with context cap (4000 chars), total prompt stays under reasonable token limits.
**Corner case covered:** User in a 50-message conversation about multiple topics → old behavior: 10 messages + massive context = potential token overflow or truncation. New behavior: capped and manageable.

---

## CLASSIFIER & ROUTING

### Fix 13: Triplicated Regex Patterns Inflate Scores
**File:** `app/rag/classifier.py`
**Mistake:** The pattern `r"\b(learning|programming|calculus|physics|chemistry|biology|engineering)\b"` appeared 3 times in `course_info`. A query with "learning" got score +3 instead of +1, making `course_info` beat `faculty_search` even for "machine learning professors."
**Fix:** Removed the 2 duplicate lines. Also removed "learning" from the pattern since it's ambiguous (could be course or faculty research).
**Corner cases covered:**
- "machine learning professors" → old: course_info (score 3) beats faculty_search (score 1). New: faculty_search wins correctly.
- "deep learning research at KU" → old: misrouted to courses. New: correctly routes to faculty.
- "programming courses" → still correctly routes to course_info (has "courses" keyword).

### Fix 14: Double Preprocessing Wastes Latency
**Files:** `app/rag/chat.py` + `app/rag/router.py`
**Mistake:** `ask()` preprocessed the query, then passed it to `router.route()` which created a new `QueryPreprocessor()` and preprocessed again. Double computation, and potentially conflicting corrections.
**Fix:** Removed preprocessing from `router.py` since `chat.py` already handles it. Router now classifies the already-preprocessed query.
**Corner case covered:** Query "machien lerning" → old: corrected to "machine learning" twice with two QueryPreprocessor instances. New: corrected once.

### Fix 15: QueryPreprocessor Created Without Subject Codes
**Files:** `app/rag/chat.py` + `app/rag/router.py`
**Mistake:** `QueryPreprocessor()` was instantiated with no `valid_subject_codes`. So `_detect_codes()` couldn't recognize subject codes, and `_fuzzy_match_subject_code()` always returned `None`. Subject code protection was effectively disabled.
**Fix:** Removed the redundant preprocessor in router.py. The preprocessor in `_clean_query_hybrid()` still handles corrections. Subject codes are now handled by the classifier which has its own `subject_codes` set.
**Corner case covered:** "EESC courses" → old: preprocessor can't recognize it as near "EECS" (empty subject codes). Now handled by classifier's own subject code matching.

### Fix 16: Synonym Expansion Breaks Valid Queries
**File:** `app/rag/query_preprocessor.py`
**Mistake:** Synonyms like `"cs" → "computer science"`, `"me" → "mechanical engineering"`, `"ce" → "civil engineering"`, `"bio" → "biology"`, `"chem" → "chemistry"`, `"phys" → "physics"`, `"class" → "course"` were always applied. So "CS 101" could become "computer science 101" and fail course code matching. "ME 301" → "mechanical engineering 301". And `"class"/"classes"` replacement was unnecessary noise.
**Fix:** Removed subject-code-like abbreviations that conflict with actual course codes (cs, me, ce, bio, chem, phys, stat, psych, econ). Removed class→course mapping. Kept non-ambiguous abbreviations (ml, ai, dl, nlp, cv, ds, calc, orgo, polisci).
**Corner cases covered:**
- "CS 101" → old: "computer science 101" (broken). New: stays "CS 101"
- "ME 301 prerequisites" → old: "mechanical engineering 301 prerequisites". New: stays "ME 301"
- "What ML courses are there?" → still correctly expands to "machine learning"
- "I need to take a bio class" → "bio" no longer expands (could be BIOL subject code)

---

## RETRIEVER BUGS

### Fix 17: Faculty Fallback Iterates Wrong Variable
**File:** `app/rag/retriever.py` — `smart_search()` method
**Mistake:** In the faculty fallback block, after calling `search_faculty_enhanced()` into `fallback_results`, the code iterated over `merged` (the original empty results) instead of `fallback_results`. The fallback always returned empty.
**Fix:** Changed `for r in merged:` to `for r in fallback_results:`.
**Corner case covered:** "Physics professors doing ML" → department filter "Physics" returns 0 results → fallback tries all departments → old: iterates empty `merged`, returns nothing. New: iterates `fallback_results`, returns ML professors from any department.

---

## RESPONSE QUALITY

### Fix 18: LLM Error Returns Generic Message Instead of Raw Context
**File:** `app/rag/chat.py` — end of `ask()` method
**Mistake:** If the final OpenAI call failed, the user got "Sorry, I encountered an error. Please try again!" even though retrieved context was available.
**Fix:** If context exists, return it directly: "I found some information but had trouble formatting it. Here's what I have:" followed by truncated raw context.
**Corner case covered:** OpenAI rate limit or timeout → user still gets useful data instead of nothing.

### Fix 19: "KU Retry" Corrupts Non-KU Queries
**File:** `app/rag/chat.py`
**Mistake:** When no context was found, query was retried as `"KU {query}"`. So "what's the weather?" became "KU what's the weather?" — a nonsensical search wasting an API call.
**Fix:** KU retry only triggers if: (a) the query isn't off-topic, and (b) the query doesn't already contain "ku" or "kansas."
**Corner case covered:** "KU dining options" → old: retried as "KU KU dining options". New: no redundant retry.

### Fix 20: Large Context Sent Without Truncation
**File:** `app/rag/chat.py`
**Mistake:** "List all EECS courses" could return thousands of lines of context, all stuffed into one LLM message. Could cause token overflow, high cost, or confused responses.
**Fix:** Context capped at 4000 characters with a truncation notice appended.
**Corner case covered:** "Show me all courses in the catalog" → old: 50K chars sent to LLM. New: 4000 chars + "[...additional results truncated]".

### Fix 21: No Input Validation on Chat Endpoint
**File:** `app/routers/api_routes.py`
**Mistake:** No max message length — users could send 1MB+ messages directly to OpenAI.
**Fix:** Added `Field(..., min_length=1, max_length=5000)` to `ChatRequest.message`.
**Corner case covered:** Bot/spam sends massive payload → old: crashes or huge API bill. New: 422 validation error.

### Fix 22: No Error Handling in Chat Endpoint
**File:** `app/routers/api_routes.py`
**Mistake:** The entire chat endpoint had zero try-except. Any failure (DB down, OpenAI timeout, etc.) returned a raw 500 error with a Python traceback.
**Fix:** Wrapped endpoint in try-except. HTTPExceptions re-raised as-is, other errors return a clean 500 with message.
**Corner case covered:** Supabase connection drops mid-request → old: ugly traceback. New: clean error response.

---

## SUMMARY TABLE

| #  | Category            | File                        | Severity | Status |
|----|---------------------|-----------------------------|----------|--------|
| 1  | Hallucination       | chat.py (SYSTEM_PROMPT)     | CRITICAL | FIXED  |
| 2  | Hallucination       | chat.py (_is_off_topic)     | HIGH     | FIXED  |
| 3  | Hallucination       | retriever.py (min_relevance)| HIGH     | FIXED  |
| 4  | Hallucination       | chat.py (no-context prompt) | HIGH     | FIXED  |
| 5  | Hallucination       | chat.py (temperature)       | MEDIUM   | FIXED  |
| 6  | Hallucination       | chat.py (context signal)    | MEDIUM   | FIXED  |
| 7  | Hallucination       | chat.py (new rules 9-10)    | MEDIUM   | FIXED  |
| 8  | Memory              | chat.py (greeting save)     | HIGH     | FIXED  |
| 9  | Memory/State        | api_routes.py (instances)   | CRITICAL | FIXED  |
| 10 | Memory              | api_routes.py (double save) | HIGH     | FIXED  |
| 11 | Memory              | chat.py (stale context)     | MEDIUM   | FIXED  |
| 12 | Memory              | chat.py (history cap)       | MEDIUM   | FIXED  |
| 13 | Classifier          | classifier.py (triplicate)  | HIGH     | FIXED  |
| 14 | Performance         | router.py (double preproc)  | MEDIUM   | FIXED  |
| 15 | Classifier          | router.py (empty codes)     | MEDIUM   | FIXED  |
| 16 | Preprocessing       | query_preprocessor.py       | HIGH     | FIXED  |
| 17 | Retriever           | retriever.py (fallback bug) | HIGH     | FIXED  |
| 18 | Response            | chat.py (error fallback)    | MEDIUM   | FIXED  |
| 19 | Response            | chat.py (KU retry)          | LOW      | FIXED  |
| 20 | Response            | chat.py (context cap)       | MEDIUM   | FIXED  |
| 21 | Validation          | api_routes.py (input)       | HIGH     | FIXED  |
| 22 | Error Handling      | api_routes.py (try-except)  | HIGH     | FIXED  |
| 23 | LLM Migration       | chat.py, classifier.py      | HIGH     | FIXED  |
| 24 | Ranking             | faculty_retriever.py        | HIGH     | FIXED  |
| 25 | Retrieval           | retriever.py, bm25_scorer   | HIGH     | FIXED  |
| 26 | Retrieval           | router.py, query_decomposer | HIGH     | FIXED  |
| 27 | Follow-up/Greeting  | chat.py (_is_greeting, ask) | CRITICAL | FIXED  |

---

## FILES MODIFIED

| File | Changes |
|------|---------|
| `app/rag/chat.py` | System prompt rewritten, off-topic gate, greeting/about history saving, context cap, relevance signal, stale context fix, KU retry fix, LLM error fallback, temperature fix, history cap, removed double preprocessing |
| `app/rag/classifier.py` | Removed triplicated regex, removed ambiguous "learning" keyword from course_info |
| `app/rag/retriever.py` | Added min_relevance threshold, fixed fallback variable bug |
| `app/rag/router.py` | Removed redundant preprocessing |
| `app/rag/query_preprocessor.py` | Removed synonym expansions that break subject codes |
| `app/routers/api_routes.py` | Persistent chat instances, fixed double saving, added input validation, added error handling |
| `app/rag/bm25_scorer.py` | **NEW** — BM25 keyword scorer + hybrid merge (RRF) for hybrid search |
| `app/rag/query_decomposer.py` | **NEW** — Decomposes multi-part questions into sub-queries |
| `app/rag/reranker.py` | **NEW** — LLM-based re-ranking using Claude Haiku as cross-encoder |
| `app/rag/context_builder.py` | **NEW** — Context compression with source citations and query-aware field selection |
| `app/rag/faculty_retriever.py` | Department affinity ranking for all departments |

---

## LLM MIGRATION: OpenAI GPT-4o-mini → Claude Haiku 4.5

### Fix 23: Migrated All Chat/Generation Calls to Claude Haiku
**Files:** `app/rag/chat.py`, `app/rag/classifier.py`, `requirements.txt`, `.env`
**What changed:**
- Replaced `openai.OpenAI` client with `anthropic.Anthropic` client in chat.py and classifier.py
- All 6 `client.chat.completions.create()` calls converted to `client.messages.create()` via `_call_haiku()` helper
- Model: `claude-haiku-4-5-20251001`
- Added `anthropic>=0.84.0` to requirements.txt
- Added `ANTHROPIC_API_KEY` placeholder to .env

**Calls migrated:**
1. Main response (`ask()`) — temperature 0.5
2. Greeting response (`_generate_greeting_response()`) — temperature 0.9
3. About bot response (`_generate_about_response()`) — temperature 0.8
4. Query cleaning (`_clean_query_hybrid()`) — temperature 0
5. Follow-up expansion (`_expand_followup_question()`) — temperature 0
6. Live course formatting (`_handle_live_course_query()`) — temperature 0.4
7. Classifier LLM fallback (`_classify_with_llm()`) — temperature 0

**What stayed on OpenAI:**
- ChromaDB embeddings (`text-embedding-3-small`) — re-embedding all data is a separate migration
- OpenAI SDK stays in requirements.txt for embeddings

**Helper function added:** `_call_haiku()` in chat.py
- Converts OpenAI-style messages to Anthropic format
- Handles role alternation (merges consecutive same-role messages)
- Ensures conversation starts with user message
- Single place to change model/params in the future

---

## DEPARTMENT AFFINITY RANKING

### Fix 24: Faculty Results Not Ranked by Department Relevance
**File:** `app/rag/faculty_retriever.py` — `_filter_by_research()`
**Mistake:** When searching for "ML professors" (no department specified), results came back in file order — Physics professors appeared before EECS professors. The system prompt told the LLM to "prefer EECS for technical topics" but the retriever returned Physics results first, so the LLM often just used whatever came first.
**Fix:** Added `TOPIC_DEPARTMENT_AFFINITY` mapping covering all major departments and research areas. When a topic matches (e.g., "machine learning" → EECS), professors from the preferred department appear first in results. Professors from other departments still appear — just ranked below. This covers:
- CS/Engineering: ML, AI, robotics, cybersecurity, etc. → EECS first
- Physics: quantum, astrophysics, particle physics → Physics first
- Chemistry: organic, computational, materials → Chemistry first
- Biology: genetics, ecology, neuroscience → Biology first
- Math: statistics, probability, algebra → Math first
- Business: finance, marketing, supply chain → Business first
- Psychology: cognitive, behavioral, clinical → Psychology first
- Law, Humanities, Health, and all other departments mapped

**Corner cases covered:**
- "ML professors" → EECS professors first, then Physics/Business professors who also do ML
- "quantum computing researchers" → Physics AND EECS both preferred (cross-disciplinary)
- "bioinformatics" → Molecular Biosciences, EECS, and Biology all preferred
- "robotics" → EECS and Mechanical Engineering both preferred
- "sociology research" → Sociology department first
- Topic with no affinity mapping → no boost, all departments equal (same as before)

---

## RAG ARCHITECTURE UPGRADES

### Fix 25: Vector-Only Search Misses Exact Keyword Matches
**Files:** `app/rag/retriever.py`, `app/rag/bm25_scorer.py` (NEW)
**Mistake:** The fallback search path used only ChromaDB vector search. Vector search finds semantically similar content but can miss exact keyword matches. For example, searching "EECS 678 operating systems" might rank a general CS document higher than the exact EECS 678 course page, because the embedding is semantically close to many CS topics.
**Fix:** Added hybrid search (BM25 + vector) using Reciprocal Rank Fusion (RRF):
- `BM25Scorer`: Lightweight keyword scorer that indexes documents and scores by term frequency / inverse document frequency
- `hybrid_merge()`: Merges vector and BM25 results using RRF (rank-based, not score-based — avoids scale mismatch)
- BM25 indices are cached per source filter to avoid re-indexing on every query
- Default weights: 60% vector, 40% BM25 — semantic similarity matters more, but exact matches get a boost
**Corner cases covered:**
- "EECS 678" → BM25 boosts exact match, vector finds semantically related courses → hybrid returns EECS 678 first
- "dining Mrs. E's" → vector might miss the apostrophe, BM25 catches the exact name
- Misspelled queries → vector still handles these (BM25 won't match, but doesn't hurt either)
- Empty BM25 results → falls back to vector-only (no degradation)

### Fix 26: Complex Multi-Part Questions Get Mediocre Results
**Files:** `app/rag/router.py`, `app/rag/query_decomposer.py` (NEW)
**Mistake:** A query like "Compare EECS 168 and EECS 268 prerequisites" was treated as a single search. Vector search on the full string returned results that partially matched both courses but didn't give complete info on either. Same problem with "prerequisites for EECS 168, EECS 268, and EECS 368."
**Fix:** Added `QueryDecomposer` that detects and splits multi-part queries:
- Detects comparison patterns: "Compare X and Y", "X vs Y", "difference between X and Y"
- Detects multiple course codes: "EECS 168 and EECS 268 prerequisites" → 2 sub-queries
- Detects list patterns: "prerequisites for X, Y, and Z" → 3 sub-queries
- Each sub-query is routed independently through the full pipeline
- Results are merged with deduplication and context separation (---dividers)
- Uses regex only (no LLM call) — adds zero latency for non-decomposable queries
**Corner cases covered:**
- "Compare EECS 168 and EECS 268" → decomposed into 2 queries, each gets full course info
- "Should I take MATH 125 or MATH 126?" → decomposed, both courses retrieved
- "prerequisites for EECS 168, 268, and 368" → 3 sub-queries
- "What is EECS 168?" → NOT decomposed (single entity, works fine as-is)
- Short queries like "EECS 168" → NOT decomposed (length check < 15 chars)

### Fix 27: Follow-Up Questions Misdetected as Greetings
**File:** `app/rag/chat.py` — `_is_greeting()` + `ask()` method
**Mistake:** Two bugs:
1. `_is_greeting()` used `q.startswith(greeting)` without word boundary checking. "his number?" starts with "hi" (a greeting), so it was treated as a greeting. Same problem with "history", "hire", "high five", etc.
2. In `ask()`, the greeting check ran BEFORE the follow-up check. Even if `_is_simple_followup()` would have correctly identified "his number?" as a follow-up (it contains "his"), it never got the chance to run.

**Fix:**
1. Added word boundary check in `_is_greeting()`: after the `startswith` match, verify the next character is a space, punctuation, or end-of-string. "his" no longer matches "hi".
2. Restructured `ask()` to check `_is_simple_followup()` FIRST when conversation history exists. Follow-ups now skip greeting, about-bot, and off-topic checks entirely.

**Corner cases covered:**
- "his number?" after asking about a professor → now correctly treated as follow-up
- "history of KU" → no longer caught as greeting ("hi")
- "hi there" → still correctly detected as greeting
- "hi" with no prior conversation → still greeting
- "thanks" after getting an answer → still greeting (no "his/her/their" follow-up indicators)
