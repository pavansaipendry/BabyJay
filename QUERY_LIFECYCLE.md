# BabyJay: What Happens When a User Sends a Query

**Every function, every decision, every file — from keystroke to response.**

---

## THE BIG PICTURE (30-second version)

```
User types "ML professors"
    ↓
[API Route] → receives HTTP POST, finds/creates chat instance
    ↓
[chat.ask()] → the brain — decides what to do with the query
    ↓
[Early exits] → Is it a greeting? About the bot? Off-topic? Follow-up?
    ↓
[Query cleaning] → Fix typos, expand abbreviations
    ↓
[Router] → Decompose complex queries, classify intent, pick a retriever
    ↓
[Retriever] → Fetch relevant data (JSON lookup OR hybrid vector+BM25 search)
    ↓
[Reranker] → Re-score results by relevance (LLM-based)
    ↓
[Context Builder] → Compress results, add source tags, remove irrelevant fields
    ↓
[LLM Call] → Claude Haiku generates a natural language response
    ↓
[Response] → Sent back to user
```

Now let's zoom into every single step.

---

## PHASE 0: THE HTTP REQUEST

**File:** `app/routers/api_routes.py`
**Function:** `chat()` (line 73)

When the user sends a message from the frontend, it hits:

```
POST /api/chat
Body: { "message": "ML professors", "conversation_id": "abc-123" }
```

### Step 0.1: Authentication
FastAPI calls `get_current_user()` to verify the JWT token. If invalid → 401 error. If valid → we know which user this is.

### Step 0.2: Conversation Setup
```python
if not conversation_id:
    # First message → create a new conversation in Supabase
    title = db.generate_title_from_message(request.message)
    conversation = db.create_conversation(user.id, title)
    conversation_id = conversation["id"]
else:
    # Continuing a conversation → verify it belongs to this user
    conversation = db.get_conversation(conversation_id, user.id)
```

### Step 0.3: Get or Create Chat Instance
```python
chat_instance = _get_or_create_chat(conversation_id)
```

**This is critical.** `_get_or_create_chat()` maintains a **dictionary of BabyJayChat instances** keyed by `conversation_id`. If the user already has a conversation going, we reuse the same instance — this preserves:
- `last_mentioned_course` — so "what about that course?" works
- `waiting_for_clarification` — so clarification follow-ups work
- `active_department_filter` — so "only EECS" works
- `recent_context` — so follow-ups can reference previous results

The cache holds up to 200 instances (FIFO eviction when full).

### Step 0.4: Load History from Database
```python
if not chat_instance._conversation_history:
    recent_messages = db.get_recent_messages(conversation_id, limit=20)
    for msg in recent_messages:
        chat_instance._conversation_history.append(...)
```

First request for a returning conversation? Load the last 20 messages from Supabase into the instance's memory so the LLM has context.

### Step 0.5: Call ask()
```python
response = chat_instance.ask(request.message, use_history=False)
```

**`use_history=False`** because the API route manages history itself (to avoid double-saving). The route manually appends messages after:

```python
chat_instance._conversation_history.append({"role": "user", "content": request.message})
chat_instance._conversation_history.append({"role": "assistant", "content": response})
db.add_message(conversation_id, "user", request.message)
db.add_message(conversation_id, "assistant", response)
```

Messages are saved in two places:
1. **Instance memory** (`_conversation_history`) — for the LLM to see in this session
2. **Supabase database** — permanent storage, survives server restarts

---

## PHASE 1: INSIDE `ask()` — THE DECISION TREE

**File:** `app/rag/chat.py`
**Function:** `ask()` (line 602)
**Class:** `BabyJayChat`

This is the brain. Every query enters here and gets routed through a decision tree.

### Step 1.0: Basic Validation
```python
if not question or not question.strip():
    return "I'd be happy to help! What would you like to know about KU?"

if not self._validate_query(question):
    return "I didn't quite understand that. Could you rephrase your question?"
```

`_validate_query()` just checks that the query has at least one alphanumeric character. Pure punctuation like "????" fails.

### Step 1.1: Follow-Up Detection (FIRST CHECK)
```python
_is_followup = self._conversation_history and self._is_simple_followup(question)
```

**Why this is first:** A follow-up like "his number?" starts with "hi", which would falsely match the greeting detector. So we check for follow-ups BEFORE greetings when there's conversation history.

**`_is_simple_followup()`** (line 424) returns True if:
- Query has ≤ 8 words, AND
- Contains pronouns/references: "his", "her", "their", "that", "this", "what about", "how about", "about it", "that course", etc.

If it IS a follow-up → **skip greeting, about-bot, and off-topic checks entirely.** Jump straight to the search pipeline.

### Step 1.2: Greeting Detection
```python
if self._is_greeting(question):
    response = self._generate_greeting_response(question)
    return response
```

**`_is_greeting()`** (line 242) checks three things:
1. **Exact match** against `GREETING_PATTERNS` list: "hi", "hello", "hey", "thanks", "bye", etc.
2. **Regex match** against `GREETING_REGEX_PATTERNS`: catches variations like "heyyyy", "how's it going", "what's up"
3. **Prefix match with word boundary**: "hi there" starts with "hi" AND next char is a space → greeting. "history" starts with "hi" but next char is "s" → NOT a greeting.

If greeting → call `_generate_greeting_response()` which sends a one-shot LLM call:
- System prompt: "You are BabyJay... respond naturally, keep it SHORT, vary your opening"
- Temperature: 0.9 (high creativity for varied greetings)
- Max tokens: 100

Returns things like: "Hey there! What can I help you with today?" (different every time)

### Step 1.3: About-Bot Detection
```python
if self._is_about_bot(question):
    response = self._generate_about_response(question)
    return response
```

**`_is_about_bot()`** (line 284) checks for phrases like:
- "who are you", "what can you do", "are you ai", "who created you", "what is babyjay"

If matched → LLM call with temperature 0.8 to generate a self-introduction.

### Step 1.4: Off-Topic Detection
```python
if self._is_off_topic(question):
    return "That's a bit outside my area! I'm all about KU — ..."
```

**`_is_off_topic()`** (line 324) is a two-layer filter:

**Layer 1 — Allow if KU-related:** If the query contains ANY of these keywords, it's NOT off-topic:
```
"ku", "kansas", "jayhawk", "professor", "course", "class", "eecs", "campus",
"dorm", "dining", "bus", "transit", "tuition", "admission", "library", "gym",
"scholarship", "housing", "building", "department", "degree", "major", "gpa", ...
```

**Layer 2 — Block if clearly off-topic:** Regex patterns catch:
- Trivia: "what is the capital of France"
- Recipes, movies, stocks, crypto
- "Write me a poem/story/essay"
- Math problems: "solve 2+2"
- Sports: NBA, NFL, FIFA
- Medical advice, diet, workout plans

**No API call.** Pure regex. Instant response.

### Step 1.5: Clarification Check
```python
if self._is_clarification_answer(question):
    return self._process_clarification_answer(question)
```

If a PREVIOUS query triggered a clarification question (e.g., user said "professors" and bot asked "which department?"), and `waiting_for_clarification` is True, this catches the answer.

**`_process_clarification_answer()`** (line 485):
1. Takes the original ambiguous query ("professors") and the answer ("EECS")
2. Combines them: "EECS professors"
3. Calls `self.ask(combined_query)` recursively — sending it through the full pipeline again

### Step 1.6: Ambiguity Detection
```python
if self._is_ambiguous(question):
    return self._generate_clarification_question(question)
```

**`_is_ambiguous()`** (line 446) triggers for very short, vague queries:
- Single vague words: "professors", "courses", "help", "info"
- Two-word vague phrases: "professors here", "courses available"
- Does NOT trigger if ≥3 words (enough context to work with)
- Does NOT trigger if it's a follow-up or department filter

If ambiguous → returns a clarification question like "What research area or department are you interested in?" Sets `waiting_for_clarification = True` so the next message is caught by Step 1.5.

---

## PHASE 2: QUERY CLEANING

**Functions:** `_needs_cleaning()`, `_clean_query_hybrid()`
**File:** `app/rag/chat.py` (lines 360-420)

### Step 2.1: Should We Clean?
```python
cleaned, was_cleaned, method = self._clean_query_hybrid(question)
```

**`_needs_cleaning()`** (line 360) checks for indicators of messy input:
- Known typos: "machien", "artifical", "robtics", "quantim", "profesors"
- Text speak: " 2 " (too), " 4 " (for), " u " (you), " r " (are), " abt " (about)
- Multiple spaces, excessive punctuation ("???", "!!!")

If the query looks clean → skip cleaning entirely (saves time).

### Step 2.2: Local Cleaning (No API Call)
If cleaning is needed, first try **local preprocessing**:

**File:** `app/rag/query_preprocessor.py`
**Class:** `QueryPreprocessor`

The preprocessor runs a 4-step pipeline:

**Step A — Normalize:** Lowercase, remove emojis/special chars, normalize whitespace.
```
"EECS  168???" → "eecs 168"
```

**Step B — Detect Subject Codes:** Recognizes patterns like "EECS 168", "AE345". Protects them from being "corrected."
- Combined codes: "EECS168" → "EECS 168"
- Fuzzy matching: "EESC" → "EECS" (if similarity ≥ 80%)

**Step C — Synonym Expansion:** Expands abbreviations:
```
"ml" → "machine learning"
"ai" → "artificial intelligence"
"prereq" → "prerequisite"
"prof" → "professor"
"calc" → "calculus"
"orgo" → "organic chemistry"
```
Will NOT expand if the abbreviation is a valid subject code (protects "CS 101" from becoming "computer science 101").

**Step D — Fuzzy Typo Correction:** Uses `rapidfuzz` library to match against a vocabulary of ~100 course-related words.
```
"machien" → "machine" (similarity 85%+)
"compter" → "computer"
"introducton" → "introduction"
```
Protected words ("the", "a", "is", "course", etc.) are never "corrected." Threshold is 85% similarity — high enough to avoid false corrections.

### Step 2.3: LLM Cleaning (Fallback)
If local cleaning didn't change anything, try Claude Haiku:
```python
llm_cleaned = _call_haiku(
    client, "Fix typos, grammar, and text speak. Return ONLY the corrected query.",
    [{"role": "user", "content": query}],
    temperature=0, max_tokens=100
)
```
Temperature 0 = deterministic. No creativity, just correction.

---

## PHASE 3: DEPARTMENT FILTER vs. FOLLOW-UP vs. NEW QUERY

Back in `ask()`, line 677-710. The cleaned query now enters the search pipeline, but HOW it enters depends on what kind of query it is.

### Path A: Department Filter
```python
if self._is_department_filter(search_question) and self.last_search_query:
```

If the user said "only EECS" or "just business" AND there was a previous faculty search:
1. Combine: `"ML professors" + "EECS"` → `"ML professors EECS"`
2. Route the combined query
3. Filter the context to only show professors from that department

**`_filter_context_by_department()`** (line 527) parses the raw context text, finds "Department:" lines, and removes any professor whose department doesn't match.

### Path B: Follow-Up Expansion
```python
if self._is_simple_followup(search_question) and use_history:
    expanded = self._expand_followup_question(search_question)
```

**`_expand_followup_question()`** (line 570) takes a vague follow-up like "his number?" and uses the LLM to rewrite it with full context:

1. Sends the last 6 conversation messages + the follow-up to Haiku
2. Prompt: "Rewrite the follow-up question to include full context"
3. Temperature 0, max 100 tokens
4. "his number?" + history about Prof Kulkarni → "What is Professor Kulkarni's phone number?"

The expanded query then goes through normal routing.

### Path C: Normal New Query
Everything else. Goes straight to the router.

---

## PHASE 4: THE ROUTER — INTENT CLASSIFICATION + RETRIEVAL

**File:** `app/rag/router.py`
**Class:** `QueryRouter`
**Function:** `route()` (line 68)

### Step 4.0: Query Decomposition
```python
if self.decomposer.should_decompose(query):
    sub_queries = self.decomposer.decompose(query)
    if len(sub_queries) > 1:
        for sq in sub_queries:
            sr = self.route(sq)  # Recursive!
        return self.decomposer.merge_sub_results(sub_results)
```

**File:** `app/rag/query_decomposer.py`
**Class:** `QueryDecomposer`

Checks if the query has multiple parts:
- "Compare EECS 168 and EECS 268" → ["EECS 168 information", "EECS 268 information"]
- "Prerequisites for EECS 168, 268, and 368" → 3 separate queries
- "Should I take MATH 125 or MATH 126?" → 2 queries

Detection uses regex only (no LLM call). If decomposed, each sub-query goes through the FULL routing pipeline independently, then results are merged with deduplication.

### Step 4.1: Classification

**File:** `app/rag/classifier.py`
**Class:** `QueryClassifier`
**Function:** `classify()` (line 153)

The classifier determines WHAT the user is asking about. It returns:
```python
{
    "intent": "faculty_search",      # What kind of question
    "entities": {                     # Extracted details
        "department": "eecs",
        "research_area": "machine learning"
    },
    "scope": "top_results",          # How many results they want
    "confidence": 0.85,
    "method": "regex"                # How we classified it
}
```

**How classification works (hybrid approach):**

**Layer 1 — Regex Scoring** (`_detect_intent_regex()`):

Each intent has a list of regex patterns. The classifier counts how many patterns match:

```python
intent_patterns = {
    "faculty_search": [
        r"\b(professor|professors|faculty|researcher)\b",    # +1 if matches
        r"\b(who teaches|who does research|expert in)\b",    # +1 if matches
        r"\b(research in|working on|studies)\b",             # +1 if matches
    ],
    "course_info": [
        r"\b[A-Z]{2,4}\s*\d{3,4}\b",                       # Course code → +3
        r"\b(course|courses|class|classes)\b",               # +1
        r"\b(prerequisite|prereq|corequisite)\b",            # +1
        ...
    ],
    "dining_info": [...],
    "transit_info": [...],
    "housing_info": [...],
    "financial_info": [...],
    ...
}
```

The intent with the highest score wins. Confidence = `0.6 + (score × 0.15)`, capped at 0.95.

**Special rule:** A course code like "EECS 168" gets +3 immediately, making course_info very hard to beat.

**Layer 2 — Entity Extraction:**

Depending on the intent, extract specific entities:

- **Faculty:** department (from alias list), research area (from keyword list), professor name (regex for "Dr. X" or "Prof Y")
- **Courses:** course code ("EECS 168"), subject ("EECS"), level ("graduate"/"undergraduate"), credits

**Layer 3 — LLM Fallback** (`_classify_with_llm()`):

If regex confidence < 0.7, send the query to Claude Haiku for classification:
```python
"Classify the user query for a university chatbot. Return JSON only."
```
Temperature 0, max 200 tokens. Returns structured JSON with intent + entities. Only used if the regex is unsure.

**Layer 4 — Scope Detection:**

Checks for words like "all", "every", "complete list", "how many" → sets scope to "complete_list" (returns more results). Otherwise "top_results".

### Step 4.2: Routing to a Retriever

Based on the classified intent, the router sends the query to the right retriever:

```
faculty_search  → FacultyRetriever (JSON lookup, ~5-20ms)
course_info     → CourseRetriever (JSON lookup, ~5-50ms)
dining_info     → CampusRetriever.search_dining() (JSON, ~1-5ms)
housing_info    → CampusRetriever.search_housing() (JSON, ~1-5ms)
transit_info    → CampusRetriever.search_transit() (JSON, ~1-5ms)
financial_info  → CampusRetriever.search_tuition() (JSON, ~1-5ms)
admission_info  → Vector search fallback (~500-1000ms)
calendar_info   → Vector search fallback (~500-1000ms)
library_info    → Vector search fallback (~500-1000ms)
general         → Vector search fallback (~500-1000ms)
```

---

## PHASE 5: THE RETRIEVERS — GETTING THE DATA

### 5A: FacultyRetriever (JSON Lookup)

**File:** `app/rag/faculty_retriever.py`
**Data:** `data/all_faculty_combined.json` (2,207+ faculty across ~50 departments)

Loaded once into memory. Three search strategies:

**Strategy 1 — By Name:**
If the classifier extracted a name entity → `search_by_name(name)`. Partial case-insensitive match across all faculty.

**Strategy 2 — By Department:**
If department detected → `get_department_faculty(dept_key)`. Returns all faculty in that department.

**Strategy 3 — By Research Area:**
`_filter_by_research()` searches each faculty member's:
- `research_interests` array
- `biography` text
- `title` field

Uses substring matching (case-insensitive). "machine learning" matches if it appears anywhere in those fields.

**Department Affinity Ranking:**
When searching by research area without a specific department, results are ranked by department affinity. `TOPIC_DEPARTMENT_AFFINITY` maps topics to preferred departments:
```
"machine learning" → EECS first, then other departments
"quantum computing" → Physics first
"supply chain" → Business first
```

Primary matches (preferred department) appear before secondary matches (other departments).

**Format for context:**
`format_for_context()` creates text like:
```
Professor: Prasad Kulkarni
Department: EECS
Email: prasadk@ku.edu
Research: software security, compiler optimizations, virtual machines
```

### 5B: CourseRetriever (JSON Lookup)

**File:** `app/rag/course_retriever.py`
**Data:** `data/courses/all_courses.json` (7,344 courses)

Three indexes built at load time:
- `_subject_index`: courses grouped by subject code (EECS, MATH, etc.)
- `_level_index`: courses grouped by level (undergraduate/graduate)
- `_code_index`: hash map for O(1) exact course code lookup

**Search strategies (tried in order):**

1. **Exact course code:** "EECS 168" → hash lookup → instant
2. **Subject filter:** "EECS courses" → return all EECS courses
3. **Level filter:** "graduate courses" → return all graduate courses
4. **General search:** Score every course across multiple fields:
   - `course_code` match: weight 10 (highest)
   - `title` match: weight 8
   - `subject` match: weight 6
   - `level` match: weight 5
   - `department` match: weight 4
   - `school` match: weight 3
   - `description` match: weight 3
   - `prerequisites` match: weight 2

Results sorted by total score. Top N returned.

The CourseRetriever has its own `QueryPreprocessor` instance initialized with valid subject codes from the course catalog.

### 5C: CampusRetriever (JSON Lookup)

**File:** `app/rag/campus_retriever.py`
**Data:** JSON files for dining, transit, housing, tuition

Simple substring matching against concatenated fields. Fastest retriever (~1-5ms).

- **Dining:** Matches against name, building, type, description
- **Transit:** Matches against route name, description, stops. Special "KU" filter for campus-only routes
- **Housing:** Flattens nested structure (residence halls, scholarship halls, apartments)
- **Tuition:** Recursively extracts fees from nested JSON

### 5D: Vector Search Fallback (Hybrid BM25 + ChromaDB)

**Files:** `app/rag/retriever.py`, `app/rag/bm25_scorer.py`
**Data:** ChromaDB at `data/vectordb/` (63 MB, ~2000-3000 documents)

When no specialized retriever exists for an intent (admissions, calendar, library, etc.), the router falls back to vector search.

**How vector search works:**

1. Your query gets sent to OpenAI's embedding API (`text-embedding-3-small`)
2. OpenAI returns a 1536-dimensional vector representing the semantic meaning
3. ChromaDB computes cosine distance between your query vector and every document vector
4. Returns the top N closest documents
5. Score = `1 - distance` (0 to 1, higher = more relevant)
6. Documents below `min_relevance=0.25` are filtered out

**How BM25 works (keyword scoring):**

Running in parallel with vector search:

1. Documents are tokenized (split into words, stopwords removed)
2. For each query term, compute:
   - TF (term frequency): how often does this word appear in the document?
   - IDF (inverse document frequency): how rare is this word across all documents?
   - BM25 score = IDF × (TF contribution with saturation)
3. Documents with matching keywords get a BM25 score

**How they're merged (Reciprocal Rank Fusion):**

```
For each document:
  RRF_score = 0.6 × (1 / (60 + vector_rank)) + 0.4 × (1 / (60 + bm25_rank))
```

Vector search gets 60% weight, BM25 gets 40%. RRF uses RANKS not raw scores (because vector scores and BM25 scores are on completely different scales).

Documents that appear in BOTH searches get higher combined scores. A document that's #1 in vector and #3 in BM25 beats one that's #2 in vector but absent from BM25.

BM25 indices are cached per source filter (e.g., one index for "dining" documents, another for "course" documents).

### 5E: Re-ranking (LLM-based)

**File:** `app/rag/reranker.py`
**Class:** `Reranker`

Only runs on **vector fallback results** when there are more than 3 results. Not used for JSON retrievers (they're already precise).

How it works:
1. Takes the top 15 results from hybrid search
2. Truncates each to 300 chars
3. Sends to Claude Haiku: "Score each document's relevance to the query from 0-10"
4. Parses the JSON array of scores
5. Re-sorts results by score
6. Returns top 5

This is a **cross-encoder** pattern. Vector search uses bi-encoders (query and document embedded separately). Cross-encoders look at query AND document together, which is more accurate but slower.

---

## PHASE 6: CONTEXT BUILDING

**File:** `app/rag/context_builder.py`
**Class:** `ContextBuilder`
**Function:** `build()` (line 21)

The retriever returns raw data. The context builder compresses it into what the LLM actually needs.

### Step 6.1: Determine Relevant Fields

Based on the query and intent, decide which fields matter:

```python
# Faculty search:
fields = {"name", "title", "department", "email", "research"}
if "office" or "where" in query:
    fields.add("office", "building", "phone")

# Course info:
fields = {"course_code", "credits", "description"}
if "prerequisite" in query:
    fields.add("prerequisites")

# Dining:
fields = {"building", "type", "hours"}
```

### Step 6.2: Format Each Result

Each result is formatted with only the relevant fields:

```
[Source: faculty_retriever]
Professor: Prasad Kulkarni
Department: EECS
Email: prasadk@ku.edu
Research: software security, compiler optimizations
```

The `[Source: ...]` tag tells the LLM where this info came from. The system prompt (Rule 11) tells the LLM to cite sources naturally: "According to the faculty directory..."

### Step 6.3: Deduplicate

Same professor/course appearing twice? Removed by name.

### Step 6.4: Truncate

Max 4000 characters. If exceeded: `context[:4000] + "\n\n[...additional results omitted]"`

---

## PHASE 7: THE LLM CALL — GENERATING THE RESPONSE

Back in `ask()`, line 761-805.

### Step 7.1: RLHF Enhancement

```python
enhanced_prompt = self.rlhf_optimizer.enhance_prompt(SYSTEM_PROMPT, question)
```

**File:** `app/rag/rlhf_optimizer.py`
**Class:** `RLHFOptimizer`

Fetches patterns from user feedback (thumbs up/down stored in Supabase). If there are learned lessons (e.g., "users dislike when you add emojis"), they get appended to the system prompt. If Supabase is unavailable → returns the base prompt unchanged.

### Step 7.2: Build the Message Array

```python
messages = [{"role": "system", "content": enhanced_prompt}]

# Add conversation history (last 6 messages)
if use_history and self._conversation_history:
    messages.extend(self._conversation_history[-6:])
```

Then add the user message — the format depends on whether we found context:

**WITH context:**
```
Here's information from KU's database (use ONLY this to answer — do not add anything beyond what is here):

[Source: faculty_retriever]
Professor: Prasad Kulkarni
Department: EECS
Email: prasadk@ku.edu
Research: software security, compiler optimizations

User's question: ML professors

Answer based ONLY on the information above. If the context doesn't answer their question, say so.
```

**WITHOUT context:**
```
User's question: Tell me about the Mars rover

I don't have specific information about this in my database right now.
Let the user know you don't have that info and suggest they check ku.edu or contact the relevant KU office.
Do NOT guess or make up any specific details like names, numbers, URLs, or office locations.
```

### Step 7.3: The `_call_haiku()` Helper

**Function:** `_call_haiku()` (line 34)

This converts the message array to Anthropic's format:

1. **Filter out system messages** from the conversation array (Anthropic takes system as a separate parameter)
2. **Merge consecutive same-role messages** — Anthropic requires alternating user/assistant. If history has two user messages in a row, they get merged with `\n\n`
3. **Ensure first message is from user** — Anthropic requires this. If not, inserts a dummy "Hello"
4. **Call the API:**
   ```python
   response = client.messages.create(
       model="claude-haiku-4-5-20251001",
       system=system_prompt,
       messages=merged,
       temperature=0.5,
       max_tokens=1500,
   )
   ```

Temperature 0.5 = balanced between creativity and accuracy. The system prompt (SYSTEM_PROMPT, 68 lines) controls everything about how the LLM responds:
- Scope restrictions (KU-only)
- Anti-hallucination rules (11 rules)
- Response style (prose vs. bullets)
- Personality ("warm and friendly, like a helpful upperclassman")
- Department prioritization for technical topics

### Step 7.4: Error Fallback

If the LLM call fails (rate limit, timeout, etc.):
```python
except Exception as e:
    if context:
        return f"I found some information but had trouble formatting it. Here's what I have:\n\n{context[:1500]}"
    return "Sorry, I'm having trouble right now. Please try again in a moment!"
```

If we have context, the user still gets useful data even without LLM formatting.

---

## PHASE 8: AFTER THE RESPONSE

### Step 8.1: Context Tracking

```python
if not self._is_simple_followup(search_question):
    if context:
        self.recent_context = context  # Save for follow-ups
    else:
        self.recent_context = ""       # Clear stale context
```

New topic → update `recent_context` so follow-ups can reference it.
Follow-up → keep the existing context (don't overwrite).

### Step 8.2: Faculty Query Tracking

```python
if any(kw in search_question.lower() for kw in ["professor", "faculty", "research", "ml", "ai"]):
    self.last_search_query = search_query
```

Saves the search query so "only EECS" department filters work on the next message.

### Step 8.3: KU Retry

If no context was found AND the query isn't off-topic AND doesn't already contain "KU":
```python
retry_query = f"KU {search_query}"
retry_results = self.router.route(retry_query)
```

Sometimes adding "KU" helps the retriever find relevant results (e.g., "parking" → "KU parking").

---

## THE SYSTEM PROMPT — What the LLM is Told

The `SYSTEM_PROMPT` (line 85, 68 lines) is the most important piece of text in the entire system. It controls:

**Scope:** What BabyJay can and cannot answer.

**11 Rules:**
1. If context provided → use ONLY it
2. If NO context → say so, suggest ku.edu, do NOT guess
3. Be conversational
4. Never say "I don't have info" if context was provided
5. NEVER make up names, locations, phone numbers, URLs
6. Professor names MUST appear in context
7. Prefer EECS for technical topics
8. List most relevant department first
9. Ignore unrelated context
10. NEVER fabricate specific numbers
11. Cite sources naturally from [Source: ...] tags

**Response Style:** Prose for 1-2 items, bullets for 3+. No markdown headings. Vary openings.

---

## COMPLETE FILE MAP

| File | Role | When Called |
|------|------|------------|
| `app/routers/api_routes.py` | HTTP endpoint, auth, DB ops | Every request |
| `app/rag/chat.py` | Decision tree, LLM calls, state management | Every request |
| `app/rag/router.py` | Query decomposition + routing to retrievers | Every non-greeting/off-topic query |
| `app/rag/classifier.py` | Intent detection (regex + LLM fallback) | Every routed query |
| `app/rag/query_decomposer.py` | Split multi-part questions | Complex queries only |
| `app/rag/faculty_retriever.py` | JSON faculty lookup | Faculty queries |
| `app/rag/course_retriever.py` | JSON course lookup | Course queries |
| `app/rag/campus_retriever.py` | JSON campus data lookup | Dining/transit/housing/tuition |
| `app/rag/retriever.py` | Hybrid BM25+vector search | Fallback queries |
| `app/rag/bm25_scorer.py` | BM25 keyword scoring | Used by retriever.py |
| `app/rag/reranker.py` | LLM re-ranking | Vector fallback results only |
| `app/rag/context_builder.py` | Compress + cite context | Every query with results |
| `app/rag/query_preprocessor.py` | Typo correction, synonym expansion | Messy queries only |
| `app/rag/rlhf_optimizer.py` | Feedback-based prompt enhancement | Every LLM call |

---

## TIMING BREAKDOWN (Typical Query)

| Step | Time | Notes |
|------|------|-------|
| HTTP + Auth | ~10ms | FastAPI + JWT |
| Early exits (greeting/off-topic) | ~0ms | Pure regex |
| Query cleaning (local) | ~5ms | If needed |
| Query cleaning (LLM) | ~500ms | Only if local fails |
| Classification (regex) | ~2ms | Usually sufficient |
| Classification (LLM fallback) | ~500ms | Only if regex unsure |
| JSON retriever | ~5-50ms | Faculty/Course/Campus |
| Vector search | ~700ms | OpenAI embedding API |
| BM25 scoring | ~5ms | Local computation |
| Re-ranking | ~500ms | Only for vector results |
| Context building | ~2ms | Local computation |
| LLM response generation | ~800-1500ms | Claude Haiku |
| **Total (fast path — JSON)** | **~1-2 seconds** | Faculty/course/dining |
| **Total (slow path — vector)** | **~2-4 seconds** | Admissions/calendar/general |

---

## STATE THAT PERSISTS BETWEEN MESSAGES

These live on the `BabyJayChat` instance (preserved across requests via `_chat_instances` cache):

| Variable | Type | Purpose |
|----------|------|---------|
| `_conversation_history` | List[Dict] | Last 6 messages sent to LLM for context |
| `recent_context` | str | Last retrieved context (for follow-up fallback) |
| `last_search_query` | str | Last faculty search (for department filters) |
| `last_mentioned_course` | str | Last course code (for "that course" references) |
| `waiting_for_clarification` | bool | Are we expecting a clarification answer? |
| `original_ambiguous_query` | str | What was the ambiguous query we asked about? |
| `active_department_filter` | str | Active department filter |

---

## DATA FILES

| File | Contents | Size |
|------|----------|------|
| `data/all_faculty_combined.json` | 2,207+ faculty, ~50 departments | Large |
| `data/courses/all_courses.json` | 7,344 courses | Large |
| `data/dining/locations.json` | Campus dining locations | Small |
| `data/transit/routes.json` | Bus routes and stops | Small |
| `data/housing/housing.json` | Dorms, apartments, meal plans | Small |
| `data/tuition/tuition_fees.json` | Tuition rates and fees | Small |
| `data/vectordb/chroma.sqlite3` | ChromaDB embeddings | 63 MB |

---

That's everything. Every function, every decision, every file. From the moment a user types a message to the moment they see a response.
