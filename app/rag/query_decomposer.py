"""
Query Decomposer for BabyJay RAG
==================================
Breaks complex multi-part questions into sub-queries for better retrieval.

Problem it solves:
    "Compare EECS 168 and EECS 268 prerequisites" is ONE query but needs TWO lookups.
    Vector search on the full string finds mediocre results for both.
    Decomposing into ["EECS 168 prerequisites", "EECS 268 prerequisites"]
    gets precise results for each, then merges them.

Uses regex patterns first (fast, no API call), falls back to LLM only for
truly complex queries that regex can't handle.
"""

import re
from typing import List, Optional


# Patterns that indicate a multi-part question
COMPARISON_PATTERNS = [
    # "Compare X and Y" / "X vs Y" / "difference between X and Y"
    r'(?:compare|vs\.?|versus|difference(?:s)? between)\s+(.+?)\s+(?:and|vs\.?|versus|&)\s+(.+?)(?:\?|$)',
    # "X or Y" (when asking which to choose)
    r'(?:should i take|which is better|choose between)\s+(.+?)\s+(?:or|vs\.?|versus)\s+(.+?)(?:\?|$)',
]

MULTI_ENTITY_PATTERNS = [
    # "Tell me about X, Y, and Z"
    r'(?:tell me about|what (?:is|are)|info on|information about)\s+(.+?),\s+(.+?)(?:,?\s+and\s+(.+?))?(?:\?|$)',
    # "X and Y courses/professors/info"
    r'(.+?)\s+and\s+(.+?)\s+(?:courses?|professors?|faculty|info|information|details)',
]

LIST_PATTERNS = [
    # "Prerequisites for X, Y, and Z"
    r'(?:prerequisites?|prereqs?|credits?|instructors?|professors?)\s+(?:for|of)\s+(.+?),\s+(.+?)(?:,?\s+and\s+(.+?))?(?:\?|$)',
]

# Course code pattern for extraction
COURSE_CODE_RE = re.compile(r'\b([A-Z]{2,5})\s*(\d{3,4})\b')


class QueryDecomposer:
    """Decomposes complex queries into simpler sub-queries."""

    def should_decompose(self, query: str) -> bool:
        """Check if a query should be decomposed into sub-queries."""
        q = query.strip()

        # Don't decompose short queries
        if len(q) < 15:
            return False

        # Check for comparison patterns
        for pattern in COMPARISON_PATTERNS:
            if re.search(pattern, q, re.IGNORECASE):
                return True

        # Check for multiple course codes
        codes = COURSE_CODE_RE.findall(q)
        if len(codes) >= 2:
            return True

        # Check for multi-entity patterns
        for pattern in MULTI_ENTITY_PATTERNS:
            if re.search(pattern, q, re.IGNORECASE):
                return True

        # Check for list patterns
        for pattern in LIST_PATTERNS:
            if re.search(pattern, q, re.IGNORECASE):
                return True

        return False

    def decompose(self, query: str) -> List[str]:
        """
        Decompose a complex query into sub-queries.

        Returns:
            List of sub-queries. If decomposition isn't needed, returns [query].
        """
        q = query.strip()

        if not self.should_decompose(q):
            return [q]

        sub_queries = []

        # Strategy 1: Multiple course codes with a shared question
        codes = COURSE_CODE_RE.findall(q)
        if len(codes) >= 2:
            # Extract what's being asked about these courses
            question_part = self._extract_question_part(q, codes)
            for subj, num in codes:
                sub_q = f"{subj} {num} {question_part}".strip()
                sub_queries.append(sub_q)
            return sub_queries if sub_queries else [q]

        # Strategy 2: Comparison patterns
        for pattern in COMPARISON_PATTERNS:
            match = re.search(pattern, q, re.IGNORECASE)
            if match:
                entities = [g for g in match.groups() if g]
                question_type = self._infer_question_type(q)
                for entity in entities:
                    entity = entity.strip().rstrip('?.,!')
                    if entity:
                        sub_queries.append(f"{entity} {question_type}".strip())
                if sub_queries:
                    return sub_queries

        # Strategy 3: List patterns (prerequisites for X, Y, Z)
        for pattern in LIST_PATTERNS:
            match = re.search(pattern, q, re.IGNORECASE)
            if match:
                question_prefix = self._extract_prefix(q, match)
                entities = [g.strip().rstrip('?.,!') for g in match.groups() if g]
                for entity in entities:
                    if entity:
                        sub_queries.append(f"{question_prefix} {entity}".strip())
                if sub_queries:
                    return sub_queries

        # Strategy 4: Multi-entity patterns
        for pattern in MULTI_ENTITY_PATTERNS:
            match = re.search(pattern, q, re.IGNORECASE)
            if match:
                entities = [g.strip().rstrip('?.,!') for g in match.groups() if g]
                for entity in entities:
                    if entity:
                        sub_queries.append(entity)
                if sub_queries:
                    return sub_queries

        # No decomposition worked
        return [q]

    def _extract_question_part(self, query: str, codes: List[tuple]) -> str:
        """Extract what's being asked about the course codes."""
        q = query.lower()
        # Remove course codes from query to get the question part
        for subj, num in codes:
            q = q.replace(f"{subj.lower()} {num}", "").replace(f"{subj.lower()}{num}", "")

        # Remove comparison words
        for word in ['compare', 'vs', 'versus', 'and', 'or', 'between', 'difference']:
            q = q.replace(word, ' ')

        q = ' '.join(q.split()).strip().rstrip('?.,!')

        # If nothing left, infer from common patterns
        if not q or len(q) < 3:
            return "information"

        return q

    def _infer_question_type(self, query: str) -> str:
        """Infer what type of information is being asked about."""
        q = query.lower()
        if any(w in q for w in ['prerequisite', 'prereq', 'require']):
            return 'prerequisites'
        if any(w in q for w in ['credit', 'hour']):
            return 'credits'
        if any(w in q for w in ['instructor', 'professor', 'who teach', 'taught by']):
            return 'instructor'
        if any(w in q for w in ['schedule', 'time', 'when']):
            return 'schedule'
        if any(w in q for w in ['compare', 'difference', 'vs']):
            return 'information'
        return ''

    def _extract_prefix(self, query: str, match: re.Match) -> str:
        """Extract the question prefix before the entity list."""
        prefix = query[:match.start()].strip()
        # Also check for the word right before the match
        pre_match = re.match(r'(.*?(?:prerequisites?|prereqs?|credits?|instructors?|professors?)\s+(?:for|of))',
                             query, re.IGNORECASE)
        if pre_match:
            return pre_match.group(1).strip()
        return prefix if prefix else "information about"

    def merge_sub_results(self, sub_results: List[dict]) -> dict:
        """
        Merge results from multiple sub-queries into a single result.

        Args:
            sub_results: List of route results from each sub-query

        Returns:
            Merged result dict with combined results and context
        """
        all_results = []
        all_contexts = []
        sources = set()

        for sr in sub_results:
            results = sr.get("results", [])
            context = sr.get("context", "")
            source = sr.get("source", "")

            all_results.extend(results)
            if context:
                all_contexts.append(context)
            if source:
                sources.add(source)

        # Deduplicate results by content/name
        seen = set()
        deduped = []
        for r in all_results:
            key = str(r.get("content", r.get("course_code", r.get("name", ""))))[:100]
            if key and key not in seen:
                seen.add(key)
                deduped.append(r)

        merged_context = "\n\n---\n\n".join(all_contexts)

        return {
            "results": deduped,
            "context": merged_context,
            "source": ", ".join(sources) if sources else "none",
            "query_info": {"intent": "multi_part", "sub_query_count": len(sub_results)},
            "result_count": len(deduped),
        }
