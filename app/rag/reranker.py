"""
Re-ranker for BabyJay RAG
==========================
Re-scores retrieved results against the original query for better precision.

Uses Claude Haiku as a lightweight cross-encoder re-ranker.
Each result is scored for relevance to the query, then results are re-sorted.

This dramatically improves retrieval quality — the top result after re-ranking
is almost always more relevant than the top result from vector search alone.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional
import anthropic
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HAIKU_MODEL = "claude-haiku-4-5-20251001"


class Reranker:
    """Re-rank retrieved results by relevance to the query."""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Re-rank results by relevance to query.

        Args:
            query: The user's original query
            results: List of dicts with at least a "content" key
            top_k: Number of top results to return

        Returns:
            Re-ranked list of results (best first), with added "rerank_score" field
        """
        if not results or len(results) <= 1:
            return results

        # Don't re-rank if we have very few results
        if len(results) <= 3:
            return results[:top_k]

        # Build the scoring prompt
        docs_text = ""
        for i, r in enumerate(results[:15]):  # Cap at 15 to control costs
            content = r.get("content", "")
            # Truncate long documents for scoring
            if len(content) > 300:
                content = content[:300] + "..."
            docs_text += f"\n[DOC {i}]: {content}\n"

        try:
            response = self.client.messages.create(
                model=HAIKU_MODEL,
                system=(
                    "You are a relevance scorer. Given a query and documents, "
                    "score each document's relevance to the query from 0-10. "
                    "Return ONLY a JSON array of scores in order. Example: [8, 2, 9, 1, 5]\n"
                    "Score 0 = completely irrelevant, 10 = perfectly answers the query."
                ),
                messages=[{
                    "role": "user",
                    "content": f"Query: {query}\n\nDocuments:{docs_text}\n\nScores (JSON array only):"
                }],
                temperature=0,
                max_tokens=100,
            )

            scores_text = response.content[0].text.strip()
            # Parse JSON array of scores — json.loads is safer than eval.
            scores_match = re.search(r'\[[\d,\s]+\]', scores_text)
            scores: List[int] = []
            if scores_match:
                try:
                    scores = [int(x) for x in json.loads(scores_match.group())]
                except (ValueError, json.JSONDecodeError):
                    scores = []
            if not scores:
                # Fallback: try to extract numbers
                scores = [int(x) for x in re.findall(r'\d+', scores_text)]

            # Attach scores and sort
            scored_results = []
            for i, r in enumerate(results[:15]):
                score = scores[i] if i < len(scores) else 0
                r_copy = dict(r)
                r_copy["rerank_score"] = score
                scored_results.append(r_copy)

            scored_results.sort(key=lambda x: x["rerank_score"], reverse=True)
            return scored_results[:top_k]

        except Exception as e:
            # On any error, return original order
            return results[:top_k]
