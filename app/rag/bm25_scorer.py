"""
BM25 Scorer for BabyJay RAG
============================
Lightweight BM25 keyword scoring to complement vector search.

Vector search (ChromaDB) excels at semantic similarity but can miss exact keyword matches.
BM25 excels at exact term matching but misses semantic similarity.
Together = hybrid search, the gold standard for production RAG.

Usage:
    scorer = BM25Scorer()
    scorer.index_documents(docs)  # One-time indexing
    results = scorer.score(query, top_k=5)
"""

import hashlib
import math
import re
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple


class BM25Scorer:
    """BM25 keyword relevance scorer for hybrid search."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: Term frequency saturation. Higher = more weight on term frequency.
            b: Length normalization. 0 = no normalization, 1 = full normalization.
        """
        self.k1 = k1
        self.b = b
        self.documents: List[Dict[str, Any]] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_freqs: Dict[str, int] = {}  # term -> number of docs containing it
        self.avg_doc_len: float = 0.0
        self.n_docs: int = 0
        self._indexed = False

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer: lowercase, split on non-alphanumeric, remove stopwords."""
        text = text.lower()
        tokens = re.findall(r'[a-z0-9]+', text)
        # Remove common stopwords that don't help with scoring
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
            'not', 'no', 'nor', 'so', 'if', 'then', 'than', 'too', 'very',
            'just', 'about', 'up', 'out', 'that', 'this', 'it', 'its', 'i',
            'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'they',
            'them', 'what', 'which', 'who', 'whom', 'how', 'when', 'where',
        }
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def index_documents(self, documents: List[Dict[str, Any]], content_key: str = "content"):
        """
        Index documents for BM25 scoring.

        Args:
            documents: List of dicts, each must have a content_key field
            content_key: Key in dict that contains the text to index
        """
        self.documents = documents
        self.doc_tokens = []
        self.doc_freqs = {}

        for doc in documents:
            content = doc.get(content_key, "")
            if isinstance(content, dict):
                content = str(content)
            tokens = self._tokenize(str(content))
            self.doc_tokens.append(tokens)

            # Count document frequency (unique terms per doc)
            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.n_docs = len(documents)
        total_tokens = sum(len(t) for t in self.doc_tokens)
        self.avg_doc_len = total_tokens / self.n_docs if self.n_docs > 0 else 0
        self._indexed = True

    def _idf(self, term: str) -> float:
        """Inverse document frequency with smoothing."""
        df = self.doc_freqs.get(term, 0)
        # BM25 IDF formula with +1 smoothing to avoid negative IDF
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)

    def score_document(self, query_tokens: List[str], doc_idx: int) -> float:
        """Score a single document against query tokens."""
        doc_tokens = self.doc_tokens[doc_idx]
        doc_len = len(doc_tokens)
        term_counts = Counter(doc_tokens)

        score = 0.0
        for term in query_tokens:
            if term not in term_counts:
                continue
            tf = term_counts[term]
            idf = self._idf(term)
            # BM25 scoring formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
            score += idf * (numerator / denominator)

        return score

    def score(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Score all documents against a query.

        Returns:
            List of (doc_index, score) tuples, sorted by score descending.
        """
        if not self._indexed:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = []
        for i in range(self.n_docs):
            s = self.score_document(query_tokens, i)
            if s > 0:
                scores.append((i, s))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search documents by BM25 score.

        Returns:
            List of document dicts with added 'bm25_score' field.
        """
        scored = self.score(query, top_k)
        results = []
        for doc_idx, score in scored:
            doc = dict(self.documents[doc_idx])
            doc['bm25_score'] = score
            results.append(doc)
        return results


def hybrid_merge(vector_results: List[Dict], bm25_results: List[Dict],
                 vector_weight: float = 0.6, bm25_weight: float = 0.4,
                 top_k: int = 5) -> List[Dict]:
    """
    Merge vector search and BM25 results using Reciprocal Rank Fusion (RRF).

    RRF is preferred over raw score merging because vector and BM25 scores
    are on different scales. RRF only uses rank positions.

    Args:
        vector_results: Results from ChromaDB vector search (must have 'content' key)
        bm25_results: Results from BM25 search (must have 'content' key)
        vector_weight: Weight for vector search ranks
        bm25_weight: Weight for BM25 ranks
        top_k: Number of results to return

    Returns:
        Merged and re-ranked results with 'hybrid_score' field
    """
    k = 60  # RRF constant (standard value from the original paper)

    def _dedup_key(result: Dict) -> str:
        """Prefer a stable metadata ID; otherwise hash the full content."""
        md = result.get("metadata") or {}
        for k_name in ("id", "_id", "doc_id", "chunk_id", "url", "profile_url", "name"):
            if md.get(k_name):
                return f"{k_name}:{md[k_name]}"
        content = str(result.get("content", ""))
        return "sha1:" + hashlib.sha1(content.encode("utf-8")).hexdigest()

    # Build content → result mapping, deduplicating by stable key
    content_to_result: Dict[str, Dict] = {}
    content_scores: Dict[str, float] = {}

    # Score vector results by rank
    for rank, result in enumerate(vector_results):
        content_key = _dedup_key(result)
        rrf_score = vector_weight * (1.0 / (k + rank + 1))

        if content_key not in content_to_result:
            content_to_result[content_key] = dict(result)
            content_scores[content_key] = 0.0
        content_scores[content_key] += rrf_score

    # Score BM25 results by rank
    for rank, result in enumerate(bm25_results):
        content_key = _dedup_key(result)
        rrf_score = bm25_weight * (1.0 / (k + rank + 1))

        if content_key not in content_to_result:
            content_to_result[content_key] = dict(result)
            content_scores[content_key] = 0.0
        content_scores[content_key] += rrf_score

    # Sort by combined RRF score
    sorted_keys = sorted(content_scores.keys(), key=lambda k: content_scores[k], reverse=True)

    merged = []
    for key in sorted_keys[:top_k]:
        result = content_to_result[key]
        result['hybrid_score'] = content_scores[key]
        merged.append(result)

    return merged
