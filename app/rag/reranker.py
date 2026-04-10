"""
Re-ranker for BabyJay RAG
==========================
Re-scores retrieved results against the original query using a local
cross-encoder model — purpose-built for passage reranking.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS MARCO (real search queries + passages)
  - Latency: ~5-30ms for 15 passages on CPU
  - Free, runs locally — no API calls

Why cross-encoder beats LLM reranking:
  - Jointly encodes (query, passage) — sees full interaction
  - Trained end-to-end for relevance scoring, not general reasoning
  - Deterministic and ~10-20x faster than an LLM API call
  - Scores are comparable across queries (stable ranking signal)

Note on environment:
  torch._dynamo is stubbed before the transformers import because
  torch 2.3.1 has a broken ONNX import path that transformers triggers
  via _prepare_4d_attention_mask_for_sdpa. The stub is safe — we never
  use torch.compile here, and is_compiling() → False is correct.
"""

import sys
import types
from functools import lru_cache
from typing import List, Dict

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_MAX_CANDIDATES = 20  # Beyond this, marginal gain is negligible
_PASSAGE_CHAR_LIMIT = 800  # ~200 tokens; cross-encoders cap at 512 tokens


def _patch_torch_dynamo() -> None:
    """
    Stub out torch._dynamo to work around a broken ONNX import in
    torch 2.3.1 that prevents transformers from loading when _dynamo is
    lazily imported during attention-mask utilities.

    Safe as long as we never use torch.compile (we don't).
    """
    if "torch._dynamo" not in sys.modules:
        stub = types.ModuleType("torch._dynamo")
        stub.is_compiling = lambda: False
        stub.config = types.SimpleNamespace(suppress_errors=True)
        sys.modules["torch._dynamo"] = stub


@lru_cache(maxsize=1)
def _load_model():
    """
    Load the cross-encoder model once and cache it for the process lifetime.

    attn_implementation="eager" bypasses PyTorch SDPA / torch.compile, which
    is broken on torch 2.3.1 with transformers 4.44.x (ONNX exporter issue).
    Eager attention is ~same speed for batch sizes < 64 on CPU.
    """
    _patch_torch_dynamo()
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        attn_implementation="eager",   # avoid broken SDPA path on torch 2.3.1
    )
    model.eval()
    return tokenizer, model


class Reranker:
    """
    Re-rank retrieved results using a local cross-encoder model.

    Usage::

        reranker = Reranker()
        ranked = reranker.rerank("machine learning courses", results, top_k=5)
    """

    def __init__(self):
        # Eagerly load so the first real query isn't slow
        self._tokenizer, self._model = _load_model()

    def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Re-rank results by relevance to the query.

        Args:
            query:   The user's original query
            results: List of dicts with at least a "content" key
            top_k:   Number of top results to return after reranking

        Returns:
            Re-ranked list (best first). Each dict gains a "rerank_score" key
            containing the raw cross-encoder logit (higher = more relevant).
        """
        if not results:
            return results
        if len(results) == 1:
            results[0].setdefault("rerank_score", 1.0)
            return results

        import torch

        candidates = results[:_MAX_CANDIDATES]

        # Build (query, passage) pairs
        pairs: List[tuple] = []
        for r in candidates:
            passage = r.get("content", "")
            if len(passage) > _PASSAGE_CHAR_LIMIT:
                passage = passage[:_PASSAGE_CHAR_LIMIT]
            pairs.append((query, passage))

        # Single batched forward pass — no gradient needed
        with torch.no_grad():
            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            scores = self._model(**inputs).logits.squeeze(-1).tolist()

        # If logits is a scalar (single pair), wrap it
        if isinstance(scores, float):
            scores = [scores]

        # Attach scores and sort descending
        scored: List[Dict] = []
        for r, score in zip(candidates, scores):
            r_copy = dict(r)
            r_copy["rerank_score"] = float(score)
            scored.append(r_copy)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Tail (results beyond _MAX_CANDIDATES) keep original order — appended
        # after scored results in case top_k > _MAX_CANDIDATES.
        tail = results[_MAX_CANDIDATES:]
        return (scored + tail)[:top_k]
