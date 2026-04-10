"""
Re-ingest all ChromaDB collections with text-embedding-3-large
===============================================================
Run this once after upgrading from text-embedding-3-small.

It will:
  1. Delete the babyjay_knowledge collection
  2. Delete the faculty collection
  3. Re-create both with text-embedding-3-large embeddings

IMPORTANT: This makes OpenAI API calls for every document.
Estimated cost: ~$0.50-2.00 depending on corpus size.

Usage (from BabyJay root):
    python app/scripts/reingest_embeddings.py
"""

import os
import sys
import time
from pathlib import Path

# Make sure BabyJay root is on the path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.utils import embedding_functions


def delete_collection(client: chromadb.PersistentClient, name: str) -> None:
    try:
        client.delete_collection(name)
        print(f"  Deleted collection: {name}")
    except Exception:
        print(f"  Collection {name!r} did not exist, skipping delete")


def reingest_babyjay_knowledge(vectordb_path: str) -> None:
    """Delete + rebuild babyjay_knowledge with text-embedding-3-large."""
    from app.rag.embeddings import reset_database, EMBEDDING_MODEL
    print(f"\n=== babyjay_knowledge ===")
    print(f"  Model: {EMBEDDING_MODEL}")
    t0 = time.time()
    collection = reset_database(vectordb_path)
    elapsed = time.time() - t0
    print(f"  Done: {collection.count()} docs in {elapsed:.1f}s")


def reingest_faculty(vectordb_path: str) -> None:
    """Delete + rebuild faculty collection with text-embedding-3-large."""
    from app.rag.embeddings import EMBEDDING_MODEL

    print(f"\n=== faculty ===")
    print(f"  Model: {EMBEDDING_MODEL}")

    # Load faculty documents
    faculty_docs_path = ROOT / "data" / "faculty_documents.json"
    if not faculty_docs_path.exists():
        print(f"  WARNING: {faculty_docs_path} not found — skipping faculty collection")
        return

    import json
    with open(faculty_docs_path) as f:
        faculty_data = json.load(f)

    # faculty_documents.json can be a list or {"documents": [...]}
    if isinstance(faculty_data, list):
        documents = faculty_data
    else:
        documents = faculty_data.get("documents", [])
    if not documents:
        print("  WARNING: No documents in faculty_documents.json")
        return

    client = chromadb.PersistentClient(path=vectordb_path)
    delete_collection(client, "faculty")

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=EMBEDDING_MODEL,
    )

    collection = client.create_collection(
        name="faculty",
        embedding_function=openai_ef,
        metadata={"description": "KU Faculty members", "embeddings": f"OpenAI {EMBEDDING_MODEL}"},
    )

    # Batch insert (ChromaDB handles batching internally)
    docs_text = []
    metadatas = []
    ids = []

    for i, doc in enumerate(documents):
        # Build rich searchable text from all relevant fields
        ri = doc.get("research_interests", [])
        if isinstance(ri, list):
            ri_str = "; ".join(ri)
        else:
            ri_str = str(ri)

        text = (
            doc.get("searchable_text")
            or doc.get("document")
            or doc.get("text")
            or (
                f"Professor: {doc.get('name', '')}\n"
                f"Department: {doc.get('department', '')}\n"
                f"Title: {doc.get('title', '')}\n"
                f"Research Interests: {ri_str}\n"
                f"Biography: {doc.get('biography', '')}"
            )
        )

        # Metadata: only scalar types allowed by ChromaDB
        meta = {
            k: (v if isinstance(v, (str, int, float, bool)) else str(v))
            for k, v in doc.items()
            if k not in ("searchable_text", "research_interests", "biography")
               and isinstance(v, (str, int, float, bool))
        }
        # Add name + department explicitly in case they were skipped
        meta.setdefault("name", doc.get("name", ""))
        meta.setdefault("department", doc.get("department", ""))

        docs_text.append(text)
        metadatas.append(meta)
        ids.append(doc.get("id", f"faculty_{i}"))

    BATCH = 100
    t0 = time.time()
    for start in range(0, len(docs_text), BATCH):
        end = min(start + BATCH, len(docs_text))
        collection.add(
            documents=docs_text[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        print(f"  Inserted {end}/{len(docs_text)}...", end="\r")

    elapsed = time.time() - t0
    print(f"\n  Done: {collection.count()} faculty docs in {elapsed:.1f}s")


if __name__ == "__main__":
    vectordb_path = str(ROOT / "data" / "vectordb")
    print(f"Vector DB path: {vectordb_path}")
    print("Starting re-ingestion with text-embedding-3-large...")

    reingest_babyjay_knowledge(vectordb_path)
    reingest_faculty(vectordb_path)

    print("\nRe-ingestion complete.")
    print("Run your tests to verify retrieval quality improved.")
