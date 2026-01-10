"""
Faculty Embeddings Generator for BabyJay
=========================================
Uses ChromaDB (already installed) instead of sentence-transformers.

USAGE (from BabyJay root folder):
    python app/scripts/generate_faculty_embeddings.py
"""

import json
import os
import chromadb
from chromadb.utils import embedding_functions

# Get the project root directory (BabyJay/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

def load_faculty_documents():
    """Load the faculty documents JSON file."""
    filepath = os.path.join(DATA_DIR, 'faculty_documents.json')
    print(f"Loading from: {filepath}")
    with open(filepath, 'r') as f:
        return json.load(f)

def generate_and_store_embeddings(documents):
    """Generate embeddings using ChromaDB's built-in embedding function."""
    
    # Use ChromaDB's default embedding function
    embedding_func = embedding_functions.DefaultEmbeddingFunction()
    
    # Create/connect to ChromaDB
    vectordb_path = os.path.join(DATA_DIR, 'vectordb')
    client = chromadb.PersistentClient(path=vectordb_path)
    
    # Delete existing faculty collection if it exists
    try:
        client.delete_collection("faculty")
        print("  Deleted existing faculty collection")
    except:
        pass
    
    # Create new faculty collection
    collection = client.create_collection(
        name="faculty",
        embedding_function=embedding_func,
        metadata={"description": "KU Faculty members"}
    )
    
    print(f"  Created new faculty collection")
    print(f"  Processing {len(documents)} faculty members...")
    
    # Prepare data with UNIQUE IDs
    ids = []
    documents_text = []
    metadatas = []
    seen_ids = set()
    duplicates = 0
    
    for idx, doc in enumerate(documents):
        # Create unique ID using index to guarantee uniqueness
        base_id = doc.get('id', f'faculty_{idx}')
        unique_id = base_id
        
        # If duplicate, append index
        if unique_id in seen_ids:
            unique_id = f"{base_id}_{idx}"
            duplicates += 1
        
        seen_ids.add(unique_id)
        ids.append(unique_id)
        documents_text.append(doc.get('searchable_text', ''))
        metadatas.append({
            'name': doc.get('name', ''),
            'department': doc.get('department', ''),
            'email': doc.get('email', ''),
            'office': doc.get('office', ''),
            'phone': doc.get('phone', ''),
            'building': doc.get('department_building', ''),
            'profile_url': doc.get('profile_url', '')
        })
    
    if duplicates > 0:
        print(f"  Note: Fixed {duplicates} duplicate IDs")
    
    # Add in batches (ChromaDB handles embedding automatically)
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end],
            documents=documents_text[i:end],
            metadatas=metadatas[i:end]
        )
        print(f"  Added {end}/{len(ids)} faculty members...")
    
    return len(ids)

def main():
    print("=" * 60)
    print("BabyJay Faculty Embeddings Generator (ChromaDB)")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    
    # Load documents
    print("\n[1/2] Loading faculty documents...")
    documents = load_faculty_documents()
    print(f"  Loaded {len(documents)} faculty members")
    
    # Generate and store embeddings
    print("\n[2/2] Generating embeddings and storing in ChromaDB...")
    count = generate_and_store_embeddings(documents)
    
    print("\n" + "=" * 60)
    print(f"✓ DONE! Added {count} faculty members to ChromaDB")
    print("=" * 60)

if __name__ == '__main__':
    main()