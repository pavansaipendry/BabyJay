"""
Regenerate faculty embeddings using OpenAI embeddings
=======================================================
This will replace the ChromaDB default embeddings with OpenAI embeddings,
which are much better at understanding semantic relationships like:
- "deep learning" → finds "machine learning", "neural networks", "AI"
"""

import os
import sys
import json
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

# Paths - handle being in app/rag/ directory
SCRIPT_DIR = Path(__file__).parent.resolve()

# If script is in app/rag/, go up 2 levels to project root
if SCRIPT_DIR.name == 'rag' and SCRIPT_DIR.parent.name == 'app':
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
elif SCRIPT_DIR.name == 'scripts':
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
else:
    # Script is at project root
    PROJECT_ROOT = SCRIPT_DIR

DATA_DIR = PROJECT_ROOT / 'data'
VECTORDB_PATH = DATA_DIR / 'vectordb'
FACULTY_JSON = DATA_DIR / 'faculty_documents.json'

def main():
    print("=" * 80)
    print("REGENERATING FACULTY EMBEDDINGS WITH OPENAI")
    print("=" * 80)
    
    # 1. Load faculty documents
    print(f"\n1. Loading faculty data from {FACULTY_JSON}...")
    with open(FACULTY_JSON, 'r') as f:
        faculty_data = json.load(f)
    print(f"   Loaded {len(faculty_data)} faculty members")
    
    # 2. Connect to ChromaDB
    print(f"\n2. Connecting to ChromaDB at {VECTORDB_PATH}...")
    client = chromadb.PersistentClient(path=str(VECTORDB_PATH))
    
    # 3. Delete old faculty collection
    print("\n3. Deleting old 'faculty' collection...")
    try:
        client.delete_collection("faculty")
        print("   ✓ Deleted old collection")
    except Exception as e:
        print(f"   (Collection didn't exist, creating new one)")
    
    # 4. Create new collection with OpenAI embeddings
    print("\n4. Creating new 'faculty' collection with OpenAI embeddings...")
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    
    collection = client.create_collection(
        name="faculty",
        embedding_function=openai_ef,
        metadata={"description": "KU Faculty with OpenAI embeddings"}
    )
    print("   ✓ Created collection")
    
    # 5. Prepare documents
    print("\n5. Preparing documents...")
    documents = []
    metadatas = []
    ids = []
    
    for fac in faculty_data:
        # Use searchable_text which has everything
        documents.append(fac['searchable_text'])
        
        # Metadata
        metadatas.append({
            'name': fac['name'],
            'department': fac['department'],
            'email': fac['email'],
            'office': fac['office'],
            'phone': fac['phone'],
            'building': fac.get('department_building', ''),
            'profile_url': fac['profile_url']
        })
        
        ids.append(fac['id'])
    
    print(f"   Prepared {len(documents)} documents")
    
    # 6. Add to collection in batches
    print("\n6. Adding documents to collection (this will take a few minutes)...")
    batch_size = 100
    
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        batch_num = (i // batch_size) + 1
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        print(f"   Processing batch {batch_num}/{total_batches} ({i+1}-{end_idx})...")
        
        collection.add(
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )
    
    # 7. Verify
    print("\n7. Verifying collection...")
    count = collection.count()
    print(f"   ✓ Collection has {count} documents")
    
    # 8. Test search
    print("\n8. Testing semantic search...")
    test_queries = [
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "robotics"
    ]
    
    for query in test_queries:
        results = collection.query(
            query_texts=[query],
            n_results=3,
            include=["metadatas", "distances"]
        )
        
        print(f"\n   Query: '{query}'")
        if results and results['metadatas'] and results['metadatas'][0]:
            for j, meta in enumerate(results['metadatas'][0], 1):
                distance = results['distances'][0][j-1]
                score = 1 / (1 + distance)
                print(f"      {j}. {meta['name']} - {meta['department']} (score: {score:.4f})")
        else:
            print("      No results")
    
    print("\n" + "=" * 80)
    print("✓ DONE! Faculty collection now uses OpenAI embeddings")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Test with: python -m app.rag.chat")
    print("  2. Ask: 'Find me professors doing deep learning research'")
    print("  3. Should now find ML/AI professors correctly!")
    

if __name__ == "__main__":
    main()