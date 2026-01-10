"""
Faculty Search Module for BabyJay
==================================
Semantic search across 2,207 KU faculty members using ChromaDB with OpenAI embeddings.

USAGE:
    from app.rag.faculty_search import FacultySearcher
    
    searcher = FacultySearcher()
    results = searcher.search("AI machine learning research")
    
    for result in results:
        print(f"{result['name']} - {result['department']}")
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional

# Get data directory path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')


class FacultySearcher:
    """
    Semantic search engine for KU faculty members using ChromaDB with OpenAI embeddings.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the faculty searcher.
        
        Args:
            data_dir: Directory containing vectordb (defaults to BabyJay/data/)
        """
        self.data_dir = data_dir if data_dir else DATA_DIR
        self.client = None
        self.collection = None
        self._connect()
    
    def _connect(self):
        """Connect to ChromaDB and get faculty collection."""
        vectordb_path = os.path.join(self.data_dir, 'vectordb')
        self.client = chromadb.PersistentClient(path=vectordb_path)
        
        # Get the faculty collection with OpenAI embedding function
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        
        self.collection = self.client.get_collection(
            name="faculty",
            embedding_function=openai_ef
        )
    
    def search(self, query: str, top_k: int = 5, 
               department_filter: Optional[str] = None) -> List[Dict]:
        """
        Search for faculty members matching the query.
        
        Args:
            query: Search query (name, research interest, topic, etc.)
            top_k: Number of results to return
            department_filter: Optional department to filter by
            
        Returns:
            List of matching faculty with scores
        """
        # If department filter specified, get more results and filter after
        fetch_k = top_k * 10 if department_filter else top_k
        
        # Query ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results and results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results['distances'] else 0
                
                # Apply department filter (case-insensitive partial match)
                if department_filter:
                    dept = metadata.get('department', '').lower()
                    if department_filter.lower() not in dept:
                        continue
                
                # Convert distance to similarity score (lower distance = higher similarity)
                score = 1 / (1 + distance)
                
                formatted_results.append({
                    'id': doc_id,
                    'name': metadata.get('name', ''),
                    'department': metadata.get('department', ''),
                    'email': metadata.get('email', ''),
                    'office': metadata.get('office', ''),
                    'phone': metadata.get('phone', ''),
                    'building': metadata.get('building', ''),
                    'profile_url': metadata.get('profile_url', ''),
                    'score': score,
                    'document': results['documents'][0][i] if results.get('documents') else ''
                })
                
                # Stop if we have enough results
                if len(formatted_results) >= top_k:
                    break
        
        return formatted_results
    
    def get_faculty_by_name(self, name: str) -> Optional[Dict]:
        """
        Find a specific faculty member by name.
        
        Args:
            name: Faculty member's name (partial match supported)
            
        Returns:
            Faculty info dict or None if not found
        """
        results = self.search(name, top_k=1)
        if results and name.lower() in results[0]['name'].lower():
            return results[0]
        return None
    
    def get_department_faculty(self, department: str, limit: int = 50) -> List[Dict]:
        """
        Get all faculty in a specific department.
        
        Args:
            department: Department name (partial match)
            limit: Maximum number to return
            
        Returns:
            List of faculty in that department
        """
        # Query with department filter
        results = self.collection.query(
            query_texts=[department],
            n_results=limit,
            include=["metadatas"]
        )
        
        faculty_list = []
        if results and results['metadatas'] and results['metadatas'][0]:
            for metadata in results['metadatas'][0]:
                if department.lower() in metadata.get('department', '').lower():
                    faculty_list.append({
                        'name': metadata.get('name', ''),
                        'department': metadata.get('department', ''),
                        'email': metadata.get('email', ''),
                        'office': metadata.get('office', '')
                    })
        
        return faculty_list
    
    def stats(self) -> Dict:
        """Get statistics about the faculty database."""
        count = self.collection.count()
        return {
            'total_faculty': count,
            'collection_name': 'faculty',
            'storage': 'ChromaDB',
            'embeddings': 'OpenAI text-embedding-3-small'
        }


# Quick test
if __name__ == '__main__':
    print("Testing Faculty Search Module...")
    print("=" * 60)
    
    # Initialize searcher
    searcher = FacultySearcher()
    
    # Print stats
    stats = searcher.stats()
    print(f"\nDatabase Stats:")
    print(f"  Total Faculty: {stats['total_faculty']}")
    print(f"  Embeddings: {stats['embeddings']}")
    
    # Test searches
    test_queries = [
        "machine learning artificial intelligence",
        "deep learning neural networks",
        "particle physics",
        "environmental sustainability",
        "social work mental health"
    ]
    
    print("\n" + "=" * 60)
    print("Sample Searches:")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = searcher.search(query, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['name']}")
            print(f"     Dept: {r['department']}")
            print(f"     Score: {r['score']:.4f}")
    
    # Test department filtering
    print("\n" + "=" * 60)
    print("Department Filter Tests:")
    print("=" * 60)
    
    print("\nQuery: 'machine learning' (EECS only)")
    results = searcher.search("machine learning", top_k=5, department_filter="Electrical")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['name']} - {r['department']} (score: {r['score']:.4f})")
    
    print("\nQuery: 'deep learning' (EECS only)")
    results = searcher.search("deep learning", top_k=5, department_filter="Electrical")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['name']} - {r['department']} (score: {r['score']:.4f})")