"""
Diagnostic script to verify faculty search quality
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.rag.faculty_search import FacultySearcher

print("=" * 80)
print("FACULTY SEARCH QUALITY DIAGNOSTIC")
print("=" * 80)

searcher = FacultySearcher()

# Test 1: Check what's actually stored for a professor
print("\n" + "=" * 80)
print("TEST 1: What's stored for 'Sankha Guria'?")
print("=" * 80)

results = searcher.search("Sankha Guria", top_k=1)
if results:
    r = results[0]
    print(f"Name: {r['name']}")
    print(f"Department: {r['department']}")
    print(f"Email: {r['email']}")
    print(f"\nDocument content (first 1000 chars):")
    print("-" * 80)
    print(r.get('document', 'NO DOCUMENT STORED')[:1000])
    print("-" * 80)
else:
    print("Not found!")

# Test 2: Search for "machine learning" and check document content
print("\n" + "=" * 80)
print("TEST 2: Search 'machine learning' - check if docs contain ML content")
print("=" * 80)

ml_results = searcher.search("machine learning", top_k=5)
for i, r in enumerate(ml_results, 1):
    print(f"\n{i}. {r['name']} - {r['department']}")
    print(f"   Score: {r['score']:.4f}")
    doc = r.get('document', '')
    if doc:
        # Check if document actually mentions ML-related terms
        ml_terms = ['machine learning', 'deep learning', 'neural', 'ai', 'artificial intelligence', 
                    'data science', 'computer vision', 'natural language']
        found_terms = [term for term in ml_terms if term.lower() in doc.lower()]
        
        if found_terms:
            print(f"   ✓ Contains ML terms: {', '.join(found_terms)}")
            # Show snippet with ML content
            for term in found_terms[:2]:
                idx = doc.lower().find(term.lower())
                if idx != -1:
                    snippet = doc[max(0, idx-50):min(len(doc), idx+100)]
                    print(f"   Snippet: ...{snippet}...")
                    break
        else:
            print(f"   ✗ NO ML terms found in document!")
            print(f"   First 200 chars: {doc[:200]}")
    else:
        print(f"   ✗ NO DOCUMENT CONTENT!")

# Test 3: Search for "deep learning EECS" specifically
print("\n" + "=" * 80)
print("TEST 3: Search 'deep learning EECS' with department filter")
print("=" * 80)

eecs_results = searcher.search("deep learning", top_k=5, department_filter="Electrical")
for i, r in enumerate(eecs_results, 1):
    print(f"\n{i}. {r['name']} - {r['department']}")
    print(f"   Score: {r['score']:.4f}")
    doc = r.get('document', '')
    
    # Check for deep learning mentions
    if 'deep learning' in doc.lower():
        print(f"   ✓ Contains 'deep learning'")
        idx = doc.lower().find('deep learning')
        snippet = doc[max(0, idx-100):min(len(doc), idx+150)]
        print(f"   Context: ...{snippet}...")
    elif 'machine learning' in doc.lower():
        print(f"   ~ Contains 'machine learning' (not deep learning)")
    else:
        print(f"   ✗ Does NOT contain 'deep learning' or 'machine learning'")
        print(f"   First 300 chars: {doc[:300]}")

# Test 4: Who actually does deep learning?
print("\n" + "=" * 80)
print("TEST 4: Finding professors who ACTUALLY mention 'deep learning' in their bio")
print("=" * 80)

# Get more results and filter manually
all_results = searcher.collection.get(
    include=["documents", "metadatas"],
    limit=2207  # Get all faculty
)

dl_faculty = []
if all_results and all_results.get('documents'):
    for i, doc in enumerate(all_results['documents']):
        if 'deep learning' in doc.lower():
            metadata = all_results['metadatas'][i]
            dl_faculty.append({
                'name': metadata.get('name', 'Unknown'),
                'department': metadata.get('department', 'Unknown'),
                'email': metadata.get('email', ''),
                'snippet': doc[max(0, doc.lower().find('deep learning')-50):
                               min(len(doc), doc.lower().find('deep learning')+100)]
            })

print(f"\nFound {len(dl_faculty)} professors with 'deep learning' in their profile:")
for i, prof in enumerate(dl_faculty[:10], 1):  # Show first 10
    print(f"\n{i}. {prof['name']} - {prof['department']}")
    print(f"   Email: {prof['email']}")
    print(f"   Snippet: ...{prof['snippet']}...")

if len(dl_faculty) == 0:
    print("\n⚠️ WARNING: NO professors have 'deep learning' in their stored documents!")
    print("This means the faculty documents may not contain research interests.")

# Test 5: Check database stats
print("\n" + "=" * 80)
print("TEST 5: Database Statistics")
print("=" * 80)

stats = searcher.stats()
print(f"Total faculty: {stats['total_faculty']}")

# Sample a few random documents to see what's stored
print("\nSample of 5 random faculty documents:")
sample = searcher.collection.get(
    include=["documents", "metadatas"],
    limit=5
)

if sample and sample.get('documents'):
    for i, doc in enumerate(sample['documents'], 1):
        meta = sample['metadatas'][i-1]
        print(f"\n{i}. {meta.get('name', 'Unknown')} - {meta.get('department', 'Unknown')}")
        print(f"   Document length: {len(doc)} chars")
        print(f"   First 200 chars: {doc[:200]}")