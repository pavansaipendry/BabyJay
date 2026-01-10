"""
Debug: Why isn't retriever finding ML professors?
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.rag.retriever import Retriever

query = "Find ML professors"

print("=" * 80)
print(f"DEBUGGING: '{query}'")
print("=" * 80)

r = Retriever()

# Check if faculty search is even initialized
print(f"\n1. Faculty searcher exists: {hasattr(r, 'faculty_searcher')}")
print(f"   Type: {type(r.faculty_searcher) if hasattr(r, 'faculty_searcher') else 'N/A'}")

# Check what smart_search detects
print(f"\n2. Running smart_search...")
results = r.smart_search(query, n_results=5)

print(f"\n3. Results breakdown:")
print(f"   Dining: {len(results.get('dining', []))}")
print(f"   Transit: {len(results.get('transit', []))}")
print(f"   Courses: {len(results.get('courses', []))}")
print(f"   Faculty: {len(results.get('faculty', []))}")
print(f"   Context length: {len(results.get('context', ''))}")

if results.get('faculty'):
    print(f"\n4. Faculty found:")
    for i, f in enumerate(results['faculty'], 1):
        meta = f.get('metadata', {})
        print(f"   {i}. {meta.get('name')} - {meta.get('department')}")
else:
    print(f"\n4. NO FACULTY FOUND!")
    print(f"   Context preview:")
    print(f"   {results.get('context', '')[:300]}")

# Check if the intent flags are working
print(f"\n5. Testing intent detection manually:")
q = query.lower()
is_professor = any(w in q for w in ['professor', 'prof', 'faculty', 'teacher', 'instructor'])
is_research = any(w in q for w in ['machine learning', 'ml ', 'research'])

print(f"   Query lowercased: '{q}'")
print(f"   Contains 'professor' keywords: {is_professor}")
print(f"   Contains 'ml' or 'machine learning': {is_research}")

# Test faculty search directly
print(f"\n6. Testing faculty_searcher directly:")
try:
    direct_results = r.faculty_searcher.search("machine learning", top_k=5)
    print(f"   Direct search found {len(direct_results)} professors:")
    for i, prof in enumerate(direct_results[:3], 1):
        print(f"   {i}. {prof['name']} - {prof['department']}")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)

if len(results.get('faculty', [])) == 0 and len(direct_results) > 0:
    print("❌ Faculty searcher WORKS but retriever ISN'T CALLING IT!")
    print("   Problem: Intent detection in smart_search() is broken")
elif len(results.get('faculty', [])) > 0:
    print("✓ Working correctly")
else:
    print("❌ Faculty searcher itself is broken")