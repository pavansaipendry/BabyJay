"""
Test file to verify the complete list fix before implementing
"""
from app.rag.retriever import Retriever

r = Retriever()

# Test: Detect "all/complete/every" in query
def wants_complete_list(query: str) -> bool:
    q = query.lower()
    complete_indicators = [
        'all ', 'every ', 'complete list', 'full list', 
        'list all', 'show all', 'all of the', 'how many'
    ]
    return any(indicator in q for indicator in complete_indicators)

# Test cases
test_queries = [
    "EECS professors",           # False - normal query
    "all EECS professors",       # True - wants complete list
    "show me all ML faculty",    # True - wants complete list
    "every professor in physics",# True - wants complete list
    "list all researchers",      # True - wants complete list
    "who does ML research",      # False - normal query
]

print("Testing complete list detection:")
for q in test_queries:
    result = wants_complete_list(q)
    print(f"  '{q}' -> {result}")
