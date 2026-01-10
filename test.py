"""
Test Fix #2: LLM-Based Query Cleaning
======================================
Tests that LLM cleans typos, grammar, and text speak
"""

import sys
sys.path.append('/Users/pavansaipendry/Documents/BabyJay')

from app.rag.chat import BabyJayChat

print("="*80)
print("TESTING FIX #2: LLM-BASED QUERY CLEANING")
print("="*80)

# Test cases: (messy_query, description, should_find_results)
test_cases = [
    # Typos
    ("machien learning professors", "ML typo", True),
    ("artifical intelligence faculty", "AI typo", True),
    ("robtics researchers", "robotics typo", True),
    ("quantim computing", "quantum typo", True),
    
    # Text speak
    ("hey i wnt 2 noe abt ml", "text speak", True),
    ("yo wat r the ai profesors", "casual + typos", True),
    ("can u tell me abt quantim stuff", "u/abt abbreviations", True),
    
    # Grammar issues
    ("professor who doing machine learning", "grammar", True),
    ("where i can found robotics faculty", "grammar", True),
    
    # Multiple issues combined
    ("hey i wnt 2 noe abt machien leraning profesors", "multiple issues", True),
    ("sup hw can i find quantim compting reseachers", "extreme mess", True),
    
    # Mixed case and punctuation
    ("MACHIEN LEARNING PROFESSORS???", "caps + punctuation", True),
    ("artifical intelligence!!!", "punctuation", True),
    
    # Natural language (should not over-clean)
    ("tell me about machine learning professors", "clean query", True),
    ("who are the AI researchers", "clean query", True),
]

chat = BabyJayChat(use_redis=False, debug=True)

results = {
    "passed": 0,
    "failed": 0,
    "errors": [],
    "cleaned_examples": []
}

print("\nRunning tests with debug ON to see LLM cleaning in action...\n")

for i, (messy_query, description, should_find) in enumerate(test_cases, 1):
    print(f"\n{'='*80}")
    print(f"[Test {i}/{len(test_cases)}] {description}")
    print(f"{'='*80}")
    print(f"Original Query: '{messy_query}'")
    
    try:
        # The LLM cleaning happens inside ask()
        response = chat.ask(messy_query)
        
        # Check if we got results
        has_results = len(response) > 200
        
        if should_find and has_results:
            print(f"✓ PASS - Found results ({len(response)} chars)")
            print(f"  Response preview: {response[:150]}...")
            results["passed"] += 1
        elif not should_find and not has_results:
            print(f"✓ PASS - Correctly returned minimal results")
            results["passed"] += 1
        else:
            print(f"✗ FAIL - Unexpected result length: {len(response)}")
            results["failed"] += 1
            results["errors"].append((messy_query, f"Expected results: {should_find}, Got: {has_results}"))
            
    except Exception as e:
        print(f"✗ FAIL - Exception: {e}")
        results["failed"] += 1
        results["errors"].append((messy_query, str(e)))

print("\n" + "="*80)
print("TEST RESULTS")
print("="*80)
print(f"Passed: {results['passed']}/{len(test_cases)}")
print(f"Failed: {results['failed']}/{len(test_cases)}")

if results["errors"]:
    print(f"\nErrors:")
    for query, error in results["errors"]:
        print(f"  '{query}': {error}")
else:
    print(f"\n✓ ALL TESTS PASSED!")

print("\n" + "="*80)
print("CLEANING EXAMPLES (from debug output above)")
print("="*80)
print("Look for lines like:")
print("  [DEBUG] LLM cleaned: 'messy query' → 'clean query'")
print("="*80)