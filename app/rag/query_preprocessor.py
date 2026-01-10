"""
Query Preprocessor for BabyJay
==============================
Handles query normalization, synonym expansion, and typo correction.

Pipeline:
    1. Input validation & normalization
    2. Subject/course code detection (protect from correction)
    3. Manual synonym expansion
    4. Fuzzy typo correction
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from rapidfuzz import fuzz, process


class QueryPreprocessor:
    """Preprocesses user queries before search."""
    
    def __init__(self, valid_subject_codes: Set[str] = None):
        """
        Initialize preprocessor.
        
        Args:
            valid_subject_codes: Set of valid subject codes (e.g., {"EECS", "AE", "MATH"})
                                 If None, will be set later via set_subject_codes()
        """
        self.valid_subject_codes = valid_subject_codes or set()
        
        # Common abbreviations and synonyms
        # Format: abbreviation -> expansion
        self.synonyms = {
            # Academic abbreviations
            "prereq": "prerequisite",
            "prereqs": "prerequisites",
            "coreq": "corequisite",
            "coreqs": "corequisites",
            "prof": "professor",
            "profs": "professors",
            "dept": "department",
            "intro": "introduction",
            "adv": "advanced",
            "grad": "graduate",
            "undergrad": "undergraduate",
            "lab": "laboratory",
            "sem": "seminar",
            "lec": "lecture",
            "rec": "recitation",
            
            # Subject abbreviations (only expand if NOT a valid subject code)
            "ml": "machine learning",
            "ai": "artificial intelligence",
            "dl": "deep learning",
            "nlp": "natural language processing",
            "cv": "computer vision",
            "ds": "data science",
            "cs": "computer science",
            "ee": "electrical engineering",
            "ce": "civil engineering",
            "me": "mechanical engineering",
            "tic": "aerospace engineering",
            "bio": "biology",
            "chem": "chemistry",
            "phys": "physics",
            "psych": "psychology",
            "econ": "economics",
            "poli sci": "political science",
            "polisci": "political science",
            "comm": "communication",
            "calc": "calculus",
            "stats": "statistics",
            "stat": "statistics",
            "orgo": "organic chemistry",
            "ochem": "organic chemistry",
            "biochem": "biochemistry",
            "tic": "aerospace",
            
            # Common student slang
            "class": "course",
            "classes": "courses",
        }
        
        # Words that should NOT be corrected (common valid words)
        self.protected_words = {
            "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
            "is", "are", "was", "were", "be", "been", "being",
            "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
            "all", "any", "some", "no", "not", "can", "could", "will", "would",
            "should", "may", "might", "must", "do", "does", "did",
            "i", "me", "my", "you", "your", "he", "she", "it", "we", "they",
            "this", "that", "these", "those", "here", "there",
            "about", "with", "from", "by", "as", "into", "through",
            "course", "courses", "class", "classes", "credit", "credits",
            "hour", "hours", "level", "department", "school", "college",
            "undergraduate", "graduate", "freshman", "sophomore", "junior", "senior",
            "fall", "spring", "summer", "winter", "semester",
            "easy", "hard", "difficult", "simple", "basic", "advanced",
            "online", "hybrid", "person",
        }
        
        # Common words in course titles/descriptions for fuzzy matching
        self.course_vocabulary = {
            "introduction", "intermediate", "advanced", "principles",
            "fundamentals", "foundations", "theory", "practice",
            "analysis", "design", "systems", "methods", "applications",
            "programming", "engineering", "science", "mathematics",
            "learning", "machine", "artificial", "intelligence",
            "data", "structures", "algorithms", "networks", "security",
            "database", "software", "hardware", "computer", "computing",
            "physics", "chemistry", "biology", "psychology", "economics",
            "history", "philosophy", "literature", "writing", "research",
            "calculus", "algebra", "geometry", "statistics", "probability",
            "organic", "inorganic", "biochemistry", "molecular", "cellular",
            "mechanics", "dynamics", "thermodynamics", "electronics",
            "communication", "media", "journalism", "business", "management",
            "accounting", "finance", "marketing", "economics", "law",
            "music", "art", "theatre", "dance", "film",
            "health", "nursing", "pharmacy", "medicine",
            "education", "teaching", "curriculum", "leadership",
            "social", "political", "international", "global",
            "environmental", "sustainability", "climate", "energy",
        }
    
    def set_subject_codes(self, codes: Set[str]):
        """Set valid subject codes (call after loading course data)."""
        self.valid_subject_codes = {c.upper() for c in codes}
    
    def preprocess(self, query: str) -> Dict:
        """
        Full preprocessing pipeline.
        
        Args:
            query: Raw user query
            
        Returns:
            Dict with:
                - original: Original query
                - normalized: After normalization
                - processed: After all corrections
                - corrections: List of corrections made
                - detected_codes: Any subject/course codes found
        """
        result = {
            "original": query,
            "normalized": "",
            "processed": "",
            "corrections": [],
            "detected_codes": [],
        }
        
        # Step 1: Input validation & normalization
        normalized = self._normalize(query)
        result["normalized"] = normalized
        
        if not normalized:
            result["processed"] = ""
            return result
        
        # Step 2: Detect and protect subject/course codes
        tokens, detected_codes = self._detect_codes(normalized)
        result["detected_codes"] = detected_codes
        
        # Step 3: Apply synonym expansion
        tokens, synonym_corrections = self._apply_synonyms(tokens)
        result["corrections"].extend(synonym_corrections)
        
        # Step 4: Apply fuzzy typo correction
        tokens, typo_corrections = self._apply_fuzzy_correction(tokens)
        result["corrections"].extend(typo_corrections)
        
        # Reconstruct query
        result["processed"] = " ".join(tokens)
        
        return result
    
    def _normalize(self, query: str) -> str:
        """Normalize input: lowercase, remove special chars, normalize whitespace."""
        if not query or not query.strip():
            return ""
        
        # Remove emojis and special characters (keep alphanumeric, spaces, hyphens)
        cleaned = re.sub(r'[^\w\s\-]', ' ', query)
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Lowercase (but we'll preserve original case for subject codes later)
        return cleaned.lower()
    
    def _detect_codes(self, query: str) -> Tuple[List[str], List[str]]:
        """
        Detect subject codes and course codes in query.
        
        Returns:
            Tuple of (tokens with codes marked, list of detected codes)
        """
        tokens = query.split()
        detected_codes = []
        processed_tokens = []
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            token_upper = token.upper()
            
            # Check for course code pattern: "EECS 168" or "EECS168"
            course_match = re.match(r'^([a-zA-Z]{2,4})(\d{3,4})$', token, re.IGNORECASE)
            if course_match:
                # Combined format like "EECS168"
                subject = course_match.group(1).upper()
                number = course_match.group(2)
                detected_codes.append(f"{subject} {number}")
                processed_tokens.append(f"{subject} {number}")
                i += 1
                continue
            
            # Check if current token is subject code and next is number
            if token_upper in self.valid_subject_codes:
                detected_codes.append(token_upper)
                if i + 1 < len(tokens) and tokens[i + 1].isdigit():
                    # Course code like "EECS 168"
                    number = tokens[i + 1]
                    detected_codes.append(f"{token_upper} {number}")
                    processed_tokens.append(f"{token_upper} {number}")
                    i += 2
                    continue
                else:
                    # Just subject code
                    processed_tokens.append(token_upper)
                    i += 1
                    continue
            
            # Check if it looks like a subject code (2-4 letters) but might be typo
            if re.match(r'^[a-zA-Z]{2,4}$', token) and token_upper not in self.valid_subject_codes:
                # Could be typo of subject code - check fuzzy match
                best_match = self._fuzzy_match_subject_code(token_upper)
                if best_match:
                    detected_codes.append(best_match)
                    processed_tokens.append(best_match)
                    i += 1
                    continue
            
            processed_tokens.append(token)
            i += 1
        
        return processed_tokens, detected_codes
    
    def _fuzzy_match_subject_code(self, token: str) -> Optional[str]:
        """Try to fuzzy match a token to a valid subject code."""
        if not self.valid_subject_codes:
            return None
        
        # Only try if token looks like it could be a subject code
        if not re.match(r'^[A-Z]{2,5}$', token):
            return None
        
        # Find best match among subject codes
        best_match, score, _ = process.extractOne(
            token, 
            list(self.valid_subject_codes),
            scorer=fuzz.ratio
        )
        
        # High threshold for subject codes (they're short, need high similarity)
        if score >= 80:
            return best_match
        
        return None
    
    def _apply_synonyms(self, tokens: List[str]) -> Tuple[List[str], List[Dict]]:
        """Apply manual synonym expansion."""
        corrections = []
        processed = []
        
        i = 0
        while i < len(tokens):
            token = tokens[i].lower()
            
            # Skip if it's a detected subject code (uppercase)
            if tokens[i].isupper() and len(tokens[i]) <= 5:
                processed.append(tokens[i])
                i += 1
                continue
            
            # Check for multi-word synonyms first (e.g., "poli sci")
            if i + 1 < len(tokens):
                two_word = f"{token} {tokens[i+1].lower()}"
                if two_word in self.synonyms:
                    expansion = self.synonyms[two_word]
                    corrections.append({
                        "type": "synonym",
                        "original": two_word,
                        "corrected": expansion
                    })
                    processed.extend(expansion.split())
                    i += 2
                    continue
            
            # Check single word synonyms
            if token in self.synonyms:
                # But don't expand if it's a valid subject code
                if token.upper() in self.valid_subject_codes:
                    processed.append(token.upper())
                else:
                    expansion = self.synonyms[token]
                    corrections.append({
                        "type": "synonym",
                        "original": token,
                        "corrected": expansion
                    })
                    processed.extend(expansion.split())
                i += 1
                continue
            
            processed.append(tokens[i])
            i += 1
        
        return processed, corrections
    
    def _apply_fuzzy_correction(self, tokens: List[str]) -> Tuple[List[str], List[Dict]]:
        """Apply fuzzy typo correction."""
        corrections = []
        processed = []
        
        for token in tokens:
            # Skip protected words, subject codes, and numbers
            if (token.lower() in self.protected_words or 
                token.isupper() or 
                token.isdigit() or
                len(token) <= 2):
                processed.append(token)
                continue
            
            # Skip if already a valid course vocabulary word
            if token.lower() in self.course_vocabulary:
                processed.append(token)
                continue
            
            # Try fuzzy match against course vocabulary
            corrected = self._fuzzy_correct_word(token)
            if corrected and corrected != token.lower():
                corrections.append({
                    "type": "typo",
                    "original": token,
                    "corrected": corrected
                })
                processed.append(corrected)
            else:
                processed.append(token)
        
        return processed, corrections
    
    def _fuzzy_correct_word(self, word: str) -> Optional[str]:
        """Try to correct a single word using fuzzy matching."""
        word_lower = word.lower()
        
        # Find best match in vocabulary
        result = process.extractOne(
            word_lower,
            list(self.course_vocabulary),
            scorer=fuzz.ratio
        )
        
        if result:
            match, score, _ = result
            # High threshold to avoid false corrections
            if score >= 85 and match != word_lower:
                return match
        
        return word_lower


# Quick test
if __name__ == "__main__":
    print("Testing Query Preprocessor")
    print("=" * 60)
    
    # Simulate valid subject codes
    valid_codes = {"EECS", "AE", "MATH", "PHSX", "CHEM", "BIOL", "PSYC", "ECON", "CS", "ME", "CE"}
    
    preprocessor = QueryPreprocessor(valid_codes)
    
    test_queries = [
        # Basic queries
        ("EECS courses", "Should detect EECS as subject code"),
        ("AE 345", "Should detect as course code"),
        ("EECS168", "Should parse combined course code"),
        
        # Synonyms
        ("ML prereqs", "Should expand ML and prereqs"),
        ("intro to AI", "Should expand intro and AI"),
        ("CS classes", "Should expand CS and classes"),
        ("grad courses", "Should expand grad"),
        
        # Typos
        ("machien learning", "Should correct machien"),
        ("introducton to programming", "Should correct introducton"),
        ("compter science", "Should correct compter"),
        
        # Combined
        ("prereqs for machien lerning", "Should fix typos and expand prereqs"),
        ("EESC courses", "Should correct EESC to EECS"),
        
        # Edge cases
        ("", "Empty query"),
        ("   ", "Whitespace only"),
        ("🎓 courses", "Emoji"),
        ("the course", "Common words - no correction"),
    ]
    
    for query, description in test_queries:
        print(f"\nQuery: '{query}'")
        print(f"  ({description})")
        
        result = preprocessor.preprocess(query)
        
        print(f"  Normalized: '{result['normalized']}'")
        print(f"  Processed:  '{result['processed']}'")
        
        if result['corrections']:
            print(f"  Corrections: {result['corrections']}")
        if result['detected_codes']:
            print(f"  Detected codes: {result['detected_codes']}")
    
    print("\n" + "=" * 60)
    print("All tests complete!")