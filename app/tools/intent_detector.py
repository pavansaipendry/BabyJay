"""
Intent Detection using Embeddings
==================================
Detects if a query needs live course data and extracts the topic.
Falls back to regex if embedding fails.

Usage:
    detector = LiveCourseIntentDetector()
    result = detector.detect("seats available for Deep Reinforcement Learning?")
    # Returns: {"needs_live": True, "intent": "seats", "topic": "Deep Reinforcement Learning", "confidence": 0.87}
"""

import os
import re
import json
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path

# Try to import OpenAI for embeddings
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()


# ==================== INTENT DEFINITIONS ====================
# These are the "anchor" phrases for each intent type
# We'll pre-compute embeddings for these

LIVE_COURSE_INTENTS = {
    "instructor": [
        "who teaches this course",
        "who is teaching",
        "who is the instructor",
        "who is the professor",
        "which professor teaches",
        "instructor for the course",
        "taught by whom",
        "who's teaching",
    ],
    "seats": [
        "are there open seats",
        "how many seats available",
        "is there space in the class",
        "can I enroll",
        "is the class full",
        "seat availability",
        "open spots in course",
        "any seats left",
        "is it open for enrollment",
    ],
    "schedule": [
        "when is the class",
        "what time does it meet",
        "class schedule",
        "what days does it meet",
        "meeting time for course",
        "when does the course meet",
    ],
    "sections": [
        "how many sections",
        "what sections are available",
        "different sections of the course",
        "section options",
        "which sections are offered",
    ],
    "general_course_info": [
        "tell me about this course",
        "course information",
        "details about the class",
        "what is this course about",
    ],
}

# Words/phrases to remove when extracting topic
FILLER_WORDS = [
    # Verbal fillers
    "um", "uh", "umm", "uhh", "erm", "er", "like", "so", "well", "actually",
    # Intent phrases
    "i want to know", "i want to", "i need to know", "can you tell me",
    "could you tell me", "please tell me", "i'm looking for", "im looking for",
    "i am looking for", "looking for", "searching for", "tell me about",
    "i want to see", "i would like to know", "id like to know",
    "can i get", "could i get", "show me",
    # Question starters
    "if there are", "are there", "is there", "do you know", "does it have",
    "do they have", "what about", "how about",
    # Articles and misc
    "any", "some", "the", "a", "an", "of", "in", "on", "at",
]

# Course-related words to remove from topic extraction
COURSE_CONTEXT_WORDS = [
    # Course terms
    "course", "class", "classes", "courses", "section", "sections",
    # Availability terms (including typos)
    "seats", "seat", "available", "availability", "open", "spots", "space",
    "avilable", "availble", "avialable", "avalable",  # common typos
    "enroll", "enrollment", "register", "registration", "full",
    # People terms
    "instructor", "professor", "prof", "teacher", "teaches", "teaching", "taught",
    "who", "whom",
    # Time/schedule terms
    "schedule", "time", "when", "where", "how many", "what time",
    # Semester terms
    "spring", "fall", "summer", "winter", "semester", 
    "2024", "2025", "2026", "2027",
    "next", "this", "current", "upcoming",
    # Prepositions that leak through
    "for", "about", "with", "from", "into",
    # Misc words that leak through
    "are", "is", "there", "if", "want", "know",
    # Career level terms (NEW)
    "graduate", "undergraduate", "grad", "undergrad", "level",
    "masters", "master", "phd", "doctoral", "bachelors", "bachelor",
]


class LiveCourseIntentDetector:
    """
    Embedding-based detector for live course lookup intents.
    Uses OpenAI embeddings with fallback to regex.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, debug: bool = False):
        self.debug = debug
        self.client = None
        self.intent_embeddings: Dict[str, np.ndarray] = {}
        self.embedding_model = "text-embedding-3-small"
        self.similarity_threshold = 0.45  # Minimum similarity to trigger
        
        # Cache directory for pre-computed embeddings
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(__file__).parent / "embeddings_cache"
        
        # Initialize OpenAI client
        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
                self._load_or_compute_intent_embeddings()
            else:
                if self.debug:
                    print("[DEBUG] No OpenAI API key found, using regex fallback")
        else:
            if self.debug:
                print("[DEBUG] OpenAI not available, using regex fallback")
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for a single text."""
        if not self.client:
            return None
        
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Embedding error: {e}")
            return None
    
    def _get_embeddings_batch(self, texts: List[str]) -> Optional[List[np.ndarray]]:
        """Get embeddings for multiple texts in one API call."""
        if not self.client:
            return None
        
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            return [np.array(item.embedding) for item in response.data]
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Batch embedding error: {e}")
            return None
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def _load_or_compute_intent_embeddings(self):
        """Load cached intent embeddings or compute them."""
        cache_file = self.cache_dir / "intent_embeddings.json"
        
        # Try to load from cache
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    cached = json.load(f)
                
                # Convert lists back to numpy arrays
                for intent, embedding in cached.items():
                    self.intent_embeddings[intent] = np.array(embedding)
                
                if self.debug:
                    print(f"[DEBUG] Loaded {len(self.intent_embeddings)} intent embeddings from cache")
                return
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Cache load failed: {e}")
        
        # Compute embeddings for each intent
        if self.debug:
            print("[DEBUG] Computing intent embeddings...")
        
        for intent, phrases in LIVE_COURSE_INTENTS.items():
            # Get embeddings for all anchor phrases
            embeddings = self._get_embeddings_batch(phrases)
            if embeddings:
                # Average the embeddings to get a single vector per intent
                self.intent_embeddings[intent] = np.mean(embeddings, axis=0)
        
        # Save to cache
        if self.intent_embeddings:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_data = {k: v.tolist() for k, v in self.intent_embeddings.items()}
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)
            
            if self.debug:
                print(f"[DEBUG] Saved {len(self.intent_embeddings)} intent embeddings to cache")
    
    def _clean_query_for_topic(self, query: str) -> str:
        """Remove filler words and intent phrases to extract topic."""
        text = query.lower().strip()
        
        # Remove punctuation at start and end
        text = text.strip("?!.,;:'\"")
        
        # Remove ellipsis and multiple dots
        text = re.sub(r'\.{2,}', ' ', text)
        text = re.sub(r'\s*\.\s*', ' ', text)
        
        # Remove filler words (as whole words)
        for filler in FILLER_WORDS:
            # Use word boundaries to avoid partial matches
            text = re.sub(rf'\b{re.escape(filler)}\b', ' ', text, flags=re.IGNORECASE)
        
        # Remove course context words (as whole words)
        for word in COURSE_CONTEXT_WORDS:
            text = re.sub(rf'\b{re.escape(word)}\b', ' ', text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove any remaining leading/trailing punctuation or short garbage
        text = re.sub(r'^[\W\d_]+', '', text)  # Leading non-word chars
        text = re.sub(r'[\W\d_]+$', '', text)  # Trailing non-word chars
        
        return text.strip()
    
    def _extract_course_code(self, query: str) -> Optional[str]:
        """Extract course code like EECS 700, MATH 125, etc."""
        match = re.search(r'\b([A-Z]{2,4})\s*(\d{3,4})\b', query, re.IGNORECASE)
        if match:
            return f"{match.group(1).upper()} {match.group(2)}"
        return None
    
    def _extract_topic(self, query: str) -> Optional[str]:
        """Extract the topic/course name from query."""
        # First, try to extract course code
        course_code = self._extract_course_code(query)
        if course_code:
            return course_code
        
        # Clean query to get topic
        topic = self._clean_query_for_topic(query)
        
        # If too short or empty, return None
        if len(topic) < 3:
            return None
        
        # Capitalize properly (title case for topics)
        topic = topic.title()
        
        return topic
    
    def _detect_with_embeddings(self, query: str) -> Optional[Dict]:
        """
        Use embeddings to detect intent.
        Returns dict with intent info or None if no match.
        """
        if not self.intent_embeddings:
            return None
        
        # Get query embedding
        query_embedding = self._get_embedding(query.lower())
        if query_embedding is None:
            return None
        
        # Compare to each intent
        best_intent = None
        best_score = 0.0
        
        for intent, intent_embedding in self.intent_embeddings.items():
            score = self._cosine_similarity(query_embedding, intent_embedding)
            if score > best_score:
                best_score = score
                best_intent = intent
        
        if self.debug:
            print(f"[DEBUG] Embedding match: {best_intent} (score: {best_score:.3f})")
        
        # Check if above threshold
        if best_score >= self.similarity_threshold:
            return {
                "intent": best_intent,
                "confidence": best_score,
                "method": "embedding"
            }
        
        return None
    
    def _detect_with_regex(self, query: str) -> Optional[Dict]:
        """
        Fallback regex-based detection.
        Returns dict with intent info or None if no match.
        """
        q = query.lower()
        
        # Instructor patterns
        if any(p in q for p in ['who teach', 'who is teaching', 'instructor', 'professor for', 'taught by']):
            return {"intent": "instructor", "confidence": 0.8, "method": "regex"}
        
        # Seats patterns
        if any(p in q for p in ['seats', 'enroll', 'is it full', 'is it open', 'space in', 'spots']):
            return {"intent": "seats", "confidence": 0.8, "method": "regex"}
        
        # Schedule patterns
        if any(p in q for p in ['when is', 'what time', 'schedule', 'what days', 'meeting time']):
            return {"intent": "schedule", "confidence": 0.8, "method": "regex"}
        
        # Sections patterns
        if any(p in q for p in ['how many section', 'sections available', 'which section']):
            return {"intent": "sections", "confidence": 0.8, "method": "regex"}
        
        # Course code with question mark (user asking about course)
        if re.match(r'^[A-Z]{2,4}\s*\d{3,4}\s*\??$', query.strip(), re.IGNORECASE):
            return {"intent": "general_course_info", "confidence": 0.9, "method": "regex"}
        
        return None
    
    def detect(self, query: str) -> Dict:
        """
        Main detection method. Uses embeddings with regex fallback.
        
        Returns:
            {
                "needs_live": bool,
                "intent": str or None,
                "topic": str or None,
                "confidence": float,
                "method": "embedding" | "regex" | "none"
            }
        """
        # Default result
        result = {
            "needs_live": False,
            "intent": None,
            "topic": None,
            "confidence": 0.0,
            "method": "none"
        }
        
        # Try embedding-based detection first
        intent_result = self._detect_with_embeddings(query)
        
        # Fall back to regex if embeddings didn't match
        if not intent_result:
            intent_result = self._detect_with_regex(query)
        
        # If we found an intent, extract topic
        if intent_result:
            topic = self._extract_topic(query)
            
            # Only mark as needs_live if we have a topic
            if topic:
                result["needs_live"] = True
                result["intent"] = intent_result["intent"]
                result["topic"] = topic
                result["confidence"] = intent_result["confidence"]
                result["method"] = intent_result["method"]
        
        if self.debug:
            print(f"[DEBUG] Intent detection: {result}")
        
        return result
    
    def clear_cache(self):
        """Clear the embeddings cache."""
        cache_file = self.cache_dir / "intent_embeddings.json"
        if cache_file.exists():
            cache_file.unlink()
        self.intent_embeddings = {}
        self._load_or_compute_intent_embeddings()


# ==================== CONVENIENCE FUNCTION ====================

_detector_instance: Optional[LiveCourseIntentDetector] = None

def detect_live_course_intent(query: str, debug: bool = False) -> Dict:
    """
    Convenience function for detecting live course intent.
    Uses a singleton detector instance.
    """
    global _detector_instance
    
    if _detector_instance is None:
        _detector_instance = LiveCourseIntentDetector(debug=debug)
    
    _detector_instance.debug = debug
    return _detector_instance.detect(query)


# ==================== CLI FOR TESTING ====================

def main():
    """Test the intent detector."""
    print("=" * 60)
    print("Live Course Intent Detector - Test Mode")
    print("=" * 60)
    
    detector = LiveCourseIntentDetector(debug=True)
    
    test_queries = [
        "who teaches EECS 700?",
        "seats available for Deep Reinforcement Learning?",
        "Um.. i want to know if there are any seats for machine learning",
        "how about EECS 700?",
        "when does MATH 125 meet?",
        "is there space in machine learning class?",
        "BSAN 460?",
        "what sections of physics 101 are available?",
        "tell me about artificial intelligence courses",
        "i want to if there are any seats are avilable for Deep Reinforcement Learning?",
        "hello",  # Should NOT trigger
        "what is the meaning of life",  # Should NOT trigger
    ]
    
    print("\nTest Results:")
    print("-" * 60)
    
    for query in test_queries:
        result = detector.detect(query)
        status = "LIVE" if result["needs_live"] else "NO"
        print(f"\nQuery: '{query}'")
        print(f"  {status} | Intent: {result['intent']} | Topic: {result['topic']}")
        print(f"  Confidence: {result['confidence']:.2f} | Method: {result['method']}")


if __name__ == "__main__":
    main()