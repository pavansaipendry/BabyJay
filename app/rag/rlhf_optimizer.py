"""
BabyJay RLHF Optimizer
=======================
Uses collected feedback (👍/👎) to improve response quality.

This implements lightweight RLHF without model fine-tuning:
1. Analyzes patterns from positive/negative feedback
2. Generates "lessons learned" from failures
3. Injects learned patterns into system prompts
4. Tracks query types that need improvement

Usage:
    from app.rag.rlhf_optimizer import RLHFOptimizer
    
    optimizer = RLHFOptimizer()
    enhanced_prompt = optimizer.enhance_prompt(base_prompt, query)
"""

import os
import json
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict
import hashlib

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")


class RLHFOptimizer:
    """
    Reinforcement Learning from Human Feedback optimizer.
    
    Learns from user feedback to improve response quality without
    fine-tuning the underlying LLM.
    """
    
    def __init__(self, cache_ttl: int = 300, debug: bool = False):
        self.debug = debug
        self.cache_ttl = cache_ttl  # How often to refresh patterns (seconds)
        self.supabase: Optional[Client] = None
        
        # Cached patterns (refreshed periodically)
        self._patterns_cache: Dict[str, Any] = {}
        self._cache_timestamp: float = 0
        
        # Query type classifications
        self.query_types = {
            "course_info": ["course", "class", "prerequisite", "prereq", "credit", "eecs", "math"],
            "faculty_search": ["professor", "prof", "faculty", "researcher", "teaches", "research"],
            "live_lookup": ["seats", "available", "enroll", "open", "section", "who teaches"],
            "campus_info": ["dining", "food", "bus", "transit", "library", "gym", "parking"],
            "general": []  # Fallback
        }
        
        # Connect to Supabase
        if SUPABASE_AVAILABLE and SUPABASE_URL and SUPABASE_KEY:
            try:
                self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
                if self.debug:
                    print("[RLHF] Connected to Supabase")
            except Exception as e:
                if self.debug:
                    print(f"[RLHF] Supabase connection failed: {e}")
    
    def _classify_query(self, query: str) -> str:
        """Classify query into a type for pattern matching."""
        q = query.lower()
        
        for query_type, keywords in self.query_types.items():
            if any(kw in q for kw in keywords):
                return query_type
        
        return "general"
    
    def _fetch_feedback(self, limit: int = 500) -> List[Dict]:
        """Fetch recent feedback from database."""
        if not self.supabase:
            return []
        
        try:
            # Get feedback from last 30 days
            cutoff = (datetime.utcnow() - timedelta(days=30)).isoformat()
            
            result = self.supabase.table("feedback")\
                .select("*")\
                .gte("created_at", cutoff)\
                .order("created_at", desc=True)\
                .limit(limit)\
                .execute()
            
            return result.data or []
        except Exception as e:
            if self.debug:
                print(f"[RLHF] Feedback fetch failed: {e}")
            return []
    
    def _analyze_patterns(self) -> Dict[str, Any]:
        """
        Analyze feedback to extract patterns.
        
        Returns patterns like:
        - Which query types have low approval
        - Common issues in negative feedback
        - Successful response characteristics
        """
        import time
        
        # Check cache
        if time.time() - self._cache_timestamp < self.cache_ttl:
            return self._patterns_cache
        
        feedback = self._fetch_feedback()
        
        if not feedback:
            return {"lessons": [], "problem_queries": [], "success_patterns": []}
        
        # Analyze by query type
        by_type = defaultdict(lambda: {"up": 0, "down": 0, "examples": []})
        
        for f in feedback:
            query_type = self._classify_query(f.get("query", ""))
            rating = f.get("rating", "")
            
            if rating == "up":
                by_type[query_type]["up"] += 1
            else:
                by_type[query_type]["down"] += 1
                by_type[query_type]["examples"].append({
                    "query": f.get("query", "")[:100],
                    "response": f.get("response", "")[:200],
                    "feedback_text": f.get("feedback_text", "")
                })
        
        # Find problem areas (approval < 70%)
        problem_types = []
        for qtype, stats in by_type.items():
            total = stats["up"] + stats["down"]
            if total >= 3:  # Need at least 3 samples
                approval = stats["up"] / total * 100
                if approval < 70:
                    problem_types.append({
                        "type": qtype,
                        "approval": round(approval, 1),
                        "examples": stats["examples"][:3]
                    })
        
        # Extract lessons from negative feedback
        lessons = self._extract_lessons(feedback)
        
        # Find success patterns from positive feedback
        success_patterns = self._extract_success_patterns(feedback)
        
        patterns = {
            "lessons": lessons,
            "problem_queries": problem_types,
            "success_patterns": success_patterns,
            "total_feedback": len(feedback),
            "overall_approval": self._calculate_approval(feedback)
        }
        
        # Cache results
        self._patterns_cache = patterns
        self._cache_timestamp = time.time()
        
        if self.debug:
            print(f"[RLHF] Analyzed {len(feedback)} feedback items")
            print(f"[RLHF] Found {len(lessons)} lessons, {len(problem_types)} problem areas")
        
        return patterns
    
    def _extract_lessons(self, feedback: List[Dict]) -> List[str]:
        """Extract actionable lessons from negative feedback."""
        lessons = []
        negative = [f for f in feedback if f.get("rating") == "down"]
        
        # Pattern: Wrong entity type returned
        wrong_entity = [f for f in negative if any(phrase in (f.get("feedback_text") or "").lower() 
                       for phrase in ["wrong", "not what i asked", "different", "asked about"])]
        if len(wrong_entity) >= 2:
            lessons.append("Pay close attention to what entity type the user is asking about (course vs professor vs location)")
        
        # Pattern: Missing information
        missing_info = [f for f in negative if any(phrase in (f.get("feedback_text") or "").lower()
                       for phrase in ["missing", "didn't include", "no information", "incomplete"])]
        if len(missing_info) >= 2:
            lessons.append("Include all relevant details in responses - users want comprehensive answers")
        
        # Pattern: Too generic
        too_generic = [f for f in negative if any(phrase in (f.get("feedback_text") or "").lower()
                      for phrase in ["generic", "vague", "not specific", "too general"])]
        if len(too_generic) >= 2:
            lessons.append("Be specific - include names, numbers, and concrete details rather than generic statements")
        
        # Pattern: Outdated information
        outdated = [f for f in negative if any(phrase in (f.get("feedback_text") or "").lower()
                   for phrase in ["outdated", "old", "not current", "wrong semester"])]
        if len(outdated) >= 2:
            lessons.append("Ensure information is current - check semester and verify data is up to date")
        
        # Pattern: Format issues
        format_issues = [f for f in negative if any(phrase in (f.get("feedback_text") or "").lower()
                        for phrase in ["too long", "too short", "format", "hard to read"])]
        if len(format_issues) >= 2:
            lessons.append("Keep responses concise and well-formatted - avoid walls of text")
        
        # Analyze response length correlation
        positive_lengths = [len(f.get("response", "")) for f in feedback if f.get("rating") == "up"]
        negative_lengths = [len(f.get("response", "")) for f in negative]
        
        if positive_lengths and negative_lengths:
            avg_positive = sum(positive_lengths) / len(positive_lengths)
            avg_negative = sum(negative_lengths) / len(negative_lengths)
            
            if avg_negative > avg_positive * 1.5:
                lessons.append("Keep responses concise - shorter, focused answers perform better")
            elif avg_negative < avg_positive * 0.5:
                lessons.append("Provide sufficient detail - very short answers may lack needed information")
        
        return lessons
    
    def _extract_success_patterns(self, feedback: List[Dict]) -> List[str]:
        """Extract patterns from successful responses."""
        patterns = []
        positive = [f for f in feedback if f.get("rating") == "up"]
        
        if not positive:
            return patterns
        
        # Analyze successful responses
        responses = [f.get("response", "") for f in positive]
        
        # Check for common elements in successful responses
        has_specifics = sum(1 for r in responses if re.search(r'\d+', r)) / len(responses)
        if has_specifics > 0.7:
            patterns.append("Include specific numbers (seats, credits, times)")
        
        has_names = sum(1 for r in responses if re.search(r'(Dr\.|Prof\.|Professor)', r)) / len(responses)
        if has_names > 0.5:
            patterns.append("Mention specific professor names when relevant")
        
        avg_length = sum(len(r) for r in responses) / len(responses)
        if 100 < avg_length < 500:
            patterns.append("Optimal response length: 100-500 characters")
        
        return patterns
    
    def _calculate_approval(self, feedback: List[Dict]) -> float:
        """Calculate overall approval rate."""
        if not feedback:
            return 0.0
        
        positive = sum(1 for f in feedback if f.get("rating") == "up")
        return round(positive / len(feedback) * 100, 1)
    
    def enhance_prompt(self, base_prompt: str, query: str) -> str:
        """
        Enhance the system prompt with learned patterns.
        
        This is the main integration point - call this before sending
        to the LLM to inject learned behaviors.
        """
        patterns = self._analyze_patterns()
        
        if not patterns.get("lessons") and not patterns.get("success_patterns"):
            return base_prompt  # No patterns yet, use base prompt
        
        # Build enhancement section
        enhancements = []
        
        # Add lessons learned
        if patterns.get("lessons"):
            enhancements.append("\nLEARNED FROM USER FEEDBACK:")
            for lesson in patterns["lessons"][:5]:  # Limit to top 5
                enhancements.append(f"- {lesson}")
        
        # Add success patterns
        if patterns.get("success_patterns"):
            enhancements.append("\nWHAT USERS APPRECIATE:")
            for pattern in patterns["success_patterns"][:3]:
                enhancements.append(f"- {pattern}")
        
        # Check if this query type is problematic
        query_type = self._classify_query(query)
        problem_types = {p["type"]: p for p in patterns.get("problem_queries", [])}
        
        if query_type in problem_types:
            problem = problem_types[query_type]
            enhancements.append(f"\n⚠️ NOTE: {query_type} queries have {problem['approval']}% approval. Be extra careful with accuracy.")
        
        if enhancements:
            enhanced_prompt = base_prompt + "\n" + "\n".join(enhancements)
            
            if self.debug:
                print(f"[RLHF] Enhanced prompt with {len(patterns.get('lessons', []))} lessons")
            
            return enhanced_prompt
        
        return base_prompt
    
    def get_query_guidance(self, query: str) -> Optional[str]:
        """
        Get specific guidance for a query based on past feedback.
        
        Returns None if no specific guidance available.
        """
        patterns = self._analyze_patterns()
        query_type = self._classify_query(query)
        
        # Check if this query type is problematic
        for problem in patterns.get("problem_queries", []):
            if problem["type"] == query_type:
                examples = problem.get("examples", [])
                if examples:
                    # Find similar past failures
                    q_lower = query.lower()
                    for ex in examples:
                        if any(word in q_lower for word in ex.get("query", "").lower().split()[:3]):
                            feedback_text = ex.get("feedback_text", "")
                            if feedback_text:
                                return f"Similar query had issue: {feedback_text}"
        
        return None
    
    def log_response(self, query: str, response: str, query_type: str = None):
        """
        Log a response for later analysis.
        
        Call this after generating a response to track patterns.
        """
        if not query_type:
            query_type = self._classify_query(query)
        
        # This could be extended to log to a file or database
        # for offline analysis
        if self.debug:
            print(f"[RLHF] Logged {query_type} response ({len(response)} chars)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RLHF system statistics."""
        patterns = self._analyze_patterns()
        
        return {
            "total_feedback_analyzed": patterns.get("total_feedback", 0),
            "overall_approval": patterns.get("overall_approval", 0),
            "lessons_learned": len(patterns.get("lessons", [])),
            "problem_areas": len(patterns.get("problem_queries", [])),
            "success_patterns": len(patterns.get("success_patterns", [])),
            "cache_age_seconds": int(
                (datetime.utcnow().timestamp() - self._cache_timestamp) 
                if self._cache_timestamp else 0
            )
        }


# ==================== INTEGRATION HELPER ====================

def integrate_rlhf_with_chat(chat_instance, optimizer: RLHFOptimizer = None):
    """
    Helper to integrate RLHF with existing BabyJayChat.
    
    Usage:
        from app.rag.rlhf_optimizer import RLHFOptimizer, integrate_rlhf_with_chat
        
        optimizer = RLHFOptimizer(debug=True)
        integrate_rlhf_with_chat(chat, optimizer)
    """
    if optimizer is None:
        optimizer = RLHFOptimizer()
    
    # Store original SYSTEM_PROMPT
    original_prompt = getattr(chat_instance, 'SYSTEM_PROMPT', None)
    
    # Create enhanced ask method
    original_ask = chat_instance.ask
    
    def enhanced_ask(question: str, use_history: bool = True) -> str:
        # Get guidance for this query
        guidance = optimizer.get_query_guidance(question)
        if guidance:
            print(f"[RLHF Guidance] {guidance}")
        
        # Call original ask
        response = original_ask(question, use_history)
        
        # Log for analysis
        optimizer.log_response(question, response)
        
        return response
    
    # Replace ask method
    chat_instance.ask = enhanced_ask
    
    return chat_instance


# ==================== CLI TESTING ====================

if __name__ == "__main__":
    print("=" * 60)
    print("BabyJay RLHF Optimizer")
    print("=" * 60)
    
    optimizer = RLHFOptimizer(debug=True)
    
    # Get current stats
    stats = optimizer.get_stats()
    print(f"\n📊 Current Stats:")
    print(f"   Feedback analyzed: {stats['total_feedback_analyzed']}")
    print(f"   Overall approval: {stats['overall_approval']}%")
    print(f"   Lessons learned: {stats['lessons_learned']}")
    print(f"   Problem areas: {stats['problem_areas']}")
    
    # Test prompt enhancement
    base_prompt = "You are BabyJay, a helpful KU assistant."
    test_query = "who teaches machine learning?"
    
    enhanced = optimizer.enhance_prompt(base_prompt, test_query)
    
    print(f"\n📝 Enhanced Prompt Preview:")
    print("-" * 40)
    print(enhanced[:500] + "..." if len(enhanced) > 500 else enhanced)
    
    # Get guidance for specific query
    guidance = optimizer.get_query_guidance(test_query)
    if guidance:
        print(f"\n⚠️ Query Guidance: {guidance}")
    else:
        print(f"\n✅ No specific concerns for this query type")
    
    print("\n" + "=" * 60)
    print("RLHF Optimizer ready for integration!")