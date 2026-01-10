"""
BabyJay Feedback System
========================
Collects user feedback (thumbs up/down) for RLHF and quality improvement.

Endpoints:
  POST /api/feedback - Submit feedback
  GET  /api/feedback/stats - Get feedback statistics (admin)
  GET  /api/feedback/export - Export all feedback (admin)
"""

import os
import json
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import hashlib

# Try to import Supabase
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("[WARNING] Supabase not available, using in-memory storage")

from dotenv import load_dotenv
load_dotenv()

# ==================== CONFIGURATION ====================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "babyjay-admin-2026")  # Change in production!

# ==================== MODELS ====================

class FeedbackRating(str, Enum):
    UP = "up"
    DOWN = "down"


class FeedbackRequest(BaseModel):
    """Request model for submitting feedback."""
    session_id: str = Field(..., description="User session ID")
    message_id: str = Field(..., description="Unique message ID")
    query: str = Field(..., description="User's original question")
    response: str = Field(..., description="Bot's response")
    rating: FeedbackRating = Field(..., description="Thumbs up or down")
    feedback_text: Optional[str] = Field(None, description="Optional text feedback")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class FeedbackResponse(BaseModel):
    """Response model after submitting feedback."""
    success: bool
    feedback_id: str
    message: str


class FeedbackStats(BaseModel):
    """Statistics about collected feedback."""
    total_feedback: int
    thumbs_up: int
    thumbs_down: int
    approval_rate: float
    today_count: int
    this_week_count: int
    top_negative_queries: List[Dict[str, Any]]
    feedback_by_day: List[Dict[str, Any]]


# ==================== STORAGE ====================

class FeedbackStore:
    """
    Handles feedback storage.
    Uses Supabase if available, falls back to in-memory.
    """
    
    def __init__(self):
        self.supabase: Optional[Client] = None
        self.memory_store: List[Dict] = []
        
        if SUPABASE_AVAILABLE and SUPABASE_URL and SUPABASE_KEY:
            try:
                self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
                print("[INFO] Feedback store connected to Supabase")
            except Exception as e:
                print(f"[WARNING] Supabase connection failed: {e}")
    
    def _generate_feedback_id(self) -> str:
        """Generate unique feedback ID."""
        return f"fb_{uuid.uuid4().hex[:12]}"
    
    def _get_user_hash(self, session_id: str, ip_address: str = "") -> str:
        """Create anonymized user hash for analytics."""
        raw = f"{session_id}:{ip_address}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
    
    async def save_feedback(
        self,
        feedback: FeedbackRequest,
        ip_address: str = "",
        user_agent: str = ""
    ) -> str:
        """Save feedback to storage. Returns feedback_id."""
        
        feedback_id = self._generate_feedback_id()
        user_hash = self._get_user_hash(feedback.session_id, ip_address)
        
        record = {
            "feedback_id": feedback_id,
            "session_id": feedback.session_id,
            "message_id": feedback.message_id,
            "user_hash": user_hash,
            "query": feedback.query[:1000],  # Limit size
            "response": feedback.response[:5000],  # Limit size
            "rating": feedback.rating.value,
            "feedback_text": feedback.feedback_text[:500] if feedback.feedback_text else None,
            "metadata": feedback.metadata or {},
            "ip_hash": hashlib.sha256(ip_address.encode()).hexdigest()[:16] if ip_address else None,
            "user_agent": user_agent[:200] if user_agent else None,
            "created_at": datetime.utcnow().isoformat(),
        }
        
        if self.supabase:
            try:
                result = self.supabase.table("feedback").insert(record).execute()
                return feedback_id
            except Exception as e:
                print(f"[ERROR] Supabase insert failed: {e}")
                # Fall back to memory
                self.memory_store.append(record)
                return feedback_id
        else:
            self.memory_store.append(record)
            return feedback_id
    
    async def get_stats(self) -> FeedbackStats:
        """Get feedback statistics."""
        
        if self.supabase:
            try:
                # Get all feedback
                result = self.supabase.table("feedback").select("*").execute()
                all_feedback = result.data
            except Exception as e:
                print(f"[ERROR] Supabase query failed: {e}")
                all_feedback = self.memory_store
        else:
            all_feedback = self.memory_store
        
        if not all_feedback:
            return FeedbackStats(
                total_feedback=0,
                thumbs_up=0,
                thumbs_down=0,
                approval_rate=0.0,
                today_count=0,
                this_week_count=0,
                top_negative_queries=[],
                feedback_by_day=[]
            )
        
        # Calculate stats
        total = len(all_feedback)
        thumbs_up = sum(1 for f in all_feedback if f.get("rating") == "up")
        thumbs_down = total - thumbs_up
        approval_rate = (thumbs_up / total * 100) if total > 0 else 0
        
        # Today's count
        today = datetime.utcnow().date().isoformat()
        today_count = sum(1 for f in all_feedback if f.get("created_at", "").startswith(today))
        
        # This week's count
        week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
        this_week_count = sum(1 for f in all_feedback if f.get("created_at", "") >= week_ago)
        
        # Top negative queries (for improvement)
        negative = [f for f in all_feedback if f.get("rating") == "down"]
        # Group by query similarity (simple: exact match for now)
        query_counts = {}
        for f in negative:
            q = f.get("query", "")[:100]
            if q in query_counts:
                query_counts[q]["count"] += 1
            else:
                query_counts[q] = {"query": q, "count": 1, "example_response": f.get("response", "")[:200]}
        
        top_negative = sorted(query_counts.values(), key=lambda x: x["count"], reverse=True)[:10]
        
        # Feedback by day (last 7 days)
        feedback_by_day = []
        for i in range(7):
            day = (datetime.utcnow() - timedelta(days=i)).date().isoformat()
            day_feedback = [f for f in all_feedback if f.get("created_at", "").startswith(day)]
            day_up = sum(1 for f in day_feedback if f.get("rating") == "up")
            day_down = len(day_feedback) - day_up
            feedback_by_day.append({
                "date": day,
                "total": len(day_feedback),
                "thumbs_up": day_up,
                "thumbs_down": day_down
            })
        
        return FeedbackStats(
            total_feedback=total,
            thumbs_up=thumbs_up,
            thumbs_down=thumbs_down,
            approval_rate=round(approval_rate, 1),
            today_count=today_count,
            this_week_count=this_week_count,
            top_negative_queries=top_negative,
            feedback_by_day=feedback_by_day
        )
    
    async def export_all(self, limit: int = 1000) -> List[Dict]:
        """Export all feedback for analysis."""
        
        if self.supabase:
            try:
                result = self.supabase.table("feedback")\
                    .select("*")\
                    .order("created_at", desc=True)\
                    .limit(limit)\
                    .execute()
                return result.data
            except Exception as e:
                print(f"[ERROR] Supabase export failed: {e}")
                return self.memory_store[-limit:]
        else:
            return self.memory_store[-limit:]
    
    async def get_training_pairs(self) -> List[Dict]:
        """
        Get feedback pairs for RLHF/DPO training.
        Returns pairs of (query, good_response, bad_response).
        """
        
        all_feedback = await self.export_all(limit=5000)
        
        # Group by similar queries
        query_groups = {}
        for f in all_feedback:
            q = f.get("query", "").lower().strip()
            if q not in query_groups:
                query_groups[q] = {"up": [], "down": []}
            
            if f.get("rating") == "up":
                query_groups[q]["up"].append(f.get("response", ""))
            else:
                query_groups[q]["down"].append(f.get("response", ""))
        
        # Create training pairs
        training_pairs = []
        for query, responses in query_groups.items():
            if responses["up"] and responses["down"]:
                training_pairs.append({
                    "query": query,
                    "chosen": responses["up"][0],  # Good response
                    "rejected": responses["down"][0]  # Bad response
                })
        
        return training_pairs


# Global store instance
feedback_store = FeedbackStore()


# ==================== API ROUTES ====================

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


def verify_admin(admin_key: str = None):
    """Simple admin verification."""
    if admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    return True


@router.post("/", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest, request: Request):
    """
    Submit user feedback (thumbs up/down).
    
    This is the main endpoint called when users click 👍 or 👎.
    """
    try:
        # Get request metadata
        ip_address = request.client.host if request.client else ""
        user_agent = request.headers.get("user-agent", "")
        
        # Save feedback
        feedback_id = await feedback_store.save_feedback(
            feedback=feedback,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        return FeedbackResponse(
            success=True,
            feedback_id=feedback_id,
            message="Thanks for your feedback!"
        )
    
    except Exception as e:
        print(f"[ERROR] Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to save feedback")


@router.get("/stats", response_model=FeedbackStats)
async def get_feedback_stats(admin_key: str = None):
    """
    Get feedback statistics (admin only).
    
    Usage: GET /api/feedback/stats?admin_key=your-secret
    """
    verify_admin(admin_key)
    
    try:
        stats = await feedback_store.get_stats()
        return stats
    except Exception as e:
        print(f"[ERROR] Stats query failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get stats")


@router.get("/export")
async def export_feedback(admin_key: str = None, limit: int = 1000):
    """
    Export all feedback for analysis (admin only).
    
    Usage: GET /api/feedback/export?admin_key=your-secret&limit=1000
    """
    verify_admin(admin_key)
    
    try:
        feedback = await feedback_store.export_all(limit=limit)
        return {"count": len(feedback), "feedback": feedback}
    except Exception as e:
        print(f"[ERROR] Export failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to export")


@router.get("/training-pairs")
async def get_training_pairs(admin_key: str = None):
    """
    Get feedback pairs for RLHF/DPO training (admin only).
    
    Returns pairs where same query got both 👍 and 👎.
    These are perfect for training preference models.
    """
    verify_admin(admin_key)
    
    try:
        pairs = await feedback_store.get_training_pairs()
        return {
            "count": len(pairs),
            "pairs": pairs,
            "note": "Use these for DPO training when you have 500+ pairs"
        }
    except Exception as e:
        print(f"[ERROR] Training pairs export failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get training pairs")


# ==================== SUPABASE SCHEMA ====================
"""
Run this SQL in Supabase to create the feedback table:

CREATE TABLE feedback (
    id BIGSERIAL PRIMARY KEY,
    feedback_id TEXT UNIQUE NOT NULL,
    session_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    user_hash TEXT,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    feedback_text TEXT,
    metadata JSONB DEFAULT '{}',
    ip_hash TEXT,
    user_agent TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for faster queries
CREATE INDEX idx_feedback_created_at ON feedback(created_at DESC);
CREATE INDEX idx_feedback_rating ON feedback(rating);
CREATE INDEX idx_feedback_session ON feedback(session_id);

-- Row Level Security (optional but recommended)
ALTER TABLE feedback ENABLE ROW LEVEL SECURITY;

-- Policy: Only service role can read/write
CREATE POLICY "Service role full access" ON feedback
    FOR ALL
    USING (auth.role() = 'service_role');
"""


# ==================== INTEGRATION WITH MAIN APP ====================
"""
Add to your main.py:

from app.api.feedback import router as feedback_router
app.include_router(feedback_router)
"""


if __name__ == "__main__":
    # Test the feedback system
    import asyncio
    
    async def test():
        print("Testing Feedback System...")
        
        # Test feedback submission
        test_feedback = FeedbackRequest(
            session_id="test-session-123",
            message_id="msg-456",
            query="seats for machine learning?",
            response="EECS 658 has 30 seats available...",
            rating=FeedbackRating.UP,
            feedback_text="Very helpful!"
        )
        
        feedback_id = await feedback_store.save_feedback(test_feedback)
        print(f"Saved feedback: {feedback_id}")
        
        # Test negative feedback
        test_feedback_neg = FeedbackRequest(
            session_id="test-session-123",
            message_id="msg-789",
            query="who is prof kulkarni?",
            response="I found SW 860 course...",  # Wrong answer
            rating=FeedbackRating.DOWN,
            feedback_text="Wrong answer, asked about professor not course"
        )
        
        feedback_id = await feedback_store.save_feedback(test_feedback_neg)
        print(f"Saved negative feedback: {feedback_id}")
        
        # Get stats
        stats = await feedback_store.get_stats()
        print(f"\n📊 Stats:")
        print(f"   Total: {stats.total_feedback}")
        print(f"   👍 {stats.thumbs_up} | 👎 {stats.thumbs_down}")
        print(f"   Approval: {stats.approval_rate}%")
        
        print("\nFeedback system working!")
    
    asyncio.run(test())