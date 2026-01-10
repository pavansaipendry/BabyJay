"""
BabyJay Rate Limiting
======================
Prevents abuse and protects OpenAI budget.

Features:
- Per-IP rate limiting
- Per-session rate limiting  
- Global daily budget limit
- Configurable limits
"""

import os
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

from dotenv import load_dotenv
load_dotenv()


# ==================== CONFIGURATION ====================

# Rate limits
REQUESTS_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
REQUESTS_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "100"))
REQUESTS_PER_DAY = int(os.getenv("RATE_LIMIT_PER_DAY", "500"))

# Budget protection
DAILY_BUDGET_LIMIT = float(os.getenv("DAILY_BUDGET_LIMIT", "50.0"))  # $50/day default
ESTIMATED_COST_PER_QUERY = 0.005  # ~$0.005 per query average

# Whitelist (bypass rate limiting)
WHITELISTED_IPS = os.getenv("WHITELISTED_IPS", "").split(",")
WHITELISTED_IPS = [ip.strip() for ip in WHITELISTED_IPS if ip.strip()]


# ==================== RATE LIMITER ====================

@dataclass
class RateLimitInfo:
    """Information about rate limit status."""
    allowed: bool
    remaining: int
    reset_at: datetime
    limit_type: str  # "minute", "hour", "day", "budget"
    message: str


class RateLimiter:
    """
    Token bucket rate limiter with multiple time windows.
    Uses in-memory storage (reset on restart).
    For production, use Redis.
    """
    
    def __init__(self):
        # Track requests per IP
        self.minute_buckets: Dict[str, list] = defaultdict(list)
        self.hour_buckets: Dict[str, list] = defaultdict(list)
        self.day_buckets: Dict[str, list] = defaultdict(list)
        
        # Track daily spend
        self.daily_spend: Dict[str, float] = defaultdict(float)
        self.daily_queries: Dict[str, int] = defaultdict(int)
        
        # Global daily tracking
        self.global_spend_today = 0.0
        self.global_queries_today = 0
        self.current_day = datetime.utcnow().date()
    
    def _get_ip_hash(self, ip: str) -> str:
        """Hash IP for privacy."""
        return hashlib.sha256(ip.encode()).hexdigest()[:16]
    
    def _cleanup_old_entries(self, bucket: list, max_age_seconds: int) -> list:
        """Remove entries older than max_age_seconds."""
        cutoff = time.time() - max_age_seconds
        return [t for t in bucket if t > cutoff]
    
    def _reset_daily_if_needed(self):
        """Reset daily counters at midnight."""
        today = datetime.utcnow().date()
        if today != self.current_day:
            self.daily_spend.clear()
            self.daily_queries.clear()
            self.global_spend_today = 0.0
            self.global_queries_today = 0
            self.current_day = today
    
    def check_rate_limit(self, ip: str, session_id: str = None) -> RateLimitInfo:
        """
        Check if request is allowed.
        Returns RateLimitInfo with status.
        """
        self._reset_daily_if_needed()
        
        # Check whitelist
        if ip in WHITELISTED_IPS:
            return RateLimitInfo(
                allowed=True,
                remaining=999,
                reset_at=datetime.utcnow() + timedelta(minutes=1),
                limit_type="none",
                message="Whitelisted"
            )
        
        ip_hash = self._get_ip_hash(ip)
        now = time.time()
        
        # Clean up old entries
        self.minute_buckets[ip_hash] = self._cleanup_old_entries(
            self.minute_buckets[ip_hash], 60
        )
        self.hour_buckets[ip_hash] = self._cleanup_old_entries(
            self.hour_buckets[ip_hash], 3600
        )
        self.day_buckets[ip_hash] = self._cleanup_old_entries(
            self.day_buckets[ip_hash], 86400
        )
        
        # Check minute limit
        minute_count = len(self.minute_buckets[ip_hash])
        if minute_count >= REQUESTS_PER_MINUTE:
            return RateLimitInfo(
                allowed=False,
                remaining=0,
                reset_at=datetime.utcnow() + timedelta(seconds=60),
                limit_type="minute",
                message=f"Rate limit exceeded. Max {REQUESTS_PER_MINUTE} requests per minute. Try again in a minute."
            )
        
        # Check hour limit
        hour_count = len(self.hour_buckets[ip_hash])
        if hour_count >= REQUESTS_PER_HOUR:
            return RateLimitInfo(
                allowed=False,
                remaining=0,
                reset_at=datetime.utcnow() + timedelta(hours=1),
                limit_type="hour",
                message=f"Hourly limit exceeded. Max {REQUESTS_PER_HOUR} requests per hour. Take a break!"
            )
        
        # Check day limit
        day_count = len(self.day_buckets[ip_hash])
        if day_count >= REQUESTS_PER_DAY:
            return RateLimitInfo(
                allowed=False,
                remaining=0,
                reset_at=datetime.utcnow().replace(hour=0, minute=0, second=0) + timedelta(days=1),
                limit_type="day",
                message=f"Daily limit exceeded. Max {REQUESTS_PER_DAY} requests per day. Come back tomorrow!"
            )
        
        # Check global budget
        if self.global_spend_today >= DAILY_BUDGET_LIMIT:
            return RateLimitInfo(
                allowed=False,
                remaining=0,
                reset_at=datetime.utcnow().replace(hour=0, minute=0, second=0) + timedelta(days=1),
                limit_type="budget",
                message="I'm taking a rest for today. Come back tomorrow! 🐦💤"
            )
        
        # All checks passed - record this request
        self.minute_buckets[ip_hash].append(now)
        self.hour_buckets[ip_hash].append(now)
        self.day_buckets[ip_hash].append(now)
        
        # Update global counters
        self.global_queries_today += 1
        self.global_spend_today += ESTIMATED_COST_PER_QUERY
        
        return RateLimitInfo(
            allowed=True,
            remaining=REQUESTS_PER_MINUTE - minute_count - 1,
            reset_at=datetime.utcnow() + timedelta(seconds=60),
            limit_type="none",
            message="OK"
        )
    
    def get_stats(self) -> Dict:
        """Get rate limiter statistics."""
        self._reset_daily_if_needed()
        
        return {
            "global_queries_today": self.global_queries_today,
            "global_spend_today": round(self.global_spend_today, 2),
            "budget_remaining": round(DAILY_BUDGET_LIMIT - self.global_spend_today, 2),
            "budget_limit": DAILY_BUDGET_LIMIT,
            "active_ips": len(self.day_buckets),
            "limits": {
                "per_minute": REQUESTS_PER_MINUTE,
                "per_hour": REQUESTS_PER_HOUR,
                "per_day": REQUESTS_PER_DAY
            }
        }
    
    def record_actual_cost(self, cost: float):
        """Record actual OpenAI cost (more accurate than estimate)."""
        self.global_spend_today += cost - ESTIMATED_COST_PER_QUERY  # Adjust from estimate


# Global instance
rate_limiter = RateLimiter()


# ==================== FASTAPI MIDDLEWARE ====================

async def rate_limit_middleware(request: Request, call_next):
    """
    FastAPI middleware for rate limiting.
    Add to your app with: app.middleware("http")(rate_limit_middleware)
    """
    # Skip rate limiting for non-chat endpoints
    if not request.url.path.startswith("/api/chat"):
        return await call_next(request)
    
    # Get client IP
    ip = request.client.host if request.client else "unknown"
    
    # Check for forwarded IP (behind proxy)
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    
    # Check rate limit
    session_id = request.headers.get("x-session-id", "")
    limit_info = rate_limiter.check_rate_limit(ip, session_id)
    
    if not limit_info.allowed:
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit_exceeded",
                "message": limit_info.message,
                "limit_type": limit_info.limit_type,
                "retry_after": limit_info.reset_at.isoformat()
            },
            headers={
                "Retry-After": str(int((limit_info.reset_at - datetime.utcnow()).total_seconds())),
                "X-RateLimit-Remaining": str(limit_info.remaining),
                "X-RateLimit-Reset": limit_info.reset_at.isoformat()
            }
        )
    
    # Add rate limit headers to response
    response = await call_next(request)
    response.headers["X-RateLimit-Remaining"] = str(limit_info.remaining)
    response.headers["X-RateLimit-Limit"] = str(REQUESTS_PER_MINUTE)
    
    return response


# ==================== DECORATOR FOR ROUTES ====================

def rate_limited(func):
    """
    Decorator for rate limiting individual routes.
    
    Usage:
        @router.post("/chat")
        @rate_limited
        async def chat(request: Request, ...):
            ...
    """
    @wraps(func)
    async def wrapper(*args, request: Request = None, **kwargs):
        if request is None:
            # Try to find request in args
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
        
        if request:
            ip = request.client.host if request.client else "unknown"
            forwarded = request.headers.get("x-forwarded-for")
            if forwarded:
                ip = forwarded.split(",")[0].strip()
            
            limit_info = rate_limiter.check_rate_limit(ip)
            
            if not limit_info.allowed:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "rate_limit_exceeded",
                        "message": limit_info.message,
                        "retry_after": limit_info.reset_at.isoformat()
                    }
                )
        
        return await func(*args, request=request, **kwargs)
    
    return wrapper


# ==================== API ROUTES ====================

from fastapi import APIRouter

router = APIRouter(prefix="/api/rate-limit", tags=["rate-limit"])


@router.get("/stats")
async def get_rate_limit_stats(admin_key: str = None):
    """Get rate limiter statistics (admin only)."""
    admin_secret = os.getenv("ADMIN_SECRET", "babyjay-admin-2026")
    if admin_key != admin_secret:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    
    return rate_limiter.get_stats()


@router.get("/check")
async def check_my_rate_limit(request: Request):
    """Check your current rate limit status."""
    ip = request.client.host if request.client else "unknown"
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    
    # Don't actually count this as a request
    ip_hash = rate_limiter._get_ip_hash(ip)
    
    minute_remaining = REQUESTS_PER_MINUTE - len(rate_limiter.minute_buckets.get(ip_hash, []))
    hour_remaining = REQUESTS_PER_HOUR - len(rate_limiter.hour_buckets.get(ip_hash, []))
    day_remaining = REQUESTS_PER_DAY - len(rate_limiter.day_buckets.get(ip_hash, []))
    
    return {
        "minute": {"remaining": max(0, minute_remaining), "limit": REQUESTS_PER_MINUTE},
        "hour": {"remaining": max(0, hour_remaining), "limit": REQUESTS_PER_HOUR},
        "day": {"remaining": max(0, day_remaining), "limit": REQUESTS_PER_DAY},
        "message": "Use these wisely! 🐦"
    }


# ==================== INTEGRATION ====================
"""
Add to your main.py:

from app.api.rate_limit import rate_limit_middleware, router as rate_limit_router

# Add middleware
app.middleware("http")(rate_limit_middleware)

# Add routes
app.include_router(rate_limit_router)
"""


if __name__ == "__main__":
    # Test rate limiter
    print("Testing Rate Limiter...")
    
    # Simulate requests
    test_ip = "192.168.1.100"
    
    for i in range(15):
        result = rate_limiter.check_rate_limit(test_ip)
        status = "✅" if result.allowed else "❌"
        print(f"{status} Request {i+1}: {result.message} (remaining: {result.remaining})")
    
    print("\n📊 Stats:")
    stats = rate_limiter.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Rate limiter working!")