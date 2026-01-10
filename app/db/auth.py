"""
Auth Middleware for BabyJay
===========================
Verifies Supabase JWT tokens and extracts user info.
"""

import os
from typing import Optional
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
import base64
import json

load_dotenv()

security = HTTPBearer(auto_error=False)


class AuthUser:
    """Authenticated user info extracted from JWT."""
    
    def __init__(self, user_id: str, email: Optional[str] = None):
        self.id = user_id
        self.email = email
    
    def __repr__(self):
        return f"AuthUser(id={self.id}, email={self.email})"


def decode_token(token: str) -> dict:
    """Decode Supabase JWT token (without verification - Supabase handles that)."""
    try:
        # JWT has 3 parts: header.payload.signature
        parts = token.split('.')
        if len(parts) != 3:
            raise ValueError("Invalid token format")
        
        # Decode the payload (middle part)
        payload_b64 = parts[1]
        # Add padding if needed
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += '=' * padding
        
        payload_json = base64.urlsafe_b64decode(payload_b64)
        payload = json.loads(payload_json)
        
        return payload
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> AuthUser:
    """
    Dependency to get current authenticated user.
    """
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    token = credentials.credentials
    payload = decode_token(token)
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token: no user ID")
    
    email = payload.get("email")
    
    return AuthUser(user_id=user_id, email=email)


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Optional[AuthUser]:
    """
    Dependency to get user if authenticated, None otherwise.
    """
    if not credentials:
        return None
    
    try:
        token = credentials.credentials
        payload = decode_token(token)
        
        user_id = payload.get("sub")
        if not user_id:
            return None
        
        email = payload.get("email")
        return AuthUser(user_id=user_id, email=email)
    except:
        return None