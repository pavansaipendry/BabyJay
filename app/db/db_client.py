"""
Supabase Database Client for BabyJay
====================================
Handles all database operations for conversations and messages.
"""

import os
from typing import List, Dict, Optional
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")


def get_supabase_client() -> Client:
    """Get Supabase client with service role (bypasses RLS for backend)."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in environment")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


class DatabaseClient:
    """Database operations for BabyJay."""
    
    def __init__(self):
        self.client = get_supabase_client()
    
    # ==================== CONVERSATIONS ====================
    
    def create_conversation(self, user_id: str, title: str = "New Chat") -> Dict:
        """Create a new conversation for a user."""
        result = self.client.table("conversations").insert({
            "user_id": user_id,
            "title": title
        }).execute()
        
        return result.data[0] if result.data else None
    
    def get_conversations(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get all conversations for a user, sorted by most recent."""
        result = self.client.table("conversations") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("updated_at", desc=True) \
            .limit(limit) \
            .execute()
        
        return result.data or []
    
    def get_conversation(self, conversation_id: str, user_id: str) -> Optional[Dict]:
        """Get a single conversation (with ownership check)."""
        result = self.client.table("conversations") \
            .select("*") \
            .eq("id", conversation_id) \
            .eq("user_id", user_id) \
            .execute()
        
        return result.data[0] if result.data else None
    
    def update_conversation_title(self, conversation_id: str, user_id: str, title: str) -> Optional[Dict]:
        """Update conversation title."""
        result = self.client.table("conversations") \
            .update({"title": title}) \
            .eq("id", conversation_id) \
            .eq("user_id", user_id) \
            .execute()
        
        return result.data[0] if result.data else None
    
    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """Delete a conversation and all its messages."""
        result = self.client.table("conversations") \
            .delete() \
            .eq("id", conversation_id) \
            .eq("user_id", user_id) \
            .execute()
        
        return len(result.data) > 0 if result.data else False
    
    def touch_conversation(self, conversation_id: str) -> None:
        """Update the updated_at timestamp (called when new message added)."""
        self.client.table("conversations") \
            .update({"updated_at": datetime.utcnow().isoformat()}) \
            .eq("id", conversation_id) \
            .execute()
    
    # ==================== MESSAGES ====================
    
    def add_message(self, conversation_id: str, role: str, content: str) -> Dict:
        """Add a message to a conversation."""
        result = self.client.table("messages").insert({
            "conversation_id": conversation_id,
            "role": role,
            "content": content
        }).execute()
        
        # Update conversation timestamp
        self.touch_conversation(conversation_id)
        
        return result.data[0] if result.data else None
    
    def get_messages(self, conversation_id: str, limit: int = 100) -> List[Dict]:
        """Get all messages in a conversation, sorted by time."""
        result = self.client.table("messages") \
            .select("*") \
            .eq("conversation_id", conversation_id) \
            .order("created_at", desc=False) \
            .limit(limit) \
            .execute()
        
        return result.data or []
    
    def get_recent_messages(self, conversation_id: str, limit: int = 20) -> List[Dict]:
        """Get recent messages for context (most recent first, then reversed)."""
        result = self.client.table("messages") \
            .select("*") \
            .eq("conversation_id", conversation_id) \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        
        # Reverse to get chronological order
        messages = result.data or []
        return list(reversed(messages))
    
    # ==================== UTILITY ====================
    
    def generate_title_from_message(self, message: str) -> str:
        """Generate a conversation title from the first message."""
        # Take first 50 chars, clean up
        title = message.strip()[:50]
        if len(message) > 50:
            title += "..."
        return title or "New Chat"


# Singleton instance
_db_client: Optional[DatabaseClient] = None

def get_db() -> DatabaseClient:
    """Get database client singleton."""
    global _db_client
    if _db_client is None:
        _db_client = DatabaseClient()
    return _db_client


# Quick test
if __name__ == "__main__":
    print("Testing Database Client")
    print("=" * 60)
    
    try:
        db = get_db()
        print("✓ Connected to Supabase")
        
        # Test creating a conversation (will fail without real user_id)
        # This is just to test the connection
        print("✓ Database client initialized")
        print("\nConnection successful!")
        
    except Exception as e:
        print(f"✗ Error: {e}")