"""
Chat API Routes for BabyJay
============================
Feedback handling moved to app/api/feedback.py
"""

from collections import OrderedDict
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
from app.db.db_client import get_db, DatabaseClient
from app.db.auth import get_current_user, get_optional_user, AuthUser
from app.rag.chat import BabyJayChat

router = APIRouter(prefix="/api", tags=["chat"])

# Persistent chat instances keyed by conversation_id to preserve state
# (last_mentioned_course, clarification state, department filters, etc.).
# OrderedDict + move_to_end gives us proper LRU eviction so a user in an
# active conversation never gets bumped by someone else starting earlier.
_chat_instances: "OrderedDict[str, BabyJayChat]" = OrderedDict()
_MAX_CACHED_INSTANCES = 200


def _get_or_create_chat(conversation_id: str) -> BabyJayChat:
    """Get existing chat instance or create new one. Preserves state across requests."""
    if conversation_id in _chat_instances:
        _chat_instances.move_to_end(conversation_id)  # mark as most recently used
        return _chat_instances[conversation_id]

    # Evict least-recently-used if cache is full
    if len(_chat_instances) >= _MAX_CACHED_INSTANCES:
        _chat_instances.popitem(last=False)

    instance = BabyJayChat(
        session_id=conversation_id,
        use_redis=False,
        debug=False
    )
    _chat_instances[conversation_id] = instance
    return instance

# Request/Response models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: Optional[str]
    title: Optional[str]

class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str

class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    created_at: str

class ConversationDetailResponse(BaseModel):
    conversation: ConversationResponse
    messages: List[MessageResponse]

class UpdateTitleRequest(BaseModel):
    title: str


# ==================== CHAT ====================

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user: AuthUser = Depends(get_current_user), # autentication verifies JWT Token.
    db: DatabaseClient = Depends(get_db)
):
    """Send a message and get a response (authenticated)."""
    conversation_id = request.conversation_id
    is_new_conversation = False

    try:
        # Create new conversation if needed
        if not conversation_id:
            title = db.generate_title_from_message(request.message)
            conversation = db.create_conversation(user.id, title)
            if not conversation:
                raise HTTPException(status_code=500, detail="Failed to create conversation")
            conversation_id = conversation["id"]
            is_new_conversation = True
        else:
            conversation = db.get_conversation(conversation_id, user.id)
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")

        # Get or create persistent chat instance (preserves follow-up state)
        chat_instance = _get_or_create_chat(conversation_id)

        # Load conversation history from DB into chat instance (if empty)
        if not chat_instance._conversation_history:
            recent_messages = db.get_recent_messages(conversation_id, limit=20)
            for msg in recent_messages:
                chat_instance._conversation_history.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # Get response — use_history=False because we manage history ourselves
        # This prevents chat.ask() from double-saving messages to its internal store
        response = chat_instance.ask(request.message, use_history=False)

        # Manually update the instance's history so follow-ups work
        chat_instance._conversation_history.append({"role": "user", "content": request.message})
        chat_instance._conversation_history.append({"role": "assistant", "content": response})

        # Save to database (single source of truth)
        db.add_message(conversation_id, "user", request.message)
        db.add_message(conversation_id, "assistant", response)

        # Get title
        if is_new_conversation:
            title = db.generate_title_from_message(request.message)
            db.update_conversation_title(conversation_id, user.id, title)
        else:
            title = conversation["title"]

        return ChatResponse(
            response=response,
            conversation_id=conversation_id,
            title=title
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@router.post("/chat/anonymous", response_model=ChatResponse)
async def chat_anonymous(request: ChatRequest):
    """Chat without authentication (no history saved)."""
    chat_instance = BabyJayChat(
        use_redis=False,
        debug=False
    )
    
    response = chat_instance.ask(request.message, use_history=False)
    
    return ChatResponse(
        response=response,
        conversation_id=None,
        title=None
    )


# ==================== CONVERSATIONS ====================

@router.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations(
    user: AuthUser = Depends(get_current_user),
    db: DatabaseClient = Depends(get_db),
    limit: int = 50
):
    """Get all conversations for the current user."""
    conversations = db.get_conversations(user.id, limit=limit)
    
    return [
        ConversationResponse(
            id=c["id"],
            title=c["title"],
            created_at=c["created_at"],
            updated_at=c["updated_at"]
        )
        for c in conversations
    ]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: str,
    user: AuthUser = Depends(get_current_user),
    db: DatabaseClient = Depends(get_db)
):
    """Get a conversation with all its messages."""
    conversation = db.get_conversation(conversation_id, user.id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    messages = db.get_messages(conversation_id)
    
    return ConversationDetailResponse(
        conversation=ConversationResponse(
            id=conversation["id"],
            title=conversation["title"],
            created_at=conversation["created_at"],
            updated_at=conversation["updated_at"]
        ),
        messages=[
            MessageResponse(
                id=m["id"],
                role=m["role"],
                content=m["content"],
                created_at=m["created_at"]
            )
            for m in messages
        ]
    )


@router.put("/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    request: UpdateTitleRequest,
    user: AuthUser = Depends(get_current_user),
    db: DatabaseClient = Depends(get_db)
):
    """Update conversation title."""
    result = db.update_conversation_title(conversation_id, user.id, request.title)
    if not result:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"success": True, "title": request.title}


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user: AuthUser = Depends(get_current_user),
    db: DatabaseClient = Depends(get_db)
):
    """Delete a conversation and all its messages."""
    success = db.delete_conversation(conversation_id, user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"success": True}


# ==================== HEALTH CHECK ====================

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "babyjay-api"}